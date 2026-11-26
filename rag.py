"""RAG system with Qdrant vector store, structured data (SQLite), caching, retries and citations."""

import asyncio
import logging
import random
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import config
from embeddings import get_embeddings
from vectorstore import QdrantVectorStoreManager
from cache import query_cache
from structured_extractor import get_table_extractor
from metrics import (
    LLM_ERRORS,
    LLM_RETRIES,
    QUERY_DURATION,
    QUERY_RESULTS,
)

logger = logging.getLogger(__name__)

REINDEXING_MESSAGE = "🔄 I'm rebuilding the index right now. Please ask again in a moment."
NOT_READY_MESSAGE = "⏳ I'm still starting up. Please try again in a moment."


def _retry_delay(attempt: int, base: float) -> float:
    """Exponential backoff with jitter, capped so a chat is never stuck long."""
    delay = base * (2 ** attempt)
    return min(30.0, delay + random.uniform(0, base))


def _is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


def _is_retryable(exc: Exception) -> bool:
    """Transient failures worth retrying; auth/validation errors are not."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (408, 409, 425, 429):
            return True
        return 500 <= status < 600

    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "429",
            "overloaded",
            "temporarily",
            "timeout",
            "timed out",
            "connection reset",
            "connection error",
            "try again",
            "service unavailable",
        )
    )


def _format_sources(docs: List[Document], limit: int = 3) -> str:
    """Build a compact ``Source: file, page N`` suffix from retrieved chunks."""
    seen: set[str] = set()
    labels: List[str] = []

    for doc in docs:
        meta = doc.metadata or {}
        name = meta.get("source") or Path(str(meta.get("file_path", ""))).name
        if not name:
            continue

        page = meta.get("page")
        if page is None:
            page = meta.get("page_number")
        try:
            page = int(page) + 1  # PyPDFLoader pages are 0-based
        except (TypeError, ValueError):
            page = None

        label = f"{name}, page {page}" if page else str(name)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= limit:
            break

    if not labels:
        return ""
    prefix = "📄 Source: " if len(labels) == 1 else "📄 Sources: "
    return "\n\n" + prefix + "; ".join(labels)


def _format_structured_results(rows: List[Dict[str, Any]], limit: int = 10) -> str:
    """Format structured query results for WhatsApp."""
    if not rows:
        return "No matching records found."

    lines = []
    for i, row in enumerate(rows[:limit], 1):
        parts = []
        if row.get("s_no"):
            parts.append(f"#{row['s_no']}")
        if row.get("name"):
            parts.append(row["name"])
        if row.get("mobile"):
            parts.append(f"📞 {row['mobile']}")
        if row.get("address"):
            parts.append(f"📍 {row['address']}")
        if row.get("district"):
            parts.append(f"🏙️ {row['district']}")
        if row.get("pin"):
            parts.append(f"📮 {row['pin']}")
        lines.append(" | ".join(parts))

    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more")

    return "\n".join(lines)


# Keywords that suggest a structured query
_STRUCTURED_KEYWORDS = {
    "count", "how many", "number of", "total",
    "mobile", "phone", "contact",
    "address", "district", "pin", "pincode", "postal",
    "member", "members",
    "list", "show", "find",
    "bidar", "kalaburagi", "gulbarga", "raichur", "hubballi", "mysore",
    "bangalore", "belgaum", "dharwad", "shimoga", "tumkur", "udupi",
    "hassan", "mandya", "kolar", "chikmagalur", "kodagu", "chamrajnagar",
    "bijapur", "bagalkot", "haveri", "gadag", "koppal", "yadgir",
    "chikkaballapur", "ramanagara", "chitradurga", "davangere", "uttara kannada",
    "dakshina kannada",
}


def _looks_like_structured_query(question: str) -> bool:
    """Heuristic: does this question look like it wants structured data?"""
    q_lower = question.lower()
    words = set(re.findall(r"[a-z]+", q_lower))
    # Check for structured keywords
    if words & _STRUCTURED_KEYWORDS:
        return True
    # Check for "X in Y" pattern (e.g., "members in Bidar")
    if re.search(r"\b(in|from|at|near)\s+\w+", q_lower):
        return True
    return False


def _extract_district(question: str) -> Optional[str]:
    """Try to extract a district name from the question."""
    districts = [
        "bagalkot", "bangalore", "bangalore rural", "belgaum", "bellary",
        "bidar", "bijapur", "chamarajanagar", "chikkaballapur", "chikmagalur",
        "chitradurga", "dakshina kannada", "davangere", "dharwad", "gadag",
        "gulbarga", "hassan", "haveri", "kalaburagi", "kodagu", "kolar",
        "koppal", "mandya", "mysore", "raichur", "ramanagara", "shimoga",
        "tumkur", "udupi", "uttara kannada", "yadgir"
    ]
    q_lower = question.lower()
    for d in districts:
        if d in q_lower:
            return d.title()
    return None


def _extract_name(question: str) -> Optional[str]:
    """Try to extract a person name from the question."""
    # Look for patterns like "mobile of X", "address of X", "X's mobile", etc.
    patterns = [
        r"(?:mobile|phone|contact|address)\s+(?:of|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:mobile|phone|contact|address)",
        r"(?:who is|find|search)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_mobile(question: str) -> Optional[str]:
    """Try to extract a mobile number from the question."""
    match = re.search(r"\b(\d{10})\b", question)
    if match:
        return match.group(1)
    return None


class ConversationMemory:
    """Per-chat conversation history for context."""

    def __init__(self, max_size: int = 3, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._history: Dict[str, List[Tuple[str, str, float]]] = {}  # chat_id -> [(q, a, time)]
        self._lock = threading.Lock()

    def add(self, chat_id: str, question: str, answer: str) -> None:
        with self._lock:
            now = time.time()
            if chat_id not in self._history:
                self._history[chat_id] = []
            self._history[chat_id].append((question, answer, now))
            # Trim to max_size
            if len(self._history[chat_id]) > self.max_size:
                self._history[chat_id] = self._history[chat_id][-self.max_size:]
            # Remove expired
            self._history[chat_id] = [
                (q, a, t) for q, a, t in self._history[chat_id]
                if now - t < self.ttl
            ]

    def get_context(self, chat_id: str) -> List[Tuple[str, str]]:
        with self._lock:
            now = time.time()
            if chat_id not in self._history:
                return []
            # Filter expired
            valid = [(q, a) for q, a, t in self._history[chat_id] if now - t < self.ttl]
            return valid[-self.max_size:]

    def clear(self, chat_id: str) -> None:
        with self._lock:
            if chat_id in self._history:
                del self._history[chat_id]


class RAGSystem:
    def __init__(self):
        # Embeddings and the Qdrant connection are created in `initialize`,
        # which the lifespan retries, so a database outage at boot no longer
        # crashes the import.
        self.embeddings = None

        self.llm = self._build_llm(config.LLM_MODEL, streaming=False)
        self._fallback_llm = None
        if (
            config.LLM_FALLBACK_MODEL
            and config.LLM_FALLBACK_MODEL != config.LLM_MODEL
        ):
            self._fallback_llm = self._build_llm(
                config.LLM_FALLBACK_MODEL, streaming=False
            )

        # Streaming LLM (created on demand)
        self._streaming_llm: Optional[ChatOpenAI] = None

        # Vector store manager (Qdrant only)
        self.vector_manager: Optional[QdrantVectorStoreManager] = None

        self.retriever = None
        self.qa_chain = None
        self._fallback_chain = None

        self._initialized = False
        self.init_error: Optional[str] = None

        # Serializes indexing/rebuilds between the watcher, the API and startup.
        self._index_lock = threading.RLock()

        # Queries may run while the index is being read, but a rebuild waits
        # for them to finish and tells new queries to retry instead of serving
        # an empty collection.
        self._state = threading.Condition()
        self._rebuilding = False
        self._active_queries = 0

        # Single-flight: identical concurrent questions share one LLM call.
        self._inflight: Dict[str, threading.Event] = {}
        self._inflight_lock = threading.Lock()

        # Conversation memory
        self.conversation = ConversationMemory(
            max_size=config.CONVERSATION_MEMORY_SIZE,
            ttl=config.CONVERSATION_TTL,
        )

    # ------------------------------------------------------------- lifecycle
    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def rebuilding(self) -> bool:
        with self._state:
            return self._rebuilding

    def initialize(self) -> None:
        """Connect to Qdrant and build the chain. Safe to call repeatedly."""
        with self._index_lock:
            if self._initialized:
                return

            logger.info("Initializing RAG system with Qdrant...")
            try:
                if self.embeddings is None:
                    self.embeddings = get_embeddings()

                self.vector_manager = QdrantVectorStoreManager(
                    embeddings=self.embeddings,
                    collection_name=config.QDRANT_COLLECTION,
                    qdrant_url=config.QDRANT_URL,
                    qdrant_api_key=config.QDRANT_API_KEY,
                )
                self._refresh_chain()
            except Exception as exc:
                self.init_error = str(exc)
                logger.error("RAG initialization failed: %s", exc)
                raise

            self._initialized = True
            self.init_error = None
            logger.info("RAG system initialized")

    # ------------------------------------------------------------------- LLM
    def _build_llm(self, model: str, streaming: bool) -> ChatOpenAI:
        if config.LLM_PROVIDER == "openrouter":
            api_key = config.OPENROUTER_API_KEY
            base_url = config.OPENROUTER_BASE_URL
            extra_headers = {
                "HTTP-Referer": "https://github.com/PrateekT1987/whatsapp_ragbot",
                "X-Title": "RAG WhatsApp Bot",
            }
        else:
            api_key = config.OPENAI_API_KEY
            base_url = None
            extra_headers = None

        return ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers,
            timeout=config.LLM_TIMEOUT,
            # Retries are handled explicitly below so we can send a "busy"
            # notice and switch to the fallback model.
            max_retries=0,
            streaming=streaming,
        )

    def _get_streaming_llm(self) -> ChatOpenAI:
        """Get or create a streaming LLM instance."""
        if self._streaming_llm is None:
            self._streaming_llm = self._build_llm(
                config.LLM_MODEL, streaming=True
            )
        return self._streaming_llm

    def _refresh_chain(self) -> None:
        """Rebuild retriever + QA chain, clamping k to the index size."""
        if self.vector_manager is None:
            return

        try:
            count = self.vector_manager.count()
        except Exception:  # noqa: BLE001
            count = config.TOP_K_RESULTS

        k = max(1, min(config.TOP_K_RESULTS, count)) if count else 1
        self.retriever = self.vector_manager.get_retriever(k=k)
        self.qa_chain = self._create_qa_chain(self.llm)
        self._fallback_chain = (
            self._create_qa_chain(self._fallback_llm)
            if self._fallback_llm is not None
            else None
        )

    def _create_qa_chain(self, llm: ChatOpenAI) -> Any:
        """Create a RAG chain; retrieval is done explicitly in `_answer`."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        return prompt | llm | StrOutputParser()

    def _create_streaming_chain(self) -> Any:
        """Create a streaming chain for token-by-token output."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        return prompt | self._get_streaming_llm() | StrOutputParser()

    # --------------------------------------------------------------- indexing
    def load_documents(self, force_reload: bool = False) -> int:
        """Load all PDF documents from the documents folder."""
        if not self._initialized or self.vector_manager is None:
            logger.warning("RAG not initialized; skipping document load")
            return 0
        with self._index_lock:
            return self._load_documents_locked(force_reload)

    def _load_documents_locked(self, force_reload: bool = False) -> int:
        if self.vector_manager is None:
            return 0

        if not config.DOCUMENTS_PATH.exists():
            logger.warning(f"Documents path {config.DOCUMENTS_PATH} does not exist")
            return 0

        pdf_files = list(config.DOCUMENTS_PATH.glob("*.pdf"))

        if not force_reload:
            indexed = {str(p) for p in self.vector_manager.get_processed_files()}
            present = {str(f) for f in pdf_files}
            removed = indexed - present
            for removed_path in removed:
                pdf = Path(removed_path)
                logger.info("PDF removed: %s; dropping its chunks", pdf.name)
                try:
                    self.vector_manager.delete_file(pdf)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not remove chunks for %s: %s", pdf.name, exc)
                self.vector_manager.remove_processed_file(pdf)

            # A changed PDF must have its old chunks removed before re-adding,
            # otherwise the index keeps both the stale and the new text.
            for pdf_file in pdf_files:
                if self.vector_manager.has_record(pdf_file) and not self.vector_manager.is_processed(pdf_file):
                    logger.info("PDF changed: %s; replacing its chunks", pdf_file.name)
                    try:
                        self.vector_manager.delete_file(pdf_file)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not remove old chunks for %s: %s", pdf_file.name, exc)
                    self.vector_manager.remove_processed_file(pdf_file)

        if not pdf_files:
            logger.info("No PDF files found in documents folder")
            return 0

        new_files = [f for f in pdf_files if not self.vector_manager.is_processed(f) or force_reload]

        if not new_files and not force_reload:
            logger.info("No new files to process")
            return 0

        logger.info(f"Processing {len(new_files)} PDF file(s)...")

        all_documents = []
        loaded_files = []
        for pdf_file in new_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                # Add metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["file_path"] = str(pdf_file)

                all_documents.extend(documents)
                loaded_files.append(pdf_file)
                logger.info(f"Loaded {len(documents)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")

        if not all_documents:
            return 0

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        splits = text_splitter.split_documents(all_documents)
        logger.info(f"Split into {len(splits)} chunks")

        # Add to vector store first: only files whose chunks were stored are
        # recorded in the manifest, so a failed add is retried next time.
        self.vector_manager.add_documents(splits)
        for pdf_file in loaded_files:
            self.vector_manager.add_processed_file(pdf_file)

        # Also extract structured data if enabled
        if config.ENABLE_TABLE_EXTRACTION:
            for pdf_file in loaded_files:
                try:
                    get_table_extractor().extract_from_pdf(pdf_file, force_reload=force_reload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Table extraction failed for %s: %s", pdf_file.name, exc)

        self._refresh_chain()
        logger.info("Documents added to vector store")

        return len(loaded_files)

    def rebuild_index(self) -> int:
        """Rebuild the entire index from scratch.

        Queries in flight are allowed to finish; new queries receive a
        "reindexing" reply instead of hitting a deleted collection.
        """
        if not self._initialized or self.vector_manager is None:
            raise RuntimeError("RAG system is not initialized")

        with self._state:
            if self._rebuilding:
                logger.info("Rebuild already in progress; ignoring duplicate request")
                return 0
            self._rebuilding = True
            while self._active_queries > 0:
                self._state.wait(timeout=1.0)

        try:
            with self._index_lock:
                logger.info("Rebuilding index...")

                # Answers generated from the old index must not survive a rebuild.
                query_cache.clear()

                try:
                    self.vector_manager.delete_collection()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not delete vector store collection: %s", exc)

                self.vector_manager = QdrantVectorStoreManager(
                    embeddings=self.embeddings,
                    collection_name=config.QDRANT_COLLECTION,
                    qdrant_url=config.QDRANT_URL,
                    qdrant_api_key=config.QDRANT_API_KEY,
                )
                self._refresh_chain()
                self.vector_manager.clear_processed_files()

                # Reload all documents
                return self.load_documents(force_reload=True)
        finally:
            with self._state:
                self._rebuilding = False
                self._state.notify_all()

    # ---------------------------------------------------------------- queries
    def _begin_query(self) -> bool:
        with self._state:
            if self._rebuilding:
                return False
            self._active_queries += 1
            return True

    def _end_query(self) -> None:
        with self._state:
            self._active_queries = max(0, self._active_queries - 1)
            self._state.notify_all()

    def query(
        self,
        question: str,
        chat_id: str = "",
        on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    ) -> str:
        """Query the RAG system (non-streaming, with cache and retries)."""
        if not self._initialized or self.qa_chain is None:
            return NOT_READY_MESSAGE

        cached = query_cache.get(question)
        if cached is not None:
            return cached

        key = query_cache.make_key(question)
        with self._inflight_lock:
            event = self._inflight.get(key)
            owner = event is None
            if owner:
                event = threading.Event()
                self._inflight[key] = event

        if not owner:
            # Another request is already answering this exact question.
            event.wait(timeout=config.LLM_TIMEOUT + 30)
            cached = query_cache.get(question)
            return cached or "⏳ That answer is still being generated. Please try again."

        if not self._begin_query():
            with self._inflight_lock:
                self._inflight.pop(key, None)
            event.set()
            return REINDEXING_MESSAGE

        try:
            response = self._answer(question, chat_id, on_retry)
            query_cache.set(question, response)
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error querying RAG")
            return self._friendly_error(exc)
        finally:
            with self._inflight_lock:
                self._inflight.pop(key, None)
            event.set()
            self._end_query()

    def _answer(
        self,
        question: str,
        chat_id: str,
        on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    ) -> str:
        # First, try structured query if it looks like one
        if config.ENABLE_TABLE_EXTRACTION and _looks_like_structured_query(question):
            structured_result = self._try_structured_query(question)
            if structured_result:
                return structured_result

        # Get conversation context
        context_history = self.conversation.get_context(chat_id) if chat_id else []
        context_str = ""
        if context_history:
            context_str = "\n".join(
                f"Previous Q: {q}\nPrevious A: {a}" for q, a in context_history
            )

        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Add conversation context to the prompt
        if context_str:
            context = f"Conversation history:\n{context_str}\n\n---\n\nDocument context:\n{context}"

        response = self._invoke_with_retries(
            self.qa_chain,
            {"context": context, "question": question},
            on_retry,
        )
        response = (response or "").strip()

        # Store in conversation memory
        if chat_id:
            self.conversation.add(chat_id, question, response)

        if config.SHOW_SOURCES:
            response += _format_sources(docs)
        return response

    def _try_structured_query(self, question: str) -> Optional[str]:
        """Try to answer using structured SQLite data."""
        try:
            q_lower = question.lower()
            extractor = get_table_extractor()

            # Count queries
            if any(kw in q_lower for kw in ("how many", "count", "number of", "total")):
                district = _extract_district(question)
                if district:
                    count = extractor.count_by_district(district)
                    return f"There are **{count}** member(s) from **{district}** district.\n\n📊 Source: Structured data (SQLite)"
                # General count
                stats = extractor.get_stats()
                return f"Total members in directory: **{stats['total_rows']}**.\n\n📊 Source: Structured data (SQLite)"

            # Search by name
            name = _extract_name(question)
            if name:
                rows = extractor.search_by_name(name)
                if rows:
                    return _format_structured_results(rows) + "\n\n📊 Source: Structured data (SQLite)"

            # Search by mobile
            mobile = _extract_mobile(question)
            if mobile:
                row = extractor.search_by_mobile(mobile)
                if row:
                    return _format_structured_results([row]) + "\n\n📊 Source: Structured data (SQLite)"

            # District query
            district = _extract_district(question)
            if district:
                rows = extractor.query(
                    "SELECT s_no, name, mobile, address, district, pin FROM members WHERE district LIKE ? ORDER BY name LIMIT 20",
                    (f"%{district}%",)
                )
                if rows:
                    return _format_structured_results(rows) + "\n\n📊 Source: Structured data (SQLite)"

        except Exception as exc:  # noqa: BLE001
            logger.warning("Structured query failed: %s", exc)

        return None

    def _invoke_with_retries(
        self,
        chain: Any,
        inputs: dict,
        on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    ) -> str:
        """Invoke a chain with bounded retries and an optional fallback model."""
        attempts = max(0, config.LLM_MAX_RETRIES) + 1
        last_exc: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                return chain.invoke(inputs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 >= attempts or not _is_retryable(exc):
                    break
                delay = _retry_delay(attempt, config.LLM_RETRY_BASE_SECONDS)
                logger.warning(
                    "LLM call failed (%s); retry %s/%s in %.1fs",
                    exc, attempt + 1, attempts - 1, delay,
                )
                LLM_RETRIES.inc()
                if on_retry is not None:
                    on_retry(attempt + 1, delay, exc)
                time.sleep(delay)

        if self._fallback_chain is not None and last_exc is not None:
            logger.warning(
                "Primary model failed; trying fallback %s", config.LLM_FALLBACK_MODEL
            )
            for attempt in range(attempts):
                try:
                    return self._fallback_chain.invoke(inputs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt + 1 >= attempts or not _is_retryable(exc):
                        break
                    delay = _retry_delay(attempt, config.LLM_RETRY_BASE_SECONDS)
                    LLM_RETRIES.inc()
                    if on_retry is not None:
                        on_retry(attempt + 1, delay, exc)
                    time.sleep(delay)

        if last_exc is not None:
            LLM_ERRORS.labels(
                kind="rate_limit" if _is_rate_limit(last_exc) else "other"
            ).inc()
            raise last_exc
        raise RuntimeError("LLM call failed")

    def _friendly_error(self, exc: Exception) -> str:
        if _is_rate_limit(exc):
            return "⏳ The model is rate-limited right now. Please try again in a minute."
        return "❌ Sorry, I couldn't answer that right now. Please try again."

    async def query_streaming(
        self,
        question: str,
        chat_id: str,
        send_chunk_callback,
        on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    ) -> str:
        """Query with streaming, sending coalesced bubbles to WhatsApp."""
        if not self._initialized or self.qa_chain is None:
            await send_chunk_callback(chat_id, NOT_READY_MESSAGE)
            return NOT_READY_MESSAGE

        cached = query_cache.get(question)
        if cached is not None:
            await send_chunk_callback(chat_id, cached)
            return cached

        if not self._begin_query():
            await send_chunk_callback(chat_id, REINDEXING_MESSAGE)
            return REINDEXING_MESSAGE

        started = time.perf_counter()
        full: List[str] = []
        try:
            chain = self._create_streaming_chain()

            # Coalesce tokens into readable bubbles instead of one per token.
            flush_at = 160
            delay = max(0, config.STREAM_CHUNK_DELAY_MS) / 1000.0

            docs = await self.retriever.ainvoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            pending: List[str] = []

            async def flush() -> None:
                if pending:
                    await send_chunk_callback(chat_id, "".join(pending))
                    pending.clear()
                    if delay:
                        await asyncio.sleep(delay)

            async for chunk in chain.astream({"context": context, "question": question}):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not text:
                    continue
                full.append(text)
                pending.append(text)
                # Flush on size, paragraph break or sentence end.
                if (
                    sum(len(t) for t in pending) >= flush_at
                    or "\n" in text
                    or (len(pending) > 1 and text.rstrip().endswith((".", "!", "?")))
                ):
                    await flush()

            await flush()
            response = "".join(full).strip()
            if config.SHOW_SOURCES:
                sources = _format_sources(docs)
                if sources:
                    response += sources
                    await send_chunk_callback(chat_id, sources.strip())

            # Store in conversation memory
            if chat_id:
                self.conversation.add(chat_id, question, response)

            # Cache the full response
            query_cache.set(question, response)
            QUERY_RESULTS.labels(result="success").inc()
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in streaming query")
            QUERY_RESULTS.labels(result="error").inc()
            if full:
                # Part of the answer already reached the user; do not repeat it.
                notice = "⚠️ The answer was cut off. Please ask again."
                await send_chunk_callback(chat_id, notice)
                return "".join(full)
            # Nothing was sent: use the non-streaming path (retries + fallback).
            try:
                response = await asyncio.to_thread(self._answer, question, chat_id, on_retry)
                await send_chunk_callback(chat_id, response)
                query_cache.set(question, response)
                return response
            except Exception as exc2:  # noqa: BLE001
                message = self._friendly_error(exc2)
                await send_chunk_callback(chat_id, message)
                return message
        finally:
            QUERY_DURATION.observe(time.perf_counter() - started)
            self._end_query()

    def get_stats(self) -> dict:
        """Get statistics about the current index."""
        cache_stats = query_cache.stats()
        if not self._initialized or self.vector_manager is None:
            return {
                "error": self.init_error or "RAG system is not initialized",
                "cache": cache_stats,
                "rebuilding": self.rebuilding,
            }
        try:
            count = self.vector_manager.count()
            structured_stats = get_table_extractor().get_stats() if config.ENABLE_TABLE_EXTRACTION else {}
            return {
                "document_count": count,
                "processed_files": len(self.vector_manager.get_processed_files()),
                "files": [f.name for f in self.vector_manager.get_processed_files()],
                "vector_store": "qdrant",
                "cache": cache_stats,
                "rebuilding": self.rebuilding,
                "structured": structured_stats,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e), "cache": cache_stats, "rebuilding": self.rebuilding}


# Global instance
rag_system = RAGSystem()