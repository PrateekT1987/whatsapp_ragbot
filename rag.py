"""RAG system with Qdrant vector store, caching, and streaming support."""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import config
from embeddings import get_embeddings
from vectorstore import QdrantVectorStoreManager
from cache import query_cache

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        # Local ONNX embeddings by default - OpenRouter has no embeddings API.
        self.embeddings = get_embeddings()

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

        # Non-streaming LLM for regular queries
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0,
            openai_api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers,
            timeout=120,
            max_retries=2,
            streaming=False,
        )

        # Streaming LLM (created on demand)
        self._streaming_llm: Optional[ChatOpenAI] = None

        # Vector store manager (Qdrant only)
        self.vector_manager = QdrantVectorStoreManager(
            embeddings=self.embeddings,
            collection_name=config.QDRANT_COLLECTION,
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
        )

        self.retriever = None
        self.qa_chain = None

        # Serializes indexing/rebuilds between the watcher, the API and startup.
        self._index_lock = threading.RLock()

    def initialize(self) -> None:
        """Initialize the vector store and QA chain."""
        logger.info("Initializing RAG system with Qdrant...")
        self._refresh_chain()
        logger.info("RAG system initialized")

    def _get_streaming_llm(self) -> ChatOpenAI:
        """Get or create a streaming LLM instance."""
        if self._streaming_llm is None:
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

            self._streaming_llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=0,
                openai_api_key=api_key,
                base_url=base_url,
                default_headers=extra_headers,
                timeout=120,
                max_retries=2,
                streaming=True,
            )
        return self._streaming_llm

    def _refresh_chain(self) -> None:
        """Rebuild retriever + QA chain, clamping k to the index size."""
        try:
            count = self.vector_manager.count()
        except Exception:  # noqa: BLE001
            count = config.TOP_K_RESULTS

        k = max(1, min(config.TOP_K_RESULTS, count)) if count else 1
        self.retriever = self.vector_manager.get_retriever(k=k)
        self._create_qa_chain()

    def _create_qa_chain(self) -> None:
        """Create the RAG QA chain (non-streaming)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}")
        ])

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _create_streaming_chain(self) -> Any:
        """Create a streaming chain for token-by-token output.

        Retrieval happens in ``query_streaming`` so the retrieved context is
        computed once and can also be used for caching.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        return prompt | self._get_streaming_llm()

    def load_documents(self, force_reload: bool = False) -> int:
        """Load all PDF documents from the documents folder."""
        with self._index_lock:
            return self._load_documents_locked(force_reload)

    def _load_documents_locked(self, force_reload: bool = False) -> int:
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
        self._refresh_chain()
        logger.info("Documents added to vector store")

        return len(loaded_files)

    def rebuild_index(self) -> int:
        """Rebuild the entire index from scratch."""
        with self._index_lock:
            logger.info("Rebuilding index...")

            # Answers generated from the old index must not survive a rebuild.
            query_cache.clear()

            # Delete the collection
            try:
                self.vector_manager.delete_collection()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not delete vector store collection: %s", exc)

            # Reinitialize vector manager
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

    def query(self, question: str) -> str:
        """Query the RAG system (non-streaming, with caching)."""
        if not self.qa_chain:
            return "RAG system not initialized. Please add documents first."

        # Check cache first
        cached = query_cache.get(question)
        if cached:
            return cached

        try:
            response = self.qa_chain.invoke(question)
            # Cache the response
            query_cache.set(question, response)
            return response
        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"Error processing your question: {str(e)}"

    async def query_streaming(self, question: str, chat_id: str, send_chunk_callback) -> str:
        """Query the RAG system with streaming, sending chunks to WhatsApp as they arrive.
        
        Args:
            question: The user's question
            chat_id: WhatsApp chat ID to send chunks to
            send_chunk_callback: Async function to send a chunk (chat_id, chunk) -> bool
            
        Returns:
            Full response string
        """
        if not self.qa_chain:
            return "RAG system not initialized. Please add documents first."

        # Check cache first
        cached = query_cache.get(question)
        if cached:
            # Send cached response in chunks
            await self._send_cached_streaming(cached, chat_id, send_chunk_callback)
            return cached

        # Create streaming chain
        chain = self._create_streaming_chain()

        # Coalesce tokens into readable bubbles instead of one per token.
        flush_at = 160

        try:
            # Retrieve once, then stream the answer for that context.
            docs = await self.retriever.ainvoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            full_response: List[str] = []
            pending: List[str] = []

            async def flush() -> None:
                if pending:
                    await send_chunk_callback(chat_id, "".join(pending))
                    pending.clear()

            async for chunk in chain.astream({"context": context, "question": question}):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not text:
                    continue
                full_response.append(text)
                pending.append(text)
                # Flush on size, paragraph break or sentence end.
                if (
                    sum(len(t) for t in pending) >= flush_at
                    or "\n" in text
                    or (len(pending) > 1 and text.rstrip().endswith((".", "!", "?")))
                ):
                    await flush()

            await flush()
            response = "".join(full_response)
            
            # Cache the full response
            query_cache.set(question, response)
            
            return response
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            error_msg = f"Error processing your question: {str(e)}"
            await send_chunk_callback(chat_id, error_msg)
            return error_msg

    async def _send_cached_streaming(self, response: str, chat_id: str, send_chunk_callback) -> None:
        """Send a cached response in one go (no need to simulate streaming)."""
        await send_chunk_callback(chat_id, response)

    def get_stats(self) -> dict:
        """Get statistics about the current index."""
        try:
            count = self.vector_manager.count()
            return {
                "document_count": count,
                "processed_files": len(self.vector_manager.get_processed_files()),
                "files": [f.name for f in self.vector_manager.get_processed_files()],
                "vector_store": "qdrant",
                "cache": query_cache.stats(),
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Global instance
rag_system = RAGSystem()