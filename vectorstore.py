"""Qdrant vector store manager with hybrid (vector + keyword) retrieval."""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from config import config

logger = logging.getLogger(__name__)

# Words that carry no retrieval signal on their own.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "what",
    "who",
    "whom",
    "whose",
    "which",
    "when",
    "where",
    "why",
    "how",
    "give",
    "gives",
    "tell",
    "show",
    "list",
    "find",
    "get",
    "me",
    "my",
    "about",
    "details",
    "detail",
    "information",
    "info",
    "please",
    "can",
    "could",
    "would",
    "should",
    "any",
    "all",
    "this",
    "that",
    "these",
    "those",
    "there",
    "their",
    "has",
    "have",
    "had",
    "it",
    "its",
}


def _normalize(text: str) -> str:
    """Lower-case and drop everything that is not a letter or digit.

    PDF text extraction glues serial numbers to names ("15Abhishek.J.Patil"),
    so substring matching on a punctuation-free string is far more reliable
    than token matching.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _query_terms(query: str) -> List[str]:
    terms: List[str] = []
    for raw in re.split(r"[^A-Za-z0-9]+", query.lower()):
        if len(raw) >= 3 and raw not in _STOPWORDS and raw not in terms:
            terms.append(raw)
    return terms


class HybridRetriever(BaseRetriever):
    """Retriever that merges keyword matches with vector similarity."""

    manager: Any
    k: int = 4

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        return self.manager.search(query, k=self.k)


class QdrantVectorStoreManager:
    """Manages Qdrant vector store operations with a unified interface."""

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "rag_documents",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "",
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.embeddings = embeddings

        self.vectorstore = self._create_qdrant(
            embeddings, collection_name, qdrant_url, qdrant_api_key
        )
        self._all_docs: Optional[List[Document]] = None

        # Persist which files are already indexed (path -> mtime). Without this
        # every restart re-added all chunks, because langchain-qdrant generates
        # a new id per insert.
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._manifest_path = config.CACHE_DIR / f"indexed_{collection_name}.json"
        self._processed_files: Dict[str, float] = self._read_manifest()

    # ------------------------------------------------------------- internals
    def _client(self):
        from qdrant_client import QdrantClient

        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key or None)

    def _create_qdrant(
        self,
        embeddings: Embeddings,
        collection_name: str,
        qdrant_url: str,
        qdrant_api_key: str,
    ) -> VectorStore:
        """Create Qdrant vector store, validating the embedding dimension."""
        # langchain-qdrant 0.1.x exports the store as `Qdrant`
        # (`QdrantVectorStore` is the name used from 0.2 onwards).
        from langchain_qdrant import Qdrant as QdrantVectorStore
        from qdrant_client.http.models import Distance, VectorParams

        client = self._client()

        # Ask the embedder instead of hardcoding 384: switching to OpenAI
        # embeddings must not silently create a mismatched collection.
        dimension = len(embeddings.embed_query("dimension probe"))

        # Ensure collection exists
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        if not exists:
            logger.info(
                "Creating Qdrant collection: %s (%s-dim)", collection_name, dimension
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        else:
            info = client.get_collection(collection_name)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                sizes = {v.size for v in vectors.values()}
                existing = sizes.pop() if len(sizes) == 1 else None
            else:
                existing = getattr(vectors, "size", None)
            if existing is not None and existing != dimension:
                raise ValueError(
                    f"Qdrant collection '{collection_name}' stores {existing}-dim "
                    f"vectors but the configured embedder produces {dimension}-dim. "
                    "Delete that collection (Qdrant dashboard) or set "
                    "QDRANT_COLLECTION to a new name, then restart."
                )

        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )

    def _load_all_documents(self) -> List[Document]:
        """Load every stored chunk once, for lexical matching."""
        if self._all_docs is not None:
            return self._all_docs

        client = self._client()
        docs: List[Document] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                content = payload.get("page_content")
                if not content:
                    continue
                docs.append(
                    Document(
                        page_content=content,
                        metadata=payload.get("metadata") or {},
                    )
                )
            if offset is None:
                break

        self._all_docs = docs
        logger.info("Loaded %s chunk(s) for keyword search", len(docs))
        return docs

    def _keyword_search(self, query: str, limit: int = 2) -> List[Document]:
        """Rank chunks by how many query terms they contain verbatim."""
        terms = _query_terms(query)
        if not terms:
            return []

        phrase = _normalize(" ".join(terms))
        docs = self._load_all_documents()
        required = 1 if len(terms) == 1 else 2

        scored: List[Tuple[int, int, Document]] = []
        for doc in docs:
            haystack = _normalize(doc.page_content)
            hits = sum(1 for term in terms if term in haystack)
            if hits < required:
                continue
            # A chunk containing the whole phrase beats one with scattered terms.
            phrase_hit = 1 if phrase and phrase in haystack else 0
            scored.append((phrase_hit, hits, doc))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [doc for _, _, doc in scored[:limit]]

    # -------------------------------------------------------------- retrieval
    def search(self, query: str, k: int = 4) -> List[Document]:
        """Hybrid search: keyword hits first, then vector similarity."""
        results: List[Document] = []
        seen: set = set()

        for doc in self._keyword_search(query):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                results.append(doc)

        try:
            for doc in self.vectorstore.similarity_search(query, k=k):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    results.append(doc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Vector search failed: %s", exc)

        return results

    def get_retriever(self, k: int = 4):
        """Get a hybrid retriever with the specified k."""
        return HybridRetriever(manager=self, k=k)

    # -------------------------------------------------------------- mutation
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Point ids are deterministic (file + chunk number), so re-running a
        partially completed indexing pass overwrites points instead of
        duplicating them after a crash.
        """
        counters: Dict[str, int] = {}
        ids: List[str] = []
        for doc in documents:
            source = str(
                doc.metadata.get("file_path") or doc.metadata.get("source") or ""
            )
            index = counters.get(source, 0)
            counters[source] = index + 1
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}")))
        self.vectorstore.add_documents(documents, ids=ids)
        self._all_docs = None

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._client().delete_collection(collection_name=self.collection_name)
        self._all_docs = None

    def delete_file(self, file_path) -> None:
        """Remove every chunk that came from one PDF."""
        from qdrant_client.http import models as qmodels

        selector = qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.file_path",
                        match=qmodels.MatchValue(value=str(file_path)),
                    )
                ]
            )
        )
        self._client().delete(
            collection_name=self.collection_name,
            points_selector=selector,
        )
        self._all_docs = None
        logger.info("Removed indexed chunks for %s", Path(file_path).name)

    def count(self) -> int:
        """Get the number of documents in the store."""
        info = self._client().get_collection(collection_name=self.collection_name)
        return info.points_count

    # ---------------------------------------------------------- file tracking
    def _read_manifest(self) -> Dict[str, float]:
        try:
            data = json.loads(self._manifest_path.read_text())
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read index manifest: %s", exc)
        return {}

    def _write_manifest(self) -> None:
        try:
            self._manifest_path.write_text(json.dumps(self._processed_files))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write index manifest: %s", exc)

    def clear_processed_files(self) -> None:
        """Forget which files are indexed (a rebuild is about to happen)."""
        self._processed_files.clear()
        try:
            self._manifest_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not remove index manifest: %s", exc)

    def add_processed_file(self, file_path) -> None:
        """Record a file as indexed, including its modification time."""
        path = Path(file_path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        self._processed_files[str(path)] = mtime
        self._write_manifest()

    def remove_processed_file(self, file_path) -> None:
        """Forget a file (used when it changed or was deleted)."""
        self._processed_files.pop(str(Path(file_path)), None)
        self._write_manifest()

    def has_record(self, file_path) -> bool:
        """True when the file is in the manifest, regardless of mtime."""
        return str(Path(file_path)) in self._processed_files

    def is_processed(self, file_path) -> bool:
        """True when the file is indexed and has not changed since."""
        path = Path(file_path)
        recorded = self._processed_files.get(str(path))
        if recorded is None:
            return False
        try:
            return abs(path.stat().st_mtime - recorded) < 1.0
        except OSError:
            return False

    def get_processed_files(self) -> List[Path]:
        """Files currently recorded as indexed."""
        return [Path(p) for p in self._processed_files]
