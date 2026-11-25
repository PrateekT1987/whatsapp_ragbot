"""Qdrant vector store manager."""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


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

        self.vectorstore = self._create_qdrant(embeddings, collection_name, qdrant_url, qdrant_api_key)
        self._processed_files: set = set()

    def _create_qdrant(
        self,
        embeddings: Embeddings,
        collection_name: str,
        qdrant_url: str,
        qdrant_api_key: str,
    ) -> VectorStore:
        """Create Qdrant vector store."""
        # langchain-qdrant 0.1.x exports the store as `Qdrant`
        # (`QdrantVectorStore` is the name used from 0.2 onwards).
        from langchain_qdrant import Qdrant as QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)

        # Ensure collection exists
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            logger.info(f"Creating Qdrant collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )

    def get_retriever(self, k: int = 4):
        """Get a retriever with the specified k."""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vectorstore.add_documents(documents)

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        from qdrant_client import QdrantClient
        client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key or None)
        client.delete_collection(collection_name=self.collection_name)

    def count(self) -> int:
        """Get the number of documents in the store."""
        from qdrant_client import QdrantClient
        client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key or None)
        info = client.get_collection(collection_name=self.collection_name)
        return info.points_count

    def clear_processed_files(self) -> None:
        """Clear the set of processed files."""
        self._processed_files.clear()

    def add_processed_file(self, file_path) -> None:
        """Add a file to the processed set."""
        self._processed_files.add(file_path)

    def is_processed(self, file_path) -> bool:
        """Check if a file has been processed."""
        return file_path in self._processed_files

    def get_processed_files(self) -> set:
        """Get the set of processed files."""
        return self._processed_files