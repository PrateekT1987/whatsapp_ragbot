"""Vector store abstraction supporting Chroma and Qdrant."""

import logging
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create(
        store_type: str,
        embeddings: Embeddings,
        persist_dir: Optional[Path] = None,
        collection_name: str = "rag_documents",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "",
    ) -> VectorStore:
        """Create a vector store instance based on configuration."""
        if store_type == "qdrant":
            return VectorStoreFactory._create_qdrant(
                embeddings, collection_name, qdrant_url, qdrant_api_key
            )
        else:
            return VectorStoreFactory._create_chroma(embeddings, persist_dir)

    @staticmethod
    def _create_chroma(
        embeddings: Embeddings, persist_dir: Optional[Path] = None
    ) -> VectorStore:
        """Create Chroma vector store."""
        from langchain_chroma import Chroma
        from chromadb.config import Settings

        if persist_dir is None:
            persist_dir = Path("./chroma_db")
        persist_dir.mkdir(parents=True, exist_ok=True)

        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )

    @staticmethod
    def _create_qdrant(
        embeddings: Embeddings,
        collection_name: str,
        qdrant_url: str,
        qdrant_api_key: str,
    ) -> VectorStore:
        """Create Qdrant vector store."""
        from langchain_qdrant import QdrantVectorStore
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
            embedding=embeddings,
        )


class VectorStoreManager:
    """Manages vector store operations with a unified interface."""

    def __init__(
        self,
        store_type: str,
        embeddings: Embeddings,
        persist_dir: Optional[Path] = None,
        collection_name: str = "rag_documents",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "",
    ):
        self.store_type = store_type
        self.collection_name = collection_name
        self.vectorstore = VectorStoreFactory.create(
            store_type, embeddings, persist_dir, collection_name, qdrant_url, qdrant_api_key
        )
        self._processed_files: set = set()

    def get_retriever(self, k: int = 4):
        """Get a retriever with the specified k."""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vectorstore.add_documents(documents)

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if self.store_type == "qdrant":
            from qdrant_client import QdrantClient
            client = QdrantClient(url=self._get_qdrant_url(), api_key=self._get_qdrant_api_key())
            client.delete_collection(collection_name=self.collection_name)
        else:
            self.vectorstore.delete_collection()

    def _get_qdrant_url(self) -> str:
        from config import config
        return config.QDRANT_URL

    def _get_qdrant_api_key(self) -> str:
        from config import config
        return config.QDRANT_API_KEY

    def count(self) -> int:
        """Get the number of documents in the store."""
        if self.store_type == "qdrant":
            from qdrant_client import QdrantClient
            client = QdrantClient(url=self._get_qdrant_url(), api_key=self._get_qdrant_api_key())
            info = client.get_collection(collection_name=self.collection_name)
            return info.points_count
        else:
            return self.vectorstore._collection.count()

    def clear_processed_files(self) -> None:
        """Clear the set of processed files."""
        self._processed_files.clear()

    def add_processed_file(self, file_path: Path) -> None:
        """Add a file to the processed set."""
        self._processed_files.add(file_path)

    def is_processed(self, file_path: Path) -> bool:
        """Check if a file has been processed."""
        return file_path in self._processed_files

    def get_processed_files(self) -> set:
        """Get the set of processed files."""
        return self._processed_files