"""Embedding backends.

OpenRouter serves chat completions only - it has no embeddings endpoint, so the
default backend is a fully local ONNX model that ChromaDB already ships with.
That keeps the bot free to run and needs no API key.

Set EMBEDDING_PROVIDER=openai to use OpenAI embeddings instead (requires
OPENAI_API_KEY).
"""
import logging
from typing import List

from langchain_core.embeddings import Embeddings

from config import config

logger = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    """LangChain adapter around Chroma's bundled ONNX MiniLM embedder.

    Downloads the model (~80 MB) once on first use, then runs offline on CPU.
    """

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()
        logger.info("Using local ONNX MiniLM embeddings (all-MiniLM-L6-v2)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, vec)) for vec in self._fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(map(float, self._fn([text])[0]))


def get_embeddings() -> Embeddings:
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "local":
        return LocalEmbeddings()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI embeddings (%s)", config.EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={config.EMBEDDING_PROVIDER!r}; "
        "use 'local' or 'openai'"
    )
