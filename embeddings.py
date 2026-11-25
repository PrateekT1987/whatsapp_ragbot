"""Embedding backends.

OpenRouter serves chat completions only - it has no embeddings endpoint, so the
default backend is a fully local ONNX model (fastembed). That keeps the bot free
to run and needs no API key.

Set EMBEDDING_PROVIDER=openai to use OpenAI embeddings instead (requires
OPENAI_API_KEY).
"""
import logging
import os
from typing import List

from langchain_core.embeddings import Embeddings

from config import config

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 is 384-dimensional, matching the Qdrant collection config.
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddings(Embeddings):
    """LangChain adapter around fastembed's ONNX MiniLM embedder.

    Downloads the model (~90 MB) once on first use, then runs offline on CPU.
    Set FASTEMBED_CACHE_PATH to keep the download across container recreates.
    """

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL)
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        logger.info("Using local ONNX embeddings (%s)", model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.query_embed(text))).tolist()


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
