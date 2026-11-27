import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


class Config:
    # ---------------------------------------------------------------- WAHA
    WAHA_BASE_URL: str = os.getenv("WAHA_BASE_URL", "http://localhost:3000")
    WAHA_API_KEY: str = os.getenv("WAHA_API_KEY", "")
    WAHA_SESSION: str = os.getenv("WAHA_SESSION", "default")
    WAHA_HMAC_KEY: str = os.getenv("WAHA_HMAC_KEY", "")
    # Only answer in the "Message Yourself" chat (safe default)
    WAHA_SELF_CHAT_ONLY: bool = _env_bool("WAHA_SELF_CHAT_ONLY", "true")
    # Bounded retries for transient WAHA failures (429/5xx/connection).
    WAHA_SEND_RETRIES: int = int(os.getenv("WAHA_SEND_RETRIES", "2"))
    WAHA_TIMEOUT: float = float(os.getenv("WAHA_TIMEOUT", "30"))

    # ---------------------------------------------------------------- LLM
    # "openrouter" (default here) or "openai"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    # Optional second :free model used when the primary stays rate-limited.
    LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "")
    # Streaming: send tokens to WhatsApp as they arrive (requires OpenRouter/OpenAI streaming support)
    LLM_STREAMING: bool = _env_bool("LLM_STREAMING", "false")
    # Retries for transient LLM failures (429, 5xx, timeouts).
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_BASE_SECONDS: float = float(os.getenv("LLM_RETRY_BASE_SECONDS", "2.0"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))
    # Append "Source: file, page N" to answers.
    SHOW_SOURCES: bool = _env_bool("SHOW_SOURCES", "true")

    # ---------------------------------------------------------- Embeddings
    # "local" (bundled ONNX MiniLM, free, no API key) or "openai" or "multilingual"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # Multilingual model (intfloat/multilingual-e5-small, 384-dim)
    MULTILINGUAL_MODEL: str = os.getenv(
        "MULTILINGUAL_MODEL", "intfloat/multilingual-e5-small"
    )

    # ---------------------------------------------------------- Vector Store (Qdrant only)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "rag_documents")

    # ---------------------------------------------------------- Structured Data (SQLite)
    # Enable table extraction from PDFs
    ENABLE_TABLE_EXTRACTION: bool = _env_bool("ENABLE_TABLE_EXTRACTION", "true")
    # SQLite database path
    STRUCTURED_DB_PATH: Path = Path(
        os.getenv("STRUCTURED_DB_PATH", "./cache/structured_data.sqlite")
    )

    # ---------------------------------------------------------- Caching
    # Enable semantic cache for query responses
    CACHE_ENABLED: bool = _env_bool("CACHE_ENABLED", "true")
    # Cache TTL in seconds (default: 1 hour)
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    # Cache directory (diskcache)
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./cache"))
    # Exact-match cache for frequent questions (in-memory, very fast)
    EXACT_CACHE_SIZE: int = int(os.getenv("EXACT_CACHE_SIZE", "1000"))
    # Semantic cache: embedding-based paraphrase matching
    SEMANTIC_CACHE_ENABLED: bool = _env_bool("SEMANTIC_CACHE_ENABLED", "false")
    SEMANTIC_CACHE_THRESHOLD: float = float(
        os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.85")
    )
    SEMANTIC_CACHE_MAX_ENTRIES: int = int(
        os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "500")
    )

    # ------------------------------------------------------------- Server
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    # Development only: restart on .py changes. Never enable in production.
    # `RELOAD` is accepted as an alias for older deployments.
    UVICORN_RELOAD: bool = _env_bool("UVICORN_RELOAD", os.getenv("RELOAD", "false"))

    # ----------------------------------------------------------- Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    # Question text is private by default: log only its length at INFO.
    LOG_QUESTION_TEXT: bool = _env_bool("LOG_QUESTION_TEXT", "false")

    # -------------------------------------------------------- Concurrency
    # At most this many LLM calls run at once across all chats.
    MAX_CONCURRENT_QUERIES: int = int(os.getenv("MAX_CONCURRENT_QUERIES", "2"))
    # Messages waiting per chat (keeps replies ordered, memory bounded).
    MAX_QUEUED_PER_CHAT: int = int(os.getenv("MAX_QUEUED_PER_CHAT", "5"))
    # Delay between streamed bubbles (WhatsApp spam/flood protection).
    STREAM_CHUNK_DELAY_MS: int = int(os.getenv("STREAM_CHUNK_DELAY_MS", "400"))
    # First retry delay for RAG/Qdrant initialization (doubles up to 60s).
    STARTUP_RETRY_SECONDS: int = int(os.getenv("STARTUP_RETRY_SECONDS", "2"))

    # ------------------------------------------------------ Conversation
    # Number of previous Q&A pairs to keep per chat for context
    CONVERSATION_MEMORY_SIZE: int = int(os.getenv("CONVERSATION_MEMORY_SIZE", "3"))
    # Max age of conversation history in seconds
    CONVERSATION_TTL: int = int(os.getenv("CONVERSATION_TTL", "3600"))

    # ---------------------------------------------------- Feedback
    # Enable /good /bad feedback commands
    FEEDBACK_ENABLED: bool = _env_bool("FEEDBACK_ENABLED", "true")
    # Path to store feedback
    FEEDBACK_DB_PATH: Path = Path(
        os.getenv("FEEDBACK_DB_PATH", "./cache/feedback.sqlite")
    )

    # -------------------------------------------------- Voice Notes
    # Enable local transcription of WhatsApp voice notes.
    WHISPER_ENABLED: bool = _env_bool("WHISPER_ENABLED", "true")
    # tiny, base, small, medium, large (tiny is prefetched into the image).
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "")  # empty = auto-detect

    # ----------------------------------------------------------------- RAG
    DOCUMENTS_PATH: Path = Path(os.getenv("DOCUMENTS_PATH", "./documents"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))

    SYSTEM_PROMPT: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful assistant that answers questions based on the provided documents. "
        "If the answer is not in the documents, say you don't know.",
    )

    def validate(self) -> list[str]:
        """Return list of missing required configuration."""
        missing: list[str] = []

        if self.LLM_PROVIDER == "openrouter" and not self.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        elif self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        elif self.LLM_PROVIDER not in ("openrouter", "openai"):
            missing.append("LLM_PROVIDER (must be 'openrouter' or 'openai')")

        if self.EMBEDDING_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY (required for EMBEDDING_PROVIDER=openai)")

        if not self.QDRANT_URL:
            missing.append("QDRANT_URL")

        return missing


config = Config()
