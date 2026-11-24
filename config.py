import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ---------------------------------------------------------------- WAHA
    WAHA_BASE_URL: str = os.getenv("WAHA_BASE_URL", "http://localhost:3000")
    WAHA_API_KEY: str = os.getenv("WAHA_API_KEY", "")
    WAHA_SESSION: str = os.getenv("WAHA_SESSION", "default")
    WAHA_HMAC_KEY: str = os.getenv("WAHA_HMAC_KEY", "")
    # Only answer in the "Message Yourself" chat (safe default)
    WAHA_SELF_CHAT_ONLY: bool = os.getenv("WAHA_SELF_CHAT_ONLY", "true").lower() in (
        "1", "true", "yes", "on"
    )

    # ---------------------------------------------------------------- LLM
    # "openrouter" (default here) or "openai"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

    # ---------------------------------------------------------- Embeddings
    # "local" (bundled ONNX MiniLM, free, no API key) or "openai"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ------------------------------------------------------------- Server
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # ----------------------------------------------------------------- RAG
    DOCUMENTS_PATH: Path = Path(os.getenv("DOCUMENTS_PATH", "./documents"))
    CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
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

        return missing


config = Config()
