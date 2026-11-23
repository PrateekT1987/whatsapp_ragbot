import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Twilio
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_NUMBER: str = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    
    # LLM Provider (OpenAI or OpenRouter)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "openrouter"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Webhook
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # RAG
    DOCUMENTS_PATH: Path = Path(os.getenv("DOCUMENTS_PATH", "./documents"))
    CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
    
    # System
    SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", 
        "You are a helpful assistant that answers questions based on the provided documents. "
        "If the answer is not in the documents, say you don't know."
    )
    
    def validate(self) -> list[str]:
        """Return list of missing required configuration."""
        missing = []
        if not self.TWILIO_ACCOUNT_SID:
            missing.append("TWILIO_ACCOUNT_SID")
        if not self.TWILIO_AUTH_TOKEN:
            missing.append("TWILIO_AUTH_TOKEN")
        # Accept either OpenAI or OpenRouter API key
        if self.LLM_PROVIDER == "openrouter" and not self.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        elif self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        return missing

config = Config()