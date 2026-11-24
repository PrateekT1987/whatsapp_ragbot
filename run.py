#!/usr/bin/env python3
"""Validate configuration and start the RAG WhatsApp bot."""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config


def check_config() -> bool:
    missing = config.validate()
    if missing:
        print("❌ Missing required configuration:")
        for item in missing:
            print(f"  - {item}")
        print("\n📝 Copy .env.example to .env and fill in the values:")
        print("   cp .env.example .env")
        return False
    return True


def check_documents_folder() -> None:
    if not config.DOCUMENTS_PATH.exists():
        config.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created documents folder: {config.DOCUMENTS_PATH}")
    else:
        pdf_count = len(list(config.DOCUMENTS_PATH.glob("*.pdf")))
        print(f"📁 Documents folder: {pdf_count} PDF file(s)")


def main() -> None:
    print("🤖 RAG WhatsApp Bot (WAHA)")
    print("=" * 40)

    if not check_config():
        sys.exit(1)

    check_documents_folder()

    print(f"\n🧠 Model:      {config.LLM_MODEL} ({config.LLM_PROVIDER})")
    print(f"🔎 Embeddings: {config.EMBEDDING_PROVIDER}")
    print(f"📡 WAHA:       {config.WAHA_BASE_URL} (session: {config.WAHA_SESSION})")
    print(f"🚀 Server:     http://{config.HOST}:{config.PORT}")

    print("\n📱 Pairing")
    print(f"   Dashboard: {config.WAHA_BASE_URL}/dashboard")
    print(f"   QR proxy:  http://localhost:{config.PORT}/qr")
    print("\n💬 First time: send any message to yourself in WhatsApp")
    print("   ('Message Yourself') and the bot will reply in the same chat.")
    print("\n" + "=" * 40)

    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)


if __name__ == "__main__":
    main()
