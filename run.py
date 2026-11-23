#!/usr/bin/env python3
"""
Entry point for running the RAG WhatsApp Bot.
This script validates configuration and starts the server.
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config


def check_config():
    """Check if all required configuration is present."""
    missing = config.validate()
    if missing:
        print("❌ Missing required configuration:")
        for item in missing:
            print(f"  - {item}")
        print("\n📝 Please copy .env.example to .env and fill in your credentials:")
        print("   cp .env.example .env")
        print("   # Then edit .env with your values")
        return False
    return True


def check_documents_folder():
    """Ensure documents folder exists."""
    if not config.DOCUMENTS_PATH.exists():
        config.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created documents folder: {config.DOCUMENTS_PATH}")
    else:
        pdf_count = len(list(config.DOCUMENTS_PATH.glob("*.pdf")))
        print(f"📁 Documents folder exists with {pdf_count} PDF file(s)")


def main():
    print("🤖 RAG WhatsApp Bot")
    print("=" * 40)
    
    if not check_config():
        sys.exit(1)
    
    check_documents_folder()
    
    print("\n🚀 Starting server...")
    print(f"   Host: {config.HOST}")
    print(f"   Port: {config.PORT}")
    print(f"   Webhook URL: {config.WEBHOOK_URL or 'Not set'}")
    print("\n📱 Configure Twilio webhook to: {config.WEBHOOK_URL}/webhook")
    print("   (Use ngrok or similar for local development)")
    print("\n" + "=" * 40)
    
    # Import and run uvicorn
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )


if __name__ == "__main__":
    main()