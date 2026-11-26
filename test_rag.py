#!/usr/bin/env python3
"""Smoke test for the RAG pipeline (no WhatsApp/WAHA involved)."""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config
from rag import rag_system


def main() -> int:
    print("🧪 Testing RAG pipeline")
    print("=" * 40)

    missing = config.validate()
    if missing:
        print("❌ Missing configuration:")
        for item in missing:
            print(f"  - {item}")
        return 1

    if not config.DOCUMENTS_PATH.exists():
        config.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

    pdf_files = list(config.DOCUMENTS_PATH.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files in {config.DOCUMENTS_PATH}")
        return 1

    print(f"📄 {len(pdf_files)} PDF file(s)")
    print(f"🔎 Embeddings: {config.EMBEDDING_PROVIDER}")

    rag_system.initialize()

    print("📥 Indexing...")
    count = rag_system.load_documents()
    print(f"✅ Indexed {count} file(s)")

    stats = rag_system.get_stats()
    print(f"📊 Chunks: {stats.get('document_count', 0)} | Files: {stats.get('processed_files', 0)}")

    question = "What is this document about?"
    print(f"\n💬 Q: {question}")
    print(f"💬 A: {rag_system.query(question)[:500]}")

    print("\n✅ RAG pipeline OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
