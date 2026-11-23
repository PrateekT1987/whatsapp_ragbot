#!/usr/bin/env python3
"""
Test script for the RAG system.
Run this to verify the RAG system works before starting the bot.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config
from rag import RAGSystem


def main():
    print("🧪 Testing RAG System")
    print("=" * 40)
    
    # Check config
    missing = config.validate()
    if missing:
        print("❌ Missing configuration:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease set up .env file first")
        return 1
    
    # Check documents
    if not config.DOCUMENTS_PATH.exists():
        print(f"📁 Creating documents folder: {config.DOCUMENTS_PATH}")
        config.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(config.DOCUMENTS_PATH.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found in documents folder")
        print(f"   Add some PDF files to {config.DOCUMENTS_PATH}")
        return 1
    
    print(f"📄 Found {len(pdf_files)} PDF file(s)")
    for f in pdf_files:
        print(f"   - {f.name}")
    
    # Initialize RAG
    print("\n🔧 Initializing RAG system...")
    rag = RAGSystem()
    rag.initialize()
    
    # Load documents
    print("📥 Loading documents...")
    count = rag.load_documents()
    print(f"✅ Loaded {count} document(s)")
    
    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 Index Stats:")
    print(f"   Chunks: {stats.get('document_count', 0)}")
    print(f"   Files: {stats.get('processed_files', 0)}")
    
    # Test query
    print("\n💬 Testing query...")
    test_question = "What is this document about?"
    print(f"Q: {test_question}")
    response = rag.query(test_question)
    print(f"A: {response[:500]}...")
    
    print("\n✅ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())