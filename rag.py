import os
from pathlib import Path
from typing import List, Optional
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import config
from embeddings import get_embeddings

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        # Local ONNX embeddings by default - OpenRouter has no embeddings API.
        self.embeddings = get_embeddings()

        if config.LLM_PROVIDER == "openrouter":
            api_key = config.OPENROUTER_API_KEY
            base_url = config.OPENROUTER_BASE_URL
            extra_headers = {
                "HTTP-Referer": "https://github.com/PrateekT1987/whatsapp_ragbot",
                "X-Title": "RAG WhatsApp Bot",
            }
        else:
            api_key = config.OPENAI_API_KEY
            base_url = None
            extra_headers = None

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0,
            openai_api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers,
            timeout=120,
            max_retries=2,
        )
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        self.qa_chain = None
        self._processed_files: set = set()

    @staticmethod
    def _chroma_settings():
        """Disable Chroma's anonymous telemetry (it also logs noisy warnings)."""
        import chromadb
        from chromadb.config import Settings

        return Settings(anonymized_telemetry=False)

    def initialize(self) -> None:
        """Initialize the vector store and QA chain."""
        logger.info("Initializing RAG system...")
        
        # Create persist directory if it doesn't exist
        config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load or create vector store
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_PERSIST_DIR),
            embedding_function=self.embeddings,
            client_settings=self._chroma_settings(),
        )
        
        self._refresh_chain()
        logger.info("RAG system initialized")

    def _refresh_chain(self) -> None:
        """Rebuild retriever + QA chain, clamping k to the index size."""
        try:
            count = self.vectorstore._collection.count()
        except Exception:  # noqa: BLE001
            count = config.TOP_K_RESULTS

        k = max(1, min(config.TOP_K_RESULTS, count)) if count else 1
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self._create_qa_chain()
    
    def _create_qa_chain(self) -> None:
        """Create the RAG QA chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def load_documents(self, force_reload: bool = False) -> int:
        """Load all PDF documents from the documents folder."""
        if not config.DOCUMENTS_PATH.exists():
            logger.warning(f"Documents path {config.DOCUMENTS_PATH} does not exist")
            return 0
        
        pdf_files = list(config.DOCUMENTS_PATH.glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in documents folder")
            return 0
        
        new_files = [f for f in pdf_files if f not in self._processed_files or force_reload]
        
        if not new_files and not force_reload:
            logger.info("No new files to process")
            return 0
        
        logger.info(f"Processing {len(new_files)} PDF file(s)...")
        
        all_documents = []
        for pdf_file in new_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["file_path"] = str(pdf_file)
                
                all_documents.extend(documents)
                self._processed_files.add(pdf_file)
                logger.info(f"Loaded {len(documents)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
        
        if not all_documents:
            return 0
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        splits = text_splitter.split_documents(all_documents)
        logger.info(f"Split into {len(splits)} chunks")
        
        # Add to vector store
        self.vectorstore.add_documents(splits)
        self._refresh_chain()
        logger.info("Documents added to vector store")

        return len(new_files)
    
    def remove_document(self, file_path: Path) -> bool:
        """Remove a document from the vector store."""
        try:
            # Chroma doesn't have direct delete by metadata, so we rebuild
            # For small scale, this is acceptable
            self.rebuild_index()
            self._processed_files.discard(file_path)
            return True
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return False
    
    def rebuild_index(self) -> int:
        """Rebuild the entire index from scratch."""
        logger.info("Rebuilding index...")

        # Delete the collection instead of the directory: chroma_db is a bind
        # mount inside Docker, and rmtree() fails with EBUSY on mount points.
        try:
            self.vectorstore.delete_collection()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete Chroma collection: %s", exc)

        config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        # Reinitialize
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_PERSIST_DIR),
            embedding_function=self.embeddings,
            client_settings=self._chroma_settings(),
        )
        self._refresh_chain()
        self._processed_files.clear()
        
        # Reload all documents
        return self.load_documents(force_reload=True)
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.qa_chain:
            return "RAG system not initialized. Please add documents first."
        
        try:
            response = self.qa_chain.invoke(question)
            return response
        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"Error processing your question: {str(e)}"
    
    def get_stats(self) -> dict:
        """Get statistics about the current index."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "document_count": count,
                "processed_files": len(self._processed_files),
                "files": [f.name for f in self._processed_files]
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Global instance
rag_system = RAGSystem()