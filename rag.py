"""RAG system with Qdrant vector store, caching, and streaming support."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from config import config
from embeddings import get_embeddings
from vectorstore import QdrantVectorStoreManager
from cache import query_cache

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM tokens."""

    def __init__(self):
        self.tokens: List[str] = []
        self.done = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.done = True

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.done = True
        logger.error(f"LLM streaming error: {error}")

    def get_response(self) -> str:
        return "".join(self.tokens)


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

        # Non-streaming LLM for regular queries
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0,
            openai_api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers,
            timeout=120,
            max_retries=2,
            streaming=False,
        )

        # Streaming LLM (created on demand)
        self._streaming_llm: Optional[ChatOpenAI] = None

        # Vector store manager (Qdrant only)
        self.vector_manager = QdrantVectorStoreManager(
            embeddings=self.embeddings,
            collection_name=config.QDRANT_COLLECTION,
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
        )

        self.retriever = None
        self.qa_chain = None
        self.streaming_chain = None

    def initialize(self) -> None:
        """Initialize the vector store and QA chain."""
        logger.info("Initializing RAG system with Qdrant...")
        self._refresh_chain()
        logger.info("RAG system initialized")

    def _get_streaming_llm(self) -> ChatOpenAI:
        """Get or create a streaming LLM instance."""
        if self._streaming_llm is None:
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

            self._streaming_llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=0,
                openai_api_key=api_key,
                base_url=base_url,
                default_headers=extra_headers,
                timeout=120,
                max_retries=2,
                streaming=True,
            )
        return self._streaming_llm

    def _refresh_chain(self) -> None:
        """Rebuild retriever + QA chain, clamping k to the index size."""
        try:
            count = self.vector_manager.count()
        except Exception:  # noqa: BLE001
            count = config.TOP_K_RESULTS

        k = max(1, min(config.TOP_K_RESULTS, count)) if count else 1
        self.retriever = self.vector_manager.get_retriever(k=k)
        self._create_qa_chain()

    def _create_qa_chain(self) -> None:
        """Create the RAG QA chain (non-streaming)."""
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

    def _create_streaming_chain(self) -> Any:
        """Create a streaming chain for token-by-token output."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            ("human", "{question}")
        ])

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        streaming_llm = self._get_streaming_llm()
        return (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | streaming_llm
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

        new_files = [f for f in pdf_files if not self.vector_manager.is_processed(f) or force_reload]

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
                self.vector_manager.add_processed_file(pdf_file)
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
        self.vector_manager.add_documents(splits)
        self._refresh_chain()
        logger.info("Documents added to vector store")

        return len(new_files)

    def rebuild_index(self) -> int:
        """Rebuild the entire index from scratch."""
        logger.info("Rebuilding index...")

        # Delete the collection
        try:
            self.vector_manager.delete_collection()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete vector store collection: %s", exc)

        # Reinitialize vector manager
        self.vector_manager = QdrantVectorStoreManager(
            embeddings=self.embeddings,
            collection_name=config.QDRANT_COLLECTION,
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
        )

        self._refresh_chain()
        self.vector_manager.clear_processed_files()

        # Reload all documents
        return self.load_documents(force_reload=True)

    def query(self, question: str) -> str:
        """Query the RAG system (non-streaming, with caching)."""
        if not self.qa_chain:
            return "RAG system not initialized. Please add documents first."

        # Check cache first
        cached = query_cache.get(question)
        if cached:
            return cached

        try:
            response = self.qa_chain.invoke(question)
            # Cache the response
            query_cache.set(question, response)
            return response
        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"Error processing your question: {str(e)}"

    async def query_streaming(self, question: str, chat_id: str, send_chunk_callback) -> str:
        """Query the RAG system with streaming, sending chunks to WhatsApp as they arrive.
        
        Args:
            question: The user's question
            chat_id: WhatsApp chat ID to send chunks to
            send_chunk_callback: Async function to send a chunk (chat_id, chunk) -> bool
            
        Returns:
            Full response string
        """
        if not self.qa_chain:
            return "RAG system not initialized. Please add documents first."

        # Check cache first
        cached = query_cache.get(question)
        if cached:
            # Send cached response in chunks
            await self._send_cached_streaming(cached, chat_id, send_chunk_callback)
            return cached

        # Create streaming chain
        chain = self._create_streaming_chain()
        callback = StreamingCallbackHandler()

        try:
            # Get relevant documents first
            docs = await self.retriever.ainvoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            # Stream the response
            full_response = []
            async for chunk in chain.astream({"context": context, "question": question}, config={"callbacks": [callback]}):
                if chunk:
                    full_response.append(chunk)
                    # Send chunk to WhatsApp
                    await send_chunk_callback(chat_id, chunk)

            response = "".join(full_response)
            
            # Cache the full response
            query_cache.set(question, response)
            
            return response
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            error_msg = f"Error processing your question: {str(e)}"
            await send_chunk_callback(chat_id, error_msg)
            return error_msg

    async def _send_cached_streaming(self, response: str, chat_id: str, send_chunk_callback) -> None:
        """Send a cached response in simulated streaming chunks."""
        # Split into words and send in small batches
        words = response.split()
        batch_size = 5
        for i in range(0, len(words), batch_size):
            chunk = " ".join(words[i:i+batch_size]) + " "
            await send_chunk_callback(chat_id, chunk)
            # Small delay to simulate streaming
            import asyncio
            await asyncio.sleep(0.05)

    def get_stats(self) -> dict:
        """Get statistics about the current index."""
        try:
            count = self.vector_manager.count()
            return {
                "document_count": count,
                "processed_files": len(self.vector_manager.get_processed_files()),
                "files": [f.name for f in self.vector_manager.get_processed_files()],
                "vector_store": "qdrant",
                "cache": query_cache.stats(),
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Global instance
rag_system = RAGSystem()