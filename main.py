import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import config
from rag import rag_system
from watcher import document_watcher
from whatsapp import whatsapp_bot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting RAG WhatsApp Bot...")
    
    # Validate config
    missing = config.validate()
    if missing:
        logger.warning(f"Missing configuration: {missing}")
        logger.warning("Copy .env.example to .env and fill in your credentials")
    
    # Initialize RAG system
    rag_system.initialize()
    
    # Load existing documents
    loaded = rag_system.load_documents()
    logger.info(f"Loaded {loaded} document(s) on startup")
    
    # Start file watcher
    document_watcher.start()
    
    logger.info("Bot started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    document_watcher.stop()
    logger.info("Bot stopped")


app = FastAPI(
    title="RAG WhatsApp Bot",
    description="A RAG-powered WhatsApp chatbot using Twilio and OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "RAG WhatsApp Bot",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    stats = rag_system.get_stats()
    return {
        "status": "healthy",
        "rag": stats,
        "config_valid": len(config.validate()) == 0
    }


@app.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
    MessageSid: str = Form(None),
    NumMedia: str = Form("0"),
):
    """
    Twilio WhatsApp webhook endpoint.
    Receives incoming messages and responds via TwiML.
    """
    logger.info(f"Webhook received from {From}: {Body[:100]}...")
    
    # Process the message
    response_text = whatsapp_bot.process_incoming_message(From, Body)
    
    # Return TwiML response
    twiml_response = whatsapp_bot.create_twiml_response(response_text)
    
    return PlainTextResponse(content=twiml_response, media_type="application/xml")


@app.post("/webhook/test")
async def test_webhook(message: str, from_number: str = "whatsapp:+15551234567"):
    """Test endpoint for simulating webhook calls."""
    response_text = whatsapp_bot.process_incoming_message(from_number, message)
    return {"response": response_text}


@app.get("/stats")
async def get_stats():
    """Get RAG system statistics."""
    return rag_system.get_stats()


@app.post("/reload")
async def reload_documents():
    """Manually trigger document reload."""
    count = rag_system.rebuild_index()
    return {"message": f"Rebuilt index with {count} document(s)", "count": count}


@app.post("/send")
async def send_message(to: str, message: str):
    """Send a WhatsApp message (for testing/broadcasting)."""
    sid = whatsapp_bot.send_message(to, message)
    if sid:
        return {"success": True, "message_sid": sid}
    else:
        raise HTTPException(status_code=500, detail="Failed to send message")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )