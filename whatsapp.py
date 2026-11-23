import logging
from typing import Optional
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from config import config
from rag import rag_system

logger = logging.getLogger(__name__)


class WhatsAppBot:
    def __init__(self):
        self.client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        self.from_number = config.TWILIO_WHATSAPP_NUMBER
    
    def send_message(self, to_number: str, message: str) -> Optional[str]:
        """Send a WhatsApp message."""
        try:
            # Ensure the number has whatsapp: prefix
            if not to_number.startswith("whatsapp:"):
                to_number = f"whatsapp:{to_number}"
            
            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            logger.info(f"Message sent to {to_number}: {message_obj.sid}")
            return message_obj.sid
        except Exception as e:
            logger.error(f"Error sending message to {to_number}: {e}")
            return None
    
    def create_twiml_response(self, message: str) -> str:
        """Create a TwiML response for webhook."""
        resp = MessagingResponse()
        resp.message(message)
        return str(resp)
    
    def process_incoming_message(self, from_number: str, body: str) -> str:
        """Process an incoming WhatsApp message and return response."""
        logger.info(f"Received message from {from_number}: {body}")
        
        # Handle special commands
        body_lower = body.strip().lower()
        
        if body_lower in ["/help", "help", "?"]:
            return self._get_help_message()
        
        if body_lower == "/stats":
            return self._get_stats_message()
        
        if body_lower == "/reload":
            return self._reload_documents()
        
        if body_lower.startswith("/add "):
            # Not applicable for file-based system, but keep for extensibility
            return "Use the documents folder to add PDF files. The system auto-indexes changes."
        
        # Query RAG system
        response = rag_system.query(body)
        return response
    
    def _get_help_message(self) -> str:
        return (
            "🤖 *RAG WhatsApp Bot Commands*\n\n"
            "Just ask me a question and I'll answer based on your documents!\n\n"
            "*Commands:*\n"
            "• `/help` - Show this help\n"
            "• `/stats` - Show document statistics\n"
            "• `/reload` - Rebuild the document index\n\n"
            "📁 *Adding Documents:*\n"
            "Place PDF files in the documents folder. The system automatically "
            "detects and indexes new/changed files."
        )
    
    def _get_stats_message(self) -> str:
        stats = rag_system.get_stats()
        if "error" in stats:
            return f"❌ Error getting stats: {stats['error']}"
        
        files_list = "\n".join(f"  • {f}" for f in stats.get("files", [])) or "  (none)"
        return (
            f"📊 *Index Statistics*\n\n"
            f"📄 Documents indexed: {stats.get('document_count', 0)} chunks\n"
            f"📁 Source files: {stats.get('processed_files', 0)}\n\n"
            f"*Files:*\n{files_list}"
        )
    
    def _reload_documents(self) -> str:
        try:
            count = rag_system.rebuild_index()
            return f"✅ Index rebuilt! Processed {count} PDF file(s)."
        except Exception as e:
            logger.error(f"Error reloading documents: {e}")
            return f"❌ Error reloading: {str(e)}"


# Global instance
whatsapp_bot = WhatsAppBot()