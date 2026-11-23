import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from rag import rag_system
from config import config

logger = logging.getLogger(__name__)


class DocumentChangeHandler(FileSystemEventHandler):
    """Handle file system events for document changes."""
    
    def __init__(self):
        super().__init__()
        self.last_rebuild = 0
        self.rebuild_cooldown = 5  # seconds
    
    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "created")
    
    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "modified")
    
    def on_deleted(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "deleted")
    
    def _handle_change(self, file_path: str, event_type: str):
        path = Path(file_path)
        
        # Only process PDF files
        if path.suffix.lower() != ".pdf":
            return
        
        # Only process files in our documents directory
        try:
            path.relative_to(config.DOCUMENTS_PATH)
        except ValueError:
            return
        
        current_time = time.time()
        if current_time - self.last_rebuild < self.rebuild_cooldown:
            logger.info(f"Skipping rebuild (cooldown), file {event_type}: {path.name}")
            return
        
        logger.info(f"Document {event_type}: {path.name}, rebuilding index...")
        self.last_rebuild = current_time
        
        # Rebuild index in background to avoid blocking
        try:
            rag_system.rebuild_index()
            logger.info("Index rebuilt successfully")
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")


class DocumentWatcher:
    """Watch documents folder for changes and auto-reindex."""
    
    def __init__(self):
        self.observer: Observer = None
        self.handler = DocumentChangeHandler()
    
    def start(self):
        """Start watching the documents folder."""
        if not config.DOCUMENTS_PATH.exists():
            config.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created documents directory: {config.DOCUMENTS_PATH}")
        
        self.observer = Observer()
        self.observer.schedule(
            self.handler,
            str(config.DOCUMENTS_PATH),
            recursive=False
        )
        self.observer.start()
        logger.info(f"Started watching {config.DOCUMENTS_PATH} for changes")
    
    def stop(self):
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching documents folder")


# Global instance
document_watcher = DocumentWatcher()