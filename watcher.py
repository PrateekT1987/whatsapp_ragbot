"""Auto re-index when PDFs are added, changed or removed.

Events are debounced (a copy produces several create/modify events) and the
index is updated only once the folder has been quiet for a few seconds. Files
that are still being written are detected with a size/mtime stability check.
Only new or changed files are embedded; removing a PDF drops its chunks.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import config
from rag import rag_system

logger = logging.getLogger(__name__)


class _PDFChangeHandler(FileSystemEventHandler):
    def __init__(self, schedule) -> None:
        super().__init__()
        self._schedule = schedule

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        path = getattr(event, "src_path", "") or getattr(event, "dest_path", "")
        if Path(path).suffix.lower() != ".pdf":
            return
        self._schedule()


class DocumentWatcher:
    """Debounced, single-worker index rebuilds."""

    def __init__(self, debounce_seconds: float = 3.0) -> None:
        self.observer: Any = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._debounce = debounce_seconds
        self._handler = _PDFChangeHandler(self._schedule)
        self._running = False
        self._pending = False

    def _schedule(self) -> None:
        with self._lock:
            self._pending = True
            # An update is already in progress; its `finally` reschedules.
            if self._running:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _files_are_stable(self) -> bool:
        """False while a PDF is still being copied/written."""
        files = list(config.DOCUMENTS_PATH.glob("*.pdf"))
        snapshot = {}
        for pdf in files:
            try:
                stat = pdf.stat()
            except OSError:
                return False
            snapshot[str(pdf)] = (stat.st_size, stat.st_mtime)

        time.sleep(1.0)

        for pdf in files:
            try:
                stat = pdf.stat()
            except OSError:
                return False
            if snapshot.get(str(pdf)) != (stat.st_size, stat.st_mtime):
                return False
        return True

    def _run(self) -> None:
        with self._lock:
            self._timer = None
            if self._running:
                return
            self._running = True
            self._pending = False

        try:
            if not config.DOCUMENTS_PATH.exists():
                return

            if not self._files_are_stable():
                logger.info("Documents still changing; delaying update")
                with self._lock:
                    self._pending = True
                return

            if not rag_system.initialized:
                logger.info("RAG not initialized yet; skipping automatic update")
                return

            # Incremental: only new/changed files are embedded, removed files
            # have their chunks dropped. A full rebuild is reserved for /reload.
            count = rag_system.load_documents()
            logger.info("Index updated automatically (%s file(s))", count)
        except Exception:  # noqa: BLE001
            logger.exception("Automatic re-index failed")
        finally:
            with self._lock:
                self._running = False
                pending = self._pending
            if pending:
                self._schedule()

    def start(self) -> None:
        if not config.DOCUMENTS_PATH.exists():
            config.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
            logger.info("Created documents directory: %s", config.DOCUMENTS_PATH)

        self.observer = Observer()
        self.observer.schedule(
            self._handler, str(config.DOCUMENTS_PATH), recursive=False
        )
        self.observer.start()
        logger.info("Started watching %s for changes", config.DOCUMENTS_PATH)

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching documents folder")


document_watcher = DocumentWatcher()
