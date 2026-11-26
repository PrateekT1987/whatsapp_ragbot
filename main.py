import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from config import config
from rag import rag_system
from waha import extract_message_id, waha
from watcher import document_watcher
from cache import query_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Keep strong references to background tasks so they are not garbage collected.
_background_tasks: set[asyncio.Task] = set()

# Per-chat queues keep replies ordered; the semaphore bounds concurrent LLM
# calls across all chats.
_dispatcher: "ChatDispatcher" = None  # type: ignore[assignment]
_last_busy_notice: dict[str, float] = {}

HELP_TEXT = (
    "🤖 **RAG Bot**\n\n"
    "Ask me anything about the indexed documents and I'll answer here.\n\n"
    "**Commands**\n"
    "• /help – this message\n"
    "• /stats – how many chunks/files are indexed\n"
    "• /reload – rebuild the index from the documents folder\n"
    "• /ping – check that the bot is alive\n"
    "• /cache – show cache statistics\n"
    "• /cache clear – clear the query cache\n"
)


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _notify_busy(chat_id: str) -> None:
    """Tell a chat it is queueing, at most once every 15 seconds."""
    now = time.time()
    if now - _last_busy_notice.get(chat_id, 0) < 15:
        return
    _last_busy_notice[chat_id] = now
    _spawn(
        waha.send_text(
            chat_id,
            "⏳ I'm still working through your earlier questions. Please wait a moment.",
        )
    )


class ChatDispatcher:
    """One ordered queue per chat, with a global cap on concurrent work.

    Without this, every webhook spawned its own task: ten quick messages meant
    ten parallel OpenRouter calls and replies could arrive out of order.
    """

    def __init__(self, concurrency: int, max_queue: int) -> None:
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._max_queue = max(1, max_queue)
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, asyncio.Task] = {}

    def submit(self, chat_id: str, payload: dict) -> bool:
        queue = self._queues.get(chat_id)
        if queue is None:
            queue = asyncio.Queue(maxsize=self._max_queue)
            self._queues[chat_id] = queue
            self._workers[chat_id] = asyncio.create_task(
                self._worker(chat_id, queue)
            )
        try:
            queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            return False

    async def _worker(self, chat_id: str, queue: asyncio.Queue) -> None:
        while True:
            payload = await queue.get()
            try:
                async with self._semaphore:
                    await process_message(payload)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Unhandled error while processing a message")
            finally:
                queue.task_done()

    async def shutdown(self) -> None:
        for task in self._workers.values():
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._workers.clear()
        self._queues.clear()


def _payload_session(data: dict, payload: dict) -> Optional[str]:
    """Extract the WAHA session name from a webhook body."""
    for container in (data, payload):
        if isinstance(container, dict):
            session = container.get("session")
            if isinstance(session, str) and session:
                return session
    nested = payload.get("_data") if isinstance(payload, dict) else None
    if isinstance(nested, dict):
        session = nested.get("session")
        if isinstance(session, str) and session:
            return session
    return None


_dispatcher = ChatDispatcher(
    config.MAX_CONCURRENT_QUERIES, config.MAX_QUEUED_PER_CHAT
)


async def _handle_command(text: str) -> str | None:
    cmd = text.strip().lower()
    if cmd in ("/help", "help", "?"):
        return HELP_TEXT
    if cmd == "/ping":
        return "pong ✅"
    if cmd == "/stats":
        stats = await asyncio.to_thread(rag_system.get_stats)
        if "error" in stats:
            return f"❌ Error reading stats: {stats['error']}"
        files = "\n".join(f"  • {f}" for f in stats.get("files", [])) or "  (none)"
        vector_store = stats.get("vector_store", "unknown")
        cache_stats = stats.get("cache", {})
        cache_info = ""
        if cache_stats:
            cache_info = f"\n**Cache**\n  Enabled: {cache_stats.get('enabled')}\n  Exact: {cache_stats.get('exact_cache_size')}/{cache_stats.get('exact_cache_max')}\n  Disk: {cache_stats.get('disk_cache_size')}\n  Hit rate: {cache_stats.get('hit_rate', 0)}"
        return (
            f"📊 **Index**\n\n"
            f"Chunks: {stats.get('document_count', 0)}\n"
            f"Files: {stats.get('processed_files', 0)}\n"
            f"Vector Store: {vector_store}{cache_info}\n\n"
            f"**Sources**\n{files}"
        )
    if cmd == "/reload":
        if rag_system.rebuilding:
            return "🔄 A rebuild is already running. Please wait."
        try:
            count = await asyncio.to_thread(rag_system.rebuild_index)
            # Clear cache on rebuild
            query_cache.clear()
            return f"✅ Index rebuilt – {count} PDF file(s) processed. Cache cleared."
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reload failed")
            return f"❌ Reload failed: {exc}"
    if cmd == "/cache":
        return f"📦 **Cache Stats**\n\n{json.dumps(query_cache.stats(), indent=2)}"
    if cmd == "/cache clear":
        query_cache.clear()
        return "✅ Query cache cleared."
    return None


# Streaming state per chat
_streaming_sessions: dict[str, list[str]] = {}


async def process_message(payload: dict) -> None:
    """Decide whether an incoming WAHA message is a question for the bot."""
    message_id = extract_message_id(payload)

    # Duplicated across `message` / `message.any` events, or a retry.
    if waha.mark_seen(message_id):
        return

    body = (payload.get("body") or "").strip()
    if not body:
        return

    msg_type = payload.get("type")
    if msg_type not in (None, "text"):
        return

    if not await waha.is_self_chat(payload):
        if config.WAHA_SELF_CHAT_ONLY:
            logger.debug("Ignoring non self-chat message from %s", payload.get("chatId"))
            return

    # Our own answer echoed back into the self-chat - never answer it.
    if waha.is_own_reply(message_id, body):
        logger.debug("Ignoring own echoed message")
        return

    chat_id = payload.get("from") or payload.get("chatId")
    if not chat_id:
        return

    logger.info("Question from %s (chatId: %s): %s",
                payload.get("from"), payload.get("chatId"), body[:120])

    try:
        reply = await _handle_command(body)
        if reply is None:
            await waha.start_typing(chat_id)
            try:
                if config.LLM_STREAMING:
                    # Streaming mode: send tokens as they arrive
                    async def send_chunk(cid: str, chunk: str) -> bool:
                        return await waha.send_chunk(cid, chunk)

                    reply = await rag_system.query_streaming(body, chat_id, send_chunk)
                else:
                    # Non-streaming: retries happen in the worker thread; tell
                    # the user once if the model is rate-limited.
                    loop = asyncio.get_running_loop()
                    busy_sent = False

                    def on_retry(attempt: int, delay: float, exc: Exception) -> None:
                        nonlocal busy_sent
                        if busy_sent:
                            return
                        busy_sent = True
                        asyncio.run_coroutine_threadsafe(
                            waha.send_text(
                                chat_id,
                                "⏳ The model is busy — retrying…",
                            ),
                            loop,
                        )

                    reply = await asyncio.to_thread(
                        rag_system.query, body, on_retry
                    )
            finally:
                await waha.stop_typing(chat_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to answer message")
        reply = f"❌ Sorry, something went wrong: {exc}"

    # For non-streaming, send the full reply
    if reply and not config.LLM_STREAMING:
        await waha.send_long_text(chat_id, reply)


async def _retry_rag_init() -> None:
    """Keep trying to connect to Qdrant after a failed startup."""
    delay = max(1, config.STARTUP_RETRY_SECONDS)
    while True:
        try:
            await asyncio.to_thread(rag_system.initialize)
            loaded = await asyncio.to_thread(rag_system.load_documents)
            logger.info("RAG ready after retry; loaded %s document(s)", loaded)
            return
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            # Dimension mismatch / invalid configuration: retrying cannot help.
            logger.error("RAG initialization cannot succeed: %s", exc)
            return
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RAG initialization failed: %s (retrying in %ss)", exc, delay
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


async def _ensure_waha_session() -> str:
    """Start the session, or restart it once if WAHA reports FAILED."""
    try:
        info = await waha.session_status()
        status = info.get("status", "unknown")

        if status == "FAILED":
            logger.warning(
                "WAHA session %r is FAILED; restarting once", config.WAHA_SESSION
            )
            await waha.restart_session()
            await asyncio.sleep(2)
            info = await waha.session_status()
            status = info.get("status", status)
        elif status in ("STOPPED", "unknown", "unreachable"):
            await waha.start_session()
            await asyncio.sleep(1)
            info = await waha.session_status()
            status = info.get("status", status)

        logger.info("WAHA session %r status: %s", config.WAHA_SESSION, status)
        if status == "WORKING":
            await waha.get_me_id(force=True)
        return status
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("WAHA not reachable yet: %s", exc)
        return "unreachable"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG WhatsApp bot (WAHA)...")

    missing = config.validate()
    if missing:
        logger.warning("Missing configuration: %s", missing)

    init_task: Optional[asyncio.Task] = None
    waha_task: Optional[asyncio.Task] = None

    # Qdrant may still be starting: serve /ready (503) immediately and connect
    # in the background with retry/backoff instead of crashing the import or
    # relying on Compose restarts.
    init_task = asyncio.create_task(_retry_rag_init())

    document_watcher.start()

    # Session recovery can take a few seconds; do not block readiness on it.
    waha_task = asyncio.create_task(_ensure_waha_session())

    logger.info("Bot ready. WAHA dashboard: %s/dashboard", config.WAHA_BASE_URL)

    try:
        yield
    finally:
        logger.info("Shutting down...")
        for task in (init_task, waha_task):
            if task is not None:
                task.cancel()
        for task in (init_task, waha_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    logger.exception("Background task failed during shutdown")
        await _dispatcher.shutdown()
        document_watcher.stop()
        await waha.aclose()


app = FastAPI(
    title="RAG WhatsApp Bot (WAHA)",
    description="Self-chat RAG assistant on WhatsApp using WAHA + FastAPI + LangChain",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "RAG WhatsApp Bot", "version": "2.0.0"}


@app.get("/health")
async def health():
    stats = await asyncio.to_thread(rag_system.get_stats)
    session = await waha.session_status()
    healthy = "error" not in stats and rag_system.initialized
    return {
        "status": "healthy" if healthy else "degraded",
        "config_valid": not config.validate(),
        "initialized": rag_system.initialized,
        "init_error": rag_system.init_error,
        "rag": stats,
        "waha": {
            "session": config.WAHA_SESSION,
            "status": session.get("status", "unknown"),
        },
    }


@app.get("/ready")
async def ready():
    """Readiness for Docker: the vector store and QA chain must work.

    WAHA pairing is reported but not required, so an unpaired session does not
    make the container permanently unhealthy.
    """
    chunks = 0
    error = rag_system.init_error if not rag_system.initialized else None

    if rag_system.initialized and rag_system.vector_manager is not None:
        try:
            chunks = await asyncio.to_thread(rag_system.vector_manager.count)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

    chain_ok = rag_system.qa_chain is not None
    is_ready = rag_system.initialized and error is None and chain_ok

    try:
        waha_info = await asyncio.wait_for(waha.session_status(), timeout=3)
    except Exception:  # noqa: BLE001
        waha_info = {"status": "unknown"}

    body = {
        "ready": is_ready,
        "initialized": rag_system.initialized,
        "qdrant": rag_system.initialized and error is None,
        "chunks": chunks,
        "chain": chain_ok,
        "rebuilding": rag_system.rebuilding,
        "waha": {
            "session": config.WAHA_SESSION,
            "status": waha_info.get("status", "unknown"),
        },
    }
    if error:
        body["error"] = error
    return JSONResponse(status_code=200 if is_ready else 503, content=body)


@app.post("/webhook")
async def waha_webhook(request: Request):
    raw = await request.body()

    # Optional HMAC verification (set WAHA_HMAC_KEY in both services).
    # WAHA sends `X-Webhook-Hmac` plus the algorithm in
    # `X-Webhook-Hmac-Algorithm` (sha512 by default); `X-Hmac-Signature` is
    # accepted for older/custom senders.
    if config.WAHA_HMAC_KEY:
        signature = request.headers.get(
            "X-Webhook-Hmac", request.headers.get("X-Hmac-Signature", "")
        )
        algorithm = request.headers.get("X-Webhook-Hmac-Algorithm", "sha512").lower()
        digest = getattr(hashlib, algorithm, None)
        if digest is None:
            logger.warning("Rejected webhook with unsupported HMAC algorithm: %s", algorithm)
            raise HTTPException(status_code=401, detail="Unsupported signature algorithm")
        expected = hmac.new(
            config.WAHA_HMAC_KEY.encode("utf-8"), raw, digest
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            logger.warning("Rejected webhook with invalid HMAC signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        data = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        logger.warning("Webhook body is not JSON")
        return JSONResponse({"status": "ignored"})

    event = data.get("event")
    # WAHA emits `message` for incoming and `message.any` for everything,
    # including our own outgoing messages - self-chat messages are outgoing.
    if event not in ("message", "message.any"):
        return JSONResponse({"status": "ignored", "event": event})

    payload = data.get("payload") or {}
    if not isinstance(payload, dict):
        return JSONResponse({"status": "ignored", "reason": "invalid payload"})

    # Cross-bot guard: a shared webhook URL or a misconfigured WAHA instance
    # must never make the personal bot answer work messages (or vice versa).
    incoming_session = _payload_session(data, payload)
    if incoming_session and incoming_session != config.WAHA_SESSION:
        logger.warning(
            "Rejected webhook for session %r (expected %r)",
            incoming_session, config.WAHA_SESSION,
        )
        return JSONResponse(
            status_code=403,
            content={"status": "ignored", "reason": "session mismatch"},
        )

    chat_id = payload.get("from") or payload.get("chatId")
    if not chat_id:
        return JSONResponse({"status": "ignored", "reason": "no chat id"})

    # Answer WAHA fast; the per-chat queue does the slow RAG work in order.
    if not _dispatcher.submit(chat_id, payload):
        logger.warning("Queue full for %s; dropping message", chat_id)
        _notify_busy(chat_id)
        return JSONResponse(
            status_code=429,
            content={"status": "busy", "reason": "chat queue full"},
        )

    return JSONResponse({"status": "queued"})


@app.post("/webhook/test")
async def test_webhook(message: str, simulate: str = "self"):
    """Simulate a self-chat message without WhatsApp (useful for smoke tests)."""
    if simulate == "self":
        payload = {"fromMe": True, "type": "text", "body": message}
        reply = await _handle_command(message)
        if reply is None:
            reply = await asyncio.to_thread(rag_system.query, message)
        return {"response": reply}
    raise HTTPException(status_code=400, detail="Only simulate=self is supported")


@app.get("/stats")
async def get_stats():
    return await asyncio.to_thread(rag_system.get_stats)


@app.post("/reload")
async def reload_documents():
    if rag_system.rebuilding:
        raise HTTPException(status_code=409, detail="Reindex already in progress")
    try:
        count = await asyncio.to_thread(rag_system.rebuild_index)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    query_cache.clear()
    return {"message": f"Rebuilt index with {count} document(s)", "count": count}


@app.post("/send")
async def send_message(chat_id: str, message: str):
    ok = await waha.send_text(chat_id, message)
    if not ok:
        raise HTTPException(status_code=502, detail="WAHA sendText failed")
    return {"success": True, "chat_id": chat_id}


async def _prepare_qr() -> tuple[Optional[bytes], str]:
    """Return (qr_png, session_status), restarting FAILED/STOPPED sessions.

    WAHA abandons a session that is not paired within its QR window, so opening
    /qr after a while would otherwise just show nothing until a manual restart.
    """
    info = await waha.session_status()
    status = info.get("status", "unknown")

    if status in ("FAILED", "STOPPED", "unknown", "unreachable"):
        if status == "FAILED":
            await waha.restart_session()
        else:
            await waha.start_session()
        image = await waha.wait_for_qr()
        info = await waha.session_status()
        status = info.get("status", status)
    else:
        image = await waha.qr_image()

    return image, status


@app.get("/qr.png")
async def qr_png():
    """Raw pairing QR (PNG). Restarts the session if it expired."""
    image, status = await _prepare_qr()
    if not image:
        raise HTTPException(
            status_code=503,
            detail=f"QR not available yet (session status: {status}). Retry in a few seconds.",
        )
    return Response(
        content=image,
        media_type="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/qr")
async def qr_page():
    """Browser-friendly pairing page with a live, auto-refreshing QR."""
    info = await waha.session_status()
    status = info.get("status", "unknown")

    if status == "WORKING":
        body = (
            "<h1>✅ WhatsApp is already paired</h1>"
            "<p>Open WhatsApp → <b>Message Yourself</b> and send a question.</p>"
            f'<p class="muted">Session: {config.WAHA_SESSION}</p>'
        )
    else:
        body = (
            "<h1>Scan to pair WhatsApp</h1>"
            f'<p class="muted">Session status: <code>{status}</code></p>'
            '<div class="frame"><img id="qr" src="/qr.png" alt="pairing QR" '
            'onerror="this.style.display=\'none\';document.getElementById(\'msg\').style.display=\'block\'">'
            '<p id="msg" style="display:none">Generating a fresh QR…</p></div>'
            "<p>WhatsApp → <b>Settings</b> → <b>Linked devices</b> → <b>Link a device</b></p>"
            "<p>This page refreshes every 10 seconds; the QR expires quickly.</p>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="10">
<title>WhatsApp pairing · RAG bot</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui, sans-serif; background:#0b141a; color:#e9edef;
         display:flex; flex-direction:column; align-items:center; justify-content:center;
         min-height:100vh; margin:0; padding:24px; text-align:center; }}
  h1 {{ font-size:1.4rem; margin:0 0 12px; }}
  p {{ margin:8px 0; }}
  .muted {{ color:#8696a0; font-size:.9rem; }}
  code {{ background:#202c33; padding:2px 6px; border-radius:4px; }}
  .frame {{ background:#fff; padding:14px; border-radius:12px; margin:18px 0; }}
  img {{ display:block; width:292px; height:292px; image-rendering:pixelated; }}
  #msg {{ color:#111; width:292px; height:292px; display:flex; align-items:center;
          justify-content:center; margin:0; font-weight:600; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


@app.post("/session/start")
async def start_session():
    return await waha.start_session()


@app.get("/session")
async def session_info():
    return await waha.session_status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.UVICORN_RELOAD,
    )