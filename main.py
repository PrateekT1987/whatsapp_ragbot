import asyncio
import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from config import config
from rag import rag_system
from waha import extract_message_id, waha
from watcher import document_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Keep strong references to background tasks so they are not garbage collected.
_background_tasks: set[asyncio.Task] = set()

HELP_TEXT = (
    "🤖 *RAG Bot*\n\n"
    "Ask me anything about the indexed documents and I'll answer here.\n\n"
    "*Commands*\n"
    "• /help – this message\n"
    "• /stats – how many chunks/files are indexed\n"
    "• /reload – rebuild the index from the documents folder\n"
    "• /ping – check that the bot is alive"
)


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


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
        return (
            f"📊 *Index*\n\n"
            f"Chunks: {stats.get('document_count', 0)}\n"
            f"Files: {stats.get('processed_files', 0)}\n\n"
            f"*Sources*\n{files}"
        )
    if cmd == "/reload":
        try:
            count = await asyncio.to_thread(rag_system.rebuild_index)
            return f"✅ Index rebuilt – {count} PDF file(s) processed."
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reload failed")
            return f"❌ Reload failed: {exc}"
    return None


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

    chat_id = payload.get("chatId") or payload.get("from")
    if not chat_id:
        return

    logger.info("Question from %s: %s", chat_id, body[:120])

    try:
        reply = await _handle_command(body)
        if reply is None:
            await waha.start_typing(chat_id)
            try:
                reply = await asyncio.to_thread(rag_system.query, body)
            finally:
                await waha.stop_typing(chat_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to answer message")
        reply = f"❌ Sorry, something went wrong: {exc}"

    await waha.send_long_text(chat_id, reply)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG WhatsApp bot (WAHA)...")

    missing = config.validate()
    if missing:
        logger.warning("Missing configuration: %s", missing)

    await asyncio.to_thread(rag_system.initialize)
    loaded = await asyncio.to_thread(rag_system.load_documents)
    logger.info("Loaded %s document(s) on startup", loaded)

    document_watcher.start()

    # Best effort: make sure the WAHA session exists / is started.
    try:
        session = await waha.start_session()
        status = session.get("status") or session.get("engine")
        logger.info("WAHA session '%s' status: %s", config.WAHA_SESSION, status)
        await waha.get_me_id(force=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("WAHA not reachable yet: %s", exc)

    logger.info("Bot ready. WAHA dashboard: %s/dashboard", config.WAHA_BASE_URL)

    yield

    logger.info("Shutting down...")
    document_watcher.stop()


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
    return {
        "status": "healthy",
        "config_valid": not config.validate(),
        "rag": stats,
        "waha": {
            "session": config.WAHA_SESSION,
            "status": session.get("status", "unknown"),
        },
    }


@app.post("/webhook")
async def waha_webhook(request: Request):
    raw = await request.body()

    # Optional HMAC-SHA512 verification (set WAHA_HMAC_KEY in both services).
    if config.WAHA_HMAC_KEY:
        signature = request.headers.get("X-Hmac-Signature", "")
        expected = hmac.new(
            config.WAHA_HMAC_KEY.encode("utf-8"), raw, hashlib.sha512
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

    # Answer WAHA fast; do the slow RAG work in the background.
    _spawn(process_message(payload))
    return JSONResponse({"status": "ok"})


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
    count = await asyncio.to_thread(rag_system.rebuild_index)
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

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
