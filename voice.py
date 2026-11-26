"""Voice note handling with Whisper transcription.

Downloads voice notes from WAHA, transcribes with Whisper, and feeds
the text into the RAG pipeline.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx

from config import config
from metrics import WAHA_SENDS

logger = logging.getLogger(__name__)

# Global whisper model (loaded on first use)
_whisper_model = None


def get_whisper_model():
    """Lazy-load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        model_name = config.WHISPER_MODEL
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model


async def download_voice_file(waha_client, message_id: str, chat_id: str) -> Optional[bytes]:
    """Download a voice note file from WAHA."""
    try:
        # WAHA stores media files - we need to get the file from the message
        # First get message details to find the media path
        resp = await waha_client._request(
            "GET",
            f"/api/{waha_client.session}/messages/{message_id}",
            retries=1,
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.warning(f"Could not get message details: {resp.status_code}")
            return None

        msg_data = resp.json()
        media_path = msg_data.get("mediaData", {}).get("filePath") or msg_data.get("filePath")
        if not media_path:
            logger.warning("No media path found in voice message")
            return None

        # Download the file
        resp = await waha_client._request(
            "GET",
            f"/api/{waha_client.session}/media",
            retries=1,
            timeout=30.0,
            params={"path": media_path},
        )
        if resp.status_code == 200:
            return resp.content
        else:
            logger.warning(f"Failed to download voice file: {resp.status_code}")
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error downloading voice file: {exc}")
        return None


async def transcribe_audio(audio_bytes: bytes, language: Optional[str] = None) -> str:
    """Transcribe audio bytes to text using Whisper."""
    loop = asyncio.get_running_loop()

    def _transcribe() -> str:
        model = get_whisper_model()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            result = model.transcribe(
                tmp_path,
                language=language or config.WHISPER_LANGUAGE or None,
                fp16=False,  # CPU inference
            )
            return result.get("text", "").strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return await loop.run_in_executor(None, _transcribe)


async def handle_voice_note(waha_client, payload: dict, chat_id: str) -> Optional[str]:
    """Process a voice note: download, transcribe, and return text.

    Returns the transcribed text, or None if handling failed.
    """
    if not config.WHISPER_ENABLED:
        logger.info("Voice notes received but WHISPER_ENABLED=false")
        return None

    message_id = payload.get("id") or payload.get("_data", {}).get("id")
    if not message_id:
        logger.warning("Voice note missing message ID")
        return None

    logger.info(f"Processing voice note from {chat_id}")

    # Download the voice file
    audio_bytes = await download_voice_file(waha_client, str(message_id), chat_id)
    if not audio_bytes:
        return None

    # Transcribe
    try:
        text = await transcribe_audio(audio_bytes)
        if not text:
            logger.warning("Whisper returned empty transcription")
            return None
        logger.info(f"Transcribed voice note ({len(text)} chars): {text[:100]}...")
        return text
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed")
        return None