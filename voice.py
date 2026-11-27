"""Voice note handling with faster-whisper transcription.

Downloads voice notes from WAHA, transcribes them locally, and feeds the text
into the normal RAG pipeline.

faster-whisper is a CTranslate2 reimplementation of Whisper: CPU-friendly and
free of the build-time setuptools/pkg_resources problems of openai-whisper.
"""

import asyncio
import logging
import os
import tempfile
from typing import Optional
from urllib.parse import urlparse

from config import config

logger = logging.getLogger(__name__)

# Global whisper model (loaded lazily on the first voice note)
_whisper_model = None

# WAHA deletes downloaded media after WHATSAPP_FILES_LIFETIME seconds (180 by
# default), so the webhook handler must fetch the file immediately.
_MEDIA_PATH_PREFIX = "/api/files/"


def get_whisper_model():
    """Lazy-load the faster-whisper model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        model_name = config.WHISPER_MODEL
        logger.info("Loading faster-whisper model: %s", model_name)
        kwargs = dict(
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
        )
        try:
            # The image prefetches the default model, so start fully offline.
            _whisper_model = WhisperModel(model_name, local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - model not cached yet
            logger.info("Whisper model %s not cached; downloading it", model_name)
            _whisper_model = WhisperModel(model_name, **kwargs)
    return _whisper_model


def _media_file_path(payload: dict) -> Optional[str]:
    """Return the WAHA /api/files/... path for a message's media, if present.

    WAHA serves downloaded media from ``{baseUrl}/api/files/<name>``. The URL
    host is WAHA's own view of itself (often ``localhost:3000``), so only the
    path is reused and the request goes through our configured base URL.
    """
    media = payload.get("media") or {}
    url = media.get("url")
    if url:
        path = urlparse(url).path
        if path.startswith(_MEDIA_PATH_PREFIX):
            return path
    return None


async def _fetch_media(waha_client, path: str) -> Optional[bytes]:
    """GET a WAHA media path and return its bytes."""
    try:
        resp = await waha_client._request(
            "GET",
            path,
            retries=2,
            headers={"Accept": "*/*"},
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Voice download failed (%s)", exc)
        return None

    if resp.status_code == 200 and resp.content:
        return resp.content
    logger.warning("Could not download voice file %s: HTTP %s", path, resp.status_code)
    return None


async def download_voice_file(waha_client, payload: dict) -> Optional[bytes]:
    """Download a voice note from WAHA.

    Prefers the ``media.url`` embedded in the webhook payload, then falls back
    to asking WAHA for the message and downloading its media.
    """
    media = payload.get("media") or {}
    if media.get("error"):
        logger.warning("WAHA could not download the voice media: %s", media["error"])

    path = _media_file_path(payload)
    if path:
        audio = await _fetch_media(waha_client, path)
        if audio:
            return audio

    # Fallback: fetch the message and use the media URL it reports.
    message_id = payload.get("id") or payload.get("_data", {}).get("id")
    if not message_id:
        return None
    try:
        resp = await waha_client._request(
            "GET",
            f"/api/{waha_client.session}/messages/{message_id}",
            retries=1,
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.warning("Could not get message details: %s", resp.status_code)
            return None

        msg_data = resp.json()
        path = _media_file_path(msg_data)
        if path:
            return await _fetch_media(waha_client, path)

        # Older WAHA versions expose a filesystem path instead of a URL.
        media_path = (msg_data.get("mediaData") or {}).get("filePath") or msg_data.get(
            "filePath"
        )
        if media_path:
            return await _fetch_media(
                waha_client, "/api/files/" + os.path.basename(media_path)
            )

        logger.warning("No media path found in voice message")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Error downloading voice file: %s", exc)
        return None


async def transcribe_audio(audio_bytes: bytes, language: Optional[str] = None) -> str:
    """Transcribe audio bytes to text using faster-whisper."""
    loop = asyncio.get_running_loop()

    def _transcribe() -> str:
        model = get_whisper_model()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            segments, _info = model.transcribe(
                tmp_path,
                language=language or config.WHISPER_LANGUAGE or None,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            return " ".join(segment.text for segment in segments).strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return await loop.run_in_executor(None, _transcribe)


async def handle_voice_note(waha_client, payload: dict, chat_id: str) -> Optional[str]:
    """Process a voice note: download, transcribe, return text (or None)."""
    if not config.WHISPER_ENABLED:
        logger.info("Voice note received but WHISPER_ENABLED=false")
        return None

    logger.info("Processing voice note from %s", chat_id)

    audio_bytes = await download_voice_file(waha_client, payload)
    if not audio_bytes:
        return None

    try:
        text = await transcribe_audio(audio_bytes)
        if not text:
            logger.warning("faster-whisper returned an empty transcription")
            return None
        logger.info("Transcribed voice note (%d chars)", len(text))
        return text
    except Exception:  # noqa: BLE001
        logger.exception("Transcription failed")
        return None
