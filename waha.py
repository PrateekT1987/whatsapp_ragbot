"""WAHA (WhatsApp HTTP API) client.

Talks to the WAHA container over its REST API and breaks the two hard problems
of a "Message Yourself" bot:

1. Echo loop - in the self-chat BOTH your question and the bot's answer are
   `fromMe: true`, so the bot would keep answering itself. Every outgoing text
   is fingerprinted *before* it is sent, and every inbound `fromMe` message that
   matches a recent fingerprint is ignored.
2. Duplicate webhooks - WAHA can deliver the same message under both the
   `message` and `message.any` events, and retries on errors. Message ids are
   remembered with a TTL.
"""

import asyncio
import hashlib
import logging
import random
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import httpx

from config import config
from cache import session_cache
from formatting import to_whatsapp
from metrics import WAHA_SENDS

logger = logging.getLogger(__name__)


def _backoff(attempt: int, retry_after: Optional[str] = None) -> float:
    """Exponential backoff with jitter, honouring a Retry-After header."""
    if retry_after:
        try:
            return min(15.0, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    base = min(8.0, 0.5 * (2**attempt))
    return base + random.uniform(0, base)


def _user_part(jid: Optional[str]) -> Optional[str]:
    """'1234:12@c.us' -> '1234'."""
    if not jid:
        return None
    return jid.split("@")[0].split(":")[0]


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def _is_self_chat_payload(
    payload: dict,
    me_id: Optional[str],
    me_lid: Optional[str],
) -> bool:
    """Pure self-chat check, kept separate so it can be unit tested.

    A "Message Yourself" message is `fromMe`, and its chat id is our own phone
    JID (`@c.us`) or LID (`@lid`). When the session id is unavailable, sender
    and recipient being the same account is a safe fallback.
    """
    if not payload.get("fromMe"):
        return False

    chat_id = payload.get("chatId") or payload.get("from")

    if me_id and _user_part(chat_id) == _user_part(me_id):
        return True

    if me_lid and chat_id and _user_part(chat_id) == _user_part(me_lid):
        return True

    sender = payload.get("from")
    recipient = payload.get("to")
    return bool(sender and recipient and _user_part(sender) == _user_part(recipient))


def extract_message_id(payload: Dict[str, Any] | None) -> Optional[str]:
    """WAHA has returned ids as a string or as a nested dict over time."""
    if not isinstance(payload, dict):
        return None

    def _from(obj: Any) -> Optional[str]:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            mid = obj.get("id")
            if isinstance(mid, str):
                return mid
            if isinstance(mid, dict):
                serialized = mid.get("_serialized") or mid.get("id")
                if isinstance(serialized, str):
                    return serialized
        return None

    return _from(payload) or _from(payload.get("_data"))


class _TTLCache:
    """Small insertion-ordered cache with per-entry TTL."""

    def __init__(self, ttl_seconds: float, max_size: int = 500) -> None:
        self._ttl = ttl_seconds
        self._max = max_size
        self._data: "OrderedDict[str, float]" = OrderedDict()

    def add(self, key: Optional[str]) -> None:
        if not key:
            return
        self._data[key] = time.time()
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def __contains__(self, key: Optional[str]) -> bool:
        if not key:
            return False
        ts = self._data.get(key)
        if ts is None:
            return False
        if time.time() - ts > self._ttl:
            self._data.pop(key, None)
            return False
        return True


class WAHAClient:
    def __init__(self) -> None:
        self.base_url = config.WAHA_BASE_URL.rstrip("/")
        self.session = config.WAHA_SESSION
        self.api_key = config.WAHA_API_KEY

        self._me_id: Optional[str] = None
        self._me_lid: Optional[str] = None
        self._http: Optional[httpx.AsyncClient] = None

        # Load cached session info
        self._load_cached_session()

        # Echo protection: 10 minutes is far longer than a webhook round-trip.
        self._sent_bodies = _TTLCache(ttl_seconds=600)
        self._sent_ids = _TTLCache(ttl_seconds=600)
        # Duplicate webhook protection.
        self._seen_ids = _TTLCache(ttl_seconds=600)

    def _load_cached_session(self) -> None:
        """Load cached me_id and me_lid from disk."""
        cached = session_cache.get_me_info(self.session)
        if cached:
            self._me_id = cached.get("me_id")
            self._me_lid = cached.get("me_lid")
            if self._me_id:
                logger.info(
                    f"Loaded cached session info for {self.session}: me_id={self._me_id}, me_lid={self._me_lid}"
                )

    # ------------------------------------------------------------- helpers
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _client(self) -> httpx.AsyncClient:
        """One pooled client for every call (no per-request TCP/TLS setup)."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=httpx.Timeout(config.WAHA_TIMEOUT, connect=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        retries: int = 0,
        retry_timeouts: bool = False,
        **kwargs,
    ) -> httpx.Response:
        """Request with bounded retries for transient failures.

        Sends are not retried on a read timeout: the first attempt may already
        have reached WhatsApp, and a retry would post the message twice.
        """
        attempts = max(0, retries) + 1
        last_exc: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                resp = await self._client().request(method, path, **kwargs)
                if (
                    resp.status_code in (429, 500, 502, 503, 504)
                    and attempt + 1 < attempts
                ):
                    delay = _backoff(attempt, resp.headers.get("Retry-After"))
                    logger.warning(
                        "WAHA %s %s -> %s; retrying in %.1fs",
                        method,
                        path,
                        resp.status_code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return resp
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ) as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise
                delay = _backoff(attempt)
                logger.warning(
                    "WAHA %s %s failed (%s); retrying in %.1fs",
                    method,
                    path,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except httpx.TimeoutException:
                if not retry_timeouts or attempt + 1 >= attempts:
                    raise
                delay = _backoff(attempt)
                logger.warning(
                    "WAHA %s %s timed out; retrying in %.1fs", method, path, delay
                )
                await asyncio.sleep(delay)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"WAHA {method} {path} failed")

    # -------------------------------------------------------- echo control
    def remember_outgoing(self, text: str, message_id: Optional[str] = None) -> None:
        self._sent_bodies.add(_fingerprint(text))
        self._sent_ids.add(message_id)

    def is_own_reply(self, message_id: Optional[str], body: str) -> bool:
        """Only messages we actually sent count as our own echo.

        The inbound dedupe cache must not be consulted here: a brand new
        question is marked as seen right before this check.
        """
        return message_id in self._sent_ids or _fingerprint(body) in self._sent_bodies

    def mark_seen(self, message_id: Optional[str]) -> bool:
        """Return True if this message id was already handled."""
        if message_id and message_id in self._seen_ids:
            return True
        self._seen_ids.add(message_id)
        return False

    # ------------------------------------------------------- self-chat test
    async def get_me_id(self, force: bool = False) -> Optional[str]:
        if self._me_id and not force:
            return self._me_id

        try:
            resp = await self._request(
                "GET", f"/api/sessions/{self.session}", retries=1, timeout=10.0
            )
            if resp.status_code != 200:
                logger.warning(
                    "Could not read session %s: %s", self.session, resp.status_code
                )
                return None
            data = resp.json()
            me = data.get("me") or {}
            me_id = me.get("id") or me.get("_serialized")
            if isinstance(me_id, dict):
                me_id = me_id.get("_serialized") or me_id.get("id")
            if isinstance(me_id, str) and me_id:
                self._me_id = me_id
                # WAHA 2026 also provides a LID (lid field) which is the
                # account-wide identifier used in message routing.
                if isinstance(me.get("lid"), str):
                    self._me_lid = me["lid"]
                logger.info("Own WhatsApp id: %s (lid: %s)", me_id, self._me_lid)
                # Persist to cache
                session_cache.set_me_info(self.session, me_id, self._me_lid)
                return me_id
        except Exception as exc:  # noqa: BLE001 - never break the webhook
            logger.warning("Could not resolve own WhatsApp id: %s", exc)
        return None

    async def is_self_chat(self, payload: dict) -> bool:
        """True when the message was written in the 'Message Yourself' chat."""
        me_id = await self.get_me_id()
        return _is_self_chat_payload(payload, me_id, self._me_lid)

    # ------------------------------------------------------------ messaging
    async def _post(
        self,
        path: str,
        payload: dict,
        *,
        retries: Optional[int] = None,
        retry_timeouts: bool = False,
    ) -> Optional[dict]:
        if retries is None:
            retries = config.WAHA_SEND_RETRIES
        try:
            resp = await self._request(
                "POST",
                path,
                retries=retries,
                retry_timeouts=retry_timeouts,
                json=payload,
            )
            if resp.status_code >= 400:
                logger.error(
                    "WAHA %s failed (%s): %s",
                    path,
                    resp.status_code,
                    resp.text[:300],
                )
                return None
            try:
                return resp.json()
            except ValueError:
                return {}
        except Exception as exc:  # noqa: BLE001
            logger.error("WAHA %s error: %s", path, exc)
            return None

    async def send_text(self, chat_id: str, text: str) -> bool:
        """Send a text message. The body is fingerprinted before sending so the
        resulting `fromMe` webhook can never be mistaken for a new question."""
        # WhatsApp does not render Markdown: convert it before fingerprinting.
        text = to_whatsapp(text)
        if not text.strip():
            return False

        # Register BEFORE the network call - the echo webhook may arrive first.
        self.remember_outgoing(text)

        data = await self._post(
            "/api/sendText",
            {"session": self.session, "chatId": chat_id, "text": text},
        )
        if data is None:
            WAHA_SENDS.labels(result="error").inc()
            return False

        self.remember_outgoing(text, extract_message_id(data))
        WAHA_SENDS.labels(result="sent").inc()
        logger.info("Sent reply to %s (%d chars)", chat_id, len(text))
        return True

    async def send_chunk(self, chat_id: str, chunk: str) -> bool:
        """Send a streaming chunk to WhatsApp.

        Note: This uses the same sendText API. For true streaming, you'd need
        a different approach (e.g., sending to a buffer and flushing periodically).
        """
        return await self.send_text(chat_id, chunk)

    async def send_long_text(self, chat_id: str, text: str, limit: int = 3500) -> None:
        """Split long answers on paragraph/line boundaries for readability."""
        for chunk in _split_message(text, limit):
            if not await self.send_text(chat_id, chunk):
                break

    async def start_typing(self, chat_id: str) -> None:
        await self._post(
            "/api/startTyping",
            {"session": self.session, "chatId": chat_id},
            retries=0,
        )

    async def stop_typing(self, chat_id: str) -> None:
        await self._post(
            "/api/stopTyping",
            {"session": self.session, "chatId": chat_id},
            retries=0,
        )

    # -------------------------------------------------------------- session
    async def session_status(self) -> dict:
        try:
            resp = await self._request(
                "GET", f"/api/sessions/{self.session}", retries=0, timeout=5.0
            )
            if resp.status_code != 200:
                return {"status": "unknown", "http_status": resp.status_code}
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unreachable", "error": str(exc)}

    async def start_session(self) -> dict:
        """Start (or resume) the WAHA session. Safe to call repeatedly."""
        data = await self._post(f"/api/sessions/{self.session}/start", {}, retries=1)
        if data is None:
            # Older/alternative WAHA builds expose POST /api/sessions instead.
            data = await self._post(
                "/api/sessions", {"name": self.session, "start": True}, retries=1
            )
        return data or {}

    async def restart_session(self) -> dict:
        """Restart a FAILED/expired session so a fresh QR can be generated."""
        data = await self._post(f"/api/sessions/{self.session}/restart", {}, retries=1)
        if data is None:
            await self._post(f"/api/sessions/{self.session}/stop", {}, retries=1)
            data = await self._post(
                f"/api/sessions/{self.session}/start", {}, retries=1
            )
        return data or {}

    async def wait_for_qr(
        self, attempts: int = 20, delay: float = 0.7
    ) -> Optional[bytes]:
        """Poll WAHA until a QR image is available (it appears a few seconds
        after the session starts)."""
        for _ in range(attempts):
            image = await self.qr_image()
            if image:
                return image
            await asyncio.sleep(delay)
        return None

    async def qr_image(self) -> Optional[bytes]:
        """Fetch the pairing QR as PNG (dashboard is the fallback)."""
        try:
            resp = await self._request(
                "GET",
                f"/api/{self.session}/auth/qr",
                retries=1,
                timeout=15.0,
                params={"format": "image"},
                headers={"Accept": "image/png"},
            )
            if resp.status_code == 200 and resp.headers.get(
                "content-type", ""
            ).startswith("image"):
                return resp.content
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch QR: %s", exc)
        return None

    def clear_cached_session(self) -> None:
        """Clear cached session info (useful when session is restarted)."""
        self._me_id = None
        self._me_lid = None
        session_cache.clear_session(self.session)


def _split_message(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        else:
            cut += 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


waha = WAHAClient()
