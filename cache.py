"""Caching layer for RAG queries and session data."""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import diskcache

from config import config
from metrics import CACHE_EVENTS

logger = logging.getLogger(__name__)


class QueryCache:
    """Multi-level LRU cache for query responses.

    Level 1: Exact match in-memory LRU (very fast, for identical questions)
    Level 2: Disk-based cache with TTL (persists across restarts)

    A hit refreshes the entry's position and timestamp, so frequently asked
    questions stay cached and the least recently used entry is evicted first.
    """

    def __init__(self):
        self.enabled = config.CACHE_ENABLED
        # key -> (value, last_used_at)
        self.exact_cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
        self.exact_cache_size = config.EXACT_CACHE_SIZE
        self.hits = 0
        self.misses = 0

        # Disk cache for persistence
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.disk_cache = diskcache.Cache(str(config.CACHE_DIR))
        self.cache_ttl = config.CACHE_TTL

    def make_key(self, question: str) -> str:
        """Create a normalized cache key from the question.

        The session name, model and collection are part of the key so that
        bots sharing a cache directory (the Compose setup mounts ./cache into
        every bot) never serve each other's answers, and changing the model
        invalidates answers produced by the previous one.
        """
        normalized = " ".join(question.lower().strip().split())
        scoped = (
            f"{config.WAHA_SESSION}:{config.LLM_MODEL}:"
            f"{config.QDRANT_COLLECTION}:{normalized}"
        )
        return hashlib.sha256(scoped.encode()).hexdigest()

    def get(self, question: str) -> Optional[str]:
        """Get cached response for a question."""
        if not self.enabled:
            return None

        key = self.make_key(question)
        now = time.time()

        # Level 1: Exact in-memory LRU
        if key in self.exact_cache:
            value, timestamp = self.exact_cache[key]
            if now - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit (exact): {question[:50]}...")
                # Refresh recency and TTL on read.
                self.exact_cache.move_to_end(key)
                self.exact_cache[key] = (value, now)
                self.hits += 1
                CACHE_EVENTS.labels(result="hit").inc()
                return value
            del self.exact_cache[key]

        # Level 2: Disk cache
        try:
            value = self.disk_cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit (disk): {question[:50]}...")
                # Promote to exact cache and extend the disk TTL.
                self._add_to_exact(key, value)
                try:
                    self.disk_cache.touch(key, expire=self.cache_ttl)
                except Exception:  # noqa: BLE001 - touch is best effort
                    pass
                self.hits += 1
                CACHE_EVENTS.labels(result="hit").inc()
                return value
        except Exception as e:
            logger.warning(f"Disk cache read error: {e}")

        self.misses += 1
        CACHE_EVENTS.labels(result="miss").inc()
        return None

    def set(self, question: str, response: str) -> None:
        """Cache a response for a question."""
        if not self.enabled:
            return

        key = self.make_key(question)

        # Add to exact cache
        self._add_to_exact(key, response)

        # Add to disk cache
        try:
            self.disk_cache.set(key, response, expire=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Disk cache write error: {e}")

    def _add_to_exact(self, key: str, value: str) -> None:
        """Add to the in-memory LRU, evicting the least recently used entry."""
        self.exact_cache[key] = (value, time.time())
        self.exact_cache.move_to_end(key)
        while len(self.exact_cache) > self.exact_cache_size:
            self.exact_cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all caches."""
        self.exact_cache.clear()
        try:
            self.disk_cache.clear()
        except Exception as e:
            logger.warning(f"Disk cache clear error: {e}")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "exact_cache_size": len(self.exact_cache),
            "exact_cache_max": self.exact_cache_size,
            "disk_cache_size": len(self.disk_cache),
            "ttl": self.cache_ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


class SessionCache:
    """Persistent cache for session data (me_id, me_lid) to avoid WAHA API calls on restart."""

    def __init__(self):
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.disk_cache = diskcache.Cache(str(config.CACHE_DIR))

    def get_me_info(self, session_name: str) -> Optional[Dict[str, str]]:
        """Get cached me_id and me_lid for a session."""
        key = f"me_info:{session_name}"
        try:
            return self.disk_cache.get(key)
        except Exception as e:
            logger.warning(f"Session cache read error: {e}")
            return None

    def set_me_info(
        self, session_name: str, me_id: str, me_lid: Optional[str] = None
    ) -> None:
        """Cache me_id and me_lid for a session."""
        key = f"me_info:{session_name}"
        data = {"me_id": me_id}
        if me_lid:
            data["me_lid"] = me_lid
        try:
            self.disk_cache.set(key, data)
        except Exception as e:
            logger.warning(f"Session cache write error: {e}")

    def clear_session(self, session_name: str) -> None:
        """Clear cached session info."""
        key = f"me_info:{session_name}"
        try:
            self.disk_cache.delete(key)
        except Exception as e:
            logger.warning(f"Session cache delete error: {e}")


# Global instances
query_cache = QueryCache()
session_cache = SessionCache()
