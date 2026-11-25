"""Caching layer for RAG queries and session data."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import diskcache

from config import config

logger = logging.getLogger(__name__)


class QueryCache:
    """Multi-level cache for query responses.
    
    Level 1: Exact match in-memory cache (very fast, for identical questions)
    Level 2: Disk-based cache with TTL (persists across restarts)
    """

    def __init__(self):
        self.enabled = config.CACHE_ENABLED
        self.exact_cache: Dict[str, tuple[str, float]] = {}  # key -> (value, timestamp)
        self.exact_cache_size = config.EXACT_CACHE_SIZE
        
        # Disk cache for persistence
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.disk_cache = diskcache.Cache(str(config.CACHE_DIR))
        self.cache_ttl = config.CACHE_TTL

    def _make_key(self, question: str) -> str:
        """Create a normalized cache key from the question.

        The WAHA session name is part of the key so that bots sharing a cache
        directory (the Compose setup mounts ./cache into every bot) never serve
        each other's answers.
        """
        # Normalize: lowercase, strip, remove extra whitespace
        normalized = " ".join(question.lower().strip().split())
        scoped = f"{config.WAHA_SESSION}:{normalized}"
        return hashlib.sha256(scoped.encode()).hexdigest()

    def get(self, question: str) -> Optional[str]:
        """Get cached response for a question."""
        if not self.enabled:
            return None

        key = self._make_key(question)
        now = time.time()

        # Level 1: Exact in-memory cache
        if key in self.exact_cache:
            value, timestamp = self.exact_cache[key]
            if now - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit (exact): {question[:50]}...")
                return value
            else:
                del self.exact_cache[key]

        # Level 2: Disk cache
        try:
            value = self.disk_cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit (disk): {question[:50]}...")
                # Promote to exact cache
                self._add_to_exact(key, value)
                return value
        except Exception as e:
            logger.warning(f"Disk cache read error: {e}")

        return None

    def set(self, question: str, response: str) -> None:
        """Cache a response for a question."""
        if not self.enabled:
            return

        key = self._make_key(question)
        
        # Add to exact cache
        self._add_to_exact(key, response)
        
        # Add to disk cache
        try:
            self.disk_cache.set(key, response, expire=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Disk cache write error: {e}")

    def _add_to_exact(self, key: str, value: str) -> None:
        """Add to exact cache with LRU eviction."""
        now = time.time()
        self.exact_cache[key] = (value, now)
        # Evict oldest if over size limit
        while len(self.exact_cache) > self.exact_cache_size:
            oldest_key = min(self.exact_cache.keys(), key=lambda k: self.exact_cache[k][1])
            del self.exact_cache[oldest_key]

    def clear(self) -> None:
        """Clear all caches."""
        self.exact_cache.clear()
        try:
            self.disk_cache.clear()
        except Exception as e:
            logger.warning(f"Disk cache clear error: {e}")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "exact_cache_size": len(self.exact_cache),
            "exact_cache_max": self.exact_cache_size,
            "disk_cache_size": len(self.disk_cache),
            "ttl": self.cache_ttl,
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

    def set_me_info(self, session_name: str, me_id: str, me_lid: Optional[str] = None) -> None:
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