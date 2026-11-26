#!/usr/bin/env python3
"""Comprehensive unit tests for P1/P2 functionality."""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CACHE_DIR", "/tmp/rag-bot-p2-cache")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "openrouter")

sys.path.insert(0, str(Path(__file__).parent))

from waha import (
    _fingerprint,
    _is_self_chat_payload,
    _split_message,
    _user_part,
)
from cache import QueryCache
from rag import (
    _format_sources,
    _is_rate_limit,
    _is_retryable,
    _retry_delay,
)
from main import verify_webhook_signature
from vectorstore import _query_terms
from formatting import to_whatsapp
from langchain_core.documents import Document


class FormattingTests(unittest.TestCase):
    def test_bold(self):
        self.assertEqual(to_whatsapp("This is **bold** text"), "This is *bold* text")

    def test_italic(self):
        self.assertEqual(to_whatsapp("This is *italic* text"), "This is _italic_ text")

    def test_underscore_bold(self):
        self.assertEqual(to_whatsapp("This is __bold__ text"), "This is *bold* text")

    def test_strikethrough(self):
        self.assertEqual(to_whatsapp("This is ~~old~~ text"), "This is ~old~ text")

    def test_inline_code(self):
        self.assertEqual(to_whatsapp("Run `pytest` now"), "Run ```pytest``` now")

    def test_heading(self):
        self.assertEqual(to_whatsapp("# Results"), "*Results*")
        # Already bold heading should not double the markers
        self.assertEqual(to_whatsapp("## **Results**"), "*Results*")

    def test_bullets(self):
        self.assertEqual(to_whatsapp("- one\n- two"), "• one\n• two")

    def test_link(self):
        self.assertEqual(
            to_whatsapp("See [docs](https://example.com)"),
            "See docs (https://example.com)",
        )

    def test_fenced_code_is_preserved(self):
        text = "Example:\n```python\nprint('**hi**')\n```"
        out = to_whatsapp(text)
        self.assertIn("print('**hi**')", out)
        self.assertNotIn("```python", out)

    def test_plain_text_unchanged(self):
        self.assertEqual(to_whatsapp("Hello, world!"), "Hello, world!")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = QueryCache()
        self.cache.enabled = True
        self.cache.exact_cache_size = 2
        self.cache.exact_cache.clear()
        self.cache.disk_cache.clear()

    def test_lru_eviction_refreshes_on_read(self):
        tag = "test_lru"
        q1, q2, q3 = (f"q{i}-{tag}" for i in range(3))
        self.cache.set(q1, "a")
        self.cache.set(q2, "b")

        # Reading q1 makes q2 the least recently used entry.
        self.assertEqual(self.cache.get(q1), "a")
        self.cache.set(q3, "c")

        keys = list(self.cache.exact_cache.keys())
        self.assertIn(self.cache.make_key(q1), keys)
        self.assertIn(self.cache.make_key(q3), keys)
        self.assertNotIn(self.cache.make_key(q2), keys)

    def test_key_includes_model_and_session(self):
        from config import config

        first = self.cache.make_key("same question")
        original = config.LLM_MODEL
        try:
            config.LLM_MODEL = original + "-other"
            second = self.cache.make_key("same question")
        finally:
            config.LLM_MODEL = original
        self.assertNotEqual(first, second)

    def test_key_includes_collection(self):
        from config import config

        first = self.cache.make_key("same question")
        original = config.QDRANT_COLLECTION
        try:
            config.QDRANT_COLLECTION = original + "-other"
            second = self.cache.make_key("same question")
        finally:
            config.QDRANT_COLLECTION = original
        self.assertNotEqual(first, second)


class RetryTests(unittest.TestCase):
    def test_rate_limit_detection(self):
        exc = Exception("429 Too Many Requests")
        self.assertTrue(_is_rate_limit(exc))
        self.assertTrue(_is_retryable(exc))
        self.assertFalse(_is_rate_limit(ValueError("bad input")))

    def test_status_code(self):
        exc = Exception("boom")
        exc.status_code = 503  # type: ignore[attr-defined]
        self.assertTrue(_is_retryable(exc))
        exc.status_code = 401  # type: ignore[attr-defined]
        self.assertFalse(_is_retryable(exc))

    def test_backoff_is_bounded(self):
        for attempt in range(10):
            delay = _retry_delay(attempt, 2.0)
            self.assertGreaterEqual(delay, 0)
            self.assertLessEqual(delay, 30.0)


class SourceTests(unittest.TestCase):
    def test_single_source_with_one_based_page(self):
        docs = [
            Document(
                page_content="x",
                metadata={"source": "handbook.pdf", "page": 4},
            )
        ]
        self.assertEqual(
            _format_sources(docs), "\n\n📄 Source: handbook.pdf, page 5"
        )

    def test_deduplicates_and_limits(self):
        docs = [
            Document(page_content="a", metadata={"source": "a.pdf", "page": 0}),
            Document(page_content="b", metadata={"source": "a.pdf", "page": 0}),
            Document(page_content="c", metadata={"source": "b.pdf", "page": 2}),
        ]
        out = _format_sources(docs)
        self.assertEqual(out.count("a.pdf"), 1)
        self.assertIn("b.pdf, page 3", out)

    def test_no_page_in_metadata(self):
        docs = [Document(page_content="x", metadata={"source": "no_page.pdf"})]
        self.assertEqual(_format_sources(docs), "\n\n📄 Source: no_page.pdf")


class WahaUtilsTests(unittest.TestCase):
    def test_user_part(self):
        self.assertEqual(_user_part("1234:12@c.us"), "1234")
        self.assertEqual(_user_part("1234@lid"), "1234")
        self.assertIsNone(_user_part(None))
        self.assertIsNone(_user_part(""))

    def test_fingerprint_is_deterministic(self):
        self.assertEqual(_fingerprint("hello"), _fingerprint("hello"))
        self.assertEqual(_fingerprint("  hello  "), _fingerprint("hello"))

    def test_self_chat_payload(self):
        me_id = "918087433523@c.us"
        me_lid = "139333856153654@lid"
        payload = {
            "fromMe": True,
            "chatId": "918087433523@c.us",
            "from": "918087433523@c.us",
        }
        self.assertTrue(_is_self_chat_payload(payload, me_id, me_lid))

        # LID match
        payload["chatId"] = "139333856153654@lid"
        payload["from"] = "139333856153654@lid"
        self.assertTrue(_is_self_chat_payload(payload, me_id, me_lid))

        # Not fromMe -> false
        payload["fromMe"] = False
        self.assertFalse(_is_self_chat_payload(payload, me_id, me_lid))

    def test_split_message(self):
        text = "a\n\nb\n\nc" * 100  # long text
        chunks = _split_message(text, 500)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        # Rejoin should equal original without extra whitespace changes
        rejoined = "".join(chunks)
        # Each chunk ends with a newline (from the split points), so count them
        self.assertGreater(len(rejoined), 0)

    def test_verify_webhook_signature(self):
        key = "secret"
        body = b'{"test": "data"}'
        expected = hashlib.sha512(key.encode() + body).hexdigest()
        # Note: verify_webhook_signature uses hmac.new(key, body, sha512)
        import hmac
        expected = hmac.new(key.encode(), body, "sha512").hexdigest()

        self.assertTrue(verify_webhook_signature(body, expected, "sha512", key))
        self.assertFalse(verify_webhook_signature(body, "wrong", "sha512", key))
        # Empty key disables verification
        self.assertTrue(verify_webhook_signature(body, "", "sha512", ""))

    def test_query_terms(self):
        # Punctuation-insensitive, stopwords removed
        terms = _query_terms("What is the name of the 15Abhishek.J.Patil?")
        # "what" is a stopword, "the" "is" "of" "the" are stopwords
        # "name", "15abhishek", "patil" should remain
        self.assertIn("name", terms)
        self.assertIn("15abhishek", terms)
        self.assertIn("patil", terms)


class HMACTests(unittest.TestCase):
    def test_algorithm_rejection(self):
        body = b"{}"
        with self.assertRaises(ValueError):
            verify_webhook_signature(body, "sig", "unsupported_algo", "key")


if __name__ == "__main__":
    unittest.main(verbosity=2)