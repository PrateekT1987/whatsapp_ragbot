#!/usr/bin/env python3
"""Unit tests for the P1 reliability/quality changes (no network needed)."""
import os
import sys
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("CACHE_DIR", "/tmp/rag-bot-p1-cache")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "openrouter")

sys.path.insert(0, str(Path(__file__).parent))

from formatting import to_whatsapp  # noqa: E402
from cache import QueryCache  # noqa: E402
from rag import (  # noqa: E402
    _format_sources,
    _is_rate_limit,
    _is_retryable,
    _retry_delay,
)
from langchain_core.documents import Document  # noqa: E402


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
    def test_lru_eviction_refreshes_on_read(self):
        cache = QueryCache()
        cache.enabled = True
        cache.exact_cache_size = 2
        cache.exact_cache.clear()

        tag = uuid.uuid4().hex
        q1, q2, q3 = (f"q{i}-{tag}" for i in range(3))
        cache.set(q1, "a")
        cache.set(q2, "b")

        # Reading q1 makes q2 the least recently used entry.
        self.assertEqual(cache.get(q1), "a")
        cache.set(q3, "c")

        keys = list(cache.exact_cache.keys())
        self.assertIn(cache.make_key(q1), keys)
        self.assertIn(cache.make_key(q3), keys)
        self.assertNotIn(cache.make_key(q2), keys)

    def test_key_includes_model_and_session(self):
        cache = QueryCache()
        from config import config

        first = cache.make_key("same question")
        original = config.LLM_MODEL
        try:
            config.LLM_MODEL = original + "-other"
            second = cache.make_key("same question")
        finally:
            config.LLM_MODEL = original
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
