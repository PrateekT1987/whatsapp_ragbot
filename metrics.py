"""Prometheus metrics for the bot.

Exposed at `/metrics`. Labels are deliberately low-cardinality: routes are
template paths, outcomes are a fixed vocabulary, and chat ids are never used as
labels (a WhatsApp account has few chats, but ids are still private).
"""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "rag_http_requests_total",
    "HTTP requests handled",
    ["method", "path", "status"],
)
HTTP_DURATION = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

WEBHOOK_EVENTS = Counter(
    "rag_webhook_events_total",
    "Incoming WAHA webhook outcomes",
    ["result"],
)

QUERY_RESULTS = Counter(
    "rag_query_results_total",
    "RAG query outcomes",
    ["result"],
)
QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "Time spent answering a question",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)

CACHE_EVENTS = Counter(
    "rag_cache_events_total",
    "Query cache lookups",
    ["result"],
)

LLM_RETRIES = Counter(
    "rag_llm_retries_total",
    "LLM calls retried after a transient failure",
)
LLM_ERRORS = Counter(
    "rag_llm_errors_total",
    "LLM calls that ultimately failed",
    ["kind"],
)

WAHA_SENDS = Counter(
    "rag_waha_sends_total",
    "Outgoing WhatsApp sends",
    ["result"],
)
