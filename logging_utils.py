"""Request-scoped logging helpers.

Every HTTP request gets an id (from `X-Request-ID` or generated), which is
attached to all log records produced while handling it. Background work started
from the webhook carries the id in the payload so a question can be followed
from WAHA's POST through the queue to the WAHA reply.
"""

import contextvars
import logging
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"


def new_request_id() -> str:
    """Short, log-friendly request id."""
    return uuid.uuid4().hex[:12]


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, with request ids on every line."""
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT)
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())
