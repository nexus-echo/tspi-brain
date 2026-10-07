"""Logging setup + unhandled-exception capture.

Why: uncaught exceptions were surfacing only as uvicorn's terse "Exception in ASGI application"
with no exception type/message, which made 500s (e.g. POST /report) impossible to debug.

What this does:
  * configure_logging() — timestamped, named log lines. TSPI_LOG_FORMAT=json emits one JSON
    object per line with the full traceback in a single field, so log shippers that split on
    newlines (CloudWatch, Loki, etc.) keep the whole traceback together.
  * ErrorLoggingMiddleware — catches any unhandled exception, logs the FULL traceback together
    with method, path, request id and an error_id, and returns a clean JSON 500 that carries the
    same error_id (also in the X-Error-ID header) so a client-side failure can be matched to the
    exact log entry.

PHI guard: request bodies, query strings and headers are NEVER logged here — only method, path
and ids. (Exception messages themselves are logged; keep PII out of raised messages.)
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone

from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("tspi.errors")

_TEXT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class JsonFormatter(logging.Formatter):
    """One JSON object per line; traceback kept inside the record (no raw newlines)."""

    _STD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():          # fields passed via extra={...}
            if k not in self._STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            out["traceback"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    fmt = (fmt or "text").lower()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


class ErrorLoggingMiddleware:
    """Pure-ASGI middleware (safe for streaming responses / background tasks)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [(b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:  # noqa: BLE001 — this IS the last-resort handler
            error_id = uuid.uuid4().hex[:12]
            method, path = scope.get("method", "?"), scope.get("path", "?")
            # exc_info -> full traceback (text: multi-line after the message; json: "traceback" field)
            log.error(
                "Unhandled %s on %s %s: %s [error_id=%s request_id=%s]",
                type(exc).__name__, method, path, exc, error_id, request_id,
                exc_info=exc,
                extra={"error_id": error_id, "request_id": request_id,
                       "method": method, "path": path},
            )
            if response_started:
                raise   # headers already sent; can't swap in a JSON body — let the server abort
            body = json.dumps({"detail": "Internal server error", "error_id": error_id,
                               "request_id": request_id}).encode()
            await send({"type": "http.response.start", "status": 500, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"x-error-id", error_id.encode()),
                (b"x-request-id", request_id.encode()),
            ]})
            await send({"type": "http.response.body", "body": body})
