"""Logging + full-traceback capture for the MCP server.

* configure_logging() — sets up the `tspi_mcp` logger on STDERR (stdout is the MCP stdio channel and
  must never carry logs). TSPI_MCP_LOG_FORMAT=json emits one JSON object per line with the
  traceback in a single field (log shippers that split on newlines keep it together).
* ToolErrorLoggingMiddleware — any exception escaping a tool/request is logged with its FULL
  traceback, the MCP method, the tool name and an error_id, then re-raised so FastMCP still
  returns a normal tool error to the client.

PHI guard: tool ARGUMENTS are never logged (they carry case data) — only method/tool name/ids.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext

log = logging.getLogger("tspi_mcp.errors")

_TEXT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class JsonFormatter(logging.Formatter):
    _STD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self._STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            out["traceback"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging() -> None:
    level = os.getenv("TSPI_MCP_LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("TSPI_MCP_LOG_FORMAT", "text").lower()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT))
    pkg = logging.getLogger("tspi_mcp")
    pkg.handlers[:] = [handler]
    pkg.setLevel(level)
    pkg.propagate = False          # don't double-log through FastMCP's own root/rich handlers


class ToolErrorLoggingMiddleware(Middleware):
    async def on_message(self, context: MiddlewareContext, call_next):
        try:
            return await call_next(context)
        except Exception as exc:
            error_id = uuid.uuid4().hex[:12]
            method = context.method or "?"
            tool = getattr(context.message, "name", None)
            log.error("Unhandled %s in %s%s: %s [error_id=%s]",
                      type(exc).__name__, method, f" (tool={tool})" if tool else "", exc, error_id,
                      exc_info=exc,
                      extra={"error_id": error_id, "mcp_method": method, "tool": tool})
            if tool:   # surface the id to the chat so a clinician can report it
                raise ToolError(f"{exc} (error_id={error_id}; see MCP server logs)") from exc
            raise
