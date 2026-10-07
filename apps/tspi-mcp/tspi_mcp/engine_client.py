"""Async HTTP client for the standalone TSPI FastAPI engine.

This is the ONLY thing that talks to the engine, and it talks to it exactly as any other REST
client would — the engine is unchanged and standalone. Errors are turned into actionable
messages the chat model can relay to the clinician.
"""
from __future__ import annotations

import json as _json
import logging
from typing import Any

import httpx

from .config import settings

_log = logging.getLogger("tspi_mcp.engine")

# Endpoints that run the LLM inside the engine -> use the longer report timeout.
_SLOW_PATHS = {"/report", "/extract"}


class EngineError(RuntimeError):
    """A failed engine call, with an actionable message."""


def _headers() -> dict[str, str]:
    # Service token authenticates the MCP to the engine; X-TSPI-* forward the clinician identity
    # the engine trusts for RBAC + audit. Identity is PER-REQUEST from the validated OAuth token
    # (P7) when the MCP is remote/authenticated, else the static pilot identity (stdio).
    from .identity import current_identity_ext
    ident = current_identity_ext()
    h = {"Accept": "application/json"}
    if settings.engine_token:
        h["Authorization"] = f"Bearer {settings.engine_token}"
    h["X-TSPI-User-Id"] = ident["user"]
    h["X-TSPI-Role"] = ident["role"]
    h["X-TSPI-Source"] = "mcp"
    if ident["clinic"]:
        h["X-TSPI-Clinic-Id"] = ident["clinic"]
    # Human-identity for report attribution (who created/approved). Optional; omitted if absent.
    if ident["name"]:
        h["X-TSPI-Name"] = ident["name"]
    if ident["email"]:
        h["X-TSPI-Email"] = ident["email"]
    return h


def _explain(status: int, body: str) -> str:
    if status == 403:
        return ("The engine refused the request (403). The most common cause is missing consent: "
                "set consent_ai_analysis=true after confirming the patient consented.")
    if status == 404:
        return "The engine could not find that record (404). Check the report_id / case_id."
    if status == 422:
        try:
            data = _json.loads(body)
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("error") == "pii_detected":
            # Engine's fail-closed P4 guard (message is already masked / PII-free).
            return ("The engine blocked this request because patient identifiers (e.g. a phone "
                    "number, ID or email) were detected in the case data (422). Remove them from "
                    f"the symptoms/notes and resend. Engine detail: {data.get('detail', '')[:300]}")
        return f"The engine rejected the input as invalid (422): {body[:400]}"
    if status >= 500:
        try:
            data = _json.loads(body)
        except ValueError:
            data = {}
        eid = data.get("error_id") if isinstance(data, dict) else None
        if eid:
            return (f"The TSPI engine hit an internal error (HTTP {status}). Share error_id "
                    f"{eid} with the engine admin to find it in the logs.")
    return f"The engine returned HTTP {status}: {body[:400]}"


def _log_http_error(method: str, path: str, resp: httpx.Response) -> None:
    """Log a failed engine call. The engine's traceback lives in the ENGINE log — we log its
    error_id/request_id so the two can be joined. Request bodies are never logged (case data);
    4xx response bodies are not logged either (validation errors can echo input), except the
    engine's PII-guard message, which is already masked."""
    try:
        data = resp.json()
    except ValueError:
        data = {}
    data = data if isinstance(data, dict) else {}
    ids = {"status": resp.status_code,
           "engine_error_id": data.get("error_id") or resp.headers.get("x-error-id"),
           "engine_request_id": data.get("request_id") or resp.headers.get("x-request-id")}
    if resp.status_code >= 500:
        _log.error("Engine %s %s -> HTTP %s (engine error_id=%s request_id=%s) body=%s",
                   method, path, resp.status_code, ids["engine_error_id"],
                   ids["engine_request_id"], resp.text[:500], extra=ids)
    elif data.get("error") == "pii_detected":
        _log.warning("Engine %s %s -> 422 PII guard: %s", method, path,
                     str(data.get("detail"))[:300], extra=ids)
    else:
        _log.warning("Engine %s %s -> HTTP %s (request_id=%s)", method, path, resp.status_code,
                     ids["engine_request_id"], extra=ids)


async def _request(method: str, path: str, *, json: Any = None,
                   params: dict | None = None) -> Any:
    url = f"{settings.engine_url.rstrip('/')}{path}"
    timeout = settings.report_timeout_s if path in _SLOW_PATHS else settings.http_timeout_s
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, json=json, params=params, headers=_headers())
    except httpx.TimeoutException as e:
        _log.error("Engine call timed out after %.0fs: %s %s (%s)", timeout, method, path,
                   type(e).__name__, exc_info=True)
        raise EngineError(
            f"The TSPI engine did not respond within {timeout:.0f}s ({method} {path}). Report "
            "generation runs an LLM and can be slow — try again shortly. If it keeps happening, "
            "raise TSPI_REPORT_TIMEOUT on the MCP (keep it above the engine's REPORT_LLM_BUDGET_S)."
        ) from e
    except httpx.ConnectError as e:
        _log.error("Engine unreachable: %s %s", method, path, exc_info=True)
        raise EngineError(
            f"Cannot reach the TSPI engine at {settings.engine_url}. Is it running? "
            f"(set TSPI_ENGINE_URL if it lives elsewhere.) Details: {e}"
        ) from e
    except httpx.HTTPError as e:
        _log.error("Network error calling engine: %s %s", method, path, exc_info=True)
        raise EngineError(f"Network error calling the TSPI engine: {e}") from e

    if resp.status_code >= 400:
        _log_http_error(method, path, resp)
        raise EngineError(_explain(resp.status_code, resp.text))
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


async def get(path: str, params: dict | None = None) -> Any:
    return await _request("GET", path, params=params)


async def post(path: str, json: Any = None, params: dict | None = None) -> Any:
    return await _request("POST", path, json=json, params=params)
