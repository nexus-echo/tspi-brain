"""FastAPI entrypoint for tspi_ai_brain.

Run locally:
    uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.deid import PIILeak
from app.error_logging import ErrorLoggingMiddleware, configure_logging

configure_logging(settings.tspi_log_level, settings.tspi_log_format)

app = FastAPI(
    title="TSPI AI Brain",
    description=(
        "The Standardized Network Phytochemicals Intelligence Platform — "
        "AI diagnostic engine behind MiHealth. Maps patient data onto the "
        "3 Keys / 9 Steps / 12 Domains / 39 Axes model and generates a "
        "de-identified TSPI Case Report."
    ),
    version="0.1.0",
)

# Outermost app-level handler: logs full traceback + error_id for any unhandled exception.
app.add_middleware(ErrorLoggingMiddleware)
app.include_router(router)


@app.exception_handler(PIILeak)
async def pii_leak_handler(request: Request, exc: PIILeak) -> JSONResponse:
    """Fail-closed P4 guard tripped: the request carries identifiable data that would reach the LLM.
    A client-correctable input problem, not a server fault -> 422 (message is PII-free/masked)."""
    logging.getLogger("tspi.deid").warning("PII guard blocked %s %s: %s",
                                           request.method, request.url.path, exc)
    return JSONResponse(status_code=422,
                        content={"detail": str(exc), "error": "pii_detected"})


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "env": settings.tspi_env, "version": app.version}
