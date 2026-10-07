"""FastAPI entrypoint for tspi_ai_brain.

Run locally:
    uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
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


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "env": settings.tspi_env, "version": app.version}
