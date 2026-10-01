"""Configuration for the TSPI MCP server (env-driven).

Nothing here changes the engine — these are client-side settings for how the MCP
layer reaches the engine and how strict the governance gates are.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Load variables from .env into the process environment.
# IMPORTANT: load the package's OWN .env explicitly (apps/tspi-mcp/.env) so auth settings
# are found no matter what cwd uvicorn/cloudflared is launched from. A bare load_dotenv()
# only searches the cwd + parents, so running from the repo root would silently miss it and
# leave TSPI_MCP_AUTH="none" (public server with NO auth). Real env vars (Docker/Coolify)
# still win — load_dotenv does not override variables already set in the environment.
_PKG_ENV = Path(__file__).resolve().parent.parent / ".env"
if _PKG_ENV.is_file():
    load_dotenv(_PKG_ENV)
else:
    load_dotenv()


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Where the standalone FastAPI engine lives.
    engine_url: str = os.getenv(
        "TSPI_ENGINE_URL",
        "http://localhost:8000",
    )

    # Optional bearer token the MCP server presents to the engine.
    engine_token: str | None = os.getenv("TSPI_ENGINE_TOKEN") or None

    # Authenticated clinician using this chat session.
    clinician_id: str = os.getenv(
        "TSPI_CLINICIAN_ID",
        "unknown-clinician",
    )

    # Role + clinic the MCP acts as.
    clinician_role: str = os.getenv(
        "TSPI_CLINICIAN_ROLE",
        "clinician",
    )

    clinic_id: str | None = os.getenv("TSPI_CLINIC_ID") or None

    # De-identification.
    strict_deident: bool = _bool(
        "TSPI_STRICT_DEIDENT",
        True,
    )

    # Transport: "stdio" or "streamable-http".
    transport: str = os.getenv(
        "TSPI_MCP_TRANSPORT",
        "stdio",
    )

    http_timeout_s: float = float(
        os.getenv("TSPI_HTTP_TIMEOUT", "60")
    )

    # --- OAuth for public/remote MCP surface ---

    # none = no incoming auth
    # jwt = validate bearer JWTs
    # workos = WorkOS AuthKit
    mcp_auth: str = os.getenv(
        "TSPI_MCP_AUTH",
        "none",
    )

    mcp_base_url: str | None = (
        os.getenv("TSPI_MCP_BASE_URL") or None
    )

    jwt_jwks_uri: str | None = (
        os.getenv("TSPI_JWT_JWKS_URI") or None
    )

    jwt_issuer: str | None = (
        os.getenv("TSPI_JWT_ISSUER") or None
    )

    jwt_audience: str | None = (
        os.getenv("TSPI_JWT_AUDIENCE") or None
    )

    workos_authkit_domain: str | None = (
        os.getenv("TSPI_WORKOS_AUTHKIT_DOMAIN") or None
    )

    # Token claim names carrying clinician identity.
    claim_user: str = os.getenv(
        "TSPI_CLAIM_USER",
        "sub",
    )

    claim_role: str = os.getenv(
        "TSPI_CLAIM_ROLE",
        "role",
    )

    claim_clinic: str = os.getenv(
        "TSPI_CLAIM_CLINIC",
        "clinic_id",
    )

    # Optional human-identity claims (for report attribution: who created/approved a plan).
    claim_email: str = os.getenv("TSPI_CLAIM_EMAIL", "email")
    claim_first_name: str = os.getenv("TSPI_CLAIM_FIRST_NAME", "first_name")
    claim_last_name: str = os.getenv("TSPI_CLAIM_LAST_NAME", "last_name")

    # --- First-party apps that log users in through the same AuthKit (e.g. TSPI Digital chat) ---
    # Their access tokens carry aud=<WorkOS environment client ID> instead of this server's
    # resource URL. Accepted ONLY when the token's client_id is in TSPI_ALLOWED_CLIENT_IDS.
    # Both are comma-separated; leave empty to accept only resource-bound tokens (default).
    extra_audiences: tuple[str, ...] = tuple(
        a.strip() for a in os.getenv("TSPI_EXTRA_AUDIENCES", "").split(",") if a.strip()
    )
    allowed_client_ids: tuple[str, ...] = tuple(
        c.strip() for c in os.getenv("TSPI_ALLOWED_CLIENT_IDS", "").split(",") if c.strip()
    )


settings = Settings()
