"""P7 — OAuth auth provider + per-request clinician identity for the (public) MCP surface.

Fail-closed: if TSPI_MCP_AUTH is set but its config is incomplete, startup raises rather than
running an UNAUTHENTICATED public MCP. Local stdio (auth=none) keeps the static-identity pilot model.
"""
from __future__ import annotations

from .config import settings


def build_auth():
    """Return a FastMCP auth provider from env, or None for local stdio (no incoming auth)."""
    mode = (settings.mcp_auth or "none").lower()
    if mode == "none":
        return None
    if mode == "jwt":
        if not (settings.jwt_jwks_uri and settings.jwt_issuer):
            raise RuntimeError("TSPI_MCP_AUTH=jwt requires TSPI_JWT_JWKS_URI and TSPI_JWT_ISSUER.")
        from fastmcp.server.auth.providers.jwt import JWTVerifier
        return JWTVerifier(jwks_uri=settings.jwt_jwks_uri, issuer=settings.jwt_issuer,
                           audience=settings.jwt_audience)
    if mode == "workos":
        if not (settings.workos_authkit_domain and settings.mcp_base_url):
            raise RuntimeError("TSPI_MCP_AUTH=workos requires TSPI_WORKOS_AUTHKIT_DOMAIN and TSPI_MCP_BASE_URL.")
        kwargs = dict(
            authkit_domain=settings.workos_authkit_domain,
            base_url=settings.mcp_base_url,
            resource_base_url=settings.mcp_base_url,
            resource_name="TSPI AI Brain",
        )
        if not settings.extra_audiences:
            from fastmcp.server.auth.providers.workos import AuthKitProvider

            return AuthKitProvider(**kwargs)
        if not settings.allowed_client_ids:
            raise RuntimeError(
                "TSPI_EXTRA_AUDIENCES requires TSPI_ALLOWED_CLIENT_IDS (the WorkOS client IDs of "
                "the first-party apps, e.g. TSPI Digital chat). Refusing to accept any-app tokens."
            )
        return build_first_party_authkit(
            extra_audiences=settings.extra_audiences,
            allowed_client_ids=settings.allowed_client_ids,
            **kwargs,
        )

    raise RuntimeError(f"Unknown TSPI_MCP_AUTH mode: {mode!r} (use none|jwt|workos).")


def build_first_party_authkit(*, extra_audiences, allowed_client_ids, **authkit_kwargs):
    """AuthKitProvider that ALSO accepts access tokens from named first-party apps.

    Default AuthKit validation (aud == this server's resource URL) is tried first and is
    unchanged. Only if it fails, the token is checked again against the same AuthKit issuer and
    JWKS with aud in `extra_audiences`, and is then accepted only when its `client_id` claim is
    one of `allowed_client_ids`. Tokens from any other app with the environment audience are
    rejected, so this does not open the server to every app in the WorkOS environment.
    """
    import logging

    from fastmcp.server.auth.providers.jwt import JWTVerifier
    from fastmcp.server.auth.providers.workos import AuthKitProvider

    log = logging.getLogger("tspi_mcp.identity")
    allowed = frozenset(allowed_client_ids)

    class FirstPartyAuthKitProvider(AuthKitProvider):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._first_party_verifier = JWTVerifier(
                jwks_uri=f"{self.authkit_domain}/oauth2/jwks",
                issuer=self.authkit_domain,
                algorithm="RS256",
                audience=list(extra_audiences),
            )

        async def verify_token(self, token):
            tok = await super().verify_token(token)
            if tok is not None:
                return tok
            tok = await self._first_party_verifier.verify_token(token)
            if tok is None:
                return None
            client_id = (tok.claims or {}).get("client_id")
            if client_id not in allowed:
                log.warning("Bearer token rejected: client_id %r is not an allowed first-party app",
                            client_id)
                return None
            return tok

    return FirstPartyAuthKitProvider(**authkit_kwargs)


def current_identity_ext() -> dict:
    """Full identity for THIS request from the validated OAuth token (else the static pilot
    identity for stdio). Includes human-identity fields (name/email) for report attribution.
    Never trusts anything the LLM can set — everything comes from the signed token's claims."""
    tok = None
    try:
        from fastmcp.server.dependencies import get_access_token
        tok = get_access_token()
    except Exception:  # noqa: BLE001 — no request/token context (e.g. stdio)
        tok = None

    if tok is not None:
        claims = getattr(tok, "claims", None) or {}
        user = claims.get(settings.claim_user) or getattr(tok, "subject", None) or settings.clinician_id
        # No fallback to the static pilot identity for a real token: a signed-in user with no
        # role/clinic in their WorkOS metadata must NOT silently become a clinician.
        role = claims.get(settings.claim_role) or "unassigned"
        clinic = claims.get(settings.claim_clinic) or None
        email = claims.get(settings.claim_email)
        first = claims.get(settings.claim_first_name)
        last = claims.get(settings.claim_last_name)
    else:
        user, role, clinic = settings.clinician_id, settings.clinician_role, settings.clinic_id
        email = first = last = None

    name = " ".join(p for p in (first, last) if p) or None
    return {
        "user": str(user),
        "role": str(role),
        "clinic": (str(clinic) if clinic else None),
        "email": (str(email) if email else None),
        "first_name": (str(first) if first else None),
        "last_name": (str(last) if last else None),
        "name": name,
    }


def current_identity() -> tuple[str, str, str | None]:
    """(user_id, role, clinic_id) — the core triple used for RBAC/tenant checks. Thin wrapper over
    current_identity_ext() so there is a single source of truth."""
    i = current_identity_ext()
    return i["user"], i["role"], i["clinic"]
