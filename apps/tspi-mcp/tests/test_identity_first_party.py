"""Tests for first-party app tokens (e.g. TSPI Digital chat) on the WorkOS AuthKit surface.

Resource-bound tokens must keep working unchanged; environment-audience tokens are accepted
only for allow-listed client IDs.
"""
import asyncio

import pytest
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair

from tspi_mcp.identity import build_first_party_authkit

AUTHKIT = "https://tspi-test.authkit.app"
BASE = "https://mcp.example.com"
RESOURCE = f"{BASE}/mcp"
ENV_CLIENT = "client_ENV123"
CHAT_APP = "client_CHAT456"


@pytest.fixture()
def keys(monkeypatch):
    kp = RSAKeyPair.generate()

    async def fake_key(self, token):  # skip the JWKS HTTP fetch
        return kp.public_key

    monkeypatch.setattr(JWTVerifier, "_get_verification_key", fake_key)
    return kp


@pytest.fixture()
def provider():
    p = build_first_party_authkit(
        extra_audiences=(ENV_CLIENT,),
        allowed_client_ids=(CHAT_APP,),
        authkit_domain=AUTHKIT,
        base_url=BASE,
        resource_base_url=BASE,
        resource_name="TSPI AI Brain",
    )
    p.set_mcp_path("/mcp")  # what http_app() does; binds aud to the resource URL
    return p


def _verify(provider, token):
    return asyncio.run(provider.verify_token(token))


def _token(kp, aud, client_id, iss=AUTHKIT):
    return kp.create_token(
        subject="user_1", issuer=iss, audience=aud,
        additional_claims={"client_id": client_id, "role": "clinician", "clinic_id": "c1"},
    )


def test_resource_bound_token_still_accepted(keys, provider):
    tok = _verify(provider, _token(keys, RESOURCE, "client_claude_dcr"))
    assert tok is not None and tok.claims["role"] == "clinician"


def test_chat_app_token_with_environment_audience_accepted(keys, provider):
    tok = _verify(provider, _token(keys, ENV_CLIENT, CHAT_APP))
    assert tok is not None and tok.claims["client_id"] == CHAT_APP


def test_other_app_with_environment_audience_rejected(keys, provider):
    assert _verify(provider, _token(keys, ENV_CLIENT, "client_SOMEONE_ELSE")) is None


def test_wrong_issuer_rejected(keys, provider):
    assert _verify(provider, _token(keys, ENV_CLIENT, CHAT_APP, iss="https://evil.example")) is None


def test_unknown_audience_rejected(keys, provider):
    assert _verify(provider, _token(keys, "some-other-api", CHAT_APP)) is None


def test_token_without_role_is_not_upgraded_to_clinician(monkeypatch):
    """A self-signed-up user with empty WorkOS metadata must not inherit the pilot role."""
    import tspi_mcp.identity as ident

    class Tok:
        claims = {"sub": "user_x", "role": "", "clinic_id": ""}
        subject = "user_x"

    monkeypatch.setattr("fastmcp.server.dependencies.get_access_token", lambda: Tok())
    i = ident.current_identity_ext()
    assert i["role"] == "unassigned" and i["clinic"] is None
