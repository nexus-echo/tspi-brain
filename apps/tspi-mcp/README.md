# TSPI AI Brain — MCP Server (Phase A)

A thin **governance + adapter** layer that lets a clinician drive the TSPI AI Brain from an AI
chat client (Claude Desktop / Claude.ai / ChatGPT) via the Model Context Protocol.

**It contains no clinical logic.** It is an HTTP client of the existing FastAPI engine, exactly
like MiHealth or any other REST client. The engine stays standalone and unchanged; MiHealth and
other clients are unaffected.

```
Doctor ─chat─► AI client ─MCP─► THIS server ─REST─► TSPI FastAPI engine (unchanged)
                                    │
                                    ├─ de-identification gate (PII never forwarded)
                                    ├─ draft / physician-approval framing
                                    └─ actionable errors
```

## Phase A tools (existing endpoints only — zero engine changes)

| Tool | Engine endpoint | Purpose |
|---|---|---|
| `tspi_engine_health` | `GET /health`, `/knowledge/health` | Is the engine up? registry/governance status |
| `tspi_screen_red_flags` | `POST /screen` | Safety screening — run first |
| `tspi_analyze_case` | `POST /analyze` | 39-axis analysis + NSS (no plan) |
| `tspi_generate_treatment_plan` | `POST /report` | Full draft plan (deliverable=false) |
| `tspi_get_treatment_plan` | `GET /reports/{id}` | Re-fetch a plan |
| `tspi_approve_treatment_plan` | `POST /validate` | Physician approve / edit / reject |
| `tspi_update_treatment_plan` | `POST /reports/{id}/override` | Structured physician edits (remove/reject module, change dose, override safety, add/remove secondary axis, note) — non-destructive; resets plan to draft for re-approval |
| `tspi_record_outcome` | `POST /outcome` | Follow-up marker for propose-only learning |
| `tspi_ask` | `POST /ask` | Ask TSPI Brain about a treatment plan (`report_id` / `case_id`) and/or the knowledge base (axes, domains, keys, modules, products, networks, safety, dosing). Grounded, cited answer; `retrieval_only=true` returns passages only. Question is PII-scanned before it leaves the MCP. |

Later phases add `tspi_extract_lab_report` (`/extract`), remote transport + OAuth, and RBAC.

## Run locally (Claude Desktop)

1. Start the engine (from `apps/ai-engine`): `uvicorn app.main:app --port 8000`.
2. Install deps here: `pip install -r requirements.txt`.
3. Copy the `tspi-ai-brain` block from `claude_desktop_config.example.json` into your Claude
   Desktop config and restart Claude Desktop.
4. In chat: *"Screen case CASE-0912 for red flags: female, 40s, CRP 21.5 mg/L, bloating."*

No public IP is required for this local/stdio setup — everything runs on the machine.

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `TSPI_ENGINE_URL` | `http://localhost:8000` | Where the engine is |
| `TSPI_ENGINE_TOKEN` | *(none)* | Bearer token to the engine (Phase B auth) |
| `TSPI_CLINICIAN_ID` | `unknown-clinician` | Recorded on approvals + audit |
| `TSPI_STRICT_DEIDENT` | `true` | Strict = reject on any detected PII; else redact + warn |
| `TSPI_MCP_TRANSPORT` | `stdio` | `stdio` (local) or `streamable-http` (remote) |

## Public deployment (OAuth — P7)

A **public** MCP endpoint MUST be OAuth-protected — an unauthenticated public MCP would let anyone
call the clinical tools. Set `TSPI_MCP_AUTH` (fail-closed: if set but misconfigured, the server
refuses to start rather than run unauthenticated):

| `TSPI_MCP_AUTH` | Use when | Required env |
|---|---|---|
| `none` (default) | local stdio (Claude Desktop) only — never public | — |
| `workos` | fastest fully-automatic remote-connector flow (DCR) | `TSPI_WORKOS_AUTHKIT_DOMAIN`, `TSPI_MCP_BASE_URL` |
| `jwt` | you already have an OIDC issuer (Auth0/Keycloak/Azure/…) | `TSPI_JWT_JWKS_URI`, `TSPI_JWT_ISSUER`, `TSPI_JWT_AUDIENCE` |

Per-request **clinician identity** comes from the validated token's claims
(`TSPI_CLAIM_USER` / `TSPI_CLAIM_ROLE` / `TSPI_CLAIM_CLINIC`, defaults `sub` / `role` / `clinic_id`)
and is forwarded to the engine as `X-TSPI-User-Id / -Role / -Clinic-Id`. Also set
`TSPI_MCP_TRANSPORT=streamable-http` and put TLS + the OAuth callback behind a reverse proxy. The
engine stays private; the MCP is the only exposed surface.

```
TSPI_MCP_AUTH=workos
TSPI_WORKOS_AUTHKIT_DOMAIN=https://your-project.authkit.app
TSPI_MCP_BASE_URL=https://mcp.your-domain.com
TSPI_MCP_TRANSPORT=streamable-http
TSPI_ENGINE_URL=http://tspi-api:8000     # private
TSPI_ENGINE_TOKEN=<one of the engine SERVICE_TOKENS>
```

## Governance invariants

- **PII never leaves this layer.** Tool schemas have no identity fields; the de-id gate drops
  unknown fields, rejects identity fields, and scrubs identifiers from free text.
- **Physician-approval gate.** Plans are AI drafts, watermarked, `deliverable=false` until approved.
- **Deterministic authority.** The engine computes everything; the chat only presents it.
- **Fail-closed.** Module output is provisional until the Module Registry is approved.

## Test

```
pip install -r requirements.txt pytest
python -m pytest tests -q
```
