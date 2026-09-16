# TSPI MCP — Stable public URL with ngrok (free static domain)

**Why ngrok:** the free plan gives you **one permanent static domain** (e.g.
`panda-new-kit.ngrok-free.app`) that does NOT change across restarts. So `TSPI_MCP_BASE_URL` and
the WorkOS resource indicator are set **once** — no hostname churn (the pain that cloudflared quick
tunnels caused). No domain purchase or DNS setup needed. This is the pilot/testing path.

> Production note: ngrok free is for dev/testing. For the real clinic deployment, front the MCP
> server with the Coolify reverse proxy on a domain you own (see TSPI_Deploy_Runbook_Coolify.md) — or
> a paid ngrok/Cloudflare named tunnel. Keep the engine (tspi-api) private either way.

---

## One-time setup (Windows)

```powershell
# 1. Create a free account at https://ngrok.com, then install ngrok:
winget install ngrok          # or download the Windows zip and unzip ngrok.exe

# 2. Add your authtoken (copy from dashboard -> "Your Authtoken"):
ngrok config add-authtoken <YOUR_AUTHTOKEN>
```

Find your free static domain in the ngrok dashboard under **Universal Gateway -> Domains**
(e.g. `panda-new-kit.ngrok-free.app`).

---

## Run it (every time)

```powershell
# Point the static domain at the MCP server's port (8080). Leave this window running.
ngrok http --url=https://panda-new-kit.ngrok-free.app 8080
#   (older ngrok versions: use --domain=panda-new-kit.ngrok-free.app instead of --url=)
```

Set once and never change (the domain is permanent):

- `apps/tspi-mcp/.env` -> `TSPI_MCP_BASE_URL=https://panda-new-kit.ngrok-free.app`
- WorkOS dashboard -> resource indicator / allowed resource = `https://panda-new-kit.ngrok-free.app/mcp`

Restart the MCP server after editing `.env` so the new base URL loads.

---

## Verify

```powershell
# Both must return 401 with the SAME resource_metadata host (= TSPI_MCP_BASE_URL):
curl.exe -i http://localhost:8080/mcp
curl.exe -i https://panda-new-kit.ngrok-free.app/mcp

# Discovery metadata resolves and points at the static host:
curl.exe -s https://panda-new-kit.ngrok-free.app/.well-known/oauth-protected-resource/mcp
```

`401` through ngrok = healthy (auth is enforced; the client will then redirect to WorkOS to log in).

---

## Gotchas

- **Port/scheme:** `ngrok http ... 8080` must target the MCP port (8080). The engine (8000) stays private.
- **Same machine:** run ngrok where uvicorn runs. ngrok in WSL can't reach a Windows `localhost:8080`.
- **Base URL must match the ngrok domain** — the `www-authenticate` header just echoes
  `TSPI_MCP_BASE_URL`. With the static domain they always agree, so set it once.
- **Free-tier interstitial:** ngrok free shows a browser warning page for HTML requests. Machine
  clients (the MCP connector, OAuth metadata fetch) are non-browser and normally skip it. If a
  `.well-known` fetch ever returns ngrok's HTML instead of JSON, the bypass header is
  `ngrok-skip-browser-warning: true`.
- **Engine stays private:** only tunnel the MCP port. Never expose tspi-api (8000) publicly.
