# TSPI MCP — Stable public URL with a named Cloudflare Tunnel (Windows)

**Why:** `cloudflared tunnel --url …` (a *quick* tunnel) mints a **new random hostname every run**, so
`TSPI_MCP_BASE_URL` and the WorkOS resource indicator go stale on every restart. A **named** tunnel
gives you one fixed hostname (e.g. `https://mcp.yourdomain.com`) that you set **once**.

**Prerequisite:** a domain added to your Cloudflare account (free plan is fine). Named tunnels route
through a Cloudflare zone, so you need a domain there. No domain → you're stuck with quick tunnels.

---

## One-time setup

```powershell
# 1. Authenticate cloudflared to your Cloudflare account (opens a browser; pick your domain).
.\Desktop\cloudflared.exe tunnel login

# 2. Create the named tunnel. This writes a credentials file <TUNNEL-ID>.json under %USERPROFILE%\.cloudflared\
.\Desktop\cloudflared.exe tunnel create tspi-mcp

# 3. Map a stable hostname to it (creates the DNS record automatically).
.\Desktop\cloudflared.exe tunnel route dns tspi-mcp mcp.yourdomain.com
```

Create a config file at `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: tspi-mcp
credentials-file: C:\Users\conta\.cloudflared\<TUNNEL-ID>.json

ingress:
  # NOTE the colon in localhost:8080 — the missing colon is what caused the earlier 502.
  - hostname: mcp.yourdomain.com
    service: http://localhost:8080
  - service: http_status:404        # required catch-all
```

---

## Run it (every time)

```powershell
.\Desktop\cloudflared.exe tunnel run tspi-mcp
```

The hostname never changes, so set these **once** and never touch them again:

- `apps/tspi-mcp/.env` → `TSPI_MCP_BASE_URL=https://mcp.yourdomain.com`
- WorkOS dashboard → resource indicator / allowed resource = `https://mcp.yourdomain.com/mcp`

Optional: install it as a Windows service so it starts on boot:
```powershell
.\Desktop\cloudflared.exe service install
```

---

## Verify

```powershell
# Origin up + auth enforced (both should be 401 with matching resource_metadata host):
curl.exe -i http://localhost:8080/mcp
curl.exe -i https://mcp.yourdomain.com/mcp

# Discovery metadata resolves and points at the stable host:
curl.exe -s https://mcp.yourdomain.com/.well-known/oauth-protected-resource/mcp
```

`401` through the tunnel = healthy. `502` = cloudflared can't reach the origin — re-check the
`service:` line (scheme `http://`, correct port `:8080`) and that the MCP server is running.

---

## Gotchas (the ones that already bit)

- **`http://localhost:8080`** — keep the colon. `localhost8080` forwards to a non-existent host → 502.
- **`http://`, not `https://`** — the MCP server speaks plain HTTP; Cloudflare terminates TLS.
- **Same machine/namespace** — run cloudflared where uvicorn runs. cloudflared in WSL cannot reach a
  Windows `localhost:8080` (different network namespace).
- **Base URL must match the live hostname** — the `www-authenticate` header just echoes
  `TSPI_MCP_BASE_URL`; it does not prove the tunnel serves that host. With a named tunnel they always agree.
