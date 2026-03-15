# VibeScholar MCP — Remote Access via Tailscale Funnel

This guide documents how to expose the VibeScholar MCP server running on one
machine (the **corpus host**) so that Claude Code on other machines can use it
over the internet, without copying the corpus.

## Architecture

```
[Client machine]                    [Tailscale Funnel]                [Corpus host]
Claude Code                         (Tailscale's infra)               WSL + Python
    │                                                                     │
    ├─ POST /mcp ──► https://HOSTNAME.tail*.ts.net ──► 127.0.0.1:8765 ──►│
    │                  (TLS termination)                 (Funnel proxy)   │
    ◄─ JSON-RPC ◄────────────────────────────────────────────────────◄────┘
```

- The MCP server runs inside **WSL** on the corpus host, using the
  `streamable-http` transport in **stateless** mode.
- **Tailscale Funnel** exposes `127.0.0.1:8765` at a public HTTPS URL,
  handling TLS and proxying. No inbound ports need to be opened on the
  corpus host.
- On the client machine, Claude Code connects to the public Funnel URL.

## Prerequisites

| Component | Where | Notes |
|-----------|-------|-------|
| Tailscale | Both machines | Free plan works. Must be on the same tailnet. |
| Tailscale Funnel | Enabled in admin console | Visit `https://login.tailscale.com/admin/machines` → node → enable Funnel. Or follow the link printed by `tailscale funnel <port>` on first run. |
| WSL (with Python 3.11+) | Corpus host | The MCP server and its dependencies run inside WSL. |
| The VibeScholar index | Corpus host | SQLite DB + FAISS index + source PDFs, built by `index_corpus.py`. |

## Setup on the Corpus Host (one-time)

### 1. Install the project

```bash
# Inside WSL
cd /mnt/c/Users/<YOU>/Desktop/ccmcp
pip install -e .
```

Confirm it works locally:

```bash
python3 server.py --transport streamable-http --host 127.0.0.1 --port 8765 &
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/mcp
# Should print: 406 (expected — endpoint wants POST, not GET)
kill %1
```

### 2. Enable and start Tailscale Funnel

From a **Windows** terminal (PowerShell or CMD — not WSL, because `tailscale`
CLI in WSL may not be installed):

```powershell
# First time: Tailscale may prompt you to enable Funnel in the admin console.
tailscale funnel --bg 8765
```

`--bg` makes it a persistent background daemon managed by the Tailscale Windows
service. It survives reboots — no scheduled task needed for Funnel itself.

Verify:

```powershell
tailscale funnel status
# Should show:
#   https://<HOSTNAME>.tail<ID>.ts.net (Funnel on)
#   |-- / proxy http://127.0.0.1:8765
```

Take note of the full `https://...ts.net` URL — this is what clients connect to.

### 3. Create the startup script

Save as `C:\Users\<YOU>\Desktop\ccmcp\start-mcp-service.bat`:

```bat
@echo off
REM Start the VibeScholar MCP server inside WSL, accessible via Tailscale Funnel.
REM Intended to be run by Task Scheduler at system startup.

REM Start WSL and the MCP server (blocks until killed)
wsl -e bash -lc "cd /mnt/c/Users/<YOU>/Desktop/ccmcp && python3 server.py --transport streamable-http --host 127.0.0.1 --port 8765"
```

### 4. Register a Windows Scheduled Task

This ensures the MCP server starts automatically on boot, even before anyone
logs in.

```powershell
$action   = New-ScheduledTaskAction -Execute 'C:\Users\<YOU>\Desktop\ccmcp\start-mcp-service.bat'
$trigger  = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId '<YOU>' -LogonType S4U -RunLevel Highest

Register-ScheduledTask `
    -TaskName 'VibeScholar MCP Server' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'Runs the VibeScholar MCP server in WSL, exposed via Tailscale Funnel' `
    -Force
```

Test it:

```powershell
Start-ScheduledTask -TaskName 'VibeScholar MCP Server'
# Wait a few seconds for model loading, then:
curl.exe http://127.0.0.1:8765/mcp
# Should return a JSON-RPC error (406 or similar) — that means the server is up.
```

### 5. (Optional) Windows Firewall

If you also want direct SSH or direct HTTP access (bypassing Funnel), you need
firewall rules. Funnel does **not** require any inbound firewall rules since all
traffic is outbound from the corpus host to Tailscale's relay.

```powershell
# Only needed for direct access, NOT for Funnel:
New-NetFirewallRule -DisplayName 'Allow SSH Inbound All' `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort 22 -Profile Any
```

## Setup on the Client Machine

### 1. DNS resolution

Tailscale's MagicDNS may resolve the Funnel hostname to the internal Tailscale
IP instead of the public Funnel proxy. If so, override it:

```bash
# Find the public IPs (from any machine):
nslookup <HOSTNAME>.tail<ID>.ts.net 8.8.8.8

# Add to /etc/hosts on the client:
echo '<PUBLIC_IP> <HOSTNAME>.tail<ID>.ts.net' | sudo tee -a /etc/hosts
```

### 2. Claude Code MCP configuration

Add to `~/.claude/settings.json` (or the project's `.claude/settings.json`):

```json
{
  "mcpServers": {
    "vibescholar": {
      "type": "streamable-http",
      "url": "https://<HOSTNAME>.tail<ID>.ts.net/mcp"
    }
  }
}
```

### 3. Verify

```bash
# Quick HTTP test:
curl -s -o /dev/null -w "%{http_code}" https://<HOSTNAME>.tail<ID>.ts.net/mcp
# Should print: 406

# Then in Claude Code, the MCP tools should appear:
#   mcp__vibescholar__search_papers
#   mcp__vibescholar__read_document
#   mcp__vibescholar__list_indexed
```

## Key implementation details

### Why stateless mode?

Tailscale Funnel does not guarantee session affinity — consecutive HTTP requests
may arrive via different internal connections. The MCP `streamable-http`
transport normally uses session IDs, which break without affinity. Setting
`mcp.settings.stateless_http = True` disables session tracking.

### Why DNS override on the client?

When both client and corpus host are on the same Tailscale tailnet, MagicDNS
resolves `*.ts.net` hostnames to internal Tailscale IPs (e.g., `100.x.x.x`).
If the WireGuard tunnel isn't establishing (common on restrictive university
networks that block UDP), traffic to the internal IP fails. The `/etc/hosts`
override forces traffic through the public Funnel proxy instead.

### Server entry point flags

```
python3 server.py --transport streamable-http --host 127.0.0.1 --port 8765
```

| Flag | Purpose |
|------|---------|
| `--transport streamable-http` | Use HTTP instead of stdio |
| `--host 127.0.0.1` | Bind to localhost only (Funnel connects locally) |
| `--port 8765` | Port that Funnel proxies to |

The server defaults to `--transport stdio` for normal local MCP use, so
existing setups are unaffected.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `curl` returns `000` or TLS error | Funnel not running | `tailscale funnel status` on corpus host; restart with `tailscale funnel --bg 8765` |
| `curl` hangs or times out | DNS resolving to Tailscale internal IP | Check with `nslookup <host> 8.8.8.8`; add `/etc/hosts` override |
| 406 from curl but MCP tools fail with "Session not found" | `stateless_http` not enabled | Ensure `mcp.settings.stateless_http = True` in `server.py` |
| Scheduled task shows "Running" but server unreachable | WSL or Python crashed | Check `wsl -l -v` (WSL running?); test `curl http://127.0.0.1:8765/mcp` locally |
| Funnel shows "not enabled on your tailnet" | Funnel not enabled in admin | Visit the URL printed by `tailscale funnel` to enable it |

## Stopping / removing

```powershell
# Stop the MCP server:
Stop-ScheduledTask -TaskName 'VibeScholar MCP Server'

# Remove the scheduled task entirely:
Unregister-ScheduledTask -TaskName 'VibeScholar MCP Server' -Confirm:$false

# Disable Funnel:
tailscale funnel --https=443 off
```
