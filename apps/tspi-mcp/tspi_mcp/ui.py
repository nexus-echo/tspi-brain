"""Interactive button panels (MCP-UI) for the TSPI Digital chat.

The chat (LibreChat) renders a `ui://` resource with mimeType text/html inside a sandboxed iframe
(scripts only: no same-origin, no downloads, no popups, no dialogs). A button can only post a
message to the chat; the chat turns it into a user message and the assistant then calls the
matching TSPI tool. So every button here is a *prompt* action, and every action is still checked
by the engine's RBAC exactly as if the user had typed it.

Panels are attached only for the first-party chat app (token client_id in TSPI_UI_CLIENT_IDS,
default TSPI_ALLOWED_CLIENT_IDS). Other MCP clients (e.g. the Claude connector) get the plain JSON
result as before, so they never pay for or see the HTML.

Buttons:
  * Generate report  : after a red-flag screen or an analysis (clinicians and patients).
  * Download PDF     : on any report (clinicians and patients).
  * Approve          : clinician/reviewer only, draft reports only, two-step confirm in the panel.
"""
from __future__ import annotations

import html
import json
from typing import Any

from .config import settings

CLINICAL_ROLES = frozenset({"clinician", "reviewer"})
UI_URI_PREFIX = "ui://tspi/"


# --------------------------------------------------------------------------- gating
def _token_client_id() -> str | None:
    try:
        from fastmcp.server.dependencies import get_access_token
        tok = get_access_token()
    except Exception:  # noqa: BLE001 — no request/token context (stdio, tests)
        return None
    if tok is None:
        return None
    claims = getattr(tok, "claims", None) or {}
    return claims.get("client_id") or getattr(tok, "client_id", None)


def ui_enabled() -> bool:
    """True when the caller is a chat client that can render MCP-UI panels."""
    mode = (settings.ui_mode or "auto").lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    client_id = _token_client_id()
    return bool(client_id) and client_id in settings.ui_client_ids


def is_clinical(role: str | None) -> bool:
    return (role or "").lower() in CLINICAL_ROLES


# --------------------------------------------------------------------------- prompts
def generate_prompt(case_id: str | None) -> str:
    case = f" for case {case_id}" if case_id else ""
    return (f"Generate the TSPI report{case} using the case details already given in this "
            "conversation.")


def download_prompt(report_id: str) -> str:
    return f"Create and give me the PDF for report {report_id}."


def approve_prompt(report_id: str) -> str:
    return (f"Approve report {report_id}. I am the reviewing clinician and I confirm this approval "
            "(confirmed with the Approve button).")


# --------------------------------------------------------------------------- html
_CSS = """
:root{--blue:#001FB8;--orange:#FF8442;--ink:#0B1B33;--muted:#4A5868;--line:#DDE3EA;--card:#fff;
--ok:#0F7B3E;--okbg:#E7F6EC;--warn:#8A4B00;--warnbg:#FFF3E3;--bad:#A4161A;--badbg:#FDECEC}
@media (prefers-color-scheme:dark){:root{--ink:#E8ECF4;--muted:#A8B3C4;--line:#2A3448;--card:#141D2F;
--okbg:#12301F;--ok:#7FD6A0;--warnbg:#33240F;--warn:#FFC27A;--badbg:#3A1416;--bad:#FF9EA1}}
*{box-sizing:border-box}html,body{margin:0;background:transparent}
body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;color:var(--ink)}
.card{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:12px 14px}
.head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.title{font-weight:600}.id{color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.badge{font-size:11px;font-weight:700;letter-spacing:.03em;border-radius:999px;padding:2px 8px;text-transform:uppercase}
.b-draft{background:var(--warnbg);color:var(--warn)}.b-ok{background:var(--okbg);color:var(--ok)}
.b-bad{background:var(--badbg);color:var(--bad)}
.note{color:var(--muted);font-size:13px;margin:4px 0 10px}
.alert{background:var(--badbg);color:var(--bad);border-radius:8px;padding:8px 10px;font-size:13px;margin:4px 0 10px}
.row{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;font-weight:600;border-radius:8px;padding:7px 14px;cursor:pointer;border:1px solid transparent}
.primary{background:var(--blue);color:#fff}.secondary{background:transparent;color:var(--blue);border-color:var(--blue)}
@media (prefers-color-scheme:dark){.secondary{color:#9DB0FF;border-color:#9DB0FF}}
.approve{background:var(--ok);color:#fff}.ghost{background:transparent;color:var(--muted);border-color:var(--line)}
button:disabled{opacity:.5;cursor:default}
.confirm{display:none;margin-top:10px;border-top:1px dashed var(--line);padding-top:10px}
.confirm p{margin:0 0 8px;font-size:13px}.sent{display:none;color:var(--muted);font-size:12px;margin-top:8px}
"""

_JS = """
(function(){
  function size(){var h=Math.ceil(document.documentElement.getBoundingClientRect().height)+2;
    parent.postMessage({type:'ui-size-change',payload:{height:h}},'*');}
  window.addEventListener('load',size);
  if(window.ResizeObserver){new ResizeObserver(size).observe(document.body);}
  var done=false;
  function send(prompt){ if(done){return;} done=true;
    document.querySelectorAll('button').forEach(function(b){b.disabled=true;});
    document.getElementById('sent').style.display='block'; size();
    parent.postMessage({type:'prompt',payload:{prompt:prompt}},'*'); }
  document.querySelectorAll('[data-prompt]').forEach(function(b){
    b.addEventListener('click',function(){ send(JSON.parse(b.getAttribute('data-prompt'))); });});
  var ask=document.getElementById('approve-ask');
  if(ask){ ask.addEventListener('click',function(){
      document.getElementById('confirm').style.display='block'; ask.disabled=true; size(); });
    document.getElementById('approve-cancel').addEventListener('click',function(){
      document.getElementById('confirm').style.display='none'; ask.disabled=false; size(); }); }
})();
"""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _button(label: str, prompt: str, cls: str) -> str:
    return (f'<button type="button" class="{cls}" data-prompt="{_esc(json.dumps(prompt))}">'
            f"{_esc(label)}</button>")


def _page(body: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{_CSS}</style></head>'
            f'<body><div class="card">{body}'
            '<div class="sent" id="sent">Sent to the assistant. See the chat for the result.</div>'
            f"</div><script>{_JS}</script></body></html>")


def screen_panel(case_id: str | None, red_flags: list | None, role: str | None) -> str | None:
    """After tspi_screen_red_flags. Patients with red flags get no Generate button: the
    assistant must send them to urgent care first."""
    flags = red_flags or []
    head = ('<div class="head"><span class="title">Red-flag screen</span>'
            f'<span class="id">{_esc(case_id)}</span></div>')
    if flags:
        if not is_clinical(role):
            return _page(head + '<div class="alert"><b>Urgent:</b> warning signs were found. '
                         "Please seek medical care now (India 112, Thailand 1669) before anything "
                         "else.</div>")
        body = (head + f'<div class="alert"><b>{len(flags)} red flag(s) found.</b> Act on them '
                "through your clinical protocols before continuing.</div>"
                '<div class="row">' + _button("Generate report anyway", generate_prompt(case_id),
                                              "secondary") + "</div>")
        return _page(body)
    body = (head + '<div class="note">No red flags found. You can generate the TSPI report.</div>'
            '<div class="row">' + _button("Generate report", generate_prompt(case_id), "primary")
            + "</div>")
    return _page(body)


def analysis_panel(case_id: str | None) -> str:
    """After tspi_analyze_case."""
    body = ('<div class="head"><span class="title">Analysis complete</span>'
            f'<span class="id">{_esc(case_id)}</span></div>'
            '<div class="note">Create the full TSPI report with the module plan.</div>'
            '<div class="row">' + _button("Generate report", generate_prompt(case_id), "primary")
            + "</div>")
    return _page(body)


def report_panel(report_id: str | None, status: str | None, deliverable: bool,
                 role: str | None) -> str | None:
    """After a report is generated, fetched, edited or decided."""
    if not report_id:
        return None
    clinical = is_clinical(role)
    st = (status or "draft").lower()
    if deliverable:
        badge, note = '<span class="badge b-ok">Approved by clinician</span>', (
            "This report has been reviewed and approved.")
    elif st in ("rejected", "reject"):
        badge, note = '<span class="badge b-bad">Rejected</span>', "This draft was rejected."
    elif clinical:
        badge, note = '<span class="badge b-draft">AI draft</span>', (
            "Pending physician review. Not for patient use until approved.")
    else:
        badge, note = '<span class="badge b-draft">Unreviewed AI draft</span>', (
            "Not reviewed or approved by a TSPI doctor. Not medical advice. "
            "Contact TSPI Digital for doctor review.")
    buttons = [_button("Download PDF", download_prompt(report_id), "primary")]
    confirm = ""
    if clinical and not deliverable and st not in ("rejected", "reject"):
        buttons.append('<button type="button" class="approve" id="approve-ask">Approve…</button>')
        confirm = ('<div class="confirm" id="confirm"><p>Approve report '
                   f'<b>{_esc(report_id)}</b>? This marks it as reviewed and releases it for '
                   "patient use, recorded under your name.</p><div class=\"row\">"
                   + _button("Confirm approval", approve_prompt(report_id), "approve")
                   + '<button type="button" class="ghost" id="approve-cancel">Cancel</button>'
                   "</div></div>")
    body = ('<div class="head"><span class="title">TSPI report</span>'
            f'<span class="id">{_esc(report_id)}</span>{badge}</div>'
            f'<div class="note">{note}</div><div class="row">{"".join(buttons)}</div>{confirm}')
    return _page(body)


# --------------------------------------------------------------------------- result wrapper
def with_panel(result: dict, panel_html: str | None, name: str):
    """Return `result` unchanged, or a ToolResult carrying the same JSON plus the UI panel.
    structured_content keeps the tool's output schema satisfied for MCP clients that check it."""
    if not panel_html or not ui_enabled():
        return result
    from fastmcp.tools.tool import ToolResult
    from mcp.types import EmbeddedResource, TextContent, TextResourceContents

    return ToolResult(
        content=[
            TextContent(type="text", text=json.dumps(result, default=str)),
            EmbeddedResource(type="resource", resource=TextResourceContents(
                uri=f"{UI_URI_PREFIX}{name}", mimeType="text/html", text=panel_html)),
        ],
        structured_content=result,
    )
