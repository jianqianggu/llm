#!/usr/bin/env python3
"""LLM Observability Monitor - polls services, logs to SQLite/JSONL, generates status dashboard.

Read-only: performs GET-only health/readiness/catalog probes, never chat completions or model loads.
"""
import json
import os
import socket
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CHECK_INTERVAL = int(os.environ.get("OBS_INTERVAL", "60"))
DB_PATH = Path(
    os.environ.get(
        "OBS_DB",
        os.path.expanduser("~/.local/share/llm-observability/status.db"),
    )
)
HTML_PATH = Path(
    os.environ.get(
        "OBS_HTML",
        os.path.expanduser("~/.local/share/llm-observability/status.html"),
    )
)
LOG_PATH = Path(
    os.environ.get(
        "OBS_LOG",
        os.path.expanduser("~/.local/share/llm-observability/status.jsonl"),
    )
)

def load_keys() -> Dict[str, str]:
    """Load API keys from environment or config files securely."""
    litellm_key = os.environ.get("OBS_API_KEY", "").strip()
    cursor_key = (
        os.environ.get("CURSOR_API_KEY", "").strip()
        or os.environ.get("CURSOR_BRIDGE_API_KEY", "").strip()
    )
    backend_key = ""

    # Check obs.env file if keys not in env
    obs_env = Path("~/.config/litellm/obs.env").expanduser()
    if obs_env.is_file():
        try:
            for line in obs_env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k == "OBS_API_KEY" and not litellm_key:
                        litellm_key = v
                    elif (k == "CURSOR_API_KEY" or k == "CURSOR_BRIDGE_API_KEY") and not cursor_key:
                        cursor_key = v
        except Exception:
            pass

    # Backend key
    bk_file = Path("~/.config/qwen35-server/api-key").expanduser()
    if bk_file.is_file():
        try:
            backend_key = bk_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    return {
        "litellm": litellm_key,
        "cursor": cursor_key,
        "backend": backend_key,
    }

KEYS = load_keys()

SERVICES = [
    {
        "name": "LiteLLM Proxy",
        "url": "http://127.0.0.1:8000/health/readiness",
        "auth": None,
        "port": 8000,
        "role": "litellm",
    },
    {
        "name": "Ollama",
        "url": "http://127.0.0.1:11434/api/tags",
        "auth": None,
        "port": 11434,
        "role": "service",
    },
    {
        "name": "cursor-api-proxy",
        "url": "http://127.0.0.1:8765/health",
        "auth": "cursor",
        "port": 8765,
        "role": "service",
    },
    {
        "name": "antigravity-proxy",
        "url": "http://127.0.0.1:8766/v1/models",
        "auth": None,
        "port": 8766,
        "role": "service",
    },
    {
        "name": "Whisper Server",
        "url": "http://127.0.0.1:8001/healthz",
        "auth": None,
        "port": 8001,
        "role": "service",
    },
    {
        "name": "Unsloth Studio",
        "url": "http://127.0.0.1:8888/api/health",
        "auth": None,
        "port": 8888,
        "role": "service",
    },
    {
        "name": "llama-proxy",
        "url": "http://127.0.0.1:8080/health",
        "auth": None,
        "port": 8080,
        "role": "proxy",
        "unit": "llama-proxy.service",
    },
    {
        "name": "llama-backend",
        "url": "http://127.0.0.1:8081/health",
        "auth": "backend",
        "port": 8081,
        "role": "backend",
        "unit": "llama-backend.service",
    },
    {
        "name": "Strata",
        "url": "http://127.0.0.1:8090/health",
        "auth": None,
        "port": 8090,
        "role": "backend",
        "unit": "strata-qwen38.service",
    },
]

def db_init():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript('''
        CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            ok INTEGER NOT NULL,
            latency_ms INTEGER,
            detail TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_checks_ts ON checks(ts);
        CREATE INDEX IF NOT EXISTS idx_checks_target ON checks(target);
    ''')
    c.commit()
    c.close()

def db_record(target: str, kind: str, ok: bool, latency_ms: int, detail: str):
    c = sqlite3.connect(DB_PATH)
    c.execute(
        "INSERT INTO checks(ts,target,kind,ok,latency_ms,detail) VALUES(?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            target,
            kind,
            1 if ok else 0,
            latency_ms,
            detail,
        ),
    )
    c.commit()
    c.close()

def log_json(record: Dict[str, Any]):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def check_tcp(port: int, timeout: float = 3.0) -> Tuple[bool, int, str]:
    """Plain TCP connection check without sending any byte/newline to avoid invalid HTTP traffic."""
    t0 = time.time()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            pass
        return True, int((time.time() - t0) * 1000), "open"
    except Exception as e:
        return False, int((time.time() - t0) * 1000), str(e)

def check_http(url: str, auth_type: Optional[str] = None, timeout: float = 5.0) -> Tuple[bool, int, str, Optional[Any]]:
    """Perform read-only GET request and parse whole response JSON without truncation."""
    t0 = time.time()
    headers = {}
    if auth_type and KEYS.get(auth_type):
        headers["Authorization"] = f"Bearer {KEYS[auth_type]}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()  # Read whole response, never truncate before JSON parse
            parsed = None
            try:
                parsed = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                pass
            return True, int((time.time() - t0) * 1000), f"HTTP {r.status}", parsed
    except urllib.error.HTTPError as e:
        return False, int((time.time() - t0) * 1000), f"HTTP {e.code}", None
    except Exception as e:
        return False, int((time.time() - t0) * 1000), str(e), None

def get_systemd_unit_state(unit: str) -> Tuple[str, str]:
    """Read-only check of systemd ActiveState and SubState (unprivileged)."""
    try:
        out = subprocess.check_output(
            ["systemctl", "show", "-p", "ActiveState", "-p", "SubState", "--value", unit],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).splitlines()
        active = out[0].strip() if len(out) > 0 else "unknown"
        sub = out[1].strip() if len(out) > 1 else "unknown"
        return active, sub
    except Exception:
        return "unknown", "unknown"

def poll_once() -> Dict[str, Any]:
    results = {"services": [], "ts": datetime.now(timezone.utc).isoformat()}
    for svc in SERVICES:
        tcp_ok, tcp_lat, tcp_det = check_tcp(svc["port"])
        http_ok = False
        http_lat = 0
        http_det = ""
        parsed_body = None
        state = "down"
        detail = tcp_det
        role = svc["role"]

        if tcp_ok:
            http_ok, http_lat, http_det, parsed_body = check_http(
                svc["url"], svc.get("auth"), timeout=4.0
            )
            detail = http_det

        # Evaluate exact health & standby states per contract
        if role == "litellm":
            # HTTP 200 readiness must require actual status healthy
            if (
                http_ok
                and isinstance(parsed_body, dict)
                and parsed_body.get("status") == "healthy"
            ):
                state = "ok"
                detail = "healthy"
                record_ok = True
            else:
                state = "down"
                detail = http_det or "not healthy"
                record_ok = False

        elif role == "proxy":
            # Proxy inactive is always service down
            if http_ok:
                state = "ok"
                if isinstance(parsed_body, dict) and parsed_body.get("loaded_model"):
                    detail = f"active: {parsed_body.get('loaded_model')}"
                else:
                    detail = "idle (no model loaded)"
                record_ok = True
            else:
                state = "down"
                detail = "service down" if not tcp_ok else http_det
                record_ok = False

        elif role == "backend":
            # Backend unit: port open vs port closed
            if tcp_ok:
                if svc["name"] == "Strata":
                    if http_ok and isinstance(parsed_body, dict) and parsed_body.get("status") == "ok":
                        loaded = parsed_body.get("loaded", False)
                        model = parsed_body.get("model", "qwen3.8-flash-next")
                        if loaded:
                            state = "ok"
                            detail = f"loaded: {model}"
                            record_ok = True
                        else:
                            # Strata health status ok + loaded false is standby, not service failure
                            state = "standby"
                            detail = "standby (unloaded)"
                            record_ok = True
                    else:
                        state = "down"
                        detail = http_det
                        record_ok = False
                else:  # llama-backend
                    if http_ok and isinstance(parsed_body, dict) and parsed_body.get("status") == "ok":
                        state = "ok"
                        detail = "ready"
                        record_ok = True
                    else:
                        state = "down"
                        detail = http_det
                        record_ok = False
            else:
                # Port closed: inspect read-only systemd unit state
                unit = svc.get("unit", "")
                active_state, sub_state = get_systemd_unit_state(unit) if unit else ("unknown", "unknown")
                if active_state == "failed" or sub_state == "failed":
                    state = "down"
                    detail = f"unit failed ({sub_state})"
                    record_ok = False
                elif active_state == "inactive":
                    state = "standby"
                    detail = "standby (inactive)"
                    record_ok = True
                else:
                    state = "standby"
                    detail = f"standby ({active_state}/{sub_state})"
                    record_ok = True

        else:  # Standard external / local service (Ollama, cursor, antigravity, whisper, unsloth)
            if http_ok:
                state = "ok"
                record_ok = True
            else:
                state = "down"
                record_ok = False

        db_record(
            svc["name"],
            "http" if tcp_ok else "tcp",
            record_ok,
            http_lat if tcp_ok else tcp_lat,
            detail,
        )
        results["services"].append({
            "name": svc["name"],
            "port": svc["port"],
            "state": state,
            "tcp_ok": tcp_ok,
            "http_ok": http_ok,
            "latency_ms": http_lat if tcp_ok else tcp_lat,
            "detail": detail,
        })

    log_json(results)
    return results

def render_html(latest: Dict[str, Any]):
    c = sqlite3.connect(DB_PATH)
    rows = c.execute(
        "SELECT ts,target,kind,ok,latency_ms FROM checks ORDER BY id DESC LIMIT 240"
    ).fetchall()
    c.close()

    cards = ""
    for svc in latest["services"]:
        st = svc["state"]
        cards += f'''<div class="card {st}">
<h3>{svc["name"]}</h3>
<div class="port">:{svc["port"]}</div>
<div class="status">{st.upper()}</div>
<div class="lat">{svc["latency_ms"]} ms</div>
<div class="det">{svc["detail"]}</div>
</div>'''

    spark = ""
    for ts, target, kind, ok, lat in reversed(rows):
        if target.startswith("model:"):
            continue
        cls = "g" if ok else "r"
        spark += f'<span class="dot {cls}" title="{ts} {target} {kind} {lat}ms"></span>'

    html = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<title>LLM Observability</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#0d1117;color:#c9d1d9;margin:24px;}}
h1{{color:#58a6ff;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin:16px 0;}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
.card.ok{{border-color:#2ea043;}}
.card.standby{{border-color:#d29922;background:#18160f;}}
.card.down{{border-color:#da3633;background:#1c0e0e;}}
.card h3{{margin:0 0 4px 0;font-size:14px;}}
.port{{color:#8b949e;font-size:12px;}}
.status{{font-weight:bold;font-size:13px;margin:4px 0;}}
.card.ok .status{{color:#3fb950;}}
.card.standby .status{{color:#e3b341;}}
.card.down .status{{color:#f85149;}}
.lat{{color:#8b949e;font-size:12px;}}
.det{{color:#6e7681;font-size:11px;word-break:break-all;}}
.spark{{display:flex;flex-wrap:wrap;gap:2px;margin:12px 0;}}
.dot{{width:8px;height:18px;border-radius:2px;}}
.dot.g{{background:#3fb950;}}
.dot.r{{background:#f85149;}}
.ts{{color:#6e7681;font-size:12px;margin-top:8px;}}
</style>
</head>
<body>
<h1>LLM Observability</h1>
<div class="ts">Last check: {latest["ts"]}</div>
<h2>Services</h2>
<div class="grid">{cards}</div>
<h2>Recent history (last 240 checks)</h2>
<div class="spark">{spark}</div>
</body>
</html>'''

    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(html, encoding="utf-8")

def main():
    db_init()
    print(
        f"[obs] starting, interval={CHECK_INTERVAL}s db={DB_PATH} html={HTML_PATH}",
        flush=True,
    )
    while True:
        try:
            r = poll_once()
            render_html(r)
            up = sum(1 for s in r["services"] if s["state"] in {"ok", "standby"})
            total = len(r["services"])
            print(f"[obs] {r['ts']} {up}/{total} operational", flush=True)
        except Exception as e:
            print(f"[obs] error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
