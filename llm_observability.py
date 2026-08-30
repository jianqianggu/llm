#!/usr/bin/env python3
"""LLM Observability Monitor - polls services, logs to SQLite, generates status dashboard."""
import json, sqlite3, time, os, socket, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path

CHECK_INTERVAL = int(os.environ.get("OBS_INTERVAL", "60"))
DB_PATH = Path(os.environ.get("OBS_DB", os.path.expanduser("~/.local/share/llm-observability/status.db")))
HTML_PATH = Path(os.environ.get("OBS_HTML", os.path.expanduser("~/src/llm/status.html")))
LOG_PATH = Path(os.environ.get("OBS_LOG", os.path.expanduser("~/.local/share/llm-observability/status.jsonl")))
API_KEY = os.environ.get("OBS_API_KEY", "sk-live-072fba7cd49fa1de5553d1259225517b")

SERVICES = [
    {"name": "LiteLLM Proxy",     "type": "http", "url": "http://127.0.0.1:8000/v1/models", "auth": True,  "port": 8000},
    {"name": "Ollama",            "type": "http", "url": "http://127.0.0.1:11434/api/tags", "auth": False, "port": 11434},
    {"name": "cursor-api-proxy",  "type": "http", "url": "http://127.0.0.1:8765/health",    "auth": True,  "port": 8765},
    {"name": "antigravity-proxy", "type": "http", "url": "http://127.0.0.1:8766/v1/models","auth": False, "port": 8766},
    {"name": "Whisper Server",    "type": "http", "url": "http://127.0.0.1:8001/healthz",   "auth": False, "port": 8001},
    {"name": "Unsloth Studio",    "type": "http", "url": "http://127.0.0.1:8888/api/health", "auth": False, "port": 8888},
]

MODELS_TO_TEST = [
    {"name": "grok-4.6-high",    "model": "grok-4.6-high"},
    {"name": "gemini-4.7-flash", "model": "gemini-4.7-flash"},
    {"name": "claude-opus-4.6",  "model": "claude-opus-4.6"},
    {"name": "fara1.5",          "model": "fara1.5"},
]

_prev = {}


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


def db_record(target, kind, ok, latency_ms, detail):
    c = sqlite3.connect(DB_PATH)
    c.execute("INSERT INTO checks(ts,target,kind,ok,latency_ms,detail) VALUES(?,?,?,?,?,?)",
              (datetime.now(timezone.utc).isoformat(), target, kind, 1 if ok else 0,
               latency_ms, detail))
    c.commit()
    c.close()


def log_json(record):
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def check_tcp(port, timeout=3):
    t0 = time.time()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.send(b"\\n")
        return True, int((time.time() - t0) * 1000), "open"
    except Exception as e:
        return False, int((time.time() - t0) * 1000), str(e)


def check_http(url, auth=False, timeout=5):
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"} if auth else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(512)
        return True, int((time.time() - t0) * 1000), f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, int((time.time() - t0) * 1000), f"HTTP {e.code}"
    except Exception as e:
        return False, int((time.time() - t0) * 1000), str(e)


def check_model(model_name, timeout=20):
    t0 = time.time()
    try:
        payload = json.dumps({"model": model_name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}).encode()
        req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                    data=payload,
                                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True, int((time.time() - t0) * 1000), "ok"
    except urllib.error.HTTPError as e:
        return False, int((time.time() - t0) * 1000), f"HTTP {e.code}"
    except Exception as e:
        return False, int((time.time() - t0) * 1000), str(e)[:120]


def poll_once():
    results = {"services": [], "models": [], "ts": datetime.now(timezone.utc).isoformat()}
    for svc in SERVICES:
        ok, lat, detail = check_tcp(svc["port"])
        db_record(svc["name"], "tcp", ok, lat, detail)
        results["services"].append({"name": svc["name"], "port": svc["port"], "tcp_ok": ok, "latency_ms": lat, "detail": detail})
        if ok:
            ok2, lat2, det2 = check_http(svc["url"], svc.get("auth", False))
            db_record(svc["name"], "http", ok2, lat2, det2)
            results["services"][-1].update({"http_ok": ok2, "http_latency_ms": lat2, "http_detail": det2})
    for m in MODELS_TO_TEST:
        ok, lat, detail = check_model(m["model"])
        db_record(f"model:{m['name']}", "chat", ok, lat, detail)
        results["models"].append({"name": m["name"], "ok": ok, "latency_ms": lat, "detail": detail})
    log_json(results)
    return results


def render_html(latest):
    c = sqlite3.connect(DB_PATH)
    rows = c.execute("SELECT ts,target,kind,ok,latency_ms FROM checks ORDER BY id DESC LIMIT 240").fetchall()
    c.close()
    cards = ""
    for svc in latest["services"]:
        status = "ok" if svc.get("http_ok", svc["tcp_ok"]) else "down"
        cards += f'''<div class="card {status}"><h3>{svc["name"]}</h3><div class="port">:{svc["port"]}</div>
        <div class="status">{status.upper()}</div>
        <div class="lat">{svc.get("http_latency_ms", svc["latency_ms"])} ms</div>
        <div class="det">{svc.get("http_detail", svc["detail"])}</div></div>'''
    mcards = ""
    for m in latest["models"]:
        status = "ok" if m["ok"] else "down"
        mcards += f'''<div class="card {status}"><h3>{m["name"]}</h3><div class="status">{status.upper()}</div>
        <div class="lat">{m["latency_ms"]} ms</div><div class="det">{m["detail"]}</div></div>'''
    spark = ""
    for ts, target, kind, ok, lat in reversed(rows):
        if target.startswith("model:"): continue
        cls = "g" if ok else "r"
        spark += f'<span class="dot {cls}" title="{ts} {target} {kind} {lat}ms"></span>'
    html = f'''<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<title>LLM Observability</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#0d1117;color:#c9d1d9;margin:24px;}}
h1{{color:#58a6ff;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin:16px 0;}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
.card.ok{{border-color:#2ea043;}}.card.down{{border-color:#da3633;background:#1c0e0e;}}
.card h3{{margin:0 0 4px 0;font-size:14px;}}
.port{{color:#8b949e;font-size:12px;}}
.status{{font-weight:bold;font-size:13px;margin:4px 0;}}
.card.ok .status{{color:#3fb950;}}.card.down .status{{color:#f85149;}}
.lat{{color:#8b949e;font-size:12px;}}
.det{{color:#6e7681;font-size:11px;word-break:break-all;}}
.spark{{display:flex;flex-wrap:wrap;gap:2px;margin:12px 0;}}
.dot{{width:8px;height:18px;border-radius:2px;}}
.dot.g{{background:#3fb950;}}.dot.r{{background:#f85149;}}
.ts{{color:#6e7681;font-size:12px;margin-top:8px;}}
</style></head><body>
<h1>LLM Observability</h1>
<div class="ts">Last check: {latest["ts"]}</div>
<h2>Services</h2><div class="grid">{cards}</div>
<h2>Model probes</h2><div class="grid">{mcards}</div>
<h2>Recent history (last 240 checks)</h2><div class="spark">{spark}</div>
</body></html>'''
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(html)


def main():
    db_init()
    print(f"[obs] starting, interval={CHECK_INTERVAL}s db={DB_PATH} html={HTML_PATH}", flush=True)
    while True:
        try:
            r = poll_once()
            render_html(r)
            up = sum(1 for s in r["services"] if s.get("http_ok", s["tcp_ok"])) + sum(1 for m in r["models"] if m["ok"])
            total = len(r["services"]) + len(r["models"])
            print(f"[obs] {r['ts']} {up}/{total} up", flush=True)
        except Exception as e:
            print(f"[obs] error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
