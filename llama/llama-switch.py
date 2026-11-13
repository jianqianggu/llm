#!/home/jianqianggu/.venvs/huggingface/bin/python
"""Restart llama-server for a GGUF, attaching --mmproj when one exists.

Installed on the XTX as `/usr/local/bin/llama-switch`. The live llama-proxy
calls this with the chat GGUF filename and, after this fix, `--mmproj <path>`.
This script also auto-detects a sibling mmproj so a direct CLI switch works.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

_BIND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BIND_DIR not in sys.path:
    sys.path.insert(0, _BIND_DIR)

from llama_mmproj import (  # noqa: E402
    MODELS_DIR,
    find_mmproj,
    is_mmproj_filename,
    iter_chat_ggufs,
    resolve_model_filename,
)

DEFAULT_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8081"))
DEFAULT_HOST = os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
API_KEY_FILE = os.environ.get(
    "LLAMA_API_KEY_FILE", "/home/jianqianggu/.config/qwen35-server/api-key"
)
PID_FILE = os.environ.get(
    "LLAMA_SERVER_PID_FILE",
    os.path.expanduser("~/.config/qwen35-server/llama-server.pid"),
)
LOG_FILE = os.environ.get(
    "LLAMA_SERVER_LOG_FILE",
    os.path.expanduser("~/.config/qwen35-server/llama-server.log"),
)
EXTRA_ARGS_FILE = os.environ.get(
    "LLAMA_SERVER_ARGS_FILE",
    os.path.expanduser("~/.config/qwen35-server/server.args"),
)
# Conservative 7900 XTX defaults. Override via server.args or LLAMA_SERVER_EXTRA_ARGS.
DEFAULT_EXTRA_ARGS = "-ngl 99 -fa on -c 32768 -np 1 --jinja"
HEALTH_TIMEOUT_S = int(os.environ.get("LLAMA_SWITCH_HEALTH_TIMEOUT", "90"))


def _read_extra_args():
    raw = os.environ.get("LLAMA_SERVER_EXTRA_ARGS", "").strip()
    if not raw and os.path.isfile(EXTRA_ARGS_FILE):
        with open(EXTRA_ARGS_FILE, encoding="utf-8") as handle:
            parts = []
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts.append(line)
            raw = " ".join(parts)
    if not raw:
        raw = DEFAULT_EXTRA_ARGS
    return strip_mmproj_args(shlex.split(raw))


def strip_mmproj_args(argv):
    out = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in {"--mmproj", "-mmproj"}:
            skip = True
            continue
        if arg.startswith("--mmproj=") or arg.startswith("-mmproj="):
            continue
        out.append(arg)
    return out


def resolve_gguf(requested, models_dir=None):
    models_dir = models_dir or MODELS_DIR
    if requested and os.path.isfile(requested):
        if is_mmproj_filename(os.path.basename(requested)):
            raise SystemExit(f"refusing to load mmproj as a chat model: {requested}")
        return os.path.abspath(requested)

    installed = list(iter_chat_ggufs(models_dir))
    by_name = {os.path.basename(path): path for path in installed}
    chosen = resolve_model_filename(requested, list(by_name))
    if chosen is None:
        raise SystemExit(f"unknown model: {requested}")
    return by_name[chosen]


def build_llama_server_cmd(
    model_path,
    mmproj_path=None,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    extra_args=None,
    api_key_file=None,
    llama_server_bin=None,
):
    bin_path = llama_server_bin or os.environ.get("LLAMA_SERVER_BIN", "llama-server")
    cmd = [
        bin_path,
        "-m",
        model_path,
        "--host",
        str(host),
        "--port",
        str(port),
    ]
    if extra_args is None:
        extra_args = _read_extra_args()
    cmd.extend(strip_mmproj_args(list(extra_args)))
    key_file = API_KEY_FILE if api_key_file is None else api_key_file
    if key_file and os.path.isfile(key_file):
        cmd.extend(["--api-key-file", key_file])
    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])
    return cmd


def _pids_on_port(port):
    try:
        output = subprocess.check_output(
            ["ss", "-lptn", f"sport = :{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        output = ""
    return [int(pid) for pid in re_pids(output)]


def re_pids(ss_output):
    import re

    return re.findall(r"pid=(\d+)", ss_output or "")


def _read_pidfile():
    try:
        with open(PID_FILE, encoding="utf-8") as handle:
            pid = int(handle.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def stop_llama_server(port=DEFAULT_PORT, timeout=30):
    pids = []
    pidfile_pid = _read_pidfile()
    if pidfile_pid:
        pids.append(pidfile_pid)
    pids.extend(_pids_on_port(port))
    # Unique, skip our own pid.
    seen = set()
    ordered = []
    for pid in pids:
        if pid in seen or pid == os.getpid():
            continue
        seen.add(pid)
        ordered.append(pid)

    for pid in ordered:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[llama-switch] SIGTERM pid {pid}", flush=True)
        except OSError:
            continue

    deadline = time.time() + timeout
    for pid in ordered:
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[llama-switch] SIGKILL pid {pid}", flush=True)
            except OSError:
                pass

    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def _api_key():
    try:
        with open(API_KEY_FILE, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def wait_healthy(url, timeout=HEALTH_TIMEOUT_S):
    health_url = url.rstrip("/") + "/health"
    headers = {}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    deadline = time.time() + timeout
    last_error = "timeout"
    while time.time() < deadline:
        try:
            request = urllib.request.Request(health_url, headers=headers)
            with urllib.request.urlopen(request, timeout=2) as response:
                body = response.read().decode("utf-8", errors="replace")
            if response.status == 200 and "ok" in body.lower():
                return True
            last_error = f"HTTP {response.status} {body[:200]}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise SystemExit(f"llama-server failed to become healthy: {last_error}")


def start_llama_server(cmd, log_file=LOG_FILE, pid_file=PID_FILE):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(pid_file) or ".", exist_ok=True)
    log_handle = open(log_file, "ab", buffering=0)
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    with open(pid_file, "w", encoding="utf-8") as handle:
        handle.write(str(process.pid))
    print(f"[llama-switch] started pid {process.pid}: {' '.join(cmd)}", flush=True)
    return process


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Switch the llama-server GGUF on port 8081")
    parser.add_argument("model", help="GGUF filename, alias (qwen3.8-uncensored), or absolute path")
    parser.add_argument("--mmproj", default=None, help="Multimodal projector GGUF (auto-detected if omitted)")
    parser.add_argument("--print-cmd", action="store_true", help="Print llama-server argv and exit")
    parser.add_argument("--dry-run", action="store_true", help="Do not stop/start the server")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--no-mmproj", action="store_true", help="Force a text-only load")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    models_dir = args.models_dir or MODELS_DIR
    model_path = resolve_gguf(args.model, models_dir=models_dir)
    if args.no_mmproj:
        mmproj = None
    elif args.mmproj:
        mmproj = os.path.abspath(args.mmproj)
        if not os.path.isfile(mmproj):
            raise SystemExit(f"mmproj not found: {mmproj}")
    else:
        mmproj = find_mmproj(model_path, models_dir=models_dir)

    cmd = build_llama_server_cmd(
        model_path,
        mmproj_path=mmproj,
        host=args.host,
        port=args.port,
    )
    if mmproj:
        print(f"[llama-switch] mmproj: {mmproj}", flush=True)
    else:
        print(f"[llama-switch] no mmproj for {os.path.basename(model_path)} (text-only)", flush=True)

    if args.print_cmd or args.dry_run:
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        return 0

    stop_llama_server(port=args.port)
    start_llama_server(cmd)
    wait_healthy(f"http://{args.host}:{args.port}")
    print(f"[llama-switch] ready: {os.path.basename(model_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
