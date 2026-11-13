#!/home/jianqianggu/.venvs/huggingface/bin/python
"""OpenAI-compatible auto-switch proxy in front of llama-server (:8081).

Live path: `/usr/local/bin/llama-proxy.py` on the 7900 XTX (uvicorn :8080).
When a chat request names a GGUF / alias, this proxy calls `llama-switch`.
Vision models must pass `--mmproj` into that switch or llama.cpp serves
text-only and returns: "image input is not supported".
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.parse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

_BIND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BIND_DIR not in sys.path:
    sys.path.insert(0, _BIND_DIR)

from llama_mmproj import (  # noqa: E402
    MODELS_DIR,
    clean_model_name,
    find_mmproj,
    is_mmproj_filename,
    iter_chat_ggufs,
    llama_switch_command,
    resolve_model_filename,
)

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
API_KEY_FILE = os.environ.get(
    "LLAMA_API_KEY_FILE", "/home/jianqianggu/.config/qwen35-server/api-key"
)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
current_loaded_model = None
current_loaded_mmproj = None
switch_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0)
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="llama.cpp Auto-Switching Model Proxy",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_api_key():
    try:
        with open(API_KEY_FILE) as api_key_file:
            return api_key_file.read().strip()
    except OSError:
        return ""


def get_installed_models():
    models = []
    for path in iter_chat_ggufs(MODELS_DIR):
        filename = os.path.basename(path)
        if is_mmproj_filename(filename):
            continue
        mmproj = find_mmproj(path, models_dir=MODELS_DIR)
        models.append(
            {
                "id": filename,
                "object": "model",
                "created": int(os.path.getmtime(path)),
                "owned_by": "llamacpp",
                "meta": {
                    "path": path,
                    "size_bytes": os.path.getsize(path),
                    "mmproj": mmproj,
                    "vision": bool(mmproj),
                },
            }
        )
    return models


def find_model(requested_model_id):
    query = clean_model_name(requested_model_id)
    installed = get_installed_models()
    by_id = {model["id"]: model for model in installed}
    chosen = resolve_model_filename(query, list(by_id))
    if chosen is None:
        return None
    return by_id[chosen]


in_flight_requests = 0
in_flight_lock = asyncio.Condition()


def _mmproj_for(model):
    path = (model.get("meta") or {}).get("path")
    explicit = (model.get("meta") or {}).get("mmproj")
    if explicit:
        return explicit
    if path:
        return find_mmproj(path, models_dir=MODELS_DIR)
    return None


async def acquire_model(requested_model_id):
    global current_loaded_model, current_loaded_mmproj, in_flight_requests

    target = find_model(requested_model_id)
    if target is None:
        return None

    target_name = target["id"]
    target_mmproj = _mmproj_for(target)

    async with in_flight_lock:
        while True:
            same_load = (
                current_loaded_model == target_name
                and current_loaded_mmproj == target_mmproj
            )
            if same_load:
                try:
                    response = await app.state.http_client.get(
                        f"{LLAMA_SERVER_URL}/health",
                        headers={"Authorization": f"Bearer {get_api_key()}"},
                        timeout=2.0,
                    )
                    if response.status_code == 200 and response.json().get("status") == "ok":
                        in_flight_requests += 1
                        return target_name
                except (httpx.HTTPError, ValueError):
                    pass

            if in_flight_requests == 0:
                break
            await in_flight_lock.wait()

        switch_cmd = llama_switch_command(target_name, target_mmproj)
        if target_mmproj:
            print(
                f"[Proxy] Auto-switching llama-server from '{current_loaded_model}' "
                f"to '{target_name}' with mmproj '{target_mmproj}'",
                flush=True,
            )
        else:
            print(
                f"[Proxy] Auto-switching llama-server from '{current_loaded_model}' "
                f"to '{target_name}' (text-only, no mmproj)",
                flush=True,
            )
        result = await asyncio.to_thread(
            subprocess.run,
            switch_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, end="", flush=True)
            in_flight_lock.notify_all()
            raise RuntimeError(f"llama-switch exited with status {result.returncode}")

        current_loaded_model = target_name
        current_loaded_mmproj = target_mmproj
        in_flight_requests += 1
        in_flight_lock.notify_all()
        return target_name


async def release_model():
    global in_flight_requests
    async with in_flight_lock:
        in_flight_requests = max(0, in_flight_requests - 1)
        in_flight_lock.notify_all()


def normalize_api_path(raw_path):
    path = urllib.parse.unquote(raw_path)
    path = re.sub(r"\s+", "", path)
    path = "/" + path.strip("/")

    # Tolerate clients that append /v1 to a base URL already ending in /v1.
    path = re.sub(r"^/v1/v1(?=/|$)", "/v1", path)

    # Tolerate clients that expect a host-only OpenAI base URL.
    if path in {"/models", "/chat/completions", "/completions", "/embeddings"}:
        path = "/v1" + path

    return path


def filtered_request_headers(request):
    return {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
        and name.lower() not in {"host", "content-length"}
    }


def filtered_response_headers(response):
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
        and name.lower() != "content-length"
    }


@app.get("/")
@app.get("/v1")
async def api_info():
    return {
        "status": "ok",
        "service": "llama.cpp OpenAI-compatible model proxy",
        "models": "/v1/models",
        "chat_completions": "/v1/chat/completions",
    }


@app.get("/health")
@app.get("/healthz")
async def health():
    try:
        response = await app.state.http_client.get(
            f"{LLAMA_SERVER_URL}/health",
            headers={"Authorization": f"Bearer {get_api_key()}"},
            timeout=2.0,
        )
        return JSONResponse(
            content=response.json(), status_code=response.status_code
        )
    except (httpx.HTTPError, ValueError):
        return JSONResponse(
            content={"status": "unavailable", "proxy": "ok"},
            status_code=503,
        )


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    models = get_installed_models()
    return {
        "object": "list",
        "data": models,
        "models": [model["id"] for model in models],
    }


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)
async def proxy_all(path, request: Request):
    if request.method == "OPTIONS":
        return Response(status_code=204)

    clean_path = normalize_api_path(path)
    if clean_path == "/v1/models":
        return await list_models()
    if clean_path in {"/", "/v1"}:
        return await api_info()

    request_body = await request.body()
    requested_model = None
    if request.method == "POST" and request_body:
        try:
            requested_model = json.loads(request_body).get("model")
        except (json.JSONDecodeError, AttributeError):
            pass

    acquired_model = False
    if requested_model:
        try:
            selected_model = await acquire_model(requested_model)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Model switch failed: {error}",
                        "type": "model_switch_error",
                    }
                },
            )
        if selected_model is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Unknown model: {requested_model}",
                        "type": "model_not_found",
                    }
                },
            )
        acquired_model = True

    headers = filtered_request_headers(request)
    api_key = get_api_key()
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    upstream_request = app.state.http_client.build_request(
        method=request.method,
        url=f"{LLAMA_SERVER_URL}{clean_path}",
        headers=headers,
        content=request_body,
    )
    try:
        upstream = await app.state.http_client.send(
            upstream_request, stream=True
        )
    except httpx.HTTPError as error:
        if acquired_model:
            await release_model()
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"llama-server unavailable: {error}",
                    "type": "upstream_error",
                }
            },
        )

    async def stream_wrapper():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            if acquired_model:
                await release_model()

    return StreamingResponse(
        stream_wrapper(),
        status_code=upstream.status_code,
        headers=filtered_response_headers(upstream),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
