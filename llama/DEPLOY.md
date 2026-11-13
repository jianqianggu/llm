# Deploy llama-proxy vision fix on the 7900 XTX

You cannot apply this from the cloud agent. Copy the files onto
`jianq-7900xtx` as user `jianqianggu` and restart the proxy so the next
model load attaches mmproj.

## Root cause

1. LiteLLM (`:8000`) marks `qwen3.8-uncensored` as `supports_vision: true`.
2. Requests go to `llama-proxy.py` (`:8080`) which auto-switches
   `llama-server` (`:8081`) via `llama-switch <GGUF-filename>`.
3. The live switch command never passed `--mmproj`. Fara worked because the
   already-running `llama-server` argv included its projector. Qwen 3.8
   Uncensored was loaded text-only even though this file is on disk:

```
/mnt/fast_models/unsloth/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
```

(mirrored under `/mnt/LLM_Models/unsloth/...`).

## Install

From a clone of this repo (the `llama/` directory):

```bash
# 1. Back up whatever is live
sudo cp -a /usr/local/bin/llama-proxy.py /usr/local/bin/llama-proxy.py.bak.$(date +%Y%m%d)
sudo cp -a /usr/local/bin/llama-switch /usr/local/bin/llama-switch.bak.$(date +%Y%m%d) 2>/dev/null || true

# 2. Install the maintained copies (all three files must sit together)
sudo install -m 0755 llama_mmproj.py /usr/local/bin/llama_mmproj.py
sudo install -m 0755 llama-proxy.py  /usr/local/bin/llama-proxy.py
sudo install -m 0755 llama-switch.py /usr/local/bin/llama-switch
```

Keep the existing llama-server flags. If the current process uses extra
arguments beyond `-m` / `--mmproj`, write them (one token-group per line or a
single shell line) to:

```
~/.config/qwen35-server/server.args
```

Example (copy from `ps aux | grep llama-server`, omitting `-m` and `--mmproj`):

```
-ngl 99
-fa on
-c 32768
-np 1
--jinja
```

Or:

```bash
export LLAMA_SERVER_EXTRA_ARGS='-ngl 99 -fa on -c 32768 -np 1 --jinja'
```

`--mmproj` in that file is ignored on purpose so a stale Fara projector is not
stuck onto the next model.

Optional env:

| Variable | Default |
|---|---|
| `LLAMA_MODELS_DIR` | `/mnt/fast_models/unsloth` |
| `LLAMA_MODELS_DIR_MIRROR` | `/mnt/LLM_Models/unsloth` |
| `LLAMA_SERVER_URL` / `--port` | `http://127.0.0.1:8081` |
| `LLAMA_API_KEY_FILE` | `~/.config/qwen35-server/api-key` |
| `LLAMA_SERVER_BIN` | `llama-server` on `PATH` |

## Restart safely

Do **not** restart LiteLLM (`:8000`) unless you also changed its config.
Restart the proxy and force one model reload so 8081 picks up mmproj.

```bash
# Find the unit that owns llama-proxy.py (names vary)
systemctl --user list-units --all '*llama*' '*proxy*'
systemctl list-units --all '*llama*' '*proxy*'

# Typical user-unit restart (adjust the unit name if different)
systemctl --user restart llama-proxy.service

# If it is not systemd-managed, replace the process on :8080 only:
#   1. note the uvicorn/python pid for llama-proxy.py
#   2. SIGTERM that pid
#   3. start:
/home/jianqianggu/.venvs/huggingface/bin/python /usr/local/bin/llama-proxy.py
```

Confirm the projector would be attached **before** touching the GPU process:

```bash
llama-switch --dry-run qwen3.8-uncensored
llama-switch --dry-run fara1.5
```

Both commands must print `--mmproj` and a real path. Then load Qwen so the
running server is no longer text-only:

```bash
llama-switch qwen3.8-uncensored
# waits until http://127.0.0.1:8081/health is ok
```

That stops whatever is on **8081** (the current Fara or Qwen process) and
starts llama-server with the detected mmproj. It does not stop LiteLLM.

If `fara1.5` vision must stay up during the copy, install the files first,
then only `llama-switch` when you can spare the GPU for a reload. The next
chat request that names `fara1.5` will also attach Fara's mmproj
(`Fara1.5-27.mmproj-q8_0.gguf` in the same GGUF directory).

## Vision smoke test (`qwen3.8-uncensored`)

1×1 PNG (replace the key file if you use a different LiteLLM master key):

```bash
KEY=$(cat ~/.config/qwen35-server/api-key)
# If LiteLLM uses a different master key:
# KEY=$(sudo grep -oP 'master_key:\s*\K\S+' ~/.config/litellm/config.yaml | tr -d '"')

IMG='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

curl -sS -X POST 'http://127.0.0.1:8000/v1/chat/completions' \
  -H "Authorization: Bearer ${KEY}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3.8-uncensored\",
    \"max_tokens\": 64,
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image in one sentence. If you can see it, start with VISION_OK.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${IMG}\"}}
      ]
    }]
  }"
```

Success: HTTP 200 and a normal assistant message (it should contain `VISION_OK`
or a description). Failure still looks like LiteLLM 500 /
`image input is not supported` / `provide the mmproj`.

Direct proxy (bypasses LiteLLM) uses port **8080** and the GGUF id or alias:

```bash
curl -sS -X POST 'http://127.0.0.1:8080/v1/chat/completions' \
  -H "Authorization: Bearer ${KEY}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3.8-uncensored\",
    \"max_tokens\": 64,
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image in one sentence.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${IMG}\"}}
      ]
    }]
  }"
```

## Fara 1.5 regression check

Same payload, `"model": "fara1.5"`. Must still return HTTP 200, not the mmproj
error. Dry-run must keep Fara's projector:

```bash
llama-switch --dry-run fara1.5
```

## After deploy

`GET /v1/models` on the proxy includes `meta.mmproj` and `meta.vision` for
GGUFs that have a projector. mmproj files themselves are not listed as chat
models.
