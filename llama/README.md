# llama.cpp auto-switch + vision (7900 XTX)

LiteLLM `qwen3.8-uncensored` was advertised as vision (`supports_vision: true`)
while the llama-server process behind `llama-proxy.py` loaded the GGUF
**without** `--mmproj`. llama.cpp then answered every image chat with:

`image input is not supported — hint: if this is unexpected, you may need to provide the mmproj`

Fara 1.5 already worked because its running `llama-server` command included
`--mmproj`. The uncensored Qwen weights have a matching projector on disk; the
switch path never passed it.

This directory is the maintained copy of the XTX proxy stack. Install notes
and the curl smoke test are in [DEPLOY.md](DEPLOY.md).

## Layout

| File | Install as |
|---|---|
| `llama_mmproj.py` | `/usr/local/bin/llama_mmproj.py` |
| `llama-proxy.py` | `/usr/local/bin/llama-proxy.py` |
| `llama-switch.py` | `/usr/local/bin/llama-switch` |

`llama-proxy.py` still lists GGUFs (skipping mmproj files) and calls
`llama-switch <gguf>`. When a sibling / Unsloth-layout mmproj exists it now
adds `--mmproj <path>`. Text-only GGUFs keep a text-only launch.

## Tests

```bash
python3 -m unittest llama.tests.test_llama_mmproj
```

From this directory:

```bash
python3 -m unittest tests.test_llama_mmproj
```
