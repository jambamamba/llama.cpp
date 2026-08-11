# llamacpp-server.sh

`llamacpp-server.sh` launches the **llama.cpp** inference server for a local
GGUF model on macOS (Apple Silicon) or Linux. It resolves everything needed
to serve a `.gguf`, opens the macOS application firewall for the binary, can
install the server as a persistent service, and can self-test the running
endpoint.

It serves the OpenAI-compatible HTTP API on `http://<host>:<port>`:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`, `GET /health`, plus a web UI at `/`

## Quick start

```bash
# Default model (Liquid AI LFM2.5-8B-A1B Q8_0, native 128 000-token context)
# — auto-downloaded on first use:
./llamacpp-server.sh

# Model already on disk:
./llamacpp-server.sh --model ~/data/models/llama.gguf

# Download the gguf, then serve:
./llamacpp-server.sh --model ~/data/models/lfm/LFM2.5-8B-A1B-Q8_0.gguf \
    --hf-repo LiquidAI/LFM2.5-8B-A1B-GGUF

# Self-test an already-running server:
./llamacpp-server.sh --test --port 8080
```

## Options

| Option | Description |
| --- | --- |
| `--model <path>` | Path to the model `.gguf` file. **Optional** — defaults to Liquid AI LFM2.5-8B-A1B Q8_0 at 128 000-token context (`~/data/models/lfm2.5-8b-a1b-q8_0/LFM2.5-8B-A1B-Q8_0.gguf`), downloaded on first use. If the given path does not exist yet and `--hf-repo` is set, it is downloaded first. |
| `--hf-repo <repo_id>` | Hugging Face repo that holds the `.gguf` (e.g. `LiquidAI/LFM2.5-8B-A1B-GGUF`). Used to download `--model` when missing. |
| `--hf-file <name>` | Filename to download from `--hf-repo`. Defaults to the basename of `--model`. |
| `--port <port>` | Port to listen on. Default: `8080`. (The vLLM server uses `8000`, so the two can run side by side.) |
| `--ctx-size <n>` | Size of the prompt context (tokens). Default: `0` = loaded from the model (128 000 for the default model). With N slots the per-slot context is ctx-size / N. |
| `--gpu-layers <n\|all>` | Layers to offload to VRAM. Default: `all` (full Metal offload on Apple Silicon). |
| `--flash-attn <on\|off\|auto>` | Flash attention use. Default: `auto`. |
| `--parallel <n>` | Number of parallel server slots. Default: `2`. |
| `--reasoning <on\|off\|auto>` | Use reasoning/thinking in chat. Default: `auto` — the default model emits chain-of-thought before answering. |
| `--reasoning-budget <n>` | Token budget for thinking: `-1` unrestricted, `0` immediate end, `N>0` budget. Default: `-1`. |
| `--metrics` | Enable the Prometheus-compatible metrics endpoint. |
| `--test` | Self-test: start the server, wait until ready, prompt it, confirm a non-empty reply, then shut the server down. If a server already answers on `--port`, it is tested and left running. Prints `PASS`/`FAIL` and timings; exits non-zero on failure. |
| `--status` | Print the current status: whether the service is installed and running (launchd/systemd), whether a server process is up, and the port + loaded model (queried from the live `/v1/models` endpoint). Starts nothing. |
| `--install-service` | Install as a launchd LaunchAgent (macOS) or systemd user unit (Linux) instead of running in the foreground. Restarts on failure, starts at login. |
| `-h, --help` | Show help. |

## Why the default model: Liquid AI LFM2.5-8B-A1B

The reference analysis in `~/repos/share/docs/lfms-vs-orinth.md` concludes that
Liquid AI LFM models are the right fit for long-context local hosting on this
machine: LFMs use a hybrid Linear State Space Model (SSM) + sparse grouped
query attention architecture, so the KV cache stays near-constant as context
grows instead of scaling O(N) like a standard transformer.

The transcript names an "LFM-40B-MoE" — that model does not exist in Liquid
AI's current lineup. The real, current long-context options (August 2026) are:

| Model | Params (active) | Context | GGUF (Q8_0) | Notes |
| --- | --- | --- | --- | --- |
| **LFM2.5-8B-A1B** | 8B (1.5B) | 128 000 | 9.01 GB | MoE, tool calling + CoT, latest LFM2.5 |
| LFM2.5-2.6B | 2.7B | 131 072 | ~2.5 GB | agentic, but not for coding/knowledge-heavy |
| LFM2-24B-A2B | 24B (2B) | 32 768 | ~25 GB | bigger, but native context too short |

**LFM2.5-8B-A1B is the pick for this MacBook (M5 Max, 128 GB):**

- **128 000-token native context** — matches the ~150k average session size.
- **8B total / 1.5B active MoE** — 8B-class quality at a fraction of the
  compute cost, built for tool calling and agentic workflows.
- **Tiny footprint** — Q8_0 is ~9 GiB of weights; the hybrid architecture
  keeps the KV cache small even at 128K context. Total wired memory is far
  below the ~50 GiB watchdog-panic threshold documented in `vllm-server.md`
  (which was measured on this same machine).
- **First-class llama.cpp support** — architecture `lfm2moe`
  (`LLM_ARCH_LFM2MOE`), official GGUF releases from `LiquidAI`.

## Measured performance (M5 Max, LFM2.5-8B-A1B Q8_0 @ 128K)

Measured against this server (llama.cpp build b10326, Metal, full offload):

| Test | Result |
| --- | --- |
| Prompt processing | **519.7 t/s** |
| Generation | **221.2 t/s** |

For comparison, the vLLM Metal server reaches ~24 tok/s on Mistral-Nemo-12B
Q8_0 — the LFM MoE + native Metal path is roughly an order of magnitude
faster on generation. Generation comfortably exceeds human reading speed.

## Reasoning / thinking behavior

The default model emits chain-of-thought before its final answer. llama.cpp
places the thinking in `message.reasoning_content` and the final answer in
`message.content`. Implications:

- A short `max_tokens` budget can be consumed by thinking alone, leaving
  `content` empty. Use a generous budget (the `--test` probe uses 256).
- To cap thinking, pass `--reasoning-budget <n>` (or `--reasoning off` to
  skip it). Per-request override is available via the `reasoning_control`
  field on `/v1/chat/completions` (see the server README).
- The `/v1/chat/completions/control` endpoint can end a long-thinking
  completion early by id.

## How the model is loaded

A `.gguf` carries weights, tokenizer, and chat template — unlike the vLLM
GGUF path, no companion `config.json` is needed. The script downloads the
model file from `--hf-repo` if it is missing; later runs need no flags.

llama.cpp applies the model's chat template through its built-in Jinja
engine (`--jinja`, enabled by default). The LFM chat template handles both
tool calling (`<|tool_call_start|>...`) and the CoT thinking tags.

## Prefix caching (KV reuse)

llama.cpp automatically reuses the KV state of any prompt prefix that has
already been computed, in-process (the server log reports `cache_n` tokens
per request). This is the same economics as vLLM's `--enable-prefix-caching`:
a long multi-turn session pays the full prefill only once, then each new turn
only prefills its newly-appended tokens.

The cache lives in the server process and dies on restart. Session length is
still bounded by `--ctx-size` (default: the model's 128 000).

## Self-test (`--test`)

`--test` boots the server (unless one already answers on the port), waits for
`GET /v1/models` to succeed (up to 600s), sends a one-shot chat completion
(`max_tokens: 256`, since the model thinks first), and passes only if the
reply is non-empty. Timings are printed and the exit status reflects the
result; a server started by the test is shut down afterward, a pre-existing
one is left running.

Example run:

```bash
$ ./llamacpp-server.sh --test
No --model given; using the default model (/Users/user01macbook377/data/models/lfm2.5-8b-a1b-q8_0/LFM2.5-8B-A1B-Q8_0.gguf).
Starting server on port 8080 for self-test (log: /tmp/llamacpp-server-test-12345.log) ...
PASS: server responded in 3s:
Hi
Stopping self-test server (PID 23456).
```

## Verify the server manually with curl

While the server is running, check it from another terminal (or from another
machine on your LAN using the Mac's LAN IP instead of `127.0.0.1`):

```bash
export BASE="http://127.0.0.1:8080"

# 1. Liveness + model metadata
curl -s $BASE/v1/models | python3 -m json.tool

# 2. Chat completion (OpenAI-style) - the model thinks, then answers:
curl -s $BASE/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lfm2.5-8b-a1b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 17 * 23? Use the calculator tool."}
    ],
    "tools": [
      {"type": "function", "function": {
        "name": "calculator",
        "description": "Multiply two integers",
        "parameters": {
          "type": "object",
          "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
          },
          "required": ["a", "b"]
        }
      }}
    ],
    "tool_choice": "auto",
    "max_tokens": 512
  }' | python3 -m json.tool

# 3. Plain completion (raw prompt)
curl -s $BASE/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Once upon a time,", "max_tokens": 64}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'

# 4. Embeddings
curl -s $BASE/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "The quick brown fox"}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("embedding dim:", len(d["data"][0]["embedding"]))'

# 5. Health endpoint
curl -si $BASE/health | head -1
```

Notes:

- The `model` field in a request may be omitted (llama.cpp serves the one
  model it was started with).
- `--test` in `llamacpp-server.sh` automates step 2 (readiness + one short
  prompt) and prints `PASS`/`FAIL`.
- From another machine on your LAN, replace `127.0.0.1` with the Mac's LAN
  IP. The Mac's firewall must allow the `llama-server` binary through — run
  `llamacpp-server.sh` once in an interactive terminal so it can add the rule.

## Service install (`--install-service`)

Writes a unit that starts at login, restarts on failure, and records logs:

- **macOS** — `~/Library/LaunchAgents/com.llamacpp.server.plist`; logs in
  `~/Library/Logs/llamacpp/`; inspect with
  `launchctl print gui/$(id -u)/com.llamacpp.server`, stop/remove with
  `launchctl bootout gui/$(id -u)/com.llamacpp.server`.
- **Linux** — `~/.config/systemd/user/llamacpp-server.service`; logs via
  `journalctl --user -u llamacpp-server.service`.

> On macOS the service is bootstrapped into the `gui/<uid>` domain, which
> only exists while a user is logged into the GUI session. If `launchctl
> bootstrap` fails with error 125, log into the Mac's screen and re-run.

## Troubleshooting

### Port already in use

The default port is `8080`. If the vLLM server holds `8000` and something
else holds `8080`, pick a free port:

```bash
./llamacpp-server.sh --port 8081
```

### Empty `content` in chat responses

The model thinks before answering; the thinking fills `reasoning_content`.
A tight `max_tokens` budget can end the response during thinking, leaving
`content` empty. Increase `max_tokens`, or cap/skip thinking:
`--reasoning-budget 0` skips thinking, `--reasoning off` disables it.

### Slow first response on a long context

A cold prefill of a 128K prompt takes a while (prompt processing is ~520 t/s,
so ~250s for a full 128K context). Subsequent turns reuse the cached prefix
KV and only prefill the newly-appended tokens. Restarting the server drops
the cache.

### `--install-service` fails over SSH after a reboot

Same as the vLLM service: the `gui` domain only exists while a user is logged
in at the Mac's display. Log in at the screen, then re-run `--install-service`.
