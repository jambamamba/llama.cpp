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

# Qwen3.6-35B-A3B at 1M context (single slot, YaRN rope scaling):
./llamacpp-server.sh --preset qwen36-35b-q4

# Model already on disk:
./llamacpp-server.sh --model ~/data/models/llama.gguf

# Download the gguf, then serve:
./llamacpp-server.sh --model ~/data/models/lfm/LFM2.5-8B-A1B-Q8_0.gguf \
    --hf-repo LiquidAI/LFM2.5-8B-A1B-GGUF

# Self-test an already-running server:
./llamacpp-server.sh --test --port 8080

# Profile a preset (one long prefill + a 256-token generation):
./llamacpp-server.sh --preset qwen36-35b-q4 --bench
```

## Options

| Option | Description |
| --- | --- |
| `--preset <name>` | Pre-configured model + context. `lfm2.5` (default), `qwen36-35b-q4`, `qwen36-35b-q4xl`, `qwen36-35b-q8`, `qwen36-27b-q4` - see [Qwen3.6 presets](#qwen36-presets-1m-context). Explicit flags (`--model`, `--ctx-size`, ...) still win over preset values. |
| `--model <path>` | Path to the model `.gguf` file. **Optional** — defaults to Liquid AI LFM2.5-8B-A1B Q8_0 at 128 000-token context (`~/data/models/lfm2.5-8b-a1b-q8_0/LFM2.5-8B-A1B-Q8_0.gguf`), downloaded on first use. If the given path does not exist yet and `--hf-repo` is set, it is downloaded first. Files already present in the Hugging Face hub cache (`~/.cache/huggingface/hub/models--*`) are reused without re-downloading. |
| `--hf-repo <repo_id>` | Hugging Face repo that holds the `.gguf` (e.g. `LiquidAI/LFM2.5-8B-A1B-GGUF`). Used to download `--model` when missing. |
| `--hf-file <name>` | Filename to download from `--hf-repo`. Defaults to the basename of `--model`. |
| `--port <port>` | Port to listen on. Default: `8080`. (The vLLM server uses `8000`, so the two can run side by side.) |
| `--ctx-size <n>` | Size of the prompt context (tokens). Default: `0` = loaded from the model (128 000 for the default model). With N slots the per-slot context is ctx-size / N. |
| `--gpu-layers <n\|all>` | Layers to offload to VRAM. Default: `all` (full Metal offload on Apple Silicon). |
| `--flash-attn <on\|off\|auto>` | Flash attention use. Default: `auto`. |
| `--parallel <n>` | Number of parallel server slots. Default: `2`, or `1` for the qwen36 presets (the full 1M context goes into one slot). |
| `--cache-type-k <type>` | KV cache quantization for K (e.g. `q8_0`). The 35B-A3B presets default to `q8_0` to halve the 1M-token KV cache. |
| `--cache-type-v <type>` | KV cache quantization for V (e.g. `q8_0`). |
| `--bench [N]` | Profile: start the server, send one long prefill (default 8192 tokens) plus a 256-token generation, print prefill/decode tok/s and process memory, then stop the server. A pre-existing server on the port is profiled in place and left running. Use for A/B comparison of presets: `for p in qwen36-35b-q4 qwen36-35b-q4xl qwen36-35b-q8 qwen36-27b-q4; do ./llamacpp-server.sh --preset "$p" --bench; done` |
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

## Qwen3.6 presets (262K context)

`--preset qwen36-*` runs the open-weights **Qwen3.6** family at the native
**262 144-token** context in a single slot. Note: **Qwen3.6-Plus itself is
closed-weights** (hosted on Alibaba Cloud Model Studio / OpenRouter); no GGUF
of it exists. The presets use its open siblings, published by Qwen
(Apache-2.0) with GGUF quants from unsloth:

| Preset | Model file | Size | Notes |
| --- | --- | --- | --- |
| `qwen36-35b-q4` | `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22.7 GB | **Already in the local HF cache** - no download |
| `qwen36-35b-q4xl` | `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 22.9 GB | Dynamic XL quant, best 4-bit quality |
| `qwen36-35b-q8` | `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-Q8_0.gguf` | 36.9 GB | Best quality, near-BF16 |
| `qwen36-27b-q4` | `unsloth/Qwen3.6-27B-GGUF` / `Qwen3.6-27B-Q4_K_M.gguf` | 16.8 GB | Dense (no MoE routing), simpler but slower at long ctx |

All presets: **single slot** (`--parallel 1`). The 35B presets quantize the
KV cache to `q8_0`.

> **1M context is not currently effective.** The script targets 1 048 576
> tokens via YaRN (factor 4) and wires the flags when `--ctx-size` exceeds
> 262 144, but llama-server caps the slot context at the model's training
> context, so the effective context is 262 144. This is deliberate upstream
> behavior (issue #22140, closed as not planned); re-check after updating
> llama.cpp, or run `--ctx-size 1048576 --port 8081` and look for the
> `capping` warning to see if it still applies.

### Why 1M context works on this machine

Qwen3.6-35B-A3B is a **hybrid** model (arch `qwen35moe` in llama.cpp): only
10 of its 40 layers are full attention; the other 30 are linear attention
(gated delta net) with a fixed-size recurrent state that does not grow with
context. Consequences:

- **KV cache is ~20 KB/token** (10 layers x 2 KV heads x 128 dim x 2 (K+V) x
  f16), so 262 144 tokens is ~5 GB at f16 - halved to ~2.5 GB with the
  preset's `q8_0` KV cache. At a true 1M context it would be ~20 GB f16 /
  ~10 GB q8_0. A dense model of the same size would need 4x that.
- Native context is **262 144**. Extending it (e.g. to 1 048 576) stretches
  RoPE by factor 4 via YaRN, which llama.cpp applies only to the RoPE
  sections of the hybrid attention layers; linear-attention layers are
  unaffected. The server's slot cap currently limits the usable context to
  the native 262 144 - see the note above.
- **Memory budget** (35B q4 preset): ~22 GB weights + ~2.5 GB KV (q8_0) +
  compute buffers ~= 28-32 GB wired - comfortable on 128 GB. The q8_0 preset
  (~37 GB weights) totals ~45 GB - it fits, but measure with `--bench`.
- `mmproj-BF16.gguf` in the same HF repo enables vision via `--mmproj` (not
  wired into the presets).

### Profiling presets against each other

```bash
for p in qwen36-35b-q4 qwen36-35b-q4xl qwen36-35b-q8 qwen36-27b-q4; do
  ./llamacpp-server.sh --preset "$p" --bench
 done
```

`--bench` reports prefill t/s (8192-token prompt), decode t/s (256 tokens)
and the server's RSS. Thinking is disabled in the probe so decode measures
the answer, not chain-of-thought. It starts and stops its own server per
preset; against a running service it profiles in place. Compare decode t/s
at equal context and pick the fastest quant whose quality you accept.

## Native 1M-context presets

`--preset llama4-scout-q4`, `--preset glm-5.3-flash-iq1` and
`--preset kimi-linear-48b-q4` run models whose **native training context is
1 048 576 tokens**, so no YaRN rope scaling is needed and the server's
slot cap (n_ctx_train) does not bite. All three use compressed attention
(chunked SWA, sparse/linear hybrid, or MLA) so the KV cache stays affordable
at 1M tokens. Verified GGUF metadata and repo file sizes:

| Preset | Model | Weights | KV @ 1M (q8_0) | Wired total |
| --- | --- | --- | --- | --- |
| `llama4-scout-q4` | Llama 4 Scout 17B-16E Q4_K_M (bartowski), 2 shards | 61.7 GiB | ~25 GB (12 full-attn layers of 48; 36 are 8192-token chunked SWA) | ~90 GB |
| `glm-5.3-flash-iq1` | GLM-5.3-Flash 320B-A18B IQ1_M (unsloth), 3 shards | 84.7 GiB | ~10 GB (hybrid sparse + linear attention) | ~100 GB, tight |
| `kimi-linear-48b-q4` | Kimi-Linear 48B-A3B Q4_K_M (bartowski) | 28.0 GiB | ~5 GB (3:1 KDA linear attention + MLA, f16) | ~40 GB |

Notes:

- One model at a time: loading two large servers concurrently can exhaust
  unified memory and panic the machine. Stop the current service before
  starting another preset (`launchctl bootout gui/$(id -u)/com.llamacpp.server`).
- GLM-5.3-Flash IQ1_M is a 1-bit dynamic quant of a 320B model: it fits 128 GB
  only at this size, and 1-bit quality loss is real. Unsloth demonstrates it
  as the intended way to run GLM-5.3-Flash on 128 GB Macs.
- Llama 4 Scout is natively multimodal; the `mmproj` file is not wired into
  the preset.
- MiniMax-M3 (1M ctx, 230B/10B) was evaluated and rejected: its smallest
  quant (UD-IQ1_M) is 119.6 GiB, leaving no room for KV + buffers on 128 GB.
- Gemma 4 tops out at 256K native context - no Gemma model does 1M.

## Kimi-Linear-48B-A3B deep dive

Why the `kimi-linear-48b-q4` preset works so well on this machine, plus a
field report on driving it from OpenCode CLI for Yocto development.
Architecture facts come from Moonshot AI's Kimi-Linear documentation and
community analyses; memory numbers are reconciled against our own
`--bench` measurements below.

### The trick: hybrid linear attention (KDA)

A standard Transformer keeps a full-attention KV cache that grows with every
layer and every token, and prefill compute grows quadratically with context.
Kimi-Linear instead alternates **Kimi Delta Attention (KDA)** layers - a
gated delta-rule linear attention with a fixed-size recurrent state - with
full MLA attention at a **3:1 ratio**. Only every fourth layer carries a
cache that grows with context; the KDA layers hold a constant-size state.
The 1M-token KV cache therefore stays in the low GB range (f16 - see the
preset table above), where a same-size dense Transformer would need tens of
GB. It is also an MoE: **48B total parameters, ~3B active per token**, which
is why Metal decode reaches 118 t/s at 1M context here (vs 36 t/s for Llama
4 Scout and 27.6 t/s for the dense Qwen3.6-27B at 256K).

### Does the memory footprint hold up?

External analyses estimate ~50 GB total at full 1M context (weights 27-30 GB
+ KV cache 15-20 GB). Our measurements are more optimistic:

| Item | External estimate | Measured / verified here |
| --- | --- | --- |
| Weights (Q4_K_M) | 27-30 GB | 28.0 GiB (30 061 058 720 bytes on disk) |
| KV cache at 1M, f16 | 15-20 GB | ~5 GB estimated from GGUF metadata; grows toward this only as the context fills |
| Total at 1M ctx | ~50 GB | 36.4 GB RSS right after load (context nearly empty); even at the pessimistic 20 GB KV the total stays ~55 GB |

Either way, the 128 GB machine keeps ample headroom for the OS, IDEs, Docker
and browsers while the server holds 1M tokens. Measured numbers are in
"Measured performance" below.

### Using Kimi-Linear from OpenCode CLI (Yocto workflow)

The server is a drop-in OpenAI-compatible endpoint, so OpenCode CLI can use
it as a local provider:

```jsonc
// ~/.config/opencode/opencode.json (provider entry)
"llamacpp": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "llama.cpp (local Kimi-Linear)",
  "options": { "baseURL": "http://127.0.0.1:8000/v1" },
  "models": {
    "kimi-linear-48b": { "name": "Kimi-Linear-48B-A3B (local, 1M ctx)" }
  }
}
```

Any `model` string is accepted - llama.cpp serves exactly one model. Start
it with `./llamacpp-server.sh --preset kimi-linear-48b-q4`, or use the
installed service (see "Current service" under "Service install").

Where it shines for Yocto:

- **Repo-wide ingestion**: an entire layer set (meta-custom,
  meta-raspberrypi, .bb / .bbappend recipes, local.conf, bblayers.conf) fits
  into the 1M window at once, so cross-file recipe reasoning needs no RAG.
- **Bitbake log analysis**: failed-build dumps such as `log.do_compile` run
  to tens of thousands of lines; Kimi-Linear takes the whole raw log and can
  pin the failure against the recipe that caused it.

Where to be careful (community benchmarks; not independently verified
here):

- **Coding tier**: reported as basic-to-intermediate, closer to a 7B-14B
  dense model than to a deep-reasoning frontier model.
- **BitBake syntax is niche**: override patterns (`do_configure:prepend`,
  `RDEPENDS:${PN}`) blend Python and shell; models of this class sometimes
  hallucinate variable semantics or deprecated Yocto syntax.
- **Thinking**: the model emits chain-of-thought before answers - keep the
  reasoning budget in mind for interactive use (see "Reasoning / thinking
  behavior").

Recommended split: run Kimi-Linear locally for long-context search, log
debugging and variable-inheritance mapping, and route hard recipe logic and
dependency loops to a hosted model through OpenCode's model router (e.g.
Qwen3.6-Plus or DeepSeek). Note that Qwen3.6-Plus itself is closed-weights;
its open siblings are available locally via the qwen36-* presets (262K
context).

## Measured performance (M5 Max, same session 2026-09-22, one model at a time)

`--bench` thinking-off probes, sequential loads, nothing else resident.
Qwen presets run 262 144 ctx (slot cap), the 1M presets run 1 048 576 ctx.

| Preset | Ctx | Prefill t/s | Decode t/s | RSS |
| --- | --- | --- | --- | --- |
| qwen36-35b-q4 | 256K | 2401 | 97.0 | 23.6 GB |
| qwen36-35b-q4xl | 256K | 2637 | 95.1 | 23.8 GB |
| qwen36-35b-q8 | 256K | 2607 | 91.2 | 33.0 GB |
| qwen36-27b-q4 | 256K | 670 | 27.6 | 30.0 GB |
| kimi-linear-48b-q4 | 1M | 2116 | 118.4 | 36.4 GB |
| llama4-scout-q4 | 1M | 635 | 36.3 | 83.4 GB |
| glm-5.3-flash-iq1 | 1M | n/a - GGUF arch `glm5next` is not supported by this build (open upstream PR); download kept on disk |

Reading: Kimi-Linear is the fastest decoder of everything measured and the
only affordable true-1M model; Scout holds 1M but is slow and heavy; the
Qwen presets are prefill champions but cap at 256K. Kimi KV cache runs f16
(72-dim compressed heads are incompatible with q8_0 blocks).

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
(`max_tokens: 256`, thinking disabled via `chat_template_kwargs`), and passes
only if the reply is non-empty. Timings are printed and the exit status
reflects the result; a server started by the test is shut down afterward, a
pre-existing one is left running.

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

### Current service (2026-09-23)

`com.llamacpp.server` runs `kimi-linear-48b-q4` on port 8000 (installed with
`--install-service --preset kimi-linear-48b-q4 --port 8000`). It replaced the
Mistral-Nemo-12B vLLM-era service and, on 2026-09-23, also displaced
`com.omlx.server` (an MLX lazy-loading server for Qwen3-8B and
Qwen3-Coder-30B-A3B-4bit) which had taken port 8000 in the meantime. To
restore omlx later: `launchctl bootout gui/$(id -u)/com.llamacpp.server &&
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.omlx.server.plist`
- but do not run both servers with models resident at the same time
(one-LLM-at-a-time rule: loading two large models can panic the machine).

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
