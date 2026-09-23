# Task handoff: run and profile Qwen3.6 presets in llamacpp-server.sh

> Status as of 2026-09-22 ~12:15 local time. Written for an agent or human
> picking up the work. Read this whole file before doing anything.
>
> UPDATE 2026-09-22 ~19:00: the task pivoted to 1M-context models (Llama 4
> Scout / GLM-5.3-Flash / Kimi-Linear). READ SECTION 6 FIRST - it has the
> current download/test state and the safety rules. Sections 2 and 4 are
> partly superseded by it.

## 1. Task and user intent

The user asked to run "Qwen3.6 Plus" with 1M token context on the local
llama.cpp server, download models as needed, and profile four model variants
to pick the best one. Confirmed decisions (user-selected):

1. **Model**: try ALL FOUR variants and profile them to compare:
   - `qwen36-35b-q4` (unsloth UD-Q4_K_M)
   - `qwen36-35b-q4xl` (unsloth UD-Q4_K_XL)
   - `qwen36-35b-q8` (unsloth Q8_0)
   - `qwen36-27b-q4` (unsloth Q4_K_M, dense)
   Rationale: Qwen3.6-**Plus** itself is closed-weights (Alibaba Cloud /
   OpenRouter hosted, no GGUF exists) - verified via web search. The open
   siblings above are the closest runnable substitutes; repo `Qwen/` originals
   are Apache-2.0.
2. **Script design**: add a `--preset` flag; LFM2.5 stays the default.
3. **Context**: user chose "1M, single slot", BUT see section 3: the llama.cpp
   server caps slot context at the model training context, so effective
   context is 262 144. User explicitly accepted "Accept 256K for now" and
   wants the 1M wiring kept for when upstream changes.
4. **Probe behavior**: disable thinking in `--test` / `--bench` probes
   (user chose "Probe with thinking off").
5. Final step once profiling is done: **swap the launchd service** to the
   winning preset (replaces Mistral-Nemo-12B on port 8000).

Machine: MacBook M5 Max, 128 GB unified memory, macOS. Repo checkout:
`~/repos/llama.cpp` (branch master). Server build exists at
`build/bin/llama-server` (b10326 + local dylibs).

## 2. What is already done

### Script changes (uncommitted, in working tree)

`git status`: modified `llamacpp-server.sh` and `llamacpp-server.md` only.

- `llamacpp-server.sh`:
  - `--preset` flag with 5 presets: `lfm2.5` (default), `qwen36-35b-q4`,
    `qwen36-35b-q4xl`, `qwen36-35b-q8`, `qwen36-27b-q4`. Preset values do not
    override explicit CLI flags (verified by fresh-process test harness).
  - qwen36 presets: single slot (`--parallel 1`), ctx 262144, KV cache
    `q8_0` on the 35B presets, YaRN wiring intact: when CTX_SIZE > 262144
    the args include `--rope-scaling yarn --yarn-orig-ctx 262144
    --rope-scale 4` (factor derived as CTX_SIZE/YARN_ORIG_CTX).
  - `--cache-type-k/--cache-type-v` flags.
  - `resolve_cache_model()`: reuses files already in the HF hub cache
    (`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/*/<file>`)
    before downloading.
  - `--bench [N]` mode: starts server (or profiles an already-running one
    in place), runs a long-prompt prefill probe (1 output token) and a
    short-prompt decode probe (256 tokens), prints tok/s + RSS, stops a
    server it started. Thinking disabled in probes via
    `chat_template_kwargs {"enable_thinking": false}`.
  - `--test` probe: same thinking-off kwargs; passes on non-empty reply.
- `llamacpp-server.md`: new "Qwen3.6 presets (262K context)" section with
  the preset table, the slot-cap limitation callout (upstream issue #22140
  closed as not planned), KV cache math, memory budgets, `--bench` usage,
  and updated `--test` docs.

### Verified working

- `--preset qwen36-35b-q4 --test` on port 8081: PASS ("Hello", instant).
- `--preset qwen36-35b-q4 --bench`: prefill 2644 t/s (8213 tokens), decode
  97 t/s (256 tokens), RSS 23.9 GB.
- Syntax check `bash -n` passes. `--help`, `--status` work.

### Downloads (SUPERSEDED - all finished, see section 6)

Durable resume script: `~/data/models/resilient-download.sh`
(copy of /tmp one; loops `curl -sL -C - --fail --retry 3 --retry-delay 5
--speed-time 30 --speed-limit 10240` until size matches, 5s between
attempts; appends to `<dest>.dl.log`).

Progress at pause:

| File | Have | Total | Pct |
| --- | --- | --- | --- |
| `~/data/models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf` | 13 634 588 672 | 16 817 244 384 | 81% |
| `~/data/models/qwen3.6-35b-a3b-q4_k_xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 9 219 059 712 | 22 853 663 008 | 40% |
| `~/data/models/qwen3.6-35b-a3b-q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` | 13 917 265 920 | 36 903 140 320 | 38% |

`qwen36-35b-q4` needed no download: it runs from
`~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

Important: child processes spawned from agent tool calls get killed when the
call ends (nohup does not survive). Anything long-running must go through a
launchd agent, e.g.:

```bash
LOG_DIR="$HOME/Library/Logs/llamacpp-dl"; mkdir -p "$LOG_DIR"
cat > "$HOME/Library/LaunchAgents/com.llamacpp.modeldownload.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.llamacpp.modeldownload</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>/tmp/model-download-wrapper.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>${LOG_DIR}/download.out.log</string>
  <key>StandardErrorPath</key><string>${LOG_DIR}/download.err.log</string>
</dict>
</plist>
PLIST
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.llamacpp.modeldownload.plist"
```

The wrapper script just runs the three `resilient-download.sh` invocations in
parallel with `&` and `wait` (recreate it if /tmp was wiped; contents
described above - one call per file with url/dest/expected-size). Unpause =
recreate wrapper + plist, bootstrap, and verify with
`pgrep -fl resilient-download` plus file growth after ~60s. Pause = `launchctl
bootout "gui/$(id -u)/com.llamacpp.modeldownload"` + remove plist + `pkill -f
resilient-download`.

URLs used (all work, resume supported via -C -):
- `https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf`
- `https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- `https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q8_0.gguf`

Expected sizes (verify before benching; a truncated file fails at load):
16 817 244 384 / 22 853 663 008 / 36 903 140 320 bytes.

### Current service state (SUPERSEDED - see section 6)

launchd service `com.llamacpp.server` (Mistral-Nemo-12B, port 8000) was
UNLOADED on 2026-09-22 (~17:00) with `launchctl bootout` before the first 1M
model test, per the user's one-model-at-a-time rule. The plist file is still
on disk at `~/Library/LaunchAgents/com.llamacpp.server.plist` (not deleted).
To restore Mistral later:
`launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.llamacpp.server.plist`

## 3. Key technical facts (why things are the way they are)

- **Server slot cap**: `tools/server/server-context.cpp` (~line 1309-1315)
  caps `n_ctx_slot` to `n_ctx_train` with a `capping` warning. This silently
  overrides any larger `--ctx-size`. Upstream knows: issue #22140, closed
  "not planned"; also #17459. User chose to accept 262 144 effective context.
  If user later wants true 1M: options are (a) local 2-line patch changing
  the cap to a warning (deviates from upstream, must be re-applied after
  pulls; user must be able to defend it), or (b) edit GGUF metadata
  context_length (makes the file lie; affects other tooling). Do NOT do this
  without the user explicitly asking again.
- **Qwen3.6-35B-A3B is a hybrid model** (arch `qwen35moe`, LLM_ARCH_QWEN35MOE
  in this repo): 40 layers, only 10 full-attention (every 4th), 30 linear
  attention (gated delta net, fixed-size state). KV cache ~20 KB/token f16:
  262 144 tokens ~ 5 GB f16 / ~2.5 GB q8_0; 1M tokens would be ~20 GB f16 /
  ~10 GB q8_0. Verified GGUF metadata: ctx 262144, n_embd 2048, 16 heads /
  2 KV heads, head_dim 128, 733 tensors, 54 kv pairs. `mmproj-BF16.gguf`
  exists in the HF repo for vision (NOT wired into presets).
- **YaRN**: generic llama.cpp support; factor comes from 1/rope_freq_scale
  (`--rope-scale 4`); `--yarn-orig-ctx 262144`. Applies only to RoPE
  sections of hybrid layers (rope.dimension_sections present in GGUF).
  Native max context is 262 144; 1M requires factor 4. Untested beyond
  native because of the slot cap.
- **Probes must disable thinking** (`chat_template_kwargs
  {"enable_thinking": false}`), otherwise Qwen3.6 spends the whole token
  budget on chain-of-thought and `message.content` comes back empty
  (first --test run failed exactly this way before the fix).
- MTP variants (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) exist and this repo has
  MTP support, but were deliberately not preset (different repo; speculative
  decoding changes benchmark comparability).

## 4. What remains (ordered)

1. **Resume downloads** (see section 2 for exact commands). Monitor
   periodically: `stat -f %z` each file vs expected size; the resilient loop
   handles reconnects itself, only intervene if a `.dl.log` shows repeated
   failures or processes are gone.
2. **Verify integrity**: after each file reaches expected size, run
   `./llamacpp-server.sh --preset <name> --test --port 8081` - a truncated
   GGUF fails to load and the log will say so.
3. **Bench all presets** with thinking-off probes:
   ```bash
   for p in qwen36-35b-q4 qwen36-35b-q4xl qwen36-35b-q8 qwen36-27b-q4; do
     ./llamacpp-server.sh --preset "$p" --bench --port 8081
   done
   ```
   (q4 is done: prefill 2644 t/s, decode 97 t/s, 23.9 GB - but re-run for a
   same-session comparison.) Consider also a long-decode bench
   (BENCH_PREDICT is a variable in the script, default 256) and a larger
   prefill (BENCH_TOKENS, default 8192) for stability. Record all numbers
   into llamacpp-server.md ("Measured performance" section has the LFM
   precedent format).
4. **Pick the winner** with the user (ask_user): quality vs speed vs memory.
   Note q8_0 (~37 GB weights) + ~2.5 GB KV + buffers ~ 45 GB wired, still OK
   on 128 GB.
5. **Swap the launchd service** to the winning preset (user already approved
   this step): re-run `./llamacpp-server.sh --install-service --preset
   <winner>` (check the script: install uses SERVE_ARGS built from preset +
   flags; the service will take over port 8000 from Mistral). Verify with
   `./llamacpp-server.sh --status` and a real chat completion on port 8000.
   Confirm the user is OK retiring Mistral-Nemo from the service.
6. **Cleanup**: remove `~/data/models/*.dl.log` and the download wrapper
   plist; optionally drop `/tmp/resilient-download.sh` copies. Update
   llamacpp-server.md "Measured performance" with final numbers. Leave the
   working tree uncommitted unless the user asks (repo rules: do not commit
   or push without explicit approval; AI-written commit messages must follow
   AGENTS.md, use `Assisted-by:` not `Co-authored-by:`).

## 5. Gotchas for the next agent

- Tool-call child processes die when the call ends: long downloads/benches
  either fit in one SYNC call with a generous timeout (benches: ~2-3 min
  each once downloaded) or need the launchd trick.
- The tool timeout maxes at 600s; a cold 17 GB download needs the launchd
  supervisor, not a single tool call.
- `--status` output lies if another llama-server runs on a different port;
  it only reports what pgrep finds.
- When testing presets, always pass `--port 8081` until the final swap, so
  the live Mistral service on 8000 is never disturbed.
- gguf-py is unusable (no numpy in either system python). Parse GGUF headers
  with the pure-python struct reader approach used earlier (see git log of
  this session if needed) - do not pip install into system python.
- The first `--test` FAIL with empty content was the thinking-budget issue,
  already fixed; if probes ever fail again with empty content, check that
  `chat_template_kwargs` survived in the payload.
- Repo is llama.cpp with strict contribution rules in AGENTS.md: no
  committing/pushing/PRs on the user's behalf, no AI-written PR
  descriptions; ASCII only (no em-dash), keep comments short and code-first.

## 6. Session update: 1M-context pivot (CURRENT WORK, 2026-09-22 ~17:00+)

### User request and rules

User wants 1M-token context; Qwen3.6 caps at 262 144 natively. A Gemini chat
recommended Llama 4 Maverick/Scout, GLM-5.3(-Flash) and DeepSeek V4-Flash.
User also asked: are there any Gemma models with 1M context?

**SAFETY RULE (user, repeated twice): only ONE LLM may be loaded at a time.
Loading two simultaneously KERNEL PANICS the machine.** Before testing any
model: check `pgrep -fl llama-server`; if anything is loaded, stop it first;
then load and test exactly one model. This is why Mistral was unloaded.

User approved downloads: Llama 4 Scout Q4_K_M + GLM-5.3-Flash IQ1_M ("Scout +
GLM IQ1_M; Let them finish"), and approved letting the Qwen downloads finish.
The Kimi-Linear 48B download was agent-initiated after research (user has not
objected; it is the best memory fit). Flag this to the user when picking the
final winner.

### Research verdict (verified against HF API file listings + this build's src/models/)

| Model | 1M ctx? | Verdict |
| --- | --- | --- |
| Llama 4 Maverick | yes | IMPOSSIBLE on 128 GB: Q4_K_M is ~245 GB, 1-bit is 114 GiB |
| DeepSeek V4-Flash | yes | REJECTED: 1-bit weights ~87+ GiB; best quant needs the dspark fork (unsupported here) |
| Gemma (incl. Gemma 4) | NO | Answer to user's question: no Gemma does 1M; Gemma 4 max is 256K native |
| MiniMax-M3 | yes | REJECTED: smallest quant 119.6 GiB |
| GLM-5.3-Flash (320B-A18B, hybrid sparse+linear attn) | yes | DOWNLOADING: unsloth UD-IQ1_M, 3 shards, ~97.6 GB total |
| Llama 4 Scout (109B-A17B, chunked SWA 8192 windows, 3:1 with full attn) | yes (10M native) | DOWNLOADING: bartowski Q4_K_M, 2 shards, ~62.9 GiB |
| Kimi-Linear 48B-A3B (MLA-style compressed KV) | yes | DOWNLOADING: bartowski Q4_K_M, single file 30 061 058 720 bytes |

### Script changes for the new presets (uncommitted)

`llamacpp-server.sh` presets added: `llama4-scout-q4`, `glm-5.3-flash-iq1`,
`kimi-linear-48b-q4`. All three default to 1 048 576 ctx, single slot.
`llamacpp-server.md` has a new section documenting them.

Kimi KV gotcha (FIXED): its compressed KV cache uses 72-dim heads, so q8_0
KV fails with "V cache type q8_0 with block size 32 does not divide
n_embd_head_v=72". Preset now defaults CACHE_TYPE_K/V to f16. If Scout or
GLM fail the same way on q8_0 KV, switch that preset to f16 too (q8_0 KV
requires head_dim divisible by 32).

### Download state (FINAL - all downloads complete, supervisor removed)

DONE at exact remote size, verified loadable: Kimi Q4_K_M (30 061 058 720),
GLM-5.3-Flash IQ1_M 3 shards (9 429 859 + 49 996 246 592 + 47 573 668 096),
Scout Q4_K_M 2 shards (39 837 275 008 + 27 708 903 776), Qwen 27B-Q4,
35B-Q8, and 35B-UD-Q4_K_XL (22 360 456 160 - note: HANDOFF section 2's XL
expected size 22 853 663 008 was STALE; the remote file is the smaller
number and the XL loads and benches fine, so it is complete). All resilient
loops stopped, launchd download plists and /tmp wrappers removed,
*.dl.log cleaned. Downloads can be redone from the URLs above if ever
needed (curl -C - resumable).

Model URLs (all resume-capable with `curl -C -`):
- Scout: `https://huggingface.co/bartowski/meta-llama_Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/meta-llama_Llama-4-Scout-17B-16E-Instruct-Q4_K_M/meta-llama_Llama-4-Scout-17B-16E-Instruct-Q4_K_M-0000{1,2}-of-00002.gguf`
- GLM: `https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/UD-IQ1_M/GLM-5.3-Flash-UD-IQ1_M-0000{1,2,3}-of-00003.gguf`
- Kimi: `https://huggingface.co/bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF/resolve/main/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf`

### Test log (sequential, one model at a time)

- qwen36-27b-q4 `--test` PASS; qwen36-35b-q4xl `--test` PASS; qwen36-35b-q8
  `--test` PASS (integrity of the finished downloads; nothing else loaded).
- kimi-linear-48b-q4: first `--test` FAILED (q8_0 KV / 72-dim heads, see
  above); after f16 fix, `--test` PASS ("Hello" in 2s). `--bench` at 1M ctx:
  prefill 2116 t/s (8209 tok), decode 118.4 t/s (200 tok), 36.4 GB RSS.
- glm-5.3-flash-iq1: CANNOT RUN on this build - GGUF arch is `glm5next`,
  unknown to llama.cpp (checked local master, origin, and ggml-org: support
  is an OPEN PR, "model : add GLM-5.3-Flash (glm5next)", plus an open
  softmax bug report for it). The 97.6 GB download is on disk and usable
  only after that PR lands locally. Do not delete the files without asking.
- llama4-scout-q4: downloads complete (shard1 exact 39 837 275 008).
  `--bench` at 1M ctx: prefill 634.7 t/s (8212 tok), decode 36.3 t/s (211
  tok), 83.4 GB RSS. Much slower and heavier than Kimi.
- NOT yet done: same-session re-bench of the qwen36 presets (old numbers:
  35b-q4 prefill 2644 t/s, decode 97 t/s, 23.9 GB at 256K).

### Remaining steps (ordered)

1. DONE: all downloads, all integrity tests, all benches (see test log and
   llamacpp-server.md "Measured performance" table). Nothing is loaded.
2. DONE 2026-09-23: user picked `kimi-linear-48b-q4` as the winner (fastest
   decode, only affordable true-1M option). GLM files stay on disk per user.
3. DONE 2026-09-23: service swapped to `kimi-linear-48b-q4` on port 8000.
   Verified: /health 200 in ~15 s, RSS 35.6 GB, real chat completion answers
   in ~1.3 s with thinking off. Port 8000 was held by `com.omlx.server` (MLX
   server, appeared 2026-09-22 17:34 after Mistral was unloaded; no model
   resident, nothing in ~/video-agent referenced it) - user chose to replace
   it: bootout com.omlx.server (plist kept on disk for restore), then
   bootstrap com.llamacpp.server. Do not run both servers with models
   resident at once (one-LLM rule). Restore omlx with: launchctl bootout
   gui/$(id -u)/com.llamacpp.server && launchctl bootstrap "gui/$(id -u)"
   ~/Library/LaunchAgents/com.omlx.server.plist
4. Cleanup is done. GLM-5.3-Flash files (~95 GB) stay on disk per user
   decision (2026-09-23); they remain unusable until the glm5next PR lands
   locally. Nothing committed (repo rules).

### New gotchas (this phase)

- Router DNS intermittently intercepts huggingface.co with a `*.eero.com`
  cert (curl TLS error 60 / "error 000"). resilient-download.sh retries
  through it; ad-hoc curl needs `--retry` or the hf-mirror.com fallback.
- Tests currently run on the default self-test port (8080) since nothing
  else is loaded; once a long-lived server is up again, go back to 8081.
- If the machine DID crash mid-test, this file is the state of record:
  re-check `pgrep -fl llama-server`, download sizes vs expected, and
  continue the test log from the top of this section.
