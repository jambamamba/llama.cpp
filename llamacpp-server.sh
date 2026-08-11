#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# llamacpp-server.sh - launch the llama.cpp inference server for a local GGUF
# model on Apple Silicon (Metal) or Linux.
#
# Runs `llama-server` against the model, exposing the OpenAI-compatible API on
# http://localhost:PORT. Can also install a launchd LaunchAgent (macOS) or a
# systemd user unit (Linux) that keeps the server running across reboots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
MODEL=""
HF_REPO=""
HF_FILE=""
PORT="8080"
ACTION="run"
CTX_SIZE="0"
N_GPU_LAYERS="all"
FLASH_ATTN="auto"
PARALLEL="2"
REASONING="auto"
REASONING_BUDGET="-1"
METRICS="off"

# Default model when --model is omitted: Liquid AI LFM2.5-8B-A1B Q8_0 (MoE,
# 8B total / 1.5B active, native 128 000-token context). Measured on this Mac
# (M5 Max, Mac17,6): ~9 GiB weights + small hybrid KV cache at 128K context,
# ~200+ tok/s generation on Metal. See llamacpp-server.md for the model
# rationale. Local copy is downloaded on first use from --hf-repo.
DEFAULT_MODEL="$HOME/data/models/lfm2.5-8b-a1b-q8_0/LFM2.5-8B-A1B-Q8_0.gguf"
DEFAULT_HF_REPO="LiquidAI/LFM2.5-8B-A1B-GGUF"
DEFAULT_HF_FILE="LFM2.5-8B-A1B-Q8_0.gguf"

usage() {
  cat <<'EOF'
Usage: llamacpp-server.sh [options]

Launches the llama.cpp inference server for a local GGUF model.

The server binds to http://localhost:PORT and serves the OpenAI-compatible
/v1/chat/completions, /v1/completions and /v1/embeddings endpoints. Full GPU
offload (Metal on Apple Silicon) and flash attention are enabled by default.
The model's chat template is applied via llama.cpp's built-in Jinja engine.

Options:
  --model <path>         Path to the model .gguf file. If omitted, the
                         default model is used: Liquid AI LFM2.5-8B-A1B Q8_0
                         (native 128 000-token context). If the file does not
                         exist yet and --hf-repo is given, it is downloaded
                         first.
  --hf-repo <repo_id>    Hugging Face repo that holds the .gguf file (e.g.
                         LiquidAI/LFM2.5-8B-A1B-GGUF). Used to download
                         --model when it is missing.
  --hf-file <name>       Filename to download from --hf-repo. Defaults to the
                         basename of --model.
  --port <port>          Port to listen on. Default: 8080.
  --ctx-size <n>         Size of the prompt context in tokens. Default: 0
                         (loaded from the model, 128 000 for the default
                         model). With N slots the per-slot context is
                         ctx-size / N.
  --gpu-layers <n|all>   Layers to offload to VRAM. Default: all.
  --flash-attn <on|off|auto>
                         Flash attention use. Default: auto.
  --parallel <n>         Number of parallel server slots. Default: 2.
  --reasoning <on|off|auto>
                         Use reasoning/thinking in chat. Default: auto.
  --reasoning-budget <n> Token budget for thinking: -1 for unrestricted,
                         0 for immediate end, N>0 for a token budget.
                         Default: -1.
  --metrics              Enable the Prometheus metrics endpoint.
  --test                 Self-test: start the server, wait for it to be
                         ready, send a short chat-completion prompt, and
                         confirm a non-empty reply, then shut the server
                         down. If a server is already answering on --port,
                         it is tested and left running. Prints PASS/FAIL and
                         timings; exits non-zero on failure.
  --status               Print the current status: whether the service is
                         installed and running, whether a server process is
                         up, and the port + loaded model (queried from the
                         live /v1/models endpoint). Starts nothing.
  --install-service      Install as a service instead of running in the
                         foreground: a launchd LaunchAgent on macOS, or a
                         systemd user unit on Linux. The service restarts on
                         failure and starts at login. Logs: see below.
  -h, --help             Show this help and exit.

Examples:
   ./llamacpp-server.sh
   ./llamacpp-server.sh --model ~/models/llama.gguf --port 8081
   ./llamacpp-server.sh --model ~/models/model.gguf --test
   # Download the gguf, then run:
   ./llamacpp-server.sh --model ~/data/models/lfm/LFM2.5-8B-A1B-Q8_0.gguf \
       --hf-repo LiquidAI/LFM2.5-8B-A1B-GGUF

Service management:
  macOS (launchd):
    service: ~/Library/LaunchAgents/com.llamacpp.server.plist
    logs:    ~/Library/Logs/llamacpp/llamacpp-server.{out,err}.log
    control: launchctl print gui/$(id -u)/com.llamacpp.server   (status)
             launchctl bootout gui/$(id -u)/com.llamacpp.server (stop/remove)
  Linux (systemd, user scope):
    unit:  ~/.config/systemd/user/llamacpp-server.service
    logs:  journalctl --user -u llamacpp-server.service
    control: systemctl --user status llamacpp-server.service
             systemctl --user stop llamacpp-server.service
EOF
}

error() {
  echo -e "Error: $*" >&2
}

warn() {
  echo -e "Warning: $*" >&2
}

# Resolve the llama-server binary: prefer the repo build, fall back to PATH.
resolve_bin() {
  if [ -x "${SCRIPT_DIR}/build/bin/llama-server" ]; then
    LLAMA_SERVER="${SCRIPT_DIR}/build/bin/llama-server"
  elif command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER="$(command -v llama-server)"
  else
    error "Cannot find the llama-server binary. Build llama.cpp first:"
    error "  cmake -B ${SCRIPT_DIR}/build -DCMAKE_BUILD_TYPE=Release"
    error "  cmake --build ${SCRIPT_DIR}/build --target llama-server"
    return 1
  fi
}

# Build the llama-server argv into the global SERVE_ARGS array.
build_serve_args() {
  SERVE_ARGS=(-m "$MODEL" --host 0.0.0.0 --port "$PORT")
  if [ -n "$CTX_SIZE" ] && [ "$CTX_SIZE" != "0" ]; then
    SERVE_ARGS+=(--ctx-size "$CTX_SIZE")
  fi
  if [ -n "$N_GPU_LAYERS" ]; then
    SERVE_ARGS+=(--n-gpu-layers "$N_GPU_LAYERS")
  fi
  if [ -n "$FLASH_ATTN" ]; then
    SERVE_ARGS+=(--flash-attn "$FLASH_ATTN")
  fi
  if [ -n "$PARALLEL" ]; then
    SERVE_ARGS+=(--parallel "$PARALLEL")
  fi
  if [ -n "$REASONING" ]; then
    SERVE_ARGS+=(--reasoning "$REASONING")
  fi
  if [ -n "$REASONING_BUDGET" ] && [ "$REASONING_BUDGET" != "-1" ]; then
    SERVE_ARGS+=(--reasoning-budget "$REASONING_BUDGET")
  fi
  if [ "$METRICS" = "on" ]; then
    SERVE_ARGS+=(--metrics)
  fi
}

# Ensure the model file exists, downloading it from --hf-repo when the given
# path is missing.
ensure_model() {
  if [ -f "$MODEL" ]; then
    return 0
  fi
  if [ -z "$HF_REPO" ]; then
    error "Model file not found: ${MODEL}"
    error "Pass --hf-repo <repo> to download it, or fix --model."
    return 1
  fi
  local file="${HF_FILE:-$(basename "$MODEL")}"
  local model_dir="$(dirname "$MODEL")"
  [ "$model_dir" = "." ] && model_dir="$PWD"
  mkdir -p "$model_dir"
  echo "Downloading ${HF_REPO}/${file} ..."
  curl -sL --fail -o "$MODEL" "https://huggingface.co/${HF_REPO}/resolve/main/${file}" \
    || { error "Download failed: ${file} was not fetched from ${HF_REPO}."; return 1; }
}

# Allow the llama-server binary through macOS's application firewall
# (macOS filters per-application, not per-port). Best-effort: no-ops on
# Linux, and warns (rather than fails) when sudo is unavailable.
allow_firewall() {
  [ "$(uname -s)" = "Darwin" ] || return 0

  local fw="/usr/libexec/ApplicationFirewall/socketfilterfw"
  local state blockall exe

  state="$("$fw" --getglobalstate 2>/dev/null)" || state=""
  case "$state" in
    *enabled*|"State = 1"*) ;;
    *) return 0 ;; # firewall off -> nothing to filter
  esac

  blockall="$("$fw" --getblockall 2>/dev/null)" || blockall=""
  if [ "$blockall" = "Block All is enabled." ]; then
    warn "Firewall is set to block all incoming connections; LAN clients"
    warn "will be refused regardless of the allow rule below."
  fi

  exe="$(cd "$(dirname "$LLAMA_SERVER")" && pwd)/$(basename "$LLAMA_SERVER")"
  if ! sudo -n "$fw" --add "$exe" >/dev/null 2>&1; then
    if [ -t 0 ]; then
      sudo "$fw" --add "$exe" >/dev/null 2>&1 \
        || warn "Could not allow $exe through the firewall (is sudo available?)."
    else
      warn "Non-interactive shell: cannot allow $exe through the firewall"
      warn "without sudo. Re-run once interactively to enable LAN access."
    fi
  fi
}

install_launchd() {
  local label="com.llamacpp.server"
  local plist_dir="$HOME/Library/LaunchAgents"
  local log_dir="$HOME/Library/Logs/llamacpp"
  local plist="$plist_dir/${label}.plist"
  local uid
  uid="$(id -u)"

  mkdir -p "$plist_dir" "$log_dir"

  local args_xml=""
  local arg
  for arg in "${SERVE_ARGS[@]}"; do
    args_xml+="    <string>${arg}</string>\n"
  done

  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${LLAMA_SERVER}</string>
$(printf '%b' "$args_xml")
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>$(dirname "$MODEL")</string>
  <key>StandardOutPath</key>
  <string>${log_dir}/llamacpp-server.out.log</string>
  <key>StandardErrorPath</key>
  <string>${log_dir}/llamacpp-server.err.log</string>
</dict>
</plist>
PLIST

  if ! plutil -lint "$plist" > /dev/null; then
    error "Generated invalid plist: ${plist}"
    return 1
  fi

  launchctl bootout "gui/${uid}" "$plist" 2>/dev/null || true
  if ! launchctl bootstrap "gui/${uid}" "$plist"; then
    error "Failed to load launchd service ${label}."
    return 1
  fi

  echo "Installed launchd service: ${plist}"
  echo "Logs: ${log_dir}/llamacpp-server.out.log and ${log_dir}/llamacpp-server.err.log"
  echo "Status: launchctl print gui/${uid}/${label}"
}

install_systemd() {
  local unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  local unit="$unit_dir/llamacpp-server.service"

  mkdir -p "$unit_dir"

  local quoted=()
  local arg
  for arg in "${SERVE_ARGS[@]}"; do
    quoted+=("$(printf '%q' "$arg")")
  done

  cat > "$unit" <<UNIT
[Unit]
Description=llama.cpp server (${MODEL})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${LLAMA_SERVER} ${quoted[*]}
WorkingDirectory=$(dirname "$MODEL")
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT

  if ! systemctl --user daemon-reload; then
    error "Failed to reload systemd (is systemd --user running?)."
    return 1
  fi
  if ! systemctl --user enable --now llamacpp-server.service; then
    error "Failed to enable/start llamacpp-server.service."
    return 1
  fi

  echo "Installed systemd user service: ${unit}"
  echo "Logs: journalctl --user -u llamacpp-server.service"
  echo "Status: systemctl --user status llamacpp-server.service"
}

run_server() {
  exec "${LLAMA_SERVER}" "${SERVE_ARGS[@]}"
}

# Start the server (if none is already answering on PORT), wait for readiness,
# prompt it via /v1/chat/completions, confirm a non-empty reply, then stop the
# server we started. Tests against an already-running server without killing it.
# The default model thinks before answering, so the probe uses a generous
# max_tokens and reads message.content (reasoning lands in reasoning_content).
test_server() {
  local url="http://127.0.0.1:${PORT}"
  local log_file="${TMPDIR:-/tmp}/llamacpp-server-test-$$.log"
  local deadline=$((SECONDS + 600))
  local reply elapsed prompt_start

  if curl -sf "${url}/v1/models" >/dev/null 2>&1; then
    echo "A server already answers on port ${PORT}; testing it (leaving it running)."
    SERVER_PID=""
  else
    echo "Starting server on port ${PORT} for self-test (log: ${log_file}) ..."
    "${LLAMA_SERVER}" "${SERVE_ARGS[@]}" >"$log_file" 2>&1 &
    SERVER_PID=$!
  fi

  while ! curl -sf "${url}/v1/models" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      error "Server did not become ready within $((deadline - SECONDS + 600))s."
      error "See the log: ${log_file}"
      [ -z "${SERVER_PID:-}" ] || kill "$SERVER_PID" 2>/dev/null || true
      return 1
    fi
    if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      error "Server process exited during startup; see the log: ${log_file}"
      return 1
    fi
    sleep 2
  done

  prompt_start="$SECONDS"
  reply="$(
    curl -s -m 180 "${url}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hello in one short word."}],"max_tokens":256,"temperature":0.2}' \
      | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("", end="")
    raise SystemExit(1)
print(d.get("choices", [{}])[0].get("message", {}).get("content", ""))
'
  )" || reply=""
  elapsed=$((SECONDS - prompt_start))

  if [ -n "$reply" ]; then
    echo "PASS: server responded in ${elapsed}s: ${reply}"
    if [ -n "${SERVER_PID:-}" ]; then
      echo "Stopping self-test server (PID ${SERVER_PID})."
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
    fi
    return 0
  fi

  error "FAIL: no usable reply within 180s (server log: ${log_file})."
  [ -z "${SERVER_PID:-}" ] || kill "$SERVER_PID" 2>/dev/null || true
  return 1
}

# Print the current state: service installed?, server process running?, and
# the port + model actually in use. Reads the live /v1/models endpoint when
# the server is reachable; never starts anything.
status_report() {
  local uname_s="$(uname -s)"
  local label="com.llamacpp.server"
  local plist="$HOME/Library/LaunchAgents/${label}.plist"
  local unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/llamacpp-server.service"
  local uid
  uid="$(id -u)"

  echo "llama.cpp server status"
  echo "-----------------------"

  # Service installed? launchd on macOS, systemd user unit on Linux.
  if [ "$uname_s" = "Darwin" ] && [ -f "$plist" ]; then
    if launchctl print "gui/${uid}/${label}" 2>/dev/null | grep -q "state = running"; then
      echo "Service:          installed (launchd ${label}) and RUNNING"
    else
      echo "Service:          installed (launchd ${label}) but NOT running"
    fi
  elif [ "$uname_s" = "Linux" ] && [ -f "$unit" ]; then
    if systemctl --user is-active --quiet llamacpp-server.service 2>/dev/null; then
      echo "Service:          installed (systemd) and RUNNING"
    else
      echo "Service:          installed (systemd) but NOT running"
    fi
  else
    echo "Service:          NOT installed"
  fi

  # Running llama-server processes (any started by hand or by the service).
  local pids pid args port model prev tok line
  pids="$(pgrep -f 'llama-server' 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    echo "Server process:   NOT running"
    return 0
  fi

  for pid in $pids; do
    args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    case "$args" in
      *llama-server*) ;;
      *) continue ;;
    esac
    port=""
    model=""
    prev=""
    for tok in $args; do
      case "$prev" in
        -m|--model) model="$tok" ;;
        --port) port="$tok" ;;
      esac
      prev="$tok"
    done
    echo "Server process:   RUNNING - PID ${pid}, port ${port:-?}"
    [ -n "$model" ] && echo "Model (args):     ${model}"

    [ -n "$port" ] || continue
    line="$(curl -sf -m 5 "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
      | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
data = d.get("data") or d.get("models") or []
m = data[0] if data else {}
mid = m.get("id") or m.get("name") or ""
ctx = (m.get("meta") or {}).get("n_ctx", "")
print(mid + "|" + str(ctx))
' 2>/dev/null || true)"
    if [ -n "$line" ]; then
      echo "Model loaded:     ${line%%|*}"
      echo "Context:          ${line##*|} tokens"
      echo "HTTP:             ok at http://127.0.0.1:${port}"
    else
      echo "HTTP:             no response on port ${port}"
    fi
  done
}

main() {
  set -euo pipefail

  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --model)
        [ $# -ge 2 ] || { error "--model requires a path argument."; exit 1; }
        MODEL="$2"
        shift 2
        ;;
      --hf-repo)
        [ $# -ge 2 ] || { error "--hf-repo requires a repo id."; exit 1; }
        HF_REPO="$2"
        shift 2
        ;;
      --hf-file)
        [ $# -ge 2 ] || { error "--hf-file requires a filename."; exit 1; }
        HF_FILE="$2"
        shift 2
        ;;
      --port)
        [ $# -ge 2 ] || { error "--port requires a port number."; exit 1; }
        PORT="$2"
        shift 2
        ;;
      --ctx-size)
        [ $# -ge 2 ] || { error "--ctx-size requires a token count."; exit 1; }
        CTX_SIZE="$2"
        shift 2
        ;;
      --gpu-layers)
        [ $# -ge 2 ] || { error "--gpu-layers requires a value."; exit 1; }
        N_GPU_LAYERS="$2"
        shift 2
        ;;
      --flash-attn)
        [ $# -ge 2 ] || { error "--flash-attn requires on|off|auto."; exit 1; }
        FLASH_ATTN="$2"
        shift 2
        ;;
      --parallel)
        [ $# -ge 2 ] || { error "--parallel requires a slot count."; exit 1; }
        PARALLEL="$2"
        shift 2
        ;;
      --reasoning)
        [ $# -ge 2 ] || { error "--reasoning requires on|off|auto."; exit 1; }
        REASONING="$2"
        shift 2
        ;;
      --reasoning-budget)
        [ $# -ge 2 ] || { error "--reasoning-budget requires a token count."; exit 1; }
        REASONING_BUDGET="$2"
        shift 2
        ;;
      --metrics)
        METRICS="on"
        shift
        ;;
      --install-service)
        ACTION="install-service"
        shift
        ;;
      --test)
        ACTION="test"
        shift
        ;;
      --status)
        ACTION="status"
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        error "Unknown argument: $1"
        echo "Run with --help for usage." >&2
        exit 1
        ;;
    esac
  done

  if [ "$ACTION" = "status" ]; then
    status_report
    exit 0
  fi

  if [ -z "$MODEL" ]; then
    echo "No --model given; using the default model (${DEFAULT_MODEL})."
    MODEL="$DEFAULT_MODEL"
    [ -n "$HF_REPO" ] || HF_REPO="$DEFAULT_HF_REPO"
    [ -n "$HF_FILE" ] || HF_FILE="$DEFAULT_HF_FILE"
  fi
  if [ ! -f "$MODEL" ] && [ -z "$HF_REPO" ]; then
    error "Model file not found: ${MODEL}"
    echo "Pass --hf-repo <repo> to download it, or run with --help for usage." >&2
    exit 1
  fi
  if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    error "Invalid port: ${PORT} (expected 1-65535)."
    exit 1
  fi

  if ! resolve_bin; then
    exit 1
  fi
  if ! ensure_model; then
    exit 1
  fi
  build_serve_args

  # Allow the listening binary through macOS's application firewall before
  # the server starts (or before the service is scheduled to start).
  allow_firewall

  case "$ACTION" in
    install-service)
      case "$(uname -s)" in
        Darwin) install_launchd ;;
        Linux)  install_systemd ;;
        *)
          error "Unsupported OS for --install-service: $(uname -s)"
          error "Supported: Darwin (launchd), Linux (systemd)."
          exit 1
          ;;
      esac
      ;;
    run)
      run_server
      ;;
    test)
      test_server
      ;;
  esac
}

main "$@"
