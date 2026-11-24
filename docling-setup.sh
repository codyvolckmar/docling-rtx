#!/usr/bin/env bash
# docling-setup.sh - set up GPU document processing on a fresh Linux machine with an NVIDIA RTX card:
# Docling (PDF/Office/HTML/image -> Markdown/JSON/HTML) on CUDA PyTorch, plus IBM's granite-docling-258M
# vision-language model served by vLLM (the fast path from Docling's RTX guide).
#
#   ./docling-setup.sh               whiptail menu
#   ./docling-setup.sh install       full install, no questions (also: --auto, -y)
#   ./docling-setup.sh start|stop|restart|status|test|uninstall
#   ./docling-setup.sh --dry-run     show what install would do, change nothing
#
# What it installs (nothing outside these, apart from apt packages and the optional systemd unit):
#   ~/.local/bin/uv                    Python/package manager (brings its own Python 3.12)
#   $DOCLING_HOME/venv                 docling + CUDA PyTorch (cu126/cu128/cu130, picked from your driver)
#   $DOCLING_HOME/vllm                 vLLM in its own venv (it pins its own torch version)
#   $DOCLING_HOME/convert.py           batch converter; launcher: ~/.local/bin/docling-convert
#   ~/.cache/huggingface               model weights (granite-docling-258M, Docling's layout/table models)
#   docling-vllm.service               systemd unit (or a background process where systemd isn't running)
# You only need the NVIDIA driver: the PyTorch and vLLM wheels ship their own CUDA runtime and cuDNN, so the
# CUDA Toolkit and cuDNN steps of the Docling RTX guide are not needed.
#
# Settings (environment or $DOCLING_HOME/config.env; the menu edits the file):
#   DOCLING_HOME=~/.docling-rtx  PORT=8000  GPU_UTIL=0.9  BATCH=auto  CUDA=auto (cu126|cu128|cu130|cpu)
#   DOCLING_GPU="RTX 4090:24564:575.64"   pretend this GPU (name:MB:driver), for --dry-run previews
set -euo pipefail

VERSION=1.0.0
MODEL=ibm-granite/granite-docling-258M
MODEL_REV=untied   # the model card's vLLM example loads this branch (untied embedding weights)
DOCLING_HOME=${DOCLING_HOME:-$HOME/.docling-rtx}
CONF=$DOCLING_HOME/config.env
LOG=$DOCLING_HOME/install.log
SERVICE=docling-vllm
SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)
REPO_RAW=${REPO_RAW:-}   # where to fetch convert.py when this script runs without the repo next to it
TEST_URL=https://arxiv.org/pdf/2501.17887   # the Docling paper, as in the model card's examples
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
DRY=0

# --- helpers ---------------------------------------------------------------------------------------------
say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then echo "   [dry-run] $*"; else "$@"; fi; }
have() { command -v "$1" >/dev/null 2>&1; }
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
interactive() { [ -t 1 ] && [ -r /dev/tty ] && have whiptail; }
wt() { whiptail --backtitle "Docling RTX setup $VERSION" "$@" 3>&1 1>&2 2>&3 </dev/tty; }

load_conf() {   # environment beats config.env beats defaults
  local p=${PORT:-} g=${GPU_UTIL:-} b=${BATCH:-} c=${CUDA:-}
  if [ -f "$CONF" ]; then . "$CONF"; fi
  PORT=${p:-${PORT:-8000}} GPU_UTIL=${g:-${GPU_UTIL:-0.9}} BATCH=${b:-${BATCH:-auto}} CUDA=${c:-${CUDA:-auto}}
}
save_conf() {
  mkdir -p "$DOCLING_HOME"
  printf 'PORT=%s\nGPU_UTIL=%s\nBATCH=%s\nCUDA=%s\n' "$PORT" "$GPU_UTIL" "$BATCH" "$CUDA" > "$CONF"
}

# --- detection -------------------------------------------------------------------------------------------
GPU_NAME="" GPU_MB="" DRIVER=""
detect_gpu() {
  if [ -n "${DOCLING_GPU:-}" ]; then   # preview a card you don't have: DOCLING_GPU="RTX 4090:24564:575.64"
    IFS=: read -r GPU_NAME GPU_MB DRIVER <<< "$DOCLING_GPU"; return
  fi
  if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  fi
}
cuda_flavor() {   # PyTorch wheel matching the driver (cu130 needs driver >= 580, cu128 >= 570, cu126 >= 560)
  [ "$CUDA" != auto ] && { echo "$CUDA"; return; }
  [ -z "$DRIVER" ] && { echo cpu; return; }
  local major=${DRIVER%%.*}
  if   [ "$major" -ge 580 ]; then echo cu130
  elif [ "$major" -ge 570 ]; then echo cu128
  elif [ "$major" -ge 560 ]; then echo cu126
  else echo old; fi
}
auto_batch() {    # Docling's RTX guide: 32 GB -> 64-128, 24 GB -> 32-64, 12 GB -> 16-32
  [ "$BATCH" != auto ] && { echo "$BATCH"; return; }
  if   [ "${GPU_MB:-0}" -ge 30000 ]; then echo 64
  elif [ "${GPU_MB:-0}" -ge 20000 ]; then echo 32
  else echo 16; fi
}
has_systemd() { [ -d /run/systemd/system ] && have systemctl; }

# --- install steps ---------------------------------------------------------------------------------------
step_preflight() {
  say "Checking the machine"
  [ "$(uname -s)" = Linux ] || die "Linux only (on Windows, run this inside WSL 2)."
  have apt-get || die "Needs an apt-based distro (Ubuntu 22.04/24.04, Debian 12)."
  [ -n "$SUDO" ] && ! have sudo && die "Run as root or install sudo."
  detect_gpu
  if [ -n "$GPU_NAME" ]; then
    ok "GPU: $GPU_NAME, $(((GPU_MB + 512) / 1024)) GB, driver $DRIVER"
    [ "$(cuda_flavor)" = old ] && die "Driver $DRIVER is too old: install driver 570 or newer (menu: NVIDIA driver)."
  else
    warn "No working NVIDIA driver (nvidia-smi failed). Install the driver first (menu: NVIDIA driver), or"
    warn "continue CPU-only: Docling works, vLLM is skipped."
  fi
  local free_gb; free_gb=$({ df -Pk "$HOME" 2>/dev/null || df -Pk /; } | awk 'NR==2{print int($4/1048576)}')
  [ "$free_gb" -ge 20 ] || warn "Only ${free_gb} GB free in $HOME; the full install needs ~20 GB."
}

step_driver() {
  detect_gpu
  if [ -n "$DRIVER" ]; then ok "NVIDIA driver $DRIVER already working"; return; fi
  say "Installing the NVIDIA driver"
  if have ubuntu-drivers || grep -qi ubuntu /etc/os-release; then
    run $SUDO apt-get update -q
    run $SUDO apt-get install -y -q ubuntu-drivers-common
    run $SUDO ubuntu-drivers install
    warn "Reboot now, then run this script again."
    [ "$DRY" = 1 ] || exit 0
  else
    die "Automatic driver install is Ubuntu-only. Install it from https://www.nvidia.com/Download/index.aspx, reboot, rerun."
  fi
}

step_apt() {
  say "System packages"
  run $SUDO apt-get update -q
  # libgl1/libglib2.0-0: OpenCV (used by Docling's OCR); build-essential: Triton compiles small kernels at runtime.
  run $SUDO apt-get install -y -q --no-install-recommends ca-certificates curl git whiptail \
    build-essential libgl1 libglib2.0-0 screen
}

step_uv() {
  if have uv; then ok "uv $(uv --version | cut -d' ' -f2) present"; return; fi
  say "Installing uv"
  run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
}

step_docling() {
  local flavor; detect_gpu; flavor=$(cuda_flavor)
  [ "$flavor" = old ] && flavor=cpu
  say "Docling + PyTorch ($flavor) in $DOCLING_HOME/venv"
  run mkdir -p "$DOCLING_HOME"
  [ -x "$DOCLING_HOME/venv/bin/python" ] || run uv venv -q --python 3.12 "$DOCLING_HOME/venv"
  export VIRTUAL_ENV=$DOCLING_HOME/venv
  local have_build   # torch already there but a different build (e.g. +cpu after a driver install): swap it
  have_build=$("$VIRTUAL_ENV/bin/python" -c 'import torch;print(torch.__version__.partition("+")[2])' 2>/dev/null || true)
  if [ -n "$have_build" ] && [ "$have_build" != "$flavor" ]; then
    say "Replacing torch +$have_build with +$flavor"; run uv pip uninstall -q torch torchvision
  fi
  run uv pip install -q --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$flavor"
  run uv pip install -q --upgrade docling
  unset VIRTUAL_ENV
  install_launchers
  [ "$DRY" = 1 ] && return
  "$DOCLING_HOME/venv/bin/python" - <<'EOF'
import torch, importlib.metadata as m
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f" ok docling {m.version('docling')}, torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {gpu}")
EOF
}

install_launchers() {
  local src=$SRC_DIR/convert.py
  if [ -f "$src" ]; then run cp "$src" "$DOCLING_HOME/convert.py"
  elif [ -n "$REPO_RAW" ]; then run curl -fsSL "$REPO_RAW/convert.py" -o "$DOCLING_HOME/convert.py"
  else warn "convert.py not found next to this script (set REPO_RAW=... to download it)."; fi
  [ -f "$SRC_DIR/GUIDE.md" ] && run cp "$SRC_DIR/GUIDE.md" "$DOCLING_HOME/GUIDE.md"
  run mkdir -p "$HOME/.local/bin"
  [ "$DRY" = 1 ] && return
  cat > "$HOME/.local/bin/docling-convert" <<EOF
#!/bin/sh
exec "$DOCLING_HOME/venv/bin/python" "$DOCLING_HOME/convert.py" "\$@"
EOF
  chmod +x "$HOME/.local/bin/docling-convert"
  ln -sf "$DOCLING_HOME/venv/bin/docling" "$HOME/.local/bin/docling"
  ok "Commands: docling-convert, docling (in ~/.local/bin)"
}

step_vllm() {
  detect_gpu
  [ -z "$DRIVER" ] && { warn "No NVIDIA driver: skipping vLLM."; return; }
  say "vLLM in $DOCLING_HOME/vllm (its own venv: it pins its own torch)"
  [ -x "$DOCLING_HOME/vllm/bin/python" ] || run uv venv -q --python 3.12 "$DOCLING_HOME/vllm"
  VIRTUAL_ENV=$DOCLING_HOME/vllm run uv pip install -q --upgrade vllm --torch-backend=auto
  [ "$DRY" = 1 ] || ok "vLLM $("$DOCLING_HOME/vllm/bin/python" -c 'import importlib.metadata as m;print(m.version("vllm"))')"
}

step_models() {
  say "Downloading models (granite-docling-258M for vLLM and inline use)"
  local py=$DOCLING_HOME/venv/bin
  run "$py/hf" download "$MODEL" --revision "$MODEL_REV" --quiet
  run "$py/hf" download "$MODEL" --quiet
  ok "Docling's layout/table/OCR models download on first use (the test step triggers it)."
}

vllm_cmd() {
  echo "$DOCLING_HOME/vllm/bin/vllm serve $MODEL --revision $MODEL_REV --served-model-name $MODEL" \
       "--host 127.0.0.1 --port $PORT --max-num-seqs 512 --max-num-batched-tokens 8192" \
       "--enable-chunked-prefill --gpu-memory-utilization $GPU_UTIL"
}

step_service() {
  [ -x "$DOCLING_HOME/vllm/bin/vllm" ] || [ "$DRY" = 1 ] || { warn "vLLM not installed: no service."; return; }
  if has_systemd; then
    say "systemd service $SERVICE (starts on boot)"
    local unit
    unit="[Unit]
Description=granite-docling-258M on vLLM (Docling VLM server)
After=network-online.target

[Service]
User=$(id -un)
Environment=HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
Environment=HF_HUB_DISABLE_TELEMETRY=1
Environment=VLLM_NO_USAGE_STATS=1
ExecStart=$(vllm_cmd)
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target"
    if [ "$DRY" = 1 ]; then echo "   [dry-run] write /etc/systemd/system/$SERVICE.service:"; echo "             ExecStart=$(vllm_cmd)"
    else echo "$unit" | $SUDO tee /etc/systemd/system/$SERVICE.service >/dev/null; fi
    run $SUDO systemctl daemon-reload
    run $SUDO systemctl enable -q $SERVICE
  else
    ok "No systemd here: the server runs as a background process (rerun 'start' after a reboot)."
  fi
}

# --- server control --------------------------------------------------------------------------------------
pidfile=$DOCLING_HOME/vllm.pid
healthy() { curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }
svc_unit() { has_systemd && [ -f /etc/systemd/system/$SERVICE.service ]; }

server_start() {
  [ -x "$DOCLING_HOME/vllm/bin/vllm" ] || die "vLLM isn't installed (menu: Install)."
  healthy && { ok "Already running on port $PORT"; return; }
  say "Starting vLLM on 127.0.0.1:$PORT (first start takes a minute or two: CUDA graphs are compiled)"
  if svc_unit; then
    $SUDO systemctl restart $SERVICE
  else
    VLLM_NO_USAGE_STATS=1 setsid nohup $(vllm_cmd) > "$DOCLING_HOME/vllm.log" 2>&1 < /dev/null &
    echo $! > "$pidfile"
  fi
  for _ in $(seq 1 120); do
    healthy && { ok "vLLM is up: http://127.0.0.1:$PORT/v1 (model $MODEL)"; return; }
    if ! svc_unit && ! kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then break; fi
    sleep 5
  done
  warn "vLLM didn't come up. Last log lines:"; server_logs | tail -25; return 1
}
server_stop() {
  if svc_unit; then $SUDO systemctl stop $SERVICE; ok "vLLM stopped"
  elif [ -f "$pidfile" ]; then kill "$(cat "$pidfile")" 2>/dev/null || true; rm -f "$pidfile"; ok "vLLM stopped"
  else ok "vLLM wasn't running"; fi
}
server_logs() {
  if svc_unit; then $SUDO journalctl -u $SERVICE -n 200 --no-pager; else cat "$DOCLING_HOME/vllm.log" 2>/dev/null; fi
}

status() {
  load_conf; detect_gpu
  echo "Docling RTX $VERSION   home: $DOCLING_HOME"
  echo "GPU:      ${GPU_NAME:-none} ${GPU_MB:+($(((GPU_MB + 512) / 1024)) GB, driver $DRIVER)}"
  if [ -x "$DOCLING_HOME/venv/bin/python" ]; then
    "$DOCLING_HOME/venv/bin/python" -c 'import torch, importlib.metadata as m; print("Docling:  %s (torch %s, CUDA %s)" % (m.version("docling"), torch.__version__, torch.cuda.is_available()))' 2>/dev/null || echo "Docling:  broken install (rerun install)"
  else echo "Docling:  not installed"; fi
  [ -x "$DOCLING_HOME/vllm/bin/vllm" ] && echo "vLLM:     installed" || echo "vLLM:     not installed"
  if healthy; then echo "Server:   UP   http://127.0.0.1:$PORT/v1  ($MODEL)"; else echo "Server:   down (port $PORT)"; fi
  svc_unit && echo "Service:  $SERVICE ($(systemctl is-enabled $SERVICE 2>/dev/null), starts on boot)"
  echo "Settings: PORT=$PORT GPU_UTIL=$GPU_UTIL BATCH=$(auto_batch) CUDA=$(cuda_flavor)"
  have nvidia-smi && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/GPU now:  /'
  return 0
}

test_run() {
  load_conf; detect_gpu
  local out=$DOCLING_HOME/test-output b; b=$(auto_batch)
  say "Test 1: standard pipeline (layout + tables + OCR on the GPU) on $TEST_URL"
  "$HOME/.local/bin/docling-convert" "$TEST_URL" -o "$out/standard" --batch "$b" --overwrite
  if healthy; then
    say "Test 2: VLM pipeline through vLLM (granite-docling-258M)"
    "$HOME/.local/bin/docling-convert" "$TEST_URL" --engine vlm -o "$out/vlm" --batch "$b" \
      --server "http://127.0.0.1:$PORT/v1/chat/completions" --overwrite
  else
    warn "vLLM isn't running: skipped the VLM test (start it, then rerun test)."
  fi
  ok "Results in $out (open the .md files)"
}

uninstall() {
  server_stop || true
  if svc_unit; then $SUDO systemctl disable -q $SERVICE; $SUDO rm -f /etc/systemd/system/$SERVICE.service; $SUDO systemctl daemon-reload; fi
  rm -f "$HOME/.local/bin/docling-convert" "$HOME/.local/bin/docling"
  rm -rf "$DOCLING_HOME"
  ok "Removed $DOCLING_HOME, the launchers and the service. Kept: uv, apt packages, the driver, ~/.cache/huggingface."
}

ALL_STEPS="preflight apt uv docling vllm models service"
install_steps() {   # $@ = steps
  load_conf
  [ "$DRY" = 1 ] || save_conf
  for s in "$@"; do "step_$s"; done
  if [[ " $* " == *" service "* ]] && [ "$DRY" = 0 ] && [ -x "$DOCLING_HOME/vllm/bin/vllm" ]; then server_start || true; fi
  [ "$DRY" = 1 ] && { echo; ok "Dry run: nothing was changed."; return; }
  echo; ok "Installed. Next:  docling-convert --help   (guide: $DOCLING_HOME/GUIDE.md)"
  [[ ":$PATH:" == *":$HOME/.local/bin:"* ]] || warn "Open a new shell (or: export PATH=\$HOME/.local/bin:\$PATH)"
}

# --- whiptail menu ---------------------------------------------------------------------------------------
menu_settings() {
  local v
  v=$(wt --title "Port" --inputbox "Port for the vLLM server (localhost only):" 9 60 "$PORT") && PORT=$v
  v=$(wt --title "GPU memory" --inputbox "Share of GPU memory vLLM reserves (0.1-0.95).\n0.9 = fastest; lower it (e.g. 0.4) to also run the standard pipeline on the same card:" 11 70 "$GPU_UTIL") && GPU_UTIL=$v
  v=$(wt --title "Batch size" --inputbox "Pages in flight per conversion (auto = from VRAM: 64 for 32 GB, 32 for 24 GB, 16 below):" 10 70 "$BATCH") && BATCH=$v
  v=$(wt --title "PyTorch CUDA build" --radiolist "PyTorch wheel for Docling (auto = from your driver: $(cuda_flavor)):" 14 70 5 \
      auto "pick from the driver" "$([ "$CUDA" = auto ] && echo ON || echo OFF)" \
      cu130 "CUDA 13.0 (driver >= 580)" "$([ "$CUDA" = cu130 ] && echo ON || echo OFF)" \
      cu128 "CUDA 12.8 (driver >= 570, RTX 50 needs this or newer)" "$([ "$CUDA" = cu128 ] && echo ON || echo OFF)" \
      cu126 "CUDA 12.6 (driver >= 560)" "$([ "$CUDA" = cu126 ] && echo ON || echo OFF)" \
      cpu "CPU only" "$([ "$CUDA" = cpu ] && echo ON || echo OFF)") && CUDA=$v
  save_conf
  if svc_unit; then step_service; wt --yesno "Settings saved. Restart vLLM now to apply them?" 8 60 && server_start_restart; fi
}
server_start_restart() { server_stop; server_start; }

pause() { read -rp "Press Enter to return to the menu..." _ </dev/tty; }

menu() {
  load_conf
  local c=1
  while true; do
    detect_gpu
    local up="down"; healthy && up="UP on :$PORT"
    c=$(wt --title "Docling RTX setup" --default-item "$c" --menu "GPU: ${GPU_NAME:-none found}${GPU_MB:+ ($(((GPU_MB + 512) / 1024)) GB)}   vLLM: $up\n\nChoose:" 21 76 11 \
      1 "Install everything (recommended)" \
      2 "Choose what to install..." \
      3 "Start vLLM server" \
      4 "Stop vLLM server" \
      5 "Status" \
      6 "Run a test conversion" \
      7 "Convert a folder of documents..." \
      8 "Settings (port, GPU memory, batch, CUDA)" \
      9 "Server log" \
      10 "How to process documents" \
      11 "Uninstall") || { clear; exit 0; }
    clear
    case $c in
      1) wt --yesno "Install:\n\n - system packages (apt)\n - uv + Python 3.12\n - Docling with CUDA PyTorch\n - vLLM + granite-docling-258M\n - $SERVICE service, started now\n\nNeeds ~20 GB of disk and sudo. Continue?" 17 64 && { install_steps $ALL_STEPS; pause; } ;;
      2) local picks
         picks=$(wt --title "Components" --checklist "Space to toggle, Enter to install:" 18 76 8 \
           driver "NVIDIA driver (Ubuntu, needs a reboot)" OFF \
           apt "System packages" ON \
           uv "uv (Python manager)" ON \
           docling "Docling + CUDA PyTorch" ON \
           vllm "vLLM server engine" ON \
           models "Download granite-docling-258M" ON \
           service "Run vLLM as a service (systemd)" ON) && { install_steps preflight $(echo "$picks" | tr -d '"'); pause; } ;;
      3) server_start || true; pause ;;
      4) server_stop; pause ;;
      5) status; pause ;;
      6) test_run || true; pause ;;
      7) local in outd eng
         in=$(wt --inputbox "Folder (or file / URL) to convert:" 9 70 "$HOME/documents") &&
         outd=$(wt --inputbox "Output folder:" 9 70 "$HOME/docling-out") &&
         eng=$(wt --radiolist "Engine:" 13 76 3 \
           standard "layout + tables + OCR models (digital PDFs, Office)" ON \
           vlm "granite-docling via vLLM (scans, formulas, code)" OFF \
           local "granite-docling in-process (no server, slower)" OFF) &&
         { "$HOME/.local/bin/docling-convert" "$in" -o "$outd" --engine "$eng" --batch "$(auto_batch)" \
             --server "http://127.0.0.1:$PORT/v1/chat/completions" || true; pause; } ;;
      8) menu_settings ;;
      9) server_logs | tail -60; pause ;;
      10) local g=$DOCLING_HOME/GUIDE.md; [ -f "$g" ] || g=$SRC_DIR/GUIDE.md
          if [ -f "$g" ]; then wt --title "GUIDE.md" --scrolltext --textbox "$g" 30 100; else wt --msgbox "GUIDE.md not found." 8 40; fi ;;
      11) wt --yesno "Remove $DOCLING_HOME, the launchers and the service?\n(Model cache in ~/.cache/huggingface is kept.)" 10 64 && { uninstall; pause; } ;;
    esac
  done
}

# --- main ------------------------------------------------------------------------------------------------
args=()
for a in "$@"; do
  case $a in
    --dry-run) DRY=1 ;;
    --auto|-y) args+=(install) ;;
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    --version) echo "$VERSION"; exit 0 ;;
    *) args+=("$a") ;;
  esac
done
load_conf
case ${args[0]:-} in
  install)
    if [ "$DRY" = 1 ]; then install_steps $ALL_STEPS
    else mkdir -p "$DOCLING_HOME"; install_steps $ALL_STEPS 2>&1 | tee -a "$LOG"; exit "${PIPESTATUS[0]}"; fi ;;
  start)   server_start ;;
  stop)    server_stop ;;
  restart) server_stop; server_start ;;
  status)  status ;;
  test)    test_run ;;
  logs)    server_logs ;;
  uninstall) uninstall ;;
  "")
    if [ "$DRY" = 1 ]; then install_steps $ALL_STEPS
    elif interactive; then menu
    else die "No terminal for the menu: use '$0 install' (or --help)."; fi ;;
  *) die "Unknown command: ${args[0]} (see --help)" ;;
esac
