#!/usr/bin/env bash
# docling-setup.sh - set up GPU document processing on a fresh Linux machine with an NVIDIA RTX card:
# Docling (PDF/Office/HTML/image -> Markdown/JSON/HTML) on CUDA PyTorch, plus IBM's granite-docling-258M
# vision-language model served by vLLM (the fast path from Docling's RTX guide).
#
#   One-line install on a new machine:
#     curl -fsSL https://raw.githubusercontent.com/codyvolckmar/docling-rtx/main/install.sh | bash
#   After install the script is also the command  docling-rtx  (same subcommands):
#   ./docling-setup.sh               whiptail menu
#   ./docling-setup.sh install       full install, no questions (also: --auto, -y)
#   ./docling-setup.sh howto         where the web page is and how to use it
#   ./docling-setup.sh web [url|start|stop|logs|public|private]   drag & drop page (public = direct link + key)
#   ./docling-setup.sh web port 8080 [41234]   the page's port [and the public port your provider maps it to]
#   ./docling-setup.sh web tunnel on|off      public https link through a Cloudflare quick tunnel (default: on)
#   ./docling-setup.sh start|stop|restart|status|test|logs|uninstall      (the vLLM server)
#   ./docling-setup.sh watch start|stop|logs   the drop folder: files put in INBOX come out as JSON in OUTBOX
#   ./docling-setup.sh --dry-run     show what install would do, change nothing
#
# What it installs (nothing outside these, apart from apt packages and the optional systemd unit):
#   ~/.local/bin/uv                    Python/package manager (brings its own Python 3.12)
#   $DOCLING_HOME/venv                 docling + CUDA PyTorch (cu126/cu128/cu130, picked from your driver)
#   $DOCLING_HOME/vllm                 vLLM in its own venv, only once the vlm engine is used (docling-rtx start)
#   $DOCLING_HOME/convert.py           batch converter; launcher: ~/.local/bin/docling-convert
#   ~/.cache/huggingface               model weights (granite-docling-258M, Docling's layout/table models)
#   docling-vllm.service               systemd unit (or a background process where systemd isn't running)
#   docling-watch.service              the drop-folder watcher, same (folders: ~/docling/inbox -> ~/docling/outbox)
# You only need the NVIDIA driver: the PyTorch and vLLM wheels ship their own CUDA runtime and cuDNN, so the
# CUDA Toolkit and cuDNN steps of the Docling RTX guide are not needed.
#
# Settings (environment or $DOCLING_HOME/config.env; the menu edits the file):
#   DOCLING_HOME=~/.docling-rtx  PORT=8000  GPU_UTIL=auto  BATCH=auto  CUDA=auto (cu126|cu128|cu130|cpu)
#   WEB_PORT=8080  WEB_PUBLIC=1 (direct link + key)  WEB_PUBLIC_PORT=  (set at install: curl ... | WEB_PORT=... bash)
#   INBOX=~/docling/inbox  OUTBOX=~/docling/outbox  WATCH_ENGINE=standard|vlm|local  WATCH_FORMATS="json"
#   DOCLING_GPU="RTX 4090:24564:575.64"   pretend this GPU (name:MB:driver), for --dry-run previews
set -euo pipefail

VERSION=1.4.3
MODEL=ibm-granite/granite-docling-258M
MODEL_REV=untied   # the model card's vLLM example loads this branch (untied embedding weights)
DOCLING_HOME=${DOCLING_HOME:-$HOME/.docling-rtx}
CONF=$DOCLING_HOME/config.env
LOG=$DOCLING_HOME/install.log
SERVICE=docling-vllm
SELF=$(readlink -f "${BASH_SOURCE[0]:-$0}" 2>/dev/null || echo "$0")   # resolves the ~/.local/bin/docling-rtx link
SRC_DIR=$(cd "$(dirname "$SELF")" 2>/dev/null && pwd || pwd)
REPO_RAW=${REPO_RAW:-https://raw.githubusercontent.com/codyvolckmar/docling-rtx/main}   # where to fetch convert.py when this script runs without the repo next to it
TEST_URL=https://arxiv.org/pdf/2501.17887   # the Docling paper, as in the model card's examples
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
# vLLM, the converter and the web page talk over 127.0.0.1: never through a pod's http_proxy
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
DRY=0
ME=$0; [ "$(basename "$0")" = docling-rtx ] && ME=docling-rtx   # how to name this command in messages

# --- helpers ---------------------------------------------------------------------------------------------
say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then echo "   [dry-run] $*"; else "$@"; fi; }
have() { command -v "$1" >/dev/null 2>&1; }
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
interactive() { [ -t 1 ] && [ -r /dev/tty ]; }
need_whiptail() {   # minimal images (Docker, rented pods) don't ship it, and the menu is what installs things
  case ${TERM:-} in ""|dumb) export TERM=xterm ;; esac
  have whiptail && return
  have apt-get || die "The menu needs whiptail: install it (package 'newt'/'whiptail'), or run '$0 install'."
  say "Installing whiptail for the menu"
  $SUDO apt-get update -q >/dev/null && $SUDO apt-get install -y -q whiptail >/dev/null \
    || die "Couldn't install whiptail. Run '$0 install' (no menu) instead."
}
wt() { whiptail --backtitle "Docling RTX setup $VERSION" "$@" 3>&1 1>&2 2>&3 </dev/tty; }

load_conf() {   # environment beats config.env beats defaults
  local p=${PORT:-} g=${GPU_UTIL:-} b=${BATCH:-} c=${CUDA:-} i=${INBOX:-} o=${OUTBOX:-} e=${WATCH_ENGINE:-} f=${WATCH_FORMATS:-} wh=${WEB_HOST:-} wp=${WEB_PORT:-} wpp=${WEB_PUBLIC_PORT:-} wt=${WEB_TUNNEL:-}
  [ "${WEB_PUBLIC:-}" = 1 ] && wh=0.0.0.0   # WEB_PUBLIC=1: direct link with a key (e.g. at install)
  if [ -f "$CONF" ]; then
    . "$CONF"
    # Config files from 1.1.0 (no CONF_VERSION) stored the computed GPU_UTIL/BATCH; go back to auto sizing.
    [ -n "${CONF_VERSION:-}" ] || GPU_UTIL="" BATCH=""
  fi
  # GPU_UTIL auto (see gpu_util): 0.5 instead of the RTX guide's 0.9, because the 258M model needs little and the
  # drop-folder watcher's standard pipeline shares the card; 0.7 on cards under 10 GB.
  PORT=${p:-${PORT:-8000}} GPU_UTIL=${g:-${GPU_UTIL:-auto}} BATCH=${b:-${BATCH:-auto}} CUDA=${c:-${CUDA:-auto}}
  INBOX=${i:-${INBOX:-$HOME/docling/inbox}} OUTBOX=${o:-${OUTBOX:-$HOME/docling/outbox}}
  WATCH_ENGINE=${e:-${WATCH_ENGINE:-standard}} WATCH_FORMATS=${f:-${WATCH_FORMATS:-json}}
  WEB_HOST=${wh:-${WEB_HOST:-127.0.0.1}} WEB_PORT=${wp:-${WEB_PORT:-8080}} WEB_PUBLIC_PORT=${wpp:-${WEB_PUBLIC_PORT:-}}
  WEB_TUNNEL=${wt:-${WEB_TUNNEL:-1}}   # 1 = public https link through a Cloudflare quick tunnel (key required)
}
save_conf() {
  mkdir -p "$DOCLING_HOME"
  printf 'CONF_VERSION=2\nPORT=%s\nGPU_UTIL=%s\nBATCH=%s\nCUDA=%s\nINBOX="%s"\nOUTBOX="%s"\nWATCH_ENGINE=%s\nWATCH_FORMATS="%s"\nWEB_HOST=%s\nWEB_PORT=%s\nWEB_PUBLIC_PORT=%s\nWEB_TUNNEL=%s\n' \
    "$PORT" "$GPU_UTIL" "$BATCH" "$CUDA" "$INBOX" "$OUTBOX" "$WATCH_ENGINE" "$WATCH_FORMATS" "$WEB_HOST" "$WEB_PORT" "$WEB_PUBLIC_PORT" "$WEB_TUNNEL" > "$CONF"
}

# --- detection -------------------------------------------------------------------------------------------
GPU_NAME="" GPU_MB="" DRIVER="" GPU_CC=""
detect_gpu() {
  if [ -n "${DOCLING_GPU:-}" ]; then   # preview a card you don't have: DOCLING_GPU="RTX 4090:24564:575.64"
    IFS=: read -r GPU_NAME GPU_MB DRIVER GPU_CC <<< "$DOCLING_GPU"; return   # optional 4th field: compute capability
  fi
  if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  fi
}
cuda_flavor() {   # PyTorch wheel for the driver: cu130 >= 580, cu128 >= 570, cu126 >= 525 (CUDA 12 minor-version
                  # compatibility: rented pods often run 535/550, and a pod user can't upgrade the host driver)
  [ "$CUDA" != auto ] && { echo "$CUDA"; return; }
  [ -z "$DRIVER" ] && { echo cpu; return; }
  local major=${DRIVER%%.*}
  if   [ "$major" -ge 580 ]; then echo cu130
  elif [ "$major" -ge 570 ]; then echo cu128
  elif [ "$major" -ge 525 ]; then echo cu126
  else echo old; fi
}
auto_batch() {    # Docling's RTX guide: 32 GB -> 64-128, 24 GB -> 32-64, 12 GB -> 16-32
  [ "$BATCH" != auto ] && { echo "$BATCH"; return; }
  if   [ "${GPU_MB:-0}" -ge 30000 ]; then echo 64
  elif [ "${GPU_MB:-0}" -ge 20000 ]; then echo 32
  elif [ "${GPU_MB:-0}" -ge 10000 ]; then echo 16
  else echo 8; fi
}
gpu_util() {      # share of VRAM vLLM takes
  [ "$GPU_UTIL" != auto ] && { echo "$GPU_UTIL"; return; }
  # vlm drop folder: vLLM does the work, the standard models aren't loaded -> most of the card. Otherwise it shares
  # the card with the standard pipeline: half (small cards need 0.7 for vLLM's startup profiling run).
  if [ "$WATCH_ENGINE" = vlm ]; then echo 0.85
  elif [ "${GPU_MB:-0}" -ge 10000 ]; then echo 0.5; else echo 0.7; fi
}
uses_vlm() { [ "$WATCH_ENGINE" = vlm ] || [ "${WITH_VLM:-}" = 1 ]; }
has_systemd() { [ -z "${DOCLING_NO_SYSTEMD:-}" ] && [ -d /run/systemd/system ] && have systemctl; }

# --- install steps ---------------------------------------------------------------------------------------
step_preflight() {
  say "Checking the machine"
  [ "$(uname -s)" = Linux ] || die "Linux only (on Windows, run this inside WSL 2)."
  have apt-get || die "Needs an apt-based distro (Ubuntu 22.04/24.04, Debian 12)."
  [ -n "$SUDO" ] && ! have sudo && die "Run as root or install sudo."
  detect_gpu
  if [ -n "$GPU_NAME" ]; then
    ok "GPU: $GPU_NAME, $(((GPU_MB + 512) / 1024)) GB, driver $DRIVER"
    [ "$(cuda_flavor)" = old ] && die "Driver $DRIVER is too old for CUDA PyTorch (needs 525+). Update the driver, or install CPU-only: CUDA=cpu $ME install"
  else
    warn "No working NVIDIA driver (nvidia-smi failed). Install the driver first (menu: NVIDIA driver), or"
    warn "continue CPU-only: Docling works, vLLM is skipped."
  fi
  local free_gb; free_gb=$({ df -Pk "$HOME" 2>/dev/null || df -Pk /; } | awk 'NR==2{print int($4/1048576)}')
  local need=8; uses_vlm && need=20   # Docling + CUDA PyTorch + models ~8 GB; vLLM adds ~12 GB
  [ "$free_gb" -ge "$need" ] || warn "Only ${free_gb} GB free in $HOME; this install needs ~${need} GB."
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
  local p missing=""
  for p in ca-certificates curl git whiptail build-essential libgl1 libglib2.0-0 screen; do
    dpkg -s "$p" >/dev/null 2>&1 || missing="$missing $p"
  done
  if [ -z "$missing" ]; then ok "System packages present"; return 0; fi
  say "System packages:$missing"
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
  [ -x "$DOCLING_HOME/venv/bin/python" ] || run uv venv -q --managed-python --python 3.12 "$DOCLING_HOME/venv"
  export VIRTUAL_ENV=$DOCLING_HOME/venv
  local have_build   # torch already there but a different build (e.g. +cpu after a driver install): swap it
  have_build=$("$VIRTUAL_ENV/bin/python" -c 'import torch;print(torch.__version__.partition("+")[2])' 2>/dev/null || true)
  if [ -n "$have_build" ] && [ "$have_build" != "$flavor" ]; then
    say "Replacing torch +$have_build with +$flavor"; run uv pip uninstall -q torch torchvision
  fi
  # One command with --torch-backend: installing docling separately would pull PyPI's default torch build over this one.
  run uv pip install -q --upgrade torch torchvision docling --torch-backend="$flavor"
  unset VIRTUAL_ENV
  install_launchers
  [ "$DRY" = 1 ] && return
  "$DOCLING_HOME/venv/bin/python" - <<'EOF'
import torch, importlib.metadata as m
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f" ok docling {m.version('docling')}, torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {gpu}")
EOF
}

install_launchers() {   # app files into $DOCLING_HOME, so the commands keep working without the cloned repo
  local f
  run mkdir -p "$DOCLING_HOME" "$HOME/.local/bin"
  for f in convert.py web.py logo.png GUIDE.md docling-setup.sh; do
    if [ -f "$SRC_DIR/$f" ]; then [ "$SRC_DIR/$f" -ef "$DOCLING_HOME/$f" ] || run cp "$SRC_DIR/$f" "$DOCLING_HOME/$f"
    elif [ -n "$REPO_RAW" ]; then run curl -fsSL "$REPO_RAW/$f" -o "$DOCLING_HOME/$f"
    else warn "$f not found next to this script (set REPO_RAW=... to download it)."; fi
  done
  [ "$DRY" = 1 ] && return
  chmod +x "$DOCLING_HOME/docling-setup.sh"
  printf '#!/bin/sh\nexec "%s" "%s" "$@"\n' "$DOCLING_HOME/venv/bin/python" "$DOCLING_HOME/convert.py" > "$HOME/.local/bin/docling-convert"
  chmod +x "$HOME/.local/bin/docling-convert"
  ln -sf "$DOCLING_HOME/venv/bin/docling" "$HOME/.local/bin/docling"
  ln -sf "$DOCLING_HOME/docling-setup.sh" "$HOME/.local/bin/docling-rtx"
  # ~/.local/bin isn't on PATH in many containers (often not for root): also link into /usr/local/bin when we
  # can, otherwise put ~/.local/bin on PATH in ~/.bashrc.
  local c
  if [ -w /usr/local/bin ]; then
    for c in docling-rtx docling-convert docling; do ln -sf "$HOME/.local/bin/$c" "/usr/local/bin/$c"; done
    ok "Commands: docling-rtx (menu), docling-convert, docling (in /usr/local/bin)"
  else
    grep -qs 'HOME/.local/bin' "$HOME/.bashrc" ||
      printf '\nexport PATH="$HOME/.local/bin:$PATH"   # docling-rtx commands\n' >> "$HOME/.bashrc"
    ok "Commands: docling-rtx (menu), docling-convert, docling (in ~/.local/bin, on PATH in new shells)"
  fi
}

step_vllm() {
  detect_gpu
  [ -z "$DRIVER" ] && { warn "No NVIDIA driver: skipping vLLM."; return; }
  say "vLLM in $DOCLING_HOME/vllm (its own venv: it pins its own torch)"
  # uv's own Python, never the system one: vLLM compiles Triton/torch.compile kernels at startup and needs Python.h,
  # which Ubuntu's python3.12 lacks without python3.12-dev ("Engine core initialization failed").
  if [ -f "$DOCLING_HOME/vllm/pyvenv.cfg" ] && ! grep -q "/uv/python/" "$DOCLING_HOME/vllm/pyvenv.cfg"; then
    say "Rebuilding the vLLM venv on uv's Python (the system Python has no C headers)"; run rm -rf "$DOCLING_HOME/vllm"
  fi
  [ -x "$DOCLING_HOME/vllm/bin/python" ] || run uv venv -q --managed-python --python 3.12 "$DOCLING_HOME/vllm"
  VIRTUAL_ENV=$DOCLING_HOME/vllm run uv pip install -q --upgrade vllm --torch-backend=auto
  [ "$DRY" = 1 ] || ok "vLLM $("$DOCLING_HOME/vllm/bin/python" -c 'import importlib.metadata as m;print(m.version("vllm"))')"
}

step_models() {
  say "Downloading granite-docling-258M for vLLM"
  run "$DOCLING_HOME/venv/bin/hf" download "$MODEL" --revision "$MODEL_REV" --quiet >/dev/null
}

vllm_cmd() {
  # Sized to the card. The RTX guide's 512 sequences x 8192 batched tokens is for 24-32 GB; on a 6 GB card the
  # startup profiling run alone leaves no room for the KV cache ("_initialize_kv_caches" failure). The model needs
  # ~23 KB of cache per token (30 layers x 3 KV heads x 64 dims, 16-bit), ~190 MB for a full 8K page.
  detect_gpu
  local seqs=512 tokens=8192 dtype=""
  if   [ "${GPU_MB:-0}" -lt 10000 ]; then seqs=32 tokens=4096
  elif [ "${GPU_MB:-0}" -lt 20000 ]; then seqs=128; fi
  # No bfloat16 before Ampere (GTX 16 / RTX 20 are compute 7.5): run in float16.
  if [ -n "$GPU_CC" ] && [ "${GPU_CC%%.*}" -lt 8 ]; then dtype=" --dtype half"; fi
  echo "$DOCLING_HOME/vllm/bin/vllm serve $MODEL --revision $MODEL_REV --served-model-name $MODEL" \
       "--host 127.0.0.1 --port $PORT --max-model-len 8192 --max-num-seqs $seqs --max-num-batched-tokens $tokens" \
       "--enable-chunked-prefill --gpu-memory-utilization $(gpu_util)$dtype"
}

write_unit() {   # $1 name, $2 description, $3 ExecStart, $4 extra [Unit] lines, $5 extra [Service] line
  local unit; unit="[Unit]
Description=$2
After=network-online.target
$4
[Service]
User=$(id -un)
Environment=HOME=$HOME
Environment=HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
Environment=HF_HUB_DISABLE_TELEMETRY=1
Environment=VLLM_NO_USAGE_STATS=1
Environment=PYTHONUNBUFFERED=1
Environment=NO_PROXY=127.0.0.1,localhost
Environment=no_proxy=127.0.0.1,localhost
${5:+$5
}ExecStart=$3
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target"
  if [ "$DRY" = 1 ]; then echo "   [dry-run] write /etc/systemd/system/$1.service:"; echo "             ExecStart=$3"
  else echo "$unit" | $SUDO tee "/etc/systemd/system/$1.service" >/dev/null; fi
  run $SUDO systemctl daemon-reload
  run $SUDO systemctl enable -q "$1"
}

step_service() {
  [ -x "$DOCLING_HOME/vllm/bin/vllm" ] || [ "$DRY" = 1 ] || { warn "vLLM not installed: no service."; return; }
  if has_systemd; then
    say "systemd service $SERVICE (starts on boot)"
    write_unit "$SERVICE" "granite-docling-258M on vLLM (Docling VLM server)" "$(vllm_cmd)" ""
  else
    ok "No systemd here: vLLM runs as a background process (started again at login)."
  fi
}

step_warmup() {   # convert a one-line PDF: Docling downloads and loads its models now, not on your first document
  [ "$DRY" = 1 ] && { echo "   [dry-run] warm-up conversion (downloads Docling's layout/table/OCR models)"; return 0; }
  say "Warming up: downloading Docling's layout, table and OCR models (first install only, ~1 min)"
  local d; d=$(mktemp -d)
  "$DOCLING_HOME/venv/bin/python" - "$d/warmup.pdf" <<'EOF'
import sys
objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
text = b"BT /F1 24 Tf 72 120 Td (Docling RTX warm-up) Tj ET"
objs[3] = b"<< /Length %d >>\nstream\n" % len(text) + text + b"\nendstream"
out, offs = bytearray(b"%PDF-1.4\n"), []
for i, o in enumerate(objs, 1):
    offs.append(len(out)); out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
x = len(out)
out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
open(sys.argv[1], "wb").write(out)
EOF
  if "$DOCLING_HOME/venv/bin/python" "$DOCLING_HOME/convert.py" "$d/warmup.pdf" -o "$d/out" --to md >"$d/log" 2>&1 &&
     grep -q "warm-up" "$d/out/warmup.md" 2>/dev/null; then ok "Models ready: the first document starts right away"
  else warn "Warm-up conversion failed (the first document will download the models). Log:"; tail -5 "$d/log"; fi
  rm -rf "$d"
}

# --- background services: the drop-folder watcher and the web page ----------------------------------------
# systemd units where systemd runs; otherwise background processes with a pid file, restarted at login
# (rented GPU pods are usually containers without systemd).
svc_unit_file() { echo "/etc/systemd/system/docling-$1.service"; }
svc_unit() { has_systemd && [ -f "$(svc_unit_file "$1")" ]; }
svc_running() {   # $1 = watch | web: the program itself runs (not just its restart loop)
  if svc_unit "$1"; then systemctl is-active -q "docling-$1"
  else [ -f "$DOCLING_HOME/$1.pid" ] && kill -0 "$(cat "$DOCLING_HOME/$1.pid")" 2>/dev/null &&
       [ -f "$DOCLING_HOME/$1.child" ] && kill -0 "$(cat "$DOCLING_HOME/$1.child")" 2>/dev/null; fi
}
web_answers() {   # our web page (not another program, e.g. Jupyter on 8080) answers on its port
  curl -s -m 3 -o /dev/null -D - "http://127.0.0.1:$WEB_PORT/logo.png" 2>/dev/null | grep -qi "^server: docling-rtx"
}
svc_start() {     # $1 name, $2 command
  [ -x "$DOCLING_HOME/venv/bin/python" ] || { warn "Docling isn't installed (menu: Install)."; return 1; }
  rm -f "$DOCLING_HOME/$1.off" "$DOCLING_HOME/$1.crashes"
  if svc_unit "$1"; then $SUDO systemctl reset-failed "docling-$1" 2>/dev/null; $SUDO systemctl restart "docling-$1"
  else
    svc_stop "$1"; rm -f "$DOCLING_HOME/$1.off"
    # A restart loop, like systemd's Restart=always: brings it back after a crash or a forced cancel (exit 75).
    # It records the program's pid (for status) and each crash (so a program that can't start isn't shown as UP).
    # setsid makes it a process group: stopping kills the loop and the program together.
    setsid nohup bash -c 'n=$1 d=$2; shift 2; while :; do "$@" & echo $! > "$d/$n.child"; wait $!; rc=$?
      [ $rc = 0 ] && break; echo "$(date +%T) exit $rc" >> "$d/$n.crashes"; sleep 3; done' \
      "docling-$1" "$1" "$DOCLING_HOME" $2 >> "$DOCLING_HOME/$1.log" 2>&1 < /dev/null &
    echo $! > "$DOCLING_HOME/$1.pid"
  fi
  sleep 4
  local bad=""
  if svc_unit "$1"; then
    [ "$(systemctl show -p NRestarts --value "docling-$1" 2>/dev/null || echo 0)" -gt 0 ] && bad=1
  else [ -s "$DOCLING_HOME/$1.crashes" ] && bad=1; fi
  [ "$1" = web ] && ! web_answers && bad=1
  if [ -n "$bad" ] || ! svc_running "$1"; then
    warn "docling-$1 isn't running properly. Its log:"; svc_logs "$1" | tail -12
    [ "$1" = web ] && warn "Is port $WEB_PORT taken by another program? Choose another: $ME web port <port>"
    return 1
  fi
}
svc_stop() {
  [ -d "$DOCLING_HOME" ] && touch "$DOCLING_HOME/$1.off"   # autostart leaves it off until it's started again
  if svc_unit "$1"; then $SUDO systemctl stop "docling-$1"
  elif [ -f "$DOCLING_HOME/$1.pid" ]; then
    local pid; pid=$(cat "$DOCLING_HOME/$1.pid")
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true   # the whole group (loop + program)
    rm -f "$DOCLING_HOME/$1.pid"
  fi
  return 0
}
svc_logs() {
  if svc_unit "$1"; then $SUDO journalctl -u "docling-$1" -n 200 --no-pager -o cat
  else tail -n 200 "$DOCLING_HOME/$1.log" 2>/dev/null; fi
}
svc_setup() {     # $1 name, $2 description, $3 command, $4 extra [Unit] lines, $5 extra [Service] line
  if has_systemd; then write_unit "docling-$1" "$2" "$3" "$4" "$5"; fi
  if [ "$DRY" = 1 ]; then return 0; fi
  svc_start "$1" "$3"
}
svc_remove() {
  svc_stop "$1"
  if svc_unit "$1"; then $SUDO systemctl disable -q "docling-$1"; $SUDO rm -f "$(svc_unit_file "$1")"; $SUDO systemctl daemon-reload; fi
  return 0
}

# Drop folder: INBOX -> OUTBOX
watch_cmd() {
  local f to=""
  for f in $WATCH_FORMATS; do to="$to --to $f"; done
  echo "$DOCLING_HOME/venv/bin/python $DOCLING_HOME/convert.py --watch $INBOX -o $OUTBOX$to" \
       "--engine $WATCH_ENGINE --batch $(auto_batch) --server http://127.0.0.1:$PORT/v1/chat/completions" \
       "--status-file $DOCLING_HOME/watch-status.json"
}
step_watch() {
  say "Drop folder: $INBOX -> $OUTBOX ($WATCH_FORMATS, engine $WATCH_ENGINE)"
  run mkdir -p "$INBOX" "$OUTBOX"
  local after=""; [ "$WATCH_ENGINE" = vlm ] && after="Wants=$SERVICE.service
After=$SERVICE.service"
  svc_setup watch "Docling drop folder: $INBOX -> $OUTBOX" "$(watch_cmd)" "$after" "" || return 0
  [ "$DRY" = 1 ] || ok "Watching $INBOX -> $OUTBOX"
}
watch_running() { svc_running watch; }
watch_start() { mkdir -p "$INBOX" "$OUTBOX"; svc_start watch "$(watch_cmd)" && ok "Watching $INBOX -> $OUTBOX. Drop files in; JSON appears in the outbox."; }
watch_stop() { svc_stop watch; ok "Drop-folder watcher stopped"; }
watch_logs() { svc_logs watch; }
watch_apply() {   # settings changed: rewrite the unit and restart, if the watcher is set up
  if svc_unit watch; then step_watch
  elif watch_running; then watch_start; fi
}

# Web page: drag & drop upload, live progress, downloads (web.py)
web_cmd() {
  echo "$DOCLING_HOME/venv/bin/python $DOCLING_HOME/web.py --inbox $INBOX --outbox $OUTBOX" \
       "--status-file $DOCLING_HOME/watch-status.json --host $WEB_HOST --port $WEB_PORT --vllm-port $PORT --config $CONF --batch $(auto_batch)" \
       "--tunnel-log $DOCLING_HOME/tunnel-cf.log"
}
needs_token() { [ "$WEB_HOST" != 127.0.0.1 ] || [ "$WEB_TUNNEL" = 1 ]; }   # anyone but this machine can reach it
web_token() {     # the key in the link: only when the page is reachable from outside (tunnel or public)
  needs_token || return 0
  [ -s "$DOCLING_HOME/web.token" ] || ( umask 077; head -c 18 /dev/urandom | base64 | tr '+/' '-_' > "$DOCLING_HOME/web.token" )
  cat "$DOCLING_HOME/web.token"
}
web_start() { DOCLING_WEB_TOKEN=$(web_token) svc_start web "$(web_cmd)"; }
step_web() {
  say "Web page on $WEB_HOST:$WEB_PORT (drag & drop, progress, downloads)"
  local env=""
  if needs_token && [ "$DRY" = 0 ]; then env="Environment=DOCLING_WEB_TOKEN=$(web_token)"; fi
  if has_systemd; then write_unit docling-web "Docling RTX web page" "$(web_cmd)" "" "$env"; fi
  [ "$DRY" = 1 ] || web_start || true
  if [ "$WEB_TUNNEL" = 1 ]; then step_tunnel; fi
  return 0
}

# Cloudflare quick tunnel: a public https://<random>.trycloudflare.com address for the page, no account and no open
# port needed (works on pods that expose nothing). The address changes when the tunnel restarts; the page and
# 'docling-rtx howto' show the current one. Uploads through it are limited to 100 MB per file (Cloudflare's limit).
CF=$DOCLING_HOME/bin/cloudflared
cloudflared_bin() { if [ -x "$CF" ]; then echo "$CF"; elif have cloudflared; then command -v cloudflared; fi; }
step_cloudflared() {
  [ -n "$(cloudflared_bin)" ] && return 0
  local arch; case $(uname -m) in x86_64) arch=amd64 ;; aarch64|arm64) arch=arm64 ;; *) arch=$(uname -m) ;; esac
  say "Installing cloudflared (Cloudflare's tunnel client) into $DOCLING_HOME/bin"
  run mkdir -p "$DOCLING_HOME/bin"
  run curl -fsSL -o "$CF.part" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch" &&
    run chmod +x "$CF.part" && run mv "$CF.part" "$CF" || { warn "Couldn't download cloudflared: no public link."; return 1; }
}
tunnel_cmd() {
  echo "$(cloudflared_bin) tunnel --no-autoupdate --logfile $DOCLING_HOME/tunnel-cf.log --url http://127.0.0.1:$WEB_PORT"
}
tunnel_url() {    # the current https://....trycloudflare.com address (the last one cloudflared announced)
  grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' "$DOCLING_HOME/tunnel-cf.log" 2>/dev/null | tail -1
}
tunnel_start() {
  [ -n "$(cloudflared_bin)" ] || step_cloudflared || return 1
  rm -f "$DOCLING_HOME/tunnel-cf.log"
  svc_start tunnel "$(tunnel_cmd)" || return 1
  local _
  for _ in $(seq 1 40); do [ -n "$(tunnel_url)" ] && break; sleep 0.5; done
  if [ -n "$(tunnel_url)" ]; then ok "Public link: $(tunnel_url)/?token=$(web_token)"
  else warn "The tunnel started but has no address yet: $ME web url (in a few seconds), log: $ME web tunnel logs"; fi
}
step_tunnel() {
  say "Public link through a Cloudflare tunnel (anyone with the link and its key can use the page)"
  [ "$DRY" = 1 ] && { echo "   [dry-run] cloudflared tunnel --url http://127.0.0.1:$WEB_PORT"; return 0; }
  step_cloudflared || return 0
  if has_systemd; then write_unit docling-tunnel "Cloudflare quick tunnel to the Docling RTX web page" "$(tunnel_cmd)" "" ""; fi
  tunnel_start || true
}
valid_port() { [[ ${1:-} =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]; }
public_port() {   # the port the outside world uses: pod providers often map the pod's port to another one
  local vast="VAST_TCP_PORT_$WEB_PORT" runpod="RUNPOD_TCP_PORT_$WEB_PORT"   # how Vast.ai / RunPod advertise it
  echo "${WEB_PUBLIC_PORT:-${!vast:-${!runpod:-$WEB_PORT}}}"
}
public_ip() {
  echo "${PUBLIC_IPADDR:-${RUNPOD_PUBLIC_IP:-$(curl -fsS -m 4 https://api.ipify.org 2>/dev/null || hostname -I | cut -d' ' -f1)}}"
}
web_url() {
  local runpod="RUNPOD_TCP_PORT_$WEB_PORT"
  if [ "$WEB_TUNNEL" = 1 ] && [ -n "$(tunnel_url)" ] && svc_running tunnel; then echo "$(tunnel_url)/?token=$(web_token)"
  elif [ "$WEB_HOST" = 127.0.0.1 ]; then   # key needed here too when the tunnel is on (same page)
    if needs_token; then echo "http://localhost:$WEB_PORT/?token=$(web_token)"; else echo "http://localhost:$WEB_PORT"; fi
  elif [ -n "${RUNPOD_POD_ID:-}" ] && [ -z "$WEB_PUBLIC_PORT" ] && [ -z "${!runpod:-}" ]; then
    # RunPod's HTTPS proxy (add the port under "Expose HTTP ports" in the pod settings)
    echo "https://$RUNPOD_POD_ID-$WEB_PORT.proxy.runpod.net/?token=$(web_token)"
  else echo "http://$(public_ip):$(public_port)/?token=$(web_token)"; fi
}
web_apply() {     # port or access changed: rewrite the services and restart them (if Docling is installed)
  save_conf
  if [ -x "$DOCLING_HOME/venv/bin/python" ]; then svc_remove tunnel; svc_remove web; step_web; fi
  howto
}
autostart() {     # no systemd: bring back what was running, at login (hooked into ~/.bashrc by step_autostart)
  has_systemd && return 0
  exec 9> "$DOCLING_HOME/autostart.lock"   # shells opened together: only one does it
  if have flock; then flock -n 9 || return 0; fi
  if [ -x "$DOCLING_HOME/vllm/bin/vllm" ] && [ ! -f "$DOCLING_HOME/vllm.off" ] && ! healthy; then server_start >/dev/null 2>&1 || true; fi
  [ -f "$DOCLING_HOME/watch.off" ] || svc_running watch || watch_start >/dev/null 2>&1 || true
  [ -f "$DOCLING_HOME/web.off" ] || svc_running web || web_start >/dev/null 2>&1 || true
  if [ "$WEB_TUNNEL" = 1 ] && [ ! -f "$DOCLING_HOME/tunnel.off" ] && ! svc_running tunnel; then tunnel_start >/dev/null 2>&1 || true; fi
  return 0
}
step_autostart() {
  has_systemd && return 0
  grep -q "docling-rtx autostart" "$HOME/.bashrc" 2>/dev/null && return 0
  say "No systemd: the services start again when you log in (hook in ~/.bashrc)"
  [ "$DRY" = 1 ] && return 0
  printf '\n# docling-rtx autostart: restarts the drop folder, web page and vLLM after a reboot (no systemd here)\n[ -x "$HOME/.local/bin/docling-rtx" ] && ( "$HOME/.local/bin/docling-rtx" autostart >/dev/null 2>&1 & )\n' >> "$HOME/.bashrc"
}

howto() {
  local tunnel=""
  if [ "$WEB_TUNNEL" = 1 ] && [ -n "$(tunnel_url)" ] && svc_running tunnel; then
    tunnel="
  │    Public link through Cloudflare: anyone with it (key included) can use the page. It changes when the
  │    tunnel restarts; 'docling-rtx web url' shows the current one. Uploads over 100 MB: copy into the inbox."
  fi
  [ -z "$tunnel" ] && [ "$WEB_HOST" != 127.0.0.1 ] && [ "$(public_port)" != "$WEB_PORT" ] && tunnel="
  │    (the page listens on port $WEB_PORT in this machine; your provider maps it to $(public_port) outside)"
  [ -z "$tunnel" ] && [ "$WEB_HOST" != 127.0.0.1 ] && [ "$(public_port)" = "$WEB_PORT" ] && tunnel="
  │    Listening on port $WEB_PORT. If your provider maps it to another public port, set it:
  │      docling-rtx web port $WEB_PORT <public port>"
  [ -z "$tunnel" ] && [ "$WEB_HOST" = 127.0.0.1 ] && tunnel="
  │    It listens on this server only. From your laptop, add  -L $WEB_PORT:127.0.0.1:$WEB_PORT  to the ssh
  │    command you use for this server, e.g.  ssh -L $WEB_PORT:127.0.0.1:$WEB_PORT root@<server> -p <port>
  │    and keep that ssh session open while you use the page."
  cat <<EOF

  ┌─ How to use ───────────────────────────────────────────────────────────────────────────────────────
  │ 1. Web page (easiest): open $(web_url) in your browser$tunnel
  │    Drag documents onto the page, watch the progress, download the JSON.
  │
  │ 2. Or copy files into   $INBOX
  │    JSON appears in      $OUTBOX    (name.json = done, name.error.json = failed)
  │
  │ docling-rtx status        is everything running?
  │ docling-rtx watch logs    what the converter is doing right now
  │ docling-rtx               the menu          Guide: $DOCLING_HOME/GUIDE.md
  └────────────────────────────────────────────────────────────────────────────────────────────────────
EOF
}

# --- server control --------------------------------------------------------------------------------------
pidfile=$DOCLING_HOME/vllm.pid
healthy() { curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }
vllm_unit() { has_systemd && [ -f /etc/systemd/system/$SERVICE.service ]; }

server_start() {
  detect_gpu
  [ -n "$DRIVER" ] || { warn "No NVIDIA GPU: the VLM server needs one."; return 1; }
  ensure_vlm || return 1
  rm -f "$DOCLING_HOME/vllm.off"
  healthy && { ok "Already running on port $PORT"; return; }
  vllm_pid_alive && server_kill   # an old one on another port (settings changed): never two vLLMs on one card
  say "Starting vLLM on 127.0.0.1:$PORT (first start takes a minute or two: CUDA graphs are compiled)"
  if vllm_unit; then
    $SUDO systemctl restart $SERVICE
  else
    VLLM_NO_USAGE_STATS=1 setsid nohup $(vllm_cmd) > "$DOCLING_HOME/vllm.log" 2>&1 < /dev/null &
    echo $! > "$pidfile"
  fi
  for _ in $(seq 1 120); do
    healthy && { ok "vLLM is up: http://127.0.0.1:$PORT/v1 (model $MODEL)"; return; }
    if ! vllm_unit && ! kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then break; fi
    sleep 5
  done
  warn "vLLM didn't come up. The errors in its log (full log: $0 logs):"
  server_logs | grep -aE "[A-Za-z]+(Error|Exception): |out of memory|Python.h" \
    | grep -avE "Engine core initialization failed|FA version 2" | tail -6
  return 1
}
vllm_pid_alive() { [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; }
server_kill() {   # background vLLM: its whole process group (API server + engine core), then wait for the VRAM
  local pid; pid=$(cat "$pidfile" 2>/dev/null) || return 0
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  rm -f "$pidfile"
}
server_stop() {
  [ -d "$DOCLING_HOME" ] && touch "$DOCLING_HOME/vllm.off"   # autostart leaves it off until 'start'
  if vllm_unit; then $SUDO systemctl stop $SERVICE; ok "vLLM stopped"
  elif vllm_pid_alive; then server_kill; ok "vLLM stopped"
  else rm -f "$pidfile"; ok "vLLM wasn't running"; fi
}
server_logs() {
  if vllm_unit; then $SUDO journalctl -u $SERVICE -n 200 --no-pager; else cat "$DOCLING_HOME/vllm.log" 2>/dev/null; fi
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
  vllm_unit && echo "Service:  $SERVICE ($(systemctl is-enabled $SERVICE 2>/dev/null), starts on boot)"
  if watch_running; then echo "Watcher:  UP   $INBOX -> $OUTBOX ($WATCH_FORMATS, engine $WATCH_ENGINE)"
  else echo "Watcher:  down ($INBOX -> $OUTBOX)"; fi
  if svc_running web; then echo "Web page: UP   $(web_url)"; else echo "Web page: down (docling-rtx web start)"; fi
  if [ "$WEB_TUNNEL" = 1 ]; then
    if svc_running tunnel && [ -n "$(tunnel_url)" ]; then echo "Tunnel:   UP   $(tunnel_url)  (Cloudflare quick tunnel)"
    else echo "Tunnel:   down (docling-rtx web tunnel on)"; fi
  fi
  echo "Settings: PORT=$PORT GPU_UTIL=$(gpu_util) BATCH=$(auto_batch) CUDA=$(cuda_flavor)"
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
  server_stop || true; svc_remove watch; svc_remove tunnel; svc_remove web
  rm -f "$HOME/.local/bin/docling-rtx"
  local c
  for c in docling-rtx docling-convert docling; do   # only our links
    [ "$(readlink "/usr/local/bin/$c" 2>/dev/null)" = "$HOME/.local/bin/$c" ] && rm -f "/usr/local/bin/$c"
  done
  if [ -f "$HOME/.bashrc" ]; then sed -i '/docling-rtx autostart/d' "$HOME/.bashrc"; fi
  if vllm_unit; then $SUDO systemctl disable -q $SERVICE; $SUDO rm -f /etc/systemd/system/$SERVICE.service; $SUDO systemctl daemon-reload; fi
  rm -f "$HOME/.local/bin/docling-convert" "$HOME/.local/bin/docling"
  rm -rf "$DOCLING_HOME"
  ok "Removed $DOCLING_HOME, the launchers and the services. Kept: your inbox/outbox folders, uv, apt packages, the driver, ~/.cache/huggingface."
}

# vLLM (several GB, minutes to install) only when it's used: WATCH_ENGINE=vlm or WITH_VLM=1. Otherwise it installs
# on first use: docling-rtx start, or choosing the vlm engine in the drop-folder menu.
ALL_STEPS="preflight apt uv docling vlm watch web autostart warmup"
step_vlm() {   # the VLM stack, if this install uses it
  if uses_vlm; then step_vllm; step_models; step_service
  else ok "VLM server (vLLM) not installed: the standard engine doesn't need it. Later: $ME start"; fi
}
ensure_vlm() {   # install vLLM on first use
  [ -x "$DOCLING_HOME/vllm/bin/vllm" ] && return 0
  say "First use of the VLM engine: installing vLLM + granite-docling (a few minutes, ~8 GB)"
  step_vllm && step_models && step_service
}
install_steps() {   # $@ = steps
  load_conf
  [ "$DRY" = 1 ] || save_conf
  for s in "$@"; do "step_$s"; done
  if [ "$DRY" = 0 ] && [ -x "$DOCLING_HOME/vllm/bin/vllm" ] && [ ! -f "$DOCLING_HOME/vllm.off" ] &&
     { [[ " $* " == *" service "* ]] || { [[ " $* " == *" vlm "* ]] && uses_vlm; }; }; then server_start || true; fi
  [ "$DRY" = 1 ] && { echo; ok "Dry run: nothing was changed."; return; }
  echo; ok "Installed."; howto
  if ! [ -L /usr/local/bin/docling-rtx ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    warn "Open a new shell for the docling-rtx command (or run: export PATH=\$HOME/.local/bin:\$PATH)"
  fi
}

# --- whiptail menu ---------------------------------------------------------------------------------------
menu_settings() {
  local v
  local old_web="$WEB_HOST:$WEB_PORT:$WEB_PUBLIC_PORT:$WEB_TUNNEL" old_vllm="$PORT:$GPU_UTIL"
  v=$(wt --title "Web page port" --inputbox "Port the web page listens on (your GPU provider may give you one):" 9 70 "$WEB_PORT") && valid_port "$v" && WEB_PORT=$v
  v=$(wt --title "Web page access" --radiolist "How you reach the web page:" 13 78 3 \
      tunnel "public https link via Cloudflare, with a key (no open port needed)" "$([ "$WEB_TUNNEL" = 1 ] && echo ON || echo OFF)" \
      private "SSH tunnel only (ssh -L ...), safest" "$([ "$WEB_TUNNEL" != 1 ] && [ "$WEB_HOST" = 127.0.0.1 ] && echo ON || echo OFF)" \
      public "direct link http://IP:PORT with a key (provider-exposed port)" "$([ "$WEB_TUNNEL" != 1 ] && [ "$WEB_HOST" != 127.0.0.1 ] && echo ON || echo OFF)") &&
    case $v in
      tunnel) WEB_TUNNEL=1 WEB_HOST=127.0.0.1 ;;
      private) WEB_TUNNEL=0 WEB_HOST=127.0.0.1 ;;
      public) WEB_TUNNEL=0 WEB_HOST=0.0.0.0 ;;
    esac
  if [ "$WEB_HOST" != 127.0.0.1 ]; then
    v=$(wt --title "Public port" --inputbox "If your provider maps port $WEB_PORT to a different public port, enter it.\nLeave empty if it's the same (detected now: $(public_port)):" 10 76 "$WEB_PUBLIC_PORT") &&
      { [ -z "$v" ] || valid_port "$v"; } && WEB_PUBLIC_PORT=$v
  fi
  v=$(wt --title "vLLM port" --inputbox "Port for the vLLM server (localhost only, nothing outside needs it):" 9 70 "$PORT") && valid_port "$v" && PORT=$v
  v=$(wt --title "GPU memory" --inputbox "Share of GPU memory vLLM reserves (0.1-0.95), or auto (0.5; 0.7 under 10 GB).\n0.5 leaves room for the standard pipeline (drop folder) on the same card; 0.9 if you only use the vlm engine:" 12 76 "$GPU_UTIL") && GPU_UTIL=$v
  v=$(wt --title "Batch size" --inputbox "Pages in flight per conversion (auto = from VRAM: 64 for 32 GB, 32 for 24 GB, 16 below):" 10 70 "$BATCH") && BATCH=$v
  v=$(wt --title "PyTorch CUDA build" --radiolist "PyTorch wheel for Docling (auto = from your driver: $(cuda_flavor)):" 14 70 5 \
      auto "pick from the driver" "$([ "$CUDA" = auto ] && echo ON || echo OFF)" \
      cu130 "CUDA 13.0 (driver >= 580)" "$([ "$CUDA" = cu130 ] && echo ON || echo OFF)" \
      cu128 "CUDA 12.8 (driver >= 570, RTX 50 needs this or newer)" "$([ "$CUDA" = cu128 ] && echo ON || echo OFF)" \
      cu126 "CUDA 12.6 (driver >= 525)" "$([ "$CUDA" = cu126 ] && echo ON || echo OFF)" \
      cpu "CPU only" "$([ "$CUDA" = cpu ] && echo ON || echo OFF)") && CUDA=$v
  save_conf
  if [ "$old_vllm" != "$PORT:$GPU_UTIL" ] && { vllm_unit || vllm_pid_alive; }; then
    vllm_unit && step_service
    wt --yesno "vLLM settings changed. Restart vLLM now to apply them?" 8 60 && { server_start_restart; pause; }
  fi
  watch_apply || true
  if [ "$old_web" != "$WEB_HOST:$WEB_PORT:$WEB_PUBLIC_PORT:$WEB_TUNNEL" ]; then web_apply; pause; fi
}
server_start_restart() { server_stop; server_start; }

menu_watch() {
  local c=1 st v
  while true; do
    st="stopped"; watch_running && st="RUNNING"
    c=$(wt --title "Drop folder" --default-item "$c" --menu "Files dropped in the inbox are converted to $WATCH_FORMATS in the outbox.\n\nWatcher: $st\nInbox:   $INBOX\nOutbox:  $OUTBOX\nEngine:  $WATCH_ENGINE\n" 21 78 6 \
      1 "Start / restart the watcher" \
      2 "Stop the watcher" \
      3 "Folders, output format, engine..." \
      4 "Watcher log" \
      5 "Latest outputs" \
      6 "Back") || return 0
    clear
    case $c in
      1) watch_start || true; pause ;;
      2) watch_stop; pause ;;
      3) v=$(wt --title "Inbox" --inputbox "Folder to watch (subfolders included):" 9 70 "$INBOX") && INBOX=$v
         v=$(wt --title "Outbox" --inputbox "Where the converted files go (keep it outside the inbox):" 9 78 "$OUTBOX") && OUTBOX=$v
         v=$(wt --title "Output" --checklist "Files per document (JSON = DoclingDocument: text, tables, layout, reading order):" 14 78 4 \
           json "structured JSON for agents/programs" "$([[ " $WATCH_FORMATS " == *" json "* ]] && echo ON || echo OFF)" \
           md "Markdown (compact, LLM-friendly reading)" "$([[ " $WATCH_FORMATS " == *" md "* ]] && echo ON || echo OFF)" \
           html "HTML" "$([[ " $WATCH_FORMATS " == *" html "* ]] && echo ON || echo OFF)" \
           txt "plain text" "$([[ " $WATCH_FORMATS " == *" txt "* ]] && echo ON || echo OFF)") && v=$(echo "$v" | tr -d '"') && [ -n "$v" ] && WATCH_FORMATS=$v
         v=$(wt --title "Engine" --radiolist "How PDFs and images are read (Office/CSV/HTML don't need it):" 13 78 3 \
           standard "layout + tables + OCR models (best for digital PDFs)" "$([ "$WATCH_ENGINE" = standard ] && echo ON || echo OFF)" \
           vlm "granite-docling via vLLM (scans, formulas, code)" "$([ "$WATCH_ENGINE" = vlm ] && echo ON || echo OFF)" \
           local "granite-docling in-process (no server)" "$([ "$WATCH_ENGINE" = local ] && echo ON || echo OFF)") && WATCH_ENGINE=$v
         save_conf
         # vlm needs the server (installed on first use), started before the watcher so its first file finds it
         if [ "$WATCH_ENGINE" = vlm ] && ! healthy; then server_start || true; fi
         watch_apply || true; pause ;;
      4) watch_logs | tail -60; pause ;;
      5) ls -lt "$OUTBOX" 2>/dev/null | head -20; pause ;;
      6) return 0 ;;
    esac
  done
}

pause() { read -rp "Press Enter to return to the menu..." _ </dev/tty; }

menu() {
  load_conf
  local c=1
  while true; do
    detect_gpu
    local up="down" wst="down"; healthy && up="UP on :$PORT"; watch_running && wst="UP"
    c=$(wt --title "Docling RTX setup" --default-item "$c" --menu "GPU: ${GPU_NAME:-none found}${GPU_MB:+ ($(((GPU_MB + 512) / 1024)) GB)}   vLLM: $up   Drop folder: $wst\n\nChoose:" 22 76 12 \
      1 "Install everything (recommended)" \
      2 "Choose what to install..." \
      3 "Drop folder: inbox -> JSON..." \
      4 "Start vLLM server" \
      5 "Stop vLLM server" \
      6 "Status" \
      7 "Run a test conversion" \
      8 "Convert a folder once..." \
      9 "Settings (web port & access, GPU memory, batch, CUDA)" \
      10 "vLLM server log" \
      11 "How to use it (web page, drop folder)" \
      12 "Uninstall") || { clear; exit 0; }
    clear
    case $c in
      1) wt --yesno "Install:\n\n - system packages (apt)\n - uv + Python 3.12\n - Docling with CUDA PyTorch\n - VLM engine (vLLM + granite-docling): only if the drop folder uses it\n - drop folder: $INBOX -> $OUTBOX\n - web page: drag & drop, progress, downloads\n\nNeeds ~8 GB of disk (20 GB with the VLM engine) and sudo. Continue?" 18 66 && { install_steps $ALL_STEPS; pause; } ;;
      2) local picks
         picks=$(wt --title "Components" --checklist "Space to toggle, Enter to install:" 21 76 11 \
           driver "NVIDIA driver (Ubuntu, needs a reboot)" OFF \
           apt "System packages" ON \
           uv "uv (Python manager)" ON \
           docling "Docling + CUDA PyTorch" ON \
           vllm "vLLM server engine (for the vlm engine)" OFF \
           models "Download granite-docling-258M (vlm engine)" OFF \
           service "Run vLLM as a service (systemd)" OFF \
           watch "Drop folder: inbox -> JSON (service)" ON \
           web "Web page: drag & drop, progress, downloads" ON \
           autostart "Restart services at login (no-systemd pods)" ON) && { install_steps preflight $(echo "$picks" | tr -d '"'); pause; } ;;
      3) menu_watch ;;
      4) server_start || true; pause ;;
      5) server_stop; pause ;;
      6) status; pause ;;
      7) test_run || true; pause ;;
      8) local in outd eng
         in=$(wt --inputbox "Folder (or file / URL) to convert:" 9 70 "$HOME/documents") &&
         outd=$(wt --inputbox "Output folder:" 9 70 "$HOME/docling-out") &&
         eng=$(wt --radiolist "Engine:" 13 76 3 \
           standard "layout + tables + OCR models (digital PDFs, Office)" ON \
           vlm "granite-docling via vLLM (scans, formulas, code)" OFF \
           local "granite-docling in-process (no server, slower)" OFF) &&
         { "$HOME/.local/bin/docling-convert" "$in" -o "$outd" --engine "$eng" --batch "$(auto_batch)" \
             --to json --to md --server "http://127.0.0.1:$PORT/v1/chat/completions" || true; pause; } ;;
      9) menu_settings ;;
      10) server_logs | tail -60; pause ;;
      11) howto; pause ;;
      12) wt --yesno "Remove $DOCLING_HOME, the launchers and the services?\n(Kept: your inbox/outbox and the model cache.)" 10 64 && { uninstall; pause; } ;;
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
  watch)
    case ${args[1]:-start} in
      start) watch_start ;; stop) watch_stop ;; logs) watch_logs ;; restart) watch_stop; watch_start ;;
      *) die "Use: $ME watch start|stop|restart|logs" ;;
    esac ;;
  web)
    case ${args[1]:-url} in
      start|restart) web_start && { [ "$WEB_TUNNEL" != 1 ] || tunnel_start; } && howto ;;
      stop) svc_stop tunnel; svc_stop web; ok "Web page stopped" ;;
      tunnel)   # web tunnel on|off|logs: the public Cloudflare link
        case ${args[2]:-on} in
          on) WEB_TUNNEL=1; web_apply ;;
          off) WEB_TUNNEL=0; web_apply ;;
          logs) cat "$DOCLING_HOME/tunnel-cf.log" 2>/dev/null | tail -40 ;;
          *) die "Use: $ME web tunnel on|off|logs" ;;
        esac ;;
      logs) svc_logs web ;;
      url) howto ;;
      public|private)   # public: all interfaces + a key in the link (pods that expose ports); private: SSH tunnel
        if [ "${args[1]}" = public ]; then WEB_HOST=0.0.0.0; else WEB_HOST=127.0.0.1; fi
        web_apply ;;
      port)   # web port <port the page listens on> [<public port your provider maps it to>]
        valid_port "${args[2]:-}" || die "Use: $ME web port <port> [public port], e.g. $ME web port 8080 41234"
        [ -z "${args[3]:-}" ] || valid_port "${args[3]}" || die "Not a port: ${args[3]}"
        WEB_PORT=${args[2]} WEB_PUBLIC_PORT=${args[3]:-}
        web_apply ;;
      *) die "Use: $ME web [url|start|stop|logs|public|private|tunnel on|off|port <port> [public port]]" ;;
    esac ;;
  howto|help) howto ;;
  autostart) autostart ;;
  uninstall) uninstall ;;
  "")
    if [ "$DRY" = 1 ]; then install_steps $ALL_STEPS
    elif interactive; then need_whiptail; menu
    else die "No terminal for the menu (stdout isn't a terminal): use '$0 install' (or --help)."; fi ;;
  *) die "Unknown command: ${args[0]} (see --help)" ;;
esac
