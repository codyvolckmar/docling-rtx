#!/usr/bin/env bash
# Summon installer for Linux - AI, Open-source, Private Intelligence.
#
#   curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash
#   curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash -s -- --dry-run      # preview, installs nothing
#   curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash -s -- ornith9b 40000 # skip the first menu
#
# Downloads summon.sh to ~/.summon/ (rerun it later for --lock / --uninstall) and starts the wizard with your
# terminal attached. Needs: Linux with apt (Ubuntu/Debian, incl. WSL 2), an NVIDIA GPU + driver, root or sudo.
# Read before you pipe: https://github.com/codyvolckmar/summon/blob/main/summon.sh
set -e

SRC=${SUMMON_SRC:-https://raw.githubusercontent.com/codyvolckmar/summon/main/summon.sh}
DIR=${SUMMON_DIR:-$HOME/.summon}

[ "$(uname -s)" = Linux ] || { echo "Summon runs on Linux. On Windows use install.ps1 (it runs Summon inside WSL 2)."; exit 1; }
command -v curl >/dev/null || { echo "Please install curl first (sudo apt-get install -y curl)."; exit 1; }

mkdir -p "$DIR"
curl -fsSL "$SRC" -o "$DIR/summon.sh.new"
bash -n "$DIR/summon.sh.new" || { echo "Downloaded summon.sh is not a valid script; try again later."; rm -f "$DIR/summon.sh.new"; exit 1; }
mv -f "$DIR/summon.sh.new" "$DIR/summon.sh"
chmod +x "$DIR/summon.sh"
echo "Summon saved to $DIR/summon.sh  (later: $DIR/summon.sh --lock | --uninstall | --help)"

# This script arrives on stdin (curl | bash), so give the wizard the real terminal for its menus.
TTY=/dev/null
if [ -r /dev/tty ] && { : < /dev/tty; } 2>/dev/null; then TTY=/dev/tty; fi

# Root is needed for apt, Docker and the firewall rule; a dry run needs nothing. sudo drops the environment,
# so pass Summon's settings (and SSH_CLIENT, which decides how the port is locked) explicitly.
case " $* " in *" --dry-run "*|*" --help "*|*" -h "*) NEED_ROOT=0 ;; *) NEED_ROOT=1 ;; esac
if [ "$(id -u)" = 0 ] || [ "$NEED_ROOT" = 0 ]; then
  exec bash "$DIR/summon.sh" "$@" < "$TTY"
fi
command -v sudo >/dev/null || { echo "Run this as root (or install sudo)."; exit 1; }
ENVS=()
for v in SSH_CLIENT SSH_CONNECTION ALLOW_IP ENGINE CTX QUANT ONLY FAST SPEC THINK_BUDGET TEMPERATURE TOP_P TOP_K \
         VERIFY_SHA256 OLLAMA_KV_CACHE_TYPE OLLAMA_KEEP_ALIVE SUMMON_GPU TERM; do
  [ -n "${!v:-}" ] && ENVS+=("$v=${!v}")
done
echo "Summon needs root for apt, Docker and the firewall: asking sudo."
exec sudo env "${ENVS[@]}" bash "$DIR/summon.sh" "$@" < "$TTY"
