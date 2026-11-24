#!/usr/bin/env bash
# One-line installer for Docling RTX:
#   curl -fsSL https://raw.githubusercontent.com/codyvolckmar/docling-rtx/main/install.sh | bash
# Clones (or updates) the repo into ~/docling-rtx and runs the full install without questions.
# Then: open the web page it prints, or run  docling-rtx  for the menu.
set -euo pipefail
REPO=${REPO:-https://github.com/codyvolckmar/docling-rtx}
DIR=${DIR:-$HOME/docling-rtx}
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
if ! command -v git >/dev/null 2>&1; then
  echo "==> Installing git"
  $SUDO apt-get update -q >/dev/null && $SUDO apt-get install -y -q git >/dev/null
fi
if [ -d "$DIR/.git" ]; then
  echo "==> Updating $DIR"; git -C "$DIR" pull -q --ff-only
else
  echo "==> Cloning $REPO into $DIR"; git clone -q "$REPO" "$DIR"
fi
exec "$DIR/docling-setup.sh" install "$@"
