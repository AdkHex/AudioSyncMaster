#!/usr/bin/env bash
# Development setup for macOS and Linux.
# Usage:  ./dev.sh [--sidecar]   (--sidecar also builds the frozen engine)
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="python/.venv"
BUILD_SIDECAR=0
[[ "${1:-}" == "--sidecar" ]] && BUILD_SIDECAR=1

echo "[1/5] Python environment"
if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r python/requirements-dev.txt

echo "[2/5] Checking for FFmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "  WARNING: ffmpeg is not on PATH."
  echo "  Install it (macOS: brew install ffmpeg) or place binaries in"
  echo "  src-tauri/resources/ffmpeg/ to have them bundled with the app."
else
  echo "  found: $(command -v ffmpeg)"
fi

echo "[3/5] Node dependencies"
npm install --no-audit --no-fund

if [[ "$BUILD_SIDECAR" == "1" ]]; then
  echo "[4/5] Building the sidecar"
  "$VENV/bin/pip" install --quiet pyinstaller

  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  TRIPLE="aarch64-apple-darwin" ;;
    Darwin-x86_64) TRIPLE="x86_64-apple-darwin" ;;
    Linux-x86_64)  TRIPLE="x86_64-unknown-linux-gnu" ;;
    *)             TRIPLE="" ;;
  esac

  "$VENV/bin/pyinstaller" --onefile --clean --noconfirm \
    --name audiosync-cli \
    --paths . \
    --hidden-import audiosync \
    --hidden-import audiosync.analyze \
    --hidden-import audiosync.batch \
    --hidden-import audiosync.correlate \
    --hidden-import audiosync.matching \
    --hidden-import audiosync.media \
    --hidden-import audiosync.mux \
    --collect-all numpy \
    --collect-all scipy \
    python/bridge.py

  mkdir -p src-tauri/bin
  cp dist/audiosync-cli src-tauri/bin/audiosync-cli
  if [[ -n "$TRIPLE" ]]; then
    cp dist/audiosync-cli "src-tauri/bin/audiosync-cli-$TRIPLE"
  fi
  echo "  sidecar written to src-tauri/bin/"
else
  echo "[4/5] Skipping sidecar build (pass --sidecar to build it)"
fi

echo "[5/5] Starting Tauri"
npm run tauri:dev
