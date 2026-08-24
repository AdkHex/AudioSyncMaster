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

  # A directory build, NOT --onefile. Onefile re-extracts its whole payload to
  # a temp folder on every launch, which measured 32-63s per run on macOS.
  # A directory build starts in ~0.12s because nothing is unpacked.
  rm -rf src-tauri/resources/engine
  "$VENV/bin/pyinstaller" --onedir --clean --noconfirm --log-level WARN \
    --distpath build/engine --workpath build/pyi --specpath build/pyi \
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
    --exclude-module scipy \
    --exclude-module matplotlib \
    --exclude-module tkinter \
    --exclude-module PIL \
    --exclude-module pandas \
    --exclude-module pytest \
    python/bridge.py

  mkdir -p src-tauri/resources
  cp -R build/engine/audiosync-cli src-tauri/resources/engine
  chmod +x src-tauri/resources/engine/audiosync-cli

  echo "  engine written to src-tauri/resources/engine/"
  echo -n "  smoke test: "
  echo '{"command":"ping"}' | ./src-tauri/resources/engine/audiosync-cli | head -2 | tail -1
else
  echo "[4/5] Skipping sidecar build (pass --sidecar to build it)"
fi

echo "[5/5] Starting Tauri"
npm run tauri:dev
