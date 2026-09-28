#!/bin/sh
# Build the Apple Vision OCR helper (macOS only) as a universal binary.
#
#   native/vision-ocr/build.sh [output-path]
#
# Default output: src-tauri/resources/tools/vision-ocr, which is where
# audiosync/subs/ocr_engines.py looks for a bundled copy (a "tools" folder
# beside the bundled "ffmpeg" folder). Set CODESIGN_IDENTITY to sign with the
# app's Developer ID (needed for notarisation); the default is an ad-hoc
# signature, enough for local and CI test runs.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/src-tauri/resources/tools/vision-ocr"}
min_macos=${VISION_OCR_MIN_MACOS:-11.0}

if [ "$(uname -s)" != "Darwin" ]; then
    echo "vision-ocr is macOS only; skipping." >&2
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Universal by default. A toolchain without an architecture's Swift runtime
# libraries (Command Line Tools on recent macOS ship arm64 only) skips that
# architecture with a warning instead of failing the whole build.
built=""
for arch in ${VISION_OCR_ARCHS:-arm64 x86_64}; do
    if swiftc -O -swift-version 5 -target "$arch-apple-macos$min_macos" \
        -o "$tmp/vision-ocr-$arch" "$here/main.swift" 2>"$tmp/$arch.log"; then
        built="$built $tmp/vision-ocr-$arch"
    else
        echo "warning: could not build vision-ocr for $arch:" >&2
        tail -3 "$tmp/$arch.log" >&2
    fi
done
[ -n "$built" ] || { echo "error: vision-ocr did not build for any architecture" >&2; exit 1; }

mkdir -p "$(dirname "$out")"
# shellcheck disable=SC2086
lipo -create -output "$out" $built
chmod 755 "$out"
codesign --force --timestamp=none --options runtime --sign "${CODESIGN_IDENTITY:--}" "$out"
"$out" --version
echo "Built $out"
