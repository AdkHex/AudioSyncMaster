"""The JSON-lines protocol every pack worker speaks with the engine.

Workers run inside a pack's own Python (mlx-whisper, faster-whisper, Demucs,
RapidOCR), never inside the shipped engine, so this file must not import
``audiosync`` or anything outside the standard library.

The engine writes one JSON request to the worker's stdin and closes it. The
worker answers with JSON lines on stdout:

    {"type": "progress", "percent": 42.0, "stage": "Recognising speech"}
    {"type": "log", "message": "Detected Japanese (99%)"}
    {"type": "result", "data": {...}}          exactly once on success
    {"type": "error", "message": "..."}        instead of a result

Libraries print to stdout freely (tqdm bars, "Detected language", C
extensions writing to fd 1), and one stray line would corrupt the protocol.
So ``main`` keeps the real stdout file descriptor for protocol lines only and
points fd 1 at stderr: whatever a library prints ends up in the stderr tail
the engine shows when a worker crashes, never in the stream it parses. (It
happens in ``main`` rather than on import so tests can import a worker's
helpers without losing their own stdout.)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional

_out = None
_lock = threading.Lock()
_last_progress: Optional[Dict[str, Any]] = None
_last_emit = time.monotonic()

#: The host gives up on a job after 30 minutes without an event. A single
#: 30-second Whisper window or a large model load never takes that long, but
#: a slow CPU separating a two-hour film with Demucs can go quiet for a long
#: stretch, so the last progress is repeated on this interval.
HEARTBEAT_S = 60.0


def _claim_stdout() -> None:
    """Keep fd 1 for protocol lines; send everything else printed to stderr."""
    global _out
    if _out is None:
        proto_fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        _out = os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)


def _emit(event: Dict[str, Any]) -> None:
    global _last_emit
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    with _lock:
        stream = _out or sys.stdout
        stream.write(line + "\n")
        stream.flush()
        _last_emit = time.monotonic()


def progress(percent: float, stage: Optional[str] = None, **extra: Any) -> None:
    """Report ``percent`` (0-100) of the current stage."""
    global _last_progress
    event: Dict[str, Any] = {"type": "progress", "percent": round(max(0.0, min(100.0, float(percent))), 2)}
    if stage:
        event["stage"] = stage
    event.update(extra)
    _last_progress = event
    _emit(event)


def log(message: str) -> None:
    _emit({"type": "log", "message": str(message)})


def _heartbeat() -> None:
    while True:
        time.sleep(HEARTBEAT_S / 4)
        if time.monotonic() - _last_emit >= HEARTBEAT_S:
            _emit(_last_progress or {"type": "progress", "percent": 0.0})


def read_wav(path: str):
    """Mono float32 samples and the rate of a 16-bit PCM WAV.

    The engine always hands workers 16-bit WAVs made by its own FFmpeg, so
    the standard library's ``wave`` is enough and no worker needs an audio
    library (or FFmpeg on its PATH) of its own.
    """
    import wave

    import numpy as np

    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{os.path.basename(path)}: expected 16-bit PCM, got {8 * width}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, rate


def main(handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Read the request, run ``handler``, and report its result or error.

    The exit code is non-zero on failure so a crash before any JSON (a
    missing library, a segfault in a native extension) is still told apart
    from a worker that simply returned nothing.
    """
    _claim_stdout()
    threading.Thread(target=_heartbeat, daemon=True).start()
    try:
        raw = sys.stdin.read()
        request = json.loads(raw) if raw.strip() else {}
        result = handler(request)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - reported to the engine
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        sys.stderr.flush()
        sys.exit(1)
    _emit({"type": "result", "data": result if result is not None else {}})
