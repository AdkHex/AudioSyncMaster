"""Silero voice-activity worker (runs in the asr-faster pack).

faster-whisper ships the Silero VAD v6 ONNX model and a small wrapper, so the
asr-faster pack gives the engine a neural speech detector at no extra cost.

Request::

    {"audio": "<media file>", "audioTrack": 0, "outDir": "<folder>" | null}

``audio`` may be the 16 kHz mono 16-bit WAV the engine already extracted
(read directly, fastest) or any media file, decoded here with PyAV -- which
faster-whisper depends on anyway -- so a caller without a WAV at hand need
not make one.

Result::

    {"hopS": 0.032, "probsPath": "<raw float32 file>"}   when outDir is given
    {"hopS": 0.032, "probs": [0.01, 0.98, ...]}          otherwise
    + "count", "duration"

One probability per 512-sample (32 ms) window; callers resample onto their
own grid. A two-hour film has 225,000 windows, which is why a file is
preferred over a JSON list.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as proto  # noqa: E402

SR = 16000
WINDOW = 512
#: Windows per model call: about 5 minutes, so progress moves steadily.
BATCH = 10000
STAGE = "Detecting speech"


def _is_engine_wav(path: str) -> bool:
    import wave

    if not path.lower().endswith(".wav"):
        return False
    try:
        with wave.open(path, "rb") as handle:
            return handle.getframerate() == SR and handle.getnchannels() == 1 and handle.getsampwidth() == 2
    except (wave.Error, EOFError, OSError):
        return False


def _decode(path: str, track: int):
    """Mono 16 kHz float32 of audio stream ``track`` of any media file."""
    import av
    import numpy as np

    parts = []
    with av.open(path) as container:
        streams = container.streams.audio
        if not streams:
            raise ValueError(f"{os.path.basename(path)} has no audio")
        stream = streams[min(max(0, track), len(streams) - 1)]
        duration = float(stream.duration * stream.time_base) if stream.duration else (
            container.duration / 1e6 if container.duration else 0.0)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        last = -1.0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
            if duration and frame.time is not None and frame.time - last >= 10:
                last = frame.time
                proto.progress(50.0 * min(1.0, frame.time / duration), "Decoding audio")
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32) / 32768.0


def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    from faster_whisper.vad import get_vad_model

    path = req["audio"]
    decoded = not _is_engine_wav(path)
    if decoded:
        audio = _decode(path, int(req.get("audioTrack") or 0))
    else:
        audio, _rate = proto.read_wav(path)
    model = get_vad_model()
    padded = np.pad(audio, (0, (-len(audio)) % WINDOW)).astype(np.float32)
    step = WINDOW * BATCH
    base, span = (50.0, 50.0) if decoded else (0.0, 100.0)
    parts = []
    for start in range(0, len(padded), step):
        # Each call restarts the model's recurrent state; a restart every
        # five minutes costs a few windows of accuracy, nothing more.
        parts.append(np.asarray(model(padded[start:start + step]), dtype=np.float32).reshape(-1))
        proto.progress(base + span * min(len(padded), start + step) / max(1, len(padded)), STAGE)
    probs = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    out: Dict[str, Any] = {"hopS": WINDOW / SR, "count": int(probs.size), "duration": len(audio) / SR}
    if req.get("outDir"):
        target = os.path.join(req["outDir"], "silero_probs.f32")
        probs.astype("<f4").tofile(target)
        out["probsPath"] = target
    else:
        out["probs"] = [round(float(p), 3) for p in probs]
    return out


if __name__ == "__main__":
    proto.main(handle)
