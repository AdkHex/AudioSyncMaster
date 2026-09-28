"""Voice isolation worker: Demucs vocals stem (runs in the demucs pack).

Request::

    {"audio": "<16-bit WAV at the model's rate, 44.1 kHz stereo>",
     "out": "<vocals WAV to write>", "outRate": 16000,
     "modelPath": "<local HTDemucs snapshot>" | null, "model": "htdemucs"}

Result: {"out", "seconds", "device", "model"}

The film is separated in 60-second chunks that overlap by two seconds and
are cross-faded, and each chunk's vocals are written out (mono, resampled to
``outRate`` for speech recognition) before the next is read. Demucs itself
wants the whole track in memory -- for a two-hour film that is 2.5 GB of
input plus four stems of output -- and a laptop running Whisper next to it
does not have that to spare.
"""

from __future__ import annotations

import os
import sys
import time
import wave
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as proto  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

CHUNK_S = 60.0
OVERLAP_S = 2.0
STAGE = "Isolating voices"


def _load_model(req: Dict[str, Any]):
    from demucs.pretrained import get_model

    local = req.get("modelPath")
    if local and os.path.isdir(local):
        # A local snapshot: load the bag it describes without the hub.
        import yaml
        from demucs.apply import BagOfModels
        from demucs.hf import load_safetensors_model

        name = req.get("model") or "htdemucs"
        with open(os.path.join(local, f"{name}.yaml"), encoding="utf-8") as handle:
            bag = yaml.safe_load(handle)
        models = [load_safetensors_model(os.path.join(local, f"{sig}.safetensors")) for sig in bag["models"]]
        model = BagOfModels(models, bag.get("weights"), bag.get("segment"))
    else:
        model = get_model(req.get("model") or "htdemucs")
    model.eval()
    return model


def _device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    import julius
    import numpy as np
    import torch
    from demucs.apply import apply_model

    proto.progress(0, "Loading Demucs")
    model = _load_model(req)
    device = _device(torch)
    vocals_index = list(model.sources).index("vocals")
    out_rate = int(req.get("outRate") or 16000)
    started = time.monotonic()

    with wave.open(req["audio"], "rb") as src, wave.open(req["out"], "wb") as dst:
        rate, channels = src.getframerate(), src.getnchannels()
        if rate != model.samplerate:
            raise ValueError(f"Demucs needs {model.samplerate} Hz input, got {rate} Hz")
        total = src.getnframes()
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(out_rate)
        chunk = int(CHUNK_S * rate)
        overlap_out = int(OVERLAP_S * out_rate)
        fade_in = np.linspace(0.0, 1.0, overlap_out, dtype=np.float32)
        carry = None  # the previous chunk's last OVERLAP_S, not yet written
        position = 0
        while position < total:
            # Read OVERLAP_S back so consecutive chunks share two seconds.
            start = max(0, position - (int(OVERLAP_S * rate) if position else 0))
            src.setpos(start)
            frames = src.readframes(min(chunk, total - start))
            pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            mix = torch.from_numpy(pcm.reshape(-1, channels).T.copy())
            if channels == 1:
                mix = mix.repeat(2, 1)
            ref = mix.mean(0)
            mean, std = ref.mean(), ref.std().clamp_min(1e-6)
            with torch.no_grad():
                sources = apply_model(model, ((mix - mean) / std)[None], device=device,
                                      split=True, overlap=0.25, progress=False)[0]
            voice = sources[vocals_index] * std + mean
            voice = julius.resample_frac(voice.mean(0), rate, out_rate).numpy().astype(np.float32)
            if carry is not None:
                n = min(len(carry), len(voice), overlap_out)
                voice[:n] = carry[:n] * (1.0 - fade_in[:n]) + voice[:n] * fade_in[:n]
            last = start + len(pcm) // channels >= total
            if last:
                body, carry = voice, None
            else:
                body, carry = voice[:-overlap_out], voice[-overlap_out:]
            dst.writeframes((np.clip(body, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
            position = start + len(pcm) // channels
            proto.progress(100.0 * position / max(1, total), STAGE)
    return {"out": req["out"], "seconds": round(time.monotonic() - started, 2), "device": device,
            "model": req.get("model") or "htdemucs"}


if __name__ == "__main__":
    proto.main(handle)
