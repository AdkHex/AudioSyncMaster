"""The voice tools' worker: speech detection and voice separation.

This file does not run in the app's engine. It runs under the Python the
voice tools install (see ``voicetools``), which has PyTorch, Demucs and
Silero VAD; the engine starts it as a child process and talks to it one JSON
line at a time on stdin/stdout. Audio goes both ways as ``.npy`` files, so
the engine keeps every read of a media file -- and with it the sample
exactness the renderer depends on -- and this side only computes.

Commands (one JSON object per line, one reply per line):

* ``{"cmd": "hello"}`` -- versions and the device in use.
* ``{"cmd": "warmup"}`` -- load both models (downloading Demucs' weights
  the first time).
* ``{"cmd": "speech", "input": X, "output": Y}`` -- X: mono float32 at
  16 kHz. Y: Silero's speech probability, SPEECH_RATE values a second;
  value ``k`` describes the 32 ms starting at sample ``128 * k`` (four
  interleaved grids of 32 ms, so a line's start is placed to 8 ms).
* ``{"cmd": "vocals", "input": X, "rate": R, "output": Y}`` -- X:
  float32 ``(n, channels)`` at R Hz, 1 or 2 channels. Y: the voices in it,
  same shape, same clock.
* ``{"cmd": "quit"}``.

Every reply has ``ok``; a failure carries ``error`` and the worker carries
on with the next command.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

SPEECH_SR = 16000
SPEECH_CHUNK = 512
SPEECH_RATE = 125  # four 32 ms grids offset by 8 ms
DEMUCS_SR = 44100


class Tools:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        threads = int(os.environ.get("AUDIOSYNC_THREADS") or 0) or (os.cpu_count() or 4)
        torch.set_num_threads(max(1, threads))
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_name = os.environ.get("AUDIOSYNC_DEMUCS_MODEL") or "htdemucs"
        self._separator = None
        self._vad = None

    def separator(self):
        if self._separator is None:
            from demucs.pretrained import get_model

            model = get_model(self.model_name)
            model.eval()
            self._separator = model
        return self._separator

    def vad(self):
        if self._vad is None:
            from silero_vad import load_silero_vad

            self._vad = load_silero_vad()
        return self._vad

    def speech(self, audio: np.ndarray) -> np.ndarray:
        """Speech probability on four 32 ms grids, 8 ms apart."""
        torch = self.torch
        model = self.vad()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        count = max(0, (len(audio) - SPEECH_CHUNK) // SPEECH_CHUNK)
        out = np.zeros(4 * count, dtype=np.float32)
        for phase in range(4):
            model.reset_states()
            start = phase * SPEECH_CHUNK // 4
            frames = torch.from_numpy(audio[start: start + count * SPEECH_CHUNK].copy())
            for i in range(count - 1 if phase else count):
                out[4 * i + phase] = float(model(frames[i * SPEECH_CHUNK:(i + 1) * SPEECH_CHUNK], SPEECH_SR))
        return out

    def vocals(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """The voices of ``audio`` ((n, channels), 1 or 2 channels) on its own clock."""
        torch = self.torch
        import julius
        from demucs.apply import apply_model

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        n, channels = audio.shape
        stereo = audio if channels == 2 else np.repeat(audio[:, :1], 2, axis=1)
        wav = torch.from_numpy(stereo.T.copy())
        if rate != DEMUCS_SR:
            wav = julius.resample_frac(wav, rate, DEMUCS_SR)
        model = self.separator()
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        sources = apply_model(model, ((wav - mean) / std)[None], device=self.device,
                              split=True, overlap=0.25, progress=False)[0]
        voice = sources[model.sources.index("vocals")].cpu() * std
        if rate != DEMUCS_SR:
            voice = julius.resample_frac(voice, DEMUCS_SR, rate)
        voice = voice.numpy().T
        if len(voice) < n:
            voice = np.concatenate([voice, np.zeros((n - len(voice), 2), np.float32)])
        voice = voice[:n]
        if channels == 1:
            voice = voice.mean(axis=1, keepdims=True)
        return voice.astype(np.float32)


def main() -> int:
    tools = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "quit":
                print(json.dumps({"ok": True}), flush=True)
                return 0
            if tools is None:
                tools = Tools()
            if command == "hello":
                import demucs
                import torch

                reply = {"ok": True, "device": tools.device, "torch": torch.__version__,
                         "demucs": getattr(demucs, "__version__", "?"), "model": tools.model_name,
                         "threads": torch.get_num_threads()}
            elif command == "warmup":
                tools.separator()
                tools.vad()
                reply = {"ok": True, "device": tools.device}
            elif command == "speech":
                curve = tools.speech(np.load(request["input"]))
                np.save(request["output"], curve)
                reply = {"ok": True, "count": int(len(curve))}
            elif command == "vocals":
                voice = tools.vocals(np.load(request["input"]), int(request["rate"]))
                np.save(request["output"], voice)
                reply = {"ok": True, "count": int(len(voice))}
            else:
                reply = {"ok": False, "error": f"unknown command {command!r}"}
        except Exception as exc:  # noqa: BLE001 - one bad request must not end the worker
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
