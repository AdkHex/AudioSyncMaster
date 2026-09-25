"""Speech in a soundtrack, in any language: where someone is talking.

Two curves, a hundred values a second, from a mono mix:

* the level of the voice band (300-3400 Hz) in dB -- where speech carries
  its energy, whatever the language;
* the voicing of each 40 ms frame (0-1): how periodic it is at a speaking
  pitch (70-400 Hz), which a voice is and most effects are not.

Their product, gated a little above the quietest fifth of the span, is the
speech activity used by the lip check (see ``lipcheck``) and to compare two
languages' line timing. It is a detector of voiced sound, not of words:
sung or hummed music counts, which is why every use of it measures a lag
rather than trusting a level. Measured on a real film's Japanese and
English centre channels, two-minute windows of it agreed on the tracks'
offset to within one frame 90-100% of the time.
"""

from __future__ import annotations

import subprocess
from typing import Optional, Tuple

import numpy as np

from .media import CancellationToken, MediaError, _popen_kwargs, ffmpeg_path

SPEECH_RATE = 100
"""Values per second of every curve here."""
# The rate a file is decoded at for them.
ANALYSIS_SR = 16000
VOICE_BAND = (300.0, 3400.0)
PITCH_RANGE = (70.0, 400.0)
FRAME_S = 0.040
# The gate opens this far above the quietest fifth of the span, fully
# this much further up.
GATE_ABOVE_DB = 6.0
GATE_RAMP_DB = 12.0


def voice_curves(samples: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """(voice-band level in dB, voicing 0-1), SPEECH_RATE values a second;
    value ``i`` describes the frame centred on ``i / SPEECH_RATE`` seconds."""
    samples = np.asarray(samples, dtype=np.float64)
    hop = max(1, int(round(sr / SPEECH_RATE)))
    window = max(hop, int(round(FRAME_S * sr)))
    count = len(samples) // hop
    if count == 0:
        return np.zeros(0), np.zeros(0)
    padded = np.concatenate([np.zeros(window // 2), samples, np.zeros(window)])
    index = np.arange(window)[None, :] + hop * np.arange(count)[:, None]
    frames = padded[index] * np.hanning(window)[None, :]
    size = 1 << int(np.ceil(np.log2(2 * window)))
    power = np.abs(np.fft.rfft(frames, size, axis=1)) ** 2
    freqs = np.fft.rfftfreq(size, 1.0 / sr)
    band = power[:, (freqs >= VOICE_BAND[0]) & (freqs <= VOICE_BAND[1])].sum(axis=1)
    level = 10.0 * np.log10(band + 1e-12)
    correlation = np.fft.irfft(power, size, axis=1)
    lo, hi = int(sr / PITCH_RANGE[1]), min(int(sr / PITCH_RANGE[0]), size // 2 - 1)
    energy = np.maximum(correlation[:, :1], 1e-12)
    voicing = np.clip((correlation[:, lo : hi + 1] / energy).max(axis=1), 0.0, 1.0)
    voicing[band < 1e-10] = 0.0
    return level, voicing


def speech_activity(samples: np.ndarray, sr: int) -> np.ndarray:
    """How much each hundredth of a second sounds like someone talking, 0-1."""
    level, voicing = voice_curves(samples, sr)
    if not len(level):
        return level
    floor = float(np.percentile(level, 20))
    gate = np.clip((level - floor - GATE_ABOVE_DB) / GATE_RAMP_DB, 0.0, 1.0)
    activity = voicing * gate
    return np.convolve(activity, np.ones(3) / 3.0, mode="same")


def speech_energy(samples: np.ndarray, sr: int) -> np.ndarray:
    """The voice band's level where the frame is voiced, in dB above the
    span's quiet: what a mouth's opening follows, as near as sound says."""
    level, voicing = voice_curves(samples, sr)
    if not len(level):
        return level
    floor = float(np.percentile(level, 20))
    return np.clip(level - floor, 0.0, None) * np.clip(voicing * 1.5, 0.0, 1.0)


def centre(left: np.ndarray, right: np.ndarray, frame: int = 1024) -> np.ndarray:
    """What of a stereo mix sits in the centre, where dialogue is mixed:
    per frequency, the mid's magnitude less what the difference carries,
    with the mid's phase. Music and effects spread across the stereo field
    fall away; a voice panned to the middle stays."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n = min(len(left), len(right))
    hop = frame // 4
    window = np.hanning(frame)
    out = np.zeros(n + frame)
    norm = np.zeros(n + frame)
    for start in range(0, max(1, n - frame), hop):
        a = np.fft.rfft(left[start : start + frame] * window, frame)
        b = np.fft.rfft(right[start : start + frame] * window, frame)
        mid, side = (a + b) / 2.0, (a - b) / 2.0
        magnitude = np.maximum(np.abs(mid) - 1.5 * np.abs(side), 0.0)
        out[start : start + frame] += np.fft.irfft(magnitude * np.exp(1j * np.angle(mid)), frame) * window
        norm[start : start + frame] += window ** 2
    return (out / np.maximum(norm, 1e-9))[:n]


def centre_activity(
    path: str, track: int, lo_s: float, hi_s: float,
    token: Optional[CancellationToken] = None, pad: bool = True,
) -> np.ndarray:
    """Speech activity of a file's centre across [lo_s, hi_s), SPEECH_RATE
    values a second. With ``pad``, value ``i`` describes ``lo_s + i /
    SPEECH_RATE`` however much of the span the file covers: what lies
    before its start or past its end is silence. Without, the read starts
    at the file's start when ``lo_s`` is before it and ends where it ends."""
    start = max(0.0, lo_s)
    command = [
        ffmpeg_path(), "-nostdin", "-v", "error", "-ss", f"{start:.6f}", "-i", path, "-t", f"{max(0.0, hi_s - start):.6f}",
        "-map", f"0:a:{max(0, track)}", "-vn", "-sn", "-dn", "-ac", "2", "-ar", str(ANALYSIS_SR),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    process = subprocess.run(command, capture_output=True, **{k: v for k, v in _popen_kwargs().items() if k not in ("stdout", "stderr")})
    if process.returncode != 0:
        raise MediaError(f"could not read {path}: {process.stderr.decode(errors='replace').strip()[-200:]}")
    if token:
        token.raise_if_cancelled()
    pcm = np.frombuffer(process.stdout, dtype=np.float32).reshape(-1, 2).astype(np.float64)
    activity = speech_activity(centre(pcm[:, 0], pcm[:, 1]), ANALYSIS_SR)
    if not pad:
        return activity
    front = int(round((start - lo_s) * SPEECH_RATE))
    want = int(round((hi_s - lo_s) * SPEECH_RATE))
    activity = np.concatenate([np.zeros(front), activity])[:want]
    return np.concatenate([activity, np.zeros(max(0, want - len(activity)))])
