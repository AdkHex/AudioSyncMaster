"""Generate synthetic audio fixtures with exactly known sync offsets.

These are the ground truth for every algorithm change. Each fixture is a pair of
WAV files plus a manifest entry stating the true offset in milliseconds, using
the convention documented in audiosync.correlate:

    positive delay  =>  the secondary track starts LATER than the primary
                        (secondary must be moved EARLIER to align)

Run:  python tests/make_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixtures")
MANIFEST = os.path.join(FIXTURE_DIR, "manifest.json")

SR = 16000
DURATION_S = 30.0


def _speechlike(seconds: float, sr: int, seed: int) -> np.ndarray:
    """Synthesize a broadband signal with speech-like bursts and silences.

    Pure noise correlates too perfectly to be a realistic test, and a pure tone
    is ambiguous under shift. Amplitude-modulated filtered noise gives a signal
    with genuine transients to lock onto, like dialogue.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    noise = rng.standard_normal(n)

    # Smooth the noise into a band-limited signal (cheap low-pass via cumulative
    # moving average) so it resembles voiced audio rather than white hiss.
    kernel = np.ones(24) / 24.0
    voiced = np.convolve(noise, kernel, mode="same")

    # Amplitude envelope: bursts of "speech" separated by near-silence.
    envelope = np.zeros(n)
    pos = 0
    while pos < n:
        burst = int(rng.uniform(0.25, 0.9) * sr)
        gap = int(rng.uniform(0.1, 0.5) * sr)
        end = min(n, pos + burst)
        envelope[pos:end] = rng.uniform(0.4, 1.0)
        pos = end + gap

    # Soften envelope edges so transients are sharp but not synthetic clicks.
    envelope = np.convolve(envelope, np.ones(128) / 128.0, mode="same")
    signal = voiced * envelope
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    return signal.astype(np.float32)


def _shift(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Return signal delayed by delay_samples (positive => starts later)."""
    if delay_samples == 0:
        return signal.copy()
    if delay_samples > 0:
        return np.concatenate([np.zeros(delay_samples, dtype=signal.dtype), signal])
    return signal[-delay_samples:].copy()


def _resample_linear(signal: np.ndarray, ratio: float) -> np.ndarray:
    """Resample by a ratio to simulate playback-speed drift (e.g. PAL speedup)."""
    n_out = int(len(signal) * ratio)
    src_idx = np.linspace(0, len(signal) - 1, n_out)
    return np.interp(src_idx, np.arange(len(signal)), signal).astype(np.float32)


def _write(name: str, signal: np.ndarray) -> str:
    path = os.path.join(FIXTURE_DIR, name)
    sf.write(path, signal, SR)
    return path


def build() -> list[dict]:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    cases: list[dict] = []

    base = _speechlike(DURATION_S, SR, seed=1)

    # --- Constant-offset cases across the range that matters in practice ---
    for offset_ms in (0.0, 50.0, -50.0, 500.0, -500.0, 5000.0):
        delay_samples = int(round(offset_ms / 1000.0 * SR))
        primary = base
        secondary = _shift(base, delay_samples)
        tag = f"offset_{str(offset_ms).replace('-', 'neg').replace('.0', '')}ms"
        cases.append(
            {
                "name": tag,
                "primary": _write(f"{tag}_primary.wav", primary),
                "secondary": _write(f"{tag}_secondary.wav", secondary),
                "true_offset_ms": offset_ms,
                "expect_match": True,
                "kind": "constant",
                "tolerance_ms": 2.0,
            }
        )

    # --- Independent audio: must be REJECTED, not answered confidently ---
    other = _speechlike(DURATION_S, SR, seed=999)
    cases.append(
        {
            "name": "unrelated",
            "primary": _write("unrelated_primary.wav", base),
            "secondary": _write("unrelated_secondary.wav", other),
            "true_offset_ms": None,
            "expect_match": False,
            "kind": "unrelated",
            "tolerance_ms": None,
        }
    )

    # --- Silent lead-in: the correct answer is still recoverable further in ---
    silent_lead = np.concatenate(
        [np.zeros(int(8.0 * SR), dtype=np.float32), base[: int(22.0 * SR)]]
    )
    cases.append(
        {
            "name": "silent_intro",
            "primary": _write("silent_intro_primary.wav", silent_lead),
            "secondary": _write(
                "silent_intro_secondary.wav", _shift(silent_lead, int(0.25 * SR))
            ),
            "true_offset_ms": 250.0,
            "expect_match": True,
            "kind": "silent_intro",
            "tolerance_ms": 2.0,
        }
    )

    # --- Drift: secondary runs slightly fast, as with a 25 vs 23.976fps dub ---
    drift_ratio = 1.0 - (1.0 / 500.0)  # secondary ~0.2% shorter => grows apart
    drifted = _resample_linear(base, drift_ratio)
    cases.append(
        {
            "name": "drift",
            "primary": _write("drift_primary.wav", base),
            "secondary": _write("drift_secondary.wav", drifted),
            "true_offset_ms": 0.0,
            "expect_match": True,
            "kind": "drift",
            # Start aligns; the end diverges by ~0.2% of duration.
            "expect_drift": True,
            "tolerance_ms": 20.0,
        }
    )

    # --- Local cut: a chunk exists in one track but not the other, partway
    # through an otherwise-constant offset. This must not be averaged away
    # into a slightly-wrong single number; it must be reported as a cut. ---
    splice_at_s = 20.0
    splice_len_s = 2.0
    lead_offset_ms = 300.0
    lead_samples = int(round(lead_offset_ms / 1000.0 * SR))
    splice_at_sample = int(splice_at_s * SR)
    splice_len_samples = int(splice_len_s * SR)
    inserted = _speechlike(splice_len_s, SR, seed=42)
    secondary_with_cut = np.concatenate(
        [
            base[:splice_at_sample],
            inserted,
            base[splice_at_sample:],
        ]
    )
    cases.append(
        {
            "name": "local_cut",
            "primary": _write("local_cut_primary.wav", base),
            "secondary": _write(
                "local_cut_secondary.wav", _shift(secondary_with_cut, lead_samples)
            ),
            "true_offset_ms": lead_offset_ms,
            "expect_match": True,
            "kind": "local_cut",
            "tolerance_ms": 20.0,
            "cut_position_s": splice_at_s,
            "cut_magnitude_ms": splice_len_s * 1000.0,
        }
    )

    # --- Minor cut: one frame of content, the smallest splice that can exist.
    # Large cuts are easy; this is the size the reconciliation has to survive,
    # because it is small enough to fit a gentle slope through and be reported
    # as a frame-rate conversion instead. ---
    one_frame_s = 1.0 / (24000.0 / 1001.0)
    frame_samples = int(round(one_frame_s * SR))
    secondary_one_frame = np.concatenate(
        [
            base[:splice_at_sample],
            _speechlike(one_frame_s, SR, seed=77)[:frame_samples],
            base[splice_at_sample:],
        ]
    )
    cases.append(
        {
            "name": "minor_cut",
            "primary": _write("minor_cut_primary.wav", base),
            "secondary": _write(
                "minor_cut_secondary.wav", _shift(secondary_one_frame, lead_samples)
            ),
            "true_offset_ms": lead_offset_ms,
            "expect_match": True,
            "kind": "local_cut",
            "tolerance_ms": 20.0,
            "cut_position_s": splice_at_s,
            "cut_magnitude_ms": frame_samples / SR * 1000.0,
        }
    )

    # --- Frame-rate conversions: the audio is the same content timed against a
    # different rate, which is what a dub laid on the wrong master is. The PAL
    # pair is the one that matters -- at 4.27% the offset moves further inside
    # a single window than the correlation can resolve, so the speed has to be
    # taken off before the pair can be measured at all. ---
    video_fps = 24000.0 / 1001.0
    for audio_fps, tag in ((25.0, "rate_pal"), (24.0, "rate_film")):
        # Content of N frames occupies N/fps seconds, so audio timed at a
        # higher rate runs correspondingly short against the same video.
        timed = _resample_linear(base, video_fps / audio_fps)
        cases.append(
            {
                "name": tag,
                "primary": _write(f"{tag}_primary.wav", base),
                "secondary": _write(f"{tag}_secondary.wav", _shift(timed, lead_samples)),
                "true_offset_ms": lead_offset_ms,
                "expect_match": True,
                "kind": "rate_change",
                "tolerance_ms": 20.0,
                "audio_fps": audio_fps,
                "video_fps": video_fps,
                # How fast the offset moves: a window at video time t finds its
                # content at video_fps/audio_fps of the way in.
                "drift_ms_per_s": (video_fps / audio_fps - 1.0) * 1000.0,
            }
        )

    # --- Quiet secondary: correlation must survive a large level difference ---
    cases.append(
        {
            "name": "quiet_secondary",
            "primary": _write("quiet_primary.wav", base),
            "secondary": _write(
                "quiet_secondary.wav", _shift(base, int(0.1 * SR)) * 0.02
            ),
            "true_offset_ms": 100.0,
            "expect_match": True,
            "kind": "level_mismatch",
            "tolerance_ms": 2.0,
        }
    )

    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump({"sample_rate": SR, "cases": cases}, handle, indent=2)

    return cases


if __name__ == "__main__":
    built = build()
    print(f"Wrote {len(built)} fixture pairs to {FIXTURE_DIR}")
    for case in built:
        truth = case["true_offset_ms"]
        label = "no match expected" if truth is None else f"{truth:+.1f} ms"
        print(f"  {case['name']:24s} {label}")
    sys.exit(0)
