"""Ground-truth tests for offset estimation.

These lock the sign convention and the reject-unrelated-audio behaviour. Both
were broken in the original implementation: every non-zero offset came back
negated, and unrelated audio produced a confident wrong answer.

Run:  python -m pytest tests/ -v      (or)      python tests/test_correlate.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.correlate import estimate_offset, onset_envelope  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "fixtures", "manifest.json")


def load_manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit("Fixtures missing. Run: python tests/make_fixtures.py")
    with open(MANIFEST, encoding="utf-8") as handle:
        return json.load(handle)


def _read(path):
    signal, sr = sf.read(path, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal, sr


def test_constant_offsets():
    """Every known offset must be recovered with the correct sign and magnitude."""
    manifest = load_manifest()
    failures = []
    for case in manifest["cases"]:
        if case["kind"] not in ("constant", "silent_intro", "level_mismatch"):
            continue
        primary, sr = _read(case["primary"])
        secondary, _ = _read(case["secondary"])
        result = estimate_offset(primary, secondary, sr, max_offset_ms=30000)

        truth = case["true_offset_ms"]
        tolerance = case["tolerance_ms"]
        if not result.matched:
            failures.append(f"{case['name']}: rejected ({result.reason})")
            continue
        error = result.delay_ms - truth
        if abs(error) > tolerance:
            failures.append(
                f"{case['name']}: expected {truth:+.1f}ms got "
                f"{result.delay_ms:+.1f}ms (error {error:+.1f}ms)"
            )
    assert not failures, "Offset mismatches:\n  " + "\n  ".join(failures)


def test_unrelated_audio_is_rejected():
    """Independent audio must be reported as no-match, never as a number."""
    manifest = load_manifest()
    cases = [c for c in manifest["cases"] if not c["expect_match"]]
    assert cases, "manifest has no negative case"
    for case in cases:
        primary, sr = _read(case["primary"])
        secondary, _ = _read(case["secondary"])
        result = estimate_offset(primary, secondary, sr, max_offset_ms=30000)
        assert not result.matched, (
            f"{case['name']}: unrelated audio produced a confident "
            f"{result.delay_ms:+.1f}ms (ratio {result.peak_ratio:.1f})"
        )


def test_sign_convention_is_documented_direction():
    """Positive delay must mean the secondary starts LATER than the primary."""
    sr = 16000
    rng = np.random.default_rng(7)
    base = np.convolve(rng.standard_normal(sr * 8), np.ones(20) / 20, mode="same")
    base = (base / np.max(np.abs(base)) * 0.7).astype(np.float32)

    delay_samples = int(0.3 * sr)
    secondary = np.concatenate(
        [np.zeros(delay_samples, dtype=np.float32), base]
    )  # secondary starts 300ms LATER

    result = estimate_offset(base, secondary, sr, max_offset_ms=5000)
    assert result.matched, f"clean synthetic pair rejected: {result.reason}"
    assert result.delay_ms > 0, (
        f"secondary starts later so delay must be POSITIVE, got {result.delay_ms:+.1f}ms"
    )
    assert abs(result.delay_ms - 300.0) < 3.0, f"got {result.delay_ms:+.1f}ms, want +300ms"


def test_silence_is_rejected_not_answered():
    """A silent segment must be reported as unusable rather than guessed at."""
    sr = 16000
    silence = np.zeros(sr * 4, dtype=np.float32)
    rng = np.random.default_rng(3)
    real = (rng.standard_normal(sr * 4) * 0.3).astype(np.float32)

    result = estimate_offset(silence, real, sr)
    assert not result.matched
    assert result.reason is not None


def test_confidence_separates_match_from_noise():
    """Confidence must be high for a true match and low for unrelated audio."""
    manifest = load_manifest()
    matched_scores, unmatched_scores = [], []
    for case in manifest["cases"]:
        primary, sr = _read(case["primary"])
        secondary, _ = _read(case["secondary"])
        result = estimate_offset(primary, secondary, sr, max_offset_ms=30000)
        (matched_scores if case["expect_match"] else unmatched_scores).append(
            result.peak_ratio
        )
    assert min(matched_scores) > max(unmatched_scores), (
        f"confidence does not separate: matched min={min(matched_scores):.1f} "
        f"unmatched max={max(unmatched_scores):.1f}"
    )


def test_envelope_is_shift_equivariant():
    """Shifting audio must shift its envelope correspondingly, not reshape it."""
    sr = 16000
    rng = np.random.default_rng(11)
    signal = (rng.standard_normal(sr * 4) * 0.4).astype(np.float32)
    env_a = onset_envelope(signal)
    env_b = onset_envelope(np.concatenate([np.zeros(320, dtype=np.float32), signal]))
    overlap = min(len(env_a), len(env_b) - 10)
    correlation = np.corrcoef(env_a[:overlap], env_b[10 : 10 + overlap])[0, 1]
    assert correlation > 0.9, f"envelope not shift-stable (r={correlation:.2f})"


def test_does_not_mutate_inputs():
    """Estimation must not modify the caller's arrays."""
    sr = 16000
    rng = np.random.default_rng(5)
    primary = (rng.standard_normal(sr * 3) * 0.4).astype(np.float32)
    secondary = primary.copy()
    before_p, before_s = primary.copy(), secondary.copy()

    estimate_offset(primary, secondary, sr)

    assert np.array_equal(primary, before_p), "primary was mutated"
    assert np.array_equal(secondary, before_s), "secondary was mutated"


def _run_all():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
