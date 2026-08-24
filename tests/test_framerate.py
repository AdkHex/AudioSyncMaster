"""Tests for explaining drift as a frame-rate mismatch."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.framerate import MAX_RATE_DRIFT_MS_PER_S, diagnose  # noqa: E402
from audiosync.media import _parse_frame_rate  # noqa: E402


def _drift_for(source_fps: float, target_fps: float) -> float:
    """Drift a source->target conversion produces, in ms per second."""
    return (1.0 - target_fps / source_fps) * 1000.0


def test_pal_speedup_is_named():
    """25fps audio against a 23.976fps video is the classic dub problem."""
    drift = _drift_for(23.976, 25.0)
    result = diagnose(drift, primary_fps=23.976, secondary_fps=25.0)
    assert result is not None and result.is_rate_mismatch, (
        f"PAL speedup not identified (drift={drift:.3f} ms/s)"
    )
    assert not result.is_likely_cut
    assert "23.976" in result.explanation and "25" in result.explanation


def test_correction_ratio_cancels_the_drift():
    """Applying the suggested ratio must remove the measured drift."""
    for source, target in ((23.976, 25.0), (25.0, 23.976), (24.0, 23.976)):
        drift = _drift_for(source, target)
        result = diagnose(drift, primary_fps=source, secondary_fps=target)
        assert result is not None and result.correction_ratio, (
            f"{source}->{target} produced no correction"
        )
        # Correcting then re-measuring should leave essentially nothing.
        residual = result.speed_ratio * result.correction_ratio
        assert abs(residual - 1.0) < 1e-9, (
            f"{source}->{target}: correction leaves {residual:.9f}, want 1.0"
        )


def test_rates_are_identified_without_metadata():
    """A rate pair should be recoverable from the drift alone."""
    drift = _drift_for(23.976, 25.0)
    result = diagnose(drift)
    assert result is not None and result.is_rate_mismatch
    assert result.source_fps and result.target_fps


def test_extreme_drift_is_reported_as_a_cut():
    """Runtimes diverging this fast are different edits, not a rate issue."""
    result = diagnose(MAX_RATE_DRIFT_MS_PER_S * 2)
    assert result is not None
    assert result.is_likely_cut, "large drift not flagged as a probable cut"
    assert not result.is_rate_mismatch
    assert "cut" in result.explanation.lower()


def test_small_drift_is_not_called_a_rate_mismatch():
    """Measurement noise must not be dressed up as a frame-rate conversion."""
    result = diagnose(0.004)
    assert result is not None
    assert not result.is_likely_cut
    assert not result.is_rate_mismatch, (
        f"noise misreported as a rate mismatch: {result.explanation}"
    )


def test_no_drift_returns_a_benign_diagnosis():
    result = diagnose(0.0)
    assert result is not None and not result.is_rate_mismatch and not result.is_likely_cut


def test_missing_drift_returns_nothing():
    assert diagnose(None) is None


def test_frame_rate_parsing_is_safe_and_correct():
    """ffprobe rates are parsed, never eval'd: this string comes from a file."""
    assert abs(_parse_frame_rate("24000/1001") - 23.976) < 0.001
    assert _parse_frame_rate("25/1") == 25.0
    assert _parse_frame_rate("30") == 30.0
    assert _parse_frame_rate("0/0") is None
    assert _parse_frame_rate("N/A") is None
    assert _parse_frame_rate(None) is None
    assert _parse_frame_rate("1/0") is None, "division by zero not handled"
    # Anything that would execute under eval must simply fail to parse.
    assert _parse_frame_rate("__import__('os').system('true')") is None
    assert _parse_frame_rate("999999") is None, "implausible rate accepted"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
