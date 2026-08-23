"""Tests for whole-file pair analysis: multi-window offsets and drift detection."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.analyze import analyze_pair, plan_windows  # noqa: E402
from audiosync.media import CancellationToken  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "fixtures", "manifest.json")


def load_manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit("Fixtures missing. Run: python tests/make_fixtures.py")
    with open(MANIFEST, encoding="utf-8") as handle:
        return json.load(handle)


def _case(name):
    for case in load_manifest()["cases"]:
        if case["name"] == name:
            return case
    raise KeyError(name)


def test_window_planning_spreads_across_file():
    positions = plan_windows(duration_s=3600.0, window_s=45.0, count=6)
    assert len(positions) == 6
    assert positions == sorted(positions), "positions must be ordered"
    assert positions[0] > 0, "first window should be inset from the start"
    assert positions[-1] + 45.0 <= 3600.0, "last window must fit inside the file"


def test_window_planning_handles_short_files():
    positions = plan_windows(duration_s=10.0, window_s=45.0, count=6)
    assert len(positions) >= 1
    assert all(p >= 0 for p in positions)


def test_constant_offset_recovered_end_to_end():
    """Full pipeline must recover known offsets with the correct sign."""
    failures = []
    for case in load_manifest()["cases"]:
        if case["kind"] not in ("constant", "level_mismatch"):
            continue
        result = analyze_pair(
            case["primary"], case["secondary"], window_s=8.0, window_count=4
        )
        truth = case["true_offset_ms"]
        if result.error:
            failures.append(f"{case['name']}: {result.error}")
            continue
        error = result.delay_ms - truth
        if abs(error) > 5.0:
            failures.append(
                f"{case['name']}: want {truth:+.1f}ms got {result.delay_ms:+.1f}ms"
            )
    assert not failures, "End-to-end offset failures:\n  " + "\n  ".join(failures)


def test_unrelated_pair_reports_error_not_number():
    case = _case("unrelated")
    result = analyze_pair(case["primary"], case["secondary"], window_s=8.0, window_count=4)
    assert result.delay_ms is None, (
        f"unrelated audio produced {result.delay_ms:+.1f}ms instead of an error"
    )
    assert result.error, "unrelated audio must carry an explanatory error"


def test_drift_is_detected():
    """A speed mismatch must surface as drift, not as a plain offset."""
    case = _case("drift")
    result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
    assert result.error is None, f"drift pair failed: {result.error}"
    assert result.drift_ms_per_s is not None, "drift was not measured"
    assert result.has_significant_drift, (
        f"0.2% speed mismatch not flagged (drift={result.drift_ms_per_s:.4f} ms/s)"
    )


def test_matched_pair_reports_no_significant_drift():
    """A constant-offset pair must not be misreported as drifting."""
    case = _case("offset_500ms")
    result = analyze_pair(case["primary"], case["secondary"], window_s=8.0, window_count=5)
    assert result.error is None
    assert not result.has_significant_drift, (
        f"constant offset misreported as drift ({result.drift_ms_per_s:.4f} ms/s)"
    )


def test_drift_pair_reports_both_midpoint_and_start_offsets():
    """With drift, delayMs is the midpoint value but corrections start at t=0.

    Applying the midpoint offset from the beginning of the file over-shifts by
    half the total drift; a 120s pair drifting 2.9 ms/s left ~150ms of residual
    delay after an otherwise correct fix.
    """
    case = _case("drift")
    result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
    assert result.error is None, f"drift pair failed: {result.error}"
    assert result.delay_at_start_ms is not None, "no start-referenced offset"
    # The two must differ by roughly half the total drift.
    expected_gap = abs(result.drift_ms_per_s) * (result.primary_duration_s or 0) / 2
    actual_gap = abs(result.delay_ms - result.delay_at_start_ms)
    assert actual_gap > 0, "start and midpoint offsets are identical despite drift"
    assert abs(actual_gap - expected_gap) < max(20.0, expected_gap * 0.5), (
        f"gap {actual_gap:.1f}ms does not match half the drift ({expected_gap:.1f}ms)"
    )


def test_no_drift_means_start_equals_midpoint():
    case = _case("offset_500ms")
    result = analyze_pair(case["primary"], case["secondary"], window_s=8.0, window_count=5)
    assert result.error is None
    assert result.delay_at_start_ms == result.delay_ms, (
        "without drift the two offsets must be the same value"
    )


def test_confidence_is_high_for_true_match():
    case = _case("offset_50ms")
    result = analyze_pair(case["primary"], case["secondary"], window_s=8.0, window_count=4)
    assert result.confidence > 0.8, f"confidence too low: {result.confidence:.2f}"


def test_cancellation_stops_analysis():
    case = _case("offset_500ms")
    token = CancellationToken()
    token.cancel()
    try:
        analyze_pair(
            case["primary"], case["secondary"], window_s=8.0, window_count=4, token=token
        )
    except Exception as exc:  # Cancelled propagates out
        assert "cancel" in str(exc).lower(), f"unexpected exception: {exc}"
    else:
        raise AssertionError("cancellation token was ignored")


def test_progress_reaches_completion():
    case = _case("offset_0ms")
    seen = []
    analyze_pair(
        case["primary"], case["secondary"],
        window_s=8.0, window_count=3, progress=seen.append,
    )
    assert seen, "no progress reported"
    assert seen == sorted(seen), f"progress went backwards: {seen}"
    assert seen[-1] == 100, f"progress ended at {seen[-1]}, not 100"
    assert len(set(seen)) > 2, f"progress was not granular: {sorted(set(seen))}"


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
