"""Tests for telling a splice apart from a speed mismatch.

The measurements here are written out by hand rather than decoded, because the
question these answer is not "can the correlator find the audio" -- that is
test_correlate's job -- but "given these six numbers, is this file one timeline
or two". Every case is a set of window offsets a real file would produce.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.segments import MIN_STEP_MS, find_step  # noqa: E402

# Six windows across a 45-minute episode, as plan_windows would place them.
POSITIONS = [54.0, 594.0, 1134.0, 1674.0, 2214.0, 2754.0]
WINDOW_S = 45.0


def _offsets(base: float, cut_ms: float, at: int, drift_ms_per_s: float = 0.0):
    return [
        base + drift_ms_per_s * position + (cut_ms if index >= at else 0.0)
        for index, position in enumerate(POSITIONS)
    ]


def test_a_step_is_found_where_it_is():
    step = find_step(POSITIONS, _offsets(300.0, 2000.0, at=4), WINDOW_S)
    assert step is not None, "a 2-second jump was not found"
    assert step.before_ms == 300.0 and step.after_ms == 2300.0
    assert step.magnitude_ms == 2000.0
    # The cut is between the last window that measured 300 and the end of the
    # first that measured 2300 -- no tighter, until something probes it.
    assert step.earliest_s == POSITIONS[3]
    assert step.latest_s == POSITIONS[4] + WINDOW_S


def test_the_delay_before_the_cut_is_not_an_average_of_both_halves():
    """The failure this whole module exists to stop.

    Reconciled as one file, six windows measuring 300/300/300/300/2300/2300
    produce a median of 1300ms: a delay that is 1000ms wrong for the first half
    and 1000ms wrong for the second, reported with high confidence because the
    windows themselves each looked perfect.
    """
    step = find_step(POSITIONS, _offsets(300.0, 2000.0, at=4), WINDOW_S)
    assert step is not None
    assert step.before_ms == 300.0, "the pre-cut level is the correctable one"
    assert step.before_ms != 1300.0


def test_steady_drift_is_not_a_cut():
    """A PAL speedup slides the offset by 42.7ms every second. It is a line."""
    for drift in (0.06, 0.5, 1.0, 4.27, 42.7):
        offsets = _offsets(300.0, 0.0, at=0, drift_ms_per_s=drift)
        assert find_step(POSITIONS, offsets, WINDOW_S) is None, (
            f"{drift} ms/s of steady drift was reported as a splice"
        )


def test_a_flat_file_has_no_step():
    assert find_step(POSITIONS, [300.0] * 6, WINDOW_S) is None


def test_a_step_smaller_than_the_bar_is_left_alone():
    """Under half a frame is not worth stopping a mux for."""
    offsets = _offsets(300.0, MIN_STEP_MS / 2.0, at=3)
    assert find_step(POSITIONS, offsets, WINDOW_S) is None


def test_a_step_must_clear_the_noise_as_well_as_the_bar():
    """40ms is over the absolute bar, and nothing at all on windows that
    already disagree by that much among themselves."""
    noisy = [300.0, 260.0, 340.0, 300.0 + 40.0, 260.0 + 40.0, 340.0 + 40.0]
    assert find_step(POSITIONS, noisy, WINDOW_S) is None


def test_one_frame_is_found_on_windows_that_agree():
    """The smallest splice that can exist, on material measured cleanly."""
    one_frame = 1000.0 / (24000.0 / 1001.0)
    offsets = _offsets(300.0, one_frame, at=3)
    step = find_step(POSITIONS, offsets, WINDOW_S)
    assert step is not None, f"a one-frame cut ({one_frame:.1f}ms) was missed"
    assert abs(step.magnitude_ms - one_frame) < 0.01


def test_a_cut_resting_on_one_window_is_marked_for_corroboration():
    """A lone deviating window is also what a bad correlation looks like."""
    lone = find_step(POSITIONS, _offsets(300.0, 2000.0, at=5), WINDOW_S)
    assert lone is not None and lone.lone_group

    supported = find_step(POSITIONS, _offsets(300.0, 2000.0, at=3), WINDOW_S)
    assert supported is not None and not supported.lone_group


def test_too_few_windows_to_ask_the_question():
    """With three points a split always fits two of them exactly."""
    assert find_step(POSITIONS[:3], [300.0, 300.0, 2300.0], WINDOW_S) is None


def test_the_position_is_the_middle_of_what_is_still_possible():
    step = find_step(POSITIONS, _offsets(300.0, 2000.0, at=4), WINDOW_S)
    assert step is not None
    assert step.earliest_s <= step.position_s <= step.latest_s
    assert abs(step.uncertainty_s - (step.latest_s - step.earliest_s) / 2.0) < 1e-9


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
