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


def test_no_drift_means_start_agrees_with_midpoint():
    """Without drift the two offsets describe the same thing, so they must agree.

    Not to the bit: delay_ms is the median of the windows and delay_at_start_ms
    is now a fitted intercept, so on a constant-offset pair they are two
    estimators of one number and differ by measurement noise. Requiring them to
    be identical is what made the intercept conditional on the drift threshold
    in the first place, and that is the bug below.
    """
    case = _case("offset_500ms")
    result = analyze_pair(case["primary"], case["secondary"], window_s=8.0, window_count=5)
    assert result.error is None
    assert abs(result.delay_at_start_ms - result.delay_ms) < 5.0, (
        f"without drift the two offsets must agree: "
        f"{result.delay_at_start_ms:.2f} vs {result.delay_ms:.2f}"
    )


def test_start_offset_ignores_drift_too_small_to_report():
    """Sub-threshold drift must not leak into the offset used for corrections.

    DRIFT_SIGNIFICANT_MS_PER_S decides whether drift is worth reporting, not
    whether it exists. While delay_at_start_ms was gated on it, anything below
    the threshold was reported as the median across the file -- the value at its
    middle -- and the error was the drift times half the duration. On a
    40-minute episode drifting a tenth of the threshold that is 6ms, which is
    what this reconstructs: windows lying exactly on a known line, so the offset
    at t=0 is known to the millisecond.
    """
    from audiosync.analyze import (  # noqa: PLC0415
        DRIFT_SIGNIFICANT_MS_PER_S,
        PairResult,
        WindowResult,
        _reconcile,
    )
    from audiosync.correlate import OffsetEstimate  # noqa: PLC0415

    duration_s = 2400.0
    true_at_zero = 4785.0
    drift = DRIFT_SIGNIFICANT_MS_PER_S / 10.0

    positions = plan_windows(duration_s, 45.0, 6)
    result = PairResult("primary.mkv", "secondary.eac3")
    result.primary_duration_s = duration_s
    for position in positions:
        offset = true_at_zero + drift * position
        result.windows.append(WindowResult(position, OffsetEstimate(offset, 0.9, 500.0)))

    _reconcile(result)

    assert not result.has_significant_drift, "this drift is deliberately below the bar"
    assert abs(result.delay_at_start_ms - true_at_zero) < 0.5, (
        f"offset at t=0 is {result.delay_at_start_ms:.1f}ms, should be "
        f"{true_at_zero:.1f}ms -- the mid-file value has leaked into it"
    )


def test_a_cut_is_reported_as_a_cut():
    """The reported failure: a splice partway through an otherwise clean pair.

    Before this was reconciled in two segments, the four pre-cut windows agreed
    perfectly, that agreement made the tolerance collapse to its floor, and the
    two post-cut windows were discarded as outliers. What survived was a flat
    fit across half the file, reported as "No meaningful drift; a single delay
    aligns the whole file" at confidence 0.99 -- with the second half two
    seconds out.
    """
    case = _case("local_cut")
    result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
    assert result.error is None, f"local_cut failed: {result.error}"
    assert result.is_likely_cut, (
        f"a {case['cut_magnitude_ms']:.0f}ms splice was not flagged "
        f"(delay={result.delay_ms:.1f}ms, confidence={result.confidence:.3f})"
    )
    assert result.cut is not None
    assert abs(result.cut.magnitude_ms - case["cut_magnitude_ms"]) < 20.0, (
        f"jump measured {result.cut.magnitude_ms:.1f}ms, truth "
        f"{case['cut_magnitude_ms']:.1f}ms"
    )


def test_the_delay_reported_for_a_cut_file_is_the_one_before_the_cut():
    """A correction is applied from t=0, so it can only serve the first segment.

    The median across every window is the average of two different files: 1300ms
    where the truth is 300 before the cut and 2300 after, a number that is wrong
    by a second in both directions and belongs to neither half.
    """
    case = _case("local_cut")
    result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
    assert result.error is None
    truth = case["true_offset_ms"]
    assert abs(result.delay_ms - truth) < 20.0, (
        f"delay {result.delay_ms:.1f}ms is not the pre-cut offset {truth:.1f}ms"
    )
    assert abs(result.delay_at_start_ms - truth) < 20.0, (
        f"the t=0 offset a correction uses is {result.delay_at_start_ms:.1f}ms"
    )


def test_the_cut_is_located_and_not_merely_flagged():
    """"Somewhere in this file" is not something a user can act on."""
    case = _case("local_cut")
    result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
    assert result.cut is not None
    truth = case["cut_position_s"]
    assert abs(result.cut.position_s - truth) <= result.cut.uncertainty_s + 1.0, (
        f"cut placed at {result.cut.position_s:.1f}s +-{result.cut.uncertainty_s:.1f}s, "
        f"truth {truth:.1f}s"
    )
    assert result.cut.uncertainty_s < 6.0, (
        f"probing narrowed the cut to +-{result.cut.uncertainty_s:.1f}s, which is "
        "no better than the windows that straddled it"
    )


def test_a_one_frame_cut_is_not_too_small_to_see():
    """The smallest splice that can exist in a video file."""
    case = _case("minor_cut")
    result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
    assert result.error is None, f"minor_cut failed: {result.error}"
    assert result.is_likely_cut, (
        f"a {case['cut_magnitude_ms']:.1f}ms cut was missed; delay reported as "
        f"{result.delay_ms:.1f}ms"
    )
    assert abs(result.cut.magnitude_ms - case["cut_magnitude_ms"]) < 5.0


def test_a_cut_is_never_offered_a_resampling_fix():
    """A splice fitted with one line becomes a slope, and a slope of the right
    size is a frame-rate conversion with an exact correction ratio. Accepting
    that offer stretches the entire file to correct one local edit."""
    for name in ("local_cut", "minor_cut"):
        case = _case(name)
        result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
        assert not result.is_rate_mismatch, (
            f"{name}: offered a resample at ratio "
            f"{result.rate_diagnosis.correction_ratio!r}"
        )
        assert result.rate_diagnosis.correction_ratio is None


def test_files_without_a_cut_are_not_given_one():
    """The expensive false positive: a cut flag stops the pair being muxed."""
    for case in load_manifest()["cases"]:
        if case["kind"] not in ("constant", "drift", "level_mismatch"):
            continue
        result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
        assert not result.is_likely_cut, (
            f"{case['name']}: invented a "
            f"{result.cut.magnitude_ms:+.1f}ms cut in a file that has none"
        )


def test_drift_is_still_drift_once_cuts_can_be_detected():
    """The two diagnoses must not have traded places."""
    case = _case("drift")
    result = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6)
    assert result.error is None
    assert not result.is_likely_cut
    assert result.has_significant_drift


def test_a_cut_resting_on_one_window_needs_corroboration():
    """With probing switched off there is nothing to tell a genuine post-cut
    window from a window that locked onto a repeated musical phrase, so a step
    resting on a single window is not claimed."""
    from audiosync.analyze import PairResult, WindowResult, _reconcile  # noqa: PLC0415
    from audiosync.correlate import OffsetEstimate  # noqa: PLC0415

    positions = plan_windows(2400.0, 45.0, 6)
    result = PairResult("primary.mkv", "secondary.eac3")
    result.window_s = 45.0
    for index, position in enumerate(positions):
        offset = 300.0 + (2000.0 if index == len(positions) - 1 else 0.0)
        result.windows.append(WindowResult(position, OffsetEstimate(offset, 0.9, 500.0)))

    step = _reconcile(result)
    assert step is not None and step.lone_group, "the proposal itself is fine"

    case = _case("offset_500ms")
    unprobed = analyze_pair(
        case["primary"], case["secondary"], window_s=8.0, window_count=5, cut_probes=0
    )
    assert not unprobed.is_likely_cut


def test_a_cut_is_not_looked_for_when_the_cut_check_is_off():
    """Off, the windows are fitted with one delay and no probe is spent
    locating a cut, so a pair with a splice costs no more than one without."""
    import audiosync.analyze as analyze  # noqa: PLC0415

    case = _case("local_cut")
    measured = []
    original = analyze._measure_window

    def counting(*args, **kwargs):
        measured.append(args[2])
        return original(*args, **kwargs)

    analyze._measure_window = counting
    try:
        checked = analyze_pair(case["primary"], case["secondary"], window_s=6.0, window_count=6, timeline=False)
        probed = len(measured)
        measured.clear()
        unchecked = analyze_pair(
            case["primary"], case["secondary"], window_s=6.0, window_count=6, timeline=False, find_cuts=False,
        )
    finally:
        analyze._measure_window = original

    assert checked.cut is not None and probed > 6, f"precondition: the cut check found and probed it ({probed})"
    assert unchecked.cut is None, unchecked.cut
    assert unchecked.to_dict()["cutPositionS"] is None
    assert len(measured) == 6, f"{len(measured)} windows measured, want the 6 survey windows alone"


def test_the_search_range_covers_the_offset_the_user_asked_for():
    """The secondary is decoded with a head start so a shifted match still fits,
    and that head start lands in the measurement. Bounding the search at the
    user's setting therefore ruled out the very offsets it was meant to allow:
    with the slider at 6s and a real offset of 5s, only the windows near the
    start of the file -- where the file's own beginning caps the head start --
    stayed in range, and the rest reported the tracks as unrelated."""
    case = _case("offset_5000ms")
    result = analyze_pair(
        case["primary"], case["secondary"], window_s=8.0, window_count=4, max_offset_ms=6000.0
    )
    used = sum(1 for w in result.windows if w.usable)
    assert used == len(result.windows), (
        f"only {used} of {len(result.windows)} windows survived a search range "
        "larger than the offset being searched for"
    )
    assert abs(result.delay_ms - case["true_offset_ms"]) < 20.0


def _speechlike(seconds: float, sr: int, seed: int) -> "np.ndarray":
    """Synthesize a broadband speech-like signal, as in make_fixtures.py."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    noise = rng.standard_normal(n)
    voiced = np.convolve(noise, np.ones(24) / 24.0, mode="same")
    envelope = np.zeros(n)
    pos = 0
    while pos < n:
        burst = int(rng.uniform(0.25, 0.9) * sr)
        gap = int(rng.uniform(0.1, 0.5) * sr)
        end = min(n, pos + burst)
        envelope[pos:end] = rng.uniform(0.4, 1.0)
        pos = end + gap
    envelope = np.convolve(envelope, np.ones(128) / 128.0, mode="same")
    signal = voiced * envelope
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    return signal.astype(np.float32)


def _write_intro_pair(tmp: str, intro_s: float, content_s: float):
    """A video with a recap the dub lacks: primary = intro + content, secondary = content."""
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    sr = 16000
    content = _speechlike(content_s, sr, seed=1)
    primary = np.concatenate([_speechlike(intro_s, sr, seed=2), content])
    primary_path = os.path.join(tmp, "primary.wav")
    secondary_path = os.path.join(tmp, "secondary.wav")
    sf.write(primary_path, primary, sr)
    sf.write(secondary_path, content, sr)
    return primary_path, secondary_path, sr


def test_a_dub_missing_the_videos_recap_is_measured_not_stretched():
    """The reported failure: a video with a recap the dub does not carry.

    The dub is 92.4s shorter than the video, which puts its duration ratio
    inside the tolerance of a 25/24 conversion -- the signature a duration
    guess used to treat as a PAL speedup and "correct" by stretching the dub
    4.17% before correlating. That produced a confident wrong delay. Two bugs
    had to come out together: the pre-emptive stretch (the measurement must
    run at the files' own speed, with a conversion taken off only on evidence),
    and the 30s cap on the search margin, which made an offset this large
    physically invisible no matter what the slider said.
    """
    import tempfile  # noqa: PLC0415

    intro_s = 92.4
    with tempfile.TemporaryDirectory() as tmp:
        primary_path, secondary_path, _ = _write_intro_pair(tmp, intro_s, 300.0)
        result = analyze_pair(
            primary_path, secondary_path,
            window_s=30.0, window_count=6, max_offset_ms=120000.0,
        )

    assert result.error is None, f"intro pair failed to measure: {result.error}"
    assert result.speed_compensation == 1.0, (
        f"the dub was stretched by {result.speed_compensation:.4f} -- its "
        "shorter duration was mistaken for a PAL conversion"
    )
    # Negative: the dub's content sits earlier in its own file than in the
    # video, which starts with the recap.
    assert result.delay_ms is not None
    assert abs(result.delay_ms - (-intro_s * 1000.0)) < 200.0, (
        f"delay {result.delay_ms:.1f}ms, want {-intro_s * 1000.0:.1f}ms"
    )
    assert not result.is_rate_mismatch and not result.is_likely_cut, (
        f"misdiagnosed a plain length difference: "
        f"{result.rate_diagnosis.explanation if result.rate_diagnosis else None}"
    )


def test_the_fast_pass_measures_a_large_offset_with_two_windows():
    """The same geometry through the fast route.

    The fast route exists because decoding the whole offset range around every
    survey window re-reads the file over and over: six windows at a five-minute
    offset decode an hour of audio. One long window plus an end check finds the
    same offset in two decodes, which is what makes a large search range usable
    in a batch.
    """
    import tempfile  # noqa: PLC0415

    intro_s = 92.4
    with tempfile.TemporaryDirectory() as tmp:
        primary_path, secondary_path, _ = _write_intro_pair(tmp, intro_s, 300.0)
        result = analyze_pair(
            primary_path, secondary_path,
            window_s=45.0, window_count=6, max_offset_ms=300000.0, prefer_fast=True,
        )

    assert result.error is None, f"fast pair failed to measure: {result.error}"
    assert len(result.windows) == 2, (
        f"the fast pass should measure two windows, got {len(result.windows)}"
    )
    assert result.speed_compensation == 1.0, (
        f"the dub was stretched by {result.speed_compensation:.4f}"
    )
    assert result.delay_ms is not None
    assert abs(result.delay_ms - (-intro_s * 1000.0)) < 200.0, (
        f"delay {result.delay_ms:.1f}ms, want {-intro_s * 1000.0:.1f}ms"
    )
    assert not result.is_rate_mismatch and not result.is_likely_cut


def test_the_fast_pass_withholds_an_answer_when_the_end_check_disagrees():
    """A drifting or cut pair must not quietly report one offset: the end check
    disagrees with the long window and hands over to the full survey."""
    import audiosync.analyze as analyze_module  # noqa: PLC0415

    from audiosync.correlate import OffsetEstimate  # noqa: PLC0415

    def fake_measure(primary_path, secondary_path, position_s, window_s, *args):
        # The long window matches one thing, the end check another.
        if window_s > 100.0:
            return OffsetEstimate(-92400.0, 0.9, 500.0)
        return OffsetEstimate(-1000.0, 0.9, 500.0)

    original = analyze_module._measure_window
    analyze_module._measure_window = fake_measure
    try:
        fast = analyze_module._fast_pair(
            "primary.mkv", "secondary.m4a", 2400.0, 2300.0,
            300000.0, None, 0, 0, 45.0,
        )
    finally:
        analyze_module._measure_window = original
    assert fast is None, "disagreeing windows must not yield a fast answer"


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


def test_a_file_measured_against_itself_is_zero_when_its_audio_starts_late():
    """The video and the dub as the same remux, whose audio starts 2 s after
    its picture. Windows near the start read the dub from zero and the video
    by seeking; before reads shared one clock that came out as a 2 s delay
    (and, in a longer film, as a 2 s "cut" after the first minute)."""
    import shutil
    import subprocess
    import tempfile

    import numpy as np

    from audiosync.media import ffmpeg_path

    root = tempfile.mkdtemp(prefix="audiosync-late-")
    try:
        path = os.path.join(root, "remux.mkv")
        subprocess.run(
            [
                ffmpeg_path(), "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=s=160x90:r=25:d=43",
                "-itsoffset", "2", "-f", "lavfi", "-i", "anoisesrc=d=40:r=16000:a=0.3:seed=5,volume='if(lt(mod(t,3),1),1,0.1)':eval=frame",
                "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "flac", path,
            ],
            check=True,
        )
        result = analyze_pair(path, path, window_s=10.0, window_count=4, max_offset_ms=60000.0, timeline=False)
        assert result.error is None, result.error
        assert result.cut is None, f"a cut was found in a file measured against itself: {result.cut}"
        assert result.delay_ms is not None and abs(result.delay_ms) < 5.0, f"measured {result.delay_ms} ms against itself"
        assert np.isfinite(result.confidence) and result.confidence > 0.9
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _bed_and_dialogue(seconds: float, sr: int, seed: int):
    """A music-and-effects bed spread across the stereo field, and a loud,
    continuous, broadband centre "dialogue" of the given seed.

    The dialogue never pauses and covers the whole band, so the full mix of
    two such tracks shares nothing a correlation can use; only the bed under
    it, which left minus right keeps and the dialogue cancels out of, can
    place one against the other.
    """
    import numpy as np  # noqa: PLC0415

    bed_rng = np.random.default_rng(1234)
    n = int(seconds * sr)
    left = np.zeros(n)
    right = np.zeros(n)
    for _ in range(int(seconds * 3)):
        at = int(bed_rng.uniform(0, seconds - 1) * sr)
        length = int(bed_rng.uniform(0.05, 0.4) * sr)
        burst = bed_rng.standard_normal(length) * np.exp(-np.linspace(0, 6, length)) * 0.05
        pan = bed_rng.uniform(0.0, 1.0)
        left[at:at + length] += burst * np.cos(pan * np.pi / 2)
        right[at:at + length] += burst * np.sin(pan * np.pi / 2)
    dialogue = np.random.default_rng(seed).standard_normal(n) * 0.3
    return np.stack([left + dialogue, right + dialogue], axis=1).astype(np.float32)


def test_a_dub_in_another_language_is_placed_by_the_music_and_effects_it_shares():
    """English and Hindi dialogue share nothing; the M&E under them does.

    The full mix of these two tracks is dominated by two different
    performances, so it cannot place them; the dialogue-free mix (left minus
    right) can, to a fraction of a millisecond, and every window after the
    first is searched only around the offset the first one found.
    """
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    from audiosync.analyze import NARROW_SEARCH_MS  # noqa: PLC0415

    sr = 16000
    shift = 1974  # samples: 123.375 ms
    original = _bed_and_dialogue(90.0, sr, seed=1)
    dub = _bed_and_dialogue(90.0, sr, seed=2)
    dub = np.concatenate([np.zeros((shift, 2), dtype=np.float32), dub])[: len(original)]
    root = tempfile.mkdtemp(prefix="audiosync-me-")
    try:
        primary, secondary = os.path.join(root, "english.wav"), os.path.join(root, "hindi.wav")
        sf.write(primary, original, sr)
        sf.write(secondary, dub, sr)
        result = analyze_pair(primary, secondary, window_s=15.0, window_count=4, max_offset_ms=10000.0, timeline=False)
        assert result.error is None, result.error
        assert abs(result.delay_ms - 123.375) < 0.5, f"measured {result.delay_ms:+.3f} ms, want +123.375"
        assert all(w.estimate.mix == "me" for w in result.windows), [w.estimate.mix for w in result.windows]
        assert all(w.estimate.confirmed for w in result.windows), "the shared M&E should confirm every window"
        assert result.search_ms == NARROW_SEARCH_MS, "windows after the first should search around its offset"
        assert result.confidence > 0.95, result.confidence
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_dual_mono_dub_is_measured_on_the_full_mix():
    """Left minus right of a dual-mono track is silence, so the M&E has
    nothing to offer and the full mix answers, as it always did."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    sr = 16000
    shift = 800  # samples: 50 ms
    mono = _speechlike(60.0, sr, seed=4)
    original = np.stack([mono, mono * 0.5], axis=1)
    dub = np.concatenate([np.zeros(shift, dtype=np.float32), mono])[: len(mono)]
    root = tempfile.mkdtemp(prefix="audiosync-dualmono-")
    try:
        primary, secondary = os.path.join(root, "video.wav"), os.path.join(root, "dub.wav")
        sf.write(primary, original, sr)
        sf.write(secondary, np.stack([dub, dub], axis=1), sr)
        result = analyze_pair(primary, secondary, window_s=10.0, window_count=4, max_offset_ms=10000.0, timeline=False)
        assert result.error is None, result.error
        assert abs(result.delay_ms - 50.0) < 0.5, f"measured {result.delay_ms:+.3f} ms, want +50"
        assert all(w.estimate.mix == "full" for w in result.windows if w.usable)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_mix_the_waveforms_confirm_speaks_for_a_window():
    from audiosync.analyze import DECISIVE_PEAK_RATIO, _choose_mix  # noqa: PLC0415
    from audiosync.correlate import OffsetEstimate  # noqa: PLC0415

    def est(delay, ratio, confirmed=False, waveform=0.0, mix="full"):
        return OffsetEstimate(delay, 0.8, ratio, confirmed=confirmed, waveform_ratio=waveform, mix=mix)

    unmatched = OffsetEstimate(None, 0.2, 5.0, "no distinct correlation peak; tracks appear unrelated")
    # A sharper envelope peak in the M&E is not enough on its own: that is
    # what a lock onto a repeated cue looks like, and the waveforms decide.
    assert _choose_mix(est(37.0, 33.0), est(-71.0, 72.0, mix="me")).delay_ms == 37.0
    assert _choose_mix(est(37.0, 33.0), est(37.4, 72.0, True, 900.0, "me")).mix == "me"
    assert _choose_mix(est(37.0, 33.0, True, 1000.0), est(-71.0, 72.0, mix="me")).delay_ms == 37.0
    # Agreeing with the full mix, the sharper M&E reading is the same answer.
    assert _choose_mix(est(37.0, 20.0), est(37.003, 40.0, mix="me")).mix == "me"
    # Where the full mix found nothing, the M&E alone must be decisive.
    assert _choose_mix(unmatched, est(12.0, DECISIVE_PEAK_RATIO - 1, mix="me")).delay_ms is None
    assert _choose_mix(unmatched, est(12.0, DECISIVE_PEAK_RATIO + 1, mix="me")).delay_ms == 12.0


def test_windows_that_agree_are_reported_as_sure():
    """Six windows on one offset are not a coincidence, however soft each
    peak was; two are not enough to say so."""
    from audiosync.analyze import AGREEMENT_CONFIDENCE_CAP, NARROW_SEARCH_MS, _agreement_confidence  # noqa: PLC0415

    assert _agreement_confidence(6, 6, NARROW_SEARCH_MS) == AGREEMENT_CONFIDENCE_CAP
    assert 0.9 < _agreement_confidence(3, 6, NARROW_SEARCH_MS) < AGREEMENT_CONFIDENCE_CAP
    assert _agreement_confidence(3, 6, 60000.0) == AGREEMENT_CONFIDENCE_CAP
    assert _agreement_confidence(2, 6, 60000.0) == 0.0


def test_a_step_of_about_a_frame_is_a_slip_and_a_bigger_one_a_cut():
    from audiosync.analyze import PairResult  # noqa: PLC0415
    from audiosync.segments import Step  # noqa: PLC0415

    def pair(fps=23.976, cut=None, edits=(), gaps=()):
        result = PairResult("video.mkv", "dub.ac3", primary_fps=fps, edits=list(edits), gaps=list(gaps))
        if cut is not None:
            result.cut = Step(before_ms=10.0, after_ms=10.0 + cut, split_position_s=600.0, earliest_s=590.0, latest_s=610.0)
        return result

    def edit(before, after, kind="missing", missing_s=0.0, extra_s=0.0):
        return {"kind": kind, "jumpMs": after - before, "offsetBeforeMs": before, "offsetAfterMs": after,
                "missingS": missing_s, "extraS": extra_s}

    assert pair(cut=41.7).is_minor_slip, "one frame at 23.976 is a slip"
    assert not pair(cut=100.0).is_minor_slip
    assert not pair(fps=None, cut=50.0).is_minor_slip, "without a frame rate the limit is 45 ms"
    assert pair(fps=60.0, cut=30.0).is_minor_slip, "at 60 fps the limit is still 45 ms, not 18"
    assert pair(edits=[edit(10, 40, missing_s=0.03), edit(40, 10, kind="extra", extra_s=0.03)]).is_minor_slip, \
        "slips that cancel stay slips"
    assert not pair(edits=[edit(10, 40, missing_s=0.03), edit(40, 70, missing_s=0.03)]).is_minor_slip, \
        "slips that add up are a cut"
    assert not pair(edits=[edit(10, 35, kind="replaced", missing_s=30.0)]).is_minor_slip, \
        "a scene the dub replaces is a cut however small the jump"
    assert pair(edits=[edit(10, 40, missing_s=0.03)], gaps=[{"reason": "tail"}]).is_minor_slip, \
        "a dub that ends early still only slips"
    assert not pair(edits=[edit(10, 40, missing_s=0.03)], gaps=[{"reason": "unmatched"}]).is_minor_slip
    assert not pair().is_minor_slip, "no step is no slip"


def test_each_window_is_reported_on_the_results_terms():
    """The window list carries the codec correction the result applied, so a
    window and the delay it fed read the same."""
    from audiosync.analyze import PairResult, WindowResult  # noqa: PLC0415
    from audiosync.correlate import OffsetEstimate  # noqa: PLC0415

    result = PairResult("video.mkv", "dub.ac3", codec_delay_ms=5.333)
    result.windows = [
        WindowResult(10.0, OffsetEstimate(42.0, 0.97, 80.0, confirmed=True, waveform_ratio=900.0, mix="me"), agrees=True),
        WindowResult(50.0, OffsetEstimate(None, 0.1, 3.0, "no distinct correlation peak; tracks appear unrelated")),
    ]
    details = result.to_dict()["windowDetails"]
    assert abs(details[0]["delayMs"] - 36.667) < 1e-9
    assert details[0]["mix"] == "me" and details[0]["confirmed"] and details[0]["agrees"]
    assert details[1]["delayMs"] is None and details[1]["reason"] and details[1]["agrees"] is None


def test_overlapping_windows_are_one_witness_not_several():
    """On a short file the windows overlap, and one lock read again by the
    next window over is the same evidence twice: six of those agreeing must
    not be reported as six independent windows agreeing."""
    from audiosync.analyze import PairResult, WindowResult, _reconcile  # noqa: PLC0415
    from audiosync.correlate import OffsetEstimate  # noqa: PLC0415

    def surveyed(step_s):
        result = PairResult("video.mkv", "dub.ac3", window_s=45.0, search_ms=2000.0)
        result.windows = [WindowResult(i * step_s, OffsetEstimate(120.0 + i * 0.1, 0.55, 14.0)) for i in range(6)]
        _reconcile(result, allow_step=False)
        return result

    overlapping = surveyed(step_s=5.0)
    assert overlapping.agreeing_windows == 6
    assert overlapping.confidence < 0.6, f"one stretch of audio read six times scored {overlapping.confidence:.2f}"
    apart = surveyed(step_s=100.0)
    assert apart.confidence > 0.95, f"six separate stretches agreeing scored {apart.confidence:.2f}"


def test_a_speed_is_settled_by_the_full_mix_or_by_the_waveforms():
    """The decisive bar was set on the full mix's envelope peak. An M&E
    reading -- a second chance per trial -- settles a speed only when the
    waveforms confirm it."""
    from fractions import Fraction  # noqa: PLC0415

    import audiosync.analyze as analyze  # noqa: PLC0415
    from audiosync.correlate import OffsetEstimate  # noqa: PLC0415

    def trial(estimate):
        original = analyze._measure_window
        analyze._measure_window = lambda *args, **kwargs: estimate
        try:
            return analyze._search_speed("v.mkv", "d.ac3", 45.0, 60000.0, None, 0, 0, [Fraction(25, 24)], 16000, 3600.0)
        finally:
            analyze._measure_window = original

    rate = int(round(16000 * 25 / 24))
    assert trial(OffsetEstimate(80.0, 0.9, 35.0, mix="full")) == rate
    assert trial(OffsetEstimate(80.0, 0.9, 35.0, mix="me")) is None, "an unconfirmed M&E peak is a second coincidence"
    assert trial(OffsetEstimate(80.0, 0.9, 20.0, confirmed=True, waveform_ratio=900.0, mix="me")) == rate


def test_a_narrowed_window_stays_inside_the_range_the_user_asked_for():
    """An anchor near the edge of the search range must not carry the next
    window past it: the dub is really 4.5 s late, the user searched 3 s."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    from audiosync.analyze import _measure_window  # noqa: PLC0415

    sr = 16000
    signal = _speechlike(60.0, sr, seed=12)
    late = np.concatenate([np.zeros(int(4.5 * sr), dtype=np.float32), signal])[: len(signal)]
    root = tempfile.mkdtemp(prefix="audiosync-narrow-")
    try:
        primary, secondary = os.path.join(root, "video.wav"), os.path.join(root, "dub.wav")
        sf.write(primary, signal, sr)
        sf.write(secondary, late, sr)
        estimate = _measure_window(primary, secondary, 20.0, 15.0, 3000.0, None, expect_ms=2900.0)
        assert not estimate.matched or estimate.delay_ms <= 3000.0 + 15.0, estimate
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_cut_bigger_than_the_narrowed_search_is_still_found():
    """Narrowed windows search 2 s around the last offset; a 3 s cut lies
    outside that, so the window after it must be measured in full."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    sr = 16000
    original = _bed_and_dialogue(120.0, sr, seed=1)
    dub = _bed_and_dialogue(120.0, sr, seed=2)
    cut_at, cut_s = int(60.0 * sr), int(3.0 * sr)
    dub = np.concatenate([dub[:cut_at], dub[cut_at + cut_s:]])
    root = tempfile.mkdtemp(prefix="audiosync-bigcut-")
    try:
        primary, secondary = os.path.join(root, "video.wav"), os.path.join(root, "dub.wav")
        sf.write(primary, original, sr)
        sf.write(secondary, dub, sr)
        result = analyze_pair(primary, secondary, window_s=15.0, window_count=6, max_offset_ms=10000.0, timeline=False)
        assert result.error is None, result.error
        assert result.cut is not None, [w.estimate.delay_ms for w in result.windows]
        assert abs(result.cut.magnitude_ms + 3000.0) < 5.0, result.cut.magnitude_ms
        assert not result.is_minor_slip
    finally:
        shutil.rmtree(root, ignore_errors=True)
