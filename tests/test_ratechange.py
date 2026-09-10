"""Tests for naming the frame-rate conversion an audio track needs.

The question a user asks is "which rate do I convert from, and to what" -- so
what these check is not that a drift number came out, but that the pair of rates
was named, and named the right way round.

Two conversions stand for the whole set. 24 -> 23.976 is 0.1%, the narrowest
real conversion and the one that stresses precision. 25 -> 23.976 is 4.27%, the
PAL speedup, and it stresses something else entirely: at that speed the offset
moves further inside a single measurement window than the correlation can
resolve, so the pair reports itself as unrelated unless the speed is taken off
before the measurement.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.analyze import analyze_pair  # noqa: E402
from audiosync.framerate import (  # noqa: E402
    diagnose,
    plan_speed_compensation,
    speed_ratio_for,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "fixtures", "manifest.json")

NTSC_FILM = 24000.0 / 1001.0


def _case(name):
    with open(MANIFEST, encoding="utf-8") as handle:
        for case in json.load(handle)["cases"]:
            if case["name"] == name:
                return case
    raise KeyError(name)


def _named(result):
    """The conversion the result names, as an (audio_fps, video_fps) pair."""
    diagnosis = result.rate_diagnosis
    if not diagnosis or not diagnosis.is_rate_mismatch:
        return None
    return diagnosis.source_fps, diagnosis.target_fps


# --- What a drift figure means -------------------------------------------


def test_the_speed_ratio_is_inverted_exactly():
    """A window at t finds its content at t/ratio, so drift is 1/ratio - 1.

    Inverting that as 1 - drift/1000 is a first-order approximation. It holds to
    a millionth at the 0.1% of a 24 -> 23.976 conversion, which is why it went
    unnoticed, and it is wrong by 0.17% at the 4.27% of a PAL speedup -- four
    times the tolerance a conversion has to be matched within.
    """
    ratio = 25.0 / NTSC_FILM
    drift = (1.0 / ratio - 1.0) * 1000.0
    assert abs(speed_ratio_for(drift) - ratio) < 1e-9, (
        f"{drift:.4f} ms/s inverted to {speed_ratio_for(drift):.9f}, want {ratio:.9f}"
    )
    # The approximation it replaced would have missed by more than the tolerance.
    assert abs((1.0 - drift / 1000.0) - ratio) / ratio > 4e-4


def test_an_impossible_drift_does_not_divide_by_zero():
    assert speed_ratio_for(-1000.0) == 0.0
    assert speed_ratio_for(-2000.0) == 0.0


# --- Deciding whether to take the speed off before measuring --------------


def test_a_pal_length_difference_is_recognised_from_the_durations():
    """4.27% short is the signature, and it has to be caught before measuring."""
    ratio = 25.0 / NTSC_FILM
    plan = plan_speed_compensation(2700.0, 2700.0 / ratio)
    assert plan is not None, "a PAL speedup was not recognised from the durations"
    assert abs(float(plan) - ratio) < 1e-9


def test_a_conversion_small_enough_to_correlate_is_left_alone():
    """24 -> 23.976 needs no help, and the durations could not resolve it anyway:
    0.1% is inside what a trimmed logo is worth."""
    assert plan_speed_compensation(2700.0, 2700.0 / (24.0 / NTSC_FILM)) is None


def test_a_plain_length_difference_is_not_treated_as_a_speed_change():
    """Half a minute of extra credits on a 45-minute episode is 1.1% -- large
    enough to notice and nothing to do with frame rate. Snapping to a standard
    conversion is what tells the two apart."""
    assert plan_speed_compensation(2700.0, 2730.0) is None
    assert plan_speed_compensation(2700.0, 2700.0) is None
    assert plan_speed_compensation(None, 2700.0) is None
    assert plan_speed_compensation(2700.0, 0.0) is None


# --- End to end -----------------------------------------------------------


def test_a_pal_speedup_is_measured_and_named():
    """The case that could not be measured at all.

    At 4.27% the alignment slides 1.9 seconds inside a 45-second window and the
    correlation peak flattens into the noise: every window reported the two
    tracks as unrelated, so there was no drift to diagnose and no conversion to
    name. Measured on real decodes, peak prominence went from 6-12 against a
    threshold of 12, to 71-207 once the speed was taken off first.
    """
    case = _case("rate_pal")
    result = analyze_pair(
        case["primary"], case["secondary"], window_s=10.0, window_count=5,
        max_offset_ms=3000.0,
    )
    assert result.error is None, f"rate_pal failed to measure: {result.error}"
    assert result.speed_compensation > 1.04, (
        "the speed was not taken off before measuring "
        f"(compensation {result.speed_compensation:.4f})"
    )
    named = _named(result)
    assert named is not None, (
        f"no conversion named; drift {result.drift_ms_per_s} ms/s, "
        f"{sum(1 for w in result.windows if w.usable)} windows usable"
    )
    audio_fps, video_fps = named
    assert abs(audio_fps - 25.0) < 0.01 and abs(video_fps - NTSC_FILM) < 0.01, (
        f"named {audio_fps:.3f} -> {video_fps:.3f}, want 25 -> 23.976"
    )


def test_the_narrowest_conversion_is_still_named():
    case = _case("rate_film")
    result = analyze_pair(
        case["primary"], case["secondary"], window_s=10.0, window_count=5,
        max_offset_ms=3000.0,
    )
    assert result.error is None, f"rate_film failed to measure: {result.error}"
    named = _named(result)
    assert named is not None, f"no conversion named; drift {result.drift_ms_per_s}"
    audio_fps, video_fps = named
    assert abs(audio_fps - 24.0) < 0.01 and abs(video_fps - NTSC_FILM) < 0.01, (
        f"named {audio_fps:.3f} -> {video_fps:.3f}, want 24 -> 23.976"
    )


def test_the_correction_offered_is_the_exact_conversion():
    """The ratio is what gets applied to every timestamp, so it is not enough
    for it to be close: it has to be the conversion itself."""
    case = _case("rate_pal")
    result = analyze_pair(
        case["primary"], case["secondary"], window_s=10.0, window_count=5,
        max_offset_ms=3000.0,
    )
    ratio = result.rate_diagnosis.correction_ratio
    assert ratio is not None
    assert abs(ratio - NTSC_FILM / 25.0) < 1e-12, (
        f"correction ratio {ratio!r} is not exactly 23.976/25"
    )


def test_a_matched_pair_is_not_given_a_conversion():
    """The expensive false positive: resampling a file that was already right."""
    case = _case("offset_500ms")
    result = analyze_pair(
        case["primary"], case["secondary"], window_s=8.0, window_count=5
    )
    assert result.speed_compensation == 1.0
    assert _named(result) is None, (
        f"invented a conversion: {result.rate_diagnosis.explanation}"
    )


# --- Every audio format ---------------------------------------------------

# codec name -> ffmpeg encoder arguments and the extension to write.
CODECS = [
    ("E-AC-3 (.eac3)", ["-c:a", "eac3", "-b:a", "640k"], "eac3"),
    ("E-AC-3 (.ec3)", ["-c:a", "eac3", "-b:a", "640k"], "ec3"),
    ("AC-3", ["-c:a", "ac3", "-b:a", "448k"], "ac3"),
    ("AAC (ADTS)", ["-c:a", "aac", "-b:a", "256k"], "aac"),
    ("AAC (.m4a)", ["-c:a", "aac", "-b:a", "256k"], "m4a"),
    ("DTS", ["-c:a", "dca", "-strict", "-2", "-ar", "48000"], "dts"),
    ("TrueHD", ["-c:a", "truehd", "-strict", "-2"], "thd"),
    ("FLAC", ["-c:a", "flac"], "flac"),
    ("Opus", ["-c:a", "libopus", "-b:a", "256k"], "opus"),
    ("MP3", ["-c:a", "libmp3lame", "-b:a", "320k"], "mp3"),
    ("MP2", ["-c:a", "mp2", "-b:a", "384k"], "mp2"),
    ("ALAC", ["-c:a", "alac"], "m4a"),
    ("PCM", ["-c:a", "pcm_s16le"], "wav"),
    ("E-AC-3 in Matroska", ["-c:a", "eac3", "-b:a", "640k"], "mka"),
    ("DTS in Matroska", ["-c:a", "dca", "-strict", "-2", "-ar", "48000"], "mka"),
]

# WMA is left out on purpose. ffmpeg's ASF demuxer decodes 85.8ms further in
# when it seeks than when it reads from the start, so on a fixture short enough
# that some windows seek and some do not, the offsets step partway through. On a
# real-length file every window seeks, the shift is identical in all of them and
# cancels out of the drift -- verified separately at 300s, where WMA names the
# conversion correctly. Nothing here can fix a demuxer, and reproducing it would
# only pin down ffmpeg's behaviour rather than this engine's.


def test_every_audio_format_names_the_same_conversion():
    """A dub arrives as whatever the release used, and the answer must not.

    The conversion is a property of the timeline, not of the encoding, so every
    one of these has to reach the same conclusion. They decode to the same
    samples; the point is that nothing upstream of the decode -- a container
    with no duration, an elementary stream ffprobe can only estimate -- quietly
    changes the answer.
    """
    if not shutil.which("ffmpeg"):
        print("      (skipped: no ffmpeg)")
        return

    case = _case("rate_pal")
    workspace = tempfile.mkdtemp(prefix="audiosync-codecs-")
    failures, checked, skipped = [], [], []
    try:
        for label, args, extension in CODECS:
            encoded = os.path.join(
                workspace, f"{label.replace(' ', '_').replace('.', '')}.{extension}"
            )
            built = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", case["secondary"]]
                + args + [encoded],
                capture_output=True,
            )
            if built.returncode != 0 or not os.path.exists(encoded):
                skipped.append(label)
                continue

            result = analyze_pair(
                case["primary"], encoded, window_s=10.0, window_count=5,
                max_offset_ms=3000.0,
            )
            checked.append(label)
            if result.error:
                failures.append(f"{label}: {result.error}")
                continue
            named = _named(result)
            if named is None:
                failures.append(
                    f"{label}: named no conversion "
                    f"(drift {result.drift_ms_per_s:.3f} ms/s)"
                )
            elif abs(named[0] - 25.0) > 0.01 or abs(named[1] - NTSC_FILM) > 0.01:
                failures.append(
                    f"{label}: named {named[0]:.3f} -> {named[1]:.3f}, want 25 -> 23.976"
                )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert checked, "no encoder available to test with"
    assert not failures, (
        f"{len(failures)} of {len(checked)} formats disagreed:\n  "
        + "\n  ".join(failures)
    )


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
