"""Tests for measuring a pair the survey cannot settle on its own.

A clean pair -- one offset from start to end -- is measured by the six-window
survey exactly as before. Anything else is laid along the whole video: a cut
bigger than the search range, several edits, a frame-rate conversion, a weak
match. These check that such a pair comes back with the right delay, every
edit listed however small, the conversion named from which rate to which, and
a correction that lands on the picture.
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import soundfile as sf  # noqa: E402

from audiosync.analyze import PairResult, WindowResult, _guard_weak_survey, analyze_pair  # noqa: E402
from audiosync.correlate import OffsetEstimate  # noqa: E402
from audiosync.dubsync import DubSyncPlan, Segment  # noqa: E402
from audiosync.media import ffmpeg_path  # noqa: E402
from audiosync.mux import build_command, plan_correction  # noqa: E402
from make_fixtures import _resample_linear  # noqa: E402
from test_dubsync import SR, Workspace, build_pair  # noqa: E402

FILM_23976 = 24000 / 1001


def _with_picture(ws, wav: str, fps_num: int, fps_den: int) -> str:
    """The audio in a Matroska file with a tiny picture stream at a real rate,
    so the video carries a frame rate as a real one does."""
    out = ws.path("video.mkv")
    duration = sf.info(wav).duration
    subprocess.run([
        ffmpeg_path(), "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=black:s=32x32:r={fps_num}/{fps_den}:d={duration}",
        "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "flac", out,
    ], check=True)
    return out


def _pal_dub(ws, pieces, total_s):
    """A dub mastered at 25fps for a 23.976fps video: it runs 25/23.976 fast."""
    build_pair(ws, total_s, pieces)
    audio, _ = sf.read(ws.path("dub.wav"), dtype="float32")
    sf.write(ws.path("dub.wav"), _resample_linear(audio, FILM_23976 / 25.0), SR)
    return _with_picture(ws, ws.path("org.wav"), 24000, 1001), ws.path("dub.wav")


def test_a_clean_pair_is_measured_by_the_survey_alone():
    with Workspace() as ws:
        build_pair(ws, 400, [("extra", 2.6), ("org", 0, 400)])
        result = analyze_pair(ws.path("org.wav"), ws.path("dub.wav"))
        survey = analyze_pair(ws.path("org.wav"), ws.path("dub.wav"), timeline=False)
        assert result.method == "survey", result.to_dict()
        assert result.delay_at_start_ms == survey.delay_at_start_ms
        assert result.confidence == survey.confidence
        assert result.edits == [] and result.warnings == []


def test_a_cut_beyond_the_search_range_is_measured_and_listed():
    """150 s missing, beyond the survey's 60 s search: the windows after it find
    nothing, and the survey used to call the file one delay."""
    with Workspace() as ws:
        build_pair(ws, 900, [("extra", 2.6), ("org", 0, 300), ("org", 450, 900)])
        result = analyze_pair(ws.path("org.wav"), ws.path("dub.wav"))
        assert result.method == "timeline"
        assert abs(result.delay_at_start_ms - 2600.0) < 3.0, result.delay_at_start_ms
        assert len(result.edits) == 1, result.edits
        edit = result.edits[0]
        assert edit["kind"] == "missing" and edit["severity"] == "major"
        assert abs(edit["sizeMs"] - 150000.0) < 20.0 and abs(edit["videoS"] - 300.0) < 1.0, edit
        assert result.is_likely_cut
        assert result.to_dict()["cutPositionS"] == edit["videoS"]


def test_every_edit_is_listed_from_one_frame_up():
    with Workspace() as ws:
        build_pair(ws, 600, [
            ("extra", 2.6), ("org", 0, 150), ("org", 150.042, 300), ("extra", 0.5), ("org", 300, 450),
            ("org", 470, 600),
        ])
        result = analyze_pair(ws.path("org.wav"), ws.path("dub.wav"))
        kinds = [(e["kind"], e["severity"], round(e["sizeMs"])) for e in result.edits]
        assert kinds == [("missing", "minor", 42), ("extra", "minor", 500), ("missing", "moderate", 20000)], kinds
        for edit, at in zip(result.edits, (150.0, 300.0, 450.0)):
            assert abs(edit["videoS"] - at) < 0.5, edit
        assert "under the 45 ms" in result.edits[0]["description"]


def test_a_pal_dub_is_measured_and_its_conversion_named():
    """The survey needed one window to clear a bar a weak dub never clears, and
    reported a PAL dub as unrelated audio."""
    with Workspace() as ws:
        video, dub = _pal_dub(ws, [("extra", 2.6), ("org", 0, 300), ("org", 305, 600)], 600)
        result = analyze_pair(video, dub)
        assert result.error is None, result.error
        assert result.method == "timeline" and result.is_rate_mismatch
        guide = result.rate_guide
        assert guide["named"] and guide["fromFps"] == 25.0 and abs(guide["toFps"] - FILM_23976) < 1e-9
        assert guide["stretch"] == {"num": 1001, "den": 960}, guide["stretch"]
        assert guide["resampleFilter"] == "aresample=16016,asetrate=15360,aresample=16000", guide["resampleFilter"]
        # The survey's convention: the dub's own clock at t=0, offset / speed.
        assert abs(result.delay_at_start_ms - 2600.0 * 960 / 1001) < 3.0, result.delay_at_start_ms
        assert abs(guide["delayWithStretchMs"] - 2600.0) < 3.0, guide["delayWithStretchMs"]
        diagnosis = result.rate_diagnosis
        assert diagnosis.source_fps == 25.0 and abs(diagnosis.correction_ratio - 960 / 1001) < 1e-9
        assert [e["kind"] for e in result.edits] == ["missing"] and result.edits[0]["frames"] == 120


def test_a_pal_correction_lands_on_the_picture():
    """Measured, corrected, measured again: the corrected dub sits on the video.
    Before, the shift was made in the dub's own time after atempo had already
    moved it onto the video's, leaving it out by delay x 4%; and atempo's
    windows moved it another 10-17 ms."""
    with Workspace() as ws:
        video, dub = _pal_dub(ws, [("extra", 2.6), ("org", 0, 400)], 400)
        measured = analyze_pair(video, dub)
        plan = plan_correction(
            video, dub, measured.delay_at_start_ms, drift_ms_per_s=measured.drift_ms_per_s,
            output_dir=ws.root,
        )
        assert plan.audio_sample_rate == SR
        subprocess.run(build_command(plan), check=True, capture_output=True)
        after = analyze_pair(video, plan.output_path)
        # Written as AAC in Matroska, with the encoder's priming marked to be
        # skipped; FFmpeg 6.1 used to read it as sound and put this 64 ms late
        # (see media.kept_priming). It lands on the picture on every build.
        assert abs(after.delay_at_start_ms) < 3.0, f"corrected dub is {after.delay_at_start_ms:+.1f} ms off"
        assert not after.is_rate_mismatch and not after.edits, after.to_dict()


def test_unrelated_audio_says_what_it_means():
    with Workspace() as ws:
        build_pair(ws, 200, [("org", 0, 200)])
        other = Workspace().__enter__()
        try:
            build_pair(other, 200, [("org", 0, 200)], bed_gain=0.0)
            result = analyze_pair(ws.path("org.wav"), other.path("dub.wav"))
        finally:
            other.__exit__()
        assert result.delay_ms is None
        assert "does not match the video anywhere" in result.error, result.error
        assert not any("survey alone" in w for w in result.warnings), result.warnings


def test_a_survey_of_one_window_is_not_trusted_on_its_own():
    matched = WindowResult(100.0, OffsetEstimate(59900.0, 0.57, 13.7))
    empty = [WindowResult(p, OffsetEstimate(None, 0.2, 5.0, "no peak")) for p in (0, 200, 300, 400, 500)]
    result = PairResult("v", "a", delay_ms=59900.0, confidence=0.57, windows=[matched] + empty)
    _guard_weak_survey(result)
    assert result.confidence < 0.5 and "Only 1 of 6 windows" in result.warnings[0]


def test_edits_are_classified_and_short_stretches_marked():
    """A cut, an insert, a replaced scene -- and a stretch too short to rule out
    a repeated cue, which is listed but marked to check."""
    dub = lambda a, b, off: Segment("dub", a, b, a + off, offset_s=off)  # noqa: E731
    fill = lambda a, b, note="dub is cut here": Segment("fill", a, b, a, note=note)  # noqa: E731
    plan = DubSyncPlan("v.mkv", "a.m4a", video_duration_s=1000.0, video_fps=25.0, segments=[
        dub(0, 200, 2.0), fill(200, 210), dub(210, 400, -8.0),          # 10 s missing
        dub(400, 600, -5.0),                                              # 3 s extra
        fill(600, 630), dub(630, 640, -15.0),                             # 30 s replaced by 20 s, 10 s stretch
        dub(640, 1000, -15.25),                                           # 250 ms missing
        fill(1000, 1010, "past dub end"),
    ])
    edits, gaps = plan.edits_and_gaps()
    assert [(e["kind"], e["severity"]) for e in edits] == [
        ("missing", "moderate"), ("extra", "moderate"), ("replaced", "moderate"), ("missing", "minor"),
    ], edits
    assert edits[0]["frames"] == 250 and edits[0]["missingS"] == 10.0
    assert edits[2]["missingS"] == 30.0 and abs(edits[2]["extraS"] - 20.0) < 1e-9
    assert [e["check"] for e in edits] == [False, False, True, True]
    assert [g["reason"] for g in gaps] == ["tail"]
    assert plan.to_dict()["edits"] == edits
