"""A dub that is only late or early: said as a delay, written and muxed as one.

A dub cut nowhere needs no syncing, only a delay. The plan says so
(``delay_only_s``), the track is written as the dub shifted -- silence where
it has nothing, none of the original -- and a mux carries the dub's own
stream, not re-encoded, delayed in the container, landing where the written
track would.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync.dubrender import RenderOptions, as_plain_delay, mux, render  # noqa: E402
from audiosync.dubsync import DubSyncPlan, Segment  # noqa: E402
from audiosync.media import ffmpeg_path, probe  # noqa: E402
from audiosync.voicefix import VoicePiece  # noqa: E402
from test_dubsync import Workspace, _plan, build_pair  # noqa: E402


def test_a_dub_that_only_starts_late_is_a_delay():
    with Workspace() as ws:
        build_pair(ws, 200, [("extra", 4.37), ("org", 0, 200)])
        plan = _plan(ws)
        assert plan.delay_only_s is not None, plan.describe()
        assert abs(plan.delay_only_s - 4.37) < 0.003, plan.describe()
        assert abs(plan.to_dict()["delayOnlyMs"] - 4370.0) < 3.0
        assert plan.describe().startswith("Nothing is cut: the dub only needs a delay of +4370 ms")


def test_a_dub_that_lacks_its_first_seconds_is_a_delay():
    with Workspace() as ws:
        build_pair(ws, 200, [("org", 7.2, 200)])
        plan = _plan(ws)
        assert plan.delay_only_s is not None and abs(plan.delay_only_s + 7.2) < 0.003, plan.describe()
        assert "moved later" in plan.describe().splitlines()[0]


def test_a_cut_dub_is_not_a_delay():
    with Workspace() as ws:
        build_pair(ws, 200, [("org", 0, 80), ("org", 95, 200)])
        plan = _plan(ws)
        assert plan.delay_only_s is None, plan.describe()
        assert plan.to_dict()["delayOnlyMs"] is None


def _plan_of(segments, **kwargs) -> DubSyncPlan:
    return DubSyncPlan("v.mkv", "d.ac3", video_duration_s=100.0, dub_duration_s=100.0, segments=segments, **kwargs)


def test_the_same_offset_either_side_of_a_fill_is_not_a_delay():
    """The dub has something else there -- a replaced scene, a dropout --
    and the original plays for a reason."""
    plan = _plan_of([
        Segment("dub", 0.0, 50.0, 1.0, offset_s=1.0),
        Segment("fill", 50.0, 60.0, 50.0, note="dub is cut here"),
        Segment("dub", 60.0, 100.0, 61.0, offset_s=1.0),
    ])
    assert plan.delay_only_s is None


def test_fills_before_the_dub_starts_and_after_it_ends_are_still_a_delay():
    plan = _plan_of([
        Segment("fill", 0.0, 3.0, 0.0, note="dub starts late"),
        Segment("dub", 3.0, 60.0, 0.0, offset_s=-3.0),
        Segment("dub", 60.0, 97.0, 57.0, offset_s=-3.0005),
        Segment("fill", 97.0, 100.0, 97.0, note="past dub end"),
    ])
    assert plan.delay_only_s is not None and abs(plan.delay_only_s + 3.0) < 0.001


def test_moved_voices_are_not_a_delay():
    plan = _plan_of([Segment("dub", 0.0, 100.0, 2.0, offset_s=2.0)])
    assert plan.delay_only_s == 2.0
    plan.voice_pieces = [VoicePiece(10.0, 20.0, 1.0, 1.0, False, "moved")]
    assert plan.delay_only_s is None


def test_a_delay_is_written_as_the_dub_shifted_with_silence_before_it():
    with Workspace() as ws:
        build_pair(ws, 120, [("org", 7.2, 120)])
        plan = _plan(ws)
        shifted = as_plain_delay(plan)
        assert [s.kind for s in shifted.segments][:2] == ["silence", "dub"], shifted.segments
        render(plan, ws.path("out.wav"), RenderOptions(codec="wav"))
        out, rate = sf.read(ws.path("out.wav"), dtype="float32")
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        out = out if out.ndim == 1 else out[:, 0]
        head = int(round(-plan.delay_only_s * rate))
        # Nothing of the original before the dub starts: silence.
        assert np.max(np.abs(out[: head - 2])) == 0.0
        # Then the dub itself, sample for sample (past the fade-in).
        a = out[head + 1000: head + 1000 + 5 * rate]
        b = dub[1000: 1000 + 5 * rate]
        assert np.max(np.abs(a - b)) < 2e-4


def _as_mkv(ws, wav: str, name: str) -> str:
    target = ws.path(name)
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", ws.path(wav), "-c:a", "flac", target], check=True)
    return target


def _mux_lands(pieces):
    with Workspace() as ws:
        build_pair(ws, 120, pieces)
        video = _as_mkv(ws, "org.wav", "org.mkv")
        plan = _plan(ws)
        plan.video_path = video
        assert plan.delay_only_s is not None, plan.describe()
        render(plan, ws.path("written.wav"), RenderOptions(codec="wav"))
        said = []
        out = mux(plan, ws.path("written.wav"), ws.path("muxed.mkv"), log=said.append)
        assert os.path.isfile(out)
        assert any("muxed the dub's own stream, not re-encoded" in line for line in said), said
        codecs = [t.codec for t in probe(out).audio_tracks]
        # The dub goes in as it is: the WAV's own PCM, not re-encoded.
        assert codecs[-1].startswith("pcm"), codecs


def test_a_late_dub_is_muxed_as_itself_moved_earlier():
    _mux_lands([("extra", 4.37), ("org", 0, 120)])


def test_an_early_dub_is_muxed_as_itself_moved_later():
    _mux_lands([("org", 7.2, 120)])



def test_a_plan_with_a_hole_between_pieces_is_written_on_the_videos_clock():
    """A gap between two pieces must not pull everything after it earlier:
    it is filled, and the second piece plays where its times say."""
    with Workspace() as ws:
        build_pair(ws, 60, [("org", 0, 60)])
        plan = _plan(ws)
        dub = plan.dub_segments[0]
        cut, hole, step = 30.0, 0.04, 0.01
        later = dub.offset_s + step  # a second offset, so the plan is not a plain delay
        first = Segment("dub", dub.start_s, cut, dub.source_start_s, offset_s=dub.offset_s, match=0.9)
        second = Segment("dub", cut + hole, dub.end_s, cut + hole + later, offset_s=later, match=0.9)
        plan.segments = [first, second] + [s for s in plan.segments if s.kind != "dub"]
        assert plan.delay_only_s is None
        said = []
        render(plan, ws.path("out.wav"), RenderOptions(codec="wav"), log=said.append)
        assert any("gap(s) or overlap(s)" in line for line in said), said
        out, rate = sf.read(ws.path("out.wav"), dtype="float32")
        dub_audio, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        out = out if out.ndim == 1 else out[:, 0]
        at = int(45 * rate)
        shift = int(round(later * rate))
        assert np.max(np.abs(out[at:at + rate] - dub_audio[at + shift:at + shift + rate])) < 2e-4


def test_every_plan_is_one_unbroken_run_of_pieces():
    with Workspace() as ws:
        build_pair(ws, 300, [("org", 0, 60), ("org", 75, 180), ("org", 182.5, 240), ("extra", 5), ("org", 240, 300)])
        plan = _plan(ws)
        for a, b in zip(plan.segments, plan.segments[1:]):
            assert abs(a.end_s - b.start_s) < 1e-6, plan.describe()
