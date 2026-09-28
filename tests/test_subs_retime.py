"""Tests for frame-rate conversion and the other retime maps.

The arithmetic is checked against exact fractions: a conversion that is
"about right" at 23.976 still drifts audibly over a two-hour film.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.media import CancellationToken, ffmpeg_path  # noqa: E402
from audiosync.subs import formats, retime  # noqa: E402
from audiosync.subs.model import Cue, SubtitleDoc  # noqa: E402
from audiosync.subs.tasks import TaskContext, TaskError  # noqa: E402

FILM = Fraction(24000, 1001)


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="subsync-retime-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def _ctx(workdir):
    return TaskContext(token=CancellationToken(), progress=lambda *_: None, log=lambda *_: None, workdir=workdir)


def _doc(*times):
    return SubtitleDoc(cues=[Cue(s, e, f"line {n}") for n, (s, e) in enumerate(times)], source_format="srt")


def test_parse_fps_snaps_the_1001_rates():
    assert retime.parse_fps("24000/1001") == FILM
    for typed in (23.976, "23.976", "23.98", 23.976023976):
        assert retime.parse_fps(typed) == FILM, typed
    assert retime.parse_fps(29.97) == Fraction(30000, 1001)
    assert retime.parse_fps("59.94 fps") == Fraction(60000, 1001)
    assert retime.parse_fps(47.952) == Fraction(48000, 1001)
    assert retime.parse_fps(119.88) == Fraction(120000, 1001)
    assert retime.parse_fps(24) == 24 and retime.parse_fps("25") == 25 and retime.parse_fps(30.0) == 30
    for bad in ("", "abc", 0, -25, "1/0"):
        try:
            retime.parse_fps(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_fps_ratio_is_exact():
    assert retime.fps_ratio(25, 23.976) == float(Fraction(25) / FILM) == 25 * 1001 / 24000
    assert retime.fps_ratio("30000/1001", 29.97) == 1.0
    assert retime.format_fps(FILM) == "23.976" and retime.format_fps(25) == "25"
    assert retime.format_fps(29.97) == "29.97"


def test_pal_to_film_slows_every_time_down():
    """Subs from a 25 fps PAL release on the 23.976 film: times x 25/23.976."""
    doc = _doc((100.0, 102.0), (3600.0, 3602.0))
    out = retime.convert_fps(doc, 25, 23.976)
    ratio = 25 * 1001 / 24000
    assert abs(out.cues[0].start - 100.0 * ratio) < 1e-9
    assert abs(out.cues[1].start - 3600.0 * ratio) < 1e-9, "no drift over an hour"
    assert out.fps == float(FILM)
    back = retime.convert_fps(out, 23.976, 25)
    assert abs(back.cues[1].start - 3600.0) < 1e-9


def test_frames_mode_keeps_microdvd_frame_numbers():
    """A MicroDVD whose frames are right but was read at the wrong rate:
    re-reading at the new rate must keep every frame number exactly."""
    source = "{1}{1}25\n{250}{300}Ten seconds in\n{90000}{90050}An hour in\n"
    doc = formats.parse(source, "microdvd")
    out = retime.convert_fps(doc, 25, 23.976, mode="frames")
    rendered = formats.render(out, "microdvd").splitlines()
    assert rendered[0] == "{1}{1}23.976" and rendered[1] == "{250}{300}Ten seconds in", rendered
    assert rendered[2] == "{90000}{90050}An hour in"
    # The "from" rate does not matter in frames mode for a frame-based file.
    assert formats.render(retime.convert_fps(doc, 30, 23.976, mode="frames"), "microdvd") == "\n".join(rendered) + "\n"
    # Time mode would *also* move frames when from != the file's rate.
    moved = formats.render(retime.convert_fps(doc, 24, 23.976, mode="time"), "microdvd").splitlines()
    assert moved[1] != rendered[1]


def test_frames_mode_on_time_based_doc_equals_time_mode():
    doc = _doc((10.0, 12.0))
    a = retime.convert_fps(doc, 25, 24, mode="frames")
    b = retime.convert_fps(doc, 25, 24, mode="time")
    assert (a.cues[0].start, a.cues[0].end) == (b.cues[0].start, b.cues[0].end)


def test_two_point():
    doc = _doc((10.0, 11.0), (110.0, 111.0), (60.0, 61.0))
    out = retime.two_point(doc, (10.0, 12.0), (110.0, 116.0))
    assert abs(out.cues[0].start - 12.0) < 1e-9 and abs(out.cues[1].start - 116.0) < 1e-9
    assert abs(out.cues[2].start - 64.0) < 1e-9 and abs(out.cues[0].end - 13.04) < 1e-9
    for a, b in (((10, 1), (10, 5)), ((10, 20), (20, 5))):
        try:
            retime.two_point(doc, a, b)
        except ValueError:
            continue
        raise AssertionError("accepted degenerate points")


def test_piecewise_moves_each_scene_and_drops_cut_ones():
    doc = _doc((1.0, 2.0), (9.5, 10.5), (15.0, 16.0), (25.0, 26.0), (40.0, 41.0))
    out = retime.piecewise(doc, [(0.0, 10.0, 0.5), (10.0, 20.0, -2.0), (20.0, 30.0, None), (30.0, 50.0, 3.0)])
    got = [(round(c.start, 6), round(c.end, 6)) for c in out.cues]
    assert got == [(1.5, 2.5), (10.0, 11.0), (13.0, 14.0), (43.0, 44.0)], got


def test_offsets_that_push_before_zero_clip_or_drop():
    out = _doc((0.2, 0.4), (0.5, 2.0)).shift(-1.0)
    assert [(c.start, c.end) for c in out.cues] == [(0.0, 1.0)]


def _write_srt(path, doc):
    formats.write(doc, path, "srt")
    return path


def test_run_task_fixed_rates_with_offset():
    with Workspace() as ws:
        source = _write_srt(ws.path("Movie.en.srt"), _doc((100.0, 102.0)))
        job = {"id": "1", "task": "fps", "input": {"subtitle": {"path": source}},
               "options": {"from": 25, "to": 23.976, "mode": "time", "offsetMs": 250, "twoPoint": None},
               "output": {"dir": ws.root}}
        result = retime.run_task(job, _ctx(ws.root))
        out = result.outputs[0].path
        assert os.path.basename(out) == "Movie.23.976fps.en.srt", out
        cue = formats.read(out).cues[0]
        assert abs(cue.start - (100.0 * 25 * 1001 / 24000 + 0.25)) < 0.0015
        assert result.report["fromFps"] == 25.0 and abs(result.report["toFps"] - 23.976) < 1e-3
        assert abs(result.report["ratio"] - 25 * 1001 / 24000) < 1e-12 and result.report["offsetMs"] == 250
        assert "25 fps to 23.976 fps" in result.summary and "+250 ms" in result.summary


def test_run_task_auto_rates():
    with Workspace() as ws:
        source = _write_srt(ws.path("Movie.srt"), _doc((10.0, 11.0)))
        job = {"id": "1", "task": "fps", "input": {"subtitle": {"path": source}},
               "options": {"from": "auto", "to": 24, "mode": "time", "offsetMs": 0}, "output": {"dir": ws.root}}
        try:
            retime.run_task(job, _ctx(ws.root))
        except TaskError as exc:
            assert "frame rate" in str(exc)
        else:
            raise AssertionError("SRT has no frame rate of its own")

        # from "auto" works for MicroDVD (its header), to "auto" reads the video.
        mdvd = ws.path("Movie.sub")
        with open(mdvd, "w", encoding="utf-8") as handle:
            handle.write("{1}{1}25\n{250}{275}Hello\n")
        video = ws.path("film.mkv")
        made = subprocess.run(
            [ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=64x48:rate=24000/1001:duration=1", "-c:v", "libx264", "-preset", "ultrafast", video],
            capture_output=True,
        )
        if made.returncode != 0:
            print("  SKIP  (ffmpeg could not make a test video)")
            return
        job = {"id": "2", "task": "fps", "input": {"subtitle": {"path": mdvd}, "video": {"path": video}},
               "options": {"from": "auto", "to": "auto", "mode": "frames", "offsetMs": 0},
               "output": {"dir": ws.root, "format": "same"}}
        result = retime.run_task(job, _ctx(ws.root))
        with open(result.outputs[0].path, encoding="utf-8") as handle:
            assert handle.read().splitlines() == ["{1}{1}23.976", "{250}{275}Hello"]
        assert result.outputs[0].path.endswith("Movie.23.976fps.sub")


def test_run_task_two_point():
    with Workspace() as ws:
        source = _write_srt(ws.path("Movie.srt"), _doc((10.0, 11.0), (110.0, 111.0)))
        job = {"id": "1", "task": "fps", "input": {"subtitle": {"path": source}},
               "options": {"from": "auto", "to": "auto", "mode": "time", "offsetMs": 0,
                           "twoPoint": {"a": [10.0, 12.0], "b": [110.0, 116.0]}},
               "output": {"dir": ws.root}}
        result = retime.run_task(job, _ctx(ws.root))
        cues = formats.read(result.outputs[0].path).cues
        assert (cues[0].start, cues[1].start) == (12.0, 116.0)
        assert abs(result.report["ratio"] - 1.04) < 1e-12 and result.report["fromFps"] is None


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
