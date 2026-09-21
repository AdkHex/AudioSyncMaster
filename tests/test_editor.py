"""The dub sync editor's back end: waveform peaks, previews of a span,
writing a plan that came back edited, and the bridge commands for them.

The app draws both tracks from ``waveform.peaks``, plays a span through
``preview_span`` and writes the edited plan through the ``dubsync``
command with ``plan`` set. What is checked here is that what it draws
and plays is what the plan says, to the sample, and that writing a track
again cannot destroy the one that was there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync import waveform  # noqa: E402
from audiosync.dubrender import RenderOptions, clip_plan, excerpt, preview_span, render  # noqa: E402
from audiosync.dubsync import DubSyncPlan, Segment  # noqa: E402
from audiosync.media import CancellationToken, Cancelled, probe  # noqa: E402
from test_bridge import run_bridge  # noqa: E402
from test_dubsync import SR, Workspace, _mkv_with_fps, build_pair  # noqa: E402


def _tone_file(ws, name="tone.wav", quiet_s=2.0, loud_s=1.0):
    """Silence, a burst, silence: the burst is the one thing to find."""
    t = np.arange(int(loud_s * SR)) / SR
    burst = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.concatenate([
        1e-4 * np.random.default_rng(1).standard_normal(int(quiet_s * SR)),
        burst,
        1e-4 * np.random.default_rng(2).standard_normal(int(quiet_s * SR)),
    ]).astype(np.float32)
    path = ws.path(name)
    sf.write(path, audio, SR)
    return path


def _hand_plan(ws, expected, total_s, fill_gain_db=0.0):
    """A plan built from build_pair's own answer, so nothing here depends
    on the engine finding it: (video_start, video_end, offset) becomes a
    stretch of dub, and the gaps between become fills."""
    pieces = []
    cursor = 0.0
    for a, b, offset in expected:
        if a > cursor:
            pieces.append(Segment("fill", cursor, a, cursor, note="dub is cut here"))
        pieces.append(Segment("dub", a, b, a + offset, offset_s=offset, match=0.5))
        cursor = b
    if cursor < total_s:
        pieces.append(Segment("fill", cursor, total_s, cursor, note="dub ends early"))
    return DubSyncPlan(
        ws.path("org.wav"), ws.path("dub.wav"), fill_gain_db=fill_gain_db,
        video_duration_s=total_s, dub_duration_s=sum(b - a for a, b, _ in expected),
        segments=pieces,
    )


# ------------------------------------------------------------------- peaks


def test_peaks_put_the_burst_in_the_bucket_where_it_plays():
    with Workspace() as ws:
        path = _tone_file(ws)
        waveform.forget()
        result = waveform.peaks(path, 0, 0.0, 5.0, 5)
        assert result["buckets"] == 5 and result["channels"] == 1
        assert result["sampleRate"] == SR
        top, bottom, rms = result["max"][0], result["min"][0], result["rms"][0]
        assert len(top) == len(bottom) == len(rms) == 5
        # The burst is a 0.5 sine: peaks at +-0.5, RMS 0.35.
        assert abs(top[2] - 0.5) < 0.01 and abs(bottom[2] + 0.5) < 0.01, (top, bottom)
        assert abs(rms[2] - 0.354) < 0.01, rms
        assert max(abs(v) for v in top[:2] + top[3:] + bottom[:2] + bottom[3:]) < 0.002
        assert abs(result["durationS"] - 5.0) < 0.01


def test_peaks_at_a_speed_are_on_the_plans_clock():
    """At speed 2 the file's second 1 falls at the video's second 2. The
    span is given on the video's clock, so the burst at file seconds 2-3
    shows at 4-6."""
    with Workspace() as ws:
        path = _tone_file(ws)
        waveform.forget()
        result = waveform.peaks(path, 0, 0.0, 10.0, 10, speed=2.0)
        loud = [i for i, v in enumerate(result["max"][0]) if v > 0.2]
        assert loud == [4, 5], result["max"][0]
        assert abs(result["durationS"] - 10.0) < 0.02


def test_peaks_beyond_the_file_are_zero_and_never_fail():
    with Workspace() as ws:
        path = _tone_file(ws)
        waveform.forget()
        result = waveform.peaks(path, 0, 4.0, 8.0, 4)
        assert result["max"][0][1:] == [0.0, 0.0, 0.0], result["max"][0]
        assert result["min"][0][1:] == [0.0, 0.0, 0.0], result["min"][0]
        # A span before the file starts is zero too.
        early = waveform.peaks(path, 0, -2.0, 0.0, 2)
        assert early["max"][0] == [0.0, 0.0]


def test_a_fine_zoom_decodes_the_span_exactly():
    """Narrower than a cache block per bucket, the samples themselves are
    read: a single 440 Hz cycle is 36 samples at 16 kHz, so a 10 ms span
    in 40 buckets shows the sine's own shape, not a block's extremes."""
    with Workspace() as ws:
        path = _tone_file(ws)
        waveform.forget()
        result = waveform.peaks(path, 0, 2.5, 2.51, 40)
        top = result["max"][0]
        bottom = result["min"][0]
        # Four and a bit cycles: the top and bottom envelopes differ bucket
        # to bucket, and each bucket's min and max are close together.
        assert max(top) > 0.45 and min(bottom) < -0.45
        assert any(t < 0.0 for t in top), "no bucket lies wholly in the trough"
        assert max(t - b for t, b in zip(top, bottom)) < 0.35


def test_the_waveform_is_decoded_once_and_kept():
    with Workspace() as ws:
        path = _tone_file(ws)
        waveform.forget()
        seen = []
        first = waveform.load(path, 0, progress=seen.append)
        assert seen and seen[-1] == 1.0
        assert waveform.is_loaded(path, 0)
        assert waveform.load(path, 0) is first
        assert first.channels == 1 and first.sample_rate == SR
        assert abs(first.duration_s - 5.0) < 0.01
        assert first.blocks == -(-first.frames // waveform.BLOCK)
        waveform.forget(path)
        assert not waveform.is_loaded(path, 0)
        assert waveform.load(path, 0) is not first


def test_a_rewritten_file_is_read_again():
    with Workspace() as ws:
        path = _tone_file(ws)
        waveform.forget()
        before = waveform.peaks(path, 0, 0.0, 5.0, 5)["max"][0]
        # Rewrite with the burst a second later; the cache must not serve the old one.
        _tone_file(ws, quiet_s=3.0, loud_s=1.0)
        os.utime(path, (os.path.getmtime(path) + 5, os.path.getmtime(path) + 5))
        after = waveform.peaks(path, 0, 0.0, 7.0, 7)["max"][0]
        # The old file is quiet across 3-4 s; the new one has its burst there.
        assert before[2] > 0.2 and before[3] < 0.01, before
        assert after[3] > 0.2 and after[1] < 0.01 and after[5] < 0.01, after


def test_a_stereo_file_gives_a_lane_per_channel():
    with Workspace() as ws:
        t = np.arange(int(2.0 * SR)) / SR
        left = 0.4 * np.sin(2 * np.pi * 220 * t)
        right = np.zeros_like(left)
        right[SR:] = 0.8 * np.sin(2 * np.pi * 660 * t[SR:])
        path = ws.path("stereo.wav")
        sf.write(path, np.stack([left, right], axis=1).astype(np.float32), SR)
        waveform.forget()
        result = waveform.peaks(path, 0, 0.0, 2.0, 2)
        assert result["channels"] == 2
        assert abs(result["max"][0][0] - 0.4) < 0.01 and abs(result["max"][0][1] - 0.4) < 0.01
        assert result["max"][1][0] < 0.01 and abs(result["max"][1][1] - 0.8) < 0.01


# ------------------------------------------------------------- clip & play


def test_clip_plan_keeps_every_piece_reading_from_where_it_read():
    with Workspace() as ws:
        expected = [(0.0, 40.0, 0.5), (50.0, 100.0, -9.5)]
        plan = _hand_plan(ws, expected, 100.0)
        clipped = clip_plan(plan, 45.0, 55.0)
        assert abs(clipped.video_duration_s - 10.0) < 1e-9
        kinds = [(s.kind, round(s.start_s, 3), round(s.end_s, 3), round(s.source_start_s, 3)) for s in clipped.segments]
        assert kinds == [("fill", 0.0, 5.0, 45.0), ("dub", 5.0, 10.0, 40.5)], kinds
        # Offsets are what they were: the clip is a window, not a re-sync.
        assert clipped.segments[1].offset_s == -9.5


def test_a_preview_plays_exactly_what_the_plan_plays_there():
    with Workspace() as ws:
        expected = build_pair(ws, 60, [("org", 0, 20), ("extra", 3.0), ("org", 25, 60)])
        plan = _hand_plan(ws, expected, 60.0)
        out = preview_span(plan, 18.0, 28.0, ws.path("preview.mp4"), with_video=False)
        assert out.endswith(".wav") and os.path.isfile(out)
        audio, rate = sf.read(out, dtype="float32")
        assert rate == SR and abs(len(audio) / SR - 10.0) < 0.002
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        org, _ = sf.read(ws.path("org.wav"), dtype="float32")
        margin = int(0.02 * SR)
        # 18-20 is dub read at 18 (offset 0); 20-25 is the original; 25-28
        # is dub read at 23 (offset -2, the 3 s of extra minus the 5 s gap).
        pieces = [
            (0.0, 2.0, dub, 18.0), (2.0, 7.0, org, 20.0), (7.0, 10.0, dub, 23.0),
        ]
        for a, b, source, at in pieces:
            lo = int(a * SR) + margin
            hi = int(b * SR) - margin
            reference = source[int(at * SR) + margin : int(at * SR) + margin + (hi - lo)]
            error = float(np.max(np.abs(audio[lo:hi] - reference)))
            assert error < 2.0 / 2 ** 23, f"preview at {a}-{b}s differs from its source by {error:.2e}"


def test_a_preview_with_the_picture_is_a_small_mp4_of_the_span():
    with Workspace() as ws:
        expected = build_pair(ws, 40, [("org", 0, 40)])
        mkv = _mkv_with_fps(ws, "24")
        plan = _hand_plan(ws, expected, 40.0)
        plan.video_path = mkv
        out = preview_span(plan, 10.0, 16.0, ws.path("preview.mp4"))
        assert out.endswith(".mp4") and os.path.isfile(out)
        info = probe(out)
        assert info.has_video and info.has_audio
        assert info.duration is not None and abs(info.duration - 6.0) < 0.3, info.duration
        # The staging WAV is gone.
        assert not os.path.exists(ws.path("preview.wav"))


# ---------------------------------------------------------- safe rewrites


def test_a_stopped_write_leaves_the_track_that_was_there():
    with Workspace() as ws:
        expected = build_pair(ws, 30, [("org", 0, 30)])
        plan = _hand_plan(ws, expected, 30.0)
        output = ws.path("out.wav")
        sf.write(output, np.zeros(SR, dtype=np.float32), SR)
        before = os.path.getsize(output)
        token = CancellationToken()
        token.cancel()
        try:
            render(plan, output, RenderOptions(codec="wav"), token=token)
        except Cancelled:
            pass
        else:
            raise AssertionError("a cancelled render did not stop")
        assert os.path.getsize(output) == before, "the previous track was overwritten"
        assert not os.path.exists(ws.path("out.part.wav")), "a partial file was left behind"


def test_a_finished_write_replaces_the_track_in_place():
    with Workspace() as ws:
        expected = build_pair(ws, 30, [("org", 0, 30)])
        plan = _hand_plan(ws, expected, 30.0)
        output = ws.path("out.wav")
        sf.write(output, np.zeros(SR, dtype=np.float32), SR)
        render(plan, output, RenderOptions(codec="wav"))
        audio, _ = sf.read(output, dtype="float32")
        assert abs(len(audio) / SR - 30.0) < 0.002
        assert sorted(os.listdir(ws.root)) == ["dub.wav", "org.wav", "out.wav"], os.listdir(ws.root)


# ------------------------------------------------------------------ bridge


def test_the_bridge_serves_peaks_and_echoes_the_request_id():
    with Workspace() as ws:
        path = _tone_file(ws)
        events, process = run_bridge([
            {"command": "waveformPeaks", "path": path, "track": 0, "startS": 0.0, "endS": 5.0,
             "buckets": 5, "requestId": "r1"},
            {"command": "waveformPeaks", "path": ws.path("missing.wav"), "startS": 0, "endS": 1, "buckets": 2,
             "requestId": "r2"},
        ])
        assert process.returncode == 0, process.stderr.decode()[:600]
        served = [e for e in events if e["type"] == "waveformPeaks"]
        assert [e.get("requestId") for e in served] == ["r1", "r2"]
        assert served[0]["max"][0][2] > 0.2 and len(served[0]["max"][0]) == 5, served[0]
        assert served[0]["channels"] == 1
        assert served[1].get("error"), served[1]
        # The first read of a file reports its progress, ending at 100.
        progress = [e for e in events if e["type"] == "waveformProgress" and e.get("requestId") == "r1"]
        assert progress and progress[-1]["percent"] == 100, progress


def test_the_bridge_reads_a_waveform_ahead_of_time():
    with Workspace() as ws:
        path = _tone_file(ws)
        events, process = run_bridge([
            {"command": "waveformBuild", "path": path, "track": 0, "requestId": "b1"},
            {"command": "waveformPeaks", "path": path, "track": 0, "startS": 0.0, "endS": 5.0,
             "buckets": 5, "requestId": "p1"},
        ])
        assert process.returncode == 0, process.stderr.decode()[:600]
        ready = [e for e in events if e["type"] == "waveformReady"]
        assert len(ready) == 1 and ready[0]["requestId"] == "b1" and not ready[0].get("error"), ready
        assert abs(ready[0]["durationS"] - 5.0) < 0.01 and ready[0]["channels"] == 1
        # Read once: the peaks that follow report no progress of their own.
        assert not [e for e in events if e["type"] == "waveformProgress" and e.get("requestId") == "p1"]
        assert [e for e in events if e["type"] == "waveformPeaks"][0]["max"][0][2] > 0.2


def test_the_bridge_writes_an_edited_plan_where_it_is_told_to():
    """An edited plan comes back with ``plan`` set: no analysis, the track
    written from those cuts to the path of the track it replaces."""
    with Workspace() as ws:
        expected = build_pair(ws, 30, [("org", 0, 30)])
        plan = _hand_plan(ws, expected, 30.0)
        # A cut placed by hand: the second half a frame later than it was.
        plan.segments = [
            Segment("dub", 0.0, 15.0, 0.0, offset_s=0.0, match=0.5),
            Segment("dub", 15.0, 30.0, 15.0 + 1.0 / 24, offset_s=1.0 / 24, match=0.5, note="set by hand"),
        ]
        output = ws.path("dub.dubsynced.wav")
        sf.write(output, np.zeros(SR, dtype=np.float32), SR)
        events, process = run_bridge([{
            "command": "dubsync", "plan": plan.to_dict(), "codec": "wav",
            "outputPath": output, "overwrite": True, "verify": False,
        }], timeout=300)
        assert process.returncode == 0, process.stderr.decode()[:600]
        done = [e for e in events if e["type"] == "dubsyncDone"]
        assert len(done) == 1 and not done[0].get("error"), done
        assert done[0]["output"]["outputPath"] == output
        # The plan came back as sent, hand note and all, not re-analysed.
        notes = [s["note"] for s in done[0]["plan"]["segments"]]
        assert notes == ["", "set by hand"], notes
        audio, _ = sf.read(output, dtype="float32")
        assert abs(len(audio) / SR - 30.0) < 0.002
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        # 20 s of the output is the dub read a frame later.
        shift = int(round(SR / 24))
        a = int(20 * SR)
        assert float(np.max(np.abs(audio[a : a + SR] - dub[a + shift : a + shift + SR]))) < 2.0 / 2 ** 23


def test_the_bridge_renders_a_preview_to_a_temporary_file():
    with Workspace() as ws:
        expected = build_pair(ws, 30, [("org", 0, 30)])
        plan = _hand_plan(ws, expected, 30.0)
        events, process = run_bridge([{
            "command": "dubsyncPreview", "plan": plan.to_dict(), "startS": 5.0, "endS": 9.0, "video": False,
        }], timeout=300)
        assert process.returncode == 0, process.stderr.decode()[:600]
        done = [e for e in events if e["type"] == "dubsyncPreviewDone"]
        assert len(done) == 1 and done[0].get("path"), done
        path = done[0]["path"]
        assert done[0]["what"] == "audio" and (done[0]["startS"], done[0]["endS"]) == (5.0, 9.0)
        try:
            assert os.path.isfile(path) and path.endswith(".wav")
            audio, rate = sf.read(path, dtype="float32")
            assert abs(len(audio) / rate - 4.0) < 0.002
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def test_the_bridge_cuts_the_players_pieces_on_a_frame():
    """The app's player asks for the picture first and takes the start it
    was cut from, then asks for the sound of exactly that span."""
    with Workspace() as ws:
        mkv = _flash_and_click(ws, fps=24, at_s=10.5)
        plan = DubSyncPlan(mkv, ws.path("dub.wav"), video_duration_s=20.0, dub_duration_s=20.0, video_fps=24.0,
                           segments=[Segment("dub", 0.0, 20.0, 0.0, offset_s=0.0, match=0.9)])
        events, process = run_bridge([
            {"command": "dubsyncPreview", "plan": plan.to_dict(), "startS": 10.3, "endS": 12.3, "what": "picture"},
            {"command": "dubsyncPreview", "plan": plan.to_dict(), "startS": 10.3, "endS": 12.3, "what": "subtitles"},
        ], timeout=300)
        assert process.returncode == 0, process.stderr.decode()[:600]
        done = [e for e in events if e["type"] == "dubsyncPreviewDone"]
        assert len(done) == 2, done
        picture, refused = done
        assert picture["what"] == "picture" and picture["path"].endswith(".mp4")
        assert 10.33 <= picture["startS"] <= 10.334 and picture["endS"] == 12.3, picture
        assert refused["path"] is None and "subtitles" in refused.get("error", ""), refused
        paths = [picture["path"]]
        events, process = run_bridge([
            {"command": "dubsyncPreview", "plan": plan.to_dict(), "startS": picture["startS"], "endS": picture["endS"], "what": "audio"},
            {"command": "dubsyncPreview", "plan": plan.to_dict(), "startS": picture["startS"], "endS": picture["endS"], "what": "original"},
        ], timeout=300)
        try:
            done = [e for e in events if e["type"] == "dubsyncPreviewDone"]
            assert [e["what"] for e in done] == ["audio", "original"], done
            for e in done:
                assert e["startS"] == picture["startS"] and e["path"].endswith(".wav"), e
                paths.append(e["path"])
            # Flash and click land together: the player will show and play them at the same instant.
            flash = [pts for pts, luma in _frame_times_and_brightness(picture["path"]) if luma > 128]
            assert len(flash) == 1 and abs(flash[0] - _click_time(done[0]["path"])) < 0.003, (flash, _click_time(done[0]["path"]))
            assert abs(_click_time(done[1]["path"]) - _click_time(done[0]["path"])) < 0.001
        finally:
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass


# ------------------------------------------------------------------ drafts


def test_the_planner_shows_its_drafts_stage_by_stage():
    """Each draft is a whole plan over the video -- dub where a stretch
    stands, fills marked not placed yet elsewhere -- and they converge on
    the plan returned: the last draft has the final stretches."""
    from audiosync.dubsync import DRAFT_NOTE, plan_dubsync

    with Workspace() as ws:
        expected = build_pair(ws, 240, [("org", 0, 100), ("extra", 4.0), ("org", 110, 240)])
        drafts = []
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None, draft=drafts.append)
        assert plan.error is None, plan.error
        assert len(drafts) >= 3, f"only {len(drafts)} drafts"
        for sketch in drafts:
            assert sketch.video_path == plan.video_path and abs(sketch.video_duration_s - 240.0) < 0.01
            cursor = 0.0
            for piece in sketch.segments:
                assert abs(piece.start_s - cursor) < 1e-6, "a draft has a hole or an overlap"
                assert piece.kind in ("dub", "fill")
                if piece.kind == "fill":
                    assert piece.note == DRAFT_NOTE
                cursor = piece.end_s
            assert abs(cursor - 240.0) < 1e-6
        # The first draft is coarse: the stretches are there, at the right offsets.
        first = [s for s in drafts[0].segments if s.kind == "dub"]
        assert len(first) >= 2
        for a, b, offset in expected:
            assert any(abs(s.offset_s - offset) < 0.05 for s in first), (offset, [s.offset_s for s in first])
        # The last draft has the final stretches at their offsets; only the
        # assembly's own touches -- the edges pulled by their uncertainty,
        # a gap bridged -- separate it from the plan.
        last = [(round(s.start_s, 1), round(s.end_s, 1), s.offset_s) for s in drafts[-1].segments if s.kind == "dub"]
        final = [(round(s.start_s, 1), round(s.end_s, 1), s.offset_s) for s in plan.segments if s.kind == "dub"]
        assert len(last) == len(final), (last, final)
        for (a0, b0, o0), (a1, b1, o1) in zip(last, final):
            assert abs(a0 - a1) <= 5.0 and abs(b0 - b1) <= 5.0 and abs(o0 - o1) < 0.005, (last, final)


def test_the_bridge_streams_drafts_per_job():
    with Workspace() as ws:
        build_pair(ws, 150, [("org", 0, 150)])
        events, process = run_bridge([{
            "command": "dubsyncBatch", "planOnly": True, "maxWorkers": 1,
            "jobs": [{"videoPath": ws.path("org.wav"), "dubPath": ws.path("dub.wav")}],
        }], timeout=600)
        assert process.returncode == 0, process.stderr.decode()[:600]
        drafts = [e for e in events if e["type"] == "dubsyncJobDraft"]
        assert drafts, "no drafts streamed"
        assert all(e["job"] == 0 and e["plan"]["segments"] for e in drafts)
        order = [e["type"] for e in events if e["type"] in ("dubsyncJobDraft", "dubsyncJobPlan")]
        assert order[-1] == "dubsyncJobPlan" and order.count("dubsyncJobPlan") == 1


# ---------------------------------------------------------- the player's excerpts


def _flash_and_click(ws, fps=24, total_s=20.0, at_s=10.5):
    """A test film: black picture with one white frame at ``at_s``, and a
    5 ms tone burst in the audio at the same instant. What a player shows
    and plays from an excerpt of it must put the two at the same time."""
    import subprocess

    from audiosync.media import ffmpeg_path

    frame = int(round(at_s * fps))
    assert abs(frame / fps - at_s) < 1e-9, "the flash must sit on a frame"
    audio = 1e-4 * np.random.default_rng(3).standard_normal(int(total_s * SR)).astype(np.float32)
    t = np.arange(int(0.005 * SR)) / SR
    start = int(round(at_s * SR))
    audio[start : start + len(t)] += 0.8 * np.sin(2 * np.pi * 1000 * t)
    sf.write(ws.path("org.wav"), audio, SR)
    sf.write(ws.path("dub.wav"), audio, SR)
    mkv = ws.path("org.mkv")
    subprocess.run(
        [
            ffmpeg_path(), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:r={fps}:d={total_s}",
            "-i", ws.path("org.wav"),
            "-vf", f"drawbox=enable='eq(n,{frame})':c=white:t=fill",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-shortest", mkv,
        ],
        check=True, capture_output=True,
    )
    return mkv


def _frame_times_and_brightness(path):
    """(pts_time, mean luma) of every frame of a file."""
    import subprocess

    from audiosync.media import ffmpeg_path

    out = subprocess.run(
        [ffmpeg_path(), "-nostdin", "-v", "error", "-i", path,
         "-vf", "signalstats,metadata=print:file=-", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    frames = []
    pts = None
    for line in out.splitlines():
        if line.startswith("frame:"):
            pts = float(line.split("pts_time:")[1].split()[0])
        elif line.startswith("lavfi.signalstats.YAVG=") and pts is not None:
            frames.append((pts, float(line.split("=")[1])))
    return frames


def _click_time(path):
    """Where the tone burst is in a file's audio, in seconds."""
    import subprocess

    from audiosync.media import ffmpeg_path

    raw = subprocess.run(
        [ffmpeg_path(), "-nostdin", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    loud = np.nonzero(np.abs(audio) > 0.3)[0]
    assert len(loud), "no burst in the audio"
    return int(loud[0]) / SR


def test_a_picture_excerpt_is_frame_accurate_against_its_sound():
    """The player draws the picture from one file and plays the sound from
    another, both cut at the same off-frame instant: the flash and the
    click must land at the same time in both, to within half a frame."""
    with Workspace() as ws:
        fps = 24
        mkv = _flash_and_click(ws, fps=fps, at_s=10.5)
        plan = DubSyncPlan(mkv, ws.path("dub.wav"), video_duration_s=20.0, dub_duration_s=20.0, video_fps=float(fps),
                           segments=[Segment("dub", 0.0, 20.0, 0.0, offset_s=0.0, match=0.9)])
        lo = 10.3  # 247.2 frames: not on the frame grid
        picture, start, end = excerpt(plan, lo, lo + 2.0, ws.path("picture.mp4"), "picture")
        # The cut was moved onto the first frame at or after 10.3 s: frame 248 at 10.333 s.
        assert 10.33 <= start <= 10.334, start
        # The end asked for is kept; only the start moves.
        assert abs(end - (lo + 2.0)) < 1e-6
        sound, sound_start, _ = excerpt(plan, start, end, ws.path("sound.mp4"), "audio")
        assert sound_start == start
        assert picture.endswith(".mp4") and sound.endswith(".wav")

        frames = _frame_times_and_brightness(picture)
        assert len(frames) >= 47, f"only {len(frames)} frames in a 2 s excerpt"
        flash = [pts for pts, luma in frames if luma > 128]
        assert len(flash) == 1, f"expected one white frame, got {flash}"
        click = _click_time(sound)
        expected = 10.5 - start
        assert abs(click - expected) < 0.001, f"click at {click:.4f}s, expected {expected:.4f}s"
        assert abs(flash[0] - click) < 0.003, (
            f"the flash is at {flash[0]:.4f}s and the click at {click:.4f}s: "
            f"{1000 * (flash[0] - click):+.1f} ms apart"
        )
        # The picture has no sound of its own.
        from audiosync.media import probe
        assert not probe(picture).has_audio
        # The outside player's file, cut from the same off-frame instant, lines up too.
        both, _, _ = excerpt(plan, lo, lo + 2.0, ws.path("both.mp4"), "both")
        flash_both = [pts for pts, luma in _frame_times_and_brightness(both) if luma > 128]
        assert len(flash_both) == 1 and abs(flash_both[0] - _click_time(both)) < 0.003, (flash_both, _click_time(both))
        # A cut already on a frame is left where it is.
        _, on_frame, _ = excerpt(plan, start, start + 1.0, ws.path("again.mp4"), "picture")
        assert abs(on_frame - start) < 1e-9, (on_frame, start)


def test_the_originals_excerpt_is_the_video_sound_at_its_own_level():
    with Workspace() as ws:
        expected = build_pair(ws, 30, [("org", 0, 30)])
        plan = _hand_plan(ws, expected, 30.0, fill_gain_db=6.0)
        out, start, end = excerpt(plan, 5.0, 9.0, ws.path("original.wav"), "original")
        assert (start, end) == (5.0, 9.0)
        audio, rate = sf.read(out, dtype="float32")
        org, _ = sf.read(ws.path("org.wav"), dtype="float32")
        assert rate == SR and abs(len(audio) / SR - 4.0) < 0.002
        margin = int(0.02 * SR)
        a = int(5.0 * SR) + margin
        n = int(4.0 * SR) - 2 * margin
        error = float(np.max(np.abs(audio[margin : margin + n] - org[a : a + n])))
        assert error < 2.0 / 2 ** 15, f"the original's excerpt differs from the original by {error:.2e}"


def test_an_unknown_excerpt_kind_is_refused():
    from audiosync.media import MediaError

    with Workspace() as ws:
        expected = build_pair(ws, 10, [("org", 0, 10)])
        plan = _hand_plan(ws, expected, 10.0)
        try:
            excerpt(plan, 0.0, 2.0, ws.path("x.mp4"), "subtitles")
        except MediaError as exc:
            assert "subtitles" in str(exc)
        else:
            raise AssertionError("an unknown kind was accepted")
