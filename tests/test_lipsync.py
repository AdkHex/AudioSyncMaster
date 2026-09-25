"""Tests for the lip-sync accuracy of the dub sync.

What these guard, each measured wrong on real material before it was fixed:
the analysis, the written track and the picture on one clock (a BluRay MKV
whose audio starts 23 ms after its picture came out 20 ms late); a quiet
pause in a dialogue-forward dub kept as the dub rather than filled with the
other language; cuts put on the picture's shot changes; a scene whose dub
drifts followed step by step; and the dialogue-free stereo difference used
as evidence wherever both tracks have one.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import audiosync.dubsync as dubsync  # noqa: E402
from audiosync.dubrender import RenderOptions, on_file_clock, render  # noqa: E402
from audiosync.dubsync import (  # noqa: E402
    ENVELOPE_RATE,
    Block,
    DubSyncPlan,
    Segment,
    _Aligner,
    _place_step,
    _settle,
    _trace_gap,
    build_envelope,
    plan_dubsync,
)
from audiosync.media import audio_lead_s, ffmpeg_path, probe  # noqa: E402
from audiosync.shots import PictureCuts, detect_cuts  # noqa: E402
from make_fixtures import _resample_linear  # noqa: E402
from test_dubsync import SR, Workspace, _bed, _dialogue, build_pair  # noqa: E402


def _ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", *args], check=True)


def _mkv(workspace, wav: str, name: str, audio_start_s: float) -> str:
    """The WAV in an MKV beside a small black picture, its audio starting
    ``audio_start_s`` into the file's clock -- as a real release's does."""
    out = workspace.path(name)
    seconds = sf.info(workspace.path(wav)).duration
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=c=black:s=64x36:r=24000/1001:d={seconds + 1:.3f}",
        "-itsoffset", f"{audio_start_s:.3f}", "-i", workspace.path(wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "flac", out,
    )
    return out


# --- one clock --------------------------------------------------------------


def test_the_analysis_counts_from_the_files_clock():
    with Workspace() as ws:
        signal = np.zeros(4 * SR, dtype=np.float32)
        signal[SR : SR + 32] = 0.9  # a click one second into the audio
        sf.write(ws.path("click.wav"), signal, SR)
        path = _mkv(ws, "click.wav", "click.mkv", 0.1)
        lead = audio_lead_s(probe(path), 0)
        assert abs(lead - 0.1) < 0.002, lead
        envelope = build_envelope(path)
        at = int(np.argmax(envelope.energy)) / ENVELOPE_RATE
        # The click is 1.0 s into the audio, which starts 0.1 s into the file.
        assert abs(at - 1.1) <= 0.004, at


def test_a_track_written_for_files_that_start_late_lands_in_sync():
    """Analysed, written and muxed on one clock: the written track, placed
    at the file's time zero as any muxer places it, sits on the original
    to within a millisecond. On two files whose audio starts 23 and 43 ms
    in, the old engine wrote it 20 ms late."""
    with Workspace() as ws:
        build_pair(ws, 120, [("org", 0, 120)])
        video = _mkv(ws, "org.wav", "org.mkv", 0.023)
        dub = _mkv(ws, "dub.wav", "dub.mkv", 0.043)
        plan = plan_dubsync(video, dub, progress=lambda p, s: None)
        assert plan.error is None, plan.error
        assert plan.timeline == "container"
        out = ws.path("out.wav")
        render(plan, out, RenderOptions(codec="wav"))
        written, rate = sf.read(out, dtype="float32")
        original, _ = sf.read(ws.path("org.wav"), dtype="float32")
        # The original on the file's clock: 23 ms of nothing, then its audio.
        original = np.concatenate([np.zeros(int(round(0.023 * rate)), dtype=np.float32), original])
        residuals = []
        reach = int(0.01 * rate)
        for at in (20, 50, 80):
            a = written[at * rate : (at + 10) * rate].astype(np.float64)
            b = original[at * rate - reach : (at + 10) * rate + reach].astype(np.float64)
            corr = np.array([np.dot(a, b[k : k + len(a)]) for k in range(2 * reach + 1)])
            residuals.append((int(np.argmax(corr)) - reach) / rate * 1000.0)
        # The bed is shared, the dialogue is not; the peak sits where the bed lines up.
        assert max(abs(r) for r in residuals) <= 1.0, residuals


def test_an_old_plan_is_moved_onto_the_files_clock():
    with Workspace() as ws:
        build_pair(ws, 20, [("org", 0, 20)])
        video = _mkv(ws, "org.wav", "org.mkv", 0.023)
        dub = _mkv(ws, "dub.wav", "dub.mkv", 0.043)
        old = DubSyncPlan(
            video, dub, video_duration_s=20.0, dub_duration_s=20.0, timeline="stream",
            segments=[Segment("dub", 0.0, 10.0, 0.0, offset_s=0.0), Segment("fill", 10.0, 20.0, 10.0)],
        )
        moved = on_file_clock(old)
        assert moved.timeline == "container"
        assert moved.segments[0].kind == "fill" and abs(moved.segments[0].end_s - 0.023) < 1e-6
        dub_piece = moved.segments[1]
        assert abs(dub_piece.start_s - 0.023) < 1e-6 and abs(dub_piece.source_start_s - 0.043) < 1e-6
        assert abs(dub_piece.offset_s - 0.020) < 1e-6
        assert abs(moved.video_duration_s - 20.023) < 1e-6
        # A plan already on the files' clock is left as it is.
        assert on_file_clock(moved) is moved


# --- quiet is not silent ----------------------------------------------------


def test_a_quiet_pause_in_the_dub_is_kept_not_filled():
    """A dialogue-forward dub ducks its bed in the pauses: -70 dBFS is quiet,
    not the silence a studio lays where a scene is missing, and the original
    must not be played over it."""
    with Workspace() as ws:
        build_pair(ws, 300, [("org", 0, 300)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        rng = np.random.default_rng(9)
        dub[int(100 * SR) : int(104 * SR)] = (3e-4 * rng.standard_normal(4 * SR)).astype(np.float32)
        sf.write(ws.path("dub.wav"), dub, SR)
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None)
        assert plan.error is None, plan.error
        assert not plan.fill_segments, plan.describe()
        assert not any("silent" in w for w in plan.warnings), plan.warnings


# --- the picture ------------------------------------------------------------


def _cuts_video(workspace, name="cuts.mkv") -> str:
    """Four solid shots of 5 s at 24 fps: cuts at exactly 5, 10 and 15 s."""
    out = workspace.path(name)
    shots = ["red", "blue", "green", "white"]
    inputs = []
    for colour in shots:
        inputs += ["-f", "lavfi", "-i", f"color=c={colour}:s=160x90:r=24:d=5"]
    graph = "".join(f"[{i}:v]" for i in range(len(shots))) + f"concat=n={len(shots)}:v=1:a=0[v]"
    _ffmpeg(*inputs, "-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", out)
    return out


def test_picture_cuts_are_found_on_the_frame():
    with Workspace() as ws:
        cuts = PictureCuts(_cuts_video(ws))
        found = cuts.within(0.0, 20.0)
        assert found is not None and len(found) == 3, found
        for got, want in zip(found, (5.0, 10.0, 15.0)):
            assert abs(got - want) <= 0.002, found
        # Where five seconds of picture could have been cut out: between
        # two cuts five seconds apart.
        pairs = cuts.pairs(0.0, 20.0, 5.0)
        assert pairs is not None and [round(p) for p in pairs] == [5, 10], pairs
        assert cuts.within(5.5, 9.5) == []


def test_motion_inside_a_shot_is_not_a_cut():
    times = np.arange(200) / 24.0
    scores = np.full(200, 0.02)
    scores[50:80] = 0.28 + 0.02 * np.sin(np.arange(30))  # a whip pan: high, but no higher than its neighbours
    scores[120] = 0.8  # a hard cut
    assert detect_cuts(times, scores) == [times[120]]


class _Flat:
    """An aligner the sound says nothing to: every offset agrees as little."""

    def __init__(self, picture):
        self.picture = picture

    def agreement(self, offset_s, lo_s, hi_s, bands=None):
        return np.zeros(int(round((hi_s - lo_s) * ENVELOPE_RATE)))


class _Cuts:
    def __init__(self, cuts):
        self.cuts = cuts

    def pairs(self, lo, hi, length):
        return [c for c in self.cuts if lo <= c <= hi and (length == 0.0 or any(abs(d - c - length) < 0.03 for d in self.cuts))]


def test_a_step_the_sound_cannot_place_goes_on_the_picture_cut():
    # The dub has two seconds the video lacks somewhere in 100-110 s; the
    # sound is flat there, the picture has one cut.
    point, how = _place_step(_Flat(_Cuts([104.2])), 100.0, 110.0, -5.0, -3.0)
    assert (round(point, 3), how) == (104.2, "picture")
    # Two cuts and a flat sound: a guess, and said so.
    point, how = _place_step(_Flat(_Cuts([101.0, 107.0])), 100.0, 110.0, -5.0, -3.0)
    assert how == "guess"
    # No picture at all: the middle, a guess.
    point, how = _place_step(_Flat(None), 100.0, 110.0, -5.0, -3.0)
    assert how == "guess" and abs(point - 105.0) < 1e-6
    # A scene cut out of the dub: the hole's two ends are both cuts, 3 s apart.
    point, how = _place_step(_Flat(_Cuts([101.0, 103.5, 106.5])), 100.0, 112.0, -3.0, -6.0)
    assert (round(point, 3), how) == (103.5, "picture")


class _Picture(_Cuts):
    available = True
    tolerance_s = 0.025

    def within(self, lo, hi):
        return [c for c in self.cuts if lo <= c <= hi]


class _Splice:
    """An aligner whose sound puts a 74 ms step at 103.2 s: the stretch
    before agrees (``before`` over ``between``) up to it, the one after
    from it; between 101.2 and 103.2 each agrees as ``between`` says."""

    def __init__(self, between):
        self.picture = _Picture([101.2, 105.3])
        self.between = between
        self.noise = np.random.default_rng(7)

    def edge_curve(self, offset_s, lo_s, hi_s):
        t = lo_s + np.arange(int(round((hi_s - lo_s) * ENVELOPE_RATE))) / ENVELOPE_RATE
        first = offset_s == -5.0
        curve = np.where(t < 101.2, 0.4 if first else 0.0, np.where(t < 103.2, self.between[0 if first else 1], 0.0 if first else 1.0))
        return curve + self.noise.normal(0.0, 0.1, len(t))


def _step_at(between):
    before = Block(0.0, 103.2, -5.0, match=0.5, end_lo=103.1, end_hi=103.3)
    after = Block(103.2, 200.0, -4.926, match=0.5, start_lo=103.1, start_hi=103.3)
    dubsync._snap_to_picture(_Splice(between), before, after, 300.0, 300.0, (90.0, 110.0), (90.0, 110.0))
    return before.end_s, after.start_s


def test_a_step_the_sound_places_mid_shot_stays_off_the_picture_cut():
    """A dub's audio spliced in the middle of a shot, 2 s after a cut, 74 ms
    at once (a real pair's, at 0:13:03): the sound says so plainly, and the
    cut is not taken over it -- that put 2 s of the scene 74 ms early. Where
    the sound cannot tell the two offsets apart, the cut still decides."""
    assert _step_at((0.4, 0.0)) == (103.2, 103.2)
    end, start = _step_at((0.2, 0.2))
    assert round(end, 3) == round(start, 3) == 101.2, (end, start)


# --- a stretch measured whole, and a gap followed through ------------------


def _aligner_for(ws):
    primary = build_envelope(ws.path("org.wav"))
    secondary = build_envelope(ws.path("dub.wav"))
    return _Aligner(primary, secondary, None, None), secondary.duration_s


def _stretch(start_s: float, end_s: float, offset_s: float) -> Block:
    return Block(start_s, end_s, offset_s, start_lo=start_s, start_hi=start_s, end_lo=end_s, end_hi=end_s)


def test_a_stretch_read_off_its_offset_goes_where_the_whole_of_it_agrees():
    """Readings scattered by a quiet scene put a real stretch 89 ms off; the
    whole stretch at once says where it is. One already there stays."""
    with Workspace() as ws:
        build_pair(ws, 120, [("extra", 3.0), ("org", 0, 120)])
        aligner, _ = _aligner_for(ws)
        block = _stretch(10.0, 100.0, 3.0 + 0.089)
        assert _settle(aligner, block)
        assert abs(block.offset_s - 3.0) < 0.002, block.offset_s
        block = _stretch(10.0, 100.0, 3.0)
        assert not _settle(aligner, block) and block.offset_s == 3.0


def test_a_scene_trimmed_twice_is_followed_through_the_gap():
    """The dub lacks a 1.5 s shot at 100 s and 1.2 s at 160 s. Given only the
    stretches either side of 90-175 s, the gap is followed through: the dub
    runs on to the first trim, sits at the middle offset between the two,
    and only the frames it lacks are left to the original."""
    with Workspace() as ws:
        build_pair(ws, 240, [("org", 0, 100), ("org", 101.5, 160), ("org", 161.2, 240)])
        aligner, dub_s = _aligner_for(ws)
        left, right = _stretch(0.0, 90.0, 0.0), _stretch(175.0, 240.0, -2.7)
        notes, warnings = [], []
        found = _trace_gap(aligner, left, right, dub_s, notes, warnings)
        assert found and len(found) == 1, (found, notes)
        middle = found[0]
        assert abs(middle.offset_s + 1.5) < 0.005, middle.offset_s
        assert abs(left.end_s - 100.0) < 0.05 and abs(middle.start_s - 101.5) < 0.05, (left.end_s, middle.start_s)
        assert abs(middle.end_s - 160.0) < 0.05 and abs(right.start_s - 161.2) < 0.05, (middle.end_s, right.start_s)
        assert notes and notes[0].startswith("followed the dub through")


def test_a_stretch_that_steps_and_steps_back_is_split_where_it_does():
    """The dub loses 50 ms at 60 s and gets them back at 120 s -- a scene
    conformed a frame and a bit out. Handed over as one stretch at the
    first offset, as the readings can leave it, it comes back as three."""
    with Workspace() as ws:
        build_pair(ws, 180, [("org", 0, 60), ("org", 60.05, 120), ("extra", 0.05), ("org", 120, 180)])
        aligner, _ = _aligner_for(ws)
        pieces = dubsync._split_block_levels(aligner, _stretch(0.0, 179.9, 0.0))
        assert pieces and len(pieces) == 3, pieces
        assert [round(p.offset_s, 3) for p in pieces] == [0.0, -0.05, 0.0], [p.offset_s for p in pieces]
        # Where a 50 ms step lies is only as sharp as the two offsets'
        # agreement differs, which here is to a fifth of a second: 50 ms
        # out for that long, where each side's own offset is exact.
        assert abs(pieces[0].end_s - 60.0) < 0.25 and abs(pieces[1].end_s - 120.0) < 0.25, [(p.start_s, p.end_s) for p in pieces]
        # One level throughout is left alone.
        assert dubsync._split_block_levels(aligner, _stretch(0.0, 59.0, 0.0)) is None


def test_no_switch_plays_the_dub_twice():
    """Two stretches that touch, the second 0.75 s back in the dub -- a
    stretch lost between them, as a wrong turn in the analysis once left
    it: the dub resumes where it left off, and the 0.75 s of video it has
    no fresh material for is the original's."""
    with Workspace() as ws:
        build_pair(ws, 120, [("org", 0, 60), ("org", 60.75, 120)])
        primary, secondary = build_envelope(ws.path("org.wav")), build_envelope(ws.path("dub.wav"))
        segments = dubsync._assemble(
            [_stretch(0.0, 60.0, 0.0), _stretch(60.0, 119.0, -0.75)],
            primary.duration_s, secondary.duration_s, primary, secondary, True, [], [],
        )
        dubs = [(round(x.start_s, 3), round(x.source_start_s, 3)) for x in segments if x.kind == "dub"]
        assert dubs == [(0.0, 0.0), (60.75, 60.0)], [(x.kind, x.start_s, x.end_s, x.source_start_s) for x in segments]
        assert [(round(x.start_s, 3), round(x.end_s, 3)) for x in segments if x.kind == "fill"][0] == (60.0, 60.75)


def test_dub_that_is_another_scene_is_not_followed():
    """Where the dub has material of its own in the gap -- 58.5 s of other
    dialogue in place of the video's 61.2 s -- its length fits the offsets
    either side, but nothing in it agrees, and nothing is claimed."""
    with Workspace() as ws:
        build_pair(ws, 240, [("org", 0, 100), ("extra", 58.5), ("org", 161.2, 240)])
        aligner, dub_s = _aligner_for(ws)
        left, right = _stretch(0.0, 90.0, 0.0), _stretch(175.0, 240.0, -2.7)
        assert _trace_gap(aligner, left, right, dub_s, [], []) is None
        assert (left.end_s, right.start_s) == (90.0, 175.0)
        # Offered only the neighbours' two offsets, each run would carry its
        # neighbour's evidence into the gap and pass on that alone; it is
        # refused because nothing agrees next to where its step would be.
        saved = dubsync._trace_levels
        dubsync._trace_levels = lambda _aligner, _lo, _hi, before, after: [before, after]
        try:
            assert _trace_gap(aligner, left, right, dub_s, [], []) is None
        finally:
            dubsync._trace_levels = saved


# --- drift inside one scene -------------------------------------------------


def test_a_drift_inside_one_scene_is_followed():
    """One scene of the dub runs 0.1% fast: held at one offset it would be
    50 ms out at its ends; followed, every part is within a few ms."""
    with Workspace() as ws:
        total = 300
        bed = _bed(total, 11)
        org = 0.6 * bed + 0.8 * _dialogue(total, seed=22)
        dub_full = 0.48 * bed + 0.8 * _dialogue(total, seed=33)
        floor = lambda n, seed: 1e-3 * np.random.default_rng(seed).standard_normal(n)  # noqa: E731
        org = (org + floor(len(org), 5)).astype(np.float32)
        dub_full = (dub_full + floor(len(dub_full), 6)).astype(np.float32)
        speed = 1.001
        middle = _resample_linear(dub_full[100 * SR : 200 * SR], 1.0 / speed)
        dub = np.concatenate([dub_full[: 100 * SR], middle, dub_full[200 * SR :]]).astype(np.float32)
        sf.write(ws.path("org.wav"), org, SR)
        sf.write(ws.path("dub.wav"), dub, SR)
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None)
        assert plan.error is None, plan.error

        def truth(v: float) -> float:
            if v < 100.0:
                return 0.0
            if v < 200.0:
                return (100.0 + (v - 100.0) / speed) - v
            return (100.0 + 100.0 / speed) - 200.0

        worst = 0.0
        for v in np.arange(105.0, 196.0, 5.0):
            piece = next(s for s in plan.segments if s.start_s <= v < s.end_s)
            assert piece.kind == "dub", plan.describe()
            worst = max(worst, abs(piece.offset_s - truth(v)) * 1000.0)
        assert worst <= 3.0, f"{worst:.1f} ms at worst inside the drifting scene\n{plan.describe()}"
        # Kept to a tenth of a millisecond.
        assert any(round(s.offset_s * 1000.0) != s.offset_s * 1000.0 for s in plan.dub_segments)


# --- the dialogue-free band -------------------------------------------------


def test_a_stereo_track_gets_the_dialogue_free_band_and_a_mono_one_does_not():
    with Workspace() as ws:
        rng = np.random.default_rng(1)
        left = rng.standard_normal(10 * SR).astype(np.float32) * 0.1
        right = rng.standard_normal(10 * SR).astype(np.float32) * 0.1
        sf.write(ws.path("wide.wav"), np.stack([left, right], axis=1), SR)
        sf.write(ws.path("narrow.wav"), np.stack([left, left], axis=1), SR)
        sf.write(ws.path("mono.wav"), left, SR)
        assert "side" in build_envelope(ws.path("wide.wav")).bands
        # Identical channels have no difference to speak of, and a mono file none at all.
        assert "side" not in build_envelope(ws.path("narrow.wav")).bands
        assert "side" not in build_envelope(ws.path("mono.wav")).bands


# --- diagnostics ------------------------------------------------------------


def test_the_plan_says_why_the_original_plays_and_how_much_dub_is_used():
    plan = DubSyncPlan(
        "v", "d", video_duration_s=100.0, dub_duration_s=95.0,
        segments=[
            Segment("fill", 0.0, 5.0, 0.0, note="dub starts late"),
            Segment("dub", 5.0, 60.0, 0.0, offset_s=-5.0, match=0.5),
            Segment("fill", 60.0, 62.0, 60.0, note="dub is cut here"),
            Segment("dub", 62.0, 100.0, 55.0, offset_s=-7.0, match=0.5),
        ],
    )
    summary = plan.to_dict()["summary"]
    assert summary["fillS"] == {"cut": 2.0, "head": 5.0}
    assert abs(summary["dubUsedS"] - 93.0) < 1e-9 and abs(summary["dubUsedShare"] - 93.0 / 95.0) < 1e-4
    assert [s["reason"] for s in plan.to_dict()["segments"]] == ["head", "", "cut", ""]
    assert "dub used: 0:01:33.000 of 0:01:35.000 (97.9%)" in plan.describe()


def test_no_part_of_the_dub_plays_twice():
    """Two stretches that read the same dub: the one that agrees with it
    keeps it, the other is cut back by exactly the overlap."""
    import audiosync.dubsync as dubsync

    class Stub:
        log = staticmethod(lambda _m: None)

        def best_agreement(self, offset_s, lo_s, hi_s):
            return 0.5 if abs(offset_s + 179.4302) < 1e-6 else 0.1

    before = dubsync.Block(2586.0, 2656.215, -133.4192)
    after = dubsync.Block(2680.596, 2741.839, -179.4302)
    placed = []
    original = dubsync._place_edge
    dubsync._place_edge = lambda aligner, a, b, p, s: placed.append((a, b))
    try:
        kept = dubsync._resolve_reuse(Stub(), [before, after], 5505.8, 5266.1)
    finally:
        dubsync._place_edge = original
    assert len(kept) == 2 and placed
    overlap = (2656.215 - 133.4192) - (2680.596 - 179.4302)
    assert abs(before.end_s - (2656.215 - overlap)) < 1e-6
    assert abs(after.start_s - 2680.596) < 1e-9
    # Nothing of the dub is read twice any more.
    assert before.end_s + before.offset_s <= after.start_s + after.offset_s + 1e-6


# --- the line check -----------------------------------------------------------


def _voiced_lines(seconds: float, starts, pitch: float, seed: int) -> np.ndarray:
    """Speech-like lines: voiced bursts at a speaking pitch with a syllable
    rhythm, starting at ``starts``, over a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    out = 0.002 * rng.standard_normal(len(t))
    for start in starts:
        length = rng.uniform(0.4, 1.8)
        span = (t >= start) & (t < start + length)
        tt = t[span] - start
        syllables = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3.5, 5.0) * tt))
        voice = sum(np.sin(2 * np.pi * pitch * k * tt) / k for k in range(1, 8))
        out[span] += 0.2 * syllables * voice
    return out.astype(np.float32)


def test_the_line_check_reads_where_the_dub_s_lines_sit():
    from audiosync.linecheck import line_check

    with Workspace() as ws:
        rng = np.random.default_rng(4)
        # Phrases of half a second to two, pauses of a third of one to three.
        starts = np.cumsum(rng.uniform(0.7, 4.0, 120))
        starts = starts[starts < 175.0]
        original = _voiced_lines(180.0, starts, 140.0, 1)
        for name, shift in (("same", 0.0), ("late", 0.1)):
            dub = _voiced_lines(180.0, starts + shift + rng.uniform(-0.03, 0.03, len(starts)), 210.0, 2)
            sf.write(ws.path("org.wav"), np.stack([original, original], axis=1), SR)
            sf.write(ws.path(f"{name}.wav"), np.stack([dub, dub], axis=1), SR)
            plan = DubSyncPlan(
                ws.path("org.wav"), ws.path(f"{name}.wav"), video_duration_s=180.0, dub_duration_s=180.0,
                segments=[Segment("dub", 0.0, 180.0, 0.0, offset_s=0.0)],
            )
            result = line_check(plan, ws.path(f"{name}.wav"))
            overall = result.to_dict()["overallMs"]
            assert result.judged, result.describe()
            assert abs(overall - shift * 1000.0) <= 15.0, (name, overall, result.describe())
