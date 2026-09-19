"""Tests for laying a cut dub onto its video.

The material is synthetic and the truth is exact: a shared music-and-effects
bed with a different "performance" on each side, and a dub assembled from
pieces of the video's timeline with known cuts, insertions and lead-ins. The
engine has to recover every stretch's offset to the millisecond and every
cut to within what the material allows, then write a track whose pieces are
the sources to the sample.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync.dubrender import RenderOptions, render  # noqa: E402
from audiosync.dubsync import (  # noqa: E402
    Block,
    _Aligner,
    DubSyncPlan,
    _level_runs,
    best_path,
    build_envelope,
    ncc_lags,
    plan_dubsync,
    verify_output,
)
from make_fixtures import _resample_linear, _speechlike  # noqa: E402

SR = 16000


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="audiosync-dubsync-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def _bed(seconds: float, seed: int) -> np.ndarray:
    """Music-and-effects-like: dense short transients, as a real stem has."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    noise = np.convolve(rng.standard_normal(n), np.ones(8) / 8.0, mode="same")
    envelope = np.zeros(n)
    pos = 0
    while pos < n:
        burst = int(rng.uniform(0.05, 0.3) * SR)
        gap = int(rng.uniform(0.03, 0.25) * SR)
        envelope[pos : min(n, pos + burst)] = rng.uniform(0.2, 1.0)
        pos += burst + gap
    envelope = np.convolve(envelope, np.ones(64) / 64.0, mode="same")
    signal = noise * envelope
    return (signal / np.max(np.abs(signal)) * 0.7).astype(np.float32)


def build_pair(workspace, total_s, pieces, bed_gain=0.6, dial_gain=0.8):
    """Write org.wav and dub.wav; return the expected stretches.

    ``pieces`` describe the dub's timeline in order: ``("org", a, b)`` is
    the video's material from a to b seconds, ``("extra", seconds)`` is
    material only the dub has. The expected stretches are
    (video_start, video_end, offset) with offset = dub time - video time.
    """
    bed = _bed(total_s, 11)
    org = (bed_gain * bed + dial_gain * _speechlike(total_s, SR, seed=22)).astype(np.float32)
    dub_full = (bed_gain * 0.8 * bed + dial_gain * _speechlike(total_s, SR, seed=33)).astype(np.float32)
    parts, expected = [], []
    cursor, seed = 0.0, 100
    for piece in pieces:
        if piece[0] == "org":
            _, a, b = piece
            parts.append(dub_full[int(round(a * SR)) : int(round(b * SR))])
            expected.append((a, b, cursor - a))
            cursor += b - a
        else:
            seed += 1
            parts.append(_speechlike(piece[1], SR, seed=seed) * 0.5)
            cursor += piece[1]
    sf.write(workspace.path("org.wav"), org, SR)
    sf.write(workspace.path("dub.wav"), np.concatenate(parts).astype(np.float32), SR)
    return expected


def _plan(workspace, **kwargs):
    plan = plan_dubsync(workspace.path("org.wav"), workspace.path("dub.wav"), progress=lambda p, s: None, **kwargs)
    assert plan.error is None, plan.error
    return plan


def _check(plan, expected, edge_tolerance_s, offset_tolerance_s=0.003):
    dubs = plan.dub_segments
    assert len(dubs) == len(expected), (
        f"{len(dubs)} stretches of dub, expected {len(expected)}:\n{plan.describe()}"
    )
    for got, (start, end, offset) in zip(dubs, expected):
        assert abs(got.offset_s - offset) <= offset_tolerance_s, (
            f"offset {got.offset_s:+.3f}s for the stretch at {start}s, expected {offset:+.3f}s"
        )
        assert abs(got.start_s - start) <= edge_tolerance_s, (
            f"stretch starts at {got.start_s:.3f}s, expected {start:.3f}s"
        )
        assert abs(got.end_s - end) <= edge_tolerance_s, (
            f"stretch ends at {got.end_s:.3f}s, expected {end:.3f}s"
        )


# --- the pieces ------------------------------------------------------------


def test_ncc_finds_a_template_where_it_is():
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(5000)
    template = signal[1200:1500].copy()
    row = ncc_lags(template, signal)
    assert len(row) == 5000 - 300 + 1
    assert int(np.argmax(row)) == 1200
    assert abs(row[1200] - 1.0) < 1e-9, "a template found in itself correlates at exactly one"
    assert np.max(np.abs(np.delete(row, 1200))) < 0.5, "nowhere else comes close"


def test_the_path_follows_the_offsets_and_marks_the_jump():
    scores = np.zeros((10, 50), dtype=np.float32)
    scores[:5, 10] = 10.0
    scores[5:, 30] = 10.0
    path, starts = best_path(scores)
    assert path == [10] * 5 + [30] * 5
    assert starts == [True, False, False, False, False, True, False, False, False, False]


def test_the_path_rests_on_nothing_for_noise():
    rng = np.random.default_rng(0)
    path, _starts = best_path(rng.standard_normal((40, 50)).astype(np.float32))
    assert all(column is None for column in path), "pure noise should place no window anywhere"


def test_one_loud_window_cannot_buy_a_jump():
    """A single window's peak at a random offset must not split the file."""
    scores = np.zeros((20, 50), dtype=np.float32)
    scores[:, 10] = 4.0
    scores[9, 40] = 12.0  # a coincidence twelve noise units high, once
    path, _ = best_path(scores)
    assert path == [10] * 20, "one window's fluke moved the path"


# --- planning ----------------------------------------------------------------


def test_cuts_are_found_with_exact_offsets():
    """The case this exists for: a dub with scenes missing and a longer logo.

    The offsets are what a listener would hear as sync, and they come out
    to the millisecond. The cuts come out where the shared bed lets them:
    this seed's bed is digitally silent for the fifth of a second before
    the first cut, so that edge is placed where the bed stops, and the
    first 0.7s of the file have almost no bed at all.
    """
    with Workspace() as ws:
        expected = build_pair(ws, 300, [
            ("extra", 0.8), ("org", 0, 60), ("org", 75, 180), ("org", 182.5, 240),
            ("extra", 5), ("org", 240, 297),
        ])
        plan = _plan(ws)
        _check(plan, expected, edge_tolerance_s=0.8)
        dubs = plan.dub_segments
        # The cuts the bed can see are placed to within a few milliseconds.
        assert abs(dubs[1].end_s - 180.0) <= 0.01, f"cut at 180s placed at {dubs[1].end_s:.3f}"
        assert abs(dubs[2].start_s - 182.5) <= 0.01, f"resume at 182.5s placed at {dubs[2].start_s:.3f}"
        assert abs(dubs[2].end_s - 240.0) <= 0.01 and abs(dubs[3].start_s - 240.0) <= 0.01, (
            "the dub-only insertion at 240s should leave the video's timeline continuous"
        )
        fills = plan.fill_segments
        assert any(abs(f.start_s - 60) < 0.8 and abs(f.end_s - 75) < 0.8 for f in fills), "the 15s cut was not filled"
        assert any(abs(f.start_s - 180) < 0.05 and abs(f.end_s - 182.5) < 0.05 for f in fills), "the 2.5s cut was not filled"
        assert fills[-1].note == "past dub end" and abs(fills[-1].start_s - 297.0) < 0.05
        # The first 0.7s could not be placed by the bed, but the dub has
        # material there at the offset the rest of the stretch sits at and
        # is not silent in it: it is kept, not filled, and the plan says so.
        assert dubs[0].start_s == 0.0, plan.describe()
        assert any(n.startswith("kept the dub across 0:00:00.000") for n in plan.notes), plan.notes
        assert abs(plan.filled_s - 20.5) < 0.5, plan.filled_s
        assert plan.speed == 1.0


def test_a_short_stretch_between_two_cuts_is_recovered():
    """Twelve seconds of dub between two cuts fill no coarse window on their
    own; they are found by searching the gap for them."""
    with Workspace() as ws:
        expected = build_pair(ws, 240, [("org", 0, 60), ("org", 80, 92), ("org", 120, 240)])
        _check(_plan(ws), expected, edge_tolerance_s=0.05)


def test_a_one_frame_step_inside_a_stretch_is_split():
    with Workspace() as ws:
        expected = build_pair(ws, 240, [("org", 0, 120), ("org", 120.042, 240)])
        _check(_plan(ws), expected, edge_tolerance_s=0.05, offset_tolerance_s=0.002)


def test_a_dub_with_an_extra_intro_is_offset_not_filled():
    with Workspace() as ws:
        expected = build_pair(ws, 200, [("extra", 30), ("org", 0, 200), ("extra", 10)])
        plan = _plan(ws)
        _check(plan, expected, edge_tolerance_s=0.1)
        # This seed's bed fades out over its last frames, which can leave a
        # sliver of the original at the tail; nothing more than that.
        assert plan.filled_s < 0.1, f"nothing is missing from this dub:\n{plan.describe()}"


def test_an_identical_dub_is_one_stretch():
    with Workspace() as ws:
        expected = build_pair(ws, 200, [("org", 0, 200)])
        plan = _plan(ws)
        _check(plan, expected, edge_tolerance_s=0.1)
        assert plan.filled_s < 0.1, plan.describe()


def test_a_dub_that_skips_material_keeps_the_video_continuous():
    with Workspace() as ws:
        expected = build_pair(ws, 240, [("org", 0, 120), ("extra", 3.0), ("org", 120, 240)])
        plan = _plan(ws)
        _check(plan, expected, edge_tolerance_s=0.1)
        assert plan.filled_s < 0.1, plan.describe()


def test_a_replaced_scene_is_filled_and_the_replacement_dropped():
    with Workspace() as ws:
        expected = build_pair(ws, 300, [("org", 0, 100), ("extra", 20), ("org", 130, 300)])
        plan = _plan(ws)
        _check(plan, expected, edge_tolerance_s=0.8)
        fills = [f for f in plan.fill_segments if f.length_s > 5]
        assert len(fills) == 1 and abs(fills[0].start_s - 100) < 0.05 and abs(fills[0].end_s - 130) < 0.05


def test_a_silent_stretch_in_the_dub_is_filled():
    with Workspace() as ws:
        build_pair(ws, 300, [("org", 0, 300)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        dub[int(100 * SR) : int(140 * SR)] = 0.0
        sf.write(ws.path("dub.wav"), dub, SR)
        plan = _plan(ws)
        fills = [f for f in plan.fill_segments if f.length_s > 5]
        assert len(fills) == 1, plan.describe()
        assert abs(fills[0].start_s - 100.0) < 0.05 and abs(fills[0].end_s - 140.0) < 0.05
        assert any("silent" in w for w in plan.warnings)


def test_a_track_already_in_sync_is_reported_as_one_stretch():
    """The check a user makes: sync a dub, then feed the synced track back in
    as the dub. It sits on the video at 0 ms throughout, so it must come
    back as one stretch at 0 ms with nothing filled and nothing to warn
    about. It did not: a 12-second head at one offset was merged with the
    hour after it at another, and the hour was played at the head's
    offset; silences in both tracks were filled with silence and reported
    as cuts; the silent leader and tail were reported as the dub starting
    late and ending early."""
    with Workspace() as ws:
        build_pair(ws, 300, [("extra", 0.8), ("org", 0, 60), ("org", 75, 180), ("org", 182.5, 240), ("org", 240, 297)])
        first = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None)
        assert first.error is None and first.filled_s > 15, first.describe()
        render(first, ws.path("synced.wav"), RenderOptions(codec="wav"))
        # Both tracks are quiet for a few seconds inside the film, as at a
        # fade to black, and at the very end.
        for name in ("org.wav", "synced.wav"):
            audio, _ = sf.read(ws.path(name), dtype="float32")
            audio[int(150 * SR) : int(153 * SR)] = 0.0
            audio[int(298 * SR) :] = 0.0
            sf.write(ws.path(name), audio, SR)
        again = plan_dubsync(ws.path("org.wav"), ws.path("synced.wav"), progress=lambda p, s: None)
        assert again.error is None, again.error
        assert len(again.dub_segments) == 1, again.describe()
        assert not again.fill_segments, again.describe()
        only = again.dub_segments[0]
        assert only.start_s == 0.0 and abs(only.end_s - 300.0) < 0.01, again.describe()
        assert abs(only.offset_s) <= 0.002, again.describe()
        assert not again.warnings, again.warnings


def test_a_step_of_a_frame_is_followed_but_never_filled():
    """A dub conformed a frame apart in one scene steps by 30 ms and back.
    Both steps are real and are followed, so the scene sits in lip-sync;
    nothing is filled, because nothing is missing: the frame the dub lacks
    at the second step is bridged. Before, the uncertain edges either side
    of the scene were pulled inward and the gaps filled with the original
    -- the other language over a scene the dub had -- for a step of 34 ms."""
    with Workspace() as ws:
        expected = build_pair(ws, 300, [("org", 0, 100), ("extra", 0.03), ("org", 100, 160), ("org", 160.03, 300)])
        plan = _plan(ws)
        dubs = plan.dub_segments
        assert [round(d.offset_s, 3) for d in dubs] == [0.0, 0.03, 0.0], plan.describe()
        assert not plan.fill_segments, plan.describe()
        assert abs(dubs[1].start_s - 100.0) < 1.0 and abs(dubs[1].end_s - 160.0) < 1.0, plan.describe()
        assert abs(expected[1][2] - 0.03) < 1e-9


def test_a_pause_in_both_tracks_is_not_a_cut():
    """Silence laid where the dub lacks a scene is a gap; silence where the
    video is silent too is a pause. Only the first is filled and reported."""
    with Workspace() as ws:
        build_pair(ws, 300, [("org", 0, 300)])
        for name, spans in (("org.wav", [(100, 104)]), ("dub.wav", [(100, 104), (200, 204)])):
            audio, _ = sf.read(ws.path(name), dtype="float32")
            for lo, hi in spans:
                audio[int(lo * SR) : int(hi * SR)] = 0.0
            sf.write(ws.path(name), audio, SR)
        plan = _plan(ws)
        fills = plan.fill_segments
        assert len(fills) == 1, plan.describe()
        assert abs(fills[0].start_s - 200.0) < 0.05 and abs(fills[0].end_s - 204.0) < 0.05, plan.describe()
        assert fills[0].note == "dub is silent here", fills[0].note
        assert len(plan.dub_segments) == 2, plan.describe()
        silent = [w for w in plan.warnings if "silent" in w]
        assert len(silent) == 1 and silent[0].startswith("the dub is silent across 0:03:19.9"), plan.warnings


def test_a_step_the_whole_piece_does_not_bear_out_is_not_split():
    """Readings are ten-second windows and three of them can agree on a
    wrong peak; the piece they cut out is then asked, whole, which offset
    it agrees with. Here a piece a frame off its neighbour agrees far
    better at the neighbour's offset, so the step is not real."""
    with Workspace() as ws:
        build_pair(ws, 300, [("org", 0, 300)])
        envelopes = {}
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None, envelopes=envelopes)
        assert plan.error is None
        aligner = _Aligner(envelopes["primary"], envelopes["secondary"], None, lambda m: None)
        before = Block(0.0, 200.0, 0.0)
        wrong = Block(200.0, 260.0, 0.034)
        assert not aligner.step_is_real(before, wrong)
        # And a real step is: the same piece with its material really a
        # frame off, made by asking for the offset the dub actually sits at.
        assert aligner.step_is_real(Block(0.0, 200.0, 0.034), Block(200.0, 260.0, 0.0))


def _bedless_pair(ws, total_s=600, lo_s=100.0, hi_s=280.0):
    """A dub identical to the video's timeline, but with no shared bed at all
    -- dialogue over nothing -- across [lo_s, hi_s). Three minutes of it, so
    the alignment path gives up on the stretch rather than coasting through:
    a minute of it is simply ridden over and decides nothing."""
    build_pair(ws, total_s, [("org", 0, total_s)])
    dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
    bed = _bed(total_s, 11)
    dub[int(lo_s * SR) : int(hi_s * SR)] -= 0.6 * 0.8 * bed[int(lo_s * SR) : int(hi_s * SR)]
    sf.write(ws.path("dub.wav"), dub, SR)


def test_a_weakly_correlating_stretch_at_the_same_offset_is_kept():
    """Three minutes where the dub has no shared bed at all is still the dub,
    and must not be replaced with the original -- but the plan says it was
    kept on trust, so the reader can check that spot."""
    with Workspace() as ws:
        _bedless_pair(ws)
        plan = _plan(ws)
        assert not [f for f in plan.fill_segments if f.length_s > 5], plan.describe()
        assert len(plan.dub_segments) == 1, plan.describe()
        # Said as a note, not a warning: nothing was changed and there is
        # nothing to check, only something the reader may like to know.
        assert any("kept the dub across 0:01:40" in n for n in plan.notes), plan.notes
        assert not any("kept the dub" in w for w in plan.warnings), plan.warnings


def test_a_replaced_unmatched_stretch_says_so():
    """Asked to fill such a stretch instead, the plan must not call it a cut:
    nothing showed the dub lacks the scene, only that it could not be
    matched, and the reader judging the fill needs to know which claim is
    being made."""
    with Workspace() as ws:
        _bedless_pair(ws)
        plan = _plan(ws, keep_unmatched_dub=False)
        long_fills = [f for f in plan.fill_segments if f.length_s > 5]
        assert len(long_fills) == 1, plan.describe()
        assert long_fills[0].note == "dub audible but did not correlate; replaced", long_fills[0].note
        assert abs(long_fills[0].start_s - 100.0) < 1.0 and abs(long_fills[0].end_s - 280.0) < 1.0, plan.describe()
        # The note is reserved for this case: a genuine cut still reads as one.
        assert not any(f.note == "dub is cut here" for f in plan.fill_segments), plan.describe()
        assert not any("kept the dub" in n for n in plan.notes + plan.warnings)


def test_unrelated_audio_is_refused():
    with Workspace() as ws:
        sf.write(ws.path("org.wav"), _speechlike(200, SR, seed=1), SR)
        sf.write(ws.path("dub.wav"), _speechlike(190, SR, seed=2), SR)
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None)
        assert plan.error, "two unrelated files must not produce a plan"
        assert not plan.dub_segments


def test_level_runs_find_alternating_offsets_but_not_a_ramp():
    """The readings of a dub conformed scene by scene alternate between two
    offsets in runs; those are steps. The readings of a dub at the wrong
    speed climb steadily; that is a ramp, and must not be chopped into
    steps -- doing so shredded the stretches the speed is read from."""
    positions = [i * 15.0 for i in range(20)]
    alternating = [104.0] * 6 + [-21.0] + [103.0] + [0.0] * 5 + [104.0] * 2 + [0.0] * 5
    runs = _level_runs(positions, alternating)
    assert [round(level) for _, _, level in runs] == [104, 0, 104, 0], runs
    assert runs[0][0] == 0.0 and runs[1][0] == 120.0, "runs start at their first reading"
    # A lone reading at another level is a reading that locked onto a repeat,
    # not a scene, and does not break the run it sits in.
    assert _level_runs(positions, [0.0] * 9 + [300.0] + [0.0] * 10) == []
    assert _level_runs(positions, [i * 15.0 for i in range(20)]) == [], "a ramp is not steps"
    # A staircase of flat treads is steps, however short the treads: every
    # reading on a tread agrees with its neighbours to the reading, which a
    # ramp's never do.
    assert [round(level) for _, _, level in _level_runs(positions, [50.0 * (i // 4) for i in range(20)])] == [0, 50, 100, 150, 200]
    # A step on top of a ramp is still a step once the ramp is taken off.
    ramp_step = [i * 15.0 + (100.0 if i >= 10 else 0.0) for i in range(20)]
    assert len(_level_runs(positions, ramp_step)) == 2, "a ramp with one step is two levels"


def test_a_scene_by_scene_conform_error_is_followed():
    """Real dubs are sometimes conformed a scene at a time, and a scene can
    land a few frames late; the readings inside one long stretch then
    alternate between two offsets. Each scene has to be played at its own
    offset, and a 100ms error left in place is visible lip-sync."""
    with Workspace() as ws:
        # Four scenes at alternating offsets: 0-80s straight, 80-150s 104ms
        # late in the dub, 150-230s straight again (the dub skips the 104ms
        # it gained), 230-300s late again.
        expected = build_pair(ws, 300, [
            ("org", 0, 80), ("extra", 0.104), ("org", 80, 150),
            ("org", 150.104, 230), ("extra", 0.104), ("org", 230, 300),
        ])
        plan = _plan(ws)
        dubs = plan.dub_segments
        offsets = [round(d.offset_s, 3) for d in dubs]
        assert offsets == [0.0, 0.104, 0.0, 0.104], f"{offsets}\n{plan.describe()}"
        # The video is continuous across the dub's insertions and lacks
        # only a tenth of a second at 150s, so next to nothing is filled:
        # the bed-less first 0.7s of this seed, and slivers at the steps.
        assert plan.filled_s < 1.5, plan.describe()
        for index, (got, (start, end, _offset)) in enumerate(zip(dubs, expected)):
            # The first 0.7s of this seed's bed is silent; every other edge
            # is a step the bed can see.
            if index > 0:
                assert abs(got.start_s - start) <= 0.25, plan.describe()
            assert abs(got.end_s - end) <= 0.25, plan.describe()


def test_a_small_speed_difference_is_detected_from_the_drift():
    """A dub timed against a 23.976fps master on a 24fps video runs 0.1%
    slow. The coarse pass hardly notices, so nothing fails outright; but
    every stretch's offset walks a millisecond a second, and measured at
    one offset the ends of a long stretch are a quarter of a second out.
    The drift is read from the coarse path and the speed cancelled."""
    with Workspace() as ws:
        expected = build_pair(ws, 600, [("org", 0, 250), ("org", 265, 600)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        sf.write(ws.path("dub.wav"), _resample_linear(dub, 1001.0 / 1000.0), SR)
        plan = _plan(ws)
        assert abs(plan.speed - 1000.0 / 1001.0) < 1e-6, f"speed {plan.speed:.6f}, expected {1000.0 / 1001.0:.6f}"
        assert not any("drifts" in w for w in plan.warnings), plan.warnings
        dubs = plan.dub_segments
        assert len(dubs) == 2, plan.describe()
        # Offsets are on the dub's corrected timeline, which is the video's.
        for got, (start, end, offset) in zip(dubs, expected):
            assert abs(got.offset_s - offset) <= 0.01, f"offset {got.offset_s:+.3f}, expected {offset:+.3f}"
            assert abs(got.start_s - start) <= 0.3 and abs(got.end_s - end) <= 0.3, plan.describe()


def test_the_dub_is_found_inside_a_quiet_gap():
    """Where the shared bed is faint, thirty-second windows at the coarse
    pass's resolution see nothing and the stretch becomes a gap; the gap
    search at the envelope's full resolution, with only the offsets the
    neighbours allow, still finds it -- and finds the cut inside it."""
    with Workspace() as ws:
        expected = build_pair(ws, 480, [("org", 0, 100), ("org", 110, 300), ("org", 303, 480)])
        # Fade the shared bed right down between 100s and 300s of the video,
        # on both sides, so only the fine search can see the dub there.
        bed = _bed(480, 11)
        quiet = slice(int(100 * SR), int(300 * SR))
        for name in ("org.wav", "dub.wav"):
            audio, _ = sf.read(ws.path(name), dtype="float32")
            scale = 0.6 * (0.8 if name == "dub.wav" else 1.0)
            if name == "dub.wav":
                # The dub's timeline: 0-100 -> 0-100, then 110-300 -> 100-290.
                audio[int(100 * SR) : int(290 * SR)] -= 0.8 * scale * bed[int(110 * SR) : int(300 * SR)]
            else:
                audio[quiet] -= 0.8 * scale * bed[quiet]
            sf.write(ws.path(name), audio, SR)
        logs = []
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None, log=logs.append)
        assert plan.error is None, plan.error
        assert any("found dub for" in line for line in logs), "the gap search was never needed; the fixture is not quiet enough"
        dubs = plan.dub_segments
        assert len(dubs) == 3, plan.describe()
        for got, (start, end, offset) in zip(dubs, expected):
            assert abs(got.offset_s - offset) <= 0.01, f"offset {got.offset_s:+.3f}, expected {offset:+.3f}\n{plan.describe()}"
        # The quiet stretch is the dub, at the right offset, across most of
        # its length. Its edges are as precise as the faint bed allows, and
        # where they are not, the dub stops short of the cut rather than
        # risking the far side of it: that is the trim, and it is deliberate.
        quiet_dub = dubs[1]
        assert quiet_dub.start_s <= 140.0 and quiet_dub.end_s >= 270.0, plan.describe()
        assert plan.filled_s < 60.0, plan.describe()


def test_a_pal_dub_is_brought_to_speed():
    """A dub timed at 25fps against a 23.976fps video, with a cut as well.

    The durations cannot say what the speed is -- the cut moves them too --
    so every standard conversion is tried on the audio itself.
    """
    with Workspace() as ws:
        expected = build_pair(ws, 300, [("org", 0, 120), ("org", 150, 300)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        ratio = (24000.0 / 1001.0) / 25.0
        sf.write(ws.path("dub.wav"), _resample_linear(dub, ratio), SR)
        plan = _plan(ws)
        assert abs(plan.speed - 1.0 / ratio) < 1e-6, f"speed {plan.speed:.6f}, expected {1 / ratio:.6f}"
        _check(plan, expected, edge_tolerance_s=0.8, offset_tolerance_s=0.02)


def test_a_two_hour_film_is_planned_quickly():
    """Five cuts totalling ten minutes across two hours, as the real case is."""
    with Workspace() as ws:
        expected = build_pair(ws, 7200, [
            ("extra", 12.0), ("org", 0, 1230), ("org", 1290, 2500), ("org", 2650, 4000),
            ("org", 4180, 5900), ("org", 5930, 7150),
        ])
        import time
        started = time.monotonic()
        plan = _plan(ws)
        elapsed = time.monotonic() - started
        _check(plan, expected, edge_tolerance_s=0.3)
        assert elapsed < 120, f"planning took {elapsed:.0f}s"


# --- rendering ---------------------------------------------------------------


def test_render_is_sample_exact_and_the_length_of_the_video():
    with Workspace() as ws:
        build_pair(ws, 300, [
            ("extra", 0.8), ("org", 0, 60), ("org", 75, 180), ("org", 182.5, 240),
            ("extra", 5), ("org", 240, 297),
        ])
        envelopes = {}
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None, envelopes=envelopes)
        assert plan.error is None
        output = ws.path("out.wav")
        result = render(plan, output, RenderOptions(codec="wav"))
        out, rate = sf.read(output, dtype="float32")
        org, _ = sf.read(ws.path("org.wav"), dtype="float32")
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        assert rate == SR
        assert len(out) == len(org), f"output is {len(out)} samples, the video {len(org)}"
        assert result.clipped_samples == 0
        gain = 10 ** (plan.fill_gain_db / 20.0)
        margin = int(0.015 * SR)  # clear of the crossfades
        for segment in plan.segments:
            a = int(round(segment.start_s * SR)) + margin
            b = int(round(segment.end_s * SR)) - margin
            if b <= a:
                continue
            source = int(round(segment.source_start_s * SR)) + margin
            reference = dub[source : source + (b - a)] if segment.kind == "dub" else org[source : source + (b - a)] * gain
            n = min(len(reference), b - a)
            error = float(np.max(np.abs(out[a : a + n] - reference[:n])))
            assert error < 2.0 / 2 ** 23, f"{segment.kind} at {segment.start_s}s differs from its source by {error:.2e}"

        finished = build_envelope(output)
        verification = verify_output(envelopes["primary"], finished, plan)
        assert verification.worst_ms is not None and verification.worst_ms <= 2.0, verification.describe()
        assert not verification.stretches


def test_a_rendered_pal_dub_verifies_against_the_video():
    with Workspace() as ws:
        build_pair(ws, 300, [("org", 0, 120), ("org", 150, 300)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        sf.write(ws.path("dub.wav"), _resample_linear(dub, (24000.0 / 1001.0) / 25.0), SR)
        envelopes = {}
        plan = plan_dubsync(ws.path("org.wav"), ws.path("dub.wav"), progress=lambda p, s: None, envelopes=envelopes)
        assert plan.error is None and plan.speed != 1.0
        output = ws.path("out.flac")
        render(plan, output, RenderOptions(codec="flac"))
        verification = verify_output(envelopes["primary"], build_envelope(output), plan)
        measured = [s for s in verification.spots if s.residual_ms is not None]
        assert measured, verification.describe()
        assert verification.worst_ms <= 5.0, verification.describe()


def test_the_plan_round_trips_through_json():
    with Workspace() as ws:
        build_pair(ws, 200, [("org", 0, 80), ("org", 100, 200)])
        plan = _plan(ws)
        copy = DubSyncPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        assert copy.to_dict() == plan.to_dict()
        assert [s.to_dict() for s in copy.segments] == [s.to_dict() for s in plan.segments]


def test_the_report_names_every_piece():
    with Workspace() as ws:
        build_pair(ws, 200, [("org", 0, 80), ("org", 100, 200)])
        text = _plan(ws).describe()
        assert "dub is cut here" in text
        assert text.count("\n  dub ") == 2 and text.count("\n  fill") >= 1


if __name__ == "__main__":
    # Write one synthetic pair to a directory, for a caller that is not Python:
    # the Rust end-to-end test builds its input this way. Usage:
    #     python tests/test_dubsync.py DIR
    # writes DIR/org.wav and DIR/dub.wav and prints the expected stretches.
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("usage: test_dubsync.py DIR", file=sys.stderr)
        sys.exit(2)
    os.makedirs(target, exist_ok=True)

    class _Dir:
        def path(self, name):  # noqa: D401 - mirrors Workspace
            return os.path.join(target, name)

    for stretch in build_pair(_Dir(), 300, [("extra", 0.8), ("org", 0, 60), ("org", 75, 180), ("org", 182.5, 240), ("extra", 5), ("org", 240, 297)]):
        print("%.1f-%.1f %+.3f" % stretch)
