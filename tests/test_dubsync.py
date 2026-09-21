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
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync.dubrender import RenderOptions, render  # noqa: E402
from audiosync.dubsync import (  # noqa: E402
    ENVELOPE_RATE,
    Block,
    _Aligner,
    DubSyncPlan,
    _continuing_candidate,
    _level_runs,
    _measure,
    _nearest_peak,
    _peak_dominance,
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
    """Music-and-effects-like: dense short transients, as a real stem has.

    Broadband, as a real bed is: an eight-sample average left it nothing
    above a kilohertz, and once whitened for the waveform pass it was
    quieter than the dialogue, which no music-and-effects stem is."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    noise = np.convolve(rng.standard_normal(n), np.ones(2) / 2.0, mode="same")
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


def _dialogue(seconds: float, seed: int) -> np.ndarray:
    """Speech-like bursts where speech lives: 300-3400 Hz.

    ``_speechlike`` is low-passed noise, most of its power below 300 Hz,
    and there two different speakers coincide by chance often enough to
    look like a match; real dialogue has little energy that low, which is
    what leaves the band to the music-and-effects bed and lets the low
    band find a scene the full band cannot. The fixture's dialogue is put
    where dialogue is, at the same peak level.
    """
    signal = _speechlike(seconds, SR, seed=seed).astype(np.float64)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / SR)
    spectrum[(freqs < 300.0) | (freqs > 3400.0)] = 0.0
    banded = np.fft.irfft(spectrum, len(signal))
    peak = np.max(np.abs(banded))
    return (banded / peak * np.max(np.abs(signal))).astype(np.float32) if peak > 0 else banded.astype(np.float32)


def build_pair(workspace, total_s, pieces, bed_gain=0.6, dial_gain=0.8):
    """Write org.wav and dub.wav; return the expected stretches.

    ``pieces`` describe the dub's timeline in order: ``("org", a, b)`` is
    the video's material from a to b seconds, ``("extra", seconds)`` is
    material only the dub has. The expected stretches are
    (video_start, video_end, offset) with offset = dub time - video time.
    """
    bed = _bed(total_s, 11)
    # Each mix has its own noise floor, as real ones do, so nothing below
    # it -- rounding, quantisation -- is shared between the two.
    floor = 1e-3
    org = (
        bed_gain * bed + dial_gain * _dialogue(total_s, seed=22)
        + floor * np.random.default_rng(5).standard_normal(len(bed))
    ).astype(np.float32)
    dub_full = (
        bed_gain * 0.8 * bed + dial_gain * _dialogue(total_s, seed=33)
        + floor * np.random.default_rng(6).standard_normal(len(bed))
    ).astype(np.float32)
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
            extra = _dialogue(piece[1], seed=seed) * 0.5
            parts.append(extra + 1e-3 * np.random.default_rng(seed).standard_normal(len(extra)).astype(np.float32))
            cursor += piece[1]
    sf.write(workspace.path("org.wav"), org, SR)
    sf.write(workspace.path("dub.wav"), np.concatenate(parts).astype(np.float32), SR)
    return expected


def _plan(workspace, **kwargs):
    plan = plan_dubsync(workspace.path("org.wav"), workspace.path("dub.wav"), progress=lambda p, s: None, **kwargs)
    assert plan.error is None, plan.error
    return plan


def _plan_at(workspace, video_name, dub_name, **kwargs):
    plan = plan_dubsync(
        workspace.path(video_name), workspace.path(dub_name),
        progress=lambda p, s: None, **kwargs,
    )
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
    to the millisecond. The cuts come out where the shared bed lets them.
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
        # The dub has material from the very start at the offset the first
        # stretch sits at: whether the bed placed it or it was kept on the
        # strength of the stretch, it is dub, not fill.
        assert dubs[0].start_s == 0.0, plan.describe()
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
        assert len(silent) == 1 and re.match(r"the dub is silent across 0:03:(19\.9|20\.0)", silent[0]), plan.warnings


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


def test_a_dub_made_of_episodes_is_found_across_the_whole_file():
    """A dub cut as episodes: the film's ending first, then a long recap,
    then the film with an opening and ending song between its parts. The
    ending sits 200 s *before* where a dub of the same edit could put it,
    outside the coarse pass's range, and is found by the wide pass. The
    songs and the recap are simply not used; nothing is trimmed from the
    video, and only the two scenes the dub lacks are filled."""
    with Workspace() as ws:
        expected = build_pair(ws, 300, [
            ("org", 200, 300), ("extra", 240), ("org", 0, 90), ("extra", 60), ("org", 100, 190),
        ])
        plan = _plan(ws)
        got = sorted((round(d.start_s), round(d.end_s), round(d.offset_s, 3)) for d in plan.dub_segments)
        assert sorted((s, e, round(o, 3)) for s, e, o in expected) == [(0, 90, 340.0), (100, 190, 390.0), (200, 300, -200.0)]
        for (start, end, offset) in expected:
            near = [g for g in got if abs(g[2] - offset) <= 0.003]
            assert near, f"no stretch at {offset:+.0f}s:\n{plan.describe()}"
            assert abs(near[0][0] - start) <= 2 and abs(near[0][1] - end) <= 2, f"stretch at {offset:+.0f}s spans {near[0][:2]}, expected {start}-{end}"
        assert abs(plan.filled_s - 20.0) < 3.0, plan.describe()


def test_a_long_uncorrelated_leader_is_filled_not_kept():
    """Dub audible before the first stretch, at the offset the stretch
    continues from, is kept a little way -- a logo the mixes differ on --
    but not for minutes on the strength of one stretch: inside the file a
    kept passage has the same offset on both sides for evidence, at the
    start it has one. On a dub made of episodes the passage was another
    episode's ending, and it was played over the film's opening."""
    with Workspace() as ws:
        build_pair(ws, 300, [("extra", 200), ("org", 190, 300)])
        plan = _plan(ws)
        first = plan.segments[0]
        assert first.kind == "fill" and first.note == "dub audible but did not correlate; replaced", plan.describe()
        assert abs(first.end_s - 190.0) < 2.0, plan.describe()
    with Workspace() as ws:
        build_pair(ws, 300, [("extra", 20), ("org", 15, 300)])
        plan = _plan(ws)
        assert plan.segments[0].kind == "dub" and plan.segments[0].start_s == 0.0, plan.describe()
        assert any(n.startswith("kept the dub across 0:00:00.000") for n in plan.notes), plan.notes


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
                audio[int(100 * SR) : int(290 * SR)] -= 0.97 * scale * bed[int(110 * SR) : int(300 * SR)]
            else:
                audio[quiet] -= 0.97 * scale * bed[quiet]
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


def _mkv_with_fps(workspace, fps_ratio: str, name: str = "org.mkv") -> str:
    """Wrap org.wav in an MKV whose video stream carries the given frame
    rate, the way a real file's metadata would. The video is 64x64 black,
    which encodes in seconds."""
    import subprocess

    from audiosync.media import ffmpeg_path

    mkv = workspace.path(name)
    subprocess.run(
        [
            ffmpeg_path(), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:r={fps_ratio}",
            "-i", workspace.path("org.wav"),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-shortest",
            mkv,
        ],
        check=True, capture_output=True,
    )
    return mkv


def test_the_videos_fps_guides_the_rate_check():
    """A 23.976fps video against a 25fps-mastered dub: the video's own
    metadata names the candidates, so the rate is verified before the
    coarse pass could misread the mismatch as a flood of cuts."""
    with Workspace() as ws:
        expected = build_pair(ws, 300, [("org", 0, 120), ("org", 150, 300)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        # Mastered at 25fps: the dub runs 25/23.976 faster than the film.
        sf.write(ws.path("dub.wav"), _resample_linear(dub, (24000.0 / 1001.0) / 25.0), SR)
        mkv = _mkv_with_fps(ws, "24000/1001")
        plan = _plan_at(ws, mkv, "dub.wav")
        assert abs(plan.speed - 25.0 / (24000.0 / 1001.0)) < 1e-6, f"speed {plan.speed:.6f}"
        assert plan.video_fps == float(24000.0 / 1001.0), plan.video_fps
        assert plan.dub_rate == 25.0, plan.dub_rate
        assert "video 23.976 fps, dub mastered at 25 fps" in plan.describe(), plan.describe()
        _check(plan, expected, edge_tolerance_s=0.8, offset_tolerance_s=0.02)
        # The one real cut (30s) is filled; the rate mismatch is not misread
        # as extra cuts.
        assert plan.filled_s < 40.0, f"{plan.filled_s:.0f}s filled: {plan.describe()}"


def test_a_matched_rate_is_verified_from_the_fps():
    """A dub mastered at the video's own rate is confirmed, not silently
    assumed: the verdict says so, and the dub is left unaltered."""
    with Workspace() as ws:
        expected = build_pair(ws, 240, [("org", 0, 120), ("org", 150, 240)])
        mkv = _mkv_with_fps(ws, "24000/1001")
        plan = _plan_at(ws, mkv, "dub.wav")
        assert abs(plan.speed - 1.0) < 1e-9, plan.speed
        assert plan.video_fps == float(24000.0 / 1001.0), plan.video_fps
        # dub_rate is rounded to 6 places, so equality is judged loosely.
        assert abs(plan.dub_rate - plan.video_fps) < 1e-5, plan.dub_rate
        assert "dub at the same rate" in plan.describe(), plan.describe()
        _check(plan, expected, edge_tolerance_s=0.3)


def test_the_plan_round_trips_through_json():
    with Workspace() as ws:
        build_pair(ws, 200, [("org", 0, 80), ("org", 100, 200)])
        plan = _plan(ws)
        copy = DubSyncPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        assert copy.to_dict() == plan.to_dict()
        assert [s.to_dict() for s in copy.segments] == [s.to_dict() for s in plan.segments]


def test_the_rate_verdict_survives_the_plan_json():
    with Workspace() as ws:
        build_pair(ws, 240, [("org", 0, 240)])
        mkv = _mkv_with_fps(ws, "24000/1001")
        plan = _plan_at(ws, mkv, "dub.wav")
        payload = plan.to_dict()
        assert payload["videoFps"] is not None and payload["dubRate"] is not None
        copy = DubSyncPlan.from_dict(json.loads(json.dumps(payload)))
        assert copy.video_fps == plan.video_fps
        assert copy.dub_rate == plan.dub_rate


def test_the_report_names_every_piece():
    with Workspace() as ws:
        build_pair(ws, 200, [("org", 0, 80), ("org", 100, 200)])
        text = _plan(ws).describe()
        assert "dub is cut here" in text
        assert text.count("\n  dub ") == 2 and text.count("\n  fill") >= 1


if __name__ == "__main__":
    # Write one synthetic pair to a directory, for a caller that is not Python:
    # the Rust end-to-end test builds its input this way. Usage:
    #     python tests/test_dubsync.py DIR [MINUTES]
    # writes DIR/org.wav and DIR/dub.wav and prints the expected stretches:
    # the five-minute pair with four cuts, or, given MINUTES, one long
    # uncut pair (for a run that must still be in flight when it is stopped).
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("usage: test_dubsync.py DIR [MINUTES]", file=sys.stderr)
        sys.exit(2)
    os.makedirs(target, exist_ok=True)

    class _Dir:
        def path(self, name):  # noqa: D401 - mirrors Workspace
            return os.path.join(target, name)

    if len(sys.argv) > 2:
        minutes = float(sys.argv[2])
        pieces = [("extra", 1.5), ("org", 0, 60 * minutes)]
        total = int(60 * minutes)
    else:
        pieces = [("extra", 0.8), ("org", 0, 60), ("org", 75, 180), ("org", 182.5, 240), ("extra", 5), ("org", 240, 297)]
        total = 300
    for stretch in build_pair(_Dir(), total, pieces):
        print("%.1f-%.1f %+.3f" % stretch)


# ---------------------------------------------------------------------------
# Bands: the low band finds what the full band cannot, the high band decides
# ---------------------------------------------------------------------------


def test_band_envelopes_are_built_on_the_full_bands_frames():
    """Every band has one value per full-band frame, describing the same
    instant: a click lands on the same frame in all of them."""
    with Workspace() as ws:
        audio = np.zeros(int(20 * SR), dtype=np.float32)
        rng = np.random.default_rng(3)
        audio += 0.01 * rng.standard_normal(len(audio)).astype(np.float32)
        click = int(12.3456 * SR)
        audio[click : click + 16] += 0.8
        sf.write(ws.path("click.wav"), audio, SR)
        env = build_envelope(ws.path("click.wav"))
        assert env.bands == ["full", "high", "low"]
        for band in ("low", "high"):
            assert len(env.band_energy[band]) == len(env.energy)
            assert len(env.onsets(band)) == len(env.onset)
        frame = int(12.3456 * ENVELOPE_RATE)
        assert abs(int(np.argmax(env.onset)) - frame) <= 2
        for band in ("low", "high"):
            # A band's frame is a 24 ms window, so its energy starts rising
            # half a window before the click and the onset leads by up to
            # that; the same lead on both tracks, so offsets are unmoved.
            peak = int(np.argmax(env.onsets(band)))
            assert -8 <= peak - frame <= 2, f"{band} onset peaks at frame {peak}, the click is at {frame}"
        # Sped up, the bands come along.
        faster = env.at_speed(0.999)
        assert set(faster.band_onset) == {"low", "high"}
        assert len(faster.band_onset["low"]) == len(faster.onset)


def test_a_scene_only_the_low_band_can_see_is_found():
    """A scene with no music: the two languages' consonants agree on
    nothing in the full band, but the bass and rumble the mixes share do
    correlate in the 30-250 Hz band. Such a scene is placed, not filled --
    and without the low band it would have been filled, which is what the
    band is for."""
    with Workspace() as ws:
        # The scene sits at its own offset (a cut before and after it), so
        # nothing can coast across it: it is found or it is filled.
        expected = build_pair(ws, 400, [("org", 0, 120), ("org", 125, 280), ("org", 285, 400)])
        # Between 125 and 280 s of the video replace the shared bed with one
        # that lives below 200 Hz only, well under the dialogue.
        bed = _bed(400, 11).astype(np.float64)
        spectrum = np.fft.rfft(bed)
        freqs = np.fft.rfftfreq(len(bed), 1.0 / SR)
        spectrum[freqs > 200.0] = 0.0
        rumble = np.fft.irfft(spectrum, len(bed))
        rumble = rumble / np.max(np.abs(rumble)) * 0.7
        for name, gain, shift in (("org.wav", 0.6, 0), ("dub.wav", 0.6 * 0.8, -5)):
            audio, _ = sf.read(ws.path(name), dtype="float32")
            lo, hi = int((125 + shift) * SR), int((280 + shift) * SR)
            audio[lo:hi] -= (gain * bed[int(125 * SR) : int(280 * SR)]).astype(np.float32)
            audio[lo:hi] += (gain * 0.12 * rumble[int(125 * SR) : int(280 * SR)]).astype(np.float32)
            sf.write(ws.path(name), audio, SR)
        plan = _plan(ws)
        # Found and placed; with a bed this faint the edges are as good as
        # a couple of seconds, and the point is that the scene is dub.
        _check(plan, expected, edge_tolerance_s=2.0)
        assert plan.filled_s < 15.0, plan.describe()
        # The full band alone gives up on much of the scene.
        originals = {name: _Aligner.__dict__[name] for name in ("detect_bands", "shared_bands")}
        try:
            _Aligner.detect_bands = property(lambda self: ["full"])
            _Aligner.shared_bands = property(lambda self: ["full"])
            plan_blind = _plan(ws)
        finally:
            for name, original in originals.items():
                setattr(_Aligner, name, original)
        assert plan_blind.filled_s > 30.0, plan_blind.describe()


def test_a_repeated_cue_is_placed_where_the_scene_continues():
    """A window whose cue recurs in the dub keeps the peak that continues
    the neighbouring stretch, not the tallest; without a neighbour, one
    that does not wind the dub back by a little; failing both, the tallest."""
    anchor = Block(100.0, 160.0, +19.0)
    candidates = [(-948.0, 40.0), (+19.06, 31.0), (+5143.0, 36.0)]
    assert _continuing_candidate(candidates, 170.0, [anchor], None) == (+19.06, 31.0)
    # No stretch nearby: the previous window sat at +19.0, so a candidate
    # two minutes back is a repeat within the scene and one that continues
    # wins, though the repeat is taller.
    nearby = [(-120.0, 40.0), (+19.06, 31.0)]
    assert _continuing_candidate(nearby, 800.0, [anchor], (785.0, +19.0)) == (+19.06, 31.0)
    # Winding back a whole episode is allowed: material out of order.
    far = [(-1500.0, 30.0), (+19.06, 8.0)]
    assert _continuing_candidate(far, 800.0, [], (785.0, +19.0)) == (-1500.0, 30.0)
    # Nothing known: the tallest.
    assert _continuing_candidate(candidates, 800.0, [], None) == (-948.0, 40.0)


def test_a_beat_does_not_fool_the_verification():
    """Music with a beat correlates at every period; the residual reported
    is the peak nearest zero among those as tall as the tallest, and the
    bands' curves are summed so a peak only one band shows is not it."""
    rate = ENVELOPE_RATE
    rng = np.random.default_rng(9)
    period = int(0.444 * rate)
    n = 60 * rate
    pulse = np.zeros(n)
    pulse[::period] = 1.0
    beat = np.convolve(pulse, [0.3, 1.0, 0.3], mode="same") + 0.05 * rng.standard_normal(n)
    other = 0.05 * rng.standard_normal(n)
    # The track sits where it should; the full band is nearly periodic.
    margin = int(1.0 * rate)
    window = 30 * rate
    start = 10 * rate
    padded = np.concatenate([np.zeros(margin), beat, np.zeros(margin)])
    residual, _match, z = _measure([beat, other], [padded[margin:-margin], other], start, window, margin, rate)
    assert residual is not None and abs(residual) < 3.0, residual
    assert z >= 5.0, z


def test_the_nearest_of_equal_peaks_is_taken_and_a_shoulder_is_not_a_peak():
    rng = np.random.default_rng(1)
    row = 0.01 * np.abs(rng.standard_normal(1001))
    for centre, height in ((300, 0.9), (500, 1.0), (700, 0.92)):
        row[centre - 5 : centre + 6] = height * np.hanning(11)
    assert _nearest_peak(row, 310) == 300
    assert _nearest_peak(row, 650) == 700
    assert _nearest_peak(row, 505) == 500
    # A bump on the tallest peak's shoulder is not another peak.
    row[520] += 0.05
    assert _nearest_peak(row, 530) == 500
    # Dominance: one peak among noise stands out, one among equals does not.
    lone = 0.02 * np.abs(rng.standard_normal(1001))
    lone[500 - 5 : 500 + 6] = np.hanning(11)
    assert _peak_dominance(lone, 500) > 5.0
    assert _peak_dominance(row, 500) < 3.0


def test_a_small_step_inside_an_uncorrelated_passage_is_bridged():
    """A scene with no shared bed, and a cut of a second somewhere inside
    it: the dub is there on both sides, one second apart, and is kept
    rather than replaced by a minute of the other language. The step is
    put where the agreement says, or in the middle when it says nothing,
    and the note says which; the second the dub lacks is filled there."""
    with Workspace() as ws:
        build_pair(ws, 300, [("org", 0, 100), ("org", 101, 300)])
        bed = _bed(300, 11)
        for name, gain, shift in (("org.wav", 0.6, 0.0), ("dub.wav", 0.6 * 0.8, 0.0)):
            audio, _ = sf.read(ws.path(name), dtype="float32")
            if name == "org.wav":
                audio[int(85 * SR) : int(145 * SR)] -= gain * bed[int(85 * SR) : int(145 * SR)]
            else:
                # The dub's timeline: video 85-100 -> dub 85-100, video 101-145 -> dub 100-144.
                audio[int(85 * SR) : int(100 * SR)] -= gain * bed[int(85 * SR) : int(100 * SR)]
                audio[int(100 * SR) : int(144 * SR)] -= gain * bed[int(101 * SR) : int(145 * SR)]
            sf.write(ws.path(name), audio, SR)
        plan = _plan(ws)
        dubs = plan.dub_segments
        assert [round(d.offset_s, 2) for d in dubs] == [0.0, -1.0], plan.describe()
        long_fills = [f for f in plan.fill_segments if f.length_s > 1.5]
        assert not long_fills, plan.describe()
        assert abs(plan.filled_s - 1.0) < 0.3, plan.describe()
        assert any("the offset steps by only +1000 ms inside it; the step was guessed" in n for n in plan.notes), plan.notes
        # The step lies inside the bedless passage, and the dub is
        # continuous across it: what precedes the fill on the video is the
        # last dub before the cut, what follows is the first after.
        cut = dubs[0].end_s
        assert 85.0 <= cut <= 145.0, plan.describe()
        assert abs(dubs[1].start_s - (cut + 1.0)) < 0.02, plan.describe()


def test_a_23976_dub_on_a_24fps_video_is_confirmed_before_the_coarse_pass():
    """The common case: a 24 fps video and a dub timed to a 23.976 master.
    The rate check that runs first, on the video's own frame rate, must
    settle it -- at 2 ms a thirty-second window at the wrong speed is
    smeared by fifteen frames -- and say so, without leaving it to the
    drift estimate, which needs minutes of matched dub that a heavily cut
    dub may not have."""
    with Workspace() as ws:
        expected = build_pair(ws, 400, [("org", 0, 120), ("org", 130, 250), ("org", 262, 400)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        sf.write(ws.path("dub.wav"), _resample_linear(dub, 1001.0 / 1000.0), SR)
        mkv = _mkv_with_fps(ws, "24")
        logs = []
        plan = plan_dubsync(mkv, ws.path("dub.wav"), progress=lambda p, s: None, log=logs.append)
        assert plan.error is None, plan.error
        assert abs(plan.speed - 1000.0 / 1001.0) < 1e-6, f"speed {plan.speed:.6f}\n" + "\n".join(logs)
        assert plan.rate_confirmed is True, plan.describe()
        assert plan.dub_rate is not None and abs(plan.dub_rate - 24000.0 / 1001.0) < 1e-3, plan.dub_rate
        assert any("dub runs at 0.999001x" in line for line in logs), "\n".join(logs)
        # Settled by the rate check, before the coarse pass ran.
        settled = next(i for i, line in enumerate(logs) if "dub runs at" in line)
        coarse = next(i for i, line in enumerate(logs) if line.startswith("coarse pass"))
        assert settled < coarse, "\n".join(logs)
        assert "dub mastered at 23.976 fps" in plan.describe(), plan.describe()
        _check(plan, expected, edge_tolerance_s=0.5, offset_tolerance_s=0.01)


def test_an_unconfirmed_rate_is_said_so():
    """A dub that shares nothing measurable with the video at any standard
    speed: the plan cannot know its rate, and must say that rather than
    quietly assume the video's."""
    with Workspace() as ws:
        build_pair(ws, 300, [("extra", 300)])
        mkv = _mkv_with_fps(ws, "24")
        plan = plan_dubsync(mkv, ws.path("dub.wav"), progress=lambda p, s: None, log=lambda m: None)
        assert plan.rate_confirmed is False, plan.describe()
        assert any("frame rate could not be confirmed" in w for w in plan.warnings), plan.warnings


def test_the_dub_rate_set_by_hand_fixes_the_speed():
    """The user knows the dub was mastered at 23.976 fps: with the video's
    24 fps that is the speed, and the audio is not asked."""
    with Workspace() as ws:
        expected = build_pair(ws, 300, [("org", 0, 300)])
        dub, _ = sf.read(ws.path("dub.wav"), dtype="float32")
        sf.write(ws.path("dub.wav"), _resample_linear(dub, 1001.0 / 1000.0), SR)
        mkv = _mkv_with_fps(ws, "24")
        logs = []
        plan = plan_dubsync(mkv, ws.path("dub.wav"), dub_rate=23.976, progress=lambda p, s: None, log=logs.append)
        assert plan.error is None, plan.error
        assert abs(plan.speed - 1000.0 / 1001.0) < 1e-9, plan.speed
        assert plan.rate_confirmed is None, plan.rate_confirmed
        assert any("said to be mastered at 23.976 fps" in line for line in logs), "\n".join(logs)
        assert not any("checking the dub's rate" in line for line in logs), "\n".join(logs)
        _check(plan, expected, edge_tolerance_s=0.5, offset_tolerance_s=0.01)
