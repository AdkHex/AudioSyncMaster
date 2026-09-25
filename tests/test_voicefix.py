"""Tests for the voice check: a dub whose voices were cut apart from its music.

Measured on a real series (Goblin E01, Hindi dub on the Korean Blu-ray): the
dub's music matched to 0.1 ms everywhere, yet around four places its voices
sat 0.8-4.5 s early or 2.3 s late, because the dub studio cut the voices
with a shorter picture and the music elsewhere. These tests build that
situation from synthetic speech and check that the voices -- only the
voices -- are found and moved, without the voice tools installed: the
detector and separator are stood in for.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync import voicefix, voicetools  # noqa: E402
from audiosync.dubrender import RenderOptions, clip_plan, render  # noqa: E402
from audiosync.dubsync import DubSyncPlan, Segment  # noqa: E402
from audiosync.voicefix import (  # noqa: E402
    SPEECH_RATE,
    SPEECH_T0,
    PlanMap,
    VoicePiece,
    _pieces_in_region,
    build_patch,
    lag_scan,
    pair_lines,
    profile,
    runs_of,
    segments,
)


def _lines(seed: int, lo: float, hi: float, quiet=()):
    """Random dialogue: lines of 0.4-2.5 s with pauses of 0.3-2 s."""
    rng = np.random.default_rng(seed)
    out, t = [], lo
    while t < hi:
        t += rng.uniform(0.3, 2.0)
        length = rng.uniform(0.4, 2.5)
        if t + length < hi and not any(a < t + length and t < b for a, b in quiet):
            out.append((t, t + length))
        t += length
    return out


def _curve(lines, duration: float) -> np.ndarray:
    """A speech-probability curve (on the detector's grid) for these lines."""
    times = SPEECH_T0 + np.arange(int(duration * SPEECH_RATE)) / SPEECH_RATE
    curve = np.full(len(times), 0.03)
    for a, b in lines:
        curve[(times >= a) & (times < b)] = 0.95
    return curve


def _plan(segments, duration=400.0):
    return DubSyncPlan("video.mkv", "dub.m4a", segments=segments, video_duration_s=duration,
                       dub_duration_s=duration, timeline="container")


def test_lag_scan_says_how_much_later_the_dub_must_move():
    rng = np.random.default_rng(1)
    reference = rng.random(500)
    reach = 40
    # the dub's copy sits 12 samples EARLY: its value for i is at i - 12
    other = np.concatenate([np.zeros(reach), reference, np.zeros(reach)])
    other = np.roll(other, -12)
    row = lag_scan(reference, other, reach)
    assert int(np.argmax(row)) - reach == 12, int(np.argmax(row)) - reach
    assert row.max() > 0.99


def test_lines_pair_when_they_start_together():
    original = [(1.0, 2.0), (3.0, 4.0), (6.0, 7.0)]
    dub = [(1.05, 2.2), (3.4, 3.9), (9.0, 9.5)]
    count, diffs = pair_lines(original, dub)
    assert count == 2 and abs(diffs[0] - 0.05) < 1e-9 and abs(diffs[1] - 0.4) < 1e-9
    times = np.arange(0, 10, 1 / SPEECH_RATE)
    found = segments(np.where((times > 1) & (times < 2), 0.9, 0.1), times)
    assert len(found) == 1 and abs(found[0][0] - 1.0) < 0.02 and abs(found[0][1] - 2.0) < 0.02


def _cut_apart():
    """A dub that lacks 3.5 s of picture at 0:03:20, with its music cut
    there but its voices cut at 0:02:30: between the two its voices play
    3.5 s early (they already sit where the next scene's music does)."""
    first, second = -10.0, -13.5
    korean = _lines(7, 5.0, 395.0, quiet=[(147.0, 155.0)])
    dub = []
    for a, b in korean:
        level = first if b <= 150.0 else second
        dub.append((a + level, b + level))
    plan = _plan([
        Segment("dub", 0.0, 200.0, first, first),
        Segment("fill", 200.0, 203.5, 200.0, None, note="dub is cut here"),
        Segment("dub", 203.5, 400.0, 203.5 + second, second),
    ])
    return plan, _curve(korean, 400.0), _curve(dub, 400.0), korean


def test_voices_cut_apart_from_their_music_are_found_and_join_the_next_scene():
    plan, video, dub, korean = _cut_apart()
    mapping = PlanMap(plan)
    scan = profile(video, dub, mapping.offset_at, 0.0, 400.0, 20.0, 10.0, 8.0)
    flagged = [w for w in scan if abs(w.lag_s) > 0.15 and w.ncc > 0.45 and w.gain > 0.12]
    assert flagged and all(abs(w.lag_s - 3.5) < 0.05 for w in flagged), [(w.start_s, w.lag_s) for w in flagged]
    assert all(w.start_s >= 130.0 for w in flagged), "only the displaced scene is flagged"
    windows = profile(video, dub, mapping.offset_at, 110.0, 240.0, 8.0, 2.0, 8.0, min_talk=0.1)
    notes: list = []
    pieces = _pieces_in_region(runs_of(windows), mapping, video, dub, 110.0, 240.0, notes.append)
    assert len(pieces) == 1, notes
    piece = pieces[0]
    assert abs(piece.shift_s - 3.5) < 1e-6 and abs(piece.level_s + 13.5) < 1e-6, piece
    assert piece.join_end and abs(piece.dub_end_s - (203.5 - 13.5)) < 1e-6, piece
    # it starts in the dub's pause where the voices were cut: after the last
    # line still in step, before the first displaced one
    last_in_step = max(b for a, b in korean if b <= 150.0) - 10.0
    first_displaced = min(a for a, b in korean if a >= 150.0) - 13.5
    assert last_in_step < piece.dub_start_s < first_displaced, (last_in_step, piece.dub_start_s, first_displaced)


def test_a_dub_in_step_with_the_lips_is_left_alone():
    korean = _lines(3, 5.0, 395.0)
    dub = [(a - 10.0, b - 10.0) for a, b in korean]
    plan = _plan([Segment("dub", 0.0, 400.0, -10.0, -10.0)])
    mapping = PlanMap(plan)
    video_curve, dub_curve = _curve(korean, 400.0), _curve(dub, 400.0)
    scan = profile(video_curve, dub_curve, mapping.offset_at, 0.0, 400.0, 20.0, 10.0, 8.0)
    assert not voicefix.candidate_regions(scan)


class _Band:
    """A stand-in separator: the 'voices' are whatever lies above 700 Hz."""

    def vocals(self, audio: np.ndarray, rate: int) -> np.ndarray:
        spectrum = np.fft.rfft(audio, axis=0)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
        spectrum[freqs < 700.0] = 0.0
        return np.fft.irfft(spectrum, len(audio), axis=0).astype(np.float32)

    def speech(self, audio):  # pragma: no cover - not used here
        raise AssertionError("not used")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


class _Reader:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio

    def read(self, start: int, count: int) -> np.ndarray:
        out = np.zeros((count, self.audio.shape[1]), dtype=np.float32)
        a, b = max(0, start), min(len(self.audio), start + count)
        if b > a:
            out[a - start:b - start] = self.audio[a:b]
        return out


def _voice_and_music(rate: int, seconds: float, voice_at):
    t = np.arange(int(seconds * rate)) / rate
    music = 0.2 * np.sin(2 * np.pi * 200.0 * t)
    voice = np.zeros_like(t)
    for a, b in voice_at:
        on = (t >= a) & (t < b)
        voice[on] = 0.3 * np.sin(2 * np.pi * 1000.0 * t[on])
    return music.astype(np.float32), voice.astype(np.float32)


def _band_energy(x: np.ndarray, rate: int, lo: float, hi: float, a: float, b: float) -> float:
    seg = x[int(a * rate):int(b * rate)]
    spectrum = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / rate)
    return float(spectrum[(freqs >= lo) & (freqs < hi)].sum())


def test_the_patch_moves_the_voices_and_leaves_the_music():
    rate = 8000
    music, voice = _voice_and_music(rate, 12.0, [(3.0, 4.0)])
    dub = (music + voice)[:, None]
    plan = _plan([Segment("dub", 0.0, 12.0, 0.0, 0.0)], duration=12.0)
    piece = VoicePiece(2.5, 4.5, -1.0, 1.0)  # these voices belong 1 s later
    patch = build_patch(plan, [piece], _Band(), _Reader(dub), _Reader(dub), rate, 1.0, 0.01)
    block = dub.copy()
    patch.apply(0, block)
    moved = block[:, 0]
    before = _band_energy(dub[:, 0], rate, 900, 1100, 3.1, 3.9)
    assert _band_energy(moved, rate, 900, 1100, 3.1, 3.9) < 1e-4 * before, "voices gone from where they were"
    assert _band_energy(moved, rate, 900, 1100, 4.1, 4.9) > 0.95 * before, "and heard 1 s later"
    music_before = _band_energy(dub[:, 0], rate, 150, 250, 2.0, 6.0)
    music_after = _band_energy(moved, rate, 150, 250, 2.0, 6.0)
    assert abs(music_after / music_before - 1.0) < 1e-3, "the music is not touched"


def test_voices_moved_into_a_fill_take_the_originals_voices_out():
    rate = 8000
    music, voice = _voice_and_music(rate, 12.0, [(3.0, 4.0)])
    dub = (music + voice)[:, None]
    theirs_music, theirs_voice = _voice_and_music(rate, 12.0, [(5.0, 5.8)])
    original = (0.5 * theirs_music + theirs_voice)[:, None]
    plan = _plan([
        Segment("dub", 0.0, 5.0, 0.0, 0.0),
        Segment("fill", 5.0, 6.0, 5.0, None),
        Segment("dub", 6.0, 12.0, 5.0, -1.0),
    ], duration=12.0)
    piece = VoicePiece(2.5, 4.5, -2.0, 2.0)  # lands at 4.5-6.5, over the fill
    patch = build_patch(plan, [piece], _Band(), _Reader(dub), _Reader(original), rate, 1.0, 0.01)
    fill = original[int(5.0 * rate):int(6.0 * rate), 0].copy()
    block = fill[:, None].copy()
    patch.apply(int(5.0 * rate), block)
    # the original's voice (1 kHz, 5.0-5.8) is gone from the fill; the moved
    # dub voice now plays there (3.0-4.0 moved to 5.0-6.0)
    theirs = _band_energy(original[:, 0], rate, 900, 1100, 5.05, 5.75)
    assert _band_energy(block[:, 0], rate, 150, 250, 0.1, 0.9) > 0.99 * _band_energy(fill, rate, 150, 250, 0.1, 0.9)
    moved_in = _band_energy(block[:, 0], rate, 900, 1100, 0.05, 0.75)
    assert abs(moved_in / theirs - (0.3 / 0.3) ** 2) < 0.1, moved_in / theirs


def test_a_render_moves_the_voices_where_the_plan_says():
    rate = 16000
    music, voice = _voice_and_music(rate, 10.0, [(2.0, 3.0)])
    with tempfile.TemporaryDirectory() as root:
        video_path = os.path.join(root, "video.wav")
        dub_path = os.path.join(root, "dub.wav")
        out_path = os.path.join(root, "out.wav")
        sf.write(video_path, np.stack([music, music], axis=1), rate, subtype="FLOAT")
        sf.write(dub_path, np.stack([music + voice, music + voice], axis=1), rate, subtype="FLOAT")
        plan = DubSyncPlan(video_path, dub_path, segments=[Segment("dub", 0.0, 10.0, 0.0, 0.0)],
                           video_duration_s=10.0, dub_duration_s=10.0, timeline="container",
                           voice_pieces=[VoicePiece(1.5, 3.5, -1.5, 1.5)])
        installed, worker = voicetools.installed, voicetools.VoiceWorker
        voicetools.installed = lambda: True
        voicetools.VoiceWorker = lambda token=None, log=None: _Band()
        try:
            result = render(plan, out_path, RenderOptions(codec="wav"))
            written, _ = sf.read(out_path, dtype="float32", always_2d=True)
            skipped = render(plan, out_path + ".off.wav", RenderOptions(codec="wav", fix_voices=False))
            plain, _ = sf.read(out_path + ".off.wav", dtype="float32", always_2d=True)
        finally:
            voicetools.installed, voicetools.VoiceWorker = installed, worker
        assert not result.warnings and not skipped.warnings, (result.warnings, skipped.warnings)
        left = written[:, 0]
        before = _band_energy(plain[:, 0], rate, 900, 1100, 2.1, 2.9)
        assert _band_energy(left, rate, 900, 1100, 2.1, 2.9) < 1e-3 * before
        assert _band_energy(left, rate, 900, 1100, 3.6, 4.4) > 0.95 * before
        assert np.allclose(left[: int(1.4 * rate)], plain[: int(1.4 * rate), 0], atol=1e-5)


def test_without_the_tools_a_plans_voice_moves_are_skipped_and_said():
    rate = 16000
    music, voice = _voice_and_music(rate, 4.0, [(1.0, 2.0)])
    with tempfile.TemporaryDirectory() as root:
        os.environ["AUDIOSYNC_VOICE_TOOLS"] = os.path.join(root, "tools")
        try:
            video_path = os.path.join(root, "video.wav")
            dub_path = os.path.join(root, "dub.wav")
            sf.write(video_path, music, rate, subtype="FLOAT")
            sf.write(dub_path, music + voice, rate, subtype="FLOAT")
            plan = DubSyncPlan(video_path, dub_path, segments=[Segment("dub", 0.0, 4.0, 0.0, 0.0)],
                               video_duration_s=4.0, dub_duration_s=4.0, timeline="container",
                               voice_pieces=[VoicePiece(0.5, 2.5, -0.5, 0.5)])
            result = render(plan, os.path.join(root, "out.wav"), RenderOptions(codec="wav"))
        finally:
            del os.environ["AUDIOSYNC_VOICE_TOOLS"]
        assert any("voice tools are not installed" in w for w in result.warnings), result.warnings


def test_plans_carry_their_voice_moves():
    plan = _plan([Segment("dub", 0.0, 400.0, -10.0, -10.0, 0.5)])
    plan.voice_pieces = [VoicePiece(100.0, 120.0, -12.0, 2.0, True, "moved")]
    again = DubSyncPlan.from_dict(plan.to_dict())
    assert again.voice_pieces[0].to_dict() == plan.voice_pieces[0].to_dict()
    assert again.to_dict()["voicePieces"][0]["videoStartS"] == 112.0
    clipped = clip_plan(plan, 100.0, 200.0)
    # on the clipped timeline the same dub plays 100 s earlier
    assert abs(clipped.voice_pieces[0].video_start_s - 12.0) < 1e-9
    assert "voices: moved" in plan.describe()


def test_the_voice_tools_install_into_the_users_data_folder():
    old = os.environ.pop("AUDIOSYNC_VOICE_TOOLS", None)
    try:
        assert voicetools.tools_dir().endswith(os.path.join("AudioSyncMaster", "voice-tools"))
        with tempfile.TemporaryDirectory() as root:
            os.environ["AUDIOSYNC_VOICE_TOOLS"] = root
            assert voicetools.tools_dir() == root
            assert voicetools.status()["installed"] is False
            if os.name == "nt":
                assert voicetools.python_path().endswith(os.path.join("Scripts", "python.exe"))
    finally:
        os.environ.pop("AUDIOSYNC_VOICE_TOOLS", None)
        if old is not None:
            os.environ["AUDIOSYNC_VOICE_TOOLS"] = old
    # every platform the app ships on has a pinned, checksummed installer
    for key in (("windows", "x86_64"), ("darwin", "arm64"), ("linux", "x86_64")):
        name, digest = voicetools.UV_ASSETS[key]
        assert name.startswith("uv-") and len(digest) == 64
    assert os.path.isfile(voicetools.worker_source())
