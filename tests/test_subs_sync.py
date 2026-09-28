"""Tests for subtitle sync: speech activity, the four engines, and the sign
of every offset.

The fixtures are built here rather than stored. Speech comes from macOS
``say`` in eight languages (elsewhere, noise shaped like syllables), laid at
known times over a synthetic score -- sustained chords, a kick on every beat,
hi-hats -- so the detector has to find voices under music, as it must in an
anime or a drama. The subtitle is then made the way real ones go wrong:
timed for a 25 fps release of a 23.976 fps film, 3.25 s late, and from an
edit with seven seconds the video does not have.

Run:  python/.venv/bin/python tests/test_subs_sync.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from audiosync.media import CancellationToken, Cancelled  # noqa: E402
from audiosync.subs import sync, vad  # noqa: E402
from audiosync.subs.model import Cue, SubtitleDoc, Word  # noqa: E402

SR = 16000

# (voice, text). Several languages, because the audio engine must not care
# which one is spoken.
PHRASES: List[Tuple[str, str]] = [
    ("Samantha", "Where were you last night? I waited for hours."),
    ("Samantha", "Don't tell me you forgot again."),
    ("Albert", "We have to leave before the storm comes."),
    ("Albert", "I never said that. You know I didn't."),
    ("Eddy (Japanese (Japan))", "今日はとても寒いですね。早く帰りましょう。"),
    ("Eddy (Japanese (Japan))", "本当にそれでいいの？"),
    ("Eddy (Japanese (Japan))", "待って、まだ話は終わってない。"),
    ("Eddy (Korean (South Korea))", "어디 가는 거야? 같이 가자."),
    ("Eddy (Korean (South Korea))", "그건 내 잘못이 아니야."),
    ("Eddy (Chinese (China mainland))", "我们明天早上再见吧。"),
    ("Eddy (Chinese (China mainland))", "你为什么不早点告诉我？"),
    ("Eddy (French (France))", "Je ne comprends pas pourquoi tu es parti."),
    ("Eddy (French (France))", "Attends-moi, j'arrive tout de suite."),
    ("Eddy (Spanish (Spain))", "No puedo creer lo que acabas de decir."),
    ("Eddy (Spanish (Spain))", "Vamos, que llegamos tarde otra vez."),
    ("Eddy (Italian (Italy))", "Non ti preoccupare, andrà tutto bene."),
    ("Eddy (Italian (Italy))", "Dove hai messo le chiavi della macchina?"),
    ("Eddy (German (Germany))", "Ich habe dich überall gesucht."),
    ("Eddy (German (Germany))", "Das ist nicht so einfach, wie du denkst."),
    ("Samantha", "Yes."),
    ("Albert", "Listen to me, just this once."),
    ("Eddy (Japanese (Japan))", "ありがとう。"),
    ("Eddy (Korean (South Korea))", "잠깐만요!"),
    ("Eddy (French (France))", "Pourquoi pas?"),
]

PAL = Fraction(25 * 1001, 24000)  # a 25 fps subtitle on a 23.976 fps video

_CACHE: Dict[str, object] = {}


# ------------------------------------------------------------------ fixtures


def _trim(x: np.ndarray) -> np.ndarray:
    """Drop the silence `say` leaves around a phrase, so cue = speech."""
    loud = np.flatnonzero(np.abs(x) > 0.02 * float(np.abs(x).max()))
    return x[loud[0]:loud[-1] + 1] if len(loud) else x


def _say_phrases() -> Optional[List[np.ndarray]]:
    say = shutil.which("say")
    if not say:
        return None
    import soundfile as sf

    folder = tempfile.mkdtemp(prefix="subsync-say-")

    def render(item):
        index, (voice, text) = item
        path = os.path.join(folder, f"{index}.wav")
        subprocess.run(
            [say, "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", path, text],
            check=True, capture_output=True, timeout=60,
        )
        data, rate = sf.read(path, dtype="float32")
        assert rate == SR
        return _trim(data if data.ndim == 1 else data.mean(axis=1))

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(render, enumerate(PHRASES)))
    except (subprocess.SubprocessError, OSError, AssertionError):
        return None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _synthetic_phrases(rng: np.random.Generator) -> List[np.ndarray]:
    """Speech-shaped noise: 300-3400 Hz, voiced with a gliding pitch, and
    chopped into syllables four to six times a second."""
    out = []
    for _ in range(len(PHRASES)):
        seconds = rng.uniform(0.8, 3.2)
        n = int(seconds * SR)
        t = np.arange(n) / SR
        pitch = rng.uniform(100, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t))
        phase = 2 * np.pi * np.cumsum(pitch) / SR
        voiced = sum(np.sin(k * phase) / k for k in range(1, 25))
        noise = rng.standard_normal(n)
        spectrum = np.fft.rfft(voiced + 0.3 * noise)
        freqs = np.fft.rfftfreq(n, 1 / SR)
        spectrum[(freqs < 300) | (freqs > 3400)] *= 0.05
        band = np.fft.irfft(spectrum, n)
        rate = rng.uniform(4, 6)
        syllables = np.clip(np.sin(2 * np.pi * rate * t + rng.uniform(0, 6)), 0, None) ** 0.7
        x = band * syllables
        out.append((0.3 * x / np.abs(x).max()).astype(np.float32))
    return out


def phrases() -> Tuple[List[np.ndarray], str]:
    if "phrases" not in _CACHE:
        rendered = _say_phrases()
        kind = "say"
        if rendered is None:
            rendered, kind = _synthetic_phrases(np.random.default_rng(7)), "synthetic"
        _CACHE["phrases"] = (rendered, kind)
    return _CACHE["phrases"]  # type: ignore[return-value]


def score_bed(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """A film score under everything: pad chords that change every bar, a
    kick on each beat and hi-hats between them."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float64)
    bar = 2.4
    roots = [110.0, 130.8, 98.0, 146.8, 123.5, 87.3]
    for start in np.arange(0, seconds, bar):
        a, b = int(start * SR), min(n, int((start + bar) * SR))
        root = roots[int(rng.integers(len(roots)))]
        seg = t[a:b] - start
        envelope = np.minimum(1.0, seg / 0.3) * np.minimum(1.0, (bar - seg) / 0.3)
        for ratio in (1.0, 1.26, 1.5, 2.0):
            for harmonic in (1, 2, 3):
                out[a:b] += envelope * np.sin(2 * np.pi * root * ratio * harmonic * seg) / (harmonic * 2.0)
    beat = 0.5
    for start in np.arange(0, seconds, beat):
        a = int(start * SR)
        m = min(n - a, int(0.25 * SR))
        seg = np.arange(m) / SR
        out[a:a + m] += 2.5 * np.sin(2 * np.pi * 55 * seg) * np.exp(-seg / 0.08)
        h = min(n - a - int(beat / 2 * SR), int(0.05 * SR))
        if h > 0:
            hat = rng.standard_normal(h) * np.exp(-np.arange(h) / SR / 0.015)
            k = a + int(beat / 2 * SR)
            out[k:k + h] += 0.3 * np.diff(hat, prepend=0.0)
    return out / np.sqrt(np.mean(out ** 2))


class Scenario:
    """A video's audio and a subtitle timed for another release of it.

    The subtitle's release runs ``ratio`` times fast and ``offset`` seconds
    ahead (subtitle time ``s`` plays at video time ``ratio * s + offset``),
    and has ``cut_s`` seconds of material at ``cut_at`` (its own time) that
    the video lacks. ``lines_in_cut`` of its lines are in that material.
    """

    def __init__(self, seed: int = 1, length_s: float = 720.0, ratio: float = float(PAL),
                 offset: float = 3.25, cut_at: Optional[float] = 360.0, cut_s: float = 7.0,
                 bed_db: float = -8.0) -> None:
        rng = np.random.default_rng(seed)
        clips, self.kind = phrases()
        self.ratio, self.offset, self.cut_at, self.cut_s = ratio, offset, cut_at, cut_s
        events = []  # (subtitle start, clip, in_cut)
        t = 4.0
        while t < length_s:
            k = int(rng.integers(len(clips)))
            dur = len(clips[k]) / SR / ratio
            if cut_at is not None and t < cut_at < t + dur + 0.3:
                t = cut_at + 0.8  # one line inside the missing material
                continue
            if cut_at is not None and cut_at <= t < cut_at + cut_s:
                if t + dur > cut_at + cut_s - 0.3:
                    t = cut_at + cut_s + 0.6
                    continue
                events.append((t, k, True))
            else:
                events.append((t, k, False))
            t += dur + rng.uniform(0.5, 2.2)
        self.lines_in_cut = sum(1 for e in events if e[2])
        video_len = self.to_video(length_s + 10.0) + 5.0
        audio = 10 ** (bed_db / 20.0) * 0.1 * score_bed(video_len, rng)
        cues = []
        self.truth: List[Tuple[float, float]] = []  # video spans of the spoken lines
        for start, k, in_cut in events:
            clip = clips[k]
            dur = len(clip) / SR
            cues.append(Cue(start, start + dur / ratio, PHRASES[k][1]))
            if in_cut:
                continue
            v = self.to_video(start)
            a = int(round(v * SR))
            gain = 0.1 / (np.sqrt(np.mean(clip ** 2)) + 1e-9) * 10 ** (rng.uniform(-3, 3) / 20)
            audio[a:a + len(clip)] += gain * clip[: max(0, len(audio) - a)]
            self.truth.append((v, v + dur))
        self.audio = audio.astype(np.float32)
        self.doc = SubtitleDoc(cues, language=None, source_format="srt")

    def to_video(self, s: float) -> float:
        if self.cut_at is not None and s >= self.cut_at + self.cut_s:
            s -= self.cut_s
        return self.ratio * s + self.offset

    @property
    def split_at(self) -> float:
        return self.ratio * self.cut_at + self.offset

    @property
    def offset_after(self) -> float:
        return self.offset - self.ratio * self.cut_s


def scenario(**kwargs) -> Scenario:
    key = "scenario" + repr(sorted(kwargs.items()))
    if key not in _CACHE:
        _CACHE[key] = Scenario(**kwargs)
    return _CACHE[key]  # type: ignore[return-value]


def probs_of(sc: Scenario) -> np.ndarray:
    key = f"probs{id(sc)}"
    if key not in _CACHE:
        _CACHE[key] = vad.activity_from_samples(sc.audio)[0]
    return _CACHE[key]  # type: ignore[return-value]


OPTIONS = {"engine": "audio", "maxOffsetS": 60, "detectFramerate": True, "allowSplits": True, "splitPenalty": 7, "vad": "energy"}


# ------------------------------------------------------------------ helpers


class Ctx:
    """A TaskContext without the task runner."""

    def __init__(self, workdir: Optional[str] = None) -> None:
        self.token = CancellationToken()
        self.stages: List[str] = []
        self.logs: List[str] = []
        self.workdir = workdir or tempfile.mkdtemp(prefix="subsync-test-")
        self.secrets: Dict[str, str] = {}

    def progress(self, percent: int, stage: str) -> None:
        if not self.stages or self.stages[-1] != stage:
            self.stages.append(stage)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def check(self) -> None:
        self.token.raise_if_cancelled()


def _write_wav(path: str, audio: np.ndarray) -> None:
    import soundfile as sf

    sf.write(path, audio, SR, subtype="PCM_16")


def _starts_error(doc: SubtitleDoc, expected: List[float]) -> float:
    got = sorted(c.start for c in doc.cues)
    assert len(got) == len(expected), f"{len(got)} cues, expected {len(expected)}"
    return float(np.max(np.abs(np.array(got) - np.array(sorted(expected)))))


# ------------------------------------------------------------------ vad


def test_vad_finds_speech_under_a_score():
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    probs = probs_of(sc)
    speech = probs > 0.5
    truth = np.zeros(len(probs), dtype=bool)
    for a, b in sc.truth:
        truth[int(a / vad.HOP_S):int(b / vad.HOP_S)] = True
    precision = (speech & truth).sum() / max(1, speech.sum())
    recall = (speech & truth).sum() / max(1, truth.sum())
    # Edges: each true line's start against the nearest run start.
    runs = vad.speech_runs(probs)
    starts = np.array([a for a, _ in runs])
    errors = [float(np.min(np.abs(starts - a))) for a, _ in sc.truth]
    median_edge = float(np.median(errors))
    _CACHE["vad_numbers"] = (precision, recall, median_edge)
    assert precision > 0.8, f"precision {precision:.2f}"
    assert recall > 0.8, f"recall {recall:.2f}"
    assert median_edge < 0.06, f"median start error {median_edge * 1000:.0f} ms"


def test_vad_ignores_the_score_alone():
    rng = np.random.default_rng(3)
    bed = (0.03 * score_bed(120.0, rng)).astype(np.float32)
    probs, _ = vad.activity_from_samples(bed)
    fraction = float((probs > 0.5).mean())
    assert fraction < 0.1, f"music alone called speech {fraction:.0%} of the time"


def test_vad_streams_a_file_and_matches_in_memory():
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    folder = tempfile.mkdtemp(prefix="subsync-vad-")
    try:
        path = os.path.join(folder, "a.wav")
        _write_wav(path, sc.audio)
        seen = []
        streamed, hop = vad.speech_activity(path, progress=seen.append)
        assert hop == vad.HOP_S
        in_memory = probs_of(sc)
        n = min(len(streamed), len(in_memory))
        assert abs(len(streamed) - len(in_memory)) <= 2
        # 16-bit quantisation only.
        assert float(np.mean(np.abs(streamed[:n] - in_memory[:n]) > 0.1)) < 0.01
        assert seen and seen[-1] == 100
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_vad_engine_statuses_and_missing_pack():
    statuses = {s["id"]: s for s in vad.engine_statuses()}
    assert statuses["energy"]["available"] is True
    assert statuses["silero"]["pack"] == vad.SILERO_PACK
    if not statuses["silero"]["available"]:
        from audiosync.subs.tasks import TaskError

        try:
            vad.speech_activity(__file__, engine="silero")
        except TaskError as exc:
            assert "pack" in str(exc).lower()
        else:
            raise AssertionError("silero without its pack should raise TaskError")


def test_hysteresis_bridges_short_gaps_without_extending_runs():
    p = np.zeros(300, dtype=np.float32)
    p[50:100] = 0.9
    p[110:160] = 0.9  # 100 ms gap: bridged
    p[250:260] = 0.9  # too short to be speech
    runs = vad._runs(vad.hysteresis(p))
    assert runs == [(50, 160)], runs


# ------------------------------------------------------------------ frame rates


def test_rate_candidates_are_exact():
    rates = {round(c.ratio, 12): c for c in sync.rate_candidates()}
    assert round(float(PAL), 12) in rates
    assert rates[round(float(PAL), 12)].label(24000 / 1001) == "25 → 23.976"
    assert round(1001 / 1000, 12) in rates  # 24 -> 23.976 and 30 -> 29.97
    assert sync.rate_candidates()[0].ratio == 1.0
    assert sync.framerate_label(1.0) is None


# ------------------------------------------------------------------ audio engine


def test_audio_offset_ratio_and_cut():
    sc = scenario()
    outcome = sync.sync_to_signal(sc.doc, probs_of(sc), OPTIONS, "audio")
    _CACHE["main_outcome"] = outcome
    assert outcome.applied, outcome.warnings
    assert outcome.ratio == float(PAL), f"ratio {outcome.ratio!r}"
    assert outcome.framerate is not None and outcome.framerate.startswith("25 → ")
    assert abs(outcome.offset_s - sc.offset) < 0.05, f"offset {outcome.offset_s:+.3f}, expected {sc.offset:+.3f}"
    assert len(outcome.splits) == 1, [s.to_dict() for s in outcome.splits]
    split = outcome.splits[0]
    assert abs(split.at_s - sc.split_at) < 1.0, f"split at {split.at_s:.2f}, expected {sc.split_at:.2f}"
    assert abs(split.offset_s - sc.offset_after) < 0.1, f"after the cut {split.offset_s:+.3f}, expected {sc.offset_after:+.3f}"
    assert outcome.cues_dropped == sc.lines_in_cut, f"dropped {outcome.cues_dropped}, expected {sc.lines_in_cut}"
    assert outcome.confidence == "high"
    # Every line that is in the video now starts where it is spoken.
    error = _starts_error(outcome.doc, [a for a, _ in sc.truth])
    assert error < 0.15, f"worst line {error * 1000:.0f} ms off"


def test_audio_plain_offset_both_signs():
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    probs = probs_of(sc)
    for shift in (3.25, -12.5):
        doc = sc.doc.shift(-shift)  # the subtitle runs `shift` early
        outcome = sync.sync_to_signal(doc, probs, {**OPTIONS, "detectFramerate": False}, "audio")
        assert abs(outcome.offset_s - shift) < 0.05, f"{shift}: got {outcome.offset_s:+.3f}"
        assert not outcome.splits
        assert outcome.ratio == 1.0


def test_audio_keeps_a_correct_subtitle():
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    outcome = sync.sync_to_signal(sc.doc, probs_of(sc), OPTIONS, "audio")
    assert outcome.ratio == 1.0 and not outcome.splits
    assert abs(outcome.offset_s) < 0.05, outcome.offset_s


def test_audio_unrelated_subtitle_scores_low():
    sc = scenario()
    other = scenario(seed=99)
    outcome = sync.sync_to_signal(other.doc, probs_of(sc), OPTIONS, "audio")
    _CACHE["unrelated"] = outcome
    assert outcome.confidence == "low", (outcome.confidence, outcome.peak_z)
    assert outcome.score < 0.25, outcome.score
    assert not outcome.applied
    assert outcome.cues_moved == 0 and outcome.doc.cues[0].start == other.doc.cues[0].start


def test_non_dialogue_cues_are_not_evidence():
    cues = [
        Cue(1.0, 2.0, "Hello."),
        Cue(3.0, 4.0, "[door slams]"),
        Cue(5.0, 6.0, "♪ la la ♪"),
        Cue(7.0, 30.0, "A very long sign"),
        Cue(31.0, 32.0, "Sign text", style="Signs"),
        Cue(33.0, 34.0, "Bye."),
        Cue(35.0, 36.0, "Again."),
    ]
    mask = sync.dialogue_mask(cues)
    assert mask.tolist() == [True, False, True, False, False, True, True], mask.tolist()


# ------------------------------------------------------------------ subtitle engine


def test_subtitle_engine_aligns_and_snaps():
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    reference = SubtitleDoc([Cue(a, b, "ref") for a, b in sc.truth])
    rng = np.random.default_rng(5)
    # The same lines, 2.5 s late, each start and end jittered by up to
    # 150 ms, as another subtitler would have timed them.
    jitter = rng.uniform(-0.15, 0.15, size=(len(sc.truth), 2))
    doc = SubtitleDoc([Cue(a + 2.5 + j0, b + 2.5 + j1, "line") for (a, b), (j0, j1) in zip(sc.truth, jitter)])
    signal = sync.activity([(c.start, c.end) for c in reference.cues])
    outcome = sync.sync_to_signal(doc, signal, {**OPTIONS, "detectFramerate": False}, "subtitle")
    assert abs(outcome.offset_s + 2.5) < 0.1, outcome.offset_s
    snapped, count = sync.snap_to_reference(outcome.doc, reference.cues)
    assert count == len(reference.cues), (count, len(reference.cues))
    assert [(c.start, c.end) for c in snapped.cues] == [(c.start, c.end) for c in reference.cues]


def test_snap_leaves_lines_split_differently():
    reference = [Cue(10.0, 14.0, "one long line"), Cue(20.0, 21.0, "short")]
    doc = SubtitleDoc([Cue(10.1, 11.5, "half"), Cue(11.6, 14.0, "other half"), Cue(20.2, 21.1, "short")])
    snapped, count = sync.snap_to_reference(doc, reference)
    assert count == 1
    assert (snapped.cues[2].start, snapped.cues[2].end) == (20.0, 21.0)
    assert snapped.cues[0].start == 10.1


# ------------------------------------------------------------------ reference engine


def test_reference_sign_two_extra_seconds_at_head():
    """The reference release has 2 s more at its head, so the subtitle timed
    for it must come out 2 s EARLIER on the video."""
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=150.0)
    folder = tempfile.mkdtemp(prefix="subsync-ref-")
    try:
        video = os.path.join(folder, "video.wav")
        reference = os.path.join(folder, "reference.wav")
        _write_wav(video, sc.audio)
        rng = np.random.default_rng(11)
        head = (0.02 * rng.standard_normal(2 * SR)).astype(np.float32)
        _write_wav(reference, np.concatenate([head, sc.audio]))
        # Timed for the reference: every line 2 s later than in the video.
        doc = SubtitleDoc([Cue(a + 2.0, b + 2.0, "x") for a, b in sc.truth])
        ctx = Ctx(folder)
        options = {**OPTIONS, "engine": "reference", "allowSplits": False}
        outcome = sync.sync_doc(doc, options, {"path": video}, {"path": reference}, ctx)
        assert abs(outcome.offset_s + 2.0) < 0.01, f"offset {outcome.offset_s:+.4f}, expected -2.000"
        error = _starts_error(outcome.doc, [a for a, _ in sc.truth])
        assert error < 0.01, error
        assert outcome.confidence == "high", outcome.confidence
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_reference_follows_a_cut_edit():
    """The reference has 2 s more at the head and 7 s more near 150 s: lines
    after the cut move 9 s earlier, and a line inside the 7 s the video does
    not have is dropped."""
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    folder = tempfile.mkdtemp(prefix="subsync-refcut-")
    try:
        video = os.path.join(folder, "video.wav")
        reference = os.path.join(folder, "reference.wav")
        rng = np.random.default_rng(2)
        head = (0.02 * rng.standard_normal(2 * SR)).astype(np.float32)
        extra = (0.05 * rng.standard_normal(7 * SR)).astype(np.float32)
        # Cut in the pause between two lines, as an editor would.
        at = min(((b0 + a1) / 2 for (_, b0), (a1, _) in zip(sc.truth, sc.truth[1:])), key=lambda t: abs(t - 150.0))
        cut = int(round(at * SR))
        _write_wav(video, sc.audio)
        _write_wav(reference, np.concatenate([head, sc.audio[:cut], extra, sc.audio[cut:]]))

        def to_reference(v: float) -> float:
            return v + 2.0 + (7.0 if v >= at else 0.0)

        cues = [Cue(to_reference(a), to_reference(a) + (b - a), "x", meta={"truth": a}) for a, b in sc.truth]
        cues.append(Cue(at + 4.0, at + 6.0, "only in the reference"))
        doc = SubtitleDoc(sorted(cues, key=lambda c: c.start))
        options = {**OPTIONS, "engine": "reference", "allowSplits": True}
        outcome = sync.sync_doc(doc, options, {"path": video}, {"path": reference}, Ctx(folder))
        assert outcome.cues_dropped == 1, outcome.report()
        assert abs(outcome.offset_s + 2.0) < 0.01, outcome.offset_s
        assert len(outcome.splits) == 1 and abs(outcome.splits[0].offset_s + 9.0) < 0.01, outcome.report()
        assert abs(outcome.splits[0].at_s - at) < 1.0, (at, outcome.report())
        worst = max(abs(c.start - c.meta["truth"]) for c in outcome.doc.cues)
        assert worst < 0.01, worst
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_reference_mappings_carry_cues():
    # Drift: the reference runs 1001/1000 slow and 500 ms late at t=0.
    mapping = sync.map_from_pair(500.0, 1.0, 1.0)
    assert abs(mapping(0.5) - 0.0) < 1e-9
    assert abs(mapping(1000.0 * 1.001 + 0.5) - 1000.0) < 1e-6

    class Seg:
        def __init__(self, start, end, source):
            self.start_s, self.end_s, self.source_start_s = start, end, source

        @property
        def length_s(self):
            return self.end_s - self.start_s

    class Plan:
        speed = 1.0
        # Reference 0-100 plays at video 0-100; reference 107-200 at video
        # 100-193: the video lacks 7 s of the reference at 100.
        dub_segments = [Seg(0.0, 100.0, 0.0), Seg(100.0, 193.0, 107.0)]

    mapping = sync.map_from_plan(Plan())
    doc = SubtitleDoc([Cue(50.0, 52.0, "a"), Cue(102.0, 104.0, "gone"), Cue(106.5, 109.0, "b"), Cue(150.0, 151.0, "c")])
    out, dropped = sync.carry(doc, mapping)
    assert dropped == 1
    assert [(round(c.start, 3), round(c.end, 3)) for c in out.cues] == [(50.0, 52.0), (100.0, 102.0), (143.0, 144.0)]


# ------------------------------------------------------------------ transcript engine


def _words(pairs: List[Tuple[str, float, float]]) -> List[Word]:
    return [Word(start, end, text) for text, start, end in pairs]


def test_transcript_alignment_matches_lines_to_words():
    words = _words([
        ("Well,", 10.0, 10.3), ("where", 10.4, 10.6), ("were", 10.6, 10.8), ("you", 10.8, 11.0),
        ("last", 11.0, 11.3), ("night?", 11.3, 11.7),
        ("uh", 13.0, 13.2), ("I", 13.5, 13.6), ("was", 13.6, 13.8), ("at", 13.8, 13.9), ("home.", 13.9, 14.4),
        ("Liar!", 16.0, 16.5),
        ("It's", 20.0, 20.2), ("the", 20.2, 20.3), ("truth,", 20.3, 20.7), ("Anna.", 20.8, 21.3),
    ])
    # The subtitle is 30 s late, drops "Well" and "uh", spells one word
    # differently, and has a line nobody says.
    doc = SubtitleDoc([
        Cue(40.2, 42.0, "Where were you\nlast night?"),
        Cue(43.4, 45.0, "<i>I was at home.</i>"),
        Cue(46.0, 47.0, "LIAR!"),
        Cue(48.0, 49.0, "[thunder]"),
        Cue(50.0, 52.0, "It's the truth, Ana."),
    ])
    out, matched, _ = sync.align_to_words(doc, words, prior=lambda t: t - 30.0)
    starts = [round(c.start, 2) for c in out.cues]
    assert matched == 4, matched
    assert starts[:3] == [10.4, 13.5, 16.0], starts
    assert starts[4] == 20.0, starts
    assert abs(out.cues[1].end - 14.4) < 1e-9
    # The unmatched line moves with its neighbours (-30 s) and keeps its length.
    assert abs(out.cues[3].start - 18.0) < 1e-6 and abs(out.cues[3].duration - 1.0) < 1e-6


def test_transcript_alignment_cjk_per_character():
    # A recogniser splits Japanese into its own words; the subtitle has no
    # spaces and a different punctuation.
    words = _words([("今日は", 5.0, 5.6), ("とても", 5.6, 6.0), ("寒い", 6.0, 6.4), ("です", 6.4, 6.7), ("ね", 6.7, 6.8),
                    ("本当", 9.0, 9.4), ("に", 9.4, 9.5), ("それで", 9.5, 9.9), ("いい", 9.9, 10.1), ("の", 10.1, 10.3)])
    doc = SubtitleDoc([Cue(5.5, 7.0, "今日はとても寒いですね。"), Cue(9.3, 10.5, "本当にそれでいいの？")])
    out, matched, _ = sync.align_to_words(doc, words)
    assert matched == 2
    assert [round(c.start, 2) for c in out.cues] == [5.0, 9.0]
    assert sync.normalize_tokens("Ça va?  Très BIEN!") == ["ca", "va", "tres", "bien"]
    assert sync.normalize_tokens("안녕 하세요") == ["안", "녕", "하", "세", "요"]


def test_align_tokens_semi_global():
    pairs = sync.align_tokens(["b", "c", "d"], ["x", "a", "b", "c", "z", "d", "y"])
    assert pairs == [(0, 2), (1, 3), (2, 5)], pairs


# ------------------------------------------------------------------ task


def test_run_task_writes_synced_subtitle_and_report():
    try:
        from audiosync.subs import formats  # noqa: F401
    except ImportError:
        print("        (skipped: formats.py not available)")
        return
    from audiosync.subs import tasks

    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    folder = tempfile.mkdtemp(prefix="subsync-task-")
    try:
        video = os.path.join(folder, "Episode.wav")
        _write_wav(video, sc.audio)
        srt = os.path.join(folder, "Episode.en.srt")
        formats.write(sc.doc.shift(4.0), srt, "srt")
        ctx = Ctx(folder)
        job = {
            "id": "1", "task": "sync",
            "input": {"subtitle": {"path": srt}, "video": {"path": video, "audioTrack": 0}},
            "options": {**OPTIONS, "detectFramerate": False},
            "output": {"format": "srt"},
        }
        result = sync.run_task(job, ctx)
        report = result.report
        assert abs(report["offsetMs"] + 4000.0) < 50, report
        for key in ("offsetMs", "ratio", "framerate", "splits", "score", "confidence", "method", "cuesMoved", "cuesDropped"):
            assert key in report, key
        assert report["cuesMoved"] == len(sc.doc.cues)
        assert result.outputs and os.path.isfile(result.outputs[0].path)
        assert ".synced" in os.path.basename(result.outputs[0].path)
        written = formats.read(result.outputs[0].path)
        assert abs(written.cues[0].start - sc.doc.cues[0].start) < 0.06
        assert "Finding speech" in ctx.stages and result.summary
        assert isinstance(tasks.TASKS["sync"], str)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_cancelling_stops_the_sync():
    sc = scenario(cut_at=None, ratio=1.0, offset=0.0, length_s=300.0)
    token = CancellationToken()
    token.cancel()
    try:
        sync.sync_to_signal(sc.doc, probs_of(sc), OPTIONS, "audio", token.raise_if_cancelled)
    except Cancelled:
        return
    raise AssertionError("a cancelled token must stop the sync")


def test_engine_statuses_shape():
    for status in sync.engine_statuses():
        assert set(status) == {"id", "label", "available", "reason", "pack"}
    ids = [s["id"] for s in sync.engine_statuses()]
    assert ids == ["audio", "subtitle", "reference", "transcript"]


# ------------------------------------------------------------------ runner


def _numbers() -> None:
    """The accuracy figures, for the record."""
    if "vad_numbers" in _CACHE:
        p, r, e = _CACHE["vad_numbers"]
        print(f"  vad: precision {p:.2f}, recall {r:.2f}, median start error {e * 1000:.0f} ms ({phrases()[1]} speech)")
    if "main_outcome" in _CACHE:
        sc, o = scenario(), _CACHE["main_outcome"]
        print(f"  audio: offset {o.offset_s:+.3f} (truth {sc.offset:+.3f}), ratio {o.ratio:.9f} (truth {float(PAL):.9f}), z {o.peak_z:.1f}, score {o.score:.2f}")
        for s in o.splits:
            print(f"         split at {s.at_s:.2f} (truth {sc.split_at:.2f}), offset {s.offset_s:+.3f} (truth {sc.offset_after:+.3f})")
    if "unrelated" in _CACHE:
        o = _CACHE["unrelated"]
        print(f"  unrelated subtitle: z {o.peak_z:.1f}, score {o.score:.2f}, confidence {o.confidence}")


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
    _numbers()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
