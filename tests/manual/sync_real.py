"""Subtitle sync on real films (manual; not part of CI).

Three kinds of check:

1.  **Films that carry their own subtitle track.** The track is taken as
    ground truth, broken the ways real subtitles are broken -- shifted,
    timed at another frame rate, from a longer or shorter cut -- and synced
    back against the film's own audio. Every line's error against its
    original time is measured.
2.  **A film with no subtitles** (the Korean drama). Lines are made from the
    Silero detector's speech runs over the whole film (speech pack), broken
    the same ways, and
    synced back with the built-in detector: two independent detectors, so
    the result is not the detector agreeing with itself. Without the pack
    the built-in detector's own runs are used, and the check says so.
3.  **The transcript engine** on English speech from ``say`` (needs a
    speech recognition pack and faster-whisper's small model).

Usage:
    python/.venv/bin/python tests/manual/sync_real.py [films] [kdrama] [transcript] [--quick]

Films are read from the paths below when present; missing ones are skipped.
``--quick`` limits every film to its first 45 minutes. Set
AUDIOSYNC_PACKS_DIR for the speech pack.
"""

from __future__ import annotations

import os
import resource
import sys
import tempfile
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from audiosync.media import CancellationToken  # noqa: E402
from audiosync.subs import sync, tasks, vad  # noqa: E402
from audiosync.subs.model import Cue, SubtitleDoc  # noqa: E402

HOME = os.path.expanduser("~")
# (label, video, subtitle track 0:s:N, audio track 0:a:N)
SUBTITLED = [
    ("anime, JA audio / EN subs", os.path.join(HOME, "Movies", "PSArips.com | Demon.Slayer.Mugen.Train.A.K.A.Kimetsu.no.Yaiba.Mugen.Ressha-Hen.2020.DUAL-AUDIO.JPN-ENG.1080p.10bit.BluRay.6CH.x265.HEVC-PSA.mkv"), 0, 1),
    ("drama, IT audio / RO subs", os.path.join(HOME, "Downloads", "Here Now aka Fino alla fine (2024) 1080p BluRay x265 10bit HEVC Italian DDP 5.1 - St0neR (1).mkv"), 1, 1),
]
UNSUBTITLED = ("K-drama, KO audio", os.path.join(HOME, "Downloads", "Video.mkv"), 0)

PAL = float(Fraction(25 * 1001, 24000))
OPTIONS = {"engine": "audio", "maxOffsetS": 60, "detectFramerate": True, "allowSplits": True, "splitPenalty": 7, "vad": "energy"}


class Ctx:
    def __init__(self) -> None:
        self.token = CancellationToken()
        self.workdir = tempfile.mkdtemp(prefix="subsync-real-")
        self.secrets: Dict[str, str] = {}

    def progress(self, percent: int, stage: str) -> None:
        pass

    def log(self, message: str) -> None:
        pass

    def check(self) -> None:
        self.token.raise_if_cancelled()


def variants(doc: SubtitleDoc, middle: float) -> List[Tuple[str, SubtitleDoc]]:
    """The ways a subtitle arrives broken. Every line keeps its true start
    in ``meta["truth"]``, which survives every retime."""
    doc = doc.copy([c.copy(meta={**c.meta, "truth": c.start}) for c in doc.cues])
    out = [
        ("as released", doc.copy()),
        ("+3.25 s", doc.shift(3.25)),
        ("-41.7 s", doc.shift(-41.7)),
        ("25 fps, +3.25 s", doc.scale(1 / PAL).shift(3.25)),
        ("24 fps (x1000/1001)", doc.scale(1000 / 1001)),
        # From a longer cut: the subtitle's release has 7 s more at `middle`.
        ("longer cut (+7 s mid)", doc.map_times(lambda t: t if t < middle else t + 7.0)),
    ]
    # From a shorter cut: 20 s of the video is not in the subtitle's release.
    kept = [c for c in doc.cues if not (middle <= c.start < middle + 20.0)]
    out.append(("shorter cut (-20 s mid)", doc.copy(kept).map_times(lambda t: t if t < middle else t - 20.0)))
    return out


def errors(result: SubtitleDoc, broken: SubtitleDoc) -> Tuple[np.ndarray, int]:
    """Per-line start error against the truth, and lines dropped."""
    errs = np.array([c.start - c.meta["truth"] for c in result.cues if "truth" in c.meta])
    return errs, len(broken.cues) - len(result.cues)


def report_row(name: str, outcome, errs: np.ndarray, dropped: int) -> None:
    a = np.abs(errs) if len(errs) else np.array([np.nan])
    splits = ", ".join(f"{s.at_s:.1f}s->{s.offset_s:+.3f}" for s in outcome.splits) or "-"
    print(
        f"    {name:24s} offset {outcome.offset_s:+8.3f}  ratio {outcome.ratio:.6f}  splits {splits:28s} "
        f"z {outcome.peak_z:5.1f}  {outcome.confidence:6s} score {outcome.score:.2f}  "
        f"lines: median {np.nanmedian(a) * 1000:5.0f} ms, p90 {np.nanpercentile(a, 90) * 1000:6.0f} ms, "
        f"within 0.1 s {np.mean(a < 0.1) * 100:5.1f}%  dropped {dropped}"
    )


def activity(path: str, track: int, engine: str = "energy", limit_s: Optional[float] = None) -> np.ndarray:
    started = time.time()
    probs, _ = vad.speech_activity(path, track, engine)
    took = time.time() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    print(f"    {engine} speech activity: {len(probs) * vad.HOP_S / 60:.1f} min of audio in {took:.1f} s, "
          f"speech {np.mean(probs > 0.5) * 100:.0f}%, peak RSS so far {rss:.0f} MB")
    if limit_s:
        probs = probs[: int(limit_s / vad.HOP_S)]
    return probs


def subtitled(quick: bool) -> None:
    for label, path, sub_track, audio_track in SUBTITLED:
        if not os.path.isfile(path):
            print(f"  skip {label}: not found")
            continue
        print(f"  {label}: {os.path.basename(path)[:60]}")
        limit = 2700.0 if quick else None
        doc = tasks.load_subtitle({"path": path, "track": sub_track}, Ctx())
        if limit:
            doc = doc.copy([c for c in doc.cues if c.end < limit])
        probs = activity(path, audio_track, limit_s=limit)
        middle = (doc.cues[len(doc.cues) // 2].start + doc.cues[len(doc.cues) // 2 - 1].end) / 2
        for name, broken in variants(doc, middle):
            started = time.time()
            outcome = sync.sync_to_signal(broken, probs, OPTIONS, "audio")
            errs, dropped = errors(outcome.doc, broken)
            report_row(name, outcome, errs, dropped)
            if name == "as released":
                print(f"      (sync itself: {time.time() - started:.2f} s)")
        others = [s for s in SUBTITLED if s[1] != path and os.path.isfile(s[1])]
        if others:
            other = tasks.load_subtitle({"path": others[0][1], "track": others[0][2]}, Ctx())
            outcome = sync.sync_to_signal(other, probs, OPTIONS, "audio")
            print(f"    {'another film subtitle':24s} applied {outcome.applied}  z {outcome.peak_z:.1f}  "
                  f"{outcome.confidence}  score {outcome.score:.2f}")


def lines_from_runs(probs: np.ndarray, until_s: float) -> SubtitleDoc:
    """Subtitle-like lines from speech runs: runs closer than 0.3 s joined,
    long ones cut at 6 s, blips dropped."""
    runs = [(a, b) for a, b in vad.speech_runs(probs) if a < until_s]
    joined: List[List[float]] = []
    for a, b in runs:
        if joined and a - joined[-1][1] < 0.3:
            joined[-1][1] = b
        else:
            joined.append([a, b])
    cues = []
    for a, b in joined:
        while b - a > 6.0:
            cues.append(Cue(a, a + 4.0, f"line {len(cues)}"))
            a += 4.2
        if b - a >= 0.5:
            cues.append(Cue(a, b, f"line {len(cues)}"))
    return SubtitleDoc(cues)


def unsubtitled(quick: bool) -> None:
    label, path, track = UNSUBTITLED
    if not os.path.isfile(path):
        print(f"  skip {label}: not found")
        return
    print(f"  {label}: {os.path.basename(path)}")
    silero_ok = any(s["id"] == "silero" and s["available"] for s in vad.engine_statuses())
    limit = 2700.0 if quick else None
    energy = activity(path, track, "energy", limit)
    if silero_ok:
        truth_probs = activity(path, track, "silero", limit)
        source = "Silero (speech pack)"
    else:
        truth_probs = energy
        source = "the built-in detector itself (circular; install the speech pack for an independent check)"
    doc = lines_from_runs(truth_probs, len(truth_probs) * vad.HOP_S)
    print(f"    lines from {source}: {len(doc.cues)}")
    middle = (doc.cues[len(doc.cues) // 2].start + doc.cues[len(doc.cues) // 2 - 1].end) / 2
    for name, broken in variants(doc, middle):
        outcome = sync.sync_to_signal(broken, energy, OPTIONS, "audio")
        errs, dropped = errors(outcome.doc, broken)
        report_row(name, outcome, errs, dropped)


ENGLISH = [
    "Where were you last night? I waited for hours.", "Don't tell me you forgot again.",
    "We have to leave before the storm comes.", "I never said that. You know I didn't.",
    "Listen to me, just this once.", "The train leaves at seven, so we should hurry.",
    "Have you seen my keys anywhere?", "Nobody told me the meeting was cancelled.",
    "It's not your fault. It never was.", "Can we talk about this tomorrow?",
    "I think someone is following us.", "Turn left at the next corner.",
    "She said she would call, but she never did.", "Why is the door open?",
    "Let's get out of here.", "I'm sorry. I didn't mean it.",
]


def transcript(quick: bool) -> None:
    """Transcript engine end to end: English speech from `say` at known
    times, a subtitle 12.3 s late and at 25 fps, faster-whisper small."""
    import shutil
    import subprocess

    import soundfile as sf

    if not shutil.which("say") or not any(s["available"] for s in __import__("audiosync.subs.generate", fromlist=["x"]).engine_statuses()):
        print("  skip: needs macOS say and a speech recognition pack")
        return
    folder = tempfile.mkdtemp(prefix="subsync-transcript-")
    rng = np.random.default_rng(4)
    clips = []
    for i, text in enumerate(ENGLISH):
        path = os.path.join(folder, f"{i}.wav")
        voice = "Samantha" if i % 2 else "Albert"
        subprocess.run(["say", "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", path, text], check=True)
        x, _ = sf.read(path, dtype="float32")
        loud = np.flatnonzero(np.abs(x) > 0.02 * np.abs(x).max())
        clips.append(x[loud[0]:loud[-1] + 1])
    t, cues, audio = 3.0, [], np.zeros(int(200 * 16000), dtype=np.float32)
    for round_ in range(3):
        for i in rng.permutation(len(ENGLISH)):
            clip = clips[i]
            a = int(t * 16000)
            if a + len(clip) >= len(audio):
                break
            audio[a:a + len(clip)] += 0.3 * clip
            cues.append(Cue(t, t + len(clip) / 16000, ENGLISH[i], meta={"truth": t}))
            t += len(clip) / 16000 + rng.uniform(0.8, 2.5)
    audio += 0.01 * rng.standard_normal(len(audio)).astype(np.float32)
    video = os.path.join(folder, "speech.wav")
    sf.write(video, audio, 16000, subtype="PCM_16")
    broken = SubtitleDoc(cues, language="en").scale(1 / PAL).shift(12.3)
    ctx = Ctx()
    started = time.time()
    options = {**OPTIONS, "engine": "transcript", "asrEngine": "faster-whisper", "asrModel": "small"}
    outcome = sync.sync_doc(broken, options, {"path": video}, None, ctx)
    errs, dropped = errors(outcome.doc, broken)
    report_row(f"transcript ({time.time() - started:.0f} s)", outcome, errs, dropped)
    shutil.rmtree(folder, ignore_errors=True)


def main() -> None:
    quick = "--quick" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not only or "films" in only:
        print("Films with their own subtitles (track = truth):")
        subtitled(quick)
    if not only or "kdrama" in only:
        print("\nFilm without subtitles:")
        unsubtitled(quick)
    if not only or "transcript" in only:
        print("\nTranscript engine:")
        transcript(quick)


if __name__ == "__main__":
    main()
