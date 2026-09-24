"""The lip check: does a dub's speech move with the mouths on screen?

Every other check in the dub sync compares sound with sound -- the dub's
music and effects with the original's -- which says where the dub master
sits, not whether its voices meet the lips. This one looks at the picture.
At spots where someone talks on screen, the opening of the largest face's
mouth, frame by frame (Apple's Vision, see ``lipsync/mouth.swift``), is
correlated with the voice energy of a soundtrack (see ``speech``); the lag
at which they agree best is where that soundtrack's speech sits against
the lips.

It is measured twice at every spot. First on the original: its own lip
sync is the reference, and it carries the method's own bias as well -- a
mouth opens a little before the sound comes out, and which lag the
agreement peaks at depends on how people talk. Then on the dub. The dub's
lag less the original's is the dub's lip error there, in any language: a
dub is performed to the picture, so its lines open and close with the
same mouths even where the words differ. The agreement is weaker than a
language's own, which is why only spots where both measurements stand
clear of their noise are judged, and why the check is for errors of
frames, not milliseconds -- the milliseconds come from the music and
effects.

macOS only: the mouths are read with Vision, compiled on first use with
the Swift compiler the Xcode command line tools install. Elsewhere, or
without them, the check reports that it could not run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .dubsync import DubSyncPlan, _clock
from .media import CancellationToken, MediaError, _run, load_audio
from .speech import SPEECH_RATE, speech_activity, speech_energy

ANALYSIS_SR = 16000
# Spots are this long, this many along the film, never closer than this.
LIP_WINDOW_S = 10.0
LIP_SPOTS = 40
# Candidate spots are tried this far apart inside the dub's stretches.
LIP_CANDIDATE_STEP_S = 30.0
# A spot is worth reading when the original talks for this share of it ...
LIP_MIN_SPEECH_SHARE = 0.3
# ... and judged when a face is found in this share of its frames.
LIP_MIN_FACE_SHARE = 0.5
# Lags looked at, either way, and the step between them.
LIP_RANGE_MS = 400.0
# A lag is a measurement when its peak stands this many noise units above
# the other lags' agreement.
LIP_MIN_STRENGTH = 2.5
# Past this the dub's voices visibly miss the lips (see dubsync.AUDIBLE_MS).
LIP_TOLERANCE_MS = 45.0
# A mouth frame this far from the nearest face-found frame is not guessed.
LIP_MAX_HOLE_S = 0.15

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lipsync", "mouth.swift")

LogFn = Callable[[str], None]


def _binary() -> Optional[str]:
    """The compiled mouth reader, built from its source on first use."""
    if platform.system() != "Darwin" or not os.path.isfile(SOURCE):
        return None
    with open(SOURCE, "rb") as handle:
        digest = hashlib.sha1(handle.read()).hexdigest()[:12]
    target = os.path.join(os.path.expanduser("~/Library/Caches/AudioSyncMaster"), f"mouth-{digest}")
    if os.path.isfile(target) and os.access(target, os.X_OK):
        return target
    compiler = shutil.which("swiftc")
    if not compiler:
        return None
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        subprocess.run([compiler, "-O", "-o", target + ".part", SOURCE], check=True, capture_output=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return None
    os.replace(target + ".part", target)
    return target


def available() -> bool:
    """Whether the lip check can run here."""
    return _binary() is not None


@dataclass
class LipSpot:
    """One spot of the check."""

    start_s: float
    end_s: float
    faces: float
    """Share of the spot's frames with a face found."""
    reference_ms: Optional[float] = None
    """Where the original's speech sits against the mouths (+ = later)."""
    reference_strength: float = 0.0
    dub_ms: Optional[float] = None
    dub_strength: float = 0.0
    note: str = ""

    @property
    def error_ms(self) -> Optional[float]:
        """The dub's lip error here: its lag less the original's."""
        if self.reference_ms is None or self.dub_ms is None:
            return None
        return self.dub_ms - self.reference_ms

    def to_dict(self) -> dict:
        return {
            "startS": self.start_s, "endS": self.end_s, "faces": self.faces,
            "referenceMs": self.reference_ms, "referenceStrength": self.reference_strength,
            "dubMs": self.dub_ms, "dubStrength": self.dub_strength,
            "errorMs": self.error_ms, "note": self.note,
        }


@dataclass
class LipCheck:
    spots: List[LipSpot] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def judged(self) -> List[LipSpot]:
        return [s for s in self.spots if s.error_ms is not None]

    def to_dict(self) -> dict:
        errors = [abs(s.error_ms) for s in self.judged]
        return {
            "spots": [s.to_dict() for s in self.spots],
            "judged": len(errors),
            "typicalMs": float(np.median(errors)) if errors else None,
            "worstMs": float(max(errors)) if errors else None,
            "withinTolerance": sum(1 for e in errors if e <= LIP_TOLERANCE_MS),
            "toleranceMs": LIP_TOLERANCE_MS,
            "error": self.error,
        }

    def describe(self) -> str:
        if self.error:
            return f"the lip check could not run: {self.error}"
        judged = self.judged
        lines = []
        if judged:
            errors = [abs(s.error_ms) for s in judged]
            within = sum(1 for e in errors if e <= LIP_TOLERANCE_MS)
            lines.append(
                f"lips: judged at {len(judged)} of {len(self.spots)} spots with a talking face; the dub's voices sit "
                f"{np.median(errors):.0f} ms from where the original's do, typically, {max(errors):.0f} ms at worst; "
                f"{within} of {len(judged)} within {LIP_TOLERANCE_MS:.0f} ms"
            )
        else:
            lines.append(f"lips: none of {len(self.spots)} spots could be judged (no clear talking face, or too weak a match)")
        for spot in self.spots:
            where = f"  {_clock(spot.start_s)} - {_clock(spot.end_s)}  faces {100 * spot.faces:3.0f}%"
            if spot.error_ms is not None:
                flag = "  <-- check" if abs(spot.error_ms) > LIP_TOLERANCE_MS else ""
                lines.append(
                    f"{where}  original {spot.reference_ms:+5.0f} ms ({spot.reference_strength:.1f})  "
                    f"dub {spot.dub_ms:+5.0f} ms ({spot.dub_strength:.1f})  error {spot.error_ms:+5.0f} ms{flag}"
                )
            else:
                lines.append(f"{where}  {spot.note}")
        return "\n".join(lines)


def _mouths(
    video_path: str, lo_s: float, hi_s: float, workdir: str, token: Optional[CancellationToken]
) -> Tuple[np.ndarray, np.ndarray]:
    """(frame times on the video's clock, mouth opening, NaN where no face)."""
    from .dubrender import excerpt  # noqa: WPS433 - dubrender imports dubsync, as this does

    binary = _binary()
    if binary is None:
        raise MediaError("the mouth reader is not available here")
    shell = DubSyncPlan(video_path, video_path, video_duration_s=hi_s + 1.0)
    clip, start, _end = excerpt(shell, lo_s, hi_s, os.path.join(workdir, f"lips_{lo_s:.3f}.mp4"), "picture", token)
    out = _run([binary, clip], 900, token, what="read the mouths")
    times, opening = [], []
    for line in out.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        times.append(start + float(row["t"]))
        opening.append(float(row["open"]) if row.get("open") is not None else math.nan)
    try:
        os.remove(clip)
    except OSError:
        pass
    return np.asarray(times), np.asarray(opening)


def _lag(
    times: np.ndarray, opening: np.ndarray, curve: np.ndarray, curve_t0: float
) -> Tuple[Optional[float], float]:
    """Where a voice curve agrees best with the mouths, in ms (+ = the
    voice later than the mouths), and how far that agreement stands above
    the agreement at other lags, in noise units."""
    found = np.isfinite(opening)
    if found.sum() < 24:
        return None, 0.0
    grid = np.arange(times[0], times[-1], 1.0 / SPEECH_RATE)
    mouth = np.interp(grid, times[found], opening[found])
    # No guessing across a stretch without a face.
    nearest = np.min(np.abs(grid[:, None] - times[found][None, :]), axis=1) if len(grid) * found.sum() < 4e6 else None
    mask = np.ones(len(grid), dtype=bool) if nearest is None else nearest <= LIP_MAX_HOLE_S
    if mask.sum() < 3 * SPEECH_RATE:
        return None, 0.0
    m = (mouth - mouth[mask].mean()) / (mouth[mask].std() + 1e-9)
    reach = int(round(LIP_RANGE_MS / 1000.0 * SPEECH_RATE))
    lags = np.arange(-reach, reach + 1)
    base = np.round((grid - curve_t0) * SPEECH_RATE).astype(int)
    agreement = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        index = base + lag
        ok = mask & (index >= 0) & (index < len(curve))
        if ok.sum() < 3 * SPEECH_RATE:
            continue
        a = curve[index[ok]]
        if a.std() < 1e-9:
            continue
        agreement[i] = float(np.corrcoef(m[ok], a)[0, 1])
    if not np.isfinite(agreement).any():
        return None, 0.0
    peak = int(np.nanargmax(agreement))
    others = np.delete(agreement, range(max(0, peak - 5), min(len(agreement), peak + 6)))
    others = others[np.isfinite(others)]
    if len(others) < 5:
        return None, 0.0
    strength = float((agreement[peak] - others.mean()) / (others.std() + 1e-9))
    shift = 0.0
    if 0 < peak < len(agreement) - 1 and np.isfinite(agreement[peak - 1]) and np.isfinite(agreement[peak + 1]):
        y0, y1, y2 = agreement[peak - 1], agreement[peak], agreement[peak + 1]
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) > 1e-12:
            shift = max(-1.0, min(1.0, 0.5 * (y0 - y2) / denominator))
    return (lags[peak] + shift) * 1000.0 / SPEECH_RATE, strength


def _voice(path: str, track: int, lo_s: float, hi_s: float, token: Optional[CancellationToken]) -> Tuple[np.ndarray, float]:
    """The voice energy of a file across a span, and the time of its first value."""
    t0 = max(0.0, lo_s)
    pcm = load_audio(path, ANALYSIS_SR, duration=hi_s - t0, offset=t0, token=token, track=track)
    return speech_energy(pcm, ANALYSIS_SR), t0


def choose_spots(
    plan: DubSyncPlan, token: Optional[CancellationToken] = None, count: int = LIP_SPOTS, window_s: float = LIP_WINDOW_S,
) -> List[Tuple[float, float]]:
    """Spots inside the dub's stretches where the original talks, spread
    along the film: every LIP_CANDIDATE_STEP_S a candidate, the one in each
    stretch of the film where the original talks most."""
    candidates: List[Tuple[float, float]] = []
    for segment in plan.dub_segments:
        start = segment.start_s + 0.5
        while start + window_s <= segment.end_s - 0.5:
            candidates.append((start, start + window_s))
            start += LIP_CANDIDATE_STEP_S
    if not candidates:
        return []
    talk: List[float] = []
    for lo, hi in candidates:
        if token:
            token.raise_if_cancelled()
        try:
            pcm = load_audio(plan.video_path, ANALYSIS_SR, duration=hi - lo, offset=lo, token=token, track=plan.video_track)
            talk.append(float(np.mean(speech_activity(pcm, ANALYSIS_SR) > 0.3)))
        except MediaError:
            talk.append(0.0)
    chosen: List[Tuple[float, float]] = []
    edges = np.linspace(0.0, plan.video_duration_s, count + 1)
    for a, b in zip(edges, edges[1:]):
        inside = [(share, spot) for share, spot in zip(talk, candidates) if a <= spot[0] < b and share >= LIP_MIN_SPEECH_SHARE]
        if inside:
            chosen.append(max(inside)[1])
    return chosen


def lip_check(
    plan: DubSyncPlan,
    dub_audio_path: str,
    dub_track: int = 0,
    spots: Optional[Sequence[Tuple[float, float]]] = None,
    token: Optional[CancellationToken] = None,
    log: Optional[LogFn] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> LipCheck:
    """Measure the dub's lip error at spots with a talking face.

    ``dub_audio_path`` is the written track, on the video's clock (its time
    zero the video's); ``spots`` default to ``choose_spots``.
    """
    say = log or (lambda _m: None)
    result = LipCheck()
    if not available():
        result.error = "it needs macOS with the Swift compiler (xcode-select --install)"
        return result
    chosen = list(spots) if spots is not None else choose_spots(plan, token)
    say(f"lip check: {len(chosen)} spots where the original talks")
    with tempfile.TemporaryDirectory(prefix="audiosync-lips-") as workdir:
        for index, (lo, hi) in enumerate(chosen):
            if token:
                token.raise_if_cancelled()
            try:
                times, opening = _mouths(plan.video_path, lo, hi, workdir, token)
            except MediaError as exc:
                result.spots.append(LipSpot(lo, hi, 0.0, note=f"picture not read: {exc}"))
                continue
            faces = float(np.mean(np.isfinite(opening))) if len(opening) else 0.0
            spot = LipSpot(lo, hi, faces)
            if faces < LIP_MIN_FACE_SHARE:
                spot.note = "no talking face for most of it"
            else:
                margin = LIP_RANGE_MS / 1000.0 + 0.2
                original, t0 = _voice(plan.video_path, plan.video_track, lo - margin, hi + margin, token)
                dub, d0 = _voice(dub_audio_path, dub_track, lo - margin, hi + margin, token)
                reference, reference_strength = _lag(times, opening, original, t0)
                lag, strength = _lag(times, opening, dub, d0)
                spot.reference_strength, spot.dub_strength = reference_strength, strength
                if reference is None or reference_strength < LIP_MIN_STRENGTH:
                    spot.note = f"the original's own speech does not follow the mouths clearly ({reference_strength:.1f})"
                    spot.reference_ms = reference
                elif lag is None or strength < LIP_MIN_STRENGTH:
                    spot.reference_ms = reference
                    spot.note = f"the dub's speech does not follow the mouths clearly ({strength:.1f})"
                else:
                    spot.reference_ms, spot.dub_ms = reference, lag
            result.spots.append(spot)
            if progress:
                progress((index + 1) / max(1, len(chosen)))
            if spot.error_ms is not None:
                say(f"  {_clock(lo)}: original {spot.reference_ms:+.0f} ms, dub {spot.dub_ms:+.0f} ms, error {spot.error_ms:+.0f} ms")
            else:
                say(f"  {_clock(lo)}: {spot.note}")
    return result
