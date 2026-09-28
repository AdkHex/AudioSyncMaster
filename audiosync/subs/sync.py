"""Re-time a subtitle to a video.

A subtitle that is out of sync is almost never out of sync at random. It was
timed for some release of the same programme, and that release differs from
this one in one of three ways, often at once:

*   **a constant offset** -- a logo or a recap at the head, a different
    encoder delay;
*   **a speed difference** -- timed at 25 fps (PAL speed-up) for a 23.976 fps
    video, or 24 for 23.976, which drifts steadily through the film;
*   **different edits** -- a scene the other release has and this one lacks,
    or the reverse, so the offset steps at each such place.

So every engine here recovers a *mapping* of that shape -- a ratio, and a
constant offset per stretch of the film -- rather than moving cues one by
one. A mapping of that shape is what a real edit produces, and forcing the
answer into it is what lets many weakly-evidenced cues vote together: a line
that sits over music or crosstalk still lands where its neighbours say.

The engines differ only in what the subtitle is compared with:

*   **audio** -- the video's own speech activity (``vad``). Any language: the
    question is *when* someone talks, not what they say.
*   **subtitle** -- a correctly timed subtitle in any language, whose cues say
    when someone talks. Afterwards, cues that clearly pair one-to-one with a
    reference cue take its exact timing.
*   **reference** -- the audio of the release the subtitle was timed for. The
    two audio tracks are measured against each other with the dub-sync
    machinery, which is exact to the millisecond, and every cue is carried
    across that mapping.
*   **transcript** -- speech recognition of the video: the subtitle's words
    are aligned to the recognised words, and each line takes the time its
    words were spoken.

Time mappings throughout: ``video_time = ratio * subtitle_time + offset``.
A positive offset moves the subtitle later.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..correlate import _fast_fft_size
from ..media import CancellationToken, MediaError, probe
from . import vad
from .model import Cue, SubtitleDoc, Word, strip_tags

HOP_S = vad.HOP_S

# --- frame rates -------------------------------------------------------------

# The rates a subtitle is realistically timed at, as the exact rationals they
# are (see framerate.COMMON_RATES for why 23.976 must be 24000/1001).
SYNC_RATES: Tuple[Fraction, ...] = (
    Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
)

# --- matching ---------------------------------------------------------------

# Every correlation curve is judged after its slow part is taken off (a
# moving average over SHARPEN_S). Speech comes in runs of seconds and
# dialogue in scenes of minutes, so the raw curve is a landscape of broad
# hills -- any offset that lays the cues over a talkative scene scores well
# -- and its tallest hill says little. What only the right offset produces is
# a *sharp* peak on top, where dozens of cue edges meet speech edges at once.
# Measured on two real films, 5-minute windows at the right offset scored a
# median z of 9 (min 4) this way against 4-5 by the raw curve, while
# subtitles of another film never passed 4.5.
SHARPEN_S = 4.0

# Peak prominence of the sharpened curve, in robust standard deviations of
# every other offset tried. A subtitle of another film peaks at 2-5; the
# right one against a feature-length film at 10-40.
REJECT_Z = 6.0
HIGH_Z = 10.0
# Around the peak, its own shoulders are left out of the noise estimate.
PEAK_SHOULDER_S = 2.0

# A ratio is kept only when its sharpened peak beats the unscaled one by
# this factor. The right conversion wins by more the longer the film: on
# 45 minutes, 24 -> 23.976 scored 1.13-1.3x the unscaled peak, whose drift
# of 2.7 s smears it; a coincidence must not turn a correct subtitle into a
# slowly drifting one, and preferring no conversion on a near-tie costs at
# most the drift of a short clip.
RATIO_MARGIN = 1.05

# Windows for finding the offsets a cut edit produces: five minutes, which
# is what it takes for the sharp peak to stand clear of chance (see
# SHARPEN_S), each window a third of the way on from the last.
WINDOW_S = 300.0
WINDOW_HOP_S = 100.0
WINDOW_MIN_CUES = 8
WINDOW_Z = 5.5
WINDOW_PEAKS = 2
# Offsets closer than this are the same offset, measured twice.
SAME_OFFSET_S = 0.2
MAX_CANDIDATES = 12

# The splitPenalty option is in these units: seconds of cue time that has to
# land on speech at the new offset (and not at the old) before a split pays
# for itself. At the default 7 that is about 3.5 s of cleanly placed
# dialogue, three or four lines.
PENALTY_UNIT = 0.5
# A stretch with fewer lines than this cannot be told apart from noise.
MIN_SEGMENT_CUES = 3
# ...and every stretch the path proposes must also stand on its own: its
# lines alone must peak at its offset, this clearly. A path can string
# together a few lucky lines at a wrong offset; a stretch whose own evidence
# points there cannot be luck.
SEGMENT_Z = 5.0
SEGMENT_AGREE_S = 0.5
# At a cut, lines on either side may overlap by this much and still be in
# order; two lines of one subtitle overlap by a few frames at most.
OVERLAP_S = 0.2

# Final placement of each stretch: the offset is searched this far either
# side of its candidate, at the 10 ms hop, then interpolated between hops.
REFINE_S = 0.4

# Cue starts are where a subtitler put the line's first word, and ends are
# where the reading time ran out -- often a second after the speech. Laid
# over speech runs, a cue that overhangs its line fits equally well anywhere
# across the overhang, and the middle of that plateau is early by half of
# it: on two real films, activity alone placed correctly timed subtitles
# 320 ms and 160 ms early. So the fine placement adds the agreement of cue
# starts with rises in speech, weighted against the activity by this much
# (each on the scale of its own spread). Weights of 1.25-3 all land those
# films within 25 ms of their own timing; below 1 the plateau can still win.
START_WEIGHT = 1.5
START_SPREAD_S = 0.15

# --- snapping (subtitle engine) --------------------------------------------

SNAP_START_S = 0.4
SNAP_DURATION_RATIO = 1.5
SNAP_DURATION_S = 0.3

# --- transcript -------------------------------------------------------------

TRANSCRIPT_CHUNK_CUES = 40
TRANSCRIPT_MARGIN_S = 30.0
MIN_MATCHED_FRACTION = 0.5
MIN_CUE_S = 0.5

# Styles and texts that are not dialogue. ASS typesetting ("signs"),
# karaoke and songs, and SDH-only lines ("[door slams]", "♪") say nothing
# about when someone talks, and a sign held for ten seconds would pull the
# correlation towards itself.
_NON_DIALOGUE_STYLE = re.compile(r"sign|song|kara|op\b|ed\b|opening|ending|title|note|typeset|insert", re.IGNORECASE)
_SDH_ONLY = re.compile(r"^\s*(?:[\[\(（【][^\]\)）】]*[\]\)）】]\s*|[♪♫#*~\-–—\s])*$")
MAX_DIALOGUE_CUE_S = 12.0

ProgressFn = Callable[[int, str], None]


# ============================================================ outcome


@dataclass
class Split:
    """Where the offset changes: from ``at_s`` on the video, ``offset_s``."""

    at_s: float
    offset_s: float

    def to_dict(self) -> Dict[str, float]:
        return {"atS": round(self.at_s, 3), "offsetMs": round(self.offset_s * 1000.0, 1)}


@dataclass
class SyncOutcome:
    doc: SubtitleDoc
    #: Offset at the start of the video, seconds (positive = later).
    offset_s: float
    ratio: float = 1.0
    splits: List[Split] = field(default_factory=list)
    #: 0-1: how well the result fits the evidence (see _fit_score).
    score: float = 0.0
    method: str = "audio"
    confidence: str = "low"
    #: e.g. "25 → 23.976" when a frame-rate conversion was applied.
    framerate: Optional[str] = None
    cues_moved: int = 0
    cues_dropped: int = 0
    warnings: List[str] = field(default_factory=list)
    #: Peak prominence (z) of the match, for the log and the tests.
    peak_z: float = 0.0
    #: False when no match was clear enough to act on: the timing was kept.
    applied: bool = True

    def report(self) -> Dict[str, Any]:
        return {
            "offsetMs": round(self.offset_s * 1000.0, 1),
            "ratio": self.ratio,
            "framerate": self.framerate,
            "splits": [s.to_dict() for s in self.splits],
            "score": round(self.score, 3),
            "confidence": self.confidence,
            "method": self.method,
            "cuesMoved": self.cues_moved,
            "cuesDropped": self.cues_dropped,
        }

    def describe(self) -> str:
        """One sentence a person can read: what was done and how sure it is."""
        engine = {
            "audio": "the video's speech",
            "subtitle": "the reference subtitle",
            "reference": "the reference release's audio",
            "transcript": "a transcript of the video",
        }.get(self.method, self.method)
        if not self.applied:
            return (
                f"No clear match between the subtitle and {engine} was found, so the timing was left "
                f"as it was (score {self.score:.2f})."
            )
        parts = [f"shifted by {_format_offset(self.offset_s)}"]
        if self.framerate:
            parts.append(f"converted {self.framerate} fps")
        elif abs(self.ratio - 1.0) > 1e-6:
            parts.append(f"scaled by {self.ratio:.6f}")
        if self.splits:
            steps = ", ".join(f"{_format_offset(s.offset_s)} from {_clock(s.at_s)}" for s in self.splits[:4])
            more = f" and {len(self.splits) - 4} more" if len(self.splits) > 4 else ""
            parts.append(f"with {len(self.splits)} scene change{'s' if len(self.splits) != 1 else ''} ({steps}{more})")
        sentence = (
            f"Matched against {engine} ({self.confidence} confidence, score {self.score:.2f}): "
            + ", ".join(parts)
        )
        if self.cues_dropped:
            lines = f"{self.cues_dropped} line{'s' if self.cues_dropped != 1 else ''}"
            sentence += f"; {lines} with no matching scene in this video {'were' if self.cues_dropped != 1 else 'was'} dropped"
        return sentence + "."


def _format_offset(seconds: float) -> str:
    return f"{seconds:+.3f} s"


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# ============================================================ frame rates


@dataclass(frozen=True)
class RateCandidate:
    ratio: float
    #: (subtitle's rate, video's rate) pairs this ratio converts between.
    pairs: Tuple[Tuple[Fraction, Fraction], ...] = ()

    def label(self, video_fps: Optional[float] = None) -> Optional[str]:
        if not self.pairs:
            return None
        chosen = self.pairs[0]
        if video_fps:
            for pair in self.pairs:
                if abs(float(pair[1]) - video_fps) < 0.01:
                    chosen = pair
                    break
        return f"{_fps(chosen[0])} → {_fps(chosen[1])}"


def _fps(rate: Fraction) -> str:
    value = float(rate)
    return f"{value:g}" if value == int(value) else f"{value:.3f}"


def rate_candidates() -> List[RateCandidate]:
    """Identity first, then every conversion between SYNC_RATES.

    A subtitle timed at ``a`` fps played against a video at ``b`` fps needs
    ``t * a / b``: timed for a 25 fps PAL release, whose picture runs 4%
    fast, it needs stretching by 25/23.976 for the 23.976 fps original.
    """
    found: Dict[Fraction, List[Tuple[Fraction, Fraction]]] = {}
    for a in SYNC_RATES:
        for b in SYNC_RATES:
            if a != b:
                found.setdefault(a / b, []).append((a, b))
    out = [RateCandidate(1.0)]
    for ratio, pairs in sorted(found.items(), key=lambda item: abs(float(item[0]) - 1.0)):
        out.append(RateCandidate(float(ratio), tuple(pairs)))
    return out


def framerate_label(ratio: float, video_fps: Optional[float] = None) -> Optional[str]:
    for candidate in rate_candidates()[1:]:
        if abs(candidate.ratio - ratio) < 1e-7:
            return candidate.label(video_fps)
    return None


# ============================================================ signals


def dialogue_mask(cues: Sequence[Cue]) -> np.ndarray:
    """Which cues are spoken dialogue (see _NON_DIALOGUE_STYLE)."""
    mask = np.ones(len(cues), dtype=bool)
    for i, cue in enumerate(cues):
        text = strip_tags(cue.text or "")
        if cue.duration <= 0 or cue.duration > MAX_DIALOGUE_CUE_S:
            mask[i] = False
        elif not text.strip() or _SDH_ONLY.match(text):
            mask[i] = False
        elif cue.style and _NON_DIALOGUE_STYLE.search(cue.style):
            mask[i] = False
    if mask.sum() < max(3, len(cues) // 4):
        # Heuristics that throw away most of a subtitle are wrong about it.
        mask = np.array([c.duration > 0 for c in cues], dtype=bool)
    return mask


def activity(spans: Sequence[Tuple[float, float]], length: Optional[int] = None, hop_s: float = HOP_S) -> np.ndarray:
    """1 where any span is on, on the hop grid (frame k covers k*hop)."""
    end = max((e for _, e in spans), default=0.0)
    n = length if length is not None else int(math.ceil(end / hop_s)) + 1
    out = np.zeros(max(0, n), dtype=np.float32)
    for start, stop in spans:
        a = max(0, int(round(start / hop_s)))
        b = min(n, int(round(stop / hop_s)))
        if b > a:
            out[a:b] = 1.0
    return out


def _starts(spans: Sequence[Tuple[float, float]], n: int, hop_s: float = HOP_S) -> np.ndarray:
    """A small bump at each span's start, for the fine placement."""
    out = np.zeros(max(0, n), dtype=np.float32)
    width = max(1, int(round(START_SPREAD_S / hop_s)))
    kernel = np.hanning(2 * width + 1).astype(np.float32)
    for start, _ in spans:
        k = int(round(start / hop_s))
        lo, hi = k - width, k + width + 1
        a, b = max(0, lo), min(n, hi)
        if b > a:
            out[a:b] = np.maximum(out[a:b], kernel[a - lo:b - lo])
    return out


def _onsets(signal: np.ndarray, hop_s: float = HOP_S) -> np.ndarray:
    """Where a 0-1 activity signal rises, smoothed like _starts."""
    rise = np.maximum(0.0, np.diff(signal.astype(np.float32), prepend=signal[:1]))
    width = max(1, int(round(START_SPREAD_S / hop_s)))
    kernel = np.hanning(2 * width + 1).astype(np.float32)
    return np.convolve(rise, kernel, mode="same").astype(np.float32)


def xcorr(weights: np.ndarray, signal: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """``out[j] = sum_t weights[t] * signal[t + lag]`` for lag = lo + j .. hi.

    Positions outside ``signal`` count as zero, so a shift that pushes cues
    off either end of the video neither gains nor loses there.
    """
    size = _fast_fft_size(len(weights) + len(signal) + max(abs(lo), abs(hi)) + 1)
    spectrum = np.fft.rfft(signal.astype(np.float64), size) * np.conj(np.fft.rfft(weights.astype(np.float64), size))
    full = np.fft.irfft(spectrum, size)
    return full[np.arange(lo, hi + 1) % size]


def peak_z(curve: np.ndarray, index: int, hop_s: float = HOP_S) -> float:
    """How far a peak stands above every other offset, in robust units."""
    shoulder = int(round(PEAK_SHOULDER_S / hop_s))
    mask = np.ones(len(curve), dtype=bool)
    mask[max(0, index - shoulder):index + shoulder + 1] = False
    rest = curve[mask]
    if len(rest) < 10:
        return 0.0
    median = float(np.median(rest))
    spread = 1.4826 * float(np.median(np.abs(rest - median)))
    if spread <= 1e-12:
        return 0.0
    return (float(curve[index]) - median) / spread


def _parabolic(curve: np.ndarray, index: int) -> float:
    if 0 < index < len(curve) - 1:
        left, mid, right = curve[index - 1], curve[index], curve[index + 1]
        denom = left - 2.0 * mid + right
        if denom < 0:
            shift = 0.5 * (left - right) / denom
            if abs(shift) <= 1.0:
                return index + float(shift)
    return float(index)


def sharpen(curve: np.ndarray, hop_s: float = HOP_S) -> np.ndarray:
    """The curve without its slow part (see SHARPEN_S)."""
    width = max(3, int(round(SHARPEN_S / hop_s)) | 1)
    if len(curve) <= width:
        return curve - float(np.mean(curve)) if len(curve) else curve
    kernel = np.ones(width) / width
    padded = np.pad(curve.astype(np.float64), width // 2, mode="edge")
    return curve - np.convolve(padded, kernel, mode="valid")


# ============================================================ the aligner


@dataclass
class Alignment:
    """A mapping from subtitle time to video time, per cue."""

    ratio: float
    #: Offset per dialogue span (in the order given to ``align``), seconds.
    offsets: np.ndarray
    #: (first cue, end cue, offset) per constant stretch, cue indexes into
    #: the dialogue cues sorted by start.
    segments: List[Tuple[int, int, float]]
    peak_z: float
    rate: RateCandidate = field(default_factory=lambda: RateCandidate(1.0))
    #: Evidence for one (scaled) span at an offset, as _Aligner.cue_gain.
    gain: Callable[[float, float, float], float] = field(default=lambda start, end, offset: 0.0, repr=False)


class _Aligner:
    """Places a list of cues on a 0-1 evidence signal: VAD probability, or
    another subtitle's activity."""

    def __init__(
        self,
        spans: Sequence[Tuple[float, float]],
        signal: np.ndarray,
        max_offset_s: float,
        check: Callable[[], None] = lambda: None,
        progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.spans = [(float(a), float(b)) for a, b in spans]
        self.signal = np.asarray(signal, dtype=np.float32)
        self.mean = float(self.signal.mean()) if len(self.signal) else 0.0
        # Evidence relative to chance: a cue over this much speech gains, one
        # over less loses, so moving cues into a talkative scene is not free.
        self.centred = (self.signal - self.mean).astype(np.float32)
        self.onsets = _onsets(self.signal)
        self.onsets -= float(self.onsets.mean()) if len(self.onsets) else 0.0
        self.max_lag = max(1, int(round(max_offset_s / HOP_S)))
        self.check = check
        self.progress = progress or (lambda percent: None)
        cum = np.concatenate([[0.0], np.cumsum(self.centred, dtype=np.float64)])
        self._cum = cum

    # -- evidence ----------------------------------------------------------

    def cue_gain(self, spans: Sequence[Tuple[float, float]], offset_s: float) -> np.ndarray:
        """Seconds of excess evidence under each span shifted by ``offset_s``."""
        n = len(self.centred)
        starts = np.array([s for s, _ in spans]) + offset_s
        ends = np.array([e for _, e in spans]) + offset_s
        a = np.clip(np.round(starts / HOP_S).astype(np.int64), 0, n)
        b = np.clip(np.round(ends / HOP_S).astype(np.int64), 0, n)
        return (self._cum[b] - self._cum[a]) * HOP_S

    def curve(self, spans: Sequence[Tuple[float, float]], lo: int, hi: int) -> np.ndarray:
        """Evidence under ``spans`` at every lag lo..hi (frames)."""
        if not spans:
            return np.zeros(hi - lo + 1)
        first = min(s for s, _ in spans)
        base = int(math.floor(first / HOP_S))
        local = [(s - base * HOP_S, e - base * HOP_S) for s, e in spans]
        weights = activity(local)
        # Only the part of the signal these lags can reach.
        s_lo = max(0, base + lo)
        s_hi = min(len(self.centred), base + len(weights) + hi + 1)
        if s_hi <= s_lo:
            return np.zeros(hi - lo + 1)
        piece = self.centred[s_lo:s_hi]
        shift = s_lo - base  # signal index 0 of piece is lag `shift` for weight 0
        return xcorr(weights, piece, lo - shift, hi - shift)

    # -- global ------------------------------------------------------------

    def best_rate(self, detect: bool) -> Tuple[RateCandidate, float, float]:
        """The ratio whose sharpened correlation peaks highest, with its
        offset and z."""
        results = []
        candidates = rate_candidates() if detect else [RateCandidate(1.0)]
        for index, rate in enumerate(candidates):
            self.check()
            scaled = [(s * rate.ratio, e * rate.ratio) for s, e in self.spans]
            curve = sharpen(self.curve(scaled, -self.max_lag, self.max_lag))
            total = sum(e - s for s, e in scaled) or 1.0
            peak = int(np.argmax(curve))
            results.append((rate, curve, peak, float(curve[peak]) / total))
            self.progress(int(100 * (index + 1) / len(candidates)))
        identity = results[0]
        chosen = identity
        best = max(results, key=lambda r: r[3])
        if best is not identity and best[3] > max(0.0, identity[3]) * RATIO_MARGIN:
            if peak_z(best[1], best[2]) >= REJECT_Z:
                chosen = best
        rate, curve, peak, _ = chosen
        offset = (_parabolic(curve, peak) - self.max_lag) * HOP_S
        return rate, offset, peak_z(curve, peak)

    # -- splits ------------------------------------------------------------

    def candidates(self, spans: Sequence[Tuple[float, float]], global_offset: float) -> List[float]:
        """Offsets that some five minutes of the subtitle clearly prefer."""
        found: List[Tuple[float, float]] = [(global_offset, float("inf"))]
        if not spans:
            return [global_offset]
        first, last = spans[0][0], max(e for _, e in spans)
        starts = np.array([s for s, _ in spans])
        apart = int(round(1.0 / HOP_S))
        position = first
        while True:
            self.check()
            lo, hi = np.searchsorted(starts, position), np.searchsorted(starts, position + WINDOW_S)
            inside = list(spans[lo:hi])
            if len(inside) >= WINDOW_MIN_CUES:
                curve = sharpen(self.curve(inside, -self.max_lag, self.max_lag))
                masked = curve.copy()
                for _ in range(WINDOW_PEAKS):
                    peak = int(np.argmax(masked))
                    z = peak_z(curve, peak)
                    if z < WINDOW_Z:
                        break
                    found.append(((_parabolic(curve, peak) - self.max_lag) * HOP_S, z))
                    masked[max(0, peak - apart):peak + apart] = -np.inf
            if position + WINDOW_S >= last:
                break
            position += WINDOW_HOP_S
        # Cluster: strongest first, anything within SAME_OFFSET_S joins it.
        found.sort(key=lambda item: -item[1])
        merged: List[float] = []
        for offset, _ in found:
            if all(abs(offset - m) > SAME_OFFSET_S for m in merged):
                merged.append(offset)
            if len(merged) >= MAX_CANDIDATES:
                break
        return merged

    def stands_alone(self, spans: Sequence[Tuple[float, float]], offset: float) -> bool:
        """Whether these lines on their own peak at ``offset`` (SEGMENT_Z)."""
        curve = sharpen(self.curve(spans, -self.max_lag, self.max_lag))
        peak = int(np.argmax(curve))
        if abs((peak - self.max_lag) * HOP_S - offset) > SEGMENT_AGREE_S:
            return False
        return peak_z(curve, peak) >= SEGMENT_Z

    def mapping_z(self, spans: Sequence[Tuple[float, float]], segments: Sequence[Tuple[int, int, float]]) -> float:
        """How sharply the whole mapping peaks: every stretch's curve around
        its own offset, summed, so a subtitle split into several stretches
        is judged on all its lines at once, like an unsplit one."""
        total: Optional[np.ndarray] = None
        for a, b, o in segments:
            centre = int(round(o / HOP_S))
            curve = self.curve(spans[a:b], centre - self.max_lag, centre + self.max_lag)
            total = curve if total is None else total + curve
        if total is None:
            return 0.0
        total = sharpen(total)
        reach = int(round(SEGMENT_AGREE_S / HOP_S))
        lo = self.max_lag - reach
        peak = lo + int(np.argmax(total[lo:self.max_lag + reach + 1]))
        return peak_z(total, peak)

    def path(self, spans: Sequence[Tuple[float, float]], offsets: Sequence[float], penalty: float) -> List[int]:
        """The offset per cue that maximises evidence minus split penalties.

        A Viterbi pass over the cues in order: staying at an offset is free,
        moving to another costs ``penalty``. The penalty is what makes a
        split need several lines agreeing after it, rather than one line
        that happens to fit elsewhere.
        """
        gains = np.stack([self.cue_gain(spans, o) for o in offsets], axis=1)
        n, k = gains.shape
        score = gains[0].copy()
        back = np.zeros((n, k), dtype=np.int32)
        for i in range(1, n):
            best = int(np.argmax(score))
            switch = score[best] - penalty
            stay = score >= switch
            back[i] = np.where(stay, np.arange(k), best)
            score = np.where(stay, score, switch) + gains[i]
        state = int(np.argmax(score))
        out = [state] * n
        for i in range(n - 1, 0, -1):
            state = int(back[i, state])
            out[i - 1] = state
        return out

    def refine(self, spans: Sequence[Tuple[float, float]], offset: float) -> float:
        """The offset of one stretch to the hop and below.

        Evidence is the cue activity plus a bump at each cue start against
        the signal's rises (see START_WEIGHT).
        """
        if not spans:
            return offset
        centre = int(round(offset / HOP_S))
        reach = int(round(REFINE_S / HOP_S))
        lo, hi = centre - reach, centre + reach
        body = self.curve(spans, lo, hi)
        first = min(s for s, _ in spans)
        base = int(math.floor(first / HOP_S))
        local = [(s - base * HOP_S, e - base * HOP_S) for s, e in spans]
        n = int(math.ceil(max(e for _, e in local) / HOP_S)) + 1
        bumps = _starts(local, n)
        s_lo = max(0, base + lo)
        s_hi = min(len(self.onsets), base + n + hi + 1)
        if s_hi > s_lo:
            edge = xcorr(bumps, self.onsets[s_lo:s_hi], lo - (s_lo - base), hi - (s_lo - base))
        else:
            edge = np.zeros_like(body)
        # Put both on the scale of their own spread before adding them.
        total = _unit(body) + START_WEIGHT * _unit(edge)
        peak = int(np.argmax(total))
        return (lo + _parabolic(total, peak)) * HOP_S


def _unit(curve: np.ndarray) -> np.ndarray:
    spread = float(np.std(curve))
    return (curve - float(np.mean(curve))) / spread if spread > 1e-12 else curve * 0.0


def align(
    spans: Sequence[Tuple[float, float]],
    signal: np.ndarray,
    options: Dict[str, Any],
    check: Callable[[], None] = lambda: None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Alignment:
    """Map dialogue spans (sorted by start) onto an evidence signal."""
    report = progress or (lambda percent, stage: None)
    aligner = _Aligner(
        spans, signal, float(options.get("maxOffsetS") or 60.0), check,
        lambda percent: report(percent, "Matching timing"),
    )
    rate, offset, z = aligner.best_rate(bool(options.get("detectFramerate", True)))
    scaled = [(s * rate.ratio, e * rate.ratio) for s, e in aligner.spans]
    segments: List[Tuple[int, int, float]] = [(0, len(scaled), offset)]
    # Splits are looked for only once the subtitle as a whole has clearly
    # been found: stretches of a subtitle that matches nowhere would each
    # find their own coincidence.
    if options.get("allowSplits", True) and len(scaled) >= 2 * MIN_SEGMENT_CUES and z >= SEGMENT_Z:
        report(0, "Looking for scene changes")
        offsets = aligner.candidates(scaled, offset)
        if len(offsets) > 1:
            penalty = float(options.get("splitPenalty", 7.0)) * PENALTY_UNIT
            states = aligner.path(scaled, offsets, penalty)
            segments = _segments(states, offsets)
            segments = _absorb_short(segments, scaled, aligner)
            segments = _verify(segments, scaled, aligner)
        report(100, "Looking for scene changes")
    # Place each stretch exactly; stretches that land on the same offset are
    # one stretch after all.
    placed = [(a, b, aligner.refine(scaled[a:b], o)) for a, b, o in segments]
    placed = _merge_same(placed, scaled, aligner)
    per_cue = np.zeros(len(scaled))
    for a, b, o in placed:
        per_cue[a:b] = o
    if len(placed) > 1:
        z = max(z, aligner.mapping_z(scaled, placed))
    return Alignment(
        rate.ratio, per_cue, placed, z, rate,
        gain=lambda start, end, o: float(aligner.cue_gain([(start, end)], o)[0]),
    )


def _verify(segments, spans, aligner: _Aligner):
    """Fold every stretch that does not stand on its own into the neighbour
    its lines fit better, weakest first, until all that remain do."""
    segments = list(segments)
    while len(segments) > 1:
        weak = [i for i, (a, b, o) in enumerate(segments) if not aligner.stands_alone(spans[a:b], o)]
        if not weak:
            break
        index = min(weak, key=lambda i: segments[i][1] - segments[i][0])
        a, b, _ = segments[index]
        neighbours = [j for j in (index - 1, index + 1) if 0 <= j < len(segments)]
        best = max(neighbours, key=lambda j: float(aligner.cue_gain(spans[a:b], segments[j][2]).sum()))
        na, nb, no = segments[best]
        segments[best] = (min(a, na), max(b, nb), no)
        del segments[index]
        # Neighbours that now hold the same offset are one stretch.
        joined: List[Tuple[int, int, float]] = []
        for seg in segments:
            if joined and abs(joined[-1][2] - seg[2]) <= SAME_OFFSET_S:
                joined[-1] = (joined[-1][0], seg[1], joined[-1][2])
            else:
                joined.append(seg)
        segments = joined
    return segments


def _segments(states: List[int], offsets: List[float]) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append((start, i, offsets[states[start]]))
            start = i
    return out


def _absorb_short(segments, spans, aligner: _Aligner):
    """Give a stretch too short to trust to whichever neighbour fits it better."""
    segments = list(segments)
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for index, (a, b, o) in enumerate(segments):
            if b - a >= MIN_SEGMENT_CUES:
                continue
            options = []
            if index > 0:
                options.append(index - 1)
            if index < len(segments) - 1:
                options.append(index + 1)
            best = max(options, key=lambda j: float(aligner.cue_gain(spans[a:b], segments[j][2]).sum()))
            na, nb, no = segments[best]
            segments[best] = (min(a, na), max(b, nb), no)
            del segments[index]
            changed = True
            break
    return segments


def _merge_same(placed, spans, aligner: _Aligner):
    merged: List[Tuple[int, int, float]] = []
    for a, b, o in placed:
        if merged and abs(merged[-1][2] - o) <= SAME_OFFSET_S:
            ma, _, _ = merged[-1]
            merged[-1] = (ma, b, aligner.refine(spans[ma:b], o))
        else:
            merged.append((a, b, o))
    return merged


# ============================================================ applying


def _apply(
    doc: SubtitleDoc,
    order: List[int],
    dialogue: np.ndarray,
    alignment: Alignment,
) -> Tuple[SubtitleDoc, List[Split], int]:
    """Carry every cue across the alignment.

    ``order`` lists the dialogue cues' indexes in ``doc.cues`` sorted by
    start, matching ``alignment.offsets``. Other cues (signs, songs) take the
    offset of the nearest dialogue line.
    """
    ratio = alignment.ratio
    cues = doc.cues
    per_rank = np.array(alignment.offsets, dtype=np.float64)
    dropped: set = set()
    splits: List[Split] = []
    for (a, b, o), (na, nb, no) in zip(alignment.segments, alignment.segments[1:]):
        gone = _settle_cut(cues, order, per_rank, (a, b, o), (na, nb, no), ratio, alignment.gain, dropped)
        kept_a = [r for r in range(a, b) if order[r] not in dropped and per_rank[r] == o] or [b - 1]
        kept_b = [r for r in range(na, nb) if order[r] not in dropped and per_rank[r] == no] or [na]
        splits.append(Split(
            at_s=_split_time(
                max(cues[order[r]].end for r in kept_a) * ratio, min(cues[order[r]].start for r in kept_b) * ratio,
                [(cues[i].start * ratio, cues[i].end * ratio) for i in gone], o, no,
            ),
            offset_s=no,
        ))

    offsets = np.zeros(len(cues))
    for rank, index in enumerate(order):
        offsets[index] = per_rank[rank]
    dialogue_starts = np.array([cues[i].start for i in order])
    for index, cue in enumerate(cues):
        if not dialogue[index] and len(order):
            offsets[index] = per_rank[int(np.argmin(np.abs(dialogue_starts - cue.start)))]
    out: List[Cue] = []
    for index, cue in enumerate(cues):
        if index in dropped:
            continue
        start, end = cue.start * ratio + offsets[index], cue.end * ratio + offsets[index]
        if end <= 0:
            dropped.add(index)
            continue
        out.append(cue.copy(start=max(0.0, start), end=end))
    out.sort(key=lambda c: (c.start, c.end))
    return doc.copy(out), splits, len(dropped)


def _settle_cut(cues, order, per_rank, before, after, ratio, gain, dropped: set) -> List[int]:
    """Decide exactly which lines fall on either side of a cut, and which
    belong to material the video lacks.

    The path chooses offsets line by line and knows nothing of order, so
    where the video lacks a stretch it can leave lines on the wrong side --
    in a scene that is all speech either offset fits -- and it has no way
    to say "this line is not in the video". An edit has a simpler shape:
    the video lacks the subtitle's ``[m, m + length)``, lines before ``m``
    keep the old offset, lines after the gap take the new one, and lines
    inside it go. So near the path's switch every possible ``m`` is tried
    and the one whose lines land on the most speech wins; a line dropped
    earns nothing, so lines are dropped only where no ``m`` keeps them in
    order. Returns the cue indexes dropped. Where the video has material
    the subtitle lacks, the path's choice is already of that shape.
    """
    (a, b, o), (na, nb, no) = before, after
    length = o - no
    if length <= 0:
        return []
    lo, hi = max(a, b - SETTLE_REACH), min(nb, na + SETTLE_REACH)
    ranks = list(range(lo, hi))
    starts = np.array([cues[order[r]].start * ratio for r in ranks])
    ends = np.array([cues[order[r]].end * ratio for r in ranks])
    gain_before = np.array([gain(s, e, o) for s, e in zip(starts, ends)])
    gain_after = np.array([gain(s, e, no) for s, e in zip(starts, ends)])
    choices = np.unique(np.concatenate([ends, starts - length]))
    best = None
    for m in choices:
        keep_before = ends <= m + OVERLAP_S
        keep_after = starts >= m + length - OVERLAP_S
        # A line cannot be on both sides.
        keep_after &= ~keep_before
        total = float(gain_before[keep_before].sum() + gain_after[keep_after].sum())
        lost = int((~keep_before & ~keep_after).sum())
        key = (round(total, 9), -lost, -m)
        if best is None or key > best[0]:
            best = (key, keep_before, keep_after)
    _, keep_before, keep_after = best
    gone = []
    for rank, kb, ka in zip(ranks, keep_before, keep_after):
        if kb:
            per_rank[rank] = o
        elif ka:
            per_rank[rank] = no
        else:
            dropped.add(order[rank])
            gone.append(order[rank])
    return gone


# How many lines either side of the path's switch _settle_cut reconsiders.
SETTLE_REACH = 30


def _split_time(
    last_end: float,
    first_start: float,
    missing: Sequence[Tuple[float, float]],
    before: float,
    after: float,
) -> float:
    """Where on the video the offset changes.

    Arguments are on the subtitle's (scaled) timeline: the end of the last
    line kept before the change, the start of the first after it, and the
    lines dropped as missing material. The change itself can sit anywhere
    between those two lines -- nothing in a subtitle says where in a pause
    a scene was cut -- so the middle of what is possible is reported. When
    the video lacks material, the dropped lines narrow that down: the
    missing stretch, ``before - after`` long, has to hold them all.
    """
    jump = after - before
    lo, hi = last_end, first_start
    if jump < 0:
        length = -jump
        hi = first_start - length
        if missing:
            lo = max(lo, max(e for _, e in missing) - length)
            hi = min(hi, min(s for s, _ in missing))
    if hi < lo:
        lo = hi = 0.5 * (lo + hi)
    return 0.5 * (lo + hi) + before


def _moved(before: SubtitleDoc, after: SubtitleDoc) -> int:
    old = sorted((c.start for c in before.cues))
    new = sorted((c.start for c in after.cues))
    if len(old) != len(new):
        return len(new)
    return int(sum(1 for a, b in zip(old, new) if abs(a - b) > 0.0005))


def _fit_score(spans_video: Sequence[Tuple[float, float]], signal: np.ndarray) -> float:
    """How much of the cue time lands on the evidence, above chance: 0 is
    chance, 1 is every cue frame on certain speech."""
    if not spans_video or len(signal) == 0:
        return 0.0
    mean = float(signal.mean())
    n = len(signal)
    inside = 0.0
    total = 0.0
    for start, end in spans_video:
        a = int(np.clip(round(start / HOP_S), 0, n))
        b = int(np.clip(round(end / HOP_S), 0, n))
        if b > a:
            inside += float(signal[a:b].sum())
            total += b - a
    if total == 0 or mean >= 1.0:
        return 0.0
    return float(np.clip((inside / total - mean) / (1.0 - mean), 0.0, 1.0))


def _confidence(z: float, fit: float) -> str:
    if z >= HIGH_Z and fit >= GOOD_FIT / 2.0:
        return "high"
    if z >= REJECT_Z and fit >= 0.1:
        return "medium"
    return "low"


# Fit of a correct subtitle against its film's own speech. Measured on two
# real films: 0.42 and 0.45 (cues run on after the line, and the detector
# misses some speech under loud score); another film's subtitle at its best
# offset: 0.04-0.05.
GOOD_FIT = 0.4


def _score(z: float, fit: float) -> float:
    """One 0-1 number: how clearly the match stood out, times how much of
    the subtitle landed on speech. Either alone can flatter: a sharp peak
    from a handful of lines, or cues spread over a film that talks all the
    time."""
    clarity = float(np.clip((z - 3.0) / (HIGH_Z - 3.0), 0.0, 1.0))
    return float(clarity * min(1.0, fit / GOOD_FIT))


def sync_to_signal(
    doc: SubtitleDoc,
    signal: np.ndarray,
    options: Dict[str, Any],
    method: str,
    check: Callable[[], None] = lambda: None,
    progress: Optional[ProgressFn] = None,
    video_fps: Optional[float] = None,
) -> SyncOutcome:
    """The audio and subtitle engines: align cues to a 0-1 evidence signal."""
    cues = doc.cues
    dialogue = dialogue_mask(cues)
    order = sorted((i for i in range(len(cues)) if dialogue[i]), key=lambda i: (cues[i].start, cues[i].end))
    if len(order) < MIN_SEGMENT_CUES:
        return SyncOutcome(doc.copy(), 0.0, method=method, applied=False,
                           warnings=["The subtitle has too few lines of dialogue to sync."])
    spans = [(cues[i].start, cues[i].end) for i in order]
    alignment = align(spans, signal, options, check, progress)
    out, splits, dropped = _apply(doc, order, dialogue, alignment)

    fit = _fit_score([(c.start, c.end) for c, keep in zip(out.cues, dialogue_mask(out.cues)) if keep], signal)
    confidence = _confidence(alignment.peak_z, fit)
    outcome = SyncOutcome(
        doc=out,
        offset_s=float(alignment.segments[0][2]) if alignment.segments else 0.0,
        ratio=alignment.ratio,
        splits=splits,
        score=_score(alignment.peak_z, fit),
        method=method,
        confidence=confidence,
        framerate=alignment.rate.label(video_fps),
        cues_dropped=dropped,
        peak_z=alignment.peak_z,
    )
    if alignment.peak_z < REJECT_Z:
        # Nothing stands out: moving the cues would be a guess, and a guess
        # can wreck a subtitle that was only slightly off. Say so instead.
        outcome.doc = doc.copy()
        outcome.applied = False
        outcome.offset_s, outcome.ratio, outcome.splits, outcome.framerate = 0.0, 1.0, [], None
        outcome.cues_dropped = 0
        outcome.confidence = "low"
        outcome.warnings.append(
            "No clear match was found: this subtitle may belong to a different cut or episode, "
            "or the video has little clear speech."
        )
    outcome.cues_moved = _moved(doc, outcome.doc) if outcome.applied else 0
    return outcome


# ============================================================ engines


def _audio(doc, options, video, ctx) -> SyncOutcome:
    if not video or not video.get("path"):
        from .tasks import TaskError

        raise TaskError("Syncing to audio needs the video.")
    path = video["path"]
    track = int(video.get("audioTrack") or 0)
    engine = options.get("vad") or "energy"
    probs, _ = vad.speech_activity(
        path, track, engine, ctx.token,
        lambda percent: ctx.progress(percent, "Finding speech"),
    )
    ctx.check()
    return sync_to_signal(doc, probs, options, "audio", ctx.check, ctx.progress, _video_fps(path, ctx.token))


def _subtitle(doc, options, reference, ctx) -> SyncOutcome:
    from .tasks import TaskError, load_subtitle

    if not reference or not reference.get("path"):
        raise TaskError("Syncing to a subtitle needs the reference subtitle.")
    ctx.progress(0, "Reading the reference")
    ref_doc = load_subtitle(reference, ctx)
    ref_cues = sorted(ref_doc.cues, key=lambda c: c.start)
    mask = dialogue_mask(ref_cues)
    signal = activity([(c.start, c.end) for c, keep in zip(ref_cues, mask) if keep])
    outcome = sync_to_signal(doc, signal, options, "subtitle", ctx.check, ctx.progress)
    if outcome.applied:
        outcome.doc, snapped = snap_to_reference(outcome.doc, ref_cues)
        if snapped:
            ctx.log(f"{snapped} lines took the reference's exact timing")
        outcome.cues_moved = _moved(doc, outcome.doc)
    return outcome


def snap_to_reference(doc: SubtitleDoc, reference: Sequence[Cue]) -> Tuple[SubtitleDoc, int]:
    """Give each cue that clearly pairs with one reference cue its timing.

    A pair is clear when each is the other's nearest by start, the starts
    are within SNAP_START_S, and the durations within SNAP_DURATION_RATIO:
    the same line, split the same way, in another language (short lines
    may differ by SNAP_DURATION_S whatever the ratio). Lines split
    differently keep the aligned timing.
    """
    if not reference or not doc.cues:
        return doc.copy(), 0
    ref_starts = np.array([c.start for c in reference])
    cues = [c.copy() for c in doc.cues]
    starts = np.array([c.start for c in cues])
    snapped = 0
    for i, cue in enumerate(cues):
        j = int(np.argmin(np.abs(ref_starts - cue.start)))
        ref = reference[j]
        if abs(ref.start - cue.start) > SNAP_START_S:
            continue
        back = int(np.argmin(np.abs(starts - ref.start)))
        if back != i:
            continue
        a, b = max(cue.duration, 1e-3), max(ref.duration, 1e-3)
        if max(a, b) / min(a, b) > SNAP_DURATION_RATIO and abs(a - b) > SNAP_DURATION_S:
            continue
        cue.start, cue.end = ref.start, ref.end
        snapped += 1
    return doc.copy(sorted(cues, key=lambda c: (c.start, c.end))), snapped


def _video_fps(path: str, token: Optional[CancellationToken]) -> Optional[float]:
    try:
        return probe(path, token).fps
    except MediaError:
        return None


# ------------------------------------------------------------ reference audio


@dataclass
class TimeMap:
    """Reference-release time -> video time, piece by piece.

    Each piece covers ``[ref_lo, ref_hi)`` of the reference and maps it by
    ``video = at + scale * ref``. Reference time outside every piece is
    material this video lacks.
    """

    pieces: List[Tuple[float, float, float, float]]  # ref_lo, ref_hi, at, scale

    def __call__(self, t: float) -> Optional[float]:
        for ref_lo, ref_hi, at, scale in self.pieces:
            if ref_lo <= t < ref_hi:
                return at + scale * t
        return None

    def nearest(self, t: float) -> float:
        """Where ``t`` lands, pulled onto the nearest piece if it has none:
        a line that starts just inside a missing scene starts where the
        video picks up again."""
        best: Optional[Tuple[float, float]] = None
        for ref_lo, ref_hi, at, scale in self.pieces:
            clamped = min(max(t, ref_lo), ref_hi)
            distance = abs(clamped - t)
            if best is None or distance < best[0]:
                best = (distance, at + scale * clamped)
        return best[1] if best else t


def map_from_pair(delay_at_start_ms: float, drift_ms_per_s: Optional[float], speed: float = 1.0) -> TimeMap:
    """The mapping analyze_pair measured, turned around.

    analyze_pair reports how late the secondary (the reference) runs against
    the primary (the video): reference analysis time = v + d(v), with d
    growing linearly by the drift, and the reference decoded ``speed`` times
    faster than it plays. So reference time T sits at
    ``v = (T * speed - d0) / (1 + k)`` on the video. A reference with two
    extra seconds at its head measures +2000 ms, and its cues move 2 s
    earlier.
    """
    d0 = delay_at_start_ms / 1000.0
    k = (drift_ms_per_s or 0.0) / 1000.0
    return TimeMap([(-math.inf, math.inf, -d0 / (1.0 + k), speed / (1.0 + k))])


def map_from_plan(plan) -> TimeMap:
    """The dub-sync plan's pieces of reference, where each sits on the video.

    A ``dub`` segment plays the reference from ``source_start_s`` on its
    analysis timeline -- reference time times ``plan.speed`` -- at
    ``start_s`` on the video.
    """
    pieces = []
    for segment in plan.dub_segments:
        ref_lo = segment.source_start_s / plan.speed
        ref_hi = (segment.source_start_s + segment.length_s) / plan.speed
        pieces.append((ref_lo, ref_hi, segment.start_s - segment.source_start_s, plan.speed))
    pieces.sort()
    return TimeMap(pieces)


def carry(doc: SubtitleDoc, mapping: TimeMap) -> Tuple[SubtitleDoc, int]:
    """Every cue through a TimeMap; cues whose middle has no place on the
    video are dropped (counted)."""
    out: List[Cue] = []
    dropped = 0
    for cue in doc.cues:
        middle = mapping(0.5 * (cue.start + cue.end))
        if middle is None:
            dropped += 1
            continue
        start = mapping(cue.start)
        end = mapping(cue.end)
        start = mapping.nearest(cue.start) if start is None else start
        end = mapping.nearest(cue.end) if end is None else end
        if end <= 0:
            dropped += 1
            continue
        if end < start:
            start, end = end, start
        out.append(cue.copy(start=max(0.0, start), end=end))
    out.sort(key=lambda c: (c.start, c.end))
    return doc.copy(out), dropped


def _reference(doc, options, video, reference, ctx) -> SyncOutcome:
    from ..analyze import analyze_pair
    from .tasks import TaskError

    if not video or not video.get("path"):
        raise TaskError("Syncing to a reference release needs the video.")
    if not reference or not reference.get("path"):
        raise TaskError("Syncing to a reference release needs that release's audio or video.")
    video_track = int(video.get("audioTrack") or 0)
    ref_track = int(reference.get("audioTrack") or 0)
    max_offset_ms = float(options.get("maxOffsetS") or 60.0) * 1000.0
    ctx.progress(0, "Measuring the two releases")
    pair = analyze_pair(
        video["path"], reference["path"], max_offset_ms=max_offset_ms, token=ctx.token,
        progress=lambda percent: ctx.progress(percent, "Measuring the two releases"),
        primary_track=video_track, secondary_track=ref_track,
    )
    ctx.check()
    usable = pair.error is None and pair.delay_at_start_ms is not None
    warnings: List[str] = []
    if usable and not pair.is_likely_cut:
        speed = pair.speed_compensation or 1.0
        mapping = map_from_pair(pair.delay_at_start_ms, pair.drift_ms_per_s, speed)
        out, dropped = carry(doc, mapping)
        k = (pair.drift_ms_per_s or 0.0) / 1000.0
        ratio = speed / (1.0 + k)
        ratio = _snap_ratio(ratio)
        outcome = SyncOutcome(
            doc=out, offset_s=mapping.pieces[0][2], ratio=ratio, method="reference",
            score=float(pair.confidence), confidence=_pair_confidence(pair.confidence),
            framerate=framerate_label(ratio, pair.primary_fps), cues_dropped=dropped,
        )
    elif options.get("allowSplits", True):
        from ..dubsync import plan_dubsync

        ctx.progress(0, "Mapping scene by scene")
        plan = plan_dubsync(
            video["path"], reference["path"], video_track=video_track, dub_track=ref_track,
            search_s=max(float(options.get("maxOffsetS") or 60.0), 30.0), token=ctx.token,
            progress=lambda percent, stage="": ctx.progress(int(percent), "Mapping scene by scene"),
            log=ctx.log,
        )
        if plan.error or not plan.dub_segments:
            raise TaskError(plan.error or "The reference audio did not match this video anywhere.")
        mapping = map_from_plan(plan)
        out, dropped = carry(doc, mapping)
        first = plan.dub_segments[0]
        pieces = plan.dub_segments
        # A step in where the reference sits is a scene change; the offset
        # reported is where the subtitle's time zero would land, as for the
        # other engines.
        splits = [
            Split(at_s=seg.start_s, offset_s=seg.start_s - seg.source_start_s)
            for prev, seg in zip(pieces, pieces[1:])
            if abs((seg.start_s - seg.source_start_s) - (prev.start_s - prev.source_start_s)) > 0.01
        ]
        matches = [s.match for s in pieces if s.match is not None]
        score = float(np.clip(np.mean(matches) * 2.0, 0.0, 1.0)) if matches else 0.5
        warnings.extend(plan.warnings)
        outcome = SyncOutcome(
            doc=out, offset_s=first.start_s - first.source_start_s, ratio=_snap_ratio(plan.speed),
            splits=splits, method="reference", score=score,
            confidence="high" if score >= 0.6 else "medium" if score >= 0.3 else "low",
            framerate=framerate_label(_snap_ratio(plan.speed), plan.video_fps), cues_dropped=dropped,
        )
    else:
        if not usable:
            raise TaskError(pair.error or "The reference audio did not match this video.")
        mapping = map_from_pair(pair.delay_at_start_ms, pair.drift_ms_per_s, pair.speed_compensation or 1.0)
        out, dropped = carry(doc, mapping)
        warnings.append("The two releases are cut differently; allow scene changes to follow the cut.")
        outcome = SyncOutcome(
            doc=out, offset_s=mapping.pieces[0][2], method="reference", score=float(pair.confidence) * 0.5,
            confidence="low", cues_dropped=dropped,
        )
    outcome.warnings.extend(warnings)
    outcome.cues_moved = _moved(doc, outcome.doc)
    return outcome


def _snap_ratio(ratio: float) -> float:
    """A measured ratio within measurement noise of a standard conversion is
    that conversion, exactly."""
    for candidate in rate_candidates():
        if abs(candidate.ratio - ratio) < 2e-4:
            return candidate.ratio
    return ratio


def _pair_confidence(confidence: float) -> str:
    return "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"


# ------------------------------------------------------------ transcript

_CJK_RE = re.compile(
    "[ᄀ-ᇿ぀-ヿ㄰-㆏ㇰ-ㇿ㐀-䶿一-鿿"
    "가-힯豈-﫿ｦ-ﾟ]"
)
_ASS_TAG = re.compile(r"\{[^}]*\}")


def normalize_tokens(text: str) -> List[str]:
    """Words to compare: casefolded, accents and punctuation gone; CJK text
    (no spaces, and recognisers split it differently) one character each."""
    text = _ASS_TAG.sub(" ", strip_tags(text or "")).replace("\\N", " ").replace("\\n", " ")
    text = unicodedata.normalize("NFKD", text.casefold())
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if category.startswith("M"):
            continue
        kept.append(" " if category[0] in "PSZC" else ch)
    tokens: List[str] = []
    for word in "".join(kept).split():
        if _CJK_RE.search(word):
            tokens.extend(ch for ch in unicodedata.normalize("NFKC", word) if not ch.isspace())
        else:
            tokens.append(word)
    return tokens


def _word_tokens(words: Sequence[Word]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Tokens of the recognised words with their times; a word that splits
    into several tokens (CJK) shares its span out evenly."""
    tokens: List[str] = []
    starts: List[float] = []
    ends: List[float] = []
    for word in words:
        parts = normalize_tokens(word.text)
        if not parts:
            continue
        step = (word.end - word.start) / len(parts)
        for n, part in enumerate(parts):
            tokens.append(part)
            starts.append(word.start + n * step)
            ends.append(word.start + (n + 1) * step)
    return tokens, np.array(starts), np.array(ends)


# Alignment scores. Subtitles condense speech, and recognisers insert and
# drop words, so both kinds of gap are cheap; a word the recogniser heard but
# the subtitle left out is the cheapest of all.
MATCH = 1.0
MISMATCH = -0.8
SKIP_CUE_TOKEN = -0.6
SKIP_WORD = -0.3


def align_tokens(a: Sequence[str], b: Sequence[str]) -> List[Tuple[int, int]]:
    """Semi-global alignment: every token of ``a`` is placed against ``b``,
    and ``b`` may run on either side for free. Returns matched (i, j) pairs
    of equal tokens.

    Row by row in numpy: within a row, a left move depends on the cell before
    it, which is a running maximum of ``cell - j * gap`` -- one
    ``maximum.accumulate`` instead of a Python loop per cell.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return []
    vocab: Dict[str, int] = {}
    ai = np.array([vocab.setdefault(t, len(vocab)) for t in a])
    bi = np.array([vocab.setdefault(t, len(vocab)) for t in b])
    cols = np.arange(n + 1)
    previous = np.zeros(n + 1)  # b's leading part is free
    moves = np.zeros((m + 1, n + 1), dtype=np.uint8)  # 0 diag, 1 up, 2 left
    moves[0, :] = 2
    for i in range(1, m + 1):
        scores = np.where(bi == ai[i - 1], MATCH, MISMATCH)
        diag = previous[:-1] + scores
        up = previous + SKIP_CUE_TOKEN
        arrive = up.copy()
        from_diag = np.zeros(n + 1, dtype=bool)
        better = diag > up[1:]
        arrive[1:] = np.where(better, diag, up[1:])
        from_diag[1:] = better
        run = np.maximum.accumulate(arrive - cols * SKIP_WORD)
        current = run + cols * SKIP_WORD
        left = current > arrive + 1e-12
        moves[i] = np.where(left, 2, np.where(from_diag, 0, 1))
        previous = current
    # b's trailing part is free too: end wherever the last row is best.
    j = int(np.argmax(previous))
    i = m
    pairs: List[Tuple[int, int]] = []
    while i > 0:
        move = moves[i, j] if j > 0 else 1
        if move == 0:
            if ai[i - 1] == bi[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def align_to_words(
    doc: SubtitleDoc,
    words: Sequence[Word],
    prior: Optional[Callable[[float], float]] = None,
    ratio: float = 1.0,
    check: Callable[[], None] = lambda: None,
) -> Tuple[SubtitleDoc, int, int]:
    """Give each cue the time its words were spoken.

    ``prior`` maps subtitle time to roughly the right video time (from a
    coarse activity match), so each chunk of cues is aligned only against
    the words near where it belongs. Cues with at least half their tokens
    matched take the matched words' start and end; the rest keep their
    length and move by the offset interpolated from their matched
    neighbours. Returns (doc, matched cues, cues with no tokens).
    """
    prior = prior or (lambda t: t)
    cues = sorted((c.copy() for c in doc.cues), key=lambda c: (c.start, c.end))
    w_tokens, w_starts, w_ends = _word_tokens(words)
    cue_tokens = [normalize_tokens(c.text) for c in cues]
    new_start: List[Optional[float]] = [None] * len(cues)
    new_end: List[Optional[float]] = [None] * len(cues)
    for first in range(0, len(cues), TRANSCRIPT_CHUNK_CUES):
        check()
        chunk = range(first, min(len(cues), first + TRANSCRIPT_CHUNK_CUES))
        a: List[str] = []
        owner: List[int] = []
        for index in chunk:
            a.extend(cue_tokens[index])
            owner.extend([index] * len(cue_tokens[index]))
        if not a:
            continue
        lo = prior(cues[chunk[0]].start) - TRANSCRIPT_MARGIN_S
        hi = prior(cues[chunk[-1]].end) + TRANSCRIPT_MARGIN_S
        j_lo = int(np.searchsorted(w_starts, lo))
        j_hi = int(np.searchsorted(w_starts, hi))
        if j_hi <= j_lo:
            continue
        pairs = align_tokens(a, w_tokens[j_lo:j_hi])
        by_cue: Dict[int, List[Tuple[int, int]]] = {}
        for i, j in pairs:
            by_cue.setdefault(owner[i], []).append((i, j + j_lo))
        for index, matched in by_cue.items():
            total = len(cue_tokens[index])
            if len(matched) < max(1, math.ceil(MIN_MATCHED_FRACTION * total)):
                continue
            new_start[index] = float(w_starts[matched[0][1]])
            new_end[index] = float(w_ends[matched[-1][1]])
    matched_idx = [i for i in range(len(cues)) if new_start[i] is not None]
    empty = sum(1 for t in cue_tokens if not t)
    out: List[Cue] = []
    if not matched_idx:
        return doc.copy(), 0, empty
    shifts = np.array([new_start[i] - cues[i].start * ratio for i in matched_idx])
    anchors = np.array([cues[i].start for i in matched_idx])
    for index, cue in enumerate(cues):
        if new_start[index] is not None:
            start, end = new_start[index], new_end[index]
            end = max(end, start + min(MIN_CUE_S, cue.duration * ratio))
        else:
            shift = float(np.interp(cue.start, anchors, shifts))
            start = cue.start * ratio + shift
            end = cue.end * ratio + shift
        out.append(cue.copy(start=max(0.0, start), end=max(0.0, end)))
    # A matched line may now run into the next; the next line's start wins.
    for prev, nxt in zip(out, out[1:]):
        if prev.end > nxt.start > prev.start:
            prev.end = max(prev.start + 0.2, nxt.start - 0.04)
    return doc.copy(out), len(matched_idx), empty


def _transcript(doc, options, video, ctx) -> SyncOutcome:
    from .tasks import TaskError

    if not video or not video.get("path"):
        raise TaskError("Syncing to a transcript needs the video.")
    try:
        from . import generate
    except ImportError as exc:
        raise TaskError("Transcript sync needs speech recognition, which is not available in this build.") from exc
    ctx.progress(0, "Recognising speech")
    words, language, _info = generate.transcribe(
        video,
        {
            "engine": options.get("asrEngine") or "mlx-whisper",
            "model": options.get("asrModel") or "large-v3-turbo",
            "language": doc.language or "auto",
            "task": "transcribe",
            "vad": True,
        },
        ctx,
    )
    ctx.check()
    if not words:
        raise TaskError("Speech recognition found no words in the video.")
    # A coarse placement first, from when words and cues are on: that is
    # what lets a subtitle that is minutes out, or at the wrong frame rate,
    # still find its words.
    signal = activity([(w.start, w.end) for w in words])
    coarse = sync_to_signal(doc, signal, {**options, "allowSplits": False}, "transcript", ctx.check, ctx.progress)
    ratio = coarse.ratio if coarse.applied else 1.0
    offset = coarse.offset_s if coarse.applied else 0.0
    ctx.progress(0, "Aligning lines to words")
    out, matched, _empty = align_to_words(doc, words, lambda t: t * ratio + offset, ratio, ctx.check)
    total = len(doc.cues) or 1
    fraction = matched / total
    outcome = SyncOutcome(
        doc=out, offset_s=offset, ratio=ratio, method="transcript",
        score=float(np.clip(fraction, 0.0, 1.0)),
        confidence="high" if fraction >= 0.6 else "medium" if fraction >= 0.3 else "low",
        framerate=coarse.framerate if coarse.applied else None,
        peak_z=coarse.peak_z,
    )
    from .languages import normalize

    if normalize(language) and normalize(doc.language) and normalize(language) != normalize(doc.language):
        outcome.warnings.append(
            f"The video's speech was recognised as {language}, but the subtitle is {doc.language}; "
            "transcript sync needs a subtitle in the spoken language."
        )
    if fraction < 0.3:
        outcome.warnings.append(f"Only {matched} of {total} lines matched the recognised speech.")
    outcome.cues_moved = _moved(doc, outcome.doc)
    return outcome


# ============================================================ entry points


def sync_doc(doc: SubtitleDoc, options: Dict[str, Any], video: Optional[dict], reference: Optional[dict], ctx) -> SyncOutcome:
    """Re-time ``doc`` with the engine ``options["engine"]`` names."""
    engine = (options or {}).get("engine") or "audio"
    options = dict(options or {})
    if engine == "audio":
        return _audio(doc, options, video, ctx)
    if engine == "subtitle":
        return _subtitle(doc, options, reference, ctx)
    if engine == "reference":
        return _reference(doc, options, video, reference, ctx)
    if engine == "transcript":
        return _transcript(doc, options, video, ctx)
    from .tasks import TaskError

    raise TaskError(f"Unknown sync engine: {engine}")


def engine_statuses() -> List[dict]:
    """The four sync engines; transcript needs speech recognition."""
    transcript_ok, reason, pack = False, "Install the speech pack", None
    try:
        from . import generate

        statuses = [s for s in generate.engine_statuses() if s.get("available")]
        transcript_ok = bool(statuses)
        if statuses:
            reason = None
        else:
            all_statuses = generate.engine_statuses()
            if all_statuses:
                reason = all_statuses[0].get("reason") or reason
                pack = all_statuses[0].get("pack")
    except Exception:  # noqa: BLE001 - a missing front end is an unavailable engine
        transcript_ok = False
    return [
        {"id": "audio", "label": "Speech in the video's audio", "available": True, "reason": None, "pack": None},
        {"id": "subtitle", "label": "A correctly timed subtitle", "available": True, "reason": None, "pack": None},
        {"id": "reference", "label": "Audio of the release it was timed for", "available": True, "reason": None, "pack": None},
        {"id": "transcript", "label": "Transcript (speech recognition)", "available": transcript_ok, "reason": reason, "pack": pack},
    ]


def run_task(job: dict, ctx) -> Any:
    """Task "sync": input.subtitle re-timed to input.video."""
    from .tasks import TaskError, TaskResult, load_subtitle, preview_of, write_subtitle

    inputs = job.get("input") or {}
    subtitle = inputs.get("subtitle")
    if not subtitle:
        raise TaskError("Choose a subtitle to sync.")
    options = dict(job.get("options") or {})
    ctx.progress(0, "Reading the subtitle")
    doc = load_subtitle(subtitle, ctx)
    if not doc.cues:
        raise TaskError("The subtitle has no lines.")
    outcome = sync_doc(doc, options, inputs.get("video"), options.get("reference"), ctx)
    ctx.check()
    output = write_subtitle(outcome.doc, subtitle, job, ".synced", ctx)
    summary = outcome.describe()
    ctx.log(summary)
    return TaskResult(
        outputs=[output],
        report=outcome.report(),
        summary=summary,
        warnings=list(outcome.warnings),
        preview=preview_of(outcome.doc),
    )
