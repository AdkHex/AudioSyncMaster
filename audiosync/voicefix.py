"""The voice check: find where a dub's voices sit away from the lips, and
move just the voices there.

Why it is needed. The planner (``dubsync``) places the dub by what the two
tracks share -- music and effects -- which is exact to a fraction of a
millisecond. That is also where the dub's voices belong, as long as the dub
studio kept its voices and its background together. Some do not: a dub
made for a shorter TV or streaming cut has its picture trimmed in places,
and its voices were cut with the picture while its music was cut
elsewhere, in one piece, where the music allowed. Measured on a real
series (Goblin E01, Hindi dub on the Korean Blu-ray): around four such
places the dub's voices sat 0.16-4.7 s early and 2.3 s late for up to two
minutes each, while its music matched to 0.1 ms. No placement of the whole
track can fix that; only the voices can move.

How it finds them. Speech is language-blind in its rhythm: a dub actor
starts and stops with the mouth on screen, so the on/off pattern of the
dub's speech follows the original's, line for line, gap for gap. With
Silero's speech probability for both tracks, the pattern of every stretch
of dialogue is compared with the dub's at lags up to +-8 s:

1. on the raw soundtracks, every 10 s (cheap: one pass of the detector),
   to find the few places worth a closer look;
2. there, on the voices alone (Demucs separates them, the only costly
   step), every 2 s, to measure the shift and where it starts and stops;
3. runs of windows agreeing on one shift become *voice pieces*: a span of
   the dub's own timeline, cut in the dub's pauses, whose voices are laid
   at their own offset instead of the plan's. A shift within 150 ms of a
   neighbouring stretch's offset takes that offset, so the moved voices
   join the stretch after them without a seam (the usual case: the voices
   already sit where the next scene's music does);
4. a piece is kept only when the lines it moves pair up with the
   original's better than they did.

How it moves them (``VoicePatch``, applied by ``dubrender.render``). The
renderer's own reader decodes the dub, so the separated voices are cut from
exactly the samples it writes; the voices are subtracted where the plan
plays them and added where they belong. The background is not touched. A
piece that lands in a fill -- the original plays there because the dub
lacks the scene -- takes the original's voices out of the fill too, so two
languages do not talk at once.

Everything here is numpy; the detector and separator run in the voice
tools' worker (``voicetools``). Without the tools the check is skipped and
says so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .media import CancellationToken, MediaError

SPEECH_RATE = 125.0  # Silero values a second (see voiceworker)
SPEECH_SR = 16000
SPEECH_T0 = 256 / SPEECH_SR  # centre of the first 32 ms chunk
SPEECH_ON, SPEECH_OFF = 0.5, 0.35

SCAN_WINDOW_S, SCAN_STEP_S = 20.0, 10.0
SCAN_RANGE_S = 8.0
# A raw-soundtrack window is worth a closer look when the dub's speech
# agrees clearly better at another lag than at the plan's.
SCAN_MIN_SHIFT_S = 0.15
SCAN_MIN_NCC = 0.45
SCAN_MIN_GAIN = 0.12
SCAN_MIN_TALK = 0.08
REGION_PAD_S = 40.0

PROFILE_WINDOW_S, PROFILE_STEP_S = 8.0, 2.0
# Below this a shift is within what dub actors themselves vary by (a dub's
# lines start ~70 ms ahead of the original's as a habit), and the line
# evidence cannot tell a real displacement from that; above it a
# displacement is plain on screen and the evidence is unambiguous.
PIECE_MIN_SHIFT_S = 0.3
PIECE_MIN_NCC = 0.55
PIECE_MIN_GAIN = 0.15
RUN_TOLERANCE_S = 0.12
SNAP_S = 0.15
# A line near a measured displacement is tried at its shift this far out.
EXTEND_S = 60.0
# A piece whose voices sit at the offset of a stretch starting this soon
# after its last line runs on into that stretch: what lies between belongs
# to the same scene, which the dub already plays in step from there.
JOIN_REACH_S = 5.0
# A move on two paired lines only needs the rhythm to agree this well.
STRONG_NCC = 0.7
SHARED_VOICE_NCC = 0.2
SEPARATE_RATE = 44100
FADE_S = 0.05

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, str], None]


@dataclass
class VoicePiece:
    """A span of the dub's own timeline whose voices go at ``level_s``
    (dub = video + level) rather than where the plan's stretch puts them."""

    dub_start_s: float
    dub_end_s: float
    level_s: float
    shift_s: float
    join_end: bool = False
    note: str = ""

    @property
    def video_start_s(self) -> float:
        return self.dub_start_s - self.level_s

    @property
    def video_end_s(self) -> float:
        return self.dub_end_s - self.level_s

    def to_dict(self) -> dict:
        return {
            "dubStartS": round(self.dub_start_s, 4), "dubEndS": round(self.dub_end_s, 4),
            "levelS": round(self.level_s, 4), "shiftS": round(self.shift_s, 4),
            "joinEnd": self.join_end, "note": self.note,
            "videoStartS": round(self.video_start_s, 4), "videoEndS": round(self.video_end_s, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoicePiece":
        return cls(
            dub_start_s=float(data["dubStartS"]), dub_end_s=float(data["dubEndS"]),
            level_s=float(data["levelS"]), shift_s=float(data.get("shiftS", 0.0) or 0.0),
            join_end=bool(data.get("joinEnd")), note=str(data.get("note") or ""),
        )


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{seconds % 60:06.3f}"


# ---------------------------------------------------------------------------
# Speech curves and their comparison (pure numpy, tested without the tools)
# ---------------------------------------------------------------------------

def curve_at(curve: np.ndarray, times: np.ndarray) -> np.ndarray:
    """A speech curve (index k at SPEECH_T0 + k / SPEECH_RATE) at arbitrary times."""
    index = (np.asarray(times, dtype=np.float64) - SPEECH_T0) * SPEECH_RATE
    return np.interp(index, np.arange(len(curve)), curve, left=0.0, right=0.0)


def lag_scan(reference: np.ndarray, other: np.ndarray, reach: int) -> np.ndarray:
    """Normalised correlation of ``reference`` (n values) against ``other``
    (n + 2 * reach values, centred) at lags -reach..reach; lag > 0 means
    ``other`` must move later to agree."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    n = len(a)
    a = a - a.mean()
    norm_a = float(np.linalg.norm(a))
    count = 2 * reach + 1
    if n == 0 or len(b) < n + 2 * reach or norm_a < 1e-9:
        return np.zeros(count)
    size = 1 << int(np.ceil(np.log2(len(b) + n)))
    # other[j + i] against a[i], for j = 0..2*reach; j = reach is lag 0
    raw = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)[:count]
    sums = np.concatenate([[0.0], np.cumsum(b)])
    squares = np.concatenate([[0.0], np.cumsum(b * b)])
    seg_sum = sums[n:n + count] - sums[:count]
    seg_sq = squares[n:n + count] - squares[:count]
    variance = np.maximum(seg_sq - seg_sum * seg_sum / n, 1e-12)
    # a is zero-mean, so the other side's mean drops out of the numerator
    ncc = raw / (norm_a * np.sqrt(variance))
    # b[j + i] matching a[i] means the dub's sound for time i sits j - reach
    # samples later in its own timeline: it must move EARLIER by that; the
    # returned lag is how much LATER the dub must move, so flip.
    return ncc[::-1]


def segments(curve: np.ndarray, times: np.ndarray, min_s: float = 0.15, bridge_s: float = 0.2) -> List[Tuple[float, float]]:
    """Speech segments (start, end) from a probability curve sampled at ``times``."""
    out: List[List[float]] = []
    on, start = False, 0
    for i, value in enumerate(curve):
        if not on and value > SPEECH_ON:
            on, start = True, i
        elif on and value < SPEECH_OFF:
            on = False
            out.append([float(times[start]), float(times[i])])
    if on and len(times):
        out.append([float(times[start]), float(times[-1])])
    merged: List[List[float]] = []
    for a, b in out:
        if merged and a - merged[-1][1] < bridge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_s]


def pair_lines(original: Sequence[Tuple[float, float]], dub: Sequence[Tuple[float, float]],
               reach_s: float = 0.6) -> Tuple[int, List[float]]:
    """How many of the original's lines have a dub line starting within
    ``reach_s`` and overlapping it, and those start differences (s)."""
    starts = np.array([a for a, _ in dub])
    ends = np.array([b for _, b in dub])
    diffs: List[float] = []
    for a, b in original:
        if not len(starts):
            break
        j = int(np.argmin(np.abs(starts - a)))
        if abs(starts[j] - a) < reach_s and ends[j] > a and starts[j] < b:
            diffs.append(float(starts[j] - a))
    return len(diffs), diffs


@dataclass
class Window:
    start_s: float
    lag_s: float
    ncc: float
    ncc_at_zero: float
    offset_s: float

    @property
    def gain(self) -> float:
        return self.ncc - self.ncc_at_zero


def profile(original: np.ndarray, dub: np.ndarray, offset_at: Callable[[float], Optional[float]],
            lo_s: float, hi_s: float, window_s: float, step_s: float, reach_s: float,
            min_talk: float = SCAN_MIN_TALK, skip: Optional[Callable[[float, float], bool]] = None) -> List[Window]:
    """Best lag of the dub's speech against the original's, window by window.

    ``original`` is on the video's clock, ``dub`` on the dub's (output)
    clock; ``offset_at(t)`` is the plan's offset at video time t (None in a
    fill). A window takes the offset at its middle.
    """
    step = 1.0 / SPEECH_RATE
    reach = int(round(reach_s * SPEECH_RATE))
    out: List[Window] = []
    t = lo_s
    while t + window_s <= hi_s + 1e-9:
        offset = offset_at(t + window_s / 2)
        if offset is not None and not (skip and skip(t, t + window_s)):
            times = t + np.arange(int(round(window_s * SPEECH_RATE))) * step
            a = curve_at(original, times)
            if float((a > SPEECH_ON).mean()) >= min_talk:
                wide = t - reach * step + np.arange(len(times) + 2 * reach) * step
                b = curve_at(dub, wide + offset)
                row = lag_scan(a, b, reach)
                k = int(np.argmax(row))
                out.append(Window(t, (k - reach) * step, float(row[k]), float(row[reach]), offset))
        t += step_s
    return out


def candidate_regions(windows: Sequence[Window], min_shift_s: float = SCAN_MIN_SHIFT_S,
                      min_ncc: float = SCAN_MIN_NCC, min_gain: float = SCAN_MIN_GAIN,
                      pad_s: float = REGION_PAD_S, window_s: float = SCAN_WINDOW_S) -> List[Tuple[float, float]]:
    """Merged spans of the video worth a closer look."""
    hits = [w for w in windows if abs(w.lag_s) > min_shift_s and w.ncc >= min_ncc and w.gain >= min_gain]
    regions: List[List[float]] = []
    for w in hits:
        lo, hi = w.start_s - pad_s, w.start_s + window_s + pad_s
        if regions and lo <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], hi)
        else:
            regions.append([lo, hi])
    return [(max(0.0, a), b) for a, b in regions]


@dataclass
class Run:
    start_s: float
    end_s: float
    shift_s: float
    windows: List[Window] = field(default_factory=list)


def runs_of(windows: Sequence[Window], window_s: float = PROFILE_WINDOW_S) -> List[Run]:
    """Consecutive windows agreeing on one displacement."""
    runs: List[Run] = []
    current: List[Window] = []

    def close() -> None:
        if len(current) >= 2:
            lags = [w.lag_s for w in current]
            runs.append(Run(current[0].start_s, current[-1].start_s + window_s, float(np.median(lags)), list(current)))

    for w in windows:
        displaced = abs(w.lag_s) > PIECE_MIN_SHIFT_S and w.ncc >= PIECE_MIN_NCC and w.gain >= PIECE_MIN_GAIN
        if displaced and current and abs(w.lag_s - float(np.median([c.lag_s for c in current]))) <= RUN_TOLERANCE_S \
                and w.start_s - current[-1].start_s <= 2 * PROFILE_STEP_S + 1e-9 \
                and abs(w.offset_s - current[-1].offset_s) < 0.5:
            current.append(w)
            continue
        close()
        current = [w] if displaced else []
    close()
    return runs


# ---------------------------------------------------------------------------
# The plan's side
# ---------------------------------------------------------------------------

class PlanMap:
    """The plan's dub stretches as (video start, video end, offset)."""

    def __init__(self, plan) -> None:
        self.dubs = [(s.start_s, s.end_s, s.offset_s) for s in plan.dub_segments if s.offset_s is not None]
        self.fills = [(s.start_s, s.end_s) for s in plan.fill_segments]
        self.plan = plan

    def offset_at(self, t: float) -> Optional[float]:
        for a, b, offset in self.dubs:
            if a <= t < b:
                return offset
        return None

    def offsets(self, times: np.ndarray) -> np.ndarray:
        out = np.full(len(times), np.nan)
        for a, b, offset in self.dubs:
            out[(times >= a) & (times < b)] = offset
        return out

    def video_of(self, dub_s: float, lo: float = -1e18, hi: float = 1e18) -> Optional[Tuple[float, float]]:
        """Where the plan plays dub time ``dub_s``: (video time, offset), the
        occurrence inside [lo, hi) first; None when the plan skips it."""
        found = None
        for a, b, offset in self.dubs:
            t = dub_s - offset
            if a <= t < b:
                if lo <= t < hi:
                    return t, offset
                found = found or (t, offset)
        return found

    def levels_near(self, t: float, reach_s: float = 90.0) -> List[float]:
        return sorted({round(o, 4) for a, b, o in self.dubs if b > t - reach_s and a < t + reach_s})

    def stretch_starting_near(self, dub_s: float, level_s: float, reach_s: float = JOIN_REACH_S) -> Optional[Tuple[float, float, float]]:
        """A stretch at ``level_s`` whose dub time begins within reach of ``dub_s``."""
        for a, b, offset in self.dubs:
            if abs(offset - level_s) <= 0.003 and abs((a + offset) - dub_s) <= reach_s:
                return a, b, offset
        return None


def _separated_speech(worker, reader, lo: float, hi: float, rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Voices of [lo, hi) (on the reader's clock) and Silero's curve of them."""
    start = int(round(lo * rate))
    audio = reader(start, int(round((hi - lo) * rate)))
    voices = worker.vocals(audio, rate)
    mono = voices.mean(axis=1) if voices.ndim == 2 else voices
    speech_in = _resample(mono, rate, SPEECH_SR)
    return voices, worker.speech(speech_in)


def _resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Band-limited enough for a speech detector: a windowed-sinc FIR by FFT."""
    if rate == target:
        return np.asarray(x, dtype=np.float32)
    n = len(x)
    m = int(round(n * target / rate))
    spectrum = np.fft.rfft(np.asarray(x, dtype=np.float64))
    keep = m // 2 + 1
    cut = np.zeros(keep, dtype=complex)
    cut[: min(keep, len(spectrum))] = spectrum[: min(keep, len(spectrum))]
    return (np.fft.irfft(cut, m) * (m / n)).astype(np.float32)


def find_voice_pieces(
    plan,
    worker,
    read_video: Callable[[int, int, int], np.ndarray],
    read_dub: Callable[[int, int, int], np.ndarray],
    speech_curves: Tuple[np.ndarray, np.ndarray],
    token: Optional[CancellationToken] = None,
    progress: Optional[ProgressFn] = None,
    log: Optional[LogFn] = None,
) -> List[VoicePiece]:
    """The voice pieces of ``plan`` (see the module notes).

    ``read_video(rate, start_sample, count)`` / ``read_dub(...)`` read mono
    or stereo float32 on each file's clock (the dub's output clock);
    ``speech_curves`` are Silero's curves of the raw video audio and dub
    (see ``speech_of_track``).
    """
    say = log or (lambda _m: None)
    step = progress or (lambda _p, _s: None)
    stage = "checking the dub's voices against the lips"
    mapping = PlanMap(plan)
    video_speech, dub_speech = speech_curves
    scan = profile(video_speech, dub_speech, mapping.offset_at, 0.0, plan.video_duration_s,
                   SCAN_WINDOW_S, SCAN_STEP_S, SCAN_RANGE_S)
    regions = candidate_regions(scan)
    if not regions:
        say(f"voices: the dub's lines follow the original's across all {len(scan)} dialogue windows")
        return []
    say(f"voices: {len(regions)} place(s) where the dub's lines may sit away from the lips: "
        + ", ".join(f"{_clock(a)}-{_clock(b)}" for a, b in regions))
    pieces: List[VoicePiece] = []
    for index, (lo, hi) in enumerate(regions):
        if token:
            token.raise_if_cancelled()
        step(int(100 * index / len(regions)), stage)
        lo = max(0.0, lo)
        hi = min(plan.video_duration_s, hi)
        offsets = [o for o in (mapping.offset_at(t) for t in np.arange(lo, hi, 1.0)) if o is not None]
        if not offsets:
            continue
        dub_lo = max(0.0, lo + min(offsets) - SCAN_RANGE_S)
        dub_hi = hi + max(offsets) + SCAN_RANGE_S
        video_voices, video_curve = _separated_speech(worker, lambda s, c: read_video(SEPARATE_RATE, s, c),
                                                      lo, hi, SEPARATE_RATE)
        dub_voices, dub_curve = _separated_speech(worker, lambda s, c: read_dub(SEPARATE_RATE, s, c),
                                                  dub_lo, dub_hi, SEPARATE_RATE)
        # curves on absolute clocks
        v_curve = np.concatenate([np.zeros(int(round(lo * SPEECH_RATE))), video_curve])
        d_curve = np.concatenate([np.zeros(int(round(dub_lo * SPEECH_RATE))), dub_curve])

        songs: List[Tuple[float, float]] = []

        def shared(a: float, b: float, off_fn=mapping.offset_at) -> bool:
            """A song both tracks carry: their voices are the same waveform."""
            offset = off_fn((a + b) / 2)
            if offset is None:
                return False
            if any(sa <= a and b <= sb for sa, sb in songs):
                return True
            i0, i1 = int((a - lo) * SEPARATE_RATE), int((b - lo) * SEPARATE_RATE)
            j0 = int((a + offset - dub_lo) * SEPARATE_RATE)
            if i0 < 0 or j0 < 0 or i1 > len(video_voices) or j0 + (i1 - i0) > len(dub_voices):
                return False
            x = np.diff(video_voices[i0:i1].mean(axis=1))
            y = np.diff(dub_voices[j0:j0 + (i1 - i0)].mean(axis=1))
            reach = int(0.02 * SEPARATE_RATE)
            if len(x) <= 2 * reach or np.linalg.norm(x) < 1e-6 or np.linalg.norm(y) < 1e-6:
                return False
            row = lag_scan(x[reach:-reach], y, reach)
            if float(row.max()) > SHARED_VOICE_NCC:
                songs.append((a, b))
                return True
            return False

        windows = profile(v_curve, d_curve, mapping.offset_at, lo, hi, PROFILE_WINDOW_S, PROFILE_STEP_S,
                          SCAN_RANGE_S, min_talk=0.1, skip=shared)
        debug = os.environ.get("AUDIOSYNC_VOICE_DEBUG")
        if debug:  # the region's curves and windows, to study a check offline
            os.makedirs(debug, exist_ok=True)
            np.savez(os.path.join(debug, f"region_{lo:.0f}.npz"), v_curve=v_curve, d_curve=d_curve, lo=lo, hi=hi,
                     windows=np.array([[w.start_s, w.lag_s, w.ncc, w.ncc_at_zero, w.offset_s] for w in windows]),
                     songs=np.array(songs).reshape(-1, 2))
        pieces.extend(_pieces_in_region(runs_of(windows), mapping, v_curve, d_curve, lo, hi, say, songs))
    step(100, stage)
    pieces = _without_overlaps(sorted(pieces, key=lambda p: p.dub_start_s), say)
    if pieces:
        moved = sum(p.dub_end_s - p.dub_start_s for p in pieces)
        say(f"voices: moved the dub's voices in {len(pieces)} place(s), {moved:.0f}s of dub in all; "
            "the music and effects stay where they are")
    else:
        say("voices: nothing to move -- where the lines looked off, moving them did not make them pair up better")
    return pieces


def _refine(run: Run, mapping: PlanMap, v_curve: np.ndarray, d_curve: np.ndarray) -> Optional[Tuple[float, float]]:
    """The run's shift to 8 ms over its whole span, snapped to a stretch's
    offset when one is as good: (shift, level)."""
    step = 1.0 / SPEECH_RATE
    offset = mapping.offset_at((run.start_s + run.end_s) / 2)
    if offset is None:
        return None
    times = np.arange(run.start_s, run.end_s, step)
    reach = int(round(0.3 * SPEECH_RATE))
    centre = run.shift_s
    wide = run.start_s - reach * step + np.arange(len(times) + 2 * reach) * step
    row = lag_scan(curve_at(v_curve, times), curve_at(d_curve, wide + offset - centre), reach)
    k = int(np.argmax(row))
    shift = centre + (k - reach) * step
    best = float(row[k])
    level = offset - shift
    for candidate in mapping.levels_near((run.start_s + run.end_s) / 2):
        cand_shift = offset - candidate
        if abs(cand_shift - shift) <= SNAP_S and abs(cand_shift) > 1e-4:
            idx = reach + int(round((cand_shift - centre) / step))
            if 0 <= idx < len(row) and row[idx] >= best - 0.03:
                return cand_shift, candidate
    return shift, level


def _pieces_in_region(runs: Sequence[Run], mapping: PlanMap, v_curve: np.ndarray, d_curve: np.ndarray,
                      lo: float, hi: float, say: LogFn,
                      songs: Sequence[Tuple[float, float]] = ()) -> List[VoicePiece]:
    """Give every dub line in [lo, hi) the shift that pairs it with the
    original's lines -- none, or one of the runs' -- and cut pieces from
    consecutive lines sharing a shift, in the dub's pauses."""
    step = 1.0 / SPEECH_RATE
    refined: List[Tuple[Run, float, float, bool]] = []
    for run in runs:
        result = _refine(run, mapping, v_curve, d_curve)
        if result is not None:
            snapped = any(abs(result[1] - level) < 1e-4 for level in mapping.levels_near((run.start_s + run.end_s) / 2))
            refined.append((run, result[0], result[1], snapped))
    if not refined:
        return []
    # Runs agreeing on a shift within SNAP_S are one displacement measured
    # in pieces: they take one offset -- a stretch's, when one of them
    # snapped to it, else the one read over the most windows.
    refined.sort(key=lambda item: item[1])
    clusters: List[List[Tuple[Run, float, float, bool]]] = []
    for item in refined:
        # two readings of one displacement meet at a stretch's offset; two
        # displacements that merely differ little (a scene 4.5 s off beside
        # a song 4.4 s off) stay apart
        if clusters and abs(item[1] - clusters[-1][-1][1]) <= SNAP_S and (
                item[3] or any(c[3] for c in clusters[-1])):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    shifts: List[Tuple[Run, float, float]] = []
    for cluster in clusters:
        chosen = next((c for c in cluster if c[3]), max(cluster, key=lambda c: len(c[0].windows)))
        extent = Run(min(c[0].start_s for c in cluster), max(c[0].end_s for c in cluster), chosen[1],
                     [w for c in cluster for w in c[0].windows])
        shifts.append((extent, chosen[1], chosen[2]))
    times = np.arange(lo, hi, step)
    korean = segments(curve_at(v_curve, times), times)
    k_starts = np.array([a for a, _ in korean]) if korean else np.zeros(0)
    offsets = [o for o in (mapping.offset_at(t) for t in np.arange(lo, hi, 1.0)) if o is not None]
    reach = max(abs(r.shift_s) for r, _, _ in shifts) + 2.0
    dub_times = np.arange(lo + min(offsets) - reach, hi + max(offsets) + reach, step)
    dub_lines = segments(curve_at(d_curve, dub_times), dub_times)

    def onset_error(start_video: float, end_video: float) -> float:
        if not len(k_starts):
            return 9.9
        j = int(np.argmin(np.abs(k_starts - start_video)))
        a, b = korean[j]
        return abs(k_starts[j] - start_video) if (b > start_video and a < end_video) else 9.9

    assigned: List[Optional[int]] = []
    barrier: List[bool] = []
    # the cluster a line already sits at the offset of (a stretch the dub
    # plays in step, which a displaced run of voices may join)
    at_level: List[Optional[int]] = []
    for a, b in dub_lines:
        current = mapping.video_of((a + b) / 2, lo, hi)
        now = (a - current[1], b - current[1]) if current else None
        at_level.append(next((k for k, (_r, _s, level) in enumerate(shifts)
                              if current is not None and abs(current[1] - level) <= 0.003), None))
        if now is not None and any(sa < now[1] and now[0] < sb for sa, sb in songs):
            # sung by both tracks: music, which the plan already lays in step
            assigned.append(None)
            barrier.append(True)
            continue
        barrier.append(False)
        candidates = [(index, (a - level, b - level)) for index, (run, _shift, level) in enumerate(shifts)
                      if max(lo, run.start_s - EXTEND_S) <= (a + b) / 2 - level <= min(hi, run.end_s + EXTEND_S)]
        if (now is None or not (lo <= (now[0] + now[1]) / 2 <= hi)) and not candidates:
            assigned.append(None)
            barrier[-1] = False
            continue
        best, best_err = None, (onset_error(*now) if now else 9.9)
        best_score = best_err
        for index, moved in candidates:
            err = onset_error(*moved)
            run = shifts[index][0]
            mid = (moved[0] + moved[1]) / 2
            outside = max(0.0, run.start_s - mid, mid - run.end_s)
            score = err + 0.01 * outside  # a line nearer a displacement takes its shift
            if err < 0.35 and err < best_err - 0.15 and score < best_score:
                best, best_score = index, score
        assigned.append(best)
    def inside_share(span: Optional[Tuple[float, float]]) -> float:
        if span is None or span[1] <= span[0]:
            return 0.0
        covered = sum(max(0.0, min(span[1], kb) - max(span[0], ka)) for ka, kb in korean)
        return covered / (span[1] - span[0])

    now_spans: List[Optional[Tuple[float, float]]] = []
    for a, b in dub_lines:
        current = mapping.video_of((a + b) / 2, lo, hi)
        now_spans.append((a - current[1], b - current[1]) if current else None)
    # Lines no shift pairs better (a breath, a word the original has no line
    # for, speech running on without a pause) between lines of one
    # displacement go along, and so do those between a displaced line and
    # the stretch it runs into: inside a displaced scene a line that meets
    # one of the original's where it is does so by chance, and left behind
    # it would talk over the lines moved past it. Only a song both tracks
    # sing stops that.
    i = 0
    while i < len(assigned):
        if assigned[i] is not None:
            j = i + 1
            while j < len(assigned) and assigned[j] is None and at_level[j] != assigned[i]:
                j += 1
            ends_same = j < len(assigned) and (assigned[j] == assigned[i] or at_level[j] == assigned[i])
            between = range(i + 1, j)
            fits = not any(barrier[k] for k in between)
            if ends_same and j > i + 1 and fits and dub_lines[j][0] - dub_lines[i][1] <= 20.0:
                for k in between:
                    assigned[k] = assigned[i]
            i = j if j > i + 1 else i + 1
        else:
            i += 1
    # A line next to a displaced one that lands inside the original's speech
    # when moved, and outside it where it is, goes with it.
    changed = True
    while changed:
        changed = False
        for i in range(len(assigned)):
            if assigned[i] is not None or barrier[i]:
                continue
            for k in (i - 1, i + 1):
                if 0 <= k < len(assigned) and assigned[k] is not None \
                        and abs(dub_lines[i][0] - dub_lines[k][0]) <= 12.0:
                    level = shifts[assigned[k]][2]
                    moved = (dub_lines[i][0] - level, dub_lines[i][1] - level)
                    if inside_share(moved) >= 0.6 and inside_share(now_spans[i]) < 0.6:
                        assigned[i] = assigned[k]
                        changed = True
                        break

    pieces: List[VoicePiece] = []
    i = 0
    while i < len(dub_lines):
        if assigned[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < len(dub_lines) and assigned[j + 1] == assigned[i]:
            j += 1
        run, shift, level = shifts[assigned[i]]
        first, last = dub_lines[i], dub_lines[j]
        before = dub_lines[i - 1][1] if i > 0 else first[0] - 1.0
        after = dub_lines[j + 1][0] if j + 1 < len(dub_lines) else last[1] + 1.0
        dub_start = first[0] - min(0.4, max(0.05, (first[0] - before) / 2))
        dub_end = last[1] + min(0.4, max(0.05, (after - last[1]) / 2))
        strength = _strength(run)
        piece = _validated(dub_start, dub_end, level, shift, korean, dub_lines[i:j + 1], mapping, say, strength)
        if piece is not None:
            pieces.append(piece)
        i = j + 1
    return pieces


def _strength(run: Run) -> float:
    """How clearly the rhythm agreed at the run's shift: the mean of the
    better half of its windows' agreement."""
    values = sorted((w.ncc for w in run.windows), reverse=True)
    return float(np.mean(values[: max(1, len(values) // 2)])) if values else 0.0


def _inside(lines: Sequence[Tuple[float, float]], korean: Sequence[Tuple[float, float]]) -> int:
    """How many lines lie mostly (60 %) inside the original's speech."""
    count = 0
    for a, b in lines:
        covered = sum(max(0.0, min(b, kb) - max(a, ka)) for ka, kb in korean)
        count += covered >= 0.6 * (b - a)
    return count


def _validated(dub_start: float, dub_end: float, level: float, shift: float,
               korean: Sequence[Tuple[float, float]], lines: Sequence[Tuple[float, float]],
               mapping: PlanMap, say: LogFn, strength: float = 1.0) -> Optional[VoicePiece]:
    """A piece, if its lines pair with the original's clearly better moved
    than where they are: three more lines starting with the original's, or
    -- for a short piece -- two, when the rhythm agreed strongly and the
    moved lines sit inside the original's speech."""
    moved = [(a - level, b - level) for a, b in lines]
    lo = min(a for a, _ in moved) - abs(shift) - 1.0
    hi = max(b for _, b in moved) + abs(shift) + 1.0
    now = []
    for a, b in lines:
        current = mapping.video_of((a + b) / 2, lo, hi)
        if current is not None:
            now.append((a - current[1], b - current[1]))
    local = [(a, b) for a, b in korean if b > lo and a < hi]
    before_n, before_d = pair_lines(local, now)
    after_n, after_d = pair_lines(local, moved)
    inside_before, inside_after = _inside(now, local), _inside(moved, local)
    many = after_n >= 3 and after_n >= before_n + 2 and after_n >= 0.5 * len(lines)
    # a short piece: at least one line starting with the original's, the
    # rest inside the original's speech, and a clear rhythm behind it
    few = (after_n >= 1 and after_n > before_n and len(lines) >= 2 and strength >= STRONG_NCC
           and inside_after >= 0.75 * len(lines) and inside_after > inside_before)
    ok = abs(shift) >= PIECE_MIN_SHIFT_S and (many or few)
    where = f"{_clock(dub_start - level)}-{_clock(dub_end - level)}"
    if not ok:
        say(f"voices: {where} looked {abs(shift) * 1000:.0f} ms {'early' if shift > 0 else 'late'}, but moving "
            f"its {len(lines)} line(s) did not pair them clearly better ({before_n} -> {after_n}, "
            f"rhythm {strength:.2f}); left as it is")
        return None
    join_end = False
    joined = mapping.stretch_starting_near(dub_end, level)
    if joined is not None:
        join_dub = joined[0] + joined[2]
        if join_dub >= lines[-1][1] - 0.02:
            dub_end, join_end = join_dub, True
            where = f"{_clock(dub_start - level)}-{_clock(dub_end - level)}"
    note = (f"the dub's voices for {where} sat {abs(shift) * 1000:.0f} ms {'early' if shift > 0 else 'late'} "
            f"of the lips while its music matched; moved {'later' if shift > 0 else 'earlier'} "
            f"(lines paired with the original's: {before_n} -> {after_n} of {len(lines)})")
    say(f"voices: {note}")
    return VoicePiece(dub_start, dub_end, level, shift, join_end, note)


def _without_overlaps(pieces: List[VoicePiece], say: LogFn) -> List[VoicePiece]:
    """Pieces may not share dub time, nor land on each other."""
    kept: List[VoicePiece] = []
    for piece in pieces:
        clash = any(p.dub_end_s > piece.dub_start_s and p.dub_start_s < piece.dub_end_s for p in kept) or any(
            p.video_end_s > piece.video_start_s and p.video_start_s < piece.video_end_s for p in kept)
        if clash:
            say(f"voices: {_clock(piece.video_start_s)}-{_clock(piece.video_end_s)} overlaps another move; left as it is")
            continue
        kept.append(piece)
    return kept


# ---------------------------------------------------------------------------
# Applying the pieces while the track is written
# ---------------------------------------------------------------------------

class VoicePatch:
    """Additions to the written track, by output sample."""

    def __init__(self) -> None:
        self.deltas: List[Tuple[int, np.ndarray]] = []

    def add(self, start: int, delta: np.ndarray) -> None:
        if len(delta):
            self.deltas.append((int(start), delta.astype(np.float32, copy=False)))

    def apply(self, position: int, samples: np.ndarray) -> None:
        """Add every delta overlapping ``samples`` (which starts at output
        sample ``position``), in place."""
        end = position + len(samples)
        for start, delta in self.deltas:
            a, b = max(start, position), min(start + len(delta), end)
            if b > a:
                samples[a - position:b - position] += delta[a - start:b - start]


def _window(count: int, fade: int, join_end: int = 0) -> np.ndarray:
    w = np.ones(count, dtype=np.float32)
    f = min(fade, count // 2)
    if f > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f, endpoint=False))).astype(np.float32)
        w[:f] = ramp
        if not join_end:
            w[count - f:] = ramp[::-1]
    if join_end:
        j = min(join_end, count)
        w[count - j:] = np.linspace(1.0, 0.0, j, endpoint=False, dtype=np.float32)
    return w


def _voice_channels(audio: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """What the separator gets from a track of any channel count, and how
    its voices go back into that layout: mono and stereo as they are; for
    more channels the centre (where dialogue is mixed), the rest untouched."""
    channels = audio.shape[1]
    if channels <= 2:
        return audio, lambda voices: voices
    centre = audio[:, 2:3]

    def back(voices: np.ndarray) -> np.ndarray:
        full = np.zeros((len(voices), channels), dtype=np.float32)
        full[:, 2] = voices[:, 0]
        return full

    return centre, back


def build_patch(
    plan,
    pieces: Sequence[VoicePiece],
    worker,
    dub_reader,
    video_reader,
    rate: int,
    fill_gain: float,
    xfade_s: float,
    token: Optional[CancellationToken] = None,
    log: Optional[LogFn] = None,
) -> VoicePatch:
    """The additions that move ``pieces``' voices, cut from the renderer's
    own readers (``reader.read(start_sample, count)`` on the output clock)."""
    say = log or (lambda _m: None)
    patch = VoicePatch()
    fade = int(FADE_S * rate)
    pad = int(4.0 * rate)
    half = int(round(xfade_s * rate / 2.0))
    # a piece joining a stretch at its own level hands its voices over across
    # the renderer's crossfade there: out as the stretch's fade comes in
    join = max(1, 2 * half)
    dubs = [(int(round(s.start_s * rate)), int(round(s.end_s * rate)), int(round(s.source_start_s * rate)))
            for s in plan.dub_segments]
    fills = [(int(round(s.start_s * rate)), int(round(s.end_s * rate))) for s in plan.fill_segments]
    for piece in sorted(pieces, key=lambda p: p.dub_start_s):
        if token:
            token.raise_if_cancelled()
        m0 = int(round(piece.dub_start_s * rate))
        m1 = int(round(piece.dub_end_s * rate)) + (half if piece.join_end else 0)
        audio = dub_reader.read(m0 - pad, (m1 - m0) + 2 * pad)
        part, back = _voice_channels(audio)
        voices = back(worker.vocals(part, rate))[pad: pad + (m1 - m0)]
        voices *= _window(m1 - m0, fade, join if piece.join_end else 0)[:, None]
        for o0, o1, s0 in dubs:
            a, b = max(m0, s0), min(m1, s0 + (o1 - o0))
            if b > a:
                patch.add(o0 + (a - s0), -voices[a - m0:b - m0])
        shift = int(round(-piece.level_s * rate))
        n0 = m0 + shift
        patch.add(n0, voices)
        for f0, f1 in fills:
            a, b = max(f0, n0), min(f1, n0 + (m1 - m0))
            if b > a:
                k0, k1 = f0 - half, f1 + half
                original = video_reader.read(k0 - pad, (k1 - k0) + 2 * pad)
                opart, oback = _voice_channels(original)
                theirs = oback(worker.vocals(opart, rate))[pad: pad + (k1 - k0)]
                theirs *= (fill_gain * _window(k1 - k0, max(1, 2 * half)))[:, None]
                patch.add(k0, -theirs)
                say(f"voices: took the original's voices out of the fill at {_clock(f0 / rate)} "
                    "where moved dub voices now play")
    return patch


# ---------------------------------------------------------------------------
# Glue: reading speech curves of whole tracks through the tools
# ---------------------------------------------------------------------------

def speech_of_track(worker, read: Callable[[int, int, int], np.ndarray], duration_s: float,
                    token: Optional[CancellationToken] = None,
                    on_fraction: Optional[Callable[[float], None]] = None,
                    chunk_s: float = 600.0, overlap_s: float = 4.0) -> np.ndarray:
    """Silero's curve of a whole track, read in chunks through ``read(rate,
    start, count)`` (mono 16 kHz). Chunks overlap so the detector has
    context at every seam; the overlap is dropped."""
    parts: List[np.ndarray] = []
    total = int(np.ceil(duration_s * SPEECH_RATE))
    t = 0.0
    while t < duration_s:
        if token:
            token.raise_if_cancelled()
        lo = max(0.0, t - overlap_s)
        hi = min(duration_s, t + chunk_s)
        audio = read(SPEECH_SR, int(round(lo * SPEECH_SR)), int(round((hi - lo) * SPEECH_SR)))
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        curve = worker.speech(mono)
        skip = int(round((t - lo) * SPEECH_RATE))
        parts.append(curve[skip:skip + int(round((hi - t) * SPEECH_RATE))])
        t = hi
        if on_fraction:
            on_fraction(min(1.0, t / max(duration_s, 1e-9)))
    curve = np.concatenate(parts) if parts else np.zeros(0)
    if len(curve) < total:
        curve = np.concatenate([curve, np.zeros(total - len(curve))])
    return curve[:total].astype(np.float32)


def check_voices(plan, token: Optional[CancellationToken] = None, progress: Optional[ProgressFn] = None,
                 log: Optional[LogFn] = None) -> List[VoicePiece]:
    """Run the voice check on a plan with the installed tools; [] when they
    are not installed (and it says so)."""
    from . import voicetools
    from .dubrender import _TrackReader

    say = log or (lambda _m: None)
    step = progress or (lambda _p, _s: None)
    if not voicetools.installed():
        say("voices: not checked -- the voice tools are not installed (Dub sync settings: Voice check)")
        return []
    if abs(plan.speed - 1.0) > 1e-9:
        say("voices: not checked -- the dub plays at a changed speed, which the voice check does not handle yet")
        return []
    readers: Dict[Tuple[str, int], object] = {}

    def reader(path: str, track: int, rate: int, channels: int):
        key = (path, track, rate, channels)
        if key not in readers:
            readers[key] = _TrackReader(path, track, rate, channels, 1.0, "resample", token)
        return readers[key]

    def read_video(rate: int, start: int, count: int) -> np.ndarray:
        channels = 1 if rate == SPEECH_SR else 2
        return reader(plan.video_path, plan.video_track, rate, channels).read(start, count)

    def read_dub(rate: int, start: int, count: int) -> np.ndarray:
        channels = 1 if rate == SPEECH_SR else 2
        return reader(plan.dub_path, plan.dub_track, rate, channels).read(start, count)

    stage = "reading the speech in both tracks"
    try:
        with voicetools.VoiceWorker(token=token, log=say) as worker:
            step(0, stage)
            video_speech = speech_of_track(worker, read_video, plan.video_duration_s, token,
                                           lambda f: step(int(50 * f), stage))
            dub_speech = speech_of_track(worker, read_dub, plan.dub_duration_s * plan.speed, token,
                                         lambda f: step(50 + int(50 * f), stage))
            return find_voice_pieces(plan, worker, read_video, read_dub, (video_speech, dub_speech),
                                     token, progress, say)
    except MediaError as exc:
        say(f"voices: the check could not run: {exc}")
        return []
    finally:
        for r in readers.values():
            r.close()
