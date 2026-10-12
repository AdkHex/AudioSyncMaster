"""Something else in the dub: a replaced scene, an insert laid into a gap.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..media import MediaError

from .plan import _clock
from .scores import ncc_lags
from .tuning import ENVELOPE_RATE, FOREIGN_ACTIVE, FOREIGN_BANDS, FOREIGN_CALIBRATION_MIN, FOREIGN_HOP_S, FOREIGN_LIMIT_CEIL, FOREIGN_LIMIT_FLOOR, FOREIGN_LIMIT_PERCENTILE, FOREIGN_LIMIT_SHARE, FOREIGN_MAX_T, FOREIGN_MIN_S, FOREIGN_REACH_S, FOREIGN_SPEECH_Z, FOREIGN_T_BLOCK_S, FOREIGN_WINDOW_S, MIN_BLOCK_S

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aligner import _Aligner
    from .scores import Block


# ---------------------------------------------------------------------------
# Something else in the dub: a replaced scene, an insert laid into a gap
# ---------------------------------------------------------------------------


def _foreign_bands(aligner: "_Aligner") -> List[str]:
    return [b for b in FOREIGN_BANDS if b == "full" or b in aligner.shared_bands]


def _typical_activity(aligner: "_Aligner", which: str, band: str) -> float:
    """The median spread of a track's onsets over FOREIGN_WINDOW_S windows:
    what "something is happening in this band" looks like for this track."""
    key = (which, band)
    if key not in aligner._activity:
        x = aligner.onsets(which, band).astype(np.float64)
        w, h = int(FOREIGN_WINDOW_S * ENVELOPE_RATE), int(FOREIGN_HOP_S * ENVELOPE_RATE)
        if len(x) < w:
            aligner._activity[key] = 0.0
        else:
            ones = np.concatenate([[0.0], np.cumsum(x)])
            squares = np.concatenate([[0.0], np.cumsum(x * x)])
            starts = np.arange(0, len(x) - w + 1, h)
            mean = (ones[starts + w] - ones[starts]) / w
            var = np.maximum((squares[starts + w] - squares[starts]) / w - mean * mean, 0.0)
            aligner._activity[key] = float(np.median(np.sqrt(var)))
    return aligner._activity[key]


def _active(aligner: "_Aligner", which: str, band: str, start: int, width: int) -> bool:
    x = aligner.onsets(which, band)[max(0, start):max(0, start) + width]
    typical = _typical_activity(aligner, which, band)
    return len(x) == width and typical > 0.0 and float(np.std(x)) >= FOREIGN_ACTIVE * typical


def _window_agreement(aligner: "_Aligner", band: str, t: float, lo_off: float, hi_off: float) -> Optional[float]:
    """The best agreement of one FOREIGN_WINDOW_S window of the video, at
    ``t``, with the dub anywhere from ``lo_off`` to ``hi_off`` (and
    FOREIGN_REACH_S either side); None where the dub has no such span."""
    rate = ENVELOPE_RATE
    w, reach = int(FOREIGN_WINDOW_S * rate), int(FOREIGN_REACH_S * rate)
    a, b = aligner.onsets("primary", band), aligner.onsets("secondary", band)
    s = int(round(t * rate))
    first = int(round((t + lo_off) * rate)) - reach
    last = int(round((t + hi_off) * rate)) + w + reach
    if s < 0 or s + w > len(a) or first < 0 or last > len(b):
        return None
    row = ncc_lags(a[s:s + w], b[first:last])
    return float(row.max()) if row.size else None


def _calibrate_foreign(aligner: "_Aligner", blocks: List[Block]) -> Optional[Dict[str, float]]:
    """Per band, the agreement below which this pair is not itself (see
    FOREIGN_LIMIT_PERCENTILE); None when too little is placed to say."""
    samples: Dict[str, List[float]] = {b: [] for b in _foreign_bands(aligner)}
    w = int(FOREIGN_WINDOW_S * ENVELOPE_RATE)
    for block in blocks:
        t = block.start_s + 1.0
        while t + FOREIGN_WINDOW_S <= block.end_s - 1.0:
            s = int(round(t * ENVELOPE_RATE))
            for band in samples:
                if band != "full" and not (_active(aligner, "primary", band, s, w) and _active(
                        aligner, "secondary", band, s + int(round(block.offset_s * ENVELOPE_RATE)), w)):
                    continue
                value = _window_agreement(aligner, band, t, block.offset_s, block.offset_s)
                if value is not None:
                    samples[band].append(value)
            t += 3 * FOREIGN_HOP_S
    limits = {}
    for band, values in samples.items():
        if len(values) < FOREIGN_CALIBRATION_MIN:
            continue
        level = FOREIGN_LIMIT_SHARE * float(np.percentile(values, FOREIGN_LIMIT_PERCENTILE))
        limits[band] = min(FOREIGN_LIMIT_CEIL, max(FOREIGN_LIMIT_FLOOR, level))
    if "full" not in limits or len(limits) < 2:
        return None
    return limits


def _foreign_runs(aligner: "_Aligner", lo_s: float, hi_s: float, lo_off: float, hi_off: float,
                  min_s: float = FOREIGN_MIN_S) -> List[Tuple[float, float]]:
    """Spans of at least ``min_s`` inside [lo_s, hi_s) where the dub,
    anywhere between the two offsets, is not the video: both carry music
    and effects there and every band agrees less than this pair ever does
    where it is placed."""
    limits = aligner.foreign_limits
    if not limits or "side" not in limits:
        return []
    rate = ENVELOPE_RATE
    w = int(FOREIGN_WINDOW_S * rate)
    mid = (lo_off + hi_off) / 2.0
    flags: List[Tuple[float, bool]] = []
    t = lo_s
    while t + FOREIGN_WINDOW_S <= hi_s + 1e-9:
        s = int(round(t * rate))
        d = int(round((t + mid) * rate))
        side = _active(aligner, "primary", "side", s, w) and _active(aligner, "secondary", "side", d, w)
        busy = side or (_active(aligner, "primary", "low", s, w) and _active(aligner, "secondary", "low", d, w))
        foreign = busy
        if busy:
            for band, limit in limits.items():
                value = _window_agreement(aligner, band, t, lo_off, hi_off)
                if value is None or value >= limit:
                    foreign = False
                    break
        flags.append((t, foreign, side))
        t += FOREIGN_HOP_S
    # A run must rest on music and effects: the stereo difference, where
    # centred dialogue does not reach, busy in both tracks across at least
    # half of it. A mono pair (or dual mono) has none, and its bass carries
    # the voices as well -- a dub that lost its bed there and kept its lines
    # reads just like another scene -- so there nothing is called foreign.
    runs: List[Tuple[float, float]] = []
    start: Optional[int] = None
    for index, (t, foreign, _side) in enumerate(flags + [(float("inf"), False, False)]):
        if foreign and start is None:
            start = index
        elif not foreign and start is not None:
            lo_t = flags[start][0]
            end = flags[index - 1][0] + FOREIGN_WINDOW_S
            sided = sum(1 for f in flags[start:index] if f[2])
            if end - lo_t >= min_s - 1e-9 and sided >= 0.5 * (index - start):
                runs.append((lo_t, min(end, hi_s)))
            start = None
    return runs


def _agreement_t(aligner: "_Aligner", offset_s: float, lo_s: float, hi_s: float) -> float:
    """How clearly the dub agrees with the video across [lo_s, hi_s) at one
    offset: the t-statistic of the agreement's FOREIGN_T_BLOCK_S means."""
    curve = aligner.agreement(offset_s, lo_s, hi_s)
    block = max(1, int(FOREIGN_T_BLOCK_S * ENVELOPE_RATE))
    means = np.array([curve[i:i + block].mean() for i in range(0, len(curve) - block + 1, block)])
    if len(means) < 4:
        return float("inf")
    return float(means.mean()) / (float(means.std(ddof=1)) / np.sqrt(len(means)) + 1e-9)


def _speech_agrees(aligner: "_Aligner", lo_s: float, hi_s: float, offset_s: float) -> Optional[bool]:
    """Whether the dub's lines follow the video's across [lo_s, hi_s) at
    ``offset_s``: True, False, or None when there is too little speech to
    say. Read on the centre of each mix (see ``speech.centre_activity``),
    at any speed."""
    primary, secondary = aligner.primary, aligner.secondary
    if not getattr(primary, "path", None) or not getattr(secondary, "path", None) or hi_s - lo_s < 5.0:
        return None
    from ..speech import SPEECH_RATE, centre_activity
    speed = secondary.speed or 1.0
    pad, reach = 1.5, 0.3
    try:
        video = centre_activity(primary.path, primary.track, lo_s, hi_s, aligner.token)
        dub_lo = (lo_s + offset_s - pad) / speed
        dub_hi = (hi_s + offset_s + pad) / speed
        dub = centre_activity(secondary.path, secondary.track, dub_lo, dub_hi, aligner.token)
    except MediaError:
        return None
    if float(np.mean(video > 0.3)) < 0.1:
        return None
    if abs(speed - 1.0) > 1e-9:
        own = (dub_lo + np.arange(len(dub)) / SPEECH_RATE) * speed
        grid = (lo_s + offset_s - pad) + np.arange(int(round((hi_s - lo_s + 2 * pad) * SPEECH_RATE))) / SPEECH_RATE
        dub = np.interp(grid, own, dub, left=0.0, right=0.0)
    row = ncc_lags(video, dub)
    if row.size < 3:
        return None
    lags = np.arange(len(row)) / SPEECH_RATE - pad
    near = np.abs(lags) <= reach
    away = np.abs(lags) > 2 * reach
    if near.sum() == 0 or away.sum() < 10:
        return None
    z = (float(row[near].max()) - float(np.median(row[away]))) / (float(np.std(row[away])) + 1e-9)
    return z >= FOREIGN_SPEECH_Z


def _split_foreign(aligner: "_Aligner", blocks: List[Block], notes: Optional[List[str]]) -> List[Block]:
    """Take out of every stretch the spans where the dub has something else
    (see ``_foreign_runs``), unless the agreement over the span or the
    lines say the scene is the same. The spans are remembered on the
    aligner, so the assembly fills them from the original rather than
    bridging them."""
    aligner.foreign_limits = _calibrate_foreign(aligner, blocks)
    if not aligner.foreign_limits:
        return blocks
    out: List[Block] = []
    for block in blocks:
        spans: List[Tuple[float, float]] = []
        for lo, hi in _foreign_runs(aligner, block.start_s, block.end_s, block.offset_s, block.offset_s):
            if _agreement_t(aligner, block.offset_s, lo, hi) >= FOREIGN_MAX_T:
                continue
            if _speech_agrees(aligner, lo, hi, block.offset_s):
                continue
            lo, hi = _foreign_edges(aligner, block.offset_s, lo, hi, block.start_s, block.end_s)
            if spans and (lo <= spans[-1][1] + FOREIGN_WINDOW_S or (
                    lo <= spans[-1][1] + 2 * FOREIGN_WINDOW_S
                    and _agreement_t(aligner, block.offset_s, spans[-1][1], lo) < FOREIGN_MAX_T)):
                # Close, and nothing between agrees either: one span, not
                # two with a few seconds of the wrong scene kept between.
                spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
            else:
                spans.append((lo, hi))
        pieces = [block]
        for lo, hi in spans:
            aligner.foreign.append((lo, hi))
            if notes is not None:
                notes.append(
                    f"the dub has something else at {_clock(lo)} - {_clock(hi)}: its music and effects agree with "
                    f"the video's nowhere near the offset there, and its lines do not follow the video's; "
                    f"the original plays"
                )
            current = pieces.pop()
            if lo - current.start_s >= MIN_BLOCK_S:
                before = copy.deepcopy(current)
                before.end_s = before.end_lo = before.end_hi = lo
                before.end_cut = None
                pieces.append(before)
            if current.end_s - hi >= MIN_BLOCK_S:
                after = copy.deepcopy(current)
                after.start_s = after.start_lo = after.start_hi = hi
                after.start_cut = None
                pieces.append(after)
            else:
                break
        out.extend(pieces)
    return out


def _foreign_edges(aligner: "_Aligner", offset_s: float, lo_s: float, hi_s: float,
                   floor_s: float, ceiling_s: float) -> Tuple[float, float]:
    """Where agreement at the offset gives way to none, near each end of a
    foreign span found on FOREIGN_HOP_S windows: the point with the most
    agreement before it and the least after (and the mirror at the end)."""
    base = max(floor_s, lo_s - FOREIGN_WINDOW_S)
    curve = aligner.agreement(offset_s, base, min(ceiling_s, hi_s + FOREIGN_WINDOW_S))
    rate = ENVELOPE_RATE
    side = int(1.5 * rate)
    cumulative = np.concatenate([[0.0], np.cumsum(curve)])

    def mean(a: int, b: int) -> float:
        a, b = max(0, a), min(len(curve), b)
        return (cumulative[b] - cumulative[a]) / max(1, b - a)

    def best(lo_t: float, hi_t: float, falling: bool) -> float:
        top, where = -np.inf, (lo_t + hi_t) / 2.0
        first = max(side, int((lo_t - base) * rate))
        last = min(len(curve) - side, int((hi_t - base) * rate))
        for i in range(first, last, max(1, int(0.05 * rate))):
            gain = mean(i - side, i) - mean(i, i + side) if falling else mean(i, i + side) - mean(i - side, i)
            if gain > top:
                top, where = gain, base + i / rate
        return where

    start = best(lo_s - FOREIGN_HOP_S, lo_s + FOREIGN_WINDOW_S, True)
    end = best(hi_s - FOREIGN_WINDOW_S, hi_s + FOREIGN_HOP_S, False)
    if end - start < FOREIGN_MIN_S / 2.0:
        return lo_s, hi_s
    return start, end


def _gap_is_foreign(aligner: "_Aligner", last: Block, block: Block) -> bool:
    """Whether the dub has something else for the gap between two
    stretches: a span already found so, or one where it agrees with the
    video at neither offset nor anywhere between (see ``_foreign_runs``)
    and its lines follow the video's at neither."""
    lo, hi = last.end_s, block.start_s
    if any(a < hi and b > lo for a, b in aligner.foreign):
        return True
    if hi - lo < FOREIGN_WINDOW_S or not aligner.foreign_limits:
        return False
    low, high = sorted((last.offset_s, block.offset_s))
    runs = _foreign_runs(aligner, lo, hi, low, high, max(FOREIGN_WINDOW_S, min(FOREIGN_MIN_S, 0.5 * (hi - lo))))
    # A gap is partly what the dub lacks -- as much as the offset falls
    # across it -- and that agrees with nothing as surely as another scene
    # does. Only what reads as something else beyond that, and plenty of
    # it, is: else the step is placed and the dub kept either side of it.
    missing = max(last.offset_s - block.offset_s, 0.0)
    unexplained = sum(b - a for a, b in runs) - missing
    if unexplained < max(FOREIGN_WINDOW_S, 0.5 * (hi - lo - missing)):
        return False
    offsets = (last.offset_s, block.offset_s)
    # Judged on the spans that read as something else, not the whole gap:
    # its ends hold a little of the neighbours' own material, which the
    # edges left there, and would lift the whole gap over the bar.
    for run_lo, run_hi in runs:
        if any(_agreement_t(aligner, offset, run_lo, run_hi) >= FOREIGN_MAX_T for offset in offsets):
            return False  # the dub is there at one side's offset: the usual bridging decides
    return not any(_speech_agrees(aligner, lo, hi, offset) for offset in offsets)
