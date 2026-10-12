"""Correlation in units of noise, the score grid, and the path through it.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..correlate import _fast_fft_size
from ..media import CancellationToken

from .tuning import BLOCK_BAR_MARGIN, COARSE_HOP_S, COARSE_WINDOW_S, FROM_NULL, JUMPED, JUMP_COST, MATCH_Z, MIN_BLOCK_Z, NULL_COST, NULL_REWARD, SHORT_RUN_WINDOWS, STAY, STAY_LEFT, STAY_RIGHT, WIDE_ANCHOR_REACH_S, WIDE_EPISODE_S, WIDE_LOCAL_S, WIDE_REWIND_SLACK_S, WOBBLE_COST


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def ncc_lags(template: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``template`` against every equal span of ``signal``.

    Returns ``len(signal) - len(template) + 1`` values, one per start
    position. Every span of the signal is normalised on its own, so a loud
    passage cannot win by being loud and a silent one scores zero rather than
    dividing by nothing. The numerator is an FFT correlation; the
    denominators come from running sums, so the whole thing costs one
    transform however many positions are tried.
    """
    template = np.asarray(template, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    width, length = len(template), len(signal)
    if width == 0 or length < width:
        return np.zeros(0, dtype=np.float64)

    centred = template - template.mean()
    template_norm = float(np.sqrt(np.sum(centred * centred)))
    positions = length - width + 1
    if template_norm < 1e-9:
        return np.zeros(positions, dtype=np.float64)

    size = _fast_fft_size(length)
    spectrum = np.fft.rfft(signal, size) * np.conj(np.fft.rfft(centred, size))
    numerator = np.fft.irfft(spectrum, size)[:positions]

    ones = np.concatenate([[0.0], np.cumsum(signal)])
    squares = np.concatenate([[0.0], np.cumsum(signal * signal)])
    sums = ones[width:] - ones[:-width]
    sums_sq = squares[width:] - squares[:-width]
    variance = np.maximum(sums_sq - sums * sums / width, 0.0)
    denominator = template_norm * np.sqrt(variance)

    out = np.zeros(positions, dtype=np.float64)
    ok = denominator > 1e-9
    out[ok] = numerator[ok] / denominator[ok]
    return np.clip(out, -1.0, 1.0)


def _parabolic(values: np.ndarray, index: int) -> float:
    """Sub-sample peak position from the three values around a maximum."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, centre, right = float(values[index - 1]), float(values[index]), float(values[index + 1])
    denom = left - 2.0 * centre + right
    if abs(denom) < 1e-12:
        return float(index)
    shift = 0.5 * (left - right) / denom
    if not math.isfinite(shift) or abs(shift) > 1.0:
        return float(index)
    return index + shift


def _nearest_peak(row: np.ndarray, centre: int, share: float = 0.85, apart: int = 50) -> int:
    """The index of the peak nearest ``centre`` among the local maxima that
    reach ``share`` of the tallest and sit at least ``apart`` samples from
    it. A rhythmic passage has a peak every beat, all about as tall, and
    the tallest is not the right one -- the one nearest where the stretch
    already sits is. A bump on the tallest peak's own shoulder is not
    another peak, hence ``apart`` (100 ms at 2 ms frames)."""
    if len(row) < 3:
        return int(np.argmax(row))
    tallest = int(np.argmax(row))
    top = float(row[tallest])
    inner = row[1:-1]
    maxima = np.flatnonzero((inner >= share * top) & (inner >= row[:-2]) & (inner >= row[2:])) + 1
    maxima = maxima[np.abs(maxima - tallest) >= apart]
    if maxima.size == 0:
        return tallest
    candidates = np.concatenate([[tallest], maxima])
    return int(candidates[np.argmin(np.abs(candidates - centre))])


def _tallest_peaks(row: np.ndarray, count: int, apart: int) -> List[int]:
    """Indices of the ``count`` tallest values of ``row`` that are at least
    ``apart`` samples from each other, tallest first."""
    out: List[int] = []
    taken = np.zeros(len(row), dtype=bool)
    work = np.array(row, dtype=np.float64)
    for _ in range(count):
        work[taken] = -np.inf
        peak = int(np.argmax(work))
        if not np.isfinite(work[peak]):
            break
        out.append(peak)
        taken[max(0, peak - apart) : peak + apart + 1] = True
    return out


def _continuing_candidate(
    candidates: Sequence[Tuple[float, float]],
    position_s: float,
    anchors: Sequence[Block],
    previous: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Which of a window's credible peaks to keep (see ``_wide_blocks``)."""
    if len(candidates) == 1:
        return candidates[0]
    nearby = [
        b for b in anchors
        if b.start_s - WIDE_ANCHOR_REACH_S <= position_s <= b.end_s + WIDE_ANCHOR_REACH_S
    ]
    if nearby:
        anchor = min(nearby, key=lambda b: max(0.0, b.start_s - position_s, position_s - b.end_s))
        local = [c for c in candidates if abs(c[0] - anchor.offset_s) <= WIDE_LOCAL_S]
        if local:
            return max(local, key=lambda c: c[1])
    if previous is not None:
        prev_position, prev_offset = previous
        # Where the dub was at the previous window's end; a candidate that
        # winds back from there by less than an episode is a repeat.
        floor = prev_position + prev_offset + (position_s - prev_position) - WIDE_REWIND_SLACK_S
        forward = [c for c in candidates if position_s + c[0] >= floor or position_s + c[0] < floor - WIDE_EPISODE_S]
        if forward:
            return max(forward, key=lambda c: c[1])
    return max(candidates, key=lambda c: c[1])


def _peak_dominance(row: np.ndarray, peak: int, apart: int = 50) -> float:
    """How many times taller the peak is than the tallest other value more
    than ``apart`` samples from it. A match stands alone; a chance
    coincidence in a sparse band has company of its own size."""
    if len(row) < 3:
        return 0.0
    away = np.abs(np.arange(len(row)) - peak) > apart
    if not away.any():
        return float("inf")
    other = float(np.max(row[away]))
    return float(row[peak]) / max(other, 1e-9)


def _noise_units(row: np.ndarray, possible: np.ndarray) -> np.ndarray:
    """Express one window's correlations as multiples of its noise floor.

    The floor is the root mean square over every offset that could be
    tried, less the tallest one percent (and always the tallest one): the
    peak being measured must not be part of its own floor, which on a
    short row it would be. For a row of dense onsets that is the standard
    deviation the median magnitude used to estimate; for a row of sparse
    ones -- a band with a few transients in it, a quiet scene -- the
    median sits near zero and made a chance coincidence of two or three
    spikes look ten floors tall, while the RMS still counts the spikes.
    Music with a repeating structure raises it, and that is the right
    direction, since every repeat is a chance for a coincidence.
    """
    out = np.full(len(row), -1e6, dtype=np.float32)
    if not possible.any():
        return out
    values = row[possible]
    magnitudes = np.abs(values).astype(np.float64)
    if len(magnitudes) > 2:
        drop = max(1, len(magnitudes) // 100)
        magnitudes = np.partition(magnitudes, len(magnitudes) - drop)[: len(magnitudes) - drop]
    sigma = float(np.sqrt(np.mean(magnitudes ** 2)))
    if sigma < 1e-6:
        out[possible] = 0.0
    else:
        out[possible] = (values / sigma).astype(np.float32)
    return out


@dataclass
class ScoreGrid:
    """Correlation of every window against every candidate offset."""

    rate: float
    """Samples per second of the pooled envelopes the grid was built on."""
    starts: np.ndarray
    """Window start, in pooled samples of the primary."""
    window: int
    """Window length in pooled samples."""
    lag_lo: int
    """Offset of column 0, in pooled samples (dub = primary + lag)."""
    scores: np.ndarray
    """Windows x lags, in noise units; hugely negative where impossible."""

    def offset_s(self, column: int) -> float:
        return (self.lag_lo + column) / self.rate

    def start_s(self, index: int) -> float:
        return float(self.starts[index]) / self.rate

    @property
    def window_s(self) -> float:
        return self.window / self.rate


def score_windows(
    primary: np.ndarray,
    secondary: np.ndarray,
    rate: float,
    starts: Sequence[int],
    window: int,
    lag_lo: int,
    lag_hi: int,
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> ScoreGrid:
    """Score every window of the primary at every offset in [lag_lo, lag_hi].

    Offsets that would place any part of the window outside the secondary
    are impossible, not merely unmeasured, and are marked so the path cannot
    rest on them: a stretch of video past the end of the dub must be a gap.
    """
    lags = lag_hi - lag_lo + 1
    grid = np.full((len(starts), lags), -1e6, dtype=np.float32)
    # A window may hang a tenth of its length past either end of the
    # secondary and still be measured, on what it does cover: otherwise the
    # last window before the dub's end is impossible at the true offset by
    # a rounding error, and the stretch it belongs to loses its last edge.
    pad = window // 10
    padded = np.concatenate([np.zeros(pad), np.asarray(secondary, dtype=np.float64), np.zeros(pad)])
    for index, start in enumerate(starts):
        if token:
            token.raise_if_cancelled()
        template = primary[start : start + window]
        if len(template) < window:
            continue
        first = max(0, start + lag_lo + pad)
        last = min(len(padded), start + window + lag_hi + pad)
        if last - first < window:
            continue
        row = ncc_lags(template, padded[first:last])
        column = (first - pad) - start - lag_lo
        possible = np.zeros(lags, dtype=bool)
        possible[column : column + len(row)] = True
        full = np.zeros(lags, dtype=np.float64)
        full[column : column + len(row)] = row
        grid[index] = _noise_units(full, possible)
        if progress:
            progress((index + 1) / len(starts))
    return ScoreGrid(rate, np.asarray(starts), window, lag_lo, grid)


def best_path(
    scores: np.ndarray,
    jump_cost: float = JUMP_COST,
    null_reward: float = NULL_REWARD,
    null_cost: float = NULL_COST,
    wobble_cost: float = WOBBLE_COST,
    first: Optional[np.ndarray] = None,
    last: Optional[np.ndarray] = None,
) -> Tuple[List[Optional[int]], List[bool]]:
    """The offset at every window that best explains the whole matrix.

    Dynamic programming over (window, offset), plus one extra state per
    window meaning "no offset applies". Staying put is free; moving one
    column costs a little; any larger move costs ``jump_cost``; the null
    state earns ``null_reward`` a window and costs ``null_cost`` to enter or
    leave. Exact, not greedy: every window's answer accounts for every other
    window.

    Args:
        first, last: optional bonuses added to the first and last windows,
            for a pass that has to begin and end at known offsets.

    Returns:
        The column chosen at each window (None for the null state), and
        whether each window began a new stretch -- reached by a jump, from
        the null state, or as the first window.
    """
    count, lags = scores.shape
    if count == 0:
        return [], []
    back = np.zeros((count, lags), dtype=np.uint8)
    back_null = np.zeros(count, dtype=np.uint8)
    best_previous = np.zeros(count, dtype=np.int64)

    value = scores[0].astype(np.float64)
    if first is not None:
        value = value + first
    if last is not None and count == 1:
        value = value + last
    value_null = null_reward

    for index in range(1, count):
        argmax = int(np.argmax(value))
        best = float(value[argmax])
        best_previous[index] = argmax

        stay = value.copy()
        code = np.full(lags, STAY, dtype=np.uint8)
        if lags > 1:
            from_left = np.empty(lags)
            from_left[0] = -np.inf
            from_left[1:] = value[:-1] - wobble_cost
            better = from_left > stay
            stay[better] = from_left[better]
            code[better] = STAY_LEFT
            from_right = np.empty(lags)
            from_right[-1] = -np.inf
            from_right[:-1] = value[1:] - wobble_cost
            better = from_right > stay
            stay[better] = from_right[better]
            code[better] = STAY_RIGHT

        jumped = best - jump_cost
        from_null = value_null - null_cost
        if from_null > jumped:
            move, move_code = from_null, FROM_NULL
        else:
            move, move_code = jumped, JUMPED
        better = move > stay
        stay[better] = move
        code[better] = move_code

        new_value = scores[index] + stay
        if index == count - 1 and last is not None:
            new_value = new_value + last

        entered = best - null_cost
        if entered > value_null:
            value_null = null_reward + entered
            back_null[index] = 1
        else:
            value_null = null_reward + value_null
            back_null[index] = 0

        value = new_value
        back[index] = code

    state: Optional[int] = None if value_null > float(value.max()) else int(np.argmax(value))
    path: List[Optional[int]] = [None] * count
    starts = [False] * count
    for index in range(count - 1, -1, -1):
        path[index] = state
        if index == 0:
            starts[0] = state is not None
            break
        if state is None:
            state = best_previous[index] if back_null[index] else None
            continue
        code = back[index, state]
        if code == STAY_LEFT:
            state -= 1
        elif code == STAY_RIGHT:
            state += 1
        elif code == JUMPED:
            starts[index] = True
            state = int(best_previous[index])
        elif code == FROM_NULL:
            starts[index] = True
            state = None
    return path, starts


@dataclass
class Block:
    """One stretch of video that one stretch of dub belongs to."""

    start_s: float
    end_s: float
    offset_s: float
    """dub time = video time + offset, on the dub's decoded timeline."""
    match: float = 0.0
    """Correlation of the two envelopes across the stretch, 0-1."""
    support: float = 0.0
    """Mean score of the windows that put it here, in noise units."""
    windows: int = 0
    start_lo: float = 0.0
    start_hi: float = 0.0
    end_lo: float = 0.0
    end_hi: float = 0.0
    """Where each edge may lie, before it is placed exactly; afterwards, the
    placed edge plus and minus its uncertainty."""
    consistency: float = 0.0
    """Fraction of the sub-window readings that agree with the offset to
    within CONSISTENT_MS."""
    readings: int = 0
    spread_ms: float = float("inf")
    """Range of the readings' offsets."""
    effects: bool = False
    """Whether the offset was read from the high band (see BANDS)."""
    follows_drift: bool = False
    """One step of a staircase that follows a drifting offset (see
    ``_follow_drift``); never merged back into its neighbours."""
    start_cut: Optional[float] = None
    end_cut: Optional[float] = None
    """The picture cut an edge was put on (see ``_snap_to_picture``). A
    later pass that fits the edge again on the sound moves it within its
    bracket; the picture's frame stands, so the edge is put back."""
    trace: List[Tuple[float, float]] = field(default_factory=list)
    """(window start, offset) for every coarse window with evidence for
    this stretch: the path's own record of where the offset sat, which is
    what a speed difference is read from."""

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def start_uncertainty_s(self) -> float:
        return (self.start_hi - self.start_lo) / 2.0

    @property
    def end_uncertainty_s(self) -> float:
        return (self.end_hi - self.end_lo) / 2.0


def _block_bar(windows: int, lags: int, overlap: float = COARSE_WINDOW_S / COARSE_HOP_S) -> float:
    """Mean score a run of this many windows must reach to be believed.

    The path is free to pick any of ``lags`` offsets, and to wander a step
    between windows, so a run of k windows at a random offset is not
    typical noise but the best of a great many tries. Its mean reaches
    about sqrt(2 ln(lags * 3^k) / k) -- were the windows independent.
    They are not: at a ten-second hop each thirty-second window shares two
    thirds of its material with the next, so k windows carry a third as
    many independent readings, and that is the k the bound is taken at.
    Measured on a two-hour synthetic pair, a run of seven coasting windows
    reached 3.45; this puts its bar at 4.4, and a real stretch's windows
    average five or more.
    """
    if windows <= 0:
        return float("inf")
    if windows <= SHORT_RUN_WINDOWS:
        # A run this short has to match outright. A real stretch this
        # short is found by the gap search instead, which looks for it
        # where the neighbours say it must be and cannot be fooled the
        # same way.
        return MATCH_Z
    effective = max(1.0, windows / max(1.0, overlap))
    tries = math.log(max(2, lags)) + effective * math.log(3.0)
    expected_max = math.sqrt(2.0 * tries / effective)
    return max(MIN_BLOCK_Z, BLOCK_BAR_MARGIN * expected_max)


def blocks_from_path(grid: ScoreGrid, path: List[Optional[int]], starts: List[bool]) -> List[Block]:
    """Group the path into stretches, one per run of windows at one offset."""
    blocks: List[Block] = []
    run: List[int] = []

    def close() -> None:
        if not run:
            return
        columns = np.array([path[i] for i in run], dtype=np.float64)
        hop_s = grid.start_s(1) - grid.start_s(0) if len(grid.starts) > 1 else grid.window_s
        # Staying at an offset is free, so past a cut the path coasts on
        # windows with no evidence at all until the next offset's evidence
        # begins. The stretch really ends somewhere in the last window that
        # had evidence for it, and begins somewhere in the first; the edges
        # are bracketed by those windows, not by where the path turned.
        #
        # Two grades of evidence, because a coasting window reaches two
        # noise units by chance every forty windows or so, and a bracket
        # that trusted one of those would exclude the real cut. A window
        # that matches outright must hold a good share of the stretch, so
        # the cut cannot be before it starts; a window with a little
        # evidence might hold a sliver of it, so the cut might be as late
        # as its end.
        weak = [i for i in run if grid.scores[i, path[i]] >= MIN_BLOCK_Z] or [run[0], run[-1]]
        strong = [i for i in run if grid.scores[i, path[i]] >= MATCH_Z] or weak
        # The stretch's support is the mean over the windows that matched
        # outright, and its window count theirs: the windows the path
        # coasted over carry no evidence either way, and averaged in they
        # hid a stretch two windows of eight and five noise units had
        # found. The bar a run must clear already scales with its length.
        support = float(np.mean([grid.scores[i, path[i]] for i in strong]))
        # The stretch itself is taken as the span of its strong windows:
        # what it is measured on. Its true edges lie further out, inside
        # the brackets, and are placed there afterwards.
        blocks.append(Block(
            start_s=grid.start_s(strong[0]),
            end_s=grid.start_s(strong[-1]) + grid.window_s,
            offset_s=grid.offset_s(float(np.median(columns))),
            support=support,
            windows=len(strong),
            start_lo=max(0.0, grid.start_s(weak[0]) - hop_s),
            start_hi=grid.start_s(strong[0]) + grid.window_s,
            end_lo=grid.start_s(strong[-1]),
            end_hi=grid.start_s(weak[-1]) + grid.window_s + hop_s,
            trace=[(grid.start_s(i), grid.offset_s(path[i])) for i in weak],
        ))
        run.clear()

    for index, column in enumerate(path):
        if column is None:
            close()
            continue
        if starts[index] and run:
            close()
        run.append(index)
    close()
    return blocks
