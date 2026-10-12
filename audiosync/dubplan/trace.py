"""Following the dub through a gap between two offsets, second by second.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..media import MediaError

from .aligner import _local_standardize, _timeline_activity
from .assemble import _dub_is_silent, _dub_level
from .edges import _place_step
from .plan import _clock
from .scores import Block, _noise_units, _parabolic, ncc_lags
from .tuning import EFFECTS_DISCOUNT, ENVELOPE_RATE, LOCAL_STATS_S, MIN_BLOCK_S, NEAR_OFFSET_S, SAME_OFFSET_S, SPEECH_ARBITER_S, TRACE_BEAT_Z, TRACE_BIN_S, TRACE_CARRY_Z, TRACE_EDGE_S, TRACE_EDGE_Z, TRACE_JUMP, TRACE_LEVEL_APART_S, TRACE_MAX_LEVELS, TRACE_MAX_STEP_S, TRACE_MIN_DUB_S, TRACE_MIN_S, TRACE_NOISE_APART_S, TRACE_NOISE_SPACING_S, TRACE_NOISE_SPAN_S, TRACE_READING_HOP_S, TRACE_READING_S, TRACE_READING_Z, TRACE_RUN_Z, TRACE_SHORT_CARRY_S, TRACE_STEP_REACH_S

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aligner import _Aligner


def _trace_gaps(
    aligner: _Aligner, blocks: List[Block], secondary_s: float, notes: Optional[List[str]], warnings: List[str],
) -> List[Block]:
    """Follow the dub through every gap it has the material for (see
    ``_trace_gap``); the stretches found are put in between."""
    out = sorted(blocks, key=lambda b: b.start_s)
    index = 0
    while index + 1 < len(out):
        found = _trace_gap(aligner, out[index], out[index + 1], secondary_s, notes, warnings)
        if found:
            out[index + 1 : index + 1] = found
            index += len(found)
        index += 1
    return out


def _trace_levels(aligner: _Aligner, lo_s: float, hi_s: float, before: float, after: float) -> List[float]:
    """The offsets a gap's dub may sit at: the neighbours' two, then the
    peaks of short readings across the gap that fall between them,
    strongest first."""
    rate = ENVELOPE_RATE
    low, high = min(before, after), max(before, after)
    window = int(round(TRACE_READING_S * rate))
    margin = int(round(TRACE_NOISE_SPAN_S * rate))
    found: List[Tuple[float, float]] = []
    at = lo_s
    while at + TRACE_READING_S <= hi_s + 1e-9:
        start = int(round(at * rate))
        for band in aligner.shared_bands:
            template = aligner.onsets("primary", band)[start : start + window]
            secondary = aligner.onsets("secondary", band)
            first = max(0, start + int(round(low * rate)) - margin)
            last = min(len(secondary), start + window + int(round(high * rate)) + margin)
            if len(template) < window or last - first < window + 2:
                continue
            row = ncc_lags(template, secondary[first:last])
            units = _noise_units(row, np.ones(len(row), dtype=bool))
            offsets = (first + np.arange(len(row)) - start) / rate
            inside = np.flatnonzero((offsets >= low - SAME_OFFSET_S) & (offsets <= high + SAME_OFFSET_S))
            if inside.size == 0:
                continue
            peak = int(inside[np.argmax(units[inside])])
            z = float(units[peak]) * (EFFECTS_DISCOUNT if band == "high" else 1.0)
            if z >= TRACE_READING_Z:
                found.append((float(offsets[0] + _parabolic(row, peak) / rate), z))
        at += TRACE_READING_HOP_S
    levels = [before, after]
    for offset, _z in sorted(found, key=lambda f: -f[1]):
        if len(levels) >= TRACE_MAX_LEVELS:
            break
        if all(abs(offset - level) >= TRACE_LEVEL_APART_S for level in levels):
            levels.append(offset)
    return levels


class _TraceScores:
    """How much the two tracks agree in each bin of a gap at any offset, in
    noise units: every shared band's agreement and the two languages'
    speech timing, each measured against offsets where nothing matches."""

    def __init__(self, aligner: _Aligner, lo_s: float, hi_s: float, levels: Sequence[float]) -> None:
        self.aligner = aligner
        self.lo_s, self.hi_s = lo_s, hi_s
        self.bins = max(1, int(round((hi_s - lo_s) / TRACE_BIN_S)))
        self.edges = lo_s + (hi_s - lo_s) * np.arange(self.bins + 1) / self.bins
        low, high = min(levels), max(levels)
        noise = [
            float(offset) for offset in np.arange(low - TRACE_NOISE_SPAN_S, high + TRACE_NOISE_SPAN_S, TRACE_NOISE_SPACING_S)
            if all(abs(offset - level) >= TRACE_NOISE_APART_S for level in levels)
        ]
        self.parts: List[Tuple[str, float]] = [(band, EFFECTS_DISCOUNT if band == "high" else 1.0) for band in aligner.shared_bands]
        self.speech: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        primary, secondary = aligner.primary, aligner.secondary
        if primary.speed == 1.0:
            # The voices themselves, where the music and effects are too
            # quiet to say anything: a dub is performed to the lips, so its
            # lines start and stop about where the original's do.
            from ..speech import SPEECH_RATE, centre_activity
            try:
                reach = TRACE_NOISE_SPAN_S + 1.0
                video = centre_activity(primary.path, primary.track, lo_s, hi_s, aligner.token)
                dub_lo = lo_s + low - reach
                dub = _timeline_activity(secondary, dub_lo, hi_s + high + reach, aligner.token)
                span = int(LOCAL_STATS_S * SPEECH_RATE)
                self.speech = (_local_standardize(video, span), _local_standardize(dub, span), dub_lo)
                self.parts.append(("speech", 1.0))
            except MediaError:
                self.speech = None
        self.sigma: Dict[str, float] = {}
        for name, _weight in self.parts:
            values = np.concatenate([self._means(name, offset) for offset in noise]) if noise else np.zeros(0)
            self.sigma[name] = float(np.std(values)) if values.size > 1 else 0.0

    def _means(self, name: str, offset_s: float) -> np.ndarray:
        """The mean agreement of one part in every bin, at one offset."""
        if name == "speech":
            from ..speech import SPEECH_RATE
            video, dub, dub_lo = self.speech
            curve = np.zeros(len(video))
            begin = int(round((self.lo_s + offset_s - dub_lo) * SPEECH_RATE))
            usable = max(0, min(len(video), len(dub) - begin))
            if begin >= 0 and usable:
                curve[:usable] = video[:usable] * dub[begin : begin + usable]
            cuts = np.round((self.edges - self.lo_s) * SPEECH_RATE).astype(int)
        else:
            curve = self.aligner.agreement(offset_s, self.lo_s, self.hi_s, [name])
            cuts = np.round((self.edges - self.lo_s) * ENVELOPE_RATE).astype(int)
        cuts = np.clip(cuts, 0, len(curve))
        return np.array([float(curve[a:b].mean()) if b > a else 0.0 for a, b in zip(cuts[:-1], cuts[1:])])

    def at(self, offset_s: float) -> np.ndarray:
        """Every bin's agreement at the offset, summed over the parts in
        noise units and scaled so pure noise has a spread of one."""
        total = np.zeros(self.bins)
        weights = 0.0
        for name, weight in self.parts:
            sigma = self.sigma.get(name, 0.0)
            if sigma <= 1e-12:
                continue
            total += weight * self._means(name, offset_s) / sigma
            weights += weight * weight
        return total / math.sqrt(weights) if weights else total


def _trace_gap(
    aligner: _Aligner, left: Block, right: Block, secondary_s: float,
    notes: Optional[List[str]], warnings: List[str],
) -> Optional[List[Block]]:
    """Follow the dub through the gap between two stretches whose offsets
    differ, where the dub has the material for it.

    The neighbours say where the dub leaves off and picks up; between those
    two points is all the dub the gap can hold, and all the video it lacks
    is the difference of the offsets. A scene both edits trimmed
    differently -- a shot shorter here, one taken out there -- is that
    material at a few offsets in turn, stepping from one neighbour's to the
    other's. The coarse and gap searches need a long window to see a quiet
    scene and a short one to see a step, and a trimmed quiet scene defeats
    both; here the offsets are only a handful (the neighbours', and any a
    short reading found between them), every second of the gap is scored at
    each in every band and in the speech timing, and one path through them
    picks the level each second is at, paying for every change. Each run of
    it must stand above the noise over its length -- a neighbour's offset
    carried on a little, one of the gap's own clearly, and above the other
    candidates too -- or the gap is left as it was. Measured on a real pair:
    a 65-second gap whose dub was 61.6 s of the same scene went on at the
    first offset for 49 s, took out a 1.84-second shot the picture shows,
    ran 11 s at the second and stepped 1.67 s to the third.
    """
    lo_s, hi_s = left.end_s, right.start_s
    gap = hi_s - lo_s
    before, after = left.offset_s, right.offset_s
    step = before - after          # > 0: the video has this much the dub lacks
    if gap < TRACE_MIN_S or not NEAR_OFFSET_S < abs(step) <= TRACE_MAX_STEP_S:
        return None
    if gap - max(step, 0.0) < TRACE_MIN_DUB_S or lo_s + before < 0.0 or hi_s + after > secondary_s:
        return None
    if _dub_is_silent(aligner.secondary, lo_s + before, hi_s + after, _dub_level(aligner.secondary, [left, right])):
        return None

    levels = _trace_levels(aligner, lo_s, hi_s, before, after)
    low, high = min(before, after), max(before, after)
    middle = sorted((level for level in levels[2:] if low < level < high), reverse=step > 0)
    order = [before] + middle + [after]
    scores = _TraceScores(aligner, lo_s, hi_s, order)
    table = np.array([scores.at(level) for level in order])
    count, bins = table.shape

    # The path: it starts at the offset before the gap and ends at the one
    # after (anything else costs a change), and moves only from the one
    # towards the other -- the dub is not played twice.
    best = table[:, 0] - np.where(np.arange(count) == 0, 0.0, TRACE_JUMP)
    back = np.zeros((bins, count), dtype=int)
    for column in range(1, bins):
        new = np.empty(count)
        for state in range(count):
            source, value = state, best[state]
            for earlier in range(state):
                if best[earlier] - TRACE_JUMP > value:
                    source, value = earlier, best[earlier] - TRACE_JUMP
            new[state] = value + table[state, column]
            back[column, state] = source
        best = new
    state = int(np.argmax(best - np.where(np.arange(count) == count - 1, 0.0, TRACE_JUMP)))
    path = [state]
    for column in range(bins - 1, 0, -1):
        state = int(back[column, state])
        path.append(state)
    path.reverse()

    runs: List[Tuple[int, int, int]] = []          # (state, first bin, last bin)
    for column, state in enumerate(path):
        if runs and runs[-1][0] == state:
            runs[-1] = (state, runs[-1][1], column)
        else:
            runs.append((state, column, column))
    judged: List[str] = []
    edge = max(1, int(round(TRACE_EDGE_S / TRACE_BIN_S)))
    for position, (state, first, last) in enumerate(runs):
        size = last - first + 1
        own = float(table[state, first : last + 1].sum()) / math.sqrt(size)
        rivals = [float(table[other, first : last + 1].sum()) / math.sqrt(size) for other in range(count) if other != state]
        carried = (state == 0 and first == 0) or (state == count - 1 and last == bins - 1)
        if carried and size * TRACE_BIN_S < TRACE_SHORT_CARRY_S:
            judged.append(f"{own:.1f}")
            continue
        if carried:
            ok = own >= TRACE_CARRY_Z
        else:
            ok = own >= TRACE_RUN_Z and own - max(rivals, default=-np.inf) >= TRACE_BEAT_Z
        # A run whose support is all at one end says nothing about the step
        # at the other -- nor that the dub there is this scene at all: the
        # dub's own material the length of what it lacks, in its place,
        # would carry a neighbour's offset on its far end alone.
        ends = []
        if position > 0 or order[state] != before:
            ends.append(table[state, first : first + edge])
        if position < len(runs) - 1 or order[state] != after:
            ends.append(table[state, max(first, last + 1 - edge) : last + 1])
        if any(float(chunk.sum()) / math.sqrt(len(chunk)) < TRACE_EDGE_Z for chunk in ends):
            ok = False
        if not ok:
            aligner.log(
                f"  {_clock(lo_s)} - {_clock(hi_s)}: the dub at {order[state]:+.3f}s over "
                f"{_clock(scores.edges[first])} - {_clock(scores.edges[last + 1])} is not clear ({own:.1f} noise units); left as it was"
            )
            return None
        judged.append(f"{own:.1f}")

    # Every change of offset is put exactly where it happens: on the
    # picture's cut when there is one to put it on, else where the
    # agreement changes, else where the path changed. A neighbour's edge
    # was placed on long windows that could not see the gap's steps, so a
    # step next to it may move it, as long as the neighbour keeps a length.
    edges = scores.edges
    pieces: List[List[float]] = [[order[state], edges[first], edges[last + 1]] for state, first, last in runs]
    pieces[0][1] = lo_s
    pieces[-1][2] = hi_s
    # The neighbours are pieces too, empty where the path left them at the
    # gap's edge, so the step into and out of the gap is placed on the
    # evidence like every other rather than where the neighbour's own edge
    # happened to be left.
    if order[runs[0][0]] != before:
        pieces.insert(0, [before, lo_s, lo_s])
    if order[runs[-1][0]] != after:
        pieces.append([after, hi_s, hi_s])
    floor = max(left.start_s + MIN_BLOCK_S, lo_s - TRACE_STEP_REACH_S)
    ceiling = min(right.end_s - MIN_BLOCK_S, hi_s + TRACE_STEP_REACH_S)
    guessed: List[float] = []
    cuts: Dict[int, Tuple[float, float]] = {}   # step -> (where the one ends, where the next starts)
    for index in range(len(pieces) - 1):
        a, b = pieces[index][0], pieces[index + 1][0]
        hole = max(0.0, a - b)
        at = pieces[index + 1][1]
        lower = floor if index == 0 else pieces[index][1]
        upper = ceiling if index + 1 == len(pieces) - 1 else pieces[index + 1][2]
        # The path knows nothing of the frames the dub lacks at a step, so
        # the span reaches that much further on.
        span_lo = max(lower, at - TRACE_STEP_REACH_S)
        span_hi = min(upper, at + TRACE_STEP_REACH_S + hole)
        if span_hi - span_lo <= hole:
            aligner.log(f"  {_clock(lo_s)} - {_clock(hi_s)}: no room for the step near {_clock(at)}; left as it was")
            return None
        point, how = _place_step(aligner, span_lo, span_hi, a, b, guess_on_sound=True)
        if "high" in aligner.shared_bands:
            # The effects first, as everywhere (see BANDS): where the music
            # ran on across a trim, every band summed puts the step between
            # where the music moved and where the picture did. Measured on a
            # real pair: the effects moved 0.8 s at 1:00:58 at ten noise
            # units a second, the bass four seconds later, and the sum put
            # the step at 1:01:02. Unless the dialogue, between the two
            # answers, is at the level the effects have already left.
            said, told = _place_step(aligner, span_lo, span_hi, a, b, bands=["high"])
            if told != "guess" and abs(said - point) > 1.0:
                between = (min(said, point), max(said, point))
                if not (
                    abs(a - b) >= SPEECH_ARBITER_S
                    and aligner.speech_choice(between[0], between[1] + max(0.0, a - b), a, b) == (a if said < point else b)
                ):
                    point, how = said, told
            elif told != "guess":
                point, how = said, told
        if how == "guess":
            point = min(max(point, span_lo), span_hi - hole)
            guessed.append(point)
        pieces[index][2] = point
        pieces[index + 1][1] = point + hole
        if how == "picture":
            cuts[index] = (point, point + hole)
    inner = pieces[1:-1]
    if any(end - start < MIN_BLOCK_S for _offset, start, end in inner) or (
        pieces[0][2] - left.start_s < MIN_BLOCK_S or right.end_s - pieces[-1][1] < MIN_BLOCK_S
    ) or any(pieces[index][2] > pieces[index + 1][1] + 1e-9 for index in range(len(pieces) - 1)):
        aligner.log(f"  {_clock(lo_s)} - {_clock(hi_s)}: the steps the dub takes there leave a piece too short; left as it was")
        return None

    left.end_s = left.end_lo = left.end_hi = pieces[0][2]
    left.end_cut = cuts[0][0] if 0 in cuts else None
    right.start_s = right.start_lo = right.start_hi = pieces[-1][1]
    last_step = len(pieces) - 2
    right.start_cut = cuts[last_step][1] if last_step in cuts else None
    found: List[Block] = []
    for offset, start, end in inner:
        block = Block(start, end, offset, start_lo=start, start_hi=start, end_lo=end, end_hi=end)
        block.match = aligner.sharp_match(block)
        block.support = float(np.mean(table[order.index(offset)]))
        found.append(block)

    steps = " then ".join(
        f"{_clock(max(start, lo_s))} - {_clock(min(end, hi_s))} at {offset:+.3f}s"
        for offset, start, end in pieces if min(end, hi_s) - max(start, lo_s) > 0.01
    )
    what = "the music and effects" + (" and the speech timing" if scores.speech is not None else "")
    if notes is not None:
        notes.append(
            f"followed the dub through {_clock(lo_s)} - {_clock(hi_s)}, where it did not correlate in long windows: "
            f"{steps} (by {what}; {', '.join(judged)} noise units)"
        )
    aligner.log(f"  followed the dub through {_clock(lo_s)} - {_clock(hi_s)}: {steps}")
    for point in guessed:
        warnings.append(
            f"the dub steps near {_clock(point)}: {what} say so, but neither the sound nor the picture "
            "clearly says where to the frame -- check the lips there"
        )
    return found
