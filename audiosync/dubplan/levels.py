"""Offsets inside a stretch: drift, two levels, music against effects.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .edges import _place_edge, _place_step
from .plan import _clock
from .scores import Block, _parabolic
from .trace import _TraceScores
from .tuning import BAND_WORDS, CONSISTENT_MS, DRIFT_CORNER_REACH_S, DRIFT_FIT_GAIN, DRIFT_GROUP_GAP_S, DRIFT_MAX_RUN, DRIFT_MAX_SCATTER_MS, DRIFT_OVER_SCATTER, DRIFT_READING_SPACING_S, DRIFT_READING_WINDOW_S, DRIFT_SPAN_MS, DRIFT_STEP_MS, ENVELOPE_RATE, FOLLOW_EFFECTS_GAP_S, FOLLOW_EFFECTS_STEP_S, MIN_BLOCK_S, NEAR_OFFSET_S, POLISH_RANGE_S, POLISH_WINDOW_S, READING_Z, REUSE_MIN_S, SAME_OFFSET_S, SPEECH_ARBITER_S, SPLIT_APART_S, SPLIT_MIN_RUN_S, SPLIT_MIN_S, SPLIT_REACH_S, SPLIT_ROUNDS, SPLIT_SECOND_SHARE, SPLIT_SECOND_Z, TRACE_BEAT_Z, TRACE_BIN_S, TRACE_JUMP, TRACE_RUN_Z, TRACE_STEP_REACH_S, WIDE_EPISODE_S

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aligner import _Aligner


def _theil_sen(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Slope and intercept of the median line through the points."""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else 0.0
    return slope, float(np.median(y - slope * x))


def _evidence_note(evidence: Sequence[Tuple[str, float]]) -> Optional[str]:
    """What the sync rests on, from the dense readings: which band read
    each window best, and how strongly the tracks agreed."""
    if not evidence:
        return None
    found = [(band, z) for band, z in evidence if z >= READING_Z]
    if not found:
        return f"the two tracks shared too little to read in {len(evidence)} dense readings"
    counts: Dict[str, int] = {}
    for band, _z in found:
        counts[band] = counts.get(band, 0) + 1
    parts = ", ".join(
        f"{BAND_WORDS.get(band, band)} {100.0 * n / len(found):.0f}%"
        for band, n in sorted(counts.items(), key=lambda item: -item[1])
    )
    strength = float(np.median([z for _band, z in found]))
    return (
        f"the tracks share their music and effects: {len(found)} of {len(evidence)} readings every "
        f"{DRIFT_READING_SPACING_S:.0f}s found them (read on {parts}), typically {strength:.0f} noise units strong"
    )


def _drift_groups(blocks: List[Block]) -> List[List[Block]]:
    """Runs of stretches a few frames apart in offset and at most
    DRIFT_GROUP_GAP_S apart in time: where a drift was cut into flat pieces
    before it was recognised -- at steps the readings took for trims, around
    passages that did not correlate -- the pieces are one ramp and are read
    as one."""
    groups: List[List[Block]] = []
    for block in blocks:
        if (
            groups
            and block.start_s - groups[-1][-1].end_s <= DRIFT_GROUP_GAP_S
            and abs(block.offset_s - groups[-1][-1].offset_s) <= NEAR_OFFSET_S
        ):
            groups[-1].append(block)
        else:
            groups.append([block])
    return groups


def _follow_drift(aligner: _Aligner, blocks: List[Block], notes: Optional[List[str]] = None) -> List[Block]:
    """Replace every run of stretches whose offset walks by a staircase that
    follows it.

    Stretches a few frames apart and close together in time are read
    together (see ``_drift_groups``), every DRIFT_READING_SPACING_S, each
    window searched around the offset of the stretch it falls in. Within
    each group the longest run of stretches whose readings lie on one line
    -- walking DRIFT_SPAN_MS or more, clearly above their scatter about it,
    fitting clearly better than one offset, and scattering no more than
    DRIFT_MAX_SCATTER_MS -- is a drift, and the stretches either side are
    looked at the same way. A run of flat pieces a trim or two apart is not
    a ramp: the readings near each step sit far off any one line.

    A drift is followed from its corners -- where its line meets the level
    of the stretch beside it, across the passage that did not correlate
    between them if there is one -- in pieces short enough that the line
    moves at most DRIFT_STEP_MS across each, each playing at the line's
    offset at its middle. The cuts go on the picture's shot changes where
    one is near -- a few milliseconds' jump at a shot change is inaudible --
    and are placed exactly, so nothing trims them back.
    """
    rate = ENVELOPE_RATE
    window = int(DRIFT_READING_WINDOW_S * rate)
    margin = int(round(POLISH_RANGE_S * rate))
    out: List[Block] = []

    def fit(xs: np.ndarray, ys: np.ndarray, length: float) -> Optional[Tuple[float, float, float]]:
        if len(xs) < 5 or xs[-1] - xs[0] < 0.6 * length:
            return None
        if len(xs) > 200:
            pick = np.linspace(0, len(xs) - 1, 200).astype(int)
            xs, ys = xs[pick], ys[pick]
        slope, intercept = _theil_sen(xs, ys)
        around_line = 1.4826 * float(np.median(np.abs(ys - (intercept + slope * xs))))
        around_one = 1.4826 * float(np.median(np.abs(ys - np.median(ys))))
        walk = abs(slope) * length
        if (
            walk < DRIFT_SPAN_MS or walk < DRIFT_OVER_SCATTER * around_line
            or around_one < DRIFT_FIT_GAIN * max(around_line, 0.1) or around_line > DRIFT_MAX_SCATTER_MS
        ):
            return None
        return slope, intercept, walk

    def staircase(run: List[Block], slope: float, intercept: float, walk: float, begin: float, finish: float) -> List[Block]:
        first, last = run[0], run[-1]
        length = finish - begin
        step_s = max(MIN_BLOCK_S, DRIFT_STEP_MS / abs(slope))
        count = max(2, int(math.ceil(length / step_s)))
        ideal = [begin + length * k / count for k in range(1, count)]
        cuts = aligner.picture.within(begin, finish) if aligner.picture is not None else None
        points: List[float] = []
        for point in ideal:
            if cuts:
                near = min(cuts, key=lambda c: abs(c - point))
                if abs(near - point) <= step_s / 2.0:
                    point = near
            if (not points or point - points[-1] >= 0.5) and finish - point >= 0.5 and point - begin >= 0.5:
                points.append(point)
        edges = [begin] + points + [finish]
        pieces = []
        for lo, hi in zip(edges, edges[1:]):
            middle = (lo + hi) / 2.0
            inside = next((b for b in run if b.start_s <= middle < b.end_s), min(run, key=lambda b: abs((b.start_s + b.end_s) / 2.0 - middle)))
            pieces.append(Block(
                lo, hi, (intercept + slope * middle) / 1000.0,
                match=inside.match, support=inside.support, windows=inside.windows,
                start_lo=lo, start_hi=lo, end_lo=hi, end_hi=hi, readings=inside.readings,
                consistency=inside.consistency, spread_ms=inside.spread_ms, effects=inside.effects,
                follows_drift=True,
            ))
        if begin == first.start_s:
            pieces[0].start_lo, pieces[0].start_hi, pieces[0].start_cut = first.start_lo, first.start_hi, first.start_cut
        if finish == last.end_s:
            pieces[-1].end_lo, pieces[-1].end_hi, pieces[-1].end_cut = last.end_lo, last.end_hi, last.end_cut
        said = (
            f"the dub drifts by {slope:+.2f} ms/s across {_clock(begin)} - {_clock(finish)} "
            f"({walk:.0f} ms end to end); followed in {len(pieces)} steps of {DRIFT_STEP_MS:.0f} ms or less"
        )
        aligner.log(f"  {said}")
        if notes is not None:
            notes.append(said)
        return pieces

    for group in _drift_groups(blocks):
        # Every reading of the group, once, labelled with its stretch.
        xs: List[float] = []
        ys: List[float] = []
        owner: List[int] = []
        for index, block in enumerate(group):
            start = block.start_s
            while start + DRIFT_READING_WINDOW_S <= block.end_s + 1e-9:
                reading = aligner._read(int(round(start * rate)), window, int(round(block.offset_s * rate)), margin)
                if reading is not None:
                    aligner.evidence.append((reading[2], reading[3]))
                if reading is not None and reading[3] >= READING_Z:
                    xs.append(start + DRIFT_READING_WINDOW_S / 2.0)
                    ys.append(reading[0] * 1000.0)
                    owner.append(index)
                start += DRIFT_READING_SPACING_S
        x, y, who = np.asarray(xs), np.asarray(ys), np.asarray(owner, dtype=int)

        def settle(lo: int, hi: int) -> List[Block]:
            """Stretches group[lo:hi], with the longest ramp among them followed."""
            best: Optional[Tuple[float, int, int, Tuple[float, float, float]]] = None
            for i in range(lo, hi):
                for j in range(i, min(hi, i + DRIFT_MAX_RUN)):
                    length = group[j].end_s - group[i].start_s
                    if length < 2 * DRIFT_READING_WINDOW_S or (best is not None and length <= best[0]):
                        continue
                    chosen = (who >= i) & (who <= j)
                    found = fit(x[chosen], y[chosen], length)
                    if found is not None:
                        best = (length, i, j, found)
            if best is None:
                return list(group[lo:hi])
            _length, i, j, (slope, intercept, walk) = best
            # Where the drift really begins and ends: where its line meets
            # the level of the flat stretch beside it. Between them lies, as
            # a rule, a passage that did not correlate -- the ramp was not
            # recognised there -- and the ramp covers it, from its corner;
            # into the flat stretch itself it reaches only a little way.
            begin, finish = group[i].start_s, group[j].end_s
            if i > lo:
                before = group[i - 1]
                corner = (before.offset_s * 1000.0 - intercept) / slope
                begin = min(begin, max(corner, before.end_s - DRIFT_CORNER_REACH_S, before.start_s + MIN_BLOCK_S))
                if begin < before.end_s:
                    before.end_s = before.end_lo = before.end_hi = begin
            if j + 1 < hi:
                after = group[j + 1]
                corner = (after.offset_s * 1000.0 - intercept) / slope
                finish = max(finish, min(corner, after.start_s + DRIFT_CORNER_REACH_S, after.end_s - MIN_BLOCK_S))
                if finish > after.start_s:
                    after.start_s = after.start_lo = after.start_hi = finish
            return settle(lo, i) + staircase(group[i : j + 1], slope, intercept, walk, begin, finish) + settle(j + 1, hi)

        out.extend(settle(0, len(group)))
    return out


def _resolve_reuse(aligner: _Aligner, blocks: List[Block], primary_s: float, secondary_s: float) -> List[Block]:
    """Play no part of the dub twice.

    A dub of the same episode runs forward: every moment of it belongs at
    one moment of the video. Where two neighbouring stretches both read the
    same stretch of dub -- one of them resting on a cue heard twice, a shot
    the video shows again, a weak match -- the voices there are right at
    most once. On a real 1.5-hour pair, 21.6 s of dub played at two places
    46 s apart. The material goes to the stretch that agrees with it
    better, measured on exactly the overlapping span of each; the other is
    cut back, and the cut between them placed afresh (on the picture where
    it can be) -- never back over what was taken from it.

    Two stretches that overlap on the video are not this: that is an edge
    the assembly settles.
    """
    out = sorted(blocks, key=lambda b: b.start_s)
    for _attempt in range(4 * len(out) + 1):
        changed = False
        for index in range(len(out) - 1):
            a, b = out[index], out[index + 1]
            if b.start_s < a.end_s - 0.05:
                continue
            a_end = a.end_s + a.offset_s
            b_start = b.start_s + b.offset_s
            overlap = a_end - b_start
            if overlap < REUSE_MIN_S or overlap > WIDE_EPISODE_S:
                continue
            overlap = min(overlap, a.length_s, b.length_s)
            a_at, b_at = a.end_s - overlap, b.start_s
            keep_a = aligner.best_agreement(a.offset_s, a.end_s - overlap, a.end_s)
            keep_b = aligner.best_agreement(b.offset_s, b.start_s, b.start_s + overlap)
            if keep_a >= keep_b:
                loser, where = b, "after"
                b.start_s += overlap
                # The edge may move on into the stretch, not back over the
                # dub it no longer plays.
                b.start_lo, b.start_hi, b.start_cut = b.start_s, b.start_s + 1.0, None
            else:
                loser, where = a, "before"
                a.end_s -= overlap
                a.end_lo, a.end_hi, a.end_cut = a.end_s - 1.0, a.end_s, None
            aligner.log(
                f"  the dub's {_clock(b_start)} - {_clock(b_start + overlap)} was placed twice, at {_clock(a_at)} and at "
                f"{_clock(b_at)}; it stays where it agrees better ({keep_a:.3f} against {keep_b:.3f}), and the "
                f"stretch {where} is cut back {overlap:.1f}s"
            )
            if loser.length_s < MIN_BLOCK_S:
                out.remove(loser)
            else:
                _place_edge(aligner, a, b, primary_s, secondary_s)
                # Whatever the placement did, the two no longer share dub.
                if loser is b:
                    b.start_s = max(b.start_s, a.end_s + a.offset_s - b.offset_s)
                else:
                    a.end_s = min(a.end_s, b.start_s + b.offset_s - a.offset_s)
            changed = True
            break
        if not changed:
            break
    return out


def _follow_effects(aligner: _Aligner, blocks: List[Block]) -> None:
    """A stretch the music placed, between stretches the effects placed at
    another offset a few frames away, is asked which the effects on its
    own span prefer -- and follows them if they prefer the neighbours'.

    The case is a picture trim the music was left running across (see
    BANDS): the dialogue and effects step with the picture, the music
    does not until it is next re-laid, and a stretch found on the music
    alone sits a few frames from the dialogue. Only a neighbour within a
    few frames and a few seconds is asked about; farther apart the two
    are different cuts.

    The mirror case is asked too: a stretch the effects placed, between
    stretches the music and dialogue placed at another offset a few
    frames away, is asked which level its own lines follow. The effects
    follow the picture on most dubs, but on one real pair they had
    stepped twenty seconds early and sat 110 ms from the lines, which a
    dialogue-arbitrated neighbour had followed (see BANDS); a stretch
    re-measured after a gap search read only the effects and came out at
    the wrong level, 109 ms from its neighbour's. Only the dialogue can
    say, and it decides only past the distance it can tell (see
    SPEECH_ARBITER_S); a frame or two either way is below what it
    resolves.
    """
    for index, block in enumerate(blocks):
        for other in (blocks[index - 1] if index else None, blocks[index + 1] if index + 1 < len(blocks) else None):
            if other is None or other.effects == block.effects:
                continue
            apart = abs(other.offset_s - block.offset_s)
            touching = min(abs(other.end_s - block.start_s), abs(other.start_s - block.end_s)) <= FOLLOW_EFFECTS_GAP_S
            if not touching or apart <= SAME_OFFSET_S or apart > FOLLOW_EFFECTS_STEP_S:
                continue
            if block.effects:
                # The effects placed this stretch, the music and dialogue
                # the neighbour: the dialogue is asked which level the
                # stretch's own lines follow, and wins only when it
                # clearly prefers the neighbour's.
                if apart >= SPEECH_ARBITER_S and aligner.speech_choice(
                    block.start_s, block.end_s, block.offset_s, other.offset_s
                ) == other.offset_s:
                    aligner.log(
                        f"  the effects put {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.3f}s; "
                        f"the dialogue prefers the neighbour's {other.offset_s:+.3f}s there"
                    )
                    block.offset_s = other.offset_s
                    block.effects = False
                    break
                continue
            rate = ENVELOPE_RATE
            window = min(POLISH_WINDOW_S, block.length_s)
            positions = [block.start_s + i * (block.length_s - window) / max(1, int(block.length_s / window)) for i in range(int(block.length_s / window) + 1)]
            if aligner._effects_prefer(positions, other.offset_s * 1000.0, block.offset_s * 1000.0) and not (
                apart >= SPEECH_ARBITER_S
                and aligner.speech_choice(block.start_s, block.end_s, block.offset_s, other.offset_s) == block.offset_s
            ):
                aligner.log(
                    f"  the music put {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.3f}s; "
                    f"the effects prefer the neighbour's {other.offset_s:+.3f}s, and the dialogue goes with the effects"
                )
                block.offset_s = other.offset_s
                block.effects = True
                break


def _two_level_path(table: np.ndarray, jump: float) -> List[int]:
    """The best path through per-bin scores at a few levels, free to start
    and end at any, paying ``jump`` for every change."""
    count, bins = table.shape
    best = table[:, 0].astype(np.float64).copy()
    back = np.zeros((bins, count), dtype=int)
    for column in range(1, bins):
        top = int(np.argmax(best))
        new = np.empty(count)
        for state in range(count):
            if best[top] - jump > best[state]:
                new[state], back[column, state] = best[top] - jump, top
            else:
                new[state], back[column, state] = best[state], state
            new[state] += table[state, column]
        best = new
    state = int(np.argmax(best))
    path = [state]
    for column in range(bins - 1, 0, -1):
        state = int(back[column, state])
        path.append(state)
    return path[::-1]


def _split_levels(aligner: _Aligner, blocks: List[Block]) -> List[Block]:
    """Split every finished stretch that is really two levels in turn.

    A dub conformed scene by scene can step a few frames and back again
    every half minute; the readings that split stretches are ten seconds
    long and a few seconds apart, and where they straddle the steps they
    see one level or a blur, so which steps they find depends on where
    they happen to fall. Measured on a real pair: a stretch 146 s long at
    one offset was, per every band, 52 ms away for 96 s of it -- the dub
    early on the lips -- and the same stretch, measured with its start
    moved by four seconds, split correctly. The whole stretch sees both
    levels at once, as two peaks; where it does, each second is scored at
    each (see _TraceScores) and one path through them says which level
    each part is at. A part must stand clearly above the noise and above
    the other level over its length, and the effects, where they clearly
    prefer the other level, decide (see BANDS); otherwise it keeps the
    stretch's own offset."""
    out: List[Block] = list(blocks)
    for _round in range(SPLIT_ROUNDS):
        split: List[Block] = []
        changed = False
        for block in out:
            pieces = _split_block_levels(aligner, block)
            split.extend(pieces if pieces else [block])
            changed = changed or bool(pieces)
        out = split
        if not changed:
            break
    return out


def _reads_as_ramp(aligner: _Aligner, block: Block, apart_s: float) -> bool:
    """Whether a stretch's own readings lie on a line that walks at least
    half of ``apart_s`` across it, as closely as a drift's do (see
    DRIFT_MAX_SCATTER_MS)."""
    found = [f for f in aligner.readings(block) if f[4] >= READING_Z]
    if len(found) < 6:
        return False
    x = np.array([f[0] for f in found], dtype=np.float64)
    y = np.array([f[1] for f in found], dtype=np.float64) * 1000.0
    slope, intercept = _theil_sen(x, y)
    scatter = float(np.median(np.abs(y - (slope * x + intercept))))
    walk = abs(slope) * float(np.ptp(x))
    return scatter <= DRIFT_MAX_SCATTER_MS and walk >= 0.5 * apart_s * 1000.0


def _split_block_levels(aligner: _Aligner, block: Block) -> Optional[List[Block]]:
    if block.follows_drift or block.length_s < SPLIT_MIN_S:
        return None
    offsets, total = aligner.span_scores(block.start_s, block.end_s, block.offset_s)
    near = np.abs(offsets - block.offset_s) <= SPLIT_REACH_S
    inner = np.flatnonzero(near[1:-1] & (total[1:-1] >= total[:-2]) & (total[1:-1] >= total[2:])) + 1
    if inner.size < 2:
        return None
    first = int(inner[np.argmax(total[inner])])
    apart = np.abs(offsets[inner] - offsets[first]) >= SPLIT_APART_S
    if not apart.any():
        return None
    candidates = inner[apart]
    second = int(candidates[np.argmax(total[candidates])])
    if total[second] < max(SPLIT_SECOND_Z, SPLIT_SECOND_SHARE * total[first]):
        return None
    rate = ENVELOPE_RATE
    levels = [float(offsets[0] + _parabolic(total, index) / rate) for index in (first, second)]
    # A stretch whose offset walks is one drift, not two levels: across a
    # ramp the whole span peaks twice, either side of its middle. Its own
    # readings lie on a line, and the drift is followed later (see
    # _follow_drift); split here, a dub 0.3 per mille slow was cut into
    # alternating levels 73 ms apart, and two pieces kept them.
    if _reads_as_ramp(aligner, block, abs(levels[1] - levels[0])):
        return None
    # The stretch's own level first: a part the path cannot vouch for
    # keeps the offset the stretch was measured at, and that level keeps
    # its exact value when a peak is it to within a few milliseconds.
    levels.sort(key=lambda level: abs(level - block.offset_s))
    if abs(levels[0] - block.offset_s) * 1000.0 <= CONSISTENT_MS:
        levels[0] = block.offset_s
    scores = _TraceScores(aligner, block.start_s, block.end_s, levels)
    table = np.array([scores.at(level) for level in levels])
    path = _two_level_path(table, TRACE_JUMP)
    runs: List[List[int]] = []
    for column, state in enumerate(path):
        if runs and runs[-1][0] == state:
            runs[-1][2] = column
        else:
            runs.append([state, column, column])
    if len(runs) < 2:
        return None
    effects = "high" in [name for name, _weight in scores.parts]
    minimum = max(1, int(round(SPLIT_MIN_RUN_S / TRACE_BIN_S)))
    for run in runs:
        state, lo, hi = run
        size = hi - lo + 1
        own = float(table[state, lo : hi + 1].sum()) / math.sqrt(size)
        other = float(table[1 - state, lo : hi + 1].sum()) / math.sqrt(size)
        if effects and scores.sigma.get("high", 0.0) > 1e-12:
            high = [float(scores._means("high", level)[lo : hi + 1].sum()) / scores.sigma["high"] / math.sqrt(size) for level in levels]
            if high[1 - state] - high[state] >= TRACE_BEAT_Z and high[1 - state] >= TRACE_RUN_Z and not (
                abs(levels[1] - levels[0]) >= SPEECH_ARBITER_S
                and aligner.speech_choice(float(scores.edges[lo]), float(scores.edges[hi + 1]), levels[state], levels[1 - state]) == levels[state]
            ):
                run[0] = 1 - state
                continue
        if state != 0 and (size < minimum or own < TRACE_RUN_Z or own - other < TRACE_BEAT_Z):
            run[0] = 0
    merged: List[List[int]] = []
    for run in runs:
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
        else:
            merged.append(list(run))
    if len(merged) < 2:
        return None
    # Each change of level on its shot change, or where the agreement
    # changes, near where the path put it.
    edges = scores.edges
    bounds = [block.start_s] + [float(edges[run[1]]) for run in merged[1:]] + [block.end_s]
    for index in range(1, len(merged)):
        a, b = levels[merged[index - 1][0]], levels[merged[index][0]]
        span_lo = max(bounds[index - 1] + MIN_BLOCK_S, bounds[index] - TRACE_STEP_REACH_S)
        span_hi = min(bounds[index + 1] - MIN_BLOCK_S, bounds[index] + TRACE_STEP_REACH_S)
        if span_hi > span_lo:
            bounds[index], _how = _place_step(aligner, span_lo, span_hi, a, b, hole=False, guess_on_sound=True)
    if any(bounds[index + 1] - bounds[index] < MIN_BLOCK_S for index in range(len(merged))):
        return None
    pieces: List[Block] = []
    for index, run in enumerate(merged):
        piece = Block(
            bounds[index], bounds[index + 1], levels[run[0]], match=block.match, support=block.support,
            windows=block.windows, start_lo=bounds[index], start_hi=bounds[index],
            end_lo=bounds[index + 1], end_hi=bounds[index + 1], effects=block.effects,
        )
        pieces.append(piece)
    pieces[0].start_lo, pieces[0].start_hi, pieces[0].start_cut = block.start_lo, block.start_hi, block.start_cut
    pieces[-1].end_lo, pieces[-1].end_hi, pieces[-1].end_cut = block.end_lo, block.end_hi, block.end_cut
    for piece in pieces:
        piece.match = aligner.sharp_match(piece)
    aligner.log(
        f"  {_clock(block.start_s)} - {_clock(block.end_s)} holds two levels {abs(levels[1] - levels[0]) * 1000.0:.0f} ms apart: "
        + ", ".join(f"{_clock(p.start_s)} - {_clock(p.end_s)} at {p.offset_s:+.4f}s" for p in pieces)
    )
    return pieces
