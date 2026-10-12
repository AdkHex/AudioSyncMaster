"""Placing every cut: on the sound, on the waveform, on the picture.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..media import MediaError

from .aligner import _agreement_threshold, _fit_end, _fit_joint, _fit_start, _local_standardize, _timeline_activity
from .plan import _clock
from .scores import Block, _parabolic
from .tuning import AGREEMENT_FLOOR, ANALYSIS_SR, CONSISTENT_FRACTION, CONSISTENT_MS, CONSISTENT_READINGS, CONTINUOUS_DUB_S, CREDIBLE_Z, EDGE_UNCERTAINTY_STEPS, ENVELOPE_RATE, LOCAL_STATS_S, MIN_BLOCK_MATCH, PICTURE_SNAP_MIN_S, PICTURE_VETO_BLOCK_S, PICTURE_VETO_Z, POLISH_RANGE_S, RECOVER_JOIN_OVERLAP_S, RELATIVE_FLOOR, RELATIVE_FLOOR_CAP, SAME_OFFSET_S, SETTLE_MARGIN, SETTLE_MIN_S, SETTLE_SHARE, SHARPEN_INTERIOR_S, SHARPEN_MAX_SPAN_S, SHARPEN_MIN_UNCERTAINTY_S, SHORT_LEVEL_APART_S, SHORT_LEVEL_BANDS, SHORT_LEVEL_BLOCK_S, SHORT_LEVEL_BRACKET_S, SHORT_LEVEL_GRID_S, SHORT_LEVEL_MAX_STEP_S, SHORT_LEVEL_MIN_S, SHORT_LEVEL_MIN_STEP_S, SHORT_LEVEL_REACH_S, SHORT_LEVEL_Z, SPEECH_SNAP_S, SPEECH_STEP_MARGIN, SPEECH_STEP_NULLS_S, STEP_EVIDENCE, STRONG_SUPPORT_Z, TIGHT_MS, TRIM_MAX_S, VERIFY_CUT_MIN_MOVE_S, VERIFY_CUT_MIN_STEP_S, VERIFY_CUT_REACH_S, VERIFY_CUT_SIDE_S, VERIFY_CUT_SMALL_STEP_S, VERIFY_CUT_SNAP_S, VERIFY_CUT_Z

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aligner import _Aligner
    from .plan import Segment


def _credible(blocks: List[Block], aligner: _Aligner) -> List[Block]:
    """Drop stretches whose measured offset correlates at the noise floor.

    Done before any edge is placed: a stretch that is really noise does not
    just waste a splice, it drags its neighbours' edges towards itself when
    they are placed against it.
    """
    long_matches = [b.match for b in blocks if b.windows >= 3]
    floor = MIN_BLOCK_MATCH
    if long_matches:
        # Relative to what this pair's real stretches score, when they
        # score well: a pair whose stretches sit at 0.3 has noise blocks
        # at 0.05 that the absolute floor alone would keep. Capped, because
        # a quiet scene in a pair that otherwise measures at 0.6 is still
        # the dub: on a real pair, stretches the gap search had just found
        # at ten to twenty noise units were thrown out by a floor of 0.18.
        floor = max(floor, min(RELATIVE_FLOOR_CAP, RELATIVE_FLOOR * float(np.median(long_matches))))
    kept = []
    for block in blocks:
        agrees = block.match >= AGREEMENT_FLOOR and (
            (block.readings >= CONSISTENT_READINGS and block.consistency >= CONSISTENT_FRACTION)
            or (block.readings == 3 and block.spread_ms <= TIGHT_MS)
            # Found outright, by many windows agreeing on one offset; the
            # fine match being modest says the scene is quiet, not absent.
            or (block.support >= STRONG_SUPPORT_Z and block.windows >= 3)
        )
        # A stretch too short to show its readings agree has to clear a
        # higher bar on correlation alone. One that fails is not lost: if
        # it is real, the gap search finds it again where its neighbours
        # say it must be, which is a far narrower search than the path's.
        bar = floor if block.readings >= 3 else 2.0 * floor
        if block.match < bar and not agrees:
            aligner.log(
                f"  dropped {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.2f}s:"
                f" match {block.match:.2f} is noise"
            )
            continue
        # A match can be high by chance where a band is sparse; a stretch
        # fewer than three windows matched outright has to show its peak
        # standing above the offsets around it as well. Three windows
        # that matched are their own evidence: a quiet scene the gap
        # search had found over thirty-seven of them measured 4.6 here.
        z = aligner.credibility(block) if block.windows < 3 else float("inf")
        if z < CREDIBLE_Z:
            aligner.log(
                f"  dropped {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.2f}s:"
                f" its peak is only {z:.1f} noise units above the offsets around it"
            )
            continue
        kept.append(block)
    return kept


def _measure_and_split(aligner: _Aligner, blocks: List[Block]) -> List[Block]:
    """Measure every stretch's offset at 2 ms, splitting any that stepped.

    The first measurement of each stretch searches wide, so a cut the
    stretch straddles is seen as a step and split; the halves are then
    measured narrowly around their own offsets.
    """
    measured: List[Block] = []
    pending = [(block, True) for block in blocks]
    while pending:
        block, wide = pending.pop(0)
        block.offset_s, block.match, _ = aligner.polish_offset(block, wide)
        split = aligner.split_at_step(block, wide)
        if split is not None:
            pending = [(part, False) for part in split] + pending
            continue
        _settle(aligner, block)
        measured.append(block)
    return sorted(measured, key=lambda b: b.start_s)


def _settle(aligner: _Aligner, block: Block) -> bool:
    """Check a stretch's offset against the whole stretch at once, and
    move it to where the whole stretch peaks when that is clearly better
    (see SETTLE_MARGIN). True when it moved.

    Measured on a real pair: a 39-second stretch of a quiet scene had its
    readings scattered over 0.6 s, the effects band's chance peaks among
    them, and their median put it at -133.759 s; the whole stretch
    correlated at 22 noise units at -133.669 s and at 5 at -133.759 --
    89 ms, which shows on the lips. A stretch the effects placed, or one
    following a drift, keeps its offset: those were read on purpose from
    one band or one part of it."""
    if block.effects or block.follows_drift or block.length_s < SETTLE_MIN_S:
        return False
    offsets, total = aligner.span_scores(block.start_s, block.end_s, block.offset_s)
    near = np.flatnonzero(np.abs(offsets - block.offset_s) <= POLISH_RANGE_S)
    if near.size == 0:
        return False
    peak = int(near[np.argmax(total[near])])
    here = int(np.argmin(np.abs(offsets - block.offset_s)))
    at_offset = float(total[max(0, here - 1) : here + 2].max())
    top = float(total[peak])
    best = float(offsets[0] + _parabolic(total, peak) / ENVELOPE_RATE)
    if abs(best - block.offset_s) * 1000.0 <= CONSISTENT_MS:
        return False
    if top - at_offset < SETTLE_MARGIN or at_offset > SETTLE_SHARE * top:
        return False
    aligner.log(
        f"  {_clock(block.start_s)} - {_clock(block.end_s)}: its readings put it at {block.offset_s:+.3f}s, "
        f"the whole stretch at {best:+.3f}s ({top:.1f} noise units against {at_offset:.1f}); the whole stretch decides"
    )
    block.offset_s = best
    return True


def _place_all_edges(aligner: _Aligner, blocks: List[Block], primary_s: float, secondary_s: float) -> None:
    for index in range(len(blocks) + 1):
        before = blocks[index - 1] if index > 0 else None
        after = blocks[index] if index < len(blocks) else None
        _place_edge(aligner, before, after, primary_s, secondary_s)


def _place_edge(
    aligner: _Aligner,
    before: Optional[Block],
    after: Optional[Block],
    primary_s: float,
    secondary_s: float,
) -> None:
    """Put the boundary between two stretches exactly where the agreement moves.

    Each edge is first placed on its own, inside the bracket it was left
    in: where the first stretch's agreement ends, and where the second's
    begins. Those two answers are then read against what the offsets
    imply. If the dub is continuous across the cut -- the usual case, a
    scene missing from the dub -- the second must begin exactly (first
    offset - second offset) after the first ends, and two independent
    placements will not land there to the millisecond; the joint placement
    is used, which ties them together. If the dub itself skips material as
    well, the independent placements are kept, because then there is
    genuinely a gap on both sides.

    That is done twice: on the onset envelopes, which always exist, and then
    on the waveforms inside the interval the envelopes left, where the two
    mixes turn out to share one. Every fit is confined to the edge's own
    bracket, so an edge already placed on the waveform is not moved by a
    later pass on the envelope.
    """
    if before is None and after is None:
        return
    lo_s = min(b for b in (before.end_lo if before else None, after.start_lo if after else None) if b is not None)
    hi_s = max(b for b in (before.end_hi if before else None, after.start_hi if after else None) if b is not None)
    lo_s, hi_s = max(0.0, lo_s), min(primary_s, hi_s)
    if hi_s <= lo_s:
        return
    # The brackets as the coarse windows left them, before the envelope
    # narrows them: the waveform pass searches these.
    coarse_before = (before.end_lo, before.end_hi) if before else None
    coarse_after = (after.start_lo, after.start_hi) if after else None
    curve_before = aligner.edge_curve(before.offset_s, lo_s, hi_s) if before else None
    curve_after = aligner.edge_curve(after.offset_s, lo_s, hi_s) if after else None
    _fit_edges(
        before, after, curve_before, curve_after,
        before.match if before else 0.0, after.match if after else 0.0,
        lo_s, float(ENVELOPE_RATE), primary_s, secondary_s,
    )
    _sharpen(aligner, before, after, primary_s, secondary_s, coarse_before, coarse_after)
    _snap_to_picture(aligner, before, after, primary_s, secondary_s, coarse_before, coarse_after)


def _fit_edges(
    before: Optional[Block],
    after: Optional[Block],
    curve_before: Optional[np.ndarray],
    curve_after: Optional[np.ndarray],
    r_before: float,
    r_after: float,
    span_lo_s: float,
    rate: float,
    primary_s: float,
    secondary_s: float,
) -> None:
    """Place the edge(s) on agreement curves that start at ``span_lo_s`` and
    run at ``rate``, each confined to its edge's bracket. Writes the result
    and its uncertainty back into the blocks."""

    def index(t: float, curve: np.ndarray) -> int:
        return max(0, min(len(curve), int(round((t - span_lo_s) * rate))))

    def about(match: float, bracket_s: float) -> float:
        steps = (EDGE_UNCERTAINTY_STEPS / max(match, 0.02)) ** 2
        return max(SHARPEN_MIN_UNCERTAINTY_S / 2.0, min(bracket_s / 2.0, steps / rate))

    def covered(lo: float, hi: float, curve: np.ndarray) -> bool:
        return hi > span_lo_s and lo < span_lo_s + len(curve) / rate

    end_s = start_s = None
    if before is not None and curve_before is not None and curve_before.size and covered(before.end_lo, before.end_hi, curve_before):
        k = _fit_end(
            curve_before, _agreement_threshold(r_before),
            index(before.end_lo, curve_before), index(before.end_hi, curve_before),
        )
        end_s = span_lo_s + k / rate
    if after is not None and curve_after is not None and curve_after.size and covered(after.start_lo, after.start_hi, curve_after):
        k = _fit_start(
            curve_after, _agreement_threshold(r_after),
            index(after.start_lo, curve_after), index(after.start_hi, curve_after),
        )
        start_s = span_lo_s + k / rate

    tied = False
    if before is not None and after is not None:
        current_end = end_s if end_s is not None else before.end_s
        current_start = start_s if start_s is not None else after.start_s
        dub_gap_s = (current_start + after.offset_s) - (current_end + before.offset_s)
        video_gap_s = max(0.0, before.offset_s - after.offset_s)
        # Tied when the dub is as good as continuous across the cut -- or
        # when the independent placements are impossible: overlapping on
        # the video, or overlapping on the dub, which would play the same
        # dub audio twice.
        tied = current_start < current_end or dub_gap_s < CONTINUOUS_DUB_S
        if tied and curve_before is not None and curve_after is not None and curve_before.size and curve_after.size:
            gap_n = int(round(video_gap_s * rate))
            lo = max(index(before.end_lo, curve_before), index(after.start_lo, curve_after) - gap_n)
            hi = min(index(before.end_hi, curve_before), index(after.start_hi, curve_after) - gap_n)
            if hi >= lo:
                k = _fit_joint(
                    curve_before, curve_after,
                    _agreement_threshold(r_before), _agreement_threshold(r_after),
                    gap_n, lo, hi,
                )
                end_s = span_lo_s + k / rate
                start_s = end_s + video_gap_s
            else:
                tied = False
        elif tied and end_s is not None and start_s is None:
            start_s = end_s + video_gap_s
        elif tied and start_s is not None and end_s is None:
            end_s = start_s - video_gap_s

    if before is not None and end_s is not None:
        bracket = before.end_hi - before.end_lo
        before.end_s = min(end_s, secondary_s - before.offset_s, primary_s)
        u = about(r_before if not tied else max(r_before, r_after), bracket)
        before.end_lo, before.end_hi = before.end_s - u, before.end_s + u
    if after is not None and start_s is not None:
        bracket = after.start_hi - after.start_lo
        after.start_s = max(start_s, -after.offset_s, 0.0)
        u = about(r_after if not tied else max(r_before, r_after), bracket)
        after.start_lo, after.start_hi = after.start_s - u, after.start_s + u


def _sharpen(
    aligner: _Aligner,
    before: Optional[Block],
    after: Optional[Block],
    primary_s: float,
    secondary_s: float,
    bracket_before: Optional[Tuple[float, float]] = None,
    bracket_after: Optional[Tuple[float, float]] = None,
) -> None:
    """Re-place an edge on the waveforms, inside the given brackets.

    The brackets default to the edges' current ones; callers with a wider
    bracket from the coarse windows pass it, and the search covers all of
    it -- the envelope's own placement is not trusted to have narrowed it
    correctly. But the envelope's placement is only ever *replaced*, never
    thrown away: an edge the waveform cannot place keeps the envelope's
    answer and the envelope's uncertainty. A version of this that widened
    the brackets first and narrowed them on success left every edge the
    waveform could not reach at the coarse bracket, tens of seconds wide,
    after the envelope had placed it to within half a second.
    """
    fit_before = before is not None and before.end_uncertainty_s >= SHARPEN_MIN_UNCERTAINTY_S
    fit_after = after is not None and after.start_uncertainty_s >= SHARPEN_MIN_UNCERTAINTY_S
    if not fit_before and not fit_after:
        return
    search_before = (bracket_before or (before.end_lo, before.end_hi)) if fit_before else None
    search_after = (bracket_after or (after.start_lo, after.start_hi)) if fit_after else None

    if search_before and search_after:
        if max(search_before[1], search_after[1]) - min(search_before[0], search_after[0]) > SHARPEN_MAX_SPAN_S:
            # A long gap: the two edges have nothing to do with each other,
            # so each is sharpened on its own, each with its own bracket.
            _sharpen(aligner, before, None, primary_s, secondary_s, bracket_before=search_before)
            _sharpen(aligner, None, after, primary_s, secondary_s, bracket_after=search_after)
            return

    edges = [e for e in (search_before, search_after) if e is not None]
    lo_s = max(0.0, min(e[0] for e in edges))
    hi_s = min(primary_s, max(e[1] for e in edges))
    if hi_s - lo_s > SHARPEN_MAX_SPAN_S:
        # Too wide to decode whole: centred on where the envelope put the
        # edge(s), which is the best guess there is.
        placed = [e for e in (before.end_s if fit_before else None, after.start_s if fit_after else None) if e is not None]
        middle = min(max(sum(placed) / len(placed), lo_s + SHARPEN_MAX_SPAN_S / 2.0), hi_s - SHARPEN_MAX_SPAN_S / 2.0)
        lo_s, hi_s = middle - SHARPEN_MAX_SPAN_S / 2.0, middle + SHARPEN_MAX_SPAN_S / 2.0

    # The decoded span reaches into each stretch beyond the bracket, so
    # there is material known to be the stretch's own to measure the
    # sample-exact offset on; and a little past it, so the cumulative sums
    # have a run-up.
    room = SHARPEN_INTERIOR_S + 1.0
    span_lo = max(0.0, lo_s - (room if fit_before else 1.0))
    span_hi = min(primary_s, hi_s + (room if fit_after else 1.0))
    if fit_before:
        span_lo = max(span_lo, before.start_s)
    if fit_after:
        span_hi = min(span_hi, after.end_s)
    if span_hi - span_lo < 1.0:
        return

    # Where the sample-exact offset is measured: material that is the
    # stretch's own beyond doubt, which is everything up to the envelope's
    # placement less its uncertainty. Falls back to the whole side of the
    # span when that leaves too little.
    curve_before = curve_after = None
    r_before = r_after = 0.0
    if fit_before:
        sure = before.end_s - before.end_uncertainty_s
        inside = (span_lo, min(span_hi, sure if sure - span_lo >= SHARPEN_INTERIOR_S else hi_s))
        got = aligner.waveform_agreement(before, span_lo, span_hi, inside)
        if got is not None:
            curve_before, r_before = got
    if fit_after:
        sure = after.start_s + after.start_uncertainty_s
        inside = (max(span_lo, sure if span_hi - sure >= SHARPEN_INTERIOR_S else lo_s), span_hi)
        got = aligner.waveform_agreement(after, span_lo, span_hi, inside)
        if got is not None:
            curve_after, r_after = got
    if curve_before is None and curve_after is None:
        return

    # Widen the brackets to the search interval only for the sides about
    # to be fitted, so the fit may move the edge anywhere the coarse
    # windows allowed; the fit narrows them again.
    if curve_before is not None:
        before.end_lo, before.end_hi = max(lo_s, search_before[0]), min(hi_s, search_before[1])
    if curve_after is not None:
        after.start_lo, after.start_hi = max(lo_s, search_after[0]), min(hi_s, search_after[1])
    _fit_edges(
        before if curve_before is not None else None,
        after if curve_after is not None else None,
        curve_before, curve_after, r_before, r_after,
        span_lo, float(ANALYSIS_SR), primary_s, secondary_s,
    )
    # One side placed on the waveform, the other not: if the dub runs
    # straight through the cut the other follows it exactly.
    if before is not None and after is not None and (curve_before is None) != (curve_after is None) and fit_before and fit_after:
        dub_gap_s = (after.start_s + after.offset_s) - (before.end_s + before.offset_s)
        if abs(dub_gap_s) < CONTINUOUS_DUB_S:
            gap = max(0.0, before.offset_s - after.offset_s)
            if curve_before is not None:
                after.start_s = max(before.end_s + gap, -after.offset_s, 0.0)
                after.start_lo, after.start_hi = before.end_lo + gap, before.end_hi + gap
            else:
                before.end_s = min(after.start_s - gap, secondary_s - before.offset_s, primary_s)
                before.end_lo, before.end_hi = after.start_lo - gap, after.start_hi - gap


def _short_level(
    aligner: _Aligner, lo_s: float, hi_s: float, first_s: float, last_s: float,
) -> Optional[Tuple[float, float, float]]:
    """(start, end, offset) of a level between ``first_s`` and ``last_s``
    that agrees better than both across part of [lo_s, hi_s), or None (see
    SHORT_LEVEL_MIN_STEP_S)."""
    size = int(round(SHORT_LEVEL_BLOCK_S * ENVELOPE_RATE))
    bands = aligner.shared_bands

    def per_band(offset_s: float) -> np.ndarray:
        rows = []
        for band in bands:
            x = aligner.agreement(offset_s, lo_s, hi_s, [band])
            count = len(x) // size
            rows.append(x[: count * size].reshape(count, size).mean(axis=1) if count else np.zeros(0))
        count = min(len(row) for row in rows) if rows else 0
        return np.array([row[:count] for row in rows]) if rows else np.zeros((0, 0))

    first_bands, last_bands = per_band(first_s), per_band(last_s)
    count = min(first_bands.shape[1], last_bands.shape[1]) if first_bands.size and last_bands.size else 0
    least = int(round(SHORT_LEVEL_MIN_S / SHORT_LEVEL_BLOCK_S))
    if count < 3 * least:
        return None
    first_bands, last_bands = first_bands[:, :count], last_bands[:, :count]
    a, c = first_bands.sum(axis=0), last_bands.sum(axis=0)
    ca = np.concatenate([[0.0], np.cumsum(a)])
    cc = np.concatenate([[0.0], np.cumsum(c)])
    one_step = float(np.max(ca + (cc[count] - cc)))
    low, high = min(first_s, last_s), max(first_s, last_s)
    best: Optional[Tuple[float, int, int, float, np.ndarray]] = None
    for level in np.arange(low + SHORT_LEVEL_APART_S, high - SHORT_LEVEL_APART_S + 1e-9, SHORT_LEVEL_GRID_S):
        level_bands = per_band(float(level))
        if level_bands.shape[1] < count:
            continue
        level_bands = level_bands[:, :count]
        cb = np.concatenate([[0.0], np.cumsum(level_bands.sum(axis=0))])
        # The best path first -> level -> last, the level held at least
        # ``least`` blocks: before i the first level, from j the last.
        left = np.maximum.accumulate(ca - cb)
        ends = np.arange(least, count + 1)
        totals = left[ends - least] + cb[ends] + cc[count] - cc[ends]
        k = int(np.argmax(totals))
        gain = float(totals[k]) - one_step
        if best is None or gain > best[0]:
            j = int(ends[k])
            i = int(np.argmax((ca - cb)[: j - least + 1]))
            best = (gain, i, j, float(level), level_bands)
    if best is None or best[0] <= 0.0:
        return None
    _gain, i, j, level, level_bands = best
    rival = np.maximum(a[i:j], c[i:j])
    margin = level_bands.sum(axis=0)[i:j] - rival
    if len(margin) < 3:
        return None
    t_stat = float(margin.mean()) / (float(margin.std(ddof=1)) / np.sqrt(len(margin)) + 1e-9)
    if t_stat < SHORT_LEVEL_Z:
        return None
    agreeing = sum(
        1 for k in range(len(bands))
        if float(np.mean(level_bands[k, i:j] - np.maximum(first_bands[k, i:j], last_bands[k, i:j]))) > 0.0
    )
    if agreeing < min(SHORT_LEVEL_BANDS, len(bands)):
        return None
    return lo_s + i * SHORT_LEVEL_BLOCK_S, lo_s + j * SHORT_LEVEL_BLOCK_S, level


def _split_short_levels(
    aligner: _Aligner, blocks: List[Block], primary_s: float, secondary_s: float, notes: Optional[List[str]] = None,
) -> List[Block]:
    """Put a level too short for the readings, between two stretches, in as
    a stretch of its own, its edges placed like every other (see
    SHORT_LEVEL_MIN_STEP_S)."""
    out = sorted(blocks, key=lambda b: b.start_s)
    index = 0
    while index + 1 < len(out):
        first, last = out[index], out[index + 1]
        index += 1
        step = abs(last.offset_s - first.offset_s)
        if not SHORT_LEVEL_MIN_STEP_S <= step <= SHORT_LEVEL_MAX_STEP_S or first.follows_drift or last.follows_drift:
            continue
        lo_s = max(first.start_s, first.end_s - SHORT_LEVEL_REACH_S)
        hi_s = min(last.end_s, last.start_s + SHORT_LEVEL_REACH_S)
        found = _short_level(aligner, lo_s, hi_s, first.offset_s, last.offset_s)
        if found is None:
            continue
        start, end, level = found
        if start - first.start_s < SHORT_LEVEL_MIN_S or last.end_s - end < SHORT_LEVEL_MIN_S:
            continue
        bracket = SHORT_LEVEL_BRACKET_S
        middle = Block(start, end, level, start_lo=start - bracket, start_hi=start + bracket,
                       end_lo=end - bracket, end_hi=end + bracket)
        middle.match = aligner.sharp_match(middle)
        first.end_s = min(first.end_s, start)
        first.end_lo, first.end_hi = start - bracket, start + bracket
        last.start_s = max(last.start_s, end)
        last.start_lo, last.start_hi = end - bracket, end + bracket
        _place_edge(aligner, first, middle, primary_s, secondary_s)
        _place_edge(aligner, middle, last, primary_s, secondary_s)
        if middle.length_s <= 0.0:
            continue
        out.insert(index, middle)
        index += 1
        if notes is not None:
            notes.append(
                f"a level of {level:+.3f}s found between {first.offset_s:+.3f}s and {last.offset_s:+.3f}s across "
                f"{_clock(middle.start_s)} - {_clock(middle.end_s)}: two trims too close together for the readings"
            )
    return out


def _verify_cuts(aligner: "_Aligner", segments: List[Segment], notes: Optional[List[str]]) -> None:
    """The last look at every cut of the assembled plan (see
    VERIFY_CUT_REACH_S): where the sound on either side clearly puts the
    switch from one stretch's offset to the next's elsewhere, the cut
    moves there -- with the frames the dub lacks still filled between
    the two -- and onto a shot change when one is that close."""
    rate = ENVELOPE_RATE
    side = int(VERIFY_CUT_SIDE_S * rate)
    reach = VERIFY_CUT_REACH_S
    index = 0
    while index < len(segments) - 1:
        first = segments[index]
        between: Optional[Segment] = None
        second = segments[index + 1]
        if second.kind != "dub" and index + 2 < len(segments):
            between, second = second, segments[index + 2]
        index += 1
        if first.kind != "dub" or second.kind != "dub" or first.offset_s is None or second.offset_s is None:
            continue
        step = first.offset_s - second.offset_s
        if abs(step) < VERIFY_CUT_MIN_STEP_S:
            continue
        hole = max(step, 0.0)
        gap = second.start_s - first.end_s
        if abs(gap - hole) > 0.05:
            continue  # a fill of another kind sits between: not a plain cut
        bands = (["high"] if abs(step) < VERIFY_CUT_SMALL_STEP_S and "high" in aligner.shared_bands else None)
        cut = first.end_s
        lo = max(first.start_s + VERIFY_CUT_SIDE_S, cut - reach)
        hi = min(second.end_s - VERIFY_CUT_SIDE_S - hole, cut + reach)
        if hi - lo < 2 * VERIFY_CUT_MIN_MOVE_S:
            continue
        before = aligner.agreement(first.offset_s, lo - VERIFY_CUT_SIDE_S, hi, bands)
        after = aligner.agreement(second.offset_s, lo + hole, hi + hole + VERIFY_CUT_SIDE_S, bands)
        if len(before) < side + 2 or len(after) < side + 2:
            continue
        cb = np.concatenate([[0.0], np.cumsum(before)])
        ca = np.concatenate([[0.0], np.cumsum(after)])
        count = min(len(before) - side, len(after) - side)
        idx = np.arange(0, count, max(1, int(0.01 * rate)))
        score = (cb[idx + side] - cb[idx]) + (ca[idx + side] - ca[idx])
        moved = float(lo + idx[int(np.argmax(score))] / rate)
        if abs(moved - cut) < VERIFY_CUT_MIN_MOVE_S:
            continue
        # Only the span between where the cut is and where the sound would
        # put it is in dispute: across it, the offset that should play
        # there must agree clearly better than the one that does, block by
        # block -- a t-statistic over quarter seconds, which a loud passage
        # elsewhere in the reach cannot sway.
        a, b = sorted((cut, moved))
        keep = aligner.agreement(first.offset_s, a, b, bands)
        other = aligner.agreement(second.offset_s, a + hole, b + hole, bands)
        n = min(len(keep), len(other))
        gain = (keep[:n] - other[:n]) if moved > cut else (other[:n] - keep[:n])
        block = max(1, int(0.25 * rate))
        means = np.array([gain[i:i + block].mean() for i in range(0, n - block + 1, block)])
        if len(means) < 3:
            continue
        t_stat = float(means.mean()) / (float(means.std(ddof=1)) / np.sqrt(len(means)) + 1e-9)
        if t_stat < VERIFY_CUT_Z:
            continue
        if aligner.picture is not None:
            shots = aligner.picture.within(moved - VERIFY_CUT_SNAP_S, moved + VERIFY_CUT_SNAP_S)
            if shots:
                moved = min(shots, key=lambda c: abs(c - moved))
        if notes is not None:
            notes.append(
                f"the cut at {_clock(cut)} moved to {_clock(moved)}: the sound on either side puts the switch "
                f"from {first.offset_s:+.3f}s to {second.offset_s:+.3f}s there"
            )
        # What lies between moves with the cut as it is: a fill of the
        # frames the dub lacks, or nothing when the assembly let the two
        # pieces meet. Setting the gap to the step instead opened a hole of
        # 40 ms where they had met, and every sample after it was written
        # 40 ms early.
        first.end_s = moved
        if between is not None:
            between.start_s, between.end_s = moved, moved + gap
            between.source_start_s = moved
        second.start_s = moved + gap
        second.source_start_s = second.start_s + second.offset_s


def _join_overlapping(blocks: List[Block]) -> List[Block]:
    """Stretches that overlap at one measured offset are one stretch: a piece found in a gap that reaches back over its
    neighbour is that neighbour reaching further than its first edge
    said. Joined, the edge is placed again on what the stretch really
    covers; left apart, each got an edge of its own, the two crossed, and
    the found piece was dropped as noise, with the cut left seconds early
    where the coarse pass had put it. Offsets are compared as measured, to
    SAME_OFFSET_S, so a trim of a frame is never joined over."""
    out: List[Block] = []
    for block in sorted(blocks, key=lambda b: b.start_s):
        if out:
            last = out[-1]
            if (abs(block.offset_s - last.offset_s) <= SAME_OFFSET_S
                    and block.start_s < last.end_s - RECOVER_JOIN_OVERLAP_S
                    and not (last.follows_drift or block.follows_drift)):
                if block.end_s > last.end_s:
                    last.end_s = block.end_s
                    last.end_lo, last.end_hi = block.end_lo, block.end_hi
                    last.end_cut = block.end_cut
                continue
        out.append(block)
    return out


def _place_step(
    aligner: _Aligner, lo_s: float, hi_s: float, before_s: float, after_s: float, hole: bool = True,
    guess_on_sound: bool = False, bands: Optional[Sequence[str]] = None,
) -> Tuple[float, str]:
    """Where inside [lo_s, hi_s] a stretch at ``before_s`` gives way to one
    at ``after_s``: the point that leaves the most agreement at the first
    offset before it and at the second after it -- after the frames the dub
    lacks, when the offset falls and ``hole`` is set.

    The picture has the last word where it can speak. A cut in the dub's
    edit is a cut in the picture, so the point is looked for among the
    video's shot changes in the span -- with, where the dub lacks frames,
    another shot change exactly that far after it (see ``PictureCuts.pairs``).
    Among those the sound chooses when it has anything to say; a lone
    candidate needs no choosing; a candidate the sound scores clearly below
    its own choice is not the place (a dissolve, a cut the picture missed),
    and the sound's choice stands. Measured on a real pair, the sound alone
    put a one-frame trim 0.66 s early and a 1.1 s insert 0.63 s early --
    both exactly on shot changes.

    Returns the point and how it was found: ``picture``, ``sound``,
    ``speech`` (the two languages' lines, where neither the sound nor the
    picture could say; see _speech_step), or ``guess`` (the middle, or one
    of several shot changes nothing could choose between). A caller that knows the step is there, and only
    asks where, sets ``guess_on_sound``: a guess is then the sound's best
    point, however faint, rather than the middle. ``bands`` limits the
    sound to those bands (all shared ones by default).
    """
    rate = ENVELOPE_RATE
    drop = max(0.0, before_s - after_s) if hole else 0.0
    room = max(0.0, hi_s - lo_s - drop)
    middle = lo_s + room / 2.0
    scores: Optional[np.ndarray] = None
    measured = False
    a = aligner.agreement(before_s, lo_s, hi_s, bands)
    b = aligner.agreement(after_s, lo_s, hi_s, bands)
    n = min(len(a), len(b))
    skip = int(round(drop * rate))
    if n >= 2 and room > 0.0 and n - skip > 0:
        a, b = a[:n], b[:n]
        level = max(float(np.abs(a).mean()), float(np.abs(b).mean()))
        if level >= AGREEMENT_FLOOR:
            ahead = np.concatenate([[0.0], np.cumsum(a)])          # agreement at the first offset up to k
            behind = np.concatenate([np.cumsum(b[::-1])[::-1], [0.0]])  # agreement at the second from k on
            last = n - skip
            scores = ahead[: last + 1] + behind[skip : skip + last + 1]
            # Said only when the choice stands out: a flat score is no placement.
            measured = float(scores.max() - np.median(scores)) > STEP_EVIDENCE * level * rate
    sound = lo_s + int(np.argmax(scores)) / rate if scores is not None else middle

    picture = aligner.picture
    candidates = picture.pairs(lo_s, lo_s + room, drop) if picture is not None and room > 0.0 else None
    if candidates and scores is not None and measured:
        def at(cut: float) -> float:
            return float(scores[min(len(scores) - 1, max(0, int(round((cut - lo_s) * rate))))])
        best = max(candidates, key=at)
        top, low = float(scores.max()), float(np.median(scores))
        if top - at(best) <= 0.5 * (top - low):
            return best, "picture"
        return sound, "sound"
    if candidates and len(candidates) == 1:
        return candidates[0], "picture"
    if measured:
        return sound, "sound"
    # The music and effects cannot say, and the picture has no one cut to
    # offer: the lines may (see _speech_step), and a shot change close to
    # where they put it is where it goes.
    heard = _speech_step(aligner, lo_s, hi_s, before_s, after_s, drop) if room > 0.0 else None
    if heard is not None:
        if candidates:
            near = min(candidates, key=lambda cut: abs(cut - heard))
            if abs(near - heard) <= SPEECH_SNAP_S:
                return near, "picture"
        return heard, "speech"
    if candidates:
        if scores is not None:
            def score_at(cut: float) -> float:
                return float(scores[min(len(scores) - 1, max(0, int(round((cut - lo_s) * rate))))])
            return max(candidates, key=score_at), "guess"
        return min(candidates, key=lambda cut: abs(cut - middle)), "guess"
    return (sound if guess_on_sound else middle), "guess"


def _speech_step(
    aligner: _Aligner, lo_s: float, hi_s: float, before_s: float, after_s: float, drop_s: float,
) -> Optional[float]:
    """Where inside [lo_s, hi_s) the dub's lines stop agreeing with the
    original's at ``before_s`` and start agreeing at ``after_s`` (``drop_s``
    later), or None when the speech does not say clearly. Measured on a
    real pair: a one-second trim the music and effects were silent about,
    guessed at the middle of a 13-second gap, sat 5.8 s later; and all 5.2 s
    of dub around a 58-second missing scene, split half to each side by a
    guess, belonged to the scene before it."""
    primary, secondary = getattr(aligner, "primary", None), getattr(aligner, "secondary", None)
    if primary is None or secondary is None or not getattr(primary, "path", None) or not getattr(secondary, "path", None):
        return None
    if primary.speed != 1.0 or hi_s - lo_s <= drop_s + 1.0:
        return None
    from ..speech import SPEECH_RATE, centre_activity
    reach = max(abs(shift) for shift in SPEECH_STEP_NULLS_S) + 0.1
    try:
        video = centre_activity(primary.path, primary.track, lo_s, hi_s, aligner.token)
        dub_a = _timeline_activity(secondary, lo_s + before_s - reach, hi_s + before_s + reach, aligner.token)
        dub_b = _timeline_activity(secondary, lo_s + after_s - reach, hi_s + after_s + reach, aligner.token)
    except MediaError:
        return None
    span = int(LOCAL_STATS_S * SPEECH_RATE)
    video, dub_a, dub_b = (_local_standardize(curve, span) for curve in (video, dub_a, dub_b))
    count = len(video)
    skip = int(round(drop_s * SPEECH_RATE))
    last = count - skip
    pad = int(round(reach * SPEECH_RATE))
    if last < 2 or len(dub_a) < 2 * pad + count or len(dub_b) < 2 * pad + count:
        return None

    def rise(shift_s: float) -> Tuple[float, int]:
        start = pad + int(round(shift_s * SPEECH_RATE))
        a = video * dub_a[start : start + count]
        b = video * dub_b[start : start + count]
        ahead = np.concatenate([[0.0], np.cumsum(a)])
        behind = np.concatenate([np.cumsum(b[::-1])[::-1], [0.0]])
        scores = ahead[: last + 1] + behind[skip : skip + last + 1]
        return float(scores.max() - np.median(scores)), int(np.argmax(scores))

    real, at = rise(0.0)
    chance = max(rise(shift)[0] for shift in SPEECH_STEP_NULLS_S)
    if real < SPEECH_STEP_MARGIN * chance:
        return None
    return lo_s + at / SPEECH_RATE


def _choose_cut(
    candidates: Sequence[float], ahead: Optional[np.ndarray], behind: Optional[np.ndarray],
    span_lo: float, gap_s: float, level: float, strict: bool = False,
) -> Optional[float]:
    """The shot change where an edge goes: the one the agreement scores best
    when the agreement says anything (see ``_place_step``), the only one when
    there is one, else none. ``ahead`` is the running agreement of the
    stretch before the edge, ``behind`` of the one after it, from the
    candidate on -- ``gap_s`` later, when the dub lacks that much."""
    rate = ENVELOPE_RATE
    skip = int(round(gap_s * rate))
    sizes = [len(x) for x in (ahead, behind) if x is not None]
    if not sizes:
        return candidates[0] if len(candidates) == 1 and not strict else None
    n = min(sizes) - 1

    def total(index: int) -> float:
        value = 0.0
        if ahead is not None:
            value += float(ahead[index])
        if behind is not None:
            value += float(behind[index + skip])
        return value

    usable = [c for c in candidates if 0 <= int(round((c - span_lo) * rate)) and int(round((c - span_lo) * rate)) + skip <= n]
    if not usable:
        return None
    everything = np.array([total(i) for i in range(0, max(1, n - skip + 1))])
    measured = level >= AGREEMENT_FLOOR and everything.size > 1 and (
        float(everything.max() - np.median(everything)) > STEP_EVIDENCE * level * rate
    )
    if measured:
        best = max(usable, key=lambda c: total(int(round((c - span_lo) * rate))))
        top, low = float(everything.max()), float(np.median(everything))
        if top - total(int(round((best - span_lo) * rate))) > 0.5 * (top - low):
            return None
        if strict and _sound_overrules(ahead, behind, skip, int(round((best - span_lo) * rate)), int(np.argmax(everything))):
            return None
        return best
    return usable[0] if len(usable) == 1 and not strict else None


def _sound_overrules(
    ahead: Optional[np.ndarray], behind: Optional[np.ndarray], skip: int, cut: int, peak: int,
) -> bool:
    """Whether the agreement, between the shot change at index ``cut`` and
    the sound's own best edge at ``peak``, says clearly that the edge is at
    ``peak``: the stretch the move would take the span from agrees better
    there, block after block."""
    lo, hi = min(cut, peak), max(cut, peak)
    block = int(round(PICTURE_VETO_BLOCK_S * ENVELOPE_RATE))
    count = (hi - lo) // block
    if count < 2:
        return False
    # Per sample: how much better the stretch before the edge agrees than
    # the one after it (the running sums' own steps).
    gain = np.zeros(hi - lo)
    if ahead is not None:
        gain += np.diff(ahead[lo : hi + 1])
    if behind is not None:
        gain += np.diff(behind[lo + skip : hi + skip + 1])
    if peak < cut:
        gain = -gain  # the sound keeps the span for the stretch after the edge
    sums = gain[: count * block].reshape(count, block).sum(axis=1)
    spread = float(np.std(sums, ddof=1))
    mean = float(sums.mean())
    if spread <= 1e-12:
        return mean > 0.0
    return mean / (spread / np.sqrt(count)) >= PICTURE_VETO_Z


def _snap_to_picture(
    aligner: _Aligner, before: Optional[Block], after: Optional[Block], primary_s: float, secondary_s: float,
    bracket_before: Optional[Tuple[float, float]] = None, bracket_after: Optional[Tuple[float, float]] = None,
) -> None:
    """Move edges the sound could only place roughly onto their shot change.

    Where the dub runs straight through the cut (the edges are tied, the
    usual case) the cut is one frame of the picture: for a trim of a few
    frames, a single shot change; for a scene missing from the dub, a pair
    of shot changes exactly the missing length apart, which across a span
    of a few seconds is usually the only such pair there is. Otherwise
    each edge is looked for on its own. The sound's own interval is asked
    first; failing that, the whole bracket the readings left the edge in,
    where only a clear choice by the sound is taken. An edge moved here is
    placed to the frame and its uncertainty says so, so it is never trimmed
    back as doubtful and the dub is kept right up to the cut; the cut is
    remembered, so a later pass that fits the edge on the sound again does
    not walk it off the frame.
    """
    picture = aligner.picture
    if picture is None or not picture.available:
        return
    tolerance = picture.tolerance_s

    def put(block: Block, is_end: bool, at: float) -> None:
        if is_end:
            block.end_s = min(at, secondary_s - block.offset_s, primary_s)
            block.end_lo, block.end_hi, block.end_cut = block.end_s - tolerance, block.end_s + tolerance, at
        else:
            block.start_s = max(at, -block.offset_s, 0.0)
            block.start_lo, block.start_hi, block.start_cut = block.start_s - tolerance, block.start_s + tolerance, at

    def windows(at: float, doubt: float, bracket: Optional[Tuple[float, float]]) -> List[Tuple[float, float, bool]]:
        """(lo, hi, needs a clear choice): the sound's interval, then the bracket."""
        out = [(max(0.0, at - doubt), min(primary_s, at + doubt), False)]
        if bracket is not None and (bracket[0] < at - doubt - 0.5 or bracket[1] > at + doubt + 0.5):
            out.append((max(0.0, min(bracket[0], at - doubt)), min(primary_s, max(bracket[1], at + doubt)), True))
        return out

    if before is not None and after is not None:
        video_gap = max(0.0, before.offset_s - after.offset_s)
        tied = abs((after.start_s - before.end_s) - video_gap) <= 0.002
        if tied:
            if before.end_cut is not None and abs(before.end_s - before.end_cut) <= 2 * tolerance:
                put(before, True, before.end_cut)
                put(after, False, before.end_cut + video_gap)
                return
            doubt = max(before.end_uncertainty_s, after.start_uncertainty_s)
            if doubt <= PICTURE_SNAP_MIN_S:
                return
            hole = video_gap if video_gap > TRIM_MAX_S else 0.0
            for lo, hi, strict in windows(before.end_s, doubt, bracket_before):
                # A lone pair of shot changes exactly the missing length
                # apart is evidence on its own; a lone single shot change is
                # not -- there is one every few seconds -- and a trim's cut
                # is only taken when the sound prefers it.
                strict = strict or hole == 0.0
                candidates = picture.pairs(lo, hi, hole)
                if not candidates:
                    continue
                span_lo, span_hi = max(0.0, lo - 1.0), min(primary_s, hi + video_gap + 1.0)
                curve_b = aligner.edge_curve(before.offset_s, span_lo, span_hi)
                curve_a = aligner.edge_curve(after.offset_s, span_lo, span_hi)
                ahead = np.concatenate([[0.0], np.cumsum(curve_b - _agreement_threshold(before.match))]) if curve_b.size else None
                behind = np.concatenate([np.cumsum((curve_a - _agreement_threshold(after.match))[::-1])[::-1], [0.0]]) if curve_a.size else None
                level = max(float(np.abs(curve_b).mean()) if curve_b.size else 0.0,
                            float(np.abs(curve_a).mean()) if curve_a.size else 0.0)
                cut = _choose_cut(candidates, ahead, behind, span_lo, video_gap, level, strict)
                if cut is not None:
                    put(before, True, cut)
                    put(after, False, cut + video_gap)
                    return
            return
    for block, is_end, bracket in ((before, True, bracket_before), (after, False, bracket_after)):
        if block is None:
            continue
        recorded = block.end_cut if is_end else block.start_cut
        at = block.end_s if is_end else block.start_s
        if recorded is not None and abs(at - recorded) <= 2 * tolerance:
            put(block, is_end, recorded)
            continue
        doubt = block.end_uncertainty_s if is_end else block.start_uncertainty_s
        if doubt <= PICTURE_SNAP_MIN_S:
            continue
        for lo, hi, _strict in windows(at, doubt, bracket):
            strict = True  # a single shot change, see above
            candidates = picture.within(lo, hi)
            if not candidates:
                continue
            span_lo, span_hi = max(0.0, lo - 1.0), min(primary_s, hi + 1.0)
            curve = aligner.edge_curve(block.offset_s, span_lo, span_hi)
            if curve.size:
                running = curve - _agreement_threshold(block.match)
                ahead = np.concatenate([[0.0], np.cumsum(running)]) if is_end else None
                behind = None if is_end else np.concatenate([np.cumsum(running[::-1])[::-1], [0.0]])
                cut = _choose_cut(candidates, ahead, behind, span_lo, 0.0, float(np.abs(curve).mean()), strict)
            else:
                cut = candidates[0] if len(candidates) == 1 and not strict else None
            if cut is not None:
                put(block, is_end, cut)
                break


def _step_words(step_s: float) -> str:
    """A step in the offset, in the unit it reads best in."""
    return f"{step_s:+.3f} s" if abs(step_s) >= 1.0 else f"{step_s * 1000.0:+.0f} ms"
