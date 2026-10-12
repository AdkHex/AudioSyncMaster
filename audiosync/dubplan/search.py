"""Finding the dub: the coarse pass, the speed, the wide pass, the gaps.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..framerate import COMMON_RATES, RATIO_TOLERANCE, speed_candidates
from ..media import CancellationToken

from .aligner import _Aligner
from .edges import _credible, _measure_and_split
from .envelope import _pool
from .plan import _clock, _duration
from .scores import Block, _block_bar, _continuing_candidate, _noise_units, _parabolic, _tallest_peaks, best_path, blocks_from_path, ncc_lags, score_windows
from .tuning import EFFECTS_RECOVER_WINDOWS, EFFECTS_RECOVER_Z, ENVELOPE_RATE, GAP_HOP_S, GAP_JUMP_COST, GAP_NULL_COST, GAP_NULL_REWARD, GAP_ONE_SIDED_MAX_S, GAP_POOL, GAP_UNUSED_MAX_S, GAP_WINDOWS_S, MATCH_Z, MIN_BLOCK_S, MIN_DRIFT_MS_PER_S, MIN_DRIFT_SPAN_S, NEAR_OFFSET_S, RECOVER_MIN_S, RECOVER_SLACK_S, RECOVER_Z, SPEED_BAND, SPEED_FINE_GAIN, SPEED_GAIN, SPEED_MIN_Z, SPEED_TRIAL_POOL, SPEED_TRIAL_WINDOWS, SPEED_TRIAL_WINDOW_S, WIDE_HOP_S, WIDE_MIN_WINDOWS, WIDE_PEAKS, WIDE_PEAK_APART_S, WIDE_SAME_OFFSET_S, WIDE_WINDOW_S, WIDE_Z, WOBBLE_COST

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .envelope import TrackEnvelope
    from .scores import ScoreGrid
    from .tuning import LogFn


def _coarse_blocks(
    aligner: _Aligner, search_s: float, report: Callable[[float, str], None], say: LogFn
) -> Tuple[ScoreGrid, List[Block]]:
    """The coarse pass, through to credible stretches with measured offsets."""
    grid = aligner.coarse(search_s, lambda f: report(40 + 20 * f, "finding the offsets"))
    path, starts = best_path(grid.scores)
    outright = sum(1 for i, c in enumerate(path) if c is not None and grid.scores[i, c] >= MATCH_Z)
    lags = grid.scores.shape[1]
    blocks = [
        b for b in blocks_from_path(grid, path, starts)
        if b.support >= _block_bar(b.windows, lags)
    ]
    blocks = _credible(_measure_and_split(aligner, blocks), aligner)
    say(
        f"at {aligner.secondary.speed:.6f}x: {outright} of {len(path)} windows match the dub outright, "
        f"{len(blocks)} credible stretch(es) covering {100 * _coverage(blocks, aligner.primary.duration_s):.0f}%"
    )
    return grid, blocks


def _estimate_drift(blocks: List[Block]) -> Tuple[Optional[float], float]:
    """How fast the offset walks, in ms per second of video, from the
    coarse path's own trace through every long stretch.

    Each stretch gives a Theil-Sen slope over its windows; the slopes are
    combined as a median weighted by how much material each was read
    from. The path's columns are quantised to the coarse grid, but a
    stretch several minutes long walks tens of columns at the speeds that
    matter, and a cut inside a stretch cannot tilt a median of slopes the
    way it would tilt a least-squares line.

    Returns (drift, span of stretches it was read from), or (None, 0).
    """
    slopes: List[Tuple[float, float]] = []
    for block in blocks:
        trace = block.trace
        if len(trace) < 6:
            continue
        span = trace[-1][0] - trace[0][0]
        if span < 60.0:
            continue
        pairs = [
            (trace[j][1] - trace[i][1]) / (trace[j][0] - trace[i][0])
            for i in range(len(trace))
            for j in range(i + 1, len(trace))
            if trace[j][0] != trace[i][0]
        ]
        if pairs:
            slopes.append((float(np.median(pairs)) * 1000.0, span))
    if not slopes:
        return None, 0.0
    values = np.array([s for s, _ in slopes])
    weights = np.array([w for _, w in slopes])
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    drift = float(values[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])
    return drift, float(weights.sum())


def _snap_speed(speed: float) -> float:
    """The standard frame-rate conversion nearest a measured speed, if one
    lies within tolerance; otherwise the speed as measured. 1000/1001 is
    what a 24 -> 23.976 conversion really is, and applying the measured
    0.99904 instead would leave 0.04 ms/s behind."""
    for numerator in COMMON_RATES:
        for denominator in COMMON_RATES:
            exact = float(numerator / denominator)
            if abs(exact - speed) <= RATIO_TOLERANCE:
                return exact
    return speed


def _fine_score(blocks: List[Block], aligner: _Aligner) -> float:
    """How well the stretches measure at 2ms: the length-weighted median
    full-band match of every stretch long enough to have been measured
    properly. The full band only: its transients are two frames wide, so
    a speed error of a millisecond a second smears them across a window
    and the match collapses, which is the symptom being judged; a band's
    onsets are twenty frames wide and hardly notice."""
    long = [(aligner.sharp_match(b), b.length_s) for b in blocks if b.windows >= 3 and b.length_s > 0]
    if not long:
        return 0.0
    values = np.array([m for m, _ in long])
    weights = np.array([w for _, w in long])
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def _speed_from_drift(
    aligner: _Aligner,
    primary: TrackEnvelope,
    secondary_native: TrackEnvelope,
    blocks: List[Block],
    search_s: float,
    report: Callable[[float, str], None],
    say: LogFn,
    token: Optional[CancellationToken],
) -> Optional[Tuple[TrackEnvelope, _Aligner, ScoreGrid, List[Block]]]:
    """Cancel a small speed difference the coarse pass tolerated.

    Returns the dub at the new speed with its aligner, grid and stretches,
    or None when there is no drift worth cancelling or cancelling it did
    not help.
    """
    drift, span = _estimate_drift(blocks)
    if drift is None or span < MIN_DRIFT_SPAN_S:
        return None
    say(f"the offset drifts by {drift:+.2f} ms/s across {_duration(span)} of matched dub")
    if abs(drift) < MIN_DRIFT_MS_PER_S:
        return None
    # The dub's clock runs at (1 + drift/1000) times the video's, so it is
    # played at the reciprocal to bring it onto the video's clock.
    current = aligner.secondary.speed
    candidate = _snap_speed(current / (1.0 + drift / 1000.0))
    if abs(candidate - current) < 1e-9:
        return None
    baseline = _fine_score(blocks, aligner)
    trial = secondary_native.at_speed(candidate)
    trial_aligner = _Aligner(primary, trial, token, say, aligner.frame_s, aligner.picture)
    grid, trial_blocks = _coarse_blocks(trial_aligner, search_s, report, say)
    score = _fine_score(trial_blocks, trial_aligner)
    say(f"  at {candidate:.6f}x the stretches measure at {score:.2f}, against {baseline:.2f} as they were")
    if not trial_blocks or score < SPEED_FINE_GAIN * baseline:
        return None
    say(f"dub runs at {candidate:.6f}x the video's speed")
    return trial, trial_aligner, grid, trial_blocks


def _wide_blocks(
    aligner: _Aligner,
    report: Callable[[float, str], None],
    say: LogFn,
    anchors: Sequence[Block] = (),
) -> List[Block]:
    """Stretches found by searching every window of the video across the
    whole dub at 2 ms (see WIDE_PASS_COVERAGE).

    Each window keeps its best offset when it stands WIDE_Z above the noise
    of the whole search; runs of consecutive windows agreeing on an offset
    are stretches, bracketed the way the coarse pass brackets its own. The
    stretches are measured and their edges placed afterwards like any other.

    "Best" is not simply tallest. A cue that recurs -- a theme heard four
    times, an episode's opening -- correlates at every occurrence, and the
    tallest peak is the occurrence mixed loudest, not the one this window
    belongs to. So every window keeps its few tallest peaks, and the one
    taken is the one that continues what is already known: within
    WIDE_LOCAL_S of the offset the coarse stretches (``anchors``) give the
    neighbourhood, when such a peak is credible on its own; failing that,
    one that does not wind the dub back a little way from the previous
    window (a repeat within the same scene rewinds by a minute or two; a
    dub made of episodes out of order rewinds by a whole episode, which is
    allowed); failing that, the tallest.
    """
    rate = ENVELOPE_RATE
    window = int(WIDE_WINDOW_S * rate)
    hop = int(WIDE_HOP_S * rate)
    primary_n = len(aligner.primary.onset)
    secondary_n = len(aligner.secondary.onset)
    if primary_n < window or secondary_n < window:
        return []
    starts = list(range(0, primary_n - window + 1, hop))
    possible = np.ones(secondary_n - window + 1, dtype=bool)
    apart = int(WIDE_PEAK_APART_S * rate)
    hits: List[Tuple[float, float, float]] = []  # (window start, offset, noise units)
    previous: Optional[Tuple[float, float]] = None  # (window start, offset) of the last hit
    for index, start in enumerate(starts):
        if aligner.token:
            aligner.token.raise_if_cancelled()
        candidates: List[Tuple[float, float]] = []  # (offset, noise units)
        for band in aligner.detect_bands:
            template = aligner.onsets("primary", band)[start : start + window]
            row = ncc_lags(template, aligner.onsets("secondary", band))
            if row.size < 3:
                continue
            units = _noise_units(row, possible)
            for peak in _tallest_peaks(units, WIDE_PEAKS, apart):
                z = float(units[peak])
                if z < WIDE_Z:
                    break
                offset = (_parabolic(row, peak) - start) / rate
                near = [c for c in candidates if abs(c[0] - offset) <= WIDE_SAME_OFFSET_S]
                if near:
                    if z > near[0][1]:
                        candidates[candidates.index(near[0])] = (offset, z)
                else:
                    candidates.append((offset, z))
        report((index + 1) / len(starts))
        if not candidates:
            continue
        position = start / rate
        chosen = _continuing_candidate(candidates, position, anchors, previous)
        hits.append((position, chosen[0], chosen[1]))
        previous = (position, chosen[0])

    blocks: List[Block] = []
    run: List[Tuple[float, float, float]] = []

    def close() -> None:
        if len(run) >= WIDE_MIN_WINDOWS:
            blocks.append(Block(
                start_s=run[0][0],
                end_s=run[-1][0] + WIDE_WINDOW_S,
                offset_s=float(np.median([h[1] for h in run])),
                support=float(np.mean([h[2] for h in run])),
                windows=len(run),
                start_lo=max(0.0, run[0][0] - WIDE_HOP_S),
                start_hi=run[0][0] + WIDE_WINDOW_S,
                end_lo=run[-1][0],
                end_hi=run[-1][0] + WIDE_WINDOW_S + WIDE_HOP_S,
                trace=[(h[0], h[1]) for h in run],
            ))
        run.clear()

    for hit in hits:
        if run and (
            abs(hit[1] - run[-1][1]) > WIDE_SAME_OFFSET_S
            or hit[0] - run[-1][0] > 2.0 * WIDE_HOP_S + 1e-6
        ):
            close()
        run.append(hit)
    close()
    say(
        f"wide pass: {len(hits)} of {len(starts)} windows of {WIDE_WINDOW_S:.0f}s found in the dub "
        f"at {WIDE_Z:.0f}+ noise units, {len(blocks)} stretch(es) of {WIDE_MIN_WINDOWS}+ windows"
    )
    return blocks


def _merge_blocks(coarse: List[Block], wide: List[Block]) -> List[Block]:
    """The coarse stretches and the wide pass's together, without overlap.

    Where the two overlap at one offset they are one stretch, spanning
    both. Where they overlap at different offsets the wide pass's is kept:
    it was measured at 2 ms across the whole dub, where the coarse pass
    coasts over windows with no evidence and searches only where a dub of
    the same edit could be.
    """
    from_wide = {id(b) for b in wide}
    kept: List[Block] = []
    for block in sorted(coarse + wide, key=lambda b: (b.start_s, b.end_s)):
        if not kept:
            kept.append(block)
            continue
        last = kept[-1]
        overlap = min(last.end_s, block.end_s) - max(last.start_s, block.start_s)
        if overlap <= 0.0:
            kept.append(block)
            continue
        if abs(block.offset_s - last.offset_s) <= NEAR_OFFSET_S:
            wide_one = block if id(block) in from_wide else last
            last.start_s = min(last.start_s, block.start_s)
            last.end_s = max(last.end_s, block.end_s)
            last.start_lo = min(last.start_lo, block.start_lo)
            last.end_hi = max(last.end_hi, block.end_hi)
            last.offset_s = wide_one.offset_s
            last.support = max(last.support, block.support)
            last.windows += block.windows
            last.trace = sorted(last.trace + block.trace)
            continue
        if id(block) in from_wide and id(last) not in from_wide:
            loser, winner = last, block
        elif id(last) in from_wide and id(block) not in from_wide:
            loser, winner = block, last
        else:
            # Both from one pass: the longer stays whole.
            loser, winner = (block, last) if last.length_s >= block.length_s else (last, block)
        if loser is last:
            last.end_s = min(last.end_s, winner.start_s)
            last.end_lo = min(last.end_lo, last.end_s)
            last.end_hi = min(last.end_hi, winner.start_s)
            if last.length_s < MIN_BLOCK_S:
                kept.pop()
            kept.append(block)
        else:
            block.start_s = max(block.start_s, winner.end_s)
            block.start_hi = max(block.start_hi, block.start_s)
            block.start_lo = max(block.start_lo, winner.end_s)
            if block.length_s >= MIN_BLOCK_S:
                kept.append(block)
    return kept


def _coverage(blocks: List[Block], duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return min(1.0, sum(b.length_s for b in blocks) / duration_s)


def _fps_speed_candidates(video_rate: Fraction) -> List[Tuple[float, float]]:
    """Every mastering rate worth trying for a dub against a video of known
    frame rate, as (playback speed, mastering rate) pairs, likeliest first.

    The video's own metadata says its rate; the dub's rate is one of the
    standard ones, so the possible speeds are exactly audio/video for each
    standard audio rate. Ordering by how close each is to 1.0 tries the
    matched rate and the small conversions before the large ones.
    """
    pairs = [
        (float(audio_rate / video_rate), float(audio_rate))
        for audio_rate in COMMON_RATES
        if audio_rate != video_rate
        and abs(float(audio_rate / video_rate) - 1.0) <= SPEED_BAND
    ]
    pairs.sort(key=lambda pair: abs(pair[0] - 1.0))
    return pairs


def _try_speeds(
    aligner: _Aligner,
    primary: TrackEnvelope,
    secondary_native: TrackEnvelope,
    search_s: float,
    say: LogFn,
    candidates: Optional[List[Tuple[float, float]]] = None,
    trials: Optional[dict] = None,
) -> Optional[TrackEnvelope]:
    """Find a standard playback speed at which the pair correlates.

    The dub's envelope is resampled rather than the dub re-decoded, since a
    speed is a time-stretch and the envelope is a curve in time. Three
    windows in the middle of the file are enough to ask: at the right speed
    the peak stands well clear of every other.

    ``candidates`` narrows the question to explicit (speed, mastering rate)
    pairs -- the frame-rate verdict the video's own metadata suggests. When
    omitted, every standard conversion is tried, ordered by the durations.
    """
    if candidates is None:
        candidates = [
            (float(c), 0.0)
            for c in speed_candidates(primary.duration_s, secondary_native.duration_s)
        ]
    candidates = [
        (speed, rate) for speed, rate in candidates
        if abs(speed - 1.0) <= SPEED_BAND and abs(speed - 1.0) > 1e-9
    ]
    if not candidates:
        return None
    pool = SPEED_TRIAL_POOL
    rate = ENVELOPE_RATE / pool
    window_s = min(SPEED_TRIAL_WINDOW_S, primary.duration_s / (SPEED_TRIAL_WINDOWS + 1))
    window = int(round(window_s * rate))
    if window < int(5.0 * rate):
        return None
    usable = primary.duration_s - window_s
    picks = [int(round((usable * (i + 1)) / (SPEED_TRIAL_WINDOWS + 1) * rate)) for i in range(SPEED_TRIAL_WINDOWS)]

    def trial_peak(envelope: TrackEnvelope, bands: Sequence[str] = ("full",)) -> float:
        """The trial windows' typical peak. With several bands, each window
        counts its strongest band: a dub re-timed with its pitch kept (a
        PAL transfer done the modern way) has its transients smeared by the
        splicing, so the full mix barely correlates, while the bass keeps
        its shape."""
        lo_s = -(max(0.0, primary.duration_s - envelope.duration_s) + search_s)
        hi_s = max(0.0, envelope.duration_s - primary.duration_s) + search_s
        best_per_window: Optional[np.ndarray] = None
        for band in bands:
            if band != "full" and (band not in primary.band_onset or band not in envelope.band_onset):
                continue
            scores = score_windows(
                _pool(primary.onsets(band), pool), _pool(envelope.onsets(band), pool), rate, picks, window,
                int(round(lo_s * rate)), int(round(hi_s * rate)), aligner.token,
            ).scores.max(axis=1)
            best_per_window = scores if best_per_window is None else np.maximum(best_per_window, scores)
        return float(np.median(best_per_window)) if best_per_window is not None else 0.0

    def ask(bands: Sequence[str]) -> Optional[Tuple[float, TrackEnvelope]]:
        baseline = trial_peak(aligner.secondary, bands)
        say(f"  at the files' own speed the trial windows peak at {baseline:.1f} noise units")
        if trials is not None:
            trials[aligner.secondary.speed] = max(baseline, trials.get(aligner.secondary.speed, 0.0))
        best: Optional[Tuple[float, TrackEnvelope]] = None
        for candidate, _mastering_rate in candidates:
            trial = secondary_native.at_speed(candidate)
            peak = trial_peak(trial, bands)
            say(f"  trying {candidate:.6f}x: peak {peak:.1f} noise units")
            if trials is not None:
                trials[candidate] = max(peak, trials.get(candidate, 0.0))
            if peak >= max(SPEED_MIN_Z, SPEED_GAIN * baseline) and (best is None or peak > best[0]):
                best = (peak, trial)
        return best

    best = ask(("full",))
    if best is None:
        others = [b for b in aligner.detect_bands if b != "full"]
        if others:
            say(f"  the full mix does not settle the speed; asking the {' and the '.join(others)} band as well")
            best = ask(["full"] + others)
    if best is not None:
        say(f"dub runs at {best[1].speed:.6f}x the video's speed")
        return best[1]
    return None


def _recover_from_gaps(
    aligner: _Aligner, blocks: List[Block], primary_s: float, secondary_s: float
) -> List[Block]:
    """Search every gap for the dub with short windows, over the few offsets
    the neighbouring stretches allow.

    The coarse pass is built to find long stretches against a quarter of a
    million possible offsets, which is why it needs long windows and a
    high price on every jump. Inside a gap the question is far smaller:
    the dub material for this span, if it exists, sits between where the
    stretch before ends and the stretch after begins, give or take a few
    seconds. Against a few hundred offsets a ten-second window is
    conclusive, a jump is cheap, and a run of nothing is called a gap after
    a few windows -- so a span with three cuts inside it comes back as
    three stretches at three offsets, each with its own bracket.
    """
    pool = GAP_POOL
    rate = ENVELOPE_RATE / pool
    found: List[Block] = []

    gaps: List[Tuple[float, float, Optional[Block], Optional[Block]]] = []
    cursor = 0.0
    for index, block in enumerate(blocks):
        if block.start_s - cursor >= RECOVER_MIN_S:
            gaps.append((cursor, block.start_s, blocks[index - 1] if index else None, block))
        cursor = max(cursor, block.end_s)
    if primary_s - cursor >= RECOVER_MIN_S:
        gaps.append((cursor, primary_s, blocks[-1] if blocks else None, None))

    for gap_lo, gap_hi, before, after in gaps:
        gap_s = gap_hi - gap_lo
        # The offsets the dub material for this gap could sit at. With a
        # neighbour on each side, between their offsets: the dub may lack
        # some of the gap (the offset falls across it) or have extra (it
        # rises), but it cannot pass either neighbour's offset by more than
        # the slack. With a neighbour on one side only, the gap's own
        # length either way, since that is the most the dub could lack or
        # have extra here.
        offsets = [b.offset_s for b in (before, after) if b is not None]
        narrow: Optional[Tuple[float, float]] = None
        if len(offsets) == 2:
            lo_s = min(offsets) - RECOVER_SLACK_S
            hi_s = max(offsets) + RECOVER_SLACK_S
            narrow = (lo_s, hi_s)
            # Unless the gap holds both at once -- a cut and an insert, a
            # TV edit's usual -- when its pieces sit anywhere the dub not
            # yet played between the two neighbours can reach: from the
            # dub where the stretch before ends to where the one after
            # begins, laid anywhere in the gap.
            dub_lo = before.end_s + before.offset_s
            dub_hi = after.start_s + after.offset_s
            if dub_hi - dub_lo > RECOVER_SLACK_S:
                lo_s = max(min(lo_s, dub_lo - gap_hi - RECOVER_SLACK_S), min(offsets) - GAP_UNUSED_MAX_S)
                hi_s = min(max(hi_s, dub_hi - gap_lo + RECOVER_SLACK_S), max(offsets) + GAP_UNUSED_MAX_S)
        else:
            reach = min(gap_s, GAP_ONE_SIDED_MAX_S) + RECOVER_SLACK_S
            lo_s = offsets[0] - reach
            hi_s = offsets[0] + reach
        lo_s = max(lo_s, -gap_hi)  # dub time cannot go negative
        hi_s = min(hi_s, secondary_s - gap_lo)
        if hi_s <= lo_s:
            continue

        # Finest scale first; each coarser scale only searches what the
        # finer ones left, so a loud scene's cuts are placed by the short
        # windows and a quiet scene is still found by the long ones.
        uncovered: List[Tuple[float, float]] = [(gap_lo, gap_hi)]
        in_gap: List[Block] = []
        for window_s in GAP_WINDOWS_S:
            remaining: List[Tuple[float, float]] = []
            for sub_lo, sub_hi in uncovered:
                if sub_hi - sub_lo < RECOVER_MIN_S:
                    continue
                pieces = _search_gap(
                    aligner, pool, rate, sub_lo, sub_hi, lo_s, hi_s,
                    min(window_s, sub_hi - sub_lo), secondary_s,
                )
                in_gap.extend(pieces)
                cursor = sub_lo
                for piece in sorted(pieces, key=lambda b: b.start_s):
                    if piece.start_s - cursor >= RECOVER_MIN_S:
                        remaining.append((cursor, piece.start_s))
                    cursor = max(cursor, piece.end_s)
                if sub_hi - cursor >= RECOVER_MIN_S:
                    remaining.append((cursor, sub_hi))
            uncovered = remaining
            if not uncovered:
                break
        found.extend(_consistent_in_gap(in_gap, narrow, [b for b in (before, after) if b is not None]))
    return found


def _consistent_in_gap(pieces: List[Block], narrow: Optional[Tuple[float, float]],
                       neighbours: List[Block]) -> List[Block]:
    """The pieces found in one gap that can stand together. One found at an
    offset near its neighbours' is kept as found; one the wider search
    found further out (see ``_recover_from_gaps``) only where it plays dub
    nothing else claims, in the dub's own order against the neighbours and
    the near pieces. A film repeats its music: on a real episode the wider
    search matched a scene the dub lacks to a cue the dub plays a minute
    later, and that piece then took its dub from the true stretch."""
    if narrow is None:
        return pieces
    near = [p for p in pieces if narrow[0] <= p.offset_s <= narrow[1]]
    anchors = neighbours + near
    kept = list(near)
    for piece in sorted((p for p in pieces if not narrow[0] <= p.offset_s <= narrow[1]), key=lambda p: -p.support):
        d0, d1 = piece.start_s + piece.offset_s, piece.end_s + piece.offset_s
        fits = True
        for other in anchors:
            a0, a1 = other.start_s + other.offset_s, other.end_s + other.offset_s
            if min(d1, a1) - max(d0, a0) > RECOVER_SLACK_S:
                fits = False  # the same dub twice
            elif other.end_s <= piece.start_s + 1e-6 and a1 > d0 + RECOVER_SLACK_S:
                fits = False  # before it in the video, after it in the dub
            elif other.start_s >= piece.end_s - 1e-6 and a0 < d1 - RECOVER_SLACK_S:
                fits = False  # after it in the video, before it in the dub
            if not fits:
                break
        if fits:
            kept.append(piece)
            anchors.append(piece)
    return sorted(kept, key=lambda p: p.start_s)


def _gap_candidates(grid: ScoreGrid, floor: float, window_s: float) -> List[Block]:
    """The stretches one gap-search grid supports: the best path through
    it, cut into runs, each kept when it clears the bar for its length."""
    path, run_starts = best_path(grid.scores, GAP_JUMP_COST, GAP_NULL_REWARD, GAP_NULL_COST, WOBBLE_COST)
    lags = grid.scores.shape[1]
    return [
        block for block in blocks_from_path(grid, path, run_starts)
        if block.support >= max(floor, _block_bar(block.windows, lags, window_s / GAP_HOP_S))
    ]


def _search_gap(
    aligner: _Aligner,
    pool: int,
    rate: float,
    gap_lo: float,
    gap_hi: float,
    lo_s: float,
    hi_s: float,
    window_s: float,
    secondary_s: float,
) -> List[Block]:
    """One scale of the gap search: windows of ``window_s`` every GAP_HOP_S
    across [gap_lo, gap_hi), over offsets [lo_s, hi_s].

    Every band is scored and the scores are summed, in noise units, and
    put back on a unit floor: a scene with no music is found by the low
    band, one with no bass by the full band. The sum rather than the best,
    because a sparse band -- a few transients in a quiet scene -- has a
    heavy-tailed noise, and its lone coincidences stood up as stretches
    when the best band was taken; a real stretch shows in more than one
    band, and the sum keeps that and not the coincidence.
    """
    window = max(1, int(round(window_s * rate)))
    hop = max(1, int(round(GAP_HOP_S * rate)))
    first = int(round(gap_lo * rate))
    last = int(round(gap_hi * rate)) - window
    starts = list(range(first, max(first, last) + 1, hop))
    grid: Optional[ScoreGrid] = None
    bands = aligner.detect_bands
    for band in bands:
        scored = score_windows(
            aligner.pooled("primary", pool, band), aligner.pooled("secondary", pool, band), rate,
            starts, window, int(round(lo_s * rate)), int(round(hi_s * rate)), aligner.token,
        )
        if grid is None:
            grid = scored
        else:
            possible = (grid.scores > -1e5) & (scored.scores > -1e5)
            grid.scores[possible] += scored.scores[possible]
            grid.scores[~possible] = -1e6
    assert grid is not None
    if len(bands) > 1:
        possible = grid.scores > -1e5
        grid.scores[possible] /= math.sqrt(len(bands))
    found: List[Block] = []
    candidates = _gap_candidates(grid, RECOVER_Z, window_s)
    if "high" in aligner.shared_bands:
        # The effects on their own, held to a higher bar: a scene whose bed
        # is footsteps and room tone shows in the 4-8 kHz band alone, but
        # that band's chance peaks are tall (see ``detect_bands``), so only
        # a run of several windows that match outright is believed, and
        # only where the other bands found nothing.
        effects = score_windows(
            aligner.pooled("primary", pool, "high"), aligner.pooled("secondary", pool, "high"), rate,
            starts, window, int(round(lo_s * rate)), int(round(hi_s * rate)), aligner.token,
        )
        for block in _gap_candidates(effects, EFFECTS_RECOVER_Z, window_s):
            if block.windows < EFFECTS_RECOVER_WINDOWS:
                continue
            if any(min(block.end_s, c.end_s) - max(block.start_s, c.start_s) > 0.0 for c in candidates):
                continue
            candidates.append(block)
    for block in sorted(candidates, key=lambda b: b.start_s):
        block.start_s = max(gap_lo, block.start_s, -block.offset_s)
        block.end_s = min(gap_hi, block.end_s, secondary_s - block.offset_s)
        if block.end_s - block.start_s < MIN_BLOCK_S:
            continue
        block.start_lo = max(gap_lo, block.start_lo)
        block.end_hi = min(gap_hi, block.end_hi)
        aligner.log(
            f"  found dub for {_clock(block.start_s)} - {_clock(block.end_s)} at "
            f"{_clock(block.start_s + block.offset_s)} ({block.offset_s:+.2f}s, "
            f"{block.support:.1f} noise units over {block.windows} windows of {window_s:.0f}s)"
        )
        found.append(block)
    return found
