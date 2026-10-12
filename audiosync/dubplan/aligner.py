"""The aligner: every measurement the planner makes of the two tracks.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..segments import find_step
from ..shots import PictureCuts
from ..media import CancellationToken, MediaError, load_audio

from .envelope import _pool
from .plan import _clock
from .scores import Block, _nearest_peak, _noise_units, _parabolic, _peak_dominance, ncc_lags, score_windows
from .tuning import AGREEMENT_FLOOR, ANALYSIS_SR, COARSE_CELL_BUDGET, COARSE_HOP_S, COARSE_POOL, COARSE_WINDOW_S, CONSISTENT_MS, CREDIBILITY_RANGE_S, CREDIBILITY_SPAN_S, DRIFT_NEIGHBOURS, EDGE_FIT_PASSES, EDGE_FIT_POSITION, EFFECTS_DISCOUNT, EFFECTS_DOMINANCE, EFFECTS_RANGE_S, EFFECTS_VOTE_SLACK_S, EFFECTS_Z, ENVELOPE_RATE, EXCURSION_MAX_S, EXCURSION_MIN_MS, LEVEL_MIN_READINGS, LEVEL_SEPARATION, LOCAL_STATS_S, MIN_BLOCK_S, MIN_LEVEL_GAP_MS, POLISH_RANGE_S, POLISH_SPACING_S, POLISH_WIDE_RANGE_S, POLISH_WINDOW_S, READING_Z, RUN_MIN_READINGS, SETTLE_BASELINE_S, SHARPEN_INTERIOR_S, SHARPEN_SEARCH_S, SHARPEN_STATS_S, SHARPEN_Z, SPEECH_ARBITER_S, SPEECH_PREFER_MARGIN, SPEECH_PREFER_Z, STEP_GAIN

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .envelope import TrackEnvelope
    from .scores import ScoreGrid
    from .tuning import LogFn


def _local_standardize(values: np.ndarray, span: int) -> np.ndarray:
    """Zero-mean, unit-variance against a sliding neighbourhood."""
    values = values.astype(np.float64)
    n = len(values)
    if n == 0:
        return values
    half = max(1, span // 2)
    padded = np.concatenate([np.full(half, values[0]), values, np.full(half, values[-1])])
    ones = np.concatenate([[0.0], np.cumsum(padded)])
    squares = np.concatenate([[0.0], np.cumsum(padded * padded)])
    width = 2 * half + 1
    sums = ones[width:] - ones[:-width]
    sums_sq = squares[width:] - squares[:-width]
    mean = sums[:n] / width
    variance = np.maximum(sums_sq[:n] / width - mean * mean, 0.0)
    std = np.sqrt(variance)
    out = np.zeros(n)
    ok = std > 1e-6
    out[ok] = (values[ok] - mean[ok]) / std[ok]
    return out


def _level_runs(positions: Sequence[float], offsets_ms: Sequence[float]) -> List[Tuple[float, float, float]]:
    """Runs of consecutive readings at one offset level, in order.

    The offsets are clustered into levels that are separated by more than
    their own scatter allows (see LEVEL_SEPARATION); a level with fewer
    than LEVEL_MIN_READINGS is noise and its readings are ignored. Each
    remaining reading is labelled with its level, and consecutive readings
    at the same level form a run. Returns (first position, last position,
    level) per run of at least RUN_MIN_READINGS, or an empty list when the
    readings sit on one level or on none -- in which case there is nothing
    to split on, and the single-step test decides.

    The clustering is done on the offsets with their trend taken off. A
    dub at the wrong speed reads as a ramp, and a ramp sorted and broken
    at gaps is a staircase: on a real pair it came back as six "steps" of
    fifty milliseconds a minute apart, which shredded the stretch the
    speed was about to be read from. A ramp is not steps; it is left whole
    and reported as drift. Steps on top of a ramp still show as gaps once
    the ramp is removed. A staircase of flat treads is steps, however
    short the treads: every reading on a tread agrees with its neighbours
    to the reading, which a ramp's never do.
    """
    positions = np.asarray(positions, dtype=np.float64)
    offsets = np.asarray(offsets_ms, dtype=np.float64)

    def cluster(values: np.ndarray) -> List[np.ndarray]:
        """Indices grouped by level, each group in ascending order of value.

        Bottom up: the sorted readings are broken wherever two neighbours
        are more than MIN_LEVEL_GAP_MS apart, then adjacent groups whose
        centres are not separated (see LEVEL_SEPARATION) are merged, the
        least separated pair first, until every adjacent pair is. Top down
        -- splitting at the widest gap whose sides are separated -- went
        wrong on three levels: a side holding two levels has the scatter of
        both, so the true valleys failed the test, and a spurious split
        inside the middle level passed it.
        """

        def scatter(members: np.ndarray) -> float:
            picked = values[members]
            return 1.4826 * float(np.median(np.abs(picked - np.median(picked))))

        def closeness(left: np.ndarray, right: np.ndarray) -> float:
            """Centre distance over the distance that would separate them;
            under one, they are one level."""
            apart = float(np.median(values[right]) - np.median(values[left]))
            return apart / max(MIN_LEVEL_GAP_MS, LEVEL_SEPARATION * max(scatter(left), scatter(right)))

        order = np.argsort(values, kind="stable")
        groups: List[np.ndarray] = []
        cut = 0
        for k in range(1, len(order) + 1):
            if k == len(order) or values[order[k]] - values[order[k - 1]] > MIN_LEVEL_GAP_MS:
                groups.append(order[cut:k])
                cut = k
        while len(groups) > 1:
            ratios = [closeness(groups[k], groups[k + 1]) for k in range(len(groups) - 1)]
            k = int(np.argmin(ratios))
            if ratios[k] >= 1.0:
                break
            groups[k : k + 2] = [np.concatenate([groups[k], groups[k + 1]])]
        return groups

    # The trend is the median slope between readings up to a tenth of the
    # stretch apart. Between all pairs, the pairs that straddle a step vote
    # for a slope, and on a staircase of three long levels they outnumber
    # the pairs within a level: the trend they elected smeared each level
    # into a ramp and no step survived. Between nearest neighbours only,
    # the slope of a noisy pair is too rough -- a few thousandths of a
    # millisecond per second, which is thirty milliseconds of tilt across a
    # feature. Pairs up to a tenth of the stretch apart straddle a step
    # rarely and average that roughness away.
    slope = 0.0
    if len(positions) >= 3 and float(np.ptp(positions)) > 1.0:
        reach = max(DRIFT_NEIGHBOURS, len(positions) // 10)
        pairs = [
            (offsets[j] - offsets[i]) / (positions[j] - positions[i])
            for i in range(len(positions))
            for j in range(i + 1, min(len(positions), i + 1 + reach))
            if positions[j] != positions[i]
        ]
        slope = float(np.median(pairs)) if pairs else 0.0
    detrended = offsets - slope * (positions - positions[0])
    clusters = cluster(detrended)
    labels: List[Optional[int]] = [None] * len(detrended)
    levels = 0
    for members in clusters:
        if len(members) < LEVEL_MIN_READINGS:
            continue
        for index in members:
            labels[int(index)] = levels
        levels += 1
    if levels < 2:
        return []

    runs: List[Tuple[int, int, int]] = []  # first index, last index, level
    for index, which in enumerate(labels):
        if which is None:
            continue
        if runs and runs[-1][2] == which:
            runs[-1] = (runs[-1][0], index, which)
        else:
            runs.append((index, index, which))
    # Drop runs too short to be a scene, then merge runs at one level that
    # a dropped run separated.
    kept: List[Tuple[int, int, int]] = []
    for first, last, which in runs:
        members = sum(1 for k in range(first, last + 1) if labels[k] == which)
        if members < RUN_MIN_READINGS:
            continue
        if kept and kept[-1][2] == which:
            kept[-1] = (kept[-1][0], last, which)
        else:
            kept.append((first, last, which))
    if len(kept) < 2:
        return []
    # Each run's level is the median of its own readings as measured, so a
    # piece carries the offset its material actually sits at.
    return [
        (float(positions[first]), float(positions[last]),
         float(np.median([offsets[k] for k in range(first, last + 1) if labels[k] == which])))
        for first, last, which in kept
    ]


def _timeline_activity(track: TrackEnvelope, lo_s: float, hi_s: float,
                       token: Optional[CancellationToken] = None) -> np.ndarray:
    """Speech activity of a track's centre across [lo_s, hi_s) of the
    timeline its envelope is on, SPEECH_RATE values a second (see
    ``speech.centre_activity``). A dub played at another speed -- 23.976
    against 24, a PAL transfer -- is read across the part of its file that
    lands there (file time t plays at t * speed) and laid on the timeline,
    so its lines are compared where they will be heard: read as it is, a
    23.976 dub's lines drift a frame from the original's every 42 seconds."""
    from ..speech import SPEECH_RATE, centre_activity
    speed = track.speed
    if abs(speed - 1.0) < 1e-12:
        return centre_activity(track.path, track.track, lo_s, hi_s, token)
    native = centre_activity(track.path, track.track, lo_s / speed, hi_s / speed, token)
    want = int(round((hi_s - lo_s) * SPEECH_RATE))
    if native.size == 0:
        return np.zeros(want)
    # Timeline value j sits at lo_s + j / R, which is file time
    # (lo_s + j / R) / speed: index j / speed of the native curve.
    return np.interp(np.arange(want) / speed, np.arange(native.size), native, right=0.0)


class _Aligner:
    """The analysis, held together so its stages can share the envelopes."""

    def __init__(
        self,
        primary: TrackEnvelope,
        secondary: TrackEnvelope,
        token: Optional[CancellationToken],
        log: Optional[LogFn],
        frame_s: Optional[float] = None,
        picture: Optional[PictureCuts] = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.token = token
        self.log = log or (lambda _message: None)
        self.frame_s = frame_s
        """One video frame, when the video's rate is known: the unit a
        picture trim comes in, which tells a trim from a music artefact."""
        self.picture = picture
        """The video's shot changes, read on demand, for putting cuts on the
        frame (see ``shots``); None where there is no picture to ask."""
        self.evidence: List[Tuple[str, float]] = []
        """(band, noise units) of every dense reading taken at the end, for
        saying what the sync rests on (see ``_evidence_note``)."""
        self._standardized: dict = {}
        self._pooled: dict = {}
        self.foreign: List[Tuple[float, float]] = []
        """Spans of the video the dub has something else for (see
        ``_split_foreign``): never bridged by the assembly."""
        self.foreign_limits: Optional[Dict[str, float]] = None
        self._activity: dict = {}

    @property
    def shared_bands(self) -> List[str]:
        """The bands both envelopes carry, ``full`` first."""
        return [b for b in self.primary.bands if b in self.secondary.bands]

    @property
    def detect_bands(self) -> List[str]:
        """The bands a stretch may be *found* by: the full band and the low
        band. The high band is bursts with silence between, whose chance
        correlations are heavy-tailed (peaks of eight or nine floors where
        nothing matches); it arbitrates the offset of a stretch already
        found (see ``_read``), it does not vouch for one."""
        return [b for b in self.shared_bands if b != "high"]

    def onsets(self, which: str, band: str = "full") -> np.ndarray:
        env = self.primary if which == "primary" else self.secondary
        return env.onsets(band)

    def pooled(self, which: str, pool: int, band: str = "full") -> np.ndarray:
        key = (which, pool, band)
        if key not in self._pooled:
            self._pooled[key] = _pool(self.onsets(which, band), pool)
        return self._pooled[key]

    # -- coarse -------------------------------------------------------------

    def coarse(self, search_s: float, progress: Optional[Callable[[float], None]] = None) -> ScoreGrid:
        primary_s = self.primary.duration_s
        secondary_s = self.secondary.duration_s
        # The dub can only sit as far behind as the material it lacks, and
        # as far ahead as the material it has extra, plus the slack. Made
        # asymmetric so a dub ten minutes short does not also search ten
        # minutes in the direction it cannot possibly go.
        lo_s = -(max(0.0, primary_s - secondary_s) + search_s)
        hi_s = max(0.0, secondary_s - primary_s) + search_s

        pool = COARSE_POOL
        while True:
            rate = ENVELOPE_RATE / pool
            window = int(round(COARSE_WINDOW_S * rate))
            hop = max(1, int(round(COARSE_HOP_S * rate)))
            count = max(1, int((primary_s * rate - window) // hop) + 1)
            lags = int(round((hi_s - lo_s) * rate)) + 1
            if count * lags <= COARSE_CELL_BUDGET or pool >= 8 * COARSE_POOL:
                break
            pool *= 2

        self.log(
            f"coarse pass: {count} windows of {COARSE_WINDOW_S:.0f}s every "
            f"{COARSE_HOP_S:.0f}s, offsets {lo_s:+.0f}s to {hi_s:+.0f}s in "
            f"{1000.0 / rate:.0f}ms steps"
        )
        primary = self.pooled("primary", pool)
        secondary = self.pooled("secondary", pool)
        starts = [i * hop for i in range(count)]
        return score_windows(
            primary, secondary, rate, starts, window,
            int(round(lo_s * rate)), int(round(hi_s * rate)),
            self.token, progress,
        )

    # -- polish -------------------------------------------------------------

    def standardized_onsets(self, which: str, band: str = "full") -> np.ndarray:
        key = (which, band)
        if key not in self._standardized:
            self._standardized[key] = _local_standardize(
                self.onsets(which, band), int(LOCAL_STATS_S * ENVELOPE_RATE)
            )
        return self._standardized[key]

    def agreement(
        self, offset_s: float, lo_s: float, hi_s: float, bands: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """How much the two envelopes agree at each 2 ms of this span, at one
        offset: the product of the two standardised onset curves, summed
        over the bands (every shared band unless ``bands`` says which).
        Each band's product averages to its correlation where the tracks
        match and to zero where they do not, so a scene that only the low
        band can see still shows its agreement, and a scene every band
        sees shows more."""
        lo = max(0, int(round(lo_s * ENVELOPE_RATE)))
        hi = min(len(self.primary.onset), int(round(hi_s * ENVELOPE_RATE)))
        if hi <= lo:
            return np.zeros(0)
        shift = int(round(offset_s * ENVELOPE_RATE))
        out = np.zeros(hi - lo)
        for band in (bands if bands is not None else self.shared_bands):
            primary = self.standardized_onsets("primary", band)
            secondary = self.standardized_onsets("secondary", band)
            s_lo, s_hi = lo + shift, hi + shift
            c_lo, c_hi = max(0, s_lo), min(len(secondary), s_hi)
            if c_hi > c_lo:
                out[c_lo - s_lo : c_hi - s_lo] += primary[c_lo - shift : c_hi - shift] * secondary[c_lo:c_hi]
        return out

    def edge_curve(self, offset_s: float, lo_s: float, hi_s: float) -> np.ndarray:
        """The agreement an edge is placed on: the full band's where it has
        any, every band's summed where it has none.

        The full band's transients are two frames wide, so its agreement
        turns at a cut within a frame or two; a band's are twenty frames
        wide, and a band that matches far better than the others -- the
        effects band on a bed the mixes share exactly -- dominates the
        sum, so that its own passing dips read as the edge. Where the
        full band is blind, a scene with no music in it, the sum is what
        there is."""
        sharp = ["full"] + (["side"] if "side" in self.shared_bands else [])
        full = self.agreement(offset_s, lo_s, hi_s, bands=sharp)
        if full.size and float(full.mean()) >= AGREEMENT_FLOOR:
            return full
        return self.agreement(offset_s, lo_s, hi_s)

    def readings(self, block: Block, range_s: float = POLISH_RANGE_S) -> List[Tuple[float, float, float]]:
        """The offset measured at 2 ms at several places inside a stretch.

        Returns (position_s, offset_s, match) per sub-window, one every
        POLISH_SPACING_S along the stretch and never fewer than three where
        the stretch is long enough for three. Each looks ``range_s`` either
        side of the stretch's offset.
        """
        rate = ENVELOPE_RATE
        window = int(POLISH_WINDOW_S * rate)
        margin = int(round(range_s * rate))
        block_lo = int(round(block.start_s * rate))
        block_hi = int(round(block.end_s * rate))
        length = block_hi - block_lo
        if length < rate:  # under a second: nothing to measure
            return []
        window = min(window, length)
        count = max(1, min(length // window, max(3, int(block.length_s / POLISH_SPACING_S))))
        if count == 1:
            starts = [block_lo + (length - window) // 2]
        else:
            step = (length - window) / (count - 1)
            starts = [block_lo + int(round(i * step)) for i in range(count)]

        shift = int(round(block.offset_s * rate))
        found: List[Tuple[float, float, float, str, float]] = []
        for start in starts:
            reading = self._read(start, window, shift, margin)
            if reading is not None:
                found.append((start / rate, *reading))
        return found

    def _read(
        self, start: int, window: int, shift: int, margin: int
    ) -> Optional[Tuple[float, float, str, float]]:
        """One window's offset, as (offset_s, match, band, noise units).

        Every band is correlated and the one that peaks highest above its
        own noise answers -- except that a confident high band (the
        effects, see BANDS) answers regardless, looked for over the wider
        EFFECTS_RANGE_S so a trim the music ran across is still seen.
        Among the effects' peaks within 15% of the tallest, the one nearest
        the stretch's own offset is taken: a beat makes peaks every period,
        all about as tall, and only the nearest is the stretch's.
        """
        rate = ENVELOPE_RATE
        best: Optional[Tuple[float, float, str, float]] = None
        for band in self.shared_bands:
            primary = self.onsets("primary", band)
            secondary = self.onsets("secondary", band)
            template = primary[start : start + window]
            effects = band == "high"
            reach = max(margin, int(round(EFFECTS_RANGE_S * rate))) if effects else margin
            # A search range that runs off an end of the dub is cut to the
            # dub, not skipped: skipped, a stretch at the very start of the
            # file had no reading at all and was dropped as noise.
            first = max(0, start + shift - reach)
            last = min(len(secondary), start + window + shift + reach)
            if last - first < window or len(template) < window:
                continue
            row = ncc_lags(template, secondary[first:last])
            if row.size < 3:
                continue
            units = _noise_units(row, np.ones(len(row), dtype=bool))
            peak = _nearest_peak(row, start + shift - first) if effects else int(np.argmax(row))
            z = float(units[peak])
            lag = (first + _parabolic(row, peak)) - start
            if effects:
                # The effects decide outright when their peak is tall and
                # alone: a band of bursts correlates by chance in several
                # places at once, and none of those is a reading.
                if z >= EFFECTS_Z and _peak_dominance(row, peak) >= EFFECTS_DOMINANCE:
                    return (lag / rate, float(row[peak]), band, z)
                # Otherwise it competes like any band, discounted: its
                # chance peaks run a third taller than the full band's.
                z *= EFFECTS_DISCOUNT
            candidate = (lag / rate, float(row[peak]), band, z)
            if best is None or z > best[3]:
                best = candidate
        return best

    def polish_offset(self, block: Block, wide: bool = False) -> Tuple[float, float, Optional[float]]:
        """Measure a stretch's offset at 2 ms, from several places inside it.

        Returns (offset_s, match, drift_ms_per_s). The offset is the median
        across the sub-windows, weighted by how well each matched, so a
        sub-window that landed on silence does not pull the answer. The
        drift is the slope of offset against position when there are enough
        readings to fit one. ``wide`` searches POLISH_WIDE_RANGE_S either
        side instead of POLISH_RANGE_S.
        """
        found = self.readings(block, POLISH_WIDE_RANGE_S if wide else POLISH_RANGE_S)
        block.readings = len(found)
        block.consistency = 0.0
        if not found:
            return block.offset_s, 0.0, None
        effects = [f for f in found if f[3] == "high" and f[4] >= EFFECTS_Z]
        others = [f for f in found if f not in effects]
        # Only once the stretch is one level: the wide first measurement
        # reads across its steps, where "the rest" is the other side of one.
        if effects and others and not wide:
            said = float(np.median([f[1] for f in effects]))
            rest = float(np.median([f[1] for f in others]))
            if abs(said - rest) >= SPEECH_ARBITER_S and self.speech_choice(block.start_s, block.end_s, said, rest) == rest:
                self.log(
                    f"  {_clock(block.start_s)} - {_clock(block.end_s)}: the effects read {said:+.3f}s, the rest "
                    f"{rest:+.3f}s, and the dialogue goes with the rest"
                )
                effects = []
        block.effects = bool(effects)
        if effects:
            # The effects are the picture's timing; where they were read
            # they decide the offset, and the other readings only say
            # whether the stretch is one level (consistency) or not.
            found = effects + [f for f in found if f not in effects]
            weights_scale = [1.0 if f in effects else 1e-3 for f in found]
        else:
            weights_scale = [1.0] * len(found)
        positions = np.array([f[0] for f in found])
        lags = np.array([f[1] for f in found])
        weights = np.maximum(np.array([f[2] for f in found]), 1e-3) * np.array(weights_scale)
        order = np.argsort(lags)
        cumulative = np.cumsum(weights[order])
        median = lags[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))]
        deciding = lags[: len(effects)] if effects else lags
        block.consistency = float(np.mean(np.abs(deciding - median) * 1000.0 <= CONSISTENT_MS))
        block.spread_ms = float(np.ptp(deciding) * 1000.0)
        drift = None
        if len(found) >= 3 and float(np.ptp(positions)) > 1.0:
            drift = float(np.polyfit(positions, lags * 1000.0, 1)[0])
        return float(median), float(np.median([f[2] for f in found])), drift

    def best_agreement(self, offset_s: float, lo_s: float, hi_s: float) -> float:
        """Mean agreement across the span at the offset, taking the best of
        the frame either side: the offset is measured to a fraction of a
        frame and the curve is only known at whole ones."""
        best = 0.0
        for frame in (-1, 0, 1):
            curve = self.agreement(offset_s + frame / ENVELOPE_RATE, lo_s, hi_s)
            if curve.size:
                best = max(best, float(curve.mean()))
        return best

    def span_scores(
        self, lo_s: float, hi_s: float, centre_s: float, reach_s: float = SETTLE_BASELINE_S,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The whole span [lo_s, hi_s) correlated at every offset within
        ``reach_s`` of ``centre_s``: (offsets, noise units summed over the
        shared bands, the high band discounted as in ``_read``).

        A quiet scene gives each ten-second reading a peak at the noise,
        and their median can land anywhere among those peaks; the whole
        stretch adds up what every window saw at the one offset that is
        right, and only there."""
        rate = ENVELOPE_RATE
        shift = int(round(centre_s * rate))
        reach = int(round(reach_s * rate))
        offsets = (shift + np.arange(-reach, reach + 1)) / rate
        total = np.zeros(len(offsets))
        # Only the part of the span the dub covers at every offset tried:
        # at the dub's first or last seconds some offsets would read past
        # its ends and score nothing, which is not the same as scoring low.
        lo = max(0, int(round(lo_s * rate)), reach - shift)
        hi = min(len(self.primary.onset), int(round(hi_s * rate)), len(self.secondary.onset) - shift - reach - 1)
        if hi - lo < rate:
            return offsets, total
        for band in self.shared_bands:
            template = self.onsets("primary", band)[lo:hi]
            secondary = self.onsets("secondary", band)
            first = lo + shift - reach
            start = max(0, first)
            stop = min(len(secondary), hi + shift + reach + 1)
            if stop - start < len(template):
                continue
            row = ncc_lags(template, secondary[start:stop])
            units = _noise_units(row, np.ones(len(row), dtype=bool)).astype(np.float64)
            if band == "high":
                units *= EFFECTS_DISCOUNT
            skip = start - first
            count = min(len(units), len(total) - skip)
            total[skip : skip + count] += units[:count]
        return offsets, total

    def sharp_match(self, block: Block) -> float:
        """The stretch's median match on the full band alone, from the
        same sub-windows ``readings`` uses (see ``_fine_score``)."""
        rate = ENVELOPE_RATE
        window = int(POLISH_WINDOW_S * rate)
        margin = int(round(POLISH_RANGE_S * rate))
        block_lo = int(round(block.start_s * rate))
        block_hi = int(round(block.end_s * rate))
        length = block_hi - block_lo
        if length < rate:
            return 0.0
        window = min(window, length)
        count = max(1, min(length // window, max(3, int(block.length_s / POLISH_SPACING_S))))
        starts = [block_lo + (length - window) // 2] if count == 1 else [
            block_lo + int(round(i * (length - window) / (count - 1))) for i in range(count)
        ]
        shift = int(round(block.offset_s * rate))
        primary, secondary = self.onsets("primary"), self.onsets("secondary")
        matches = []
        for start in starts:
            template = primary[start : start + window]
            first = max(0, start + shift - margin)
            last = min(len(secondary), start + window + shift + margin)
            if last - first < window or len(template) < window:
                continue
            row = ncc_lags(template, secondary[first:last])
            if row.size:
                matches.append(float(row.max()))
        return float(np.median(matches)) if matches else 0.0

    def credibility(self, block: Block) -> float:
        """How far the stretch's own correlation peak stands above the
        noise of the offsets around it, in the band that sees it best.

        Taken over up to CREDIBILITY_SPAN_S of the stretch's middle and a
        second of offsets either side. A short stretch can correlate well
        by chance in a sparse band -- a few coincident spikes -- and its
        match alone does not say so; the peak's height against the other
        offsets does.
        """
        rate = ENVELOPE_RATE
        length = min(block.length_s, CREDIBILITY_SPAN_S)
        if length < 0.5:
            return 0.0
        middle = (block.start_s + block.end_s) / 2.0
        # The span is kept inside both files: past either end of the dub
        # there is nothing to correlate, and a stretch at the end of the
        # film is measured on what it does cover.
        primary_s = len(self.primary.onset) / rate
        secondary_s = len(self.secondary.onset) / rate
        lo_s = max(0.0, middle - length / 2.0, -block.offset_s)
        hi_s = min(primary_s, middle + length / 2.0, secondary_s - block.offset_s)
        if hi_s - lo_s < 0.5:
            return 0.0
        start = int(round(lo_s * rate))
        window = int(round((hi_s - lo_s) * rate))
        shift = int(round(block.offset_s * rate))
        reach = int(round(CREDIBILITY_RANGE_S * rate))
        best = 0.0
        for band in self.detect_bands:
            primary = self.onsets("primary", band)
            secondary = self.onsets("secondary", band)
            template = primary[start : start + window]
            first = max(0, start + shift - reach)
            last = min(len(secondary), start + window + shift + reach)
            if len(template) < window or last - first < window + 10 or template.std() < 1e-9:
                continue
            row = ncc_lags(template, secondary[first:last])
            if row.size < 10:
                continue
            centre = start + shift - first
            peak = _nearest_peak(row, centre)
            away = np.abs(np.arange(len(row)) - peak) > int(0.25 * rate)
            if away.sum() < 10:
                continue
            noise = float(np.sqrt(np.mean(row[away] ** 2)))
            best = max(best, float(row[peak]) / max(noise, 1e-9))
        return best

    def step_is_real(self, before: Block, after: Block) -> bool:
        """Whether two adjacent pieces really sit at different offsets.

        The shorter piece is taken whole and asked which offset it agrees
        with better, its own or its neighbour's. Its own readings put it
        where it is, but readings are ten-second windows and can agree on
        a wrong peak; the span as a whole cannot.
        """
        short, long = (before, after) if before.length_s <= after.length_s else (after, before)
        own = self.best_agreement(short.offset_s, short.start_s, short.end_s)
        other = self.best_agreement(long.offset_s, short.start_s, short.end_s)
        return own > 0.0 and own >= STEP_GAIN * other

    def split_at_step(self, block: Block, wide: bool = False) -> Optional[List[Block]]:
        """Split a stretch where its own readings say the offset stepped.

        The same question ``segments`` answers for a whole file, asked of
        one stretch: do these readings sit on one level or two? A step
        between two readings puts the cut somewhere between them; the exact
        placement comes afterwards, with the bracket handed over so the
        search covers the whole of that span.
        """
        readings = self.readings(block, POLISH_WIDE_RANGE_S if wide else POLISH_RANGE_S)
        if len(readings) < 4:
            return None
        # Only readings that found something: a sub-window on silence reads
        # a random lag, and one of those is not a step. Judged by how far
        # the reading's peak stands above its row, not by the match: the
        # bands' matches are not on one scale (a broad band peaks higher),
        # and half the median of a mixed set threw out real readings.
        found = [f for f in readings if f[4] >= READING_Z]
        if len(found) < 4:
            return None
        positions = [f[0] for f in found]
        offsets_ms = [f[1] * 1000.0 for f in found]

        # Several levels, visited more than once: a dub conformed scene by
        # scene, with some scenes a few frames out, reads as runs of
        # readings alternating between two offsets. One step cannot
        # describe that, and asked for one step the fit refuses, so the
        # whole stretch would be played at one offset with a third of it a
        # tenth of a second late. Group the readings by level and split at
        # every change of run instead -- after the effects have had their
        # say about which levels are real.
        runs = _level_runs(positions, offsets_ms)
        runs, offsets_ms = self._effects_rule(found, runs, offsets_ms)
        if len(runs) >= 2:
            return self._split_at_runs(block, runs)

        step = find_step(positions, offsets_ms, POLISH_WINDOW_S)
        if step is None or step.lone_group:
            # One reading on its own side of a step is what a reading that
            # locked onto a repeat looks like; it needs company to be a cut.
            return None
        lo_s = max(block.start_s, step.earliest_s - 1.0)
        hi_s = min(block.end_s, step.latest_s + 1.0)
        head = Block(block.start_s, step.split_position_s, step.before_ms / 1000.0,
                     support=block.support, windows=block.windows,
                     start_lo=block.start_lo, start_hi=block.start_hi, end_lo=lo_s, end_hi=hi_s)
        tail = Block(step.split_position_s, block.end_s, step.after_ms / 1000.0,
                     support=block.support, windows=block.windows,
                     start_lo=lo_s, start_hi=hi_s, end_lo=block.end_lo, end_hi=block.end_hi)
        head.offset_s, head.match, _ = self.polish_offset(head)
        tail.offset_s, tail.match, _ = self.polish_offset(tail)
        if not self.step_is_real(head, tail):
            return None
        from .edges import _place_edge
        _place_edge(self, head, tail, self.primary.duration_s, self.secondary.duration_s)
        if head.length_s < MIN_BLOCK_S or tail.length_s < MIN_BLOCK_S:
            return None
        self.log(
            f"  a {step.magnitude_ms:+.0f}ms step inside {_clock(block.start_s)} - {_clock(block.end_s)}"
            f" at {_clock(head.end_s)}"
        )
        return [head, tail]

    def _effects_rule(
        self,
        found: List[Tuple[float, float, float, str, float]],
        runs: List[Tuple[float, float, float]],
        offsets_ms: List[float],
    ) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """Fold the levels the effects contradict into the effects' level.

        Where any run of readings is backed by the high band (see BANDS),
        a run that is not is the music's level, and the music is not the
        picture's timing: its readings are moved to the nearest
        effects-backed level. Two effects-backed levels one after the other
        are both kept, whatever their distance: a step is not always a whole
        number of the video's frames (a real TV dub stepped 52.5 ms back and
        forth, 1.26 frames, in every band), and folding one into the other
        played a scene a quarter of a minute at the wrong level. Runs that
        end up at one level are joined. Returns the runs and the readings'
        offsets as relabelled, so the single-step test that follows sees
        the same picture.
        """
        if len(runs) < 2:
            return runs, offsets_ms

        def members(run: Tuple[float, float, float]) -> List[int]:
            first_s, last_s, level = run
            out = []
            for index, f in enumerate(found):
                if first_s - 1e-6 <= f[0] <= last_s + 1e-6:
                    nearest = min(runs, key=lambda r: abs(r[2] - offsets_ms[index]))
                    if nearest[2] == level:
                        out.append(index)
            return out

        backing = []
        for run in runs:
            effects = [i for i in members(run) if found[i][3] == "high" and found[i][4] >= EFFECTS_Z]
            backing.append((len(effects), sum(found[i][4] for i in effects)))

        # An excursion: a run a good fraction of a second from the level
        # either side of it, over in under a minute, with the level the
        # same before and after. A dub does not gain a second and lose
        # the same second again eighteen seconds later; a cue that repeats
        # at that interval reads exactly so. Unless the effects put it
        # there, it is folded into the level around it. (A step of a
        # frame or two that comes back is a scene conformed a frame out,
        # and is real; the size tells the two apart.)
        levels = [run[2] for run in runs]
        folded = False
        for index in range(1, len(runs) - 1):
            before_level, after_level = levels[index - 1], levels[index + 1]
            size = abs(levels[index] - before_level)
            if (
                abs(before_level - after_level) <= MIN_LEVEL_GAP_MS
                and size >= EXCURSION_MIN_MS
                and runs[index + 1][0] - runs[index][0] <= EXCURSION_MAX_S
                and not backing[index][0]
            ):
                self.log(
                    f"  a {size:+.0f}ms excursion at {_clock(runs[index][0])} returns to "
                    f"{before_level / 1000.0:+.3f}s within {runs[index + 1][0] - runs[index][0]:.0f}s: a repeat, not a cut"
                )
                levels[index] = before_level
                folded = True
        if folded:
            relabelled = list(offsets_ms)
            for index, run in enumerate(runs):
                for i in members(run):
                    relabelled[i] = levels[index]
            joined: List[Tuple[float, float, float]] = []
            for run, level in zip(runs, levels):
                if joined and abs(joined[-1][2] - level) <= 1e-9:
                    joined[-1] = (joined[-1][0], run[1], level)
                else:
                    joined.append((run[0], run[1], level))
            runs, offsets_ms = (joined if len(joined) >= 2 else []), relabelled
            if len(runs) < 2:
                return runs, offsets_ms
            backing = []
            for run in runs:
                effects = [i for i in members(run) if found[i][3] == "high" and found[i][4] >= EFFECTS_Z]
                backing.append((len(effects), sum(found[i][4] for i in effects)))
        if not any(count for count, _ in backing):
            return runs, offsets_ms

        levels = [run[2] for run in runs]
        strong = [index for index, (count, _) in enumerate(backing) if count]
        moved = False
        for index, run in enumerate(runs):
            if backing[index][0]:
                continue
            target = min(strong, key=lambda j: abs(levels[j] - levels[index]))
            if abs(levels[target] - levels[index]) <= MIN_LEVEL_GAP_MS:
                levels[index] = levels[target]
                continue
            # The run had no confident effects reading of its own. Asked
            # on the run's windows, do the effects sit at the run's level
            # (a real step every band agrees on) or at the effects run's
            # level (the music ran across a trim the effects followed)?
            positions = [found[i][0] for i in members(run)]
            if not self._effects_prefer(positions, levels[target], levels[index]):
                continue
            if abs(levels[target] - levels[index]) >= SPEECH_ARBITER_S * 1000.0 and self.speech_choice(
                run[0], run[1] + POLISH_WINDOW_S, levels[index] / 1000.0, levels[target] / 1000.0,
            ) == levels[index] / 1000.0:
                self.log(
                    f"  the music sits at {levels[index] / 1000.0:+.3f}s around {_clock(run[0])}, the effects at "
                    f"{levels[target] / 1000.0:+.3f}s; the dialogue goes with the music there"
                )
                continue
            self.log(
                f"  the music sits at {levels[index] / 1000.0:+.3f}s around {_clock(run[0])}; the effects say "
                f"{levels[target] / 1000.0:+.3f}s, and the dialogue goes with the effects"
            )
            levels[index] = levels[target]
            moved = True
        if not moved and len(set(levels)) == len(levels):
            return runs, offsets_ms
        relabelled = list(offsets_ms)
        for index, run in enumerate(runs):
            for i in members(run):
                relabelled[i] = levels[index]
        joined: List[Tuple[float, float, float]] = []
        for run, level in zip(runs, levels):
            if joined and abs(joined[-1][2] - level) <= 1e-9:
                joined[-1] = (joined[-1][0], run[1], level)
            else:
                joined.append((run[0], run[1], level))
        return (joined if len(joined) >= 2 else []), relabelled

    def speech_choice(self, lo_s: float, hi_s: float, first_s: float, second_s: float) -> Optional[float]:
        """Which of two offsets the dub's lines follow across [lo_s, hi_s),
        or None when the speech does not say (see SPEECH_ARBITER_S)."""
        primary, secondary = self.primary, self.secondary
        if primary.speed != 1.0 or hi_s - lo_s < 5.0:
            return None
        if not getattr(primary, "path", None) or not getattr(secondary, "path", None):
            return None
        from ..speech import SPEECH_RATE, centre_activity
        low, high, pad = min(first_s, second_s), max(first_s, second_s), 1.5
        try:
            video = centre_activity(primary.path, primary.track, lo_s, hi_s, self.token)
            dub = _timeline_activity(secondary, lo_s + low - pad, hi_s + high + pad, self.token)
        except MediaError:
            return None
        row = ncc_lags(video, dub)
        if row.size < 3:
            return None
        offsets = low - pad + np.arange(len(row)) / SPEECH_RATE
        peak = int(np.argmax(row))
        away = np.abs(offsets - offsets[peak]) > 0.15
        if away.sum() < 10:
            return None
        spread = float(np.std(row[away])) + 1e-9
        z = [(float(row[int(np.argmin(np.abs(offsets - level)))]) - float(np.median(row[away]))) / spread
             for level in (first_s, second_s)]
        best = int(np.argmax(z))
        if z[best] >= SPEECH_PREFER_Z and z[best] - z[1 - best] >= SPEECH_PREFER_MARGIN:
            return (first_s, second_s)[best]
        return None

    def _effects_prefer(self, positions: Sequence[float], level_ms: float, other_ms: float) -> bool:
        """Whether the high band, read on windows at ``positions``, agrees
        better at ``level_ms`` than at ``other_ms`` (a twentieth better,
        so a band that sees nothing at either says nothing)."""
        if "high" not in self.shared_bands or not positions:
            return False
        rate = ENVELOPE_RATE
        window = int(POLISH_WINDOW_S * rate)
        slack = int(round(EFFECTS_VOTE_SLACK_S * rate))
        primary = self.onsets("primary", "high")
        secondary = self.onsets("secondary", "high")

        def agreement_at(offset_ms: float) -> float:
            shift = int(round(offset_ms / 1000.0 * rate))
            total = 0.0
            for position in positions:
                start = int(round(position * rate))
                template = primary[start : start + window]
                first, last = start + shift - slack, start + window + shift + slack
                if len(template) < window or first < 0 or last > len(secondary) or template.std() < 1e-9:
                    continue
                row = ncc_lags(template, secondary[first:last])
                if row.size:
                    total += float(row.max())
            return total

        return agreement_at(level_ms) > 1.05 * agreement_at(other_ms)

    def _split_at_runs(self, block: Block, runs: List[Tuple[float, float, float]]) -> Optional[List[Block]]:
        """Cut a stretch into one piece per run of readings at one level.

        Each piece takes the level as its offset and the run's span as its
        extent; the boundary between two pieces lies somewhere between the
        last reading of one run and the first of the next, and is placed
        there exactly afterwards.
        """
        pieces: List[Block] = []
        for index, (first_s, last_s, level_ms) in enumerate(runs):
            start_s = block.start_s if index == 0 else (runs[index - 1][1] + POLISH_WINDOW_S + first_s) / 2.0
            end_s = block.end_s if index == len(runs) - 1 else (last_s + POLISH_WINDOW_S + runs[index + 1][0]) / 2.0
            piece = Block(
                start_s, end_s, level_ms / 1000.0,
                support=block.support, windows=block.windows,
                start_lo=block.start_lo if index == 0 else max(block.start_s, runs[index - 1][1]),
                start_hi=block.start_hi if index == 0 else min(block.end_s, first_s + POLISH_WINDOW_S),
                end_lo=block.end_lo if index == len(runs) - 1 else max(block.start_s, last_s),
                end_hi=block.end_hi if index == len(runs) - 1 else min(block.end_s, runs[index + 1][0] + POLISH_WINDOW_S),
            )
            piece.offset_s, piece.match, _ = self.polish_offset(piece)
            # Measured on its own material, a piece may turn out to sit at
            # the same offset as the one before it -- the level it was
            # grouped by was a reading or two off. That is not a step, and
            # splitting there would put a splice, and a sliver of the
            # original, into an unbroken scene. Nor is a step the shorter
            # piece does not bear out when taken whole; the two are then
            # one piece, measured again as one.
            if pieces and (
                abs(piece.offset_s - pieces[-1].offset_s) * 1000.0 <= CONSISTENT_MS
                or not self.step_is_real(pieces[-1], piece)
            ):
                joined = pieces[-1]
                joined.end_s, joined.end_lo, joined.end_hi = piece.end_s, piece.end_lo, piece.end_hi
                joined.offset_s, joined.match, _ = self.polish_offset(joined)
                continue
            pieces.append(piece)
        if len(pieces) < 2:
            return None
        for index in range(1, len(pieces)):
            from .edges import _place_edge
            _place_edge(self, pieces[index - 1], pieces[index], self.primary.duration_s, self.secondary.duration_s)
        pieces = [piece for piece in pieces if piece.length_s >= MIN_BLOCK_S]
        if len(pieces) < 2:
            return None
        steps = ", ".join(f"{1000.0 * (b.offset_s - a.offset_s):+.0f}ms at {_clock(a.end_s)}" for a, b in zip(pieces, pieces[1:]))
        self.log(f"  the offset steps inside {_clock(block.start_s)} - {_clock(block.end_s)}: {steps}")
        return pieces

    # -- waveform ----------------------------------------------------------

    def waveform(self, which: str, lo_s: float, hi_s: float) -> Optional[np.ndarray]:
        """Decode a span of one track at ANALYSIS_SR, on its analysis timeline.

        For the dub that timeline is its own time times the speed it was
        analysed at, so the seek and the length are divided back out and the
        decoder asked for the compensating rate -- the same trick the
        envelope was built with.
        """
        env = self.primary if which == "primary" else self.secondary
        if hi_s <= lo_s:
            return None
        # Past either end of the file there is silence, which is what a
        # stretch that runs off the end of the dub has to be measured
        # against; so the span is clipped to the file and padded back out.
        clip_lo, clip_hi = max(0.0, lo_s), min(env.duration_s, hi_s)
        want = int(round((hi_s - lo_s) * ANALYSIS_SR))
        if clip_hi - clip_lo < 0.05:
            return np.zeros(want)
        try:
            pcm = load_audio(
                env.path, ANALYSIS_SR, duration=(clip_hi - clip_lo) / env.speed,
                offset=clip_lo / env.speed, token=self.token, track=env.track,
            ).astype(np.float64)
        except MediaError:
            return None
        if abs(env.speed - 1.0) > 1e-12:
            # Stretched here, by the exact factor, rather than by asking the
            # decoder for a compensating rate: that rate has to be a whole
            # number, and rounding it is a drift of a sample every few
            # seconds -- nothing to the envelope, but the whitened waveform
            # decorrelates within two samples, so a span forty seconds long
            # had lost its agreement by the far end.
            n = int(round(len(pcm) * env.speed))
            pcm = np.interp(np.arange(n) / env.speed, np.arange(len(pcm)), pcm)
        lead = int(round((clip_lo - lo_s) * ANALYSIS_SR))
        out = np.zeros(want)
        take = min(len(pcm), want - lead)
        if take > 0:
            out[lead : lead + take] = pcm[:take]
        return out

    def waveform_agreement(
        self, block: Block, lo_s: float, hi_s: float, inside: Tuple[float, float]
    ) -> Optional[Tuple[np.ndarray, float]]:
        """The waveform-level agreement of a stretch across [lo_s, hi_s).

        Returns the per-sample agreement and the waveform correlation it was
        judged by, or None when the two waveforms do not demonstrably share
        anything at this offset -- in which case the envelope's placement is
        the only one there is.

        The sample-exact offset, the polarity and the correlation are taken
        on the few seconds of ``inside`` -- the part of the span known to
        belong to the stretch -- where the envelopes agree most. They are
        taken on the same decode as the span itself: a seek through a lossy
        codec does not always land with the same slop, and a lag measured
        on one decode applied to another can be a few samples out, which
        for whitened signals is the difference between agreement and none.
        """
        primary = self.waveform("primary", lo_s, hi_s)
        secondary = self.waveform("secondary", lo_s + block.offset_s, hi_s + block.offset_s)
        if primary is None or secondary is None:
            return None
        n = min(len(primary), len(secondary))
        # Whitened by a first difference. Audio is not white: its power
        # sits low, so neighbouring samples agree with each other and a
        # second of it has far fewer independent samples than its sample
        # rate says. Flattening the spectrum restores them. Measured on the
        # synthetic pair, the shared bed's correlation doubles and the
        # correlation between unrelated passages drops by two thirds.
        primary = np.diff(primary[:n])
        secondary = np.diff(secondary[:n])
        n -= 1

        # The interior: the SHARPEN_INTERIOR_S window inside `inside` where
        # the envelopes agree most, so it is material the dub really has.
        in_lo, in_hi = max(lo_s, inside[0]), min(hi_s, inside[1])
        length = min(SHARPEN_INTERIOR_S, in_hi - in_lo)
        if length < 1.0:
            return None
        # The full band chooses the interior: the waveform is whitened, and
        # what it can lock onto is what the full band's sharp transients
        # see, not what a band's broad ones do.
        curve = self.agreement(block.offset_s, in_lo, in_hi, bands=["full"])
        window = max(1, int(length * ENVELOPE_RATE))
        if curve.size < window:
            return None
        totals = np.concatenate([[0.0], np.cumsum(curve)])
        means = (totals[window:] - totals[:-window]) / window
        best = int(np.argmax(means))
        if means[best] < AGREEMENT_FLOOR:
            return None
        interior = (in_lo + best / ENVELOPE_RATE, in_lo + best / ENVELOPE_RATE + length)

        i0 = max(0, int(round((interior[0] - lo_s) * ANALYSIS_SR)))
        i1 = min(n, int(round((interior[1] - lo_s) * ANALYSIS_SR)))
        if i1 - i0 < ANALYSIS_SR:
            return None
        a = primary[i0:i1] - primary[i0:i1].mean()
        b = secondary[i0:i1] - secondary[i0:i1].mean()
        reach = int(SHARPEN_SEARCH_S * ANALYSIS_SR)
        a_norm = float(np.sqrt(np.dot(a, a)))
        if a_norm < 1e-9:
            return None
        lags = np.arange(-reach, reach + 1)
        corr = np.zeros(len(lags))
        for index, lag in enumerate(lags):
            aa = a[max(0, lag) : len(a) + min(0, lag)]
            bb = b[max(0, -lag) : len(b) + min(0, -lag)]
            b_norm = float(np.sqrt(np.dot(bb, bb)))
            if b_norm > 1e-9:
                corr[index] = float(np.dot(aa, bb)) / (a_norm * b_norm)
        peak = int(np.argmax(np.abs(corr)))
        z = float(_noise_units(np.abs(corr), np.ones(len(corr), dtype=bool))[peak])
        if z < SHARPEN_Z:
            return None
        polarity = 1.0 if corr[peak] >= 0 else -1.0
        lag = int(lags[peak])
        # Shift the dub by the lag found, so the two line up to the sample.
        # corr[lag] pairs primary[i + lag] with secondary[i], so a positive
        # lag means the dub runs early by that much and is moved later.
        # Whitened signals decorrelate within a sample, so this has to be
        # exact: the original code shifted the other way, which turned a
        # one-sample misalignment into two and flattened the agreement.
        if lag > 0:
            secondary = np.concatenate([np.zeros(lag), secondary[:-lag]])
        elif lag < 0:
            secondary = np.concatenate([secondary[-lag:], np.zeros(-lag)])
        span = int(SHARPEN_STATS_S * ANALYSIS_SR)
        agreement = polarity * _local_standardize(primary, span) * _local_standardize(secondary, span)
        return agreement, abs(float(corr[peak]))


def _agreement_threshold(match: float) -> float:
    """Where to start: a quarter of the match, not half.

    The level of agreement varies along a stretch with what is playing --
    dialogue over quiet ambience masks the shared bed, an action scene does
    not -- so a threshold taken from one place can sit above the agreement
    somewhere else and read a matched passage as unmatched. Starting low
    errs towards keeping the dub; the fit then moves the threshold to the
    midpoint of the levels it actually finds either side of the edge.
    """
    return max(AGREEMENT_FLOOR, EDGE_FIT_POSITION * match)


def _fit_end(curve: np.ndarray, threshold: float, lo: int, hi: int) -> int:
    """Index in [lo, hi] where a stretch's agreement ends, by a two-level fit.

    The cumulative sum of (agreement - threshold) peaks where the level
    drops from "matched" to "not". The threshold that separates the two
    best sits between the two levels, which are only known once the edge
    is; so the two are estimated from each other in turn. The curve may
    run beyond [lo, hi] -- the sums need their run-up -- but the answer is
    confined to it, which is what keeps an edge already placed precisely
    from being moved by a later, coarser pass.
    """
    n = len(curve)
    lo, hi = max(0, min(lo, n)), max(0, min(hi, n))
    if n == 0 or hi < lo:
        return lo
    index = hi
    for _ in range(EDGE_FIT_PASSES):
        total = np.concatenate([[0.0], np.cumsum(curve - threshold)])
        index = lo + int(np.argmax(total[lo : hi + 1]))
        if index <= 0 or index >= n:
            break
        inside = float(curve[:index].mean())
        outside = max(0.0, float(curve[index:].mean()))
        threshold = max(AGREEMENT_FLOOR, outside + EDGE_FIT_POSITION * (inside - outside))
    return index


def _fit_start(curve: np.ndarray, threshold: float, lo: int, hi: int) -> int:
    """Index in [lo, hi] where a stretch's agreement begins; ``_fit_end`` mirrored."""
    n = len(curve)
    if n == 0:
        return lo
    return n - _fit_end(curve[::-1], threshold, n - hi, n - lo)


def _fit_joint(
    before: np.ndarray, after: np.ndarray,
    threshold_before: float, threshold_after: float,
    gap: int, lo: int, hi: int,
) -> int:
    """One cut serving both stretches: the first ends at the index returned,
    within [lo, hi], and the second begins ``gap`` samples later. Levels are
    re-estimated as in ``_fit_end``, each stretch on its own side."""
    n = min(len(before), len(after))
    lo, hi = max(0, lo), min(hi, n - gap)
    if gap >= n or n == 0 or hi < lo:
        return lo
    index = hi
    for _ in range(EDGE_FIT_PASSES):
        rise = np.concatenate([[0.0], np.cumsum(before[:n] - threshold_before)])
        fall = np.concatenate([[0.0], np.cumsum((after[:n] - threshold_after)[::-1])])[::-1]
        combined = rise[: n + 1 - gap] + fall[gap:]
        index = lo + int(np.argmax(combined[lo : hi + 1]))
        if index <= 0 or index + gap >= n:
            break
        inside_b = float(before[:index].mean())
        outside_b = max(0.0, float(before[index:n].mean()))
        threshold_before = max(AGREEMENT_FLOOR, outside_b + EDGE_FIT_POSITION * (inside_b - outside_b))
        inside_a = float(after[index + gap : n].mean())
        outside_a = max(0.0, float(after[:index + gap].mean()))
        threshold_after = max(AGREEMENT_FLOOR, outside_a + EDGE_FIT_POSITION * (inside_a - outside_a))
    return index
