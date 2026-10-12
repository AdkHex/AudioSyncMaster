"""Measuring the written track against the video.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..media import CancellationToken

from .envelope import _pool
from .plan import _clock
from .scores import _nearest_peak, _noise_units, _parabolic, _peak_dominance, ncc_lags
from .tuning import AUDIBLE_MS, EFFECTS_DOMINANCE, EFFECTS_Z, ENVELOPE_RATE, MATCH_Z, STRETCH_MS, VERIFY_SPOTS, VERIFY_SPOT_RANGE_S, VERIFY_SPOT_WINDOW_S, VERIFY_STRETCH_WINDOWS, VERIFY_SWEEP_POOL, VERIFY_SWEEP_RANGE_S, VERIFY_SWEEP_WINDOW_S, VERIFY_SWEEP_Z

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .envelope import TrackEnvelope
    from .plan import DubSyncPlan


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class SpotCheck:
    position_s: float
    residual_ms: Optional[float]
    match: float
    note: str = ""


@dataclass
class Stretch:
    start_s: float
    end_s: float
    residual_ms: float
    windows: int


@dataclass
class Verification:
    """How the finished track sits against the original, measured."""

    spots: List[SpotCheck] = field(default_factory=list)
    sweep_windows: int = 0
    sweep_measured: int = 0
    sweep_within_audible: int = 0
    sweep_within_1ms: int = 0
    sweep_within_5ms: int = 0
    sweep_typical_ms: Optional[float] = None
    sweep_worst_ms: Optional[float] = None
    stretches: List[Stretch] = field(default_factory=list)
    fills_skipped: int = 0
    lines: Optional[dict] = None
    """The line check (see ``linecheck``): where the dub's lines sit against
    the original's, which are in sync with the lips."""
    lines_text: Optional[str] = None

    @property
    def typical_ms(self) -> Optional[float]:
        values = [abs(s.residual_ms) for s in self.spots if s.residual_ms is not None]
        return float(np.median(values)) if values else None

    @property
    def worst_ms(self) -> Optional[float]:
        values = [abs(s.residual_ms) for s in self.spots if s.residual_ms is not None]
        return float(max(values)) if values else None

    def to_dict(self) -> dict:
        return {
            "spots": [
                {"positionS": s.position_s, "residualMs": s.residual_ms, "match": s.match, "note": s.note}
                for s in self.spots
            ],
            "typicalMs": self.typical_ms,
            "worstMs": self.worst_ms,
            "sweepWindows": self.sweep_windows,
            "sweepMeasured": self.sweep_measured,
            "sweepWithinAudible": self.sweep_within_audible,
            "sweepWithin1Ms": self.sweep_within_1ms,
            "sweepWithin5Ms": self.sweep_within_5ms,
            "sweepTypicalMs": self.sweep_typical_ms,
            "sweepWorstMs": self.sweep_worst_ms,
            "stretches": [
                {"startS": s.start_s, "endS": s.end_s, "residualMs": s.residual_ms, "windows": s.windows}
                for s in self.stretches
            ],
            "lines": self.lines,
        }

    def describe(self) -> str:
        lines = []
        measured = [s for s in self.spots if s.residual_ms is not None]
        if measured:
            lines.append(
                f"the finished track sits {_ms(self.typical_ms)}ms from the original typically and "
                f"{_ms(self.worst_ms)}ms at its worst, measured at {len(measured)} spots. "
                f"Lip sync starts to show around {AUDIBLE_MS:.0f}ms; the measurement resolves a fraction of a millisecond."
            )
        else:
            lines.append("none of the spot checks could be measured")
        for spot in self.spots:
            if spot.residual_ms is None:
                lines.append(f"  at {_clock(spot.position_s)}: {spot.note}")
            else:
                direction = "early" if spot.residual_ms < 0 else "late"
                lines.append(
                    f"  at {_clock(spot.position_s)}: {_ms(abs(spot.residual_ms))}ms {direction}, "
                    f"match {spot.match:.2f}"
                )
        if self.sweep_windows:
            share = 100.0 * self.sweep_within_audible / max(1, self.sweep_measured)
            fine = 100.0 * self.sweep_within_1ms / max(1, self.sweep_measured)
            lines.append(
                f"swept the whole runtime: {self.sweep_measured} of {self.sweep_windows} windows could be "
                f"measured (the rest are silence, music-only, or filled from the original), and "
                f"{share:.0f}% of them are within {AUDIBLE_MS:.0f}ms of the original, {fine:.0f}% within 1ms"
                + (
                    f" (typically {_ms(self.sweep_typical_ms)}ms, worst {_ms(self.sweep_worst_ms)}ms)"
                    if self.sweep_typical_ms is not None else ""
                )
            )
        if self.stretches:
            lines.append(f"{len(self.stretches)} stretch(es) are more than {STRETCH_MS:.0f}ms out, where lip sync shows:")
            for stretch in self.stretches:
                lines.append(
                    f"  {_clock(stretch.start_s)} - {_clock(stretch.end_s)}  out by "
                    f"{'+' if stretch.residual_ms >= 0 else ''}{_ms(stretch.residual_ms)}ms  ({stretch.windows} windows)"
                )
        elif self.sweep_measured:
            lines.append(f"no stretch is more than {STRETCH_MS:.0f}ms out")
        if self.lines_text:
            lines.append(self.lines_text)
        return "\n".join(lines)


def _ms(value: Optional[float]) -> str:
    """Milliseconds as they read best: the tenth under ten."""
    if value is None:
        return "?"
    return f"{value:.1f}" if abs(value) < 10.0 else f"{value:.0f}"


def _in_fill(plan: DubSyncPlan, lo_s: float, hi_s: float) -> bool:
    for segment in plan.fill_segments:
        if segment.start_s < hi_s and segment.end_s > lo_s:
            return True
    return False


def verify_output(
    primary: TrackEnvelope,
    output: TrackEnvelope,
    plan: DubSyncPlan,
    token: Optional[CancellationToken] = None,
) -> Verification:
    """Measure the finished track against the original along the runtime.

    Two views: a dozen long windows, each of which is a precise measurement,
    and a sweep of short windows every few seconds that can catch a short
    stretch the spots fell between. Windows in filled regions are skipped --
    the fill is the original, so it matches trivially and would flatter the
    numbers.
    """
    result = Verification()
    rate = ENVELOPE_RATE
    duration = min(primary.duration_s, output.duration_s)
    bands = [b for b in primary.bands if b in output.bands]

    # Spot checks.
    window = int(VERIFY_SPOT_WINDOW_S * rate)
    margin = int(VERIFY_SPOT_RANGE_S * rate)
    usable = duration - VERIFY_SPOT_WINDOW_S
    if usable > 0:
        inset = min(usable * 0.02, 5.0)
        step = (usable - 2 * inset) / max(1, VERIFY_SPOTS - 1)
        for index in range(VERIFY_SPOTS):
            if token:
                token.raise_if_cancelled()
            position = inset + step * index
            start = int(position * rate)
            if _in_fill(plan, position, position + VERIFY_SPOT_WINDOW_S):
                result.spots.append(SpotCheck(position, None, 0.0, "filled from the original"))
                result.fills_skipped += 1
                continue
            residual, match, z = _measure(
                [primary.onsets(b) for b in bands], [output.onsets(b) for b in bands], start, window, margin, rate, bands
            )
            if residual is None or z < MATCH_Z:
                result.spots.append(SpotCheck(position, None, match, "silence or music-only; not measurable"))
            else:
                result.spots.append(SpotCheck(position, residual, match))

    # Sweep.
    pool = VERIFY_SWEEP_POOL
    sweep_rate = rate / pool
    p = [_pool(primary.onsets(b), pool) for b in bands]
    o = [_pool(output.onsets(b), pool) for b in bands]
    window = int(VERIFY_SWEEP_WINDOW_S * sweep_rate)
    margin = int(VERIFY_SWEEP_RANGE_S * sweep_rate)
    residuals: List[Tuple[float, float]] = []
    count = int(duration // VERIFY_SWEEP_WINDOW_S)
    for index in range(count):
        if token:
            token.raise_if_cancelled()
        position = index * VERIFY_SWEEP_WINDOW_S
        if _in_fill(plan, position, position + VERIFY_SWEEP_WINDOW_S):
            continue
        result.sweep_windows += 1
        start = int(position * sweep_rate)
        residual, _match, z = _measure(p, o, start, window, margin, sweep_rate, bands)
        if residual is None or z < VERIFY_SWEEP_Z:
            continue
        result.sweep_measured += 1
        residuals.append((position, residual))
        if abs(residual) <= AUDIBLE_MS:
            result.sweep_within_audible += 1
        if abs(residual) <= 5.0:
            result.sweep_within_5ms += 1
        if abs(residual) <= 1.0:
            result.sweep_within_1ms += 1
    if residuals:
        magnitudes = [abs(r) for _, r in residuals]
        result.sweep_typical_ms = float(np.median(magnitudes))
        result.sweep_worst_ms = float(max(magnitudes))
        # Runs of consecutive measured windows that are all out the same way.
        run: List[Tuple[float, float]] = []
        for position, residual in residuals + [(float("inf"), 0.0)]:
            if abs(residual) > STRETCH_MS and (not run or position - run[-1][0] <= 2 * VERIFY_SWEEP_WINDOW_S):
                run.append((position, residual))
                continue
            if len(run) >= VERIFY_STRETCH_WINDOWS:
                result.stretches.append(Stretch(
                    run[0][0], run[-1][0] + VERIFY_SWEEP_WINDOW_S,
                    float(np.median([r for _, r in run])), len(run),
                ))
            run = []
            if abs(residual) > STRETCH_MS:
                run.append((position, residual))
    return result


def _measure(
    primary: Sequence[np.ndarray], other: Sequence[np.ndarray], start: int, window: int, margin: int, rate: float,
    bands: Optional[Sequence[str]] = None,
) -> Tuple[Optional[float], float, float]:
    """Residual offset (ms, positive = other is late) of one window.

    ``primary`` and ``other`` are the same bands of the two tracks. Each
    band's correlation is put in its own noise units and the bands are
    summed: a beat gives the full band a peak every period, about as tall
    as the true one, and only the true one is in every band. Among the
    summed peaks within 15% of the tallest, the one nearest zero lag is
    the answer -- this is a check that the track sits where it should,
    and a track that does gives a peak there.
    """
    total: Optional[np.ndarray] = None
    effects: Optional[np.ndarray] = None
    best_match = 0.0
    first, last = start - margin, start + window + margin
    for index, (p, o) in enumerate(zip(primary, other)):
        template = p[start : start + window]
        if len(template) < window or first < 0 or last > len(o):
            continue
        row = ncc_lags(template, o[first:last])
        if row.size < 3:
            continue
        if bands is not None and index < len(bands) and bands[index] == "high":
            effects = row.astype(np.float64)
        units = _noise_units(row, np.ones(len(row), dtype=bool))
        total = units.astype(np.float64) if total is None else total + units
        best_match = max(best_match, float(row.max()))
    if total is None:
        return None, 0.0, 0.0
    if effects is not None:
        # The effects decide here as they do in the plan (see BANDS): a
        # cue laid twice puts the music's peak a tenth of a second from
        # the dialogue's, and it is the dialogue the check is about.
        peak = _nearest_peak(effects, margin)
        units = _noise_units(effects, np.ones(len(effects), dtype=bool))
        if float(units[peak]) >= EFFECTS_Z and _peak_dominance(effects, peak) >= EFFECTS_DOMINANCE:
            lag = (first + _parabolic(effects, peak)) - start
            return lag / rate * 1000.0, best_match, float(units[peak])
    peak = _nearest_peak(total, margin)
    away = np.abs(np.arange(len(total)) - peak) > max(3, int(0.25 * rate))
    noise = float(np.std(total[away])) if away.sum() > 8 else 1.0
    z = float(total[peak]) / max(noise, 1e-9)
    lag = (first + _parabolic(total, peak)) - start
    return lag / rate * 1000.0, best_match, z
