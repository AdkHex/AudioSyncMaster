"""Explain drift in terms of frame rate, and classify how fixable it is.

Reporting "0.417 ms/s of drift" is accurate but leaves the user to work out
what to do. Almost all steady drift in dubbed material comes from one cause: a
track mastered at a different frame rate. Naming that cause turns an opaque
number into an instruction.

The classic case is PAL speedup. A 23.976fps film sped to 25fps runs 4.27%
short; the same audio laid against the original runs 4.27% long. Both show up
as a constant drift slope, and both have an exact correction factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Frame rates that real releases actually use.
COMMON_RATES: List[float] = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]

# How close a measured speed ratio must sit to a known conversion before it is
# named. Real conversions are widely spaced -- the smallest, 24 -> 23.976, is
# 0.1% -- so this must be tighter than that gap or neighbouring rates get
# confused, while still absorbing measurement noise.
RATIO_TOLERANCE = 0.0004

# Below this the drift is measurement noise, not a real speed difference.
# Matches the analysis layer's own "worth reporting" floor: 0.05 ms/s is 0.14
# seconds of skew across a 45-minute episode.
MIN_MEANINGFUL_DRIFT_MS_PER_S = 0.05

# The largest drift any standard conversion can produce. 23.976 -> 25 is the
# most extreme pair in real use at 42.7 ms/s; anything meaningfully beyond that
# cannot be explained by frame rate. Steady drift that large means the two
# files run at genuinely different lengths -- different cuts, or a file with
# scenes added or removed.
MAX_RATE_DRIFT_MS_PER_S = 45.0


@dataclass
class RateDiagnosis:
    """What the measured drift implies about the two files' frame rates."""

    drift_ms_per_s: float
    speed_ratio: float
    """How much faster the secondary runs than the primary. 1.0 means neither."""

    source_fps: Optional[float] = None
    target_fps: Optional[float] = None
    """The rate pair that explains the drift, when one fits."""

    is_rate_mismatch: bool = False
    is_likely_cut: bool = False
    explanation: str = ""

    @property
    def correction_ratio(self) -> Optional[float]:
        """atempo factor that cancels the mismatch, if one was identified."""
        if not self.is_rate_mismatch or self.speed_ratio <= 0:
            return None
        return 1.0 / self.speed_ratio

    def to_dict(self) -> dict:
        return {
            "driftMsPerS": self.drift_ms_per_s,
            "speedRatio": self.speed_ratio,
            "sourceFps": self.source_fps,
            "targetFps": self.target_fps,
            "isRateMismatch": self.is_rate_mismatch,
            "isLikelyCut": self.is_likely_cut,
            "explanation": self.explanation,
            "correctionRatio": self.correction_ratio,
        }


def _format_fps(value: float) -> str:
    return f"{value:g}" if value == int(value) else f"{value:.3f}".rstrip("0")


def diagnose(
    drift_ms_per_s: Optional[float],
    primary_fps: Optional[float] = None,
    secondary_fps: Optional[float] = None,
) -> Optional[RateDiagnosis]:
    """Explain a measured drift rate.

    Args:
        drift_ms_per_s: how fast the required offset changes. Positive means the
            secondary falls further behind as the file plays.
        primary_fps: frame rate of the reference, when known.
        secondary_fps: frame rate of the secondary's source, when known.

    Returns:
        A diagnosis, or None when there is no meaningful drift to explain.
    """
    if drift_ms_per_s is None:
        return None

    # An offset growing by D ms every second means the secondary's timeline
    # runs slow by D/1000 relative to the primary.
    speed_ratio = 1.0 - (drift_ms_per_s / 1000.0)

    # Too small to mean anything: report it as benign rather than inventing a
    # frame-rate story for what is really measurement noise.
    if abs(drift_ms_per_s) < MIN_MEANINGFUL_DRIFT_MS_PER_S:
        return RateDiagnosis(
            drift_ms_per_s=drift_ms_per_s,
            speed_ratio=speed_ratio,
            explanation="No meaningful drift; a single delay aligns the whole file.",
        )

    if abs(drift_ms_per_s) > MAX_RATE_DRIFT_MS_PER_S:
        return RateDiagnosis(
            drift_ms_per_s=drift_ms_per_s,
            speed_ratio=speed_ratio,
            is_likely_cut=True,
            explanation=(
                "The offset changes far too fast for a frame-rate difference. "
                "These are probably different cuts of the same title, with "
                "scenes added or removed. No single delay can align them."
            ),
        )

    # Both rates known: check whether they alone account for the drift.
    #
    # The comparison is relative, not absolute. A tolerance that suits ratios
    # near 1.0 is far too tight once the ratio itself is 4% from unity, because
    # the same fractional measurement error becomes a much larger absolute one.
    if primary_fps and secondary_fps and primary_fps > 0 and secondary_fps > 0:
        implied = primary_fps / secondary_fps
        relative_error = abs(implied - speed_ratio) / max(implied, 1e-9)
        if abs(implied - 1.0) > 1e-6 and relative_error <= RATIO_TOLERANCE:
            return RateDiagnosis(
                drift_ms_per_s=drift_ms_per_s,
                speed_ratio=implied,
                source_fps=secondary_fps,
                target_fps=primary_fps,
                is_rate_mismatch=True,
                explanation=(
                    f"The audio was timed against a {_format_fps(secondary_fps)}fps "
                    f"source, but this video is {_format_fps(primary_fps)}fps. "
                    "Resampling the audio corrects it exactly."
                ),
            )

    # Otherwise search the common rate pairs for one that fits the measurement.
    best: Optional[tuple] = None
    for source in COMMON_RATES:
        for target in COMMON_RATES:
            if source == target:
                continue
            ratio = target / source
            error = abs(ratio - speed_ratio) / max(ratio, 1e-9)
            if error <= RATIO_TOLERANCE and (best is None or error < best[0]):
                best = (error, source, target, ratio)

    if best is not None:
        _, source, target, ratio = best
        return RateDiagnosis(
            drift_ms_per_s=drift_ms_per_s,
            speed_ratio=ratio,
            source_fps=source,
            target_fps=target,
            is_rate_mismatch=True,
            explanation=(
                f"The drift matches a {_format_fps(source)}fps to "
                f"{_format_fps(target)}fps conversion. Resampling the audio "
                "corrects it exactly."
                if source and target
                else "The audio runs at a slightly different speed. Resampling corrects it."
            ),
        )

    return RateDiagnosis(
        drift_ms_per_s=drift_ms_per_s,
        speed_ratio=speed_ratio,
        explanation=(
            "The offset drifts steadily but does not match a standard frame-rate "
            "conversion. Resampling still corrects it, but check the result."
        ),
    )
