"""Pair analysis: sample a file at several points, then reconcile the results.

The original implementation measured exactly two windows (head and tail) and
called the pair "high confidence" when the two agreed. That conflates two very
different questions -- *did we align the right audio?* and *does the alignment
hold across the file?* -- and two equally wrong measurements that happen to
agree score as the best possible result.

Here the two are separated:

*   **confidence** comes from correlation peak prominence per window: did we
    actually find this audio in that audio?
*   **drift** comes from a line fitted across all windows: does the required
    offset change as the file plays?

Sampling several windows also makes the estimate robust. A single window that
lands on silence or a music-only passage no longer decides the whole file; the
median of the surviving windows does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .codecdelay import describe as describe_codec_delay
from .codecdelay import relative_codec_delay_ms
from .correlate import OffsetEstimate, estimate_offset
from .framerate import RateDiagnosis, diagnose
from .media import CancellationToken, MediaError, load_audio, probe

ANALYSIS_SR = 16000

# Drift beyond this is a real speed mismatch (e.g. PAL 25fps vs 23.976fps),
# not measurement noise. 0.05 ms/s over an hour is 180 ms of accumulated skew.
DRIFT_SIGNIFICANT_MS_PER_S = 0.05


@dataclass
class WindowResult:
    """One measurement at one point in the file."""

    position_s: float
    estimate: OffsetEstimate

    @property
    def usable(self) -> bool:
        return self.estimate.matched


@dataclass
class PairResult:
    """Everything learned about one primary/secondary pair."""

    primary_path: str
    secondary_path: str
    delay_ms: Optional[float] = None
    """Representative offset. With drift this is the value at the file midpoint,
    which is the best single number to quote; corrections must use
    `delay_at_start_ms` instead, since they are applied from t=0."""
    confidence: float = 0.0
    drift_ms_per_s: Optional[float] = None
    delay_at_start_ms: Optional[float] = None
    """Offset extrapolated back to t=0. Equals delay_ms when there is no drift."""
    start_delay_ms: Optional[float] = None
    end_delay_ms: Optional[float] = None
    windows: List[WindowResult] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None
    primary_duration_s: Optional[float] = None
    secondary_duration_s: Optional[float] = None
    primary_track: int = 0
    secondary_track: int = 0
    primary_fps: Optional[float] = None
    secondary_fps: Optional[float] = None
    rate_diagnosis: Optional[RateDiagnosis] = None
    """Why the file drifts, when it does: a frame-rate conversion, or a cut."""
    codec_delay_ms: float = 0.0
    """Codec delay already removed from delay_ms. Some formats decode shifted
    from where the source sat, which lands in the measurement when the two
    files use different codecs."""
    primary_codec: Optional[str] = None
    secondary_codec: Optional[str] = None

    @property
    def is_likely_cut(self) -> bool:
        return bool(self.rate_diagnosis and self.rate_diagnosis.is_likely_cut)

    @property
    def is_rate_mismatch(self) -> bool:
        return bool(self.rate_diagnosis and self.rate_diagnosis.is_rate_mismatch)

    @property
    def primary_name(self) -> str:
        return os.path.basename(self.primary_path)

    @property
    def secondary_name(self) -> str:
        return os.path.basename(self.secondary_path)

    @property
    def has_significant_drift(self) -> bool:
        return (
            self.drift_ms_per_s is not None
            and abs(self.drift_ms_per_s) > DRIFT_SIGNIFICANT_MS_PER_S
        )

    @property
    def total_drift_ms(self) -> Optional[float]:
        """Offset change from start to end of the overlapping region."""
        if self.drift_ms_per_s is None or not self.primary_duration_s:
            return None
        return self.drift_ms_per_s * self.primary_duration_s

    def to_dict(self) -> dict:
        return {
            "primaryPath": self.primary_path,
            "secondaryPath": self.secondary_path,
            "videoFile": self.primary_name,
            "audioFile": self.secondary_name,
            "delayMs": self.delay_ms,
            "delayAtStartMs": self.delay_at_start_ms,
            "confidence": self.confidence,
            "driftMsPerS": self.drift_ms_per_s,
            "totalDriftMs": self.total_drift_ms,
            "hasSignificantDrift": self.has_significant_drift,
            "startDelayMs": self.start_delay_ms,
            "endDelayMs": self.end_delay_ms,
            "windowsUsed": sum(1 for w in self.windows if w.usable),
            "windowsTotal": len(self.windows),
            "error": self.error,
            "elapsedMs": self.elapsed_ms,
            "primaryDurationS": self.primary_duration_s,
            "secondaryDurationS": self.secondary_duration_s,
            "primaryTrack": self.primary_track,
            "secondaryTrack": self.secondary_track,
            "primaryFps": self.primary_fps,
            "secondaryFps": self.secondary_fps,
            "isLikelyCut": self.is_likely_cut,
            "isRateMismatch": self.is_rate_mismatch,
            "codecDelayMs": self.codec_delay_ms,
            "primaryCodec": self.primary_codec,
            "secondaryCodec": self.secondary_codec,
            "rateDiagnosis": self.rate_diagnosis.to_dict() if self.rate_diagnosis else None,
        }


def _track_of(info, index: int):
    """Codec and sample rate of the stream actually being compared."""
    tracks = getattr(info, "audio_tracks", None) or []
    if 0 <= index < len(tracks):
        track = tracks[index]
        return track.codec, track.sample_rate
    return info.audio_codec, info.sample_rate


def plan_windows(
    duration_s: float, window_s: float, count: int
) -> List[float]:
    """Choose start positions spread across the file.

    Windows are inset from both ends: the very start is often silence or a
    logo sting, and the very end is often credits over music.
    """
    usable = max(0.0, duration_s - window_s)
    if usable <= 0:
        return [0.0]
    if count <= 1:
        return [usable / 2.0]

    margin = min(usable * 0.02, 5.0)
    first, last = margin, usable - margin
    if last <= first:
        return [usable / 2.0]
    step = (last - first) / (count - 1)
    return [first + step * i for i in range(count)]


def analyze_pair(
    primary_path: str,
    secondary_path: str,
    window_s: float = 45.0,
    window_count: int = 6,
    max_offset_ms: float = 60000.0,
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[int], None]] = None,
    primary_track: int = 0,
    secondary_track: int = 0,
) -> PairResult:
    """Measure the offset between two media files.

    Args:
        window_s: seconds of audio per measurement window.
        window_count: how many points across the file to measure.
        max_offset_ms: reject alignments implying a larger shift than this.
        progress: called with 0-100 as windows complete.
        primary_track: which audio stream of the primary to compare.
        secondary_track: which audio stream of the secondary to compare.
    """
    result = PairResult(
        primary_path,
        secondary_path,
        primary_track=primary_track,
        secondary_track=secondary_track,
    )

    def report(percent: int) -> None:
        if progress:
            progress(max(0, min(100, percent)))

    try:
        report(0)
        if token:
            token.raise_if_cancelled()

        # One probe per file: duration and frame rate come from the same call,
        # so identifying a rate mismatch later costs nothing extra.
        primary_info = probe(primary_path, token)
        secondary_info = probe(secondary_path, token)
        primary_duration = primary_info.duration
        secondary_duration = secondary_info.duration
        result.primary_duration_s = primary_duration
        result.secondary_duration_s = secondary_duration
        result.primary_fps = primary_info.fps
        result.secondary_fps = secondary_info.fps

        # Track-specific codec details, since a container's first stream is not
        # necessarily the one being compared.
        primary_stream = _track_of(primary_info, primary_track)
        secondary_stream = _track_of(secondary_info, secondary_track)
        result.primary_codec = primary_stream[0]
        result.secondary_codec = secondary_stream[0]
        result.codec_delay_ms = relative_codec_delay_ms(
            primary_stream[0], primary_stream[1],
            secondary_stream[0], secondary_stream[1],
        )

        if not primary_duration or primary_duration <= 0:
            result.error = f"Could not read duration of {result.primary_name}"
            return result
        if not secondary_duration or secondary_duration <= 0:
            result.error = f"Could not read duration of {result.secondary_name}"
            return result

        # Only measure where both files actually have content.
        overlap = min(primary_duration, secondary_duration)
        effective_window = min(window_s, max(5.0, overlap / 3.0))
        positions = plan_windows(overlap, effective_window, window_count)

        report(5)
        for index, position in enumerate(positions):
            if token:
                token.raise_if_cancelled()
            estimate = _measure_window(
                primary_path,
                secondary_path,
                position,
                effective_window,
                max_offset_ms,
                token,
                primary_track,
                secondary_track,
            )
            result.windows.append(WindowResult(position, estimate))
            report(5 + int(90 * (index + 1) / len(positions)))

        _reconcile(result)

        # Remove the part of the measurement that is codec alignment rather
        # than real sync. Applied to every offset, so a correction can never
        # be double-counted or missed on one of them.
        if result.codec_delay_ms:
            for attribute in ("delay_ms", "delay_at_start_ms", "start_delay_ms", "end_delay_ms"):
                value = getattr(result, attribute)
                if value is not None:
                    setattr(result, attribute, value - result.codec_delay_ms)

        result.rate_diagnosis = diagnose(
            result.drift_ms_per_s, result.primary_fps, result.secondary_fps
        )
        report(100)
        return result

    except MediaError as exc:
        result.error = str(exc)
        return result


def _measure_window(
    primary_path: str,
    secondary_path: str,
    position_s: float,
    window_s: float,
    max_offset_ms: float,
    token: Optional[CancellationToken],
    primary_track: int = 0,
    secondary_track: int = 0,
) -> OffsetEstimate:
    """Measure one window, padding the secondary so a shifted match still fits.

    The secondary is decoded with extra margin on both sides. Without it, a
    window near a genuine offset of several seconds would be comparing two
    non-overlapping spans of audio and would correctly find nothing.
    """
    margin_s = min(max_offset_ms / 1000.0, 30.0)
    secondary_start = max(0.0, position_s - margin_s)
    secondary_window = window_s + margin_s + (position_s - secondary_start)

    try:
        primary = load_audio(
            primary_path,
            ANALYSIS_SR,
            duration=window_s,
            offset=position_s,
            token=token,
            track=primary_track,
        )
        secondary = load_audio(
            secondary_path,
            ANALYSIS_SR,
            duration=secondary_window,
            offset=secondary_start,
            token=token,
            track=secondary_track,
        )
    except MediaError as exc:
        return OffsetEstimate(None, 0.0, 0.0, str(exc))

    estimate = estimate_offset(
        primary, secondary, ANALYSIS_SR, max_offset_ms=max_offset_ms
    )
    if not estimate.matched:
        return estimate

    # Convert the within-window measurement back to a whole-file offset,
    # correcting for the head start given to the secondary.
    absolute = estimate.delay_ms - (position_s - secondary_start) * 1000.0
    return OffsetEstimate(
        delay_ms=absolute,
        confidence=estimate.confidence,
        peak_ratio=estimate.peak_ratio,
    )


def _reconcile(result: PairResult) -> None:
    """Combine per-window measurements into one answer plus a drift estimate."""
    usable = [w for w in result.windows if w.usable]

    if not usable:
        reasons = [w.estimate.reason for w in result.windows if w.estimate.reason]
        result.error = reasons[0] if reasons else "No usable measurement windows"
        result.confidence = 0.0
        return

    offsets = np.array([w.estimate.delay_ms for w in usable], dtype=np.float64)
    positions = np.array([w.position_s for w in usable], dtype=np.float64)
    confidences = np.array([w.estimate.confidence for w in usable], dtype=np.float64)

    # Median resists a single window that locked onto a repeated musical phrase.
    median_offset = float(np.median(offsets))

    # Discard windows far from consensus before fitting drift, so one outlier
    # cannot tilt the line and manufacture drift that is not there.
    spread = float(np.median(np.abs(offsets - median_offset)))
    tolerance = max(50.0, spread * 4.0)
    keep = np.abs(offsets - median_offset) <= tolerance
    if keep.sum() >= 2:
        offsets, positions, confidences = offsets[keep], positions[keep], confidences[keep]

    result.delay_ms = float(np.median(offsets))
    result.confidence = float(np.mean(confidences))

    # Agreement across windows is corroborating evidence; wide disagreement
    # means something is wrong even if each individual peak looked sharp.
    if len(offsets) >= 2:
        disagreement = float(np.std(offsets))
        if disagreement > 500.0:
            result.confidence *= 0.5
        elif disagreement > 100.0:
            result.confidence *= 0.8

    result.start_delay_ms = float(offsets[int(np.argmin(positions))])
    result.end_delay_ms = float(offsets[int(np.argmax(positions))])

    # Fit offset against position: the slope is the drift rate.
    if len(offsets) >= 3 and float(np.ptp(positions)) > 1.0:
        slope, intercept = np.polyfit(positions, offsets, 1)
        result.drift_ms_per_s = float(slope)
        if abs(slope) > DRIFT_SIGNIFICANT_MS_PER_S:
            # With real drift no single number describes the whole file. Quote
            # the midpoint value as the representative offset, but keep the
            # t=0 intercept as well: a correction is applied from the start of
            # the file, so using the midpoint there over-shifts by half the
            # total drift.
            midpoint = float(np.mean(positions))
            result.delay_ms = float(np.polyval([slope, intercept], midpoint))
            result.delay_at_start_ms = float(intercept)
    elif len(offsets) == 2 and float(np.ptp(positions)) > 1.0:
        span = positions.max() - positions.min()
        slope = float((offsets[-1] - offsets[0]) / span)
        result.drift_ms_per_s = slope
        if abs(slope) > DRIFT_SIGNIFICANT_MS_PER_S:
            result.delay_at_start_ms = float(offsets[0] - slope * positions[0])

    if result.delay_at_start_ms is None:
        result.delay_at_start_ms = result.delay_ms
