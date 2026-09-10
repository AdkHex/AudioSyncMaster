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
from fractions import Fraction
from typing import Callable, List, Optional

import numpy as np

from .codecdelay import describe as describe_codec_delay
from .codecdelay import relative_codec_delay_ms
from .correlate import OffsetEstimate, estimate_offset
from .framerate import (
    RateDiagnosis,
    diagnose,
    plan_speed_compensation,
    speed_candidates,
)
from .media import CancellationToken, MediaError, load_audio, probe
from .segments import Step, find_step

ANALYSIS_SR = 16000

# Drift beyond this is a real speed mismatch (e.g. PAL 25fps vs 23.976fps),
# not measurement noise. 0.05 ms/s over an hour is 180 ms of accumulated skew.
DRIFT_SIGNIFICANT_MS_PER_S = 0.05

# Below this many usable windows a pair cannot be described: two is the minimum
# for any statement about drift and three for a fitted one. Falling short is the
# signal to go looking for a speed the durations failed to predict, because a
# pair that is really a rate conversion looks exactly like an unrelated one
# until the speed comes off.
MIN_USABLE_WINDOWS = 3

# Peak prominence that settles which trial speed is the right one. Measured on
# real decodes of a PAL pair, the correct compensation scores 71-207 and every
# wrong one 6-12, so anything in between separates them with room to spare.
DECISIVE_PEAK_RATIO = 30.0

# How many speeds to try before giving up. The durations order the search, so
# the answer is normally first or second; this only bounds how long a pair that
# was never going to match can take.
MAX_SPEED_TRIALS = 6


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
    """Offset extrapolated back to t=0, which is where a correction is applied.

    Fitted from every usable window whenever there is more than one, so it stays
    the value at the start of the file even when the drift is too small to
    report. It only falls back to delay_ms when a single window is all there
    was, and then there is nothing to extrapolate from."""
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
    cut: Optional[Step] = None
    """The splice found in this pair, when the offsets step rather than drift.

    Its presence is what makes `delay_ms` describe the part of the file before
    the cut rather than an average of both halves."""
    window_s: Optional[float] = None
    """Length of each measurement window, which bounds how late a cut can be."""
    speed_compensation: float = 1.0
    """Playback-speed difference undone during decoding so the windows could be
    correlated at all. 1.0 means the files were measured as they are."""
    codec_delay_ms: float = 0.0
    """Codec delay already removed from delay_ms. Some formats decode shifted
    from where the source sat, which lands in the measurement when the two
    files use different codecs."""
    primary_codec: Optional[str] = None
    secondary_codec: Optional[str] = None

    @property
    def is_likely_cut(self) -> bool:
        return bool(self.cut) or bool(
            self.rate_diagnosis and self.rate_diagnosis.is_likely_cut
        )

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
            "cutPositionS": self.cut.position_s if self.cut else None,
            "cutUncertaintyS": self.cut.uncertainty_s if self.cut else None,
            "cutMagnitudeMs": self.cut.magnitude_ms if self.cut else None,
            "isRateMismatch": self.is_rate_mismatch,
            "speedCompensation": self.speed_compensation,
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
    cut_probes: int = 6,
) -> PairResult:
    """Measure the offset between two media files.

    Args:
        window_s: seconds of audio per measurement window.
        window_count: how many points across the file to measure.
        max_offset_ms: reject alignments implying a larger shift than this.
        progress: called with 0-100 as windows complete.
        primary_track: which audio stream of the primary to compare.
        secondary_track: which audio stream of the secondary to compare.
        cut_probes: how many extra short windows may be spent narrowing down a
            suspected cut. Zero skips it, which also means a cut resting on a
            single window can no longer be corroborated and is not reported.
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
            primary_info.container_format, secondary_info.container_format,
        )

        if not primary_duration or primary_duration <= 0:
            result.error = f"Could not read duration of {result.primary_name}"
            return result
        if not secondary_duration or secondary_duration <= 0:
            result.error = f"Could not read duration of {result.secondary_name}"
            return result

        # A large speed difference has to come off before anything is
        # correlated: past about half a percent the alignment moves further
        # inside a single window than the correlation can resolve, and the pair
        # reports itself as unrelated rather than as a rate mismatch.
        #
        # ffmpeg needs an integer output rate, so what actually gets applied is
        # the rounded one. Deriving the ratio back from that rate rather than
        # carrying the ideal keeps every later conversion exact.
        compensation = plan_speed_compensation(primary_duration, secondary_duration)
        secondary_rate = ANALYSIS_SR
        if compensation is not None:
            secondary_rate = max(1, int(round(ANALYSIS_SR * float(compensation))))

        def sweep(rate: int):
            """Measure every window with the secondary running at this speed."""
            # Only where both files have content, in the primary's timeline --
            # which is what the secondary's duration becomes once the speed
            # difference is taken off it.
            span = min(primary_duration, secondary_duration * (rate / ANALYSIS_SR))
            length = min(window_s, max(5.0, span / 3.0))
            spots = plan_windows(span, length, window_count)
            measured = []
            for index, position in enumerate(spots):
                if token:
                    token.raise_if_cancelled()
                measured.append(WindowResult(position, _measure_window(
                    primary_path, secondary_path, position, length,
                    max_offset_ms, token, primary_track, secondary_track, rate,
                )))
                report(5 + int(75 * (index + 1) / len(spots)))
            return length, measured

        report(5)
        effective_window, windows = sweep(secondary_rate)

        # Too little to work with, which is what a rate conversion looks like
        # when the durations did not predict it -- and they often cannot, since
        # a dub that starts late or an episode with different credits moves the
        # ratio further than the conversion does. Ask the audio instead.
        if sum(1 for w in windows if w.usable) < MIN_USABLE_WINDOWS:
            alternative = _search_speed(
                primary_path, secondary_path, effective_window, max_offset_ms,
                token, primary_track, secondary_track,
                speed_candidates(primary_duration, secondary_duration),
                secondary_rate, primary_duration,
            )
            if alternative is not None and alternative != secondary_rate:
                secondary_rate = alternative
                effective_window, windows = sweep(secondary_rate)

        result.speed_compensation = secondary_rate / ANALYSIS_SR
        result.window_s = effective_window
        result.windows = windows

        # First pass proposes a split; probing then either pins the cut down or
        # withdraws it, and the second pass re-derives every number from the
        # answer. Re-running is cheaper than unpicking a provisional result,
        # and it keeps one code path deciding what a segmented file reports.
        step = _reconcile(result)
        if step is not None:
            confirmed = _localize_cut(
                step,
                primary_path,
                secondary_path,
                effective_window,
                max_offset_ms,
                token,
                primary_track,
                secondary_track,
                cut_probes,
                secondary_rate,
            )
            if confirmed:
                _reconcile(result, cut=step)
            else:
                _reconcile(result, allow_step=False)
        report(95)

        # Remove the part of the measurement that is codec alignment rather
        # than real sync. Applied to every offset, so a correction can never
        # be double-counted or missed on one of them.
        if result.codec_delay_ms:
            for attribute in ("delay_ms", "delay_at_start_ms", "start_delay_ms", "end_delay_ms"):
                value = getattr(result, attribute)
                if value is not None:
                    setattr(result, attribute, value - result.codec_delay_ms)
            if result.cut is not None:
                # Both levels, so the jump between them -- which is what the
                # cut actually is -- comes out unchanged.
                result.cut.before_ms -= result.codec_delay_ms
                result.cut.after_ms -= result.codec_delay_ms

        result.rate_diagnosis = diagnose(
            result.drift_ms_per_s,
            result.primary_fps,
            result.secondary_fps,
            cut_position_s=result.cut.position_s if result.cut else None,
            cut_magnitude_ms=result.cut.magnitude_ms if result.cut else None,
            cut_uncertainty_s=result.cut.uncertainty_s if result.cut else None,
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
    secondary_rate: int = ANALYSIS_SR,
) -> OffsetEstimate:
    """Measure one window, padding the secondary so a shifted match still fits.

    The secondary is decoded with extra margin on both sides. Without it, a
    window near a genuine offset of several seconds would be comparing two
    non-overlapping spans of audio and would correctly find nothing.

    Args:
        secondary_rate: the rate to decode the secondary at. Asking ffmpeg for
            ANALYSIS_SR * ratio and then reading the samples as ANALYSIS_SR is
            a time-stretch by that ratio, done by the decoder's own resampler
            rather than by interpolating afterwards. That is how a PAL-sped dub
            is brought onto the video's clock before being correlated; at
            ANALYSIS_SR it is an ordinary decode and changes nothing.
    """
    # How much of the primary's clock one second of decoded secondary covers.
    speed = secondary_rate / ANALYSIS_SR

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
        # Both the seek and the length are in the secondary's own timeline, and
        # the span asked for is the one that becomes `secondary_window` long
        # once stretched.
        secondary = load_audio(
            secondary_path,
            secondary_rate,
            duration=secondary_window / speed,
            offset=secondary_start / speed,
            token=token,
            track=secondary_track,
        )
    except MediaError as exc:
        return OffsetEstimate(None, 0.0, 0.0, str(exc))

    # The search bound has to cover the head start as well as the offset being
    # looked for, because the measurement it bounds is the sum of the two. Left
    # at max_offset_ms it quietly rules out every offset the user asked for:
    # with the slider at 6s and a real offset of 5s, only the windows near the
    # start of the file -- where there is no room for a full head start --
    # stayed inside the range, and the other three reported the two tracks as
    # unrelated. One surviving window is also no window at all for drift, which
    # needs three.
    head_start_ms = (position_s - secondary_start) * 1000.0
    estimate = estimate_offset(
        primary, secondary, ANALYSIS_SR, max_offset_ms=max_offset_ms + head_start_ms
    )
    if not estimate.matched:
        return estimate

    # Convert the within-window measurement back to a whole-file offset,
    # correcting for the head start given to the secondary.
    absolute = estimate.delay_ms - head_start_ms

    # Undo the compensation in the reported figure. The measurement was made on
    # a secondary running at the primary's speed, but every number this result
    # carries describes the files the user actually has: the content matching
    # primary time t sits at compensated time t + absolute, which is that over
    # `speed` in the secondary's own timeline.
    #
    # Doing it here rather than at the end means nothing downstream has to know
    # a compensation happened. The drift fitted across windows comes out as the
    # real one, and the offset at t=0 is unchanged either way.
    if speed != 1.0:
        absolute = ((position_s + absolute / 1000.0) / speed - position_s) * 1000.0

    return OffsetEstimate(
        delay_ms=absolute,
        confidence=estimate.confidence,
        peak_ratio=estimate.peak_ratio,
    )


def _search_speed(
    primary_path: str,
    secondary_path: str,
    window_s: float,
    max_offset_ms: float,
    token: Optional[CancellationToken],
    primary_track: int,
    secondary_track: int,
    candidates: List[Fraction],
    already_tried: int,
    primary_duration: float,
) -> Optional[int]:
    """Find the playback speed that makes this pair correlate, by trying them.

    Reached only when measuring the pair as planned produced almost nothing,
    which is the same symptom as two unrelated files. The difference is that a
    rate conversion has a speed at which the two do line up, and the standard
    conversions are a short list -- so the question can be settled by asking.

    One window is enough to ask with, because the answer is not subtle: at the
    right speed the correlation peak stands 6 to 17 times higher than at any
    wrong one. A pair that really is unrelated fails every trial and costs a
    handful of decodes it was going to fail with anyway.

    Returns:
        The decode rate to measure at, or None to leave the pair as it is.
    """
    # A window in the middle: far enough in that a drifting pair has separated,
    # and not the credits at either end.
    position = max(0.0, (primary_duration - window_s) / 2.0)

    best = None
    tried = 0
    for candidate in candidates:
        rate = max(1, int(round(ANALYSIS_SR * float(candidate))))
        if rate == already_tried:
            continue
        if tried >= MAX_SPEED_TRIALS:
            break
        tried += 1
        if token:
            token.raise_if_cancelled()

        estimate = _measure_window(
            primary_path, secondary_path, position, window_s,
            max_offset_ms, token, primary_track, secondary_track, rate,
        )
        # Decisive, not merely matched. Six trials are six chances at a
        # coincidence, so the bar here has to be higher than for a single
        # measurement: at the ordinary threshold, trying speeds on two
        # unrelated releases eventually found one that "worked" and reported a
        # confident seven-second delay between films that share no audio.
        if estimate.peak_ratio >= DECISIVE_PEAK_RATIO and (
            best is None or estimate.peak_ratio > best[0]
        ):
            best = (estimate.peak_ratio, rate)
            break

    return best[1] if best is not None else None


def _localize_cut(
    step: Step,
    primary_path: str,
    secondary_path: str,
    window_s: float,
    max_offset_ms: float,
    token: Optional[CancellationToken],
    primary_track: int,
    secondary_track: int,
    probes: int,
    secondary_rate: int = ANALYSIS_SR,
) -> bool:
    """Narrow down where the cut is, and decide whether to believe in it.

    The survey windows only say the offset changed somewhere between two of
    them, which on a 45-minute file is a seven-minute answer. Short windows
    placed by bisection turn that into something a user can seek to: each probe
    reads one of the two levels already measured and rules out the half of the
    bracket it sits in.

    The same probes are the corroboration. A window that locked onto a repeated
    musical phrase is indistinguishable from a genuine post-cut window while it
    is the only one out there -- but the phrase does not repeat at every
    position, so a probe near it reads the ordinary offset instead, and the
    step is withdrawn. Segments with two windows of their own already
    corroborate each other and only need locating.

    Returns:
        Whether the step survived. Probing never invents a cut; it only refuses
        to confirm one.
    """
    if probes <= 0:
        return not step.lone_group

    # Shorter than a survey window, because the bracket cannot be narrowed
    # below the length of the probe reading it.
    probe_window = max(2.0, min(window_s / 3.0, 15.0))

    lower, upper = step.earliest_s, step.latest_s
    saw_before = saw_after = False

    for _ in range(probes):
        if upper - lower <= 2.0 * probe_window:
            break
        if token:
            token.raise_if_cancelled()

        middle = (lower + upper) / 2.0
        estimate = _measure_window(
            primary_path,
            secondary_path,
            middle,
            probe_window,
            max_offset_ms,
            token,
            primary_track,
            secondary_track,
            secondary_rate,
        )
        side = _side_of_cut(estimate, step)
        if side is None:
            # Silence, music, or an offset matching neither level. Nothing here
            # is evidence either way, and the next probe would land beside it.
            break

        if side == "before":
            saw_before = True
            # The cut is after this probe starts. Not after it ends: a probe
            # whose tail crosses the cut still reports the level that fills
            # most of it.
            lower = middle
        else:
            saw_after = True
            upper = min(upper, middle + probe_window)

    step.earliest_s, step.latest_s = lower, upper

    if step.lone_group:
        return saw_before and saw_after
    return True


def _side_of_cut(estimate: OffsetEstimate, step: Step) -> Optional[str]:
    """Which of the cut's two levels this measurement agrees with, if either."""
    if not estimate.matched or estimate.delay_ms is None:
        return None
    to_before = abs(estimate.delay_ms - step.before_ms)
    to_after = abs(estimate.delay_ms - step.after_ms)
    if min(to_before, to_after) > abs(step.magnitude_ms) * 0.25:
        return None
    return "before" if to_before < to_after else "after"


def _reconcile(
    result: PairResult,
    cut: Optional[Step] = None,
    allow_step: bool = True,
) -> Optional[Step]:
    """Combine per-window measurements into one answer plus a drift estimate.

    Args:
        cut: a step already found and confirmed, whose grouping to reuse.
        allow_step: whether to look for one. False after probing withdrew it.

    Returns:
        The step it is working from, when there is one. On the first pass that
        is a proposal the caller still has to confirm.
    """
    usable = sorted(
        (w for w in result.windows if w.usable), key=lambda w: w.position_s
    )

    if not usable:
        reasons = [w.estimate.reason for w in result.windows if w.estimate.reason]
        result.error = reasons[0] if reasons else "No usable measurement windows"
        result.confidence = 0.0
        result.cut = None
        return None

    offsets = np.array([w.estimate.delay_ms for w in usable], dtype=np.float64)
    positions = np.array([w.position_s for w in usable], dtype=np.float64)
    confidences = np.array([w.estimate.confidence for w in usable], dtype=np.float64)

    # Ask whether the file is one timeline or two before discarding anything.
    # The old order threw the answer away first: a splice leaves the post-cut
    # windows sitting far from the median, the trim below deleted them as
    # outliers, and what was left -- four windows in perfect agreement -- was
    # then reported as proof that one delay aligns the whole file.
    step = cut
    if step is None and allow_step:
        step = find_step(positions, offsets, result.window_s or 0.0)

    after_cut = _split_mask(positions, step)
    model = _model_values(positions, offsets, after_cut)

    # Discard windows far from what the file is doing, so one bad measurement
    # cannot tilt the drift line. Distance is now measured from that model
    # rather than from a single median, which is what lets a coherent minority
    # survive: post-cut windows sit on their own segment's level, so they are
    # no longer outliers to anything.
    residual = np.abs(offsets - model)
    spread = float(np.median(residual))
    tolerance = max(50.0, spread * 4.0)
    keep = residual <= tolerance
    if keep.sum() >= 2 and not keep.all():
        offsets, positions = offsets[keep], positions[keep]
        confidences, after_cut = confidences[keep], after_cut[keep]
        # A segment can be trimmed out of existence, and a step with nothing on
        # one side of it is not a step.
        if step is not None and (after_cut.all() or not after_cut.any()):
            step = None
            after_cut = _split_mask(positions, None)
        model = _model_values(positions, offsets, after_cut)

    result.cut = step

    # With a cut, the offset before it is the one a correction can act on. The
    # median across every window is the average of two different files, a value
    # that is wrong for both halves -- 320ms where the truth was 300 before the
    # cut and 342 after.
    fitted = ~after_cut if step is not None else np.ones(len(offsets), dtype=bool)
    result.delay_ms = float(np.median(offsets[fitted]))
    result.confidence = float(np.mean(confidences))

    # Agreement is corroborating evidence; wide disagreement means something is
    # wrong even if each individual peak looked sharp. Measured against the
    # model, so a clean two-level file is not punished for the gap between its
    # levels: that the offset changed is already reported as a cut, and how
    # well each window was measured is a separate question this answers.
    if len(offsets) >= 2:
        disagreement = float(np.std(offsets - model))
        if disagreement > 500.0:
            result.confidence *= 0.5
        elif disagreement > 100.0:
            result.confidence *= 0.8

    result.start_delay_ms = float(offsets[0])
    result.end_delay_ms = float(offsets[-1])

    # Fit offset against position: the slope is the drift rate.
    #
    # The intercept is taken whenever a line could be fitted, not only when the
    # drift is large enough to report. DRIFT_SIGNIFICANT_MS_PER_S answers "is
    # this worth telling the user about, and does it need a speed correction?";
    # it is not a claim that anything below it is zero. Gating the intercept on
    # it meant that below the threshold `delay_at_start_ms` was quietly the
    # median across the whole file -- the value at its middle, not at its
    # start -- while still being named and consumed as the t=0 offset.
    #
    # The error that hides there is the drift times half the duration, so it
    # grows with the file and peaks just under the threshold: 0.005 ms/s over a
    # 40-minute episode is 6 ms, and 0.04 ms/s is 47 ms, all reported as no
    # drift at all. Windows are spread from ~2% in to ~2% from the end, so
    # reaching t=0 is a short extrapolation from a fit spanning the file.
    #
    # It costs a little noise when the drift is genuinely zero, and that is the
    # whole trade: measured over 20000 runs at 6 windows, RMSE at t=0 goes from
    # 0.46 ms (median) to 0.72 ms (intercept) with no drift, against 5.93 ms
    # versus 0.72 ms at a tenth of the threshold. A quarter-millisecond is
    # worth paying to delete an error of tens.
    #
    # Past a cut the line is fitted to the pre-cut windows alone. Fitting it
    # through both segments is what turned a 500ms splice into a 1.18 ms/s
    # slope, close enough to 23.976 -> 24 to be reported as a rate mismatch
    # with an exact resampling ratio -- stretching a whole episode to correct a
    # local edit.
    fit_offsets, fit_positions = offsets[fitted], positions[fitted]
    result.drift_ms_per_s = None
    result.delay_at_start_ms = None
    if len(fit_offsets) >= 3 and float(np.ptp(fit_positions)) > 1.0:
        slope, intercept = np.polyfit(fit_positions, fit_offsets, 1)
        result.drift_ms_per_s = float(slope)
        result.delay_at_start_ms = float(intercept)
        if abs(slope) > DRIFT_SIGNIFICANT_MS_PER_S:
            # With real drift no single number describes the whole file. Quote
            # the midpoint value as the representative offset; corrections use
            # the t=0 intercept above, since applying the midpoint value from
            # the start of the file over-shifts by half the total drift.
            midpoint = float(np.mean(fit_positions))
            result.delay_ms = float(np.polyval([slope, intercept], midpoint))
    elif len(fit_offsets) == 2 and float(np.ptp(fit_positions)) > 1.0:
        # Two points cannot separate drift from noise, but the extrapolation
        # back to t=0 is short enough that the intercept is still barely more
        # than the first window's own value.
        span = fit_positions.max() - fit_positions.min()
        slope = float((fit_offsets[-1] - fit_offsets[0]) / span)
        result.drift_ms_per_s = slope
        result.delay_at_start_ms = float(fit_offsets[0] - slope * fit_positions[0])

    if result.delay_at_start_ms is None:
        result.delay_at_start_ms = result.delay_ms

    return step


def _split_mask(positions: np.ndarray, step: Optional[Step]) -> np.ndarray:
    """Which windows fall on the far side of the cut."""
    if step is None:
        return np.zeros(len(positions), dtype=bool)
    return positions >= step.split_position_s


def _model_values(
    positions: np.ndarray, offsets: np.ndarray, after_cut: np.ndarray
) -> np.ndarray:
    """What the file says each window should have measured.

    Two levels when a cut splits them, a line when the offsets slide, one level
    otherwise -- and every version of it is fitted robustly, so the reference an
    outlier is judged against is never one that outlier helped set.

    Judging a drifting file against a single level is what made the outlier trim
    inert exactly where it was needed most. On a PAL-sped pair the offsets
    legitimately span nearly two seconds, so the spread they are compared
    against is enormous, the tolerance opens to cover everything, and one bad
    window survives to tilt the drift line -- which is the one number the whole
    frame-rate diagnosis is read from.
    """
    if after_cut.any():
        before = float(np.median(offsets[~after_cut]))
        after = float(np.median(offsets[after_cut]))
        return np.where(after_cut, after, before)

    if len(offsets) >= 3 and float(np.ptp(positions)) > 1.0:
        slope, intercept = _robust_line(positions, offsets)
        return slope * positions + intercept

    return np.full(len(offsets), float(np.median(offsets)))


def _robust_line(positions: np.ndarray, offsets: np.ndarray) -> tuple:
    """Theil-Sen: the median of the slopes through every pair of windows.

    Least squares cannot be used to decide what least squares should ignore --
    one window far from the truth drags the line towards itself and then looks
    reasonable against it. The median slope has to be dragged by a third of the
    windows before it moves, which is more bad windows than a usable pair has.
    """
    slopes = [
        (offsets[j] - offsets[i]) / (positions[j] - positions[i])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
        if positions[j] != positions[i]
    ]
    if not slopes:
        return 0.0, float(np.median(offsets))
    slope = float(np.median(slopes))
    return slope, float(np.median(offsets - slope * positions))
