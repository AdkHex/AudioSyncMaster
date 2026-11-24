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

import math
import os
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, List, Optional

import numpy as np

from .codecdelay import describe as describe_codec_delay
from .codecdelay import relative_codec_delay_ms
from .correlate import (
    ENVELOPE_HOP,
    MIN_PEAK_RATIO,
    NO_PEAK,
    SILENT_PRIMARY,
    SILENT_SECONDARY,
    OffsetEstimate,
    _refine_against_waveform,
    estimate_offset,
)
from .framerate import (
    COMMON_RATES,
    MAX_RATE_DRIFT_MS_PER_S,
    RateDiagnosis,
    conversion_guide,
    diagnose,
    speed_candidates,
    speed_ratio_for,
)
from .media import Cancelled, CancellationToken, MediaError, load_audio, probe
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

# A trial speed whose one window found the dub without settling it is asked
# again at two more places in the file. Three windows landing within this of
# one offset, out of a search range of seconds, is a speed that fits: a dub
# sped up with its pitch kept (PAL with pitch correction) never confirms on
# the waveforms and peaks at 12-15 on its envelope, under DECISIVE_PEAK_RATIO
# at every window, yet its windows agree to a few milliseconds. Wide enough
# for the envelope's scatter through a time-stretch, which smears transients
# by a frame of its own.
CORROBORATION_MS = 30.0

# The envelope prominence a window must clear to count towards corroboration,
# and in a sweep through a speed change to land beside the window before it.
# Under MIN_PEAK_RATIO, because the agreement is the evidence: unrelated audio
# peaks anywhere in the range, and three such peaks falling within
# CORROBORATION_MS of each other in a range of seconds is a chance of about
# one in ten million per trial. A pitch-kept PAL dub at its right speed peaks
# at 8-17 here; unrelated audio at its own median of about 5.
CORROBORATION_PEAK_RATIO = 7.0

# Only the likeliest speeds -- the durations order them -- are corroborated;
# the rest still have to settle in one window. Unrelated audio clears the
# lowered bar at random, so corroborating every trial tripled the time a pair
# with no match takes to say so, for conversions the durations all but rule
# out.
CORROBORATED_TRIALS = 3

# ... and only speeds within this of the ratio the two files' durations
# suggest. Durations are a weak witness -- a dub without the video's recap runs
# 4% short with no speed change at all -- so they never rule a speed out of
# the search, only out of being settled on weak windows. A pitch-kept PAL dub
# sits within 0.1% of its speed by them; an unrelated pair of the same length
# is 4% from every PAL candidate.
CORROBORATION_DURATION_SHARE = 0.02

# How many times a drifting survey may be measured again with the speed taken
# off. The first pass removes nearly all of it; a second catches a conversion
# the first snapped to a neighbouring one (25/24 for 25/23.976).
MAX_RATE_REFINEMENTS = 2

# A measured drift is taken to be a standard conversion when it lies this close
# to one, as a share of that conversion's own drift (with RATE_SNAP_FLOOR as
# the least). Smeared windows measure a 24 -> 23.976 drift of 1.0 ms/s as
# 0.92 ms/s; the nearest other conversions sit 100% away.
RATE_SNAP_SHARE = 0.15
RATE_SNAP_FLOOR_MS_PER_S = 0.03

# The fast route: one long window for the offset, one short check near the end.
# Decoding the whole offset range as padding around every survey window
# re-reads the file over and over -- six windows at a five-minute offset
# decode an hour of audio -- so a large search range gets this instead: a
# single long window finds the offset, and a short window near the end of the
# file, decoded around the predicted position rather than the whole range,
# catches a false lock or a drifting pair by disagreeing with it.
FAST_WINDOW_S = 300.0
# Below this span the fast window leaves no room for a separate end check, and
# the survey is cheap anyway because short files cap the decode naturally.
FAST_MIN_SPAN_S = 180.0
# Slack the end check gives the predicted offset, in seconds. Absorbs
# measurement noise; real drift is far larger and pushes the check's answer
# out of agreement instead.
FAST_VERIFY_MARGIN_S = 5.0
# If the end check disagrees with the long window by more than this, the pair
# is drifting or cut and the full survey takes over.
FAST_AGREEMENT_MS = 500.0
# How long the end check is. The survey's window length is calibrated for the
# correlation threshold, so it is reused rather than invented.
FAST_CHECK_WINDOW_S = 45.0

# A survey is the whole answer only when it is clean (see _survey_is_clean):
# confidence at the "high" band, and every window within this of the others.
# 20ms is the smallest step a cut makes that the step search looks for.
CLEAN_MIN_CONFIDENCE = 0.75
CLEAN_MAX_SPREAD_MS = 20.0

# How far either side of the offset the previous window found a survey window
# searches once one has found it. Six windows each decoding the whole search
# range around themselves re-read minutes of the dub per window -- 165 s of it
# for a 45 s window at the default 60 s range -- to confirm an offset the first
# window already placed. A cut smaller than this is still found where it is;
# anything the narrowed window cannot settle is measured again in full (see
# ``sweep`` in ``analyze_pair``), so narrowing never decides an answer alone.
NARROW_SEARCH_MS = 2000.0

# A narrowed window that lands within this of the offset it was centred on
# agrees with the window before it, and is taken as measured. One that lands
# anywhere else in its range is either a cut or a stray peak, which only the
# waveforms (``OffsetEstimate.confirmed``) or the full search can tell apart.
NARROW_AGREEMENT_MS = 20.0

# The full mix and the dialogue-free mix found the same offset when they are
# this close; then the sharper of the two speaks for the window.
MIX_AGREEMENT_MS = 5.0

# Windows that sit this close to what the whole file is doing agree with each
# other (see ``_agreement_confidence``). Wide enough for the envelope's own
# scatter, which is well under a millisecond, and far narrower than anything
# a coincidence would land in.
AGREEMENT_MS = 5.0

# The most that windows agreeing may claim on their own. Agreement says the
# offset is real; it cannot rule out that both files share a mistake.
AGREEMENT_CONFIDENCE_CAP = 0.99

# A step no larger than this many of the video's frames is a slip, not a
# different cut: one frame, plus room for the measurement either side. Below
# it the offset after the step is still inside what a viewer can notice
# (ITU-R BT.1359 puts that at 45 ms for sound early, 125 ms late), so the
# delay before it is worth applying; above it the two halves need their own.
SLIP_MAX_FRAMES = 1.1

# What a slip may always measure, and all it may when the video's frame rate
# is unknown: the BT.1359 threshold for sound arriving early, the stricter of
# its two.
SLIP_FALLBACK_MS = 45.0

# Spans the dub does not cover that leave a slip a slip: it starts late, ends
# early, or is silent there. A span it covers with other material, or matched
# nothing in, is a difference in the cut.
SLIP_TOLERATED_GAPS = frozenset({"head", "tail", "silent"})

# Where the survey's progress ends and the timeline's begins, when it may run.
TIMELINE_PROGRESS_FROM = 60

# A timeline result is fully trusted once the dub covers this much of the
# video, and the stretch the delay is read from must be this long to stand on
# its own (see _timeline_confidence).
TIMELINE_FULL_COVERAGE = 0.8
TIMELINE_MIN_STRETCH_S = 30.0

# The most a survey resting on fewer than MIN_USABLE_WINDOWS windows, with
# nothing else confirming it, may claim: under MkvBatchMux's and the report's
# 0.5 bar for using a delay unchecked.
WEAK_SURVEY_CONFIDENCE = 0.45

# How many new windows on each side of a loose step may be spent asking whether
# they read that side's level (see _levels_reproduced).
STEP_CORROBORATIONS = 2

# The shortest such window worth measuring between two of the survey's.
MIN_CORROBORATION_S = 10.0

# Windows that each found the dub but sit this far apart, typically, from what
# the file is doing cannot place it to better than that, however sure each
# one was. Half the 45 ms BT.1359 puts lip sync at: past it, a delay read from
# them can be visibly wrong. Measured on a pitch-kept PAL dub over music that
# repeats every few seconds, six windows scattered by +-150 ms and their delay
# came out 220 ms wrong at 68%.
SCATTER_MS = 25.0

# Student's t at 98%, two-sided, for 1 to 10 degrees of freedom; more than that
# is taken as the last. A slope has to stand this many of its standard errors
# from none to be a drift at all (see _slope_measured). The first, for a
# survey of three windows, is at 95%: 98% asks for 31.8, which a 24 -> 23.976
# dub smeared by its own drift does not reach on a short file.
STUDENT_T_98 = (12.71, 6.96, 4.54, 3.75, 3.36, 3.14, 3.00, 2.90, 2.82, 2.76)

# Windows at one end of the file further than this many times the typical
# scatter from the rest are somewhere else, not scattered (see
# _note_stray_stretch).
STRAY_SCATTERS = 4.0

# Shorter than this, a file has too little audio to measure a delay from:
# fewer than two of the shortest windows the survey uses.
MIN_MEASURABLE_S = 10.0

# How a plan says a file decoded with errors (dubsync.plan_dubsync); the one
# plan warning a measurement repeats.
DAMAGE_MARK = " is damaged: FFmpeg hit "

# This many edits marked to check (see dubsync.SHORT_STRETCH_S) are one region
# to look at, not a list of independent edits, and are said to be.
DOUBTFUL_RUN = 3


@dataclass
class WindowResult:
    """One measurement at one point in the file."""

    position_s: float
    estimate: OffsetEstimate
    agrees: Optional[bool] = None
    """Whether this window sits with the others (AGREEMENT_MS); None when it
    found nothing to agree with."""

    @property
    def usable(self) -> bool:
        return self.estimate.matched

    def to_dict(self, codec_delay_ms: float = 0.0) -> dict:
        """The window as a details view shows it, on the result's terms: the
        codec delay the result removed is removed here too."""
        estimate = self.estimate
        return {
            "positionS": self.position_s,
            "matched": estimate.matched,
            "delayMs": estimate.delay_ms - codec_delay_ms if estimate.matched else None,
            "confidence": estimate.confidence,
            "peakRatio": estimate.peak_ratio,
            "confirmed": estimate.confirmed,
            "waveformRatio": estimate.waveform_ratio if estimate.confirmed else None,
            "mix": estimate.mix,
            "agrees": self.agrees,
            "reason": estimate.reason,
        }


@dataclass
class PairResult:
    """Everything learned about one primary/secondary pair."""

    primary_path: str
    secondary_path: str
    delay_ms: Optional[float] = None
    """The offset. Where nothing drifts it is the one a correction applies, the
    same number as `delay_at_start_ms`. With drift no one number describes the
    file: this is then the value at its midpoint, and corrections must use
    `delay_at_start_ms`, since they are applied from t=0."""
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
    method: str = "survey"
    """How the numbers were reached. ``survey``: the windows alone, which is
    the whole answer for a file that is one offset throughout. ``timeline``:
    the dub laid along the whole video (see ``_measure_timeline``), because the
    survey saw a cut, a drift, a speed change, windows that found nothing, or
    too weak a match to stand on its own."""
    edits: List[dict] = field(default_factory=list)
    """Every place the dub departs from the video, however small, from the
    timeline route (``DubSyncPlan.edits_and_gaps``). Empty for a survey."""
    gaps: List[dict] = field(default_factory=list)
    """Spans of the video the dub does not cover, from the timeline route."""
    warnings: List[str] = field(default_factory=list)
    """Things to check before trusting the delay."""
    rate_guide: Optional[dict] = None
    """The frame-rate conversion the dub needs, from which rate to which
    (``framerate.conversion_guide``); None when it runs at the video's speed."""
    timeline: Optional[dict] = None
    """The whole-timeline plan the timeline route measured from, so a caller
    that wants it need not plan the pair a second time."""
    timeline_description: Optional[str] = None
    """The plan as a table (``DubSyncPlan.describe``), for a details view."""
    secondary_sample_rate: Optional[int] = None
    """The compared track's sample rate, for the exact resample filter the
    frame-rate guide writes out."""
    timeline_error: Optional[str] = None
    """Why the timeline route could not measure the pair, when it was tried
    and could not."""
    search_ms: Optional[float] = None
    """How far either side of its centre the narrowest window searched, which
    is what a coincidence had to land inside (see ``_agreement_confidence``)."""
    agreeing_windows: int = 0
    """How many windows sit within AGREEMENT_MS of what the file is doing."""

    @property
    def slip_limit_ms(self) -> float:
        """The largest step that is a slip rather than a cut: SLIP_MAX_FRAMES of
        the video, or SLIP_FALLBACK_MS where that is more -- at 50 or 60 fps a
        frame is under the smallest step a cut is looked for at all."""
        if self.primary_fps and self.primary_fps > 0:
            return max(1000.0 / self.primary_fps * SLIP_MAX_FRAMES, SLIP_FALLBACK_MS)
        return SLIP_FALLBACK_MS

    @property
    def is_minor_slip(self) -> bool:
        """Whether every step in the pair is a slip of about a frame or less.

        Measured from the start, so slips that add up -- five 40 ms steps the
        same way are 200 ms by the end -- are a cut, not five slips. A frame
        rate mismatch is never a slip, and neither is any stretch of the
        video the dub does not follow: a scene it replaces or misses inside an
        edit, or a span it matched nothing in. Only the dub starting late or
        ending early, or falling silent, leaves a slip a slip.
        """
        if not self.is_likely_cut or self.is_rate_mismatch:
            return False
        if any(gap.get("reason") not in SLIP_TOLERATED_GAPS for gap in self.gaps):
            return False
        limit = self.slip_limit_ms
        if self.edits:
            first = self.edits[0].get("offsetBeforeMs")
            for edit in self.edits:
                after = edit.get("offsetAfterMs")
                if first is None or after is None or edit.get("kind") == "replaced":
                    return False
                lost_ms = 1000.0 * max(edit.get("missingS") or 0.0, edit.get("extraS") or 0.0)
                if lost_ms > limit or abs(after - first) > limit:
                    return False
            return True
        if self.cut is not None:
            return abs(self.cut.magnitude_ms) <= limit
        return False

    @property
    def is_likely_cut(self) -> bool:
        return bool(self.cut) or bool(self.edits) or bool(
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
            "agreeingWindows": self.agreeing_windows,
            "windowDetails": [w.to_dict(self.codec_delay_ms) for w in self.windows],
            "error": self.error,
            "elapsedMs": self.elapsed_ms,
            "primaryDurationS": self.primary_duration_s,
            "secondaryDurationS": self.secondary_duration_s,
            "primaryTrack": self.primary_track,
            "secondaryTrack": self.secondary_track,
            "primaryFps": self.primary_fps,
            "secondaryFps": self.secondary_fps,
            "isLikelyCut": self.is_likely_cut,
            "isMinorSlip": self.is_minor_slip,
            "slipLimitMs": self.slip_limit_ms,
            "cutPositionS": self.cut.position_s if self.cut else (self.edits[0]["videoS"] if self.edits else None),
            "cutUncertaintyS": self.cut.uncertainty_s if self.cut else (self.edits[0]["uncertaintyS"] if self.edits else None),
            "cutMagnitudeMs": self.cut.magnitude_ms if self.cut else (self.edits[0]["jumpMs"] if self.edits else None),
            "isRateMismatch": self.is_rate_mismatch,
            "speedCompensation": self.speed_compensation,
            "codecDelayMs": self.codec_delay_ms,
            "primaryCodec": self.primary_codec,
            "secondaryCodec": self.secondary_codec,
            "rateDiagnosis": self.rate_diagnosis.to_dict() if self.rate_diagnosis else None,
            "method": self.method,
            "edits": list(self.edits),
            "gaps": list(self.gaps),
            "warnings": list(self.warnings),
            "rateGuide": self.rate_guide,
            "timeline": self.timeline,
            "timelineDescription": self.timeline_description,
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
    prefer_fast: bool = False,
    timeline: bool = True,
    find_cuts: bool = True,
    find_speed: bool = True,
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
        prefer_fast: try the fast route first. One long window and an end check
            replace the survey when they agree; anything they cannot settle
            falls through to the full survey unchanged.
        timeline: when the survey is not clean -- a window found nothing, the
            windows disagree, the offset steps or drifts, the speed had to be
            changed, or the match is weak -- lay the dub along the whole video
            and measure from that instead (see ``_measure_timeline``). A clean
            survey is returned exactly as it was measured.
        find_cuts: report a step between the survey windows as a cut, and
            probe for where it is. False reports one delay and no cut, and
            probes only for whether a step the windows do not settle is there:
            the delay is the one before it either way, so the setting decides
            what is said about a pair and never its delay.
        find_speed: when too few windows find the dub at its own speed, try
            the standard frame-rate conversions before giving up.
    """
    result = PairResult(
        primary_path,
        secondary_path,
        primary_track=primary_track,
        secondary_track=secondary_track,
    )

    # The survey's share of the progress bar: all of it when it is the whole
    # answer, the first part when the timeline may follow it.
    survey_share = TIMELINE_PROGRESS_FROM / 100.0 if timeline else 1.0

    def report(percent: int) -> None:
        if progress:
            progress(max(0, min(100, int(percent * survey_share))))

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
        result.secondary_sample_rate = int(secondary_stream[1]) if secondary_stream[1] else None
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
        # Said up front, in the file's terms, rather than left to surface as a
        # decoder's "Invalid argument" or a correlation that found nothing.
        for info, path, track, role in (
            (primary_info, primary_path, primary_track, "video"),
            (secondary_info, secondary_path, secondary_track, "dub"),
        ):
            problem = _unusable(info, path, track, role)
            if problem:
                result.error = problem
                return result
        if os.path.realpath(primary_path) == os.path.realpath(secondary_path) and primary_track == secondary_track:
            result.warnings.append(
                "The video and the dub are the same file and track, so the delay is zero by definition. "
                "Check the pairing."
            )

        # Measure at the files' own speed first. A duration ratio can sit
        # inside the tolerance of a standard conversion without being one -- a
        # dub that lacks the video's recap runs ~4% shorter, which lands on
        # 25/24 -- and stretching before measuring turns that pair into a
        # confident wrong answer. A real conversion is taken off below, in
        # `_search_speed`, only after the native measurement has failed and the
        # audio itself confirms the rate.
        secondary_rate = ANALYSIS_SR

        def sweep(rate: int, progress_from: int = 5, progress_to: int = 80):
            """Measure every window with the secondary running at this speed.

            Returns the window length, the windows, and how far either side of
            its centre the narrowest window searched (``PairResult.search_ms``).
            """
            # Only where both files have content, in the primary's timeline --
            # which is what the secondary's duration becomes once the speed
            # difference is taken off it.
            speed = rate / ANALYSIS_SR
            span = min(primary_duration, secondary_duration * speed)
            length = min(window_s, max(5.0, span / 3.0))
            spots = plan_windows(span, length, window_count)
            measured = []
            searched = max_offset_ms
            # The offset the last window found, which the next one searches
            # around. Held on the clock the windows are measured on: at the
            # files' own speed that is the offset itself, and with a speed
            # taken off it is the offset on the stretched clock, which a real
            # conversion holds constant however far the native one drifts.
            anchor = None
            anchored_at = None
            narrowing = max_offset_ms > NARROW_SEARCH_MS
            # A window measured through a speed change is often a weak one --
            # a dub sped up with its pitch kept peaks under MIN_PEAK_RATIO even
            # at the right speed -- and searched within NARROW_SEARCH_MS of the
            # windows before it, landing beside them is the evidence a strong
            # peak would otherwise have to be.
            narrow_bar = MIN_PEAK_RATIO if rate == ANALYSIS_SR else CORROBORATION_PEAK_RATIO
            # Before any window has found the dub, the first is searched around
            # zero, where most pairs sit, and given the full range only if that
            # does not settle it. The full range reads the whole of it from the
            # dub -- two minutes of a 4K remux, a fifth of what a pair read.
            # Nothing short of the waveforms agreeing settles it, and one try
            # that does not is enough to stop trying.
            near_zero = narrowing
            from_zero = None
            for index, position in enumerate(spots):
                if token:
                    token.raise_if_cancelled()
                estimate = None
                if anchor is not None and narrowing:
                    # Overlapping the anchor's window, landing beside it is
                    # partly the same audio agreeing with itself, not evidence.
                    bar = narrow_bar if position >= anchored_at + length else MIN_PEAK_RATIO
                    narrow = _measure_window(
                        primary_path, secondary_path, position, length,
                        max_offset_ms, token, primary_track, secondary_track, rate,
                        expect_ms=anchor, min_peak_ratio=bar,
                    )
                    if narrow.matched and (
                        narrow.confirmed
                        or abs(_compensated_ms(position, narrow.delay_ms, speed) - anchor) <= NARROW_AGREEMENT_MS
                    ):
                        estimate = narrow
                        searched = NARROW_SEARCH_MS
                elif anchor is None and near_zero:
                    near = _measure_window(
                        primary_path, secondary_path, position, length,
                        max_offset_ms, token, primary_track, secondary_track, rate,
                        expect_ms=0.0,
                    )
                    if near.matched and near.confirmed:
                        estimate = near
                        searched = NARROW_SEARCH_MS
                        from_zero = index
                    else:
                        near_zero = False
                if estimate is None:
                    estimate = _measure_window(
                        primary_path, secondary_path, position, length,
                        max_offset_ms, token, primary_track, secondary_track, rate,
                    )
                if estimate.matched:
                    anchor = _compensated_ms(position, estimate.delay_ms, speed)
                    anchored_at = position
                measured.append(WindowResult(position, estimate))
                report(progress_from + int((progress_to - progress_from) * (index + 1) / len(spots)))

            # A loop in the music can confirm a period away from a true offset
            # that lies outside the search around zero. The windows after it
            # then find the dub elsewhere, and the first is given the full
            # range after all, as it always had.
            if from_zero is not None:
                others = [
                    _compensated_ms(w.position_s, w.estimate.delay_ms, speed) for i, w in enumerate(measured)
                    if i != from_zero and w.estimate.matched
                ]
                first = measured[from_zero]
                if others and abs(
                    float(np.median(others)) - _compensated_ms(first.position_s, first.estimate.delay_ms, speed)
                ) > NARROW_AGREEMENT_MS:
                    measured[from_zero] = WindowResult(spots[from_zero], _measure_window(
                        primary_path, secondary_path, spots[from_zero], length,
                        max_offset_ms, token, primary_track, secondary_track, rate,
                    ))
            return length, measured, searched

        report(5)
        result.search_ms = max_offset_ms
        fast = None
        if prefer_fast:
            fast = _fast_pair(
                primary_path, secondary_path, primary_duration, secondary_duration,
                max_offset_ms, token, primary_track, secondary_track, window_s,
            )
            if fast is not None:
                report(50)

        if fast is not None:
            effective_window, windows = fast
            result.speed_compensation = 1.0
        else:
            effective_window, windows, result.search_ms = sweep(secondary_rate)

            # Too little to work with, which is what a rate conversion looks
            # like when the durations did not predict it -- and they often
            # cannot, since a dub that starts late or an episode with different
            # credits moves the ratio further than the conversion does. Ask the
            # audio instead.
            if find_speed and sum(1 for w in windows if w.usable) < MIN_USABLE_WINDOWS:
                alternative = _search_speed(
                    primary_path, secondary_path, effective_window, max_offset_ms,
                    token, primary_track, secondary_track,
                    speed_candidates(primary_duration, secondary_duration),
                    secondary_rate, primary_duration, secondary_duration,
                )
                if alternative is not None and alternative != secondary_rate:
                    secondary_rate = alternative
                    effective_window, windows, result.search_ms = sweep(secondary_rate)

            result.speed_compensation = secondary_rate / ANALYSIS_SR

        result.window_s = effective_window
        result.windows = windows

        def locate(step: Step) -> bool:
            return _localize_cut(
                step, primary_path, secondary_path, effective_window, max_offset_ms, token,
                primary_track, secondary_track, cut_probes, secondary_rate,
            )

        def settle(step: Optional[Step]) -> Optional[Step]:
            """The step, if it is one; otherwise every number again without it.

            A level no windows share is reproduced here, from both sides. A
            single window's is ``_localize_cut``'s to reproduce: below with the
            cut check on, here with it off, where nothing else calls it.
            """
            if step is None:
                return None
            if step.lone_group:
                believed = find_cuts or locate(step)
            else:
                believed = not step.loose or (cut_probes > 0 and _levels_reproduced(
                    step, result.windows, primary_path, secondary_path, effective_window,
                    max_offset_ms, token, primary_track, secondary_track, secondary_rate,
                ))
            return step if believed else _reconcile(result, allow_step=False)

        # First pass proposes a split; probing then either pins the cut down or
        # withdraws it, and the second pass re-derives every number from the
        # answer. Re-running is cheaper than unpicking a provisional result,
        # and it keeps one code path deciding what a segmented file reports.
        #
        # With the cut check off no cut is reported and none is probed for that
        # the windows already show, but a step is still one: an average of its
        # two levels fits neither half of the film (131 ms out on one whose
        # windows had placed it exactly), and it made a pair's delay depend on
        # a setting. The delay is then the one before the step, as it is with
        # the check on, and a note says so.
        step = settle(_reconcile(result))

        # A steady drift the windows measured through is a speed difference
        # they were smeared by: at 24 against 23.976 a 45 s window slides 45 ms
        # inside itself, so its peak is blunt, rarely confirmed, and lands
        # wherever the loudest moment of the window sits -- up to half that
        # from the truth, which is what the delay at the start inherits. Taking
        # the speed off and measuring again makes every window sharp. Only kept
        # when the windows really are sharper for it (see _sharper), and never
        # for a step, which no speed explains.
        if fast is None and find_speed and step is None:
            for _ in range(MAX_RATE_REFINEMENTS):
                rate = _compensating_rate(result, secondary_rate)
                if rate is None:
                    break
                trial_window, trial, trial_search = sweep(rate, 80, 92)
                if not _sharper(trial, windows):
                    break
                secondary_rate, effective_window, windows = rate, trial_window, trial
                result.speed_compensation = secondary_rate / ANALYSIS_SR
                result.window_s = effective_window
                result.windows = windows
                result.search_ms = trial_search
                step = settle(_reconcile(result))
                if step is not None:
                    break

        if step is not None and find_cuts:
            if locate(step):
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
            cut_position_s=result.cut.position_s if result.cut and find_cuts else None,
            cut_magnitude_ms=result.cut.magnitude_ms if result.cut and find_cuts else None,
            cut_uncertainty_s=result.cut.uncertainty_s if result.cut and find_cuts else None,
        )
        if result.cut is not None and result.cut.levels > 2:
            result.warnings.append(
                f"The offset changes at least {result.cut.levels - 1} times across the file, not once: the dub was "
                f"edited in several places (scenes cut or trimmed). The delay aligns only the part before the first "
                f"change, about {_clock_of(result.cut.position_s)} in; the whole-timeline check maps every one."
            )
        if not _survey_is_clean(result):
            measured = timeline and _measure_timeline(
                result, primary_path, secondary_path, max_offset_ms, token,
                primary_track, secondary_track,
                (lambda percent: progress(TIMELINE_PROGRESS_FROM + int(percent * (100 - TIMELINE_PROGRESS_FROM) / 100)))
                if progress else None,
            )
            if not measured:
                _guard_weak_survey(result)
                _guard_scattered_survey(result)
                _note_stray_stretch(result)
                if result.cut is not None and not find_cuts and result.cut.levels <= 2:
                    result.warnings.append(
                        f"The offset changes by {result.cut.magnitude_ms / 1000.0:+.3f} s at about "
                        f"{_clock_of(result.cut.position_s)}, so the delay aligns only the part before it. "
                        "The cut check finds where."
                    )
                if result.error:
                    result.error = _explain_no_match(result, timeline, find_speed)
        if not find_cuts:
            # Off means not reported; the numbers above already rest on it.
            result.cut = None
        if progress:
            progress(100)
        return result

    except MediaError as exc:
        result.error = str(exc)
        return result


def _survey_is_clean(result: PairResult) -> bool:
    """Whether the survey is the whole answer: one offset across the file.

    Every window found the dub, they agree to within a fraction of a frame,
    nothing steps, nothing drifts, the files ran at their own speed and the
    match is strong. A survey like that has nothing a longer look would add,
    and it is returned untouched, so a clean pair measures exactly as it
    always has.
    """
    if result.error or result.delay_ms is None or not result.windows:
        return False
    usable = [w for w in result.windows if w.usable]
    if len(usable) < len(result.windows):
        return False
    if result.is_likely_cut or result.is_rate_mismatch or result.has_significant_drift:
        return False
    if abs(result.speed_compensation - 1.0) > 1e-9:
        return False
    if result.confidence < CLEAN_MIN_CONFIDENCE:
        return False
    offsets = [w.estimate.delay_ms for w in usable]
    return max(offsets) - min(offsets) <= CLEAN_MAX_SPREAD_MS


def _measure_timeline(
    result: PairResult,
    primary_path: str,
    secondary_path: str,
    max_offset_ms: float,
    token: Optional[CancellationToken],
    primary_track: int,
    secondary_track: int,
    progress: Optional[Callable[[int], None]],
) -> bool:
    """Measure the pair by laying the dub along the whole video.

    Six windows answer one question well -- is this one offset? -- and every
    other question badly. Past a cut bigger than the search range the windows
    find nothing and the survey calls the file one delay; two cuts are one
    step to it, placed to within minutes; a frame-rate conversion needs one
    window to clear a bar a weak dub never clears; and a single window that
    locked onto a repeated cue is reported as the answer. The dub-sync planner
    reads both tracks end to end and finds every stretch, every cut and the
    speed, so when the survey is not clean it is the better witness.

    Returns:
        Whether the result now comes from the timeline. False leaves the
        survey's numbers in place with a warning saying why.
    """
    from .dubsync import DEFAULT_SEARCH_S, plan_dubsync

    try:
        plan = plan_dubsync(
            primary_path, secondary_path,
            video_track=primary_track, dub_track=secondary_track,
            search_s=max(DEFAULT_SEARCH_S, max_offset_ms / 1000.0),
            token=token,
            progress=(lambda percent, _stage: progress(percent)) if progress else None,
        )
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - the survey still stands without it
        result.timeline_error = f"{type(exc).__name__}: {exc}"
        if result.error is None:
            result.warnings.append(
                f"The whole-timeline check stopped with an error ({result.timeline_error}), "
                "so this delay rests on the survey alone."
            )
        return False
    if plan.error or not plan.dub_segments:
        result.timeline_error = plan.error or "nothing could be placed"
        if result.error is None:
            result.warnings.append(
                f"The whole-timeline check could not lay the dub on the video ({result.timeline_error}), "
                "so this delay rests on the survey alone."
            )
        result.warnings.extend(w for w in plan.warnings if DAMAGE_MARK in w and w not in result.warnings)
        return False
    _apply_timeline(result, plan)
    return True


def _apply_timeline(result: PairResult, plan) -> None:
    """Restate the result from a whole-timeline plan, in the survey's terms.

    The plan's offsets are on the video's clock, with the dub already brought
    to the video's speed. The survey reports the files the user has: at video
    time t the dub's own time is (t + offset) / speed, so its offset there is
    that less t, and at t=0 it is offset / speed. Reported that way the delay
    means what a survey's does, with or without a speed change; the delay to
    pair with a timestamp stretch is in ``rate_guide`` instead.
    """
    speed = plan.speed if plan.speed and plan.speed > 0 else 1.0
    dubs = plan.dub_segments
    first, last = dubs[0], dubs[-1]
    codec = result.codec_delay_ms

    def native_ms(offset_s: float, at_s: float) -> float:
        return ((at_s + offset_s) / speed - at_s) * 1000.0 - codec

    changed_speed = abs(speed - 1.0) > 1e-9
    result.method = "timeline"
    result.error = None
    result.cut = None
    result.speed_compensation = speed
    result.drift_ms_per_s = (1.0 / speed - 1.0) * 1000.0 if changed_speed else 0.0
    result.delay_at_start_ms = native_ms(first.offset_s, 0.0)
    # As the survey quotes it: with a drift, the value at the middle of the
    # stretch a delay can act on.
    result.delay_ms = (
        native_ms(first.offset_s, (first.start_s + first.end_s) / 2.0) if changed_speed
        else result.delay_at_start_ms
    )
    result.start_delay_ms = native_ms(first.offset_s, first.start_s)
    result.end_delay_ms = native_ms(last.offset_s, last.end_s)
    result.confidence = _timeline_confidence(plan)
    result.edits, result.gaps = plan.edits_and_gaps()
    result.rate_guide = conversion_guide(
        plan.video_fps or result.primary_fps, speed, plan.dub_duration_s, plan.rate_confirmed,
        offset_ms=first.offset_s * 1000.0 - codec,
        sample_rate=result.secondary_sample_rate,
    )
    # The plan's own warnings are written for laying the dub ("filled from
    # the original") and stay with it in ``timeline``; a measurement carries
    # the ones that bear on the delay.
    result.warnings.extend(w for w in plan.warnings if DAMAGE_MARK in w and w not in result.warnings)
    if plan.rate_confirmed is False:
        result.warnings.append(
            "The dub's frame rate could not be confirmed from the audio, so it was taken to be the video's. "
            "If the dub was mastered at another rate (23.976 against 24 is the common case), the delay will drift."
        )
    result.timeline = plan.to_dict()
    result.timeline_description = plan.describe()
    doubtful = [edit for edit in result.edits if edit.get("check")]
    if len(doubtful) >= DOUBTFUL_RUN:
        result.warnings.insert(0, (
            f"{len(doubtful)} of the {len(result.edits)} edits rest on short stretches of dub, from "
            f"{_clock_of(doubtful[0]['videoS'])} to {_clock_of(doubtful[-1]['videoEndS'])}: the two versions "
            "diverge there (often different end credits or a re-edited scene), and some of those edits may be "
            "a repeated passage matched in the wrong place. Check that part as a whole."
        ))

    guide = result.rate_guide
    first_edit = result.edits[0] if result.edits else None
    story = []
    if guide:
        story.append(guide["instruction"])
    if result.edits:
        story.append(
            f"The dub departs from the video {len(result.edits)} time{'s' if len(result.edits) != 1 else ''}; "
            f"the first is at {first_edit['description']} The delay aligns everything before it."
        )
    if not story:
        story.append("The whole timeline agrees with one delay.")
    named = bool(guide and guide["named"])
    result.rate_diagnosis = RateDiagnosis(
        drift_ms_per_s=result.drift_ms_per_s,
        speed_ratio=guide["speed"] if guide else 1.0,
        source_fps=guide["dubFps"] if named else None,
        target_fps=guide["videoFps"] if named else None,
        is_rate_mismatch=bool(guide),
        is_likely_cut=bool(result.edits),
        cut_position_s=first_edit["videoS"] if first_edit else None,
        cut_magnitude_ms=first_edit["jumpMs"] if first_edit else None,
        explanation=" ".join(story),
    )


def _unusable(info, path: str, track: int, role: str) -> Optional[str]:
    """Why a file cannot be measured as it stands, or None if it can."""
    name = os.path.basename(path)
    if not info.has_audio:
        return f"{name} has no audio track{' to compare against' if role == 'video' else ''}."
    tracks = getattr(info, "audio_tracks", None) or []
    if tracks and not 0 <= track < len(tracks):
        return (
            f"{name} has {len(tracks)} audio track{'s' if len(tracks) != 1 else ''}, so track {track + 1} "
            "does not exist. Choose a track the file has."
        )
    if info.duration and info.duration < MIN_MEASURABLE_S:
        return (
            f"{name} is only {info.duration:.1f} s long; measuring a delay needs at least "
            f"{MIN_MEASURABLE_S:.0f} s of audio."
        )
    return None


def _explain_no_match(
    result: PairResult, searched_timeline: bool = True, searched_speed: bool = True
) -> str:
    """The reason no delay could be found, in terms of the files.

    By the time this is asked the survey, and the whole-timeline search when
    it was switched on, have come up empty, so the survey's per-window reason
    ("no distinct correlation peak") is only half the story and names nothing
    a user can check. A file problem the timeline ran into outranks it;
    silence is said of the track that is silent; otherwise the two are
    different audio, or audio only a check that was switched off could place.
    """
    reasons = [w.estimate.reason for w in result.windows if w.estimate.reason]
    planned = getattr(result, "timeline_error", None)
    if planned and "matched" not in planned and "placed" not in planned:
        return planned
    if reasons and all(reason == SILENT_SECONDARY for reason in reasons):
        return (
            f"The dub ({result.secondary_name}) is silent everywhere it was sampled. It may be an empty or muted "
            "track, or the wrong track of the file."
        )
    if reasons and all(reason == SILENT_PRIMARY for reason in reasons):
        return (
            f"The video's audio ({result.primary_name}) is silent everywhere it was sampled, so there is nothing "
            "to compare the dub with. Choose another reference track."
        )
    if reasons and all(reason in (NO_PEAK, SILENT_PRIMARY, SILENT_SECONDARY) for reason in reasons):
        if not searched_timeline:
            skipped = ([] if searched_speed else ["frame-rate"]) + ["whole-timeline"]
            return (
                "The dub does not match the video at any of the sample windows. They are probably different audio "
                "-- another episode or cut, a different film, or a track that is not this video's dub. Check the "
                f"pairing and the chosen tracks, or turn on the {' and '.join(skipped)} "
                f"check{'s' if len(skipped) > 1 else ''} in Settings to search further."
            )
        return (
            "The dub does not match the video anywhere: neither the sample windows nor a search of the whole "
            "timeline found it. They are probably different audio -- another episode or cut, a different film, "
            "or a track that is not this video's dub. Check the pairing and the chosen tracks."
        )
    return result.error


def _clock_of(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000.0))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours}:{minutes:02d}:{rest / 1000.0:06.3f}"


def _timeline_confidence(plan) -> float:
    """How far to trust a timeline result, from how much of the video the dub
    was found across.

    The planner keeps only stretches that cleared its own evidence bars, so
    what is left to judge is coverage: a dub found across most of the video is
    the same audio beyond reasonable doubt, one found in a corner of it may
    not be. The stretch the delay comes from must also be long enough to stand
    on its own, and a frame rate the audio did not confirm caps it.
    """
    dubs = plan.dub_segments
    covered = sum(s.length_s for s in dubs) / max(plan.video_duration_s, 1e-9)
    confidence = 0.5 + (0.95 - 0.5) * min(1.0, covered / TIMELINE_FULL_COVERAGE)
    if dubs[0].length_s < TIMELINE_MIN_STRETCH_S or plan.rate_confirmed is False:
        confidence = min(confidence, 0.7)
    return confidence


def _guard_weak_survey(result: PairResult) -> None:
    """Keep a survey that barely found the dub from passing as a measurement.

    One or two windows out of six are what a coincidence looks like: a
    repeated cue, a stinger, a sound effect used twice. Nothing else confirmed
    them, so the delay is held below the confidence at which it would be used
    without a look, and the reason is said.
    """
    total = len(result.windows)
    usable = sum(1 for w in result.windows if w.usable)
    if result.delay_ms is None or total < MIN_USABLE_WINDOWS or usable >= MIN_USABLE_WINDOWS:
        return
    result.confidence = min(result.confidence, WEAK_SURVEY_CONFIDENCE)
    result.warnings.append(
        f"Only {usable} of {total} windows found the dub and nothing else confirmed it, so this delay "
        "may be a repeated passage rather than the real offset. Check it before using it."
    )


def _guard_scattered_survey(result: PairResult) -> None:
    """Keep a survey whose windows disagree from passing as a measurement.

    Agreement is what the confidence of a survey mostly rests on; windows
    that each found the dub somewhere different, by more than SCATTER_MS, have
    not agreed on anything. The delay is the middle of what they read, which
    many of them place far better than a few: it is held back when the middle
    itself is that uncertain, and the scatter is only said when it is not -- a
    dub that wanders about the video by 30 ms over twenty windows has one best
    delay, and they know it to about ten. A cut is the other reason windows
    disagree, and it is reported as one, so it is left alone here.
    """
    if result.error or result.delay_ms is None or result.cut is not None:
        return
    usable = sorted((w for w in result.windows if w.usable), key=lambda w: w.position_s)
    if len(usable) < MIN_USABLE_WINDOWS:
        return
    positions = np.array([w.position_s for w in usable], dtype=np.float64)
    offsets = np.array([w.estimate.delay_ms for w in usable], dtype=np.float64)
    model = _model_values(positions, offsets, np.zeros(len(offsets), dtype=bool))
    scatter = float(np.median(np.abs(offsets - model)))
    if scatter <= SCATTER_MS:
        return
    # A minority that agrees can still be the answer, the rest having found
    # nothing but noise: unless agreeing that often is what windows scattered
    # this far do by chance.
    agreeing = [w for w in usable if w.agrees]
    if len(agreeing) >= MIN_USABLE_WINDOWS and (
        sum(w.estimate.confirmed for w in agreeing) >= MIN_USABLE_WINDOWS
        or _agreement_confidence(len(agreeing), len(usable), scatter) >= 0.5
    ):
        return
    # The standard error of a median, from the scatter's own robust scale.
    uncertainty = 1.2533 * 1.4826 * scatter / math.sqrt(len(usable))
    if uncertainty > SCATTER_MS:
        result.confidence = min(result.confidence, WEAK_SURVEY_CONFIDENCE)
        result.warnings.append(
            f"The windows found the dub but disagree with each other by about {scatter:.0f} ms, so the delay is "
            "only that accurate. Check it by ear, or turn on the whole-timeline check."
        )
    else:
        result.warnings.append(
            f"The windows disagree with each other by about {scatter:.0f} ms, and the delay is the middle of what "
            "they read: parts of the film may sit that far from it. Check a scene or two."
        )


def _note_stray_stretch(result: PairResult) -> None:
    """Say so when the windows at one end of the file sit together away from
    the rest, and no cut came of it.

    Two or more in a row on the same side of the delay, by more than lip sync
    tolerates and more than the windows scatter anyway, may be a stretch the
    dub was edited in, or the credits' music, or a run of bad readings. One
    delay still fits everything else, so it stands; which of the three it is
    takes a look at a scene there, and only this says where.
    """
    if result.error or result.delay_ms is None or result.cut is not None or result.has_significant_drift:
        return
    usable = sorted((w for w in result.windows if w.usable), key=lambda w: w.position_s)
    if len(usable) < MIN_USABLE_WINDOWS + 2:
        return
    positions = np.array([w.position_s for w in usable], dtype=np.float64)
    offsets = np.array([w.estimate.delay_ms for w in usable], dtype=np.float64)
    away = offsets - _model_values(positions, offsets, np.zeros(len(offsets), dtype=bool))
    limit = max(SLIP_FALLBACK_MS, STRAY_SCATTERS * float(np.median(np.abs(away))))

    def run(values) -> int:
        count = 0
        while count < len(values) and abs(values[count]) > limit and values[count] * values[0] > 0:
            count += 1
        return count

    head, tail = run(away), run(away[::-1])
    for count, stretch, where in (
        (head, away[:head], f"Up to about {_clock_of(float(positions[max(head - 1, 0)]) + (result.window_s or 0.0))}"),
        (tail, away[len(away) - tail:], f"From about {_clock_of(float(positions[-max(tail, 1)]))} to the end"),
    ):
        # A minority, or it is the rest that strayed and the delay with it.
        if count < 2 or 2 * count >= len(usable):
            continue
        typical = float(np.median(stretch))
        result.warnings.append(
            f"{where}, the dub reads about {abs(typical) / 1000.0:.2f} s {'later' if typical > 0 else 'earlier'} than "
            "in the rest of the film, but not steadily enough to call it a cut. Check a scene there."
        )


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
    expect_ms: Optional[float] = None,
    min_peak_ratio: float = MIN_PEAK_RATIO,
    search_waveform: bool = True,
) -> OffsetEstimate:
    """Measure one window, padding the secondary so a shifted match still fits.

    The secondary is decoded with extra margin on both sides. Without it, a
    window near a genuine offset of several seconds would be comparing two
    non-overlapping spans of audio and would correctly find nothing.

    Both tracks are measured twice from one stereo decode: as the full mix,
    and as the music and effects alone (see ``_side``). A dub replaces the
    dialogue and keeps the M&E, so the dialogue in the full mix is noise the
    two tracks do not share -- English in a 5.1 centre against Hindi in a
    2.0 phantom centre scored 92% where the M&E alone scored 99-100%, and
    was up to 0.38 ms out where the M&E was within 0.09 ms.

    Args:
        secondary_rate: the rate to decode the secondary at. Asking ffmpeg for
            ANALYSIS_SR * ratio and then reading the samples as ANALYSIS_SR is
            a time-stretch by that ratio, done by the decoder's own resampler
            rather than by interpolating afterwards. That is how a PAL-sped dub
            is brought onto the video's clock before being correlated; at
            ANALYSIS_SR it is an ordinary decode and changes nothing.
        expect_ms: the offset to search around, NARROW_SEARCH_MS either
            side, instead of the whole range around zero. On the clock the
            window is measured on: with a speed taken off, the stretched one
            (``_compensated_ms``), where a real conversion's offset holds still.
        min_peak_ratio: the envelope prominence a match must clear
            (``correlate.estimate_offset``). Lowered only where agreement with
            other windows is what decides.
        search_waveform: let an unconfirmed window search the waveforms over
            its whole range (``correlate._search_waveform``). Off where
            speeds are being tried: a wrong speed confirms nothing anywhere,
            and searching for it at every one tripled what a pair with no
            match took to say so.
    """
    # How much of the primary's clock one second of decoded secondary covers.
    speed = secondary_rate / ANALYSIS_SR

    # The full head start the user asked for, with no cap of its own. Capping
    # the margin below max_offset_ms quietly made every offset beyond the cap
    # invisible: the matching audio was never decoded, so the correlation ran
    # on unrelated audio and produced a confident wrong answer. The UI bounds
    # max_offset_ms, so this is only ever as large as the user explicitly
    # asked to search.
    margin_s = max_offset_ms / 1000.0
    centre_s = 0.0
    if expect_ms is not None:
        margin_s = NARROW_SEARCH_MS / 1000.0
        # On the onset envelope's frame grid, as the full search always is
        # (it starts whole seconds before the window): a dub read from an
        # arbitrary fraction of a frame is framed out of step with the video,
        # which moved a drifting pair's windows by a millisecond or two.
        frame_s = ENVELOPE_HOP / ANALYSIS_SR
        centre_s = round(expect_ms / 1000.0 / frame_s) * frame_s
    secondary_start = max(0.0, position_s + centre_s - margin_s)
    secondary_window = position_s + centre_s + margin_s + window_s - secondary_start
    if secondary_window <= 0.0:
        return OffsetEstimate(None, 0.0, 0.0, "the expected position lies before the dub starts")

    try:
        primary = load_audio(
            primary_path,
            ANALYSIS_SR,
            duration=window_s,
            offset=position_s,
            token=token,
            track=primary_track,
            channels=2,
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
            channels=2,
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
    if expect_ms is None:
        bounds = {"max_offset_ms": max_offset_ms + head_start_ms}
    else:
        # Around the expected offset, and never past the range the user asked
        # to search: an anchor near its edge would otherwise carry the next
        # window beyond it.
        centre_ms = head_start_ms + expect_ms
        lowest = max(centre_ms - NARROW_SEARCH_MS, head_start_ms - max_offset_ms)
        highest = min(centre_ms + NARROW_SEARCH_MS, head_start_ms + max_offset_ms)
        if highest <= lowest:
            return OffsetEstimate(None, 0.0, 0.0, "the expected offset lies outside the search range")
        bounds = {"search_ms": (lowest, highest)}
    bounds["min_peak_ratio"] = min_peak_ratio
    full_primary, full_secondary = _mix(primary), _mix(secondary)
    full = estimate_offset(full_primary, full_secondary, ANALYSIS_SR, **bounds, search_waveform=False)
    if full.confirmed:
        # The waveforms already agree through the dialogue, which places the
        # window to a fraction of a sample; the M&E could only say the same.
        estimate = full
    else:
        side_primary, side_secondary = _side(primary), _side(secondary)
        me = replace(
            estimate_offset(side_primary, side_secondary, ANALYSIS_SR, **bounds, search_waveform=False), mix="me"
        )
        estimate = _choose_mix(full, me)
        if estimate is me and full.matched and abs(me.delay_ms - full.delay_ms) > MIX_AGREEMENT_MS:
            # Music that repeats sample for sample -- a loop, a reprised cue --
            # confirms one period away as well as at the right offset. The
            # dialogue does not repeat, so where the full mix points and the
            # M&E confirms there too, that is the offset.
            at_full = _refine_against_waveform(side_primary, side_secondary, ANALYSIS_SR, full.delay_ms)
            if at_full is not None:
                estimate = replace(me, delay_ms=at_full[0], waveform_ratio=at_full[1])
        if search_waveform and not estimate.confirmed:
            # Neither mix's envelope led to waveforms that agree. Search the
            # waveforms themselves over the whole range -- the full mix, then
            # the M&E -- before settling for an envelope's answer: on music
            # with a steady beat the envelope's peaks stand level with each
            # other and the wrong one can win in every window alike
            # (correlate._search_waveform). Only here, once both cheap reads
            # have failed, because it costs a transform of the whole range.
            for mix, a, b in (("full", full_primary, full_secondary), ("me", side_primary, side_secondary)):
                searched = estimate_offset(a, b, ANALYSIS_SR, **bounds, search_waveform=True)
                if searched.confirmed:
                    estimate = replace(searched, mix=mix)
                    break
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

    return replace(estimate, delay_ms=absolute)


def _mix(stereo: np.ndarray) -> np.ndarray:
    """The full mix of a stereo decode: to the sample the mono decode FFmpeg
    makes itself (its mono downmix is (L + R) / sqrt(2) of its stereo one)."""
    return (stereo[:, 0] + stereo[:, 1]) * np.float32(1.0 / math.sqrt(2.0))


def _side(stereo: np.ndarray) -> np.ndarray:
    """The mix with everything panned dead centre cancelled out.

    FFmpeg's stereo downmix puts the centre channel into both sides equally,
    so left minus right removes it -- the dialogue of a 5.1 mix, and the
    phantom-centre dialogue of a 2.0 one -- and keeps the music and effects,
    which are spread across the stereo field and are what a dub shares with
    the original. A mono or dual-mono track has nothing left, which
    ``estimate_offset`` rejects as silence, so the full mix answers instead.
    """
    return (stereo[:, 0] - stereo[:, 1]) * np.float32(1.0 / math.sqrt(2.0))


def _choose_mix(full: OffsetEstimate, me: OffsetEstimate) -> OffsetEstimate:
    """Which of a window's two measurements speaks for it.

    Whichever the waveforms confirm, first: GCC-PHAT agreement is evidence
    noise cannot produce, while an envelope peak alone can -- the M&E's
    envelope once locked 108 ms off with a sharper peak than the full mix's
    correct one, and the waveforms rejected it. Without that, the full mix
    stands, as it always has, unless the M&E agrees with it and is sharper,
    or found the dub decisively where the full mix found nothing.
    """
    if me.confirmed and (not full.confirmed or me.waveform_ratio > full.waveform_ratio):
        return me
    if full.confirmed:
        return full
    if not full.matched:
        return me if me.matched and me.peak_ratio >= DECISIVE_PEAK_RATIO else full
    if (
        me.matched
        and abs(me.delay_ms - full.delay_ms) <= MIX_AGREEMENT_MS
        and me.peak_ratio > full.peak_ratio
    ):
        return me
    return full


def _fast_pair(
    primary_path: str,
    secondary_path: str,
    primary_duration: float,
    secondary_duration: float,
    max_offset_ms: float,
    token: Optional[CancellationToken],
    primary_track: int,
    secondary_track: int,
    window_s: float,
):
    """One long window plus an end check: the script-speed route.

    A single long window finds the offset with the same correlation as the
    survey, in one decode instead of six. The end check exists because one
    window is not an answer: a window that locked onto a repeated musical
    phrase looks exactly like a real match until another position disagrees
    with it. The check decodes only around the predicted position -- a small
    margin, not the whole offset range -- so it costs seconds.

    Returns:
        (window_s, windows) when the fast route settled the pair, or None to
        hand over to the full survey.
    """
    span = min(primary_duration, secondary_duration)
    if span < FAST_MIN_SPAN_S:
        return None

    length = min(FAST_WINDOW_S, span)
    first_pos = min(5.0, span * 0.02)
    first = _measure_window(
        primary_path, secondary_path, first_pos, length,
        max_offset_ms, token, primary_track, secondary_track, ANALYSIS_SR,
    )
    if not first.matched:
        return None

    # Near the end, where any drift or cut has moved the offset furthest. The
    # inset matches the survey's, so the check never sits in the credits
    # themselves.
    check_length = min(FAST_CHECK_WINDOW_S, window_s)
    check_pos = max(span - check_length - min(span * 0.02, 5.0), first_pos + 60.0)
    if check_pos <= first_pos:
        return None

    margin_s = abs(first.delay_ms) / 1000.0 + FAST_VERIFY_MARGIN_S
    check = _measure_window(
        primary_path, secondary_path, check_pos, check_length,
        margin_s * 1000.0, token, primary_track, secondary_track, ANALYSIS_SR,
    )
    if not check.matched:
        return None
    if abs(check.delay_ms - first.delay_ms) > FAST_AGREEMENT_MS:
        return None

    return length, [WindowResult(first_pos, first), WindowResult(check_pos, check)]


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
    secondary_duration: Optional[float] = None,
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
            min_peak_ratio=CORROBORATION_PEAK_RATIO, search_waveform=False,
        )
        # Decisive, not merely matched. Six trials are six chances at a
        # coincidence, so the bar here has to be higher than for a single
        # measurement: at the ordinary threshold, trying speeds on two
        # unrelated releases eventually found one that "worked" and reported a
        # confident seven-second delay between films that share no audio.
        #
        # The bar was set on the full mix's envelope peak, so it is applied to
        # that and nothing else; a reading of the M&E alone -- a second chance
        # per trial -- settles a speed only once the waveforms confirm it,
        # which noise cannot do (see correlate.MIN_REFINE_PEAK_RATIO).
        decisive = _decisive(estimate)
        # Short of that, windows elsewhere in the file agreeing with this one
        # settle it: a coincidence does not repeat at the same offset three
        # times over (see CORROBORATION_MS).
        plausible = (
            secondary_duration is None or secondary_duration <= 0
            or abs(float(candidate) - primary_duration / secondary_duration) <= CORROBORATION_DURATION_SHARE
        )
        if not decisive and tried <= CORROBORATED_TRIALS and plausible:
            decisive = _corroborated(
                primary_path, secondary_path, window_s, max_offset_ms, token,
                primary_track, secondary_track, rate, primary_duration, position, estimate,
            )
        if decisive and (best is None or estimate.peak_ratio > best[0]):
            best = (estimate.peak_ratio, rate)
            break

    return best[1] if best is not None else None


def _decisive(estimate: OffsetEstimate) -> bool:
    """Whether one window settles a trial speed by itself (see _search_speed)."""
    return estimate.matched and (
        estimate.confirmed or (estimate.mix == "full" and estimate.peak_ratio >= DECISIVE_PEAK_RATIO)
    )


def _compensated_ms(position_s: float, native_ms: float, speed: float) -> float:
    """A window's offset on the stretched clock it was measured on: the
    inverse of the conversion at the end of ``_measure_window``. Constant
    across the file when the speed is the right one."""
    return ((position_s + native_ms / 1000.0) * speed - position_s) * 1000.0


def _corroborated(
    primary_path: str,
    secondary_path: str,
    window_s: float,
    max_offset_ms: float,
    token: Optional[CancellationToken],
    primary_track: int,
    secondary_track: int,
    rate: int,
    primary_duration: float,
    first_position: float,
    first: OffsetEstimate,
) -> bool:
    """Whether a trial speed holds at two more places in the file.

    Three windows that heard different audio -- the one already measured, if
    it found anything, and up to three more -- must land within
    CORROBORATION_MS of one offset on the stretched clock, or one of the new
    ones must be decisive on its own as the first would have had to be.
    Windows that overlap are one observation read twice (see _independent):
    on a short file the places below share audio, and a coincidence in what
    they share would repeat at the same offset in each. Stops as soon as
    three are out of reach, so a speed that fits nothing costs one window
    more than it used to.
    """
    speed = rate / ANALYSIS_SR
    span = max(0.0, primary_duration - window_s)
    found = []
    if first.matched:
        found.append((first_position, _compensated_ms(first_position, first.delay_ms, speed)))
    places = [span * share for share in (0.15, 0.85, 0.32, 0.68)]
    if _independent([first_position] + places, window_s) < 3:
        return False
    for index, position in enumerate(places):
        remaining = len(places) - index
        if len(found) + remaining < 3:
            return False
        if token:
            token.raise_if_cancelled()
        estimate = _measure_window(
            primary_path, secondary_path, position, window_s,
            max_offset_ms, token, primary_track, secondary_track, rate,
            min_peak_ratio=CORROBORATION_PEAK_RATIO, search_waveform=False,
        )
        if _decisive(estimate):
            return True
        if estimate.matched:
            found.append((position, _compensated_ms(position, estimate.delay_ms, speed)))
        elif not found:
            # Two places in, nothing found at either: not this speed.
            return False
        for _, centre in found:
            agreeing = [at for at, value in found if abs(value - centre) <= CORROBORATION_MS]
            if _independent(agreeing, window_s) >= 3:
                return True
    return False


def _standard_ratios() -> List[Fraction]:
    """Every speed difference two standard frame rates make."""
    ratios = set()
    for audio_rate in COMMON_RATES:
        for video_rate in COMMON_RATES:
            if audio_rate != video_rate:
                ratios.add(audio_rate / video_rate)
    return sorted(ratios)


STANDARD_RATIOS = _standard_ratios()


def _drift_for(ratio: float) -> float:
    """The drift, in ms per second, a dub running ``ratio`` as fast shows."""
    return (1.0 / ratio - 1.0) * 1000.0


def _compensating_rate(result: PairResult, current_rate: int) -> Optional[int]:
    """The decode rate that takes the survey's drift off the dub, or None.

    The drift is the files' own (windows report it on the dub's native clock
    whatever speed they were measured at), so the rate follows from it
    directly, snapped to the standard conversion it sits beside when there is
    one: a smeared survey reads 24 -> 23.976 as 0.92 ms/s, and stretching by
    that instead of by 1.001 would leave 0.08 ms/s behind. None when there is
    no drift worth taking off, too few windows to trust it, or the rate it
    asks for is the one already used.
    """
    drift = result.drift_ms_per_s
    usable = sum(1 for w in result.windows if w.usable)
    if drift is None or usable < MIN_USABLE_WINDOWS:
        return None
    if abs(drift) <= DRIFT_SIGNIFICANT_MS_PER_S or abs(drift) > MAX_RATE_DRIFT_MS_PER_S:
        return None
    ratio = speed_ratio_for(drift)
    if ratio <= 0.0:
        return None
    nearest = None
    for standard in STANDARD_RATIOS:
        expected = _drift_for(float(standard))
        gap = abs(drift - expected)
        if gap <= max(RATE_SNAP_FLOOR_MS_PER_S, RATE_SNAP_SHARE * abs(expected)) and (
            nearest is None or gap < nearest[0]
        ):
            nearest = (gap, float(standard))
    if nearest is not None:
        ratio = nearest[1]
    rate = int(round(ANALYSIS_SR * ratio))
    return None if rate == current_rate else rate


def _sharper(trial: List[WindowResult], before: List[WindowResult]) -> bool:
    """Whether a sweep with the speed taken off measured better than without.

    It must find the dub in at least as many windows, with at least as many
    confirmed by the waveforms, and then find it in more of them, confirm
    more, or be surer of them on average. More windows counts for itself: a
    pitch-kept PAL dub stretched by 25/24 instead of its 25/23.976 found the
    dub in three windows and the right speed in all six, at a lower average
    because the three it gained were its weakest. A drift that was never a
    speed -- several small edits a line happens to fit -- gains nothing from
    stretching and is left as it was.
    """
    def tally(windows: List[WindowResult]):
        usable = [w for w in windows if w.usable]
        confirmed = sum(1 for w in usable if w.estimate.confirmed)
        mean = float(np.mean([w.estimate.confidence for w in usable])) if usable else 0.0
        return len(usable), confirmed, mean

    trial_usable, trial_confirmed, trial_mean = tally(trial)
    usable, confirmed, mean = tally(before)
    if trial_usable < max(MIN_USABLE_WINDOWS, usable) or trial_confirmed < confirmed:
        return False
    return trial_usable > usable or trial_confirmed > confirmed or trial_mean > mean + 0.02


def _levels_reproduced(
    step: Step,
    windows: List[WindowResult],
    primary_path: str,
    secondary_path: str,
    window_s: float,
    max_offset_ms: float,
    token: Optional[CancellationToken],
    primary_track: int,
    secondary_track: int,
    secondary_rate: int,
) -> bool:
    """Whether new audio from each side of a loose step reads that side's level.

    Windows that scatter by tens of milliseconds can be split somewhere along
    the file into two groups with different medians whether or not anything
    was cut, and no arithmetic on the same windows tells the two apart: the
    split was chosen to make them differ. Audio the split was not chosen from
    does. A cut holds its level everywhere on its side, so windows between the
    ones the survey placed, nearest the step and clear of it, read it too; a
    split drawn through scatter predicts them no better than the single offset
    does.
    """
    speed = secondary_rate / ANALYSIS_SR
    positions = sorted(w.position_s for w in windows if w.usable)
    middle_ms = (step.before_ms + step.after_ms) / 2.0
    # One narrowed search takes in both levels, with room to land elsewhere.
    narrow = max_offset_ms > NARROW_SEARCH_MS and abs(step.magnitude_ms) < NARROW_SEARCH_MS
    for side, held in (
        ("before", [p for p in positions if p < step.split_position_s][::-1]),
        ("after", [p for p in positions if p >= step.split_position_s]),
    ):
        seen = 0
        for first, second in list(zip(held[:-1], held[1:]))[:STEP_CORROBORATIONS]:
            if token:
                token.raise_if_cancelled()
            # Clear of both windows where there is room to be: on a short file
            # a full window between them would be mostly the same audio again.
            first, second = min(first, second), max(first, second)
            room = second - first - window_s
            length = min(window_s, room) if room >= MIN_CORROBORATION_S else window_s
            spot = (first + second + window_s - length) / 2.0
            estimate = _measure_window(
                primary_path, secondary_path, spot, length, max_offset_ms, token,
                primary_track, secondary_track, secondary_rate,
                expect_ms=_compensated_ms(spot, middle_ms, speed) if narrow else None,
            )
            read = _side_of_cut(estimate, step)
            if read is not None and read != side:
                return False
            seen += read == side
        if not seen:
            return False
    return True


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
    keep = _near(offsets, model)
    kept_windows = usable
    # As they were, for asking how easily a line runs through them.
    every_position, every_offset = positions, offsets
    if keep.sum() >= 2 and not keep.all():
        kept_windows = [w for w, kept in zip(usable, keep) if kept]
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
    penalty = 1.0
    if len(offsets) >= 2:
        disagreement = float(np.std(offsets - model))
        if disagreement > 500.0:
            penalty = 0.5
        elif disagreement > 100.0:
            penalty = 0.8

    # Each window's own peak says how sharply it matched; windows landing on
    # the same offset say far more, and the mean above never counted it. Six
    # windows within a few milliseconds of each other, out of a search range
    # of seconds, are not a coincidence however soft each peak was -- that
    # is what a dub in a different language looks like, and it was reported
    # at 63%.
    agreeing = np.abs(offsets - model) <= AGREEMENT_MS
    for window in result.windows:
        window.agrees = None
    for window in usable:
        window.agrees = False
    for window, agrees in zip(kept_windows, agreeing):
        window.agrees = bool(agrees)
    result.agreeing_windows = int(agreeing.sum())
    # Only windows that heard different audio are independent witnesses: on
    # a short file the windows overlap, and one lock read again by the next
    # window over is the same evidence twice.
    span = result.window_s or 0.0
    independent = _independent(
        [w.position_s for w, agrees in zip(kept_windows, agreeing) if agrees], span
    )
    result.confidence = penalty * max(
        result.confidence,
        _agreement_confidence(independent, _independent([w.position_s for w in result.windows], span), result.search_ms),
    )

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
        before = every_position < step.split_position_s if step is not None else np.ones(len(every_position), dtype=bool)
        if abs(slope) > DRIFT_SIGNIFICANT_MS_PER_S and not _slope_measured(
            every_position[before], every_offset[before], fit_positions, fit_offsets
        ):
            # Steep, but no steeper than a line through this scatter comes out
            # by itself: six weak windows spread over 250 ms fitted 0.15 ms/s,
            # and its t=0 was 364 ms from a delay their middle had right to 3.
            # A conversion moves every window by hundreds of times what they
            # scatter by, so it never comes near this.
            result.drift_ms_per_s = None
            result.delay_at_start_ms = result.delay_ms
        elif abs(slope) <= DRIFT_SIGNIFICANT_MS_PER_S:
            # Nothing drifts, so there is one delay, and every number that
            # names it is the same one: an app showing delay_ms beside another
            # applying delay_at_start_ms showed two delays for one pair, 1.6
            # ms apart where the windows scattered by tens. How far it leans
            # on the slope is how well the windows measured one: all the way
            # when they agree to a fraction of a millisecond (the few
            # milliseconds a drift too small to report adds up to are kept),
            # not at all when they scatter, where carrying a line back to t=0
            # only carries the scatter there.
            trust = _slope_trust(fit_positions, fit_offsets, float(slope), float(intercept))
            result.delay_at_start_ms = trust * float(intercept) + (1.0 - trust) * result.delay_ms
            result.delay_ms = result.delay_at_start_ms
        else:
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
        if abs(slope) <= DRIFT_SIGNIFICANT_MS_PER_S:
            result.delay_ms = result.delay_at_start_ms

    if result.delay_at_start_ms is None:
        result.delay_at_start_ms = result.delay_ms

    return step


def _slope_evidence(positions: np.ndarray, offsets: np.ndarray) -> float:
    """How many of its own standard errors the slope through these windows
    stands from none, squared."""
    freedom = len(offsets) - 2
    spread = float(np.sum((positions - positions.mean()) ** 2))
    if freedom < 1 or spread <= 0.0:
        return 0.0
    slope, intercept = np.polyfit(positions, offsets, 1)
    variance = float(np.sum((offsets - (slope * positions + intercept)) ** 2)) / freedom
    return float(slope) ** 2 * spread / max(variance, 1e-12)


def _student_t(freedom: int) -> float:
    """What a slope that is not there reaches one time in fifty, in its own
    standard errors (Student's t, two-sided)."""
    if freedom < 1:
        return math.inf
    return STUDENT_T_98[min(freedom, len(STUDENT_T_98)) - 1]


def _slope_measured(
    positions: np.ndarray, offsets: np.ndarray, kept_positions: np.ndarray, kept_offsets: np.ndarray
) -> bool:
    """Whether the windows measured the drift fitted through the ones kept.

    Every window first, the trimmed ones included: those left are the ones a
    line was found to run through, and say nothing about how easily one was.
    One window that found something else entirely would hide a real drift from
    that, so the ones kept may speak for themselves when they are nearly all
    of them and far surer than that takes. Over simulated surveys scatter
    passes one time in fifty, where it was taken for a drift every time, and a
    real conversion with a stray window among six or more never fails.
    """
    if _slope_evidence(positions, offsets) >= _student_t(len(offsets) - 2) ** 2:
        return True
    return (
        3 * len(kept_offsets) > 2 * len(offsets)
        and _slope_evidence(kept_positions, kept_offsets) >= (3.0 * _student_t(len(kept_offsets) - 2)) ** 2
    )


def _slope_trust(positions: np.ndarray, offsets: np.ndarray, slope: float, intercept: float) -> float:
    """How far a slope too small to be a drift can be believed, 0 to 1.

    Windows that do not follow the line (half of them further than
    AGREEMENT_MS from it) measured no drift this small, whatever it happens to
    come to: a dub that wanders about the video by tens of milliseconds fits
    one as steep as it is meaningless, and the middle of what they read is the
    delay that suits the whole film. Windows that do are believed by how many
    of the slope's own standard errors it stands from none, as one minus the
    share of it their scatter would produce by itself (the positive-part
    shrinkage estimator), which over simulated pairs is never materially worse
    at t=0 than the line and up to a third better where there is no drift.
    """
    if float(np.median(np.abs(offsets - (slope * positions + intercept)))) > AGREEMENT_MS:
        return 0.0
    squared_t = _slope_evidence(positions, offsets)
    return max(0.0, 1.0 - 1.0 / squared_t) if squared_t > 0.0 else 0.0


def _independent(positions: List[float], window_s: float) -> int:
    """How many of these windows can be counted without any two sharing
    audio: the most that fit one after another without overlapping."""
    count, free_from = 0, -math.inf
    for position in sorted(positions):
        if position >= free_from:
            count += 1
            free_from = position + window_s
    return count


def _agreement_confidence(agreeing: int, total: int, search_ms: Optional[float]) -> float:
    """How sure ``agreeing`` of ``total`` windows make the offset.

    One minus the chance of it happening by accident: windows that each
    landed anywhere in their search range falling within AGREEMENT_MS of a
    line two of them define. Two are spent on the line, since the file may
    drift, and any set of that many windows could have been the one.
    """
    if agreeing < MIN_USABLE_WINDOWS or not search_ms or search_ms <= 0:
        return 0.0
    landing = min(1.0, AGREEMENT_MS / search_ms)
    chance = math.comb(total, agreeing) * landing ** (agreeing - 2)
    return float(min(AGREEMENT_CONFIDENCE_CAP, max(0.0, 1.0 - chance)))


def _split_mask(positions: np.ndarray, step: Optional[Step]) -> np.ndarray:
    """Which windows fall on the far side of the cut."""
    if step is None:
        return np.zeros(len(positions), dtype=bool)
    return positions >= step.split_position_s


def _near(offsets: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Which windows are near enough to what the file is doing to be part of it."""
    residual = np.abs(offsets - model)
    return residual <= max(50.0, 4.0 * float(np.median(residual)))


def _model_values(
    positions: np.ndarray, offsets: np.ndarray, after_cut: np.ndarray
) -> np.ndarray:
    """What the file says each window should have measured.

    Two levels when a cut splits them, a line when the offsets measurably
    slide, one level otherwise -- and every version of it is fitted robustly,
    so the reference an outlier is judged against is never one that outlier
    helped set.

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

    level = np.full(len(offsets), float(np.median(offsets)))
    if len(offsets) >= 3 and float(np.ptp(positions)) > 1.0:
        slope, intercept = _robust_line(positions, offsets)
        line = slope * positions + intercept
        # Only a slope the windows measured. Through scatter there is always
        # one that runs close by a few of them, and they were then counted as
        # agreeing: three of six spread over 250 ms, which made 91%.
        near = _near(offsets, line)
        if _slope_measured(positions, offsets, positions[near], offsets[near]):
            return line
    return level


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
