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
from fractions import Fraction
from typing import List, Optional

# Frame rates that real releases actually use, as the exact rationals they
# actually are.
#
# "23.976" is shorthand. The rate is 24000/1001, and the difference is not
# cosmetic here: a ratio built from the decimal makes 24 -> 23.976 come out
# 1000/999 where the true conversion is 1001/1000. That is 1e-6, which sounds
# like nothing until it is the correction factor applied to every timestamp in
# a 45-minute episode -- about 2.7ms of drift left behind by the fix that was
# supposed to remove it.
#
# The same set, expressed the same way, lives in MkvBatchMux as `EXACT_RATES`
# in `delayConversion.ts` and `COMMON_RATES` in `audioFps.ts`; keep the three
# in step.
COMMON_RATES: List[Fraction] = [
    Fraction(24000, 1001),  # 23.976
    Fraction(24),
    Fraction(25),
    Fraction(30000, 1001),  # 29.97
    Fraction(30),
    Fraction(50),
    Fraction(60000, 1001),  # 59.94
    Fraction(60),
]

# How far a frame rate read from a file may sit from a standard rate and still
# be treated as that rate. 23.976 and 24 are 0.024 apart and take different
# ratios, so this stays well under half that gap; matches FPS_TOLERANCE in
# MkvBatchMux's delayConversion.ts.
FPS_TOLERANCE = 0.01

# Beyond this much speed difference a window cannot be correlated at all. The
# offset slides by ratio x window_s *inside* the window, so a 45-second window
# on a PAL-sped dub smears the alignment across 1.9 seconds and the correlation
# peak flattens into the noise. Measured on real decodes, peak prominence at
# 4.27% falls to 6-12 against a match threshold of 12: the pair reports
# "tracks appear unrelated" at every useful window length.
MAX_UNCOMPENSATED_RATIO = 0.005

# How far a ratio of two durations may sit from a standard conversion and still
# be taken as one. Far looser than RATIO_TOLERANCE because durations differ for
# reasons that have nothing to do with frame rate -- a trimmed logo, trailing
# silence, encoder padding -- and it only has to be tight enough that an
# ordinary length difference is not mistaken for a conversion.
DURATION_TOLERANCE = 0.005

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

    drift_ms_per_s: Optional[float]
    """None when a cut left too few windows on one side to fit a line through."""
    speed_ratio: float
    """How much faster the secondary runs than the primary. 1.0 means neither."""

    source_fps: Optional[float] = None
    target_fps: Optional[float] = None
    """The rate pair that explains the drift, when one fits."""

    is_rate_mismatch: bool = False
    is_likely_cut: bool = False
    cut_position_s: Optional[float] = None
    cut_magnitude_ms: Optional[float] = None
    """Where the offset jumps and by how much, when a splice was located."""
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
            "cutPositionS": self.cut_position_s,
            "cutMagnitudeMs": self.cut_magnitude_ms,
            "explanation": self.explanation,
            "correctionRatio": self.correction_ratio,
        }


def plan_speed_compensation(
    primary_duration: Optional[float],
    secondary_duration: Optional[float],
) -> Optional[Fraction]:
    """The playback-speed difference to undo before correlating, if any.

    A PAL-sped dub runs 4.27% short, and that is far too much to correlate
    through: the alignment moves further inside one window than the window can
    resolve. The speed has to come off before the measurement, not after it.

    The estimate comes from the two durations rather than from a measurement,
    because a measurement is exactly what is not available yet. That is a crude
    signal -- but crude in precisely the right place. It cannot separate 24 from
    23.976 (0.1% apart, inside the noise of a trimmed logo), and it does not
    need to: those correlate perfectly well uncompensated. What it separates
    easily is 25 from 23.976, 4.27% apart, which is the case that fails.

    Snapping to a standard conversion rather than using the raw ratio is the
    safeguard. Two files whose lengths differ because one has extra credits
    would otherwise be "compensated" for a speed change that never happened;
    a length difference has no reason to land on 25/23.976.

    Returns:
        The exact ratio to undo, or None to measure the files as they are.
    """
    if not primary_duration or not secondary_duration:
        return None
    if primary_duration <= 0 or secondary_duration <= 0:
        return None

    # How much faster the secondary runs than the primary. Content of N frames
    # occupies N/fps seconds, so this ratio is the audio's rate over the
    # video's -- the same quantity a measured drift resolves to.
    ratio = primary_duration / secondary_duration
    if abs(ratio - 1.0) <= MAX_UNCOMPENSATED_RATIO:
        return None

    best: Optional[tuple] = None
    for audio_rate in COMMON_RATES:
        for video_rate in COMMON_RATES:
            if audio_rate == video_rate:
                continue
            candidate = audio_rate / video_rate
            error = abs(float(candidate) - ratio) / ratio
            if error <= DURATION_TOLERANCE and (best is None or error < best[0]):
                best = (error, candidate)

    return best[1] if best is not None else None


def speed_candidates(
    primary_duration: Optional[float],
    secondary_duration: Optional[float],
) -> List[Fraction]:
    """Every conversion worth trying to correlate through, likeliest first.

    `plan_speed_compensation` believes the durations. This does not: it only
    uses them to decide what to try first. Two files differ in length for
    reasons that have nothing to do with speed -- a dub that starts later, an
    episode with a different set of credits -- and half a percent of that is
    enough to stop the ratio landing on a conversion, which leaves a PAL pair
    unmeasurable for a reason unrelated to why it is hard to measure.

    Ordering by the durations keeps the search short in the cases where they
    were roughly right, without depending on them being right.
    """
    ratios = {Fraction(1)}
    for audio_rate in COMMON_RATES:
        for video_rate in COMMON_RATES:
            if audio_rate != video_rate:
                ratios.add(audio_rate / video_rate)

    # Only conversions big enough to break a correlation are worth trying: the
    # small ones measure perfectly well as they are, and trying them would just
    # be a slower way of measuring the pair unchanged.
    worth_trying = [
        ratio for ratio in ratios
        if abs(float(ratio) - 1.0) > MAX_UNCOMPENSATED_RATIO
    ]

    observed = None
    if primary_duration and secondary_duration and secondary_duration > 0:
        observed = primary_duration / secondary_duration

    if observed is None:
        worth_trying.sort(key=lambda ratio: abs(float(ratio) - 1.0))
    else:
        worth_trying.sort(key=lambda ratio: abs(float(ratio) - observed))
    return worth_trying


def speed_ratio_for(drift_ms_per_s: float) -> float:
    """How much faster the secondary runs, from how fast the offset moves.

    A window at primary time t finds its content at secondary time t/ratio, so
    the offset it measures is t*(1/ratio - 1) and the slope of that is the
    drift. Inverting it is therefore 1/(1 + drift/1000), not 1 - drift/1000.

    The two agree to a millionth at the drift a 24 -> 23.976 conversion makes,
    which is why the approximation survived. They do not agree at 25 -> 23.976:
    there it is wrong by 0.17%, four times the tolerance a conversion has to be
    named within, so the most common conversion in dubbed material could not be
    identified even from a perfect measurement.
    """
    factor = 1.0 + (drift_ms_per_s / 1000.0)
    if factor <= 1e-9:
        # The secondary would have to run infinitely fast. Not a real file.
        return 0.0
    return 1.0 / factor


def _format_fps(value: float) -> str:
    return f"{value:g}" if value == int(value) else f"{value:.3f}".rstrip("0")


def _format_jump(magnitude_ms: float) -> str:
    """The size of a cut, in whichever unit reads as a quantity rather than a
    number: 80ms is eighty milliseconds, 2000ms is two seconds."""
    if abs(magnitude_ms) < 1000.0:
        return f"{magnitude_ms:+.0f}ms"
    return f"{magnitude_ms / 1000.0:+.2f}s"


def _format_clock(seconds: float) -> str:
    """A position a user can type into a player."""
    minutes, secs = divmod(int(round(max(0.0, seconds))), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _exact_rate(fps: float) -> Optional[Fraction]:
    """The standard rate a measured frame rate is, or None if it is not one.

    A codec's own frame rate lands here too -- 31.25 for AC-3, 46.875 for
    AAC-LC -- and must not be mistaken for a timing rate.
    """
    best, best_error = None, FPS_TOLERANCE
    for rate in COMMON_RATES:
        error = abs(float(rate) - fps)
        if error < best_error:
            best, best_error = rate, error
    return best


def diagnose(
    drift_ms_per_s: Optional[float],
    primary_fps: Optional[float] = None,
    secondary_fps: Optional[float] = None,
    cut_position_s: Optional[float] = None,
    cut_magnitude_ms: Optional[float] = None,
    cut_uncertainty_s: Optional[float] = None,
) -> Optional[RateDiagnosis]:
    """Explain a measured drift rate, or a cut found in place of one.

    Args:
        drift_ms_per_s: how fast the required offset changes. Positive means the
            secondary falls further behind as the file plays.
        primary_fps: frame rate of the reference, when known.
        secondary_fps: frame rate of the secondary's source, when known.
        cut_magnitude_ms: how far the offset jumps at a splice, when the
            analysis found one. Its presence settles the diagnosis outright.
        cut_position_s: where that jump is, in the reference's timeline.
        cut_uncertainty_s: how tightly that position was pinned down.

    Returns:
        A diagnosis, or None when there is neither drift nor a cut to explain.
    """
    if drift_ms_per_s is None and cut_magnitude_ms is None:
        return None

    speed_ratio = speed_ratio_for(drift_ms_per_s or 0.0)

    # A located cut outranks every rate story below, and has to: a 500ms splice
    # fitted with one line becomes a 1.18 ms/s slope, which sits inside the
    # tolerance of 23.976 -> 24 and was reported as a rate mismatch correctable
    # "exactly" by resampling. Nothing about the file is a speed problem, and
    # the offered fix stretches the whole episode to paper over one edit.
    if cut_magnitude_ms is not None:
        where = ""
        if cut_position_s is not None:
            where = f" about {_format_clock(cut_position_s)} in"
            if cut_uncertainty_s and cut_uncertainty_s >= 1.0:
                where += f" (give or take {cut_uncertainty_s:.0f}s)"
        return RateDiagnosis(
            drift_ms_per_s=drift_ms_per_s,
            speed_ratio=speed_ratio,
            is_likely_cut=True,
            cut_position_s=cut_position_s,
            cut_magnitude_ms=cut_magnitude_ms,
            explanation=(
                f"The offset jumps by {_format_jump(cut_magnitude_ms)}{where} "
                "and stays there. These are different cuts of the same title: "
                "the delay below aligns everything before the jump, and nothing "
                "after it."
            ),
        )

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
        # Snapped to exact rationals first, so the ratio between two standard
        # rates is the conversion itself rather than a rounding of it.
        exact_primary = _exact_rate(primary_fps)
        exact_secondary = _exact_rate(secondary_fps)
        if exact_primary is not None and exact_secondary is not None:
            implied = float(exact_primary / exact_secondary)
            primary_fps, secondary_fps = float(exact_primary), float(exact_secondary)
        else:
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
    #
    # speed_ratio is how much faster the secondary runs than the primary, so a
    # candidate pair fits when audio_fps / video_fps equals it. Naming the
    # variables that way round matters: it is the audio that gets resampled, so
    # source_fps must be the rate the audio was timed at and target_fps the
    # video's rate it has to become. Deriving them the other way round reports a
    # correction that reads as though the video were being changed.
    # When the video's own rate is known, only conversions *onto* that rate can
    # be true, and the search has to be held to it.
    #
    # Three conversions share the 1001/1000 factor -- 24 -> 23.976, 30 -> 29.97
    # and 60 -> 59.94 -- so a drift of about a millisecond per second fits all
    # three equally and the search picked whichever scored marginally better.
    # On a 23.976fps BluRay that came back as "the audio was timed against a
    # 23.976fps source, but this video is 24fps": a rate the engine had just
    # been told the video does not have, attached to a resample offered as
    # exact. Refusing to name one is the right answer there -- a drift that no
    # conversion onto *this* video explains is a fact worth reporting, not a
    # gap to fill with the nearest rate from another video.
    required_video_rate = _exact_rate(primary_fps) if primary_fps else None

    best: Optional[tuple] = None
    for audio_rate in COMMON_RATES:
        for video_rate in COMMON_RATES:
            if video_rate == audio_rate:
                continue
            if required_video_rate is not None and video_rate != required_video_rate:
                continue
            # Divided as rationals and converted once, so the ratio carries no
            # rounding of its own into the correction factor built from it.
            ratio = float(audio_rate / video_rate)
            error = abs(ratio - speed_ratio) / max(ratio, 1e-9)
            if error <= RATIO_TOLERANCE and (best is None or error < best[0]):
                best = (error, float(audio_rate), float(video_rate), ratio)

    if best is not None:
        _, audio_fps, video_fps, ratio = best
        return RateDiagnosis(
            drift_ms_per_s=drift_ms_per_s,
            speed_ratio=ratio,
            source_fps=audio_fps,
            target_fps=video_fps,
            is_rate_mismatch=True,
            explanation=(
                f"The audio was timed against a {_format_fps(audio_fps)}fps "
                f"source, but this video is {_format_fps(video_fps)}fps. "
                "Resampling the audio corrects it exactly."
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
