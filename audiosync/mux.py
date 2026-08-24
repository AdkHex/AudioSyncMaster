"""Apply a measured offset by writing a corrected output file.

The original tool only ever *reported* offsets, leaving the user to translate a
number into a working ffmpeg invocation by hand. This module closes that loop.

Two correction shapes are supported:

*   **constant delay** -- the audio needs shifting by a fixed amount.
*   **drift** -- the audio also runs at a slightly different rate (a 25fps vs
    23.976fps telecine difference, typically), so it is resampled as well as
    shifted.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import List, Optional

from .media import CancellationToken, MediaError, ffmpeg_path, probe, _run

# Below this the correction is inaudible; muxing would waste time and disk.
NEGLIGIBLE_DELAY_MS = 5.0

MUX_TIMEOUT_S = 3600


@dataclass
class MuxPlan:
    """A described, inspectable correction, produced before anything is written."""

    video_path: str
    audio_path: str
    output_path: str
    delay_ms: float
    speed_ratio: Optional[float] = None
    copy_video: bool = True

    @property
    def needs_resample(self) -> bool:
        return self.speed_ratio is not None and abs(self.speed_ratio - 1.0) > 1e-9

    def describe(self) -> str:
        parts = []
        if abs(self.delay_ms) >= NEGLIGIBLE_DELAY_MS:
            direction = "later" if self.delay_ms > 0 else "earlier"
            parts.append(f"shift audio {abs(self.delay_ms):.1f}ms {direction}")
        else:
            parts.append("no significant shift")
        if self.needs_resample:
            parts.append(f"resample audio by {self.speed_ratio:.6f}x to correct drift")
        return "; ".join(parts)


def build_command(plan: MuxPlan) -> List[str]:
    """Construct the ffmpeg command for a plan.

    A positive delay means the secondary audio starts LATER than the video, so
    aligning it means pulling the audio EARLIER.

    The two directions need different mechanisms:

    *   Pulling audio earlier means *discarding* its first N milliseconds, done
        with an input ``-ss`` on the audio. Expressing this as a negative
        ``-itsoffset`` does not work: every container normalisation option
        (notably ``-avoid_negative_ts``) shifts the negative timestamps back to
        zero, silently cancelling the correction and writing an unchanged file.
    *   Pushing audio later means *inserting* silence, which ``adelay`` does
        explicitly. Using ``-itsoffset`` here would rely on the player honouring
        a positive start offset, which not all of them do.

    Both are applied to the stream itself, so the result is correct regardless
    of how the container or the player treats timestamps.
    """
    command = [ffmpeg_path(), "-nostdin", "-y", "-i", plan.video_path]

    # Without a rate change, a positive delay is just a decode start point, so
    # an input -ss keeps the audio stream copyable. With a rate change the shift
    # has to happen in the filter chain instead (see below), which necessarily
    # re-encodes.
    trim_at_input = plan.delay_ms > 1e-6 and not plan.needs_resample
    if trim_at_input:
        command.extend(["-ss", f"{plan.delay_ms / 1000.0:.6f}"])
    command.extend(["-i", plan.audio_path])

    command.extend(["-map", "0:v:0", "-map", "1:a:0"])
    command.extend(["-c:v", "copy" if plan.copy_video else "libx264"])

    filters: List[str] = []

    # Rate correction comes FIRST. atempo rescales every timestamp after it, so
    # a shift applied beforehand is itself scaled, leaving a residual offset
    # proportional to the shift even though the drift is gone.
    if plan.needs_resample:
        # atempo is limited to 0.5-2.0 per instance; drift corrections are
        # always tiny, so a single instance always suffices here.
        filters.append(f"atempo={plan.speed_ratio:.9f}")

        # Shift on the rate-corrected timeline.
        if plan.delay_ms > 1e-6:
            filters.append(f"atrim=start={plan.delay_ms / 1000.0:.6f}")
            filters.append("asetpts=PTS-STARTPTS")

    if plan.delay_ms < -1e-6:
        filters.append(f"adelay={-plan.delay_ms:.3f}:all=1")

    if filters:
        command.extend(["-filter:a", ",".join(filters), "-c:a", "aac", "-b:a", "320k"])
    else:
        command.extend(["-c:a", "copy"])

    command.append(plan.output_path)
    return command


def command_string(plan: MuxPlan) -> str:
    """Shell-quoted command, for showing the user what will run."""
    return " ".join(shlex.quote(part) for part in build_command(plan))


def plan_correction(
    video_path: str,
    audio_path: str,
    delay_ms: float,
    output_path: Optional[str] = None,
    drift_ms_per_s: Optional[float] = None,
    output_dir: Optional[str] = None,
    suffix: str = ".synced",
) -> MuxPlan:
    """Describe the correction for a measured pair without performing it."""
    if output_path is None:
        stem, ext = os.path.splitext(os.path.basename(video_path))
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{stem}{suffix}{ext or '.mkv'}")

    speed_ratio = None
    if drift_ms_per_s is not None and abs(drift_ms_per_s) > 1e-6:
        # Offset grows by drift_ms_per_s each second, so the audio is running
        # slow by that fraction; speeding it up by the reciprocal cancels it.
        speed_ratio = 1.0 + (drift_ms_per_s / 1000.0)
        if not 0.5 <= speed_ratio <= 2.0:
            speed_ratio = None

    return MuxPlan(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        delay_ms=delay_ms,
        speed_ratio=speed_ratio,
    )


def apply_correction(
    plan: MuxPlan,
    token: Optional[CancellationToken] = None,
    overwrite: bool = False,
) -> str:
    """Execute a plan, writing the corrected file. Returns the output path."""
    if not os.path.isfile(plan.video_path):
        raise MediaError(f"Video not found: {plan.video_path}")
    if not os.path.isfile(plan.audio_path):
        raise MediaError(f"Audio not found: {plan.audio_path}")

    if os.path.exists(plan.output_path) and not overwrite:
        raise MediaError(
            f"Output already exists: {os.path.basename(plan.output_path)}. "
            "Enable overwrite to replace it."
        )
    if os.path.abspath(plan.output_path) == os.path.abspath(plan.video_path):
        raise MediaError("Refusing to overwrite the source video in place.")

    parent = os.path.dirname(os.path.abspath(plan.output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    source = probe(plan.video_path, token)
    if not source.has_video:
        raise MediaError(f"{os.path.basename(plan.video_path)} has no video stream to mux into")

    _run(
        build_command(plan),
        MUX_TIMEOUT_S,
        token,
        what=f"write {os.path.basename(plan.output_path)}",
    )

    if not os.path.isfile(plan.output_path) or os.path.getsize(plan.output_path) == 0:
        raise MediaError(f"ffmpeg reported success but produced no output for {plan.output_path}")
    return plan.output_path


PREVIEW_TIMEOUT_S = 180


def build_preview_command(
    video_path: str,
    audio_path: str,
    delay_ms: float,
    position_s: float,
    duration_s: float,
    output_path: str,
    video_track: int = 0,
    audio_track: int = 0,
    drift_ms_per_s: Optional[float] = None,
) -> List[str]:
    """ffmpeg command for a short aligned excerpt.

    The audio is taken from wherever the measured delay says it should be, so
    playing the result is a direct test of the measurement: if the number is
    right, the excerpt is in sync.
    """
    audio_start = max(0.0, position_s + (delay_ms / 1000.0))

    command = [ffmpeg_path(), "-nostdin", "-y"]

    # Input seeks, so the cost does not scale with how far into the file the
    # excerpt sits. A preview is a quick check; decoding two hours to reach it
    # would defeat the point. Exactness is not needed here -- the ear cannot
    # resolve a keyframe's worth of start position, and both inputs shift
    # together anyway.
    command.extend(["-ss", f"{position_s:.6f}", "-i", video_path])
    command.extend(["-ss", f"{audio_start:.6f}", "-i", audio_path])

    command.extend([
        "-t", f"{duration_s:.6f}",
        # The video's own audio is irrelevant here: the point is to hear the
        # dub against the picture, so only the video stream is taken from it.
        "-map", "0:v:0",
        "-map", f"1:a:{max(0, audio_track)}",
    ])

    # With drift, the excerpt only stays aligned if the rate is corrected too.
    if drift_ms_per_s is not None and abs(drift_ms_per_s) > 1e-6:
        ratio = 1.0 + (drift_ms_per_s / 1000.0)
        if 0.5 <= ratio <= 2.0:
            command.extend(["-filter:a", f"atempo={ratio:.9f}"])

    command.extend([
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest", output_path,
    ])
    return command


def extract_preview(
    video_path: str,
    audio_path: str,
    delay_ms: float,
    position_s: float,
    duration_s: float,
    output_path: str,
    token: Optional[CancellationToken] = None,
    video_track: int = 0,
    audio_track: int = 0,
    drift_ms_per_s: Optional[float] = None,
) -> str:
    """Render a short aligned excerpt so a result can be checked by ear.

    Hearing a few seconds line up is far more convincing than a confidence
    score, and it is the only way to settle the borderline cases.
    """
    if not os.path.isfile(video_path):
        raise MediaError(f"Video not found: {video_path}")
    if not os.path.isfile(audio_path):
        raise MediaError(f"Audio not found: {audio_path}")

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    _run(
        build_preview_command(
            video_path, audio_path, delay_ms, position_s, duration_s, output_path,
            video_track, audio_track, drift_ms_per_s,
        ),
        PREVIEW_TIMEOUT_S,
        token,
        what="render preview",
    )

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise MediaError("ffmpeg produced no preview output")
    return output_path


def choose_preview_position(duration_s: Optional[float], window_s: float = 12.0) -> float:
    """Pick a point worth listening to.

    A third of the way in avoids both the opening logo and the credits, which
    are the two places most likely to be silent or music-only.
    """
    if not duration_s or duration_s <= window_s:
        return 0.0
    return max(0.0, min(duration_s / 3.0, duration_s - window_s))
