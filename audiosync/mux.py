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

import json
import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .framerate import exact_speed_filters
from .media import PROBE_TIMEOUT_S, CancellationToken, MediaError, ffmpeg_path, ffprobe_path, probe, _run

_log = logging.getLogger(__name__)

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
    #: The video's subtitle streams to carry over, as ``(N of 0:s:N, codec
    #: to write it with)``. None = not probed yet: every subtitle stream and
    #: attachment is copied when the output is Matroska, none otherwise.
    subtitles: Optional[List[Tuple[int, str]]] = None
    #: Carry font attachments (Matroska only; ASS subtitles need them).
    attachments: bool = True
    #: What had to be left out, for the user.
    warnings: List[str] = field(default_factory=list)
    #: The audio's own sample rate, which a sample-exact rate change needs;
    #: None falls back to atempo.
    audio_sample_rate: Optional[int] = None

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
            how = (
                "pitch follows the speed, as a frame-rate conversion's does" if self.audio_sample_rate
                else "pitch kept"
            )
            parts.append(f"resample audio by {self.speed_ratio:.6f}x to correct drift ({how})")
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
    # The video's subtitles and fonts ride along unchanged: only the audio is
    # retimed, the video timeline is not, so they are still in sync. Without
    # this every synced file silently lost its subtitle tracks.
    matroska = _container_kind(plan.output_path) == "matroska"
    if plan.subtitles is None:
        if matroska:
            command.extend(["-map", "0:s?", "-map", "0:t?", "-c:s", "copy"])
    else:
        for index, _codec in plan.subtitles:
            command.extend(["-map", f"0:s:{index}"])
        if plan.attachments and matroska:
            command.extend(["-map", "0:t?"])
        for position, (_index, codec) in enumerate(plan.subtitles):
            command.extend([f"-c:s:{position}", codec])
    command.extend(["-c:v", "copy" if plan.copy_video else "libx264"])

    filters: List[str] = []

    # The delay is where the video's first moment sits on the dub's own clock.
    # Once atempo has put the dub on the video's clock, that same moment sits
    # delay / atempo from the start, and that is the shift to make there:
    # trimming the unconverted figure leaves the dub out by delay * (1 - atempo)
    # -- 107ms on a 2.6s PAL delay, 3.8s on a 92s one.
    shift_ms = plan.delay_ms / plan.speed_ratio if plan.needs_resample else plan.delay_ms

    # Rate correction comes FIRST. atempo rescales every timestamp after it, so
    # a shift applied beforehand is itself scaled, leaving a residual offset
    # proportional to the shift even though the drift is gone.
    if plan.needs_resample:
        if plan.audio_sample_rate:
            filters.extend(exact_speed_filters(plan.speed_ratio, plan.audio_sample_rate))
        else:
            # atempo is limited to 0.5-2.0 per instance; drift corrections are
            # always tiny, so a single instance always suffices here.
            filters.append(f"atempo={plan.speed_ratio:.9f}")

        # Shift on the rate-corrected timeline.
        if shift_ms > 1e-6:
            filters.append(f"atrim=start={shift_ms / 1000.0:.6f}")
            filters.append("asetpts=PTS-STARTPTS")

    if shift_ms < -1e-6:
        filters.append(f"adelay={-shift_ms:.3f}:all=1")

    if filters:
        command.extend(["-filter:a", ",".join(filters), "-c:a", "aac", "-b:a", "320k"])
    else:
        command.extend(["-c:a", "copy"])

    command.append(plan.output_path)
    return command


#: Subtitle codecs each output container takes as they are, and the text
#: codec other text subtitles are converted to (None: text is left out too).
_SUBTITLE_SUPPORT = {
    "matroska": ({"subrip", "ass", "ssa", "webvtt", "hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"}, "srt"),
    "webm": ({"webvtt"}, "webvtt"),
    "mp4": ({"mov_text"}, "mov_text"),
    "mpegts": ({"dvb_subtitle", "dvb_teletext"}, None),
}
_TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}


def _container_kind(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mkv", ".mka", ".mks", ".mk3d"):
        return "matroska"
    if ext == ".webm":
        return "webm"
    if ext in (".mp4", ".m4v", ".mov", ".m4a"):
        return "mp4"
    if ext in (".ts", ".m2ts", ".mts"):
        return "mpegts"
    return None


def _probe_subtitles(path: str, token: Optional[CancellationToken]) -> Tuple[List[str], int]:
    """Codec of every subtitle stream (0:s:N order) and the attachment count."""
    stdout = _run(
        [ffprobe_path(), "-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "json", path],
        PROBE_TIMEOUT_S, token, what=f"probe {os.path.basename(path)}",
    )
    streams = json.loads(stdout.decode("utf-8", errors="replace") or "{}").get("streams") or []
    codecs = [s.get("codec_name") or "unknown" for s in streams if s.get("codec_type") == "subtitle"]
    attachments = sum(1 for s in streams if s.get("codec_type") == "attachment")
    return codecs, attachments


def plan_subtitles(plan: MuxPlan, token: Optional[CancellationToken] = None) -> MuxPlan:
    """Decide which of the video's subtitle streams the output can hold.

    A stream the output container cannot take (PGS into MP4, say) is left
    out with a warning instead of failing the whole correction: the audio
    fix is what was asked for.
    """
    try:
        codecs, attachments = _probe_subtitles(plan.video_path, token)
    except (MediaError, ValueError):
        return plan
    kind = _container_kind(plan.output_path)
    copyable, text_codec = _SUBTITLE_SUPPORT.get(kind, (set(), None))
    kept: List[Tuple[int, str]] = []
    dropped: List[str] = []
    for index, codec in enumerate(codecs):
        if codec in copyable:
            kept.append((index, "copy"))
        elif text_codec and codec in _TEXT_SUBTITLE_CODECS:
            kept.append((index, text_codec))
        else:
            dropped.append(f"track {index + 1} ({codec})")
    plan.subtitles = kept
    plan.attachments = kind == "matroska"
    plan.warnings = []
    if dropped:
        plan.warnings.append(
            f"Left out subtitle {', '.join(dropped)}: {os.path.splitext(plan.output_path)[1] or 'this'} "
            "files cannot hold that format. Write an .mkv to keep every subtitle."
        )
    if attachments and kind != "matroska":
        plan.warnings.append(f"Left out {attachments} attachment(s) (fonts): only .mkv output can hold them.")
    return plan


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

    plan = MuxPlan(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        delay_ms=delay_ms,
        speed_ratio=speed_ratio,
    )
    # Probed now so the command shown before running is the one that runs.
    if os.path.isfile(video_path):
        plan_subtitles(plan)
    if plan.needs_resample and os.path.isfile(audio_path):
        try:
            info = probe(audio_path)
            rate = (info.audio_tracks[0].sample_rate if info.audio_tracks else None) or info.sample_rate
            plan.audio_sample_rate = int(rate) if rate else None
        except MediaError:
            plan.audio_sample_rate = None  # atempo still corrects it, less exactly
    return plan


def apply_correction(
    plan: MuxPlan,
    token: Optional[CancellationToken] = None,
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Execute a plan, writing the corrected file. Returns the output path.

    ``log`` receives a warning for every subtitle stream or attachment the
    output container could not hold (also kept in ``plan.warnings``).
    """
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
    if plan.subtitles is None:
        plan_subtitles(plan, token)
    for warning in plan.warnings:
        (log or _log.warning)(warning)

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
