"""Media decoding and probing.

Everything that shells out to ffmpeg/ffprobe lives here, so binary discovery,
timeouts and process cleanup have exactly one implementation.

Two behaviours differ deliberately from the original code:

*   ``-ss`` is placed AFTER ``-i``. Before the input it is a fast but
    keyframe-accurate seek, which lands wherever the nearest keyframe happens
    to be -- injecting up to several hundred milliseconds of unmeasured error
    into exactly the end-of-file analysis that end-delay depends on. After the
    input the seek is sample-accurate.
*   Subprocesses are started in their own process group and are always killed
    on timeout or cancellation, so a cancelled run cannot leave orphaned ffmpeg
    processes consuming the machine.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".ts", ".wmv", ".flv"}
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".aac", ".flac", ".ogg", ".opus", ".m4a",
    ".eac3", ".ac3", ".dts", ".wma", ".mka",
}

DECODE_TIMEOUT_S = 300
PROBE_TIMEOUT_S = 60

# How much audio to decode accurately before a requested offset. The coarse
# input seek lands on a packet boundary somewhere before this point; decoding
# through the remainder restores sample-exact positioning. Comfortably longer
# than any real container's packet interval, and short enough that the cost
# does not depend on how far into the file the window sits.
SEEK_PREROLL_S = 20.0


class MediaError(RuntimeError):
    """Raised when a media file cannot be decoded or probed."""


class Cancelled(RuntimeError):
    """Raised when a cancellation token is tripped mid-operation."""


class CancellationToken:
    """Thread-safe cancellation flag shared across a batch.

    Tracks live subprocesses so cancelling terminates in-flight ffmpeg work
    immediately rather than waiting for each decode to finish.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            _terminate(process)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled("operation cancelled")

    def register(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)
        # Cancellation may have arrived between the check and the spawn.
        if self._event.is_set():
            _terminate(process)

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)


def _terminate(process: subprocess.Popen) -> None:
    """Kill a process and its children, tolerating races with normal exit."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.kill()
        else:
            # Kill the whole group so ffmpeg's own children go too.
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except Exception:  # noqa: BLE001 - already exiting
            pass


def _popen_kwargs() -> dict:
    kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    if os.name == "nt":
        # Prevent a console window flashing up for every decode.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _bundled_dir() -> Optional[str]:
    """Locate ffmpeg binaries bundled alongside the frozen executable."""
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(exe_dir, "resources", "ffmpeg"))
        candidates.append(os.path.join(exe_dir, "ffmpeg"))
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "resources", "ffmpeg"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(root, "src-tauri", "resources", "ffmpeg"))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def _find_binary(name: str) -> str:
    """Resolve ffmpeg/ffprobe, preferring a bundled copy over PATH.

    The original code resolved a bundled binary in Rust but called the bare name
    from Python, so a machine without ffmpeg on PATH failed every file even
    though a bundled copy was present.
    """
    exe = f"{name}.exe" if os.name == "nt" else name
    override = os.environ.get(f"AUDIOSYNC_{name.upper()}")
    if override and os.path.isfile(override):
        return override

    bundled_dir = _bundled_dir()
    if bundled_dir:
        candidate = os.path.join(bundled_dir, exe)
        if os.path.isfile(candidate):
            return candidate

    found = shutil.which(name)
    if found:
        return found
    return name  # Let the spawn fail with a clear message.


def ffmpeg_path() -> str:
    return _find_binary("ffmpeg")


def ffprobe_path() -> str:
    return _find_binary("ffprobe")


def has_ffmpeg() -> bool:
    path = ffmpeg_path()
    return os.path.isfile(path) or shutil.which(path) is not None


@dataclass
class AudioTrack:
    """One selectable audio stream inside a container.

    A file often carries several: the original language, a dub, a commentary.
    Comparing against the wrong one produces a confident measurement of two
    tracks that were never meant to align.
    """

    index: int
    """Position among audio streams, i.e. the N in ffmpeg's ``-map 0:a:N``."""
    codec: Optional[str] = None
    language: Optional[str] = None
    title: Optional[str] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    bit_rate: Optional[int] = None
    """Bits per second, when the container declares it. Absent for lossless
    and for streams whose bitrate is only knowable by decoding them."""
    is_default: bool = False

    @property
    def label(self) -> str:
        """Human-readable description for a track picker."""
        parts = [f"Track {self.index + 1}"]
        if self.language and self.language.lower() not in ("und", "unknown"):
            parts.append(self.language.upper())
        if self.title:
            parts.append(self.title)
        details = []
        if self.codec:
            details.append(self.codec.upper())
        if self.channels:
            details.append(
                {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}.get(
                    self.channels, f"{self.channels}ch"
                )
            )
        if details:
            parts.append(f"({', '.join(details)})")
        return " · ".join(parts)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "codec": self.codec,
            "language": self.language,
            "title": self.title,
            "channels": self.channels,
            "sampleRate": self.sample_rate,
            "bitRate": self.bit_rate,
            "isDefault": self.is_default,
            "label": self.label,
        }


@dataclass
class MediaInfo:
    duration: Optional[float]
    has_audio: bool
    has_video: bool
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    audio_tracks: List["AudioTrack"] = field(default_factory=list)
    fps: Optional[float] = None
    """Video frame rate, when the file has a video stream. A dub timed against
    a 25fps PAL master drifts steadily against a 23.976fps source, and naming
    that cause is far more actionable than reporting the drift alone."""

    container_format: Optional[str] = None
    """ffprobe's format_name. Tells a real container from a raw elementary
    stream, which is what decides whether AC3 decoder priming survives into a
    measurement."""


def _parse_frame_rate(raw: Optional[str]) -> Optional[float]:
    """Parse ffprobe's ``num/den`` frame rate.

    Parsed rather than eval'd: this string comes straight out of a media file's
    metadata, and eval on untrusted input is arbitrary code execution.
    """
    if not raw or raw in ("0/0", "N/A"):
        return None
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            den = float(denominator)
            if den == 0:
                return None
            value = float(numerator) / den
        else:
            value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if 0 < value < 1000 else None


def probe(path: str, token: Optional[CancellationToken] = None) -> MediaInfo:
    """Read stream metadata via ffprobe."""
    if not os.path.isfile(path):
        raise MediaError(f"File not found: {path}")

    command = [
        ffprobe_path(), "-v", "error",
        # format_name distinguishes a real container from a raw elementary
        # stream, which decides whether codec priming reaches a measurement.
        "-show_entries", "format=duration,format_name",
        "-show_streams", "-of", "json", path,
    ]
    stdout = _run(command, PROBE_TIMEOUT_S, token, what=f"probe {os.path.basename(path)}")

    try:
        payload = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise MediaError(f"Could not parse ffprobe output for {os.path.basename(path)}") from exc

    has_audio = has_video = False
    codec = sample_rate = channels = None
    fps: Optional[float] = None
    tracks: List[AudioTrack] = []

    for stream in payload.get("streams", []) or []:
        kind = stream.get("codec_type")
        if kind == "audio":
            has_audio = True
            tags = stream.get("tags") or {}
            disposition = stream.get("disposition") or {}

            stream_rate = None
            if stream.get("sample_rate"):
                try:
                    stream_rate = int(stream["sample_rate"])
                except (TypeError, ValueError):
                    stream_rate = None

            # Matroska keeps the per-stream bitrate in a tag rather than the
            # stream itself, so check both before giving up.
            stream_bitrate = None
            raw_bitrate = (
                stream.get("bit_rate")
                or tags.get("BPS")
                or tags.get("BPS-eng")
            )
            if raw_bitrate:
                try:
                    stream_bitrate = int(raw_bitrate)
                except (TypeError, ValueError):
                    stream_bitrate = None

            tracks.append(
                AudioTrack(
                    index=len(tracks),
                    codec=stream.get("codec_name"),
                    language=tags.get("language") or tags.get("LANGUAGE"),
                    title=tags.get("title") or tags.get("TITLE"),
                    channels=stream.get("channels"),
                    sample_rate=stream_rate,
                    bit_rate=stream_bitrate,
                    is_default=bool(disposition.get("default")),
                )
            )

            codec = codec or stream.get("codec_name")
            sample_rate = sample_rate or stream_rate
            channels = channels or stream.get("channels")
        elif kind == "video":
            # Cover art is a video stream by codec type but is not real video.
            if stream.get("disposition", {}).get("attached_pic"):
                continue
            has_video = True
            fps = fps or _parse_frame_rate(stream.get("r_frame_rate"))

    duration = None
    raw_duration = (payload.get("format") or {}).get("duration")
    if raw_duration not in (None, "N/A"):
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = None

    if duration is None:
        # Some containers only carry duration on the stream, not the format.
        for stream in payload.get("streams", []) or []:
            value = stream.get("duration")
            if value not in (None, "N/A"):
                try:
                    duration = float(value)
                    break
                except (TypeError, ValueError):
                    continue

    return MediaInfo(
        duration,
        has_audio,
        has_video,
        codec,
        sample_rate,
        channels,
        audio_tracks=tracks,
        fps=fps,
        container_format=(payload.get("format") or {}).get("format_name"),
    )


def get_duration(path: str, token: Optional[CancellationToken] = None) -> Optional[float]:
    try:
        return probe(path, token).duration
    except MediaError:
        return None


def load_audio(
    path: str,
    sr: int,
    duration: Optional[float] = None,
    offset: float = 0.0,
    token: Optional[CancellationToken] = None,
    track: int = 0,
) -> np.ndarray:
    """Decode a mono float32 segment at the requested sample rate.

    Uses ffmpeg for every format. The original code had three overlapping
    loaders (soundfile, librosa, ffmpeg) whose fallbacks silently disagreed
    about seek semantics; one decoder means one behaviour to reason about.

    Args:
        track: which audio stream to decode, as the N in ``-map 0:a:N``. Files
            frequently carry several (original language, dub, commentary), and
            without this every comparison silently used the first one.
    """
    if not os.path.isfile(path):
        raise MediaError(f"File not found: {path}")
    if token:
        token.raise_if_cancelled()

    command = [ffmpeg_path(), "-nostdin"]

    # Seeking in two stages: jump most of the way with an input seek, then let
    # the decoder run accurately through the last few seconds.
    #
    # An input seek (-ss before -i) is near-instant but lands on a container
    # packet boundary, so it cannot be trusted to be sample-exact on its own.
    # An output seek (-ss after -i) is sample-exact but decodes every frame
    # from the start of the file to get there -- for a window two hours into a
    # movie that is two hours of wasted decoding, per window, per file.
    #
    # Doing the coarse jump first and the accurate seek only over SEEK_PREROLL_S
    # keeps the exact-sample guarantee while making the cost independent of how
    # far into the file the window sits. Measured on a 2-hour AAC track this is
    # ~29x faster over a six-window pass, and the residual shift is identical
    # for both files of a pair, so the measured *difference* is unchanged.
    if offset > 0:
        preroll = min(SEEK_PREROLL_S, offset)
        coarse = offset - preroll
        if coarse > 0:
            command.extend(["-ss", f"{coarse:.6f}"])
        command.extend(["-i", path, "-ss", f"{preroll:.6f}"])
    else:
        command.extend(["-i", path])

    if duration is not None:
        command.extend(["-t", f"{duration:.6f}"])
    command.extend([
        "-map", f"0:a:{max(0, track)}",
        "-vn", "-sn", "-dn",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1", "-",
    ])

    what = os.path.basename(path)
    if track:
        what = f"{what} (track {track + 1})"
    stdout = _run(command, DECODE_TIMEOUT_S, token, what=f"decode {what}")
    if not stdout:
        raise MediaError(f"No audio decoded from {os.path.basename(path)}")

    samples = np.frombuffer(stdout, dtype=np.float32)
    if samples.size == 0:
        raise MediaError(f"No audio samples in {os.path.basename(path)}")
    # Copy off the read-only buffer so downstream code may write freely.
    return np.array(samples, dtype=np.float32)


def _run(
    command: list[str],
    timeout: int,
    token: Optional[CancellationToken],
    what: str,
) -> bytes:
    """Run a subprocess with cancellation, timeout and guaranteed cleanup."""
    if token:
        token.raise_if_cancelled()

    try:
        process = subprocess.Popen(command, **_popen_kwargs())
    except FileNotFoundError as exc:
        raise MediaError(
            f"{os.path.basename(command[0])} not found. Install FFmpeg or bundle it "
            f"in src-tauri/resources/ffmpeg."
        ) from exc
    except OSError as exc:
        raise MediaError(f"Could not start {os.path.basename(command[0])}: {exc}") from exc

    if token:
        token.register(process)
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(process)
            process.communicate()
            raise MediaError(f"Timed out after {timeout}s: {what}")

        if token and token.cancelled:
            raise Cancelled("operation cancelled")

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
            tail = detail[-1] if detail else f"exit code {process.returncode}"
            raise MediaError(f"Failed to {what}: {tail}")
        return stdout
    finally:
        if token:
            token.unregister(process)
        if process.poll() is None:
            _terminate(process)


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS


def is_media_file(path: str) -> bool:
    return is_video_file(path) or is_audio_file(path)
