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

import functools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".ts", ".wmv", ".flv"}
# Every extension an external dub actually arrives as. Deliberately generous:
# this only decides what a folder scan offers to pair up, and probing rejects
# anything that turns out not to carry audio. Being short here is the worse
# failure -- a .ec3 or .thd track was simply invisible, with nothing to explain
# why. Kept in step with AUDIO_EXTENSIONS in MkvBatchMux's shared/lib/extensions.ts.
AUDIO_EXTENSIONS = {
    # Dolby
    ".ac3", ".eac3", ".ec3", ".thd", ".truehd", ".mlp",
    # DTS
    ".dts", ".dtsma", ".dtshd",
    # MPEG and friends
    ".aac", ".m4a", ".m4b", ".mp3", ".mp2", ".mpa",
    # Lossless and open formats
    ".flac", ".wav", ".w64", ".aiff", ".aif", ".caf", ".alac",
    ".ogg", ".oga", ".opus", ".ape", ".tak", ".tta", ".wv",
    # Containers that commonly hold nothing but audio
    ".mka", ".wma",
}

# ffprobe format names that are a bare sequence of frames rather than a real
# container. These carry no duration field, so ffprobe either divides the file
# size by a nominal bitrate or gives up: raw ADTS AAC came back 14% short of a
# 57.5s file, and raw TrueHD came back "N/A" and failed the pair outright. A
# duration that wrong makes a PAL speedup look like nothing in particular, and
# the pair then cannot be measured at all -- and bare .eac3/.ac3/.aac/.thd is
# exactly the shape an external dub arrives in.
ELEMENTARY_STREAM_FORMATS = frozenset({
    "aac", "ac3", "eac3", "dts", "truehd", "mlp", "mp3", "spdif",
})

DECODE_TIMEOUT_S = 300
PROBE_TIMEOUT_S = 60

# How much audio to decode accurately before a requested offset. The coarse
# input seek lands on a packet boundary somewhere before this point; decoding
# through the remainder restores sample-exact positioning. Comfortably longer
# than any real container's packet interval, and short enough that the cost
# does not depend on how far into the file the window sits.
SEEK_PREROLL_S = 20.0


class MediaError(RuntimeError):
    """Raised when a media file cannot be decoded or probed.

    The message is for the user; ``detail`` keeps FFmpeg's own words, all of
    them, for working out what went wrong."""

    def __init__(self, message: str, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.detail = detail


# FFmpeg's ways of saying what is wrong with a file, and what that means to
# someone who has to fix it. Matched against the whole of FFmpeg's error
# output, in order, because the line that names the cause is rarely the last
# one: "matches no streams" is followed by "Error opening output files:
# Invalid argument", which on its own says nothing.
_FAILURES = [
    ("permission denied", "{name} cannot be read (permission denied). Check its permissions, or close the program that has it open."),
    ("moov atom not found", "{name} is incomplete: its index is missing, which usually means the download or copy stopped early."),
    ("matches no streams", "{name} has no audio track {track}. Choose a track the file has."),
    ("does not contain any stream", "{name} has no audio in it."),
    ("decoder not found", "{name} uses a codec this FFmpeg build cannot decode. Install a full FFmpeg build."),
    ("unknown decoder", "{name} uses a codec this FFmpeg build cannot decode. Install a full FFmpeg build."),
    ("not currently supported", "{name} uses a codec feature this FFmpeg build cannot decode. Install a newer FFmpeg build."),
    ("encrypt", "{name} is encrypted (DRM) and cannot be decoded."),
    ("drm", "{name} is encrypted (DRM) and cannot be decoded."),
    ("end of file", "{name} ends early: it is incomplete or damaged."),
    ("invalid data found", "{name} is damaged, incomplete, or not a media file; FFmpeg could not read it."),
    ("ebml header parsing failed", "{name} is not a valid Matroska file, or it is damaged."),
    ("could not find codec parameters", "{name} is damaged or incomplete; FFmpeg could not work out what its audio is."),
    ("no such file", "{name} was not found. It may have been moved or renamed, or be on a drive that is not connected."),
]


def describe_failure(path: str, detail: Optional[str], track: int = 0) -> str:
    """What FFmpeg's error output means for this file, in words a user can act
    on, with FFmpeg's own last line kept after it for anyone diagnosing."""
    name = os.path.basename(path) or path
    text = (detail or "").strip()
    lowered = text.lower()
    last = text.splitlines()[-1].strip() if text else ""
    for needle, message in _FAILURES:
        if needle in lowered:
            said = message.format(name=name, track=track + 1)
            return f"{said} (FFmpeg: {last})" if last else said
    return f"FFmpeg could not read {name}: {last or 'no reason given'}"


def check_readable(path: str) -> None:
    """Refuse a path that cannot be a readable media file, saying why, before
    FFmpeg is asked to make sense of it."""
    name = os.path.basename(path.rstrip("/\\")) or path
    if os.path.isdir(path):
        raise MediaError(f"{name} is a folder, not a media file.")
    if not os.path.isfile(path):
        folder = os.path.dirname(path)
        where = (
            f"its folder {folder} is missing too" if folder and not os.path.isdir(folder)
            else f"it is not in {folder}" if folder else "it does not exist"
        )
        raise MediaError(
            f"{name} was not found ({where}). It may have been moved or renamed, or be on a drive that is "
            "not connected."
        )
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise MediaError(f"{name} cannot be read: {exc.strerror or exc}.") from exc
    if size == 0:
        raise MediaError(f"{name} is empty (0 bytes); the download or copy did not finish.")


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
            # Kill the tree, not just the process: ffmpeg from Chocolatey or
            # Scoop is a shim that runs the real ffmpeg as a child, and a
            # venv's python.exe is a launcher for the real interpreter.
            # Killing only the parent left the child running with our pipes
            # open, so a cancelled read waited for it forever.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=10,
            )
            if process.poll() is None:
                process.kill()
        else:
            # Kill the whole group so ffmpeg's own children go too.
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
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
    start_time: Optional[float] = None
    """Where the stream's first sample sits on the file's clock, in seconds,
    as ffprobe reads it. See ``audio_lead_s``."""

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

    start_time: Optional[float] = None
    """The file's own time zero, as ffprobe reads it: the earliest stream."""


def audio_lead_s(info: MediaInfo, track: int = 0) -> float:
    """How far into the file's clock an audio stream's first sample sits.

    ffmpeg seeks, and a muxer places a track, on the file's clock, whose
    zero is its earliest stream; a decode from the top starts at the
    stream's own first sample. The two differ wherever the audio starts
    after the picture -- 23 ms on a real BluRay MKV -- and a timeline
    counted from the first sample is out by that much against every seek,
    every picture frame and every mux. Decodes that are to share the
    file's clock are padded at the front by this much.
    """
    if not info.audio_tracks or info.start_time is None:
        return 0.0
    stream = info.audio_tracks[min(max(0, track), len(info.audio_tracks) - 1)]
    if stream.start_time is None:
        return 0.0
    lead = stream.start_time - info.start_time
    return lead if math.isfinite(lead) and lead > 0.0 else 0.0


# The most encoder priming a container marks to be skipped: AAC's 1024 samples
# are 128 ms at 8 kHz, HE-AAC's 2048 at 16 kHz the same; AC-3 and E-AC-3 are
# 256 samples, Opus 312. A file starting further before zero than this is not
# priming, and is left as it is.
MAX_PRIMING_S = 0.25


@functools.lru_cache(maxsize=256)
def _probed(path: str, size: int, mtime_ns: int, reader: str) -> MediaInfo:
    # ``reader`` is only a key: what a file reports depends on which build reads it.
    return probe(path)


def _stream_of(info: MediaInfo, track: int) -> Optional[AudioTrack]:
    return info.audio_tracks[min(max(0, track), len(info.audio_tracks) - 1)] if info.audio_tracks else None


@functools.lru_cache(maxsize=256)
def _origins(path: str, track: int, size: int, mtime_ns: int, reader: str) -> Tuple[float, float]:
    info = _probed(path, size, mtime_ns, reader)
    stream = _stream_of(info, track)
    track_start = stream.start_time if stream and stream.start_time is not None else 0.0
    return (info.start_time if info.start_time is not None else 0.0), track_start


def _is_dts(path: str, track: int) -> bool:
    """Whether this track is DTS, whose core can be decoded on its own."""
    try:
        stat = os.stat(path)
        info = _probed(path, stat.st_size, stat.st_mtime_ns, ffprobe_path())
    except (OSError, MediaError):
        return False
    stream = _stream_of(info, track)
    codec = stream.codec if stream else info.audio_codec
    return (codec or "").lower() == "dts"


@functools.lru_cache(maxsize=256)
def _first_packet_s(path: str, track: int, size: int, mtime_ns: int, reader: str) -> Optional[float]:
    # The stream's start_time is not enough: Matroska reports 0 for a track
    # whose first block sits seconds into the file. Its first packet does not.
    output = _run(
        [
            ffprobe_path(), "-v", "error",
            "-select_streams", f"a:{max(0, track)}",
            "-show_entries", "packet=pts_time",
            "-read_intervals", "%+#1",
            "-of", "csv=p=0",
            path,
        ],
        PROBE_TIMEOUT_S,
        None,
        what=f"probe {os.path.basename(path)}",
    )
    return _parse_float(output.decode(errors="replace").strip().splitlines()[0]) if output.strip() else None


def track_lead_s(path: str, track: int = 0) -> float:
    """How far into the file's clock this audio track's first sample sits:
    ``audio_lead_s`` for a file on disk, read from the first packet. Zero
    when it starts with the file, or the file cannot be read here (the
    decode that follows says why)."""
    try:
        stat = os.stat(path)
        key = (path, int(track), stat.st_size, stat.st_mtime_ns, ffprobe_path())
        file_start, _ = _origins(*key)
        first = _first_packet_s(*key)
    except (OSError, MediaError, IndexError):
        return 0.0
    if first is None:
        return 0.0
    lead = first - file_start
    return lead if math.isfinite(lead) and lead > 0.0 else 0.0


def kept_priming(path: str, track: int = 0) -> Tuple[float, float]:
    """Encoder priming this FFmpeg build decodes as sound, for a seek and for a
    decode from the top, in seconds.

    A container marks the priming an encoder put in front of the audio (1024
    samples of AAC, 256 of AC-3 or E-AC-3) to be skipped. FFmpeg 7 and later
    skip it. FFmpeg 6.1 -- what Ubuntu 24.04 packages -- decodes it as sound
    and reports the stream as starting that far before zero, and everything
    then comes out that much late: 64 ms on a 16 kHz AAC track in Matroska,
    21 ms at 48 kHz, 5.3 ms on E-AC-3. Measured with the same file both ways,
    it was the reader and never the writer.

    Two numbers, because the two ways of decoding are off by different
    amounts: a seek is taken from the file's earliest timestamp, a decode
    from the top starts at this track's first sample.

    Returns:
        (seek, top): how much later to seek, and how much to drop from the
        front of a decode from the top. Zero for a build that skips priming,
        a container that trims it itself, or a file starting at zero.
    """
    try:
        stat = os.stat(path)
        file_start, track_start = _origins(path, int(track), stat.st_size, stat.st_mtime_ns, ffprobe_path())
    except (OSError, MediaError, IndexError):
        return 0.0, 0.0

    def priming(start: float) -> float:
        return -start if math.isfinite(start) and -MAX_PRIMING_S <= start < 0.0 else 0.0

    return priming(file_start), priming(track_start)


def _parse_float(raw) -> Optional[float]:
    if raw in (None, "N/A", ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def choose_frame_rate(nominal: Optional[float], average: Optional[float]) -> Optional[float]:
    """The rate a video's picture actually plays at, from ffprobe's two answers.

    ``r_frame_rate`` is the rate the timestamps are written at and
    ``avg_frame_rate`` frames over duration. They agree on almost every file,
    and where they do not the nominal one is the one that is wrong for timing:
    interlaced 29.97 reports its field rate, 59.94; soft-telecined film on a
    DVD or broadcast stream reports 29.97 while its pictures, and so its
    runtime, are 23.976 film. A dub is mastered to the runtime, so the average
    is taken whenever it is a standard rate the nominal one is not.
    """
    from .framerate import exact_rate

    nominal_rate = exact_rate(nominal) if nominal else None
    average_rate = exact_rate(average) if average else None
    if nominal_rate is not None and average_rate is not None:
        return float(average_rate) if average_rate != nominal_rate else float(nominal_rate)
    if average_rate is not None:
        return float(average_rate)
    if nominal_rate is not None:
        return float(nominal_rate)
    return average or nominal


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
    check_readable(path)

    command = [
        ffprobe_path(), "-v", "error",
        # format_name distinguishes a real container from a raw elementary
        # stream, which decides whether codec priming reaches a measurement.
        "-show_entries", "format=duration,format_name,start_time",
        "-show_streams", "-of", "json", path,
    ]
    try:
        stdout = _run(command, PROBE_TIMEOUT_S, token, what=f"probe {os.path.basename(path)}")
    except MediaError as exc:
        if exc.detail is None:
            raise
        raise MediaError(describe_failure(path, exc.detail), exc.detail) from exc

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
                    start_time=_parse_float(stream.get("start_time")),
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
            fps = fps or choose_frame_rate(
                _parse_frame_rate(stream.get("r_frame_rate")),
                _parse_frame_rate(stream.get("avg_frame_rate")),
            )

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

    container_format = (payload.get("format") or {}).get("format_name")
    if duration is None or _is_elementary_stream(container_format):
        # Read where the last packet actually ends. Demuxing only, no decoding,
        # so it costs a tenth of a second on the files that need it and is
        # exact rather than inferred from a bitrate.
        exact = _packet_duration(path, token)
        if exact is not None:
            duration = exact

    return MediaInfo(
        duration,
        has_audio,
        has_video,
        codec,
        sample_rate,
        channels,
        audio_tracks=tracks,
        fps=fps,
        container_format=container_format,
        start_time=_parse_float((payload.get("format") or {}).get("start_time")),
    )


def _is_elementary_stream(container_format: Optional[str]) -> bool:
    """Whether ffprobe can only estimate this format's duration."""
    if not container_format:
        return False
    names = {part.strip().lower() for part in container_format.split(",")}
    return bool(names & ELEMENTARY_STREAM_FORMATS)


def _packet_duration(
    path: str, token: Optional[CancellationToken] = None
) -> Optional[float]:
    """Where the audio actually ends, from the last packet rather than a guess.

    Returns None on any failure: an unreliable duration is still better than no
    analysis, so this only ever improves on what ffprobe reported.
    """
    command = [
        ffprobe_path(), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "packet=pts_time,duration_time",
        "-of", "csv=p=0", path,
    ]
    try:
        stdout = _run(
            command, PROBE_TIMEOUT_S, token,
            what=f"measure {os.path.basename(path)}",
        )
    except MediaError:
        return None

    for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            end = float(parts[0]) + float(parts[1])
        except ValueError:
            continue
        if end > 0:
            return end
    return None


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
    channels: int = 1,
) -> np.ndarray:
    """Decode a float32 segment at the requested sample rate.

    Uses ffmpeg for every format. The original code had three overlapping
    loaders (soundfile, librosa, ffmpeg) whose fallbacks silently disagreed
    about seek semantics; one decoder means one behaviour to reason about.

    Args:
        track: which audio stream to decode, as the N in ``-map 0:a:N``. Files
            frequently carry several (original language, dub, commentary), and
            without this every comparison silently used the first one.
        channels: 1 for the mono mix, as a flat array; 2 for FFmpeg's stereo
            downmix, as (samples, 2). The stereo downmix folds the centre
            channel into both sides equally, so left minus right is the mix
            with the dialogue taken out (see ``analyze._measure_window``),
            and (left + right) / sqrt(2) is the mono mix to the sample.
    """
    check_readable(path)
    if token:
        token.raise_if_cancelled()

    # Every read is on the file's clock, whose zero is its earliest stream. A
    # seek lands there; a read that does not seek starts at this track's own
    # first sample instead. Where the audio starts after the picture -- 5 s on
    # some remuxes -- those two disagree by that much, and a pair measured
    # with both kinds of read (the dub read from zero near the start, the
    # video sought to) found an offset of exactly that size in its first
    # windows and reported a cut that was never there. Before the track has
    # started there is nothing to decode: read from its first sample, which
    # is a seek like every other, and put the gap back as silence.
    requested = duration
    silence_s = 0.0
    lead = track_lead_s(path, track)
    if lead > 0.0 and offset < lead:
        silence_s = lead - offset
        offset = lead
        if duration is not None:
            duration -= silence_s
            if duration <= 0.0:
                return _silence(int(round(requested * sr)), channels)

    # On a build that decodes encoder priming as sound, the file's zero is
    # that far into what it decodes (see ``kept_priming``). Every read here is
    # a seek -- one from zero included, once there is priming to pass -- and
    # FFmpeg takes a seek from the file's earliest timestamp, so it is the
    # file's priming that is added, not this track's.
    offset = offset + kept_priming(path, track)[0]

    # A DTS-HD Master Audio track is a lossy DTS core plus a lossless
    # residual added to it sample for sample, so the core alone sits exactly
    # where the full decode does (checked on FFmpeg's DTS-HD MA samples: lag
    # 0) and decodes ~5x faster -- 0.49 s against 2.35 s for five minutes of
    # 7.1. What the residual adds is detail far below anything a 16 kHz
    # analysis can see. A stream with no core decodes nothing this way and
    # is decoded in full instead.
    if _is_dts(path, track):
        try:
            return _decode(path, sr, duration, offset, token, track, channels, silence_s, core_only=True)
        except MediaError:
            if token:
                token.raise_if_cancelled()
    return _decode(path, sr, duration, offset, token, track, channels, silence_s)


def _silence(samples: int, channels: int) -> np.ndarray:
    shape = samples if channels == 1 else (samples, channels)
    return np.zeros(shape, dtype=np.float32)


def _decode(
    path: str,
    sr: int,
    duration: Optional[float],
    offset: float,
    token: Optional[CancellationToken],
    track: int,
    channels: int,
    silence_s: float,
    core_only: bool = False,
) -> np.ndarray:
    """One FFmpeg decode for ``load_audio``, with the offset already on the
    file's clock and the silence in front of the track already measured."""
    command = [ffmpeg_path(), "-nostdin"]
    if core_only:
        command.extend(["-core_only", "1"])

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
        "-ar", str(sr), "-ac", str(channels), "-",
    ])

    what = os.path.basename(path)
    if track:
        what = f"{what} (track {track + 1})"
    try:
        stdout = _run(command, DECODE_TIMEOUT_S, token, what=f"decode {what}")
    except MediaError as exc:
        if exc.detail is None:
            raise
        raise MediaError(describe_failure(path, exc.detail, track), exc.detail) from exc
    if not stdout:
        raise MediaError(f"No audio decoded from {os.path.basename(path)}")

    samples = np.frombuffer(stdout, dtype=np.float32)
    if samples.size == 0:
        raise MediaError(f"No audio samples in {os.path.basename(path)}")
    if channels > 1:
        samples = samples[: samples.size - samples.size % channels].reshape(-1, channels)
    if silence_s > 0.0:
        return np.concatenate([_silence(int(round(silence_s * sr)), channels), samples])
    # Copy off the read-only buffer so downstream code may write freely.
    return np.array(samples, dtype=np.float32)


def stream_audio(
    path: str,
    sr: int,
    track: int = 0,
    channels: int = 1,
    token: Optional[CancellationToken] = None,
    block_s: float = 4.0,
    audio_filter: Optional[str] = None,
    lead_s: float = 0.0,
    issues: Optional[List[str]] = None,
) -> Iterator[np.ndarray]:
    """Decode a whole stream from t=0 in fixed blocks, without ever seeking.

    ``load_audio`` holds a decoded window in memory and seeks to reach it. For
    a whole feature film neither is acceptable: two hours at 16 kHz mono is
    half a gigabyte per track, and a seek is only exact for some codecs --
    measured with an impulse, an input ``-ss`` through AAC lands 8 samples
    early, while a decode from the start lands on the sample for every codec.
    Reading the file once from the top and handing out blocks keeps memory
    flat and makes every position exact by construction, which is what lets
    the dub-sync analysis and the dub-sync render agree to the sample: both
    see each file through this one path.

    Yields float32 arrays shaped ``(samples, channels)`` (``(samples,)`` when
    mono). The final block is whatever is left. Cancelling stops the decode
    mid-file and kills ffmpeg.

    Args:
        audio_filter: an ffmpeg ``-af`` chain applied before the format
            conversion, for the callers that need a time-stretch.
        lead_s: silence yielded before the first decoded sample, so the
            stream's own first sample lands where it sits on the file's
            clock (see ``audio_lead_s``) and position ``i`` of what is
            yielded is the file's time ``i / sr`` -- the time a seek, a
            picture frame and a muxer all use.
        issues: given a list, the decode errors FFmpeg reported and decoded
            past are added to it: a damaged file that still decodes.
    """
    check_readable(path)
    if token:
        token.raise_if_cancelled()

    command = [ffmpeg_path(), "-nostdin", "-v", "error", "-i", path,
               "-map", f"0:a:{max(0, track)}", "-vn", "-sn", "-dn"]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend([
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", str(max(1, channels)), "-",
    ])

    frame_bytes = 4 * max(1, channels)
    block_bytes = max(frame_bytes, int(block_s * sr) * frame_bytes)
    what = f"decode {os.path.basename(path)}"

    try:
        process = subprocess.Popen(command, **_popen_kwargs())
    except FileNotFoundError as exc:
        raise MediaError(
            f"{os.path.basename(command[0])} not found. Install FFmpeg or bundle it "
            f"in src-tauri/resources/ffmpeg."
        ) from exc
    except OSError as exc:
        raise MediaError(f"Could not start {os.path.basename(command[0])}: {exc}") from exc

    # stderr is drained on its own thread: ffmpeg blocks once the pipe fills,
    # and a decoder that prints a warning per frame would otherwise stall a
    # two-hour decode after the first few kilobytes of complaints.
    stderr_chunks: list[bytes] = []
    drain = threading.Thread(
        target=lambda: stderr_chunks.append(process.stderr.read()), daemon=True
    )
    drain.start()
    if token:
        token.register(process)

    produced = 0
    # Priming this build decodes as sound, dropped before anything is yielded
    # (see ``kept_priming``); zero on a build that skips it.
    drop = int(round(kept_priming(path, track)[1] * sr))
    try:
        assert process.stdout is not None
        pad = int(round(max(0.0, lead_s) * sr))
        if pad:
            yield np.zeros((pad, channels), dtype=np.float32) if channels > 1 else np.zeros(pad, dtype=np.float32)
        pending = b""
        while True:
            if token:
                token.raise_if_cancelled()
            chunk = process.stdout.read(block_bytes)
            if not chunk:
                break
            pending += chunk
            usable = len(pending) - (len(pending) % frame_bytes)
            if usable == 0:
                continue
            block = np.frombuffer(pending[:usable], dtype=np.float32)
            pending = pending[usable:]
            produced += usable // frame_bytes
            if drop:
                skip = min(drop, len(block) // channels)
                block = block[skip * channels:]
                drop -= skip
                if block.size == 0:
                    continue
            yield block.reshape(-1, channels) if channels > 1 else block.copy()
        process.wait()
        drain.join(timeout=5)
        if token and token.cancelled:
            raise Cancelled("operation cancelled")
        detail = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise MediaError(describe_failure(path, detail or f"exit code {process.returncode}", track), detail)
        if produced == 0:
            raise MediaError(f"No audio decoded from {os.path.basename(path)}")
        if detail and issues is not None:
            # At -v error anything FFmpeg says is an error it decoded past:
            # damaged packets it skipped or concealed. The audio is still
            # usable, but not across the damage, and the reader should know.
            issues.extend(line.strip() for line in detail.splitlines() if line.strip())
    finally:
        if token:
            token.unregister(process)
        if process.poll() is None:
            _terminate(process)
            # On Windows a killed ffmpeg keeps the file open until it has
            # actually exited, so a read stopped early (a voice move reads
            # a span from the middle) would leave the file locked against
            # deleting or overwriting it a moment later.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


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
            raise MediaError(
                f"Gave up after {timeout}s trying to {what}; the file may be on a slow or disconnected drive, "
                "or FFmpeg stalled on a damaged part of it."
            )

        if token and token.cancelled:
            raise Cancelled("operation cancelled")

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            tail = detail.splitlines()[-1] if detail else f"exit code {process.returncode}"
            raise MediaError(f"Failed to {what}: {tail}", detail or f"exit code {process.returncode}")
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
