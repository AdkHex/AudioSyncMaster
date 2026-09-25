"""Write the synced dub a plan describes, and optionally mux it into the video.

The plan says which piece of which file sits at every moment of the video's
timeline. This module reads those pieces and joins them, in the video's
order, into one track exactly the video's length: dub where the dub belongs,
the original where it does not, each seam crossfaded so nothing clicks.

Everything is read through ffmpeg with the same two-stage seek ``load_audio``
uses, which lands on the sample for every codec it was measured with and
eight samples early for AAC -- a sixth of a millisecond. Nothing is held in
memory beyond a chunk: a five-hour 7.1 track goes through the same few
megabytes as a stereo short.

A dub that runs at a different speed is decoded at the rate that cancels the
difference, in chunks. The compensating rate has to be a whole number and
usually is not exactly one; the rounding is a drift of a few microseconds a
second, which over two hours would be tens of milliseconds, so each chunk
starts afresh from an exact position and the drift never grows past what a
minute can accumulate.
"""

from __future__ import annotations

import copy
import os
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from .codecdelay import codec_delay_ms
from .dubsync import DubSyncPlan, Segment
from .voicefix import VoicePiece
from .media import (
    SEEK_PREROLL_S,
    CancellationToken,
    Cancelled,
    MediaError,
    _popen_kwargs,
    _terminate,
    audio_lead_s,
    ffmpeg_path,
    probe,
    stream_audio,
)

# Crossfade at every seam. Ten milliseconds is below what the ear resolves
# as a fade and above what it hears as a click.
XFADE_S = 0.010
# A stretch of dub is decoded in pieces this long, so a rounded compensating
# rate cannot drift further than a minute's worth: under a millisecond.
CHUNK_S = 60.0
# How much audio each decode hands over at a time.
BLOCK_S = 4.0
ENCODE_TIMEOUT_S = 4 * 3600

# Output codecs, with the bitrate used per channel count when none is given.
# Lossless formats ignore the bitrate.
CODECS = {
    "flac": {"args": ["-c:a", "flac", "-compression_level", "5"], "ext": ".flac"},
    "wav": {"args": ["-c:a", "pcm_s24le", "-rf64", "auto"], "ext": ".wav"},
    "aac": {"args": ["-c:a", "aac"], "ext": ".m4a", "bitrate": {1: "128k", 2: "256k", 6: "640k", 8: "768k"}},
    "ac3": {"args": ["-c:a", "ac3"], "ext": ".ac3", "bitrate": {1: "192k", 2: "256k", 6: "640k"}},
    "eac3": {"args": ["-c:a", "eac3"], "ext": ".eac3", "bitrate": {1: "192k", 2: "256k", 6: "768k", 8: "1024k"}},
    "opus": {"args": ["-c:a", "libopus"], "ext": ".opus", "bitrate": {1: "96k", 2: "160k", 6: "384k", 8: "512k"}},
}

# 16-bit WAV for the app's playback excerpts, which every webview decodes;
# not offered as an output codec, so it lives outside CODECS.
_EXCERPT_WAV = {"args": ["-c:a", "pcm_s16le"], "ext": ".wav"}

ProgressFn = Callable[[int, str], None]


@dataclass
class RenderOptions:
    codec: str = "flac"
    bitrate: Optional[str] = None
    sample_rate: Optional[int] = None
    """Output rate. Default: the dub's own."""
    channels: Optional[int] = None
    """Output channel count. Default: the dub's own; the original is
    conformed to it where it fills."""
    xfade_s: float = XFADE_S
    stretch: str = "resample"
    """How a dub at another speed is brought onto the video's clock:
    ``resample`` changes rate and pitch together, exactly as undoing a
    frame-rate conversion should; ``atempo`` keeps the pitch."""
    exact: bool = True
    """Read each file once from the top rather than seeking to every piece
    (see ``_TrackReader``): exact to the sample, at the cost of decoding
    what lies before the first piece. Off for previews, which read a few
    seconds from anywhere in the film and must be ready at once."""
    fix_voices: bool = True
    """Move the dub's voices where the plan's voice pieces say (see
    ``voicefix``). Needs the voice tools and the exact reads above: a voice
    is taken out of the mix again only when it is cut from exactly the
    samples written, so a preview (which seeks) plays the plan without
    them."""


@dataclass
class RenderResult:
    output_path: str
    sample_rate: int
    channels: int
    seconds: float
    clipped_samples: int = 0
    command: str = ""
    muxed_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outputPath": self.output_path,
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "seconds": self.seconds,
            "clippedSamples": self.clipped_samples,
            "muxedPath": self.muxed_path,
            "warnings": list(self.warnings),
        }


# What "the same codec as the dub" means for each codec ffprobe reports.
# Anything not listed cannot be written back as itself (TrueHD and DTS have
# no encoder here), and becomes FLAC: lossless, so nothing is lost twice.
SAME_AS_DUB = {
    "aac": "aac", "ac3": "ac3", "eac3": "eac3", "opus": "opus", "flac": "flac",
    "pcm_s16le": "wav", "pcm_s24le": "wav", "pcm_s32le": "wav", "pcm_f32le": "wav",
}


def resolve_codec(codec: Optional[str], plan: DubSyncPlan, token: Optional[CancellationToken] = None) -> str:
    """Turn a codec choice into one of CODECS. ``same`` (or None) follows the
    dub's own codec where there is an encoder for it, FLAC otherwise."""
    if codec and codec != "same":
        if codec not in CODECS:
            raise MediaError(f"Unknown codec {codec!r}; choose from {', '.join(CODECS)}")
        return codec
    try:
        info = probe(plan.dub_path, token)
    except MediaError:
        return "flac"
    tracks = info.audio_tracks
    own = tracks[plan.dub_track].codec if plan.dub_track < len(tracks) else info.audio_codec
    return SAME_AS_DUB.get((own or "").lower(), "flac")


def default_output_path(plan: DubSyncPlan, codec: str, output_dir: Optional[str] = None) -> str:
    stem = os.path.splitext(os.path.basename(plan.dub_path))[0]
    directory = output_dir or os.path.dirname(os.path.abspath(plan.video_path))
    return os.path.join(directory, f"{stem}.dubsynced{CODECS[codec]['ext']}")


def _decode_command(
    path: str, track: int, rate: int, channels: int, offset_s: float, duration_s: float,
    audio_filter: Optional[str] = None,
) -> List[str]:
    """ffmpeg decode of one span, two-stage seek, raw float output."""
    command = [ffmpeg_path(), "-nostdin", "-v", "error"]
    if offset_s > 0:
        preroll = min(SEEK_PREROLL_S, offset_s)
        coarse = offset_s - preroll
        if coarse > 0:
            command.extend(["-ss", f"{coarse:.6f}"])
        command.extend(["-i", path, "-ss", f"{preroll:.6f}"])
    else:
        command.extend(["-i", path])
    command.extend(["-t", f"{duration_s:.6f}", "-map", f"0:a:{max(0, track)}", "-vn", "-sn", "-dn"])
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend(["-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(rate), "-ac", str(channels), "-"])
    return command


def _stream(command: List[str], channels: int, token: Optional[CancellationToken], what: str) -> Iterator[np.ndarray]:
    """Run a decode and yield ``(samples, channels)`` float32 blocks."""
    frame_bytes = 4 * channels
    block_bytes = int(BLOCK_S * 48000) * frame_bytes
    try:
        process = subprocess.Popen(command, **_popen_kwargs())
    except (FileNotFoundError, OSError) as exc:
        raise MediaError(f"Could not start ffmpeg: {exc}") from exc
    stderr_chunks: List[bytes] = []
    drain = threading.Thread(target=lambda: stderr_chunks.append(process.stderr.read()), daemon=True)
    drain.start()
    if token:
        token.register(process)
    try:
        assert process.stdout is not None
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
            block = np.frombuffer(pending[:usable], dtype=np.float32).reshape(-1, channels)
            pending = pending[usable:]
            yield block
        process.wait()
        drain.join(timeout=5)
        if token and token.cancelled:
            raise Cancelled("operation cancelled")
        if process.returncode != 0:
            detail = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
            tail = detail.splitlines()[-1] if detail else f"exit code {process.returncode}"
            raise MediaError(f"Failed to {what}: {tail}")
    finally:
        if token:
            token.unregister(process)
        if process.poll() is None:
            _terminate(process)


def read_span(
    path: str,
    track: int,
    rate: int,
    channels: int,
    start_sample: int,
    count: int,
    speed: float = 1.0,
    stretch: str = "resample",
    token: Optional[CancellationToken] = None,
    on_chunk: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Exactly ``count`` samples of a file, from ``start_sample`` on the
    output clock, as ``(count, channels)`` float32.

    ``on_chunk`` is told how many samples have been read so far, once per
    decoded chunk: a two-hour stretch is one span, and without this a
    caller has nothing to report for the minutes it takes.

    On the output clock: a file played at ``speed`` has its sample ``i``
    at output sample ``i * speed``, so the span is sought at ``start /
    speed`` in the file's own time and decoded at the compensating rate.
    Short reads past the end of the file are padded with silence; negative
    starts are padded at the front.
    """
    out = np.zeros((count, channels), dtype=np.float32)
    if count <= 0:
        return out
    written = 0
    if start_sample < 0:
        written = min(count, -start_sample)
        start_sample = 0
    chunk = int(CHUNK_S * rate)
    while written < count:
        want = min(chunk, count - written)
        offset_s = (start_sample + written) / rate / speed
        duration_s = want / rate / speed
        if stretch == "atempo" and abs(speed - 1.0) > 1e-12:
            decode_rate = rate
            audio_filter = f"atempo={1.0 / speed:.9f}"
        else:
            decode_rate = max(1, int(round(rate * speed)))
            audio_filter = None
        # A little extra, so a rounded rate that comes up short does not
        # leave a hole; anything over is dropped.
        command = _decode_command(path, track, decode_rate, channels, offset_s, duration_s * 1.001 + 0.05, audio_filter)
        got = 0
        for block in _stream(command, channels, token, f"decode {os.path.basename(path)}"):
            take = min(len(block), want - got)
            if take > 0:
                out[written + got : written + got + take] = block[:take]
                got += take
            if got >= want:
                break
        written += want
        if on_chunk:
            on_chunk(written)
        if got < want:
            # Past the end of the file: the rest stays silent, and there is
            # nothing further on to read.
            break
    return out


class _Encoder:
    """An ffmpeg process fed raw float samples on stdin."""

    def __init__(self, output_path: str, rate: int, channels: int, codec: str,
                 bitrate: Optional[str], token: Optional[CancellationToken]) -> None:
        spec = _EXCERPT_WAV if codec == "wav16" else CODECS[codec]
        command = [
            ffmpeg_path(), "-nostdin", "-v", "error", "-y",
            "-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-i", "-",
        ]
        command.extend(spec["args"])
        chosen = bitrate or spec.get("bitrate", {}).get(channels)
        if chosen and "bitrate" in spec:
            command.extend(["-b:a", chosen])
        command.append(output_path)
        self.command = command
        self.token = token
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, **_popen_kwargs())
        except (FileNotFoundError, OSError) as exc:
            raise MediaError(f"Could not start ffmpeg: {exc}") from exc
        self._stderr: List[bytes] = []
        self._drain = threading.Thread(target=lambda: self._stderr.append(self.process.stderr.read()), daemon=True)
        self._drain.start()
        if token:
            token.register(self.process)

    def write(self, samples: np.ndarray) -> None:
        if self.token:
            self.token.raise_if_cancelled()
        try:
            self.process.stdin.write(np.ascontiguousarray(samples, dtype=np.float32).tobytes())
        except BrokenPipeError as exc:
            raise MediaError(f"ffmpeg stopped accepting audio: {self._error_tail()}") from exc

    def _error_tail(self) -> str:
        detail = b"".join(self._stderr).decode("utf-8", errors="replace").strip()
        return detail.splitlines()[-1] if detail else "no error output"

    def close(self) -> None:
        try:
            self.process.stdin.close()
            self.process.wait(timeout=ENCODE_TIMEOUT_S)
        finally:
            if self.token:
                self.token.unregister(self.process)
            if self.process.poll() is None:
                _terminate(self.process)
        self._drain.join(timeout=5)
        if self.token and self.token.cancelled:
            raise Cancelled("operation cancelled")
        if self.process.returncode != 0:
            raise MediaError(f"ffmpeg failed to write the output: {self._error_tail()}")

    def abort(self) -> None:
        _terminate(self.process)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# Decoded audio kept behind the reading position, so a piece that starts a
# little before the last one ended -- a crossfade, a step back of a few
# frames at a trim -- is served without decoding the file again.
READER_HISTORY_S = 20.0


def _decode_plan(rate: int, speed: float, stretch: str, native_rate: Optional[int]) -> Tuple[int, Optional[str], bool]:
    """How to decode a file so its samples land on an output clock of
    ``rate`` at ``speed``: (decode rate, ffmpeg filter, exact).

    Exact when the conversion is a ratio of whole numbers: asking the
    decoder for ``rate * speed`` when that is an integer (48048 Hz for
    1.001, 50050 Hz for 25/23.976), or relabelling the file's own rate as
    ``native / speed`` and resampling that to ``rate`` when that one is
    (48048 Hz again, for 1000/1001). Otherwise the rounded rate drifts a
    few microseconds a second and the caller reads in chunks instead.
    """
    if abs(speed - 1.0) < 1e-12:
        return rate, None, True
    if stretch == "atempo":
        return rate, f"atempo={1.0 / speed:.9f}", False
    target = rate * speed
    if abs(target - round(target)) < 1e-6:
        return int(round(target)), None, True
    if native_rate:
        relabel = native_rate / speed
        if abs(relabel - round(relabel)) < 1e-6:
            return rate, f"asetrate={int(round(relabel))},aresample={rate}", True
    return max(1, int(round(target))), None, False


class _TrackReader:
    """One track, decoded once from the top and handed out by position on
    the output clock.

    A seek is only as exact as the container's timestamps: Matroska keeps
    them to the millisecond, so a read that seeks lands up to half a
    millisecond either side of where it was asked for, and a different
    amount at every piece. Read from the top, sample ``i`` is sample ``i``
    -- and on the file's clock, padded at the front exactly as the
    analysis is (see ``audio_lead_s``), so a piece is read from exactly
    where the analysis measured it. Pieces come in plan order and almost
    always move forward through the file; a piece that reaches back further
    than READER_HISTORY_S starts the decode over.
    """

    def __init__(
        self, path: str, track: int, rate: int, channels: int, speed: float, stretch: str,
        token: Optional[CancellationToken], native_rate: Optional[int] = None,
    ) -> None:
        self.path, self.track, self.rate, self.channels = path, track, rate, channels
        self.token = token
        self.decode_rate, self.audio_filter, self.exact = _decode_plan(rate, speed, stretch, native_rate)
        # The padding is asked for in seconds at the rate stream_audio
        # decodes at; on the output clock it is the file's lead times speed.
        lead = audio_lead_s(probe(path, token), track)
        self.pad_s = lead * speed * rate / self.decode_rate if self.audio_filter is None else lead * speed
        self.keep = int(READER_HISTORY_S * rate)
        self.restarts = 0
        self._open()

    def _open(self) -> None:
        self._blocks = stream_audio(
            self.path, self.decode_rate, track=self.track, channels=self.channels, token=self.token,
            block_s=BLOCK_S, audio_filter=self.audio_filter, lead_s=self.pad_s,
        )
        self._buffer = np.zeros((0, self.channels), dtype=np.float32)
        self._at = 0  # output sample of _buffer[0]
        self._ended = False

    def close(self) -> None:
        blocks = getattr(self, "_blocks", None)
        if blocks is not None:
            blocks.close()

    def _pull(self) -> bool:
        """Append the next decoded block, keeping READER_HISTORY_S behind."""
        if self._ended:
            return False
        try:
            block = next(self._blocks)
        except StopIteration:
            self._ended = True
            return False
        block = block.reshape(-1, self.channels) if block.ndim == 1 else block
        drop = max(0, len(self._buffer) - self.keep)
        self._buffer = np.concatenate([self._buffer[drop:], block.astype(np.float32, copy=False)])
        self._at += drop
        return True

    def read(self, start: int, count: int, on_block: Optional[Callable[[int], None]] = None) -> np.ndarray:
        """``count`` samples from output sample ``start``; silence before the
        file and past its end."""
        out = np.zeros((count, self.channels), dtype=np.float32)
        if count <= 0:
            return out
        begin = max(0, start)
        if begin < self._at:
            self.close()
            self.restarts += 1
            self._open()
        filled = begin - start
        while filled < count:
            offset = start + filled - self._at
            if offset < len(self._buffer):
                take = min(count - filled, len(self._buffer) - offset)
                out[filled:filled + take] = self._buffer[offset:offset + take]
                filled += take
                if on_block:
                    on_block(filled)
                continue
            if not self._pull():
                break
        return out


def on_file_clock(plan: DubSyncPlan, token: Optional[CancellationToken] = None) -> DubSyncPlan:
    """The plan with every time on the files' own clocks.

    Plans made before the analysis moved onto the file's clock (see
    ``audio_lead_s``) counted each file from its audio's first sample. On
    a file whose audio starts after its picture that is out by the
    difference against every seek and every mux -- 23 ms on a real BluRay
    MKV -- so such a plan is moved onto the file's clock before it is
    written: the video's times by its lead, the dub's by its own.
    """
    if plan.timeline == "container":
        return plan
    video_lead = audio_lead_s(probe(plan.video_path, token), plan.video_track)
    dub_lead = audio_lead_s(probe(plan.dub_path, token), plan.dub_track) * plan.speed
    moved = copy.deepcopy(plan)
    moved.timeline = "container"
    if video_lead < 1e-9 and dub_lead < 1e-9:
        return moved
    segments: List[Segment] = []
    if video_lead > 0.0005:
        segments.append(Segment("fill", 0.0, video_lead, 0.0, note="before the original's audio starts"))
    for segment in moved.segments:
        segment.start_s += video_lead
        segment.end_s += video_lead
        if segment.kind == "dub":
            segment.source_start_s += dub_lead
            segment.offset_s = (segment.offset_s or 0.0) + dub_lead - video_lead
        else:
            segment.source_start_s += video_lead
        segments.append(segment)
    if segments and video_lead <= 0.0005:
        segments[0].start_s = 0.0
    moved.segments = segments
    for piece in moved.voice_pieces:
        piece.dub_start_s += dub_lead
        piece.dub_end_s += dub_lead
        piece.level_s += dub_lead - video_lead
    moved.video_duration_s += video_lead
    return moved


def render(
    plan: DubSyncPlan,
    output_path: str,
    options: Optional[RenderOptions] = None,
    token: Optional[CancellationToken] = None,
    progress: Optional[ProgressFn] = None,
    log: Optional[Callable[[str], None]] = None,
) -> RenderResult:
    """Write the track the plan describes."""
    options = options or RenderOptions()
    say = log or (lambda _m: None)
    plan = on_file_clock(plan, token) if plan.segments else plan
    if options.codec not in CODECS and options.codec != "wav16":
        raise MediaError(f"Unknown codec {options.codec!r}; choose from {', '.join(CODECS)}")
    if not plan.segments:
        raise MediaError("The plan has no segments to render")
    for path, what in ((plan.video_path, "Video"), (plan.dub_path, "Dub")):
        if not os.path.isfile(path):
            raise MediaError(f"{what} not found: {path}")
    if os.path.abspath(output_path) in (os.path.abspath(plan.video_path), os.path.abspath(plan.dub_path)):
        raise MediaError("Refusing to overwrite a source file")

    dub_info = probe(plan.dub_path, token)
    dub_track = dub_info.audio_tracks[plan.dub_track] if plan.dub_track < len(dub_info.audio_tracks) else None
    rate = options.sample_rate or (dub_track.sample_rate if dub_track else None) or dub_info.sample_rate or 48000
    channels = options.channels or (dub_track.channels if dub_track else None) or dub_info.channels or 2
    if options.codec in ("ac3", "eac3") and rate not in (32000, 44100, 48000):
        rate = 48000
    if options.codec == "opus" and rate != 48000:
        rate = 48000

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    total = int(round(plan.video_duration_s * rate))
    half = int(round(options.xfade_s * rate / 2.0))
    xfade = 2 * half
    # A raw AC3 or E-AC3 stream decodes 256 samples late, and nothing in a
    # bare elementary stream tells a player to trim them. Written 256
    # samples early, it plays on the sample. Formats in a container carry
    # their own priming information and need nothing.
    raw_ext = os.path.splitext(output_path)[1].lower().lstrip(".")
    skip = 0
    if options.codec in ("ac3", "eac3") and raw_ext in ("ac3", "eac3", "ec3"):
        skip = int(round(codec_delay_ms(options.codec, rate, options.codec) / 1000.0 * rate))
        if skip:
            say(f"writing {skip} samples early to cancel the raw {options.codec} decoder delay")
    fill_gain = float(10.0 ** (plan.fill_gain_db / 20.0))
    voice_warnings: List[str] = []
    patch = _voice_patch(plan, rate, channels, dub_track, options, fill_gain, token, say, voice_warnings)
    say(
        f"writing {channels} channel(s) at {rate} Hz as {options.codec}, "
        f"{len(plan.dub_segments)} dub piece(s) and {len(plan.fill_segments)} fill(s), "
        f"{options.xfade_s * 1000:.0f}ms crossfades"
    )

    # Written beside the output and renamed onto it at the end, so a write
    # that is stopped or fails leaves whatever was there -- the previous
    # track, when the cuts are being written again by hand -- untouched.
    # The extension stays put: ffmpeg picks the container by it.
    stem, ext = os.path.splitext(output_path)
    staging = f"{stem}.part{ext}"
    encoder = _Encoder(staging, rate, channels, options.codec, options.bitrate, token)
    clipped = 0
    written = 0
    to_skip = skip

    def emit(samples: np.ndarray) -> int:
        """Write samples, dropping the first `skip` of the whole track."""
        nonlocal to_skip
        if to_skip:
            drop = min(to_skip, len(samples))
            samples = samples[drop:]
            to_skip -= drop
        if len(samples):
            encoder.write(samples)
        return len(samples)
    # The previous piece's fading tail: exactly `xfade` samples, starting
    # half a crossfade before that piece's end on the output timeline. The
    # next piece is read from half a crossfade before its own start -- the
    # same instant -- and fades in over the same span, so adding the two
    # is the crossfade.
    held = np.zeros((0, channels), dtype=np.float32)
    # One reader per file, each decoding its file once from the top, when
    # the reads can be exact (see _TrackReader); otherwise every piece is
    # sought, in chunks that bound the drift of a rounded rate.
    dub_reader: Optional[_TrackReader] = None
    video_reader: Optional[_TrackReader] = None
    if options.exact:
        native = dub_track.sample_rate if dub_track else None
        if _decode_plan(rate, plan.speed, options.stretch, native)[2]:
            dub_reader = _TrackReader(plan.dub_path, plan.dub_track, rate, channels, plan.speed,
                                      options.stretch, token, native)
        video_reader = _TrackReader(plan.video_path, plan.video_track, rate, channels, 1.0, "resample", token)
    try:
        for index, segment in enumerate(plan.segments):
            if token:
                token.raise_if_cancelled()
            start = int(round(segment.start_s * rate))
            end = int(round(segment.end_s * rate)) if index < len(plan.segments) - 1 else total
            end = min(end, total)
            if end <= start:
                continue
            lead = half if len(held) else 0
            trail = half if end < total else 0
            if end - start < 2 * xfade:
                # Too short to fade at both ends; joined hard instead.
                trail = 0
            length = end - start + lead + trail

            source = int(round(segment.source_start_s * rate)) - lead
            stage = f"writing {os.path.basename(output_path)}"

            def on_chunk(read: int, base: int = written) -> None:
                if progress:
                    progress(int(100 * min(base + read, total) / max(1, total)), stage)

            if segment.kind == "dub":
                if dub_reader is not None:
                    audio = dub_reader.read(source, length, on_chunk)
                else:
                    audio = read_span(
                        plan.dub_path, plan.dub_track, rate, channels, source, length,
                        plan.speed, options.stretch, token, on_chunk,
                    )
            elif video_reader is not None:
                audio = video_reader.read(source, length, on_chunk)
            else:
                audio = read_span(
                    plan.video_path, plan.video_track, rate, channels, source, length,
                    1.0, "resample", token, on_chunk,
                )
            if segment.kind != "dub" and abs(plan.fill_gain_db) > 1e-6:
                audio *= fill_gain

            if lead:
                ramp = min(xfade, length)
                audio[:ramp] *= np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)[:, None]
                overlap = min(len(held), length)
                audio[:overlap] += held[:overlap]
            if trail:
                audio[length - xfade:] *= np.linspace(1.0, 0.0, xfade, endpoint=False, dtype=np.float32)[:, None]
                held = audio[length - xfade:].copy()
                body = audio[: length - xfade]
            else:
                held = np.zeros((0, channels), dtype=np.float32)
                body = audio

            if patch is not None:
                patch.apply(written, body)
            over = np.abs(body) > 1.0
            if over.any():
                clipped += int(over.sum())
                np.clip(body, -1.0, 1.0, out=body)
            written += len(body)
            emit(body)
            if progress:
                progress(int(100 * min(written, total) / max(1, total)), stage)
        if len(held):
            if patch is not None:
                patch.apply(written, held)
            np.clip(held, -1.0, 1.0, out=held)
            written += len(held)
            emit(held)
        if written < total:
            tail = np.zeros((total - written, channels), dtype=np.float32)
            if patch is not None:
                patch.apply(written, tail)
                np.clip(tail, -1.0, 1.0, out=tail)
            emit(tail)
            written = total
        if skip:
            # What was dropped at the head is made up at the tail, so the
            # track is still exactly the video's length once decoded.
            encoder.write(np.zeros((skip, channels), dtype=np.float32))
        encoder.close()
    except BaseException:
        encoder.abort()
        _discard(staging)
        raise
    finally:
        for reader in (dub_reader, video_reader):
            if reader is not None:
                reader.close()

    if not os.path.isfile(staging) or os.path.getsize(staging) == 0:
        _discard(staging)
        raise MediaError(f"ffmpeg reported success but produced no output for {output_path}")
    os.replace(staging, output_path)
    result = RenderResult(
        output_path, rate, channels, written / rate, clipped,
        " ".join(shlex.quote(part) for part in encoder.command),
    )
    if clipped:
        result.warnings.append(
            f"{clipped} samples clipped; the fills are {plan.fill_gain_db:+.1f} dB and the original "
            f"peaks close to full scale -- rerun with a lower --fill-gain if it is audible"
        )
    result.warnings.extend(voice_warnings)
    return result


def _voice_patch(plan: DubSyncPlan, rate: int, channels: int, dub_track, options: RenderOptions,
                 fill_gain: float, token: Optional[CancellationToken], say: Callable[[str], None],
                 warnings: List[str]):
    """The voice pieces' additions (see ``voicefix.build_patch``), or None
    when there are none to make or they cannot be made -- said in
    ``warnings`` when the plan asked for them."""
    if not plan.voice_pieces or not options.fix_voices:
        return None
    from . import voicefix, voicetools

    count = len(plan.voice_pieces)
    moves = f"the plan moves the dub's voices in {count} place{'s' if count != 1 else ''}"
    native = dub_track.sample_rate if dub_track else None
    if not options.exact or not _decode_plan(rate, plan.speed, options.stretch, native)[2]:
        if options.exact:
            warnings.append(f"{moves}, but at this rate and speed the dub cannot be read exactly enough "
                            "to take its voices out; written without those moves")
        return None
    if not voicetools.installed():
        warnings.append(f"{moves}, but the voice tools are not installed; written without those moves")
        return None
    say(f"separating the dub's voices for the {count} voice move{'s' if count != 1 else ''}")
    dub_reader = _TrackReader(plan.dub_path, plan.dub_track, rate, channels, plan.speed,
                              options.stretch, token, native)
    video_reader = _TrackReader(plan.video_path, plan.video_track, rate, channels, 1.0, "resample", token)
    try:
        with voicetools.VoiceWorker(token=token, log=say) as worker:
            return voicefix.build_patch(plan, plan.voice_pieces, worker, dub_reader, video_reader, rate,
                                        fill_gain, options.xfade_s, token, say)
    except MediaError as exc:
        warnings.append(f"{moves}, but the voice tools failed ({exc}); written without those moves")
        return None
    finally:
        dub_reader.close()
        video_reader.close()


def mux(
    plan: DubSyncPlan,
    audio_path: str,
    output_path: Optional[str] = None,
    language: Optional[str] = None,
    title: Optional[str] = None,
    token: Optional[CancellationToken] = None,
    overwrite: bool = False,
) -> str:
    """Add the synced track to a copy of the video, every original stream kept."""
    if output_path is None:
        stem, _ext = os.path.splitext(os.path.basename(plan.video_path))
        output_path = os.path.join(os.path.dirname(os.path.abspath(plan.video_path)), f"{stem}.dubsynced.mkv")
    if os.path.abspath(output_path) == os.path.abspath(plan.video_path):
        raise MediaError("Refusing to overwrite the source video in place")
    if os.path.exists(output_path) and not overwrite:
        raise MediaError(f"Output already exists: {os.path.basename(output_path)}")
    info = probe(plan.video_path, token)
    new_index = len(info.audio_tracks)
    command = [
        ffmpeg_path(), "-nostdin", "-v", "error", "-y",
        "-i", plan.video_path, "-i", audio_path,
        "-map", "0", "-map", "1:a:0", "-c", "copy",
    ]
    if language:
        command.extend([f"-metadata:s:a:{new_index}", f"language={language}"])
    if title:
        command.extend([f"-metadata:s:a:{new_index}", f"title={title}"])
    # Staged and renamed like the track, so a stopped mux never leaves a
    # truncated copy of the video where a good one was.
    stem, ext = os.path.splitext(output_path)
    staging = f"{stem}.part{ext}"
    command.append(staging)
    from .media import _run  # noqa: WPS433 - shared subprocess handling
    try:
        _run(command, ENCODE_TIMEOUT_S, token, what=f"mux {os.path.basename(output_path)}")
    except BaseException:
        _discard(staging)
        raise
    if not os.path.isfile(staging) or os.path.getsize(staging) == 0:
        _discard(staging)
        raise MediaError(f"ffmpeg produced no output for {output_path}")
    os.replace(staging, output_path)
    return output_path


# ---------------------------------------------------------------------------
# A preview of a span: for the editor, to hear a cut before applying it
# ---------------------------------------------------------------------------

# How far a preview may reach: enough to hear a scene, short enough to be
# ready in seconds.
PREVIEW_MAX_S = 120.0


def clip_plan(plan: DubSyncPlan, lo_s: float, hi_s: float) -> DubSyncPlan:
    """The plan restricted to [lo_s, hi_s), on a timeline starting at 0.

    Each piece keeps reading from where it read before -- its source moves
    by exactly what was cut off its front -- so the clipped plan plays the
    same audio the whole one plays across that span."""
    lo_s = max(0.0, lo_s)
    hi_s = min(plan.video_duration_s, hi_s)
    pieces: List[Segment] = []
    for segment in plan.segments:
        start = max(segment.start_s, lo_s)
        end = min(segment.end_s, hi_s)
        if end - start <= 0.0:
            continue
        pieces.append(Segment(
            segment.kind, start - lo_s, end - lo_s,
            segment.source_start_s + (start - segment.start_s),
            segment.offset_s, segment.match, segment.note, segment.uncertainty_s,
        ))
    clipped = DubSyncPlan(
        plan.video_path, plan.dub_path, plan.video_track, plan.dub_track,
        speed=plan.speed, fill_gain_db=plan.fill_gain_db,
        video_duration_s=hi_s - lo_s, dub_duration_s=plan.dub_duration_s,
        video_fps=plan.video_fps, dub_rate=plan.dub_rate, segments=pieces,
        timeline=plan.timeline,
        voice_pieces=[
            VoicePiece(p.dub_start_s, p.dub_end_s, p.level_s + lo_s, p.shift_s, p.join_end, p.note)
            for p in plan.voice_pieces
        ],
    )
    return clipped


EXCERPT_KINDS = ("both", "audio", "picture", "original")
# A cut is started this much before the frame it is meant to start on,
# so the frame is unambiguously on the kept side of the seek point; the
# picture is then this much late against the sound, which is nothing.
FRAME_SEEK_SLACK_S = 0.0005


def first_frame_offset(video_path: str, lo_s: float, token: Optional[CancellationToken] = None) -> float:
    """How far past ``lo_s`` the first picture frame ffmpeg keeps after
    seeking there sits, in seconds: between zero and one frame.

    ffmpeg cuts accurately -- frames before the seek point are decoded
    and dropped -- but it then starts the output's clock at the first
    frame it kept, not at the seek point, so a cut that starts between
    two frames comes out up to a frame early against sound cut at the
    same instant. Knowing the offset, the cut can be started on the frame
    itself instead.
    """
    from .media import _run  # noqa: WPS433 - shared subprocess handling

    # The metadata filter prints only frames that carry metadata, so one
    # entry is added to every frame first; the print goes to stdout.
    command = [
        ffmpeg_path(), "-nostdin", "-v", "error", "-ss", f"{lo_s:.6f}", "-i", video_path,
        "-map", "0:v:0", "-frames:v", "1",
        "-vf", "metadata=mode=add:key=first:value=1,metadata=print:file=-", "-f", "null", "-",
    ]
    out = _run(command, 600, token, what=f"find the first frame of {os.path.basename(video_path)}")
    match = re.search(rb"pts_time:\s*(-?[0-9.]+)", out)
    if not match:
        raise MediaError(f"Could not find a picture frame at {lo_s:.3f}s in {os.path.basename(video_path)}")
    return max(0.0, float(match.group(1)))


def frame_aligned_start(video_path: str, lo_s: float, token: Optional[CancellationToken] = None) -> float:
    """``lo_s`` moved forward onto the first picture frame at or after it
    (less the seek slack), so a cut started there has that frame at its
    time 0 and sound cut at the same instant lines up with it."""
    offset = first_frame_offset(video_path, lo_s, token)
    if offset <= FRAME_SEEK_SLACK_S:
        return lo_s
    return lo_s + offset - FRAME_SEEK_SLACK_S


def excerpt(
    plan: DubSyncPlan,
    lo_s: float,
    hi_s: float,
    output_path: str,
    what: str = "both",
    token: Optional[CancellationToken] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[str, float, float]:
    """Render [lo_s, hi_s) of the plan as a small file to play. Returns the
    path and the span actually cut, which for a picture starts on a frame.

    ``what`` chooses the piece: ``audio`` is the synced track alone as a
    16-bit WAV (what the written file will contain there, to the sample);
    ``original`` is the video's own audio across the span, the same way,
    for A/B against it; ``picture`` is a low-resolution copy of the
    picture with no sound, for a player that carries its own; ``both`` is
    the picture with the synced track under it, for an outside player.

    Every piece is on one clock: its time 0 is the start returned. A cut
    with a picture is moved onto the first frame at or after ``lo_s``
    (see ``first_frame_offset``), so the frame at its time 0 is the frame
    that was there; a player asking for the sound of the same span must
    use the start returned here. No B-frames and a keyframe every half
    second, so a player can step frame by frame without waiting.
    """
    if what not in EXCERPT_KINDS:
        raise MediaError(f"Unknown excerpt {what!r}; choose from {', '.join(EXCERPT_KINDS)}")
    if what in ("picture", "both"):
        if not os.path.isfile(plan.video_path):
            raise MediaError(f"Video not found: {plan.video_path}")
        lo_s = frame_aligned_start(plan.video_path, lo_s, token)
    hi_s = min(hi_s, lo_s + PREVIEW_MAX_S)
    if hi_s - lo_s < 0.25:
        raise MediaError("The preview span is too short")
    root, _ext = os.path.splitext(output_path)
    from .media import _run  # noqa: WPS433 - shared subprocess handling

    if what == "picture":
        if not os.path.isfile(plan.video_path):
            raise MediaError(f"Video not found: {plan.video_path}")
        command = [
            ffmpeg_path(), "-nostdin", "-v", "error", "-y",
            "-ss", f"{lo_s:.6f}", "-t", f"{hi_s - lo_s:.6f}", "-i", plan.video_path,
            "-map", "0:v:0", "-an", "-dn", "-sn", "-map_chapters", "-1",
            "-vf", "scale=-2:'min(480,ih)'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-bf", "0", "-g", "12", "-keyint_min", "12", "-sc_threshold", "0",
            "-movflags", "+faststart", output_path,
        ]
        _run(command, ENCODE_TIMEOUT_S, token, what=f"preview {os.path.basename(output_path)}")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise MediaError("ffmpeg produced no preview")
        return output_path, lo_s, hi_s

    if what == "original":
        # The original across the span, at the dub's rate and channel count
        # so it mixes with the synced excerpt as it is: one fill, no gain.
        source = DubSyncPlan(
            plan.video_path, plan.dub_path, plan.video_track, plan.dub_track,
            speed=1.0, fill_gain_db=0.0, video_duration_s=hi_s - lo_s, dub_duration_s=plan.dub_duration_s,
            segments=[Segment("fill", 0.0, hi_s - lo_s, lo_s)],
        )
        wav_path = root + ".wav"
        render(source, wav_path, RenderOptions(codec="wav16", exact=False), token=token, log=log)
        return wav_path, lo_s, hi_s

    clipped = clip_plan(plan, lo_s, hi_s)
    if not clipped.segments:
        raise MediaError("Nothing to preview in that span")
    wav_path = root + ".wav"
    render(clipped, wav_path, RenderOptions(codec="wav16", exact=False), token=token, log=log)
    if what == "audio":
        return wav_path, lo_s, hi_s
    command = [
        ffmpeg_path(), "-nostdin", "-v", "error", "-y",
        "-ss", f"{lo_s:.6f}", "-t", f"{hi_s - lo_s:.6f}", "-i", plan.video_path,
        "-i", wav_path,
        "-map", "0:v:0", "-map", "1:a:0", "-dn", "-sn", "-map_chapters", "-1",
        "-vf", "scale=-2:'min(480,ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", output_path,
    ]
    _run(command, ENCODE_TIMEOUT_S, token, what=f"preview {os.path.basename(output_path)}")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise MediaError("ffmpeg produced no preview")
    try:
        os.remove(wav_path)
    except OSError:
        pass
    return output_path, lo_s, hi_s


def preview_span(
    plan: DubSyncPlan,
    lo_s: float,
    hi_s: float,
    output_path: str,
    with_video: bool = True,
    token: Optional[CancellationToken] = None,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Render [lo_s, hi_s) of the plan as a small file to play: the synced
    audio alone as a WAV, or laid under a low-resolution copy of the
    picture as an MP4. See ``excerpt``."""
    path, _lo, _hi = excerpt(plan, lo_s, hi_s, output_path, "both" if with_video else "audio", token, log)
    return path
