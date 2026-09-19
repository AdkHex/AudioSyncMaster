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

import os
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

import numpy as np

from .codecdelay import codec_delay_ms
from .dubsync import DubSyncPlan
from .media import (
    SEEK_PREROLL_S,
    CancellationToken,
    Cancelled,
    MediaError,
    _popen_kwargs,
    _terminate,
    ffmpeg_path,
    probe,
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
        spec = CODECS[codec]
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
    if options.codec not in CODECS:
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
    say(
        f"writing {channels} channel(s) at {rate} Hz as {options.codec}, "
        f"{len(plan.dub_segments)} dub piece(s) and {len(plan.fill_segments)} fill(s), "
        f"{options.xfade_s * 1000:.0f}ms crossfades"
    )

    encoder = _Encoder(output_path, rate, channels, options.codec, options.bitrate, token)
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
                audio = read_span(
                    plan.dub_path, plan.dub_track, rate, channels, source, length,
                    plan.speed, options.stretch, token, on_chunk,
                )
            else:
                audio = read_span(
                    plan.video_path, plan.video_track, rate, channels, source, length,
                    1.0, "resample", token, on_chunk,
                )
                if abs(plan.fill_gain_db) > 1e-6:
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

            over = np.abs(body) > 1.0
            if over.any():
                clipped += int(over.sum())
                np.clip(body, -1.0, 1.0, out=body)
            written += len(body)
            emit(body)
            if progress:
                progress(int(100 * min(written, total) / max(1, total)), stage)
        if len(held):
            np.clip(held, -1.0, 1.0, out=held)
            written += len(held)
            emit(held)
        if written < total:
            emit(np.zeros((total - written, channels), dtype=np.float32))
            written = total
        if skip:
            # What was dropped at the head is made up at the tail, so the
            # track is still exactly the video's length once decoded.
            encoder.write(np.zeros((skip, channels), dtype=np.float32))
        encoder.close()
    except BaseException:
        encoder.abort()
        raise

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise MediaError(f"ffmpeg reported success but produced no output for {output_path}")
    result = RenderResult(
        output_path, rate, channels, written / rate, clipped,
        " ".join(shlex.quote(part) for part in encoder.command),
    )
    if clipped:
        result.warnings.append(
            f"{clipped} samples clipped; the fills are {plan.fill_gain_db:+.1f} dB and the original "
            f"peaks close to full scale -- rerun with a lower --fill-gain if it is audible"
        )
    return result


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
    command.append(output_path)
    from .media import _run  # noqa: WPS433 - shared subprocess handling
    _run(command, ENCODE_TIMEOUT_S, token, what=f"mux {os.path.basename(output_path)}")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise MediaError(f"ffmpeg produced no output for {output_path}")
    return output_path
