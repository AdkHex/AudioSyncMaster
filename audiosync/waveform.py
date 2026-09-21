"""Waveforms for the app's dub sync view.

The app draws the video's own audio and the dub laid under it the way an
audio editor does: per channel, the highest and lowest sample in each
pixel's span as a filled shape, with the RMS inside it in a lighter tone.
That is what a person reads alignment from -- a transient is a spike that
lands in the same pixel on both tracks when they are in sync -- and it
must hold at every zoom, from the whole film on one screen to a few
frames across it.

Reading two hours of 48 kHz stereo at every scroll is out of the question,
so each file is decoded once, at its own rate, into blocks of 256 samples
(5.3 ms at 48 kHz): the minimum, maximum and mean square of each block
per channel, 32 MB for a two-hour stereo track. A view wider than a block
per pixel is served from those; a view narrower than that decodes exactly
the span in sight, which is at most a few seconds. Multichannel tracks are
shown as the stereo downmix ffmpeg makes of them: six lanes would say
nothing more about where a cut is.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .dubrender import read_span
from .media import CancellationToken, probe, stream_audio

# Samples per block of the cache.
BLOCK = 256
# Two tracks per pair, and a pair or two being looked at.
CACHE_SIZE = 4
# Decode in half-minute pieces: a few megabytes each at 48 kHz stereo.
DECODE_BLOCK_S = 30.0

ProgressFn = Callable[[float], None]


@dataclass
class Waveform:
    """One track's peaks, per channel, in blocks of BLOCK samples."""

    path: str
    track: int
    sample_rate: int
    channels: int
    frames: int
    """Samples decoded, in the file's own time."""
    mins: np.ndarray
    """(channels, blocks) float32."""
    maxs: np.ndarray
    meansq: np.ndarray

    @property
    def duration_s(self) -> float:
        return self.frames / float(self.sample_rate)

    @property
    def blocks(self) -> int:
        return int(self.mins.shape[1])


_cache: "OrderedDict[Tuple[str, int, float, int], Waveform]" = OrderedDict()
_lock = threading.Lock()


def _key(path: str, track: int) -> Tuple[str, int, float, int]:
    """The file as it is now: a rewritten track must not show stale peaks."""
    try:
        stat = os.stat(path)
        return (os.path.abspath(path), int(track), stat.st_mtime, stat.st_size)
    except OSError:
        return (os.path.abspath(path), int(track), 0.0, 0)


def build(
    path: str,
    track: int = 0,
    token: Optional[CancellationToken] = None,
    progress: Optional[ProgressFn] = None,
) -> Waveform:
    """Decode a track once into block peaks. Blocking; minutes for a film
    only if the decoder is slow, seconds as a rule."""
    info = probe(path, token)
    stream = info.audio_tracks[track] if track < len(info.audio_tracks) else None
    rate = int((stream.sample_rate if stream else None) or info.sample_rate or 48000)
    source_channels = int((stream.channels if stream else None) or info.channels or 2)
    channels = 1 if source_channels == 1 else 2
    expected = max(1.0, float(info.duration or 0.0)) * rate

    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    meansq: List[np.ndarray] = []
    remainder = np.zeros((0, channels), dtype=np.float32)
    frames = 0

    def fold(samples: np.ndarray) -> None:
        # (n * BLOCK, channels) -> one row of peaks per block.
        shaped = samples.reshape(-1, BLOCK, channels)
        mins.append(shaped.min(axis=1))
        maxs.append(shaped.max(axis=1))
        meansq.append(np.mean(np.square(shaped, dtype=np.float32), axis=1))

    for piece in stream_audio(path, rate, track=track, channels=channels, token=token, block_s=DECODE_BLOCK_S):
        if piece.ndim == 1:
            piece = piece.reshape(-1, 1)
        frames += len(piece)
        if len(remainder):
            piece = np.concatenate([remainder, piece])
        usable = (len(piece) // BLOCK) * BLOCK
        if usable:
            fold(piece[:usable])
        remainder = piece[usable:]
        if progress:
            progress(min(1.0, frames / expected))
    if len(remainder):
        padded = np.zeros((BLOCK, channels), dtype=np.float32)
        padded[: len(remainder)] = remainder
        fold(padded)
    if progress:
        progress(1.0)

    if mins:
        stack = lambda rows: np.concatenate(rows).T.astype(np.float32, copy=False)  # noqa: E731
        return Waveform(path, track, rate, channels, frames, stack(mins), stack(maxs), stack(meansq))
    empty = np.zeros((channels, 0), dtype=np.float32)
    return Waveform(path, track, rate, channels, 0, empty, empty.copy(), empty.copy())


def load(
    path: str,
    track: int = 0,
    token: Optional[CancellationToken] = None,
    progress: Optional[ProgressFn] = None,
) -> Waveform:
    """The track's peaks, decoded once and kept."""
    key = _key(path, track)
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
    built = build(path, track, token, progress)
    with _lock:
        _cache[key] = built
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return built


def is_loaded(path: str, track: int = 0) -> bool:
    with _lock:
        return _key(path, track) in _cache


def forget(path: Optional[str] = None) -> None:
    """Drop cached waveforms -- all of them, or one file's."""
    with _lock:
        for key in list(_cache):
            if path is None or key[0] == os.path.abspath(path):
                del _cache[key]


def _reduce_spans(values: np.ndarray, starts: np.ndarray, ends: np.ndarray, ufunc) -> np.ndarray:
    """``ufunc`` reduced over values[starts[i]:ends[i]] for every i, zero
    where the span is empty or lies outside the values.

    One reduceat call: the spans are interleaved as [s0, e0, s1, e1, ...],
    which makes reduceat produce each span's reduction at the even
    positions and the gaps (or, for overlapping spans, a single element)
    at the odd ones, which are dropped. One padding element keeps every
    index in range, since a span may end at len(values)."""
    n = len(values)
    out = np.zeros(len(starts), dtype=np.float32)
    if n == 0:
        return out
    s = np.clip(starts, 0, n)
    e = np.clip(ends, 0, n)
    have = e > s
    if not have.any():
        return out
    idx = np.empty(2 * int(have.sum()), dtype=np.int64)
    idx[0::2] = s[have]
    idx[1::2] = e[have]
    padded = np.append(values, values.dtype.type(0))
    out[have] = ufunc.reduceat(padded, idx)[0::2]
    return out


def _peaks_over(
    mins: np.ndarray, maxs: np.ndarray, squares: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> Dict[str, List[List[float]]]:
    """Per channel, min/max/rms of each [starts[i], ends[i]) span of rows
    that are themselves minima, maxima and mean squares (or the samples
    three times over)."""
    result: Dict[str, List[List[float]]] = {"min": [], "max": [], "rms": []}
    counts = np.maximum(1, np.clip(ends, 0, None) - np.clip(starts, 0, None)).astype(np.float32)
    for ch in range(mins.shape[0]):
        result["min"].append(_round(_reduce_spans(mins[ch], starts, ends, np.minimum)))
        result["max"].append(_round(_reduce_spans(maxs[ch], starts, ends, np.maximum)))
        sums = _reduce_spans(squares[ch], starts, ends, np.add)
        result["rms"].append(_round(np.sqrt(sums / counts)))
    return result


def _spans(edges: np.ndarray, rows: int, limit: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Bucket edges (float, in rows) to [start, end) row spans, at least one
    row wide, empty where the bucket lies wholly outside [0, limit).

    Edges are rounded to the nearest row, so neighbouring buckets never
    share a row and a bucket reaches at most half a row past its edge.
    ``limit`` is where the material really ends, in rows, when the last
    row is only partly filled: a bucket starting past it shows nothing."""
    end_of_material = float(rows) if limit is None else limit
    starts = np.round(edges[:-1]).astype(np.int64)
    ends = np.maximum(starts + 1, np.round(edges[1:]).astype(np.int64))
    inside = (edges[1:] > 0) & (edges[:-1] < end_of_material)
    starts = np.clip(starts, 0, rows)
    ends = np.where(inside, np.clip(ends, 0, rows), starts)
    return starts, ends


def _round(values: np.ndarray) -> List[float]:
    return [round(float(v), 4) for v in values]


def peaks(
    path: str,
    track: int,
    start_s: float,
    end_s: float,
    buckets: int,
    speed: float = 1.0,
    token: Optional[CancellationToken] = None,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, object]:
    """Per channel, the lowest and highest sample and the RMS in each of
    ``buckets`` equal spans of [start_s, end_s), on the timeline the plan
    uses for the file.

    A dub the plan plays at ``speed`` sits on the video's clock, where
    its own second ``t`` falls at ``t * speed``; the span is given on
    that clock and read from the file at ``t / speed``. Beyond the file's
    ends the values are zero.
    """
    wave = load(path, track, token, progress)
    buckets = max(1, int(buckets))
    speed = max(speed, 1e-9)
    span_s = max(0.0, end_s - start_s)
    per_bucket = span_s / buckets / speed * wave.sample_rate  # file samples per bucket

    if per_bucket >= BLOCK or wave.blocks == 0 or span_s <= 0:
        edges = np.linspace(start_s, end_s, buckets + 1) / speed * wave.sample_rate / BLOCK
        starts, ends = _spans(edges, wave.blocks, wave.frames / BLOCK)
        result = _peaks_over(wave.mins, wave.maxs, wave.meansq, starts, ends)
    else:
        # Finer than the cache: decode exactly what is in sight.
        start_sample = int(round(start_s * wave.sample_rate))
        count = max(1, int(round(span_s * wave.sample_rate)))
        samples = read_span(path, track, wave.sample_rate, wave.channels, start_sample, count, speed, token=token)
        columns = samples.T.astype(np.float32, copy=False)
        starts, ends = _spans(np.linspace(0, count, buckets + 1), count)
        result = _peaks_over(columns, columns, np.square(columns), starts, ends)

    return {
        "startS": float(start_s),
        "endS": float(end_s),
        "buckets": buckets,
        "channels": wave.channels,
        "sampleRate": wave.sample_rate,
        "durationS": wave.duration_s * speed,
        **result,
    }
