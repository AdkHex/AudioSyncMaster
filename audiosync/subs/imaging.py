"""A small image toolkit for bitmap subtitles, numpy only.

The shipped engine is built with numpy and nothing else (no Pillow), yet OCR
has to read and write PNGs, enlarge subtitle lines, and split fill from
outline. Everything here works on plain ``uint8`` arrays: ``HxW`` grey,
``HxWx3`` RGB or ``HxWx4`` RGBA with straight (not premultiplied) alpha,
which is what PGS and VobSub palettes describe.

PNG support covers what real tools write: 1/2/4/8/16-bit grey, grey+alpha,
RGB, RGBA and palette images with ``tRNS``, every filter type. Adam7
interlacing is rejected with a clear error; no subtitle tool produces it.
"""

from __future__ import annotations

import struct
import zlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Rec. 601 luma weights. Subtitle fills are judged against their outline in
#: the same image, so the exact matrix does not matter; 601 matches what most
#: OCR engines assume when they convert to grey themselves.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class ImageError(ValueError):
    """An image could not be decoded (corrupt or unsupported PNG)."""


# --------------------------------------------------------------------- PNG


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def encode_png(image: np.ndarray, compress_level: int = 6) -> bytes:
    """8-bit PNG of a grey, grey+alpha, RGB or RGBA array.

    Rows are written with filter 0 (none): subtitle images are mostly flat
    runs that zlib already compresses well, and skipping the filter search
    keeps encoding fast in pure numpy.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 2, 3, 4):
        raise ImageError(f"Cannot encode an image of shape {arr.shape}")
    height, width, channels = arr.shape
    if height == 0 or width == 0:
        raise ImageError("Cannot encode an empty image")
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    raw = np.empty((height, 1 + width * channels), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = arr.reshape(height, width * channels)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"".join(
        (
            PNG_SIGNATURE,
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(raw.tobytes(), compress_level)),
            _chunk(b"IEND", b""),
        )
    )


def write_png(path: str, image: np.ndarray) -> str:
    with open(path, "wb") as handle:
        handle.write(encode_png(image))
    return path


def read_png(path: str) -> np.ndarray:
    """RGBA ``uint8`` array of a PNG file."""
    with open(path, "rb") as handle:
        return decode_png(handle.read())


def _paeth_row(line: bytes, prev: bytes, bpp: int) -> bytearray:
    out = bytearray(line)
    for i in range(len(out)):
        a = out[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            pred = a
        elif pb <= pc:
            pred = b
        else:
            pred = c
        out[i] = (out[i] + pred) & 0xFF
    return out


def _average_row(line: bytes, prev: bytes, bpp: int) -> bytearray:
    out = bytearray(line)
    for i in range(len(out)):
        a = out[i - bpp] if i >= bpp else 0
        out[i] = (out[i] + ((a + prev[i]) >> 1)) & 0xFF
    return out


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> np.ndarray:
    """Undo PNG's per-row filters.

    None/Sub/Up vectorise; Average and Paeth depend on the pixel just
    reconstructed to their left, so they run as a Python loop per row. Those
    rows are rare in subtitle-sized images, and correctness beats speed here.
    """
    if len(raw) < height * (stride + 1):
        raise ImageError("PNG data is truncated")
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(height):
        kind = raw[pos]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos + 1)
        pos += stride + 1
        if kind == 0:
            cur = line.copy()
        elif kind == 1:
            cur = (np.cumsum(line.reshape(-1, bpp).astype(np.int64), axis=0) & 0xFF).astype(np.uint8).reshape(-1)
        elif kind == 2:
            cur = line + prev
        elif kind == 3:
            cur = np.frombuffer(_average_row(line.tobytes(), prev.tobytes(), bpp), dtype=np.uint8)
        elif kind == 4:
            cur = np.frombuffer(_paeth_row(line.tobytes(), prev.tobytes(), bpp), dtype=np.uint8)
        else:
            raise ImageError(f"Unknown PNG filter type {kind}")
        out[y] = cur
        prev = out[y]
    return out


def decode_png(data: bytes) -> np.ndarray:
    """Decode a PNG to an ``HxWx4`` RGBA ``uint8`` array."""
    if not data.startswith(PNG_SIGNATURE):
        raise ImageError("Not a PNG file")
    pos = len(PNG_SIGNATURE)
    header = None
    palette: Optional[np.ndarray] = None
    transparency: Optional[bytes] = None
    idat: List[bytes] = []
    while pos + 8 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, pos)
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload[:13])
        elif kind == b"PLTE":
            palette = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        elif kind == b"tRNS":
            transparency = payload
        elif kind == b"IDAT":
            idat.append(payload)
        elif kind == b"IEND":
            break
    if header is None:
        raise ImageError("PNG has no header")
    width, height, depth, color_type, _compression, _filter, interlace = header
    if interlace:
        raise ImageError("Interlaced PNGs are not supported")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None or depth not in (1, 2, 4, 8, 16):
        raise ImageError(f"Unsupported PNG (colour type {color_type}, depth {depth})")
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise ImageError(f"Corrupt PNG data: {exc}") from exc
    bits = depth * channels
    stride = (width * bits + 7) // 8
    bpp = max(1, bits // 8)
    rows = _unfilter(raw, height, stride, bpp)

    if depth == 16:
        samples = rows.reshape(height, width * channels, 2)[:, :, 0]
    elif depth == 8:
        samples = rows[:, : width * channels]
    else:
        # Sub-byte samples: unpack bits, then regroup per sample.
        unpacked = np.unpackbits(rows, axis=1)[:, : width * depth]
        weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint8)
        samples = (unpacked.reshape(height, width, depth) * weights).sum(axis=2).astype(np.uint8)
        if color_type == 0:
            samples = (samples.astype(np.uint16) * 255 // ((1 << depth) - 1)).astype(np.uint8)
    samples = samples.reshape(height, width, channels)

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    if color_type == 3:
        if palette is None:
            raise ImageError("Palette PNG without a palette")
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[: len(palette), :3] = palette
        lut[:, 3] = 255
        if transparency:
            alpha = np.frombuffer(transparency, dtype=np.uint8)[:256]
            lut[: len(alpha), 3] = alpha
        return lut[samples[:, :, 0]]
    if color_type in (0, 4):
        rgba[:, :, :3] = samples[:, :, :1]
        rgba[:, :, 3] = samples[:, :, 1] if color_type == 4 else 255
    else:
        rgba[:, :, :3] = samples[:, :, :3]
        rgba[:, :, 3] = samples[:, :, 3] if color_type == 6 else 255
    return rgba


# ------------------------------------------------------------- resampling


def _bilinear_axis(arr: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    n_in = arr.shape[axis]
    if n_in == n_out:
        return arr
    centres = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    centres = np.clip(centres, 0, n_in - 1)
    lo = np.floor(centres).astype(np.intp)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = (centres - lo).astype(np.float32)
    shape = [1] * arr.ndim
    shape[axis] = n_out
    frac = frac.reshape(shape)
    a = np.take(arr, lo, axis=axis)
    b = np.take(arr, hi, axis=axis)
    return a * (1.0 - frac) + b * frac


def _area_axis(arr: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Exact box-filter shrink: each output pixel averages the source span it
    covers, fractional edges included. Bilinear would alias text strokes."""
    n_in = arr.shape[axis]
    if n_in == n_out:
        return arr
    moved = np.moveaxis(arr, axis, 0).astype(np.float64)
    cumulative = np.concatenate([np.zeros((1,) + moved.shape[1:]), np.cumsum(moved, axis=0)], axis=0)
    edges = np.linspace(0.0, n_in, n_out + 1)
    lo = np.minimum(np.floor(edges).astype(np.intp), n_in - 1)
    frac = (edges - lo).reshape((-1,) + (1,) * (moved.ndim - 1))
    integral = cumulative[lo] + frac * moved[lo]
    out = (integral[1:] - integral[:-1]) / (n_in / n_out)
    return np.moveaxis(out.astype(np.float32), 0, axis)


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with bilinear interpolation when enlarging and area averaging
    when shrinking, per axis. Keeps the input dtype."""
    arr = np.asarray(image)
    width, height = max(1, int(width)), max(1, int(height))
    work = arr.astype(np.float32)
    for axis, n_out in ((0, height), (1, width)):
        n_in = work.shape[axis]
        work = _bilinear_axis(work, n_out, axis) if n_out >= n_in else _area_axis(work, n_out, axis)
    if arr.dtype == np.uint8:
        return np.clip(np.rint(work), 0, 255).astype(np.uint8)
    return work.astype(arr.dtype)


def scale(image: np.ndarray, factor: float) -> np.ndarray:
    arr = np.asarray(image)
    if abs(factor - 1.0) < 1e-6:
        return arr.copy()
    h, w = arr.shape[:2]
    return resize(arr, int(round(w * factor)), int(round(h * factor)))


def pad(image: np.ndarray, border: int, value: int = 255) -> np.ndarray:
    arr = np.asarray(image)
    widths = [(border, border), (border, border)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, widths, mode="constant", constant_values=value)


# ------------------------------------------------------------ measurements


def luma(rgba: np.ndarray) -> np.ndarray:
    """Luma 0..255 (float32) of an RGB/RGBA array, ignoring alpha."""
    return np.asarray(rgba)[..., :3].astype(np.float32) @ LUMA_WEIGHTS


def otsu_threshold(values: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Otsu's threshold over 0..255 values; returns (threshold, separation).

    ``separation`` is the between-class variance divided by the total
    variance (0..1): near 1 for two clean tones such as a white fill and a
    black outline, near 0 when there is really only one tone.
    """
    vals = np.clip(np.asarray(values, dtype=np.float64).ravel(), 0, 255)
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    hist = np.bincount(np.rint(vals).astype(np.intp), weights=w, minlength=256)[:256]
    total = hist.sum()
    if total <= 0:
        return 128.0, 0.0
    levels = np.arange(256, dtype=np.float64)
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_t = mu[-1]
    variance = float((p * (levels - mu_t) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    between[~np.isfinite(between)] = 0.0
    k = int(np.argmax(between))
    separation = float(between[k] / variance) if variance > 0 else 0.0
    return k + 0.5, separation


def runs(profile: np.ndarray, min_value: float = 0.0) -> List[Tuple[int, int]]:
    """Half-open ``(start, end)`` spans where ``profile > min_value``."""
    on = np.asarray(profile) > min_value
    if not on.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.view(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """``(top, bottom, left, right)`` half-open box of the true pixels."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def flatten(rgba: np.ndarray, background: Sequence[int] = (96, 96, 96)) -> np.ndarray:
    """Composite RGBA over a solid colour -> RGB, for review images that must
    show white text with a black outline legibly."""
    arr = np.asarray(rgba).astype(np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(np.rint(arr[:, :, :3] * alpha + bg * (1.0 - alpha)), 0, 255).astype(np.uint8)


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Square dilation of a boolean mask by ``radius`` pixels."""
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ys = slice(max(0, dy), h + min(0, dy))
            yd = slice(max(0, -dy), h + min(0, -dy))
            xs = slice(max(0, dx), w + min(0, dx))
            xd = slice(max(0, -dx), w + min(0, -dx))
            out[yd, xd] |= mask[ys, xs]
    return out
