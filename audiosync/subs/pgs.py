"""HDMV PGS (Blu-ray ``.sup``): parse, render, rewrite palettes, encode.

A PGS stream is a sequence of *segments*, each with a 13-byte header::

    "PG"  PTS(4, 90 kHz)  DTS(4)  type(1)  size(2)

grouped into *display sets* that start with a Presentation Composition
Segment (PCS) and end with an END segment. Between them come window (WDS),
palette (PDS) and object (ODS) definitions. What is on screen is decided by
the PCS alone: it lists which objects to show where, with which palette.

Two properties shape this module:

* **State persists within an epoch.** Objects and palettes defined once stay
  usable until the next *epoch start*. A later display set may only change
  the palette (fades), only reposition, or re-send everything unchanged
  (*acquisition points*, so a player that seeks mid-stream can decode). So
  the parser keeps the epoch's objects, palettes and windows, and a display
  set is rendered from that state, not from its own segments.
* **An event is what stays on screen unchanged.** A subtitle starts at a PCS
  that shows it and ends at the next PCS that shows something else (or
  nothing). An acquisition point that re-sends the same picture does not end
  it. Objects far apart vertically (a sign at the top while dialogue runs at
  the bottom) become separate events, so each gets its own cue, timing and
  alignment.

Colours are limited-range YCbCr: BT.709 for HD, BT.601 for SD (576 lines or
fewer), matching what Blu-ray authoring tools and players use.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

PTS_HZ = 90000

SEG_PDS = 0x14
SEG_ODS = 0x15
SEG_PCS = 0x16
SEG_WDS = 0x17
SEG_END = 0x80

STATE_NORMAL = 0x00
STATE_ACQUISITION_POINT = 0x40
STATE_EPOCH_START = 0x80

OBJECT_CROPPED = 0x80
OBJECT_FORCED = 0x40

ODS_FIRST = 0x80
ODS_LAST = 0x40

#: Largest segment payload (the size field is 16 bits).
MAX_SEGMENT = 0xFFFF
#: Longest run one RLE code can express.
MAX_RUN = 0x3FFF
#: Frame-rate byte in the PCS. Players ignore it; 0x10 is what every
#: authoring tool writes regardless of the actual rate.
FRAME_RATE_CODE = 0x10

#: A stream that ends while something is still shown (no clearing display
#: set) gets this duration for its last events rather than none at all.
OPEN_EVENT_DURATION_S = 5.0

#: Objects whose vertical gap is at most this many times the taller one's
#: height belong to the same caption (two lines authored as two objects).
#: Anything further apart -- top sign plus bottom dialogue -- stays separate.
GROUP_GAP_FACTOR = 1.0


@dataclass
class BitmapEvent:
    """One picture on screen for one interval: a PGS or VobSub caption.

    ``rgba`` is straight-alpha ``HxWx4 uint8``; ``x``/``y`` is its top-left
    corner on a ``video_size`` (width, height) frame.
    """

    start: float
    end: float
    rgba: np.ndarray
    x: int
    y: int
    forced: bool = False
    video_size: Tuple[int, int] = (1920, 1080)

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])

    @property
    def duration(self) -> float:
        return self.end - self.start


# ------------------------------------------------------------ colour space

#: (Kr, Kb) of each matrix.
_MATRICES = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


def matrix_for_height(height: int) -> str:
    return "bt601" if height <= 576 else "bt709"


def ycbcr_to_rgb(y, cb, cr, matrix: str = "bt709") -> np.ndarray:
    """Limited-range YCbCr (0..255 codes) -> RGB ``uint8``; broadcasts."""
    kr, kb = _MATRICES[matrix]
    kg = 1.0 - kr - kb
    luma = (np.asarray(y, dtype=np.float64) - 16.0) * (255.0 / 219.0)
    pb = (np.asarray(cb, dtype=np.float64) - 128.0) * (255.0 / 224.0)
    pr = (np.asarray(cr, dtype=np.float64) - 128.0) * (255.0 / 224.0)
    r = luma + 2.0 * (1.0 - kr) * pr
    b = luma + 2.0 * (1.0 - kb) * pb
    g = (luma - kr * r - kb * b) / kg
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def rgb_to_ycbcr(rgb: np.ndarray, matrix: str = "bt709") -> np.ndarray:
    """RGB ``uint8`` (..., 3) -> limited-range (Y, Cb, Cr) ``uint8``."""
    kr, kb = _MATRICES[matrix]
    kg = 1.0 - kr - kb
    arr = np.asarray(rgb, dtype=np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = kr * r + kg * g + kb * b
    y = 16.0 + luma * (219.0 / 255.0)
    cb = 128.0 + (b - luma) / (2.0 * (1.0 - kb)) * (224.0 / 255.0)
    cr = 128.0 + (r - luma) / (2.0 * (1.0 - kr)) * (224.0 / 255.0)
    return np.clip(np.rint(np.stack([y, cb, cr], axis=-1)), 0, 255).astype(np.uint8)


# -------------------------------------------------------------------- RLE


def rle_decode(data: bytes, width: int, height: int) -> np.ndarray:
    """PGS run-length data -> ``height x width`` palette indices.

    Codes: a non-zero byte is one pixel of that colour; ``00`` escapes to
    ``00 00`` end of line, ``00 0L`` L zeros, ``00 4L LL`` long zeros,
    ``00 8L CC`` L of colour C, ``00 CL LL CC`` long run of C. Literal pixels
    are copied a whole stretch at a time (bytes.find to the next escape), so
    the Python loop runs per run, not per byte.
    """
    data = bytes(data)
    out = np.zeros((height, width), dtype=np.uint8)
    row = bytearray()
    y = 0
    pos = 0
    n = len(data)
    find = data.find

    def store() -> None:
        count = min(len(row), width)
        if count:
            out[y, :count] = np.frombuffer(bytes(row[:count]), dtype=np.uint8)

    while pos < n and y < height:
        escape = find(b"\x00", pos)
        if escape < 0:
            escape = n
        if escape > pos:
            row += data[pos:escape]
            pos = escape
        if pos + 1 >= n:
            break
        code = data[pos + 1]
        if code == 0:
            store()
            y += 1
            row = bytearray()
            pos += 2
            continue
        flag = code & 0xC0
        if flag == 0x00:
            row += bytes(code & 0x3F)
            pos += 2
        elif flag == 0x40:
            if pos + 2 >= n:
                break
            row += bytes(((code & 0x3F) << 8) | data[pos + 2])
            pos += 3
        elif flag == 0x80:
            if pos + 2 >= n:
                break
            row += bytes((data[pos + 2],)) * (code & 0x3F)
            pos += 3
        else:
            if pos + 3 >= n:
                break
            row += bytes((data[pos + 3],)) * (((code & 0x3F) << 8) | data[pos + 2])
            pos += 4
    if row and y < height:
        store()
    return out


def rle_encode(indices: np.ndarray) -> bytes:
    """``HxW`` palette indices -> PGS run-length data.

    Colour 0 gets the compact zero-run codes, so callers put the transparent
    background at index 0. One or two pixels of another colour are written
    literally (cheaper than a 3-byte run code).
    """
    arr = np.asarray(indices, dtype=np.uint8)
    height, width = arr.shape
    out = bytearray()
    for row in arr:
        change = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], change))
        lengths = np.diff(np.concatenate((starts, [width])))
        for colour, length in zip(row[starts].tolist(), lengths.tolist()):
            while length > 0:
                run = min(length, MAX_RUN)
                if colour == 0:
                    if run < 64:
                        out += bytes((0, run))
                    else:
                        out += bytes((0, 0x40 | (run >> 8), run & 0xFF))
                elif run < 3:
                    out += bytes((colour,)) * run
                elif run < 64:
                    out += bytes((0, 0x80 | run, colour))
                else:
                    out += bytes((0, 0xC0 | (run >> 8), run & 0xFF, colour))
                length -= run
        out += b"\x00\x00"
    return bytes(out)


# --------------------------------------------------------------- segments


@dataclass
class Segment:
    offset: int
    pts: int
    dts: int
    kind: int
    payload: int
    size: int


def iter_segments(data: bytes) -> Iterator[Segment]:
    """Every well-formed segment; resynchronises on the next ``PG`` after
    junk, and stops at a truncated tail."""
    pos = 0
    n = len(data)
    while pos + 13 <= n:
        if data[pos : pos + 2] != b"PG":
            nxt = data.find(b"PG", pos + 1)
            if nxt < 0:
                return
            pos = nxt
            continue
        pts, dts, kind, size = struct.unpack_from(">IIBH", data, pos + 2)
        if pos + 13 + size > n:
            return
        yield Segment(pos, pts, dts, kind, pos + 13, size)
        pos += 13 + size


@dataclass
class _CompositionObject:
    object_id: int
    window_id: int
    x: int
    y: int
    forced: bool
    crop: Optional[Tuple[int, int, int, int]] = None


@dataclass
class _Composition:
    pts: int
    width: int
    height: int
    number: int
    state: int
    palette_update: bool
    palette_id: int
    objects: List[_CompositionObject] = field(default_factory=list)


def _parse_pcs(data: bytes, seg: Segment) -> Optional[_Composition]:
    if seg.size < 11:
        return None
    p = seg.payload
    width, height, _rate, number, state, update, palette_id, count = struct.unpack_from(">HHBHBBBB", data, p)
    comp = _Composition(seg.pts, width, height, number, state, bool(update & 0x80), palette_id)
    q = p + 11
    end = p + seg.size
    for _ in range(count):
        if q + 8 > end:
            break
        object_id, window_id, flags, x, y = struct.unpack_from(">HBBHH", data, q)
        q += 8
        crop = None
        if flags & OBJECT_CROPPED:
            if q + 8 > end:
                break
            crop = struct.unpack_from(">HHHH", data, q)
            q += 8
        comp.objects.append(_CompositionObject(object_id, window_id, x, y, bool(flags & OBJECT_FORCED), crop))
    return comp


@dataclass
class _Object:
    version: int
    width: int
    height: int
    data: bytearray
    complete: bool


@dataclass
class _Placed:
    rgba: np.ndarray
    x: int
    y: int
    forced: bool

    @property
    def bottom(self) -> int:
        return self.y + self.rgba.shape[0]

    @property
    def right(self) -> int:
        return self.x + self.rgba.shape[1]


class _Epoch:
    """Objects, palettes and windows defined since the last epoch start."""

    def __init__(self) -> None:
        self.objects: Dict[int, _Object] = {}
        self.palettes: Dict[int, np.ndarray] = {}
        self.windows: Dict[int, Tuple[int, int, int, int]] = {}
        self.decoded: Dict[bytes, np.ndarray] = {}

    def palette_rgba(self, palette_id: int, matrix: str) -> np.ndarray:
        """256x4 RGBA lookup table; undefined entries are transparent."""
        entries = self.palettes.get(palette_id)
        lut = np.zeros((256, 4), dtype=np.uint8)
        if entries is None:
            return lut
        lut[:, :3] = ycbcr_to_rgb(entries[:, 0], entries[:, 1], entries[:, 2], matrix)
        lut[:, 3] = entries[:, 3]
        lut[lut[:, 3] == 0] = 0
        return lut

    def indices(self, obj: _Object) -> np.ndarray:
        """Decoded object, cached by content: acquisition points re-send the
        same bitmap, and decoding is the expensive part of reading a .sup."""
        key = hashlib.sha1(bytes(obj.data)).digest() + struct.pack(">HH", obj.width, obj.height)
        cached = self.decoded.get(key)
        if cached is None:
            cached = rle_decode(bytes(obj.data), obj.width, obj.height)
            self.decoded[key] = cached
        return cached


def _render(comp: _Composition, epoch: _Epoch) -> List[_Placed]:
    matrix = matrix_for_height(comp.height)
    lut = epoch.palette_rgba(comp.palette_id, matrix)
    placed: List[_Placed] = []
    for item in comp.objects:
        obj = epoch.objects.get(item.object_id)
        if obj is None or not obj.complete or obj.width == 0 or obj.height == 0:
            continue
        idx = epoch.indices(obj)
        x, y = item.x, item.y
        if item.crop is not None:
            cx, cy, cw, ch = item.crop
            idx = idx[cy : cy + ch, cx : cx + cw]
        window = epoch.windows.get(item.window_id)
        if window is not None:
            wx, wy, ww, wh = window
            left, top = max(x, wx), max(y, wy)
            right = min(x + idx.shape[1], wx + ww)
            bottom = min(y + idx.shape[0], wy + wh)
            if right <= left or bottom <= top:
                continue
            idx = idx[top - y : bottom - y, left - x : right - x]
            x, y = left, top
        if idx.size == 0:
            continue
        placed.append(_Placed(lut[idx], x, y, item.forced))
    return placed


def _composite(parts: Sequence[_Placed]) -> _Placed:
    if len(parts) == 1:
        return parts[0]
    left = min(p.x for p in parts)
    top = min(p.y for p in parts)
    right = max(p.right for p in parts)
    bottom = max(p.bottom for p in parts)
    canvas = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
    for p in parts:
        region = canvas[p.y - top : p.bottom - top, p.x - left : p.right - left]
        visible = p.rgba[:, :, 3] > 0
        region[visible] = p.rgba[visible]
    return _Placed(canvas, left, top, parts[0].forced)


def _group(placed: List[_Placed]) -> List[_Placed]:
    """Merge objects that form one caption; keep far-apart ones separate."""
    groups: List[List[_Placed]] = []
    for part in sorted(placed, key=lambda p: (p.y, p.x)):
        if groups:
            last = groups[-1]
            bottom = max(p.bottom for p in last)
            tallest = max(max(p.rgba.shape[0] for p in last), part.rgba.shape[0])
            if part.forced == last[0].forced and part.y - bottom <= GROUP_GAP_FACTOR * tallest:
                last.append(part)
                continue
        groups.append([part])
    return [_composite(g) for g in groups]


def _key(item: _Placed) -> bytes:
    digest = hashlib.sha1(item.rgba.tobytes()).digest()
    return digest + struct.pack(">IIII?", item.x, item.y, item.rgba.shape[1], item.rgba.shape[0], item.forced)


def read_events(path: str) -> List[BitmapEvent]:
    """Every caption in a ``.sup`` file, in time order."""
    with open(path, "rb") as handle:
        data = handle.read()
    return parse_events(data)


def parse_events(data: bytes) -> List[BitmapEvent]:
    epoch = _Epoch()
    events: List[BitmapEvent] = []
    active: Dict[bytes, Tuple[int, _Placed, Tuple[int, int]]] = {}
    pending: Optional[_Composition] = None
    last_pts = 0

    def close(key: bytes, end_pts: int) -> None:
        start_pts, item, size = active.pop(key)
        if end_pts > start_pts:
            events.append(
                BitmapEvent(start_pts / PTS_HZ, end_pts / PTS_HZ, item.rgba, item.x, item.y, item.forced, size)
            )

    def finish(comp: _Composition) -> None:
        items = _group(_render(comp, epoch))
        shown = {_key(item): item for item in items}
        for key in [k for k in active if k not in shown]:
            close(key, comp.pts)
        for key, item in shown.items():
            if key not in active:
                active[key] = (comp.pts, item, (comp.width, comp.height))

    for seg in iter_segments(data):
        if seg.kind == SEG_PCS:
            if pending is not None:
                finish(pending)
            pending = _parse_pcs(data, seg)
            if pending is not None:
                last_pts = max(last_pts, pending.pts)
                if pending.state == STATE_EPOCH_START:
                    epoch = _Epoch()
        elif seg.kind == SEG_WDS:
            p = seg.payload
            count = data[p] if seg.size else 0
            for i in range(count):
                q = p + 1 + 9 * i
                if q + 9 > seg.payload + seg.size:
                    break
                window_id, x, y, w, h = struct.unpack_from(">BHHHH", data, q)
                epoch.windows[window_id] = (x, y, w, h)
        elif seg.kind == SEG_PDS:
            if seg.size < 2:
                continue
            palette_id = data[seg.payload]
            entries = epoch.palettes.get(palette_id)
            if entries is None:
                entries = np.zeros((256, 4), dtype=np.uint8)
                entries[:, 0] = 16
                entries[:, 1:3] = 128
                epoch.palettes[palette_id] = entries
            body = np.frombuffer(data, dtype=np.uint8, count=(seg.size - 2) // 5 * 5, offset=seg.payload + 2)
            body = body.reshape(-1, 5)
            # Stored order is Y, Cr, Cb, A; kept here as Y, Cb, Cr, A.
            entries[body[:, 0]] = body[:, [1, 3, 2, 4]]
        elif seg.kind == SEG_ODS:
            if seg.size < 4:
                continue
            object_id, version, sequence = struct.unpack_from(">HBB", data, seg.payload)
            if sequence & ODS_FIRST:
                if seg.size < 11:
                    continue
                width, height = struct.unpack_from(">HH", data, seg.payload + 7)
                chunk = data[seg.payload + 11 : seg.payload + seg.size]
                epoch.objects[object_id] = _Object(version, width, height, bytearray(chunk), bool(sequence & ODS_LAST))
            else:
                obj = epoch.objects.get(object_id)
                if obj is not None and not obj.complete:
                    obj.data += data[seg.payload + 4 : seg.payload + seg.size]
                    obj.complete = bool(sequence & ODS_LAST)
        elif seg.kind == SEG_END:
            if pending is not None:
                finish(pending)
                pending = None
    if pending is not None:
        finish(pending)
    for key in list(active):
        close(key, active[key][0] + int(OPEN_EVENT_DURATION_S * PTS_HZ))
    events.sort(key=lambda e: (e.start, e.y, e.x))
    return events


def count_captions(path: str) -> int:
    """Display sets that show something, without decoding any bitmap: a
    cheap cue count for probing. Acquisition points are counted too, so this
    is an upper bound on ``len(read_events(path))``."""
    with open(path, "rb") as handle:
        data = handle.read()
    count = 0
    for seg in iter_segments(data):
        if seg.kind == SEG_PCS and seg.size >= 11 and data[seg.payload + 10] > 0:
            if data[seg.payload + 7] != STATE_ACQUISITION_POINT:
                count += 1
    return count


# --------------------------------------------------------- palette rewrite


def adjust_palette(
    in_path: str,
    out_path: str,
    fn: Callable[[int, int, int, int], Tuple[int, int, int, int]],
) -> int:
    """Rewrite every palette entry through ``fn(y, cb, cr, a)``.

    Only the four colour bytes of each PDS entry change; timing, objects,
    windows and even junk between segments are copied byte for byte, so the
    result stays valid for any player that accepted the original. Returns
    the number of palette entries rewritten.
    """
    with open(in_path, "rb") as handle:
        data = handle.read()
    out = bytearray(data)
    cache: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}
    changed = 0
    for seg in iter_segments(data):
        if seg.kind != SEG_PDS:
            continue
        end = seg.payload + seg.size
        p = seg.payload + 2
        while p + 5 <= end:
            y, cr, cb, a = data[p + 1], data[p + 2], data[p + 3], data[p + 4]
            key = (y, cb, cr, a)
            mapped = cache.get(key)
            if mapped is None:
                ny, ncb, ncr, na = fn(y, cb, cr, a)
                mapped = tuple(int(min(255, max(0, round(v)))) for v in (ny, ncb, ncr, na))
                cache[key] = mapped  # type: ignore[assignment]
            out[p + 1], out[p + 2], out[p + 3], out[p + 4] = mapped[0], mapped[2], mapped[1], mapped[3]
            changed += 1
            p += 5
    with open(out_path, "wb") as handle:
        handle.write(bytes(out))
    return changed


# ---------------------------------------------------------------- encoder


def _median_cut(colors: np.ndarray, weights: np.ndarray, target: int) -> np.ndarray:
    """Group unique colours into at most ``target`` boxes; returns the box
    index of every colour. Splits the box with the largest weighted error
    at its weighted median along its widest channel."""
    labels = np.zeros(len(colors), dtype=np.intp)
    boxes: List[np.ndarray] = [np.arange(len(colors))]

    def error(members: np.ndarray) -> float:
        if len(members) < 2:
            return 0.0
        c = colors[members]
        w = weights[members][:, None]
        mean = (c * w).sum(axis=0) / w.sum()
        return float((((c - mean) ** 2) * w).sum())

    errors = [error(boxes[0])]
    while len(boxes) < target:
        worst = int(np.argmax(errors))
        if errors[worst] <= 0:
            break
        members = boxes[worst]
        c = colors[members]
        channel = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
        order = members[np.argsort(c[:, channel], kind="stable")]
        cumulative = np.cumsum(weights[order])
        cut = int(np.searchsorted(cumulative, cumulative[-1] / 2.0)) + 1
        cut = min(max(cut, 1), len(order) - 1)
        first, second = order[:cut], order[cut:]
        boxes[worst] = first
        errors[worst] = error(first)
        boxes.append(second)
        errors.append(error(second))
    for i, members in enumerate(boxes):
        labels[members] = i
    return labels


def quantize(rgba_list: Sequence[np.ndarray], max_colors: int = 255) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Shared palette (``Kx4`` RGBA, entry 0 transparent) and index maps for
    the objects of one display set. Exact when they use few enough colours."""
    pixels = np.concatenate([np.asarray(a, dtype=np.uint8).reshape(-1, 4) for a in rgba_list])
    pixels = pixels.copy()
    pixels[pixels[:, 3] == 0] = 0
    codes = pixels.view(np.uint32).ravel()
    uniq, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    opaque = uniq != 0  # the all-zero code is the transparent pixel
    colours = uniq[opaque].view(np.uint8).reshape(-1, 4)
    limit = max_colors - 1
    if len(colours) <= limit:
        palette = np.vstack([np.zeros((1, 4), dtype=np.uint8), colours])
        mapping = np.zeros(len(uniq), dtype=np.uint8)
        mapping[opaque] = np.arange(1, len(colours) + 1, dtype=np.uint8)
    else:
        weights = counts[opaque].astype(np.float64)
        # Alpha errors show more than colour errors on a subtitle's edge.
        features = colours.astype(np.float64) * np.array([1.0, 1.0, 1.0, 1.5])
        labels = _median_cut(features, weights, limit)
        k = int(labels.max()) + 1
        sums = np.zeros((k, 4))
        np.add.at(sums, labels, colours.astype(np.float64) * weights[:, None])
        totals = np.bincount(labels, weights=weights, minlength=k)
        means = np.clip(np.rint(sums / totals[:, None]), 0, 255).astype(np.uint8)
        means[means[:, 3] == 0] = 0
        palette = np.vstack([np.zeros((1, 4), dtype=np.uint8), means])
        mapping = np.zeros(len(uniq), dtype=np.uint8)
        mapping[opaque] = (labels + 1).astype(np.uint8)
    flat = mapping[inverse.ravel()]
    maps: List[np.ndarray] = []
    offset = 0
    for a in rgba_list:
        h, w = a.shape[:2]
        maps.append(flat[offset : offset + h * w].reshape(h, w))
        offset += h * w
    return palette, maps


def palette_as_decoded(palette: np.ndarray, matrix: str) -> np.ndarray:
    """What a decoder shows for an RGBA palette after the YCbCr round trip."""
    ycc = rgb_to_ycbcr(palette[:, :3], matrix)
    out = np.zeros_like(palette)
    out[:, :3] = ycbcr_to_rgb(ycc[:, 0], ycc[:, 1], ycc[:, 2], matrix)
    out[:, 3] = palette[:, 3]
    out[out[:, 3] == 0] = 0
    return out


def as_encoded(rgba: np.ndarray, video_height: int = 1080, max_colors: int = 255) -> np.ndarray:
    """``rgba`` exactly as ``write_sup`` + ``read_events`` will return it:
    quantised to the palette, then through limited-range YCbCr and back.
    ``max_colors`` is the caption's budget: 255 alone on screen, 128 while
    it shares the screen (and so the palette) with one other caption."""
    palette, (indices,) = quantize([rgba], max_colors)
    return palette_as_decoded(palette, matrix_for_height(video_height))[indices]


def _requantize(rgba: np.ndarray, max_colors: int) -> np.ndarray:
    palette, (indices,) = quantize([rgba], max_colors)
    return palette[indices]


def _colour_budgets(timed: List[Tuple[int, int, "_Placed"]]) -> List[int]:
    """Palette entries each caption may use: 254 shared by the most captions
    it is ever on screen with, plus transparency. A caption keeps the same
    colours in every display set it appears in, so an overlap starting or
    ending does not re-quantise it -- which would look like a new caption."""
    crowd = [1] * len(timed)
    starting: Dict[int, List[int]] = {}
    ending: Dict[int, List[int]] = {}
    for i, (s, e, _) in enumerate(timed):
        starting.setdefault(s, []).append(i)
        ending.setdefault(e, []).append(i)
    live: set = set()
    for t in sorted(set(starting) | set(ending)):
        live.difference_update(ending.get(t, ()))
        live.update(starting.get(t, ()))
        for i in live:
            crowd[i] = max(crowd[i], len(live))
    return [254 // min(k, 254) + 1 for k in crowd]


def _segment(kind: int, pts: int, payload: bytes, dts: int = 0) -> bytes:
    if len(payload) > MAX_SEGMENT:
        raise ValueError("PGS segment payload too large")
    return b"PG" + struct.pack(">IIBH", pts & 0xFFFFFFFF, dts & 0xFFFFFFFF, kind, len(payload)) + payload


def _ods_segments(object_id: int, pts: int, width: int, height: int, rle: bytes) -> List[bytes]:
    first_room = MAX_SEGMENT - 11
    later_room = MAX_SEGMENT - 4
    chunks = [rle[:first_room]]
    rest = rle[first_room:]
    while rest:
        chunks.append(rest[:later_room])
        rest = rest[later_room:]
    out = []
    for i, chunk in enumerate(chunks):
        flags = (ODS_FIRST if i == 0 else 0) | (ODS_LAST if i == len(chunks) - 1 else 0)
        header = struct.pack(">HBB", object_id, 0, flags)
        if i == 0:
            length = len(rle) + 4
            header += struct.pack(">BH", length >> 16, length & 0xFFFF) + struct.pack(">HH", width, height)
        out.append(_segment(SEG_ODS, pts, header + chunk))
    return out


def _clip_to_frame(ev: BitmapEvent, size: Tuple[int, int]) -> Optional[_Placed]:
    w, h = size
    left, top = max(0, ev.x), max(0, ev.y)
    right, bottom = min(w, ev.x + ev.width), min(h, ev.y + ev.height)
    if right <= left or bottom <= top:
        return None
    rgba = np.asarray(ev.rgba, dtype=np.uint8)[top - ev.y : bottom - ev.y, left - ev.x : right - ev.x]
    return _Placed(rgba, left, top, ev.forced)


def _overlaps(a: _Placed, b: _Placed) -> bool:
    return a.x < b.right and b.x < a.right and a.y < b.bottom and b.y < a.bottom


def _objects_for(active: List[_Placed]) -> List[_Placed]:
    """At most two non-overlapping objects (PGS allows two windows, and
    windows may not overlap); anything more is composited together."""
    parts = sorted(active, key=lambda p: (p.y, p.x))
    merged = True
    while merged:
        merged = False
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if _overlaps(parts[i], parts[j]):
                    parts[i] = _composite([parts[i], parts[j]])
                    del parts[j]
                    merged = True
                    break
            if merged:
                break
    while len(parts) > 2:
        gaps = [parts[i + 1].y - parts[i].bottom for i in range(len(parts) - 1)]
        i = int(np.argmin(gaps))
        parts[i : i + 2] = [_composite(parts[i : i + 2])]
    return parts


def encode_display_sets(events: Sequence[BitmapEvent], video_size: Tuple[int, int]) -> bytes:
    width, height = int(video_size[0]), int(video_size[1])
    matrix = matrix_for_height(height)
    timed = []
    for ev in events:
        start, end = int(round(ev.start * PTS_HZ)), int(round(ev.end * PTS_HZ))
        if end > start:
            placed = _clip_to_frame(ev, (width, height))
            if placed is not None:
                timed.append((start, end, placed))
    for (start, end, placed), budget in zip(list(timed), _colour_budgets(timed)):
        placed.rgba = _requantize(placed.rgba, budget)
    starting: Dict[int, List[int]] = {}
    ending: Dict[int, List[int]] = {}
    for i, (s, e, _) in enumerate(timed):
        starting.setdefault(s, []).append(i)
        ending.setdefault(e, []).append(i)
    out = bytearray()
    number = 0
    live: set = set()
    shown: Tuple[int, ...] = ()
    windows: List[Tuple[int, int, int, int]] = []
    for t in sorted(set(starting) | set(ending)):
        live.difference_update(ending.get(t, ()))
        live.update(starting.get(t, ()))
        now = tuple(sorted(live))
        if now == shown:
            continue
        shown = now
        if not now:
            pcs = struct.pack(">HHBHBBBB", width, height, FRAME_RATE_CODE, number & 0xFFFF, STATE_NORMAL, 0, 0, 0)
            out += _segment(SEG_PCS, t, pcs)
            wds = bytes([len(windows)]) + b"".join(struct.pack(">BHHHH", i, *w) for i, w in enumerate(windows))
            out += _segment(SEG_WDS, t, wds)
            out += _segment(SEG_END, t, b"")
            number += 1
            continue
        objects = _objects_for([timed[i][2] for i in now])
        palette, maps = quantize([o.rgba for o in objects])
        ycc = rgb_to_ycbcr(palette[:, :3], matrix)
        pcs = struct.pack(
            ">HHBHBBBB", width, height, FRAME_RATE_CODE, number & 0xFFFF, STATE_EPOCH_START, 0, 0, len(objects)
        )
        for i, o in enumerate(objects):
            pcs += struct.pack(">HBBHH", i, i, OBJECT_FORCED if o.forced else 0, o.x, o.y)
        out += _segment(SEG_PCS, t, pcs)
        windows = [(o.x, o.y, o.rgba.shape[1], o.rgba.shape[0]) for o in objects]
        wds = bytes([len(windows)]) + b"".join(struct.pack(">BHHHH", i, *w) for i, w in enumerate(windows))
        out += _segment(SEG_WDS, t, wds)
        pds = bytearray(b"\x00\x00")
        for i, (entry, (y, cb, cr)) in enumerate(zip(palette, ycc)):
            pds += bytes((i, int(y), int(cr), int(cb), int(entry[3])))
        out += _segment(SEG_PDS, t, bytes(pds))
        for i, (o, indices) in enumerate(zip(objects, maps)):
            for seg in _ods_segments(i, t, o.rgba.shape[1], o.rgba.shape[0], rle_encode(indices)):
                out += seg
        out += _segment(SEG_END, t, b"")
        number += 1
    return bytes(out)


def write_sup(events: Sequence[BitmapEvent], path: str, video_size: Optional[Tuple[int, int]] = None) -> str:
    """Encode captions as a ``.sup``.

    Every change of what is on screen becomes one self-contained display set
    (an epoch start with its own windows, palette and objects), and the end
    of the last caption shown becomes an empty composition, so any decoder
    can start anywhere. Overlapping captions share a display set; more than
    two, or overlapping rectangles, are composited because PGS allows two
    non-overlapping windows. Colours are quantised per display set to at
    most 255 palette entries plus transparency.
    """
    if video_size is None:
        video_size = events[0].video_size if events else (1920, 1080)
    with open(path, "wb") as handle:
        handle.write(encode_display_sets(events, video_size))
    return path
