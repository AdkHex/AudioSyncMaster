"""DVD VobSub (``.idx`` + ``.sub``): parse and render to bitmaps.

The ``.idx`` is a text index: frame size, the 16-colour palette (RGB hex),
and per language stream an ``id:`` line followed by one
``timestamp: HH:MM:SS:mmm, filepos: HEX`` line per subtitle. The ``.sub`` is
an MPEG program stream holding only subtitle packets: pack headers, then PES
packets on private stream 1 (``0xBD``) whose first payload byte is the
sub-stream id ``0x20 + N``.

A subtitle picture (SPU) is split across as many PES packets as it needs;
its first two bytes give its total size, so packets of the same sub-stream
are appended until that size is reached. Inside the SPU, control sequences
say when to show (``0x01``, or ``0x00`` for *forced* display), when to hide
(``0x02``), which 4 of the 16 palette colours and alphas to use (``0x03``,
``0x04``), where on screen (``0x05``) and where the two interlaced fields'
2-bit run-length data start (``0x06``).

Times come from the ``.idx`` (like FFmpeg and every player), because that is
where authoring tools and users apply delays; the PES timestamp is only a
fallback for SPUs the index does not list.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .pgs import BitmapEvent

#: Control-sequence delays count in units of 1024 ticks of the 90 kHz clock.
DELAY_UNIT_S = 1024.0 / 90000.0
#: Shown this long when an SPU never says when to stop and nothing follows.
OPEN_EVENT_DURATION_S = 5.0

DEFAULT_PALETTE = [
    (0, 0, 0), (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (255, 255, 255), (0, 0, 0), (128, 128, 128), (255, 255, 255),
    (0, 0, 0), (128, 128, 128), (255, 255, 255), (0, 0, 0),
    (128, 128, 128), (255, 255, 255), (0, 0, 0), (128, 128, 128),
]  # fmt: skip

_TIMESTAMP_RE = re.compile(r"timestamp:\s*(-?)(\d+):(\d+):(\d+)[:.,](\d+)\s*,\s*filepos:\s*([0-9a-fA-F]+)")
_DELAY_RE = re.compile(r"delay:\s*(-?)(\d+):(\d+):(\d+)[:.,](\d+)")


@dataclass
class IndexStream:
    #: Language code as written after ``id:`` (usually ISO 639-1).
    language: Optional[str]
    #: N of sub-stream ``0x20 + N``.
    index: int
    #: (seconds, file position) of each subtitle, in index order.
    entries: List[Tuple[float, int]] = field(default_factory=list)


@dataclass
class VobSubIndex:
    size: Tuple[int, int] = (720, 480)
    palette: List[Tuple[int, int, int]] = field(default_factory=lambda: list(DEFAULT_PALETTE))
    origin: Tuple[int, int] = (0, 0)
    langidx: int = 0
    streams: List[IndexStream] = field(default_factory=list)
    #: ``custom colors: ON`` replaces each SPU's colour choice.
    custom_colors: Optional[List[Tuple[int, int, int]]] = None
    custom_transparent: Optional[List[bool]] = None

    def stream(self, which: Optional[int] = None) -> Optional[IndexStream]:
        """Stream by sub-stream index; default the ``langidx`` one, else the
        first that has entries."""
        want = self.langidx if which is None else which
        for s in self.streams:
            if s.index == want:
                return s
        return next((s for s in self.streams if s.entries), self.streams[0] if self.streams else None)


def _hms(sign: str, h: str, m: str, s: str, ms: str) -> float:
    value = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    return -value if sign == "-" else value


def _hex_colour(token: str) -> Tuple[int, int, int]:
    token = token.strip().lstrip("#")
    value = int(token, 16) if token else 0
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def read_index(idx_path: str) -> VobSubIndex:
    with open(idx_path, "rb") as handle:
        text = handle.read().decode("latin-1")
    index = VobSubIndex()
    current: Optional[IndexStream] = None
    offset_s = 0.0
    delay_s = 0.0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "size":
            m = re.match(r"(\d+)\s*x\s*(\d+)", value)
            if m:
                index.size = (int(m.group(1)), int(m.group(2)))
        elif key == "org":
            m = re.match(r"(-?\d+)\s*,\s*(-?\d+)", value)
            if m:
                index.origin = (int(m.group(1)), int(m.group(2)))
        elif key == "palette":
            colours = [_hex_colour(t) for t in value.split(",") if t.strip()]
            if colours:
                index.palette = (colours + list(DEFAULT_PALETTE))[:16]
        elif key == "custom colors":
            if value.upper().startswith("ON"):
                m = re.search(r"tridx:\s*([01]{4})", value)
                c = re.search(r"colors:\s*(.+)$", value)
                if c:
                    index.custom_colors = [_hex_colour(t) for t in c.group(1).split(",")[:4]]
                    tridx = m.group(1) if m else "0000"
                    index.custom_transparent = [ch == "1" for ch in tridx]
        elif key == "time offset":
            m = re.match(r"(-?\d+)", value)
            if m:
                offset_s = int(m.group(1)) / 1000.0
        elif key == "langidx":
            if value.lstrip("-").isdigit():
                index.langidx = int(value)
        elif key == "id":
            m = re.match(r"([^,\s]*)\s*(?:,\s*index:\s*(\d+))?", value)
            language = m.group(1) if m and m.group(1) and m.group(1) != "--" else None
            number = int(m.group(2)) if m and m.group(2) else len(index.streams)
            current = IndexStream(language, number)
            index.streams.append(current)
            delay_s = 0.0
        elif key == "delay":
            m = _DELAY_RE.match(line)
            if m:
                delay_s += _hms(*m.groups())
        elif key == "timestamp":
            m = _TIMESTAMP_RE.match(line)
            if m:
                if current is None:
                    current = IndexStream(None, 0)
                    index.streams.append(current)
                seconds = _hms(*m.groups()[:5]) + delay_s + offset_s
                current.entries.append((seconds, int(m.group(6), 16)))
    return index


# ----------------------------------------------------------- program stream


@dataclass
class _Packet:
    #: File offset of the pack (what ``filepos`` in the .idx points at).
    pack_offset: int
    pts: Optional[int]
    stream: int
    payload: bytes


def _parse_pts(b: bytes, p: int) -> int:
    return (
        ((b[p] >> 1) & 0x07) << 30
        | b[p + 1] << 22
        | (b[p + 2] >> 1) << 15
        | b[p + 3] << 7
        | (b[p + 4] >> 1)
    )


def iter_packets(data: bytes) -> List[_Packet]:
    """Every subpicture PES packet of an MPEG-PS ``.sub``."""
    packets: List[_Packet] = []
    pos = 0
    n = len(data)
    pack_offset = 0
    while pos + 4 <= n:
        if data[pos : pos + 3] != b"\x00\x00\x01":
            nxt = data.find(b"\x00\x00\x01", pos + 1)
            if nxt < 0:
                break
            pos = nxt
            continue
        code = data[pos + 3]
        if code == 0xBA:
            pack_offset = pos
            if pos + 5 > n:
                break
            if data[pos + 4] & 0xC0 == 0x40:  # MPEG-2
                if pos + 14 > n:
                    break
                pos += 14 + (data[pos + 13] & 0x07)
            else:  # MPEG-1
                pos += 12
            continue
        if code == 0xB9:
            pos += 4
            continue
        if pos + 6 > n:
            break
        length = struct.unpack_from(">H", data, pos + 4)[0]
        start = pos + 6
        end = min(n, start + length)
        if code == 0xBD and end > start:
            p = start
            pts = None
            if data[p] & 0xC0 == 0x80:  # MPEG-2 PES header
                flags = data[p + 1]
                header_len = data[p + 2]
                if flags & 0x80 and p + 8 <= end:
                    pts = _parse_pts(data, p + 3)
                p += 3 + header_len
            else:  # MPEG-1: stuffing, optional STD buffer, PTS
                while p < end and data[p] == 0xFF:
                    p += 1
                if p < end and data[p] & 0xC0 == 0x40:
                    p += 2
                if p < end and data[p] & 0xF0 in (0x20, 0x30):
                    pts = _parse_pts(data, p)
                    p += 10 if data[p] & 0xF0 == 0x30 else 5
                elif p < end and data[p] == 0x0F:
                    p += 1
            if p < end:
                stream = data[p]
                if 0x20 <= stream <= 0x3F:
                    packets.append(_Packet(pack_offset, pts, stream - 0x20, data[p + 1 : end]))
        pos = start + length
    return packets


@dataclass
class _Spu:
    pack_offset: int
    pts: Optional[int]
    data: bytearray


def assemble(packets: List[_Packet], stream: int, starts: Optional[set] = None) -> List[_Spu]:
    """Join each SPU's packets in order until its declared size is reached.

    A packet at a position the index lists always starts a new SPU, so one
    damaged SPU cannot swallow the next; otherwise packets continue the SPU
    in progress.
    """
    out: List[_Spu] = []
    current: Optional[_Spu] = None
    expected = 0
    for pkt in packets:
        if pkt.stream != stream:
            continue
        known_start = starts is not None and pkt.pack_offset in starts
        if current is not None and len(current.data) < expected and not known_start:
            current.data += pkt.payload
        else:
            if len(pkt.payload) < 2:
                current = None
                continue
            current = _Spu(pkt.pack_offset, pkt.pts, bytearray(pkt.payload))
            expected = struct.unpack_from(">H", pkt.payload, 0)[0]
        if current is not None and len(current.data) >= expected:
            out.append(current)
            current = None
    return out


# --------------------------------------------------------------------- SPU


@dataclass
class _SpuPicture:
    start_delay: float
    stop_delay: Optional[float]
    forced: bool
    x: int
    y: int
    indices: np.ndarray
    colours: Tuple[int, int, int, int]
    alphas: Tuple[int, int, int, int]


def _decode_field(data: bytes, offset: int, width: int, rows: int, limit: int) -> np.ndarray:
    """2-bit run-length data of one field. Codes are 1-4 nibbles::

        n n c c              run 1-3
        0 0 n n  n n c c     run 4-15
        0 0 0 0  n n n n  n n c c   run 16-63
        0 0 0 0  0 0 n n  n n n n  n n c c   run 64-255 (0 = to end of line)

    and every line starts on a byte boundary."""
    out = np.zeros((rows, width), dtype=np.uint8)
    nibble = offset * 2
    end = limit * 2

    def get() -> int:
        nonlocal nibble
        if nibble >= end:
            return 0
        byte = data[nibble >> 1]
        value = (byte >> 4) if nibble & 1 == 0 else (byte & 0x0F)
        nibble += 1
        return value

    for y in range(rows):
        x = 0
        while x < width:
            if nibble >= end:
                return out
            v = get()
            if v < 0x4:
                v = (v << 4) | get()
                if v < 0x10:
                    v = (v << 4) | get()
                    if v < 0x40:
                        v = (v << 4) | get()
            run = v >> 2
            colour = v & 0x3
            if run == 0:
                run = width - x
            run = min(run, width - x)
            if colour:
                out[y, x : x + run] = colour
            x += run
        if nibble & 1:
            nibble += 1
    return out


def decode_spu(spu: bytes) -> Optional[_SpuPicture]:
    if len(spu) < 4:
        return None
    size, control = struct.unpack_from(">HH", spu, 0)
    size = min(size, len(spu))
    colours = (0, 1, 2, 3)
    alphas = (0, 15, 15, 15)
    x1 = y1 = 0
    x2 = y2 = -1
    top = bottom = None
    start_delay: Optional[float] = None
    stop_delay: Optional[float] = None
    forced = False
    seen = set()
    offset = control
    while offset + 4 <= size and offset not in seen:
        seen.add(offset)
        delay, next_offset = struct.unpack_from(">HH", spu, offset)
        when = delay * DELAY_UNIT_S
        p = offset + 4
        while p < size:
            cmd = spu[p]
            p += 1
            if cmd == 0x00:
                forced = True
                start_delay = when if start_delay is None else start_delay
            elif cmd == 0x01:
                start_delay = when if start_delay is None else start_delay
            elif cmd == 0x02:
                # 0xFFFF (12.4 minutes) is what encoders write when they did
                # not know the end -- FFmpeg does for PGS input. Treat it as
                # "until the next subtitle", not as a real duration.
                stop_delay = when if delay < 0xFFFF else None
            elif cmd == 0x03 and p + 2 <= size:
                a, b = spu[p], spu[p + 1]
                colours = (b & 0x0F, b >> 4, a & 0x0F, a >> 4)
                p += 2
            elif cmd == 0x04 and p + 2 <= size:
                a, b = spu[p], spu[p + 1]
                alphas = (b & 0x0F, b >> 4, a & 0x0F, a >> 4)
                p += 2
            elif cmd == 0x05 and p + 6 <= size:
                c = spu[p : p + 6]
                x1 = (c[0] << 4) | (c[1] >> 4)
                x2 = ((c[1] & 0x0F) << 8) | c[2]
                y1 = (c[3] << 4) | (c[4] >> 4)
                y2 = ((c[4] & 0x0F) << 8) | c[5]
                p += 6
            elif cmd == 0x06 and p + 4 <= size:
                top, bottom = struct.unpack_from(">HH", spu, p)
                p += 4
            elif cmd == 0x07 and p + 2 <= size:
                # Change colour/contrast per region: skipped (rare, karaoke).
                p += struct.unpack_from(">H", spu, p)[0]
            else:  # 0xFF end of sequence, or unknown: stop this sequence
                break
        if next_offset == offset:
            break
        offset = next_offset
    width, height = x2 - x1 + 1, y2 - y1 + 1
    if width <= 0 or height <= 0 or top is None or bottom is None:
        return None
    even = _decode_field(spu, top, width, (height + 1) // 2, control)
    odd = _decode_field(spu, bottom, width, height // 2, control)
    indices = np.zeros((height, width), dtype=np.uint8)
    indices[0::2] = even
    indices[1::2] = odd
    return _SpuPicture(start_delay or 0.0, stop_delay, forced, x1, y1, indices, colours, alphas)


def _to_rgba(pic: _SpuPicture, index: VobSubIndex) -> np.ndarray:
    lut = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        if index.custom_colors:
            rgb = index.custom_colors[i]
            alpha = 0 if index.custom_transparent and index.custom_transparent[i] else pic.alphas[i] * 17
        else:
            rgb = index.palette[pic.colours[i] & 0x0F]
            alpha = pic.alphas[i] * 17
        lut[i, :3] = rgb
        lut[i, 3] = alpha
    lut[lut[:, 3] == 0] = 0
    return lut[pic.indices]


def _trim(rgba: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Drop fully transparent borders; DVD SPUs are often full-width."""
    visible = rgba[:, :, 3] > 0
    rows = np.flatnonzero(visible.any(axis=1))
    if rows.size == 0:
        return rgba[:0, :0], 0, 0
    cols = np.flatnonzero(visible.any(axis=0))
    return rgba[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1], int(cols[0]), int(rows[0])


def sub_path_for(idx_path: str) -> str:
    base = os.path.splitext(idx_path)[0]
    for ext in (".sub", ".SUB"):
        if os.path.isfile(base + ext):
            return base + ext
    return base + ".sub"


def read_events(idx_path: str, stream: Optional[int] = None) -> List[BitmapEvent]:
    """Every subtitle of one stream (default: the ``langidx`` one)."""
    if idx_path.lower().endswith(".sub"):
        idx_path = os.path.splitext(idx_path)[0] + ".idx"
    index = read_index(idx_path)
    with open(sub_path_for(idx_path), "rb") as handle:
        data = handle.read()
    chosen = index.stream(stream)
    stream_id = chosen.index if chosen else 0
    times = {pos: t for t, pos in (chosen.entries if chosen else [])}
    spus = assemble(iter_packets(data), stream_id, set(times))

    decoded = []
    for spu in spus:
        pic = decode_spu(bytes(spu.data))
        if pic is None:
            continue
        base = times.get(spu.pack_offset)
        if base is None:
            if spu.pts is None:
                continue
            base = spu.pts / 90000.0
        decoded.append((base, pic))
    decoded.sort(key=lambda item: item[0])

    events: List[BitmapEvent] = []
    for i, (base, pic) in enumerate(decoded):
        start = base + pic.start_delay
        # A DVD shows one subpicture per stream at a time: the next one
        # replaces this one even when this one never says "stop".
        following = decoded[i + 1][0] + decoded[i + 1][1].start_delay if i + 1 < len(decoded) else None
        if pic.stop_delay is not None and pic.stop_delay > pic.start_delay:
            end = base + pic.stop_delay
        else:
            end = following if following is not None else start + OPEN_EVENT_DURATION_S
        if following is not None and following > start:
            end = min(end, following)
        rgba, dx, dy = _trim(_to_rgba(pic, index))
        if rgba.size == 0 or end <= start:
            continue
        x = pic.x + dx + index.origin[0]
        y = pic.y + dy + index.origin[1]
        events.append(BitmapEvent(start, end, np.ascontiguousarray(rgba), x, y, pic.forced, index.size))
    return events
