"""Tests for the bitmap subtitle formats: PGS (.sup) and VobSub (.idx/.sub).

PGS streams here are built segment by segment with a small builder of the
tests' own (not the module's encoder), so a mistake shared by the encoder
and the decoder cannot hide. VobSub gets the same treatment: a minimal DVD
SPU + MPEG-PS writer below produces exact pixels to compare. The real-world
path -- FFmpeg decoding our .sup and re-encoding it as DVD subtitles in an
MKV, mkvextract writing .idx/.sub -- runs when those tools are installed.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from audiosync.subs import imaging, pgs, vobsub  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subs_fixtures import Workspace, render, run_module, skippable, tool  # noqa: E402


# ------------------------------------------------------------ PGS builder


def seg(kind, pts, payload):
    return b"PG" + struct.pack(">IIBH", pts, 0, kind, len(payload)) + payload


def pcs(pts, objects, number=0, state=0x80, palette_update=False, size=(1920, 1080)):
    body = struct.pack(">HHBHBBBB", size[0], size[1], 0x10, number, state, 0x80 if palette_update else 0, 0,
                       len(objects))
    for obj in objects:
        oid, wid, x, y = obj["id"], obj.get("window", obj["id"]), obj["x"], obj["y"]
        flags = (0x40 if obj.get("forced") else 0) | (0x80 if obj.get("crop") else 0)
        body += struct.pack(">HBBHH", oid, wid, flags, x, y)
        if obj.get("crop"):
            body += struct.pack(">HHHH", *obj["crop"])
    return seg(0x16, pts, body)


def wds(pts, windows):
    return seg(0x17, pts, bytes([len(windows)]) + b"".join(struct.pack(">BHHHH", *w) for w in windows))


def pds(pts, entries, version=0):
    return seg(0x14, pts, bytes([0, version]) + b"".join(bytes(e) for e in entries))


def ods(pts, oid, indices, version=0):
    rle = pgs.rle_encode(indices)
    h, w = indices.shape
    length = len(rle) + 4
    body = struct.pack(">HBB", oid, version, 0xC0) + struct.pack(">BH", length >> 16, length & 0xFFFF)
    return seg(0x15, pts, body + struct.pack(">HH", w, h) + rle)


def end(pts):
    return seg(0x80, pts, b"")


#: (id, Y, Cr, Cb, A): 1 white, 2 black, 3 half-transparent grey.
PALETTE = [(1, 235, 128, 128, 255), (2, 16, 128, 128, 255), (3, 126, 128, 128, 128)]


def glyphs(h=40, w=120, seed=0):
    """Palette indices that look like outlined text: white bars with a
    black border and a grey soft edge."""
    rng = np.random.default_rng(seed)
    idx = np.zeros((h, w), dtype=np.uint8)
    for x in range(4, w - 8, 9):
        top = int(rng.integers(3, 10))
        idx[top : h - 4, x : x + 5] = 2
        idx[top + 1 : h - 5, x + 1 : x + 4] = 1
        idx[top - 1, x : x + 5] = 3
    return idx


def lut(matrix="bt709"):
    table = np.zeros((256, 4), dtype=np.uint8)
    for i, y, cr, cb, a in PALETTE:
        table[i, :3] = pgs.ycbcr_to_rgb(y, cb, cr, matrix)
        table[i, 3] = a
    return table


# ------------------------------------------------------------------- RLE


def test_rle_decodes_the_spec_codes():
    # 5 | 00 03: three 0s | 00 8A 07: ten 7s | 00 41 00: 256 0s | 00 C1 00 09: 256 9s | 00 00
    data = bytes([5, 0, 3, 0, 0x8A, 7, 0, 0x41, 0x00, 0, 0xC1, 0x00, 9, 0, 0])
    row = pgs.rle_decode(data, 1 + 3 + 10 + 256 + 256, 1)[0]
    expected = [5] + [0] * 3 + [7] * 10 + [0] * 256 + [9] * 256
    assert row.tolist() == expected


def test_rle_round_trip_including_long_runs():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 4, size=(30, 200)).astype(np.uint8)
    img[5, :] = 0
    img[6, :] = 200
    img[7, 10:12] = 9  # two-pixel literal run
    wide = np.zeros((2, 20000), dtype=np.uint8)  # runs longer than one code allows
    wide[1, 100:19000] = 17
    for case in (img, wide):
        assert np.array_equal(pgs.rle_decode(pgs.rle_encode(case), case.shape[1], case.shape[0]), case)


def test_rle_decoder_tolerates_missing_end_of_line():
    data = bytes([4, 4, 0, 0, 7, 7])  # second line never terminated
    out = pgs.rle_decode(data, 2, 2)
    assert out.tolist() == [[4, 4], [7, 7]]


# ---------------------------------------------------------------- colour


def test_palette_conversion_limited_range():
    for matrix in ("bt709", "bt601"):
        assert pgs.ycbcr_to_rgb(235, 128, 128, matrix).tolist() == [255, 255, 255]
        assert pgs.ycbcr_to_rgb(16, 128, 128, matrix).tolist() == [0, 0, 0]
    # The same code means a different red under the two matrices.
    red709 = pgs.ycbcr_to_rgb(63, 102, 240, "bt709")
    red601 = pgs.ycbcr_to_rgb(63, 102, 240, "bt601")
    assert red709.tolist() != red601.tolist()
    assert pgs.matrix_for_height(576) == "bt601" and pgs.matrix_for_height(1080) == "bt709"
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
    for matrix in ("bt709", "bt601"):
        ycc = pgs.rgb_to_ycbcr(rgb, matrix)
        back = pgs.ycbcr_to_rgb(ycc[:, 0], ycc[:, 1], ycc[:, 2], matrix)
        assert np.abs(back.astype(int) - rgb).max() <= 3


# ---------------------------------------------------------------- events


def test_events_follow_compositions_acquisition_points_and_palette_updates():
    idx = glyphs()
    stream = b"".join([
        # t=1s: epoch start shows object 0
        pcs(90000, [{"id": 0, "x": 900, "y": 950}], number=0),
        wds(90000, [(0, 900, 950, idx.shape[1], idx.shape[0])]),
        pds(90000, PALETTE),
        ods(90000, 0, idx),
        end(90000),
        # t=2s: acquisition point re-sends the same thing -> same event
        pcs(180000, [{"id": 0, "x": 900, "y": 950}], number=1, state=0x40),
        wds(180000, [(0, 900, 950, idx.shape[1], idx.shape[0])]),
        pds(180000, PALETTE, version=1),
        ods(180000, 0, idx, version=1),
        end(180000),
        # t=3s: palette-only update (a fade step) -> a new picture
        pcs(270000, [{"id": 0, "x": 900, "y": 950}], number=2, state=0x00, palette_update=True),
        pds(270000, [(1, 235, 128, 128, 128)], version=2),
        end(270000),
        # t=4.5s: empty composition clears the screen
        pcs(405000, [], number=3, state=0x00),
        wds(405000, [(0, 900, 950, idx.shape[1], idx.shape[0])]),
        end(405000),
    ])
    events = pgs.parse_events(stream)
    assert [(e.start, e.end) for e in events] == [(1.0, 3.0), (3.0, 4.5)], [(e.start, e.end) for e in events]
    first, second = events
    assert (first.x, first.y, first.video_size) == (900, 950, (1920, 1080))
    assert np.array_equal(first.rgba, lut()[idx])
    # Only the white entry changed: its alpha halved in the second picture.
    white = idx == 1
    assert (second.rgba[white][:, 3] == 128).all() and (first.rgba[white][:, 3] == 255).all()
    assert np.array_equal(second.rgba[~white], first.rgba[~white])


def test_far_apart_objects_are_separate_events_and_close_ones_merge():
    top, bottom, second_line = glyphs(seed=1), glyphs(seed=2), glyphs(seed=3)
    h = top.shape[0]
    stream = b"".join([
        pcs(0, [{"id": 0, "x": 100, "y": 50, "forced": True}, {"id": 1, "x": 800, "y": 950}]),
        wds(0, [(0, 100, 50, 120, h), (1, 800, 950, 120, h)]),
        pds(0, PALETTE), ods(0, 0, top), ods(0, 1, bottom), end(0),
        pcs(90000, []), end(90000),
        # Two lines authored as two objects 10 px apart: one caption.
        pcs(180000, [{"id": 0, "x": 800, "y": 900}, {"id": 1, "x": 800, "y": 900 + h + 10}]),
        wds(180000, [(0, 800, 900, 120, h), (1, 800, 900 + h + 10, 120, h)]),
        pds(180000, PALETTE), ods(180000, 0, bottom), ods(180000, 1, second_line), end(180000),
        pcs(270000, []), end(270000),
    ])
    events = pgs.parse_events(stream)
    assert len(events) == 3, len(events)
    a, b, c = events
    assert (a.y, a.forced, b.y, b.forced) == (50, True, 950, False)
    assert a.start == b.start == 0.0 and a.end == b.end == 1.0
    assert (c.x, c.y, c.rgba.shape[:2]) == (800, 900, (2 * h + 10, 120))
    assert np.array_equal(c.rgba[:h], lut()[bottom]) and np.array_equal(c.rgba[h + 10 :], lut()[second_line])
    assert (c.rgba[h : h + 10] == 0).all()


def test_cropping_and_window_clipping():
    idx = glyphs(h=40, w=120)
    stream = b"".join([
        pcs(0, [{"id": 0, "x": 500, "y": 600, "crop": (10, 5, 60, 30)}]),
        wds(0, [(0, 500, 600, 60, 30)]),
        pds(0, PALETTE), ods(0, 0, idx), end(0),
        pcs(90000, [{"id": 0, "x": 500, "y": 600}], state=0x00),  # the window cuts it down
        pds(90000, PALETTE), end(90000),
        pcs(180000, []), end(180000),
    ])
    first, second = pgs.parse_events(stream)
    assert np.array_equal(first.rgba, lut()[idx[5:35, 10:70]])
    assert (first.x, first.y) == (500, 600)
    assert np.array_equal(second.rgba, lut()[idx[:30, :60]])


def test_epoch_start_forgets_old_objects():
    idx = glyphs()
    stream = b"".join([
        pcs(0, [{"id": 0, "x": 0, "y": 0}]), wds(0, [(0, 0, 0, 120, 40)]), pds(0, PALETTE), ods(0, 0, idx), end(0),
        # New epoch refers to object 0 without defining it: nothing is shown.
        pcs(90000, [{"id": 0, "x": 0, "y": 0}], state=0x80), end(90000),
    ])
    events = pgs.parse_events(stream)
    assert [(e.start, e.end) for e in events] == [(0.0, 1.0)]


def test_stream_without_final_clear_gets_default_duration():
    idx = glyphs()
    stream = pcs(9000, [{"id": 0, "x": 0, "y": 0}]) + pds(9000, PALETTE) + ods(9000, 0, idx) + end(9000)
    (event,) = pgs.parse_events(stream)
    assert abs(event.end - event.start - pgs.OPEN_EVENT_DURATION_S) < 1e-9


# ------------------------------------------------------------- round trip


def _outlined_rgba(h=90, w=700, seed=0):
    """Many colours and alpha levels, so write_sup must quantise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    fill = ((xx // 7) % 3 == 0) & (yy > 12) & (yy < h - 12)
    outline = imaging.dilate(fill, 2) & ~fill
    soft = imaging.dilate(fill, 3) & ~imaging.dilate(fill, 2)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[fill] = (255, 255, 255, 255)
    rgba[outline] = (0, 0, 0, 255)
    rgba[soft, 3] = rng.integers(20, 200, size=int(soft.sum()))
    rgba[soft, :3] = rng.integers(0, 90, size=(int(soft.sum()), 1))
    rgba[fill & (xx % 5 == 0), :3] = (255, 230, 0)
    return rgba


def test_write_sup_round_trip_is_exact():
    rgba = _outlined_rgba()
    small = _outlined_rgba(60, 300, seed=1)
    events = [
        pgs.BitmapEvent(1.001, 3.5, rgba, 600, 900, False, (1920, 1080)),
        pgs.BitmapEvent(2.0, 4.0, small, 100, 60, True, (1920, 1080)),
        pgs.BitmapEvent(3600.123456, 3602.5, small, 50, 50, False, (1920, 1080)),
    ]
    with Workspace() as ws:
        first = pgs.write_sup(events, ws.path("a.sup"), (1920, 1080))
        back = pgs.read_events(first)
        assert len(back) == 3
        # The first two overlap in time and so share a palette: 127 colours
        # each (plus transparency); the third has all 254 to itself.
        budgets = [128, 128, 255]
        for src, got, budget in zip(events, back, budgets):
            assert round(got.start * 90000) == round(src.start * 90000)
            assert round(got.end * 90000) == round(src.end * 90000)
            assert (got.x, got.y, got.forced, got.video_size) == (src.x, src.y, src.forced, (1920, 1080))
            # Exactly what the palette + YCbCr round trip predicts...
            assert np.array_equal(got.rgba, pgs.as_encoded(src.rgba, 1080, budget))
            # ...which is visually the source: alpha is kept exactly where
            # the image needed no quantisation, colours within a few codes.
            assert np.abs(got.rgba.astype(int) - src.rgba).max() <= 40
        # Encoding what was decoded changes nothing at all.
        again = pgs.read_events(pgs.write_sup(back, ws.path("b.sup"), (1920, 1080)))
        for a, b in zip(back, again):
            assert np.array_equal(a.rgba, b.rgba) and (a.start, a.end, a.x, a.y) == (b.start, b.end, b.x, b.y)


def test_large_objects_are_fragmented_across_segments():
    rng = np.random.default_rng(5)
    noise = np.zeros((300, 800, 4), dtype=np.uint8)
    noise[..., :3] = rng.integers(0, 4, size=(300, 800, 1)) * 80
    noise[..., 3] = 255
    with Workspace() as ws:
        path = pgs.write_sup([pgs.BitmapEvent(0.0, 1.0, noise, 0, 0)], ws.path("big.sup"), (1920, 1080))
        with open(path, "rb") as handle:
            data = handle.read()
        ods_segments = [s for s in pgs.iter_segments(data) if s.kind == pgs.SEG_ODS]
        assert len(ods_segments) >= 2, len(ods_segments)
        (event,) = pgs.read_events(path)
        assert np.array_equal(event.rgba, pgs.as_encoded(noise))


def test_adjust_palette_changes_only_palette_bytes():
    events = [pgs.BitmapEvent(0.5, 2.0, _outlined_rgba(), 600, 900), pgs.BitmapEvent(3.0, 4.0, _outlined_rgba(seed=2), 600, 900)]
    with Workspace() as ws:
        src = pgs.write_sup(events, ws.path("in.sup"), (1920, 1080))
        with open(src, "rb") as handle:
            original = handle.read()
        # Junk between segments must survive too.
        junk = original[:13] + b"garbage!" + original[13:]
        with open(src, "wb") as handle:
            handle.write(junk)
        same = ws.path("same.sup")
        pgs.adjust_palette(src, same, lambda y, cb, cr, a: (y, cb, cr, a))
        with open(same, "rb") as handle:
            assert handle.read() == junk
        dim = ws.path("dim.sup")
        count = pgs.adjust_palette(src, dim, lambda y, cb, cr, a: (16 + (y - 16) // 2, cb, cr, a))
        with open(dim, "rb") as handle:
            out = handle.read()
        assert len(out) == len(junk) and count > 0
        palette_bytes = set()
        for s in pgs.iter_segments(junk):
            if s.kind == pgs.SEG_PDS:
                for p in range(s.payload + 2, s.payload + s.size, 5):
                    palette_bytes.add(p + 1)  # only Y was changed
        changed = {i for i, (a, b) in enumerate(zip(junk, out)) if a != b}
        assert changed and changed <= palette_bytes
        before, after = pgs.read_events(src), pgs.read_events(dim)
        assert [(e.start, e.end, e.x, e.y) for e in before] == [(e.start, e.end, e.x, e.y) for e in after]
        white = (before[0].rgba[..., :3] == 255).all(axis=2)
        assert after[0].rgba[white][:, 0].max() < 140


def test_count_captions_ignores_acquisition_points():
    idx = glyphs()
    stream = b"".join([
        pcs(0, [{"id": 0, "x": 0, "y": 0}]), pds(0, PALETTE), ods(0, 0, idx), end(0),
        pcs(9000, [{"id": 0, "x": 0, "y": 0}], state=0x40), end(9000),
        pcs(18000, []), end(18000),
    ])
    with Workspace() as ws:
        with open(ws.path("c.sup"), "wb") as handle:
            handle.write(stream)
        assert pgs.count_captions(ws.path("c.sup")) == 1


@skippable
def test_rendered_text_round_trips_exactly():
    """A real CoreText-rendered caption (anti-aliased, outlined): encode,
    decode, identical to the predicted palette + colour conversion."""
    rgba = render([{"lines": [{"text": "Round trip, exactly."}], "font": "HelveticaNeue-Medium",
                            "size": 52, "outline": 4}])[0]
    with Workspace() as ws:
        (event,) = pgs.read_events(pgs.write_sup([pgs.BitmapEvent(10.0, 12.0, rgba, 400, 900)], ws.path("r.sup"),
                                                 (1920, 1080)))
        assert np.array_equal(event.rgba, pgs.as_encoded(rgba))
        assert (event.rgba[..., 3] == pgs.as_encoded(rgba)[..., 3]).all()
        assert np.abs(event.rgba.astype(int) - rgba).max() <= 64


# ---------------------------------------------------------------- VobSub


def _nibbles_for_run(length, colour, to_end):
    if to_end:
        return [0, 0, 0, colour]
    value = (length << 2) | colour
    if length < 4:
        return [value]
    if length < 16:
        return [value >> 4, value & 0xF]
    if length < 64:
        return [value >> 8, (value >> 4) & 0xF, value & 0xF]
    return [value >> 12, (value >> 8) & 0xF, (value >> 4) & 0xF, value & 0xF]


def encode_field(rows):
    nibbles = []
    for row in rows:
        change = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], change))
        lengths = np.diff(np.concatenate((starts, [len(row)])))
        for n, (c, length) in enumerate(zip(row[starts].tolist(), lengths.tolist())):
            last = n == len(starts) - 1
            while length > 0:
                if last and length > 255:
                    nibbles += _nibbles_for_run(0, c, True)
                    break
                run = min(length, 255)
                nibbles += _nibbles_for_run(run, c, False)
                length -= run
        if len(nibbles) % 2:
            nibbles.append(0)
    return bytes((nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles), 2))


def encode_spu(indices, x, y, colours=(0, 1, 2, 3), alphas=(0, 15, 15, 15), forced=False, stop_units=None):
    h, w = indices.shape
    top, bottom = encode_field(indices[0::2]), encode_field(indices[1::2])
    top_off = 4
    bottom_off = top_off + len(top)
    ctrl = bottom_off + len(bottom)
    c = colours
    a = alphas
    x2, y2 = x + w - 1, y + h - 1
    seq1 = bytes([0x00 if forced else 0x01, 0x03, (c[3] << 4) | c[2], (c[1] << 4) | c[0],
                  0x04, (a[3] << 4) | a[2], (a[1] << 4) | a[0],
                  0x05, x >> 4, ((x & 0xF) << 4) | (x2 >> 8), x2 & 0xFF, y >> 4, ((y & 0xF) << 4) | (y2 >> 8), y2 & 0xFF,
                  0x06]) + struct.pack(">HH", top_off, bottom_off) + b"\xff"
    seq1_len = 4 + len(seq1)
    if stop_units is None:
        control = struct.pack(">HH", 0, ctrl) + seq1
    else:
        second = ctrl + seq1_len
        control = struct.pack(">HH", 0, second) + seq1 + struct.pack(">HH", stop_units, second) + b"\x02\xff"
    body = top + bottom + control
    return struct.pack(">HH", 4 + len(body), ctrl) + body


def _pts_bytes(pts):
    return bytes([0x21 | ((pts >> 29) & 0x0E), (pts >> 22) & 0xFF, ((pts >> 14) & 0xFE) | 1, (pts >> 7) & 0xFF,
                  ((pts << 1) & 0xFE) | 1])


def packetise(spu, stream, pts):
    """SPU -> 2048-byte MPEG-2 PS packs, padded like real .sub files."""
    out = b""
    first = True
    pos = 0
    while pos < len(spu):
        header = b"\x81\x80\x05" + _pts_bytes(pts) if first else b"\x81\x00\x00"
        room = 2048 - 14 - 6 - len(header) - 1
        chunk = spu[pos : pos + room]
        pos += len(chunk)
        pes_body = header + bytes([0x20 + stream]) + chunk
        pack = b"\x00\x00\x01\xba\x44\x00\x04\x00\x04\x01\x01\x89\xc3\xf8"
        pack += b"\x00\x00\x01\xbd" + struct.pack(">H", len(pes_body)) + pes_body
        pad = 2048 - len(pack)
        if pad >= 6:
            pack += b"\x00\x00\x01\xbe" + struct.pack(">H", pad - 6) + b"\xff" * (pad - 6)
        out += pack
        first = False
    return out


def write_vobsub(base, subs, size=(720, 480)):
    palette = ["000000", "ffffff", "808080", "ff0000"] + ["000000"] * 12
    data = b""
    streams = {}
    for sub in subs:
        streams.setdefault(sub["stream"], []).append((sub["time"], len(data)))
        data += packetise(sub["spu"], sub["stream"], int(sub["time"] * 90000))
    lines = ["# VobSub index file, v7 (do not modify this line!)", f"size: {size[0]}x{size[1]}",
             "palette: " + ", ".join(palette), "langidx: 0"]
    for stream, entries in sorted(streams.items()):
        lines.append(f"id: {'en' if stream == 0 else 'ja'}, index: {stream}")
        if stream == 1:
            lines.append("delay: 00:00:00:500")
        for t, pos in entries:
            ms = int(round(t * 1000))
            stamp = f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}:{ms % 1000:03d}"
            lines.append(f"timestamp: {stamp}, filepos: {pos:09x}")
    with open(base + ".idx", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(base + ".sub", "wb") as handle:
        handle.write(data)
    return base + ".idx"


def _dvd_picture(h, w, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(h, w)).astype(np.uint8)
    idx[:, : w // 3] = np.where(idx[:, : w // 3] == 0, 1, idx[:, : w // 3])  # long runs + no empty border
    idx[5, :] = 3
    idx[0, :] = 2
    idx[-1, :] = 2
    idx[:, 0] = 2
    idx[:, -1] = 2
    return idx


def test_vobsub_decodes_exact_pixels_times_and_forced():
    big = _dvd_picture(151, 500, 1)  # several KB: spans packets
    small = _dvd_picture(20, 64, 2)
    subs = [
        {"stream": 0, "time": 1.0, "spu": encode_spu(big, 100, 300, colours=(0, 1, 0, 2), stop_units=176)},
        {"stream": 0, "time": 4.0, "spu": encode_spu(small, 30, 40, forced=True)},
        {"stream": 1, "time": 2.0, "spu": encode_spu(small, 50, 60, colours=(0, 3, 0, 2))},
        {"stream": 0, "time": 6.0, "spu": encode_spu(small, 30, 40, alphas=(0, 8, 15, 15), stop_units=88)},
    ]
    assert len(subs[0]["spu"]) > 4000
    rgb = np.array([(0, 0, 0), (255, 255, 255), (128, 128, 128), (255, 0, 0)], dtype=np.uint8)
    with Workspace() as ws:
        idx_path = write_vobsub(ws.path("movie"), subs)
        index = vobsub.read_index(idx_path)
        assert index.size == (720, 480) and [s.language for s in index.streams] == ["en", "ja"]
        events = vobsub.read_events(idx_path)
        assert [round(e.start, 3) for e in events] == [1.0, 4.0, 6.0]
        # Stop after 176 * 1024 / 90000 s; open-ended ones end at the next.
        assert abs(events[0].end - (1.0 + 176 * 1024 / 90000)) < 1e-6
        assert events[1].end == 6.0 and abs(events[2].end - (6.0 + 88 * 1024 / 90000)) < 1e-6
        assert [e.forced for e in events] == [False, True, False]
        assert (events[0].x, events[0].y, events[0].video_size) == (100, 300, (720, 480))
        colour_map = np.array([0, 1, 0, 2])  # pixel value -> palette entry
        expected = np.zeros(big.shape + (4,), dtype=np.uint8)
        expected[..., :3] = rgb[colour_map[big]]
        expected[..., 3] = np.where(big == 0, 0, 255)
        expected[big == 0] = 0
        assert np.array_equal(events[0].rgba, expected)
        assert (events[2].rgba[small == 1][:, 3] == 8 * 17).all()
        # The second stream, with its index delay.
        (ja,) = vobsub.read_events(idx_path, stream=1)
        assert abs(ja.start - 2.5) < 1e-9 and (ja.x, ja.y) == (50, 60)
        assert (ja.rgba[small == 1][:, :3] == (255, 0, 0)).all()
        # Reading via the .sub path works too.
        assert len(vobsub.read_events(ws.path("movie.sub"))) == 3


@skippable
def test_ffmpeg_dvdsub_and_mkvextract_path():
    """Our .sup -> FFmpeg (PGS decoder, DVD encoder, MKV) -> mkvextract
    .idx/.sub -> our VobSub reader. Checks the encoder against FFmpeg's
    decoder and the reader against real tools' output."""
    ffmpeg, mkvextract = tool("ffmpeg"), tool("mkvextract")
    rgba = _outlined_rgba()
    events = [pgs.BitmapEvent(1.0, 2.5, rgba, 600, 900), pgs.BitmapEvent(3.0, 4.2, rgba[:, :350], 100, 80, True)]
    with Workspace() as ws:
        sup = pgs.write_sup(events, ws.path("in.sup"), (1920, 1080))
        mkv = ws.path("dvd.mkv")
        # -fix_sub_duration: FFmpeg's PGS decoder does not know when a
        # caption ends until the next display set; this makes it pass the
        # end on, so the DVD encoder writes a real stop command.
        subprocess.run([ffmpeg, "-v", "error", "-y", "-copyts", "-fix_sub_duration", "-i", sup, "-map", "0:s",
                        "-c:s", "dvdsub", "-f", "matroska", mkv], check=True)
        subprocess.run([mkvextract, mkv, "tracks", f"0:{ws.path('dvd.idx')}"], check=True, capture_output=True)
        back = vobsub.read_events(ws.path("dvd.idx"))
        assert [round(e.start, 2) for e in back] == [1.0, 3.0], [e.start for e in back]
        # DVD stop delays count in 1024/90000 s units (11.4 ms), rounded down.
        assert abs(back[0].end - 2.5) < 0.012 and abs(back[1].end - 4.2) < 0.012, [e.end for e in back]
        # The PGS forced flag survives FFmpeg into the DVD "forced start" command.
        assert [e.forced for e in back] == [False, True]
        assert back[0].video_size == (1920, 1080)
        for src, got in zip(events, back):
            want = src.rgba[..., 3] > 0
            top, bottom, left, right = imaging.bbox(want)
            assert (got.x, got.y) == (src.x + left, src.y + top)
            assert got.rgba.shape[:2] == (bottom - top, right - left)
            # A DVD picture has four colours: every solid pixel of the source
            # survives, and the soft edge is either dropped or made solid --
            # nothing appears where the source was transparent.
            have = got.rgba[..., 3] >= 200
            solid = src.rgba[top:bottom, left:right, 3] >= 200
            visible = src.rgba[top:bottom, left:right, 3] > 0
            assert not (solid & ~have).any()
            assert not ((got.rgba[..., 3] > 0) & ~visible).any()


def test_png_codec_round_trip():
    rng = np.random.default_rng(0)
    for shape in ((7, 5), (9, 4, 3), (6, 3, 4), (5, 5, 2)):
        img = rng.integers(0, 256, size=shape).astype(np.uint8)
        back = imaging.decode_png(imaging.encode_png(img))
        if img.ndim == 2:
            img = np.dstack([img, img, img, np.full(img.shape, 255, np.uint8)])
        elif img.shape[2] == 2:
            img = np.dstack([img[..., :1]] * 3 + [img[..., 1:]])
        elif img.shape[2] == 3:
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        assert np.array_equal(back, img)


if __name__ == "__main__":
    sys.exit(1 if run_module(dict(globals())) else 0)
