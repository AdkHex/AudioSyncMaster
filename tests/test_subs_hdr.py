"""Tests for HDR subtitle brightness (task "hdrSubs").

The PGS checks use a hand-built stream so the expected bytes are known
exactly -- every byte outside the palette entries must survive -- and a
stream from ``pgs.write_sup`` decoded back to pixels, so the result is what
a player would draw.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from audiosync.media import CancellationToken, ffmpeg_path  # noqa: E402
from audiosync.subs import hdr_subs, tasks  # noqa: E402
from audiosync.subs.tasks import TaskError  # noqa: E402


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="subsync-hdr-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)

    def write(self, name, text, encoding="utf-8"):
        with open(self.path(name), "w", encoding=encoding, newline="") as handle:
            handle.write(text)
        return self.path(name)


def _run(path, out_dir, track=None, **options):
    ref = {"path": path}
    if track is not None:
        ref["track"] = track
    job = {"id": "t", "task": "hdrSubs", "input": {"subtitle": ref}, "options": options,
           "output": {"dir": out_dir}}
    return tasks.run_job(job, CancellationToken(), lambda p, s: None, lambda m: None)


def _read(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


# ------------------------------------------------------------ hand-made PGS

# Palette of the hand-built stream: (id, Y, Cr, Cb, A) in PDS byte order.
PALETTE = [
    (0, 16, 128, 128, 0),      # transparent
    (1, 235, 128, 128, 255),   # white fill
    (2, 16, 128, 128, 255),    # black outline
    (3, 126, 128, 128, 255),   # grey anti-aliasing
    (4, 210, 146, 16, 200),    # yellow, part transparent
]


def _segment(kind, payload, pts=90000):
    return b"PG" + struct.pack(">IIBH", pts, 0, kind, len(payload)) + payload


def _hand_built_sup():
    """One caption (8x2 pixels using palette 1-4) and a clearing display set."""
    width, height = 8, 2
    rle = b""
    for row in ([1, 1, 2, 3, 4, 1, 2, 3], [2, 2, 3, 3, 1, 1, 4, 4]):
        rle += bytes(row) + b"\x00\x00"
    pcs = struct.pack(">HHBHBBBB", 1920, 1080, 0x10, 1, 0x80, 0, 0, 1) + struct.pack(">HBBHH", 0, 0, 0, 900, 1000)
    wds = struct.pack(">B", 1) + struct.pack(">BHHHH", 0, 900, 1000, width, height)
    pds = struct.pack(">BB", 0, 0) + b"".join(struct.pack(">BBBBB", *entry) for entry in PALETTE)
    ods_data = struct.pack(">HH", width, height) + rle
    ods = struct.pack(">HBB", 0, 0, 0xC0) + len(ods_data).to_bytes(3, "big") + ods_data
    clear = struct.pack(">HHBHBBBB", 1920, 1080, 0x10, 2, 0x00, 0, 0, 0)
    return (_segment(0x16, pcs) + _segment(0x17, wds) + _segment(0x14, pds) + _segment(0x15, ods)
            + _segment(0x80, b"") + _segment(0x16, clear, pts=270000) + _segment(0x17, wds, pts=270000)
            + _segment(0x80, b"", pts=270000))


def _pds_entries(data):
    """(offset of entry, (id, Y, Cr, Cb, A)) for every palette entry."""
    out = []
    pos = 0
    while pos + 13 <= len(data):
        kind = data[pos + 10]
        size = int.from_bytes(data[pos + 11:pos + 13], "big")
        payload = pos + 13
        if kind == 0x14:
            p = payload + 2
            while p + 5 <= payload + size:
                out.append((p, tuple(data[p:p + 5])))
                p += 5
        pos = payload + size
    return out


def test_mapping_numbers_follow_the_linear_light_model():
    mapping, _ = hdr_subs.mapping_from_options({})
    assert abs(mapping.factor - 0.203) < 1e-12
    # 0.203 ** (1 / 2.4) = 0.5147 of the code range.
    assert mapping.map_rgb255(255, 255, 255) == (131, 131, 131)
    assert mapping.map_rgb255(0, 0, 0) == (0, 0, 0)
    assert mapping.map_pgs(235, 128, 128, 255) == (129, 128, 128, 255)
    assert mapping.map_pgs(16, 128, 128, 255) == (16, 128, 128, 255)
    assert mapping.map_pgs(235, 128, 128, 0) == (235, 128, 128, 0), "transparent entries untouched"
    half, _ = hdr_subs.mapping_from_options({"mode": "percent", "brightnessPercent": 50})
    assert half.map_rgb255(255, 255, 255) == (191, 191, 191)
    # Anti-aliasing keeps its proportion to the fill in light.
    grey = np.array(mapping.map_rgb255(128, 128, 128)) / 255.0
    assert abs((grey[0] ** 2.4) / 0.203 - (128 / 255) ** 2.4) < 0.003


def test_pgs_palette_rewrite_keeps_every_other_byte():
    with Workspace() as ws:
        source = ws.path("film.sup")
        with open(source, "wb") as handle:
            handle.write(_hand_built_sup())
        result = _run(source, ws.root)
        out = result.outputs[0].path
        assert os.path.basename(out) == "film.hdr.sup", out
        with open(source, "rb") as handle:
            before = handle.read()
        with open(out, "rb") as handle:
            after = handle.read()
        assert len(before) == len(after)
        entries = _pds_entries(after)
        colour_bytes = {offset + k for offset, _ in entries for k in (1, 2, 3)}
        diffs = [i for i in range(len(before)) if before[i] != after[i]]
        assert diffs and set(diffs) <= colour_bytes, f"bytes outside palette colours changed: {diffs}"
        got = {entry[0]: entry[1:] for _, entry in entries}
        assert got[0] == (16, 128, 128, 0), got[0]
        assert got[1] == (129, 128, 128, 255), got[1]
        assert got[2] == (16, 128, 128, 255), got[2]
        assert got[3] == (73, 128, 128, 255), got[3]
        assert got[4][3] == 200 and got[4][0] < 210 * 0.6, got[4]
        assert result.report["entriesChanged"] == 3, result.report
        assert result.report["paletteEntries"] == 5, result.report
        assert result.report["format"] == "pgs" and abs(result.report["factor"] - 0.203) < 1e-9


def test_pgs_recolour_to_yellow_keeps_outline_black():
    with Workspace() as ws:
        source = ws.path("film.sup")
        with open(source, "wb") as handle:
            handle.write(_hand_built_sup())
        result = _run(source, ws.root, color="yellow")
        with open(result.outputs[0].path, "rb") as handle:
            got = {entry[0]: entry[1:] for _, entry in _pds_entries(handle.read())}
        y, cr, cb, a = got[1]
        assert cb < 100 and cr > 128 and 100 < y < 140, got[1]
        assert got[2] == (16, 128, 128, 255), got[2]


def test_pgs_from_write_sup_decodes_dimmer():
    from audiosync.subs import pgs

    with Workspace() as ws:
        rgba = np.zeros((20, 60, 4), np.uint8)
        rgba[4:16, 4:56] = (0, 0, 0, 255)       # outline box
        rgba[6:14, 6:54] = (255, 255, 255, 255)  # white fill
        event = pgs.BitmapEvent(start=1.0, end=3.0, rgba=rgba, x=100, y=900, video_size=(1920, 1080))
        source = pgs.write_sup([event], ws.path("made.sup"), (1920, 1080))
        result = _run(source, ws.root)
        before = pgs.read_events(source)[0].rgba
        after = pgs.read_events(result.outputs[0].path)[0].rgba
        assert np.array_equal(before[..., 3], after[..., 3]), "alpha changed"
        fill = after[8:12, 10:50, :3]
        assert abs(int(fill.mean()) - 131) <= 2, fill.mean()
        assert after[5, 5, :3].max() <= 2, "outline no longer black"


def test_image_to_text_and_text_to_sup_are_refused():
    with Workspace() as ws:
        sup = ws.path("film.sup")
        with open(sup, "wb") as handle:
            handle.write(_hand_built_sup())
        srt = ws.write("film.srt", "1\n00:00:01,000 --> 00:00:02,000\nHi\n")
        for path, fmt, words in ((sup, "ass", "OCR"), (srt, "sup", "PGS")):
            try:
                _run(path, ws.root, outputFormat=fmt)
            except TaskError as exc:
                assert words in str(exc), str(exc)
            else:
                raise AssertionError(f"{os.path.basename(path)} -> {fmt} accepted")


# ---------------------------------------------------------------------- ASS

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2.5,1,2,40,40,50,1
Style: Sign,Arial,40,&H000000FF,&H000000FF,&H00101010,&H00000000,0,0,0,0,100,100,0,0,1,2,0,8,40,40,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello {\\c&HFFFFFF&}white{\\1c&H0000FF&} red {\\3c&H000000&\\clip(0,0,100,100)}end
Comment: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\c&HFFFFFF&}comment untouched
Dialogue: 0,0:00:04.00,0:00:05.00,Sign,,0,0,0,,{\\an8}EXIT
"""


def test_ass_fill_colours_dim_and_the_rest_is_untouched():
    with Workspace() as ws:
        source = ws.write("film.ass", ASS)
        result = _run(source, ws.root)
        out = result.outputs[0].path
        assert os.path.basename(out) == "film.hdr.ass", out
        text = _read(out)
        before, after = ASS.splitlines(), text.splitlines()
        assert len(before) == len(after)
        changed = [(b, a) for b, a in zip(before, after) if b != a]
        assert [a[:13] for _, a in changed] == ["Style: Defaul", "Style: Sign,A", "Dialogue: 0,0"], changed
        assert "Style: Default,Arial,52,&H00838383,&H00000083,&H00000000,&H80000000," in text
        assert "Style: Sign,Arial,40,&H00000083,&H00000083,&H00101010,&H00000000," in text
        assert "{\\c&H838383&}white{\\1c&H000083&} red {\\3c&H000000&\\clip(0,0,100,100)}end" in text
        assert "{\\c&HFFFFFF&}comment untouched" in text
        assert result.report["entriesChanged"] == 6, result.report


def test_ass_recolour_changes_styles_and_grey_inline_tags_only():
    with Workspace() as ws:
        source = ws.write("film.ass", ASS)
        text = _read(_run(source, ws.root, color="yellow").outputs[0].path)
        # Style fills take the colour at their own luminance; outline stays.
        assert "Style: Default,Arial,52,&H00008383,&H00000083,&H00000000," in text, text
        assert "Style: Sign,Arial,40,&H00004545," in text, text
        # White inline tag -> yellow; red inline tag keeps its hue, dimmed.
        assert "{\\c&H008383&}white{\\1c&H000083&}" in text, text


def test_ssa_decimal_colours():
    ssa = (
        "[Script Info]\nScriptType: v4.00\n\n[V4 Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, "
        "Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding\n"
        "Style: Default,Arial,52,16777215,255,0,-2147483648,0,0,1,2,1,2,40,40,50,0,1\n\n[Events]\n"
        "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: Marked=0,0:00:01.00,0:00:02.00,Default,,0000,0000,0000,,Hello\n"
    )
    with Workspace() as ws:
        text = _read(_run(ws.write("old.ssa", ssa), ws.root).outputs[0].path)
        assert "Style: Default,Arial,52,8618883,131,0,-2147483648," in text, text


# ---------------------------------------------------------------- SRT / VTT

SRT = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
<font color="#FF0000">Red</font> and <i>plain</i>
"""


def test_srt_same_format_wraps_text_in_a_font_colour():
    with Workspace() as ws:
        result = _run(ws.write("film.en.srt", SRT), ws.root)
        out = result.outputs[0].path
        assert out.endswith(".srt") and ".hdr" in os.path.basename(out), out
        text = _read(out)
        assert '<font color="#838383">Hello</font>' in text, text
        assert '<font color="#830000">Red</font>' in text, text
        assert text.count('<font color="#838383">') == 2, text
        assert "player" in result.summary, result.summary


def test_srt_to_ass_uses_a_styled_colour():
    with Workspace() as ws:
        result = _run(ws.write("film.srt", SRT), ws.root, outputFormat="ass")
        text = _read(result.outputs[0].path)
        assert result.outputs[0].format == "ass"
        style = next(line for line in text.splitlines() if line.startswith("Style: Default"))
        assert style.split(",")[3] == "&H00838383", style
        assert "\\c&H000083&" in text, text


def test_vtt_same_format_adds_a_cue_style():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n"
    with Workspace() as ws:
        text = _read(_run(ws.write("film.vtt", vtt), ws.root).outputs[0].path)
        assert text.startswith("WEBVTT"), text
        assert "::cue {\n  color: #838383;\n}" in text, text
        assert text.index("STYLE") < text.index("-->"), "STYLE must precede the cues"


def test_embedded_text_track_is_extracted_first():
    with Workspace() as ws:
        srt = ws.write("in.srt", SRT)
        video = ws.path("movie.mkv")
        subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=5",
                        "-i", srt, "-map", "0", "-map", "1", "-c:v", "libx264", "-c:s", "srt", video],
                       check=True, capture_output=True)
        result = _run(video, ws.root, track=0)
        out = result.outputs[0].path
        assert os.path.basename(out).startswith("movie.track1.hdr"), out
        assert '<font color="#838383">Hello</font>' in _read(out)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
