"""Tests for reading and writing subtitle formats.

Every reader is fed the kind of file people actually have -- broken SRT
numbering, Netflix TTML in ticks, legacy code pages -- and every writer is
checked by reading its output back, so a format that only looks right in a
text editor cannot pass.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.media import CancellationToken  # noqa: E402
from audiosync.subs import formats  # noqa: E402
from audiosync.subs.model import Cue, SubtitleDoc  # noqa: E402
from audiosync.subs.tasks import TaskContext, TaskError  # noqa: E402


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="subsync-formats-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)

    def write(self, name, data):
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path


def _ctx(workdir):
    return TaskContext(token=CancellationToken(), progress=lambda *_: None, log=lambda *_: None, workdir=workdir)


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------- SRT

MESSY_SRT = (
    "﻿1\r\n00:00:01,000 --> 00:00:02.5\r\n{\\an8}<i>Hello</i>\r\nworld\r\n\r\n\r\n"
    "00:00:03,000 --> 00:00:04,000 X1:10 X2:20 Y1:5 Y2:9\r\nNo index here\r\n"
    "7\r\n00:00:05,000-->00:00:06,000\r\n<font color=\"#ff0000\">last</font>\r\n\r\n"
)


def test_srt_lenient_parsing():
    doc = formats.parse(MESSY_SRT, "srt")
    assert [c.text for c in doc.cues] == ["<i>Hello</i>\nworld", "No index here", '<font color="#ff0000">last</font>'], \
        [c.text for c in doc.cues]
    assert _close(doc.cues[0].end, 2.5), "a one-digit fraction is tenths"
    assert doc.cues[0].align == 8 and doc.cues[1].align is None
    assert _close(doc.cues[2].start, 5.0)


def test_srt_roundtrip_and_rounding():
    doc = SubtitleDoc(cues=[
        Cue(0.0, 1.0004, "a"),
        Cue(59.9996, 61.5, "<b>b</b>\nsecond line", align=8),
        Cue(62.0, 63.0, ""),  # nothing to show: not written
    ])
    text = formats.render(doc, "srt")
    assert "00:01:00,000 --> 00:01:01,500" in text, "rounding carried into the minutes"
    assert "{\\an8}<b>b</b>" in text
    back = formats.parse(text, "srt")
    assert len(back.cues) == 2 and back.cues[1].align == 8 and back.cues[1].text == "<b>b</b>\nsecond line"


# ---------------------------------------------------------------------- ASS

ASS = r"""[Script Info]
; made by hand
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1
Style: Sign,Arial,30,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,8,10,10,10,1

[Fonts]
fontname: x.ttf
M3+U

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,translator note
Dialogue: 0,0:00:01.00,0:00:02.50,Default,Bob,0,0,0,,{\an8\i1}Hi{\i0} there\Nnext {\c&H0000FF&}red{\bord3\fs20}x
Dialogue: 1,0:00:03.00,0:00:04.00,Sign,,0,0,0,,{\p1}m 0 0 l 100 0 100 100{\p0}
Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,{\t(0,100,\i1)\b1}bold\hA{\b0} {\u1}u{\u0}
"""


def test_ass_tags_become_markup():
    doc = formats.parse(ASS, "ass")
    first, drawing, third = doc.cues
    assert first.text == '<i>Hi</i> there\nnext <font color="#FF0000">redx</font>', first.text
    assert first.align == 8 and first.speaker == "Bob" and first.style == "Default"
    assert "\\bord3" in first.raw, "the original line is kept in raw"
    assert drawing.text == "" and drawing.meta.get("drawing") is True and drawing.raw.startswith("{\\p1}")
    assert third.text == "<b>bold A</b> <u>u</u>", repr(third.text)
    assert doc.play_res == (1280, 720)
    assert "[Fonts]" in doc.ass_header and "Style: Sign" in doc.ass_header and "[Events]" not in doc.ass_header


def test_ass_untouched_roundtrip_is_exact():
    doc = formats.parse(ASS, "ass")
    rendered = formats.render(doc, "ass")
    original = [line for line in ASS.splitlines() if line.startswith(("Dialogue:", "Comment:"))]
    written = [line for line in rendered.splitlines() if line.startswith(("Dialogue:", "Comment:"))]
    assert written == original, "\n".join(written)
    again = formats.parse(rendered, "ass")
    assert again.ass_header == doc.ass_header


def test_ass_retime_keeps_override_tags():
    shifted = formats.parse(ASS, "ass").shift(1.0)
    lines = [line for line in formats.render(shifted, "ass").splitlines() if line.startswith("Dialogue:")]
    assert lines[0] == ("Dialogue: 0,0:00:02.00,0:00:03.50,Default,Bob,0,0,0,,"
                        "{\\an8\\i1}Hi{\\i0} there\\Nnext {\\c&H0000FF&}red{\\bord3\\fs20}x"), lines[0]
    assert lines[1].startswith("Dialogue: 1,0:00:04.00,0:00:05.00,Sign,"), "layer and style kept"
    edited = formats.parse(ASS, "ass")
    edited.cues[0].text = "<i>Changed</i> line"
    line = [x for x in formats.render(edited, "ass").splitlines() if x.startswith("Dialogue:")][0]
    assert line.endswith(",Bob,0,0,0,,{\\an8\\i1}Changed{\\i0} line"), line


def test_ass_written_from_srt_has_a_sane_default_style():
    doc = formats.parse(MESSY_SRT, "srt")
    text = formats.render(doc, "ass")
    assert "PlayResX: 1920" in text and "PlayResY: 1080" in text
    style = [line for line in text.splitlines() if line.startswith("Style: Default,")][0]
    assert style.split(",")[2] == "54", style  # 5% of 1080
    assert "{\\an8\\i1}Hello{\\i0}\\Nworld" in text
    assert "{\\c&H0000FF&}last{\\c}" in text
    back = formats.parse(text, "ass")
    assert [c.text for c in back.cues] == [c.text.replace("ff0000", "FF0000") for c in doc.cues]


def test_ssa_legacy_alignment_and_format():
    ssa = (
        "[Script Info]\nScriptType: v4.00\n\n[V4 Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n\n"
        "[Events]\nFormat: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: Marked=0,0:00:01.00,0:00:02.00,Default,,0000,0000,0000,,{\\a6}Top line\n"
    )
    doc = formats.parse(ssa, "ssa")
    assert doc.cues[0].align == 8 and doc.cues[0].text == "Top line"
    assert formats.render(doc, "ssa").splitlines()[-1] == ssa.splitlines()[-1]


# ------------------------------------------------------------------- WebVTT

VTT = """WEBVTT - with a header
Kind: captions

STYLE
::cue { color: yellow }

NOTE this is a comment
that spans lines

REGION
id:fred

intro
00:01.000 --> 00:02.500 line:0 align:start
<v Mary>Hi &amp; <i>welcome</i></v>

00:00:03.000 --> 00:00:04.000 line:90%
<c.yellow>Coloured</c> <ruby>漢<rt>かん</rt></ruby>字

01:00:05.000 --> 01:00:06.000
<00:00:05.500>karaoke &lt;tag&gt;
"""


def test_vtt_parse():
    doc = formats.parse(VTT, "vtt")
    assert len(doc.cues) == 3, [c.text for c in doc.cues]
    first, second, third = doc.cues
    assert _close(first.start, 1.0) and _close(first.end, 2.5)
    assert first.text == "Hi & <i>welcome</i>" and first.speaker == "Mary" and first.align == 7
    assert second.text == "Coloured 漢字" and second.align is None
    assert _close(third.start, 3605.0) and third.text == "karaoke <tag>"


def test_vtt_roundtrip():
    doc = formats.parse(VTT, "vtt")
    text = formats.render(doc, "vtt")
    assert text.startswith("WEBVTT\n") and "00:00:01.000 --> 00:00:02.500 line:0 align:start" in text
    assert "&lt;tag&gt;" in text and "Hi &amp; <i>welcome</i>" in text
    back = formats.parse(text, "vtt")
    assert [(c.text, c.align, c.speaker) for c in back.cues] == [(c.text, c.align, c.speaker) for c in doc.cues]


# --------------------------------------------------------------------- TTML

NETFLIX_TTML = """<?xml version="1.0" encoding="utf-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttp="http://www.w3.org/ns/ttml#parameter"
    xmlns:tts="http://www.w3.org/ns/ttml#styling" ttp:tickRate="10000000" xml:lang="ja">
  <head>
    <styling>
      <style xml:id="s1" tts:fontStyle="italic"/>
      <style xml:id="s2" style="s1" tts:color="#FFFF00"/>
    </styling>
    <layout>
      <region xml:id="top" tts:origin="10% 10%" tts:extent="80% 80%" tts:displayAlign="before"/>
      <region xml:id="bottom" tts:origin="10% 10%" tts:extent="80% 80%" tts:displayAlign="after"/>
    </layout>
  </head>
  <body>
    <div>
      <p begin="10000000t" end="25000000t" region="bottom">First
        line<br/>second <span style="s1">italic</span></p>
      <p begin="30000000t" end="40000000t" region="top"><span tts:fontStyle="italic">Sign</span></p>
      <p begin="50000000t" end="60000000t" style="s2">Yellow italic</p>
    </div>
  </body>
</tt>
"""

FRAMES_DFXP = """<tt xmlns="http://www.w3.org/2006/10/ttaf1" xmlns:ttp="http://www.w3.org/2006/10/ttaf1#parameter"
    ttp:frameRate="24" ttp:frameRateMultiplier="1000 1001">
  <body><div begin="00:00:10.000">
    <p begin="00:00:01:12" end="00:00:02:00">Offset by the div</p>
    <div begin="2s"><p begin="500ms" dur="1.5s">Nested</p></div>
    <p begin="00:01:00.25" end="00:01:01.5">Clock</p>
  </div></body>
</tt>"""


def test_ttml_netflix_ticks_styles_regions():
    doc = formats.parse(NETFLIX_TTML, "ttml")
    assert doc.language == "ja"
    assert [(c.start, c.end) for c in doc.cues] == [(1.0, 2.5), (3.0, 4.0), (5.0, 6.0)]
    first, sign, yellow = doc.cues
    assert first.text == "First line\nsecond <i>italic</i>", repr(first.text)
    assert first.align is None and sign.align == 8 and sign.text == "<i>Sign</i>"
    assert yellow.text == '<i><font color="#FFFF00">Yellow italic</font></i>', yellow.text


def test_ttml_frames_and_nested_time_containers():
    doc = formats.parse(FRAMES_DFXP, "ttml")
    rate = 24 * 1000 / 1001
    first, nested, clock = doc.cues
    assert _close(first.start, 10 + 1 + 12 / rate) and _close(first.end, 12.0)
    assert _close(nested.start, 12.5) and _close(nested.end, 14.0)
    assert _close(clock.start, 70.25) and _close(clock.end, 71.5)


def test_ttml_writer_is_valid_imsc_and_roundtrips():
    doc = formats.parse(NETFLIX_TTML, "ttml")
    text = formats.render(doc, "ttml")
    root = ET.fromstring(text.encode("utf-8"))
    assert root.tag == "{http://www.w3.org/ns/ttml}tt"
    assert root.get("{http://www.w3.org/XML/1998/namespace}lang") == "ja"
    assert root.get("{http://www.w3.org/ns/ttml#parameter}profile").endswith("imsc1/text")
    back = formats.parse(text, "ttml")
    assert [(c.start, c.end, c.text, c.align) for c in back.cues] == \
        [(c.start, c.end, c.text, c.align) for c in doc.cues]


# ----------------------------------------------------------------- MicroDVD

MICRODVD = "{1}{1}23.976\n{24}{48}{y:i}Italic line|plain line\n{72}{96}{Y:i}All|italic\n{120}{}No end\n"


def test_microdvd_header_fps_and_styles():
    doc = formats.parse(MICRODVD, "microdvd")
    assert _close(doc.fps, 24000 / 1001), "23.976 snaps to the exact NTSC film rate"
    assert _close(doc.cues[0].start, 24 * 1001 / 24000)
    assert doc.cues[0].text == "<i>Italic line</i>\nplain line"
    assert doc.cues[1].text == "<i>All</i>\n<i>italic</i>"
    assert doc.cues[2].end > doc.cues[2].start


def test_microdvd_needs_a_frame_rate():
    try:
        formats.parse("{24}{48}No header\n", "microdvd")
    except TaskError as exc:
        assert "frame rate" in str(exc)
    else:
        raise AssertionError("parsed frames without knowing the rate")
    doc = formats.parse("{25}{50}At 25\n", "microdvd", fps=25)
    assert _close(doc.cues[0].start, 1.0) and _close(doc.cues[0].end, 2.0)
    try:
        formats.render(SubtitleDoc(cues=[Cue(1, 2, "x")]), "microdvd")
    except TaskError:
        pass
    else:
        raise AssertionError("wrote frames without a rate")


def test_microdvd_roundtrip_keeps_frames():
    doc = formats.parse(MICRODVD, "microdvd")
    text = formats.render(doc, "microdvd")
    assert text.splitlines()[0] == "{1}{1}23.976"
    assert text.splitlines()[1] == "{24}{48}{y:i}Italic line|plain line"
    assert formats.render(formats.parse(text, "microdvd"), "microdvd") == text


# ---------------------------------------------------------------------- SBV


def test_sbv_roundtrip():
    sbv = "0:00:01.000,0:00:02.500\nHello\nthere\n\n0:01:00.100,0:01:01.000\nNext[br]line\n"
    doc = formats.parse(sbv, "sbv")
    assert [c.text for c in doc.cues] == ["Hello\nthere", "Next\nline"]
    assert _close(doc.cues[1].start, 60.1)
    assert formats.parse(formats.render(doc, "sbv"), "sbv").cues[1].text == "Next\nline"


# ---------------------------------------------------------------- detection


def test_detect_format_by_content_then_extension():
    with Workspace() as ws:
        assert formats.detect_format(ws.write("a.txt", MESSY_SRT)) == "srt"
        assert formats.detect_format(ws.write("really_vtt.srt", VTT)) == "vtt"
        assert formats.detect_format(ws.write("x.ass", ASS)) == "ass"
        assert formats.detect_format(ws.write("netflix.xml", NETFLIX_TTML)) == "ttml"
        assert formats.detect_format(ws.write("frames.sub", MICRODVD)) == "microdvd"
        assert formats.detect_format(ws.write("bitmap.sub", b"\x00\x00\x01\xba" + b"\x00" * 64)) == "vobsub"
        assert formats.detect_format(ws.write("bitmap.idx", "# VobSub index file, v7\n")) == "vobsub"
        assert formats.detect_format(ws.write("x.sup", b"PG" + b"\x00" * 20)) == "pgs"
        assert formats.detect_format(ws.write("yt.sbv", "0:00:01.000,0:00:02.000\nhi\n")) == "sbv"
        assert formats.detect_format(ws.write("utf16.srt", MESSY_SRT.encode("utf-16"))) == "srt"
        assert formats.detect_format(ws.write("empty.srt", b"")) == "srt"
        assert formats.detect_format(ws.write("junk.bin", b"\x01\x02")) == "unknown"


JAPANESE = "1\n00:00:01,000 --> 00:00:02,000\n俺はお前を待っていたんだ。\n\n2\n00:00:03,000 --> 00:00:04,000\nどうしてここに来たの？\n"
CHINESE = "1\n00:00:01,000 --> 00:00:02,000\n我一直在等你。\n\n2\n00:00:03,000 --> 00:00:04,000\n这是我们的秘密，不要告诉别人。\n"
TRADITIONAL = "1\n00:00:01,000 --> 00:00:02,000\n你為什麼來這裡？\n\n2\n00:00:03,000 --> 00:00:04,000\n這是我們的秘密，不要告訴別人。\n"
KOREAN = "1\n00:00:01,000 --> 00:00:02,000\n나는 너를 기다리고 있었어.\n\n2\n00:00:03,000 --> 00:00:04,000\n왜 여기에 왔어요?\n"
FRENCH = "1\n00:00:01,000 --> 00:00:02,000\nJe t'attendais depuis très longtemps.\n\n2\n00:00:03,000 --> 00:00:04,000\nC’est notre secret – n’en parle à personne, garçon.\n"
RUSSIAN = "1\n00:00:01,000 --> 00:00:02,000\nЯ ждал тебя очень долго.\nПочему ты пришёл сюда?\n"
GREEK = "1\n00:00:01,000 --> 00:00:02,000\nΣε περίμενα πολύ καιρό.\nΓιατί ήρθες εδώ;\n"


def test_detect_encoding_legacy_code_pages():
    cases = [
        (JAPANESE, "shift_jis", "cp932", "ja"),
        (JAPANESE, "euc-jp", "euc-jp", "ja"),
        (CHINESE, "gb18030", "gb18030", "zh"),
        (CHINESE, "gb2312", "gb18030", "zh"),
        (TRADITIONAL, "big5", "big5", "zh"),
        (KOREAN, "cp949", "cp949", "ko"),
        (FRENCH, "cp1252", "cp1252", "fr"),
        (RUSSIAN, "cp1251", "cp1251", "ru"),
        (GREEK, "cp1253", "cp1253", "el"),
    ]
    for text, codec, expected, language in cases:
        data = text.encode(codec)
        assert formats.detect_encoding(data, language) == expected, (codec, "with hint")
        assert formats.detect_encoding(data) == expected, (codec, formats.detect_encoding(data))
        assert data.decode(formats.detect_encoding(data)) == text


def test_detect_encoding_boms_and_utf8():
    text = "1\n00:00:01,000 --> 00:00:02,000\nCafé ☕\n"
    assert formats.detect_encoding(text.encode("utf-8")) == "utf-8"
    assert formats.detect_encoding(b"\xef\xbb\xbf" + text.encode("utf-8")) == "utf-8-sig"
    assert formats.detect_encoding(text.encode("utf-16")) == "utf-16"
    assert formats.detect_encoding(text.encode("utf-32")) == "utf-32"
    assert formats.detect_encoding(text.encode("utf-16-le")) == "utf-16-le"


def test_read_decodes_normalises_and_guesses_language():
    with Workspace() as ws:
        path = ws.write("Movie.2019.ja.srt", JAPANESE.replace("\n", "\r\n").encode("cp932"))
        doc = formats.read(path)
        assert doc.meta["encoding"] == "cp932" and doc.language == "ja"
        assert doc.cues[1].text == "どうしてここに来たの？"
        forced = ws.write("Movie.fre.forced.srt", ("﻿" + FRENCH).encode("utf-8"))
        doc = formats.read(forced)
        assert doc.language == "fr" and doc.cues[0].text.startswith("Je")
        try:
            formats.read(ws.write("x.sup", b"PG" + b"\x00" * 20))
        except TaskError as exc:
            assert "OCR" in str(exc)
        else:
            raise AssertionError("read an image subtitle as text")


def test_language_from_filename():
    cases = {
        "Movie.ja.srt": "ja", "Movie.jpn.srt": "ja", "Movie.English.srt": "en", "Movie.pt-BR.srt": "pt",
        "Movie.eng.forced.srt": "en", "Movie.en.SDH.srt": "en", "Movie.zh-Hans.ass": "zh",
        "Movie.srt": None, "Movie.2019.1080p.srt": None, "Show.S01E02.srt": None, "Movie.hi.srt": "hi",
        "Movie.en.HI.srt": "en",
    }
    for name, expected in cases.items():
        assert formats.language_from_filename(name) == expected, (name, formats.language_from_filename(name))


def test_output_format_resolution():
    assert formats.resolve_output_format("same", "ass") == "ass"
    assert formats.resolve_output_format("same", "pgs") == "srt"
    assert formats.resolve_output_format("same", None) == "srt"
    assert formats.resolve_output_format("sub", "srt") == "microdvd"
    assert formats.resolve_output_format("sup", "srt") == "pgs"
    assert formats.extension_for("microdvd") == "sub" and formats.extension_for("vtt") == "vtt"
    try:
        formats.resolve_output_format("docx", "srt")
    except TaskError:
        pass
    else:
        raise AssertionError("accepted an unknown format")


def test_write_uses_extension_and_utf8():
    with Workspace() as ws:
        doc = formats.parse(JAPANESE, "srt")
        path = formats.write(doc, ws.path("out.vtt"))
        with open(path, encoding="utf-8") as handle:
            assert handle.read().startswith("WEBVTT")
        assert [n for n in os.listdir(ws.root) if n.endswith(".part")] == [], "staging file left behind"
        formats.write(doc, ws.path("out.sub"), fps=25)
        assert formats.read(ws.path("out.sub")).cues[1].text == "どうしてここに来たの？"


def test_convert_task_srt_to_ass_and_microdvd():
    from audiosync.subs.formats import run_convert_task

    with Workspace() as ws:
        source = ws.write("Movie.srt", MESSY_SRT)
        job = {"id": "1", "task": "convert", "input": {"subtitle": {"path": source}},
               "options": {"fps": None}, "output": {"format": "ass", "dir": ws.root}}
        result = run_convert_task(job, _ctx(ws.root))
        out = result.outputs[0]
        assert out.path == ws.path("Movie.ass") and out.format == "ass", out.path
        assert formats.read(out.path).cues[0].align == 8
        assert result.report["cues"] == 3 and "SRT to ASS" in result.summary
        job["output"]["format"] = "sub"
        try:
            run_convert_task(job, _ctx(ws.root))
        except TaskError as exc:
            assert "frame rate" in str(exc)
        else:
            raise AssertionError("wrote MicroDVD without a rate")
        job["options"]["fps"] = 25
        out = run_convert_task(job, _ctx(ws.root)).outputs[0]
        assert out.path.endswith("Movie.sub")
        with open(out.path, encoding="utf-8") as handle:
            assert handle.readline().strip() == "{1}{1}25"
        job["output"]["format"] = "same"
        out = run_convert_task(job, _ctx(ws.root)).outputs[0]
        assert out.path == ws.path("Movie.converted.srt"), "never over the source"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
