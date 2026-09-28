"""Read and write every text subtitle format into and out of ``SubtitleDoc``.

SRT, ASS/SSA, WebVTT, TTML/DFXP/IMSC, MicroDVD and YouTube SBV all become
the one model in ``model.py``: float-second times and text with ``\\n``
between lines plus the small ``<i>/<b>/<u>/<font color>`` subset. Rounding to
a format's precision happens only in the writers, so a chain of retimes never
accumulates error.

Real-world files are messy, so the readers are lenient on purpose: SRTs with
dots for commas, missing or garbled indices, stray blank lines and trailing
``X1:`` coordinates; TTML timed in frames or in Netflix's 10 MHz ticks;
encodings that were never declared. A strict parser that rejects half the
subtitles people actually have would be worse than useless here.

ASS is the one format with more styling than the model holds. Its original
dialogue text stays in ``cue.raw`` and the untouched line in
``cue.meta["ass_line"]``, so a retimed-but-otherwise-untouched ASS file keeps
every override tag, karaoke timing and drawing exactly as authored.
"""

from __future__ import annotations

import html
import os
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, List, Optional, Tuple
from xml.sax.saxutils import escape as _xml_escape

from . import languages
from .model import Cue, SubtitleDoc, strip_tags
from .tasks import TaskError, TaskResult

TEXT_FORMATS = ("srt", "ass", "ssa", "vtt", "ttml", "microdvd", "sbv")

_EXTENSIONS = {
    "srt": "srt", "ass": "ass", "ssa": "ssa", "vtt": "vtt", "ttml": "ttml",
    "microdvd": "sub", "sbv": "sbv", "pgs": "sup", "vobsub": "idx",
}

_FORMAT_BY_EXT = {
    ".srt": "srt", ".ass": "ass", ".ssa": "ssa", ".vtt": "vtt", ".webvtt": "vtt",
    ".ttml": "ttml", ".dfxp": "ttml", ".xml": "ttml", ".sub": "microdvd",
    ".sbv": "sbv", ".sup": "pgs", ".idx": "vobsub",
}

#: Extensions a subtitle file arrives with; tracks.probe_file uses this too.
SUBTITLE_EXTENSIONS = frozenset(_FORMAT_BY_EXT)

NBSP = "\u00a0"


# ================================================================ time helpers


def _frac(digits: Optional[str]) -> float:
    """"5" -> 0.5, "50" -> 0.5, "500" -> 0.5: a decimal fraction, whatever
    the digit count, which is how a sloppy "00:00:01,5" was meant."""
    return int(digits) / (10 ** len(digits)) if digits else 0.0


def _split_ms(t: float, unit: int = 1000) -> Tuple[int, int, int, int]:
    """h, m, s, fraction-in-``unit`` with one rounding step, so 59.9996 s
    becomes 1:00.000 instead of 0:59.1000."""
    total = int(round(max(0.0, t) * unit))
    frac = total % unit
    seconds = total // unit
    return seconds // 3600, (seconds // 60) % 60, seconds % 60, frac


def _srt_time(t: float) -> str:
    h, m, s, ms = _split_ms(t)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(t: float) -> str:
    h, m, s, ms = _split_ms(t)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _sbv_time(t: float) -> str:
    h, m, s, ms = _split_ms(t)
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


def _ass_time(t: float) -> str:
    h, m, s, cs = _split_ms(t, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ============================================================== markup helpers
#
# The model's text is plain text plus <i>, <b>, <u>, <font color="#RRGGBB">.
# Writers that have their own styling syntax walk it as tokens.

_MARKUP_RE = re.compile(r"<(/?)(i|b|u|font)(\s[^>]*)?>", re.IGNORECASE)
_COLOR_ATTR_RE = re.compile(r"""color\s*=\s*["']?\s*([#\w(),.\s%]+?)\s*["']?(?:\s|/?$|>)""", re.IGNORECASE)

_NAMED_COLORS = {
    "white": "FFFFFF", "black": "000000", "red": "FF0000", "lime": "00FF00", "green": "008000",
    "blue": "0000FF", "yellow": "FFFF00", "cyan": "00FFFF", "aqua": "00FFFF", "magenta": "FF00FF",
    "fuchsia": "FF00FF", "gray": "808080", "grey": "808080", "silver": "C0C0C0", "orange": "FFA500",
    "purple": "800080", "maroon": "800000", "navy": "000080", "olive": "808000", "teal": "008080",
}


def _color_hex(value: Optional[str]) -> Optional[str]:
    """``#rrggbb``/``#rrggbbaa``/``#rgb``/``rgb(r,g,b)``/a CSS name -> ``RRGGBB``."""
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("#"):
        digits = value[1:]
        if len(digits) == 3 and re.fullmatch(r"[0-9a-f]{3}", digits):
            return "".join(c * 2 for c in digits).upper()
        if len(digits) in (6, 8) and re.fullmatch(r"[0-9a-f]+", digits):
            return digits[:6].upper()
        return None
    match = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", value)
    if match:
        return "".join(f"{min(255, int(v)):02X}" for v in match.groups())
    return _NAMED_COLORS.get(value)


def _markup_tokens(text: str) -> Iterator[Tuple[str, str, Optional[str]]]:
    """("text", s, None) / ("open", tag, colour-or-None) / ("close", tag, None)."""
    pos = 0
    for match in _MARKUP_RE.finditer(text):
        if match.start() > pos:
            yield "text", text[pos:match.start()], None
        closing, tag, attrs = match.group(1), match.group(2).lower(), match.group(3) or ""
        if closing:
            yield "close", tag, None
        else:
            colour = None
            if tag == "font":
                found = _COLOR_ATTR_RE.search(attrs + " ")
                colour = _color_hex(found.group(1)) if found else None
            yield "open", tag, colour
        pos = match.end()
    if pos < len(text):
        yield "text", text[pos:], None


class _TagBuilder:
    """Builds model markup from on/off style switches, keeping tags nested.

    ASS and TTML switch styles on and off in any order (``{\\i1}a{\\b1}b{\\i0}c``);
    SRT players want properly nested tags, so closing a tag that is not the
    innermost closes and reopens the ones inside it.
    """

    def __init__(self) -> None:
        self.parts: List[str] = []
        self.stack: List[Tuple[str, Optional[str]]] = []

    @staticmethod
    def _open_str(tag: str, colour: Optional[str]) -> str:
        return f'<font color="#{colour}">' if tag == "font" else f"<{tag}>"

    def is_open(self, tag: str) -> bool:
        return any(name == tag for name, _ in self.stack)

    def open(self, tag: str, colour: Optional[str] = None) -> None:
        if tag == "font":
            self.close("font")
        elif self.is_open(tag):
            return
        self.stack.append((tag, colour))
        self.parts.append(self._open_str(tag, colour))

    def close(self, tag: str) -> None:
        if not self.is_open(tag):
            return
        reopen: List[Tuple[str, Optional[str]]] = []
        while self.stack:
            name, colour = self.stack.pop()
            self.parts.append(f"</{name}>")
            if name == tag:
                break
            reopen.append((name, colour))
        for name, colour in reversed(reopen):
            self.stack.append((name, colour))
            self.parts.append(self._open_str(name, colour))

    def close_all(self) -> None:
        while self.stack:
            self.parts.append(f"</{self.stack.pop()[0]}>")

    def text(self, value: str) -> None:
        if value:
            self.parts.append(value)

    def result(self) -> str:
        self.close_all()
        out = "".join(self.parts)
        # Drop tag pairs left empty by a switch that changed nothing visible.
        previous = None
        while previous != out:
            previous = out
            out = re.sub(r"<(i|b|u)></\1>|<font color=\"#[0-9A-F]{6}\"></font>", "", out)
        return out


def _only_markup(text: str) -> str:
    """Keep the model's tags, drop any other HTML-ish tag."""
    return re.sub(r"</?(?!(?:i|b|u|font)\b)[a-zA-Z][^>]*>", "", text)


# ===================================================================== SRT

_SRT_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\s*[,.]\s*(\d{1,3}))?"
_SRT_TIMING_RE = re.compile(rf"^\s*{_SRT_TIME}\s*-+\s*>\s*{_SRT_TIME}(.*)$")
_AN_TAG_RE = re.compile(r"\{\\(an|a)(\d{1,2})\}")
_BRACE_TAG_RE = re.compile(r"\{\\[^}]*\}")


def _srt_seconds(h: Optional[str], m: str, s: str, frac: Optional[str]) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + _frac(frac)


def _parse_srt(text: str) -> SubtitleDoc:
    lines = text.split("\n")
    timings = [(i, m) for i, m in ((i, _SRT_TIMING_RE.match(line)) for i, line in enumerate(lines)) if m]
    cues: List[Cue] = []
    for n, (i, match) in enumerate(timings):
        g = match.groups()
        start = _srt_seconds(*g[0:4])
        end = _srt_seconds(*g[4:8])
        stop = timings[n + 1][0] if n + 1 < len(timings) else len(lines)
        body = [line.rstrip() for line in lines[i + 1:stop]]
        while body and not body[-1].strip():
            body.pop()
        # The next cue's index sits just before its timing line; with the
        # blank separator missing it would otherwise read as dialogue.
        if n + 1 < len(timings) and body and body[-1].strip().isdigit():
            body.pop()
        body = [line for line in body if line.strip()]
        raw = "\n".join(body)
        align = None
        found = _AN_TAG_RE.search(raw)
        if found:
            value = int(found.group(2))
            align = value if found.group(1) == "an" and 1 <= value <= 9 else _legacy_align(value)
        body_text = _BRACE_TAG_RE.sub("", raw)
        cues.append(Cue(start=start, end=end, text=body_text.strip(), align=align))
    return SubtitleDoc(cues=cues, source_format="srt")


def _render_srt(doc: SubtitleDoc) -> str:
    out: List[str] = []
    n = 0
    for cue in _for_writing(doc):
        n += 1
        text = cue.text
        if cue.align and cue.align != 2:
            text = f"{{\\an{cue.align}}}" + text
        out.append(f"{n}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{text}\n")
    return "\n".join(out)


def _for_writing(doc: SubtitleDoc) -> List[Cue]:
    """Cues in time order, without the ones that have nothing to show
    (ASS drawings, emptied lines) -- an empty SRT cue breaks many players."""
    return sorted((c for c in doc.cues if strip_tags(c.text).strip()), key=lambda c: (c.start, c.end))


# ================================================================== ASS/SSA

_ASS_EVENT_FORMAT = ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
_SSA_EVENT_FORMAT = ["marked", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
_ASS_TIME_RE = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})(?:[.:](\d+))?\s*$")

# Longest names first so \bord is not read as \b and \iclip not as \i.
_ASS_TAG_NAMES = sorted(
    ["1c", "2c", "3c", "4c", "1a", "2a", "3a", "4a", "alpha", "an", "a", "blur", "bord", "be", "b",
     "clip", "c", "fade", "fad", "fax", "fay", "fe", "fn", "frx", "fry", "frz", "fr", "fscx", "fscy",
     "fsp", "fs", "iclip", "i", "kf", "ko", "k", "K", "move", "org", "pbo", "pos", "p", "q", "r",
     "shad", "s", "t", "u", "xbord", "ybord", "xshad", "yshad"],
    key=len, reverse=True,
)


def _legacy_align(value: int) -> Optional[int]:
    """SSA ``\\a`` numbering (1-3 bottom, 5-7 top, 9-11 middle) -> numpad."""
    return {1: 1, 2: 2, 3: 3, 5: 7, 6: 8, 7: 9, 9: 4, 10: 5, 11: 6}.get(value)


def _parse_ass_time(value: str) -> float:
    match = _ASS_TIME_RE.match(value)
    if not match:
        raise ValueError(value)
    h, m, s, frac = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + _frac(frac)


def _split_ass_tags(block: str) -> List[Tuple[str, str]]:
    """``\\i1\\pos(1,2)\\t(0,50,\\b1)`` -> [(i, 1), (pos, (1,2)), (t, (0,50,\\b1))].

    Backslashes inside parentheses belong to the enclosing tag (\\t animates
    other tags), so the split tracks nesting.
    """
    pieces: List[str] = []
    current: Optional[str] = None
    depth = 0
    for ch in block:
        if ch == "\\" and depth == 0:
            if current is not None:
                pieces.append(current)
            current = ""
            continue
        if current is None:
            continue  # a comment in braces before any tag
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        current += ch
    if current is not None:
        pieces.append(current)
    out = []
    for piece in pieces:
        name = next((n for n in _ASS_TAG_NAMES if piece.startswith(n)), None)
        if name is None:
            continue
        out.append((name, piece[len(name):].strip()))
    return out


def _ass_colour(arg: str) -> Optional[str]:
    """``&HBBGGRR&`` (optionally with alpha ``&HAABBGGRR``) -> ``RRGGBB``."""
    match = re.search(r"&?H?([0-9A-Fa-f]{1,8})&?", arg)
    if not match:
        return None
    value = int(match.group(1), 16) & 0xFFFFFF
    blue, green, red = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    return f"{red:02X}{green:02X}{blue:02X}"


def ass_to_markup(raw: str) -> Tuple[str, Optional[int], bool]:
    """ASS dialogue text -> (model text, alignment, whether it had a drawing)."""
    builder = _TagBuilder()
    align: Optional[int] = None
    drawing = False
    had_drawing = False
    pos = 0
    for match in re.finditer(r"\{([^}]*)\}", raw):
        if match.start() > pos and not drawing:
            builder.text(_ass_plain(raw[pos:match.start()]))
        pos = match.end()
        for name, arg in _split_ass_tags(match.group(1)):
            if name == "i":
                builder.open("i") if arg[:1] == "1" else builder.close("i")
            elif name == "b":
                on = arg.isdigit() and (arg == "1" or int(arg) >= 600)
                builder.open("b") if on else builder.close("b")
            elif name == "u":
                builder.open("u") if arg[:1] == "1" else builder.close("u")
            elif name in ("c", "1c"):
                colour = _ass_colour(arg) if arg else None
                builder.open("font", colour) if colour else builder.close("font")
            elif name == "an" and arg.isdigit() and 1 <= int(arg) <= 9:
                align = align or int(arg)  # the first one wins, as in libass
            elif name == "a" and arg.isdigit():
                align = align or _legacy_align(int(arg))
            elif name == "p" and arg.isdigit():
                drawing = int(arg) > 0
                had_drawing = had_drawing or drawing
            elif name == "r":
                builder.close_all()
    if pos < len(raw) and not drawing:
        builder.text(_ass_plain(raw[pos:]))
    return builder.result(), align, had_drawing


def _ass_plain(segment: str) -> str:
    return segment.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", NBSP)


def markup_to_ass(text: str, align: Optional[int] = None) -> str:
    """Model text -> ASS dialogue text with override tags."""
    parts: List[Tuple[bool, str]] = []  # (is_tag, value)
    if align and align != 2:
        parts.append((True, f"\\an{align}"))
    colours: List[str] = []
    switches = {"i": "\\i", "b": "\\b", "u": "\\u"}
    for kind, value, colour in _markup_tokens(text):
        if kind == "text":
            parts.append((False, value.replace("\n", "\\N")))
        elif value in switches:
            parts.append((True, f"{switches[value]}{1 if kind == 'open' else 0}"))
        elif value == "font" and kind == "open":
            colours.append(colour or "")
            if colour:
                parts.append((True, f"\\c&H{colour[4:6]}{colour[2:4]}{colour[0:2]}&"))
        elif value == "font" and kind == "close" and colours:
            colours.pop()
            previous = next((c for c in reversed(colours) if c), None)
            parts.append((True, f"\\c&H{previous[4:6]}{previous[2:4]}{previous[0:2]}&" if previous else "\\c"))
    # Adjacent switches share one override block: {\an8\i1} not {\an8}{\i1}.
    out: List[str] = []
    tags: List[str] = []
    for is_tag, value in parts + [(False, "")]:
        if is_tag:
            tags.append(value)
            continue
        if tags:
            out.append("{" + "".join(tags) + "}")
            tags = []
        out.append(value)
    return "".join(out)


def _parse_ass(text: str, fmt: str) -> SubtitleDoc:
    header: List[str] = []
    section: Optional[str] = None
    event_format: Optional[List[str]] = None
    cues: List[Cue] = []
    comments: List[Tuple[int, str]] = []
    play_x = play_y = None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.lower()
            if section != "[events]":
                if header and header[-1].strip():
                    header.append("")
                header.append(stripped)
            continue
        if section is None:
            continue
        if section != "[events]":
            header.append(line.rstrip())
            if section == "[script info]":
                key, _, value = stripped.partition(":")
                if key.strip().lower() == "playresx" and value.strip().isdigit():
                    play_x = int(value.strip())
                elif key.strip().lower() == "playresy" and value.strip().isdigit():
                    play_y = int(value.strip())
            continue
        lowered = stripped.lower()
        if lowered.startswith("format:"):
            event_format = [f.strip().lower() for f in stripped[7:].split(",")]
            continue
        kind = "dialogue" if lowered.startswith("dialogue:") else "comment" if lowered.startswith("comment:") else None
        if kind is None:
            continue
        if kind == "comment":
            comments.append((len(cues), line.rstrip()))
            continue
        names = event_format or (_SSA_EVENT_FORMAT if fmt == "ssa" else _ASS_EVENT_FORMAT)
        body = stripped.split(":", 1)[1].lstrip()
        values = body.split(",", len(names) - 1)
        if len(values) < len(names):
            continue
        fields = dict(zip(names, values))
        try:
            start = _parse_ass_time(fields.get("start", ""))
            end = _parse_ass_time(fields.get("end", ""))
        except ValueError:
            continue
        raw = fields.get("text", "")
        plain, align, drawing = ass_to_markup(raw)
        style = (fields.get("style") or "").strip() or None
        speaker = (fields.get("name") or "").strip() or None
        cue = Cue(start=start, end=end, text=plain, style=style, raw=raw, align=align, speaker=speaker)
        cue.meta.update({
            "ass_fields": {k: v for k, v in fields.items() if k not in ("start", "end", "text")},
            "ass_line": line.rstrip(),
            "ass_text": plain,
            "ass_align": align,
            "ass_order": len(cues),
            "ass_src": (start, end, plain, style, speaker, align),
        })
        if drawing and not strip_tags(plain).strip():
            cue.text = ""
            cue.meta["ass_text"] = ""
            cue.meta["ass_src"] = (start, end, "", style, speaker, align)
            cue.meta["drawing"] = True
        cues.append(cue)
    while header and not header[-1].strip():
        header.pop()
    doc = SubtitleDoc(cues=cues, source_format=fmt, ass_header="\n".join(header) or None)
    if play_x and play_y:
        doc.play_res = (play_x, play_y)
    if comments:
        doc.meta["ass_comments"] = comments
    return doc


def _default_ass_header(doc: SubtitleDoc, ssa: bool) -> str:
    width, height = doc.play_res or (1920, 1080)
    scale = height / 1080.0
    size = max(8, round(height * 0.05))
    outline = round(max(0.5, 2.5 * scale), 1)
    shadow = round(max(0.0, 1.0 * scale), 1)
    margin_v = round(height * 0.045)
    margin_h = round(width * 0.02)
    title = doc.title or "Subsync"
    if ssa:
        return "\n".join([
            "[Script Info]", f"Title: {title}", "ScriptType: v4.00", "WrapStyle: 0",
            f"PlayResX: {width}", f"PlayResY: {height}", "",
            "[V4 Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, "
            "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding",
            f"Style: Default,Arial,{size},16777215,255,0,-2147483648,0,0,1,{outline},{shadow},2,"
            f"{margin_h},{margin_h},{margin_v},0,1",
        ])
    return "\n".join([
        "[Script Info]", f"Title: {title}", "ScriptType: v4.00+", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "YCbCr Matrix: None", f"PlayResX: {width}", f"PlayResY: {height}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Arial,{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,"
        f"{outline},{shadow},2,{margin_h},{margin_h},{margin_v},1",
    ])


def _render_ass(doc: SubtitleDoc, fmt: str) -> str:
    ssa = fmt == "ssa"
    header = doc.ass_header or _default_ass_header(doc, ssa)
    styles = re.findall(r"^\s*Style:\s*([^,\n]*)", header, flags=re.MULTILINE | re.IGNORECASE)
    styles = [s.strip() for s in styles]
    fallback = "Default" if "Default" in styles or not styles else styles[0]
    same_format = doc.source_format == fmt
    names = _SSA_EVENT_FORMAT if ssa else _ASS_EVENT_FORMAT
    first = "Marked" if ssa else "Layer"
    out = [header, "", "[Events]",
           f"Format: {first}, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    comments = list(doc.meta.get("ass_comments") or []) if doc.source_format in ("ass", "ssa") else []

    def flush_comments(before: Optional[int]) -> None:
        while comments and (before is None or comments[0][0] <= before):
            out.append(comments.pop(0)[1])

    for cue in doc.cues:
        meta = cue.meta
        order = meta.get("ass_order")
        if order is not None:
            flush_comments(order)
        if meta.get("drawing") and not same_format and not cue.raw:
            continue
        src = meta.get("ass_src")
        if same_format and meta.get("ass_line") and src == (cue.start, cue.end, cue.text, cue.style, cue.speaker, cue.align):
            out.append(meta["ass_line"])
            continue
        if cue.raw is not None and cue.text == meta.get("ass_text"):
            body = cue.raw
            if cue.align != meta.get("ass_align"):
                body = re.sub(r"\\an?\d+", "", body).replace("{}", "")
                if cue.align:
                    body = f"{{\\an{cue.align}}}" + body
        else:
            if not strip_tags(cue.text).strip():
                continue
            body = markup_to_ass(cue.text, cue.align)
        fields = dict(meta.get("ass_fields") or {})
        style = cue.style if cue.style in styles else fallback
        speaker = cue.speaker or ""
        values = []
        for name in names:
            if name == "start":
                values.append(_ass_time(cue.start))
            elif name == "end":
                values.append(_ass_time(cue.end))
            elif name == "style":
                values.append(style)
            elif name == "name":
                values.append(speaker.replace(",", ";"))
            elif name == "text":
                values.append(body)
            elif name == "marked":
                values.append(fields.get("marked", "Marked=0"))
            elif name == "effect":
                values.append(fields.get("effect", ""))
            else:
                values.append(fields.get(name, "0"))
        out.append("Dialogue: " + ",".join(values))
    flush_comments(None)
    return "\n".join(out) + "\n"


# =================================================================== WebVTT

_VTT_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_VTT_TIMING_RE = re.compile(rf"^\s*{_VTT_TIME}\s*-->\s*{_VTT_TIME}(.*)$")


def _vtt_align(settings: str) -> Optional[int]:
    values = dict(part.split(":", 1) for part in settings.split() if ":" in part)
    row = 0  # 0 bottom, 1 middle, 2 top
    line = values.get("line", "").split(",")[0].strip()
    if line.endswith("%"):
        try:
            pct = float(line[:-1])
            row = 2 if pct < 35 else 1 if pct < 65 else 0
        except ValueError:
            pass
    elif line:
        try:
            row = 2 if float(line) >= 0 else 0
        except ValueError:
            pass
    col = {"start": 0, "left": 0, "end": 2, "right": 2}.get(values.get("align", "").strip(), 1)
    align = (1, 4, 7)[row] + col
    return None if align == 2 else align


def _vtt_text(body: str) -> Tuple[str, Optional[str]]:
    speaker = None
    found = re.search(r"<v(?:\.[\w.-]+)?\s+([^>]+)>", body)
    if found:
        speaker = found.group(1).strip()
    body = re.sub(r"<rt>.*?</rt>|<rp>.*?</rp>", "", body, flags=re.DOTALL)
    body = re.sub(r"</?(?:v|c|ruby|lang|rt|rp)(?:[.\s][^>]*)?>", "", body)
    body = re.sub(r"<\d[\d:.]*>", "", body)  # karaoke timestamps
    body = re.sub(r"<(/?)(i|b|u)\.[^>]*>", r"<\1\2>", body)  # <i.loud> -> <i>
    return html.unescape(_only_markup(body)), speaker


def _parse_vtt(text: str) -> SubtitleDoc:
    cues: List[Cue] = []
    blocks = re.split(r"\n\s*\n", text)
    for index, block in enumerate(blocks):
        lines = block.strip("\n").split("\n")
        if not lines or not lines[0].strip():
            continue
        head = lines[0].strip()
        if index == 0 and head.startswith("WEBVTT"):
            # Header text may continue on following lines until the blank.
            if not any("-->" in line for line in lines):
                continue
        if head.startswith(("NOTE", "STYLE", "REGION")):
            continue
        timing_at = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_at is None:
            continue
        match = _VTT_TIMING_RE.match(lines[timing_at])
        if not match:
            continue
        g = match.groups()
        start = _srt_seconds(*g[0:4])
        end = _srt_seconds(*g[4:8])
        text_body, speaker = _vtt_text("\n".join(lines[timing_at + 1:]))
        text_body = "\n".join(line.strip() for line in text_body.split("\n") if line.strip())
        cues.append(Cue(start=start, end=end, text=text_body, align=_vtt_align(g[8] or ""), speaker=speaker))
    return SubtitleDoc(cues=cues, source_format="vtt")


def _vtt_escape(text: str) -> str:
    """Escape &, < and > except around the tags WebVTT itself understands."""
    out = []
    pos = 0
    for match in re.finditer(r"</?(?:i|b|u)>", text):
        out.append(html.escape(text[pos:match.start()], quote=False))
        out.append(match.group(0))
        pos = match.end()
    out.append(html.escape(text[pos:], quote=False))
    return "".join(out)


def _render_vtt(doc: SubtitleDoc) -> str:
    out = ["WEBVTT", ""]
    for cue in _for_writing(doc):
        settings = ""
        if cue.align and cue.align != 2:
            row = (cue.align - 1) // 3
            col = (cue.align - 1) % 3
            parts = []
            if row == 2:
                parts.append("line:0")
            elif row == 1:
                parts.append("line:50%")
            if col != 1:
                parts.append("align:" + ("start" if col == 0 else "end"))
            settings = " " + " ".join(parts) if parts else ""
        text = re.sub(r"</?font[^>]*>", "", cue.text)
        text = _vtt_escape(text)
        if cue.speaker:
            text = f"<v {cue.speaker}>" + text
        out.append(f"{_vtt_time(cue.start)} --> {_vtt_time(cue.end)}{settings}")
        out.append(text)
        out.append("")
    return "\n".join(out)


# ====================================================================== SBV

_SBV_TIMING_RE = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})\.(\d{1,3})\s*,\s*(\d+):(\d{1,2}):(\d{1,2})\.(\d{1,3})\s*$")


def _parse_sbv(text: str) -> SubtitleDoc:
    cues: List[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip("\n").split("\n")
        if not lines:
            continue
        match = _SBV_TIMING_RE.match(lines[0])
        if not match:
            continue
        g = match.groups()
        start = _srt_seconds(*g[0:4])
        end = _srt_seconds(*g[4:8])
        body = "\n".join(line.strip() for line in lines[1:] if line.strip()).replace("[br]", "\n")
        cues.append(Cue(start=start, end=end, text=body))
    return SubtitleDoc(cues=cues, source_format="sbv")


def _render_sbv(doc: SubtitleDoc) -> str:
    out = []
    for cue in _for_writing(doc):
        out.append(f"{_sbv_time(cue.start)},{_sbv_time(cue.end)}\n{strip_tags(cue.text)}\n")
    return "\n".join(out)


# ================================================================= MicroDVD

_MDVD_LINE_RE = re.compile(r"^\s*\{(\d+)\}\{(\d*)\}(.*)$")


def _format_fps(fps: float) -> str:
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def _parse_microdvd(text: str, fps: Optional[float]) -> SubtitleDoc:
    from .retime import snap_fps  # retime imports nothing from here at load

    entries = []
    header_fps: Optional[float] = None
    for line in text.split("\n"):
        match = _MDVD_LINE_RE.match(line)
        if not match:
            continue
        start, end, body = int(match.group(1)), match.group(2), match.group(3)
        # "{1}{1}23.976" (or {0}{0}) as the first line declares the rate.
        if not entries and header_fps is None and end and int(end) == start and start <= 1:
            try:
                header_fps = float(body.strip().replace(",", "."))
                continue
            except ValueError:
                pass
        entries.append((start, int(end) if end else None, body))
    rate = fps or header_fps
    if not rate or rate <= 0:
        raise TaskError(
            "This MicroDVD (.sub) file is timed in frames and does not say its frame rate. "
            "Choose the frame rate of the video it was made for."
        )
    rate = float(snap_fps(rate))
    cues: List[Cue] = []
    for n, (start, end, body) in enumerate(entries):
        if end is None:
            nxt = entries[n + 1][0] if n + 1 < len(entries) else start + int(rate * 3)
            end = max(start + 1, nxt)
        cues.append(Cue(start=start / rate, end=end / rate, text=_microdvd_text(body)))
    doc = SubtitleDoc(cues=cues, source_format="microdvd", fps=rate)
    doc.meta["frames"] = [(s, e) for s, e, _ in entries]
    return doc


def _microdvd_text(body: str) -> str:
    whole: Dict[str, str] = {}
    lead = re.match(r"^((?:\{[A-Z]:[^}]*\})*)", body)
    if lead:
        for code in re.findall(r"\{([A-Z]):([^}]*)\}", lead.group(1)):
            whole[code[0].lower()] = code[1]
    lines = []
    for line in body.split("|"):
        local = dict(whole)
        for key, value in re.findall(r"\{([a-zA-Z]):([^}]*)\}", line):
            local[key.lower()] = value
        plain = re.sub(r"\{[^}]*\}", "", line).strip()
        if not plain:
            continue
        flags = (local.get("y") or "").lower()
        if "u" in flags:
            plain = f"<u>{plain}</u>"
        if "i" in flags:
            plain = f"<i>{plain}</i>"
        if "b" in flags:
            plain = f"<b>{plain}</b>"
        colour = local.get("c")
        if colour and colour.startswith("$") and len(colour) >= 7:
            bgr = colour[1:7]
            plain = f'<font color="#{(bgr[4:6] + bgr[2:4] + bgr[0:2]).upper()}">{plain}</font>'
        lines.append(plain)
    return "\n".join(lines)


def _render_microdvd(doc: SubtitleDoc, fps: Optional[float]) -> str:
    rate = fps or doc.fps
    if not rate:
        raise TaskError("MicroDVD (.sub) is timed in frames; choose the frame rate of the video.")
    out = [f"{{1}}{{1}}{_format_fps(rate)}"]
    for cue in _for_writing(doc):
        parts = []
        for line in cue.text.split("\n"):
            stripped = line.strip()
            italic = stripped.lower().startswith("<i>") and stripped.lower().endswith("</i>") \
                and stripped.lower().count("<i>") == 1
            plain = strip_tags(stripped)
            parts.append(("{y:i}" if italic else "") + plain)
        start = int(round(cue.start * rate))
        end = max(start + 1, int(round(cue.end * rate)))
        out.append(f"{{{start}}}{{{end}}}" + "|".join(parts))
    return "\n".join(out) + "\n"


# ===================================================================== TTML

_CLOCK_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+)|:(\d+)(?:\.(\d+))?)?$")
_OFFSET_RE = re.compile(r"^(\d+(?:\.\d+)?)(h|ms|m|s|f|t)$")


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _attrs(el: ET.Element) -> Dict[str, str]:
    return {_local(k): v for k, v in el.attrib.items()}


class _TtmlClock:
    def __init__(self, root_attrs: Dict[str, str]) -> None:
        frame_rate = float(root_attrs.get("frameRate") or 30)
        multiplier = (root_attrs.get("frameRateMultiplier") or "1 1").split()
        try:
            frame_rate *= float(multiplier[0]) / float(multiplier[1])
        except (IndexError, ValueError, ZeroDivisionError):
            pass
        self.frame_rate = frame_rate
        self.sub_frame_rate = float(root_attrs.get("subFrameRate") or 1)
        if root_attrs.get("tickRate"):
            self.tick_rate = float(root_attrs["tickRate"])
        elif root_attrs.get("frameRate"):
            self.tick_rate = frame_rate * self.sub_frame_rate
        else:
            self.tick_rate = 1.0

    def seconds(self, value: str) -> Optional[float]:
        value = value.strip()
        match = _CLOCK_RE.match(value)
        if match:
            h, m, s, frac, frames, sub = match.groups()
            total = int(h) * 3600 + int(m) * 60 + int(s) + _frac(frac)
            if frames:
                total += (int(frames) + (int(sub) / self.sub_frame_rate if sub else 0)) / self.frame_rate
            return total
        match = _OFFSET_RE.match(value)
        if match:
            number, unit = float(match.group(1)), match.group(2)
            return {
                "h": number * 3600, "m": number * 60, "s": number, "ms": number / 1000,
                "f": number / self.frame_rate, "t": number / self.tick_rate,
            }[unit]
        return None


def _percent_pair(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if not value:
        return None
    parts = value.split()
    if len(parts) != 2 or not all(p.endswith("%") for p in parts):
        return None
    try:
        return float(parts[0][:-1]), float(parts[1][:-1])
    except ValueError:
        return None


def _parse_ttml(text: str) -> SubtitleDoc:
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise TaskError(f"This TTML/DFXP file is not valid XML: {exc}") from exc
    root_attrs = _attrs(root)
    clock = _TtmlClock(root_attrs)

    # Styles and regions by xml:id; a style may reference other styles.
    raw_styles: Dict[str, Dict[str, str]] = {}
    regions: Dict[str, Dict[str, str]] = {}
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "style" and "id" in _attrs(el):
            raw_styles[_attrs(el)["id"]] = _attrs(el)
        elif tag == "region" and "id" in _attrs(el):
            attrs = _attrs(el)
            for child in el:
                if _local(child.tag) == "style":
                    attrs = {**_attrs(child), **attrs}
            regions[attrs["id"]] = attrs

    def resolve_style(ids: str, seen: Tuple[str, ...] = ()) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for sid in ids.split():
            if sid in seen or sid not in raw_styles:
                continue
            style = raw_styles[sid]
            if style.get("style"):
                out.update(resolve_style(style["style"], seen + (sid,)))
            out.update({k: v for k, v in style.items() if k not in ("id", "style")})
        return out

    def effective(el: ET.Element, inherited: Dict[str, str]) -> Dict[str, str]:
        attrs = _attrs(el)
        out = dict(inherited)
        if attrs.get("style"):
            out.update(resolve_style(attrs["style"]))
        out.update({k: v for k, v in attrs.items() if k not in ("style", "begin", "end", "dur", "id", "region")})
        if attrs.get("region"):
            out["_region"] = attrs["region"]
        return out

    cues: List[Cue] = []

    def text_of(el: ET.Element, styles: Dict[str, str], parent: Dict[str, str], builder: _TagBuilder) -> None:
        applied = _apply_ttml_style(styles, parent, builder)
        if el.text:
            builder.text(re.sub(r"\s+", " ", el.text))
        for child in el:
            tag = _local(child.tag)
            if tag == "br":
                builder.text("\n")
            elif tag == "span":
                text_of(child, effective(child, styles), styles, builder)
            elif tag not in ("metadata", "set", "image"):
                text_of(child, styles, styles, builder)
            if child.tail:
                builder.text(re.sub(r"\s+", " ", child.tail))
        for tag in reversed(applied):
            builder.close(tag)

    def emit(el: ET.Element, start: float, end: float, styles: Dict[str, str], parent: Dict[str, str]) -> None:
        builder = _TagBuilder()
        text_of(el, styles, parent, builder)
        body = "\n".join(line.strip() for line in builder.result().split("\n"))
        body = "\n".join(line for line in body.split("\n") if strip_tags(line).strip())
        if not strip_tags(body).strip():
            return
        cues.append(Cue(start=start, end=end, text=body, align=_ttml_align(styles, regions)))

    def walk(el: ET.Element, offset: float, parent_end: Optional[float], inherited: Dict[str, str]) -> None:
        attrs = _attrs(el)
        begin = clock.seconds(attrs["begin"]) if attrs.get("begin") else None
        start = offset + (begin or 0.0)
        end: Optional[float] = None
        if attrs.get("end"):
            value = clock.seconds(attrs["end"])
            end = offset + value if value is not None else None
        elif attrs.get("dur"):
            value = clock.seconds(attrs["dur"])
            end = start + value if value is not None else None
        if end is None:
            end = parent_end
        elif parent_end is not None:
            end = min(end, parent_end)
        styles = effective(el, inherited)
        tag = _local(el.tag)
        if tag == "p":
            if end is not None:
                emit(el, start, end, styles, inherited)
                return
            # Timing on the spans instead of the paragraph: one cue per span.
            for span in el.iter():
                span_attrs = _attrs(span)
                if span is el or _local(span.tag) != "span" or not span_attrs.get("begin"):
                    continue
                s = clock.seconds(span_attrs["begin"])
                e = clock.seconds(span_attrs.get("end", "")) if span_attrs.get("end") else None
                if s is None or e is None:
                    continue
                emit(span, start + s, start + e, effective(span, styles), styles)
            return
        for child in el:
            if _local(child.tag) in ("head", "metadata"):
                continue
            walk(child, start, end, styles)

    for child in root:
        if _local(child.tag) == "body":
            walk(child, 0.0, None, {})
    lang = root_attrs.get("lang")
    doc = SubtitleDoc(cues=cues, source_format="ttml", language=languages.normalize(lang) or (lang or None))
    return doc


def _apply_ttml_style(styles: Dict[str, str], parent: Dict[str, str], builder: _TagBuilder) -> List[str]:
    """Open the model tags this element's style adds over its parent's."""
    applied: List[str] = []

    def flag(style: Dict[str, str], key: str, on: str) -> bool:
        return on in (style.get(key) or "").lower()

    for key, on, tag in (("fontWeight", "bold", "b"), ("fontStyle", "italic", "i"),
                         ("fontStyle", "oblique", "i"), ("textDecoration", "underline", "u")):
        if flag(styles, key, on) and not flag(parent, key, on) and not builder.is_open(tag):
            builder.open(tag)
            applied.append(tag)
    colour = _color_hex(styles.get("color"))
    if colour and colour != "FFFFFF" and colour != _color_hex(parent.get("color")):
        builder.open("font", colour)
        applied.append("font")
    return applied


def _ttml_align(styles: Dict[str, str], regions: Dict[str, Dict[str, str]]) -> Optional[int]:
    region = dict(regions.get(styles.get("_region", ""), {}))
    region.update({k: v for k, v in styles.items() if k in ("displayAlign", "origin", "extent")})
    display = (region.get("displayAlign") or "").lower()
    origin = _percent_pair(region.get("origin"))
    extent = _percent_pair(region.get("extent"))
    top = False
    if display == "before":
        top = origin is None or origin[1] < 50
    elif not display and origin is not None:
        centre = origin[1] + (extent[1] / 2 if extent else 0)
        top = centre < 40
    text_align = (styles.get("textAlign") or region.get("textAlign") or "").lower()
    col = 0 if text_align in ("left", "start") else 2 if text_align in ("right", "end") else 1
    if not top and col == 1:
        return None
    return (7 if top else 1) + col


def _render_ttml(doc: SubtitleDoc) -> str:
    lang = doc.language or ""
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttp="http://www.w3.org/ns/ttml#parameter" '
        'xmlns:tts="http://www.w3.org/ns/ttml#styling" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" '
        'ttp:profile="http://www.w3.org/ns/ttml/profile/imsc1/text" ttp:timeBase="media" '
        f'xml:lang="{_xml_escape(lang, {chr(34): "&quot;"})}">',
        "  <head>",
        "    <styling>",
        '      <style xml:id="default" tts:fontFamily="proportionalSansSerif" tts:fontSize="100%" '
        'tts:color="white" tts:textAlign="center" tts:backgroundColor="transparent"/>',
        "    </styling>",
        "    <layout>",
        '      <region xml:id="bottom" tts:origin="10% 10%" tts:extent="80% 80%" tts:displayAlign="after"/>',
        '      <region xml:id="top" tts:origin="10% 10%" tts:extent="80% 80%" tts:displayAlign="before"/>',
        "    </layout>",
        "  </head>",
        '  <body style="default">',
        "    <div>",
    ]
    for cue in _for_writing(doc):
        region = "top" if cue.align in (7, 8, 9) else "bottom"
        text_align = ""
        if cue.align in (1, 4, 7):
            text_align = ' tts:textAlign="start"'
        elif cue.align in (3, 6, 9):
            text_align = ' tts:textAlign="end"'
        out.append(
            f'      <p begin="{_vtt_time(cue.start)}" end="{_vtt_time(cue.end)}" region="{region}"{text_align}>'
            f"{_markup_to_ttml(cue.text)}</p>"
        )
    out += ["    </div>", "  </body>", "</tt>", ""]
    return "\n".join(out)


def _markup_to_ttml(text: str) -> str:
    attrs = {"i": 'tts:fontStyle="italic"', "b": 'tts:fontWeight="bold"', "u": 'tts:textDecoration="underline"'}
    out: List[str] = []
    depth = 0
    for kind, value, colour in _markup_tokens(text):
        if kind == "text":
            out.append("<br/>".join(_xml_escape(part) for part in value.split("\n")))
        elif kind == "open":
            if value == "font":
                out.append(f'<span tts:color="#{colour}">' if colour else "<span>")
            else:
                out.append(f"<span {attrs[value]}>")
            depth += 1
        elif depth:
            out.append("</span>")
            depth -= 1
    out.extend("</span>" for _ in range(depth))
    return "".join(out)


# ================================================================ detection


def _decode_head(data: bytes) -> str:
    """Enough of a file, as text, to recognise its format."""
    for bom, codec in ((b"\xff\xfe\x00\x00", "utf-32-le"), (b"\x00\x00\xfe\xff", "utf-32-be"),
                       (b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")):
        if data.startswith(bom):
            return data[len(bom):].decode(codec, errors="ignore")
    utf16 = _utf16_without_bom(data)
    if utf16:
        return data.decode(utf16, errors="ignore")
    return data.decode("latin-1")


def sniff_format(text: str, ext: str = "") -> str:
    """Format from content, falling back to the extension."""
    head = text.lstrip("\ufeff \t\r\n")
    lowered = head[:4000].lower()
    if head.startswith("WEBVTT"):
        return "vtt"
    if "[script info]" in lowered or re.search(r"^\s*\[v4\+? styles\]", lowered, re.MULTILINE) or \
            re.search(r"^\s*dialogue:\s*", lowered, re.MULTILINE):
        if "[v4+ styles]" in lowered or "v4.00+" in lowered:
            return "ass"
        if "[v4 styles]" in lowered or "scripttype: v4.00" in lowered:
            return "ssa"
        return "ssa" if ext == ".ssa" else "ass"
    if lowered.startswith("<") and re.search(r"<(?:\w+:)?tt[\s>]", lowered[:3000]):
        return "ttml"
    if lowered.startswith("# vobsub index file"):
        return "vobsub"
    first = next((line for line in head.split("\n") if line.strip()), "")
    if _MDVD_LINE_RE.match(first):
        return "microdvd"
    if _SBV_TIMING_RE.match(first):
        return "sbv"
    for line in head.split("\n")[:50]:
        if _SRT_TIMING_RE.match(line):
            return "srt"
    # Any XML could be named .xml; only the content says it is TTML.
    fmt = _FORMAT_BY_EXT.get(ext) if ext != ".xml" else None
    return fmt if fmt and fmt not in ("pgs", "vobsub") else "unknown"


def detect_format(path: str) -> str:
    """One of TEXT_FORMATS, "pgs", "vobsub" or "unknown". Content first,
    extension second: a .sub may be MicroDVD text or a VobSub bitmap stream,
    and plenty of .srt files are really WebVTT."""
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as handle:
            data = handle.read(65536)
    except OSError:
        return _FORMAT_BY_EXT.get(ext, "unknown")
    if data.startswith(b"PG") and len(data) >= 13:
        return "pgs"
    if data.startswith(b"\x00\x00\x01\xba"):  # MPEG-PS pack: VobSub .sub
        return "vobsub"
    if ext == ".sup":
        return "pgs"
    fmt = sniff_format(_decode_head(data), ext)
    if fmt == "unknown" and ext == ".idx":
        return "vobsub"
    return fmt


# --------------------------------------------------------------- encodings

_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)

#: Language -> the legacy encodings its subtitles were written in, best first.
_ENCODING_HINTS: Dict[str, Tuple[str, ...]] = {
    "ja": ("cp932", "euc-jp"),
    "zh": ("gb18030", "big5"), "yue": ("big5", "gb18030"),
    "ko": ("cp949",),
    **{code: ("cp1251",) for code in ("ru", "uk", "bg", "sr", "mk", "be", "kk", "mn", "tg")},
    **{code: ("cp1250",) for code in ("cs", "pl", "hu", "sk", "sl", "hr", "ro", "bs", "sq")},
    "el": ("cp1253",),
    **{code: ("cp1254",) for code in ("tr", "az")},
    **{code: ("cp1255",) for code in ("he", "yi")},
    **{code: ("cp1256",) for code in ("ar", "fa", "ur", "ps")},
    "th": ("cp874",),
}

_DEFAULT_ORDER = ("cp1252", "cp932", "gb18030", "big5", "cp949", "cp1251", "cp1250",
                  "cp1253", "cp1254", "cp1255", "cp1256", "cp874", "euc-jp")

# The most frequent characters of Chinese (simplified and traditional forms)
# and Korean dialogue. A wrong CJK decoding still yields valid, even common,
# characters -- Korean read as GB18030 is all everyday hanzi -- but never the
# right *mix*: these few dozen carry a third or more of real text.
_TOP_HANZI = frozenset(
    "的一是不了人我在有他这這个個们們中来來上大为為和国國地到以说說时時要就出会會可也你对對生能而子那得于於着著"
    "下自之年过過发發后後作里裡用道行所然家种種事成方多经經么麼去法学學如都同现現当當没沒动動面起看定天分还還进進"
    "好小部其些主样樣理心她本前开開但因只从從想实實日吗嗎吧呢啊什谁誰怎"
)
_TOP_HANGUL = frozenset(
    "이다는에가하고지요서기의도을를어게아나해있수한사그리로거내만습니까면들시자라대할제전보말우주오여정일것인안상잘네데저왜뭐니무좀죠겠었"
)

# The same idea for single-byte scripts: the letters that make up about half
# of running text. Greek read as CP1251 is all valid Cyrillic, but the wrong
# Cyrillic.
_TOP_LETTERS = {
    "cp1251": frozenset("оеаинтсрвлі"),
    "cp1253": frozenset("αοειτνσςηρ"),
    "cp1255": frozenset("יוהאלרמתבנ"),
    "cp1256": frozenset("اليمونهرتب"),
    "cp874": frozenset("านอรเงกมยดวต่้"),
}

_COMMON_PUNCT = frozenset("“”‘’–—…«»¡¿°·•\u00a0、。「」『』！？（），：；～・ー\u3000《》〈〉【】")


def _utf16_without_bom(data: bytes) -> Optional[str]:
    sample = data[:4000]
    if len(sample) < 20:
        return None
    even = sample[0::2].count(0)
    odd = sample[1::2].count(0)
    half = len(sample) / 2
    if odd > half * 0.3 and even < half * 0.05:
        return "utf-16-le"
    if even > half * 0.3 and odd < half * 0.05:
        return "utf-16-be"
    return None


def _is_han(o: int) -> bool:
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF


def _plausibility(text: str, encoding: str) -> float:
    """How much ``text`` looks like real writing for ``encoding``'s scripts."""
    sample = text[:100000]
    chars = [c for c in sample if ord(c) > 127]
    if not chars:
        return 1.0
    family = encoding
    total = 0.0
    top = 0
    has_kana = any(0x3040 <= ord(c) <= 0x30FF for c in chars)
    nonascii_letters = 0
    for ch in chars:
        o = ord(ch)
        if 0x80 <= o <= 0x9F or 0xE000 <= o <= 0xF8FF or ch == "\ufffd":
            total -= 1.0
            continue
        if ch in _COMMON_PUNCT:
            total += 1.0
            continue
        if family in ("cp932", "euc-jp"):
            if 0x3040 <= o <= 0x30FF:
                total += 1.0
                top += 1
            elif _is_han(o):
                total += (1.0 if _jis_level1(ch) else 0.5) * (1.0 if has_kana else 0.6)
            elif 0xFF61 <= o <= 0xFF9F:
                total += 0.1
            elif 0xFF01 <= o <= 0xFF5E:
                total += 0.8
        elif family in ("gb18030", "big5"):
            if _is_han(o):
                total += _hanzi_commonness(ch, family)
                top += ch in _TOP_HANZI
            elif 0x3040 <= o <= 0x30FF:
                total += 0.2
            elif 0xFF01 <= o <= 0xFF5E:
                total += 0.8
        elif family == "cp949":
            if 0xAC00 <= o <= 0xD7A3:
                total += 1.0 if _in_ksx1001(ch) else 0.3
                top += ch in _TOP_HANGUL
            elif _is_han(o):
                total += 0.3
            elif 0xFF01 <= o <= 0xFF5E:
                total += 0.8
        else:
            category = unicodedata.category(ch)
            if not category.startswith("L") and not category.startswith("M"):
                total += 0.2 if category.startswith(("P", "S")) else 0.0
                continue
            nonascii_letters += 1
            home = {
                "cp1251": (0x0400, 0x04FF), "cp1253": (0x0370, 0x03FF), "cp1255": (0x0590, 0x05FF),
                "cp1256": (0x0600, 0x06FF), "cp874": (0x0E00, 0x0E7F),
            }.get(family, (0x00C0, 0x024F))
            if home[0] <= o <= home[1] or (family == "cp1256" and 0xFB50 <= o <= 0xFEFF):
                total += 1.0
    score = total / len(chars)
    if encoding in ("cp932", "euc-jp", "gb18030", "big5", "cp949"):
        return score + 1.5 * min(1.0, (top / len(chars)) / 0.2)
    ascii_letters = sum(1 for c in sample if c.isascii() and c.isalpha())
    share = nonascii_letters / max(1, ascii_letters + nonascii_letters)
    if encoding in ("cp1250", "cp1252", "cp1254"):
        # Western text is mostly ASCII letters with the odd accent; a page of
        # nothing but accented letters is another script read as Latin.
        return (score + 1.0) * (0.3 if share > 0.4 else 1.0)
    # And the reverse: Cyrillic, Greek, Hebrew, Arabic or Thai dialogue is
    # nearly all non-ASCII letters, not French with its accents turned into
    # the odd Cyrillic letter.
    # Those few letters are ~55% of real text and ~30% of a wrong decoding.
    common = _TOP_LETTERS.get(encoding, frozenset())
    letters = [unicodedata.normalize("NFD", c)[0].lower() for c in chars if unicodedata.category(c).startswith("L")]
    frequent = sum(1 for c in letters if c in common) / max(1, len(letters))
    bonus = min(1.0, max(0.0, (frequent - 0.3) / 0.2))
    # Wrong code pages scatter capitals through words ("ресЯменб").
    inner_caps = sum(1 for a, b in zip(sample, sample[1:]) if a.islower() and b.isupper() and ord(b) > 127)
    cased = inner_caps > 0.05 * max(1, nonascii_letters)
    return (score + bonus) * (0.3 if share < 0.5 else 1.0) * (0.5 if cased else 1.0)


def _jis_level1(ch: str) -> bool:
    try:
        lead = ch.encode("cp932")[0]
    except UnicodeEncodeError:
        return False
    return 0x88 <= lead <= 0x98


def _hanzi_commonness(ch: str, family: str) -> float:
    try:
        if family == "gb18030":
            lead = ch.encode("gb2312")[0]
            return 1.0 if 0xB0 <= lead <= 0xD7 else 0.6
        lead = ch.encode("big5")[0]
        return 1.0 if 0xA4 <= lead <= 0xC6 else 0.5
    except UnicodeEncodeError:
        return 0.15


def _in_ksx1001(ch: str) -> bool:
    try:
        ch.encode("euc-kr")
        return True
    except UnicodeEncodeError:
        return False


def detect_encoding(data: bytes, language: Optional[str] = None) -> str:
    """The encoding a subtitle file was written in.

    BOMs are trusted, then strict UTF-8 (which almost never decodes legacy
    text by accident). After that every candidate legacy code page that
    decodes the bytes cleanly is scored by how plausible the result is as
    writing in that code page's script, with the language hint (from the
    file name or the user) tried first and given a small bonus.
    """
    for bom, name in _BOMS:
        if data.startswith(bom):
            return name
    utf16 = _utf16_without_bom(data)
    if utf16:
        return utf16
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    hinted = _ENCODING_HINTS.get(languages.normalize(language) or "", ()) if language else ()
    if language and not hinted and languages.normalize(language):
        hinted = ("cp1252",)
    order = list(hinted) + [e for e in _DEFAULT_ORDER if e not in hinted]
    best, best_score = None, float("-inf")
    for candidate in order:
        try:
            decoded = data.decode(candidate)
        except UnicodeDecodeError:
            continue
        score = _plausibility(decoded, candidate) + (0.15 if candidate in hinted else 0.0)
        if score > best_score + 1e-9:
            best, best_score = candidate, score
    return best or "cp1252"


# -------------------------------------------------------- language from name

_NAME_FLAGS = {"forced", "sdh", "cc", "default", "full", "signs", "songs", "hi", "track", "sub", "subs",
               "subtitle", "subtitles", "dialog", "dialogue", "commentary"}


def language_from_filename(path: str) -> Optional[str]:
    """``Movie.ja.srt``, ``Movie.jpn.forced.srt``, ``Movie.English.srt``,
    ``Movie.pt-BR.srt`` -> an ISO 639-1 code; None when the name has no tag."""
    stem = os.path.splitext(os.path.basename(path))[0]
    tokens = stem.split(".")
    if len(tokens) < 2:
        return None
    for token in reversed(tokens[1:][-3:]):
        cleaned = token.strip(" []()_")
        lowered = cleaned.lower()
        if lowered in _NAME_FLAGS and not (lowered == "hi" and cleaned == "hi"):
            continue
        if re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,4})?|[A-Za-z]{4,}", cleaned):
            code = languages.normalize(cleaned)
            if code:
                return code
        if lowered.startswith("track") or lowered.isdigit():
            continue
        break
    return None


# ================================================================ public API


def parse(text: str, fmt: str, fps: Optional[float] = None) -> SubtitleDoc:
    """Parse subtitle text already decoded to str."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    fmt = "microdvd" if fmt == "sub" else fmt
    if fmt == "srt":
        return _parse_srt(text)
    if fmt in ("ass", "ssa"):
        return _parse_ass(text, fmt)
    if fmt == "vtt":
        return _parse_vtt(text)
    if fmt == "ttml":
        return _parse_ttml(text)
    if fmt == "microdvd":
        return _parse_microdvd(text, fps)
    if fmt == "sbv":
        return _parse_sbv(text)
    raise TaskError(f"Not a text subtitle format: {fmt}")


def read(path: str, encoding: Optional[str] = None, fps: Optional[float] = None) -> SubtitleDoc:
    """Read a text subtitle file of any supported format and encoding."""
    if not os.path.isfile(path):
        raise TaskError(f"Subtitle not found: {path}")
    fmt = detect_format(path)
    if fmt in ("pgs", "vobsub"):
        raise TaskError(f"{os.path.basename(path)} is an image subtitle ({fmt.upper()}); run OCR on it first.")
    with open(path, "rb") as handle:
        data = handle.read()
    hint = language_from_filename(path)
    chosen = encoding or detect_encoding(data, hint)
    try:
        text = data.decode(chosen, errors="replace")
    except LookupError as exc:
        raise TaskError(f"Unknown text encoding: {chosen}") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    if fmt == "unknown":
        fmt = sniff_format(text, os.path.splitext(path)[1].lower())
    if fmt not in TEXT_FORMATS:
        raise TaskError(f"{os.path.basename(path)} is not a subtitle format Subsync can read.")
    doc = parse(text, fmt, fps)
    doc.meta["encoding"] = chosen
    doc.meta["path"] = path
    if not doc.language:
        doc.language = hint
    return doc


def resolve_output_format(wanted: Optional[str], source_format: Optional[str]) -> str:
    """"same" -> the source's text format (else SRT); "sub" -> MicroDVD;
    "sup" -> "pgs" (written by the PGS encoder, not here)."""
    wanted = (wanted or "same").lower()
    if wanted == "same":
        return source_format if source_format in TEXT_FORMATS else "srt"
    if wanted == "sub":
        return "microdvd"
    if wanted in ("sup", "pgs"):
        return "pgs"
    if wanted in ("webvtt",):
        return "vtt"
    if wanted in ("dfxp", "xml", "imsc"):
        return "ttml"
    if wanted in TEXT_FORMATS:
        return wanted
    raise TaskError(f"Unknown subtitle output format: {wanted}")


def extension_for(fmt: str) -> str:
    return _EXTENSIONS.get(fmt, fmt)


def render(doc: SubtitleDoc, fmt: str, fps: Optional[float] = None) -> str:
    if fmt not in TEXT_FORMATS:
        fmt = resolve_output_format(fmt, doc.source_format)
    if fmt == "srt":
        return _render_srt(doc)
    if fmt in ("ass", "ssa"):
        return _render_ass(doc, fmt)
    if fmt == "vtt":
        return _render_vtt(doc)
    if fmt == "ttml":
        return _render_ttml(doc)
    if fmt == "microdvd":
        return _render_microdvd(doc, fps)
    if fmt == "sbv":
        return _render_sbv(doc)
    raise TaskError(f"Cannot write {fmt} as text; it needs the image subtitle encoder.")


def write(doc: SubtitleDoc, path: str, fmt: Optional[str] = None, fps: Optional[float] = None) -> str:
    """Write ``doc`` as UTF-8; ``fmt`` defaults to the path's extension.

    Written to a temporary name and renamed, so a failure never leaves half a
    subtitle where a good one (possibly the source) was.
    """
    if fmt is None:
        fmt = _FORMAT_BY_EXT.get(os.path.splitext(path)[1].lower())
        if fmt is None or fmt in ("pgs", "vobsub"):
            raise TaskError(f"Cannot tell which subtitle format to write from {os.path.basename(path)}")
    content = render(doc, fmt, fps)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, staging = tempfile.mkstemp(prefix=".subsync-", suffix=".part", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            out.write(content)
        os.replace(staging, path)
    except BaseException:
        try:
            os.remove(staging)
        except OSError:
            pass
        raise
    return path


def run_convert_task(job: dict, ctx) -> TaskResult:
    """Task "convert": one subtitle (file or track) to ``output.format``."""
    from .tasks import load_subtitle, preview_of, write_subtitle

    ref = (job.get("input") or {}).get("subtitle")
    if not ref:
        raise TaskError("Choose a subtitle to convert.")
    options = job.get("options") or {}
    fps = options.get("fps") or None
    fps = float(fps) if fps else None
    ctx.progress(10, "Reading")
    doc = load_subtitle(ref, ctx, fps=fps)
    if fps:
        doc.fps = fps
    wanted = (job.get("output") or {}).get("format") or "same"
    target = resolve_output_format(wanted, doc.source_format)
    if target == "pgs":
        raise TaskError("Converting text to PGS (.sup) is done by the HDR subtitles tool.")
    if target == "microdvd" and not doc.fps:
        raise TaskError("MicroDVD (.sub) is timed in frames; set the frame rate to write it.")
    ctx.check()
    ctx.progress(60, "Writing")
    source_ext = os.path.splitext(ref.get("path") or "")[1].lower().lstrip(".")
    suffix = "" if ref.get("track") is not None or source_ext != extension_for(target) else ".converted"
    output = write_subtitle(doc, ref, job, suffix, ctx)
    ctx.progress(100, "Done")
    source = (doc.source_format or "?").upper()
    return TaskResult(
        outputs=[output],
        report={"from": doc.source_format, "to": target, "cues": len(doc.cues), "encoding": doc.meta.get("encoding")},
        summary=f"Converted {len(doc.cues)} cues from {source} to {target.upper()}.",
        preview=preview_of(doc),
    )

