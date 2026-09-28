"""Subtitle brightness for HDR / Dolby Vision video (task "hdrSubs").

Why subtitles glare on HDR
--------------------------
Almost every subtitle is authored for SDR: white is "100%", and on an SDR
display that is ~100 nits, the same level as a white shirt in the picture.
An HDR picture has no "100%". PQ encodes absolute luminance, and diffuse
white in a well-graded HDR film sits near 203 nits (ITU-R BT.2408 "HDR
reference white", which is also 75% HLG), with only speculars going to
1000+ nits. So the player has to *choose* how bright a subtitle's white is,
and many choose far too high:

* mpv (``vo=gpu-next``) puts image subtitles (PGS/VobSub) at
  ``--image-subs-hdr-peak`` = 1000 nits by default, and text subtitles at
  ``--sub-hdr-peak=auto`` = the reference white (203 nits). Source: mpv
  DOCS/man/options.rst.
* Hardware players and TV apps composite subtitles as SDR graphics into the
  HDR output. Users of Plex, Infuse, Kodi-based boxes and TV apps report
  subtitles "way too bright" on HDR (e.g. the OSMC forum thread
  "Over-bright subtitles in HDR", mpv issues #6368, #8325, #13680): these
  map SDR white to the display's peak or a fixed high level.
* On UHD Blu-ray the disc's own PGS are authored for the HDR title, so they
  are usually fine; the glare comes from SDR subtitles laid on HDR video --
  a 1080p Blu-ray PGS or fan-made ASS/SRT muxed into a UHD remux.

The model used here
-------------------
A subtitle colour is an SDR display-referred R'G'B' value: the player
shows full white at ``assumedPeakNits`` and everything else by the BT.1886
curve (gamma 2.4, zero black) below it -- exactly what mpv does with its
subtitle peak. To make white land at ``targetNits`` instead, every colour's
*linear light* is multiplied by ``factor = targetNits / assumedPeakNits``
and re-encoded: code values scale by ``factor ** (1/2.4)``. For the
defaults (203 / 1000) that is x0.203 in light, x0.515 in code: PGS white
Y'=235 becomes 129, ASS &HFFFFFF becomes &H838383. ``percent`` mode uses
``brightnessPercent / 100`` as the light factor directly. Doing it in
light, not code, keeps anti-aliased edges and the outline in the same
proportion to the fill, so the text looks the same, only dimmer.

``color`` recolours the fill: the new colour is the chosen colour at the
entry's own luminance (black outline stays black, anti-aliasing stays
proportional). PGS palettes are recoloured entirely; in ASS the styles'
PrimaryColour is recoloured, while inline ``\\c`` overrides are recoloured
only when they are near-grey, because a coloured inline tag is deliberate
typesetting (a red sign) that should dim but keep its hue. Secondary
(karaoke) colours dim but keep their colour; outline and shadow colours are
left alone. SRT colour is a ``<font color>`` tag, which some players ignore.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import colorimetry
from .tasks import TaskContext, TaskError, TaskOutput, TaskResult, output_path, preview_of

DEFAULTS: Dict[str, Any] = {
    "mode": "nits",
    "targetNits": colorimetry.REFERENCE_WHITE_NITS,
    "assumedPeakNits": 1000,
    "brightnessPercent": 60,
    "color": "keep",
    "outputFormat": "same",
}

NAMED_COLOURS = {
    "white": (255, 255, 255),
    # 75% grey: the "soft white" players offer as a subtitle colour.
    "gray": (191, 191, 191),
    "grey": (191, 191, 191),
    "yellow": (255, 255, 0),
}

SUFFIX = ".hdr"

#: A colour whose channels differ by at most this (0..255) counts as grey,
#: which is what dialogue text is; anything more is deliberate colour.
NEUTRAL_SPREAD = 24


@dataclass
class ColourMapping:
    """One subtitle colour -> its dimmed (and optionally recoloured) self."""

    #: Multiplier on linear light (1.0 = unchanged).
    factor: float
    #: Replacement fill colour as R'G'B' 0..255, or None to keep hue.
    colour: Optional[Tuple[int, int, int]] = None
    gamma: float = colorimetry.SDR_GAMMA

    @property
    def code_scale(self) -> float:
        """How code values scale for a neutral colour."""
        return self.factor ** (1.0 / self.gamma)

    def map_rgb(self, rgb, recolour: bool = True) -> np.ndarray:
        """R'G'B' (0..1, shape (..., 3)) -> mapped R'G'B' (0..1)."""
        rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
        light = colorimetry.bt1886_eotf(rgb, self.gamma)
        if recolour and self.colour is not None:
            target = colorimetry.bt1886_eotf(np.asarray(self.colour, dtype=np.float64) / 255.0, self.gamma)
            lum = light @ colorimetry.BT709_LUMA
            light = target * lum[..., None]
        light = light * self.factor
        # A brightening factor can push a channel past white; scaling the
        # colour keeps its hue where clipping one channel would shift it.
        top = light.max(axis=-1, keepdims=True)
        light = np.where(top > 1.0, light / np.maximum(top, 1e-12), light)
        return colorimetry.bt1886_inverse_eotf(light, self.gamma)

    def map_rgb255(self, r: int, g: int, b: int, recolour: bool = True) -> Tuple[int, int, int]:
        out = self.map_rgb(np.array([r, g, b], dtype=np.float64) / 255.0, recolour)
        return tuple(int(v) for v in np.clip(np.rint(out * 255.0), 0, 255))  # type: ignore[return-value]

    def map_pgs(self, y: int, cb: int, cr: int, a: int) -> Tuple[int, int, int, int]:
        """A PGS palette entry (limited-range BT.709 Y'CbCr, 8-bit)."""
        if a == 0:
            return y, cb, cr, a  # invisible; changing it would only churn bytes
        rgb = colorimetry.ycbcr_to_rgb(y, cb, cr, colorimetry.BT709_LUMA, bits=8)
        mapped = self.map_rgb(rgb, recolour=True)
        ny, ncb, ncr = colorimetry.rgb_to_ycbcr(mapped, colorimetry.BT709_LUMA, bits=8)
        return int(ny), int(ncb), int(ncr), a


def parse_colour(value: Any) -> Optional[Tuple[int, int, int]]:
    """"keep" -> None; a name or ``#rgb``/``#rrggbb`` -> (r, g, b)."""
    text = str(value or "keep").strip().lower()
    if text in ("", "keep", "same", "none"):
        return None
    if text in NAMED_COLOURS:
        return NAMED_COLOURS[text]
    match = re.fullmatch(r"#?([0-9a-f]{3}|[0-9a-f]{6})", text)
    if not match:
        raise TaskError(f"Unknown colour {value!r}: use keep, white, gray, yellow or #RRGGBB.")
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def mapping_from_options(options: Dict[str, Any]) -> Tuple[ColourMapping, List[str]]:
    opts = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None}}
    warnings: List[str] = []
    mode = str(opts.get("mode") or "nits")
    if mode == "nits":
        try:
            target = float(opts["targetNits"])
            peak = float(opts["assumedPeakNits"])
        except (TypeError, ValueError):
            raise TaskError("Target and assumed peak must be numbers of nits.") from None
        if not (1 <= target <= 10000 and 1 <= peak <= 10000):
            raise TaskError("Target and assumed peak must be between 1 and 10000 nits.")
        factor = target / peak
    elif mode == "percent":
        try:
            factor = float(opts["brightnessPercent"]) / 100.0
        except (TypeError, ValueError):
            raise TaskError("Brightness must be a percentage.") from None
        if not 0.01 <= factor <= 2.0:
            raise TaskError("Brightness must be between 1% and 200%.")
    else:
        raise TaskError(f"Unknown mode: {mode}")
    if factor > 1.0:
        warnings.append("That brightens the subtitles; white cannot go above full white, so bright "
                        "colours clip.")
    return ColourMapping(factor=factor, colour=parse_colour(opts.get("color"))), warnings


def _hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _is_neutral(r: int, g: int, b: int) -> bool:
    return max(r, g, b) - min(r, g, b) <= NEUTRAL_SPREAD


# --------------------------------------------------------------- ASS / SSA

_INLINE_COLOUR_RE = re.compile(r"\\([12]?)c&[Hh]([0-9A-Fa-f]{1,8})&?")
_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")


def _parse_ass_colour(value: str) -> Optional[Tuple[int, int, int, int, str]]:
    """``&HAABBGGRR`` / ``&HBBGGRR`` / SSA decimal -> (r, g, b, a, shape)."""
    text = value.strip()
    try:
        if text[:2].lower() == "&h":
            digits = text[2:].rstrip("&")
            n = int(digits, 16)
            shape = f"hex{len(digits)}" + ("&" if text.endswith("&") else "")
        else:
            n = int(text)
            shape = "dec-" if n < 0 else "dec"
            n &= 0xFFFFFFFF
    except ValueError:
        return None
    return n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF, shape


def _format_ass_colour(r: int, g: int, b: int, a: int, shape: str) -> str:
    n = (a << 24) | (b << 16) | (g << 8) | r
    if shape.startswith("dec"):
        if shape == "dec-" and n >= 0x80000000:
            n -= 0x100000000
        return str(n)
    width = int(re.sub(r"\D", "", shape) or 8)
    digits = f"{n:08X}" if width > 6 or a else f"{n & 0xFFFFFF:06X}"
    return "&H" + digits + ("&" if shape.endswith("&") else "")


def rewrite_ass(text: str, mapping: ColourMapping) -> Tuple[str, int]:
    """Rewrite fill colours in ASS/SSA text; returns (text, colours changed).

    Everything except the colour values is kept byte for byte: styles and
    events are edited in place, never re-rendered.
    """
    changed = 0
    section = ""
    style_fields: List[str] = []
    out_lines: List[str] = []

    def map_value(value: str, recolour: bool) -> str:
        nonlocal changed
        parsed = _parse_ass_colour(value)
        if parsed is None:
            return value
        r, g, b, a, shape = parsed
        nr, ng, nb = mapping.map_rgb255(r, g, b, recolour=recolour)
        if (nr, ng, nb) == (r, g, b):
            return value
        changed += 1
        # Keep the surrounding whitespace of the field.
        lead = value[: len(value) - len(value.lstrip())]
        trail = value[len(value.rstrip()):]
        return lead + _format_ass_colour(nr, ng, nb, a, shape) + trail

    def inline(match: "re.Match[str]") -> str:
        nonlocal changed
        which = match.group(1)
        n = int(match.group(2), 16) & 0xFFFFFF
        r, g, b = n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF
        recolour = which in ("", "1") and _is_neutral(r, g, b)
        nr, ng, nb = mapping.map_rgb255(r, g, b, recolour=recolour)
        if (nr, ng, nb) == (r, g, b):
            return match.group(0)
        changed += 1
        return f"\\{which}c&H{nb:02X}{ng:02X}{nr:02X}&"

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        header = _SECTION_RE.match(body)
        if header:
            section = header.group(1).strip().lower()
            out_lines.append(line)
            continue
        key, sep, rest = body.partition(":")
        key_l = key.strip().lower()
        if section in ("v4+ styles", "v4 styles", "v4 styles+") and sep:
            if key_l == "format":
                style_fields = [f.strip().lower() for f in rest.split(",")]
            elif key_l == "style" and style_fields:
                values = rest.split(",", len(style_fields) - 1)
                for i, name in enumerate(style_fields[: len(values)]):
                    if name == "primarycolour":
                        values[i] = map_value(values[i], recolour=True)
                    elif name == "secondarycolour":
                        values[i] = map_value(values[i], recolour=False)
                body = key + sep + ",".join(values)
        elif section == "events" and key_l == "dialogue" and sep:
            body = key + sep + _INLINE_COLOUR_RE.sub(inline, rest)
        out_lines.append(body + ending)
    return "".join(out_lines), changed


# ---------------------------------------------------------------- SRT / VTT

_FONT_COLOUR_RE = re.compile(r'(<font\b[^>]*?\bcolor\s*=\s*["\']?)(#?[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3})(["\']?)', re.IGNORECASE)


def rewrite_markup(text: str, mapping: ColourMapping, wrap: bool = True) -> Tuple[str, int]:
    """Map ``<font color>`` values in cue text and wrap it in the mapped
    fill colour, so plain text (which players draw white) dims too."""
    changed = 0

    def repl(match: "re.Match[str]") -> str:
        nonlocal changed
        rgb = parse_colour(match.group(2) if match.group(2).startswith("#") else "#" + match.group(2))
        assert rgb is not None
        new = mapping.map_rgb255(*rgb, recolour=_is_neutral(*rgb))
        changed += int(new != rgb)
        return match.group(1) + _hex(new) + match.group(3)

    out = _FONT_COLOUR_RE.sub(repl, text)
    if wrap and out.strip():
        fill = mapping.map_rgb255(255, 255, 255)
        out = f'<font color="{_hex(fill)}">{out}</font>'
        changed += 1
    return out, changed


def _vtt_with_style(rendered: str, fill: Tuple[int, int, int]) -> str:
    """Put a ``::cue`` colour rule in a STYLE block before the first cue
    (WebVTT: STYLE blocks must precede cues)."""
    style = f"STYLE\n::cue {{\n  color: {_hex(fill)};\n}}\n"
    head, sep, rest = rendered.partition("\n\n")
    return head + "\n\n" + style + "\n" + rest if sep else rendered + "\n\n" + style


# -------------------------------------------------------------------- task


def _resolve_input(ref: Dict[str, Any], ctx: TaskContext) -> str:
    path = ref.get("path")
    if not path or not os.path.isfile(path):
        raise TaskError(f"Subtitle not found: {path}")
    if ref.get("track") is None:
        return path
    from . import tracks  # noqa: PLC0415 - lazy by design

    ctx.progress(5, "Extracting the subtitle track")
    return tracks.extract(path, int(ref["track"]), ctx.workdir, ctx.token)


def _read_text(path: str) -> Tuple[str, bool]:
    from . import formats  # noqa: PLC0415

    with open(path, "rb") as handle:
        data = handle.read()
    encoding = formats.detect_encoding(data)
    had_bom = data.startswith(b"\xef\xbb\xbf")
    return data.decode(encoding, errors="replace").lstrip("﻿"), had_bom


def _write_text(path: str, text: str, bom: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def run_task(job: dict, ctx: TaskContext) -> TaskResult:
    from . import formats  # noqa: PLC0415

    options = {**DEFAULTS, **(job.get("options") or {})}
    mapping, warnings = mapping_from_options(options)
    ref = (job.get("input") or {}).get("subtitle") or {}
    source = _resolve_input(ref, ctx)
    fmt = formats.detect_format(source)
    wanted = str(options.get("outputFormat") or "same").lower()
    white = mapping.map_rgb255(255, 255, 255)
    report: Dict[str, Any] = {
        "format": fmt,
        "mode": options.get("mode"),
        "factor": round(mapping.factor, 4),
        "codeScale": round(mapping.code_scale, 4),
        "targetNits": float(options.get("targetNits") or 0),
        "assumedPeakNits": float(options.get("assumedPeakNits") or 0),
        "color": str(options.get("color") or "keep"),
        "white": _hex(white),
    }
    how = (f"white at ~{report['targetNits']:g} nits instead of ~{report['assumedPeakNits']:g}"
           if options.get("mode", "nits") == "nits" else f"{mapping.factor * 100:g}% of the light")
    ctx.progress(20, "Mapping colours")

    if fmt == "pgs":
        if wanted not in ("same", "sup", "pgs"):
            raise TaskError("A PGS subtitle is an image; turn it into text with OCR before writing ASS.")
        from . import pgs  # noqa: PLC0415

        changed: set = set()

        def fn(y: int, cb: int, cr: int, a: int) -> Tuple[int, int, int, int]:
            out = mapping.map_pgs(y, cb, cr, a)
            if out != (y, cb, cr, a):
                changed.add((y, cb, cr, a))
            return out

        out_path = output_path(ref, job, SUFFIX, "sup")
        entries = pgs.adjust_palette(source, out_path, fn)
        report.update(entriesChanged=len(changed), paletteEntries=entries,
                      whiteY=int(mapping.map_pgs(235, 128, 128, 255)[0]))
        summary = (f"Rewrote {len(changed)} PGS palette colours ({entries} palette entries in all) so "
                   f"subtitles show {how}: x{mapping.factor:.3f} in light, full-white Y' 235 -> "
                   f"{report['whiteY']}. Pictures, timing and positions are unchanged.")
        ctx.log(summary)
        return TaskResult(outputs=[TaskOutput(path=out_path, kind="subtitle", format="pgs")],
                          report=report, summary=summary, warnings=warnings)

    if fmt in ("vobsub",):
        raise TaskError("VobSub (DVD) subtitles are not adjusted here; DVDs are never HDR. OCR them first "
                        "to get text you can restyle.")
    if fmt not in formats.TEXT_FORMATS:
        raise TaskError(f"Not a subtitle this tool can adjust: {os.path.basename(source)}")
    if wanted in ("sup", "pgs"):
        raise TaskError("Text subtitles cannot be written as PGS here; choose same format or ASS.")

    text, bom = _read_text(source)
    doc = formats.parse(text, fmt)
    language = doc.language

    if fmt in ("ass", "ssa") and wanted in ("same", fmt):
        new_text, changed_n = rewrite_ass(text, mapping)
        out_fmt = fmt
        out_path = output_path(ref, job, SUFFIX, formats.extension_for(fmt), language)
        _write_text(out_path, new_text, bom)
        note = ""
    elif wanted == "ass" or (wanted == "same" and fmt not in ("srt", "vtt")):
        if wanted == "same":
            raise TaskError(f"{fmt.upper()} has no colour styling this tool can set; choose ASS output.")
        rendered = formats.render(doc, "ass")
        new_text, changed_n = rewrite_ass(rendered, mapping)
        out_fmt = "ass"
        out_path = output_path(ref, job, SUFFIX, "ass", language)
        _write_text(out_path, new_text, bom=False)
        note = " Converted to ASS with the colour in the style, which every ASS renderer honours."
    elif fmt == "srt":
        changed_n = 0
        for cue in doc.cues:
            cue.text, n = rewrite_markup(cue.text, mapping)
            changed_n += n
        out_fmt = "srt"
        out_path = output_path(ref, job, SUFFIX, "srt", language)
        formats.write(doc, out_path, "srt")
        note = (" SRT has no styles, so the colour is a <font color> tag on every cue; players that "
                "ignore it (many TV apps, burn-in transcodes) still draw their own white -- use ASS "
                "output where that happens.")
    else:  # vtt, same format
        rendered = formats.render(doc, "vtt")
        new_text = _vtt_with_style(rendered, white)
        changed_n = len(doc.cues)
        out_fmt = "vtt"
        out_path = output_path(ref, job, SUFFIX, "vtt", language)
        _write_text(out_path, new_text, bom=False)
        note = (" WebVTT colour is a ::cue STYLE rule; browsers honour it, many players do not -- "
                "use ASS output where it is ignored.")

    report.update(entriesChanged=changed_n)
    summary = (f"Set subtitle fill colours so they show {how} (x{mapping.factor:.3f} in light; white "
               f"becomes {_hex(white)}), {changed_n} colour values changed.{note}")
    ctx.log(f"Wrote {os.path.basename(out_path)}")
    preview = None
    try:
        preview = preview_of(formats.read(out_path))
    except Exception:  # noqa: BLE001 - the preview is a convenience
        preview = None
    return TaskResult(
        outputs=[TaskOutput(path=out_path, kind="subtitle", format=out_fmt, language=language)],
        report=report, summary=summary, warnings=warnings, preview=preview,
    )
