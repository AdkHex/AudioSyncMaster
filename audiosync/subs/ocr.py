"""OCR task: image subtitles (Blu-ray PGS, DVD VobSub) -> text cues.

Pipeline, per caption bitmap:

1. **Ink.** Subtitles are a light fill (white, yellow) inside a dark outline
   on a transparent background. OCR engines read dark text on white best and
   stumble over outlines, so the fill is separated from the outline by
   luminance among the opaque pixels (Otsu), with the outline recognised as
   the tone that borders the transparent background. The result is an ink
   map: 1 where the fill is, soft at its anti-aliased edges, 0 elsewhere.
2. **Lines.** Rows with ink form text lines (horizontal projection). Tiny
   fragments (accents, dots) join their neighbour; for CJK, a short line
   sitting right above a taller one is ruby/furigana and is dropped.
3. **Per-line image.** Black on white, enlarged by ``upscale``, padded. Each
   line is also checked for italics (the shear that makes its vertical
   strokes most upright) and for intentional wide gaps (CJK spaces).
   Tall narrow CJK captions are vertical text: each column is cut into
   characters that are laid out left to right, so every engine can read it.
4. **Engines.** Identical line images (PGS fades repeat a caption with only
   the palette changed) are read once. Lines below ``minConfidence`` go to
   the fallback engine; lines still below it are flagged for review.
5. **Text.** Observations are joined in reading order (furigana-sized ones
   above the text dropped), common OCR errors fixed, italics tagged, and
   consecutive captions with identical text that touch in time merged.

Output is written with ``tasks.write_subtitle`` (SRT unless asked
otherwise), named ``<stem>.<lang>.srt``; forced captions can be kept, kept
alone, or written to a second ``.forced`` file.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import imaging, languages
from .model import Cue, SubtitleDoc
from .pgs import BitmapEvent
from .tasks import TaskContext, TaskError, TaskOutput, TaskResult, preview_of, write_subtitle

#: Mirrors ``DEFAULT_OPTIONS.ocr`` in src/lib/subsync/types.ts.
DEFAULT_OPTIONS: Dict[str, Any] = {
    "engine": "vision",
    "language": "ja",
    "fallbackEngine": "none",
    "minConfidence": 0.6,
    "removeFurigana": True,
    "detectItalics": True,
    "keepPositions": True,
    "forced": "all",
    "fixCommonErrors": True,
    "upscale": 2,
    "claudeModel": "claude-sonnet-5",
    "saveFlaggedImages": False,
}

# Thresholds. Heights are fill-mask heights (the outline is not ink).

#: A row band shorter than this share of the tallest band is a fragment
#: (accent, dot, the tail of a glyph) and joins its nearest neighbour.
FRAGMENT_RATIO = 0.3
#: CJK ruby is set at about half the base size: a band at most this share of
#: the band right below it, and close to it, is furigana.
FURIGANA_MAX_RATIO = 0.62
FURIGANA_MAX_GAP = 0.6
#: Italic: the best upright-making shear must be at least this slant
#: (tan of ~7 degrees) and sharpen the column histogram by this factor.
ITALIC_MIN_SLANT = 0.12
ITALIC_MIN_GAIN = 1.12
#: A column gap at least this many line heights wide is a typed space.
WIDE_GAP_RATIO = 0.6
#: Enlarged line images stop growing at this height (px): engines gain
#: nothing beyond it and large images cost time (and Claude tokens).
MAX_LINE_HEIGHT_PX = 200
#: Captions that touch within this gap and read the same are one caption
#: (PGS fades are a run of captions that differ only in palette).
MERGE_GAP_S = 0.05
#: Flagged review images are shown on this background.
REVIEW_BACKGROUND = (96, 96, 96)


@dataclass
class TextLine:
    #: Ink crop of the line (float 0..1) in bitmap pixels.
    ink: np.ndarray
    top: int = 0
    bottom: int = 0
    italic: bool = False
    #: Typed spaces seen in the image (wide gaps between glyphs).
    wide_gaps: int = 0
    vertical: bool = False
    image: Optional[np.ndarray] = None
    key: str = ""
    text: str = ""
    confidence: float = 0.0
    engine: str = ""
    furigana_dropped: int = 0


@dataclass
class PreparedEvent:
    event: BitmapEvent
    lines: List[TextLine] = field(default_factory=list)
    vertical: bool = False
    furigana_removed: int = 0


@dataclass
class OcrOutcome:
    doc: SubtitleDoc
    report: Dict[str, Any]
    warnings: List[str]
    #: (cue, event) pairs still below minConfidence, for review images.
    flagged: List[Tuple[Cue, BitmapEvent]]


def options_with_defaults(options: Optional[dict]) -> Dict[str, Any]:
    merged = dict(DEFAULT_OPTIONS)
    merged.update({k: v for k, v in (options or {}).items() if v is not None})
    return merged


# ------------------------------------------------------------------- ink


def ink_map(rgba: np.ndarray) -> np.ndarray:
    """Fill-ness 0..1 of each pixel (see module docstring, step 1)."""
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    peak_alpha = float(alpha.max()) if alpha.size else 0.0
    if peak_alpha <= 0.0:
        return alpha
    lum = imaging.luma(rgba)
    # "Opaque" relative to the caption's own peak: a caption mid-fade is
    # uniformly faint but has the same structure.
    opaque = alpha > 0.5 * peak_alpha
    total = int(opaque.sum())
    if total < 8:
        return _normalised(alpha)
    threshold, separation = imaging.otsu_threshold(lum[opaque])
    bright = opaque & (lum > threshold)
    dark = opaque & ~bright
    nb, nd = int(bright.sum()), int(dark.sum())
    if separation < 0.5 or min(nb, nd) < 0.03 * total:
        # One tone: plain text without an outline, so the glyphs are simply
        # the opaque pixels.
        return _normalised(alpha)
    near_background = imaging.dilate(alpha < 0.1 * peak_alpha, 1)
    touch_bright = float((bright & near_background).sum()) / nb
    touch_dark = float((dark & near_background).sum()) / nd
    lo, hi = float(lum[dark].mean()), float(lum[bright].mean())
    span = max(hi - lo, 1.0)
    if touch_bright <= touch_dark:
        fill = np.clip((lum - lo) / span, 0.0, 1.0)
    else:  # dark text with a light outline
        fill = np.clip((hi - lum) / span, 0.0, 1.0)
    return _normalised(alpha * fill)


def _normalised(ink: np.ndarray) -> np.ndarray:
    """A caption mid-fade is the same caption at lower alpha: scale it to
    full strength so it reads (and de-duplicates) like the visible one."""
    peak = float(ink.max()) if ink.size else 0.0
    if 0.05 < peak < 0.98:
        ink = np.clip(ink / peak, 0.0, 1.0)
    return ink.astype(np.float32)


# ----------------------------------------------------------------- lines


def _bands(mask: np.ndarray) -> List[List[int]]:
    return [[a, b] for a, b in imaging.runs(mask.sum(axis=1))]


def _merge_fragments(bands: List[List[int]]) -> List[List[int]]:
    changed = True
    while changed and len(bands) > 1:
        changed = False
        tallest = max(b - a for a, b in bands)
        for i, (a, b) in enumerate(bands):
            if b - a >= FRAGMENT_RATIO * tallest:
                continue
            up = a - bands[i - 1][1] if i > 0 else 1 << 30
            down = bands[i + 1][0] - b if i + 1 < len(bands) else 1 << 30
            if min(up, down) > 0.6 * tallest:
                continue
            j = i - 1 if up <= down else i + 1
            lo, hi = min(i, j), max(i, j)
            bands[lo:hi + 1] = [[bands[lo][0], bands[hi][1]]]
            changed = True
            break
    return bands


def _columns(mask: np.ndarray) -> Tuple[int, int]:
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]), int(cols[-1]) + 1) if cols.size else (0, 0)


def split_lines(mask: np.ndarray, cjk: bool, remove_furigana: bool) -> Tuple[List[List[int]], int]:
    """Row bands of text lines, and how many furigana bands were dropped."""
    bands = _merge_fragments(_bands(mask))
    removed = 0
    if cjk and remove_furigana and len(bands) > 1:
        kept: List[List[int]] = []
        for i, (a, b) in enumerate(bands):
            if i + 1 < len(bands):
                na, nb = bands[i + 1]
                below = nb - na
                left, right = _columns(mask[a:b])
                bl, br = _columns(mask[na:nb])
                if (
                    b - a <= FURIGANA_MAX_RATIO * below
                    and na - b <= FURIGANA_MAX_GAP * below
                    and left >= bl - below
                    and right <= br + below
                    and (right - left) < 0.8 * (br - bl)
                ):
                    removed += 1
                    continue
            kept.append([a, b])
        bands = kept
    return bands, removed


def slant(mask: np.ndarray) -> Tuple[float, float]:
    """(shear, gain): the horizontal shear per pixel of height that makes the
    strokes most vertical, and how much sharper the column histogram gets
    than unsheared. Italic text leans right, so its shear is positive."""
    ys, xs = np.nonzero(mask)
    if xs.size < 40:
        return 0.0, 1.0
    height_above = (mask.shape[0] - 1 - ys).astype(np.float64)
    best, best_score, base = 0.0, -1.0, 1.0
    for s in np.arange(-0.15, 0.46, 0.025):
        shifted = np.rint(xs - s * height_above).astype(np.int64)
        hist = np.bincount(shifted - shifted.min()).astype(np.float64)
        score = float((hist * hist).sum())
        if abs(s) < 1e-9:
            base = score
        if score > best_score:
            best, best_score = float(s), score
    return best, best_score / max(base, 1e-9)


def wide_gaps(mask: np.ndarray, line_height: int) -> int:
    cols = imaging.runs(mask.sum(axis=0))
    return sum(1 for (_, b), (c, _) in zip(cols, cols[1:]) if c - b >= WIDE_GAP_RATIO * line_height)


def line_image(ink: np.ndarray, upscale: float) -> np.ndarray:
    """Black-on-white ``uint8`` image of one line, enlarged and padded."""
    height = max(1, ink.shape[0])
    factor = max(1.0, min(float(upscale or 1), MAX_LINE_HEIGHT_PX / height))
    grey = (255.0 * (1.0 - np.clip(ink, 0.0, 1.0))).astype(np.float32)
    grey = imaging.scale(grey, factor) if factor != 1.0 else grey
    border = int(round(0.35 * height * factor)) + 4
    return imaging.pad(np.clip(np.rint(grey), 0, 255).astype(np.uint8), border, 255)


def vertical_lines(ink: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """Vertical CJK text -> one horizontal ink strip per column, columns in
    reading order (right to left). Each column is cut into roughly square
    character cells (a character is about as tall as the column is wide),
    which are then placed side by side."""
    spans = imaging.runs(mask.sum(axis=0))
    if not spans:
        return []
    widest = max(b - a for a, b in spans)
    columns: List[List[int]] = []
    for a, b in spans:  # narrow slivers (punctuation offsets) join a column
        if columns and (b - a < 0.35 * widest or a - columns[-1][1] < 0.1 * widest) and b - a < 0.6 * widest:
            columns[-1][1] = b
        else:
            columns.append([a, b])
    strips: List[np.ndarray] = []
    for a, b in reversed(columns):
        width = b - a
        cells: List[List[int]] = []
        for s, e in imaging.runs(mask[:, a:b].sum(axis=1)):
            if cells and e - cells[-1][0] <= 1.25 * width:
                cells[-1][1] = e
            else:
                cells.append([s, e])
        if not cells:
            continue
        side = max(width, max(e - s for s, e in cells))
        gap = max(2, int(0.12 * width))
        strip = np.zeros((side, len(cells) * (width + gap)), dtype=np.float32)
        x = 0
        for s, e in cells:
            glyph = ink[s:e, a:b]
            top = (side - glyph.shape[0]) // 2
            strip[top : top + glyph.shape[0], x : x + width] = glyph
            x += width + gap
        strips.append(strip[:, : max(1, x - gap)])
    return strips


def is_vertical(event: BitmapEvent, mask: np.ndarray) -> bool:
    """Vertical CJK text: a narrow block whose columns are further apart
    than its rows. CJK glyphs sit on a grid either way, so shape alone is
    ambiguous (a 3x2 block reads either way); line spacing is what differs,
    because the gap between lines is always wider than between characters."""
    rows = imaging.runs(mask.sum(axis=1))
    cols = imaging.runs(mask.sum(axis=0))
    if not rows or not cols:
        return False
    h, w = rows[-1][1] - rows[0][0], cols[-1][1] - cols[0][0]
    if w > 0.25 * event.video_size[0] or h < 1.1 * w:
        return False
    if len(cols) == 1:
        return h >= 1.8 * w
    col_gap = max(c - b for (_, b), (c, _) in zip(cols, cols[1:]))
    row_gaps = sorted(c - b for (_, b), (c, _) in zip(rows, rows[1:]))
    row_gap = row_gaps[len(row_gaps) // 2] if row_gaps else 0
    widest = max(b - a for a, b in cols)
    return col_gap >= max(1.5 * row_gap, 0.25 * widest)


def prepare_event(event: BitmapEvent, options: dict, language: Optional[str]) -> PreparedEvent:
    prepared = PreparedEvent(event)
    ink = ink_map(event.rgba)
    mask = ink > 0.5
    if not mask.any():
        return prepared
    cjk = languages.is_cjk(language)
    upscale = float(options.get("upscale") or 2)
    if cjk and is_vertical(event, mask):
        prepared.vertical = True
        for strip in vertical_lines(ink, mask):
            line = TextLine(ink=strip, vertical=True)
            line.image = line_image(strip, upscale)
            prepared.lines.append(line)
        return prepared
    bands, prepared.furigana_removed = split_lines(mask, cjk, bool(options.get("removeFurigana")))
    for i, (a, b) in enumerate(bands):
        # One pixel of soft edge above/below the band, never into a neighbour.
        lo = max(a - 1, bands[i - 1][1] if i else 0)
        hi = min(b + 1, bands[i + 1][0] if i + 1 < len(bands) else ink.shape[0])
        left, right = _columns(mask[a:b])
        left, right = max(0, left - 1), min(ink.shape[1], right + 1)
        crop = ink[lo:hi, left:right]
        line_mask = mask[a:b, left:right]
        line = TextLine(ink=crop, top=a, bottom=b)
        line.wide_gaps = wide_gaps(line_mask, b - a)
        if options.get("detectItalics"):
            shear, gain = slant(line_mask)
            line.italic = shear >= ITALIC_MIN_SLANT and gain >= ITALIC_MIN_GAIN
        line.image = line_image(crop, upscale)
        prepared.lines.append(line)
    return prepared


def _image_key(image: np.ndarray) -> str:
    """Identity of a line for de-duplication: its shape and binarised
    pixels, so palette-only differences (fades) read once."""
    bits = np.packbits(image < 128)
    return hashlib.sha1(bits.tobytes() + str(image.shape).encode()).hexdigest()


# ------------------------------------------------------------ observations


def combine_observations(observations: Sequence[dict], cjk: bool, remove_furigana: bool) -> Tuple[str, float, int]:
    """Text and confidence of one line from an engine's observations."""
    obs = [o for o in observations if str(o.get("text") or "").strip()]
    if not obs:
        return "", 0.0, 0
    dropped = 0
    boxed = [o for o in obs if o.get("box")]
    if len(boxed) == len(obs) and len(obs) > 1:
        if cjk and remove_furigana:
            tallest = max(obs, key=lambda o: o["box"][3])
            h, top = tallest["box"][3], tallest["box"][1]
            keep = []
            for o in obs:
                small = o["box"][3] < FURIGANA_MAX_RATIO * h
                above = o["box"][1] + o["box"][3] <= top + 0.35 * h
                if o is not tallest and small and above:
                    dropped += 1
                else:
                    keep.append(o)
            obs = keep
        obs = sorted(obs, key=lambda o: o["box"][0])
    text = " ".join(str(o["text"]).strip() for o in obs)
    weights = [max(1, len(str(o["text"]).strip())) for o in obs]
    conf = sum(float(o.get("confidence") or 0.0) * w for o, w in zip(obs, weights)) / sum(weights)
    return text, conf, dropped


# --------------------------------------------------------------- fixes

_KATAKANA = "ァ-ヺーㇰ-ㇿｦ-ﾟ"
_KANA = "ぁ-ゖゝゞ" + _KATAKANA
_KANJI = "㐀-䶿一-鿿豈-﫿々"
_CJK = _KANA + _KANJI + "가-힯　-〿！-｠"
_CJK_RE = re.compile(f"[{_CJK}]")

_FULLWIDTH = {"!": "！", "?": "？", ":": "：", ";": "；", "(": "（", ")": "）", "~": "～"}


def _cjk_punctuation(text: str, language: str) -> str:
    comma = "、" if language == "ja" else "，"

    def near_cjk(i: int) -> bool:
        return (i > 0 and bool(_CJK_RE.match(text[i - 1]))) or (i + 1 < len(text) and bool(_CJK_RE.match(text[i + 1])))

    text = re.sub(f"(?<=[{_CJK}])\\s*(\\.\\.\\.|・・・)\\s*", "…", text)
    text = re.sub(f"\\s*(\\.\\.\\.|・・・)\\s*(?=[{_CJK}])", "…", text)
    out = []
    for i, ch in enumerate(text):
        if ch in _FULLWIDTH and near_cjk(i):
            out.append(_FULLWIDTH[ch])
        elif ch == "," and near_cjk(i):
            out.append(comma)
        elif ch == "." and i > 0 and _CJK_RE.match(text[i - 1]) and not (i + 1 < len(text) and text[i + 1].isalnum()):
            out.append("。")
        else:
            out.append(ch)
    return "".join(out)


def _long_vowel(text: str) -> str:
    """ー (long vowel) and 一 (one) look alike; context decides."""
    # 一 right after katakana, not starting a kanji compound (チーム一丸).
    text = re.sub(f"(?<=[{_KATAKANA}])一(?![{_KANJI}])", "ー", text)
    # Dash-like marks after katakana are the long vowel too (コ-ヒ-).
    text = re.sub(f"(?<=[{_KATAKANA}])[-－−―‐](?=[{_KANA}]|$|[\\s！？。、…」』）])", "ー", text)
    # ー between kanji, after a kanji at the end, or opening a kanji word.
    text = re.sub(f"(?<=[{_KANJI}])ー(?=[{_KANJI}]|$|[\\s！？。、…」』）])", "一", text)
    text = re.sub(f"(^|[^{_KANA}\\w])ー(?=[{_KANJI}])", "\\1一", text)
    return text


def _cjk_spaces(text: str, keep: int) -> str:
    """Spaces between two CJK characters are OCR noise unless the image had
    that many wide gaps (typed spaces, a common Japanese pause mark)."""
    pattern = f"(?<=[{_CJK}])[ 　]+(?=[{_CJK}])"
    if len(re.findall(pattern, text)) <= keep:
        return text
    return re.sub(pattern, "", text)


def fix_common_errors(text: str, language: Optional[str], wide_gaps: int = 0) -> str:
    code = languages.normalize(language) or ""
    if languages.is_cjk(code) or (not code and _CJK_RE.search(text)):
        text = _cjk_spaces(text, wide_gaps)
        text = _cjk_punctuation(text, code or "ja")
        if code in ("ja", ""):
            text = _long_vowel(text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()
    # Latin scripts. A pipe is never subtitle text; it is a misread I.
    text = re.sub(r"(?<![|])\|(?![|])", "I", text)
    if code in ("en", ""):
        # Stray spaces around an apostrophe inside English contractions.
        text = re.sub(r"(\w) ?(['’]) ?(s|t|re|ve|ll|d|m)\b", r"\1\2\3", text)
        # l for I only where no English word could be meant.
        text = re.sub(r"\bl(?=['’](?:m|ll|ve|d)\b)", "I", text)
        text = re.sub(r"(?<![\w'’-])l(?![\w'’-])", "I", text)
    if code != "fr":  # French spaces before ! ? ; : are correct
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # A comma glued to the next word ("know,I") lost its space; digits
    # ("1,000") are left alone.
    text = re.sub(r"(?<=[^\W\d_]),(?=[^\W\d_])", ", ", text)
    text = re.sub(r"\.\s\.\s\.", "...", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


_VERTICAL_FORMS = str.maketrans({
    "︙": "…", "︰": "‥", "︱": "ー", "︲": "–", "︵": "（", "︶": "）", "﹁": "「", "﹂": "」",
    "﹃": "『", "﹄": "』", "︑": "、", "︒": "。", "︗": "【", "︘": "】",
})  # fmt: skip


def fix_vertical_forms(text: str) -> str:
    """Vertical presentation forms, and the rotated long-vowel bar an engine
    reads as a pipe, back to horizontal text."""
    text = text.translate(_VERTICAL_FORMS)
    return re.sub(f"(?<=[{_KANA}])[|｜丨]", "ー", text)


# ------------------------------------------------------------------ cues


def merge_touching(cues: List[Cue], gap: float = MERGE_GAP_S) -> Tuple[List[Cue], int]:
    """Join captions with identical text and placement that touch in time."""
    out: List[Cue] = []
    open_by_key: Dict[tuple, Cue] = {}
    merged = 0
    for cue in sorted(cues, key=lambda c: (c.start, c.y or 0)):
        key = (cue.text, cue.align, cue.forced)
        prev = open_by_key.get(key)
        if prev is not None and cue.start - prev.end <= gap and cue.start >= prev.start:
            prev.end = max(prev.end, cue.end)
            if cue.confidence is not None:
                prev.confidence = max(prev.confidence or 0.0, cue.confidence)
            prev.meta["flagged"] = bool(prev.meta.get("flagged")) and bool(cue.meta.get("flagged"))
            merged += 1
            continue
        out.append(cue)
        open_by_key[key] = cue
    return out, merged


def _alignment(event: BitmapEvent, vertical: bool) -> Optional[int]:
    width, height = event.video_size
    if vertical:
        return 9 if event.x + event.width / 2 > width / 2 else 7
    if event.y + event.height / 2 < height / 3:
        return 8
    return None


# -------------------------------------------------------------- pipeline


def _recognize_lines(
    lines: List[TextLine],
    engine: str,
    language: Optional[str],
    options: dict,
    ctx: TaskContext,
    stage: str,
) -> Dict[str, Any]:
    """Read every distinct line image once with ``engine``."""
    from . import imaging as im
    from . import ocr_engines

    unique: Dict[str, TextLine] = {}
    for line in lines:
        unique.setdefault(line.key, line)
    folder = os.path.join(ctx.workdir, f"lines-{engine}")
    os.makedirs(folder, exist_ok=True)
    keys = list(unique)
    paths = []
    for i, key in enumerate(keys):
        path = os.path.join(folder, f"{i:05d}.png")
        im.write_png(path, unique[key].image)
        paths.append(path)

    def progress(done: int, total: int) -> None:
        ctx.progress(int(100 * done / max(1, total)), stage)

    reads = ocr_engines.recognize(engine, paths, language, options, ctx, progress)
    cjk = languages.is_cjk(language)
    by_key: Dict[str, Any] = {}
    errors = 0
    for key, read in zip(keys, reads):
        if read.error:
            errors += 1
        text, conf, dropped = combine_observations(read.observations, cjk, bool(options.get("removeFurigana")))
        by_key[key] = (text, conf, dropped)
    return {"by_key": by_key, "errors": errors}


def ocr_events(
    events: Sequence[BitmapEvent],
    options: dict,
    ctx: TaskContext,
    language: Optional[str],
    source_format: str = "pgs",
) -> OcrOutcome:
    """Recognise ``events`` into a doc; no file I/O besides scratch PNGs."""
    options = options_with_defaults(options)
    engine = str(options["engine"])
    fallback = str(options.get("fallbackEngine") or "none")
    min_conf = float(options.get("minConfidence") or 0.0)
    language = languages.normalize(language) or (language or None)
    warnings: List[str] = []

    prepared: List[PreparedEvent] = []
    for i, event in enumerate(events):
        if i % 25 == 0:
            ctx.check()
            ctx.progress(int(100 * i / max(1, len(events))), "Preparing images")
        prepared.append(prepare_event(event, options, language))
    lines = [line for p in prepared for line in p.lines]
    for line in lines:
        line.key = _image_key(line.image)

    label = _engine_label(engine)
    first = _recognize_lines(lines, engine, language, options, ctx, f"Reading text ({label})")
    for line in lines:
        line.text, line.confidence, line.furigana_dropped = first["by_key"][line.key]
        line.engine = engine
    if first["errors"]:
        warnings.append(f"{label} could not read {first['errors']} line image(s).")

    fallback_used = 0
    if fallback not in ("none", "", engine):
        unsure = [line for line in lines if line.confidence < min_conf or not line.text]
        if unsure:
            ctx.check()
            second = _recognize_lines(unsure, fallback, language, options, ctx,
                                      f"Re-reading unsure lines ({_engine_label(fallback)})")
            fallback_used = len({line.key for line in unsure})
            for line in unsure:
                text, conf, dropped = second["by_key"][line.key]
                if text and (conf > line.confidence or not line.text):
                    line.text, line.confidence, line.furigana_dropped = text, conf, dropped
                    line.engine = fallback

    cues: List[Cue] = []
    flagged: List[Tuple[Cue, BitmapEvent]] = []
    furigana = italics = vertical = empty = 0
    for p in prepared:
        event = p.event
        furigana += p.furigana_removed + sum(line.furigana_dropped for line in p.lines)
        texts: List[str] = []
        confs: List[float] = []
        for line in p.lines:
            text = line.text
            if line.vertical:
                text = fix_vertical_forms(text)
            if options.get("fixCommonErrors"):
                text = fix_common_errors(text, language, line.wide_gaps)
            if not text:
                continue
            confs.append(line.confidence)
            if line.italic and not languages.is_cjk(language):
                text = f"<i>{text}</i>"
                italics += 1
            texts.append(text)
        if not texts:
            empty += 1
            continue
        vertical += int(p.vertical)
        conf = round(min(confs), 3)
        cue = Cue(
            start=event.start,
            end=event.end,
            text="\n".join(texts),
            align=_alignment(event, p.vertical) if options.get("keepPositions") else None,
            x=event.x if options.get("keepPositions") else None,
            y=event.y if options.get("keepPositions") else None,
            forced=bool(event.forced),
            confidence=conf,
        )
        cue.meta["engines"] = sorted({line.engine for line in p.lines})
        if conf < min_conf or p.vertical:
            cue.meta["flagged"] = True
            flagged.append((cue, event))
        cues.append(cue)

    cues, merged = merge_touching(cues)
    flagged = [(c, e) for c, e in flagged if any(c is kept for kept in cues) and c.meta.get("flagged")]
    if empty:
        warnings.append(f"{empty} caption(s) had no readable text and were left out.")
    if vertical:
        warnings.append(f"{vertical} caption(s) were vertical text; check them.")
    doc = SubtitleDoc(
        cues=cues,
        language=language,
        source_format=source_format,
        play_res=tuple(events[0].video_size) if events else None,
    )
    report = {
        "events": len(events),
        "cues": len(cues),
        "lowConfidence": sum(1 for c in cues if c.meta.get("flagged")),
        "forced": sum(1 for c in cues if c.forced),
        "engine": engine,
        "fallbackUsed": fallback_used,
        "furiganaRemoved": furigana,
        "italics": italics,
        "vertical": vertical,
        "merged": merged,
        "empty": empty,
        "lineImages": len({line.key for line in lines}),
    }
    return OcrOutcome(doc, report, warnings, flagged)


def _engine_label(engine: str) -> str:
    from .ocr_engines import ENGINE_LABELS

    return ENGINE_LABELS.get(engine, engine)


# ------------------------------------------------------------------ input


@dataclass
class _Source:
    events: List[BitmapEvent]
    source_format: str
    language: Optional[str]


def _language_from_name(path: str) -> Optional[str]:
    for part in reversed(os.path.basename(path).split(".")[1:-1]):
        code = languages.normalize(part)
        if code and len(part) in (2, 3):
            return code
    return None


def _read_file(path: str) -> _Source:
    from . import pgs, vobsub

    lower = path.lower()
    if lower.endswith((".idx", ".sub")):
        idx = os.path.splitext(path)[0] + ".idx"
        if not os.path.isfile(idx):
            raise TaskError(f"The VobSub index {os.path.basename(idx)} is missing next to the .sub file.")
        index = vobsub.read_index(idx)
        stream = index.stream()
        events = vobsub.read_events(idx)
        lang = languages.normalize(stream.language) if stream else None
        return _Source(events, "vobsub", lang or _language_from_name(path))
    with open(path, "rb") as handle:
        head = handle.read(2)
    if head == b"PG" or lower.endswith(".sup"):
        return _Source(pgs.read_events(path), "pgs", _language_from_name(path))
    raise TaskError(f"{os.path.basename(path)} is not a PGS (.sup) or VobSub (.idx/.sub) subtitle.")


def load_source(ref: dict, ctx: TaskContext) -> _Source:
    """Bitmap captions of a .sup, an .idx/.sub, or an image track in a video."""
    from . import tracks

    path = ref.get("path")
    if not path or not os.path.isfile(path):
        raise TaskError(f"Subtitle not found: {path}")
    lower = path.lower()
    if lower.endswith((".sup", ".idx", ".sub")) and ref.get("track") is None:
        return _read_file(path)
    listed = tracks.list_tracks(path, ctx.token)
    if ref.get("track") is not None:
        info = next((t for t in listed if t.index == int(ref["track"])), None)
        if info is None:
            raise TaskError(f"{os.path.basename(path)} has no subtitle track {int(ref['track']) + 1}.")
    else:
        info = next((t for t in listed if t.kind == "image"), None)
        if info is None:
            raise TaskError(f"{os.path.basename(path)} has no image subtitle track to OCR.")
    if info.kind != "image":
        raise TaskError(f"Track {info.index + 1} is a text subtitle already; OCR is for PGS/VobSub.")
    ctx.progress(0, "Extracting the subtitle track")
    extracted = tracks.extract(path, info, ctx.workdir, ctx.token)
    source = _read_file(extracted)
    source.language = languages.normalize(info.language) or source.language
    return source


def _output_ref(ref: dict, language: Optional[str]) -> dict:
    """The reference outputs are named after; a ``Movie.ja.sup`` source must
    give ``Movie.ja.srt``, not ``Movie.ja.ja.srt``."""
    path = ref.get("path") or "subtitle"
    stem, ext = os.path.splitext(path)
    if ref.get("track") is None and language and "." in os.path.basename(stem):
        head, _, tail = stem.rpartition(".")
        if languages.normalize(tail) == languages.normalize(language) and len(tail) in (2, 3):
            return dict(ref, path=head + ext)
    return ref


def _save_flagged(flagged: List[Tuple[Cue, BitmapEvent]], subtitle_path: str) -> Optional[str]:
    if not flagged:
        return None
    folder = os.path.splitext(subtitle_path)[0] + ".flagged"
    os.makedirs(folder, exist_ok=True)
    rows = ["#\tstart\tend\tconfidence\ttext"]
    for n, (cue, event) in enumerate(sorted(flagged, key=lambda item: item[0].start), 1):
        stamp = _stamp(cue.start)
        name = f"{n:04d}_{stamp}.png"
        imaging.write_png(os.path.join(folder, name), imaging.flatten(event.rgba, REVIEW_BACKGROUND))
        cue.meta["ocrImage"] = os.path.join(folder, name)
        text = cue.text.replace("\n", " / ").replace("\t", " ")
        rows.append(f"{n}\t{_stamp(cue.start)}\t{_stamp(cue.end)}\t{cue.confidence}\t{text}")
    with open(os.path.join(folder, "flagged.tsv"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    return folder


def _stamp(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}-{ms // 60000 % 60:02d}-{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def engine_statuses(secrets: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    from . import ocr_engines

    return ocr_engines.engine_statuses(secrets)


def run_task(job: dict, ctx: TaskContext) -> TaskResult:
    options = options_with_defaults(job.get("options"))
    ref = (job.get("input") or {}).get("subtitle")
    if not ref:
        raise TaskError("Choose an image subtitle (.sup, .idx/.sub, or a video with PGS/VobSub) to OCR.")
    ctx.progress(0, "Reading the subtitle")
    source = load_source(ref, ctx)
    wanted = str(options.get("language") or "").strip()
    language = source.language if wanted.lower() in ("", "auto") else (languages.normalize(wanted) or wanted)
    if not source.events:
        raise TaskError("The subtitle has no captions.")

    events = source.events
    forced_mode = str(options.get("forced") or "all")
    if forced_mode == "forcedOnly":
        events = [e for e in events if e.forced]
        if not events:
            return TaskResult(summary="No forced captions in this subtitle; nothing was written.",
                              warnings=["The subtitle has no forced captions."],
                              report={"events": len(source.events), "cues": 0, "forced": 0})

    outcome = ocr_events(events, options, ctx, language, source.source_format)
    ctx.progress(100, "Writing")
    out_job = dict(job)
    output = dict(job.get("output") or {})
    if output.get("format") == "sup":
        output["format"] = "srt"  # OCR makes text; SUP output is for HDR subtitles
    out_job["output"] = output
    out_ref = _output_ref(ref, language)
    doc = outcome.doc
    if not doc.cues:
        raise TaskError("No text could be read from the subtitle images.")

    outputs: List[TaskOutput] = []
    main = write_subtitle(doc, out_ref, out_job, "", ctx, language=language)
    outputs.append(main)
    forced_cues = [c for c in doc.cues if c.forced]
    if forced_mode == "split":
        if forced_cues:
            forced_job = dict(out_job, output=dict(output, suffix=(output.get("suffix") or "") + ".forced"))
            forced_out = write_subtitle(doc.copy([c.copy() for c in forced_cues]), out_ref, forced_job, ".forced",
                                        ctx, language=language)
            forced_out.label = "Forced captions"
            outputs.append(forced_out)
        else:
            outcome.warnings.append("No forced captions to split out.")
    if options.get("saveFlaggedImages"):
        folder = _save_flagged(outcome.flagged, main.path)
        if folder:
            outputs.append(TaskOutput(path=folder, kind="folder", label="Lines to review"))

    report = dict(outcome.report)
    report["language"] = language
    report["source"] = source.source_format
    label = _engine_label(str(options["engine"]))
    review = report["lowConfidence"]
    summary = f"Read {report['cues']} captions with {label}"
    if report["fallbackUsed"]:
        summary += f" ({report['fallbackUsed']} lines re-read with {_engine_label(str(options['fallbackEngine']))})"
    summary += f"; {review} need review." if review else "; all above the confidence threshold."
    return TaskResult(outputs=outputs, report=report, summary=summary, warnings=outcome.warnings,
                      preview=preview_of(doc))
