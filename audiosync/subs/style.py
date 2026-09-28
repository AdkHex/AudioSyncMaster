"""Timed-text style: per-language rules, line breaking, checking and fixing.

Every Subsync path that produces text ends here. Generate cuts recognised
words into subtitles (``segment_words``), Translate and OCR re-flow lines
(``break_lines``), and the Style task checks a finished file against the
Netflix Timed Text Style Guide for its language and repairs what can be
repaired without touching a word (``check`` / ``fix``).

The numbers live in ``style_presets.py`` next to their sources. This module
holds the behaviour, and two decisions shape all of it:

* **Width is counted the way the guide counts it.** Japanese and Chinese
  limits are in full-width characters with half-width ones at 0.5, Korean
  counts Latin letters, spaces and punctuation as 0.5, and Thai does not
  count tone marks or upper/lower vowels. A single ``len()`` would pass
  Japanese lines twice as long as the guide allows and fail every Thai line.
* **Fixing never changes words.** Only times and line breaks move. A cue
  that is too long is split at a sentence or clause boundary and its time is
  shared by character count; the words themselves stay exactly as written,
  styling tags included.
"""

from __future__ import annotations

import bisect
import itertools
import math
import os
import re
import subprocess
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import style_presets as P
from .languages import normalize
from .model import Cue, SubtitleDoc, Word

#: 23.976 fps: the most common rate for the drama/film content these rules
#: were written for, and what the guides' own frame examples assume.
DEFAULT_FPS = 24000 / 1001

#: A silence this long between recognised words ends a subtitle: people
#: hear it as the end of a thought, and a subtitle bridging it would sit on
#: screen over nothing.
PAUSE_S = 0.6

SENTENCE_END = ".?!。？！…‥"
CLAUSE_END = ",;:，、；：–—"
DASHES = "-–"
UNSPACED = frozenset({"ja", "zh", "yue"})

#: Japanese subtitles separate two sentences on one screen with a
#: full-width space where there is no 。; it is kept as written and is a
#: break point that consumes itself, like a space in spaced scripts.
FULL_SPACE = "\u3000"
_SPACES = (" ", FULL_SPACE)

#: Cost of a break the language does not allow: a particle cut from its word
#: (誰に | も), a prenominal from its noun (この | 子), a kanji stem from its
#: kana (立ち | 上がる). Far above any preference, so every layout without
#: one wins, yet below an over-long line, which is a harder rule still.
#: ``segment_words`` never accepts one: it ends the subtitle earlier instead.
FORBID = 200.0

#: A silence this long after a Japanese word that does not continue the
#: sentence (no particle, te-form or conjunctive ending) is a sentence end
#: the recogniser left unpunctuated: 隠そう [pause] やば.
JA_SENTENCE_PAUSE_S = 0.3

# Styling markup inside cue text: the model's HTML subset plus ASS override
# blocks that survive in SRT files ({\an8}). Both are zero-width.
_TAG_RE = re.compile(r"(</?(?:i|b|u|font)(?:\s[^>]*)?>|\{\\[^}]*\})", re.IGNORECASE)
_WORD_CHARS = re.compile(r"[\w'’]+", re.UNICODE)

SEVERITY = {
    "empty": "error",
    "overlap": "error",
    "cpl_exceeded": "error",
    "too_many_lines": "error",
    "cps_exceeded": "error",
    "too_short": "warning",
    "too_long": "warning",
    "gap_too_small": "warning",
    "shot_change_proximity": "warning",
    "bad_line_break": "warning",
    "unbalanced": "info",
}


# ------------------------------------------------------------------ rules


@dataclass
class StyleRules:
    language: Optional[str]
    preset: str
    name: str
    source: str
    fps: float
    #: Characters per line, in the guide's own units (full-width for CJK).
    cpl: float
    #: Characters per second.
    cps: float
    max_lines: int
    min_duration_s: float
    max_duration_s: float
    min_gap_frames: float
    #: "Half a second" in whole frames (12 at 23.976/24/25, 15 at 30).
    half_second_frames: int
    tail_s: float = 0.5
    pause_s: float = PAUSE_S
    dash: str = "- "
    dash_second_only: bool = False
    italics: bool = True
    width: str = "latin"
    children: bool = False
    sdh: bool = False

    @property
    def frame_s(self) -> float:
        return 1.0 / self.fps

    @property
    def min_gap_s(self) -> float:
        return self.min_gap_frames / self.fps

    @property
    def half_second_s(self) -> float:
        return self.half_second_frames / self.fps

    def frames(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "preset": self.preset,
            "name": self.name,
            "source": self.source,
            "fps": round(self.fps, 3),
            "cpl": self.cpl,
            "cps": self.cps,
            "maxLines": self.max_lines,
            "minDurationS": round(self.min_duration_s, 3),
            "maxDurationS": self.max_duration_s,
            "minGapFrames": self.min_gap_frames,
            "chainBelowFrames": self.half_second_frames,
            "shotChangeWindowFrames": self.half_second_frames,
            "dash": self.dash,
            "italics": self.italics,
            "children": self.children,
            "sdh": self.sdh,
        }


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and math.isfinite(number) else None


def rules_for(language: Optional[str], options: Optional[dict] = None, fps: Optional[float] = None) -> StyleRules:
    """The rules for ``language`` with the user's overrides applied.

    ``options`` is a ``StyleOptions`` dict (camelCase); ``None`` values mean
    "the guide's value". An unknown language gets the general Netflix
    figures. Frame-based values (the 2-frame gap, the half-second windows)
    are converted with ``fps``, which defaults to 23.976 when the video's
    rate is not known.
    """
    options = options or {}
    lang = normalize(language) if language and language != "auto" else None
    preset = P.PRESETS.get(lang or "", P.DEFAULT)
    rate = _positive(fps) or DEFAULT_FPS
    children = bool(options.get("children"))
    sdh = bool(options.get("sdh"))

    if sdh:
        cps = preset.cps_sdh_children if children else preset.cps_sdh
    else:
        cps = preset.cps_children if children else preset.cps
    cpl = preset.cpl_sdh if sdh and preset.cpl_sdh else preset.cpl
    max_lines = preset.max_lines_sdh if sdh and preset.max_lines_sdh else preset.max_lines
    timing = P.GENERAL_TIMING
    rules = StyleRules(
        language=lang,
        preset="custom" if options.get("preset") == "custom" else "netflix",
        name=preset.name,
        source=preset.source,
        fps=rate,
        cpl=float(cpl),
        cps=float(cps),
        max_lines=int(max_lines),
        min_duration_s=float(preset.min_duration_s or timing["min_duration_s"]),
        max_duration_s=float(timing["max_duration_s"]),
        min_gap_frames=float(timing["min_gap_frames"]),
        # round() is banker's: 11.988 -> 12, 12.5 -> 12 (25 fps), 14.985 -> 15.
        half_second_frames=max(1, int(round(rate * timing["half_second_s"]))),
        tail_s=float(timing["tail_s"]),
        dash=preset.dash,
        dash_second_only=preset.dash_second_only,
        italics=preset.italics,
        width=preset.width,
        children=children,
        sdh=sdh,
    )
    for key, attr, cast in (
        ("cps", "cps", float),
        ("cpl", "cpl", float),
        ("maxLines", "max_lines", int),
        ("minDurationS", "min_duration_s", float),
        ("maxDurationS", "max_duration_s", float),
        ("minGapFrames", "min_gap_frames", float),
    ):
        value = _positive(options.get(key))
        if value is not None:
            setattr(rules, attr, cast(value))
    rules.max_lines = max(1, rules.max_lines)
    rules.max_duration_s = max(rules.max_duration_s, rules.min_duration_s)
    return rules


# ------------------------------------------------------------ measurement


def _mode_for(language: Optional[str]) -> str:
    lang = normalize(language) if language else None
    return P.PRESETS.get(lang or "", P.DEFAULT).width


def _char_width(ch: str, mode: str) -> float:
    if ch in "\n\r":
        return 0.0
    if mode == "latin":
        return 1.0
    if mode == "thai":
        return 0.0 if unicodedata.category(ch) in ("Mn", "Me") else 1.0
    wide = unicodedata.east_asian_width(ch) in ("W", "F")
    if mode == "korean":
        return 1.0 if wide and not unicodedata.category(ch).startswith("P") else 0.5
    return 1.0 if wide else 0.5  # cjk


def strip_markup(text: str) -> str:
    return _TAG_RE.sub("", text)


def text_width(text: str, language: Optional[str] = None, mode: Optional[str] = None) -> float:
    """Characters as the language's guide counts them, markup excluded."""
    mode = mode or _mode_for(language)
    return sum(_char_width(ch, mode) for ch in strip_markup(text))


def _rules_width(text: str, rules: StyleRules) -> float:
    return text_width(text, mode=rules.width)


# ----------------------------------------------------------- line breaking
#
# The text is "flowed": markup is lifted out into zero-width anchors keyed by
# position, line breaks become ordinary separators, and the plain string is
# laid out. Writing the markup back at the same positions keeps every tag
# balanced across the new breaks without having to understand it.


@dataclass
class _Flow:
    text: str
    #: position in ``text`` -> markup that sits before that character.
    tags: Dict[int, List[str]]
    #: The breaks the input had: position -> True when the break consumed a
    #: space (spaced scripts) or False when it sat between two characters.
    breaks: Dict[int, bool]


def _flow(text: str, unspaced: bool) -> _Flow:
    chars: List[str] = []
    tags: Dict[int, List[str]] = {}
    breaks: Dict[int, bool] = {}
    pending_break = False
    for piece in _TAG_RE.split(text.replace("\r\n", "\n")):
        if not piece:
            continue
        if _TAG_RE.fullmatch(piece):
            tags.setdefault(len(chars), []).append(piece)
            continue
        for ch in piece:
            if ch == "\n" or ch.isspace():
                if ch == "\n" and chars:
                    pending_break = True
                elif chars and chars[-1] not in _SPACES:
                    chars.append(FULL_SPACE if unspaced and ch == FULL_SPACE else " ")
                continue
            if pending_break:
                pending_break = False
                prev = chars[-1] if chars else ""
                if prev in _SPACES:
                    breaks[len(chars) - 1] = True
                elif unspaced and not (_is_ascii_word(prev) and _is_ascii_word(ch)):
                    breaks[len(chars)] = False
                else:
                    chars.append(" ")
                    breaks[len(chars) - 1] = True
            chars.append(ch)
    # Trailing spaces go; markup anchored past them moves to the new end.
    while chars and chars[-1] in _SPACES:
        chars.pop()
        breaks.pop(len(chars), None)
    end = len(chars)
    for pos in sorted(p for p in tags if p > end):
        tags.setdefault(end, []).extend(tags.pop(pos))
    return _Flow("".join(chars), tags, breaks)


def _is_ascii_word(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalnum()


def _render(flow: _Flow, breaks: Dict[int, bool]) -> str:
    out: List[str] = []
    text = flow.text
    for i in range(len(text) + 1):
        anchored = flow.tags.get(i, [])
        if i in breaks:
            # Closing markup stays on the line it closes, opening markup
            # moves to the line it opens.
            out.extend(t for t in anchored if t.startswith("</"))
            out.append("\n")
            out.extend(t for t in anchored if not t.startswith("</"))
            if i < len(text) and not breaks[i]:
                out.append(text[i])
            continue
        out.extend(anchored)
        if i < len(text):
            out.append(text[i])
    return "".join(out)


def _lines_of(text: str, breaks: Dict[int, bool]) -> List[Tuple[int, int]]:
    """(start, end) spans of each line of ``text`` for the given breaks."""
    spans = []
    start = 0
    for pos in sorted(breaks):
        spans.append((start, pos))
        start = pos + 1 if breaks[pos] else pos
    spans.append((start, len(text)))
    return spans


def _speaker_starts(flow: _Flow) -> List[int]:
    """Positions where a dual-speaker line begins ("-Hi." / "- Hi.")."""
    text = flow.text
    starts = []
    for i, ch in enumerate(text):
        if ch not in DASHES:
            continue
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if not nxt or nxt in DASHES:
            continue
        if i == 0:
            starts.append(0)
            continue
        before = text[:i].rstrip()
        at_line_start = (i - 1) in flow.breaks or i in flow.breaks
        after_sentence = bool(before) and before[-1] in SENTENCE_END + "\"”»」』)" and text[i - 1] in _SPACES
        if (at_line_start or after_sentence) and (nxt == " " or not nxt.isspace()):
            starts.append(i)
    return starts


def _last_word(s: str) -> str:
    words = _WORD_CHARS.findall(s)
    return words[-1] if words else ""


def _first_word(s: str) -> str:
    words = _WORD_CHARS.findall(s)
    return words[0] if words else ""


def _break_score(left: str, right: str, lang: Optional[str], unspaced: bool, consumed: bool) -> float:
    """Bonus (negative) or penalty for a break between ``left`` and ``right``.

    ``FORBID`` or more means the break splits something that must stay
    together (see ``_ja_cost``).
    """
    score = 0.0
    tail = left.rstrip()
    last = tail[-1] if tail else ""
    if last and last in SENTENCE_END:
        score -= 30
    elif last and (last in CLAUSE_END or (unspaced and last in P.CJK_BREAK_AFTER)):
        score -= 22
    if unspaced:
        if consumed:
            # A space between two Latin words is part of an English phrase
            # quoted in CJK text, not the phrase separator Chinese uses in
            # place of a comma (or Japanese between two sentences).
            if _is_ascii_word(tail[-1:]) and _is_ascii_word(right[:1]):
                return score + 8
            # In Japanese the full-width space stands for a sentence end.
            return score - (30 if lang == "ja" else 22)
        if not last or last in SENTENCE_END + CLAUSE_END + P.CJK_BREAK_AFTER:
            if lang == "ja" and last in _CLOSING and right[:1] in P.JA_ATTACHING and not _ja_word_start(tail, right):
                # 「待って」|と言った: the quotative particle belongs to the quote.
                return score + FORBID
            return score
        if lang == "ja":
            return score + _ja_cost(tail, right)
        if lang in ("zh", "yue"):
            return score + _zh_cost(tail, right)
        return score + 8
    words_before = tail.split()
    word = _last_word(tail).lower()
    nxt = _first_word(right).lower()
    keep = P.KEEP_WITH_NEXT.get(lang or "", frozenset())
    before = P.BREAK_BEFORE.get(lang or "", frozenset())
    if not last or last.isalnum():
        if word in keep or (words_before and words_before[-1].lower() in keep):
            score += 25
        if nxt in P.KEEP_WITH_PREVIOUS.get(lang or "", frozenset()):
            score += 25
        if nxt in before:
            score -= 12
        first_right = _first_word(right)
        last_left = _last_word(tail)
        # "John / Smith", "New / York": two capitalised words in a row that
        # are not the start of a sentence are probably one name.
        if (
            last_left[:1].isupper()
            and first_right[:1].isupper()
            and len(words_before) > 1
            and words_before[-2][-1:] not in SENTENCE_END
            and last_left not in ("I",)
        ):
            score += 15
        if last_left.isdigit() and first_right[:1].isalpha():
            score += 10
    return score


def _script(ch: str) -> str:
    code = ord(ch) if ch else 0
    if 0x3040 <= code <= 0x309F:
        return "hira"
    if 0x30A0 <= code <= 0x30FF or 0xFF66 <= code <= 0xFF9F or 0x31F0 <= code <= 0x31FF:
        return "kata"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or ch == "々":
        return "han"
    if ch.isdigit():
        return "num"
    if ch.isalpha() and (ch.isascii() or 0xFF21 <= code <= 0xFF5A):
        return "latin"
    return "other"


_CONTENT = ("han", "kata", "latin", "num")
_CLOSING = "」』）)】》〉］｝”’"


def _ja_particle(tail: str, right: str = "") -> Optional[str]:
    """The particle ``tail`` ends with, if it plausibly is one.

    A one-kana particle counts after kanji, katakana or Latin (映画を, 駅に,
    iPhoneで). After hiragana the same kana is often inside a word -- the が
    of ありがとう, the に of なにか -- so there it counts only when a kanji
    or katakana word follows, which a word's own kana almost never does
    (子供たちは|今, ここに|足). を is never anything but a particle.
    """
    for particle in P.JA_PARTICLES:
        if tail.endswith(particle):
            if len(particle) > 1 or particle == "を" or _script(tail[-2:-1]) in _CONTENT:
                return particle
            if _script(right[:1]) in _CONTENT:
                return particle
            return None
    return None


def _ja_prenominal(tail: str) -> bool:
    for word in P.JA_PRENOMINALS:
        if tail.endswith(word):
            # など|の is a particle pair ("such as"), not どの ("which").
            return not (word == "どの" and tail.endswith("などの"))
    return False


def _ja_word_start(tail: str, right: str) -> bool:
    """``right`` begins with a hiragana word that starts a phrase."""
    for word in P.JA_WORD_STARTS:
        if right.startswith(word):
            # After a content word a leading で/も/と is that word's
            # particle: 学校|でも is 学校でも, 今日|もう is not a new phrase.
            return not (word[0] in P.JA_ATTACHING and _script(tail[-1:]) != "hira")
    return False


def _ja_cost(tail: str, right: str) -> float:
    """Cost of a Japanese line break between ``tail`` and ``right``.

    No dictionary is available, so this reads the closed word classes and
    the script changes that decide most breaks. Kana after kanji is its
    inflection (戦|う, 立ち|上がる); a particle belongs to the word before it
    and a particle after a particle to the pair (誰に|も); a prenominal to
    the noun after it (この|子). The good breaks are after a particle that is
    followed by a new content word (背中に|隠そう), and before one of the few
    hiragana words that begin a phrase (しまった|この子たち).
    """
    a, b = tail[-1:], right[:1]
    sa, sb = _script(a), _script(b)
    if _ja_prenominal(tail):
        return FORBID
    if sa == "num" and sb in ("han", "hira", "kata"):
        return FORBID  # 7|件, 3|つ: a number and its counter are one word
    starts_word = _ja_word_start(tail, right)
    if b in P.JA_ATTACHING and not starts_word:
        after = right[1:2]
        if sa in _CONTENT:
            return FORBID  # 足|を, 人|は: the particle or ending of that word
        # After hiragana the kana may begin a word (とても, やば), but when
        # the next character is not hiragana, or a second particle follows,
        # it can only be a particle: 誰に|も断ち, ここ|に足.
        if not after or _script(after) != "hira" or right[:2] in P.JA_PARTICLES:
            return FORBID
    particle = _ja_particle(tail, right)
    if particle:
        if particle in ("ね", "よ", "か"):
            return -12  # sentence-final: the thought ended here
        content = sb in _CONTENT or b in P.NO_LINE_END or starts_word
        if particle == "の":
            return -3 if content else 2
        return -12 if content else -6
    if starts_word:
        return -8
    if sa == "han" and sb == "hira":
        return FORBID  # 戦|うんだ: a kanji stem and its okurigana
    if sa == "hira" and _script(tail[-2:-1]) == "han" and sb in ("han", "hira"):
        return FORBID  # 立ち|上がる, 打ち|ひしがれ, 断ち|切る: a compound verb
    if sa == "hira" and sb in _CONTENT:
        return 0  # kanji or katakana after kana usually starts a new word
    if sa == "hira" and sb == "hira":
        # Deep inside the kana that follow a kanji stem the break is almost
        # always mid-inflection (打ちひし|がれ). In a run of plain kana it is
        # unknown, but still more likely inside a word than not: costly
        # enough that balance alone never picks it over a script change
        # (ありがとうございました|本当に, not ござ|いました).
        k = len(tail)
        while k > 0 and _script(tail[k - 1]) == "hira":
            k -= 1
        return 35 if k > 0 and _script(tail[k - 1]) == "han" else 25
    if sa == "kata" and sb == "kata":
        return 30  # one loanword
    if sa == "kata" and sb == "hira":
        return 15  # a katakana stem's kana (サボ|る)
    return 8


def _zh_cost(tail: str, right: str) -> float:
    """Chinese has no inflection to read, so only the certain cases count: a
    number keeps its measure word, a particle stays with the word it closes,
    and a sentence-final particle is the clause end unpunctuated speech has."""
    a, b = tail[-1:], right[:1]
    sa, sb = _script(a), _script(b)
    if sa == "num" and sb == "han":
        return FORBID  # 3|个, 10|点
    if b in P.ZH_ATTACHING:
        return 30
    if a in P.ZH_FINAL and sb in _CONTENT:
        return -10
    if right.startswith(P.ZH_BREAK_BEFORE):
        return -6
    return 8


def _layout_penalty(
    text: str,
    breaks: Dict[int, bool],
    rules: StyleRules,
    lang: Optional[str],
    unspaced: bool,
    bounds: Optional[Dict[int, float]] = None,
) -> float:
    spans = _lines_of(text, breaks)
    widths = [_rules_width(text[a:b], rules) for a, b in spans]
    # Balance is a tie-breaker, never a reason to split a word: for CJK a
    # character carries more text than a Latin letter, so a per-character
    # weight tuned for 42 Latin characters would let a few characters of
    # imbalance outweigh a real word boundary.
    scale = 42.0 / max(1.0, rules.cpl) * (0.6 if unspaced else 1.0)
    penalty = 0.0
    for w in widths:
        if w > rules.cpl + 1e-9:
            penalty += 10000 + 1000 * (w - rules.cpl)
    for pos, consumed in breaks.items():
        left = text[:pos]
        right = text[pos + 1:] if consumed else text[pos:]
        penalty += _break_score(left, right, lang, unspaced, consumed)
        if bounds is not None and not consumed:
            # The recogniser's own word boundaries: breaking on one cannot
            # split a word, and one it heard a pause at is a phrase end.
            penalty += -6 - 20 * min(bounds[pos], 0.6) if pos in bounds else 15
    for (a, b), (c, d), wa, wb in zip(spans, spans[1:], widths, widths[1:]):
        diff = wa - wb
        # Bottom-heavy pyramid: a longer top line costs more than a longer
        # bottom one.
        penalty += scale * (diff * 1.0 if diff > 0 else -diff * 0.5)
        if not unspaced:
            top_words = len(text[a:b].split())
            if top_words == 1:
                penalty += 20
            elif top_words == 2:
                penalty += 6
    return penalty


def _forbidden(text: str, breaks: Dict[int, bool], lang: Optional[str], unspaced: bool) -> bool:
    """Whether any of ``breaks`` is one the language does not allow."""
    return any(
        _break_score(text[:pos], text[pos + 1:] if consumed else text[pos:], lang, unspaced, consumed) >= FORBID / 2
        for pos, consumed in breaks.items()
    )


def _is_run_char(ch: str) -> bool:
    """Part of a half-width run -- a Latin word, 3.5, 10:30, 50%, COVID-19 --
    or its full-width equivalent; CJK text never breaks inside one."""
    if not ch or ch.isspace():
        return False
    return ch.isascii() or _script(ch) in ("latin", "num")


def _candidates(text: str, unspaced: bool) -> List[Tuple[int, bool]]:
    out: List[Tuple[int, bool]] = []
    for i, ch in enumerate(text):
        if ch in _SPACES:
            if 0 < i < len(text) - 1:
                out.append((i, True))
            continue
        if not unspaced or i == 0:
            continue
        prev = text[i - 1]
        if prev in _SPACES:
            continue
        if _is_run_char(prev) and _is_run_char(ch):
            continue
        if ch in P.NO_LINE_START or prev in P.NO_LINE_END:
            continue
        out.append((i, False))
    return out


def _search(
    text: str,
    rules: StyleRules,
    lang: Optional[str],
    unspaced: bool,
    max_lines: int,
    candidates: List[Tuple[int, bool]],
    bounds: Optional[Dict[int, float]],
) -> Tuple[float, Dict[int, bool]]:
    """The lowest-penalty layout using only ``candidates``, and its penalty."""
    best: Optional[Tuple[float, Dict[int, bool]]] = None
    for lines in range(2, max_lines + 1):
        combos = itertools.combinations(candidates, lines - 1)
        if math.comb(len(candidates), lines - 1) > 20000:
            combos = iter([_greedy(text, candidates, rules, lines)])
        for combo in combos:
            breaks = dict(combo)
            penalty = _layout_penalty(text, breaks, rules, lang, unspaced, bounds)
            if best is None or penalty < best[0]:
                best = (penalty, breaks)
        if best is not None and best[0] < 10000:
            break  # the fewest lines that fit
    return best if best else (math.inf, {})


def _best_layout(
    text: str,
    rules: StyleRules,
    lang: Optional[str],
    unspaced: bool,
    max_lines: int,
    bounds: Optional[Dict[int, float]] = None,
) -> Dict[int, bool]:
    if _rules_width(text, rules) <= rules.cpl + 1e-9 or max_lines <= 1:
        return {}
    candidates = _candidates(text, unspaced)
    if not candidates:
        return {}
    if bounds is not None:
        # The recogniser's word boundaries first: a layout built only from
        # them cannot split a word. Its tokens are sub-word pieces for
        # unspaced scripts, though, so the language rules still apply, and
        # only a clean layout (fits, nothing forbidden) ends the search.
        on_words = [c for c in candidates if c[1] or c[0] in bounds]
        if on_words:
            pen, laid = _search(text, rules, lang, unspaced, max_lines, on_words, bounds)
            if pen < 10000 and not _forbidden(text, laid, lang, unspaced):
                return laid
            full_pen, full = _search(text, rules, lang, unspaced, max_lines, candidates, bounds)
            return laid if pen <= full_pen else full
    return _search(text, rules, lang, unspaced, max_lines, candidates, bounds)[1]


def _greedy(text: str, candidates: List[Tuple[int, bool]], rules: StyleRules, lines: int) -> Tuple[Tuple[int, bool], ...]:
    """Evenly spaced breaks, for the rare custom layouts with many lines."""
    total = _rules_width(text, rules)
    chosen: List[Tuple[int, bool]] = []
    for k in range(1, lines):
        target = total * k / lines
        pick = min(candidates, key=lambda c: abs(_rules_width(text[: c[0]], rules) - target))
        if pick not in chosen:
            chosen.append(pick)
    return tuple(sorted(chosen))


def _layout(
    text: str, rules: StyleRules, language: Optional[str], bounds: Optional[Dict[int, float]] = None
) -> Tuple[_Flow, Dict[int, bool]]:
    lang = normalize(language) if language else rules.language
    unspaced = lang in UNSPACED
    flow = _flow(text, unspaced)
    speakers = [s for s in _speaker_starts(flow) if s > 0]
    if speakers:
        # One speaker per line, always; a speaker's line that is too long
        # stays too long (check() reports it) rather than breaking the
        # one-speaker-per-line convention.
        breaks: Dict[int, bool] = {}
        for pos in speakers:
            if flow.text[pos - 1] in _SPACES:
                breaks[pos - 1] = True
            else:
                breaks[pos] = False
        return flow, breaks
    return flow, _best_layout(flow.text, rules, lang, unspaced, rules.max_lines, bounds)


def break_lines(text: str, rules: StyleRules, language: Optional[str] = None) -> str:
    """Re-flow ``text`` into at most ``rules.max_lines`` lines of ``rules.cpl``.

    One line when it fits (every guide: "keep to one line unless it exceeds
    the character limitation"). Otherwise break after punctuation or before
    a conjunction or preposition, never after an article or between names,
    preferring a bottom-heavy pyramid. Dual-speaker cues keep one speaker per
    line. Markup is kept balanced; only whitespace and line breaks change.
    When nothing fits, the least-bad layout is returned and ``check`` flags it.
    """
    flow, breaks = _layout(text, rules, language)
    return _render(flow, breaks)


def _break_words(text: str, rules: StyleRules, lang: Optional[str], bounds: Optional[Dict[int, float]]) -> str:
    """``break_lines`` for plain text whose word boundaries are known."""
    flow, breaks = _layout(text, rules, lang, bounds)
    return _render(flow, breaks)


# ----------------------------------------------------------------- issues


@dataclass
class Issue:
    index: int
    type: str
    message: str
    severity: str = "warning"
    time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"index": self.index, "type": self.type, "message": self.message, "severity": self.severity}
        if self.time is not None:
            out["time"] = round(self.time, 3)
        return out


def _issue(index: int, kind: str, message: str, cue: Optional[Cue] = None) -> Issue:
    return Issue(index, kind, message, SEVERITY.get(kind, "warning"), cue.start if cue else None)


def _cps(cue: Cue, rules: StyleRules) -> float:
    width = _rules_width(cue.text.replace("\n", ""), rules)
    return width / max(cue.duration, 1e-3)


def _line_issues(index: int, cue: Cue, rules: StyleRules, lang: Optional[str]) -> List[Issue]:
    issues: List[Issue] = []
    lines = [line.strip() for line in cue.plain.split("\n")]
    if len(lines) > rules.max_lines:
        issues.append(_issue(index, "too_many_lines", f"{len(lines)} lines; the limit is {rules.max_lines}.", cue))
    widths = [_rules_width(line, rules) for line in lines]
    worst = max(widths) if widths else 0
    if worst > rules.cpl + 1e-9:
        issues.append(_issue(index, "cpl_exceeded", f"A line is {_fmt(worst)} characters; the limit is {_fmt(rules.cpl)}.", cue))
    if issues:
        return issues
    unspaced = lang in UNSPACED
    flow, best = _layout(cue.text, rules, lang)
    if any(s > 0 for s in _speaker_starts(flow)):
        if len(lines) < 2:
            issues.append(_issue(index, "bad_line_break", "Two speakers share a line; give each its own.", cue))
        return issues
    current = flow.breaks
    if current == best:
        return issues
    current_pen = _layout_penalty(flow.text, current, rules, lang, unspaced)
    best_pen = _layout_penalty(flow.text, best, rules, lang, unspaced)
    if len(lines) > 1 and not best:
        # Fits on one line. A break after a sentence or clause is a
        # legitimate choice; any other break is just unnecessary.
        if any(flow.text[:p].rstrip()[-1:] in SENTENCE_END + CLAUSE_END + P.CJK_BREAK_AFTER for p in current):
            return issues
        issues.append(_issue(index, "bad_line_break", "Fits on one line.", cue))
    elif current_pen - best_pen > 15:
        issues.append(_issue(index, "bad_line_break", "A better line break exists: " + break_lines(cue.text, rules, lang).replace("\n", " / "), cue))
    elif len(widths) > 1 and min(widths) < 0.33 * max(widths):
        issues.append(_issue(index, "unbalanced", "Lines are very uneven in length.", cue))
    return issues


def _fmt(value: float) -> str:
    return f"{value:g}"


def check(doc: SubtitleDoc, rules: StyleRules, shot_changes: Optional[Sequence[float]] = None) -> List[Issue]:
    """Every rule the cue list breaks, cue by cue, in the doc's own order."""
    lang = rules.language or normalize(doc.language)
    issues: List[Issue] = []
    tol = 0.002  # SRT stores milliseconds; 5/6 s is written as 0.833
    cues = doc.cues
    for i, cue in enumerate(cues):
        if not cue.plain.strip():
            issues.append(_issue(i, "empty", "The subtitle has no text.", cue))
            continue
        issues.extend(_line_issues(i, cue, rules, lang))
        if cue.duration < rules.min_duration_s - tol:
            issues.append(_issue(i, "too_short", f"On screen {cue.duration:.2f} s; the minimum is {rules.min_duration_s:.2f} s.", cue))
        if cue.duration > rules.max_duration_s + tol:
            issues.append(_issue(i, "too_long", f"On screen {cue.duration:.2f} s; the maximum is {_fmt(rules.max_duration_s)} s.", cue))
        cps = _cps(cue, rules)
        if cps > rules.cps + 0.05:
            issues.append(_issue(i, "cps_exceeded", f"Reads at {cps:.1f} characters per second; the limit is {_fmt(rules.cps)}.", cue))

    order = sorted(range(len(cues)), key=lambda k: (cues[k].start, cues[k].end))
    for a, b in zip(order, order[1:]):
        first, second = cues[a], cues[b]
        if not first.plain.strip() or not second.plain.strip():
            continue
        gap = second.start - first.end
        frames = rules.frames(gap)
        if gap < -0.0005 and _same_layer(first, second):
            issues.append(_issue(a, "overlap", f"Overlaps the next subtitle by {-gap:.3f} s.", first))
        elif gap >= -0.0005 and frames < rules.min_gap_frames:
            issues.append(_issue(a, "gap_too_small", f"{max(frames, 0)} frames before the next subtitle; the minimum is {_fmt(rules.min_gap_frames)}.", first))
        elif rules.min_gap_frames < frames < rules.half_second_frames:
            issues.append(_issue(a, "gap_too_small", f"A {frames}-frame gap should be closed to {_fmt(rules.min_gap_frames)} frames.", first))

    if shot_changes:
        cuts = sorted(shot_changes)
        for i, cue in enumerate(cues):
            if cue.plain.strip():
                message = _shot_message(cue, cuts, rules)
                if message:
                    issues.append(_issue(i, "shot_change_proximity", message, cue))
    issues.sort(key=lambda issue: issue.index)
    return issues


def _same_layer(a: Cue, b: Cue) -> bool:
    """Cues at different positions (a sign at the top, dialogue at the
    bottom) may overlap on purpose."""
    return (a.align or 2) == (b.align or 2) and not (a.forced ^ b.forced)


def _nearest(cuts: Sequence[float], t: float) -> Optional[float]:
    if not cuts:
        return None
    k = bisect.bisect_left(cuts, t)
    options = [cuts[j] for j in (k - 1, k) if 0 <= j < len(cuts)]
    return min(options, key=lambda c: abs(c - t))


def _shot_message(cue: Cue, cuts: Sequence[float], rules: StyleRules) -> Optional[str]:
    half = rules.half_second_frames
    gap = int(round(rules.min_gap_frames))
    sc = _nearest(cuts, cue.start)
    if sc is not None:
        d = rules.frames(cue.start - sc)
        if 0 < d < half:
            return f"Starts {d} frames after a shot change; start on the cut."
        if -half < d < 0 and cue.end > sc + rules.frame_s:
            return f"Starts {-d} frames before a shot change; start on the cut or half a second before it."
    sc = _nearest(cuts, cue.end)
    if sc is not None:
        e = rules.frames(cue.end - sc)
        if -half < e < -gap:
            return f"Ends {-e} frames before a shot change; end {gap} frames before the cut."
        if -gap < e < half:
            where = f"{e} frames after" if e >= 0 else f"{-e} frame before"
            return f"Ends {where} a shot change; end {gap} frames before the cut or half a second after it."
    return None


# -------------------------------------------------------------------- fix


def fix(doc: SubtitleDoc, rules: StyleRules, shot_changes: Optional[Sequence[float]] = None) -> Tuple[SubtitleDoc, List[Issue]]:
    """Repair what ``check`` finds, changing only times and line breaks.

    The order matters: lines are re-broken first (a cue that fits once
    re-broken needs no split), over-long cues are split next (the split
    points depend on the text), then durations and gaps are set, then
    in/out times are snapped to shot changes, and a last pass restores the
    gap rules the snapping may have disturbed. Returns the new doc and one
    Issue per repair, indexed into the new doc.
    """
    lang = rules.language or normalize(doc.language)
    work = doc.sorted()
    fixed: List[Issue] = []

    for i, cue in enumerate(work.cues):
        if not cue.plain.strip():
            continue
        kinds = [x.type for x in _line_issues(i, cue, rules, lang)]
        if kinds:
            text = break_lines(cue.text, rules, lang)
            if text != cue.text:
                cue.text = text
                fixed.append(_issue(i, kinds[0], "Re-broke the lines.", cue))

    cues: List[Cue] = []
    for cue in work.cues:
        parts = _split(cue, rules, lang) if cue.plain.strip() else [cue]
        if len(parts) > 1:
            kind = "too_long" if cue.duration > rules.max_duration_s + 0.002 else "too_many_lines"
            for part in parts:
                fixed.append(_issue(len(cues), kind, f"Split a long subtitle into {len(parts)}.", part))
                cues.append(part)
        else:
            cues.append(cue)

    _retime(cues, rules, fixed, extend=True)
    if shot_changes:
        _snap(cues, sorted(shot_changes), rules, fixed)
        _retime(cues, rules, fixed, extend=False)
    return work.copy(cues), _dedupe(fixed)


def _dedupe(issues: List[Issue]) -> List[Issue]:
    seen = set()
    out = []
    for issue in issues:
        key = (issue.index, issue.type, issue.message)
        if key not in seen:
            seen.add(key)
            out.append(issue)
    out.sort(key=lambda x: x.index)
    return out


def _needs_split(cue: Cue, rules: StyleRules, lang: Optional[str]) -> bool:
    if cue.duration > rules.max_duration_s + 0.002:
        return True
    lines = cue.plain.split("\n")
    if len(lines) > rules.max_lines:
        return True
    return any(_rules_width(line, rules) > rules.cpl + 1e-9 for line in lines)


def _split(cue: Cue, rules: StyleRules, lang: Optional[str], depth: int = 0) -> List[Cue]:
    """Split at the best sentence/clause boundary, recursively, sharing the
    time by character count."""
    if depth > 6 or not _needs_split(cue, rules, lang):
        return [cue]
    unspaced = lang in UNSPACED
    flow = _flow(cue.text, unspaced)
    text = flow.text
    total = _rules_width(text, rules)
    if total <= 0:
        return [cue]
    speakers = set(s for s in _speaker_starts(flow) if s > 0)
    capacity = rules.cpl * rules.max_lines
    best: Optional[Tuple[float, int, bool]] = None
    for pos, consumed in _candidates(text, unspaced):
        left = text[:pos].rstrip()
        right = text[pos + 1:] if consumed else text[pos:]
        wl = _rules_width(left, rules)
        share = wl / total
        score = 80 * abs(share - 0.5) + _break_score(left, right, lang, unspaced, consumed)
        if (pos + 1 if consumed else pos) in speakers:
            score -= 60
        if left[-1:] in SENTENCE_END:
            # Between two subtitles a sentence end matters more than inside
            # one: the viewer's eye resets with the new subtitle.
            score -= 25
        for part_w in (wl, total - wl):
            if part_w > capacity:
                score += 10 * (part_w / capacity)  # the recursion splits it again
        if best is None or score < best[0]:
            best = (score, pos, consumed)
    if best is None:
        return [cue]
    _, pos, consumed = best
    first_text, second_text = _split_markup(cue.text, flow, pos, consumed)
    share = _rules_width(text[:pos], rules) / total
    cut = cue.start + cue.duration * share
    gap = min(rules.min_gap_s, cue.duration * 0.05)
    if cue.duration >= 2 * rules.min_duration_s + gap:
        # Share by characters, but never below the minimum duration: a
        # short "Yes." split off a long line would otherwise flash by.
        cut = min(max(cut, cue.start + rules.min_duration_s + gap), cue.end - rules.min_duration_s)
    first = cue.copy(end=max(cue.start + rules.frame_s, cut - gap), raw=None)
    second = cue.copy(start=cut, raw=None)
    first.text = break_lines(first_text, rules, lang)
    second.text = break_lines(second_text, rules, lang)
    return _split(first, rules, lang, depth + 1) + _split(second, rules, lang, depth + 1)


def _split_markup(text: str, flow: _Flow, pos: int, consumed: bool) -> Tuple[str, str]:
    """Cut tagged text at a flow position, closing open tags at the end of
    the first half and reopening them at the start of the second."""
    start = pos + 1 if consumed else pos
    first_tags: Dict[int, List[str]] = {k: list(v) for k, v in flow.tags.items() if k < pos}
    rest_tags: Dict[int, List[str]] = {}
    for tag in flow.tags.get(pos, []):
        # Markup at the cut: closings end the first half, openings begin the second.
        if tag.startswith("</"):
            first_tags.setdefault(pos, []).append(tag)
        else:
            rest_tags.setdefault(0, []).append(tag)
    for k, v in flow.tags.items():
        if k > pos and k >= start:
            rest_tags.setdefault(k - start, []).extend(v)
    first = _render(_Flow(flow.text[:pos], first_tags, {}), {})
    second = _render(_Flow(flow.text[start:], rest_tags, {}), {})
    open_tags = _open_tags(first)
    if open_tags:
        first += "".join(f"</{name}>" for name, _ in reversed(open_tags))
        second = "".join(tag for _, tag in open_tags) + second
    return first, second


def _open_tags(text: str) -> List[Tuple[str, str]]:
    stack: List[Tuple[str, str]] = []
    for tag in re.findall(r"</?(?:i|b|u|font)(?:\s[^>]*)?>", text, re.IGNORECASE):
        name = re.match(r"</?(\w+)", tag).group(1).lower()
        if tag.startswith("</"):
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == name:
                    del stack[k]
                    break
        else:
            stack.append((name, tag))
    return stack


def _retime(cues: List[Cue], rules: StyleRules, fixed: List[Issue], extend: bool) -> None:
    """Overlaps, minimum duration, reading speed, and the 2-frame chain.

    Time is only ever taken from empty space: a cue grows into the gap after
    it (then before it), and never into its neighbour.
    """
    gap = rules.min_gap_s
    tol = 0.0005
    for i, cue in enumerate(cues):
        if not cue.plain.strip():
            continue
        nxt = next((c for c in cues[i + 1:] if c.plain.strip() and _same_layer(cue, c)), None)
        prev = next((c for c in reversed(cues[:i]) if c.plain.strip() and _same_layer(cue, c)), None)
        limit = nxt.start - gap if nxt else math.inf
        floor = prev.end + gap if prev else 0.0

        if nxt and cue.end > limit + tol:
            kind = "overlap" if cue.end > nxt.start + tol else "gap_too_small"
            if limit - cue.start >= min(rules.min_duration_s, cue.duration) - tol:
                cue.end = limit
            else:
                # Trimming would leave nothing readable: push the next cue
                # instead, if it can spare the time.
                wanted = cue.start + min(rules.min_duration_s, cue.duration) + gap
                if nxt.end - wanted >= rules.min_duration_s:
                    nxt.start = wanted
                cue.end = max(cue.start + rules.frame_s, nxt.start - gap)
                if cue.end > nxt.start - gap and nxt.end - (cue.end + gap) >= rules.frame_s:
                    nxt.start = cue.end + gap  # never leave an overlap behind
            limit = nxt.start - gap
            fixed.append(_issue(i, kind, f"Ended it {_fmt(rules.min_gap_frames)} frames before the next subtitle.", cue))

        if extend or cue.duration < rules.min_duration_s - 0.002:
            if cue.duration < rules.min_duration_s - 0.002:
                was = cue.duration
                cue.end = max(cue.end, min(cue.start + rules.min_duration_s, limit))
                if cue.duration < rules.min_duration_s - 0.002:
                    start = max(floor, min(cue.start, cue.end - rules.min_duration_s))
                    if prev and start < cue.start and 0 < start - floor < rules.half_second_s:
                        start = floor  # chain to the previous cue rather than leave a short gap
                    cue.start = start
                if cue.duration < rules.min_duration_s - 0.002:
                    _borrow(cue, prev, nxt, rules)
                    limit = nxt.start - gap if nxt else math.inf
                if cue.duration > was + 0.001:
                    fixed.append(_issue(i, "too_short", f"Lengthened to {cue.duration:.2f} s.", cue))
            if extend:
                need = _rules_width(cue.text.replace("\n", ""), rules) / rules.cps
                if cue.duration < need - 0.002:
                    target = min(cue.start + need, limit, cue.start + rules.max_duration_s)
                    if target > cue.end + 0.001:
                        cue.end = target
                        fixed.append(_issue(i, "cps_exceeded", f"Lengthened to {cue.duration:.2f} s for reading speed.", cue))

        if nxt:
            frames = rules.frames(nxt.start - cue.end)
            if rules.min_gap_frames < frames < rules.half_second_frames:
                if nxt.start - gap - cue.start <= rules.max_duration_s + 0.0005:
                    cue.end = nxt.start - gap
                    fixed.append(_issue(i, "gap_too_small", f"Closed a {frames}-frame gap to {_fmt(rules.min_gap_frames)} frames.", cue))
                elif nxt.start - rules.half_second_s - cue.start >= rules.min_duration_s:
                    # Closing would pass the maximum duration: open the gap
                    # to half a second instead, the other permitted size.
                    cue.end = nxt.start - rules.half_second_s
                    fixed.append(_issue(i, "gap_too_small", "Opened a short gap to half a second.", cue))


def _needed(cue: Cue, rules: StyleRules) -> float:
    """The time a cue needs: its reading time, and at least the minimum."""
    return max(rules.min_duration_s, _rules_width(cue.text.replace("\n", ""), rules) / rules.cps)


def _borrow(cue: Cue, prev: Optional[Cue], nxt: Optional[Cue], rules: StyleRules) -> None:
    """Take the time a too-short cue lacks from neighbours that can spare
    it (the guideline's "borrowing time"), next cue first."""
    gap = rules.min_gap_s
    for neighbour in (nxt, prev):
        deficit = rules.min_duration_s - cue.duration
        if neighbour is None or deficit <= 0.001:
            return
        spare = neighbour.duration - _needed(neighbour, rules)
        if spare <= 0.001:
            continue
        take = min(spare, deficit)
        if neighbour is nxt and nxt.start - (cue.end + gap) < 0.0005:
            nxt.start += take
            cue.end = nxt.start - gap
        elif neighbour is prev and cue.start - (prev.end + gap) < 0.0005:
            prev.end -= take
            cue.start = prev.end + gap


def _snap(cues: List[Cue], cuts: List[float], rules: StyleRules, fixed: List[Issue]) -> None:
    """The timing guideline's shot-change rules, in-times then out-times."""
    frame = rules.frame_s
    half = rules.half_second_frames
    gap_frames = int(round(rules.min_gap_frames))
    gap = rules.min_gap_s
    for i, cue in enumerate(cues):
        if not cue.plain.strip():
            continue
        prev_end = max((c.end for c in cues[:i] if _same_layer(c, cue) and c.plain.strip()), default=-math.inf)
        sc = _nearest(cuts, cue.start)
        if sc is not None:
            d = rules.frames(cue.start - sc)
            if 0 < d < half:
                # Dialogue starting within half a second after the cut
                # starts on the cut.
                cue.start = sc
                fixed.append(_issue(i, "shot_change_proximity", "Moved the in-time onto the shot change.", cue))
            elif -half < d < 0 and cue.end > sc + frame:
                # A subtitle that crosses a cut must start at least half a
                # second before it, or on it.
                early = sc - half * frame
                on_cut_ok = cue.end - sc >= rules.min_duration_s
                early_ok = early >= prev_end + gap
                if on_cut_ok and (-d <= half / 2 or not early_ok):
                    cue.start = sc
                elif early_ok:
                    cue.start = early
                if cue.start in (sc, early):
                    fixed.append(_issue(i, "shot_change_proximity", "Moved the in-time off the shot change.", cue))

        next_start = min((c.start for c in cues[i + 1:] if _same_layer(c, cue) and c.plain.strip()), default=math.inf)
        sc = _nearest(cuts, cue.end)
        if sc is None:
            continue
        e = rules.frames(cue.end - sc)
        earlier = sc - gap_frames * frame
        later = sc + half * frame
        new_end = None
        if -half < e < -gap_frames:
            if earlier <= next_start - gap + 1e-6:
                new_end = earlier
        elif -gap_frames < e < half:
            fits_earlier = earlier - cue.start >= rules.min_duration_s - 0.002
            fits_later = later <= next_start - gap + 1e-6 and later - cue.start <= rules.max_duration_s
            if fits_earlier and (e <= half // 2 or not fits_later):
                new_end = earlier
            elif fits_later:
                new_end = later
            elif fits_earlier:
                new_end = earlier
        if new_end is not None and abs(new_end - cue.end) > 1e-6:
            cue.end = new_end
            fixed.append(_issue(i, "shot_change_proximity", "Moved the out-time to respect the shot change.", cue))


# ----------------------------------------------------------- segmentation


@dataclass
class _Token:
    start: float
    end: float
    text: str
    #: What joins it to the token before: "", " ", or FULL_SPACE for a
    #: Japanese sentence end the recogniser heard as a pause but did not
    #: punctuate.
    sep: str
    speaker: Optional[str]
    probability: Optional[float]


_ATTACHING = re.compile(r"^(?:[.,!?;:…%)\]}»”’'\"、。，．！？：；」』）]+|n't|'[a-z]{1,2}|’[a-z]{1,2})$", re.IGNORECASE)
_FRENCH_SPACED = re.compile(r"^[?!:;»]+$")


def _tokens(words: Sequence[Word], lang: Optional[str]) -> List[_Token]:
    raw = [w for w in words if (w.text or "").strip()]
    raw.sort(key=lambda w: (w.start, w.end))
    unspaced = lang in UNSPACED
    leading = [w.text[:1].isspace() for w in raw[1:]]
    whisper_style = bool(leading) and sum(leading) >= 0.5 * len(leading)
    out: List[_Token] = []
    for k, w in enumerate(raw):
        text = w.text.strip()
        prev = out[-1].text if out else ""
        if not out:
            space = False
        elif unspaced:
            space = _is_ascii_word(prev[-1:]) and _is_ascii_word(text[:1])
        elif whisper_style:
            space = w.text[:1].isspace()
        elif lang == "fr" and _FRENCH_SPACED.match(text):
            space = True
        else:
            space = not _ATTACHING.match(text)
        sep = " " if space else ""
        if not sep and out and lang == "ja" and _ja_sentence_pause(prev, text, float(w.start) - out[-1].end):
            sep = FULL_SPACE
        out.append(_Token(float(w.start), max(float(w.end), float(w.start)), text, sep, w.speaker, w.probability))
    return out


def _ja_sentence_pause(prev: str, nxt: str, pause: float) -> bool:
    """Whether a pause between two Japanese words is an unpunctuated
    sentence end. Speakers also pause after a topic (刀は…) or a te-form, so
    the word before must not end in a particle or a conjunctive ending, and
    the word after must not be a particle or ending itself.

    The recogniser's own word is the unit here: a particle comes as a word
    of its own (は, には), while a word that merely starts with the same kana
    carries more hiragana after it (やば, とても).
    """
    if pause < JA_SENTENCE_PAUSE_S or not prev or not nxt:
        return False
    if prev[-1] in SENTENCE_END + CLAUSE_END + P.CJK_BREAK_AFTER or prev.endswith(P.JA_CONTINUES):
        return False
    if nxt[0] in P.NO_LINE_START or nxt[0] in SENTENCE_END + CLAUSE_END + P.CJK_BREAK_AFTER:
        return False
    if nxt in P.JA_PARTICLES or nxt in P.JA_CONTINUES:
        return False
    return nxt[0] not in P.JA_ATTACHING or _script(nxt[1:2]) == "hira"


def _join(tokens: Sequence[_Token]) -> str:
    parts = []
    for k, tok in enumerate(tokens):
        if k:
            parts.append(tok.sep)
        parts.append(tok.text)
    return "".join(parts)


def _bounds(tokens: Sequence[_Token], lang: Optional[str]) -> Optional[Dict[int, float]]:
    """Positions between tokens in ``_join(tokens)``, for unspaced scripts,
    each with the pause the recogniser heard there (seconds).

    Line breaking uses these instead of re-deriving boundaries from the
    joined text: a boundary it knows cannot be inside a word, and one with a
    pause is where the speaker ended a phrase.
    """
    if lang not in UNSPACED:
        return None
    out: Dict[int, float] = {}
    pos = 0
    for k, tok in enumerate(tokens):
        if k:
            pos += len(tok.sep)
            out[pos] = max(0.0, tok.start - tokens[k - 1].end)
        pos += len(tok.text)
    return out


def _fits(text: str, rules: StyleRules, lang: Optional[str], bounds: Optional[Dict[int, float]] = None) -> bool:
    width = _rules_width(text, rules)
    if width <= rules.cpl:
        return True
    if width > rules.cpl * rules.max_lines:
        return False
    flow, breaks = _layout(text, rules, lang, bounds)
    if bounds is not None and any(not consumed and pos not in bounds for pos, consumed in breaks.items()):
        return False  # only fits by breaking inside a word
    if _forbidden(flow.text, breaks, lang, lang in UNSPACED):
        return False  # only fits by breaking where the language does not allow it
    laid = _render(flow, breaks)
    lines = strip_markup(laid).split("\n")
    return len(lines) <= rules.max_lines and all(_rules_width(line, rules) <= rules.cpl + 1e-9 for line in lines)


def _ends_sentence(text: str) -> bool:
    return text.rstrip("\"'”’»」』)）")[-1:] in SENTENCE_END


def _split_point(group: List[_Token], rules: StyleRules, lang: Optional[str]) -> int:
    """Where to end a subtitle that cannot take the next word: the index of
    the first word carried into the next subtitle (``len(group)`` = none).

    Carrying a short tail forward to split at a clause boundary or a pause
    reads far better than cutting wherever the line ran out.
    """
    capacity = rules.cpl * rules.max_lines
    unspaced = lang in UNSPACED
    best_k, best_score = len(group), math.inf
    for k in range(1, len(group) + 1):
        head, tail = group[:k], group[k:]
        head_text = _join(head)
        head_w = _rules_width(head_text, rules)
        tail_w = _rules_width(_join(tail), rules) if tail else 0.0
        score = 30 * tail_w / capacity
        if head_w < 0.3 * capacity:
            score += 30
        last = head[-1]
        if _ends_sentence(last.text):
            score -= 40
        elif last.text[-1:] in CLAUSE_END + P.CJK_BREAK_AFTER:
            score -= 25
        if tail:
            pause = tail[0].start - last.end
            score -= min(pause, 0.6) * 40
            if not unspaced:
                word = _last_word(last.text).lower()
                if word in P.KEEP_WITH_NEXT.get(lang or "", frozenset()):
                    score += 25
                first = _first_word(tail[0].text).lower()
                if first in P.KEEP_WITH_PREVIOUS.get(lang or "", frozenset()):
                    score += 25
                if first in P.BREAK_BEFORE.get(lang or "", frozenset()):
                    score -= 10
            elif last.text[-1:] not in SENTENCE_END + CLAUSE_END + P.CJK_BREAK_AFTER:
                # A subtitle boundary is a line break too: the same rules
                # that keep 誰に|も on one line keep it in one subtitle.
                score += _break_score(head_text, _join(tail), lang, True, bool(tail[0].sep))
        if score < best_score:
            best_k, best_score = k, score
    return best_k


def segment_words(words: Sequence[Word], rules: StyleRules, language: Optional[str] = None) -> List[Cue]:
    """Cut recognised words into subtitles that follow ``rules``.

    A subtitle ends at a long pause, a change of speaker, the end of a
    sentence (short exclamations may share one), or before it would outgrow
    its lines, its maximum duration or what can be read in that duration.
    Its time runs from the first word's start to the last word's end plus
    half a second of linger, then the same gap and duration rules ``fix``
    applies.
    """
    lang = (normalize(language) if language else None) or rules.language
    tokens = _tokens(words, lang)
    if not tokens:
        return []
    capacity = min(rules.cpl * rules.max_lines, rules.cps * rules.max_duration_s)
    groups: List[List[_Token]] = []
    current: List[_Token] = []
    for tok in tokens:
        if not current:
            current.append(tok)
            continue
        prev = current[-1]
        pause = tok.start - prev.end
        text_now = _join(current)
        candidate = _join(current + [tok])
        if (tok.speaker and prev.speaker and tok.speaker != prev.speaker) or pause >= rules.pause_s:
            groups.append(current)
            current = [tok]
            continue
        if _ends_sentence(prev.text) and (pause >= 0.3 or _rules_width(text_now, rules) > 0.4 * rules.cpl):
            groups.append(current)
            current = [tok]
            continue
        if tok.sep == FULL_SPACE and _rules_width(text_now, rules) > 0.4 * rules.cpl:
            # An unpunctuated sentence end (see _ja_sentence_pause) after a
            # sentence long enough to stand alone: its own subtitle, as if
            # the recogniser had written the 。. A short one stays, set off
            # by the full-width space.
            groups.append(current)
            current = [tok]
            continue
        overflow = (
            _rules_width(candidate, rules) > capacity
            or tok.end - current[0].start > rules.max_duration_s
            or not _fits(candidate, rules, lang, _bounds(current + [tok], lang))
        )
        if overflow:
            k = _split_point(current, rules, lang)
            carried = current[k:] + [tok]
            if k < len(current) and (
                not _fits(_join(carried), rules, lang, _bounds(carried, lang)) or carried[-1].end - carried[0].start > rules.max_duration_s
            ):
                k, carried = len(current), [tok]
            groups.append(current[:k])
            current = carried
            continue
        current.append(tok)
    if current:
        groups.append(current)

    cues: List[Cue] = []
    for n, group in enumerate(groups):
        start = group[0].start
        end = group[-1].end + rules.tail_s
        if n + 1 < len(groups):
            end = min(end, groups[n + 1][0].start - rules.min_gap_s)
        end = max(end, min(group[-1].end, start + rules.max_duration_s), start + rules.frame_s)
        end = min(end, start + rules.max_duration_s)
        probs = [t.probability for t in group if t.probability is not None]
        speakers = {t.speaker for t in group if t.speaker}
        cues.append(Cue(
            start=start,
            end=end,
            text=_break_words(_join(group), rules, lang, _bounds(group, lang)),
            speaker=speakers.pop() if len(speakers) == 1 else None,
            confidence=round(sum(probs) / len(probs), 3) if probs else None,
            meta={"words": [{"start": t.start, "end": t.end, "text": t.text} for t in group]},
        ))
    _retime(cues, rules, [], extend=True)
    return cues


# ----------------------------------------------------------- shot changes


def shot_changes(
    video_path: str,
    token: Any = None,
    progress: Optional[Callable[[float], None]] = None,
    threshold: float = 10.0,
) -> List[float]:
    """Times (s) of the first frame of each new shot, from FFmpeg.

    ``scdet`` (FFmpeg 4.4+) reports each cut with its frame's exact time;
    older builds fall back to ``select='gt(scene,..)',showinfo``. The video
    is scaled down to 192 px wide first: detection needs only coarse
    structure and this makes the decode, not the analysis, the cost.
    ``progress`` gets 0..1 from the time decoded so far.
    """
    from ..media import Cancelled, MediaError, _popen_kwargs, _terminate, ffmpeg_path, probe

    if not os.path.isfile(video_path):
        raise MediaError(f"Video not found: {video_path}")
    if token:
        token.raise_if_cancelled()
    duration = None
    try:
        duration = probe(video_path, token).duration
    except MediaError:
        pass
    ffmpeg = ffmpeg_path()
    if _has_filter(ffmpeg, "scdet"):
        vf = f"scale=192:-2,scdet=threshold={threshold:g}"
        pattern = re.compile(r"lavfi\.scd\.time:\s*([0-9.]+)")
    else:
        vf = f"scale=192:-2,select='gt(scene,{threshold / 100 * 3:.2f})',showinfo"
        pattern = re.compile(r"pts_time:\s*([0-9.]+)")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-i", video_path,
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf,
        "-progress", "pipe:1", "-f", "null", "-",
    ]
    try:
        process = subprocess.Popen(command, **_popen_kwargs())
    except OSError as exc:
        raise MediaError(f"Could not start FFmpeg: {exc}") from exc
    times: List[float] = []
    tail: List[str] = []

    def read_stderr() -> None:
        assert process.stderr is not None
        for raw in process.stderr:
            line = raw.decode("utf-8", errors="replace")
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))
            else:
                tail.append(line.strip())
                del tail[:-5]

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    if token:
        token.register(process)
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            if token and token.cancelled:
                break
            line = raw.decode("ascii", errors="replace").strip()
            if progress and duration and line.startswith("out_time_us="):
                try:
                    progress(min(1.0, max(0.0, int(line.split("=", 1)[1]) / 1e6 / duration)))
                except ValueError:
                    pass
        process.wait()
        reader.join(timeout=5)
        if token and token.cancelled:
            raise Cancelled("operation cancelled")
        if process.returncode != 0:
            raise MediaError(f"Shot-change detection failed: {tail[-1] if tail else process.returncode}")
    finally:
        if token:
            token.unregister(process)
        if process.poll() is None:
            _terminate(process)
    if progress:
        progress(1.0)
    return sorted(set(round(t, 6) for t in times if t > 0))


_FILTERS: Dict[str, bool] = {}


def _has_filter(ffmpeg: str, name: str) -> bool:
    key = f"{ffmpeg}:{name}"
    if key not in _FILTERS:
        try:
            listing = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, timeout=30).stdout
            _FILTERS[key] = bool(re.search(rb"\s" + name.encode() + rb"\s", listing))
        except (OSError, subprocess.SubprocessError):
            _FILTERS[key] = False
    return _FILTERS[key]


# ------------------------------------------------------------------- task


def _video_fps(path: str, token: Any) -> Optional[float]:
    try:
        from . import tracks

        info = tracks.video_info(path, token)
        if info and info.get("fps"):
            return float(info["fps"])
    except Exception:  # noqa: BLE001 - tracks may be missing or the probe fail
        pass
    try:
        from ..media import probe

        return probe(path, token).fps
    except Exception:  # noqa: BLE001
        return None


def _by_type(issues: Sequence[Issue]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for issue in issues:
        counts[issue.type] = counts.get(issue.type, 0) + 1
    return counts


def run_task(job: dict, ctx: Any):
    from .tasks import TaskError, TaskResult, load_subtitle, preview_of, write_subtitle

    options = job.get("options") or {}
    inputs = job.get("input") or {}
    ref = inputs.get("subtitle")
    if not ref:
        raise TaskError("Choose a subtitle to check.")
    video = inputs.get("video") or {}
    video_path = video.get("path") if isinstance(video, dict) else video
    if video_path and not os.path.isfile(video_path):
        raise TaskError(f"Video not found: {video_path}")

    ctx.progress(0, "Reading")
    fps = _video_fps(video_path, ctx.token) if video_path else None
    doc = load_subtitle(ref, ctx, fps=fps)
    fps = fps or doc.fps
    wanted = options.get("language")
    language = normalize(wanted) if wanted and wanted != "auto" else normalize(doc.language)
    rules = rules_for(language, options, fps)
    ctx.log(f"Rules: {rules.name}, {_fmt(rules.cpl)} characters per line, {_fmt(rules.cps)} per second, "
            f"{rules.max_lines} lines, at {rules.fps:.3f} fps")

    cuts: Optional[List[float]] = None
    if video_path and options.get("snapToShotChanges", True):
        ctx.progress(0, "Finding shot changes")
        cuts = shot_changes(video_path, ctx.token, lambda f: ctx.progress(int(f * 100), "Finding shot changes"))
        ctx.log(f"Found {len(cuts)} shot changes")
    ctx.check()

    ctx.progress(0, "Checking")
    before = check(doc, rules, cuts)
    result_doc = doc
    after = before
    outputs = []
    repairs: List[Issue] = []
    if options.get("fix", True):
        ctx.progress(50, "Fixing")
        result_doc, repairs = fix(doc, rules, cuts)
        after = check(result_doc, rules, cuts)
        outputs.append(write_subtitle(result_doc, ref, job, ".netflix", ctx))
    ctx.progress(100, "Done")

    report = {
        "preset": rules.preset,
        "language": rules.language,
        "rules": rules.to_dict(),
        "cues": len(doc.cues),
        "shotChanges": len(cuts) if cuts is not None else None,
        "issuesBefore": len(before),
        "issuesAfter": len(after),
        "fixes": len(repairs),
        "byType": _by_type(after if options.get("fix", True) else before),
        "byTypeBefore": _by_type(before),
        "issues": [issue.to_dict() for issue in (after if options.get("fix", True) else before)[:500]],
    }
    if options.get("fix", True):
        summary = (f"{len(before)} issues against the {rules.name} guide; {len(repairs)} repairs left "
                   f"{len(after)} that need a person (wording or reading speed).")
    else:
        summary = f"{len(before)} issues against the {rules.name} guide in {len(doc.cues)} subtitles."
    return TaskResult(outputs=outputs, report=report, summary=summary, preview=preview_of(result_doc))
