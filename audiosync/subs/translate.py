"""Subtitle translation: the scene-by-scene pipeline behind the Translate tool.

Line-by-line machine translation produces subtitles nobody would ship. A
Japanese sentence often runs over two or three cues with the verb in the
last one, subjects and pronouns are left unsaid, and who is speaking to whom
decides the register. So the chat engines (Claude, OpenAI-compatible,
Ollama) get a whole scene per request, with:

- a system prompt that is the same for every scene of a job (glossary, show
  context, honorifics and formality policy, the target language's streaming
  conventions), so the provider's prompt cache pays for it once;
- the end of the previous scene with its translation, for continuity, and
  the next few source lines as look-ahead;
- each cue as ``{id, text, seconds, max_chars}``, where ``max_chars`` is
  the reading-speed budget (CPS x duration, capped by CPL x lines).

The answer is structured JSON ``{translations: [{id, text}]}``. Every id
must come back exactly once; missing or empty lines are asked for again.
With quality "accurate" a second pass reviews each scene against the source
and returns only corrections. With fitReadingSpeed, lines still over budget
get a condense request, then the style module lays out the line breaks.

DeepL and Google get the same scenes (DeepL also gets the neighbouring
lines as ``context``) but no review or condensing, because those need a
model that can reason about the text.

Every model or service answer is cached on disk under the user cache
folder, keyed by the engine, model, prompt version and the exact request.
A 2-hour film that crashes or is cancelled at scene 20 of 25 resumes from
the cache and pays for five scenes, not twenty-five.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import languages
from .model import Cue, SubtitleDoc
from .tasks import TaskContext, TaskError, TaskOutput, TaskResult
from .translate_engines import (
    SERVICE_NAMES,
    ENDPOINTS,
    ClaudeClient,
    DeepLClient,
    GoogleClient,
    OpenAICompatibleClient,
    Refused,
    Transport,
    Truncated,
    ollama_models,
    scrub,
)

#: Bump when a prompt or schema changes, so cached answers to the old
#: prompt are not reused.
PROMPT_VERSION = "2026-09-26.1"

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
#: Gaps longer than this usually mean a new scene.
SCENE_GAP_S = 3.0
#: Previous-scene lines sent for continuity, and next lines as look-ahead.
CONTEXT_LINES = 10
LOOKAHEAD_LINES = 5
CONDENSE_BATCH = 40

DEFAULTS: Dict[str, Any] = {
    "engine": "claude",
    "source": "auto",
    "target": "en",
    "model": "",
    "baseUrl": None,
    "quality": "accurate",
    "glossary": "",
    "context": "",
    "honorifics": "keep",
    "formality": "default",
    "keepSdh": True,
    "fitReadingSpeed": True,
    "bilingual": False,
    "batchSize": 60,
}

LLM_ENGINES = ("claude", "openai", "ollama")
NO_SPACE_LANGS = ("ja", "zh", "yue")

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                    "issue": {"type": "string"},
                },
                "required": ["id", "text", "issue"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["corrections"],
    "additionalProperties": False,
}

DETECT_SCHEMA = {
    "type": "object",
    "properties": {"language": {"type": "string"}},
    "required": ["language"],
    "additionalProperties": False,
}


# ------------------------------------------------------------ style rules


@dataclass
class Rules:
    cpl: float
    cps: float
    max_lines: int
    dash: str
    #: style.StyleRules when the style module is importable.
    native: Any = None


#: Used only while style.py is missing or fails: Netflix values per script.
_FALLBACK_RULES = {"ja": (13, 4, 2), "zh": (16, 9, 2), "yue": (16, 9, 2), "ko": (16, 12, 2), "th": (35, 15, 2)}


def rules_for(language: str, fps: Optional[float] = None) -> Rules:
    lang = languages.normalize(language) or "en"
    try:
        from . import style

        native = style.rules_for(language, {}, fps)
        return Rules(float(native.cpl), float(native.cps), int(native.max_lines),
                     str(getattr(native, "dash", "-") or "-"), native)
    except Exception:  # noqa: BLE001 -- style is optional here
        cpl, cps, lines = _FALLBACK_RULES.get(lang, (42, 17, 2))
        return Rules(cpl, cps, lines, "- " if lang == "fr" else "-")


def text_width(text: str, language: str) -> float:
    """Reading length the Netflix way (no tags, no line breaks)."""
    try:
        from . import style

        return float(style.text_width(text.replace("\n", ""), language))
    except Exception:  # noqa: BLE001
        plain = _strip_tags(text).replace("\n", "")
        if languages.normalize(language) in NO_SPACE_LANGS:
            return sum(0.5 if ord(ch) < 0x2E80 else 1.0 for ch in plain)
        return float(len(plain))


def break_lines(text: str, rules: Rules, language: str) -> str:
    if rules.native is not None:
        try:
            from . import style

            return style.break_lines(text, rules.native, language)
        except Exception:  # noqa: BLE001
            pass
    return _fallback_break(text, rules, language)


def _fallback_break(text: str, rules: Rules, language: str) -> str:
    """Two lines, bottom-heavy, split at the most balanced point (after
    punctuation where possible); tags are carried across the break."""
    if text_width(text, language) <= rules.cpl or rules.max_lines < 2:
        return text
    no_space = languages.normalize(language) in NO_SPACE_LANGS
    units = re.findall(r"(?:<[^>]+>)*[^<](?:<\/[^>]+>)*", text) if no_space else text.split(" ")
    joiner = "" if no_space else " "
    best, best_score = None, None
    for k in range(1, len(units)):
        top, bottom = joiner.join(units[:k]), joiner.join(units[k:])
        wt, wb = text_width(top, language), text_width(bottom, language)
        score = max(wt, wb) + (0.5 if wt > wb else 0) - (3 if re.search(r"[,.;:!?、。，！？]\W*$", _strip_tags(top)) else 0)
        if best_score is None or score < best_score:
            best, best_score = k, score
    if best is None:
        return text
    top, bottom = joiner.join(units[:best]), joiner.join(units[best:])
    if top.count("<i>") > top.count("</i>"):
        top, bottom = top + "</i>", "<i>" + bottom
    return f"{top}\n{bottom}"


_TAG_RE = re.compile(r"</?(?:i|b|u|font)(?:\s[^>]*)?>", re.I)
_NON_ITALIC_TAG_RE = re.compile(r"</?(?:b|u|font)(?:\s[^>]*)?>", re.I)


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


# ----------------------------------------------------------------- inputs


@dataclass
class Item:
    """One cue as the pipeline sees it; ``id`` is 1-based and stable."""

    id: int
    index: int
    start: float
    end: float
    text: str
    budget: int

    @property
    def seconds(self) -> float:
        return round(max(0.0, self.end - self.start), 2)

    @property
    def plain(self) -> str:
        return _strip_tags(self.text)


def parse_glossary(text: str) -> List[Tuple[str, str]]:
    """``源氏 = Genji`` per line (also ``=>``, ``->``, ``→`` or a tab)."""
    pairs: List[Tuple[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s*(?:=>|->|→|=|\t)\s*", line, maxsplit=1)
        if len(parts) == 2 and parts[0] and parts[1]:
            pairs.append((parts[0], parts[1]))
    return pairs


_SDH_SPANS = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|［[^］]*］|【[^】]*】|〔[^〕]*〕")
_SPEAKER = re.compile(
    r"^(?P<lead>(?:<i>)?\s*(?:[-–‐－]\s*)?)(?:[A-Z][A-Z0-9 .'’&-]{0,24}|[぀-ヿ一-鿿]{1,8})\s*[:：]\s*"
)


def strip_sdh(text: str) -> str:
    """Remove sound descriptions and speaker labels; "" if nothing is left."""
    lines = []
    for line in text.split("\n"):
        line = _SDH_SPANS.sub("", line)
        line = _SPEAKER.sub(lambda m: m.group("lead"), line)
        line = re.sub(r"<i>\s*</i>", "", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if _strip_tags(line).strip(" -–‐－") == "":
            continue
        lines.append(line)
    if len(lines) == 1:
        # A two-speaker cue that lost one speaker no longer needs a dash.
        lines[0] = re.sub(r"^((?:<i>)?)\s*[-–‐－]\s*", r"\1", lines[0])
    return "\n".join(lines)


def detect_script(texts: List[str]) -> Optional[str]:
    """Languages a script alone identifies; None for Latin, Cyrillic..."""
    sample = "".join(texts[:300])
    if not sample:
        return None
    counts = {
        "ja": len(re.findall(r"[぀-ヿ]", sample)),
        "ko": len(re.findall(r"[가-힯]", sample)),
        "han": len(re.findall(r"[一-鿿]", sample)),
        "th": len(re.findall(r"[฀-๿]", sample)),
        "el": len(re.findall(r"[Ͱ-Ͽ]", sample)),
        "he": len(re.findall(r"[֐-׿]", sample)),
    }
    letters = max(1, len(re.findall(r"\w", sample)))
    if counts["ja"] > 0.05 * letters:
        return "ja"
    if counts["ko"] > 0.3 * letters:
        return "ko"
    if counts["han"] > 0.3 * letters:
        return "zh"
    for code in ("th", "el", "he"):
        if counts[code] > 0.3 * letters:
            return code
    return None


def split_scenes(items: List[Item], batch_size: int, gap_s: float = SCENE_GAP_S) -> List[List[Item]]:
    """Requests of at most ``batch_size`` cues, cut at scene gaps.

    Short scenes are packed together (fewer requests, and the neighbouring
    scene is still useful context); a scene longer than a batch is cut at
    its widest pause, preferring the end of a sentence.
    """
    batch_size = max(1, int(batch_size))
    segments: List[List[Item]] = []
    for item in items:
        if segments and item.start - segments[-1][-1].end <= gap_s:
            segments[-1].append(item)
        else:
            segments.append([item])
    batches: List[List[Item]] = []
    current: List[Item] = []
    for segment in segments:
        if len(current) + len(segment) <= batch_size:
            current.extend(segment)
            continue
        if current:
            batches.append(current)
            current = []
        while len(segment) > batch_size:
            cut = _best_cut(segment, batch_size)
            batches.append(segment[:cut])
            segment = segment[cut:]
        current = list(segment)
    if current:
        batches.append(current)
    return batches


def _best_cut(segment: List[Item], batch_size: int) -> int:
    lo = max(1, (batch_size * 2) // 3)
    best, best_score = batch_size, -1.0
    for cut in range(lo, batch_size + 1):
        gap = segment[cut].start - segment[cut - 1].end
        ends_sentence = bool(re.search(r"[.!?。！？…♪」』]\W*$", segment[cut - 1].plain.strip()))
        score = gap + (1.0 if ends_sentence else 0.0)
        if score > best_score:
            best, best_score = cut, score
    return best


# ---------------------------------------------------------------- prompts

_CONVENTIONS = {
    "en": "spell out numbers from one to ten; use an ellipsis only for a pause or a trailing-off sentence, "
    "never to join two cues; double quotation marks",
    "es": "opening ¿ and ¡ marks",
    "fr": "a non-breaking space before ? ! : ; and inside « » guillemets",
    "de": "„…“ quotation marks",
    "it": "« » quotation marks",
    "ja": "full-width punctuation, no spaces between words, 「」 for quotations",
    "zh": "full-width punctuation, no spaces between words",
    "ko": "spaces between words as in standard Korean, no sentence-final period on dialogue",
}

_SOURCE_NOTES = {
    "ja": "Japanese leaves subjects and objects unsaid and puts the verb last. Who is speaking, to whom, their "
    "gender and relationship show through pronouns (俺, 僕, 私, あたし), sentence endings and keigo; use those "
    "cues, and the show context, to choose voice and pronouns.",
    "ko": "Korean often drops subjects; speech levels (banmal vs jondaetmal) and address terms show the "
    "relationship between speakers; reflect it in register.",
    "zh": "Chinese often drops subjects and has no tense marking; infer them from the scene.",
}


def _language_label(code: str) -> str:
    name = languages.name_of(code)
    return f"{name} ({code})" if code and "-" in code else name


def build_brief(opts: Dict[str, Any], source: Optional[str], target: str, rules: Rules) -> str:
    """The part of every system prompt that describes the job itself."""
    src_code = languages.normalize(source) if source else None
    tgt_code = languages.normalize(target) or target
    tgt = _language_label(target)
    parts: List[str] = []
    note = _SOURCE_NOTES.get(src_code or "")
    if note:
        parts.append(note)
    if src_code in ("ja", "ko", "zh", None):
        if opts.get("honorifics") == "localize":
            parts.append(
                "Honorifics: do not keep romanized suffixes (-san, -kun, -chan, -sama, -senpai, oppa, sunbae). "
                f"Convey the relationship the way {tgt} would: first names, titles, Mr./Ms., or register."
            )
        else:
            parts.append(
                "Honorifics: keep them as romanized suffixes and address terms where the source uses them "
                "(Tanaka-san, Yuki-chan, senpai, sensei, -sama, oppa, sunbae), because the audience expects them."
            )
    formality = opts.get("formality")
    if formality == "more":
        parts.append(f"Register: lean formal and polite in {tgt} wherever the scene allows.")
    elif formality == "less":
        parts.append(f"Register: lean casual and colloquial in {tgt} wherever the scene allows.")
    else:
        parts.append(
            "Register: match each speaker to who they are talking to (formal or familiar address, "
            "for example tu/vous or du/Sie, follows the relationship)."
        )
    conventions = _CONVENTIONS.get(tgt_code, "")
    parts.append(
        f"Follow the Netflix Timed Text Style Guide for {tgt}"
        + (f": {conventions}" if conventions else "")
        + f". Two speakers in one cue: one line each, each starting with \"{rules.dash}\"."
    )
    if opts.get("keepSdh"):
        parts.append(
            "Sound descriptions ([door slams]) and speaker labels are part of these subtitles (SDH): translate "
            f"them too, written the way {tgt} SDH subtitles write them."
        )
    glossary = parse_glossary(opts.get("glossary") or "")
    if glossary:
        lines = "\n".join(f"- {src} → {dst}" for src, dst in glossary)
        parts.append("Glossary. Whenever a source term appears, use this rendering verbatim:\n" + lines)
    context = (opts.get("context") or "").strip()
    if context:
        parts.append(
            "About the show (use it for tone, relationships, gender and pronouns; do not add it to lines):\n"
            + context
        )
    return "\n\n".join(parts)


def translate_system(source: Optional[str], target: str, brief: str) -> str:
    src = _language_label(source) if source else "the source language"
    tgt = _language_label(target)
    return f"""You are a senior audiovisual translator subtitling a film or series from {src} into {tgt}. The subtitles must read like the ones on a major streaming service: faithful to meaning, intent and tone, and written in the natural spoken {tgt} a native screenwriter would write, never translationese.

How to translate
- Translate what the character means, not word for word. Keep register, humour, attitude, profanity level and deliberate ambiguity. Do not soften, censor, explain or add anything.
- Subtitles are read at speed. Prefer short, idiomatic phrasing, and drop fillers, stammers and repetition that carry no meaning before you drop content. Each cue has a max_chars budget (characters including spaces); keep within it whenever the meaning allows.
- A sentence often runs over two or three cues. You may move words between the cues of one sentence so each reads naturally in {tgt} word order, but keep each cue close to what is being said while it is on screen. Never leave a cue empty when its source is not empty, and never pack a whole sentence into one cue.
- Keep names, places and terms consistent across the whole film. Romanize names the standard way (Hepburn for Japanese, Revised Romanization for Korean, pinyin for Mandarin) unless the glossary says otherwise.
- If who is speaking, or their gender, cannot be worked out, choose wording that stays neutral rather than guessing.
- Song lyrics: translate them for meaning and keep them in italics.
- Formatting: keep <i>…</i> around the words that match italic source text (off-screen voices, thoughts, lyrics, on-screen text). Use a line break only between two speakers in one cue; all other line breaking is done afterwards. No other markup and no translator notes.

{brief}

Input and output
The user message is JSON. "cues" are the lines to translate, each {{id, text, seconds, max_chars}}. "previous_scene" (already translated) and "next_lines" are context only; never translate or return them. When "already_translated" is present, those cues of this scene are done; translate only the ones in "cues" so they fit with them. Return {{"translations": [{{"id", "text"}}]}} with exactly one entry for every id in "cues", in the same order, and no other ids."""


def review_system(source: Optional[str], target: str, brief: str) -> str:
    src = _language_label(source) if source else "the source language"
    tgt = _language_label(target)
    return f"""You are the reviewing editor for {src} to {tgt} subtitles, checking a translator's draft against the source before it is delivered to a streaming service.

For each cue, compare the source with the draft and correct only real problems:
- mistranslation, including misread idioms, negation, numbers, and who does what to whom;
- meaning that was left out or added;
- wrong speaker, addressee, gender or pronouns for the scene;
- names and terms that differ from the glossary or from how they were rendered before;
- phrasing a native {tgt} viewer would find unnatural or stilted, or a register that does not suit the speaker;
- a line far over its max_chars budget when a shorter wording keeps the meaning.
Lines that are already correct and natural stay as they are; a different but equally good wording is not an error. Keep <i> tags. Words may have been moved between the cues of one sentence on purpose; that is fine when the sentence as a whole is right.

{brief}

The user message is JSON: "cues" are {{id, source, draft, max_chars}}; "previous_scene" is context. Return {{"corrections": [{{"id", "text", "issue"}}]}}, with the full corrected text of each cue you change and a few words naming the problem. Return an empty list when the draft needs no change."""


def condense_system(target: str, brief: str) -> str:
    tgt = _language_label(target)
    return f"""You shorten {tgt} subtitles that would be on screen for too short a time to read. Rewrite each cue to fit within max_chars characters (including spaces) and keep its meaning, tone and every piece of information the viewer needs. Cut fillers, redundancy and what the picture already shows first. Keep names, glossary terms, <i> tags and speaker dashes. If a cue cannot fit without losing meaning, return the shortest faithful version.

{brief}

The user message is JSON: "cues" are {{id, source, text, max_chars}}, where text is the current translation and source is the original line. Return {{"translations": [{{"id", "text"}}]}} with one entry for every id."""


DETECT_SYSTEM = (
    "You identify the language of subtitle text. Answer with its ISO 639-1 code (for example en, ja, es) "
    'as {"language": "<code>"}.'
)


# ------------------------------------------------------------------ cache


def cache_dir() -> str:
    """User cache folder for translation answers (never holds keys)."""
    env = os.environ.get("AUDIOSYNC_CACHE_DIR")
    if env:
        base = env
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches/AudioSyncMaster")
    elif os.name == "nt":
        base = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "AudioSyncMaster", "Cache")
    else:
        base = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "AudioSyncMaster")
    return os.path.join(base, "translate")


class AnswerCache:
    def __init__(self, directory: Optional[str] = None):
        self.dir = directory or cache_dir()

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps([PROMPT_VERSION, *parts], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key[:2], key + ".json")

    def get(self, key: str) -> Any:
        try:
            with open(self._path(key), encoding="utf-8") as fh:
                return json.load(fh)["answer"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: str, answer: Any) -> None:
        path = self._path(key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"answer": answer}, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            pass  # a read-only or full disk costs money on a re-run, nothing more


# --------------------------------------------------------------- pipeline


@dataclass
class Stats:
    scenes: int = 0
    cached: int = 0
    reviewed_changes: int = 0
    condensed: int = 0
    changes: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Pipeline:
    def __init__(self, ctx: TaskContext, client: Any, cache: AnswerCache, stats: Stats):
        self.ctx = ctx
        self.client = client
        self.cache = cache
        self.stats = stats

    def ask(self, kind: str, system: str, payload: dict, schema: dict, name: str, effort: str) -> dict:
        user = json.dumps(payload, ensure_ascii=False, indent=1)
        key = AnswerCache.key(self.client.cache_identity, kind, system, user)
        hit = self.cache.get(key)
        if isinstance(hit, dict):
            self.stats.cached += 1
            return hit
        self.ctx.check()
        answer = self.client.complete(system, user, schema, name, effort)
        if not isinstance(answer, dict):
            raise ValueError("the model's answer was not a JSON object")
        self.cache.put(key, answer)
        return answer

    # -- draft

    def translate_items(
        self, system: str, items: List[Item], prev: List[dict], ahead: List[str], depth: int = 0
    ) -> Dict[int, str]:
        result: Dict[int, str] = {it.id: "" for it in items if not it.plain.strip()}
        pending = [it for it in items if it.plain.strip()]
        for attempt in range(3):
            if not pending:
                break
            payload: Dict[str, Any] = {}
            if prev:
                payload["previous_scene"] = prev
            done = [it for it in items if it.id in result and it.plain.strip()]
            if done:
                payload["already_translated"] = [
                    {"id": it.id, "source": it.text, "translation": result[it.id]} for it in done
                ]
            payload["cues"] = [
                {"id": it.id, "text": it.text, "seconds": it.seconds, "max_chars": it.budget} for it in pending
            ]
            if ahead:
                payload["next_lines"] = ahead
            try:
                answer = self.ask(f"translate#{attempt}", system, payload, TRANSLATE_SCHEMA,
                                  "subtitle_translations", "high")
            except (Truncated, Refused) as exc:
                if len(pending) > 1 and depth < 5:
                    half = len(pending) // 2
                    first, second = pending[:half], pending[half:]
                    got = self.translate_items(system, first, prev, [it.text for it in second[:LOOKAHEAD_LINES]], depth + 1)
                    result.update(got)
                    prev2 = (prev + [{"source": it.text, "translation": got.get(it.id, "")} for it in first])[-CONTEXT_LINES:]
                    result.update(self.translate_items(system, second, prev2, ahead, depth + 1))
                    return result
                reason = "declined by the model" if isinstance(exc, Refused) else "too long for one answer"
                self.stats.warnings.append(f"Line {pending[0].id} was {reason}; it keeps the source text.")
                return result
            except ValueError:
                continue
            got = validate_translations(answer.get("translations"), pending)
            result.update(got)
            pending = [it for it in pending if it.id not in got]
        return result

    # -- review

    def review(self, system: str, items: List[Item], draft: Dict[int, str], prev: List[dict]) -> Dict[int, str]:
        cues = [
            {"id": it.id, "source": it.text, "draft": draft[it.id], "max_chars": it.budget}
            for it in items
            if it.id in draft and it.plain.strip()
        ]
        if not cues:
            return draft
        payload: Dict[str, Any] = {"previous_scene": prev} if prev else {}
        payload["cues"] = cues
        try:
            answer = self.ask("review", system, payload, REVIEW_SCHEMA, "subtitle_review", "high")
        except (Truncated, Refused, ValueError):
            self.stats.warnings.append(f"The review of lines {items[0].id}-{items[-1].id} did not complete; "
                                       "the first draft was kept.")
            return draft
        out = dict(draft)
        by_id = {it.id: it for it in items}
        for entry in answer.get("corrections") or []:
            try:
                cue_id = int(entry.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            text = str(entry.get("text") or "").strip()
            if cue_id not in out or cue_id not in by_id or not text or text == out[cue_id]:
                continue
            out[cue_id] = fix_tags(by_id[cue_id].text, text)
            self.stats.reviewed_changes += 1
            if len(self.stats.changes) < 300:
                self.stats.changes.append({"id": cue_id, "issue": str(entry.get("issue") or "")[:200]})
        return out

    # -- condense

    def condense(self, system: str, items: List[Item], texts: Dict[int, str], target: str) -> Dict[int, str]:
        over = [it for it in items if texts.get(it.id) and text_width(texts[it.id], target) > it.budget]
        out = dict(texts)
        for start in range(0, len(over), CONDENSE_BATCH):
            batch = over[start : start + CONDENSE_BATCH]
            self.ctx.progress(int(100 * start / max(1, len(over))), "Condensing fast lines")
            payload = {"cues": [{"id": it.id, "source": it.text, "text": texts[it.id], "max_chars": it.budget}
                                for it in batch]}
            try:
                answer = self.ask("condense", system, payload, TRANSLATE_SCHEMA, "subtitle_condensed", "medium")
            except (Truncated, Refused, ValueError):
                continue
            got = validate_translations(answer.get("translations"), batch)
            for it in batch:
                new = got.get(it.id)
                if new and text_width(new, target) < text_width(out[it.id], target):
                    out[it.id] = fix_tags(it.text, new)
                    self.stats.condensed += 1
        return out


def validate_translations(entries: Any, pending: List[Item]) -> Dict[int, str]:
    """Answers for the asked ids only: first valid one per id, non-empty
    unless the source is empty."""
    by_id = {it.id: it for it in pending}
    out: Dict[int, str] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            cue_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        item = by_id.get(cue_id)
        if item is None or cue_id in out:
            continue
        text = str(entry.get("text") or "").strip()
        if not text and item.plain.strip():
            continue
        out[cue_id] = fix_tags(item.text, text)
    return out


def fix_tags(source: str, text: str) -> str:
    """Balanced <i> tags; a fully italic source stays fully italic."""
    text = _NON_ITALIC_TAG_RE.sub("", text)
    if text.count("<i>") != text.count("</i>"):
        text = re.sub(r"</?i>", "", text)
    src = source.strip()
    whole_italic = src.startswith("<i>") and src.endswith("</i>") and src.count("<i>") == 1
    if whole_italic and "<i>" not in text and text:
        text = f"<i>{text}</i>"
    return text


def layout(text: str, rules: Rules, language: str) -> str:
    """Final line breaks: dialogue keeps one speaker per line; everything
    else is re-broken to the target's line length."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    starts_with_dash = [bool(re.match(r"^(?:<i>)?\s*[-–‐－]", line)) for line in lines]
    if len(lines) > 1 and all(starts_with_dash):
        return "\n".join(lines)
    joiner = "" if languages.normalize(language) in NO_SPACE_LANGS else " "
    flat = joiner.join(lines).replace("</i>" + joiner + "<i>", joiner)
    return break_lines(flat, rules, language)


# ---------------------------------------------------------------- engines


def _make_client(engine: str, opts: Dict[str, Any], ctx: TaskContext) -> Tuple[Any, str]:
    secrets = ctx.secrets or {}
    service = SERVICE_NAMES.get(engine, engine)
    transport = Transport(service=service, secrets=secrets, check=ctx.check,
                          log=lambda m: ctx.log(scrub(m, secrets)))
    model = str(opts.get("model") or "").strip()
    if engine == "claude":
        key = secrets.get("anthropic")
        if not key:
            raise TaskError("Add an Anthropic API key in Settings to translate with Claude.")
        if not model or not model.startswith("claude"):
            model = DEFAULT_CLAUDE_MODEL
        return ClaudeClient(key, model, transport), model
    if engine == "openai":
        base = str(opts.get("baseUrl") or ENDPOINTS["openai"]).rstrip("/")
        key = secrets.get("openai")
        local = re.match(r"https?://(localhost|127\.0\.0\.1|\[::1\])[:/]", base + "/") is not None
        if not key and not local:
            raise TaskError("Add an OpenAI API key in Settings, or set a local OpenAI-compatible URL.")
        if not model or (model.startswith("claude") and "api.openai.com" in base):
            raise TaskError("Choose a model for the OpenAI-compatible engine.")
        transport.keyed = bool(key)
        return OpenAICompatibleClient("openai", key, model, base, transport), model
    if engine == "ollama":
        base = str(opts.get("baseUrl") or ENDPOINTS["ollama"]).rstrip("/")
        if not re.search(r"/v\d+$", base):
            base += "/v1"
        transport.keyed = False
        if not model or model.startswith("claude"):
            installed = ollama_models(base, timeout=2.0)
            if installed is None:
                raise TaskError(f"Ollama is not answering at {base}. Start it with `ollama serve`.")
            if not installed:
                raise TaskError("Ollama has no models installed. Pull one first, e.g. `ollama pull qwen3:14b`.")
            model = installed[0]
        return OpenAICompatibleClient("ollama", None, model, base, transport), model
    if engine == "deepl":
        key = secrets.get("deepl")
        if not key:
            raise TaskError("Add a DeepL API key in Settings to translate with DeepL.")
        return DeepLClient(key, transport), "deepl"
    if engine == "google":
        key = secrets.get("google")
        if not key:
            raise TaskError("Add a Google Cloud API key in Settings to translate with Google.")
        return GoogleClient(key, transport), "nmt"
    raise TaskError(f"Unknown translation engine: {engine}")


class _MTAdapter:
    """Gives DeepL/Google the cache identity the pipeline keys answers by."""

    def __init__(self, client: Any, source: Optional[str], target: str, formality: str):
        self.client = client
        self.source, self.target, self.formality = source, target, formality
        self.cache_identity = f"{client.engine}|{source}|{target}|{formality}"


def _translate_mt(pipe: Pipeline, adapter: _MTAdapter, scenes: List[List[Item]], items: List[Item]) -> Dict[int, str]:
    client = adapter.client
    result: Dict[int, str] = {}
    position = {it.id: n for n, it in enumerate(items)}
    for n, scene in enumerate(scenes):
        pipe.ctx.progress(int(100 * n / max(1, len(scenes))), f"Translating scene {n + 1}/{len(scenes)}")
        for start in range(0, len(scene), client.batch_limit):
            chunk = [it for it in scene[start : start + client.batch_limit] if it.plain.strip()]
            if not chunk:
                continue
            first, last = position[chunk[0].id], position[chunk[-1].id]
            context = "\n".join(it.plain for it in items[max(0, first - CONTEXT_LINES) : first] + items[last + 1 : last + 1 + LOOKAHEAD_LINES])
            texts = [it.text for it in chunk]
            key = AnswerCache.key(adapter.cache_identity, "mt", texts, context)
            answer = pipe.cache.get(key)
            if isinstance(answer, list) and len(answer) == len(texts):
                pipe.stats.cached += 1
            else:
                pipe.ctx.check()
                answer = client.translate(texts, adapter.source, adapter.target, context, adapter.formality)
                pipe.cache.put(key, answer)
            for it, text in zip(chunk, answer):
                if text.strip():
                    result[it.id] = fix_tags(it.text, text)
    return result


# ------------------------------------------------------------ public API


def translate_doc(doc: SubtitleDoc, options: Dict[str, Any], ctx: TaskContext) -> Tuple[SubtitleDoc, Dict[str, Any]]:
    """Translate ``doc``; returns the translated doc and the job report.

    Each output cue keeps its timing and carries the source text in
    ``meta["sourceText"]`` (used for the bilingual file). The report's
    ``warnings`` list is for the job's warnings, not the report card.
    """
    opts = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None}}
    engine = str(opts["engine"])
    target = str(opts.get("target") or "").strip()
    if not languages.normalize(target):
        raise TaskError(f"Choose a target language (got '{target or 'none'}').")
    stats = Stats()
    secrets = ctx.secrets or {}

    # 1. Cues to translate: SDH dropped first so it is never paid for.
    cues: List[Cue] = []
    sdh_removed = 0
    for cue in doc.cues:
        if not opts.get("keepSdh"):
            stripped = strip_sdh(cue.text)
            if not _strip_tags(stripped).strip():
                sdh_removed += 1
                continue
            cue = cue.copy(text=stripped)
        cues.append(cue)

    rules = rules_for(target, doc.fps)
    items: List[Item] = []
    for n, cue in enumerate(cues):
        duration = max(0.0, cue.end - cue.start)
        budget = int(max(1, math.floor(min(rules.cps * duration, rules.cpl * rules.max_lines))))
        text = _NON_ITALIC_TAG_RE.sub("", cue.text).strip()
        items.append(Item(id=n + 1, index=n, start=cue.start, end=cue.end, text=text, budget=budget))

    client, model = _make_client(engine, opts, ctx)
    cache = AnswerCache()
    pipe = Pipeline(ctx, client, cache, stats)

    # 2. Source language.
    source_opt = str(opts.get("source") or "auto").strip()
    source: Optional[str] = None if source_opt.lower() == "auto" else source_opt
    if source is None:
        source = languages.normalize(doc.language) or detect_script([it.plain for it in items])
    if source is None and engine in LLM_ENGINES and items:
        sample = [it.plain for it in items if it.plain.strip()][:30]
        try:
            answer = pipe.ask("detect", DETECT_SYSTEM, {"lines": sample}, DETECT_SCHEMA, "language", "low")
            source = languages.normalize(str(answer.get("language") or ""))
        except (Truncated, Refused, ValueError):
            source = None
    if source and source.lower() == target.lower():
        raise TaskError(f"The subtitle is already in {languages.name_of(target)}.")

    scenes = split_scenes(items, int(opts.get("batchSize") or DEFAULTS["batchSize"]))
    stats.scenes = len(scenes)
    translations: Dict[int, str] = {}

    # 3. Translate (and review) scene by scene.
    if engine in LLM_ENGINES:
        brief = build_brief(opts, source, target, rules)
        t_system = translate_system(source, target, brief)
        r_system = review_system(source, target, brief)
        accurate = opts.get("quality") == "accurate"
        prev: List[dict] = []
        position = {it.id: n for n, it in enumerate(items)}
        for n, scene in enumerate(scenes):
            ctx.progress(int(100 * n / max(1, len(scenes))), f"Translating scene {n + 1}/{len(scenes)}")
            last = position[scene[-1].id]
            ahead = [it.text for it in items[last + 1 : last + 1 + LOOKAHEAD_LINES]]
            draft = pipe.translate_items(t_system, scene, prev, ahead)
            if accurate:
                ctx.progress(int(100 * (n + 0.5) / max(1, len(scenes))), f"Reviewing scene {n + 1}/{len(scenes)}")
                draft = pipe.review(r_system, scene, draft, prev)
            translations.update(draft)
            prev = [{"source": it.text, "translation": draft.get(it.id, "")}
                    for it in scene if it.plain.strip()][-CONTEXT_LINES:]
        if opts.get("fitReadingSpeed"):
            translations = pipe.condense(condense_system(target, brief), items, translations, target)
    else:
        if parse_glossary(opts.get("glossary") or ""):
            stats.warnings.append(f"{SERVICE_NAMES[engine]} does not use the glossary; names may vary. "
                                  "Use Claude or another chat model to enforce it.")
        adapter = _MTAdapter(client, source, target, str(opts.get("formality") or "default"))
        pipe.client = adapter
        translations = _translate_mt(pipe, adapter, scenes, items)
        detected = getattr(client, "detected", None)
        source = source or detected

    # 4. Assemble: final line layout; untranslated lines keep their source.
    ctx.progress(100, "Laying out lines")
    out_cues: List[Cue] = []
    untranslated: List[int] = []
    for it, cue in zip(items, cues):
        text = translations.get(it.id)
        if text is None:
            if it.plain.strip():
                untranslated.append(it.id)
            text = cue.text
        else:
            text = layout(text, rules, target) if text else ""
        meta = dict(cue.meta)
        meta["sourceText"] = cue.text
        out_cues.append(cue.copy(text=text, meta=meta))
    if untranslated:
        shown = ", ".join(f"#{i}" for i in untranslated[:12]) + (" ..." if len(untranslated) > 12 else "")
        stats.warnings.append(f"{len(untranslated)} line(s) could not be translated and keep the source text: {shown}")

    out = doc.copy(out_cues)
    out.language = target
    out.meta["translatedFrom"] = source

    usage = getattr(client, "usage", None)
    report: Dict[str, Any] = {
        "engine": engine,
        "model": model,
        "source": source or "auto",
        "target": target,
        "lines": len(items),
        "scenes": stats.scenes,
        "requests": usage.requests if usage else 0,
        "inputTokens": usage.input_tokens if usage else 0,
        "outputTokens": usage.output_tokens if usage else 0,
        "characters": usage.characters if usage else 0,
        "reviewedChanges": stats.reviewed_changes,
        "condensed": stats.condensed,
        "cached": stats.cached,
        "untranslated": len(untranslated),
        "sdhRemoved": sdh_removed,
        "changes": stats.changes,
        "warnings": [scrub(w, secrets) for w in stats.warnings],
    }
    return out, report


def bilingual_doc(doc: SubtitleDoc) -> SubtitleDoc:
    """Source line(s) above the translation, from ``meta["sourceText"]``."""
    cues = []
    for cue in doc.cues:
        source = str(cue.meta.get("sourceText") or "").strip()
        text = f"{source}\n{cue.text}" if source and source != cue.text else cue.text
        cues.append(cue.copy(text=text, raw=None))
    return doc.copy(cues)


def engine_statuses(secrets: Dict[str, str]) -> List[Dict[str, Any]]:
    secrets = secrets or {}

    def keyed(engine_id: str, label: str, secret: str, what: str) -> Dict[str, Any]:
        ok = bool(secrets.get(secret))
        return {"id": engine_id, "label": label, "available": ok,
                "reason": None if ok else f"Add {what} in Settings", "pack": None}

    ollama_up = ollama_models(timeout=0.3) is not None
    return [
        keyed("claude", "Claude", "anthropic", "an Anthropic API key"),
        keyed("openai", "OpenAI-compatible", "openai", "an OpenAI API key"),
        keyed("deepl", "DeepL", "deepl", "a DeepL API key"),
        keyed("google", "Google Translate", "google", "a Google Cloud API key"),
        {"id": "ollama", "label": "Ollama (local)", "available": ollama_up,
         "reason": None if ollama_up else "Start Ollama (ollama serve) on this computer", "pack": None},
    ]


def _ref_for_naming(ref: dict, source: Optional[str]) -> dict:
    """``movie.ja.srt`` translated to English is ``movie.en.srt``, not
    ``movie.ja.en.srt``: drop a trailing tag naming the source language."""
    path = ref.get("path") or ""
    if ref.get("track") is not None or not source:
        return ref
    stem, ext = os.path.splitext(path)
    head, dot, tag = stem.rpartition(".")
    if dot and head and languages.normalize(tag) == languages.normalize(source) and len(tag) <= 7:
        return {**ref, "path": head + ext}
    return ref


def run_task(job: dict, ctx: TaskContext) -> TaskResult:
    from . import tasks

    ref = (job.get("input") or {}).get("subtitle")
    if not ref:
        raise TaskError("Choose a subtitle to translate.")
    options = job.get("options") or {}
    doc = tasks.load_subtitle(ref, ctx)
    ctx.check()
    out, report = translate_doc(doc, options, ctx)
    warnings = report.pop("warnings", [])
    target = report["target"]
    naming = _ref_for_naming(ref, report.get("source"))
    label = f"{languages.name_of(target)} ({SERVICE_NAMES.get(report['engine'], report['engine'])})"
    outputs: List[TaskOutput] = []
    main = tasks.write_subtitle(out, naming, job, "", ctx, language=target)
    main.label = label
    outputs.append(main)
    if options.get("bilingual"):
        extra = tasks.write_subtitle(bilingual_doc(out), naming, job, ".bilingual", ctx, language=target)
        extra.label = f"{languages.name_of(report.get('source'))} + {languages.name_of(target)}"
        outputs.append(extra)
    return TaskResult(outputs=outputs, report=report, summary=_summary(report), warnings=warnings,
                      preview=tasks.preview_of(out))


def _summary(report: Dict[str, Any]) -> str:
    service = SERVICE_NAMES.get(report["engine"], report["engine"])
    who = service if report["engine"] in ("deepl", "google") else f"{service} ({report['model']})"
    source = languages.name_of(report["source"]) if report["source"] != "auto" else "the source language"
    text = (f"Translated {report['lines']:,} lines from {source} to {languages.name_of(report['target'])} "
            f"with {who} in {report['scenes']} scene{'s' if report['scenes'] != 1 else ''}")
    extras = []
    if report["reviewedChanges"]:
        extras.append(f"the review corrected {report['reviewedChanges']} line{'s' if report['reviewedChanges'] != 1 else ''}")
    if report["condensed"]:
        extras.append(f"{report['condensed']} line{'s were' if report['condensed'] != 1 else ' was'} condensed to fit reading speed")
    if report["cached"]:
        extras.append(f"{report['cached']} answer{'s' if report['cached'] != 1 else ''} came from the cache")
    if report["untranslated"]:
        extras.append(f"{report['untranslated']} kept the source text")
    return text + ("; " + ", ".join(extras) if extras else "") + "."
