"""The one in-memory shape every subtitle engine reads and writes.

A parser turns SRT, ASS, WebVTT, TTML, PGS (after OCR) or a transcript into a
``SubtitleDoc``; every engine -- sync, frame rate, translation, style,
tone-mapping -- takes a doc and returns a new one; a writer turns it back
into a file. Keeping one model means an OCR'd Blu-ray track can be synced,
translated and written as ASS without any engine knowing where it came from.

Times are float seconds on the video's timeline. Rounding to a format's
precision (SRT milliseconds, ASS centiseconds, frames) happens in the writer
only, so a chain of retimes never accumulates rounding error.

Text is plain text with ``\\n`` between lines and a small HTML-like subset for
styling -- ``<i>``, ``<b>``, ``<u>`` and ``<font color="#RRGGBB">`` -- the
subset SRT players already understand. ASS input keeps its original line in
``raw`` so an untouched cue round-trips with every override tag intact.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_TAG_RE = re.compile(r"</?(?:i|b|u|font)(?:\s[^>]*)?>", re.IGNORECASE)


@dataclass
class Cue:
    start: float
    end: float
    text: str
    #: ASS style name; None for formats without styles.
    style: Optional[str] = None
    #: The original ASS dialogue text with override tags. Writers use it only
    #: while ``text`` still equals what the parser derived from it.
    raw: Optional[str] = None
    #: Numpad alignment 1-9 (2 = bottom centre, 8 = top centre), or None for
    #: the format's default.
    align: Optional[int] = None
    #: Explicit position in video pixels (PGS/VobSub origin), if known.
    x: Optional[int] = None
    y: Optional[int] = None
    #: Forced subtitles (signs, foreign dialogue) that show with subs "off".
    forced: bool = False
    #: ASS "Name"/actor field, or a speaker label from diarisation.
    speaker: Optional[str] = None
    #: 0..1 from OCR or speech recognition; None when not machine-made.
    confidence: Optional[float] = None
    #: Engine-specific extras (word timings, OCR image path, source ids).
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def plain(self) -> str:
        """Text without styling tags, for measuring and translating."""
        return _TAG_RE.sub("", self.text)

    @property
    def lines(self) -> List[str]:
        return self.plain.split("\n")

    def copy(self, **changes: Any) -> "Cue":
        clone = copy.deepcopy(self)
        for key, value in changes.items():
            setattr(clone, key, value)
        return clone

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"start": round(self.start, 4), "end": round(self.end, 4), "text": self.text}
        for key in ("style", "align", "x", "y", "speaker", "confidence"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.forced:
            out["forced"] = True
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cue":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text", "")),
            style=data.get("style"),
            align=data.get("align"),
            x=data.get("x"),
            y=data.get("y"),
            forced=bool(data.get("forced", False)),
            speaker=data.get("speaker"),
            confidence=data.get("confidence"),
        )


@dataclass
class SubtitleDoc:
    cues: List[Cue] = field(default_factory=list)
    #: ISO 639-1 where known ("ja", "en"), else ISO 639-2 or None.
    language: Optional[str] = None
    #: Where it came from: srt, ass, ssa, vtt, ttml, microdvd, sbv, pgs,
    #: vobsub, asr. Writers use it for "same format" output.
    source_format: Optional[str] = None
    #: ASS/SSA [Script Info] + [V4+ Styles] (+ [Fonts]/[Graphics]) sections,
    #: verbatim, so a restyled or retimed ASS file keeps its look.
    ass_header: Optional[str] = None
    #: Video size the positions refer to (ASS PlayResX/Y, PGS video size).
    play_res: Optional[tuple] = None
    #: Frame rate for frame-based formats (MicroDVD) and frame snapping.
    fps: Optional[float] = None
    title: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy(self, cues: Optional[List[Cue]] = None) -> "SubtitleDoc":
        clone = copy.deepcopy(self) if cues is None else copy.copy(self)
        if cues is not None:
            clone.cues = cues
            clone.meta = dict(self.meta)
        return clone

    def sorted(self) -> "SubtitleDoc":
        return self.copy(sorted((c.copy() for c in self.cues), key=lambda c: (c.start, c.end)))

    def map_times(self, fn: Callable[[float], float], drop_negative: bool = True) -> "SubtitleDoc":
        """Every retime is this: a function from old time to new time.

        Offsets, frame-rate scaling, two-point syncs and per-scene cut maps
        are all expressed as ``fn``. A cue that ends up entirely before zero
        is dropped (it fell off the start of the video); one that straddles
        zero is clipped.
        """
        out: List[Cue] = []
        for cue in self.cues:
            start, end = fn(cue.start), fn(cue.end)
            if end < start:
                start, end = end, start
            if drop_negative and end <= 0:
                continue
            out.append(cue.copy(start=max(0.0, start), end=end))
        return self.copy(out)

    def shift(self, seconds: float) -> "SubtitleDoc":
        return self.map_times(lambda t: t + seconds)

    def scale(self, ratio: float, anchor: float = 0.0) -> "SubtitleDoc":
        return self.map_times(lambda t: anchor + (t - anchor) * ratio)

    def to_dict(self, limit: Optional[int] = None) -> Dict[str, Any]:
        cues = self.cues if limit is None else self.cues[:limit]
        return {
            "language": self.language,
            "format": self.source_format,
            "count": len(self.cues),
            "cues": [c.to_dict() for c in cues],
        }


@dataclass
class Word:
    """One recognised word with its timing, from speech recognition."""

    start: float
    end: float
    text: str
    probability: Optional[float] = None
    speaker: Optional[str] = None


def strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text)
