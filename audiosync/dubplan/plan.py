"""The plan: its pieces, what it says, and how it is written down.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..framerate import _format_fps, conversion_guide
from ..voicefix import VoicePiece

from .tuning import DELAY_ONLY_TOLERANCE_S, EDIT_MIN_MS, LIP_SYNC_VISIBLE_MS, MAJOR_EDIT_S, MINOR_EDIT_S, REPLACED_MIN_S, SHORT_STRETCH_S


# The note on a fill that replaced such a stretch, when asked to. Matched on
# by the app and the CLI report, so it is one string.
UNMATCHED_NOTE = "dub audible but did not correlate; replaced"
# Why each fill plays the original, as a word the app can count by. Derived
# from the fill's note, so plans written before it have it too.
FILL_REASONS = {
    "dub starts late": "head",
    "before the original's audio starts": "head",
    "past dub end": "tail",
    "dub is cut here": "cut",
    "dub is silent here": "silent",
    UNMATCHED_NOTE: "unmatched",
    "not placed yet": "draft",
}

# What each kind of gap means for someone watching.
GAP_WORDS = {
    "head": "before the dub starts; the original plays",
    "tail": "after the dub ends; the original plays",
    "silent": "the dub is silent here",
    "unmatched": "the dub plays but does not match the video here (replaced or re-edited material) -- check it",
    "cut": "the dub has nothing for this part of the video",
    "draft": "not placed yet",
}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """One piece of the output track."""

    kind: str
    """``dub`` or ``fill``."""
    start_s: float
    end_s: float
    """Where it sits on the video's timeline."""
    source_start_s: float
    """Where it is read from: the dub's decoded timeline for ``dub``, the
    video's own for ``fill``."""
    offset_s: Optional[float] = None
    match: Optional[float] = None
    note: str = ""
    uncertainty_s: float = 0.0
    """How far either edge of this piece might really sit from where it was
    placed, given how strongly the tracks agreed around it."""

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def reason(self) -> str:
        """For a fill, why the original plays there (see FILL_REASONS)."""
        if self.kind != "fill":
            return ""
        return FILL_REASONS.get(self.note, "cut")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "startS": self.start_s,
            "endS": self.end_s,
            "sourceStartS": self.source_start_s,
            "offsetS": self.offset_s,
            "match": self.match,
            "note": self.note,
            "reason": self.reason,
            "uncertaintyS": self.uncertainty_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        return cls(
            kind=data["kind"],
            start_s=float(data["startS"]),
            end_s=float(data["endS"]),
            source_start_s=float(data["sourceStartS"]),
            offset_s=data.get("offsetS"),
            match=data.get("match"),
            note=data.get("note") or "",
            uncertainty_s=float(data.get("uncertaintyS", 0.0) or 0.0),
        )


def delay_only_text(offset_s: float, speed: float = 1.0) -> str:
    """A delay-only verdict in words, in the app's sign convention: a
    positive delay is a dub that plays late and is moved earlier."""
    ms = 1000.0 * offset_s
    if abs(ms) < 0.5:
        verdict = "Nothing is cut and nothing needs moving: the dub is already in sync"
    else:
        size = f"{abs(ms):.1f}" if abs(ms) < 10 else f"{abs(ms):.0f}"
        verdict = (
            f"Nothing is cut: the dub only needs a delay of {'+' if ms > 0 else '-'}{size} ms "
            f"(it plays {size} ms {'late' if ms > 0 else 'early'}, so it is moved "
            f"{'earlier' if ms > 0 else 'later'})"
        )
    if abs(speed - 1.0) > 1e-9:
        verdict += f", played at {speed:.6f}x for the frame rate"
    return verdict


@dataclass
class DubSyncPlan:
    """Everything needed to write the synced track, and why it looks so."""

    video_path: str
    dub_path: str
    video_track: int = 0
    dub_track: int = 0
    speed: float = 1.0
    """Playback-speed factor the dub was decoded at to line up with the
    video. 1.0 means the files run at the same speed."""
    fill_gain_db: float = 0.0
    """Gain applied to the video's audio where it fills a gap, so it sits at
    the dub's level."""
    video_duration_s: float = 0.0
    dub_duration_s: float = 0.0
    """The dub's own length, before any speed change."""
    video_fps: Optional[float] = None
    """The video container's exact standard frame rate, when one was read
    from the file (24000/1001 for 23.976). None for a bare audio file."""
    dub_rate: Optional[float] = None
    """The rate the dub was mastered at, as implied by ``speed`` against
    ``video_fps``. Equal to ``video_fps`` when the rates match."""
    rate_confirmed: Optional[bool] = None
    """Whether the audio itself bore the rate verdict out: at the speed
    settled on, the trial windows correlated sharply. False when neither the
    files' own speed nor any standard conversion did, in which case the
    verdict is a guess and the plan says so; None when the video carried no
    frame rate, or the speed was set by hand."""
    segments: List[Segment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    """Things the reader should check: a fill, a drift, a stretch replaced."""
    notes: List[str] = field(default_factory=list)
    """Things the reader may like to know, that need no checking: the dub
    kept across a span it did not correlate in."""
    error: Optional[str] = None
    doubt_s: float = 0.0
    """Seconds of the fills that are doubt about where a cut lies, not
    material the dub lacks (see ``_trim_uncertain_edges``)."""
    voice_pieces: List[VoicePiece] = field(default_factory=list)
    """Spans of the dub whose voices are laid at their own offset rather than
    their stretch's: where the dub's voices were cut apart from its music
    (see ``voicefix``). Empty unless the voice check ran and moved some."""
    timeline: str = "container"
    """Which clock the times are on. ``container``: each file's own clock,
    time zero its earliest stream -- the clock every seek, picture frame
    and mux uses (see ``media.audio_lead_s``). ``stream``: plans made
    before that, counted from each audio stream's first sample; they are
    moved onto the files' clocks before they are written
    (``dubrender.on_file_clock``)."""

    @property
    def dub_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.kind == "dub"]

    @property
    def fill_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.kind == "fill"]

    @property
    def filled_s(self) -> float:
        return sum(s.length_s for s in self.fill_segments)

    @property
    def delay_only_s(self) -> Optional[float]:
        """The dub's offset, when that is all the plan is: one unbroken
        stretch of dub at one offset, with the original only before the
        dub's first moment and after its last. Such a dub is not cut
        anywhere and needs nothing but a delay (and, where the rates
        differ, a speed); None when anything is cut, inserted, replaced
        or moved.

        Two stretches either side of a fill at the same offset do not
        make a delay: the dub has something else there -- a replaced
        scene, a dropout -- and the original plays for a reason.
        """
        dubs = self.dub_segments
        if not dubs or self.voice_pieces or any(s.offset_s is None for s in dubs):
            return None
        offsets = [float(s.offset_s) for s in dubs]
        if max(offsets) - min(offsets) > DELAY_ONLY_TOLERANCE_S:
            return None
        first, last = dubs[0].start_s, dubs[-1].end_s
        for segment in self.fill_segments:
            if segment.end_s > first + 1e-6 and segment.start_s < last - 1e-6:
                return None
        return float(np.median(offsets))

    def summary(self) -> dict:
        """How much of the dub is used, and why the original plays where it
        does: seconds per fill reason, and of the cuts how much is doubt."""
        by_reason: Dict[str, float] = {}
        for segment in self.fill_segments:
            by_reason[segment.reason] = by_reason.get(segment.reason, 0.0) + segment.length_s
        used = sum(s.length_s for s in self.dub_segments) / max(self.speed, 1e-9)
        return {
            "dubUsedS": round(used, 3),
            "dubDurationS": round(self.dub_duration_s, 3),
            "dubUsedShare": round(used / self.dub_duration_s, 4) if self.dub_duration_s > 0 else None,
            "fillS": {reason: round(seconds, 3) for reason, seconds in sorted(by_reason.items())},
            "doubtS": round(self.doubt_s, 3),
        }

    def edits_and_gaps(self) -> Tuple[List[dict], List[dict]]:
        """Every place the dub departs from the video, as a list a person can
        check one by one, and every span of the video the dub does not cover.

        An edit is a change of offset between two stretches of dub: material
        the dub lacks, material only the dub has, or a scene one replaced with
        the other. Each is reported however small -- a one-frame trim as much
        as a ten-minute scene -- with where it is on both timelines and how
        precisely it was placed. What is left of the fills (the dub starting
        late, ending early, falling silent, or not matching) are the gaps.
        """
        speed = self.speed if self.speed and self.speed > 0 else 1.0
        frame_s = 1.0 / self.video_fps if self.video_fps else None
        ordered = sorted(self.segments, key=lambda s: s.start_s)
        edits: List[dict] = []
        consumed: set = set()
        previous: Optional[Segment] = None
        between: List[Segment] = []
        for segment in ordered:
            if segment.kind == "fill":
                between.append(segment)
                continue
            if segment.offset_s is None:
                continue
            if previous is not None:
                jump_s = segment.offset_s - previous.offset_s
                if abs(jump_s) * 1000.0 >= EDIT_MIN_MS:
                    edits.append(self._edit(len(edits) + 1, previous, segment, between, speed, frame_s))
                    consumed.update(id(fill) for fill in between)
            previous, between = segment, []

        gaps = []
        for fill in (s for s in ordered if s.kind == "fill" and id(s) not in consumed):
            gaps.append({
                "startS": fill.start_s,
                "endS": fill.end_s,
                "lengthS": fill.length_s,
                "reason": fill.reason,
                "uncertaintyS": fill.uncertainty_s,
                "description": (
                    f"{_clock(fill.start_s)} - {_clock(fill.end_s)} ({_amount(fill.length_s)}): "
                    + GAP_WORDS.get(fill.reason, fill.note or fill.reason)
                ),
            })
        return edits, gaps

    def _edit(
        self, index: int, before: Segment, after: Segment, fills: List[Segment],
        speed: float, frame_s: Optional[float],
    ) -> dict:
        jump_s = after.offset_s - before.offset_s
        # The video with no dub between the two stretches, and the dub between
        # them that matches no video. A pure cut leaves only the first, a pure
        # insert only the second; a scene the dub replaced leaves both.
        video_gap = max(0.0, after.start_s - before.end_s)
        dub_gap = video_gap + jump_s
        size_s = abs(jump_s)
        uncertainty = max([before.uncertainty_s, after.uncertainty_s] + [f.uncertainty_s for f in fills])
        if jump_s < 0:
            kind = "replaced" if dub_gap > max(REPLACED_MIN_S, 0.25 * video_gap) else "missing"
            # Doubt about where the cut lies widens the unplayed span beyond
            # what is actually missing; the difference is how far off it may be.
            uncertainty = max(uncertainty, (video_gap - size_s) / 2.0) if kind == "missing" else uncertainty
        else:
            kind = "replaced" if video_gap > max(REPLACED_MIN_S, 0.25 * max(dub_gap, 0.0)) else "extra"
        # Graded on the millisecond it was measured to, so a one-second cut
        # measured at 0.99998 s is not "minor".
        graded = round(size_s, 3)
        severity = "minor" if graded < MINOR_EDIT_S else "moderate" if graded < MAJOR_EDIT_S else "major"
        frames = int(round(size_s / frame_s)) if frame_s else None

        at = _clock(before.end_s)
        framed = f" ({frames} frame{'s' if frames != 1 else ''})" if frames else ""
        if kind == "missing":
            what = f"the dub is missing {_amount(size_s)}{framed} of the video"
            if video_gap > size_s + 0.05:
                what += f", somewhere in {_clock(before.end_s)} - {_clock(after.start_s)}"
        elif kind == "extra":
            what = (
                f"the dub has {_amount(size_s)}{framed} the video does not "
                f"(dub {_clock((before.end_s + before.offset_s) / speed)} - "
                f"{_clock((before.end_s + before.offset_s + size_s) / speed)})"
            )
        else:
            what = (
                f"{_amount(video_gap)} of the video (to {_clock(after.start_s)}) is replaced in the dub by "
                f"{_amount(max(dub_gap, 0.0))} of other material"
            )
        drift = f"from here it plays {_amount(size_s)} {'early' if jump_s < 0 else 'late'}"
        if size_s * 1000.0 < LIP_SYNC_VISIBLE_MS:
            drift += ", under the 45 ms where lip-sync error becomes visible"
        about = f" (placed to within {uncertainty:.2f}s)" if uncertainty >= 0.05 else ""
        # A real edit sits between stretches minutes long. One resting on a
        # few seconds of dub is also what a repeated cue matched in the wrong
        # place looks like -- in end credits the two versions do not share,
        # say -- and a match score cannot tell them apart, since a repeated
        # cue matches itself perfectly. So it is listed, and marked to check.
        shortest = min(before.length_s, after.length_s)
        # A plain bool: the stretches' times are often numpy numbers, and a
        # numpy bool stops the whole plan being written as JSON.
        check = bool(shortest < SHORT_STRETCH_S)
        if check:
            drift += (
                f"; it rests on a {shortest:.1f} s stretch of dub, too short to rule out a repeated "
                "passage -- check it by ear"
            )
        return {
            "index": index,
            "kind": kind,
            "severity": severity,
            "videoS": before.end_s,
            "videoEndS": after.start_s,
            "dubS": (before.end_s + before.offset_s) / speed,
            "dubEndS": (after.start_s + after.offset_s) / speed,
            "jumpMs": jump_s * 1000.0,
            "sizeMs": size_s * 1000.0,
            "missingS": size_s if kind == "missing" else max(0.0, video_gap) if kind == "replaced" else 0.0,
            "extraS": size_s if kind == "extra" else max(0.0, dub_gap) if kind == "replaced" else 0.0,
            "frames": frames,
            "uncertaintyS": uncertainty,
            "offsetBeforeMs": before.offset_s * 1000.0,
            "offsetAfterMs": after.offset_s * 1000.0,
            "shortestStretchS": shortest,
            "check": check,
            "description": f"{at}{about}: {what}; {drift}.",
        }

    def rate_guide(self) -> Optional[dict]:
        """The frame-rate conversion the dub needs, from which rate to which."""
        first = self.dub_segments[0] if self.dub_segments else None
        return conversion_guide(
            self.video_fps, self.speed, self.dub_duration_s, self.rate_confirmed,
            offset_ms=first.offset_s * 1000.0 if first is not None and first.offset_s is not None else None,
        )

    def to_dict(self) -> dict:
        edits, gaps = self.edits_and_gaps()
        return {
            "version": 1,
            "videoPath": self.video_path,
            "dubPath": self.dub_path,
            "videoTrack": self.video_track,
            "dubTrack": self.dub_track,
            "speed": self.speed,
            "fillGainDb": self.fill_gain_db,
            "videoDurationS": self.video_duration_s,
            "dubDurationS": self.dub_duration_s,
            "videoFps": self.video_fps,
            "dubRate": self.dub_rate,
            "rateConfirmed": self.rate_confirmed,
            "segments": [s.to_dict() for s in self.segments],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "error": self.error,
            "filledS": self.filled_s,
            "summary": self.summary(),
            "timeline": self.timeline,
            "voicePieces": [piece.to_dict() for piece in self.voice_pieces],
            "edits": edits,
            "gaps": gaps,
            "rateGuide": self.rate_guide(),
            "delayOnlyMs": None if self.delay_only_s is None else round(1000.0 * self.delay_only_s, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DubSyncPlan":
        return cls(
            video_path=data["videoPath"],
            dub_path=data["dubPath"],
            video_track=int(data.get("videoTrack", 0) or 0),
            dub_track=int(data.get("dubTrack", 0) or 0),
            speed=float(data.get("speed", 1.0) or 1.0),
            fill_gain_db=float(data.get("fillGainDb", 0.0) or 0.0),
            video_duration_s=float(data.get("videoDurationS", 0.0) or 0.0),
            dub_duration_s=float(data.get("dubDurationS", 0.0) or 0.0),
            video_fps=data.get("videoFps"),
            dub_rate=data.get("dubRate"),
            rate_confirmed=data.get("rateConfirmed"),
            segments=[Segment.from_dict(s) for s in data.get("segments", [])],
            warnings=list(data.get("warnings") or []),
            notes=list(data.get("notes") or []),
            error=data.get("error"),
            timeline=data.get("timeline") or "stream",
            doubt_s=float((data.get("summary") or {}).get("doubtS", 0.0) or 0.0),
            voice_pieces=[VoicePiece.from_dict(piece) for piece in data.get("voicePieces") or []],
        )

    def describe(self) -> str:
        """The plan as a table, one line per piece of the output."""
        lines = []
        dubs, fills = self.dub_segments, self.fill_segments
        delay = self.delay_only_s
        if delay is not None:
            lines.append(delay_only_text(delay, self.speed))
        head = (
            f"{len(dubs)} stretch{'es' if len(dubs) != 1 else ''} of dub, "
            f"{len(fills)} fill{'s' if len(fills) != 1 else ''} from the original "
            f"({_clock(self.filled_s)} in all)"
        )
        if self.video_fps is not None:
            # The frame-rate verdict, read off the video's own metadata rather
            # than left to the symptoms: matched, or mastered at another rate.
            # dub_rate is rounded to 6 places, so equality is judged loosely.
            if self.dub_rate is not None and abs(self.dub_rate - self.video_fps) > 1e-6:
                head += (
                    f", video {_format_fps(self.video_fps)} fps, dub mastered at "
                    f"{_format_fps(self.dub_rate)} fps, played at {self.speed:.6f}x"
                )
            else:
                head += f", video {_format_fps(self.video_fps)} fps, dub at the same rate"
            if self.rate_confirmed is False:
                head += " (not confirmed by the audio)"
        elif abs(self.speed - 1.0) > 1e-9:
            head += f", dub played at {self.speed:.6f}x"
        if fills:
            head += f", fills at {self.fill_gain_db:+.1f} dB"
        lines.append(head)
        summary = self.summary()
        if summary["dubUsedShare"] is not None:
            words = {"cut": "the dub lacks the scene", "head": "before the dub starts", "tail": "after it ends",
                     "silent": "the dub is silent", "unmatched": "replaced, did not correlate", "draft": "not placed yet"}
            parts = [f"{_duration(seconds)} {words.get(reason, reason)}" for reason, seconds in summary["fillS"].items() if seconds > 0]
            if self.doubt_s > 0.05:
                parts.append(f"of the cuts {_duration(self.doubt_s)} is doubt about where they lie")
            lines.append(
                f"  dub used: {_clock(summary['dubUsedS'])} of {_clock(summary['dubDurationS'])} "
                f"({100.0 * summary['dubUsedShare']:.1f}%)" + (f"; the original plays where {'; '.join(parts)}" if parts else "")
            )
        for segment in self.segments:
            span = f"{_clock(segment.start_s)} - {_clock(segment.end_s)}"
            if segment.kind == "dub":
                lines.append(
                    f"  dub   {span}  <- dub {_clock(segment.source_start_s)}"
                    f"  {segment.offset_s:+9.3f}s  match {segment.match:.2f}"
                )
            else:
                about = f"+/-{segment.uncertainty_s:.1f}s" if segment.uncertainty_s >= 0.05 else ""
                lines.append(
                    f"  fill  {span}  <- org {_clock(segment.source_start_s)}"
                    f"  {'':>10}  {_duration(segment.length_s):>9}  {segment.note}  {about}"
                )
        replaced = sum(1 for s in self.fill_segments if s.note == UNMATCHED_NOTE)
        if replaced:
            lines.append(
                f"  ! {replaced} fill{'s' if replaced != 1 else ''} replaced dub that did not correlate "
                f"rather than dub that is missing; check {'those spots' if replaced != 1 else 'that spot'}"
            )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        for note in self.notes:
            lines.append(f"  - {note}")
        for piece in self.voice_pieces:
            lines.append(f"  ~ voices: {piece.note}")
        return "\n".join(lines)


def _clock(seconds: float) -> str:
    # Rounded to the millisecond before it is split, or 299.9996 s reads
    # "0:04:60.000".
    ms = int(round(max(0.0, seconds) * 1000.0))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours}:{minutes:02d}:{rest / 1000.0:06.3f}"


def _amount(seconds: float) -> str:
    """An edit's size at the precision it was measured to: 42 ms, 5.000 s,
    10m 00.000s."""
    seconds = round(abs(seconds), 3)
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, rest = divmod(int(round(seconds * 1000.0)), 60_000)
    return f"{minutes}m {rest / 1000.0:06.3f}s"


def _duration(seconds: float) -> str:
    # To the tenth first: 359.96 s is 6m 00.0s, not 5m 60.0s.
    tenths = int(round(seconds * 10.0))
    if tenths >= 600:
        minutes, rest = divmod(tenths, 600)
        return f"{minutes}m {rest / 10.0:04.1f}s"
    return f"{tenths / 10.0:.1f}s"


DRAFT_NOTE = "not placed yet"
