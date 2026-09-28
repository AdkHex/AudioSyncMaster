"""Retiming: frame-rate conversion, two-point syncs and per-scene offsets.

Every retime is a map from old time to new time (``SubtitleDoc.map_times``);
this module builds the maps.

Frame rates are handled as exact fractions. "23.976" is really 24000/1001,
and the difference is not academic: 23.976 vs 24000/1001 drifts 3.6 ms per
hour, and a rate typed as 29.97 against a real 30000/1001 video is 3.6 ms per
hour off too. So every approximation of a 1001 rate snaps to the exact one
before any ratio is taken.

Direction: subtitles timed for a 25 fps PAL release (which runs 4% fast) and
played on the 23.976 fps film need every time multiplied by 25/23.976 --
frame N sits at N/25 s on one and N/23.976 s on the other. That is
``new = old * from / to``.
"""

from __future__ import annotations

import os
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, Union

from .model import Cue, SubtitleDoc
from .tasks import TaskError, TaskResult, load_subtitle, preview_of, write_subtitle

Number = Union[int, float, Fraction, str]

#: Nominal integer rates whose NTSC variant runs at n * 1000/1001.
_NTSC_BASES = (24, 30, 48, 60, 120)

# Close enough to an NTSC rate to be that rate typed short (23.976, 23.98,
# 29.97, 59.94, 119.88), yet far from the integer rate beside it (0.024 away
# at 24).
_SNAP_TOLERANCE = 0.01


def snap_fps(value: Number) -> Fraction:
    """A frame rate as an exact fraction; 1001-rate approximations snapped."""
    # Fraction(str(x)) keeps 23.976 as 23976/1000, not float noise.
    rate = value if isinstance(value, Fraction) else Fraction(str(value).strip())
    if rate <= 0:
        raise ValueError(f"Frame rate must be positive: {value}")
    for base in _NTSC_BASES:
        exact = Fraction(base * 1000, 1001)
        if rate != exact and abs(float(rate) - float(exact)) < _SNAP_TOLERANCE:
            return exact
    return rate


def parse_fps(value: Number) -> Fraction:
    """``"24000/1001"``, ``"23.976"``, ``25``, ``"29.97 fps"`` -> exact Fraction."""
    if isinstance(value, Fraction):
        return snap_fps(value)
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"Frame rate must be positive: {value}")
        return snap_fps(Fraction(str(value)))
    text = str(value).strip().lower().replace("fps", "").strip().replace(",", ".")
    if not text:
        raise ValueError("Empty frame rate")
    if text in ("pal",):
        return Fraction(25)
    if text in ("ntsc",):
        return Fraction(30000, 1001)
    if text in ("film",):
        return Fraction(24000, 1001)
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            rate = Fraction(num.strip()) / Fraction(den.strip())
        else:
            rate = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Not a frame rate: {value}") from exc
    if rate <= 0 or rate > 1000:
        raise ValueError(f"Not a frame rate: {value}")
    return snap_fps(rate)


def fps_ratio(from_fps: Number, to_fps: Number) -> float:
    """``from / to``: what every time is multiplied by. Exact for 1001 rates."""
    return float(parse_fps(from_fps) / parse_fps(to_fps))


def format_fps(value: Number) -> str:
    """24000/1001 -> "23.976", 25 -> "25", 30000/1001 -> "29.97"."""
    return f"{float(parse_fps(value)):.3f}".rstrip("0").rstrip(".")


def convert_fps(doc: SubtitleDoc, from_fps: Number, to_fps: Number, mode: str = "time") -> SubtitleDoc:
    """Retime a doc made for a ``from_fps`` video onto a ``to_fps`` one.

    ``mode="time"``: a speed change. The subtitle is correct for a release
    that plays at ``from_fps``; the target plays the same frames at
    ``to_fps``, so every time scales by from/to (PAL speed-up/slow-down).

    ``mode="frames"``: the frame numbers are right and only the rate used to
    turn them into times was wrong or has changed. For a MicroDVD source the
    frames are the ones in the file -- ``round(t * doc.fps)`` -- and are kept
    exactly, whatever ``from_fps`` says; they are re-read at ``to_fps``. For a
    time-based doc (SRT, ASS...) there are no frame numbers of record, so the
    frames are ``t * from_fps`` and the result is the same scaling as "time"
    mode without any rounding.
    """
    to_rate = parse_fps(to_fps)
    if mode not in ("time", "frames"):
        raise ValueError(f"Unknown frame-rate mode: {mode}")
    if mode == "frames" and doc.source_format == "microdvd" and doc.fps:
        file_rate = float(snap_fps(doc.fps))
        to_float = float(to_rate)
        out = doc.map_times(lambda t: round(t * file_rate) / to_float)
    else:
        ratio = float(parse_fps(from_fps) / to_rate)
        out = doc.map_times(lambda t: t * ratio)
    out.fps = float(to_rate)
    return out


def two_point(doc: SubtitleDoc, a: Sequence[float], b: Sequence[float]) -> SubtitleDoc:
    """Linear retime through two known points, each ``(subtitle time,
    correct time)`` in seconds -- typically the first and last line.

    Two points fix both the offset and the rate, so this corrects an
    unknown frame-rate mismatch as well as a shift.
    """
    (a_src, a_dst), (b_src, b_dst) = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    if abs(b_src - a_src) < 1e-6:
        raise ValueError("The two sync points must be at different subtitle times.")
    scale = (b_dst - a_dst) / (b_src - a_src)
    if scale <= 0:
        raise ValueError("The sync points are in the wrong order: the second must come after the first.")
    return doc.map_times(lambda t: a_dst + (t - a_src) * scale)


def two_point_scale(a: Sequence[float], b: Sequence[float]) -> float:
    return (float(b[1]) - float(a[1])) / (float(b[0]) - float(a[0]))


def piecewise(
    doc: SubtitleDoc,
    pieces: List[Tuple[float, float, Optional[float]]],
) -> SubtitleDoc:
    """Different offsets for different stretches of the subtitle.

    ``pieces`` are ``(src_start, src_end, offset_s)`` on the subtitle's own
    timeline; an offset of None drops the cues in that stretch (a scene the
    target edit cut). A cue takes the offset of the piece holding its start,
    so a line straddling an edit point moves as one, never stretched across
    the cut. Before the first piece the first offset applies, after the last
    the last one, and in a gap the piece before it.
    """
    if not pieces:
        return doc.copy()
    ordered = sorted(pieces, key=lambda p: p[0])

    def offset_for(t: float) -> Optional[float]:
        chosen = ordered[0]
        for piece in ordered:
            if piece[0] <= t:
                chosen = piece
            else:
                break
        return chosen[2]

    out: List[Cue] = []
    for cue in doc.cues:
        offset = offset_for(cue.start)
        if offset is None:
            continue
        start, end = cue.start + offset, cue.end + offset
        if end <= 0:
            continue
        out.append(cue.copy(start=max(0.0, start), end=end))
    return doc.copy(out)


# ---------------------------------------------------------------------- task


def run_task(job: dict, ctx) -> TaskResult:
    """Task "fps": convert a subtitle's timing between frame rates, or by two
    known points, then apply an optional offset."""
    inputs = job.get("input") or {}
    ref = inputs.get("subtitle")
    if not ref:
        raise TaskError("Choose a subtitle to convert.")
    options = job.get("options") or {}
    offset_ms = float(options.get("offsetMs") or 0.0)
    mode = options.get("mode") or "time"
    twopoint = options.get("twoPoint")
    raw_from = options.get("from", "auto")
    raw_to = options.get("to", "auto")

    from_rate: Optional[Fraction] = None
    if raw_from not in (None, "auto"):
        try:
            from_rate = parse_fps(raw_from)
        except ValueError as exc:
            raise TaskError(str(exc)) from exc

    ctx.progress(5, "Reading")
    # A frame-based source is read at the "from" rate when one is given.
    doc = load_subtitle(ref, ctx, fps=float(from_rate) if from_rate else None)
    ctx.check()

    report: dict = {"offsetMs": offset_ms}
    if twopoint:
        try:
            a, b = twopoint["a"], twopoint["b"]
            scale = two_point_scale(a, b)
            result = two_point(doc, a, b)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise TaskError(f"Two-point sync: {exc}") from exc
        report.update({"fromFps": None, "toFps": None, "ratio": scale,
                       "twoPoint": {"a": list(a), "b": list(b)}})
        described = (f"Retimed {len(result.cues)} cues through two points "
                     f"({_fmt_clock(a[0])} -> {_fmt_clock(a[1])}, {_fmt_clock(b[0])} -> {_fmt_clock(b[1])}; "
                     f"rate x{scale:.6f})")
        suffix = ".retimed"
    else:
        if from_rate is None:
            if not doc.fps:
                raise TaskError("The subtitle does not say which frame rate it was timed for; choose the "
                                "\"from\" frame rate.")
            from_rate = parse_fps(doc.fps)
        if raw_to in (None, "auto"):
            video = inputs.get("video") or {}
            if not video.get("path"):
                raise TaskError("Choose the video to read the target frame rate from, or pick a frame rate.")
            from .tracks import video_info

            info = video_info(video["path"], ctx.token)
            if not info or not info.get("fps"):
                raise TaskError(f"Could not read the frame rate of {os.path.basename(video['path'])}.")
            to_rate = parse_fps(info["fps"])
        else:
            try:
                to_rate = parse_fps(raw_to)
            except ValueError as exc:
                raise TaskError(str(exc)) from exc
        ratio = float(from_rate / to_rate)
        ctx.progress(40, "Converting")
        result = convert_fps(doc, from_rate, to_rate, mode=mode)
        report.update({"fromFps": float(from_rate), "toFps": float(to_rate), "ratio": ratio, "mode": mode})
        described = (f"Converted {len(result.cues)} cues from {format_fps(from_rate)} fps to "
                     f"{format_fps(to_rate)} fps (x{ratio:.6f})")
        suffix = f".{format_fps(to_rate)}fps"

    warnings: List[str] = []
    if abs(offset_ms) > 1e-9:
        result = result.shift(offset_ms / 1000.0)
        described += f", then shifted {offset_ms:+.0f} ms"
    if len(result.cues) < len(doc.cues):
        warnings.append(f"{len(doc.cues) - len(result.cues)} cues fell before the start of the video and were dropped.")
    if report.get("ratio") == 1.0 and abs(offset_ms) < 1e-9:
        warnings.append("The frame rates are the same and there is no offset, so the timing is unchanged.")

    ctx.check()
    ctx.progress(80, "Writing")
    output = write_subtitle(result, ref, job, suffix, ctx)
    ctx.progress(100, "Done")
    return TaskResult(outputs=[output], report=report, summary=described + ".", warnings=warnings,
                      preview=preview_of(result))


def _fmt_clock(seconds: float) -> str:
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{int(seconds // 3600)}:{int(seconds // 60) % 60:02d}:{seconds % 60:06.3f}"
