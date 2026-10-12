"""Turning stretches into the plan's pieces.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .aligner import _Aligner
from .edges import _place_step, _step_words
from .foreign import _gap_is_foreign
from .plan import Segment, UNMATCHED_NOTE, _clock
from .scores import Block
from .tuning import BRIDGE_EXPOSURE_S, BRIDGE_SOUND_EXPOSURE_S, BRIDGE_STEP_S, DUB_SILENT_FLOOR, DUB_SILENT_RATIO, EDGE_KEEP_MAX_S, EDGE_TRIM_MIN_S, ENVELOPE_RATE, FILL_AUDIBLE_RATIO, FILL_GAIN_PIECE_S, GUESS_WARN_S, MAX_FILL_GAIN_DB, MIN_BLOCK_S, MIN_FILL_S, MIN_SILENT_S, NEAR_OFFSET_S, REWIND_GUARD_MAX_S, SAME_OFFSET_S

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .envelope import TrackEnvelope


def _fill_gain_db(primary: TrackEnvelope, secondary: TrackEnvelope, blocks: List[Block]) -> float:
    """How much louder the dub is than the original, where both exist.

    One number for the whole file: the two mixes differ by a mastering
    level, not scene by scene, and a per-fill gain would pump.
    """
    rate = ENVELOPE_RATE
    piece = int(FILL_GAIN_PIECE_S * rate)
    ratios, weights = [], []
    for block in blocks:
        lo = int(round(block.start_s * rate))
        hi = int(round(block.end_s * rate))
        shift = int(round(block.offset_s * rate))
        # Measured piece by piece, so the answer depends on the material
        # and not on where the stretches happen to be cut: as one ratio
        # per stretch, the same pair read -3.8 dB with a stretch split in
        # three and -7.1 dB with it whole.
        for start in range(lo, hi, piece):
            p = primary.energy[start : min(hi, start + piece)]
            s = secondary.energy[max(0, start + shift) : max(0, min(hi, start + piece) + shift)]
            n = min(len(p), len(s))
            if n < rate:
                continue
            p_rms = float(np.sqrt(np.mean(p[:n].astype(np.float64) ** 2)))
            s_rms = float(np.sqrt(np.mean(s[:n].astype(np.float64) ** 2)))
            if p_rms < 1e-5 or s_rms < 1e-5:
                continue
            ratios.append(20.0 * math.log10(s_rms / p_rms))
            weights.append(n)
    if not ratios:
        return 0.0
    order = np.argsort(ratios)
    cumulative = np.cumsum(np.array(weights, dtype=np.float64)[order])
    median = float(np.array(ratios)[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])
    return float(max(-MAX_FILL_GAIN_DB, min(MAX_FILL_GAIN_DB, median)))


def _is_quiet(track: TrackEnvelope, lo_s: float, hi_s: float, reference_rms: float, ratio: float) -> bool:
    """Whether the track's level across ``lo_s``-``hi_s`` of its own time is
    under ``ratio`` of ``reference_rms``. An empty span is quiet."""
    rate = ENVELOPE_RATE
    lo = max(0, int(round(lo_s * rate)))
    hi = min(len(track.energy), int(round(hi_s * rate)))
    if hi <= lo:
        return True
    rms = float(np.sqrt(np.mean(track.energy[lo:hi].astype(np.float64) ** 2)))
    return rms < ratio * reference_rms


def _dub_is_silent(secondary: TrackEnvelope, lo_s: float, hi_s: float, reference_rms: float) -> bool:
    return _is_quiet(secondary, lo_s, hi_s, min(reference_rms, DUB_SILENT_FLOOR / DUB_SILENT_RATIO), DUB_SILENT_RATIO)


def _level_over(track: TrackEnvelope, spans: Sequence[Tuple[float, float]]) -> float:
    """RMS of the track's energy across the spans, in its own time."""
    rate = ENVELOPE_RATE
    parts = []
    for lo_s, hi_s in spans:
        lo = max(0, int(round(lo_s * rate)))
        hi = min(len(track.energy), int(round(hi_s * rate)))
        if hi > lo:
            parts.append(track.energy[lo:hi])
    if not parts:
        return 0.0
    joined = np.concatenate(parts).astype(np.float64)
    return float(np.sqrt(np.mean(joined * joined)))


def _dub_level(secondary: TrackEnvelope, blocks: List[Block]) -> float:
    """The dub's level across the stretches, in the dub's own time."""
    return _level_over(secondary, [(b.start_s + b.offset_s, b.end_s + b.offset_s) for b in blocks])


def _original_level(primary: TrackEnvelope, blocks: List[Block]) -> float:
    """The original's level across the stretches the dub covers: what a
    scene sounds like in it, to judge whether a span has one."""
    return _level_over(primary, [(b.start_s, b.end_s) for b in blocks])


def _trim_uncertain_edges(blocks: List[Block], trims: Optional[List[Tuple[float, float]]] = None) -> List[Block]:
    """Pull every poorly placed edge inward by its uncertainty.

    The edge itself is the best estimate of the cut; the uncertainty is how
    far either side the cut may really be. Stopping the dub that far short
    of the estimate keeps it on its own side of the cut whichever way the
    estimate erred. A stretch too uncertain at both ends to keep anything
    is dropped, which is the same statement made honestly: nothing about
    it could be placed.
    """
    kept: List[Block] = []
    for index, block in enumerate(blocks):
        before = blocks[index - 1] if index > 0 else None
        after = blocks[index + 1] if index + 1 < len(blocks) else None
        start_u, end_u = block.start_uncertainty_s, block.end_uncertainty_s
        # An edge facing a stretch at nearly the same offset is not a cut:
        # the scene runs on across it, a frame or two out at most, and
        # there is no wrong side to keep off. Trimming it opened a gap
        # that was filled with the original, twenty seconds of the other
        # language over a scene the dub had, for a step of 34 ms.
        if start_u >= EDGE_TRIM_MIN_S and not (before and abs(before.offset_s - block.offset_s) <= NEAR_OFFSET_S):
            if trims is not None:
                trims.append((block.start_s, block.start_s + start_u))
            block.start_s += start_u
        if end_u >= EDGE_TRIM_MIN_S and not (after and abs(after.offset_s - block.offset_s) <= NEAR_OFFSET_S):
            if trims is not None:
                trims.append((block.end_s - end_u, block.end_s))
            block.end_s -= end_u
        if block.length_s >= MIN_BLOCK_S:
            kept.append(block)
    return kept


def _split_silences(
    blocks: List[Block], primary: TrackEnvelope, secondary: TrackEnvelope, warnings: List[str]
) -> List[Block]:
    """Cut every stretch where the dub falls silent while the original does not.

    A dub with a scene missing does not always have it cut out; sometimes
    it has silence laid in its place, so the timeline runs on and the path
    coasts straight through. The dub's own energy says where: a run of
    near-nothing a second or longer, inside a stretch, is a gap -- when the
    original has something there. Where the original is quiet too it is a
    pause in both, and cutting it would splice twice, fill silence with
    silence and report a cut the dub does not have.
    """
    rate = ENVELOPE_RATE
    level = _dub_level(secondary, blocks)
    original_level = _original_level(primary, blocks)
    if level <= 0.0:
        return blocks
    quiet = secondary.energy < min(DUB_SILENT_RATIO * level, DUB_SILENT_FLOOR)
    out: List[Block] = []
    for block in blocks:
        lo = max(0, int(round((block.start_s + block.offset_s) * rate)))
        hi = min(len(quiet), int(round((block.end_s + block.offset_s) * rate)))
        if hi - lo < int(MIN_SILENT_S * rate):
            out.append(block)
            continue
        flags = quiet[lo:hi]
        # Runs of consecutive quiet frames.
        edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
        cursor = block.start_s
        pieces: List[Block] = []
        for run_lo, run_hi in zip(edges[::2], edges[1::2]):
            if (run_hi - run_lo) / rate < MIN_SILENT_S:
                continue
            silent_from = block.start_s + run_lo / rate
            silent_to = block.start_s + run_hi / rate
            if _is_quiet(primary, silent_from, silent_to, original_level, FILL_AUDIBLE_RATIO):
                continue
            if silent_from - cursor > 0:
                pieces.append(Block(cursor, silent_from, block.offset_s, block.match, block.support, block.windows,
                                    start_lo=block.start_lo if cursor == block.start_s else cursor,
                                    start_hi=block.start_hi if cursor == block.start_s else cursor,
                                    end_lo=silent_from, end_hi=silent_from))
            warnings.append(
                f"the dub is silent across {_clock(silent_from)} - {_clock(silent_to)}; filled from the original"
            )
            cursor = silent_to
        if cursor == block.start_s:
            out.append(block)
            continue
        if block.end_s - cursor > 0:
            pieces.append(Block(cursor, block.end_s, block.offset_s, block.match, block.support, block.windows,
                                start_lo=cursor, start_hi=cursor, end_lo=block.end_lo, end_hi=block.end_hi))
        out.extend(pieces)
    return out


def _join(last: Block, block: Block, weaker: bool = False) -> None:
    """Extend ``last`` over ``block``, which sits at the same offset.

    The offset kept is the longer stretch's: it was measured from more
    material. Taking the first stretch's regardless put a twelve-second
    head's offset over the hour that followed it. ``weaker`` records the
    join as only as good as its weaker side, for a join across material
    that did not correlate.
    """
    if block.length_s > last.length_s:
        last.offset_s = block.offset_s
    last.end_s = max(last.end_s, block.end_s)
    last.end_lo, last.end_hi = block.end_lo, block.end_hi
    last.match = min(last.match, block.match) if weaker else max(last.match, block.match)


def _assemble(
    blocks: List[Block],
    primary_s: float,
    secondary_s: float,
    primary: TrackEnvelope,
    secondary: TrackEnvelope,
    keep_unmatched_dub: bool,
    warnings: List[str],
    notes: List[str],
    aligner: Optional[_Aligner] = None,
    doubt: Optional[List[Tuple[float, float]]] = None,
) -> List[Segment]:
    """Turn stretches into a contiguous list of pieces covering the video.

    ``doubt`` is told the spans of the fills that are there only because a
    cut could not be placed exactly (see ``_trim_uncertain_edges``)."""
    trims: List[Tuple[float, float]] = []
    blocks = _trim_uncertain_edges(sorted((b for b in blocks if b.length_s > 0), key=lambda b: b.start_s), trims)
    blocks = _split_silences(blocks, primary, secondary, warnings)
    if not blocks:
        return []

    # Merge what is really one stretch.
    merged: List[Block] = []
    for block in blocks:
        if merged:
            last = merged[-1]
            gap = block.start_s - last.end_s
            if abs(block.offset_s - last.offset_s) <= SAME_OFFSET_S and gap <= MIN_FILL_S and not (
                last.follows_drift or block.follows_drift
            ):
                _join(last, block)
                continue
            if gap < 0:
                block.start_s = last.end_s
                if block.length_s <= 0:
                    continue
            # A switch that goes back in the dub by more than a few frames,
            # onto what it has just played, would play that part twice: the
            # video it lacks there is the original's. Stretches reach the
            # assembly with that hole left by the edges; this holds whatever
            # way they came. (A dub whose episodes run in another order goes
            # back by minutes onto material it has not played: not this.)
            played_to = last.end_s + last.offset_s
            resumes = block.start_s + block.offset_s
            rewind = played_to - resumes
            if NEAR_OFFSET_S < rewind <= REWIND_GUARD_MAX_S and block.end_s + block.offset_s > last.start_s + last.offset_s:
                block.start_s = block.start_lo = block.start_hi = block.start_s + rewind
                if block.length_s <= 0:
                    continue
        merged.append(block)

    dub_level = _dub_level(secondary, merged)
    original_level = _original_level(primary, merged)

    def dub_audible(lo_s: float, hi_s: float, offset_s: float) -> bool:
        return not _dub_is_silent(secondary, lo_s + offset_s, hi_s + offset_s, dub_level)

    def original_audible(lo_s: float, hi_s: float) -> bool:
        return not _is_quiet(primary, lo_s, hi_s, original_level, FILL_AUDIBLE_RATIO)

    # A gap between two stretches at one offset, with the dub audible in
    # it, is the dub not correlating -- not the dub missing. Kept as the dub
    # by default; when the caller asked for these to be filled instead, the
    # fill is marked as this kind, because it is a different claim from "the
    # dub is cut here": nothing shows the dub *lacks* the scene, only that
    # it could not be matched, and a reader deciding whether to trust the
    # fill needs to know which. Where the two sides sit a few milliseconds
    # apart each keeps its own offset and they meet in the middle of the
    # gap: extending the first side's offset over the second put a
    # twelve-second head's offset over an hour, nineteen milliseconds out.
    unmatched: set = set()
    kept: List[Block] = [merged[0]]
    aligner = aligner or _Aligner(primary, secondary, None, None)
    for block in merged[1:]:
        last = kept[-1]
        gap = block.start_s - last.end_s
        bridge = False
        step = last.offset_s - block.offset_s   # > 0: the video has this much the dub lacks
        exposure = (gap - max(step, 0.0)) / 2.0  # dub a wrongly placed step would misplace
        placed: Optional[Tuple[float, str]] = None
        foreign = gap > MIN_FILL_S and _gap_is_foreign(aligner, last, block)
        if foreign:
            unmatched.add(id(block))
            notes.append(
                f"the original plays across {_clock(last.end_s)} - {_clock(block.start_s)}: the dub has something "
                f"else there -- its music and effects agree with the video's at neither side's offset"
            )
        if (
            not foreign and gap > MIN_FILL_S and NEAR_OFFSET_S < abs(step)
            and gap - max(step, 0.0) >= 0.0
            and keep_unmatched_dub and dub_audible(last.end_s, block.start_s, last.offset_s)
        ):
            placed = _place_step(aligner, last.end_s, block.start_s, last.offset_s, block.offset_s)
        if placed is not None and (
            placed[1] == "picture"
            or (placed[1] in ("sound", "speech") and exposure <= BRIDGE_SOUND_EXPOSURE_S)
            or abs(step) <= BRIDGE_STEP_S or exposure <= BRIDGE_EXPOSURE_S
        ):
            # The dub is there but did not correlate, and the offset steps
            # somewhere inside: a scene the two edits trim differently. Put
            # the step on the picture's cut, or where the agreement changes
            # from one offset to the other -- placed so, the dub is kept
            # across however long the gap is, since nothing is guessed.
            # Where neither can say, only a small step or a short span is
            # guessed at (the middle), which misplaces at most ``exposure``
            # seconds of dub, said in the note. The frames the dub lacks
            # are filled at the step.
            cut, found = placed
            how = {
                "picture": f"put on the picture cut at {_clock(cut)}",
                "sound": "placed where the agreement changes",
                "speech": f"placed where the two languages' lines change, {_clock(cut)}",
            }.get(found, (
                "exact: the gap is the missing length" if exposure < 0.05
                else f"guessed; up to {exposure:.1f}s of dub may sit {abs(step):.1f}s off"
            ))
            said = (
                f"kept the dub across {_clock(last.end_s)} - {_clock(block.start_s)}: it did not "
                f"correlate there, and the offset steps by {_step_words(step)} inside it; the step was {how}"
            )
            if found == "guess" and exposure >= GUESS_WARN_S:
                # Nothing placed it, and a second or more of dub on the wrong
                # side of it is on the wrong lips: said where it is seen.
                warnings.append(said + f" ({_clock(cut)}) -- check the lips there")
            else:
                notes.append(said)
            last.end_s = last.end_lo = last.end_hi = cut
            block.start_s = block.start_lo = block.start_hi = cut + max(step, 0.0)
            kept.append(block)
            continue
        if not foreign and gap > MIN_FILL_S and abs(block.offset_s - last.offset_s) <= NEAR_OFFSET_S:
            if dub_audible(last.end_s, block.start_s, last.offset_s):
                if keep_unmatched_dub:
                    step_ms = abs(block.offset_s - last.offset_s) * 1000.0
                    how = "is the same" if step_ms <= SAME_OFFSET_S * 1000.0 else f"differs by only {step_ms:.0f} ms"
                    notes.append(
                        f"kept the dub across {_clock(last.end_s)} - {_clock(block.start_s)}: it did not "
                        f"correlate there, but the offset {how} either side and the dub is not silent"
                    )
                    bridge = True
                else:
                    unmatched.add(id(block))
            elif not original_audible(last.end_s, block.start_s):
                # Quiet in both: silence either way, so the dub runs on
                # rather than being spliced twice around a fill of nothing.
                bridge = True
        if bridge:
            if abs(block.offset_s - last.offset_s) <= SAME_OFFSET_S:
                _join(last, block, weaker=True)
                continue
            # Two offsets a few frames apart: a picture trim, whose switch
            # sits on a shot change; the sound decides among them, else it
            # is the middle of the gap.
            switch, found = _place_step(aligner, last.end_s, block.start_s, last.offset_s, block.offset_s, hole=False)
            if notes and notes[-1].startswith(f"kept the dub across {_clock(last.end_s)} "):
                if found == "guess" and block.start_s - last.end_s >= 0.5:
                    # Neither the sound nor the picture said where the two
                    # offsets meet; up to half the gap may sit a few frames
                    # out. Worth a look, so it is said where it is seen.
                    notes.pop()
                    warnings.append(
                        f"kept the dub across {_clock(last.end_s)} - {_clock(block.start_s)}, where it did not correlate; "
                        f"its two sides sit {abs(block.offset_s - last.offset_s) * 1000.0:.0f} ms apart and where one gives "
                        f"way to the other ({_clock(switch)}) is a guess -- check the lips there"
                    )
                elif found != "guess":
                    notes[-1] += "; the switch is " + (
                        f"on the picture cut at {_clock(switch)}" if found == "picture" else f"where the agreement changes, {_clock(switch)}"
                    )
            last.end_s = last.end_lo = last.end_hi = switch
            block.start_s = block.start_lo = block.start_hi = switch
        kept.append(block)
    merged = kept

    # The ends of the video, by the same rule: dub audible up to the end is
    # kept, or replaced and marked when asked; dub silent there is filled
    # from the original where the original has something, and left as the
    # dub where it has not -- silence for silence is not a fill, and was
    # being reported as the dub starting late and ending early.
    first = merged[0]
    reach = max(0.0, -first.offset_s)  # the earliest moment the dub has material for
    leading_note = "dub starts late"
    if first.start_s - reach > MIN_FILL_S:
        if dub_audible(reach, first.start_s, first.offset_s):
            if keep_unmatched_dub and first.start_s - reach <= min(EDGE_KEEP_MAX_S, first.length_s):
                notes.append(
                    f"kept the dub across {_clock(reach)} - {_clock(first.start_s)}: it did not "
                    f"correlate there, but the offset carries on from the stretch after it and the dub is not silent"
                )
                first.start_s = first.start_lo = first.start_hi = reach
            else:
                leading_note = UNMATCHED_NOTE
        elif original_audible(reach, first.start_s):
            if reach <= MIN_FILL_S:
                leading_note = "dub is silent here"
        else:
            first.start_s = first.start_lo = first.start_hi = reach
    last = merged[-1]
    reach = min(primary_s, secondary_s - last.offset_s)  # the latest moment it has material for
    trailing_note = "past dub end"
    if reach - last.end_s > MIN_FILL_S:
        if dub_audible(last.end_s, reach, last.offset_s):
            if keep_unmatched_dub and reach - last.end_s <= min(EDGE_KEEP_MAX_S, last.length_s):
                notes.append(
                    f"kept the dub across {_clock(last.end_s)} - {_clock(reach)}: it did not "
                    f"correlate there, but the offset carries on from the stretch before it and the dub is not silent"
                )
                last.end_s = last.end_lo = last.end_hi = reach
            else:
                trailing_note = UNMATCHED_NOTE
        elif original_audible(last.end_s, reach):
            if primary_s - reach <= MIN_FILL_S:
                trailing_note = "dub is silent here"
        else:
            last.end_s = last.end_lo = last.end_hi = reach

    segments: List[Segment] = []
    cursor = 0.0
    previous: Optional[Block] = None
    for block in merged:
        if block.start_s - cursor > MIN_FILL_S:
            if not segments:
                note = leading_note
            elif id(block) in unmatched:
                note = UNMATCHED_NOTE
            elif (
                abs(block.offset_s - previous.offset_s) <= NEAR_OFFSET_S
                if previous else False
            ):
                # The dub runs on through the gap at one offset but has
                # nothing audible in it: silence laid where a scene was.
                note = "dub is silent here"
            else:
                note = "dub is cut here"
            about = max(block.start_uncertainty_s, previous.end_uncertainty_s if previous else 0.0)
            segments.append(Segment("fill", cursor, block.start_s, cursor, note=note, uncertainty_s=about))
        elif block.start_s > cursor:
            block.start_s = cursor
        segments.append(Segment(
            "dub", block.start_s, block.end_s, block.start_s + block.offset_s,
            offset_s=block.offset_s, match=block.match,
            uncertainty_s=max(block.start_uncertainty_s, block.end_uncertainty_s),
        ))
        cursor = block.end_s
        previous = block
    if primary_s - cursor > MIN_FILL_S:
        segments.append(Segment(
            "fill", cursor, primary_s, cursor, note=trailing_note,
            uncertainty_s=previous.end_uncertainty_s if previous else 0.0,
        ))
    elif segments:
        segments[-1].end_s = primary_s

    if doubt is not None:
        # What of the trimmed doubt is still original in the end: a bridge
        # laid afterwards may have kept the dub across some of it.
        for lo, hi in trims:
            for segment in segments:
                if segment.kind == "fill":
                    a, b = max(lo, segment.start_s), min(hi, segment.end_s)
                    if b - a > MIN_FILL_S:
                        doubt.append((a, b))

    # To a tenth of a millisecond: the offsets are measured to a fraction of
    # one, the render places every piece to the sample, and a plan rounded
    # to the millisecond threw half of one away at every piece.
    for segment in segments:
        segment.start_s = round(segment.start_s, 4)
        segment.end_s = round(segment.end_s, 4)
        segment.source_start_s = round(segment.source_start_s, 4)
        if segment.offset_s is not None:
            segment.offset_s = round(segment.offset_s, 4)
        segment.uncertainty_s = round(segment.uncertainty_s, 3)
    return [s for s in segments if s.end_s > s.start_s]
