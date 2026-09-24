"""The line check: do the dub's lines start and stop where the original's do?

A dub is performed to the picture: its actors start a line when the mouth
on screen opens and stop when it closes, in any language. So along a scene
the dub's speech -- switching on and off, line by line -- follows the
original's, give or take a syllable, and the lag at which the two agree
best is where the dub's voices sit against the original's, which are in
sync with the lips by definition. One line says little (a performance
lands a tenth of a second either side of the mouth); a minute of them says
where the dub sits to within a frame. This is the check of the dub's lip
sync that needs no picture; the milliseconds come from the music and
effects, and this looks for what they cannot see -- a stretch sitting on
the wrong beat of a cue, a scene whose voices were laid a few frames off.

Both tracks are read in stereo and reduced to their centre (see
``speech.centre``), where dialogue is mixed, so the music and effects the
two share do not agree for the voices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .dubsync import DubSyncPlan, _clock, ncc_lags
from .media import CancellationToken, MediaError
from .speech import SPEECH_RATE, centre_activity

# Windows of this many seconds of dialogue, this far apart. The agreement
# is taken over lags up to LINE_BASELINE_MS either way, which is what its
# peak is measured against; a peak further out than LINE_RANGE_MS is no
# reading of where the lines sit (they agree on nothing near sync).
LINE_WINDOW_S = 60.0
LINE_STEP_S = 30.0
LINE_RANGE_MS = 300.0
LINE_BASELINE_MS = 2000.0
# A window is judged when both tracks talk for this share of it and their
# agreement peaks this many deviations above the agreement at other lags.
LINE_MIN_TALK = 0.2
LINE_MIN_Z = 3.5
# One performance lands a tenth of a second either side of the mouth, so a
# single window is only called out past this; the median over all of them
# is the dub's lip sync.
LINE_TOLERANCE_MS = 150.0

LogFn = Callable[[str], None]


@dataclass
class LineWindow:
    start_s: float
    end_s: float
    lag_ms: Optional[float]
    z: float
    note: str = ""

    def to_dict(self) -> dict:
        return {"startS": self.start_s, "endS": self.end_s, "lagMs": self.lag_ms, "z": self.z, "note": self.note}


@dataclass
class LineCheck:
    windows: List[LineWindow] = field(default_factory=list)

    @property
    def judged(self) -> List[LineWindow]:
        return [w for w in self.windows if w.lag_ms is not None]

    def to_dict(self) -> dict:
        lags = [abs(w.lag_ms) for w in self.judged]
        return {
            "windows": [w.to_dict() for w in self.windows],
            "judged": len(lags),
            "typicalMs": float(np.median(lags)) if lags else None,
            "overallMs": float(np.median([w.lag_ms for w in self.judged])) if lags else None,
            "worstMs": float(max(lags)) if lags else None,
            "withinTolerance": sum(1 for lag in lags if lag <= LINE_TOLERANCE_MS),
            "toleranceMs": LINE_TOLERANCE_MS,
        }

    def describe(self) -> str:
        judged = self.judged
        if not judged:
            return f"lines: none of {len(self.windows)} dialogue windows could be judged"
        lags = [abs(w.lag_ms) for w in judged]
        signed = float(np.median([w.lag_ms for w in judged]))
        within = sum(1 for lag in lags if lag <= LINE_TOLERANCE_MS)
        lines = [
            f"lines: the dub's speech follows the original's across {len(judged)} of {len(self.windows)} dialogue "
            f"windows of {LINE_WINDOW_S:.0f}s; overall it sits {signed:+.0f} ms from the original's lines "
            f"(a window reads to about a tenth of a second: typically {np.median(lags):.0f} ms, at worst "
            f"{max(lags):.0f} ms; {within} of {len(judged)} within {LINE_TOLERANCE_MS:.0f} ms)"
        ]
        for window in judged:
            if abs(window.lag_ms) > LINE_TOLERANCE_MS:
                lines.append(
                    f"  {_clock(window.start_s)} - {_clock(window.end_s)}: the dub's lines sit "
                    f"{window.lag_ms:+.0f} ms from the original's ({window.z:.1f}) -- check the lips there"
                )
        return "\n".join(lines)


def _activity(path: str, track: int, lo_s: float, hi_s: float, token: Optional[CancellationToken]) -> np.ndarray:
    """Speech activity of a file's centre across [lo_s, hi_s), SPEECH_RATE a second."""
    return centre_activity(path, track, lo_s, hi_s, token, pad=False)


def line_check(
    plan: DubSyncPlan,
    written_path: str,
    written_track: int = 0,
    token: Optional[CancellationToken] = None,
    log: Optional[LogFn] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> LineCheck:
    """Compare the written track's lines with the original's, window by window,
    inside the dub's stretches."""
    say = log or (lambda _m: None)
    result = LineCheck()
    windows: List[Tuple[float, float]] = []
    for segment in plan.dub_segments:
        start = segment.start_s
        while start + LINE_WINDOW_S <= segment.end_s:
            windows.append((start, start + LINE_WINDOW_S))
            start += LINE_STEP_S
    reach = int(round(LINE_BASELINE_MS / 1000.0 * SPEECH_RATE))
    near = int(round(LINE_RANGE_MS / 1000.0 * SPEECH_RATE))
    pad = LINE_BASELINE_MS / 1000.0 + 0.1
    for index, (lo, hi) in enumerate(windows):
        if token:
            token.raise_if_cancelled()
        try:
            original = _activity(plan.video_path, plan.video_track, lo, hi, token)
            dub = _activity(written_path, written_track, lo - pad, hi + pad, token)
            # A read that ran into either end of the file comes back short;
            # silence stands in for what lies beyond, so every position of
            # the dub's curve stays where the lags are counted from.
            front = int(round(max(0.0, pad - lo) * SPEECH_RATE))
            want = int(round((hi - lo + 2 * pad) * SPEECH_RATE))
            dub = np.concatenate([np.zeros(front), dub])
            if len(dub) < want:
                dub = np.concatenate([dub, np.zeros(want - len(dub))])
        except MediaError as exc:
            result.windows.append(LineWindow(lo, hi, None, 0.0, str(exc)))
            continue
        talk = min(float(np.mean(original > 0.3)), float(np.mean(dub > 0.3)))
        if talk < LINE_MIN_TALK:
            result.windows.append(LineWindow(lo, hi, None, 0.0, "too little dialogue"))
            continue
        first = int(round(pad * SPEECH_RATE)) - reach
        row = ncc_lags(original, dub[max(0, first):])
        if row.size < 2 * reach + 1:
            result.windows.append(LineWindow(lo, hi, None, 0.0, "too short"))
            continue
        row = row[: 2 * reach + 1]
        peak = int(np.argmax(row))
        away = np.abs(np.arange(len(row)) - peak) > max(10, near // 10)
        z = float((row[peak] - np.median(row[away])) / (np.std(row[away]) + 1e-9)) if away.any() else 0.0
        shift = 0.0
        if 0 < peak < len(row) - 1:
            y0, y1, y2 = row[peak - 1], row[peak], row[peak + 1]
            denominator = y0 - 2 * y1 + y2
            if abs(denominator) > 1e-12:
                shift = max(-1.0, min(1.0, 0.5 * (y0 - y2) / denominator))
        lag = (peak + shift - reach) * 1000.0 / SPEECH_RATE
        if z < LINE_MIN_Z:
            result.windows.append(LineWindow(lo, hi, None, z, "the lines do not agree clearly"))
        elif abs(lag) > LINE_RANGE_MS:
            result.windows.append(LineWindow(lo, hi, None, z, f"the lines agree best {lag:+.0f} ms away: no reading near sync"))
        else:
            result.windows.append(LineWindow(lo, hi, lag, z))
        if progress:
            progress((index + 1) / max(1, len(windows)))
    say(result.describe())
    return result
