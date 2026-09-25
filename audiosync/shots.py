"""Picture cuts: where the video changes shot, to put a dub's cuts on the frame.

A dub made for another edit of the film differs from the video by whole
scenes, recaps and trims, and every one of those differences starts and
ends where an editor cut the picture. The sound says roughly where such a
cut is -- to a second or two where the two mixes share little -- but the
picture says exactly: a shot change is one frame. So wherever the engine
knows a cut lies somewhere in a span, the span's shot changes are the
candidates, and the one the sound prefers, or the only one there is, is
where the cut goes.

Only the spans asked about are decoded, scaled to SHOT_WIDTH pixels, and
ffmpeg scores every frame against the one before it (the ``select``
filter's scene score). A hard cut scores 0.4 to 1.0 against 0.00 to 0.05
inside a shot, but fast motion -- animation on twos, a whip pan -- reaches
0.3, so a cut must also stand out from the frames around it.

Times are on the file's clock, like the analysis (see
``media.audio_lead_s``): a frame found ``t`` seconds into a decode that
was sought to ``lo`` sits at ``lo + t``.
"""

from __future__ import annotations

import bisect
import re
from typing import Callable, List, Optional, Tuple

import numpy as np

from .media import CancellationToken, MediaError, _run, ffmpeg_path, probe

# Width the picture is scored at. The scene score is a mean difference
# over the frame; a hard cut changes every part of it, so detail does not
# matter and the decode is cheaper.
SHOT_WIDTH = 160
# A frame scoring at least this against the one before it, ...
SHOT_MIN_SCORE = 0.3
# ... this many times the median of the frames around it, plus a floor ...
SHOT_CONTRAST = 4.0
SHOT_CONTRAST_FLOOR = 0.05
# ... over this many frames either side, and the highest among them.
SHOT_CONTEXT = 12
# Picture decoded around a span so its first frames have neighbours.
SHOT_MARGIN_S = 1.0
# At most this much picture is decoded in one run, in seconds of video, on
# top of the film's own length: spans are kept once read, so the budget is
# reached only by reading most of the film, which a 1.5-hour episode with
# a TV edit did in the first stages -- and at 30 minutes the cuts placed
# last, often the hardest, fell back to the sound.
SHOT_BUDGET_S = 600.0

LogFn = Callable[[str], None]


def detect_cuts(times: np.ndarray, scores: np.ndarray) -> List[float]:
    """The shot changes among scored frames: the time of each frame that
    starts a new shot."""
    cuts: List[float] = []
    for index in range(1, len(scores)):
        score = float(scores[index])
        if score < SHOT_MIN_SCORE:
            continue
        lo, hi = max(0, index - SHOT_CONTEXT), min(len(scores), index + SHOT_CONTEXT + 1)
        window = scores[lo:hi]
        if score < float(window.max()):
            continue
        around = np.delete(window, index - lo)
        if around.size and score < SHOT_CONTRAST * float(np.median(around)) + SHOT_CONTRAST_FLOOR:
            continue
        cuts.append(float(times[index]))
    return cuts


def scene_scores(
    path: str, lo_s: float, hi_s: float, token: Optional[CancellationToken] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """(frame times, scene scores) for the picture across [lo_s, hi_s)."""
    lo_s = max(0.0, lo_s)
    command = [
        ffmpeg_path(), "-nostdin", "-v", "error", "-ss", f"{lo_s:.6f}", "-i", path,
        "-t", f"{max(0.0, hi_s - lo_s):.6f}", "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"scale={SHOT_WIDTH}:-2:flags=fast_bilinear,select=gte(scene\\,0),"
               "metadata=print:key=lavfi.scene_score:file=-",
        "-f", "null", "-",
    ]
    out = _run(command, 3600, token, what="read the picture").decode("utf-8", errors="replace")
    times: List[float] = []
    scores: List[float] = []
    pending: Optional[float] = None
    for line in out.splitlines():
        found = re.search(r"pts_time:\s*(-?[0-9.]+)", line)
        if found:
            pending = float(found.group(1))
            continue
        found = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if found and pending is not None:
            times.append(lo_s + pending)
            scores.append(float(found.group(1)))
            pending = None
    return np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float64)


class PictureCuts:
    """The shot changes of one video, read on demand, span by span."""

    def __init__(
        self,
        video_path: str,
        token: Optional[CancellationToken] = None,
        log: Optional[LogFn] = None,
        budget_s: float = SHOT_BUDGET_S,
    ) -> None:
        self.path = video_path
        self.token = token
        self.log = log or (lambda _message: None)
        self.budget_s = budget_s
        self.spent_s = 0.0
        self.frame_s: Optional[float] = None
        try:
            info = probe(video_path, token)
            self.available = bool(info.has_video)
            if info.fps:
                self.frame_s = 1.0 / float(info.fps)
            self.budget_s += float(info.duration or 0.0)
        except MediaError:
            self.available = False
        self._scanned: List[Tuple[float, float]] = []
        self._cuts: List[float] = []
        self._warned = False

    @property
    def tolerance_s(self) -> float:
        """How far apart two times may be and still be the same frame."""
        return 0.6 * (self.frame_s or 1.0 / 24.0)

    def _missing(self, lo_s: float, hi_s: float) -> List[Tuple[float, float]]:
        """The parts of [lo_s, hi_s) not decoded yet."""
        parts: List[Tuple[float, float]] = []
        cursor = lo_s
        for a, b in self._scanned:
            if b <= cursor:
                continue
            if a >= hi_s:
                break
            if a > cursor:
                parts.append((cursor, min(a, hi_s)))
            cursor = max(cursor, b)
            if cursor >= hi_s:
                break
        if cursor < hi_s:
            parts.append((cursor, hi_s))
        return [(a, b) for a, b in parts if b - a > 1e-3]

    def _mark(self, lo_s: float, hi_s: float) -> None:
        spans = sorted(self._scanned + [(lo_s, hi_s)])
        merged: List[Tuple[float, float]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self._scanned = merged

    def within(self, lo_s: float, hi_s: float) -> Optional[List[float]]:
        """The shot changes in [lo_s, hi_s], or None when the picture cannot
        say: no video, the decode failed, or the run's budget is spent."""
        if not self.available or hi_s <= lo_s:
            return None
        lo_s = max(0.0, lo_s)
        for a, b in self._missing(lo_s, hi_s):
            if self.spent_s + (b - a) > self.budget_s:
                if not self._warned:
                    self._warned = True
                    self.log(f"  the picture was read for {self.spent_s / 60:.0f} min; the rest of the cuts are placed by the sound")
                return None
            if self.token:
                self.token.raise_if_cancelled()
            try:
                times, scores = scene_scores(self.path, a - SHOT_MARGIN_S, b + SHOT_MARGIN_S, self.token)
            except MediaError as exc:
                self.available = False
                self.log(f"  the picture could not be read ({exc}); cuts are placed by the sound")
                return None
            self.spent_s += (b - a) + 2 * SHOT_MARGIN_S
            for cut in detect_cuts(times, scores):
                if a <= cut < b:
                    bisect.insort(self._cuts, cut)
            self._mark(a, b)
        first = bisect.bisect_left(self._cuts, lo_s)
        last = bisect.bisect_right(self._cuts, hi_s)
        return self._cuts[first:last]

    def pairs(self, lo_s: float, hi_s: float, length_s: float) -> Optional[List[float]]:
        """Shot changes ``c`` in [lo_s, hi_s] with another shot change
        ``length_s`` after it, to within a frame: where a stretch of picture
        of that length could have been cut out. None when the picture
        cannot say."""
        cuts = self.within(lo_s, hi_s + length_s + self.tolerance_s)
        if cuts is None:
            return None
        found = []
        for cut in cuts:
            if cut > hi_s:
                break
            target = cut + length_s
            index = bisect.bisect_left(cuts, target - self.tolerance_s)
            if index < len(cuts) and abs(cuts[index] - target) <= self.tolerance_s:
                found.append(cut)
        return found
