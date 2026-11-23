"""Tell one timeline from two: find the step a splice leaves in the offsets.

A frame-rate mismatch bends the required offset gradually. A cut moves it all
at once and leaves it there. Fitted with a single line the two are easy to
confuse -- six windows straddling a two-second splice fit a gentle slope whose
residuals look small, and that slope lands inside the tolerance of a real
23.976 -> 24 conversion, so the file gets diagnosed as needing a resample.
Rejecting the line means fitting the other model as well and asking which one
the measurements actually prefer.

The comparison needs no penalty term, because both models spend the same two
parameters: a line spends slope and intercept, a step spends the level before
and the level after. Whichever leaves less unexplained is the better
description of the file. With at most a few dozen windows every possible split
can simply be tried, so there is nothing to approximate.

What this deliberately does not model is a cut *and* drift together, which
would need four parameters and could no longer be compared to the line for
free. A file with both is reported as whichever it resembles more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Fewer windows than this and the question is meaningless: with three points a
# split always fits two of them exactly, so every file looks like a cut.
MIN_WINDOWS = 4

# How much better the step has to explain the measurements before it wins.
# Like-for-like at two parameters each, so this is not a handicap but a margin:
# the step must leave at most half the unexplained variance the line does.
STEP_PREFERENCE = 0.5

# Steps smaller than this are not worth stopping a mux for. Half a frame at
# 24fps, and inside what a different mix of the same scene can move a
# correlation peak by on its own.
MIN_STEP_MS = 20.0

# However large in absolute terms, the step must also stand clear of how much
# the windows disagree among themselves. Five sigma against a scale that a real
# step cannot inflate: at the measured noise floor a one-frame cut is still
# found every time, and a file with no cut in it is claimed to have one in
# fewer than one run in ten thousand.
NOISE_MULTIPLE = 5.0

# Windows measured against a shared music-and-effects bed scatter by about
# 0.3ms, and synthetic material scatters by nothing at all. A zero scale would
# make the noise test vacuous, so it never falls below this.
NOISE_FLOOR_MS = 1.0

# Two windows read the same level of a staircase when they agree this closely.
# Matched windows land within a fraction of a millisecond of each other on a
# shared music-and-effects bed and within a few through a dub's own mix;
# agreement this tight between windows minutes apart is what a drift never
# shows (see _first_of_staircase).
LEVEL_MATCH_MS = 5.0

# How far apart the line must put two windows that agree before their
# agreement refutes it. Four times LEVEL_MATCH_MS, so the slope of a real but
# small drift -- which moves close windows by less than their own scatter --
# is never mistaken for a staircase.
STAIRCASE_CONTRADICTION_MS = 4.0 * LEVEL_MATCH_MS


@dataclass
class Step:
    """A discontinuity in the offsets, and the span it must lie in."""

    before_ms: float
    after_ms: float

    split_position_s: float
    """Where windows stop belonging to the first segment and start belonging to
    the second. Held as a position rather than an index so it survives the
    outlier trim and any windows added later."""

    earliest_s: float
    latest_s: float
    """Bracket on the cut itself, in the primary's timeline. It starts as wide
    as the two windows that straddle it -- the cut is somewhere after the last
    pre-cut window began and before the first post-cut window ended -- and
    probing narrows it."""

    lone_group: bool = False
    """One side of the split is a single window. That is also exactly what a
    window which locked onto a repeated musical phrase looks like, so this side
    has to be reproduced somewhere else before the cut is believed."""

    levels: int = 2
    """How many different offsets the windows hold in order: 2 for one cut,
    more for a file cut in several places, of which this step is the first."""

    @property
    def magnitude_ms(self) -> float:
        """How far the offset jumps, signed: positive means it grows."""
        return self.after_ms - self.before_ms

    @property
    def position_s(self) -> float:
        return (self.earliest_s + self.latest_s) / 2.0

    @property
    def uncertainty_s(self) -> float:
        return (self.latest_s - self.earliest_s) / 2.0


def _residual_sum(values: np.ndarray) -> float:
    """Squared error left by describing these values with their own mean."""
    if len(values) == 0:
        return 0.0
    return float(np.sum((values - values.mean()) ** 2))


def _line_residual_sum(positions: np.ndarray, offsets: np.ndarray) -> float:
    slope, intercept = np.polyfit(positions, offsets, 1)
    return float(np.sum((offsets - (slope * positions + intercept)) ** 2))


def _noise_scale(offsets: np.ndarray) -> float:
    """How much neighbouring windows disagree, when nothing is going on.

    Measured from the gaps between consecutive windows rather than from the
    scatter around the chosen split. Distance from the split is the obvious
    scale and the wrong one: the split was picked to make exactly that distance
    small, so the noise it reports is the noise the search already minimised,
    and the threshold built on it lets marginal steps through -- measured, a
    file with no cut in it was claimed to have one six times as often.

    A step shows up in exactly one of these gaps, which the median discards
    along with the rest of the tail. Drift shows up in all of them and inflates
    the scale, which is the right direction: a file that is genuinely sliding
    should need a bigger jump before the slide is called a splice.
    """
    gaps = np.abs(np.diff(offsets))
    if len(gaps) < 2:
        return 0.0
    # Median absolute value of a difference of two independent samples, scaled
    # back to the standard deviation of one of them.
    return float(np.median(gaps)) * 1.0483


def find_step(
    positions: Sequence[float],
    offsets: Sequence[float],
    window_s: float = 0.0,
) -> Optional[Step]:
    """Find the single best place to split these measurements, if splitting wins.

    Args:
        positions: window start positions in the primary, ascending.
        offsets: the offset each window measured, in milliseconds.
        window_s: how long each window is, which bounds how late the cut can be.

    Returns:
        The step, or None when one line describes the file at least as well.
    """
    positions = np.asarray(positions, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    count = len(offsets)

    if count < MIN_WINDOWS or float(np.ptp(positions)) <= 0.0:
        return None

    line_error = _line_residual_sum(positions, offsets)

    best_index, best_error = 1, None
    for index in range(1, count):
        error = _residual_sum(offsets[:index]) + _residual_sum(offsets[index:])
        if best_error is None or error < best_error:
            best_index, best_error = index, error

    if best_error > line_error * STEP_PREFERENCE:
        # One step loses to the line; several may not. A dub cut in three
        # places reads 700, -3300, -3300, -11300, -14300, -14300 ms: no single
        # split explains that better than a slope does, and the slope (-23
        # ms/s) was reported as a drift whose t=0 intercept, 34 ms from the
        # truth, went into the delay field. A drift cannot do what those
        # windows did, though -- read the same offset to the millisecond 164
        # seconds apart -- so the line is tested against that before it wins.
        return _first_of_staircase(positions, offsets, window_s)

    # Least squares chose the split, because that is what makes the comparison
    # with the line fair. The levels it reports are medians, which one bad
    # window inside a segment cannot drag.
    before = float(np.median(offsets[:best_index]))
    after = float(np.median(offsets[best_index:]))

    scale = max(NOISE_FLOOR_MS, _noise_scale(offsets))
    if abs(after - before) < max(MIN_STEP_MS, NOISE_MULTIPLE * scale):
        # Consecutive windows disagreeing is also what several cuts look
        # like, each in its own gap; a staircase is told apart by its
        # repeated levels instead.
        return _first_of_staircase(positions, offsets, window_s)

    last_before = float(positions[best_index - 1])
    first_after = float(positions[best_index])
    return Step(
        before_ms=before,
        after_ms=after,
        split_position_s=(last_before + first_after) / 2.0,
        earliest_s=last_before,
        latest_s=first_after + window_s,
        lone_group=min(best_index, count - best_index) < 2,
    )


def _first_of_staircase(
    positions: np.ndarray,
    offsets: np.ndarray,
    window_s: float,
) -> Optional[Step]:
    """The first step of a file the offset changes in several times, or None
    when a line describes the windows after all.

    The line is refused only on evidence a drift cannot produce: two windows
    LEVEL_MATCH_MS apart or closer whose distance in the file means the line
    has them STAIRCASE_CONTRADICTION_MS or more apart. A real conversion moves
    every window, so its windows never repeat an offset; a file cut in
    places holds each offset between its cuts, and its windows do. The first
    level is then the run of windows from the start that agree, and the step
    is where the next window leaves it.
    """
    count = len(offsets)
    slope, intercept = np.polyfit(positions, offsets, 1)
    residual = offsets - (slope * positions + intercept)
    if float(np.sqrt(np.mean(residual ** 2))) < MIN_STEP_MS:
        return None
    contradicted = False
    for i in range(count):
        for j in range(i + 1, count):
            if (
                abs(offsets[j] - offsets[i]) <= LEVEL_MATCH_MS
                and abs(slope) * (positions[j] - positions[i]) >= STAIRCASE_CONTRADICTION_MS
            ):
                contradicted = True
    if not contradicted:
        return None
    # A staircase also has to look like one: every change the same way --
    # scenes the dub lacks each pull the offset down, scenes it adds each push
    # it up -- and levels that hold across more than one window, two windows
    # at least sharing one. Windows scattering about one offset repeat a value
    # by chance, but they wander up and down and rarely twice.
    changes = [
        current - previous for previous, current in zip(offsets[:-1], offsets[1:])
        if abs(current - previous) > LEVEL_MATCH_MS
    ]
    if not changes or not (all(c > 0 for c in changes) or all(c < 0 for c in changes)):
        return None
    if _level_count(offsets) > count - 2:
        return None

    first_run = 1
    while first_run < count and abs(offsets[first_run] - offsets[0]) <= LEVEL_MATCH_MS:
        first_run += 1
    if first_run >= count:
        return None
    before = float(np.median(offsets[:first_run]))
    # The level the second run holds: the windows that agree with the first
    # one past the step.
    second = [first_run]
    while first_run + len(second) < count and abs(
        offsets[first_run + len(second)] - offsets[first_run]
    ) <= LEVEL_MATCH_MS:
        second.append(first_run + len(second))
    after = float(np.median(offsets[second]))
    if abs(after - before) < MIN_STEP_MS:
        return None
    last_before = float(positions[first_run - 1])
    first_after = float(positions[first_run])
    return Step(
        before_ms=before,
        after_ms=after,
        split_position_s=(last_before + first_after) / 2.0,
        earliest_s=last_before,
        latest_s=first_after + window_s,
        # Either level on a single window is what one stray reading looks
        # like, so probing has to reproduce it before the step is believed.
        lone_group=first_run < 2 or len(second) < 2,
        levels=_level_count(offsets),
    )


def _level_count(offsets: np.ndarray) -> int:
    """How many distinct offsets, LEVEL_MATCH_MS apart or more, the windows
    hold in the order they come: the steps a staircase has, plus one."""
    levels = 1
    for previous, current in zip(offsets[:-1], offsets[1:]):
        if abs(current - previous) > LEVEL_MATCH_MS:
            levels += 1
    return levels
