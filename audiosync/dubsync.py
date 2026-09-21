"""Lay a cut dub onto the video it belongs to, filling what it lacks.

A dub is very often a different edit of the same film: a scene the dubbing
studio never received, a recap trimmed for broadcast, a longer logo at the
head. Measured as one pair it reports "different cut" and stops, because no
single delay describes it. This module goes further: it works out, along the
whole runtime, which stretch of the dub belongs at each moment of the video,
and where nothing does, it puts the video's own audio there instead. The
result is a track the length of the video that plays the dub wherever the dub
exists and the original wherever it does not.

How the timeline is recovered
-----------------------------
Both tracks are reduced to onset-strength envelopes (see ``correlate``), which
is what survives the dub being a different performance: dialogue differs, but
the music and effects underneath it are the same stems, and their transients
sit at the same instants. The video's envelope is cut into windows and each
window is correlated against the dub over the whole plausible range of
offsets, which gives a score for every (moment, offset) pair.

Those scores are not read off one window at a time. A window that happens to
land on a repeated musical phrase produces a confident peak at the wrong
offset, and reading windows independently would splice the track there. The
offsets are instead found as one path through the whole matrix, which stays
at one offset for free and pays a fixed price to jump, so a jump has to be
earned by the windows after it agreeing with each other -- the same reason
``segments`` compares a step against a line instead of trusting any one
window. A third possibility sits alongside every offset: that no part of the
dub belongs here. It earns a little per window, so a long stretch nothing
matches is called a gap rather than a coincidence.

The path is found coarsely first, at 20 ms over the whole file, then again
finely inside each place it changes, then each boundary is placed to the
onset envelope's own 2 ms resolution by asking exactly where the agreement
between the two tracks stops. The offset inside every stretch is measured
last, at that same resolution, from the stretch alone.

Scores are compared in units of each window's own noise. The threshold that
separates a real match from a coincidence was calibrated on real dubs for the
single-pair analysis (``correlate.MIN_PEAK_RATIO``), and expressing every
decision here as a multiple of the noise floor is what lets that calibration
carry over to windows of a different length.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .correlate import ENVELOPE_HOP, _fast_fft_size
from .framerate import COMMON_RATES, RATIO_TOLERANCE, _format_fps, exact_rate, speed_candidates
from .segments import find_step
from .media import CancellationToken, MediaError, load_audio, probe, stream_audio

ANALYSIS_SR = 16000
ENVELOPE_RATE = ANALYSIS_SR // ENVELOPE_HOP  # 500 Hz: one value per 2 ms

# --- bands -------------------------------------------------------------------

# Besides the full-band envelope, two band-limited ones, computed in the same
# pass. A dub shares its music and effects bed with the original but not its
# dialogue, and the dialogue sits in the middle of the spectrum: in a scene
# with no music the full-band onsets are two languages' consonants and agree
# on nothing, while the 30-250 Hz band (bass, footsteps, room rumble) and
# the 4-8 kHz band (ambience, foley, sibilance of the bed) still correlate.
# On a real pair the low band found 21 of 23 minutes the full band had
# given up on (z 9-20 where the full band sat at 3-5).
#
# The high band is more than a fallback. At a small picture trim (a few
# frames) the dub's dialogue and effects follow the picture but its music
# is often left running, so for a while the music sits a whole number of
# frames from the effects. The dialogue, recorded to the picture, goes with
# the effects: the ambience steps exactly at the shot changes at the
# effects' offset and not at the music's. So wherever the high band is
# confident it decides, and a music-only level that is not a whole number
# of frames from the effects level is an artefact (a cue laid twice, a
# beat alias), never a cut.
BANDS: Dict[str, Tuple[float, float]] = {"low": (30.0, 250.0), "high": (4000.0, 8000.0)}
BAND_WINDOW = 384
"""STFT window for the band envelopes, in samples at ANALYSIS_SR: 24 ms,
41.7 Hz per bin, so the low band is six bins wide and a transient is
still a 24 ms bump."""
# The high band's reading overrides the others' when it stands this far
# above its noise; below, the bands are pooled by whichever peaks highest.
EFFECTS_Z = 10.0
# How far the effects are looked for around a stretch's offset: a trim of
# up to twelve frames, which is what a music-versus-effects split looks like.
EFFECTS_RANGE_S = 0.5
# ...and its peak must be this many times the size of the row's other
# peaks (see _peak_dominance).
EFFECTS_DOMINANCE = 3.0
# The high band's noise units, scaled to the full band's when it competes
# with the other bands as a reader (its chance peaks run about 1.3x).
EFFECTS_DISCOUNT = 0.75
# Slack when the high band is asked which of two levels it agrees with.
EFFECTS_VOTE_SLACK_S = 0.012
# A reading whose peak is not this far above its own row found nothing:
# the tallest of a few hundred chance offsets reaches three, of a
# thousand three and a half.
READING_Z = 4.5
# How far from a whole number of frames two effects levels may sit and
# still both be picture trims. Readings resolve 2 ms; a doubled music cue
# sits at any distance (104, 270, 20 ms were measured) and a beat alias
# at its period.
FRAME_TOLERANCE_MS = 12.0
# How far apart, in offset and in time, a music-placed stretch and an
# effects-placed neighbour may be for the effects to be asked about it.
FOLLOW_EFFECTS_STEP_S = 0.5
FOLLOW_EFFECTS_GAP_S = 3.0
# An excursion (see _Aligner._effects_rule): at least this large, over
# within this long.
EXCURSION_MIN_MS = 250.0
EXCURSION_MAX_S = 60.0
# A stretch whose own peak does not stand this far above the offsets around
# it, in any band, is a coincidence (see _Aligner.credibility).
CREDIBLE_Z = 5.0
CREDIBILITY_SPAN_S = 30.0
CREDIBILITY_RANGE_S = 1.0

# --- coarse pass: the whole file at once -----------------------------------

# Envelope frames pooled per coarse sample. 10 frames is 20 ms, fine enough
# that a transient still lands in the same bin on both sides, coarse enough
# that a two-hour film is a few hundred thousand samples.
COARSE_POOL = 10
# Thirty seconds, not twenty: on a synthetic pair whose dialogue outweighs
# its shared bed, the weakest window's peak rises from 1.0 to 2.6 noise
# units with the extra ten seconds, and the fine pass places the boundary
# regardless of how long the coarse window was.
COARSE_WINDOW_S = 30.0
COARSE_HOP_S = 10.0
# The score matrix is windows x candidate offsets, in float32. Past this many
# cells the pooling is doubled: a coarser offset grid costs nothing, since
# every offset is re-measured at full resolution afterwards, while a matrix
# that does not fit in memory costs the run.
COARSE_CELL_BUDGET = 40_000_000
# How far beyond the difference in the two durations the offset may wander.
# The durations bound the total that was cut, not where the pieces sit: a dub
# with a minute more logo at the head and a minute less recap in the middle
# is the same length as the video and two minutes out either way.
DEFAULT_SEARCH_S = 120.0

# --- the path through the scores, in units of a window's noise ------------

# What a jump between two offsets costs. Two windows of a solid match pay for
# it (a real match stands 8 or more noise units above the floor; see
# MATCH_Z), and the largest coincidence a single window can produce over a
# quarter of a million offsets is under five, so a fluke never does.
JUMP_COST = 15.0
# Moving one grid step between consecutive windows. Nearly free, because the
# peak genuinely wanders a step with the material, but not entirely, so the
# path does not random-walk through a quiet passage.
WOBBLE_COST = 0.5
# Declaring that nothing in the dub belongs here earns this per window, and
# entering or leaving that state costs NULL_COST. Staying at a wrong offset
# earns about zero per window, so a stretch of no match has to run for
# 2 * NULL_COST / NULL_REWARD windows before it is called a gap: two minutes
# at the coarse hop. Cuts do not depend on this -- they show as a jump and the
# gap is deduced from it -- so it only has to catch the head, the tail, and
# a dub that went silent, all of which run longer than that.
NULL_REWARD = 1.0
NULL_COST = 6.0
# A window whose best peak stands this far above its own noise floor is a
# match on its own, with no help from its neighbours. Reported, not relied
# on: the path integrates evidence across windows, and on a pair whose
# dialogue outweighs the shared bed a genuine match may sit at three.
MATCH_Z = 5.0
# A stretch whose windows average less than this is not a stretch of the dub;
# it is the path resting on a coincidence. It becomes a gap. This is the
# floor; the bar a short stretch has to clear is higher, see `_block_bar`.
MIN_BLOCK_Z = 2.0
# Margin over the largest mean a run of windows at some random offset can
# reach by selection alone.
BLOCK_BAR_MARGIN = 1.3
# Runs of this many windows or fewer must reach MATCH_Z instead.
SHORT_RUN_WINDOWS = 3
# A stretch whose offset, measured at 2 ms from inside it, correlates below
# this is noise: ten-second windows at the envelope rate have a noise floor
# near 0.015, and the largest coincidence across the search range is under
# 0.04. Real stretches on the weakest synthetic pair sit at 0.10.
MIN_BLOCK_MATCH = 0.05
# A stretch below that floor is still believed when its readings agree:
# sub-windows measured on their own land on the same offset to within this
# when the dub is there, and scatter over the whole search range when it
# is not. Through AAC and AC3 a real stretch on the weakest synthetic pair
# reads 0.04, under the floor, with every reading within 2 ms of the rest.
CONSISTENT_MS = 6.0
CONSISTENT_FRACTION = 0.75
CONSISTENT_READINGS = 4
# The relative floor: this fraction of the median match of the long
# stretches, never above the cap.
RELATIVE_FLOOR = 0.25
RELATIVE_FLOOR_CAP = 0.10
# A stretch whose windows averaged this many noise units is real whatever
# its fine match says: the largest coincidence a coarse window can produce
# is under five, and this is the mean over at least three of them.
STRONG_SUPPORT_Z = 8.0
# Three readings that all land within this of each other are as good as
# four that mostly agree: the chance of that from noise over a 240 ms
# range is a few in ten thousand.
TIGHT_MS = 2.0

# --- recovering the dub inside the gaps ------------------------------------

# Every gap at least this long is searched again, with windows a third the
# coarse length every few seconds, over only the offsets the neighbouring
# stretches allow. The coarse pass needs two minutes of nothing before it
# calls a gap, and a 30s window straddles any cut shorter than itself; a
# dub whose edit differs from the video's every couple of minutes -- a
# streaming master against a different region's disc -- leaves it with
# gaps that hold most of the dub. On a real pair the single-template
# search this replaces found the dub in those gaps and then could not
# measure it, because one offset does not describe a span with three cuts
# inside it. Short windows at a small hop find the pieces separately.
RECOVER_MIN_S = 6.0
# Window lengths tried in each gap, finest first. A scene with a loud
# shared bed is found by the short windows, which place its cuts closely;
# a quiet one needs the long windows, whose evidence adds up over a
# minute or more where a ten-second window sees only noise. Where a
# finer scale found the dub, the coarser scales are not asked.
GAP_WINDOWS_S = (10.0, 30.0, 90.0)
GAP_HOP_S = 5.0
# The gap search runs on the envelope at its full 2 ms resolution, not the
# coarse pass's 20 ms bins. Real music and effects are sharp spikes, and
# in a quiet scene the shared ones are outnumbered by each mix's own
# dialogue onsets; in a 20 ms bin the shared spike is diluted by ten times
# its weight of unrelated ones and the correlation sinks into the noise.
# Measured on a real pair's quiet six-minute gap: 30-second windows scored
# 3 noise units at 20 ms and 6 to 36 at 2 ms. The coarse pass cannot
# afford 2 ms across a quarter of a million offsets; the gap search, over
# a few thousand, can.
GAP_POOL = 1
# How far either side of the neighbours' offsets the search extends; at a
# gap with only one neighbour, how far beyond the gap's own length, up to
# a cap that keeps the head and tail searches bounded.
RECOVER_SLACK_S = 3.0
GAP_ONE_SIDED_MAX_S = 120.0
# The path costs inside a gap. A few hundred offsets, not a quarter of a
# million, so the largest coincidence is under four noise units, a jump
# is cheap to earn and a run of nothing is called so after a few windows.
GAP_JUMP_COST = 8.0
GAP_NULL_REWARD = 1.0
GAP_NULL_COST = 4.0
# Mean score a stretch found in a gap must reach; equivalent to RECOVER_Z
# in the search this replaces.
RECOVER_Z = 6.0
RECOVER_ROUNDS = 2
# The effects band searching a gap on its own (see _search_gap).
EFFECTS_RECOVER_Z = 7.0
EFFECTS_RECOVER_WINDOWS = 3

# --- polishing, at the envelope's own resolution --------------------------

POLISH_WINDOW_S = 10.0
# The coarse grid is 20 ms and the path may sit a couple of steps off the
# peak, so the true offset is within this of the coarse one.
POLISH_RANGE_S = 0.12
# On a stretch's first measurement the search is wider: a stretch found
# by a long window may hold a cut of a second or two that the window
# straddled, and readings that can only look 120 ms either side of the
# stretch's one offset read nothing on the far side of such a cut, so the
# step is never seen and the stretch is measured as noise. With room to
# find the far side, the readings show the step and the stretch is split.
POLISH_WIDE_RANGE_S = 1.5
# Envelopes are standardised over this much context before being multiplied,
# so a loud passage cannot outvote a quiet one when the boundary is placed.
LOCAL_STATS_S = 2.0
# How precisely an edge can be placed depends on how strongly the two tracks
# agree: the agreement curve is a random walk with a drift of a fraction of
# the match either side of the edge, and the walk's noise beats that drift
# for about (K/r)^2 steps. That is the half-width of the interval an edge is
# reported with, and the interval the next pass searches. K is set
# generously: the envelope pass is only the first estimate, and the cost of
# a wide interval is a few more seconds of decoding, while the cost of a
# narrow one is a waveform pass boxed in short of the real cut. Measured on
# the synthetic pairs, the envelope pass at a match of 0.12 lands within
# 3.3 s; at K=6 the interval is 5 s.
EDGE_UNCERTAINTY_STEPS = 6.0

# --- sharpening edges on the waveform -------------------------------------

# The music and effects under a dub are usually the same stems as under the
# original, so the two waveforms share a component -- weakly, since each is
# mostly its own dialogue, but at 16 kHz there are thirty-two times as many
# samples per second as in the onset envelope, and an edge placed on the
# waveform is placed thirty-two times as finely for the same agreement.
# This is only attempted where the waveforms demonstrably agree: a peak in
# their correlation, within a few milliseconds of the envelope's offset,
# standing this many noise units above the rest.
SHARPEN_Z = 8.0
# How far either side of the envelope's offset the waveform peak is sought.
# The envelope is exact to a couple of milliseconds; this is the margin.
SHARPEN_SEARCH_S = 0.004
# How much of the stretch's interior is used to find that peak, and how
# much context the waveform is standardised over.
SHARPEN_INTERIOR_S = 4.0
SHARPEN_STATS_S = 0.1
# An edge already placed this precisely is not worth a decode.
SHARPEN_MIN_UNCERTAINTY_S = 0.02
# The waveform is searched across the whole bracket the coarse windows
# left, not the narrower interval the envelope placement claims: at a
# match of 0.12 the envelope's placement was seven seconds out while
# claiming five, and an edge searched for in the wrong interval is found
# at the interval's end. The decode is bounded all the same.
SHARPEN_MAX_SPAN_S = 60.0
# Two edges placed independently that leave less than this between them on
# the dub's timeline are treated as one cut the dub runs straight through,
# and placed together. Placing them together pools the evidence of both
# sides, so it is preferred whenever the independent placements do not
# clearly contradict it: they scatter by a few hundred milliseconds where
# the shared bed happens to be quiet at the cut, while a dub that genuinely
# skips material skips far more than this.
CONTINUOUS_DUB_S = 1.0
# A stretch is measured every this many seconds along its length, so a step
# in the offset is caught wherever it falls.
POLISH_SPACING_S = 7.5
# Inside one stretch the offset should hold still. A slope past this is a
# speed difference, and is reported rather than silently averaged.
DRIFT_WARN_MS_PER_S = 0.5

# --- assembling the plan --------------------------------------------------

# An edge placed no better than this is pulled in by its uncertainty before
# the plan is assembled, so the dub only plays where the evidence says the
# dub belongs and the original takes the doubt. Half a second of the
# original where the dub existed is a small loss; half a second of dub
# from the far side of a cut is the wrong scene in the right language,
# which is the one thing this whole module exists to avoid.
EDGE_TRIM_MIN_S = 0.25

# Stretches of dub shorter than this are dropped in favour of the original.
# A one-second island of dub between two fills is a splice the ear notices
# twice for no benefit.
MIN_BLOCK_S = 1.0
# Two offsets closer than this describe one stretch of dub, not two.
SAME_OFFSET_S = 0.005
# A gap shorter than this between two stretches of dub at the same offset is
# absorbed rather than filled: a 15 ms sliver of the original is a click.
MIN_FILL_S = 0.02
# Where the path found no match but the offset is the same on both sides and
# the dub is not silent there, the dub is kept. A weakly correlating scene is
# still the dub, and replacing it with the original because it correlated
# weakly would put the wrong language over a scene that had the right one.
# The dub is treated as silent below this fraction of its own level.
DUB_SILENT_RATIO = 0.03
# The original has something worth bringing in only above this fraction of
# its own level: room tone at -28 dB under a silent dub is not a scene the
# dub lacks, and filling it reports a cut that is not there.
FILL_AUDIBLE_RATIO = 0.1
# Two stretches whose offsets differ by no more than this are the same
# scene of the dub, conformed a frame or two apart: nothing is missing
# between them. A gap between them is bridged, not filled -- the worst a
# bridge can do is play a few seconds this far out of lip-sync, where a fill
# puts the other language over a scene the dub has -- and each side keeps
# its own offset. This is the offset at which lip-sync starts to show; a
# step larger than it may be a shot cut out, and is treated as a cut.
NEAR_OFFSET_S = 0.1
# Dub audible before the first stretch or after the last, at the offset the
# stretch continues from, is kept only this far: as far as the stretch is
# long, and never more than this. Inside the file a kept passage has the
# same offset on both sides for evidence; at the file's ends there is one
# side, and a seven-second stretch cannot vouch for four minutes -- on a
# dub made of episodes it vouched for another episode's ending.
EDGE_KEEP_MAX_S = 30.0
# A gap the dub is audible across, whose two sides sit up to this far
# apart, is bridged with the step placed inside it; so is any such gap
# where a wrongly placed step would misplace no more than
# BRIDGE_EXPOSURE_S of dub. Beyond both, the original fills it.
BRIDGE_STEP_S = 2.0
BRIDGE_EXPOSURE_S = 5.0
# How far the best step placement must stand above the median one, in
# seconds of the mean agreement, to count as measured rather than guessed.
STEP_EVIDENCE = 1.0
# How far the original is allowed to be re-levelled to sit among the dub.
MAX_FILL_GAIN_DB = 12.0
# The level difference is read in pieces this long across every stretch.
FILL_GAIN_PIECE_S = 60.0
# The note on a fill that replaced such a stretch, when asked to. Matched on
# by the app and the CLI report, so it is one string.
UNMATCHED_NOTE = "dub audible but did not correlate; replaced"

# --- choosing a speed -----------------------------------------------------

# A dub timed against a 23.976fps master and laid on a 24fps video runs
# 0.1% slow: a millisecond a second. The coarse pass hardly notices -- its
# windows are pooled to 20ms -- but the fine measurement inside a stretch
# is at 2ms, and ten seconds of drift inside one of its windows smears the
# correlation flat. Measured on a real pair, every stretch read 0.06-0.14
# uncompensated and 0.30-0.73 at the right speed, and the ends of every
# long stretch were a quarter of a second out. The drift is read from the
# coarse path itself: where a stretch's offset column walks steadily, the
# slope is the speed.
#
# Below this slope, or with less matched material than this to read it
# from, the drift is left alone: a 45-minute episode at 0.2 ms/s is 0.5s
# end to end, and the per-stretch measurement absorbs it.
MIN_DRIFT_MS_PER_S = 0.25
MIN_DRIFT_SPAN_S = 240.0
# A trial speed must raise the fine match by this factor to be believed.
SPEED_FINE_GAIN = 1.3
SPEED_ROUNDS = 2

# When the path leaves this fraction of the windows or more unplaced, or
# resting on nothing, the pair may be a rate conversion rather than a bad
# dub, and the standard speeds are tried before believing it.
SPEED_RETRY_FRACTION = 0.3

# --- the wide pass ----------------------------------------------------------

# When the credible stretches cover less of the video than this after the
# coarse pass and the speed trials, every window of the video is searched
# across the whole dub at the envelope's full 2 ms. It costs a minute or two
# on a feature and can only add stretches, so it runs whenever more than a
# tenth of the video is without dub. The coarse pass looks
# only within the offsets a dub of the same edit can reach, and at 20 to
# 160 ms: a dub made of episodes -- recaps, openings and endings inserted,
# the episodes in any order -- puts scenes at offsets far outside that, and
# its onsets are too sharp to survive the pooling. On such a pair the coarse
# pass matched 21 windows of 698 and left 93 minutes filled, while at 2 ms
# every window of those minutes matched at 8 to 58 noise units.
WIDE_PASS_COVERAGE = 0.9
WIDE_WINDOW_S = 30.0
WIDE_HOP_S = 15.0
# A window's best offset counts when this far above the noise across the
# whole dub: the search range is the file, so the bar is higher than the
# coarse pass's MATCH_Z.
WIDE_Z = 7.0
# Consecutive windows whose offsets agree to within this are one stretch;
# it takes three, because across a whole file two windows can agree on a
# coincidence when the music repeats, and a stretch found by the wide pass
# goes into a gap that would otherwise be filled with the original.
WIDE_SAME_OFFSET_S = 0.05
WIDE_MIN_WINDOWS = 3
# Peaks a wide window keeps for the choice described in _wide_blocks, and
# how far apart they must be to count as different occurrences.
WIDE_PEAKS = 4
WIDE_PEAK_APART_S = 3.0
# A candidate this close to a neighbouring coarse stretch's offset is the
# same scene continuing (a cut inside a scene is seconds, not minutes).
WIDE_LOCAL_S = 60.0
# How far a coarse stretch's offset reaches as the neighbourhood's.
WIDE_ANCHOR_REACH_S = 120.0
# A rewind smaller than this against the previous window is a repeat of a
# cue within the scene; a larger one is material out of order.
WIDE_REWIND_SLACK_S = 2.0
WIDE_EPISODE_S = 300.0
# A trial speed is believed when its peak stands this far above the noise
# and this many times above what the files managed at their own speed.
# Several speeds are tried, and each is a chance for a coincidence, so the
# bar is a ratio against the baseline rather than a level: on real decodes
# the right speed scores 6 to 17 times any wrong one (see analyze.py).
SPEED_MIN_Z = 6.0
SPEED_GAIN = 1.5
SPEED_TRIAL_WINDOWS = 5
# Trial windows are long. A true match's peak grows with the square root
# of the window while the largest coincidence does not, and at a wrong
# speed a longer window only smears further; two minutes turns a peak of
# 5.6 on the weakest synthetic pair into one of 11 against a floor of 4.4.
SPEED_TRIAL_WINDOW_S = 120.0
# Every standard conversion inside this band is tried. The durations cannot
# order the search here as they do for a whole pair -- a dub with scenes
# cut is shorter for reasons that have nothing to do with speed, and on a
# synthetic pair with a 30-second cut they pointed at 1.2x and the true
# 1.0427x was never reached -- and a trial costs a few windows, so all of
# them are cheap. Beyond eleven percent no frame-rate conversion a dub
# could have been through exists.
SPEED_BAND = 0.11

# --- silence inside the dub -----------------------------------------------

# The dub going quiet for this long while the video is not is a gap in the
# dub, not a pause in it: filled from the original.
MIN_SILENT_S = 1.0

# --- checking the result --------------------------------------------------

VERIFY_SPOTS = 12
VERIFY_SPOT_WINDOW_S = 30.0
VERIFY_SPOT_RANGE_S = 2.0
VERIFY_SWEEP_WINDOW_S = 10.0
VERIFY_SWEEP_POOL = 5  # 10 ms
VERIFY_SWEEP_RANGE_S = 1.0
# The sweep searches two hundred offsets, whose largest coincidence is
# about 3.3 noise units; a single window at 4.5 can still be a fluke, so a
# stretch has to be at least two windows long to be reported.
VERIFY_SWEEP_Z = 4.5
VERIFY_STRETCH_WINDOWS = 2
# Past this the drift is audible as lip-sync error.
AUDIBLE_MS = 100.0
STRETCH_MS = 200.0

ProgressFn = Callable[[int, str], None]
LogFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


@dataclass
class TrackEnvelope:
    """One track reduced to a 500 Hz frame-energy curve and its onsets.

    The energy is kept as well as the onsets because the onsets are derived
    from it and the derivation is cheap: a trial speed is a resample of the
    energy curve followed by that derivation, which is how speeds are tried
    without decoding the file again.
    """

    path: str
    track: int
    energy: np.ndarray
    """RMS of each ENVELOPE_HOP-sample frame, float32."""
    onset: np.ndarray
    """Half-wave-rectified rise in log energy, one shorter than ``energy``."""
    speed: float = 1.0
    """Playback-speed factor applied when decoding: the timeline these curves
    are on is the file's own time multiplied by this."""
    rate: int = ENVELOPE_RATE
    band_energy: Dict[str, np.ndarray] = field(default_factory=dict)
    """Frame energy of each band in BANDS, same frames as ``energy``."""
    band_onset: Dict[str, np.ndarray] = field(default_factory=dict)
    """Onsets of each band, derived like ``onset``."""

    @property
    def duration_s(self) -> float:
        return len(self.energy) / self.rate

    def onsets(self, band: str) -> np.ndarray:
        """The onset curve of one band; ``full`` is the whole spectrum."""
        return self.onset if band == "full" else self.band_onset[band]

    @property
    def bands(self) -> List[str]:
        """``full`` first, then every band this track carries."""
        return ["full"] + sorted(self.band_onset)

    @classmethod
    def from_energy(
        cls, path: str, track: int, energy: np.ndarray, speed: float = 1.0,
        band_energy: Optional[Dict[str, np.ndarray]] = None,
    ):
        bands = {k: v.astype(np.float32) for k, v in (band_energy or {}).items()}
        return cls(
            path, track, energy.astype(np.float32), _onsets(energy), speed,
            band_energy=bands, band_onset={k: _band_onsets(v) for k, v in bands.items()},
        )

    def at_speed(self, speed: float) -> "TrackEnvelope":
        """The same track as it would decode at another playback speed."""
        if abs(speed - self.speed) < 1e-9:
            return self
        factor = speed / self.speed
        n = int(len(self.energy) * factor)
        source = np.arange(n, dtype=np.float64) / factor
        energy = np.interp(source, np.arange(len(self.energy)), self.energy)
        bands = {
            k: np.interp(source, np.arange(len(v)), v) for k, v in self.band_energy.items()
        }
        return TrackEnvelope.from_energy(self.path, self.track, energy, speed, bands)


def _onsets(energy: np.ndarray) -> np.ndarray:
    """``correlate.onset_envelope``, from frame energies already computed."""
    if len(energy) < 2:
        return np.zeros(0, dtype=np.float32)
    log_energy = np.log1p(energy.astype(np.float64) * 1000.0)
    return np.maximum(0.0, np.diff(log_energy)).astype(np.float32)


# Where a band's typical frame lands on the onset curve's log knee: the
# same place a full-band frame of 0.01 RMS lands, which is what the knee in
# ``_onsets`` was set for.
BAND_KNEE = 10.0


def _band_onsets(energy: np.ndarray) -> np.ndarray:
    """``_onsets`` of a band, with the band's own level put on the knee.

    ``_onsets`` compresses with log1p(1000 * energy), a knee set for the
    full-band RMS of a film. A band carries far less -- the 4-8 kHz band a
    hundredth -- and below the knee the curve is not the log rise it is
    above it but the raw difference, a few tall spikes where the band is
    loudest and nothing between: correlated, that gives heavy-tailed noise
    and chance peaks of eight or nine floors. Scaling the band so its
    median frame sits at the knee makes its curve as dense as the full
    band's, and the noise floors comparable.
    """
    if len(energy) == 0:
        return np.zeros(0, dtype=np.float32)
    typical = float(np.median(energy))
    scale = BAND_KNEE / (1000.0 * max(typical, 1e-9))
    return _onsets(energy.astype(np.float64) * scale)


def build_envelope(
    path: str,
    track: int = 0,
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[float], None]] = None,
    expected_duration_s: Optional[float] = None,
    speed: float = 1.0,
) -> TrackEnvelope:
    """Decode a whole track once, keeping only its 500 Hz energy curve.

    Args:
        speed: decode at this playback speed. Asking ffmpeg for
            ANALYSIS_SR * speed and reading the samples as ANALYSIS_SR is a
            time-stretch by that factor, done by the decoder's resampler.
        progress: called with 0.0-1.0 as the file is read, when the duration
            is known.
    """
    rate = max(1, int(round(ANALYSIS_SR * speed)))
    hop = ENVELOPE_HOP
    energies: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    seen = 0
    bands = _BandFrames()
    for block in stream_audio(path, rate, track=track, token=token, block_s=30.0):
        if carry.size:
            block = np.concatenate([carry, block])
        frames = len(block) // hop
        if frames:
            shaped = block[: frames * hop].reshape(frames, hop).astype(np.float64)
            energies.append(np.sqrt(np.mean(shaped * shaped, axis=1) + 1e-12).astype(np.float32))
        bands.push(block[: frames * hop])
        carry = block[frames * hop:]
        seen += len(block) - len(carry)
        if progress and expected_duration_s:
            progress(min(1.0, seen / rate / expected_duration_s))
    if not energies:
        raise MediaError(f"No audio decoded from {os.path.basename(path)}")
    energy = np.concatenate(energies)
    return TrackEnvelope.from_energy(path, track, energy, speed, bands.finish(len(energy)))


class _BandFrames:
    """The band-limited frame energies of a stream, built block by block.

    A Hann window of BAND_WINDOW samples is centred on every ENVELOPE_HOP
    frame of the full-band envelope, so frame ``i`` of a band and frame
    ``i`` of the full band describe the same instant; the samples the
    window needs beyond the block are kept back until the next block
    arrives. Energy is the RMS of the band's part of the windowed frame,
    on the same scale as the full-band RMS.
    """

    def __init__(self) -> None:
        self.window = np.hanning(BAND_WINDOW).astype(np.float32)
        freqs = np.fft.rfftfreq(BAND_WINDOW, 1.0 / ANALYSIS_SR)
        self.masks = {name: (freqs >= lo) & (freqs < hi) for name, (lo, hi) in BANDS.items()}
        # Parseval for a one-sided spectrum of a Hann-windowed frame.
        self.scale = 2.0 / (BAND_WINDOW * float(np.sum(self.window ** 2)))
        self.lead = BAND_WINDOW // 2 - ENVELOPE_HOP // 2
        self.buffer = np.zeros(self.lead, dtype=np.float32)
        self.done = 0
        self.frames: Dict[str, List[np.ndarray]] = {name: [] for name in BANDS}

    def push(self, samples: np.ndarray) -> None:
        self.buffer = np.concatenate([self.buffer, samples.astype(np.float32)])
        count = (len(self.buffer) - BAND_WINDOW) // ENVELOPE_HOP + 1
        if count <= 0:
            return
        shaped = np.lib.stride_tricks.as_strided(
            self.buffer, shape=(count, BAND_WINDOW),
            strides=(self.buffer.strides[0] * ENVELOPE_HOP, self.buffer.strides[0]),
        )
        power = np.abs(np.fft.rfft(shaped * self.window, axis=1)) ** 2
        for name, mask in self.masks.items():
            self.frames[name].append(np.sqrt(power[:, mask].sum(axis=1) * self.scale + 1e-12).astype(np.float32))
        self.buffer = self.buffer[count * ENVELOPE_HOP:].copy()
        self.done += count

    def finish(self, frames: int) -> Dict[str, np.ndarray]:
        """The band energies, padded or cut to ``frames`` frames."""
        self.push(np.zeros(BAND_WINDOW, dtype=np.float32))
        out = {}
        for name, parts in self.frames.items():
            curve = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if len(curve) < frames:
                curve = np.concatenate([curve, np.full(frames - len(curve), curve[-1] if len(curve) else 0.0, dtype=np.float32)])
            out[name] = curve[:frames]
        return out


def _pool(values: np.ndarray, factor: int) -> np.ndarray:
    """Max over consecutive groups, so a transient keeps its height."""
    if factor <= 1:
        return values.astype(np.float64)
    n = len(values) // factor
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return values[: n * factor].reshape(n, factor).max(axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def ncc_lags(template: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``template`` against every equal span of ``signal``.

    Returns ``len(signal) - len(template) + 1`` values, one per start
    position. Every span of the signal is normalised on its own, so a loud
    passage cannot win by being loud and a silent one scores zero rather than
    dividing by nothing. The numerator is an FFT correlation; the
    denominators come from running sums, so the whole thing costs one
    transform however many positions are tried.
    """
    template = np.asarray(template, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    width, length = len(template), len(signal)
    if width == 0 or length < width:
        return np.zeros(0, dtype=np.float64)

    centred = template - template.mean()
    template_norm = float(np.sqrt(np.sum(centred * centred)))
    positions = length - width + 1
    if template_norm < 1e-9:
        return np.zeros(positions, dtype=np.float64)

    size = _fast_fft_size(length)
    spectrum = np.fft.rfft(signal, size) * np.conj(np.fft.rfft(centred, size))
    numerator = np.fft.irfft(spectrum, size)[:positions]

    ones = np.concatenate([[0.0], np.cumsum(signal)])
    squares = np.concatenate([[0.0], np.cumsum(signal * signal)])
    sums = ones[width:] - ones[:-width]
    sums_sq = squares[width:] - squares[:-width]
    variance = np.maximum(sums_sq - sums * sums / width, 0.0)
    denominator = template_norm * np.sqrt(variance)

    out = np.zeros(positions, dtype=np.float64)
    ok = denominator > 1e-9
    out[ok] = numerator[ok] / denominator[ok]
    return np.clip(out, -1.0, 1.0)


def _parabolic(values: np.ndarray, index: int) -> float:
    """Sub-sample peak position from the three values around a maximum."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, centre, right = float(values[index - 1]), float(values[index]), float(values[index + 1])
    denom = left - 2.0 * centre + right
    if abs(denom) < 1e-12:
        return float(index)
    shift = 0.5 * (left - right) / denom
    if not math.isfinite(shift) or abs(shift) > 1.0:
        return float(index)
    return index + shift


def _nearest_peak(row: np.ndarray, centre: int, share: float = 0.85, apart: int = 50) -> int:
    """The index of the peak nearest ``centre`` among the local maxima that
    reach ``share`` of the tallest and sit at least ``apart`` samples from
    it. A rhythmic passage has a peak every beat, all about as tall, and
    the tallest is not the right one -- the one nearest where the stretch
    already sits is. A bump on the tallest peak's own shoulder is not
    another peak, hence ``apart`` (100 ms at 2 ms frames)."""
    if len(row) < 3:
        return int(np.argmax(row))
    tallest = int(np.argmax(row))
    top = float(row[tallest])
    inner = row[1:-1]
    maxima = np.flatnonzero((inner >= share * top) & (inner >= row[:-2]) & (inner >= row[2:])) + 1
    maxima = maxima[np.abs(maxima - tallest) >= apart]
    if maxima.size == 0:
        return tallest
    candidates = np.concatenate([[tallest], maxima])
    return int(candidates[np.argmin(np.abs(candidates - centre))])


def _tallest_peaks(row: np.ndarray, count: int, apart: int) -> List[int]:
    """Indices of the ``count`` tallest values of ``row`` that are at least
    ``apart`` samples from each other, tallest first."""
    out: List[int] = []
    taken = np.zeros(len(row), dtype=bool)
    work = np.array(row, dtype=np.float64)
    for _ in range(count):
        work[taken] = -np.inf
        peak = int(np.argmax(work))
        if not np.isfinite(work[peak]):
            break
        out.append(peak)
        taken[max(0, peak - apart) : peak + apart + 1] = True
    return out


def _continuing_candidate(
    candidates: Sequence[Tuple[float, float]],
    position_s: float,
    anchors: Sequence[Block],
    previous: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Which of a window's credible peaks to keep (see ``_wide_blocks``)."""
    if len(candidates) == 1:
        return candidates[0]
    nearby = [
        b for b in anchors
        if b.start_s - WIDE_ANCHOR_REACH_S <= position_s <= b.end_s + WIDE_ANCHOR_REACH_S
    ]
    if nearby:
        anchor = min(nearby, key=lambda b: max(0.0, b.start_s - position_s, position_s - b.end_s))
        local = [c for c in candidates if abs(c[0] - anchor.offset_s) <= WIDE_LOCAL_S]
        if local:
            return max(local, key=lambda c: c[1])
    if previous is not None:
        prev_position, prev_offset = previous
        # Where the dub was at the previous window's end; a candidate that
        # winds back from there by less than an episode is a repeat.
        floor = prev_position + prev_offset + (position_s - prev_position) - WIDE_REWIND_SLACK_S
        forward = [c for c in candidates if position_s + c[0] >= floor or position_s + c[0] < floor - WIDE_EPISODE_S]
        if forward:
            return max(forward, key=lambda c: c[1])
    return max(candidates, key=lambda c: c[1])


def _peak_dominance(row: np.ndarray, peak: int, apart: int = 50) -> float:
    """How many times taller the peak is than the tallest other value more
    than ``apart`` samples from it. A match stands alone; a chance
    coincidence in a sparse band has company of its own size."""
    if len(row) < 3:
        return 0.0
    away = np.abs(np.arange(len(row)) - peak) > apart
    if not away.any():
        return float("inf")
    other = float(np.max(row[away]))
    return float(row[peak]) / max(other, 1e-9)


def _noise_units(row: np.ndarray, possible: np.ndarray) -> np.ndarray:
    """Express one window's correlations as multiples of its noise floor.

    The floor is the root mean square over every offset that could be
    tried, less the tallest one percent (and always the tallest one): the
    peak being measured must not be part of its own floor, which on a
    short row it would be. For a row of dense onsets that is the standard
    deviation the median magnitude used to estimate; for a row of sparse
    ones -- a band with a few transients in it, a quiet scene -- the
    median sits near zero and made a chance coincidence of two or three
    spikes look ten floors tall, while the RMS still counts the spikes.
    Music with a repeating structure raises it, and that is the right
    direction, since every repeat is a chance for a coincidence.
    """
    out = np.full(len(row), -1e6, dtype=np.float32)
    if not possible.any():
        return out
    values = row[possible]
    magnitudes = np.abs(values).astype(np.float64)
    if len(magnitudes) > 2:
        drop = max(1, len(magnitudes) // 100)
        magnitudes = np.partition(magnitudes, len(magnitudes) - drop)[: len(magnitudes) - drop]
    sigma = float(np.sqrt(np.mean(magnitudes ** 2)))
    if sigma < 1e-6:
        out[possible] = 0.0
    else:
        out[possible] = (values / sigma).astype(np.float32)
    return out


@dataclass
class ScoreGrid:
    """Correlation of every window against every candidate offset."""

    rate: float
    """Samples per second of the pooled envelopes the grid was built on."""
    starts: np.ndarray
    """Window start, in pooled samples of the primary."""
    window: int
    """Window length in pooled samples."""
    lag_lo: int
    """Offset of column 0, in pooled samples (dub = primary + lag)."""
    scores: np.ndarray
    """Windows x lags, in noise units; hugely negative where impossible."""

    def offset_s(self, column: int) -> float:
        return (self.lag_lo + column) / self.rate

    def start_s(self, index: int) -> float:
        return float(self.starts[index]) / self.rate

    @property
    def window_s(self) -> float:
        return self.window / self.rate


def score_windows(
    primary: np.ndarray,
    secondary: np.ndarray,
    rate: float,
    starts: Sequence[int],
    window: int,
    lag_lo: int,
    lag_hi: int,
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> ScoreGrid:
    """Score every window of the primary at every offset in [lag_lo, lag_hi].

    Offsets that would place any part of the window outside the secondary
    are impossible, not merely unmeasured, and are marked so the path cannot
    rest on them: a stretch of video past the end of the dub must be a gap.
    """
    lags = lag_hi - lag_lo + 1
    grid = np.full((len(starts), lags), -1e6, dtype=np.float32)
    # A window may hang a tenth of its length past either end of the
    # secondary and still be measured, on what it does cover: otherwise the
    # last window before the dub's end is impossible at the true offset by
    # a rounding error, and the stretch it belongs to loses its last edge.
    pad = window // 10
    padded = np.concatenate([np.zeros(pad), np.asarray(secondary, dtype=np.float64), np.zeros(pad)])
    for index, start in enumerate(starts):
        if token:
            token.raise_if_cancelled()
        template = primary[start : start + window]
        if len(template) < window:
            continue
        first = max(0, start + lag_lo + pad)
        last = min(len(padded), start + window + lag_hi + pad)
        if last - first < window:
            continue
        row = ncc_lags(template, padded[first:last])
        column = (first - pad) - start - lag_lo
        possible = np.zeros(lags, dtype=bool)
        possible[column : column + len(row)] = True
        full = np.zeros(lags, dtype=np.float64)
        full[column : column + len(row)] = row
        grid[index] = _noise_units(full, possible)
        if progress:
            progress((index + 1) / len(starts))
    return ScoreGrid(rate, np.asarray(starts), window, lag_lo, grid)


# ---------------------------------------------------------------------------
# The path
# ---------------------------------------------------------------------------

STAY_LEFT, STAY, STAY_RIGHT, JUMPED, FROM_NULL = 0, 1, 2, 3, 4


def best_path(
    scores: np.ndarray,
    jump_cost: float = JUMP_COST,
    null_reward: float = NULL_REWARD,
    null_cost: float = NULL_COST,
    wobble_cost: float = WOBBLE_COST,
    first: Optional[np.ndarray] = None,
    last: Optional[np.ndarray] = None,
) -> Tuple[List[Optional[int]], List[bool]]:
    """The offset at every window that best explains the whole matrix.

    Dynamic programming over (window, offset), plus one extra state per
    window meaning "no offset applies". Staying put is free; moving one
    column costs a little; any larger move costs ``jump_cost``; the null
    state earns ``null_reward`` a window and costs ``null_cost`` to enter or
    leave. Exact, not greedy: every window's answer accounts for every other
    window.

    Args:
        first, last: optional bonuses added to the first and last windows,
            for a pass that has to begin and end at known offsets.

    Returns:
        The column chosen at each window (None for the null state), and
        whether each window began a new stretch -- reached by a jump, from
        the null state, or as the first window.
    """
    count, lags = scores.shape
    if count == 0:
        return [], []
    back = np.zeros((count, lags), dtype=np.uint8)
    back_null = np.zeros(count, dtype=np.uint8)
    best_previous = np.zeros(count, dtype=np.int64)

    value = scores[0].astype(np.float64)
    if first is not None:
        value = value + first
    if last is not None and count == 1:
        value = value + last
    value_null = null_reward

    for index in range(1, count):
        argmax = int(np.argmax(value))
        best = float(value[argmax])
        best_previous[index] = argmax

        stay = value.copy()
        code = np.full(lags, STAY, dtype=np.uint8)
        if lags > 1:
            from_left = np.empty(lags)
            from_left[0] = -np.inf
            from_left[1:] = value[:-1] - wobble_cost
            better = from_left > stay
            stay[better] = from_left[better]
            code[better] = STAY_LEFT
            from_right = np.empty(lags)
            from_right[-1] = -np.inf
            from_right[:-1] = value[1:] - wobble_cost
            better = from_right > stay
            stay[better] = from_right[better]
            code[better] = STAY_RIGHT

        jumped = best - jump_cost
        from_null = value_null - null_cost
        if from_null > jumped:
            move, move_code = from_null, FROM_NULL
        else:
            move, move_code = jumped, JUMPED
        better = move > stay
        stay[better] = move
        code[better] = move_code

        new_value = scores[index] + stay
        if index == count - 1 and last is not None:
            new_value = new_value + last

        entered = best - null_cost
        if entered > value_null:
            value_null = null_reward + entered
            back_null[index] = 1
        else:
            value_null = null_reward + value_null
            back_null[index] = 0

        value = new_value
        back[index] = code

    state: Optional[int] = None if value_null > float(value.max()) else int(np.argmax(value))
    path: List[Optional[int]] = [None] * count
    starts = [False] * count
    for index in range(count - 1, -1, -1):
        path[index] = state
        if index == 0:
            starts[0] = state is not None
            break
        if state is None:
            state = best_previous[index] if back_null[index] else None
            continue
        code = back[index, state]
        if code == STAY_LEFT:
            state -= 1
        elif code == STAY_RIGHT:
            state += 1
        elif code == JUMPED:
            starts[index] = True
            state = int(best_previous[index])
        elif code == FROM_NULL:
            starts[index] = True
            state = None
    return path, starts


@dataclass
class Block:
    """One stretch of video that one stretch of dub belongs to."""

    start_s: float
    end_s: float
    offset_s: float
    """dub time = video time + offset, on the dub's decoded timeline."""
    match: float = 0.0
    """Correlation of the two envelopes across the stretch, 0-1."""
    support: float = 0.0
    """Mean score of the windows that put it here, in noise units."""
    windows: int = 0
    start_lo: float = 0.0
    start_hi: float = 0.0
    end_lo: float = 0.0
    end_hi: float = 0.0
    """Where each edge may lie, before it is placed exactly; afterwards, the
    placed edge plus and minus its uncertainty."""
    consistency: float = 0.0
    """Fraction of the sub-window readings that agree with the offset to
    within CONSISTENT_MS."""
    readings: int = 0
    spread_ms: float = float("inf")
    """Range of the readings' offsets."""
    effects: bool = False
    """Whether the offset was read from the high band (see BANDS)."""
    trace: List[Tuple[float, float]] = field(default_factory=list)
    """(window start, offset) for every coarse window with evidence for
    this stretch: the path's own record of where the offset sat, which is
    what a speed difference is read from."""

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def start_uncertainty_s(self) -> float:
        return (self.start_hi - self.start_lo) / 2.0

    @property
    def end_uncertainty_s(self) -> float:
        return (self.end_hi - self.end_lo) / 2.0


def _block_bar(windows: int, lags: int, overlap: float = COARSE_WINDOW_S / COARSE_HOP_S) -> float:
    """Mean score a run of this many windows must reach to be believed.

    The path is free to pick any of ``lags`` offsets, and to wander a step
    between windows, so a run of k windows at a random offset is not
    typical noise but the best of a great many tries. Its mean reaches
    about sqrt(2 ln(lags * 3^k) / k) -- were the windows independent.
    They are not: at a ten-second hop each thirty-second window shares two
    thirds of its material with the next, so k windows carry a third as
    many independent readings, and that is the k the bound is taken at.
    Measured on a two-hour synthetic pair, a run of seven coasting windows
    reached 3.45; this puts its bar at 4.4, and a real stretch's windows
    average five or more.
    """
    if windows <= 0:
        return float("inf")
    if windows <= SHORT_RUN_WINDOWS:
        # A run this short has to match outright. A real stretch this
        # short is found by the gap search instead, which looks for it
        # where the neighbours say it must be and cannot be fooled the
        # same way.
        return MATCH_Z
    effective = max(1.0, windows / max(1.0, overlap))
    tries = math.log(max(2, lags)) + effective * math.log(3.0)
    expected_max = math.sqrt(2.0 * tries / effective)
    return max(MIN_BLOCK_Z, BLOCK_BAR_MARGIN * expected_max)


def blocks_from_path(grid: ScoreGrid, path: List[Optional[int]], starts: List[bool]) -> List[Block]:
    """Group the path into stretches, one per run of windows at one offset."""
    blocks: List[Block] = []
    run: List[int] = []

    def close() -> None:
        if not run:
            return
        columns = np.array([path[i] for i in run], dtype=np.float64)
        hop_s = grid.start_s(1) - grid.start_s(0) if len(grid.starts) > 1 else grid.window_s
        # Staying at an offset is free, so past a cut the path coasts on
        # windows with no evidence at all until the next offset's evidence
        # begins. The stretch really ends somewhere in the last window that
        # had evidence for it, and begins somewhere in the first; the edges
        # are bracketed by those windows, not by where the path turned.
        #
        # Two grades of evidence, because a coasting window reaches two
        # noise units by chance every forty windows or so, and a bracket
        # that trusted one of those would exclude the real cut. A window
        # that matches outright must hold a good share of the stretch, so
        # the cut cannot be before it starts; a window with a little
        # evidence might hold a sliver of it, so the cut might be as late
        # as its end.
        weak = [i for i in run if grid.scores[i, path[i]] >= MIN_BLOCK_Z] or [run[0], run[-1]]
        strong = [i for i in run if grid.scores[i, path[i]] >= MATCH_Z] or weak
        # The stretch's support is the mean over the windows that matched
        # outright, and its window count theirs: the windows the path
        # coasted over carry no evidence either way, and averaged in they
        # hid a stretch two windows of eight and five noise units had
        # found. The bar a run must clear already scales with its length.
        support = float(np.mean([grid.scores[i, path[i]] for i in strong]))
        # The stretch itself is taken as the span of its strong windows:
        # what it is measured on. Its true edges lie further out, inside
        # the brackets, and are placed there afterwards.
        blocks.append(Block(
            start_s=grid.start_s(strong[0]),
            end_s=grid.start_s(strong[-1]) + grid.window_s,
            offset_s=grid.offset_s(float(np.median(columns))),
            support=support,
            windows=len(strong),
            start_lo=max(0.0, grid.start_s(weak[0]) - hop_s),
            start_hi=grid.start_s(strong[0]) + grid.window_s,
            end_lo=grid.start_s(strong[-1]),
            end_hi=grid.start_s(weak[-1]) + grid.window_s + hop_s,
            trace=[(grid.start_s(i), grid.offset_s(path[i])) for i in weak],
        ))
        run.clear()

    for index, column in enumerate(path):
        if column is None:
            close()
            continue
        if starts[index] and run:
            close()
        run.append(index)
    close()
    return blocks


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

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "startS": self.start_s,
            "endS": self.end_s,
            "sourceStartS": self.source_start_s,
            "offsetS": self.offset_s,
            "match": self.match,
            "note": self.note,
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
    segments: List[Segment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    """Things the reader should check: a fill, a drift, a stretch replaced."""
    notes: List[str] = field(default_factory=list)
    """Things the reader may like to know, that need no checking: the dub
    kept across a span it did not correlate in."""
    error: Optional[str] = None

    @property
    def dub_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.kind == "dub"]

    @property
    def fill_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.kind == "fill"]

    @property
    def filled_s(self) -> float:
        return sum(s.length_s for s in self.fill_segments)

    def to_dict(self) -> dict:
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
            "segments": [s.to_dict() for s in self.segments],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "error": self.error,
            "filledS": self.filled_s,
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
            segments=[Segment.from_dict(s) for s in data.get("segments", [])],
            warnings=list(data.get("warnings") or []),
            notes=list(data.get("notes") or []),
            error=data.get("error"),
        )

    def describe(self) -> str:
        """The plan as a table, one line per piece of the output."""
        lines = []
        dubs, fills = self.dub_segments, self.fill_segments
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
        elif abs(self.speed - 1.0) > 1e-9:
            head += f", dub played at {self.speed:.6f}x"
        if fills:
            head += f", fills at {self.fill_gain_db:+.1f} dB"
        lines.append(head)
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
        return "\n".join(lines)


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600.0)
    minutes, rest = divmod(rest, 60.0)
    return f"{int(hours)}:{int(minutes):02d}:{rest:06.3f}"


def _duration(seconds: float) -> str:
    if seconds >= 60.0:
        minutes, rest = divmod(seconds, 60.0)
        return f"{int(minutes)}m {rest:04.1f}s"
    return f"{seconds:.1f}s"


def _local_standardize(values: np.ndarray, span: int) -> np.ndarray:
    """Zero-mean, unit-variance against a sliding neighbourhood."""
    values = values.astype(np.float64)
    n = len(values)
    if n == 0:
        return values
    half = max(1, span // 2)
    padded = np.concatenate([np.full(half, values[0]), values, np.full(half, values[-1])])
    ones = np.concatenate([[0.0], np.cumsum(padded)])
    squares = np.concatenate([[0.0], np.cumsum(padded * padded)])
    width = 2 * half + 1
    sums = ones[width:] - ones[:-width]
    sums_sq = squares[width:] - squares[:-width]
    mean = sums[:n] / width
    variance = np.maximum(sums_sq[:n] / width - mean * mean, 0.0)
    std = np.sqrt(variance)
    out = np.zeros(n)
    ok = std > 1e-6
    out[ok] = (values[ok] - mean[ok]) / std[ok]
    return out


# Two levels are told apart when their centres sit further apart than both
# a floor of two envelope frames and LEVEL_SEPARATION times the wider of the
# two levels' own scatter. The scatter is what makes it safe: readings on a
# weakly correlating pair wander a few milliseconds and must not be split on,
# while readings on a strongly correlating pair land within a fifth of a
# millisecond and steps of a few milliseconds between them are real. With a
# fixed 20 ms gap, a track whose pieces sat at 7, 18 and 32 ms -- hundreds
# of readings at each -- was played at one offset, with an hour of it 11 ms
# out. A level needs LEVEL_MIN_READINGS readings to be believed, and a run
# of readings at a level needs RUN_MIN_READINGS in a row to be a scene
# rather than a reading that locked onto a repeat.
MIN_LEVEL_GAP_MS = 4.0
LEVEL_SEPARATION = 5.0
LEVEL_MIN_READINGS = 2
RUN_MIN_READINGS = 2
# The trend taken off before clustering is measured between readings at
# least this many apart, or a tenth of the stretch, whichever is more.
DRIFT_NEIGHBOURS = 3
# A step between two pieces is believed when the shorter piece, taken whole,
# agrees with the dub this many times better at its own offset than at its
# neighbour's. Ten-second readings on a weakly correlating scene can lock
# onto a secondary peak three times running; the piece those readings cut
# out was played 38 ms out, and its whole span agreed two and a half times
# better at the neighbour's offset.
STEP_GAIN = 1.5


def _level_runs(positions: Sequence[float], offsets_ms: Sequence[float]) -> List[Tuple[float, float, float]]:
    """Runs of consecutive readings at one offset level, in order.

    The offsets are clustered into levels that are separated by more than
    their own scatter allows (see LEVEL_SEPARATION); a level with fewer
    than LEVEL_MIN_READINGS is noise and its readings are ignored. Each
    remaining reading is labelled with its level, and consecutive readings
    at the same level form a run. Returns (first position, last position,
    level) per run of at least RUN_MIN_READINGS, or an empty list when the
    readings sit on one level or on none -- in which case there is nothing
    to split on, and the single-step test decides.

    The clustering is done on the offsets with their trend taken off. A
    dub at the wrong speed reads as a ramp, and a ramp sorted and broken
    at gaps is a staircase: on a real pair it came back as six "steps" of
    fifty milliseconds a minute apart, which shredded the stretch the
    speed was about to be read from. A ramp is not steps; it is left whole
    and reported as drift. Steps on top of a ramp still show as gaps once
    the ramp is removed. A staircase of flat treads is steps, however
    short the treads: every reading on a tread agrees with its neighbours
    to the reading, which a ramp's never do.
    """
    positions = np.asarray(positions, dtype=np.float64)
    offsets = np.asarray(offsets_ms, dtype=np.float64)

    def cluster(values: np.ndarray) -> List[np.ndarray]:
        """Indices grouped by level, each group in ascending order of value.

        Bottom up: the sorted readings are broken wherever two neighbours
        are more than MIN_LEVEL_GAP_MS apart, then adjacent groups whose
        centres are not separated (see LEVEL_SEPARATION) are merged, the
        least separated pair first, until every adjacent pair is. Top down
        -- splitting at the widest gap whose sides are separated -- went
        wrong on three levels: a side holding two levels has the scatter of
        both, so the true valleys failed the test, and a spurious split
        inside the middle level passed it.
        """

        def scatter(members: np.ndarray) -> float:
            picked = values[members]
            return 1.4826 * float(np.median(np.abs(picked - np.median(picked))))

        def closeness(left: np.ndarray, right: np.ndarray) -> float:
            """Centre distance over the distance that would separate them;
            under one, they are one level."""
            apart = float(np.median(values[right]) - np.median(values[left]))
            return apart / max(MIN_LEVEL_GAP_MS, LEVEL_SEPARATION * max(scatter(left), scatter(right)))

        order = np.argsort(values, kind="stable")
        groups: List[np.ndarray] = []
        cut = 0
        for k in range(1, len(order) + 1):
            if k == len(order) or values[order[k]] - values[order[k - 1]] > MIN_LEVEL_GAP_MS:
                groups.append(order[cut:k])
                cut = k
        while len(groups) > 1:
            ratios = [closeness(groups[k], groups[k + 1]) for k in range(len(groups) - 1)]
            k = int(np.argmin(ratios))
            if ratios[k] >= 1.0:
                break
            groups[k : k + 2] = [np.concatenate([groups[k], groups[k + 1]])]
        return groups

    # The trend is the median slope between readings up to a tenth of the
    # stretch apart. Between all pairs, the pairs that straddle a step vote
    # for a slope, and on a staircase of three long levels they outnumber
    # the pairs within a level: the trend they elected smeared each level
    # into a ramp and no step survived. Between nearest neighbours only,
    # the slope of a noisy pair is too rough -- a few thousandths of a
    # millisecond per second, which is thirty milliseconds of tilt across a
    # feature. Pairs up to a tenth of the stretch apart straddle a step
    # rarely and average that roughness away.
    slope = 0.0
    if len(positions) >= 3 and float(np.ptp(positions)) > 1.0:
        reach = max(DRIFT_NEIGHBOURS, len(positions) // 10)
        pairs = [
            (offsets[j] - offsets[i]) / (positions[j] - positions[i])
            for i in range(len(positions))
            for j in range(i + 1, min(len(positions), i + 1 + reach))
            if positions[j] != positions[i]
        ]
        slope = float(np.median(pairs)) if pairs else 0.0
    detrended = offsets - slope * (positions - positions[0])
    clusters = cluster(detrended)
    labels: List[Optional[int]] = [None] * len(detrended)
    levels = 0
    for members in clusters:
        if len(members) < LEVEL_MIN_READINGS:
            continue
        for index in members:
            labels[int(index)] = levels
        levels += 1
    if levels < 2:
        return []

    runs: List[Tuple[int, int, int]] = []  # first index, last index, level
    for index, which in enumerate(labels):
        if which is None:
            continue
        if runs and runs[-1][2] == which:
            runs[-1] = (runs[-1][0], index, which)
        else:
            runs.append((index, index, which))
    # Drop runs too short to be a scene, then merge runs at one level that
    # a dropped run separated.
    kept: List[Tuple[int, int, int]] = []
    for first, last, which in runs:
        members = sum(1 for k in range(first, last + 1) if labels[k] == which)
        if members < RUN_MIN_READINGS:
            continue
        if kept and kept[-1][2] == which:
            kept[-1] = (kept[-1][0], last, which)
        else:
            kept.append((first, last, which))
    if len(kept) < 2:
        return []
    # Each run's level is the median of its own readings as measured, so a
    # piece carries the offset its material actually sits at.
    return [
        (float(positions[first]), float(positions[last]),
         float(np.median([offsets[k] for k in range(first, last + 1) if labels[k] == which])))
        for first, last, which in kept
    ]


class _Aligner:
    """The analysis, held together so its stages can share the envelopes."""

    def __init__(
        self,
        primary: TrackEnvelope,
        secondary: TrackEnvelope,
        token: Optional[CancellationToken],
        log: Optional[LogFn],
        frame_s: Optional[float] = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.token = token
        self.log = log or (lambda _message: None)
        self.frame_s = frame_s
        """One video frame, when the video's rate is known: the unit a
        picture trim comes in, which tells a trim from a music artefact."""
        self._standardized: dict = {}
        self._pooled: dict = {}

    @property
    def shared_bands(self) -> List[str]:
        """The bands both envelopes carry, ``full`` first."""
        return [b for b in self.primary.bands if b in self.secondary.bands]

    @property
    def detect_bands(self) -> List[str]:
        """The bands a stretch may be *found* by: the full band and the low
        band. The high band is bursts with silence between, whose chance
        correlations are heavy-tailed (peaks of eight or nine floors where
        nothing matches); it arbitrates the offset of a stretch already
        found (see ``_read``), it does not vouch for one."""
        return [b for b in self.shared_bands if b != "high"]

    def onsets(self, which: str, band: str = "full") -> np.ndarray:
        env = self.primary if which == "primary" else self.secondary
        return env.onsets(band)

    def pooled(self, which: str, pool: int, band: str = "full") -> np.ndarray:
        key = (which, pool, band)
        if key not in self._pooled:
            self._pooled[key] = _pool(self.onsets(which, band), pool)
        return self._pooled[key]

    # -- coarse -------------------------------------------------------------

    def coarse(self, search_s: float, progress: Optional[Callable[[float], None]] = None) -> ScoreGrid:
        primary_s = self.primary.duration_s
        secondary_s = self.secondary.duration_s
        # The dub can only sit as far behind as the material it lacks, and
        # as far ahead as the material it has extra, plus the slack. Made
        # asymmetric so a dub ten minutes short does not also search ten
        # minutes in the direction it cannot possibly go.
        lo_s = -(max(0.0, primary_s - secondary_s) + search_s)
        hi_s = max(0.0, secondary_s - primary_s) + search_s

        pool = COARSE_POOL
        while True:
            rate = ENVELOPE_RATE / pool
            window = int(round(COARSE_WINDOW_S * rate))
            hop = max(1, int(round(COARSE_HOP_S * rate)))
            count = max(1, int((primary_s * rate - window) // hop) + 1)
            lags = int(round((hi_s - lo_s) * rate)) + 1
            if count * lags <= COARSE_CELL_BUDGET or pool >= 8 * COARSE_POOL:
                break
            pool *= 2

        self.log(
            f"coarse pass: {count} windows of {COARSE_WINDOW_S:.0f}s every "
            f"{COARSE_HOP_S:.0f}s, offsets {lo_s:+.0f}s to {hi_s:+.0f}s in "
            f"{1000.0 / rate:.0f}ms steps"
        )
        primary = self.pooled("primary", pool)
        secondary = self.pooled("secondary", pool)
        starts = [i * hop for i in range(count)]
        return score_windows(
            primary, secondary, rate, starts, window,
            int(round(lo_s * rate)), int(round(hi_s * rate)),
            self.token, progress,
        )

    # -- polish -------------------------------------------------------------

    def standardized_onsets(self, which: str, band: str = "full") -> np.ndarray:
        key = (which, band)
        if key not in self._standardized:
            self._standardized[key] = _local_standardize(
                self.onsets(which, band), int(LOCAL_STATS_S * ENVELOPE_RATE)
            )
        return self._standardized[key]

    def agreement(
        self, offset_s: float, lo_s: float, hi_s: float, bands: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """How much the two envelopes agree at each 2 ms of this span, at one
        offset: the product of the two standardised onset curves, summed
        over the bands (every shared band unless ``bands`` says which).
        Each band's product averages to its correlation where the tracks
        match and to zero where they do not, so a scene that only the low
        band can see still shows its agreement, and a scene every band
        sees shows more."""
        lo = max(0, int(round(lo_s * ENVELOPE_RATE)))
        hi = min(len(self.primary.onset), int(round(hi_s * ENVELOPE_RATE)))
        if hi <= lo:
            return np.zeros(0)
        shift = int(round(offset_s * ENVELOPE_RATE))
        out = np.zeros(hi - lo)
        for band in (bands if bands is not None else self.shared_bands):
            primary = self.standardized_onsets("primary", band)
            secondary = self.standardized_onsets("secondary", band)
            s_lo, s_hi = lo + shift, hi + shift
            c_lo, c_hi = max(0, s_lo), min(len(secondary), s_hi)
            if c_hi > c_lo:
                out[c_lo - s_lo : c_hi - s_lo] += primary[c_lo - shift : c_hi - shift] * secondary[c_lo:c_hi]
        return out

    def edge_curve(self, offset_s: float, lo_s: float, hi_s: float) -> np.ndarray:
        """The agreement an edge is placed on: the full band's where it has
        any, every band's summed where it has none.

        The full band's transients are two frames wide, so its agreement
        turns at a cut within a frame or two; a band's are twenty frames
        wide, and a band that matches far better than the others -- the
        effects band on a bed the mixes share exactly -- dominates the
        sum, so that its own passing dips read as the edge. Where the
        full band is blind, a scene with no music in it, the sum is what
        there is."""
        full = self.agreement(offset_s, lo_s, hi_s, bands=["full"])
        if full.size and float(full.mean()) >= AGREEMENT_FLOOR:
            return full
        return self.agreement(offset_s, lo_s, hi_s)

    def readings(self, block: Block, range_s: float = POLISH_RANGE_S) -> List[Tuple[float, float, float]]:
        """The offset measured at 2 ms at several places inside a stretch.

        Returns (position_s, offset_s, match) per sub-window, one every
        POLISH_SPACING_S along the stretch and never fewer than three where
        the stretch is long enough for three. Each looks ``range_s`` either
        side of the stretch's offset.
        """
        rate = ENVELOPE_RATE
        window = int(POLISH_WINDOW_S * rate)
        margin = int(round(range_s * rate))
        block_lo = int(round(block.start_s * rate))
        block_hi = int(round(block.end_s * rate))
        length = block_hi - block_lo
        if length < rate:  # under a second: nothing to measure
            return []
        window = min(window, length)
        count = max(1, min(length // window, max(3, int(block.length_s / POLISH_SPACING_S))))
        if count == 1:
            starts = [block_lo + (length - window) // 2]
        else:
            step = (length - window) / (count - 1)
            starts = [block_lo + int(round(i * step)) for i in range(count)]

        shift = int(round(block.offset_s * rate))
        found: List[Tuple[float, float, float, str, float]] = []
        for start in starts:
            reading = self._read(start, window, shift, margin)
            if reading is not None:
                found.append((start / rate, *reading))
        return found

    def _read(
        self, start: int, window: int, shift: int, margin: int
    ) -> Optional[Tuple[float, float, str, float]]:
        """One window's offset, as (offset_s, match, band, noise units).

        Every band is correlated and the one that peaks highest above its
        own noise answers -- except that a confident high band (the
        effects, see BANDS) answers regardless, looked for over the wider
        EFFECTS_RANGE_S so a trim the music ran across is still seen.
        Among the effects' peaks within 15% of the tallest, the one nearest
        the stretch's own offset is taken: a beat makes peaks every period,
        all about as tall, and only the nearest is the stretch's.
        """
        rate = ENVELOPE_RATE
        best: Optional[Tuple[float, float, str, float]] = None
        for band in self.shared_bands:
            primary = self.onsets("primary", band)
            secondary = self.onsets("secondary", band)
            template = primary[start : start + window]
            effects = band == "high"
            reach = max(margin, int(round(EFFECTS_RANGE_S * rate))) if effects else margin
            # A search range that runs off an end of the dub is cut to the
            # dub, not skipped: skipped, a stretch at the very start of the
            # file had no reading at all and was dropped as noise.
            first = max(0, start + shift - reach)
            last = min(len(secondary), start + window + shift + reach)
            if last - first < window or len(template) < window:
                continue
            row = ncc_lags(template, secondary[first:last])
            if row.size < 3:
                continue
            units = _noise_units(row, np.ones(len(row), dtype=bool))
            peak = _nearest_peak(row, start + shift - first) if effects else int(np.argmax(row))
            z = float(units[peak])
            lag = (first + _parabolic(row, peak)) - start
            if effects:
                # The effects decide outright when their peak is tall and
                # alone: a band of bursts correlates by chance in several
                # places at once, and none of those is a reading.
                if z >= EFFECTS_Z and _peak_dominance(row, peak) >= EFFECTS_DOMINANCE:
                    return (lag / rate, float(row[peak]), band, z)
                # Otherwise it competes like any band, discounted: its
                # chance peaks run a third taller than the full band's.
                z *= EFFECTS_DISCOUNT
            candidate = (lag / rate, float(row[peak]), band, z)
            if best is None or z > best[3]:
                best = candidate
        return best

    def polish_offset(self, block: Block, wide: bool = False) -> Tuple[float, float, Optional[float]]:
        """Measure a stretch's offset at 2 ms, from several places inside it.

        Returns (offset_s, match, drift_ms_per_s). The offset is the median
        across the sub-windows, weighted by how well each matched, so a
        sub-window that landed on silence does not pull the answer. The
        drift is the slope of offset against position when there are enough
        readings to fit one. ``wide`` searches POLISH_WIDE_RANGE_S either
        side instead of POLISH_RANGE_S.
        """
        found = self.readings(block, POLISH_WIDE_RANGE_S if wide else POLISH_RANGE_S)
        block.readings = len(found)
        block.consistency = 0.0
        if not found:
            return block.offset_s, 0.0, None
        effects = [f for f in found if f[3] == "high" and f[4] >= EFFECTS_Z]
        block.effects = bool(effects)
        if effects:
            # The effects are the picture's timing; where they were read
            # they decide the offset, and the other readings only say
            # whether the stretch is one level (consistency) or not.
            found = effects + [f for f in found if f not in effects]
            weights_scale = [1.0 if f in effects else 1e-3 for f in found]
        else:
            weights_scale = [1.0] * len(found)
        positions = np.array([f[0] for f in found])
        lags = np.array([f[1] for f in found])
        weights = np.maximum(np.array([f[2] for f in found]), 1e-3) * np.array(weights_scale)
        order = np.argsort(lags)
        cumulative = np.cumsum(weights[order])
        median = lags[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))]
        deciding = lags[: len(effects)] if effects else lags
        block.consistency = float(np.mean(np.abs(deciding - median) * 1000.0 <= CONSISTENT_MS))
        block.spread_ms = float(np.ptp(deciding) * 1000.0)
        drift = None
        if len(found) >= 3 and float(np.ptp(positions)) > 1.0:
            drift = float(np.polyfit(positions, lags * 1000.0, 1)[0])
        return float(median), float(np.median([f[2] for f in found])), drift

    def best_agreement(self, offset_s: float, lo_s: float, hi_s: float) -> float:
        """Mean agreement across the span at the offset, taking the best of
        the frame either side: the offset is measured to a fraction of a
        frame and the curve is only known at whole ones."""
        best = 0.0
        for frame in (-1, 0, 1):
            curve = self.agreement(offset_s + frame / ENVELOPE_RATE, lo_s, hi_s)
            if curve.size:
                best = max(best, float(curve.mean()))
        return best

    def sharp_match(self, block: Block) -> float:
        """The stretch's median match on the full band alone, from the
        same sub-windows ``readings`` uses (see ``_fine_score``)."""
        rate = ENVELOPE_RATE
        window = int(POLISH_WINDOW_S * rate)
        margin = int(round(POLISH_RANGE_S * rate))
        block_lo = int(round(block.start_s * rate))
        block_hi = int(round(block.end_s * rate))
        length = block_hi - block_lo
        if length < rate:
            return 0.0
        window = min(window, length)
        count = max(1, min(length // window, max(3, int(block.length_s / POLISH_SPACING_S))))
        starts = [block_lo + (length - window) // 2] if count == 1 else [
            block_lo + int(round(i * (length - window) / (count - 1))) for i in range(count)
        ]
        shift = int(round(block.offset_s * rate))
        primary, secondary = self.onsets("primary"), self.onsets("secondary")
        matches = []
        for start in starts:
            template = primary[start : start + window]
            first = max(0, start + shift - margin)
            last = min(len(secondary), start + window + shift + margin)
            if last - first < window or len(template) < window:
                continue
            row = ncc_lags(template, secondary[first:last])
            if row.size:
                matches.append(float(row.max()))
        return float(np.median(matches)) if matches else 0.0

    def credibility(self, block: Block) -> float:
        """How far the stretch's own correlation peak stands above the
        noise of the offsets around it, in the band that sees it best.

        Taken over up to CREDIBILITY_SPAN_S of the stretch's middle and a
        second of offsets either side. A short stretch can correlate well
        by chance in a sparse band -- a few coincident spikes -- and its
        match alone does not say so; the peak's height against the other
        offsets does.
        """
        rate = ENVELOPE_RATE
        length = min(block.length_s, CREDIBILITY_SPAN_S)
        if length < 0.5:
            return 0.0
        middle = (block.start_s + block.end_s) / 2.0
        # The span is kept inside both files: past either end of the dub
        # there is nothing to correlate, and a stretch at the end of the
        # film is measured on what it does cover.
        primary_s = len(self.primary.onset) / rate
        secondary_s = len(self.secondary.onset) / rate
        lo_s = max(0.0, middle - length / 2.0, -block.offset_s)
        hi_s = min(primary_s, middle + length / 2.0, secondary_s - block.offset_s)
        if hi_s - lo_s < 0.5:
            return 0.0
        start = int(round(lo_s * rate))
        window = int(round((hi_s - lo_s) * rate))
        shift = int(round(block.offset_s * rate))
        reach = int(round(CREDIBILITY_RANGE_S * rate))
        best = 0.0
        for band in self.detect_bands:
            primary = self.onsets("primary", band)
            secondary = self.onsets("secondary", band)
            template = primary[start : start + window]
            first = max(0, start + shift - reach)
            last = min(len(secondary), start + window + shift + reach)
            if len(template) < window or last - first < window + 10 or template.std() < 1e-9:
                continue
            row = ncc_lags(template, secondary[first:last])
            if row.size < 10:
                continue
            centre = start + shift - first
            peak = _nearest_peak(row, centre)
            away = np.abs(np.arange(len(row)) - peak) > int(0.25 * rate)
            if away.sum() < 10:
                continue
            noise = float(np.sqrt(np.mean(row[away] ** 2)))
            best = max(best, float(row[peak]) / max(noise, 1e-9))
        return best

    def step_is_real(self, before: Block, after: Block) -> bool:
        """Whether two adjacent pieces really sit at different offsets.

        The shorter piece is taken whole and asked which offset it agrees
        with better, its own or its neighbour's. Its own readings put it
        where it is, but readings are ten-second windows and can agree on
        a wrong peak; the span as a whole cannot.
        """
        short, long = (before, after) if before.length_s <= after.length_s else (after, before)
        own = self.best_agreement(short.offset_s, short.start_s, short.end_s)
        other = self.best_agreement(long.offset_s, short.start_s, short.end_s)
        return own > 0.0 and own >= STEP_GAIN * other

    def split_at_step(self, block: Block, wide: bool = False) -> Optional[List[Block]]:
        """Split a stretch where its own readings say the offset stepped.

        The same question ``segments`` answers for a whole file, asked of
        one stretch: do these readings sit on one level or two? A step
        between two readings puts the cut somewhere between them; the exact
        placement comes afterwards, with the bracket handed over so the
        search covers the whole of that span.
        """
        readings = self.readings(block, POLISH_WIDE_RANGE_S if wide else POLISH_RANGE_S)
        if len(readings) < 4:
            return None
        # Only readings that found something: a sub-window on silence reads
        # a random lag, and one of those is not a step. Judged by how far
        # the reading's peak stands above its row, not by the match: the
        # bands' matches are not on one scale (a broad band peaks higher),
        # and half the median of a mixed set threw out real readings.
        found = [f for f in readings if f[4] >= READING_Z]
        if len(found) < 4:
            return None
        positions = [f[0] for f in found]
        offsets_ms = [f[1] * 1000.0 for f in found]

        # Several levels, visited more than once: a dub conformed scene by
        # scene, with some scenes a few frames out, reads as runs of
        # readings alternating between two offsets. One step cannot
        # describe that, and asked for one step the fit refuses, so the
        # whole stretch would be played at one offset with a third of it a
        # tenth of a second late. Group the readings by level and split at
        # every change of run instead -- after the effects have had their
        # say about which levels are real.
        runs = _level_runs(positions, offsets_ms)
        runs, offsets_ms = self._effects_rule(found, runs, offsets_ms)
        if len(runs) >= 2:
            return self._split_at_runs(block, runs)

        step = find_step(positions, offsets_ms, POLISH_WINDOW_S)
        if step is None or step.lone_group:
            # One reading on its own side of a step is what a reading that
            # locked onto a repeat looks like; it needs company to be a cut.
            return None
        lo_s = max(block.start_s, step.earliest_s - 1.0)
        hi_s = min(block.end_s, step.latest_s + 1.0)
        head = Block(block.start_s, step.split_position_s, step.before_ms / 1000.0,
                     support=block.support, windows=block.windows,
                     start_lo=block.start_lo, start_hi=block.start_hi, end_lo=lo_s, end_hi=hi_s)
        tail = Block(step.split_position_s, block.end_s, step.after_ms / 1000.0,
                     support=block.support, windows=block.windows,
                     start_lo=lo_s, start_hi=hi_s, end_lo=block.end_lo, end_hi=block.end_hi)
        head.offset_s, head.match, _ = self.polish_offset(head)
        tail.offset_s, tail.match, _ = self.polish_offset(tail)
        if not self.step_is_real(head, tail):
            return None
        _place_edge(self, head, tail, self.primary.duration_s, self.secondary.duration_s)
        if head.length_s < MIN_BLOCK_S or tail.length_s < MIN_BLOCK_S:
            return None
        self.log(
            f"  a {step.magnitude_ms:+.0f}ms step inside {_clock(block.start_s)} - {_clock(block.end_s)}"
            f" at {_clock(head.end_s)}"
        )
        return [head, tail]

    def _effects_rule(
        self,
        found: List[Tuple[float, float, float, str, float]],
        runs: List[Tuple[float, float, float]],
        offsets_ms: List[float],
    ) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """Fold the levels the effects contradict into the effects' level.

        Where any run of readings is backed by the high band (see BANDS),
        a run that is not is the music's level, and the music is not the
        picture's timing: its readings are moved to the nearest
        effects-backed level. Two effects-backed levels that are not a
        whole number of frames apart cannot both be picture trims, so the
        one with less behind it moves to the other. Runs that end up at one
        level are joined. Returns the runs and the readings' offsets as
        relabelled, so the single-step test that follows sees the same
        picture.
        """
        if len(runs) < 2:
            return runs, offsets_ms

        def members(run: Tuple[float, float, float]) -> List[int]:
            first_s, last_s, level = run
            out = []
            for index, f in enumerate(found):
                if first_s - 1e-6 <= f[0] <= last_s + 1e-6:
                    nearest = min(runs, key=lambda r: abs(r[2] - offsets_ms[index]))
                    if nearest[2] == level:
                        out.append(index)
            return out

        backing = []
        for run in runs:
            effects = [i for i in members(run) if found[i][3] == "high" and found[i][4] >= EFFECTS_Z]
            backing.append((len(effects), sum(found[i][4] for i in effects)))

        # An excursion: a run a good fraction of a second from the level
        # either side of it, over in under a minute, with the level the
        # same before and after. A dub does not gain a second and lose
        # the same second again eighteen seconds later; a cue that repeats
        # at that interval reads exactly so. Unless the effects put it
        # there, it is folded into the level around it. (A step of a
        # frame or two that comes back is a scene conformed a frame out,
        # and is real; the size tells the two apart.)
        levels = [run[2] for run in runs]
        folded = False
        for index in range(1, len(runs) - 1):
            before_level, after_level = levels[index - 1], levels[index + 1]
            size = abs(levels[index] - before_level)
            if (
                abs(before_level - after_level) <= MIN_LEVEL_GAP_MS
                and size >= EXCURSION_MIN_MS
                and runs[index + 1][0] - runs[index][0] <= EXCURSION_MAX_S
                and not backing[index][0]
            ):
                self.log(
                    f"  a {size:+.0f}ms excursion at {_clock(runs[index][0])} returns to "
                    f"{before_level / 1000.0:+.3f}s within {runs[index + 1][0] - runs[index][0]:.0f}s: a repeat, not a cut"
                )
                levels[index] = before_level
                folded = True
        if folded:
            relabelled = list(offsets_ms)
            for index, run in enumerate(runs):
                for i in members(run):
                    relabelled[i] = levels[index]
            joined: List[Tuple[float, float, float]] = []
            for run, level in zip(runs, levels):
                if joined and abs(joined[-1][2] - level) <= 1e-9:
                    joined[-1] = (joined[-1][0], run[1], level)
                else:
                    joined.append((run[0], run[1], level))
            runs, offsets_ms = (joined if len(joined) >= 2 else []), relabelled
            if len(runs) < 2:
                return runs, offsets_ms
            backing = []
            for run in runs:
                effects = [i for i in members(run) if found[i][3] == "high" and found[i][4] >= EFFECTS_Z]
                backing.append((len(effects), sum(found[i][4] for i in effects)))
        if not any(count for count, _ in backing):
            return runs, offsets_ms

        levels = [run[2] for run in runs]
        strong = [index for index, (count, _) in enumerate(backing) if count]
        moved = False
        for index, run in enumerate(runs):
            if backing[index][0]:
                continue
            target = min(strong, key=lambda j: abs(levels[j] - levels[index]))
            if abs(levels[target] - levels[index]) <= MIN_LEVEL_GAP_MS:
                levels[index] = levels[target]
                continue
            # The run had no confident effects reading of its own. Asked
            # on the run's windows, do the effects sit at the run's level
            # (a real step every band agrees on) or at the effects run's
            # level (the music ran across a trim the effects followed)?
            positions = [found[i][0] for i in members(run)]
            if not self._effects_prefer(positions, levels[target], levels[index]):
                continue
            self.log(
                f"  the music sits at {levels[index] / 1000.0:+.3f}s around {_clock(run[0])}; the effects say "
                f"{levels[target] / 1000.0:+.3f}s, and the dialogue goes with the effects"
            )
            levels[index] = levels[target]
            moved = True
        if self.frame_s:
            frame_ms = 1000.0 * self.frame_s
            for index in strong:
                for other in strong:
                    if other <= index or levels[index] == levels[other]:
                        continue
                    gap = abs(levels[index] - levels[other])
                    if abs(gap / frame_ms - round(gap / frame_ms)) * frame_ms > FRAME_TOLERANCE_MS:
                        weak, keep = (index, other) if backing[index] < backing[other] else (other, index)
                        self.log(
                            f"  the effects read {levels[weak] / 1000.0:+.3f}s around {_clock(runs[weak][0])}, "
                            f"{gap / frame_ms:.2f} frames from {levels[keep] / 1000.0:+.3f}s: not a trim, folded"
                        )
                        levels[weak] = levels[keep]
                        moved = True
        if not moved and len(set(levels)) == len(levels):
            return runs, offsets_ms
        relabelled = list(offsets_ms)
        for index, run in enumerate(runs):
            for i in members(run):
                relabelled[i] = levels[index]
        joined: List[Tuple[float, float, float]] = []
        for run, level in zip(runs, levels):
            if joined and abs(joined[-1][2] - level) <= 1e-9:
                joined[-1] = (joined[-1][0], run[1], level)
            else:
                joined.append((run[0], run[1], level))
        return (joined if len(joined) >= 2 else []), relabelled

    def _effects_prefer(self, positions: Sequence[float], level_ms: float, other_ms: float) -> bool:
        """Whether the high band, read on windows at ``positions``, agrees
        better at ``level_ms`` than at ``other_ms`` (a twentieth better,
        so a band that sees nothing at either says nothing)."""
        if "high" not in self.shared_bands or not positions:
            return False
        rate = ENVELOPE_RATE
        window = int(POLISH_WINDOW_S * rate)
        slack = int(round(EFFECTS_VOTE_SLACK_S * rate))
        primary = self.onsets("primary", "high")
        secondary = self.onsets("secondary", "high")

        def agreement_at(offset_ms: float) -> float:
            shift = int(round(offset_ms / 1000.0 * rate))
            total = 0.0
            for position in positions:
                start = int(round(position * rate))
                template = primary[start : start + window]
                first, last = start + shift - slack, start + window + shift + slack
                if len(template) < window or first < 0 or last > len(secondary) or template.std() < 1e-9:
                    continue
                row = ncc_lags(template, secondary[first:last])
                if row.size:
                    total += float(row.max())
            return total

        return agreement_at(level_ms) > 1.05 * agreement_at(other_ms)

    def _split_at_runs(self, block: Block, runs: List[Tuple[float, float, float]]) -> Optional[List[Block]]:
        """Cut a stretch into one piece per run of readings at one level.

        Each piece takes the level as its offset and the run's span as its
        extent; the boundary between two pieces lies somewhere between the
        last reading of one run and the first of the next, and is placed
        there exactly afterwards.
        """
        pieces: List[Block] = []
        for index, (first_s, last_s, level_ms) in enumerate(runs):
            start_s = block.start_s if index == 0 else (runs[index - 1][1] + POLISH_WINDOW_S + first_s) / 2.0
            end_s = block.end_s if index == len(runs) - 1 else (last_s + POLISH_WINDOW_S + runs[index + 1][0]) / 2.0
            piece = Block(
                start_s, end_s, level_ms / 1000.0,
                support=block.support, windows=block.windows,
                start_lo=block.start_lo if index == 0 else max(block.start_s, runs[index - 1][1]),
                start_hi=block.start_hi if index == 0 else min(block.end_s, first_s + POLISH_WINDOW_S),
                end_lo=block.end_lo if index == len(runs) - 1 else max(block.start_s, last_s),
                end_hi=block.end_hi if index == len(runs) - 1 else min(block.end_s, runs[index + 1][0] + POLISH_WINDOW_S),
            )
            piece.offset_s, piece.match, _ = self.polish_offset(piece)
            # Measured on its own material, a piece may turn out to sit at
            # the same offset as the one before it -- the level it was
            # grouped by was a reading or two off. That is not a step, and
            # splitting there would put a splice, and a sliver of the
            # original, into an unbroken scene. Nor is a step the shorter
            # piece does not bear out when taken whole; the two are then
            # one piece, measured again as one.
            if pieces and (
                abs(piece.offset_s - pieces[-1].offset_s) * 1000.0 <= CONSISTENT_MS
                or not self.step_is_real(pieces[-1], piece)
            ):
                joined = pieces[-1]
                joined.end_s, joined.end_lo, joined.end_hi = piece.end_s, piece.end_lo, piece.end_hi
                joined.offset_s, joined.match, _ = self.polish_offset(joined)
                continue
            pieces.append(piece)
        if len(pieces) < 2:
            return None
        for index in range(1, len(pieces)):
            _place_edge(self, pieces[index - 1], pieces[index], self.primary.duration_s, self.secondary.duration_s)
        pieces = [piece for piece in pieces if piece.length_s >= MIN_BLOCK_S]
        if len(pieces) < 2:
            return None
        steps = ", ".join(f"{1000.0 * (b.offset_s - a.offset_s):+.0f}ms at {_clock(a.end_s)}" for a, b in zip(pieces, pieces[1:]))
        self.log(f"  the offset steps inside {_clock(block.start_s)} - {_clock(block.end_s)}: {steps}")
        return pieces

    # -- waveform ----------------------------------------------------------

    def waveform(self, which: str, lo_s: float, hi_s: float) -> Optional[np.ndarray]:
        """Decode a span of one track at ANALYSIS_SR, on its analysis timeline.

        For the dub that timeline is its own time times the speed it was
        analysed at, so the seek and the length are divided back out and the
        decoder asked for the compensating rate -- the same trick the
        envelope was built with.
        """
        env = self.primary if which == "primary" else self.secondary
        if hi_s <= lo_s:
            return None
        # Past either end of the file there is silence, which is what a
        # stretch that runs off the end of the dub has to be measured
        # against; so the span is clipped to the file and padded back out.
        clip_lo, clip_hi = max(0.0, lo_s), min(env.duration_s, hi_s)
        want = int(round((hi_s - lo_s) * ANALYSIS_SR))
        if clip_hi - clip_lo < 0.05:
            return np.zeros(want)
        try:
            pcm = load_audio(
                env.path, ANALYSIS_SR, duration=(clip_hi - clip_lo) / env.speed,
                offset=clip_lo / env.speed, token=self.token, track=env.track,
            ).astype(np.float64)
        except MediaError:
            return None
        if abs(env.speed - 1.0) > 1e-12:
            # Stretched here, by the exact factor, rather than by asking the
            # decoder for a compensating rate: that rate has to be a whole
            # number, and rounding it is a drift of a sample every few
            # seconds -- nothing to the envelope, but the whitened waveform
            # decorrelates within two samples, so a span forty seconds long
            # had lost its agreement by the far end.
            n = int(round(len(pcm) * env.speed))
            pcm = np.interp(np.arange(n) / env.speed, np.arange(len(pcm)), pcm)
        lead = int(round((clip_lo - lo_s) * ANALYSIS_SR))
        out = np.zeros(want)
        take = min(len(pcm), want - lead)
        if take > 0:
            out[lead : lead + take] = pcm[:take]
        return out

    def waveform_agreement(
        self, block: Block, lo_s: float, hi_s: float, inside: Tuple[float, float]
    ) -> Optional[Tuple[np.ndarray, float]]:
        """The waveform-level agreement of a stretch across [lo_s, hi_s).

        Returns the per-sample agreement and the waveform correlation it was
        judged by, or None when the two waveforms do not demonstrably share
        anything at this offset -- in which case the envelope's placement is
        the only one there is.

        The sample-exact offset, the polarity and the correlation are taken
        on the few seconds of ``inside`` -- the part of the span known to
        belong to the stretch -- where the envelopes agree most. They are
        taken on the same decode as the span itself: a seek through a lossy
        codec does not always land with the same slop, and a lag measured
        on one decode applied to another can be a few samples out, which
        for whitened signals is the difference between agreement and none.
        """
        primary = self.waveform("primary", lo_s, hi_s)
        secondary = self.waveform("secondary", lo_s + block.offset_s, hi_s + block.offset_s)
        if primary is None or secondary is None:
            return None
        n = min(len(primary), len(secondary))
        # Whitened by a first difference. Audio is not white: its power
        # sits low, so neighbouring samples agree with each other and a
        # second of it has far fewer independent samples than its sample
        # rate says. Flattening the spectrum restores them. Measured on the
        # synthetic pair, the shared bed's correlation doubles and the
        # correlation between unrelated passages drops by two thirds.
        primary = np.diff(primary[:n])
        secondary = np.diff(secondary[:n])
        n -= 1

        # The interior: the SHARPEN_INTERIOR_S window inside `inside` where
        # the envelopes agree most, so it is material the dub really has.
        in_lo, in_hi = max(lo_s, inside[0]), min(hi_s, inside[1])
        length = min(SHARPEN_INTERIOR_S, in_hi - in_lo)
        if length < 1.0:
            return None
        # The full band chooses the interior: the waveform is whitened, and
        # what it can lock onto is what the full band's sharp transients
        # see, not what a band's broad ones do.
        curve = self.agreement(block.offset_s, in_lo, in_hi, bands=["full"])
        window = max(1, int(length * ENVELOPE_RATE))
        if curve.size < window:
            return None
        totals = np.concatenate([[0.0], np.cumsum(curve)])
        means = (totals[window:] - totals[:-window]) / window
        best = int(np.argmax(means))
        if means[best] < AGREEMENT_FLOOR:
            return None
        interior = (in_lo + best / ENVELOPE_RATE, in_lo + best / ENVELOPE_RATE + length)

        i0 = max(0, int(round((interior[0] - lo_s) * ANALYSIS_SR)))
        i1 = min(n, int(round((interior[1] - lo_s) * ANALYSIS_SR)))
        if i1 - i0 < ANALYSIS_SR:
            return None
        a = primary[i0:i1] - primary[i0:i1].mean()
        b = secondary[i0:i1] - secondary[i0:i1].mean()
        reach = int(SHARPEN_SEARCH_S * ANALYSIS_SR)
        a_norm = float(np.sqrt(np.dot(a, a)))
        if a_norm < 1e-9:
            return None
        lags = np.arange(-reach, reach + 1)
        corr = np.zeros(len(lags))
        for index, lag in enumerate(lags):
            aa = a[max(0, lag) : len(a) + min(0, lag)]
            bb = b[max(0, -lag) : len(b) + min(0, -lag)]
            b_norm = float(np.sqrt(np.dot(bb, bb)))
            if b_norm > 1e-9:
                corr[index] = float(np.dot(aa, bb)) / (a_norm * b_norm)
        peak = int(np.argmax(np.abs(corr)))
        z = float(_noise_units(np.abs(corr), np.ones(len(corr), dtype=bool))[peak])
        if z < SHARPEN_Z:
            return None
        polarity = 1.0 if corr[peak] >= 0 else -1.0
        lag = int(lags[peak])
        # Shift the dub by the lag found, so the two line up to the sample.
        # corr[lag] pairs primary[i + lag] with secondary[i], so a positive
        # lag means the dub runs early by that much and is moved later.
        # Whitened signals decorrelate within a sample, so this has to be
        # exact: the original code shifted the other way, which turned a
        # one-sample misalignment into two and flattened the agreement.
        if lag > 0:
            secondary = np.concatenate([np.zeros(lag), secondary[:-lag]])
        elif lag < 0:
            secondary = np.concatenate([secondary[-lag:], np.zeros(-lag)])
        span = int(SHARPEN_STATS_S * ANALYSIS_SR)
        agreement = polarity * _local_standardize(primary, span) * _local_standardize(secondary, span)
        return agreement, abs(float(corr[peak]))

# Below this level of agreement there is nothing to place an edge on.
AGREEMENT_FLOOR = 0.02
# How many times the edge and the levels either side of it are re-estimated
# from each other. The second pass already agrees with the third to a few
# samples; the third is insurance.
EDGE_FIT_PASSES = 3
# Where between the two levels the threshold sits: this far up from the
# level outside the stretch towards the level inside it. Halfway would be
# the textbook choice if the level inside held steady, but it does not --
# it drops wherever dialogue masks the shared bed -- and the two mistakes
# are not equally bad. Ending a stretch early plays half a second of the
# original where the dub existed; ending it late plays half a second of dub
# from the wrong scene. Sitting nearer the outside level keeps a weakly
# agreeing passage in the dub, at the cost of letting a noise excursion
# carry the edge a few milliseconds past a cut.
EDGE_FIT_POSITION = 0.25


def _agreement_threshold(match: float) -> float:
    """Where to start: a quarter of the match, not half.

    The level of agreement varies along a stretch with what is playing --
    dialogue over quiet ambience masks the shared bed, an action scene does
    not -- so a threshold taken from one place can sit above the agreement
    somewhere else and read a matched passage as unmatched. Starting low
    errs towards keeping the dub; the fit then moves the threshold to the
    midpoint of the levels it actually finds either side of the edge.
    """
    return max(AGREEMENT_FLOOR, EDGE_FIT_POSITION * match)


def _fit_end(curve: np.ndarray, threshold: float, lo: int, hi: int) -> int:
    """Index in [lo, hi] where a stretch's agreement ends, by a two-level fit.

    The cumulative sum of (agreement - threshold) peaks where the level
    drops from "matched" to "not". The threshold that separates the two
    best sits between the two levels, which are only known once the edge
    is; so the two are estimated from each other in turn. The curve may
    run beyond [lo, hi] -- the sums need their run-up -- but the answer is
    confined to it, which is what keeps an edge already placed precisely
    from being moved by a later, coarser pass.
    """
    n = len(curve)
    lo, hi = max(0, min(lo, n)), max(0, min(hi, n))
    if n == 0 or hi < lo:
        return lo
    index = hi
    for _ in range(EDGE_FIT_PASSES):
        total = np.concatenate([[0.0], np.cumsum(curve - threshold)])
        index = lo + int(np.argmax(total[lo : hi + 1]))
        if index <= 0 or index >= n:
            break
        inside = float(curve[:index].mean())
        outside = max(0.0, float(curve[index:].mean()))
        threshold = max(AGREEMENT_FLOOR, outside + EDGE_FIT_POSITION * (inside - outside))
    return index


def _fit_start(curve: np.ndarray, threshold: float, lo: int, hi: int) -> int:
    """Index in [lo, hi] where a stretch's agreement begins; ``_fit_end`` mirrored."""
    n = len(curve)
    if n == 0:
        return lo
    return n - _fit_end(curve[::-1], threshold, n - hi, n - lo)


def _fit_joint(
    before: np.ndarray, after: np.ndarray,
    threshold_before: float, threshold_after: float,
    gap: int, lo: int, hi: int,
) -> int:
    """One cut serving both stretches: the first ends at the index returned,
    within [lo, hi], and the second begins ``gap`` samples later. Levels are
    re-estimated as in ``_fit_end``, each stretch on its own side."""
    n = min(len(before), len(after))
    lo, hi = max(0, lo), min(hi, n - gap)
    if gap >= n or n == 0 or hi < lo:
        return lo
    index = hi
    for _ in range(EDGE_FIT_PASSES):
        rise = np.concatenate([[0.0], np.cumsum(before[:n] - threshold_before)])
        fall = np.concatenate([[0.0], np.cumsum((after[:n] - threshold_after)[::-1])])[::-1]
        combined = rise[: n + 1 - gap] + fall[gap:]
        index = lo + int(np.argmax(combined[lo : hi + 1]))
        if index <= 0 or index + gap >= n:
            break
        inside_b = float(before[:index].mean())
        outside_b = max(0.0, float(before[index:n].mean()))
        threshold_before = max(AGREEMENT_FLOOR, outside_b + EDGE_FIT_POSITION * (inside_b - outside_b))
        inside_a = float(after[index + gap : n].mean())
        outside_a = max(0.0, float(after[:index + gap].mean()))
        threshold_after = max(AGREEMENT_FLOOR, outside_a + EDGE_FIT_POSITION * (inside_a - outside_a))
    return index


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
    return _is_quiet(secondary, lo_s, hi_s, reference_rms, DUB_SILENT_RATIO)


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


def plan_dubsync(
    video_path: str,
    dub_path: str,
    video_track: int = 0,
    dub_track: int = 0,
    search_s: float = DEFAULT_SEARCH_S,
    speed: Optional[float] = None,
    fill_gain_db: Optional[float] = None,
    keep_unmatched_dub: bool = True,
    token: Optional[CancellationToken] = None,
    progress: Optional[ProgressFn] = None,
    log: Optional[LogFn] = None,
    envelopes: Optional[dict] = None,
) -> DubSyncPlan:
    """Work out which stretch of the dub belongs at every moment of the video.

    Args:
        search_s: how far beyond the duration difference to look for the dub.
        speed: playback-speed factor to decode the dub at; None tries the
            standard frame-rate conversions if the files will not correlate
            as they are.
        fill_gain_db: gain for the original where it fills a gap; None
            measures the level difference between the two tracks.
        keep_unmatched_dub: keep the dub across a stretch that did not
            correlate when the offset is unchanged either side of it and the
            dub is not silent there. Off, such a stretch is filled from the
            original.
        envelopes: a dict to leave the decoded envelopes in, keyed
            ``primary`` and ``secondary``, so a verification pass need not
            decode the original again.
    """
    plan = DubSyncPlan(video_path, dub_path, video_track, dub_track)
    say = log or (lambda _m: None)

    def report(percent: float, stage: str) -> None:
        if progress:
            progress(int(max(0, min(100, percent))), stage)

    try:
        report(0, "probing")
        video_info = probe(video_path, token)
        dub_info = probe(dub_path, token)
        if not video_info.has_audio:
            plan.error = f"{os.path.basename(video_path)} has no audio to compare against"
            return plan
        if not dub_info.has_audio:
            plan.error = f"{os.path.basename(dub_path)} has no audio"
            return plan
        plan.video_duration_s = float(video_info.duration or 0.0)
        plan.dub_duration_s = float(dub_info.duration or 0.0)

        report(2, "reading the original")
        primary = build_envelope(
            video_path, video_track, token,
            lambda f: report(2 + 18 * f, "reading the original"),
            plan.video_duration_s,
        )
        report(20, "reading the dub")
        secondary_native = build_envelope(
            dub_path, dub_track, token,
            lambda f: report(20 + 18 * f, "reading the dub"),
            plan.dub_duration_s,
        )
        # The decoded length is the truth; the probe was only a progress hint.
        plan.video_duration_s = primary.duration_s
        plan.dub_duration_s = secondary_native.duration_s
        if envelopes is not None:
            envelopes["primary"] = primary
            envelopes["secondary"] = secondary_native
        say(
            f"original {_clock(primary.duration_s)}, dub {_clock(secondary_native.duration_s)}: "
            f"dub is {_duration(abs(primary.duration_s - secondary_native.duration_s))} "
            f"{'shorter' if secondary_native.duration_s < primary.duration_s else 'longer'}"
        )

        secondary = secondary_native.at_speed(speed) if speed else secondary_native
        frame_s = None
        if video_info.fps and exact_rate(float(video_info.fps)) is not None:
            frame_s = 1.0 / float(exact_rate(float(video_info.fps)))
        aligner = _Aligner(primary, secondary, token, say, frame_s)

        if speed is None and video_info.fps:
            # Verify the frame rate before anything else trusts an offset. The
            # video's own metadata names its rate, the dub's mastering rate is
            # one of the standard ones, so the possible speedups are a short
            # list: ask each on the audio itself, before the coarse pass could
            # misread a rate mismatch as a flood of cuts. A bare audio file
            # carries no rate; for that the symptom-driven passes below stand.
            video_rate = exact_rate(float(video_info.fps))
            if video_rate is not None:
                plan.video_fps = float(video_rate)
                report(38, "checking the frame rate")
                say(f"video is {_format_fps(plan.video_fps)} fps; checking the dub's rate")
                trial = _try_speeds(
                    aligner, primary, secondary_native, search_s, say,
                    candidates=_fps_speed_candidates(video_rate),
                )
                if trial is not None:
                    secondary = trial
                    aligner = _Aligner(primary, secondary, token, say, frame_s)

        report(40, "finding the offsets")
        grid, blocks = _coarse_blocks(aligner, search_s, report, say)

        if speed is None and _coverage(blocks, primary.duration_s) <= SPEED_RETRY_FRACTION:
            # Too little to trust as they are. A rate conversion looks
            # exactly like this -- the peak smears across the window -- and
            # it has a short list of possible values, so try them.
            trial = _try_speeds(aligner, primary, secondary_native, search_s, say)
            if trial is not None:
                secondary = trial
                aligner = _Aligner(primary, secondary, token, say, frame_s)
                grid, blocks = _coarse_blocks(aligner, search_s, report, say)

        if speed is None and blocks:
            # A small speed difference does not stop the coarse pass; it
            # shows as every stretch's offset walking steadily. Read the
            # slope, try the speed that cancels it, keep it if the fine
            # measurement agrees.
            for _round in range(SPEED_ROUNDS):
                trial = _speed_from_drift(
                    aligner, primary, secondary_native, blocks, search_s, report, say, token
                )
                if trial is None:
                    break
                secondary, aligner, grid, blocks = trial
        plan.speed = secondary.speed
        if plan.video_fps is not None:
            # The rate the dub was mastered at, implied by the final speed:
            # the dub runs at ``speed`` times the video's clock.
            plan.dub_rate = round(plan.speed * plan.video_fps, 6)

        if _coverage(blocks, primary.duration_s) < WIDE_PASS_COVERAGE:
            wide = _wide_blocks(aligner, lambda f: report(56 + 4 * f, "searching the whole dub"), say, blocks)
            if wide:
                blocks = _credible(_measure_and_split(aligner, _merge_blocks(blocks, wide)), aligner)
                say(
                    f"with the wide pass: {len(blocks)} credible stretch(es) covering "
                    f"{100 * _coverage(blocks, primary.duration_s):.0f}%"
                )

        if not blocks:
            plan.error = "No part of the dub could be matched to the video"
            return plan

        report(60, "placing the cuts")
        blocks = _refine(aligner, grid, blocks, primary.duration_s, secondary.duration_s, report, plan.warnings)
        if not blocks:
            plan.error = "No part of the dub could be matched to the video"
            return plan

        report(85, "assembling the plan")
        if fill_gain_db is None:
            fill_gain_db = _fill_gain_db(primary, secondary, blocks)
        plan.fill_gain_db = float(fill_gain_db)
        plan.segments = _assemble(
            blocks, primary.duration_s, secondary.duration_s, primary, secondary,
            keep_unmatched_dub, plan.warnings, plan.notes, aligner,
        )
        if not plan.dub_segments:
            plan.error = "No part of the dub could be placed on the video"
            return plan
        report(100, "done")
        return plan
    except MediaError as exc:
        plan.error = str(exc)
        return plan


def _coarse_blocks(
    aligner: _Aligner, search_s: float, report: Callable[[float, str], None], say: LogFn
) -> Tuple[ScoreGrid, List[Block]]:
    """The coarse pass, through to credible stretches with measured offsets."""
    grid = aligner.coarse(search_s, lambda f: report(40 + 20 * f, "finding the offsets"))
    path, starts = best_path(grid.scores)
    outright = sum(1 for i, c in enumerate(path) if c is not None and grid.scores[i, c] >= MATCH_Z)
    lags = grid.scores.shape[1]
    blocks = [
        b for b in blocks_from_path(grid, path, starts)
        if b.support >= _block_bar(b.windows, lags)
    ]
    blocks = _credible(_measure_and_split(aligner, blocks), aligner)
    say(
        f"at {aligner.secondary.speed:.6f}x: {outright} of {len(path)} windows match the dub outright, "
        f"{len(blocks)} credible stretch(es) covering {100 * _coverage(blocks, aligner.primary.duration_s):.0f}%"
    )
    return grid, blocks


def _estimate_drift(blocks: List[Block]) -> Tuple[Optional[float], float]:
    """How fast the offset walks, in ms per second of video, from the
    coarse path's own trace through every long stretch.

    Each stretch gives a Theil-Sen slope over its windows; the slopes are
    combined as a median weighted by how much material each was read
    from. The path's columns are quantised to the coarse grid, but a
    stretch several minutes long walks tens of columns at the speeds that
    matter, and a cut inside a stretch cannot tilt a median of slopes the
    way it would tilt a least-squares line.

    Returns (drift, span of stretches it was read from), or (None, 0).
    """
    slopes: List[Tuple[float, float]] = []
    for block in blocks:
        trace = block.trace
        if len(trace) < 6:
            continue
        span = trace[-1][0] - trace[0][0]
        if span < 60.0:
            continue
        pairs = [
            (trace[j][1] - trace[i][1]) / (trace[j][0] - trace[i][0])
            for i in range(len(trace))
            for j in range(i + 1, len(trace))
            if trace[j][0] != trace[i][0]
        ]
        if pairs:
            slopes.append((float(np.median(pairs)) * 1000.0, span))
    if not slopes:
        return None, 0.0
    values = np.array([s for s, _ in slopes])
    weights = np.array([w for _, w in slopes])
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    drift = float(values[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])
    return drift, float(weights.sum())


def _snap_speed(speed: float) -> float:
    """The standard frame-rate conversion nearest a measured speed, if one
    lies within tolerance; otherwise the speed as measured. 1000/1001 is
    what a 24 -> 23.976 conversion really is, and applying the measured
    0.99904 instead would leave 0.04 ms/s behind."""
    for numerator in COMMON_RATES:
        for denominator in COMMON_RATES:
            exact = float(numerator / denominator)
            if abs(exact - speed) <= RATIO_TOLERANCE:
                return exact
    return speed


def _fine_score(blocks: List[Block], aligner: _Aligner) -> float:
    """How well the stretches measure at 2ms: the length-weighted median
    full-band match of every stretch long enough to have been measured
    properly. The full band only: its transients are two frames wide, so
    a speed error of a millisecond a second smears them across a window
    and the match collapses, which is the symptom being judged; a band's
    onsets are twenty frames wide and hardly notice."""
    long = [(aligner.sharp_match(b), b.length_s) for b in blocks if b.windows >= 3 and b.length_s > 0]
    if not long:
        return 0.0
    values = np.array([m for m, _ in long])
    weights = np.array([w for _, w in long])
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def _speed_from_drift(
    aligner: _Aligner,
    primary: TrackEnvelope,
    secondary_native: TrackEnvelope,
    blocks: List[Block],
    search_s: float,
    report: Callable[[float, str], None],
    say: LogFn,
    token: Optional[CancellationToken],
) -> Optional[Tuple[TrackEnvelope, _Aligner, ScoreGrid, List[Block]]]:
    """Cancel a small speed difference the coarse pass tolerated.

    Returns the dub at the new speed with its aligner, grid and stretches,
    or None when there is no drift worth cancelling or cancelling it did
    not help.
    """
    drift, span = _estimate_drift(blocks)
    if drift is None or span < MIN_DRIFT_SPAN_S:
        return None
    say(f"the offset drifts by {drift:+.2f} ms/s across {_duration(span)} of matched dub")
    if abs(drift) < MIN_DRIFT_MS_PER_S:
        return None
    # The dub's clock runs at (1 + drift/1000) times the video's, so it is
    # played at the reciprocal to bring it onto the video's clock.
    current = aligner.secondary.speed
    candidate = _snap_speed(current / (1.0 + drift / 1000.0))
    if abs(candidate - current) < 1e-9:
        return None
    baseline = _fine_score(blocks, aligner)
    trial = secondary_native.at_speed(candidate)
    trial_aligner = _Aligner(primary, trial, token, say, aligner.frame_s)
    grid, trial_blocks = _coarse_blocks(trial_aligner, search_s, report, say)
    score = _fine_score(trial_blocks, trial_aligner)
    say(f"  at {candidate:.6f}x the stretches measure at {score:.2f}, against {baseline:.2f} as they were")
    if not trial_blocks or score < SPEED_FINE_GAIN * baseline:
        return None
    say(f"dub runs at {candidate:.6f}x the video's speed")
    return trial, trial_aligner, grid, trial_blocks


def _wide_blocks(
    aligner: _Aligner,
    report: Callable[[float, str], None],
    say: LogFn,
    anchors: Sequence[Block] = (),
) -> List[Block]:
    """Stretches found by searching every window of the video across the
    whole dub at 2 ms (see WIDE_PASS_COVERAGE).

    Each window keeps its best offset when it stands WIDE_Z above the noise
    of the whole search; runs of consecutive windows agreeing on an offset
    are stretches, bracketed the way the coarse pass brackets its own. The
    stretches are measured and their edges placed afterwards like any other.

    "Best" is not simply tallest. A cue that recurs -- a theme heard four
    times, an episode's opening -- correlates at every occurrence, and the
    tallest peak is the occurrence mixed loudest, not the one this window
    belongs to. So every window keeps its few tallest peaks, and the one
    taken is the one that continues what is already known: within
    WIDE_LOCAL_S of the offset the coarse stretches (``anchors``) give the
    neighbourhood, when such a peak is credible on its own; failing that,
    one that does not wind the dub back a little way from the previous
    window (a repeat within the same scene rewinds by a minute or two; a
    dub made of episodes out of order rewinds by a whole episode, which is
    allowed); failing that, the tallest.
    """
    rate = ENVELOPE_RATE
    window = int(WIDE_WINDOW_S * rate)
    hop = int(WIDE_HOP_S * rate)
    primary_n = len(aligner.primary.onset)
    secondary_n = len(aligner.secondary.onset)
    if primary_n < window or secondary_n < window:
        return []
    starts = list(range(0, primary_n - window + 1, hop))
    possible = np.ones(secondary_n - window + 1, dtype=bool)
    apart = int(WIDE_PEAK_APART_S * rate)
    hits: List[Tuple[float, float, float]] = []  # (window start, offset, noise units)
    previous: Optional[Tuple[float, float]] = None  # (window start, offset) of the last hit
    for index, start in enumerate(starts):
        if aligner.token:
            aligner.token.raise_if_cancelled()
        candidates: List[Tuple[float, float]] = []  # (offset, noise units)
        for band in aligner.detect_bands:
            template = aligner.onsets("primary", band)[start : start + window]
            row = ncc_lags(template, aligner.onsets("secondary", band))
            if row.size < 3:
                continue
            units = _noise_units(row, possible)
            for peak in _tallest_peaks(units, WIDE_PEAKS, apart):
                z = float(units[peak])
                if z < WIDE_Z:
                    break
                offset = (_parabolic(row, peak) - start) / rate
                near = [c for c in candidates if abs(c[0] - offset) <= WIDE_SAME_OFFSET_S]
                if near:
                    if z > near[0][1]:
                        candidates[candidates.index(near[0])] = (offset, z)
                else:
                    candidates.append((offset, z))
        report((index + 1) / len(starts))
        if not candidates:
            continue
        position = start / rate
        chosen = _continuing_candidate(candidates, position, anchors, previous)
        hits.append((position, chosen[0], chosen[1]))
        previous = (position, chosen[0])

    blocks: List[Block] = []
    run: List[Tuple[float, float, float]] = []

    def close() -> None:
        if len(run) >= WIDE_MIN_WINDOWS:
            blocks.append(Block(
                start_s=run[0][0],
                end_s=run[-1][0] + WIDE_WINDOW_S,
                offset_s=float(np.median([h[1] for h in run])),
                support=float(np.mean([h[2] for h in run])),
                windows=len(run),
                start_lo=max(0.0, run[0][0] - WIDE_HOP_S),
                start_hi=run[0][0] + WIDE_WINDOW_S,
                end_lo=run[-1][0],
                end_hi=run[-1][0] + WIDE_WINDOW_S + WIDE_HOP_S,
                trace=[(h[0], h[1]) for h in run],
            ))
        run.clear()

    for hit in hits:
        if run and (
            abs(hit[1] - run[-1][1]) > WIDE_SAME_OFFSET_S
            or hit[0] - run[-1][0] > 2.0 * WIDE_HOP_S + 1e-6
        ):
            close()
        run.append(hit)
    close()
    say(
        f"wide pass: {len(hits)} of {len(starts)} windows of {WIDE_WINDOW_S:.0f}s found in the dub "
        f"at {WIDE_Z:.0f}+ noise units, {len(blocks)} stretch(es) of {WIDE_MIN_WINDOWS}+ windows"
    )
    return blocks


def _merge_blocks(coarse: List[Block], wide: List[Block]) -> List[Block]:
    """The coarse stretches and the wide pass's together, without overlap.

    Where the two overlap at one offset they are one stretch, spanning
    both. Where they overlap at different offsets the wide pass's is kept:
    it was measured at 2 ms across the whole dub, where the coarse pass
    coasts over windows with no evidence and searches only where a dub of
    the same edit could be.
    """
    from_wide = {id(b) for b in wide}
    kept: List[Block] = []
    for block in sorted(coarse + wide, key=lambda b: (b.start_s, b.end_s)):
        if not kept:
            kept.append(block)
            continue
        last = kept[-1]
        overlap = min(last.end_s, block.end_s) - max(last.start_s, block.start_s)
        if overlap <= 0.0:
            kept.append(block)
            continue
        if abs(block.offset_s - last.offset_s) <= NEAR_OFFSET_S:
            wide_one = block if id(block) in from_wide else last
            last.start_s = min(last.start_s, block.start_s)
            last.end_s = max(last.end_s, block.end_s)
            last.start_lo = min(last.start_lo, block.start_lo)
            last.end_hi = max(last.end_hi, block.end_hi)
            last.offset_s = wide_one.offset_s
            last.support = max(last.support, block.support)
            last.windows += block.windows
            last.trace = sorted(last.trace + block.trace)
            continue
        if id(block) in from_wide and id(last) not in from_wide:
            loser, winner = last, block
        elif id(last) in from_wide and id(block) not in from_wide:
            loser, winner = block, last
        else:
            # Both from one pass: the longer stays whole.
            loser, winner = (block, last) if last.length_s >= block.length_s else (last, block)
        if loser is last:
            last.end_s = min(last.end_s, winner.start_s)
            last.end_lo = min(last.end_lo, last.end_s)
            last.end_hi = min(last.end_hi, winner.start_s)
            if last.length_s < MIN_BLOCK_S:
                kept.pop()
            kept.append(block)
        else:
            block.start_s = max(block.start_s, winner.end_s)
            block.start_hi = max(block.start_hi, block.start_s)
            block.start_lo = max(block.start_lo, winner.end_s)
            if block.length_s >= MIN_BLOCK_S:
                kept.append(block)
    return kept


def _coverage(blocks: List[Block], duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return min(1.0, sum(b.length_s for b in blocks) / duration_s)


def _fps_speed_candidates(video_rate: Fraction) -> List[Tuple[float, float]]:
    """Every mastering rate worth trying for a dub against a video of known
    frame rate, as (playback speed, mastering rate) pairs, likeliest first.

    The video's own metadata says its rate; the dub's rate is one of the
    standard ones, so the possible speeds are exactly audio/video for each
    standard audio rate. Ordering by how close each is to 1.0 tries the
    matched rate and the small conversions before the large ones.
    """
    pairs = [
        (float(audio_rate / video_rate), float(audio_rate))
        for audio_rate in COMMON_RATES
        if audio_rate != video_rate
        and abs(float(audio_rate / video_rate) - 1.0) <= SPEED_BAND
    ]
    pairs.sort(key=lambda pair: abs(pair[0] - 1.0))
    return pairs


def _try_speeds(
    aligner: _Aligner,
    primary: TrackEnvelope,
    secondary_native: TrackEnvelope,
    search_s: float,
    say: LogFn,
    candidates: Optional[List[Tuple[float, float]]] = None,
) -> Optional[TrackEnvelope]:
    """Find a standard playback speed at which the pair correlates.

    The dub's envelope is resampled rather than the dub re-decoded, since a
    speed is a time-stretch and the envelope is a curve in time. Three
    windows in the middle of the file are enough to ask: at the right speed
    the peak stands well clear of every other.

    ``candidates`` narrows the question to explicit (speed, mastering rate)
    pairs -- the frame-rate verdict the video's own metadata suggests. When
    omitted, every standard conversion is tried, ordered by the durations.
    """
    if candidates is None:
        candidates = [
            (float(c), 0.0)
            for c in speed_candidates(primary.duration_s, secondary_native.duration_s)
        ]
    candidates = [
        (speed, rate) for speed, rate in candidates
        if abs(speed - 1.0) <= SPEED_BAND and abs(speed - 1.0) > 1e-9
    ]
    if not candidates:
        return None
    pool = COARSE_POOL
    rate = ENVELOPE_RATE / pool
    window_s = min(SPEED_TRIAL_WINDOW_S, primary.duration_s / (SPEED_TRIAL_WINDOWS + 1))
    window = int(round(window_s * rate))
    if window < int(5.0 * rate):
        return None
    usable = primary.duration_s - window_s
    picks = [int(round((usable * (i + 1)) / (SPEED_TRIAL_WINDOWS + 1) * rate)) for i in range(SPEED_TRIAL_WINDOWS)]
    primary_pooled = _pool(primary.onset, pool)

    def trial_peak(envelope: TrackEnvelope) -> float:
        lo_s = -(max(0.0, primary.duration_s - envelope.duration_s) + search_s)
        hi_s = max(0.0, envelope.duration_s - primary.duration_s) + search_s
        scores = score_windows(
            primary_pooled, _pool(envelope.onset, pool), rate, picks, window,
            int(round(lo_s * rate)), int(round(hi_s * rate)), aligner.token,
        ).scores
        return float(np.median(scores.max(axis=1)))

    baseline = trial_peak(aligner.secondary)
    say(f"  at the files' own speed the trial windows peak at {baseline:.1f} noise units")

    best: Optional[Tuple[float, TrackEnvelope]] = None
    for candidate, _mastering_rate in candidates:
        trial = secondary_native.at_speed(candidate)
        peak = trial_peak(trial)
        say(f"  trying {candidate:.6f}x: peak {peak:.1f} noise units")
        if peak >= max(SPEED_MIN_Z, SPEED_GAIN * baseline) and (best is None or peak > best[0]):
            best = (peak, trial)
    if best is not None:
        say(f"dub runs at {best[1].speed:.6f}x the video's speed")
        return best[1]
    return None


def _refine(
    aligner: _Aligner,
    grid: ScoreGrid,
    coarse: List[Block],
    primary_s: float,
    secondary_s: float,
    report: Callable[[float, str], None],
    warnings: List[str],
) -> List[Block]:
    """From coarse stretches to exact ones.

    Each stretch is measured at 2 ms from many places inside it, which also
    finds any step the coarse pass was too blunt to see. Each edge is then
    placed at 2 ms from the agreement curve, inside the bracket the coarse
    windows left it in. Finally every gap long enough to hold one is searched
    for a stretch of dub the coarse windows straddled, and any found is
    placed the same way.
    """
    report(60, "measuring the offsets")
    blocks = _credible(_measure_and_split(aligner, coarse), aligner)

    report(70, "placing the cuts")
    _place_all_edges(aligner, blocks, primary_s, secondary_s)

    for round_index in range(RECOVER_ROUNDS):
        report(74 + 3 * round_index, "looking for dub inside the gaps")
        recovered = _recover_from_gaps(aligner, blocks, primary_s, secondary_s)
        if not recovered:
            break
        blocks = _credible(
            _measure_and_split(aligner, sorted(blocks + recovered, key=lambda b: b.start_s)), aligner
        )
        _place_all_edges(aligner, blocks, primary_s, secondary_s)

    blocks = [b for b in blocks if b.length_s >= MIN_BLOCK_S]
    # The edges moved: a stretch whose start was pulled back over material
    # the readings put at another level (an edge placed inside the bracket
    # an early, coarse split left it) now shows that level in its own
    # readings, and is split once more, on exactly what is inside it.
    report(83, "checking every stretch once more")
    resplit = _credible(_measure_and_split(aligner, blocks), aligner)
    if len(resplit) != len(blocks):
        blocks = resplit
        _place_all_edges(aligner, blocks, primary_s, secondary_s)
        blocks = [b for b in blocks if b.length_s >= MIN_BLOCK_S]
    for block in blocks:
        # The offset was measured before the edges moved; measure it again on
        # exactly the material that is now inside the stretch.
        block.offset_s, block.match, drift = aligner.polish_offset(block)
        if drift is not None and abs(drift) > DRIFT_WARN_MS_PER_S and block.length_s >= 120.0:
            warnings.append(
                f"the dub drifts by {drift:+.2f} ms/s across {_clock(block.start_s)} - "
                f"{_clock(block.end_s)}; it may run at a different speed (see --speed)"
            )
    _follow_effects(aligner, blocks)
    return blocks


def _follow_effects(aligner: _Aligner, blocks: List[Block]) -> None:
    """A stretch the music placed, between stretches the effects placed at
    another offset a few frames away, is asked which the effects on its
    own span prefer -- and follows them if they prefer the neighbours'.

    The case is a picture trim the music was left running across (see
    BANDS): the dialogue and effects step with the picture, the music
    does not until it is next re-laid, and a stretch found on the music
    alone sits a few frames from the dialogue. Only a neighbour within a
    few frames and a few seconds is asked about; farther apart the two
    are different cuts.
    """
    for index, block in enumerate(blocks):
        if block.effects:
            continue
        for other in (blocks[index - 1] if index else None, blocks[index + 1] if index + 1 < len(blocks) else None):
            if other is None or not other.effects:
                continue
            apart = abs(other.offset_s - block.offset_s)
            touching = min(abs(other.end_s - block.start_s), abs(other.start_s - block.end_s)) <= FOLLOW_EFFECTS_GAP_S
            if not touching or apart <= SAME_OFFSET_S or apart > FOLLOW_EFFECTS_STEP_S:
                continue
            rate = ENVELOPE_RATE
            window = min(POLISH_WINDOW_S, block.length_s)
            positions = [block.start_s + i * (block.length_s - window) / max(1, int(block.length_s / window)) for i in range(int(block.length_s / window) + 1)]
            if aligner._effects_prefer(positions, other.offset_s * 1000.0, block.offset_s * 1000.0):
                aligner.log(
                    f"  the music put {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.3f}s; "
                    f"the effects prefer the neighbour's {other.offset_s:+.3f}s, and the dialogue goes with the effects"
                )
                block.offset_s = other.offset_s
                block.effects = True
                break


def _credible(blocks: List[Block], aligner: _Aligner) -> List[Block]:
    """Drop stretches whose measured offset correlates at the noise floor.

    Done before any edge is placed: a stretch that is really noise does not
    just waste a splice, it drags its neighbours' edges towards itself when
    they are placed against it.
    """
    long_matches = [b.match for b in blocks if b.windows >= 3]
    floor = MIN_BLOCK_MATCH
    if long_matches:
        # Relative to what this pair's real stretches score, when they
        # score well: a pair whose stretches sit at 0.3 has noise blocks
        # at 0.05 that the absolute floor alone would keep. Capped, because
        # a quiet scene in a pair that otherwise measures at 0.6 is still
        # the dub: on a real pair, stretches the gap search had just found
        # at ten to twenty noise units were thrown out by a floor of 0.18.
        floor = max(floor, min(RELATIVE_FLOOR_CAP, RELATIVE_FLOOR * float(np.median(long_matches))))
    kept = []
    for block in blocks:
        agrees = block.match >= AGREEMENT_FLOOR and (
            (block.readings >= CONSISTENT_READINGS and block.consistency >= CONSISTENT_FRACTION)
            or (block.readings == 3 and block.spread_ms <= TIGHT_MS)
            # Found outright, by many windows agreeing on one offset; the
            # fine match being modest says the scene is quiet, not absent.
            or (block.support >= STRONG_SUPPORT_Z and block.windows >= 3)
        )
        # A stretch too short to show its readings agree has to clear a
        # higher bar on correlation alone. One that fails is not lost: if
        # it is real, the gap search finds it again where its neighbours
        # say it must be, which is a far narrower search than the path's.
        bar = floor if block.readings >= 3 else 2.0 * floor
        if block.match < bar and not agrees:
            aligner.log(
                f"  dropped {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.2f}s:"
                f" match {block.match:.2f} is noise"
            )
            continue
        # A match can be high by chance where a band is sparse; a stretch
        # fewer than three windows matched outright has to show its peak
        # standing above the offsets around it as well. Three windows
        # that matched are their own evidence: a quiet scene the gap
        # search had found over thirty-seven of them measured 4.6 here.
        z = aligner.credibility(block) if block.windows < 3 else float("inf")
        if z < CREDIBLE_Z:
            aligner.log(
                f"  dropped {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.2f}s:"
                f" its peak is only {z:.1f} noise units above the offsets around it"
            )
            continue
        kept.append(block)
    return kept


def _measure_and_split(aligner: _Aligner, blocks: List[Block]) -> List[Block]:
    """Measure every stretch's offset at 2 ms, splitting any that stepped.

    The first measurement of each stretch searches wide, so a cut the
    stretch straddles is seen as a step and split; the halves are then
    measured narrowly around their own offsets.
    """
    measured: List[Block] = []
    pending = [(block, True) for block in blocks]
    while pending:
        block, wide = pending.pop(0)
        block.offset_s, block.match, _ = aligner.polish_offset(block, wide)
        split = aligner.split_at_step(block, wide)
        if split is not None:
            pending = [(part, False) for part in split] + pending
            continue
        measured.append(block)
    return sorted(measured, key=lambda b: b.start_s)


def _place_all_edges(aligner: _Aligner, blocks: List[Block], primary_s: float, secondary_s: float) -> None:
    for index in range(len(blocks) + 1):
        before = blocks[index - 1] if index > 0 else None
        after = blocks[index] if index < len(blocks) else None
        _place_edge(aligner, before, after, primary_s, secondary_s)


def _place_edge(
    aligner: _Aligner,
    before: Optional[Block],
    after: Optional[Block],
    primary_s: float,
    secondary_s: float,
) -> None:
    """Put the boundary between two stretches exactly where the agreement moves.

    Each edge is first placed on its own, inside the bracket it was left
    in: where the first stretch's agreement ends, and where the second's
    begins. Those two answers are then read against what the offsets
    imply. If the dub is continuous across the cut -- the usual case, a
    scene missing from the dub -- the second must begin exactly (first
    offset - second offset) after the first ends, and two independent
    placements will not land there to the millisecond; the joint placement
    is used, which ties them together. If the dub itself skips material as
    well, the independent placements are kept, because then there is
    genuinely a gap on both sides.

    That is done twice: on the onset envelopes, which always exist, and then
    on the waveforms inside the interval the envelopes left, where the two
    mixes turn out to share one. Every fit is confined to the edge's own
    bracket, so an edge already placed on the waveform is not moved by a
    later pass on the envelope.
    """
    if before is None and after is None:
        return
    lo_s = min(b for b in (before.end_lo if before else None, after.start_lo if after else None) if b is not None)
    hi_s = max(b for b in (before.end_hi if before else None, after.start_hi if after else None) if b is not None)
    lo_s, hi_s = max(0.0, lo_s), min(primary_s, hi_s)
    if hi_s <= lo_s:
        return
    # The brackets as the coarse windows left them, before the envelope
    # narrows them: the waveform pass searches these.
    coarse_before = (before.end_lo, before.end_hi) if before else None
    coarse_after = (after.start_lo, after.start_hi) if after else None
    curve_before = aligner.edge_curve(before.offset_s, lo_s, hi_s) if before else None
    curve_after = aligner.edge_curve(after.offset_s, lo_s, hi_s) if after else None
    _fit_edges(
        before, after, curve_before, curve_after,
        before.match if before else 0.0, after.match if after else 0.0,
        lo_s, float(ENVELOPE_RATE), primary_s, secondary_s,
    )
    _sharpen(aligner, before, after, primary_s, secondary_s, coarse_before, coarse_after)


def _fit_edges(
    before: Optional[Block],
    after: Optional[Block],
    curve_before: Optional[np.ndarray],
    curve_after: Optional[np.ndarray],
    r_before: float,
    r_after: float,
    span_lo_s: float,
    rate: float,
    primary_s: float,
    secondary_s: float,
) -> None:
    """Place the edge(s) on agreement curves that start at ``span_lo_s`` and
    run at ``rate``, each confined to its edge's bracket. Writes the result
    and its uncertainty back into the blocks."""

    def index(t: float, curve: np.ndarray) -> int:
        return max(0, min(len(curve), int(round((t - span_lo_s) * rate))))

    def about(match: float, bracket_s: float) -> float:
        steps = (EDGE_UNCERTAINTY_STEPS / max(match, 0.02)) ** 2
        return max(SHARPEN_MIN_UNCERTAINTY_S / 2.0, min(bracket_s / 2.0, steps / rate))

    def covered(lo: float, hi: float, curve: np.ndarray) -> bool:
        return hi > span_lo_s and lo < span_lo_s + len(curve) / rate

    end_s = start_s = None
    if before is not None and curve_before is not None and curve_before.size and covered(before.end_lo, before.end_hi, curve_before):
        k = _fit_end(
            curve_before, _agreement_threshold(r_before),
            index(before.end_lo, curve_before), index(before.end_hi, curve_before),
        )
        end_s = span_lo_s + k / rate
    if after is not None and curve_after is not None and curve_after.size and covered(after.start_lo, after.start_hi, curve_after):
        k = _fit_start(
            curve_after, _agreement_threshold(r_after),
            index(after.start_lo, curve_after), index(after.start_hi, curve_after),
        )
        start_s = span_lo_s + k / rate

    tied = False
    if before is not None and after is not None:
        current_end = end_s if end_s is not None else before.end_s
        current_start = start_s if start_s is not None else after.start_s
        dub_gap_s = (current_start + after.offset_s) - (current_end + before.offset_s)
        video_gap_s = max(0.0, before.offset_s - after.offset_s)
        # Tied when the dub is as good as continuous across the cut -- or
        # when the independent placements are impossible: overlapping on
        # the video, or overlapping on the dub, which would play the same
        # dub audio twice.
        tied = current_start < current_end or dub_gap_s < CONTINUOUS_DUB_S
        if tied and curve_before is not None and curve_after is not None and curve_before.size and curve_after.size:
            gap_n = int(round(video_gap_s * rate))
            lo = max(index(before.end_lo, curve_before), index(after.start_lo, curve_after) - gap_n)
            hi = min(index(before.end_hi, curve_before), index(after.start_hi, curve_after) - gap_n)
            if hi >= lo:
                k = _fit_joint(
                    curve_before, curve_after,
                    _agreement_threshold(r_before), _agreement_threshold(r_after),
                    gap_n, lo, hi,
                )
                end_s = span_lo_s + k / rate
                start_s = end_s + video_gap_s
            else:
                tied = False
        elif tied and end_s is not None and start_s is None:
            start_s = end_s + video_gap_s
        elif tied and start_s is not None and end_s is None:
            end_s = start_s - video_gap_s

    if before is not None and end_s is not None:
        bracket = before.end_hi - before.end_lo
        before.end_s = min(end_s, secondary_s - before.offset_s, primary_s)
        u = about(r_before if not tied else max(r_before, r_after), bracket)
        before.end_lo, before.end_hi = before.end_s - u, before.end_s + u
    if after is not None and start_s is not None:
        bracket = after.start_hi - after.start_lo
        after.start_s = max(start_s, -after.offset_s, 0.0)
        u = about(r_after if not tied else max(r_before, r_after), bracket)
        after.start_lo, after.start_hi = after.start_s - u, after.start_s + u


def _sharpen(
    aligner: _Aligner,
    before: Optional[Block],
    after: Optional[Block],
    primary_s: float,
    secondary_s: float,
    bracket_before: Optional[Tuple[float, float]] = None,
    bracket_after: Optional[Tuple[float, float]] = None,
) -> None:
    """Re-place an edge on the waveforms, inside the given brackets.

    The brackets default to the edges' current ones; callers with a wider
    bracket from the coarse windows pass it, and the search covers all of
    it -- the envelope's own placement is not trusted to have narrowed it
    correctly. But the envelope's placement is only ever *replaced*, never
    thrown away: an edge the waveform cannot place keeps the envelope's
    answer and the envelope's uncertainty. A version of this that widened
    the brackets first and narrowed them on success left every edge the
    waveform could not reach at the coarse bracket, tens of seconds wide,
    after the envelope had placed it to within half a second.
    """
    fit_before = before is not None and before.end_uncertainty_s >= SHARPEN_MIN_UNCERTAINTY_S
    fit_after = after is not None and after.start_uncertainty_s >= SHARPEN_MIN_UNCERTAINTY_S
    if not fit_before and not fit_after:
        return
    search_before = (bracket_before or (before.end_lo, before.end_hi)) if fit_before else None
    search_after = (bracket_after or (after.start_lo, after.start_hi)) if fit_after else None

    if search_before and search_after:
        if max(search_before[1], search_after[1]) - min(search_before[0], search_after[0]) > SHARPEN_MAX_SPAN_S:
            # A long gap: the two edges have nothing to do with each other,
            # so each is sharpened on its own, each with its own bracket.
            _sharpen(aligner, before, None, primary_s, secondary_s, bracket_before=search_before)
            _sharpen(aligner, None, after, primary_s, secondary_s, bracket_after=search_after)
            return

    edges = [e for e in (search_before, search_after) if e is not None]
    lo_s = max(0.0, min(e[0] for e in edges))
    hi_s = min(primary_s, max(e[1] for e in edges))
    if hi_s - lo_s > SHARPEN_MAX_SPAN_S:
        # Too wide to decode whole: centred on where the envelope put the
        # edge(s), which is the best guess there is.
        placed = [e for e in (before.end_s if fit_before else None, after.start_s if fit_after else None) if e is not None]
        middle = min(max(sum(placed) / len(placed), lo_s + SHARPEN_MAX_SPAN_S / 2.0), hi_s - SHARPEN_MAX_SPAN_S / 2.0)
        lo_s, hi_s = middle - SHARPEN_MAX_SPAN_S / 2.0, middle + SHARPEN_MAX_SPAN_S / 2.0

    # The decoded span reaches into each stretch beyond the bracket, so
    # there is material known to be the stretch's own to measure the
    # sample-exact offset on; and a little past it, so the cumulative sums
    # have a run-up.
    room = SHARPEN_INTERIOR_S + 1.0
    span_lo = max(0.0, lo_s - (room if fit_before else 1.0))
    span_hi = min(primary_s, hi_s + (room if fit_after else 1.0))
    if fit_before:
        span_lo = max(span_lo, before.start_s)
    if fit_after:
        span_hi = min(span_hi, after.end_s)
    if span_hi - span_lo < 1.0:
        return

    # Where the sample-exact offset is measured: material that is the
    # stretch's own beyond doubt, which is everything up to the envelope's
    # placement less its uncertainty. Falls back to the whole side of the
    # span when that leaves too little.
    curve_before = curve_after = None
    r_before = r_after = 0.0
    if fit_before:
        sure = before.end_s - before.end_uncertainty_s
        inside = (span_lo, min(span_hi, sure if sure - span_lo >= SHARPEN_INTERIOR_S else hi_s))
        got = aligner.waveform_agreement(before, span_lo, span_hi, inside)
        if got is not None:
            curve_before, r_before = got
    if fit_after:
        sure = after.start_s + after.start_uncertainty_s
        inside = (max(span_lo, sure if span_hi - sure >= SHARPEN_INTERIOR_S else lo_s), span_hi)
        got = aligner.waveform_agreement(after, span_lo, span_hi, inside)
        if got is not None:
            curve_after, r_after = got
    if curve_before is None and curve_after is None:
        return

    # Widen the brackets to the search interval only for the sides about
    # to be fitted, so the fit may move the edge anywhere the coarse
    # windows allowed; the fit narrows them again.
    if curve_before is not None:
        before.end_lo, before.end_hi = max(lo_s, search_before[0]), min(hi_s, search_before[1])
    if curve_after is not None:
        after.start_lo, after.start_hi = max(lo_s, search_after[0]), min(hi_s, search_after[1])
    _fit_edges(
        before if curve_before is not None else None,
        after if curve_after is not None else None,
        curve_before, curve_after, r_before, r_after,
        span_lo, float(ANALYSIS_SR), primary_s, secondary_s,
    )
    # One side placed on the waveform, the other not: if the dub runs
    # straight through the cut the other follows it exactly.
    if before is not None and after is not None and (curve_before is None) != (curve_after is None) and fit_before and fit_after:
        dub_gap_s = (after.start_s + after.offset_s) - (before.end_s + before.offset_s)
        if abs(dub_gap_s) < CONTINUOUS_DUB_S:
            gap = max(0.0, before.offset_s - after.offset_s)
            if curve_before is not None:
                after.start_s = max(before.end_s + gap, -after.offset_s, 0.0)
                after.start_lo, after.start_hi = before.end_lo + gap, before.end_hi + gap
            else:
                before.end_s = min(after.start_s - gap, secondary_s - before.offset_s, primary_s)
                before.end_lo, before.end_hi = after.start_lo - gap, after.start_hi - gap


def _recover_from_gaps(
    aligner: _Aligner, blocks: List[Block], primary_s: float, secondary_s: float
) -> List[Block]:
    """Search every gap for the dub with short windows, over the few offsets
    the neighbouring stretches allow.

    The coarse pass is built to find long stretches against a quarter of a
    million possible offsets, which is why it needs long windows and a
    high price on every jump. Inside a gap the question is far smaller:
    the dub material for this span, if it exists, sits between where the
    stretch before ends and the stretch after begins, give or take a few
    seconds. Against a few hundred offsets a ten-second window is
    conclusive, a jump is cheap, and a run of nothing is called a gap after
    a few windows -- so a span with three cuts inside it comes back as
    three stretches at three offsets, each with its own bracket.
    """
    pool = GAP_POOL
    rate = ENVELOPE_RATE / pool
    found: List[Block] = []

    gaps: List[Tuple[float, float, Optional[Block], Optional[Block]]] = []
    cursor = 0.0
    for index, block in enumerate(blocks):
        if block.start_s - cursor >= RECOVER_MIN_S:
            gaps.append((cursor, block.start_s, blocks[index - 1] if index else None, block))
        cursor = max(cursor, block.end_s)
    if primary_s - cursor >= RECOVER_MIN_S:
        gaps.append((cursor, primary_s, blocks[-1] if blocks else None, None))

    for gap_lo, gap_hi, before, after in gaps:
        gap_s = gap_hi - gap_lo
        # The offsets the dub material for this gap could sit at. With a
        # neighbour on each side, between their offsets: the dub may lack
        # some of the gap (the offset falls across it) or have extra (it
        # rises), but it cannot pass either neighbour's offset by more than
        # the slack. With a neighbour on one side only, the gap's own
        # length either way, since that is the most the dub could lack or
        # have extra here.
        offsets = [b.offset_s for b in (before, after) if b is not None]
        if len(offsets) == 2:
            lo_s = min(offsets) - RECOVER_SLACK_S
            hi_s = max(offsets) + RECOVER_SLACK_S
        else:
            reach = min(gap_s, GAP_ONE_SIDED_MAX_S) + RECOVER_SLACK_S
            lo_s = offsets[0] - reach
            hi_s = offsets[0] + reach
        lo_s = max(lo_s, -gap_hi)  # dub time cannot go negative
        hi_s = min(hi_s, secondary_s - gap_lo)
        if hi_s <= lo_s:
            continue

        # Finest scale first; each coarser scale only searches what the
        # finer ones left, so a loud scene's cuts are placed by the short
        # windows and a quiet scene is still found by the long ones.
        uncovered: List[Tuple[float, float]] = [(gap_lo, gap_hi)]
        for window_s in GAP_WINDOWS_S:
            remaining: List[Tuple[float, float]] = []
            for sub_lo, sub_hi in uncovered:
                if sub_hi - sub_lo < RECOVER_MIN_S:
                    continue
                pieces = _search_gap(
                    aligner, pool, rate, sub_lo, sub_hi, lo_s, hi_s,
                    min(window_s, sub_hi - sub_lo), secondary_s,
                )
                found.extend(pieces)
                cursor = sub_lo
                for piece in sorted(pieces, key=lambda b: b.start_s):
                    if piece.start_s - cursor >= RECOVER_MIN_S:
                        remaining.append((cursor, piece.start_s))
                    cursor = max(cursor, piece.end_s)
                if sub_hi - cursor >= RECOVER_MIN_S:
                    remaining.append((cursor, sub_hi))
            uncovered = remaining
            if not uncovered:
                break
    return found


def _gap_candidates(grid: ScoreGrid, floor: float, window_s: float) -> List[Block]:
    """The stretches one gap-search grid supports: the best path through
    it, cut into runs, each kept when it clears the bar for its length."""
    path, run_starts = best_path(grid.scores, GAP_JUMP_COST, GAP_NULL_REWARD, GAP_NULL_COST, WOBBLE_COST)
    lags = grid.scores.shape[1]
    return [
        block for block in blocks_from_path(grid, path, run_starts)
        if block.support >= max(floor, _block_bar(block.windows, lags, window_s / GAP_HOP_S))
    ]


def _search_gap(
    aligner: _Aligner,
    pool: int,
    rate: float,
    gap_lo: float,
    gap_hi: float,
    lo_s: float,
    hi_s: float,
    window_s: float,
    secondary_s: float,
) -> List[Block]:
    """One scale of the gap search: windows of ``window_s`` every GAP_HOP_S
    across [gap_lo, gap_hi), over offsets [lo_s, hi_s].

    Every band is scored and the scores are summed, in noise units, and
    put back on a unit floor: a scene with no music is found by the low
    band, one with no bass by the full band. The sum rather than the best,
    because a sparse band -- a few transients in a quiet scene -- has a
    heavy-tailed noise, and its lone coincidences stood up as stretches
    when the best band was taken; a real stretch shows in more than one
    band, and the sum keeps that and not the coincidence.
    """
    window = max(1, int(round(window_s * rate)))
    hop = max(1, int(round(GAP_HOP_S * rate)))
    first = int(round(gap_lo * rate))
    last = int(round(gap_hi * rate)) - window
    starts = list(range(first, max(first, last) + 1, hop))
    grid: Optional[ScoreGrid] = None
    bands = aligner.detect_bands
    for band in bands:
        scored = score_windows(
            aligner.pooled("primary", pool, band), aligner.pooled("secondary", pool, band), rate,
            starts, window, int(round(lo_s * rate)), int(round(hi_s * rate)), aligner.token,
        )
        if grid is None:
            grid = scored
        else:
            possible = (grid.scores > -1e5) & (scored.scores > -1e5)
            grid.scores[possible] += scored.scores[possible]
            grid.scores[~possible] = -1e6
    assert grid is not None
    if len(bands) > 1:
        possible = grid.scores > -1e5
        grid.scores[possible] /= math.sqrt(len(bands))
    found: List[Block] = []
    candidates = _gap_candidates(grid, RECOVER_Z, window_s)
    if "high" in aligner.shared_bands:
        # The effects on their own, held to a higher bar: a scene whose bed
        # is footsteps and room tone shows in the 4-8 kHz band alone, but
        # that band's chance peaks are tall (see ``detect_bands``), so only
        # a run of several windows that match outright is believed, and
        # only where the other bands found nothing.
        effects = score_windows(
            aligner.pooled("primary", pool, "high"), aligner.pooled("secondary", pool, "high"), rate,
            starts, window, int(round(lo_s * rate)), int(round(hi_s * rate)), aligner.token,
        )
        for block in _gap_candidates(effects, EFFECTS_RECOVER_Z, window_s):
            if block.windows < EFFECTS_RECOVER_WINDOWS:
                continue
            if any(min(block.end_s, c.end_s) - max(block.start_s, c.start_s) > 0.0 for c in candidates):
                continue
            candidates.append(block)
    for block in sorted(candidates, key=lambda b: b.start_s):
        block.start_s = max(gap_lo, block.start_s, -block.offset_s)
        block.end_s = min(gap_hi, block.end_s, secondary_s - block.offset_s)
        if block.end_s - block.start_s < MIN_BLOCK_S:
            continue
        block.start_lo = max(gap_lo, block.start_lo)
        block.end_hi = min(gap_hi, block.end_hi)
        aligner.log(
            f"  found dub for {_clock(block.start_s)} - {_clock(block.end_s)} at "
            f"{_clock(block.start_s + block.offset_s)} ({block.offset_s:+.2f}s, "
            f"{block.support:.1f} noise units over {block.windows} windows of {window_s:.0f}s)"
        )
        found.append(block)
    return found


def _trim_uncertain_edges(blocks: List[Block]) -> List[Block]:
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
            block.start_s += start_u
        if end_u >= EDGE_TRIM_MIN_S and not (after and abs(after.offset_s - block.offset_s) <= NEAR_OFFSET_S):
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
    quiet = secondary.energy < DUB_SILENT_RATIO * level
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


def _place_step(aligner: _Aligner, lo_s: float, hi_s: float, before_s: float, after_s: float) -> Tuple[float, bool]:
    """Where inside [lo_s, hi_s] a stretch at ``before_s`` gives way to one
    at ``after_s``: the point that leaves the most agreement at the first
    offset before it and at the second after it (after the frames the dub
    lacks, when the offset falls). Returns the point and whether the
    agreement said anything at all; when it did not, the middle."""
    rate = ENVELOPE_RATE
    drop = max(0.0, before_s - after_s)
    room = max(0.0, hi_s - lo_s - drop)
    a = aligner.agreement(before_s, lo_s, hi_s)
    b = aligner.agreement(after_s, lo_s, hi_s)
    n = min(len(a), len(b))
    if n < 2 or room <= 0.0:
        return lo_s + room / 2.0, False
    a, b = a[:n], b[:n]
    if max(float(np.abs(a).mean()), float(np.abs(b).mean())) < AGREEMENT_FLOOR:
        return lo_s + room / 2.0, False
    skip = int(round(drop * rate))
    ahead = np.concatenate([[0.0], np.cumsum(a)])          # agreement at the first offset up to k
    behind = np.concatenate([np.cumsum(b[::-1])[::-1], [0.0]])  # agreement at the second from k on
    last = n - skip
    if last <= 0:
        return lo_s + room / 2.0, False
    scores = ahead[: last + 1] + behind[skip : skip + last + 1]
    k = int(np.argmax(scores))
    # Said only when the choice stands out: a flat score is no placement.
    spread = float(scores.max() - np.median(scores))
    measured = spread > STEP_EVIDENCE * max(float(np.abs(a).mean()), float(np.abs(b).mean())) * rate
    return (lo_s + k / rate) if measured else lo_s + room / 2.0, measured


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
) -> List[Segment]:
    """Turn stretches into a contiguous list of pieces covering the video."""
    blocks = _trim_uncertain_edges(sorted((b for b in blocks if b.length_s > 0), key=lambda b: b.start_s))
    blocks = _split_silences(blocks, primary, secondary, warnings)
    if not blocks:
        return []

    # Merge what is really one stretch.
    merged: List[Block] = []
    for block in blocks:
        if merged:
            last = merged[-1]
            gap = block.start_s - last.end_s
            if abs(block.offset_s - last.offset_s) <= SAME_OFFSET_S and gap <= MIN_FILL_S:
                _join(last, block)
                continue
            if gap < 0:
                block.start_s = last.end_s
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
        if (
            gap > MIN_FILL_S and NEAR_OFFSET_S < abs(step)
            and (abs(step) <= BRIDGE_STEP_S or exposure <= BRIDGE_EXPOSURE_S)
            and gap - max(step, 0.0) >= 0.0
            and keep_unmatched_dub and dub_audible(last.end_s, block.start_s, last.offset_s)
        ):
            # The dub is there but did not correlate, and the offset steps
            # somewhere inside: a scene the two edits trim differently. Put
            # the step where the agreement changes from one offset to the
            # other; where neither offset agrees anywhere, in the middle,
            # which misplaces at most ``exposure`` seconds of dub, said in
            # the note. The frames the dub lacks are filled at the step.
            cut, measured = _place_step(aligner, last.end_s, block.start_s, last.offset_s, block.offset_s)
            how = "placed where the agreement changes" if measured else f"guessed; up to {exposure:.1f}s of dub may sit {abs(step):.1f}s off"
            notes.append(
                f"kept the dub across {_clock(last.end_s)} - {_clock(block.start_s)}: it did not "
                f"correlate there, but the offset steps by only {step * 1000.0:+.0f} ms inside it; the step was {how}"
            )
            last.end_s = last.end_lo = last.end_hi = cut
            block.start_s = block.start_lo = block.start_hi = cut + max(step, 0.0)
            kept.append(block)
            continue
        if gap > MIN_FILL_S and abs(block.offset_s - last.offset_s) <= NEAR_OFFSET_S:
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
            middle = (last.end_s + block.start_s) / 2.0
            last.end_s = last.end_lo = last.end_hi = middle
            block.start_s = block.start_lo = block.start_hi = middle
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

    for segment in segments:
        segment.start_s = round(segment.start_s, 3)
        segment.end_s = round(segment.end_s, 3)
        segment.source_start_s = round(segment.source_start_s, 3)
        if segment.offset_s is not None:
            segment.offset_s = round(segment.offset_s, 3)
        segment.uncertainty_s = round(segment.uncertainty_s, 2)
    return [s for s in segments if s.end_s > s.start_s]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class SpotCheck:
    position_s: float
    residual_ms: Optional[float]
    match: float
    note: str = ""


@dataclass
class Stretch:
    start_s: float
    end_s: float
    residual_ms: float
    windows: int


@dataclass
class Verification:
    """How the finished track sits against the original, measured."""

    spots: List[SpotCheck] = field(default_factory=list)
    sweep_windows: int = 0
    sweep_measured: int = 0
    sweep_within_audible: int = 0
    sweep_typical_ms: Optional[float] = None
    sweep_worst_ms: Optional[float] = None
    stretches: List[Stretch] = field(default_factory=list)
    fills_skipped: int = 0

    @property
    def typical_ms(self) -> Optional[float]:
        values = [abs(s.residual_ms) for s in self.spots if s.residual_ms is not None]
        return float(np.median(values)) if values else None

    @property
    def worst_ms(self) -> Optional[float]:
        values = [abs(s.residual_ms) for s in self.spots if s.residual_ms is not None]
        return float(max(values)) if values else None

    def to_dict(self) -> dict:
        return {
            "spots": [
                {"positionS": s.position_s, "residualMs": s.residual_ms, "match": s.match, "note": s.note}
                for s in self.spots
            ],
            "typicalMs": self.typical_ms,
            "worstMs": self.worst_ms,
            "sweepWindows": self.sweep_windows,
            "sweepMeasured": self.sweep_measured,
            "sweepWithinAudible": self.sweep_within_audible,
            "sweepTypicalMs": self.sweep_typical_ms,
            "sweepWorstMs": self.sweep_worst_ms,
            "stretches": [
                {"startS": s.start_s, "endS": s.end_s, "residualMs": s.residual_ms, "windows": s.windows}
                for s in self.stretches
            ],
        }

    def describe(self) -> str:
        lines = []
        measured = [s for s in self.spots if s.residual_ms is not None]
        if measured:
            lines.append(
                f"the finished track sits {self.typical_ms:.0f}ms from the original typically and "
                f"{self.worst_ms:.0f}ms at its worst, measured at {len(measured)} spots. "
                f"Lip-sync starts to show around {AUDIBLE_MS:.0f}ms; the measurement resolves 2ms."
            )
        else:
            lines.append("none of the spot checks could be measured")
        for spot in self.spots:
            if spot.residual_ms is None:
                lines.append(f"  at {_clock(spot.position_s)}: {spot.note}")
            else:
                direction = "early" if spot.residual_ms < 0 else "late"
                lines.append(
                    f"  at {_clock(spot.position_s)}: {abs(spot.residual_ms):.0f}ms {direction}, "
                    f"match {spot.match:.2f}"
                )
        if self.sweep_windows:
            share = 100.0 * self.sweep_within_audible / max(1, self.sweep_measured)
            lines.append(
                f"swept the whole runtime: {self.sweep_measured} of {self.sweep_windows} windows could be "
                f"measured (the rest are silence, music-only, or filled from the original), and "
                f"{share:.0f}% of them are within {AUDIBLE_MS:.0f}ms of the original"
                + (
                    f" (typically {self.sweep_typical_ms:.0f}ms, worst {self.sweep_worst_ms:.0f}ms)"
                    if self.sweep_typical_ms is not None else ""
                )
            )
        if self.stretches:
            lines.append(f"{len(self.stretches)} stretch(es) are more than {STRETCH_MS:.0f}ms out and would be audible:")
            for stretch in self.stretches:
                lines.append(
                    f"  {_clock(stretch.start_s)} - {_clock(stretch.end_s)}  out by "
                    f"{stretch.residual_ms:+.0f}ms  ({stretch.windows} windows)"
                )
        elif self.sweep_measured:
            lines.append(f"no stretch is more than {STRETCH_MS:.0f}ms out")
        return "\n".join(lines)


def _in_fill(plan: DubSyncPlan, lo_s: float, hi_s: float) -> bool:
    for segment in plan.fill_segments:
        if segment.start_s < hi_s and segment.end_s > lo_s:
            return True
    return False


def verify_output(
    primary: TrackEnvelope,
    output: TrackEnvelope,
    plan: DubSyncPlan,
    token: Optional[CancellationToken] = None,
) -> Verification:
    """Measure the finished track against the original along the runtime.

    Two views: a dozen long windows, each of which is a precise measurement,
    and a sweep of short windows every few seconds that can catch a short
    stretch the spots fell between. Windows in filled regions are skipped --
    the fill is the original, so it matches trivially and would flatter the
    numbers.
    """
    result = Verification()
    rate = ENVELOPE_RATE
    duration = min(primary.duration_s, output.duration_s)
    bands = [b for b in primary.bands if b in output.bands]

    # Spot checks.
    window = int(VERIFY_SPOT_WINDOW_S * rate)
    margin = int(VERIFY_SPOT_RANGE_S * rate)
    usable = duration - VERIFY_SPOT_WINDOW_S
    if usable > 0:
        inset = min(usable * 0.02, 5.0)
        step = (usable - 2 * inset) / max(1, VERIFY_SPOTS - 1)
        for index in range(VERIFY_SPOTS):
            if token:
                token.raise_if_cancelled()
            position = inset + step * index
            start = int(position * rate)
            if _in_fill(plan, position, position + VERIFY_SPOT_WINDOW_S):
                result.spots.append(SpotCheck(position, None, 0.0, "filled from the original"))
                result.fills_skipped += 1
                continue
            residual, match, z = _measure(
                [primary.onsets(b) for b in bands], [output.onsets(b) for b in bands], start, window, margin, rate, bands
            )
            if residual is None or z < MATCH_Z:
                result.spots.append(SpotCheck(position, None, match, "silence or music-only; not measurable"))
            else:
                result.spots.append(SpotCheck(position, residual, match))

    # Sweep.
    pool = VERIFY_SWEEP_POOL
    sweep_rate = rate / pool
    p = [_pool(primary.onsets(b), pool) for b in bands]
    o = [_pool(output.onsets(b), pool) for b in bands]
    window = int(VERIFY_SWEEP_WINDOW_S * sweep_rate)
    margin = int(VERIFY_SWEEP_RANGE_S * sweep_rate)
    residuals: List[Tuple[float, float]] = []
    count = int(duration // VERIFY_SWEEP_WINDOW_S)
    for index in range(count):
        if token:
            token.raise_if_cancelled()
        position = index * VERIFY_SWEEP_WINDOW_S
        if _in_fill(plan, position, position + VERIFY_SWEEP_WINDOW_S):
            continue
        result.sweep_windows += 1
        start = int(position * sweep_rate)
        residual, _match, z = _measure(p, o, start, window, margin, sweep_rate, bands)
        if residual is None or z < VERIFY_SWEEP_Z:
            continue
        result.sweep_measured += 1
        residuals.append((position, residual))
        if abs(residual) <= AUDIBLE_MS:
            result.sweep_within_audible += 1
    if residuals:
        magnitudes = [abs(r) for _, r in residuals]
        result.sweep_typical_ms = float(np.median(magnitudes))
        result.sweep_worst_ms = float(max(magnitudes))
        # Runs of consecutive measured windows that are all out the same way.
        run: List[Tuple[float, float]] = []
        for position, residual in residuals + [(float("inf"), 0.0)]:
            if abs(residual) > STRETCH_MS and (not run or position - run[-1][0] <= 2 * VERIFY_SWEEP_WINDOW_S):
                run.append((position, residual))
                continue
            if len(run) >= VERIFY_STRETCH_WINDOWS:
                result.stretches.append(Stretch(
                    run[0][0], run[-1][0] + VERIFY_SWEEP_WINDOW_S,
                    float(np.median([r for _, r in run])), len(run),
                ))
            run = []
            if abs(residual) > STRETCH_MS:
                run.append((position, residual))
    return result


def _measure(
    primary: Sequence[np.ndarray], other: Sequence[np.ndarray], start: int, window: int, margin: int, rate: float,
    bands: Optional[Sequence[str]] = None,
) -> Tuple[Optional[float], float, float]:
    """Residual offset (ms, positive = other is late) of one window.

    ``primary`` and ``other`` are the same bands of the two tracks. Each
    band's correlation is put in its own noise units and the bands are
    summed: a beat gives the full band a peak every period, about as tall
    as the true one, and only the true one is in every band. Among the
    summed peaks within 15% of the tallest, the one nearest zero lag is
    the answer -- this is a check that the track sits where it should,
    and a track that does gives a peak there.
    """
    total: Optional[np.ndarray] = None
    effects: Optional[np.ndarray] = None
    best_match = 0.0
    first, last = start - margin, start + window + margin
    for index, (p, o) in enumerate(zip(primary, other)):
        template = p[start : start + window]
        if len(template) < window or first < 0 or last > len(o):
            continue
        row = ncc_lags(template, o[first:last])
        if row.size < 3:
            continue
        if bands is not None and index < len(bands) and bands[index] == "high":
            effects = row.astype(np.float64)
        units = _noise_units(row, np.ones(len(row), dtype=bool))
        total = units.astype(np.float64) if total is None else total + units
        best_match = max(best_match, float(row.max()))
    if total is None:
        return None, 0.0, 0.0
    if effects is not None:
        # The effects decide here as they do in the plan (see BANDS): a
        # cue laid twice puts the music's peak a tenth of a second from
        # the dialogue's, and it is the dialogue the check is about.
        peak = _nearest_peak(effects, margin)
        units = _noise_units(effects, np.ones(len(effects), dtype=bool))
        if float(units[peak]) >= EFFECTS_Z and _peak_dominance(effects, peak) >= EFFECTS_DOMINANCE:
            lag = (first + _parabolic(effects, peak)) - start
            return lag / rate * 1000.0, best_match, float(units[peak])
    peak = _nearest_peak(total, margin)
    away = np.abs(np.arange(len(total)) - peak) > max(3, int(0.25 * rate))
    noise = float(np.std(total[away])) if away.sum() > 8 else 1.0
    z = float(total[peak]) / max(noise, 1e-9)
    lag = (first + _parabolic(total, peak)) - start
    return lag / rate * 1000.0, best_match, z
