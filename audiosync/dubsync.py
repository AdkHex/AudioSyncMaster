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
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .correlate import ENVELOPE_HOP, _fast_fft_size
from .framerate import COMMON_RATES, RATIO_TOLERANCE, _format_fps, exact_rate, speed_candidates
from .segments import find_step
from .shots import PictureCuts
from .media import Cancelled, CancellationToken, MediaError, audio_lead_s, load_audio, probe, stream_audio
from .voicefix import VoicePiece

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
# Where the effects band would overrule the other bands by at least this
# much, the dialogue itself is asked (see _Aligner.speech_choice): the
# effects follow the picture on some dubs and not on others -- measured on a
# real pair, the dub's lines sat with the music, 110 ms from the effects,
# where the effects band had stepped twenty seconds early, and 160 ms from
# them over half a scene -- and a line a frame or so out is below what the
# speech can tell. The speech decides when one level's lines agree this
# many deviations above their noise and this many above the other's.
SPEECH_ARBITER_S = 0.08
SPEECH_PREFER_Z = 2.5
SPEECH_PREFER_MARGIN = 1.0
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
# A stretch's offset is the median of its ten-second readings; the whole
# stretch at once is then asked too (see _settle). Where the whole stretch
# peaks elsewhere within POLISH_RANGE_S, by this many noise units, and the
# readings' offset scores under this share of that peak, the whole stretch
# decides. Its noise is taken over offsets this far either side, and only
# stretches this long are asked, since shorter ones are a reading or two.
SETTLE_MARGIN = 3.0
SETTLE_SHARE = 0.5
SETTLE_BASELINE_S = 2.0
SETTLE_MIN_S = 10.0
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
# The dub is treated as silent below this fraction of its own level --
# and below DUB_SILENT_FLOOR as well. Silence a studio lays where a scene
# was is digital silence or the codec's noise floor, -90 dBFS and under;
# a dialogue-forward mix with its bed ducked sits at -60 to -75 dBFS in
# its pauses, which the ratio alone (-54 dBFS on a real TV-style mix) took
# for silence, filling eleven seconds of pauses in a half-hour with the
# other language's room tone.
DUB_SILENT_RATIO = 0.03
DUB_SILENT_FLOOR = 1e-4  # -80 dBFS, frame RMS
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
# A step the sound placed, with no shot change to confirm it, keeps the dub
# across a gap only while a misplaced step could misplace at most this much
# of it; a step on a picture cut, however long the gap.
BRIDGE_SOUND_EXPOSURE_S = 30.0
# A guessed step that may misplace this much dub is a warning, not a note.
GUESS_WARN_S = 0.5
# The most a switch may go back in the dub and still be the same scene
# replayed, which the assembly leaves a hole for (see _assemble).
REWIND_GUARD_MAX_S = 30.0
# How far the best step placement must stand above the median one, in
# seconds of the mean agreement, to count as measured rather than guessed.
STEP_EVIDENCE = 1.0
# The largest doubts about a cut's place are listed for review, at most this many.
REVIEW_DOUBT_MAX = 12
# How far the original is allowed to be re-levelled to sit among the dub.
MAX_FILL_GAIN_DB = 12.0
# The level difference is read in pieces this long across every stretch.
FILL_GAIN_PIECE_S = 60.0
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
# Trial windows are thirty seconds at the envelope's full 2 ms resolution.
# They used to be two minutes at 20 ms, which told a PAL conversion (4%)
# apart at once but not a 24-against-23.976 one: a tenth of a percent
# smears a two-minute window by 120 ms, six bins of 20 ms, and the peak
# survives -- on a real pair the files' own speed and the right one both
# scored 2.9, and the verdict was left to the drift estimate, which needs
# minutes of matched dub. At 2 ms a thirty-second window is smeared by
# fifteen frames at the wrong speed and the peak collapses, while at the
# right speed the transients stay aligned across the whole window.
SPEED_TRIAL_WINDOW_S = 30.0
SPEED_TRIAL_POOL = 1
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
# The sweep reads at the envelope's full 2 ms: pooled to 10 ms it could only
# say a track was within a few milliseconds, and lip sync is judged finer.
VERIFY_SWEEP_POOL = 1
VERIFY_SWEEP_RANGE_S = 1.0
# The sweep searches two hundred offsets, whose largest coincidence is
# about 3.3 noise units; a single window at 4.5 can still be a fluke, so a
# stretch has to be at least two windows long to be reported.
VERIFY_SWEEP_Z = 4.5
VERIFY_STRETCH_WINDOWS = 2
# Past this the dub is visibly out of lip sync: sound 45 ms early is where
# viewers start to see it (ITU-R BT.1359; late sound is tolerated to about
# 125 ms), and the stricter bound is held both ways. A stretch of the sweep
# this far out, over more than one window, is reported.
AUDIBLE_MS = 45.0
STRETCH_MS = 45.0

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


# The difference of a track's stereo downmix, L - R. Dialogue is mixed to
# the centre, so it cancels here, and what is left is the width of the
# music and effects -- the part a dub shares with the original, with
# neither language's voices over it. Measured on a real pair (one film's
# Japanese and English 5.1): after alignment 93% of the difference's
# waveform was shared against 52% of the mono downmix's, and a 30-second
# window correlated at 31.2 noise units against 22.9. A dub mixed
# dialogue-forward, its bed ducked under the voices, keeps that bed here at
# full strength relative to itself.
SIDE_FILTER = "aformat=channel_layouts=stereo,pan=mono|c0=0.5*c0-0.5*c1"
# A near-mono mix has little in its difference but codec noise, whose
# onsets match nothing; the band is kept only where the difference carries
# at least this fraction of the mono level.
SIDE_MIN_RATIO = 0.05


def _frame_energies(blocks, hop: int, on_samples: Optional[Callable[[int], None]] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """RMS of every ``hop``-sample frame of a stream of mono blocks, and the
    whole-frame samples themselves, block by block, for the band pass."""
    energies: List[np.ndarray] = []
    kept: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    seen = 0
    for block in blocks:
        if carry.size:
            block = np.concatenate([carry, block])
        frames = len(block) // hop
        if frames:
            shaped = block[: frames * hop].reshape(frames, hop).astype(np.float64)
            energies.append(np.sqrt(np.mean(shaped * shaped, axis=1) + 1e-12).astype(np.float32))
        kept.append(block[: frames * hop])
        carry = block[frames * hop:]
        seen += frames * hop
        if on_samples:
            on_samples(seen)
    return (np.concatenate(energies) if energies else np.zeros(0, dtype=np.float32)), kept


def _side_energy(
    path: str, track: int, rate: int, token: Optional[CancellationToken], lead_s: float,
) -> np.ndarray:
    """Frame energy of the track's stereo difference (see SIDE_FILTER)."""
    blocks = stream_audio(path, rate, track=track, token=token, block_s=30.0, audio_filter=SIDE_FILTER, lead_s=lead_s)
    energy, _ = _frame_energies(blocks, ENVELOPE_HOP)
    return energy


def build_envelope(
    path: str,
    track: int = 0,
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[float], None]] = None,
    expected_duration_s: Optional[float] = None,
    speed: float = 1.0,
    lead_s: Optional[float] = None,
    channels: Optional[int] = None,
) -> TrackEnvelope:
    """Decode a whole track once, keeping only its 500 Hz energy curve.

    The curve is on the file's own clock (see ``audio_lead_s``): time zero
    is the file's, not the audio stream's first sample, which is the time
    every seek, picture frame and mux uses. A track whose audio starts
    after its picture is padded at the front by the difference.

    Args:
        speed: decode at this playback speed. Asking ffmpeg for
            ANALYSIS_SR * speed and reading the samples as ANALYSIS_SR is a
            time-stretch by that factor, done by the decoder's resampler.
        progress: called with 0.0-1.0 as the file is read, when the duration
            is known.
        lead_s: where the stream's first sample sits on the file's clock;
            probed when not given.
        channels: the track's channel count; probed when not given. A
            track of two or more channels also gets the ``side`` band (see
            SIDE_FILTER), decoded alongside the mono one.
    """
    if lead_s is None or channels is None:
        info = probe(path, token)
        if lead_s is None:
            lead_s = audio_lead_s(info, track)
        if channels is None:
            stream = info.audio_tracks[track] if track < len(info.audio_tracks) else None
            channels = int((stream.channels if stream else None) or info.channels or 1)
    rate = max(1, int(round(ANALYSIS_SR * speed)))
    hop = ENVELOPE_HOP

    side: dict = {}
    helper: Optional[threading.Thread] = None
    if channels >= 2:
        def read_side() -> None:
            try:
                side["energy"] = _side_energy(path, track, rate, token, lead_s)
            except (MediaError, Cancelled) as exc:
                side["error"] = exc
        helper = threading.Thread(target=read_side, name="side-band", daemon=True)
        helper.start()

    bands = _BandFrames()

    def mono_blocks():
        for block in stream_audio(path, rate, track=track, token=token, block_s=30.0, lead_s=lead_s):
            yield block

    def on_samples(seen: int) -> None:
        if progress and expected_duration_s:
            progress(min(1.0, seen / rate / expected_duration_s))

    energies: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    seen = 0
    for block in mono_blocks():
        if carry.size:
            block = np.concatenate([carry, block])
        frames = len(block) // hop
        if frames:
            shaped = block[: frames * hop].reshape(frames, hop).astype(np.float64)
            energies.append(np.sqrt(np.mean(shaped * shaped, axis=1) + 1e-12).astype(np.float32))
        bands.push(block[: frames * hop])
        carry = block[frames * hop:]
        seen += len(block) - len(carry)
        on_samples(seen)
    if helper is not None:
        helper.join()
    if token is not None and token.cancelled:
        raise Cancelled("operation cancelled")
    if not energies:
        raise MediaError(f"No audio decoded from {os.path.basename(path)}")
    energy = np.concatenate(energies)
    band_energy = bands.finish(len(energy))
    difference = side.get("energy")
    if difference is not None and len(difference):
        if len(difference) < len(energy):
            difference = np.concatenate([difference, np.full(len(energy) - len(difference), difference[-1], dtype=np.float32)])
        difference = difference[: len(energy)]
        level = float(np.sqrt(np.mean(energy.astype(np.float64) ** 2)))
        if float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))) >= SIDE_MIN_RATIO * level:
            band_energy["side"] = difference
    return TrackEnvelope.from_energy(path, track, energy, speed, band_energy)


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
    follows_drift: bool = False
    """One step of a staircase that follows a drifting offset (see
    ``_follow_drift``); never merged back into its neighbours."""
    start_cut: Optional[float] = None
    end_cut: Optional[float] = None
    """The picture cut an edge was put on (see ``_snap_to_picture``). A
    later pass that fits the edge again on the sound moves it within its
    bracket; the picture's frame stands, so the edge is put back."""
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
            "rateConfirmed": self.rate_confirmed,
            "segments": [s.to_dict() for s in self.segments],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "error": self.error,
            "filledS": self.filled_s,
            "summary": self.summary(),
            "timeline": self.timeline,
            "voicePieces": [piece.to_dict() for piece in self.voice_pieces],
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


def _duration(seconds: float) -> str:
    # To the tenth first: 359.96 s is 6m 00.0s, not 5m 60.0s.
    tenths = int(round(seconds * 10.0))
    if tenths >= 600:
        minutes, rest = divmod(tenths, 600)
        return f"{minutes}m {rest / 10.0:04.1f}s"
    return f"{tenths / 10.0:.1f}s"


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
        picture: Optional[PictureCuts] = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.token = token
        self.log = log or (lambda _message: None)
        self.frame_s = frame_s
        """One video frame, when the video's rate is known: the unit a
        picture trim comes in, which tells a trim from a music artefact."""
        self.picture = picture
        """The video's shot changes, read on demand, for putting cuts on the
        frame (see ``shots``); None where there is no picture to ask."""
        self.evidence: List[Tuple[str, float]] = []
        """(band, noise units) of every dense reading taken at the end, for
        saying what the sync rests on (see ``_evidence_note``)."""
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
        sharp = ["full"] + (["side"] if "side" in self.shared_bands else [])
        full = self.agreement(offset_s, lo_s, hi_s, bands=sharp)
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
        others = [f for f in found if f not in effects]
        # Only once the stretch is one level: the wide first measurement
        # reads across its steps, where "the rest" is the other side of one.
        if effects and others and not wide:
            said = float(np.median([f[1] for f in effects]))
            rest = float(np.median([f[1] for f in others]))
            if abs(said - rest) >= SPEECH_ARBITER_S and self.speech_choice(block.start_s, block.end_s, said, rest) == rest:
                self.log(
                    f"  {_clock(block.start_s)} - {_clock(block.end_s)}: the effects read {said:+.3f}s, the rest "
                    f"{rest:+.3f}s, and the dialogue goes with the rest"
                )
                effects = []
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

    def span_scores(
        self, lo_s: float, hi_s: float, centre_s: float, reach_s: float = SETTLE_BASELINE_S,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The whole span [lo_s, hi_s) correlated at every offset within
        ``reach_s`` of ``centre_s``: (offsets, noise units summed over the
        shared bands, the high band discounted as in ``_read``).

        A quiet scene gives each ten-second reading a peak at the noise,
        and their median can land anywhere among those peaks; the whole
        stretch adds up what every window saw at the one offset that is
        right, and only there."""
        rate = ENVELOPE_RATE
        shift = int(round(centre_s * rate))
        reach = int(round(reach_s * rate))
        offsets = (shift + np.arange(-reach, reach + 1)) / rate
        total = np.zeros(len(offsets))
        # Only the part of the span the dub covers at every offset tried:
        # at the dub's first or last seconds some offsets would read past
        # its ends and score nothing, which is not the same as scoring low.
        lo = max(0, int(round(lo_s * rate)), reach - shift)
        hi = min(len(self.primary.onset), int(round(hi_s * rate)), len(self.secondary.onset) - shift - reach - 1)
        if hi - lo < rate:
            return offsets, total
        for band in self.shared_bands:
            template = self.onsets("primary", band)[lo:hi]
            secondary = self.onsets("secondary", band)
            first = lo + shift - reach
            start = max(0, first)
            stop = min(len(secondary), hi + shift + reach + 1)
            if stop - start < len(template):
                continue
            row = ncc_lags(template, secondary[start:stop])
            units = _noise_units(row, np.ones(len(row), dtype=bool)).astype(np.float64)
            if band == "high":
                units *= EFFECTS_DISCOUNT
            skip = start - first
            count = min(len(units), len(total) - skip)
            total[skip : skip + count] += units[:count]
        return offsets, total

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
            if abs(levels[target] - levels[index]) >= SPEECH_ARBITER_S * 1000.0 and self.speech_choice(
                run[0], run[1] + POLISH_WINDOW_S, levels[index] / 1000.0, levels[target] / 1000.0,
            ) == levels[index] / 1000.0:
                self.log(
                    f"  the music sits at {levels[index] / 1000.0:+.3f}s around {_clock(run[0])}, the effects at "
                    f"{levels[target] / 1000.0:+.3f}s; the dialogue goes with the music there"
                )
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

    def speech_choice(self, lo_s: float, hi_s: float, first_s: float, second_s: float) -> Optional[float]:
        """Which of two offsets the dub's lines follow across [lo_s, hi_s),
        or None when the speech does not say (see SPEECH_ARBITER_S)."""
        primary, secondary = self.primary, self.secondary
        if primary.speed != 1.0 or secondary.speed != 1.0 or hi_s - lo_s < 5.0:
            return None
        if not getattr(primary, "path", None) or not getattr(secondary, "path", None):
            return None
        from .speech import SPEECH_RATE, centre_activity
        low, high, pad = min(first_s, second_s), max(first_s, second_s), 1.5
        try:
            video = centre_activity(primary.path, primary.track, lo_s, hi_s, self.token)
            dub = centre_activity(secondary.path, secondary.track, lo_s + low - pad, hi_s + high + pad, self.token)
        except MediaError:
            return None
        row = ncc_lags(video, dub)
        if row.size < 3:
            return None
        offsets = low - pad + np.arange(len(row)) / SPEECH_RATE
        peak = int(np.argmax(row))
        away = np.abs(offsets - offsets[peak]) > 0.15
        if away.sum() < 10:
            return None
        spread = float(np.std(row[away])) + 1e-9
        z = [(float(row[int(np.argmin(np.abs(offsets - level)))]) - float(np.median(row[away]))) / spread
             for level in (first_s, second_s)]
        best = int(np.argmax(z))
        if z[best] >= SPEECH_PREFER_Z and z[best] - z[1 - best] >= SPEECH_PREFER_MARGIN:
            return (first_s, second_s)[best]
        return None

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
    dub_rate: Optional[float] = None,
    draft: Optional[Callable[[DubSyncPlan], None]] = None,
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
        dub_rate: the frame rate the dub was mastered at, when the user
            knows it; with the video's own rate this fixes the speed, and
            the audio is not asked. Ignored when ``speed`` is given.
        draft: told the plan as it stands after each stage -- coarse
            stretches, then measured, then with the cuts placed, then with
            the gaps searched -- so a viewer can watch the dub being laid
            onto the video. Each draft is a whole plan, with the parts of
            the video not yet placed shown as fills marked ``not placed
            yet``; nothing in it is final until the plan is returned.
    """
    plan = DubSyncPlan(video_path, dub_path, video_track, dub_track)
    say = log or (lambda _m: None)

    def report(percent: float, stage: str) -> None:
        if progress:
            progress(int(max(0, min(100, percent))), stage)

    def show(blocks: List[Block], speed_now: float) -> None:
        if draft is None or not blocks:
            return
        sketch = DubSyncPlan(
            video_path, dub_path, video_track, dub_track, speed=speed_now,
            video_duration_s=plan.video_duration_s, dub_duration_s=plan.dub_duration_s,
            video_fps=plan.video_fps,
            dub_rate=round(speed_now * plan.video_fps, 6) if plan.video_fps else None,
            rate_confirmed=plan.rate_confirmed,
            segments=_draft_segments(blocks, plan.video_duration_s, plan.dub_duration_s),
        )
        draft(sketch)

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
            plan.video_duration_s, lead_s=audio_lead_s(video_info, video_track),
        )
        report(20, "reading the dub")
        secondary_native = build_envelope(
            dub_path, dub_track, token,
            lambda f: report(20 + 18 * f, "reading the dub"),
            plan.dub_duration_s, lead_s=audio_lead_s(dub_info, dub_track),
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

        video_rate = exact_rate(float(video_info.fps)) if video_info.fps else None
        if speed is None and dub_rate is not None:
            mastering = exact_rate(float(dub_rate))
            if video_rate is None:
                say(f"the dub was said to be mastered at {_format_fps(float(dub_rate))} fps, but the video carries no frame rate; the audio decides")
            elif mastering is None:
                say(f"{dub_rate} is not a standard frame rate; the audio decides")
            else:
                speed = float(mastering / video_rate)
                plan.video_fps = float(video_rate)
                say(
                    f"video is {_format_fps(plan.video_fps)} fps, dub said to be mastered at "
                    f"{_format_fps(float(mastering))} fps: played at {speed:.6f}x"
                )
        secondary = secondary_native.at_speed(speed) if speed else secondary_native
        frame_s = 1.0 / float(video_rate) if video_rate is not None else None
        # The video's shot changes, for putting cuts on the frame (see
        # ``shots``). Given to every aligner from the first measurement on:
        # the first time an edge is placed it is placed in the widest
        # bracket it will ever have, and that is where the picture has to
        # be asked.
        picture = PictureCuts(video_path, token, say) if video_info.has_video else None
        aligner = _Aligner(primary, secondary, token, say, frame_s, picture)

        if speed is None and video_info.fps:
            # Verify the frame rate before anything else trusts an offset. The
            # video's own metadata names its rate, the dub's mastering rate is
            # one of the standard ones, so the possible speedups are a short
            # list: ask each on the audio itself, before the coarse pass could
            # misread a rate mismatch as a flood of cuts. A bare audio file
            # carries no rate; for that the symptom-driven passes below stand.
            if video_rate is not None:
                plan.video_fps = float(video_rate)
                report(38, "checking the frame rate")
                say(f"video is {_format_fps(plan.video_fps)} fps; checking the dub's rate")
                trials: dict = {}
                trial = _try_speeds(
                    aligner, primary, secondary_native, search_s, say,
                    candidates=_fps_speed_candidates(video_rate), trials=trials,
                )
                if trial is not None:
                    secondary = trial
                    aligner = _Aligner(primary, secondary, token, say, frame_s, picture)
                    plan.rate_confirmed = True
                elif trials.get(1.0, 0.0) >= SPEED_MIN_Z:
                    # The files' own speed correlates sharply: the rates match.
                    plan.rate_confirmed = True
                else:
                    # Neither the files' own speed nor any conversion the
                    # video's rate allows correlated sharply. Everything below
                    # may still find the drift and correct it; if not, the
                    # plan is made at the files' own speed, and the reader
                    # must know that was never confirmed.
                    plan.rate_confirmed = False
                    tried = ", ".join(f"{spd:.6f}x {peak:.1f}" for spd, peak in sorted(trials.items()))
                    say(f"  the dub's rate could not be confirmed from the audio (peaks: {tried})")

        report(40, "finding the offsets")
        grid, blocks = _coarse_blocks(aligner, search_s, report, say)
        show(blocks, secondary.speed)

        if speed is None and _coverage(blocks, primary.duration_s) <= SPEED_RETRY_FRACTION:
            # Too little to trust as they are. A rate conversion looks
            # exactly like this -- the peak smears across the window -- and
            # it has a short list of possible values, so try them.
            trial = _try_speeds(aligner, primary, secondary_native, search_s, say)
            if trial is not None:
                secondary = trial
                aligner = _Aligner(primary, secondary, token, say, frame_s, picture)
                grid, blocks = _coarse_blocks(aligner, search_s, report, say)
                show(blocks, secondary.speed)

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
                show(blocks, secondary.speed)
        plan.speed = secondary.speed
        if plan.video_fps is not None:
            # The rate the dub was mastered at, implied by the final speed:
            # the dub runs at ``speed`` times the video's clock.
            plan.dub_rate = round(plan.speed * plan.video_fps, 6)
            if plan.rate_confirmed is False and abs(plan.speed - 1.0) > 1e-9:
                # The drift found what the trial could not.
                plan.rate_confirmed = True
            if plan.rate_confirmed is False:
                plan.warnings.append(
                    f"the video is {_format_fps(plan.video_fps)} fps and the dub's frame rate could not be "
                    f"confirmed from the audio, so it was taken to be the same; if the dub was mastered at "
                    f"another rate (23.976 against 24 is the common case), set the dub's rate and sync again"
                )

        if _coverage(blocks, primary.duration_s) < WIDE_PASS_COVERAGE:
            wide = _wide_blocks(aligner, lambda f: report(56 + 4 * f, "searching the whole dub"), say, blocks)
            if wide:
                blocks = _credible(_measure_and_split(aligner, _merge_blocks(blocks, wide)), aligner)
                say(
                    f"with the wide pass: {len(blocks)} credible stretch(es) covering "
                    f"{100 * _coverage(blocks, primary.duration_s):.0f}%"
                )
                show(blocks, secondary.speed)

        if not blocks:
            plan.error = "No part of the dub could be matched to the video"
            return plan

        report(60, "placing the cuts")
        blocks = _refine(
            aligner, grid, blocks, primary.duration_s, secondary.duration_s, report, plan.warnings,
            draft=lambda current: show(current, secondary.speed), notes=plan.notes,
        )
        said = _evidence_note(aligner.evidence)
        if said:
            plan.notes.insert(0, said)
            say(said)
        if not blocks:
            plan.error = "No part of the dub could be matched to the video"
            return plan

        report(85, "assembling the plan")
        if fill_gain_db is None:
            fill_gain_db = _fill_gain_db(primary, secondary, blocks)
        plan.fill_gain_db = float(fill_gain_db)
        doubt: List[Tuple[float, float]] = []
        plan.segments = _assemble(
            blocks, primary.duration_s, secondary.duration_s, primary, secondary,
            keep_unmatched_dub, plan.warnings, plan.notes, aligner, doubt,
        )
        plan.doubt_s = sum(hi - lo for lo, hi in doubt)
        for lo, hi in sorted(doubt, key=lambda span: span[0] - span[1])[:REVIEW_DOUBT_MAX]:
            plan.warnings.append(
                f"the cut near {_clock((lo + hi) / 2.0)} could only be placed to within {hi - lo:.1f}s, so the "
                f"original plays across {_clock(lo)} - {_clock(hi)} rather than risk the wrong scene -- check it"
            )
        if not plan.dub_segments:
            plan.error = "No part of the dub could be placed on the video"
            return plan
        if picture is not None and picture.spent_s > 0.0:
            say(f"read {picture.spent_s / 60.0:.1f} min of the picture to put cuts on the frame")
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
    trial_aligner = _Aligner(primary, trial, token, say, aligner.frame_s, aligner.picture)
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
    trials: Optional[dict] = None,
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
    pool = SPEED_TRIAL_POOL
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
    if trials is not None:
        trials[aligner.secondary.speed] = baseline

    best: Optional[Tuple[float, TrackEnvelope]] = None
    for candidate, _mastering_rate in candidates:
        trial = secondary_native.at_speed(candidate)
        peak = trial_peak(trial)
        say(f"  trying {candidate:.6f}x: peak {peak:.1f} noise units")
        if trials is not None:
            trials[candidate] = peak
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
    draft: Optional[Callable[[List[Block]], None]] = None,
    notes: Optional[List[str]] = None,
) -> List[Block]:
    """From coarse stretches to exact ones.

    Each stretch is measured at 2 ms from many places inside it, which also
    finds any step the coarse pass was too blunt to see. Each edge is then
    placed at 2 ms from the agreement curve, inside the bracket the coarse
    windows left it in. Finally every gap long enough to hold one is searched
    for a stretch of dub the coarse windows straddled, and any found is
    placed the same way. ``draft`` sees the stretches after each of those.
    """
    show = draft or (lambda _blocks: None)
    report(60, "measuring the offsets")
    blocks = _credible(_measure_and_split(aligner, coarse), aligner)
    show(blocks)

    report(70, "placing the cuts")
    _place_all_edges(aligner, blocks, primary_s, secondary_s)
    show(blocks)

    for round_index in range(RECOVER_ROUNDS):
        report(74 + 3 * round_index, "looking for dub inside the gaps")
        recovered = _recover_from_gaps(aligner, blocks, primary_s, secondary_s)
        if not recovered:
            break
        blocks = _credible(
            _measure_and_split(aligner, sorted(blocks + recovered, key=lambda b: b.start_s)), aligner
        )
        _place_all_edges(aligner, blocks, primary_s, secondary_s)
        show(blocks)

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
    settled = []
    for index, block in enumerate(blocks):
        # The offset was measured before the edges moved; measure it again on
        # exactly the material that is now inside the stretch.
        block.offset_s, block.match, drift = aligner.polish_offset(block)
        if _settle(aligner, block):
            settled.append(index)
    for index in settled:
        # An offset the whole stretch moved had its edges placed on the
        # agreement at the old one.
        _place_edge(aligner, blocks[index - 1] if index else None, blocks[index], primary_s, secondary_s)
        _place_edge(aligner, blocks[index], blocks[index + 1] if index + 1 < len(blocks) else None, primary_s, secondary_s)
    blocks = [b for b in blocks if b.length_s >= MIN_BLOCK_S]
    # A stretch that is two levels in turn, which the readings missed.
    blocks = _split_levels(aligner, blocks)
    # No part of the dub played twice: where two stretches claim the same
    # dub, the one that agrees with it keeps it.
    blocks = _resolve_reuse(aligner, blocks, primary_s, secondary_s)
    # A stretch whose offset walks is followed step by step, so its ends
    # are as much in sync as its middle; the note says where. (A slope the
    # readings show without lying on one line is not a drift -- on a real
    # pair it was a stretch whose start still reached over an insert the
    # assembly then cut it at -- and is not reported as one.)
    blocks = _follow_drift(aligner, blocks, notes)
    _follow_effects(aligner, blocks)
    # A gap the dub has the material for, between two offsets, is followed
    # through second by second.
    blocks = _trace_gaps(aligner, blocks, secondary_s, notes, warnings)
    show(blocks)
    return blocks


DRAFT_NOTE = "not placed yet"


def _draft_segments(blocks: List[Block], video_s: float, dub_s: float) -> List[Segment]:
    """The stretches as they stand, as a whole plan's worth of pieces: a
    dub piece per stretch, and the rest of the video as fills marked not
    placed yet. Overlaps -- a stretch whose edge has not been placed yet
    reaching into the next -- are cut at the earlier stretch's end."""
    pieces: List[Segment] = []
    cursor = 0.0
    for block in sorted(blocks, key=lambda b: b.start_s):
        start = max(block.start_s, cursor)
        end = min(block.end_s, video_s)
        if end - start < 0.01:
            continue
        if start > cursor:
            pieces.append(Segment("fill", cursor, start, cursor, note=DRAFT_NOTE))
        pieces.append(Segment(
            "dub", start, end, start + block.offset_s, offset_s=block.offset_s, match=block.match,
            uncertainty_s=max(block.start_uncertainty_s, 0.0),
        ))
        cursor = end
    if cursor < video_s:
        pieces.append(Segment("fill", cursor, video_s, cursor, note=DRAFT_NOTE))
    return pieces


# --- following a drift inside a stretch ------------------------------------

# A stretch whose offset walks at least this far from one end to the other
# is followed rather than held at one offset: a dub conformed piece by
# piece can have one scene at a slightly different speed (a two-minute
# opening at +1.13 ms/s on a real episode), and held at its median offset
# such a scene is 70 ms out at its ends -- enough to see on the lips.
DRIFT_SPAN_MS = 4.0
# ... in steps of at most this, so no part is further than half of it from
# where the readings put it, ...
DRIFT_STEP_MS = 2.0
# ... read this densely: windows this long, this far apart.
DRIFT_READING_WINDOW_S = 6.0
DRIFT_READING_SPACING_S = 2.0
# A walk is only believed when it is this many times the readings' own
# scatter about the line, and the line fits this much better than one offset.
DRIFT_OVER_SCATTER = 6.0
DRIFT_FIT_GAIN = 2.0
# ... and the readings sit this close to the line: a ramp's readings scatter
# a millisecond or two about it, a run of flat pieces a trim apart sits tens
# of milliseconds off any one line near every step.
DRIFT_MAX_SCATTER_MS = 3.0


def _theil_sen(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Slope and intercept of the median line through the points."""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else 0.0
    return slope, float(np.median(y - slope * x))


# Stretches this close together, at offsets within NEAR_OFFSET_S, are read
# as one scene when a drift is looked for.
DRIFT_GROUP_GAP_S = 20.0
# How far a drift's line may reach into the flat stretch beside it, to the
# corner where the two meet.
DRIFT_CORNER_REACH_S = 10.0
# The most stretches one drift is looked for across, which bounds the
# search on a scene cut into many pieces.
DRIFT_MAX_RUN = 16


BAND_WORDS = {
    "full": "the full mix", "side": "the stereo difference (no dialogue)",
    "low": "the bass band", "high": "the effects band",
}


def _evidence_note(evidence: Sequence[Tuple[str, float]]) -> Optional[str]:
    """What the sync rests on, from the dense readings: which band read
    each window best, and how strongly the tracks agreed."""
    if not evidence:
        return None
    found = [(band, z) for band, z in evidence if z >= READING_Z]
    if not found:
        return f"the two tracks shared too little to read in {len(evidence)} dense readings"
    counts: Dict[str, int] = {}
    for band, _z in found:
        counts[band] = counts.get(band, 0) + 1
    parts = ", ".join(
        f"{BAND_WORDS.get(band, band)} {100.0 * n / len(found):.0f}%"
        for band, n in sorted(counts.items(), key=lambda item: -item[1])
    )
    strength = float(np.median([z for _band, z in found]))
    return (
        f"the tracks share their music and effects: {len(found)} of {len(evidence)} readings every "
        f"{DRIFT_READING_SPACING_S:.0f}s found them (read on {parts}), typically {strength:.0f} noise units strong"
    )


def _drift_groups(blocks: List[Block]) -> List[List[Block]]:
    """Runs of stretches a few frames apart in offset and at most
    DRIFT_GROUP_GAP_S apart in time: where a drift was cut into flat pieces
    before it was recognised -- at steps the readings took for trims, around
    passages that did not correlate -- the pieces are one ramp and are read
    as one."""
    groups: List[List[Block]] = []
    for block in blocks:
        if (
            groups
            and block.start_s - groups[-1][-1].end_s <= DRIFT_GROUP_GAP_S
            and abs(block.offset_s - groups[-1][-1].offset_s) <= NEAR_OFFSET_S
        ):
            groups[-1].append(block)
        else:
            groups.append([block])
    return groups


def _follow_drift(aligner: _Aligner, blocks: List[Block], notes: Optional[List[str]] = None) -> List[Block]:
    """Replace every run of stretches whose offset walks by a staircase that
    follows it.

    Stretches a few frames apart and close together in time are read
    together (see ``_drift_groups``), every DRIFT_READING_SPACING_S, each
    window searched around the offset of the stretch it falls in. Within
    each group the longest run of stretches whose readings lie on one line
    -- walking DRIFT_SPAN_MS or more, clearly above their scatter about it,
    fitting clearly better than one offset, and scattering no more than
    DRIFT_MAX_SCATTER_MS -- is a drift, and the stretches either side are
    looked at the same way. A run of flat pieces a trim or two apart is not
    a ramp: the readings near each step sit far off any one line.

    A drift is followed from its corners -- where its line meets the level
    of the stretch beside it, across the passage that did not correlate
    between them if there is one -- in pieces short enough that the line
    moves at most DRIFT_STEP_MS across each, each playing at the line's
    offset at its middle. The cuts go on the picture's shot changes where
    one is near -- a few milliseconds' jump at a shot change is inaudible --
    and are placed exactly, so nothing trims them back.
    """
    rate = ENVELOPE_RATE
    window = int(DRIFT_READING_WINDOW_S * rate)
    margin = int(round(POLISH_RANGE_S * rate))
    out: List[Block] = []

    def fit(xs: np.ndarray, ys: np.ndarray, length: float) -> Optional[Tuple[float, float, float]]:
        if len(xs) < 5 or xs[-1] - xs[0] < 0.6 * length:
            return None
        if len(xs) > 200:
            pick = np.linspace(0, len(xs) - 1, 200).astype(int)
            xs, ys = xs[pick], ys[pick]
        slope, intercept = _theil_sen(xs, ys)
        around_line = 1.4826 * float(np.median(np.abs(ys - (intercept + slope * xs))))
        around_one = 1.4826 * float(np.median(np.abs(ys - np.median(ys))))
        walk = abs(slope) * length
        if (
            walk < DRIFT_SPAN_MS or walk < DRIFT_OVER_SCATTER * around_line
            or around_one < DRIFT_FIT_GAIN * max(around_line, 0.1) or around_line > DRIFT_MAX_SCATTER_MS
        ):
            return None
        return slope, intercept, walk

    def staircase(run: List[Block], slope: float, intercept: float, walk: float, begin: float, finish: float) -> List[Block]:
        first, last = run[0], run[-1]
        length = finish - begin
        step_s = max(MIN_BLOCK_S, DRIFT_STEP_MS / abs(slope))
        count = max(2, int(math.ceil(length / step_s)))
        ideal = [begin + length * k / count for k in range(1, count)]
        cuts = aligner.picture.within(begin, finish) if aligner.picture is not None else None
        points: List[float] = []
        for point in ideal:
            if cuts:
                near = min(cuts, key=lambda c: abs(c - point))
                if abs(near - point) <= step_s / 2.0:
                    point = near
            if (not points or point - points[-1] >= 0.5) and finish - point >= 0.5 and point - begin >= 0.5:
                points.append(point)
        edges = [begin] + points + [finish]
        pieces = []
        for lo, hi in zip(edges, edges[1:]):
            middle = (lo + hi) / 2.0
            inside = next((b for b in run if b.start_s <= middle < b.end_s), min(run, key=lambda b: abs((b.start_s + b.end_s) / 2.0 - middle)))
            pieces.append(Block(
                lo, hi, (intercept + slope * middle) / 1000.0,
                match=inside.match, support=inside.support, windows=inside.windows,
                start_lo=lo, start_hi=lo, end_lo=hi, end_hi=hi, readings=inside.readings,
                consistency=inside.consistency, spread_ms=inside.spread_ms, effects=inside.effects,
                follows_drift=True,
            ))
        if begin == first.start_s:
            pieces[0].start_lo, pieces[0].start_hi, pieces[0].start_cut = first.start_lo, first.start_hi, first.start_cut
        if finish == last.end_s:
            pieces[-1].end_lo, pieces[-1].end_hi, pieces[-1].end_cut = last.end_lo, last.end_hi, last.end_cut
        said = (
            f"the dub drifts by {slope:+.2f} ms/s across {_clock(begin)} - {_clock(finish)} "
            f"({walk:.0f} ms end to end); followed in {len(pieces)} steps of {DRIFT_STEP_MS:.0f} ms or less"
        )
        aligner.log(f"  {said}")
        if notes is not None:
            notes.append(said)
        return pieces

    for group in _drift_groups(blocks):
        # Every reading of the group, once, labelled with its stretch.
        xs: List[float] = []
        ys: List[float] = []
        owner: List[int] = []
        for index, block in enumerate(group):
            start = block.start_s
            while start + DRIFT_READING_WINDOW_S <= block.end_s + 1e-9:
                reading = aligner._read(int(round(start * rate)), window, int(round(block.offset_s * rate)), margin)
                if reading is not None:
                    aligner.evidence.append((reading[2], reading[3]))
                if reading is not None and reading[3] >= READING_Z:
                    xs.append(start + DRIFT_READING_WINDOW_S / 2.0)
                    ys.append(reading[0] * 1000.0)
                    owner.append(index)
                start += DRIFT_READING_SPACING_S
        x, y, who = np.asarray(xs), np.asarray(ys), np.asarray(owner, dtype=int)

        def settle(lo: int, hi: int) -> List[Block]:
            """Stretches group[lo:hi], with the longest ramp among them followed."""
            best: Optional[Tuple[float, int, int, Tuple[float, float, float]]] = None
            for i in range(lo, hi):
                for j in range(i, min(hi, i + DRIFT_MAX_RUN)):
                    length = group[j].end_s - group[i].start_s
                    if length < 2 * DRIFT_READING_WINDOW_S or (best is not None and length <= best[0]):
                        continue
                    chosen = (who >= i) & (who <= j)
                    found = fit(x[chosen], y[chosen], length)
                    if found is not None:
                        best = (length, i, j, found)
            if best is None:
                return list(group[lo:hi])
            _length, i, j, (slope, intercept, walk) = best
            # Where the drift really begins and ends: where its line meets
            # the level of the flat stretch beside it. Between them lies, as
            # a rule, a passage that did not correlate -- the ramp was not
            # recognised there -- and the ramp covers it, from its corner;
            # into the flat stretch itself it reaches only a little way.
            begin, finish = group[i].start_s, group[j].end_s
            if i > lo:
                before = group[i - 1]
                corner = (before.offset_s * 1000.0 - intercept) / slope
                begin = min(begin, max(corner, before.end_s - DRIFT_CORNER_REACH_S, before.start_s + MIN_BLOCK_S))
                if begin < before.end_s:
                    before.end_s = before.end_lo = before.end_hi = begin
            if j + 1 < hi:
                after = group[j + 1]
                corner = (after.offset_s * 1000.0 - intercept) / slope
                finish = max(finish, min(corner, after.start_s + DRIFT_CORNER_REACH_S, after.end_s - MIN_BLOCK_S))
                if finish > after.start_s:
                    after.start_s = after.start_lo = after.start_hi = finish
            return settle(lo, i) + staircase(group[i : j + 1], slope, intercept, walk, begin, finish) + settle(j + 1, hi)

        out.extend(settle(0, len(group)))
    return out


# A stretch whose dub reaches back over this much of the dub the stretch
# before it played is playing that dub twice; a rewind longer than
# WIDE_EPISODE_S is a dub made of episodes in another order, and allowed.
REUSE_MIN_S = 0.25


def _resolve_reuse(aligner: _Aligner, blocks: List[Block], primary_s: float, secondary_s: float) -> List[Block]:
    """Play no part of the dub twice.

    A dub of the same episode runs forward: every moment of it belongs at
    one moment of the video. Where two neighbouring stretches both read the
    same stretch of dub -- one of them resting on a cue heard twice, a shot
    the video shows again, a weak match -- the voices there are right at
    most once. On a real 1.5-hour pair, 21.6 s of dub played at two places
    46 s apart. The material goes to the stretch that agrees with it
    better, measured on exactly the overlapping span of each; the other is
    cut back, and the cut between them placed afresh (on the picture where
    it can be) -- never back over what was taken from it.

    Two stretches that overlap on the video are not this: that is an edge
    the assembly settles.
    """
    out = sorted(blocks, key=lambda b: b.start_s)
    for _attempt in range(4 * len(out) + 1):
        changed = False
        for index in range(len(out) - 1):
            a, b = out[index], out[index + 1]
            if b.start_s < a.end_s - 0.05:
                continue
            a_end = a.end_s + a.offset_s
            b_start = b.start_s + b.offset_s
            overlap = a_end - b_start
            if overlap < REUSE_MIN_S or overlap > WIDE_EPISODE_S:
                continue
            overlap = min(overlap, a.length_s, b.length_s)
            a_at, b_at = a.end_s - overlap, b.start_s
            keep_a = aligner.best_agreement(a.offset_s, a.end_s - overlap, a.end_s)
            keep_b = aligner.best_agreement(b.offset_s, b.start_s, b.start_s + overlap)
            if keep_a >= keep_b:
                loser, where = b, "after"
                b.start_s += overlap
                # The edge may move on into the stretch, not back over the
                # dub it no longer plays.
                b.start_lo, b.start_hi, b.start_cut = b.start_s, b.start_s + 1.0, None
            else:
                loser, where = a, "before"
                a.end_s -= overlap
                a.end_lo, a.end_hi, a.end_cut = a.end_s - 1.0, a.end_s, None
            aligner.log(
                f"  the dub's {_clock(b_start)} - {_clock(b_start + overlap)} was placed twice, at {_clock(a_at)} and at "
                f"{_clock(b_at)}; it stays where it agrees better ({keep_a:.3f} against {keep_b:.3f}), and the "
                f"stretch {where} is cut back {overlap:.1f}s"
            )
            if loser.length_s < MIN_BLOCK_S:
                out.remove(loser)
            else:
                _place_edge(aligner, a, b, primary_s, secondary_s)
                # Whatever the placement did, the two no longer share dub.
                if loser is b:
                    b.start_s = max(b.start_s, a.end_s + a.offset_s - b.offset_s)
                else:
                    a.end_s = min(a.end_s, b.start_s + b.offset_s - a.offset_s)
            changed = True
            break
        if not changed:
            break
    return out


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

    The mirror case is asked too: a stretch the effects placed, between
    stretches the music and dialogue placed at another offset a few
    frames away, is asked which level its own lines follow. The effects
    follow the picture on most dubs, but on one real pair they had
    stepped twenty seconds early and sat 110 ms from the lines, which a
    dialogue-arbitrated neighbour had followed (see BANDS); a stretch
    re-measured after a gap search read only the effects and came out at
    the wrong level, 109 ms from its neighbour's. Only the dialogue can
    say, and it decides only past the distance it can tell (see
    SPEECH_ARBITER_S); a frame or two either way is below what it
    resolves.
    """
    for index, block in enumerate(blocks):
        for other in (blocks[index - 1] if index else None, blocks[index + 1] if index + 1 < len(blocks) else None):
            if other is None or other.effects == block.effects:
                continue
            apart = abs(other.offset_s - block.offset_s)
            touching = min(abs(other.end_s - block.start_s), abs(other.start_s - block.end_s)) <= FOLLOW_EFFECTS_GAP_S
            if not touching or apart <= SAME_OFFSET_S or apart > FOLLOW_EFFECTS_STEP_S:
                continue
            if block.effects:
                # The effects placed this stretch, the music and dialogue
                # the neighbour: the dialogue is asked which level the
                # stretch's own lines follow, and wins only when it
                # clearly prefers the neighbour's.
                if apart >= SPEECH_ARBITER_S and aligner.speech_choice(
                    block.start_s, block.end_s, block.offset_s, other.offset_s
                ) == other.offset_s:
                    aligner.log(
                        f"  the effects put {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.3f}s; "
                        f"the dialogue prefers the neighbour's {other.offset_s:+.3f}s there"
                    )
                    block.offset_s = other.offset_s
                    block.effects = False
                    break
                continue
            rate = ENVELOPE_RATE
            window = min(POLISH_WINDOW_S, block.length_s)
            positions = [block.start_s + i * (block.length_s - window) / max(1, int(block.length_s / window)) for i in range(int(block.length_s / window) + 1)]
            if aligner._effects_prefer(positions, other.offset_s * 1000.0, block.offset_s * 1000.0) and not (
                apart >= SPEECH_ARBITER_S
                and aligner.speech_choice(block.start_s, block.end_s, block.offset_s, other.offset_s) == block.offset_s
            ):
                aligner.log(
                    f"  the music put {_clock(block.start_s)} - {_clock(block.end_s)} at {block.offset_s:+.3f}s; "
                    f"the effects prefer the neighbour's {other.offset_s:+.3f}s, and the dialogue goes with the effects"
                )
                block.offset_s = other.offset_s
                block.effects = True
                break


# --- a stretch that holds two levels ------------------------------------------

# A finished stretch at least this long whose whole span correlates at a
# second offset within SPLIT_REACH_S of its own (six frames: a trim the
# music ran across), at least SPLIT_APART_S away, this many noise units
# tall and this share of the first, is asked
# second by second which level each part of it is at (see _split_levels).
SPLIT_MIN_S = 20.0
SPLIT_REACH_S = 0.25
SPLIT_APART_S = 0.02
SPLIT_SECOND_Z = 8.0
SPLIT_SECOND_SHARE = 0.35
# A part must be this long to stand on its own.
SPLIT_MIN_RUN_S = 8.0
# A part split off may itself hold two levels; asked again, this many times.
SPLIT_ROUNDS = 3


def _two_level_path(table: np.ndarray, jump: float) -> List[int]:
    """The best path through per-bin scores at a few levels, free to start
    and end at any, paying ``jump`` for every change."""
    count, bins = table.shape
    best = table[:, 0].astype(np.float64).copy()
    back = np.zeros((bins, count), dtype=int)
    for column in range(1, bins):
        top = int(np.argmax(best))
        new = np.empty(count)
        for state in range(count):
            if best[top] - jump > best[state]:
                new[state], back[column, state] = best[top] - jump, top
            else:
                new[state], back[column, state] = best[state], state
            new[state] += table[state, column]
        best = new
    state = int(np.argmax(best))
    path = [state]
    for column in range(bins - 1, 0, -1):
        state = int(back[column, state])
        path.append(state)
    return path[::-1]


def _split_levels(aligner: _Aligner, blocks: List[Block]) -> List[Block]:
    """Split every finished stretch that is really two levels in turn.

    A dub conformed scene by scene can step a few frames and back again
    every half minute; the readings that split stretches are ten seconds
    long and a few seconds apart, and where they straddle the steps they
    see one level or a blur, so which steps they find depends on where
    they happen to fall. Measured on a real pair: a stretch 146 s long at
    one offset was, per every band, 52 ms away for 96 s of it -- the dub
    early on the lips -- and the same stretch, measured with its start
    moved by four seconds, split correctly. The whole stretch sees both
    levels at once, as two peaks; where it does, each second is scored at
    each (see _TraceScores) and one path through them says which level
    each part is at. A part must stand clearly above the noise and above
    the other level over its length, and the effects, where they clearly
    prefer the other level, decide (see BANDS); otherwise it keeps the
    stretch's own offset."""
    out: List[Block] = list(blocks)
    for _round in range(SPLIT_ROUNDS):
        split: List[Block] = []
        changed = False
        for block in out:
            pieces = _split_block_levels(aligner, block)
            split.extend(pieces if pieces else [block])
            changed = changed or bool(pieces)
        out = split
        if not changed:
            break
    return out


def _split_block_levels(aligner: _Aligner, block: Block) -> Optional[List[Block]]:
    if block.follows_drift or block.length_s < SPLIT_MIN_S:
        return None
    offsets, total = aligner.span_scores(block.start_s, block.end_s, block.offset_s)
    near = np.abs(offsets - block.offset_s) <= SPLIT_REACH_S
    inner = np.flatnonzero(near[1:-1] & (total[1:-1] >= total[:-2]) & (total[1:-1] >= total[2:])) + 1
    if inner.size < 2:
        return None
    first = int(inner[np.argmax(total[inner])])
    apart = np.abs(offsets[inner] - offsets[first]) >= SPLIT_APART_S
    if not apart.any():
        return None
    candidates = inner[apart]
    second = int(candidates[np.argmax(total[candidates])])
    if total[second] < max(SPLIT_SECOND_Z, SPLIT_SECOND_SHARE * total[first]):
        return None
    rate = ENVELOPE_RATE
    levels = [float(offsets[0] + _parabolic(total, index) / rate) for index in (first, second)]
    # The stretch's own level first: a part the path cannot vouch for
    # keeps the offset the stretch was measured at, and that level keeps
    # its exact value when a peak is it to within a few milliseconds.
    levels.sort(key=lambda level: abs(level - block.offset_s))
    if abs(levels[0] - block.offset_s) * 1000.0 <= CONSISTENT_MS:
        levels[0] = block.offset_s
    scores = _TraceScores(aligner, block.start_s, block.end_s, levels)
    table = np.array([scores.at(level) for level in levels])
    path = _two_level_path(table, TRACE_JUMP)
    runs: List[List[int]] = []
    for column, state in enumerate(path):
        if runs and runs[-1][0] == state:
            runs[-1][2] = column
        else:
            runs.append([state, column, column])
    if len(runs) < 2:
        return None
    effects = "high" in [name for name, _weight in scores.parts]
    minimum = max(1, int(round(SPLIT_MIN_RUN_S / TRACE_BIN_S)))
    for run in runs:
        state, lo, hi = run
        size = hi - lo + 1
        own = float(table[state, lo : hi + 1].sum()) / math.sqrt(size)
        other = float(table[1 - state, lo : hi + 1].sum()) / math.sqrt(size)
        if effects and scores.sigma.get("high", 0.0) > 1e-12:
            high = [float(scores._means("high", level)[lo : hi + 1].sum()) / scores.sigma["high"] / math.sqrt(size) for level in levels]
            if high[1 - state] - high[state] >= TRACE_BEAT_Z and high[1 - state] >= TRACE_RUN_Z and not (
                abs(levels[1] - levels[0]) >= SPEECH_ARBITER_S
                and aligner.speech_choice(float(scores.edges[lo]), float(scores.edges[hi + 1]), levels[state], levels[1 - state]) == levels[state]
            ):
                run[0] = 1 - state
                continue
        if state != 0 and (size < minimum or own < TRACE_RUN_Z or own - other < TRACE_BEAT_Z):
            run[0] = 0
    merged: List[List[int]] = []
    for run in runs:
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
        else:
            merged.append(list(run))
    if len(merged) < 2:
        return None
    # Each change of level on its shot change, or where the agreement
    # changes, near where the path put it.
    edges = scores.edges
    bounds = [block.start_s] + [float(edges[run[1]]) for run in merged[1:]] + [block.end_s]
    for index in range(1, len(merged)):
        a, b = levels[merged[index - 1][0]], levels[merged[index][0]]
        span_lo = max(bounds[index - 1] + MIN_BLOCK_S, bounds[index] - TRACE_STEP_REACH_S)
        span_hi = min(bounds[index + 1] - MIN_BLOCK_S, bounds[index] + TRACE_STEP_REACH_S)
        if span_hi > span_lo:
            bounds[index], _how = _place_step(aligner, span_lo, span_hi, a, b, hole=False, guess_on_sound=True)
    if any(bounds[index + 1] - bounds[index] < MIN_BLOCK_S for index in range(len(merged))):
        return None
    pieces: List[Block] = []
    for index, run in enumerate(merged):
        piece = Block(
            bounds[index], bounds[index + 1], levels[run[0]], match=block.match, support=block.support,
            windows=block.windows, start_lo=bounds[index], start_hi=bounds[index],
            end_lo=bounds[index + 1], end_hi=bounds[index + 1], effects=block.effects,
        )
        pieces.append(piece)
    pieces[0].start_lo, pieces[0].start_hi, pieces[0].start_cut = block.start_lo, block.start_hi, block.start_cut
    pieces[-1].end_lo, pieces[-1].end_hi, pieces[-1].end_cut = block.end_lo, block.end_hi, block.end_cut
    for piece in pieces:
        piece.match = aligner.sharp_match(piece)
    aligner.log(
        f"  {_clock(block.start_s)} - {_clock(block.end_s)} holds two levels {abs(levels[1] - levels[0]) * 1000.0:.0f} ms apart: "
        + ", ".join(f"{_clock(p.start_s)} - {_clock(p.end_s)} at {p.offset_s:+.4f}s" for p in pieces)
    )
    return pieces


# --- following the dub through a gap between two offsets ------------------

# A gap at least this long between two stretches whose offsets differ by
# more than NEAR_OFFSET_S and at most TRACE_MAX_STEP_S, with at least
# TRACE_MIN_DUB_S of dub between where the one leaves off and the other
# picks up, is followed through bin by bin (see _trace_gap).
TRACE_MIN_S = 5.0
TRACE_MAX_STEP_S = 30.0
TRACE_MIN_DUB_S = 4.0
TRACE_BIN_S = 1.0
# The offsets tried besides the neighbours': the peaks, between the two, of
# readings this long this far apart that stand this far above their noise.
TRACE_READING_S = 6.0
TRACE_READING_HOP_S = 2.0
TRACE_READING_Z = 5.0
TRACE_MAX_LEVELS = 8
# Two candidates closer than this are one: a second of speech timing
# cannot tell them apart, and a trim that small is the edges' business.
TRACE_LEVEL_APART_S = 0.2
# The bins are measured against offsets at least this far from every
# candidate, out to this far past them: what the agreement is where
# nothing matches.
TRACE_NOISE_APART_S = 0.25
TRACE_NOISE_SPAN_S = 3.0
TRACE_NOISE_SPACING_S = 0.13
# The path may change offset at this price, in noise units.
TRACE_JUMP = 6.0
# Over its bins, a neighbour's offset carried into the gap must stand this
# far above the noise; an offset of the gap's own this far, and this far
# above every other candidate over the same bins.
TRACE_CARRY_Z = 2.5
TRACE_RUN_Z = 4.0
TRACE_BEAT_Z = 2.0
# ... and the evidence must reach its steps: over this much of it next to
# each change of offset, this far above the noise.
TRACE_EDGE_S = 10.0
TRACE_EDGE_Z = 1.5
# A neighbour's offset carried this few seconds into the gap is not judged
# on them: a second or two is noise either way, and the step at its end is
# placed on the evidence like any other.
TRACE_SHORT_CARRY_S = 3.0
# How far either side of the bin the path changed offset in the step is
# looked for, on the sound and the picture (see _place_step).
TRACE_STEP_REACH_S = 5.0


def _trace_gaps(
    aligner: _Aligner, blocks: List[Block], secondary_s: float, notes: Optional[List[str]], warnings: List[str],
) -> List[Block]:
    """Follow the dub through every gap it has the material for (see
    ``_trace_gap``); the stretches found are put in between."""
    out = sorted(blocks, key=lambda b: b.start_s)
    index = 0
    while index + 1 < len(out):
        found = _trace_gap(aligner, out[index], out[index + 1], secondary_s, notes, warnings)
        if found:
            out[index + 1 : index + 1] = found
            index += len(found)
        index += 1
    return out


def _trace_levels(aligner: _Aligner, lo_s: float, hi_s: float, before: float, after: float) -> List[float]:
    """The offsets a gap's dub may sit at: the neighbours' two, then the
    peaks of short readings across the gap that fall between them,
    strongest first."""
    rate = ENVELOPE_RATE
    low, high = min(before, after), max(before, after)
    window = int(round(TRACE_READING_S * rate))
    margin = int(round(TRACE_NOISE_SPAN_S * rate))
    found: List[Tuple[float, float]] = []
    at = lo_s
    while at + TRACE_READING_S <= hi_s + 1e-9:
        start = int(round(at * rate))
        for band in aligner.shared_bands:
            template = aligner.onsets("primary", band)[start : start + window]
            secondary = aligner.onsets("secondary", band)
            first = max(0, start + int(round(low * rate)) - margin)
            last = min(len(secondary), start + window + int(round(high * rate)) + margin)
            if len(template) < window or last - first < window + 2:
                continue
            row = ncc_lags(template, secondary[first:last])
            units = _noise_units(row, np.ones(len(row), dtype=bool))
            offsets = (first + np.arange(len(row)) - start) / rate
            inside = np.flatnonzero((offsets >= low - SAME_OFFSET_S) & (offsets <= high + SAME_OFFSET_S))
            if inside.size == 0:
                continue
            peak = int(inside[np.argmax(units[inside])])
            z = float(units[peak]) * (EFFECTS_DISCOUNT if band == "high" else 1.0)
            if z >= TRACE_READING_Z:
                found.append((float(offsets[0] + _parabolic(row, peak) / rate), z))
        at += TRACE_READING_HOP_S
    levels = [before, after]
    for offset, _z in sorted(found, key=lambda f: -f[1]):
        if len(levels) >= TRACE_MAX_LEVELS:
            break
        if all(abs(offset - level) >= TRACE_LEVEL_APART_S for level in levels):
            levels.append(offset)
    return levels


class _TraceScores:
    """How much the two tracks agree in each bin of a gap at any offset, in
    noise units: every shared band's agreement and the two languages'
    speech timing, each measured against offsets where nothing matches."""

    def __init__(self, aligner: _Aligner, lo_s: float, hi_s: float, levels: Sequence[float]) -> None:
        self.aligner = aligner
        self.lo_s, self.hi_s = lo_s, hi_s
        self.bins = max(1, int(round((hi_s - lo_s) / TRACE_BIN_S)))
        self.edges = lo_s + (hi_s - lo_s) * np.arange(self.bins + 1) / self.bins
        low, high = min(levels), max(levels)
        noise = [
            float(offset) for offset in np.arange(low - TRACE_NOISE_SPAN_S, high + TRACE_NOISE_SPAN_S, TRACE_NOISE_SPACING_S)
            if all(abs(offset - level) >= TRACE_NOISE_APART_S for level in levels)
        ]
        self.parts: List[Tuple[str, float]] = [(band, EFFECTS_DISCOUNT if band == "high" else 1.0) for band in aligner.shared_bands]
        self.speech: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        primary, secondary = aligner.primary, aligner.secondary
        if primary.speed == 1.0 and secondary.speed == 1.0:
            # The voices themselves, where the music and effects are too
            # quiet to say anything: a dub is performed to the lips, so its
            # lines start and stop about where the original's do.
            from .speech import SPEECH_RATE, centre_activity
            try:
                reach = TRACE_NOISE_SPAN_S + 1.0
                video = centre_activity(primary.path, primary.track, lo_s, hi_s, aligner.token)
                dub_lo = lo_s + low - reach
                dub = centre_activity(secondary.path, secondary.track, dub_lo, hi_s + high + reach, aligner.token)
                span = int(LOCAL_STATS_S * SPEECH_RATE)
                self.speech = (_local_standardize(video, span), _local_standardize(dub, span), dub_lo)
                self.parts.append(("speech", 1.0))
            except MediaError:
                self.speech = None
        self.sigma: Dict[str, float] = {}
        for name, _weight in self.parts:
            values = np.concatenate([self._means(name, offset) for offset in noise]) if noise else np.zeros(0)
            self.sigma[name] = float(np.std(values)) if values.size > 1 else 0.0

    def _means(self, name: str, offset_s: float) -> np.ndarray:
        """The mean agreement of one part in every bin, at one offset."""
        if name == "speech":
            from .speech import SPEECH_RATE
            video, dub, dub_lo = self.speech
            curve = np.zeros(len(video))
            begin = int(round((self.lo_s + offset_s - dub_lo) * SPEECH_RATE))
            usable = max(0, min(len(video), len(dub) - begin))
            if begin >= 0 and usable:
                curve[:usable] = video[:usable] * dub[begin : begin + usable]
            cuts = np.round((self.edges - self.lo_s) * SPEECH_RATE).astype(int)
        else:
            curve = self.aligner.agreement(offset_s, self.lo_s, self.hi_s, [name])
            cuts = np.round((self.edges - self.lo_s) * ENVELOPE_RATE).astype(int)
        cuts = np.clip(cuts, 0, len(curve))
        return np.array([float(curve[a:b].mean()) if b > a else 0.0 for a, b in zip(cuts[:-1], cuts[1:])])

    def at(self, offset_s: float) -> np.ndarray:
        """Every bin's agreement at the offset, summed over the parts in
        noise units and scaled so pure noise has a spread of one."""
        total = np.zeros(self.bins)
        weights = 0.0
        for name, weight in self.parts:
            sigma = self.sigma.get(name, 0.0)
            if sigma <= 1e-12:
                continue
            total += weight * self._means(name, offset_s) / sigma
            weights += weight * weight
        return total / math.sqrt(weights) if weights else total


def _trace_gap(
    aligner: _Aligner, left: Block, right: Block, secondary_s: float,
    notes: Optional[List[str]], warnings: List[str],
) -> Optional[List[Block]]:
    """Follow the dub through the gap between two stretches whose offsets
    differ, where the dub has the material for it.

    The neighbours say where the dub leaves off and picks up; between those
    two points is all the dub the gap can hold, and all the video it lacks
    is the difference of the offsets. A scene both edits trimmed
    differently -- a shot shorter here, one taken out there -- is that
    material at a few offsets in turn, stepping from one neighbour's to the
    other's. The coarse and gap searches need a long window to see a quiet
    scene and a short one to see a step, and a trimmed quiet scene defeats
    both; here the offsets are only a handful (the neighbours', and any a
    short reading found between them), every second of the gap is scored at
    each in every band and in the speech timing, and one path through them
    picks the level each second is at, paying for every change. Each run of
    it must stand above the noise over its length -- a neighbour's offset
    carried on a little, one of the gap's own clearly, and above the other
    candidates too -- or the gap is left as it was. Measured on a real pair:
    a 65-second gap whose dub was 61.6 s of the same scene went on at the
    first offset for 49 s, took out a 1.84-second shot the picture shows,
    ran 11 s at the second and stepped 1.67 s to the third.
    """
    lo_s, hi_s = left.end_s, right.start_s
    gap = hi_s - lo_s
    before, after = left.offset_s, right.offset_s
    step = before - after          # > 0: the video has this much the dub lacks
    if gap < TRACE_MIN_S or not NEAR_OFFSET_S < abs(step) <= TRACE_MAX_STEP_S:
        return None
    if gap - max(step, 0.0) < TRACE_MIN_DUB_S or lo_s + before < 0.0 or hi_s + after > secondary_s:
        return None
    if _dub_is_silent(aligner.secondary, lo_s + before, hi_s + after, _dub_level(aligner.secondary, [left, right])):
        return None

    levels = _trace_levels(aligner, lo_s, hi_s, before, after)
    low, high = min(before, after), max(before, after)
    middle = sorted((level for level in levels[2:] if low < level < high), reverse=step > 0)
    order = [before] + middle + [after]
    scores = _TraceScores(aligner, lo_s, hi_s, order)
    table = np.array([scores.at(level) for level in order])
    count, bins = table.shape

    # The path: it starts at the offset before the gap and ends at the one
    # after (anything else costs a change), and moves only from the one
    # towards the other -- the dub is not played twice.
    best = table[:, 0] - np.where(np.arange(count) == 0, 0.0, TRACE_JUMP)
    back = np.zeros((bins, count), dtype=int)
    for column in range(1, bins):
        new = np.empty(count)
        for state in range(count):
            source, value = state, best[state]
            for earlier in range(state):
                if best[earlier] - TRACE_JUMP > value:
                    source, value = earlier, best[earlier] - TRACE_JUMP
            new[state] = value + table[state, column]
            back[column, state] = source
        best = new
    state = int(np.argmax(best - np.where(np.arange(count) == count - 1, 0.0, TRACE_JUMP)))
    path = [state]
    for column in range(bins - 1, 0, -1):
        state = int(back[column, state])
        path.append(state)
    path.reverse()

    runs: List[Tuple[int, int, int]] = []          # (state, first bin, last bin)
    for column, state in enumerate(path):
        if runs and runs[-1][0] == state:
            runs[-1] = (state, runs[-1][1], column)
        else:
            runs.append((state, column, column))
    judged: List[str] = []
    edge = max(1, int(round(TRACE_EDGE_S / TRACE_BIN_S)))
    for position, (state, first, last) in enumerate(runs):
        size = last - first + 1
        own = float(table[state, first : last + 1].sum()) / math.sqrt(size)
        rivals = [float(table[other, first : last + 1].sum()) / math.sqrt(size) for other in range(count) if other != state]
        carried = (state == 0 and first == 0) or (state == count - 1 and last == bins - 1)
        if carried and size * TRACE_BIN_S < TRACE_SHORT_CARRY_S:
            judged.append(f"{own:.1f}")
            continue
        if carried:
            ok = own >= TRACE_CARRY_Z
        else:
            ok = own >= TRACE_RUN_Z and own - max(rivals, default=-np.inf) >= TRACE_BEAT_Z
        # A run whose support is all at one end says nothing about the step
        # at the other -- nor that the dub there is this scene at all: the
        # dub's own material the length of what it lacks, in its place,
        # would carry a neighbour's offset on its far end alone.
        ends = []
        if position > 0 or order[state] != before:
            ends.append(table[state, first : first + edge])
        if position < len(runs) - 1 or order[state] != after:
            ends.append(table[state, max(first, last + 1 - edge) : last + 1])
        if any(float(chunk.sum()) / math.sqrt(len(chunk)) < TRACE_EDGE_Z for chunk in ends):
            ok = False
        if not ok:
            aligner.log(
                f"  {_clock(lo_s)} - {_clock(hi_s)}: the dub at {order[state]:+.3f}s over "
                f"{_clock(scores.edges[first])} - {_clock(scores.edges[last + 1])} is not clear ({own:.1f} noise units); left as it was"
            )
            return None
        judged.append(f"{own:.1f}")

    # Every change of offset is put exactly where it happens: on the
    # picture's cut when there is one to put it on, else where the
    # agreement changes, else where the path changed. A neighbour's edge
    # was placed on long windows that could not see the gap's steps, so a
    # step next to it may move it, as long as the neighbour keeps a length.
    edges = scores.edges
    pieces: List[List[float]] = [[order[state], edges[first], edges[last + 1]] for state, first, last in runs]
    pieces[0][1] = lo_s
    pieces[-1][2] = hi_s
    # The neighbours are pieces too, empty where the path left them at the
    # gap's edge, so the step into and out of the gap is placed on the
    # evidence like every other rather than where the neighbour's own edge
    # happened to be left.
    if order[runs[0][0]] != before:
        pieces.insert(0, [before, lo_s, lo_s])
    if order[runs[-1][0]] != after:
        pieces.append([after, hi_s, hi_s])
    floor = max(left.start_s + MIN_BLOCK_S, lo_s - TRACE_STEP_REACH_S)
    ceiling = min(right.end_s - MIN_BLOCK_S, hi_s + TRACE_STEP_REACH_S)
    guessed: List[float] = []
    cuts: Dict[int, Tuple[float, float]] = {}   # step -> (where the one ends, where the next starts)
    for index in range(len(pieces) - 1):
        a, b = pieces[index][0], pieces[index + 1][0]
        hole = max(0.0, a - b)
        at = pieces[index + 1][1]
        lower = floor if index == 0 else pieces[index][1]
        upper = ceiling if index + 1 == len(pieces) - 1 else pieces[index + 1][2]
        # The path knows nothing of the frames the dub lacks at a step, so
        # the span reaches that much further on.
        span_lo = max(lower, at - TRACE_STEP_REACH_S)
        span_hi = min(upper, at + TRACE_STEP_REACH_S + hole)
        if span_hi - span_lo <= hole:
            aligner.log(f"  {_clock(lo_s)} - {_clock(hi_s)}: no room for the step near {_clock(at)}; left as it was")
            return None
        point, how = _place_step(aligner, span_lo, span_hi, a, b, guess_on_sound=True)
        if "high" in aligner.shared_bands:
            # The effects first, as everywhere (see BANDS): where the music
            # ran on across a trim, every band summed puts the step between
            # where the music moved and where the picture did. Measured on a
            # real pair: the effects moved 0.8 s at 1:00:58 at ten noise
            # units a second, the bass four seconds later, and the sum put
            # the step at 1:01:02. Unless the dialogue, between the two
            # answers, is at the level the effects have already left.
            said, told = _place_step(aligner, span_lo, span_hi, a, b, bands=["high"])
            if told != "guess" and abs(said - point) > 1.0:
                between = (min(said, point), max(said, point))
                if not (
                    abs(a - b) >= SPEECH_ARBITER_S
                    and aligner.speech_choice(between[0], between[1] + max(0.0, a - b), a, b) == (a if said < point else b)
                ):
                    point, how = said, told
            elif told != "guess":
                point, how = said, told
        if how == "guess":
            point = min(max(point, span_lo), span_hi - hole)
            guessed.append(point)
        pieces[index][2] = point
        pieces[index + 1][1] = point + hole
        if how == "picture":
            cuts[index] = (point, point + hole)
    inner = pieces[1:-1]
    if any(end - start < MIN_BLOCK_S for _offset, start, end in inner) or (
        pieces[0][2] - left.start_s < MIN_BLOCK_S or right.end_s - pieces[-1][1] < MIN_BLOCK_S
    ) or any(pieces[index][2] > pieces[index + 1][1] + 1e-9 for index in range(len(pieces) - 1)):
        aligner.log(f"  {_clock(lo_s)} - {_clock(hi_s)}: the steps the dub takes there leave a piece too short; left as it was")
        return None

    left.end_s = left.end_lo = left.end_hi = pieces[0][2]
    left.end_cut = cuts[0][0] if 0 in cuts else None
    right.start_s = right.start_lo = right.start_hi = pieces[-1][1]
    last_step = len(pieces) - 2
    right.start_cut = cuts[last_step][1] if last_step in cuts else None
    found: List[Block] = []
    for offset, start, end in inner:
        block = Block(start, end, offset, start_lo=start, start_hi=start, end_lo=end, end_hi=end)
        block.match = aligner.sharp_match(block)
        block.support = float(np.mean(table[order.index(offset)]))
        found.append(block)

    steps = " then ".join(
        f"{_clock(max(start, lo_s))} - {_clock(min(end, hi_s))} at {offset:+.3f}s"
        for offset, start, end in pieces if min(end, hi_s) - max(start, lo_s) > 0.01
    )
    what = "the music and effects" + (" and the speech timing" if scores.speech is not None else "")
    if notes is not None:
        notes.append(
            f"followed the dub through {_clock(lo_s)} - {_clock(hi_s)}, where it did not correlate in long windows: "
            f"{steps} (by {what}; {', '.join(judged)} noise units)"
        )
    aligner.log(f"  followed the dub through {_clock(lo_s)} - {_clock(hi_s)}: {steps}")
    for point in guessed:
        warnings.append(
            f"the dub steps near {_clock(point)}: {what} say so, but neither the sound nor the picture "
            "clearly says where to the frame -- check the lips there"
        )
    return found


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
        _settle(aligner, block)
        measured.append(block)
    return sorted(measured, key=lambda b: b.start_s)


def _settle(aligner: _Aligner, block: Block) -> bool:
    """Check a stretch's offset against the whole stretch at once, and
    move it to where the whole stretch peaks when that is clearly better
    (see SETTLE_MARGIN). True when it moved.

    Measured on a real pair: a 39-second stretch of a quiet scene had its
    readings scattered over 0.6 s, the effects band's chance peaks among
    them, and their median put it at -133.759 s; the whole stretch
    correlated at 22 noise units at -133.669 s and at 5 at -133.759 --
    89 ms, which shows on the lips. A stretch the effects placed, or one
    following a drift, keeps its offset: those were read on purpose from
    one band or one part of it."""
    if block.effects or block.follows_drift or block.length_s < SETTLE_MIN_S:
        return False
    offsets, total = aligner.span_scores(block.start_s, block.end_s, block.offset_s)
    near = np.flatnonzero(np.abs(offsets - block.offset_s) <= POLISH_RANGE_S)
    if near.size == 0:
        return False
    peak = int(near[np.argmax(total[near])])
    here = int(np.argmin(np.abs(offsets - block.offset_s)))
    at_offset = float(total[max(0, here - 1) : here + 2].max())
    top = float(total[peak])
    best = float(offsets[0] + _parabolic(total, peak) / ENVELOPE_RATE)
    if abs(best - block.offset_s) * 1000.0 <= CONSISTENT_MS:
        return False
    if top - at_offset < SETTLE_MARGIN or at_offset > SETTLE_SHARE * top:
        return False
    aligner.log(
        f"  {_clock(block.start_s)} - {_clock(block.end_s)}: its readings put it at {block.offset_s:+.3f}s, "
        f"the whole stretch at {best:+.3f}s ({top:.1f} noise units against {at_offset:.1f}); the whole stretch decides"
    )
    block.offset_s = best
    return True


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
    _snap_to_picture(aligner, before, after, primary_s, secondary_s, coarse_before, coarse_after)


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


def _place_step(
    aligner: _Aligner, lo_s: float, hi_s: float, before_s: float, after_s: float, hole: bool = True,
    guess_on_sound: bool = False, bands: Optional[Sequence[str]] = None,
) -> Tuple[float, str]:
    """Where inside [lo_s, hi_s] a stretch at ``before_s`` gives way to one
    at ``after_s``: the point that leaves the most agreement at the first
    offset before it and at the second after it -- after the frames the dub
    lacks, when the offset falls and ``hole`` is set.

    The picture has the last word where it can speak. A cut in the dub's
    edit is a cut in the picture, so the point is looked for among the
    video's shot changes in the span -- with, where the dub lacks frames,
    another shot change exactly that far after it (see ``PictureCuts.pairs``).
    Among those the sound chooses when it has anything to say; a lone
    candidate needs no choosing; a candidate the sound scores clearly below
    its own choice is not the place (a dissolve, a cut the picture missed),
    and the sound's choice stands. Measured on a real pair, the sound alone
    put a one-frame trim 0.66 s early and a 1.1 s insert 0.63 s early --
    both exactly on shot changes.

    Returns the point and how it was found: ``picture``, ``sound``,
    ``speech`` (the two languages' lines, where neither the sound nor the
    picture could say; see _speech_step), or ``guess`` (the middle, or one
    of several shot changes nothing could choose between). A caller that knows the step is there, and only
    asks where, sets ``guess_on_sound``: a guess is then the sound's best
    point, however faint, rather than the middle. ``bands`` limits the
    sound to those bands (all shared ones by default).
    """
    rate = ENVELOPE_RATE
    drop = max(0.0, before_s - after_s) if hole else 0.0
    room = max(0.0, hi_s - lo_s - drop)
    middle = lo_s + room / 2.0
    scores: Optional[np.ndarray] = None
    measured = False
    a = aligner.agreement(before_s, lo_s, hi_s, bands)
    b = aligner.agreement(after_s, lo_s, hi_s, bands)
    n = min(len(a), len(b))
    skip = int(round(drop * rate))
    if n >= 2 and room > 0.0 and n - skip > 0:
        a, b = a[:n], b[:n]
        level = max(float(np.abs(a).mean()), float(np.abs(b).mean()))
        if level >= AGREEMENT_FLOOR:
            ahead = np.concatenate([[0.0], np.cumsum(a)])          # agreement at the first offset up to k
            behind = np.concatenate([np.cumsum(b[::-1])[::-1], [0.0]])  # agreement at the second from k on
            last = n - skip
            scores = ahead[: last + 1] + behind[skip : skip + last + 1]
            # Said only when the choice stands out: a flat score is no placement.
            measured = float(scores.max() - np.median(scores)) > STEP_EVIDENCE * level * rate
    sound = lo_s + int(np.argmax(scores)) / rate if scores is not None else middle

    picture = aligner.picture
    candidates = picture.pairs(lo_s, lo_s + room, drop) if picture is not None and room > 0.0 else None
    if candidates and scores is not None and measured:
        def at(cut: float) -> float:
            return float(scores[min(len(scores) - 1, max(0, int(round((cut - lo_s) * rate))))])
        best = max(candidates, key=at)
        top, low = float(scores.max()), float(np.median(scores))
        if top - at(best) <= 0.5 * (top - low):
            return best, "picture"
        return sound, "sound"
    if candidates and len(candidates) == 1:
        return candidates[0], "picture"
    if measured:
        return sound, "sound"
    # The music and effects cannot say, and the picture has no one cut to
    # offer: the lines may (see _speech_step), and a shot change close to
    # where they put it is where it goes.
    heard = _speech_step(aligner, lo_s, hi_s, before_s, after_s, drop) if room > 0.0 else None
    if heard is not None:
        if candidates:
            near = min(candidates, key=lambda cut: abs(cut - heard))
            if abs(near - heard) <= SPEECH_SNAP_S:
                return near, "picture"
        return heard, "speech"
    if candidates:
        if scores is not None:
            def score_at(cut: float) -> float:
                return float(scores[min(len(scores) - 1, max(0, int(round((cut - lo_s) * rate))))])
            return max(candidates, key=score_at), "guess"
        return min(candidates, key=lambda cut: abs(cut - middle)), "guess"
    return (sound if guess_on_sound else middle), "guess"


# Where neither the music and effects nor the picture can place a step, the
# two languages' lines can: a dub is performed to the lips, so its speech
# starts and stops about where the original's does. Their agreement is
# summed the way the sound's is (see _place_step), and the rise of that sum
# is only a placement when it is this many times the largest rise the same
# sum reaches with both offsets moved by these amounts, where nothing lines
# up: long spans of dialogue rise by chance, and the bar has to grow with them.
SPEECH_STEP_NULLS_S = (-2.1, -1.3, -0.7, 0.7, 1.3, 2.1)
SPEECH_STEP_MARGIN = 1.5


def _speech_step(
    aligner: _Aligner, lo_s: float, hi_s: float, before_s: float, after_s: float, drop_s: float,
) -> Optional[float]:
    """Where inside [lo_s, hi_s) the dub's lines stop agreeing with the
    original's at ``before_s`` and start agreeing at ``after_s`` (``drop_s``
    later), or None when the speech does not say clearly. Measured on a
    real pair: a one-second trim the music and effects were silent about,
    guessed at the middle of a 13-second gap, sat 5.8 s later; and all 5.2 s
    of dub around a 58-second missing scene, split half to each side by a
    guess, belonged to the scene before it."""
    primary, secondary = getattr(aligner, "primary", None), getattr(aligner, "secondary", None)
    if primary is None or secondary is None or not getattr(primary, "path", None) or not getattr(secondary, "path", None):
        return None
    if primary.speed != 1.0 or secondary.speed != 1.0 or hi_s - lo_s <= drop_s + 1.0:
        return None
    from .speech import SPEECH_RATE, centre_activity
    reach = max(abs(shift) for shift in SPEECH_STEP_NULLS_S) + 0.1
    try:
        video = centre_activity(primary.path, primary.track, lo_s, hi_s, aligner.token)
        dub_a = centre_activity(secondary.path, secondary.track, lo_s + before_s - reach, hi_s + before_s + reach, aligner.token)
        dub_b = centre_activity(secondary.path, secondary.track, lo_s + after_s - reach, hi_s + after_s + reach, aligner.token)
    except MediaError:
        return None
    span = int(LOCAL_STATS_S * SPEECH_RATE)
    video, dub_a, dub_b = (_local_standardize(curve, span) for curve in (video, dub_a, dub_b))
    count = len(video)
    skip = int(round(drop_s * SPEECH_RATE))
    last = count - skip
    pad = int(round(reach * SPEECH_RATE))
    if last < 2 or len(dub_a) < 2 * pad + count or len(dub_b) < 2 * pad + count:
        return None

    def rise(shift_s: float) -> Tuple[float, int]:
        start = pad + int(round(shift_s * SPEECH_RATE))
        a = video * dub_a[start : start + count]
        b = video * dub_b[start : start + count]
        ahead = np.concatenate([[0.0], np.cumsum(a)])
        behind = np.concatenate([np.cumsum(b[::-1])[::-1], [0.0]])
        scores = ahead[: last + 1] + behind[skip : skip + last + 1]
        return float(scores.max() - np.median(scores)), int(np.argmax(scores))

    real, at = rise(0.0)
    chance = max(rise(shift)[0] for shift in SPEECH_STEP_NULLS_S)
    if real < SPEECH_STEP_MARGIN * chance:
        return None
    return lo_s + at / SPEECH_RATE


# How far a step the speech placed may be from a shot change and still be
# put on it: the speech places to a few tenths of a second.
SPEECH_SNAP_S = 0.5


# An edge the sound placed no better than this is looked for among the
# picture's shot changes: a frame and a half.
PICTURE_SNAP_MIN_S = 0.06


def _choose_cut(
    candidates: Sequence[float], ahead: Optional[np.ndarray], behind: Optional[np.ndarray],
    span_lo: float, gap_s: float, level: float, strict: bool = False,
) -> Optional[float]:
    """The shot change where an edge goes: the one the agreement scores best
    when the agreement says anything (see ``_place_step``), the only one when
    there is one, else none. ``ahead`` is the running agreement of the
    stretch before the edge, ``behind`` of the one after it, from the
    candidate on -- ``gap_s`` later, when the dub lacks that much."""
    rate = ENVELOPE_RATE
    skip = int(round(gap_s * rate))
    sizes = [len(x) for x in (ahead, behind) if x is not None]
    if not sizes:
        return candidates[0] if len(candidates) == 1 and not strict else None
    n = min(sizes) - 1

    def total(index: int) -> float:
        value = 0.0
        if ahead is not None:
            value += float(ahead[index])
        if behind is not None:
            value += float(behind[index + skip])
        return value

    usable = [c for c in candidates if 0 <= int(round((c - span_lo) * rate)) and int(round((c - span_lo) * rate)) + skip <= n]
    if not usable:
        return None
    everything = np.array([total(i) for i in range(0, max(1, n - skip + 1))])
    measured = level >= AGREEMENT_FLOOR and everything.size > 1 and (
        float(everything.max() - np.median(everything)) > STEP_EVIDENCE * level * rate
    )
    if measured:
        best = max(usable, key=lambda c: total(int(round((c - span_lo) * rate))))
        top, low = float(everything.max()), float(np.median(everything))
        if top - total(int(round((best - span_lo) * rate))) > 0.5 * (top - low):
            return None
        if strict and _sound_overrules(ahead, behind, skip, int(round((best - span_lo) * rate)), int(np.argmax(everything))):
            return None
        return best
    return usable[0] if len(usable) == 1 and not strict else None


# A lone shot change is taken over the sound's own placement only while the
# sound cannot tell the two apart. Between them, in blocks this long, one
# side's offset agreeing better block after block, by this many standard
# errors, is the sound placing the edge itself: a dub's audio can be spliced
# where the picture has no cut. Measured on a real pair: a 74 ms splice in
# the middle of a shot, 2.1 s after the nearest cut, which a snap put the
# dub 74 ms early across.
PICTURE_VETO_BLOCK_S = 0.25
PICTURE_VETO_Z = 3.0


def _sound_overrules(
    ahead: Optional[np.ndarray], behind: Optional[np.ndarray], skip: int, cut: int, peak: int,
) -> bool:
    """Whether the agreement, between the shot change at index ``cut`` and
    the sound's own best edge at ``peak``, says clearly that the edge is at
    ``peak``: the stretch the move would take the span from agrees better
    there, block after block."""
    lo, hi = min(cut, peak), max(cut, peak)
    block = int(round(PICTURE_VETO_BLOCK_S * ENVELOPE_RATE))
    count = (hi - lo) // block
    if count < 2:
        return False
    # Per sample: how much better the stretch before the edge agrees than
    # the one after it (the running sums' own steps).
    gain = np.zeros(hi - lo)
    if ahead is not None:
        gain += np.diff(ahead[lo : hi + 1])
    if behind is not None:
        gain += np.diff(behind[lo + skip : hi + skip + 1])
    if peak < cut:
        gain = -gain  # the sound keeps the span for the stretch after the edge
    sums = gain[: count * block].reshape(count, block).sum(axis=1)
    spread = float(np.std(sums, ddof=1))
    mean = float(sums.mean())
    if spread <= 1e-12:
        return mean > 0.0
    return mean / (spread / np.sqrt(count)) >= PICTURE_VETO_Z


# A step in the offset no larger than this is a picture trim -- a few
# frames taken off one side of a shot change -- and sits on one cut; a
# larger one is a stretch of picture cut out, between two.
TRIM_MAX_S = 0.5


def _snap_to_picture(
    aligner: _Aligner, before: Optional[Block], after: Optional[Block], primary_s: float, secondary_s: float,
    bracket_before: Optional[Tuple[float, float]] = None, bracket_after: Optional[Tuple[float, float]] = None,
) -> None:
    """Move edges the sound could only place roughly onto their shot change.

    Where the dub runs straight through the cut (the edges are tied, the
    usual case) the cut is one frame of the picture: for a trim of a few
    frames, a single shot change; for a scene missing from the dub, a pair
    of shot changes exactly the missing length apart, which across a span
    of a few seconds is usually the only such pair there is. Otherwise
    each edge is looked for on its own. The sound's own interval is asked
    first; failing that, the whole bracket the readings left the edge in,
    where only a clear choice by the sound is taken. An edge moved here is
    placed to the frame and its uncertainty says so, so it is never trimmed
    back as doubtful and the dub is kept right up to the cut; the cut is
    remembered, so a later pass that fits the edge on the sound again does
    not walk it off the frame.
    """
    picture = aligner.picture
    if picture is None or not picture.available:
        return
    tolerance = picture.tolerance_s

    def put(block: Block, is_end: bool, at: float) -> None:
        if is_end:
            block.end_s = min(at, secondary_s - block.offset_s, primary_s)
            block.end_lo, block.end_hi, block.end_cut = block.end_s - tolerance, block.end_s + tolerance, at
        else:
            block.start_s = max(at, -block.offset_s, 0.0)
            block.start_lo, block.start_hi, block.start_cut = block.start_s - tolerance, block.start_s + tolerance, at

    def windows(at: float, doubt: float, bracket: Optional[Tuple[float, float]]) -> List[Tuple[float, float, bool]]:
        """(lo, hi, needs a clear choice): the sound's interval, then the bracket."""
        out = [(max(0.0, at - doubt), min(primary_s, at + doubt), False)]
        if bracket is not None and (bracket[0] < at - doubt - 0.5 or bracket[1] > at + doubt + 0.5):
            out.append((max(0.0, min(bracket[0], at - doubt)), min(primary_s, max(bracket[1], at + doubt)), True))
        return out

    if before is not None and after is not None:
        video_gap = max(0.0, before.offset_s - after.offset_s)
        tied = abs((after.start_s - before.end_s) - video_gap) <= 0.002
        if tied:
            if before.end_cut is not None and abs(before.end_s - before.end_cut) <= 2 * tolerance:
                put(before, True, before.end_cut)
                put(after, False, before.end_cut + video_gap)
                return
            doubt = max(before.end_uncertainty_s, after.start_uncertainty_s)
            if doubt <= PICTURE_SNAP_MIN_S:
                return
            hole = video_gap if video_gap > TRIM_MAX_S else 0.0
            for lo, hi, strict in windows(before.end_s, doubt, bracket_before):
                # A lone pair of shot changes exactly the missing length
                # apart is evidence on its own; a lone single shot change is
                # not -- there is one every few seconds -- and a trim's cut
                # is only taken when the sound prefers it.
                strict = strict or hole == 0.0
                candidates = picture.pairs(lo, hi, hole)
                if not candidates:
                    continue
                span_lo, span_hi = max(0.0, lo - 1.0), min(primary_s, hi + video_gap + 1.0)
                curve_b = aligner.edge_curve(before.offset_s, span_lo, span_hi)
                curve_a = aligner.edge_curve(after.offset_s, span_lo, span_hi)
                ahead = np.concatenate([[0.0], np.cumsum(curve_b - _agreement_threshold(before.match))]) if curve_b.size else None
                behind = np.concatenate([np.cumsum((curve_a - _agreement_threshold(after.match))[::-1])[::-1], [0.0]]) if curve_a.size else None
                level = max(float(np.abs(curve_b).mean()) if curve_b.size else 0.0,
                            float(np.abs(curve_a).mean()) if curve_a.size else 0.0)
                cut = _choose_cut(candidates, ahead, behind, span_lo, video_gap, level, strict)
                if cut is not None:
                    put(before, True, cut)
                    put(after, False, cut + video_gap)
                    return
            return
    for block, is_end, bracket in ((before, True, bracket_before), (after, False, bracket_after)):
        if block is None:
            continue
        recorded = block.end_cut if is_end else block.start_cut
        at = block.end_s if is_end else block.start_s
        if recorded is not None and abs(at - recorded) <= 2 * tolerance:
            put(block, is_end, recorded)
            continue
        doubt = block.end_uncertainty_s if is_end else block.start_uncertainty_s
        if doubt <= PICTURE_SNAP_MIN_S:
            continue
        for lo, hi, _strict in windows(at, doubt, bracket):
            strict = True  # a single shot change, see above
            candidates = picture.within(lo, hi)
            if not candidates:
                continue
            span_lo, span_hi = max(0.0, lo - 1.0), min(primary_s, hi + 1.0)
            curve = aligner.edge_curve(block.offset_s, span_lo, span_hi)
            if curve.size:
                running = curve - _agreement_threshold(block.match)
                ahead = np.concatenate([[0.0], np.cumsum(running)]) if is_end else None
                behind = None if is_end else np.concatenate([np.cumsum(running[::-1])[::-1], [0.0]])
                cut = _choose_cut(candidates, ahead, behind, span_lo, 0.0, float(np.abs(curve).mean()), strict)
            else:
                cut = candidates[0] if len(candidates) == 1 and not strict else None
            if cut is not None:
                put(block, is_end, cut)
                break


def _step_words(step_s: float) -> str:
    """A step in the offset, in the unit it reads best in."""
    return f"{step_s:+.3f} s" if abs(step_s) >= 1.0 else f"{step_s * 1000.0:+.0f} ms"


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
        if (
            gap > MIN_FILL_S and NEAR_OFFSET_S < abs(step)
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
    sweep_within_1ms: int = 0
    sweep_within_5ms: int = 0
    sweep_typical_ms: Optional[float] = None
    sweep_worst_ms: Optional[float] = None
    stretches: List[Stretch] = field(default_factory=list)
    fills_skipped: int = 0
    lines: Optional[dict] = None
    """The line check (see ``linecheck``): where the dub's lines sit against
    the original's, which are in sync with the lips."""
    lines_text: Optional[str] = None

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
            "sweepWithin1Ms": self.sweep_within_1ms,
            "sweepWithin5Ms": self.sweep_within_5ms,
            "sweepTypicalMs": self.sweep_typical_ms,
            "sweepWorstMs": self.sweep_worst_ms,
            "stretches": [
                {"startS": s.start_s, "endS": s.end_s, "residualMs": s.residual_ms, "windows": s.windows}
                for s in self.stretches
            ],
            "lines": self.lines,
        }

    def describe(self) -> str:
        lines = []
        measured = [s for s in self.spots if s.residual_ms is not None]
        if measured:
            lines.append(
                f"the finished track sits {_ms(self.typical_ms)}ms from the original typically and "
                f"{_ms(self.worst_ms)}ms at its worst, measured at {len(measured)} spots. "
                f"Lip sync starts to show around {AUDIBLE_MS:.0f}ms; the measurement resolves a fraction of a millisecond."
            )
        else:
            lines.append("none of the spot checks could be measured")
        for spot in self.spots:
            if spot.residual_ms is None:
                lines.append(f"  at {_clock(spot.position_s)}: {spot.note}")
            else:
                direction = "early" if spot.residual_ms < 0 else "late"
                lines.append(
                    f"  at {_clock(spot.position_s)}: {_ms(abs(spot.residual_ms))}ms {direction}, "
                    f"match {spot.match:.2f}"
                )
        if self.sweep_windows:
            share = 100.0 * self.sweep_within_audible / max(1, self.sweep_measured)
            fine = 100.0 * self.sweep_within_1ms / max(1, self.sweep_measured)
            lines.append(
                f"swept the whole runtime: {self.sweep_measured} of {self.sweep_windows} windows could be "
                f"measured (the rest are silence, music-only, or filled from the original), and "
                f"{share:.0f}% of them are within {AUDIBLE_MS:.0f}ms of the original, {fine:.0f}% within 1ms"
                + (
                    f" (typically {_ms(self.sweep_typical_ms)}ms, worst {_ms(self.sweep_worst_ms)}ms)"
                    if self.sweep_typical_ms is not None else ""
                )
            )
        if self.stretches:
            lines.append(f"{len(self.stretches)} stretch(es) are more than {STRETCH_MS:.0f}ms out, where lip sync shows:")
            for stretch in self.stretches:
                lines.append(
                    f"  {_clock(stretch.start_s)} - {_clock(stretch.end_s)}  out by "
                    f"{'+' if stretch.residual_ms >= 0 else ''}{_ms(stretch.residual_ms)}ms  ({stretch.windows} windows)"
                )
        elif self.sweep_measured:
            lines.append(f"no stretch is more than {STRETCH_MS:.0f}ms out")
        if self.lines_text:
            lines.append(self.lines_text)
        return "\n".join(lines)


def _ms(value: Optional[float]) -> str:
    """Milliseconds as they read best: the tenth under ten."""
    if value is None:
        return "?"
    return f"{value:.1f}" if abs(value) < 10.0 else f"{value:.0f}"


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
        if abs(residual) <= 5.0:
            result.sweep_within_5ms += 1
        if abs(residual) <= 1.0:
            result.sweep_within_1ms += 1
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
