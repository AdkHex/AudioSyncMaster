"""The numbers the planner is tuned by, each with what it was measured on.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from ..correlate import ENVELOPE_HOP


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
# confident it decides, and a music-only level the effects do not share is
# folded into theirs (a cue laid twice, a beat alias, music run across a
# trim), never a cut.
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

# The last look at every cut: the sound around it, VERIFY_CUT_REACH_S either
# way, may say the switch from one offset to the next lies elsewhere --
# where an early, coarse bracket or a picture cut chosen without the sound
# left it. Moved only when, across the span between the two, the offset
# that should play there agrees better than the one that does with a
# t-statistic of VERIFY_CUT_Z, and at least VERIFY_CUT_MIN_MOVE_S away.
VERIFY_CUT_REACH_S = 6.0
VERIFY_CUT_SIDE_S = 4.0
VERIFY_CUT_Z = 3.0
VERIFY_CUT_MIN_MOVE_S = 0.2
VERIFY_CUT_MIN_STEP_S = 0.02
VERIFY_CUT_SNAP_S = 0.06
# A step this small is between two levels of one scene -- a trim of a few
# frames, or a dub whose background holds two copies of itself a tenth of a
# second apart -- where the music agrees at both and the effects decide
# (see BANDS): the cut is judged on the effects band alone.
VERIFY_CUT_SMALL_STEP_S = 0.3

# A span where the dub has something other than the video -- a replaced
# scene, an insert laid into a gap -- is told from a quiet one the offsets
# simply did not reach: there both tracks carry music and effects (the
# bass and the stereo difference, where centred dialogue does not reach),
# and they agree nowhere near the offset, in every band, for at least
# FOREIGN_MIN_S. What "agree" means is read off the pair itself: half the
# level a quarter of its own placed windows fall below -- a quartile, so a
# replaced scene inside the stretches sampled cannot drag it down -- held
# between FOREIGN_LIMIT_FLOOR and FOREIGN_LIMIT_CEIL, since a TV dub off
# another master agrees far less than two tracks of one Blu-ray.
FOREIGN_MIN_S = 10.0
FOREIGN_WINDOW_S = 4.0
FOREIGN_HOP_S = 2.0
FOREIGN_REACH_S = 0.25
FOREIGN_BANDS = ("full", "low", "side")
FOREIGN_ACTIVE = 0.5
FOREIGN_LIMIT_PERCENTILE = 25.0
FOREIGN_LIMIT_SHARE = 0.5
FOREIGN_LIMIT_FLOOR = 0.10
FOREIGN_LIMIT_CEIL = 0.35
FOREIGN_CALIBRATION_MIN = 30
# The dub's lines following the video's this clearly at the offset say the
# scene is the same after all (its music replaced, say), and it is kept.
FOREIGN_SPEECH_Z = 4.0
# Windows can each agree too little and the span still be the video's --
# a passage whose effects were mixed differently. Over the whole span, the
# agreement at the offset (every band, half a second at a time) must show
# nothing: a t-statistic under FOREIGN_MAX_T. Measured: a replaced scene
# reads -2.2 to +0.5, a weakly agreeing passage of the right scene 4.5, a
# plain one 14 to 20.
FOREIGN_MAX_T = 2.0
FOREIGN_T_BLOCK_S = 0.5
# A level too short for the readings: two trims a few seconds apart read as
# one step, because readings of POLISH_WINDOW_S every POLISH_SPACING_S see a
# level four seconds long once at most. Between two stretches whose offsets
# differ by SHORT_LEVEL_MIN_STEP_S to SHORT_LEVEL_MAX_STEP_S, every offset
# in between (on a SHORT_LEVEL_GRID_S grid, SHORT_LEVEL_APART_S clear of
# both) is asked over SHORT_LEVEL_REACH_S either side, in blocks of
# SHORT_LEVEL_BLOCK_S. One that agrees better than both neighbours across
# at least SHORT_LEVEL_MIN_S -- by a t-statistic of SHORT_LEVEL_Z over its
# blocks, and in at least SHORT_LEVEL_BANDS bands -- is a stretch of its
# own. Measured on the bench: trims 4.7 s apart, where every band preferred
# the level between them in every two-second window by 0.6-1.1.
SHORT_LEVEL_MIN_STEP_S = 0.04
SHORT_LEVEL_MAX_STEP_S = 0.5
SHORT_LEVEL_REACH_S = 10.0
SHORT_LEVEL_BLOCK_S = 0.5
SHORT_LEVEL_MIN_S = 1.5
SHORT_LEVEL_Z = 3.0
SHORT_LEVEL_BANDS = 2
SHORT_LEVEL_APART_S = 0.012
SHORT_LEVEL_GRID_S = 0.002
# The bracket each new edge is placed in, either side of the block edge.
SHORT_LEVEL_BRACKET_S = 0.75
# A reading whose peak is not this far above its own row found nothing:
# the tallest of a few hundred chance offsets reaches three, of a
# thousand three and a half.
READING_Z = 4.5
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
# How far past its neighbours' offsets a gap is searched for dub nobody
# has played yet (see ``_recover_from_gaps``).
GAP_UNUSED_MAX_S = 180.0
# A stretch found in a gap at its neighbour's measured offset, and
# overlapping it by more than this, is the neighbour reaching further.
# Only an overlap: pieces that merely meet were kept apart for a reason,
# and joined, a faint scene's offset was measured 12 ms out.
RECOVER_JOIN_OVERLAP_S = 0.01
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
# A step larger than BRIDGE_STEP_S that nothing placed puts whatever lands
# on the wrong side of it seconds away -- another moment of the film, not
# a lip a little out -- so only a second of it is risked; past that the
# original fills the gap. At five seconds a 22 s step was guessed across a
# gap and 4.6 s of an insert played over the picture.
BRIDGE_EXPOSURE_S = 1.0
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

# A track shorter than this cannot be laid on anything: a coarse window is 30s.
MIN_PLAN_S = 10.0

# Frame energy (RMS) below which a whole track is silence: -80 dBFS, well under
# any real mix's noise floor and above a dithered digital silence.
SILENT_ENERGY = 1e-4

# --- reporting the edits ----------------------------------------------------

# A change of offset smaller than this between two stretches is measurement,
# not an edit: the same floor ``segments.MIN_STEP_MS`` splits a stretch at.
EDIT_MIN_MS = 20.0

# How an edit is graded. A trim under a second is a frame or two of picture
# (minor); up to half a minute is a line or a shot (moderate); beyond that a
# scene (major). Every one is listed whatever its grade.
MINOR_EDIT_S = 1.0
MAJOR_EDIT_S = 30.0

# The dub carrying this much of its own material inside a cut makes the cut a
# replaced scene rather than a plain one. Below it, the unmatched dub is the
# slack of a cut placed to within a second or so.
REPLACED_MIN_S = 1.0

# An edit resting on a stretch of dub shorter than this is marked to check:
# long enough for a line or a shot, too short to rule out a repeated cue.
SHORT_STRETCH_S = 30.0

# Where lip-sync error becomes visible: ITU-R BT.1359's detectability bound
# for sound arriving early, the stricter of its two.
LIP_SYNC_VISIBLE_MS = 45.0

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


# Where a band's typical frame lands on the onset curve's log knee: the
# same place a full-band frame of 0.01 RMS lands, which is what the knee in
# ``_onsets`` was set for.
BAND_KNEE = 10.0


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


# ---------------------------------------------------------------------------
# The path
# ---------------------------------------------------------------------------

STAY_LEFT, STAY, STAY_RIGHT, JUMPED, FROM_NULL = 0, 1, 2, 3, 4


# Stretches of dub this close in offset are one offset: a delay.
DELAY_ONLY_TOLERANCE_S = 0.001


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


# A stretch whose dub reaches back over this much of the dub the stretch
# before it played is playing that dub twice; a rewind longer than
# WIDE_EPISODE_S is a dub made of episodes in another order, and allowed.
REUSE_MIN_S = 0.25


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


# Where neither the music and effects nor the picture can place a step, the
# two languages' lines can: a dub is performed to the lips, so its speech
# starts and stops about where the original's does. Their agreement is
# summed the way the sound's is (see _place_step), and the rise of that sum
# is only a placement when it is this many times the largest rise the same
# sum reaches with both offsets moved by these amounts, where nothing lines
# up: long spans of dialogue rise by chance, and the bar has to grow with them.
SPEECH_STEP_NULLS_S = (-2.1, -1.3, -0.7, 0.7, 1.3, 2.1)
SPEECH_STEP_MARGIN = 1.5


# How far a step the speech placed may be from a shot change and still be
# put on it: the speech places to a few tenths of a second.
SPEECH_SNAP_S = 0.5


# An edge the sound placed no better than this is looked for among the
# picture's shot changes: a frame and a half.
PICTURE_SNAP_MIN_S = 0.06


# A lone shot change is taken over the sound's own placement only while the
# sound cannot tell the two apart. Between them, in blocks this long, one
# side's offset agreeing better block after block, by this many standard
# errors, is the sound placing the edge itself: a dub's audio can be spliced
# where the picture has no cut. Measured on a real pair: a 74 ms splice in
# the middle of a shot, 2.1 s after the nearest cut, which a snap put the
# dub 74 ms early across.
PICTURE_VETO_BLOCK_S = 0.25
PICTURE_VETO_Z = 3.0


# A step in the offset no larger than this is a picture trim -- a few
# frames taken off one side of a shot change -- and sits on one cut; a
# larger one is a stretch of picture cut out, between two.
TRIM_MAX_S = 0.5
