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

Where the parts live
--------------------
This module keeps the planner's top level (``plan_dubsync`` and the order
of its passes, ``_refine``) and re-exports every name of its parts, which
are in ``audiosync.dubplan``: ``tuning`` (every number the planner is tuned
by, with what it was measured on), ``envelope``, ``scores`` (correlation,
the score grid and the path), ``plan`` (the plan and its pieces),
``aligner`` (every measurement of the two tracks), ``search`` (coarse pass,
speed, wide pass, gaps), ``levels`` (drift, two levels, music against
effects), ``trace`` (a gap second by second), ``edges`` (placing every
cut), ``foreign`` (another scene in the dub), ``assemble`` and ``verify``.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .framerate import _format_fps, exact_rate
from .media import CancellationToken, MediaError, audio_lead_s, probe
from .shots import PictureCuts

# Everything the planner's parts define, importable from here as before.
from .dubplan.tuning import (  # noqa: F401
    ANALYSIS_SR,
    ENVELOPE_RATE,
    BANDS,
    BAND_WINDOW,
    EFFECTS_Z,
    EFFECTS_RANGE_S,
    EFFECTS_DOMINANCE,
    EFFECTS_DISCOUNT,
    EFFECTS_VOTE_SLACK_S,
    SPEECH_ARBITER_S,
    SPEECH_PREFER_Z,
    SPEECH_PREFER_MARGIN,
    VERIFY_CUT_REACH_S,
    VERIFY_CUT_SIDE_S,
    VERIFY_CUT_Z,
    VERIFY_CUT_MIN_MOVE_S,
    VERIFY_CUT_MIN_STEP_S,
    VERIFY_CUT_SNAP_S,
    VERIFY_CUT_SMALL_STEP_S,
    FOREIGN_MIN_S,
    FOREIGN_WINDOW_S,
    FOREIGN_HOP_S,
    FOREIGN_REACH_S,
    FOREIGN_BANDS,
    FOREIGN_ACTIVE,
    FOREIGN_LIMIT_PERCENTILE,
    FOREIGN_LIMIT_SHARE,
    FOREIGN_LIMIT_FLOOR,
    FOREIGN_LIMIT_CEIL,
    FOREIGN_CALIBRATION_MIN,
    FOREIGN_SPEECH_Z,
    FOREIGN_MAX_T,
    FOREIGN_T_BLOCK_S,
    SHORT_LEVEL_MIN_STEP_S,
    SHORT_LEVEL_MAX_STEP_S,
    SHORT_LEVEL_REACH_S,
    SHORT_LEVEL_BLOCK_S,
    SHORT_LEVEL_MIN_S,
    SHORT_LEVEL_Z,
    SHORT_LEVEL_BANDS,
    SHORT_LEVEL_APART_S,
    SHORT_LEVEL_GRID_S,
    SHORT_LEVEL_BRACKET_S,
    READING_Z,
    FOLLOW_EFFECTS_STEP_S,
    FOLLOW_EFFECTS_GAP_S,
    EXCURSION_MIN_MS,
    EXCURSION_MAX_S,
    CREDIBLE_Z,
    CREDIBILITY_SPAN_S,
    CREDIBILITY_RANGE_S,
    COARSE_POOL,
    COARSE_WINDOW_S,
    COARSE_HOP_S,
    COARSE_CELL_BUDGET,
    DEFAULT_SEARCH_S,
    JUMP_COST,
    WOBBLE_COST,
    NULL_REWARD,
    NULL_COST,
    MATCH_Z,
    MIN_BLOCK_Z,
    BLOCK_BAR_MARGIN,
    SHORT_RUN_WINDOWS,
    MIN_BLOCK_MATCH,
    CONSISTENT_MS,
    CONSISTENT_FRACTION,
    CONSISTENT_READINGS,
    RELATIVE_FLOOR,
    RELATIVE_FLOOR_CAP,
    STRONG_SUPPORT_Z,
    TIGHT_MS,
    RECOVER_MIN_S,
    GAP_WINDOWS_S,
    GAP_HOP_S,
    GAP_POOL,
    RECOVER_SLACK_S,
    GAP_UNUSED_MAX_S,
    RECOVER_JOIN_OVERLAP_S,
    GAP_ONE_SIDED_MAX_S,
    GAP_JUMP_COST,
    GAP_NULL_REWARD,
    GAP_NULL_COST,
    RECOVER_Z,
    RECOVER_ROUNDS,
    EFFECTS_RECOVER_Z,
    EFFECTS_RECOVER_WINDOWS,
    POLISH_WINDOW_S,
    POLISH_RANGE_S,
    POLISH_WIDE_RANGE_S,
    SETTLE_MARGIN,
    SETTLE_SHARE,
    SETTLE_BASELINE_S,
    SETTLE_MIN_S,
    LOCAL_STATS_S,
    EDGE_UNCERTAINTY_STEPS,
    SHARPEN_Z,
    SHARPEN_SEARCH_S,
    SHARPEN_INTERIOR_S,
    SHARPEN_STATS_S,
    SHARPEN_MIN_UNCERTAINTY_S,
    SHARPEN_MAX_SPAN_S,
    CONTINUOUS_DUB_S,
    POLISH_SPACING_S,
    EDGE_TRIM_MIN_S,
    MIN_BLOCK_S,
    SAME_OFFSET_S,
    MIN_FILL_S,
    DUB_SILENT_RATIO,
    DUB_SILENT_FLOOR,
    FILL_AUDIBLE_RATIO,
    NEAR_OFFSET_S,
    EDGE_KEEP_MAX_S,
    BRIDGE_STEP_S,
    BRIDGE_EXPOSURE_S,
    BRIDGE_SOUND_EXPOSURE_S,
    GUESS_WARN_S,
    REWIND_GUARD_MAX_S,
    STEP_EVIDENCE,
    REVIEW_DOUBT_MAX,
    MAX_FILL_GAIN_DB,
    FILL_GAIN_PIECE_S,
    MIN_PLAN_S,
    SILENT_ENERGY,
    EDIT_MIN_MS,
    MINOR_EDIT_S,
    MAJOR_EDIT_S,
    REPLACED_MIN_S,
    SHORT_STRETCH_S,
    LIP_SYNC_VISIBLE_MS,
    MIN_DRIFT_MS_PER_S,
    MIN_DRIFT_SPAN_S,
    SPEED_FINE_GAIN,
    SPEED_ROUNDS,
    SPEED_RETRY_FRACTION,
    WIDE_PASS_COVERAGE,
    WIDE_WINDOW_S,
    WIDE_HOP_S,
    WIDE_Z,
    WIDE_SAME_OFFSET_S,
    WIDE_MIN_WINDOWS,
    WIDE_PEAKS,
    WIDE_PEAK_APART_S,
    WIDE_LOCAL_S,
    WIDE_ANCHOR_REACH_S,
    WIDE_REWIND_SLACK_S,
    WIDE_EPISODE_S,
    SPEED_MIN_Z,
    SPEED_GAIN,
    SPEED_TRIAL_WINDOWS,
    SPEED_TRIAL_WINDOW_S,
    SPEED_TRIAL_POOL,
    SPEED_BAND,
    MIN_SILENT_S,
    VERIFY_SPOTS,
    VERIFY_SPOT_WINDOW_S,
    VERIFY_SPOT_RANGE_S,
    VERIFY_SWEEP_WINDOW_S,
    VERIFY_SWEEP_POOL,
    VERIFY_SWEEP_RANGE_S,
    VERIFY_SWEEP_Z,
    VERIFY_STRETCH_WINDOWS,
    AUDIBLE_MS,
    STRETCH_MS,
    ProgressFn,
    LogFn,
    BAND_KNEE,
    SIDE_FILTER,
    SIDE_MIN_RATIO,
    STAY_LEFT,
    STAY,
    STAY_RIGHT,
    JUMPED,
    FROM_NULL,
    DELAY_ONLY_TOLERANCE_S,
    MIN_LEVEL_GAP_MS,
    LEVEL_SEPARATION,
    LEVEL_MIN_READINGS,
    RUN_MIN_READINGS,
    DRIFT_NEIGHBOURS,
    STEP_GAIN,
    AGREEMENT_FLOOR,
    EDGE_FIT_PASSES,
    EDGE_FIT_POSITION,
    DRIFT_SPAN_MS,
    DRIFT_STEP_MS,
    DRIFT_READING_WINDOW_S,
    DRIFT_READING_SPACING_S,
    DRIFT_OVER_SCATTER,
    DRIFT_FIT_GAIN,
    DRIFT_MAX_SCATTER_MS,
    DRIFT_GROUP_GAP_S,
    DRIFT_CORNER_REACH_S,
    DRIFT_MAX_RUN,
    BAND_WORDS,
    REUSE_MIN_S,
    SPLIT_MIN_S,
    SPLIT_REACH_S,
    SPLIT_APART_S,
    SPLIT_SECOND_Z,
    SPLIT_SECOND_SHARE,
    SPLIT_MIN_RUN_S,
    SPLIT_ROUNDS,
    TRACE_MIN_S,
    TRACE_MAX_STEP_S,
    TRACE_MIN_DUB_S,
    TRACE_BIN_S,
    TRACE_READING_S,
    TRACE_READING_HOP_S,
    TRACE_READING_Z,
    TRACE_MAX_LEVELS,
    TRACE_LEVEL_APART_S,
    TRACE_NOISE_APART_S,
    TRACE_NOISE_SPAN_S,
    TRACE_NOISE_SPACING_S,
    TRACE_JUMP,
    TRACE_CARRY_Z,
    TRACE_RUN_Z,
    TRACE_BEAT_Z,
    TRACE_EDGE_S,
    TRACE_EDGE_Z,
    TRACE_SHORT_CARRY_S,
    TRACE_STEP_REACH_S,
    SPEECH_STEP_NULLS_S,
    SPEECH_STEP_MARGIN,
    SPEECH_SNAP_S,
    PICTURE_SNAP_MIN_S,
    PICTURE_VETO_BLOCK_S,
    PICTURE_VETO_Z,
    TRIM_MAX_S,
)
from .dubplan.envelope import (  # noqa: F401
    TrackEnvelope,
    _onsets,
    _band_onsets,
    _frame_energies,
    _side_energy,
    build_envelope,
    _BandFrames,
    _pool,
)
from .dubplan.scores import (  # noqa: F401
    ncc_lags,
    _parabolic,
    _nearest_peak,
    _tallest_peaks,
    _continuing_candidate,
    _peak_dominance,
    _noise_units,
    ScoreGrid,
    score_windows,
    best_path,
    Block,
    _block_bar,
    blocks_from_path,
)
from .dubplan.plan import (  # noqa: F401
    UNMATCHED_NOTE,
    FILL_REASONS,
    GAP_WORDS,
    Segment,
    delay_only_text,
    DubSyncPlan,
    _clock,
    _amount,
    _duration,
    DRAFT_NOTE,
)
from .dubplan.aligner import (  # noqa: F401
    _local_standardize,
    _level_runs,
    _timeline_activity,
    _Aligner,
    _agreement_threshold,
    _fit_end,
    _fit_start,
    _fit_joint,
)
from .dubplan.search import (  # noqa: F401
    _coarse_blocks,
    _estimate_drift,
    _snap_speed,
    _fine_score,
    _speed_from_drift,
    _wide_blocks,
    _merge_blocks,
    _coverage,
    _fps_speed_candidates,
    _try_speeds,
    _recover_from_gaps,
    _consistent_in_gap,
    _gap_candidates,
    _search_gap,
)
from .dubplan.levels import (  # noqa: F401
    _theil_sen,
    _evidence_note,
    _drift_groups,
    _follow_drift,
    _resolve_reuse,
    _follow_effects,
    _two_level_path,
    _split_levels,
    _reads_as_ramp,
    _split_block_levels,
)
from .dubplan.trace import (  # noqa: F401
    _trace_gaps,
    _trace_levels,
    _TraceScores,
    _trace_gap,
)
from .dubplan.edges import (  # noqa: F401
    _credible,
    _measure_and_split,
    _settle,
    _place_all_edges,
    _place_edge,
    _fit_edges,
    _sharpen,
    _short_level,
    _split_short_levels,
    _verify_cuts,
    _join_overlapping,
    _place_step,
    _speech_step,
    _choose_cut,
    _sound_overrules,
    _snap_to_picture,
    _step_words,
)
from .dubplan.foreign import (  # noqa: F401
    _foreign_bands,
    _typical_activity,
    _active,
    _window_agreement,
    _calibrate_foreign,
    _foreign_runs,
    _agreement_t,
    _speech_agrees,
    _split_foreign,
    _foreign_edges,
    _gap_is_foreign,
)
from .dubplan.assemble import (  # noqa: F401
    _fill_gain_db,
    _is_quiet,
    _dub_is_silent,
    _level_over,
    _dub_level,
    _original_level,
    _trim_uncertain_edges,
    _split_silences,
    _join,
    _assemble,
)
from .dubplan.verify import (  # noqa: F401
    SpotCheck,
    Stretch,
    Verification,
    _ms,
    _in_fill,
    verify_output,
    _measure,
)



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
        damage: Dict[str, List[str]] = {video_path: [], dub_path: []}
        primary = build_envelope(
            video_path, video_track, token,
            lambda f: report(2 + 18 * f, "reading the original"),
            plan.video_duration_s, lead_s=audio_lead_s(video_info, video_track),
            issues=damage[video_path],
        )
        report(20, "reading the dub")
        secondary_native = build_envelope(
            dub_path, dub_track, token,
            lambda f: report(20 + 18 * f, "reading the dub"),
            plan.dub_duration_s, lead_s=audio_lead_s(dub_info, dub_track),
            issues=damage.setdefault(dub_path, []),
        )
        for path, lines in damage.items():
            if lines:
                plan.warnings.append(
                    f"{os.path.basename(path)} is damaged: FFmpeg hit {len(lines)} decoding error"
                    f"{'s' if len(lines) != 1 else ''} reading it and decoded past them (first: {lines[0]}). "
                    "Anything measured across the damage may be off; a clean copy of the file is the fix."
                )
        # The decoded length is the truth; the probe was only a progress hint.
        plan.video_duration_s = primary.duration_s
        plan.dub_duration_s = secondary_native.duration_s
        # Said in the file's terms rather than as "no part of the dub could be
        # matched", which is true of a silent or two-second dub but names
        # nothing to fix.
        for envelope, path, role in ((primary, video_path, "video"), (secondary_native, dub_path, "dub")):
            name = os.path.basename(path)
            if envelope.duration_s < MIN_PLAN_S:
                plan.error = (
                    f"{name} is only {envelope.duration_s:.1f} s long; laying a dub on a video needs at least "
                    f"{MIN_PLAN_S:.0f} s of audio."
                )
                return plan
            if envelope.energy.size == 0 or float(np.max(envelope.energy)) < SILENT_ENERGY:
                plan.error = (
                    f"The {role}'s audio ({name}) is silent from start to end. It may be an empty or muted "
                    "track, or the wrong track of the file."
                )
                return plan
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
        _verify_cuts(aligner, plan.segments, plan.notes)
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
        blocks = _join_overlapping(blocks)
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
    # Where the dub carries another scene inside a stretch -- replaced, or
    # an insert the offsets either side happen to meet across -- the
    # original plays, and the assembly does not bridge it again.
    blocks = _split_foreign(aligner, blocks, notes)
    # Two trims a few seconds apart, read as one step: the level between
    # them is put in on its own.
    blocks = _split_short_levels(aligner, blocks, primary_s, secondary_s, notes)
    show(blocks)
    return blocks


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
