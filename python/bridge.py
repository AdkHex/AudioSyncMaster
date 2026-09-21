"""stdin/stdout bridge between the Tauri host and the analysis engine.

Protocol
--------
The host writes one JSON request per line on stdin and reads newline-delimited
JSON events from stdout. Reading commands line-by-line (rather than slurping
stdin to EOF as the original did) means the host can send ``cancel`` while a
run is in flight, and removes the deadlock risk of writing a large request into
a pipe whose reader is not yet draining stdout.

All field names crossing this boundary are camelCase, matching the Rust structs
exactly. The original mixed conventions -- Python emitted ``elapsed_ms`` while
Rust deserialized ``elapsedMs`` with no rename -- so timing data silently
became null on the completion path.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
else:
    BASE_DIR = ROOT_DIR
for candidate in (BASE_DIR, ROOT_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from audiosync.batch import BatchEvents, BatchOptions, run_batch, summarize
    from audiosync.matching import (
        MAX_COMPARE_INPUTS,
        MatchPair,
        list_media,
        match_folders,
        match_lists,
        match_movies,
        pair_every_combination,
        pair_movie_mode,
        validate_pattern,
    )
    from audiosync.dubrender import EXCERPT_KINDS, RenderOptions, default_output_path, excerpt, resolve_codec
    from audiosync.dubrender import mux as mux_dub
    from audiosync.dubrender import render as render_dub
    from audiosync.dubsync import DubSyncPlan, build_envelope, plan_dubsync, verify_output
    from audiosync.waveform import is_loaded as waveform_loaded
    from audiosync.waveform import load as waveform_load
    from audiosync.waveform import peaks as waveform_peaks
    from audiosync.media import Cancelled, CancellationToken, MediaError, has_ffmpeg, probe
    from audiosync.mux import (
        apply_correction,
        choose_preview_position,
        command_string,
        extract_preview,
        plan_correction,
    )
except Exception as exc:  # noqa: BLE001
    sys.stderr.write(f"Failed to import audiosync package: {exc}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

_write_lock = threading.Lock()
_active_token: CancellationToken | None = None
_token_lock = threading.Lock()


def emit(payload: dict) -> None:
    """Write one event. Serialized so concurrent workers cannot interleave."""
    with _write_lock:
        try:
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, ValueError):
            # Host went away; nothing useful left to do.
            pass


def emit_log(message: str) -> None:
    emit({"type": "log", "message": message})


def emit_error(message: str, fatal: bool = False) -> None:
    emit({"type": "error", "message": message, "fatal": fatal})


def _events() -> BatchEvents:
    return BatchEvents(
        on_log=emit_log,
        on_pair_start=lambda name: emit({"type": "fileStart", "file": name}),
        on_pair_progress=lambda name, percent: emit(
            {"type": "fileProgress", "file": name, "percent": percent}
        ),
        on_pair_done=lambda result: emit({"type": "result", **result.to_dict()}),
        on_progress=lambda done, total, current: emit(
            {"type": "progress", "processed": done, "total": total, "current": current}
        ),
    )


def _set_token(token: CancellationToken | None) -> None:
    global _active_token
    with _token_lock:
        _active_token = token


def handle_analyze(request: dict) -> None:
    """Run a full analysis batch described by the request."""
    if not has_ffmpeg():
        emit_error(
            "FFmpeg was not found. Install it and ensure it is on PATH, or bundle "
            "it in src-tauri/resources/ffmpeg.",
            fatal=True,
        )
        emit({"type": "done", "results": [], "summary": summarize([])})
        return

    mode = request.get("mode")
    options = BatchOptions(
        window_s=float(request.get("windowSeconds", 45.0)),
        window_count=int(request.get("windowCount", 6)),
        max_offset_ms=float(request.get("maxOffsetMs", 60000.0)),
        max_workers=int(request.get("maxWorkers", 3)),
        fast=bool(request.get("fast", False)),
    )

    # A pairing the user corrected by hand wins outright. Re-matching here
    # would silently undo their edit, which is worse than not offering the edit.
    explicit = request.get("pairs")
    if explicit:
        pairs = []
        for entry in explicit:
            video = entry.get("primaryPath")
            audio = entry.get("secondaryPath")
            if not video or not audio:
                continue
            if not os.path.isfile(video) or not os.path.isfile(audio):
                emit_log(f"Skipping a pair whose files are missing: {os.path.basename(video or '?')}")
                continue
            pairs.append(
                MatchPair(
                    video,
                    audio,
                    entry.get("key") or os.path.basename(video),
                    entry.get("method") or "chosen by hand",
                    float(entry.get("score", 1.0) or 1.0),
                    primary_track=int(entry.get("primaryTrack", request.get("videoTrack", 0)) or 0),
                    secondary_track=int(entry.get("secondaryTrack", request.get("audioTrack", 0)) or 0),
                )
            )

        if not pairs:
            emit_error("None of the chosen pairs could be used.", fatal=True)
            emit({"type": "done", "results": [], "summary": summarize([])})
            return

        emit_log(f"Using {len(pairs)} pair(s) supplied by the app.")
        emit({
            "type": "pairs",
            "pairs": [pair.to_dict() for pair in pairs],
            "method": "supplied",
            "unmatchedPrimary": [],
            "unmatchedSecondary": [],
            "warning": None,
        })

    elif mode == "movie":
        audio_file = request.get("audioFile")
        if not audio_file or not os.path.isfile(audio_file):
            emit_error("Select an audio file for movie mode.", fatal=True)
            emit({"type": "done", "results": [], "summary": summarize([])})
            return

        video_files = request.get("videoFiles") or []
        if not video_files:
            folder = request.get("videoFolder")
            video_files = list_media(folder, "video") if folder else []
        video_files = [p for p in video_files if os.path.isfile(p)]

        if not video_files:
            emit_error("No video files found to analyse.", fatal=True)
            emit({"type": "done", "results": [], "summary": summarize([])})
            return

        pairs = pair_movie_mode(
            video_files,
            audio_file,
            primary_track=int(request.get("videoTrack", 0) or 0),
            secondary_track=int(request.get("audioTrack", 0) or 0),
        )
        emit(
            {
                "type": "pairs",
                "pairs": [pair.to_dict() for pair in pairs],
                "method": "movie mode",
                "unmatchedPrimary": [],
                "unmatchedSecondary": [],
                "warning": None,
            }
        )

    elif mode == "compare":
        # Every video against every audio: answers "which release was this dub
        # timed for?" rather than assuming the pairing is already known.
        video_files = [p for p in (request.get("videoFiles") or []) if os.path.isfile(p)]
        audio_files = [p for p in (request.get("audioFiles") or []) if os.path.isfile(p)]

        if not video_files or not audio_files:
            emit_error(
                "Compare mode needs at least one video and one audio file.", fatal=True
            )
            emit({"type": "done", "results": [], "summary": summarize([])})
            return

        if len(video_files) > MAX_COMPARE_INPUTS or len(audio_files) > MAX_COMPARE_INPUTS:
            emit_error(
                f"Compare mode allows up to {MAX_COMPARE_INPUTS} files per side; "
                f"got {len(video_files)} video and {len(audio_files)} audio. "
                "The work grows as the product of both sides.",
                fatal=True,
            )
            emit({"type": "done", "results": [], "summary": summarize([])})
            return

        pairs = pair_every_combination(
            video_files,
            audio_files,
            primary_track=int(request.get("videoTrack", 0) or 0),
            secondary_track=int(request.get("audioTrack", 0) or 0),
        )
        emit_log(
            f"Comparing {len(video_files)} video file(s) against "
            f"{len(audio_files)} audio file(s): {len(pairs)} combinations."
        )
        emit(
            {
                "type": "pairs",
                "pairs": [pair.to_dict() for pair in pairs],
                "method": "every combination",
                "unmatchedPrimary": [],
                "unmatchedSecondary": [],
                "warning": None,
            }
        )

    elif mode == "series":
        video_folder = request.get("videoFolder")
        audio_folder = request.get("audioFolder")
        if not video_folder or not audio_folder:
            emit_error("Select both a video folder and an audio folder.", fatal=True)
            emit({"type": "done", "results": [], "summary": summarize([])})
            return

        pattern = request.get("matchPattern") or None
        if pattern:
            valid, problem = validate_pattern(pattern)
            if not valid:
                emit_error(f"Match pattern rejected: {problem}", fatal=True)
                emit({"type": "done", "results": [], "summary": summarize([])})
                return

        report = match_folders(video_folder, audio_folder, pattern)
        video_track = int(request.get("videoTrack", 0) or 0)
        audio_track = int(request.get("audioTrack", 0) or 0)
        if video_track or audio_track:
            for pair in report.pairs:
                pair.primary_track = video_track
                pair.secondary_track = audio_track
        emit({"type": "pairs", **report.to_dict()})
        if report.warning:
            emit_log(report.warning)
        pairs = report.pairs
        if not pairs:
            emit_error(report.warning or "No file pairs could be matched.", fatal=True)
            emit({"type": "done", "results": [], "summary": summarize([])})
            return
    else:
        emit_error(f"Unknown mode: {mode}", fatal=True)
        emit({"type": "done", "results": [], "summary": summarize([])})
        return

    token = CancellationToken()
    _set_token(token)
    try:
        results = run_batch(pairs, options, token, _events())
    finally:
        _set_token(None)

    payload = [result.to_dict() for result in results]
    emit(
        {
            "type": "done",
            "results": payload,
            "summary": summarize(results),
            "cancelled": token.cancelled,
        }
    )


def handle_preview_pairs(request: dict) -> None:
    """Show what would be paired, without analysing anything."""
    mode = request.get("mode")
    if mode == "series":
        pattern = request.get("matchPattern") or None
        if pattern:
            valid, problem = validate_pattern(pattern)
            if not valid:
                emit({"type": "pairs", "pairs": [], "warning": problem, "method": "none"})
                return
        report = match_folders(
            request.get("videoFolder") or "",
            request.get("audioFolder") or "",
            pattern,
        )
        emit({"type": "pairs", **report.to_dict()})
        return

    if mode == "compare":
        videos = [p for p in (request.get("videoFiles") or []) if os.path.isfile(p)]
        audios = [p for p in (request.get("audioFiles") or []) if os.path.isfile(p)]
        pairs = pair_every_combination(videos, audios)
        over = len(videos) > MAX_COMPARE_INPUTS or len(audios) > MAX_COMPARE_INPUTS
        emit(
            {
                "type": "pairs",
                "pairs": [pair.to_dict() for pair in pairs],
                "method": "every combination",
                "unmatchedPrimary": [],
                "unmatchedSecondary": [],
                "warning": (
                    f"Up to {MAX_COMPARE_INPUTS} files per side are allowed."
                    if over
                    else None
                ),
            }
        )
        return

    if mode == "dubsync":
        # Each episode or movie against its own dub: episode numbers where the
        # names carry them, filename similarity otherwise. The movies scope
        # skips episode matching entirely, since a movie title never carries
        # one -- its pairing is similarity alone.
        videos = [p for p in (request.get("videoFiles") or []) if os.path.isfile(p)]
        dubs = [p for p in (request.get("audioFiles") or []) if os.path.isfile(p)]
        if not videos and request.get("videoFolder"):
            videos = list_media(request["videoFolder"], "video")
        if not dubs and request.get("audioFolder"):
            dubs = list_media(request["audioFolder"])
        report = (
            match_movies(videos, dubs)
            if request.get("dubKind") == "movies"
            else match_lists(videos, dubs)
        )
        emit({"type": "pairs", **report.to_dict()})
        return

    audio_file = request.get("audioFile") or ""
    video_files = request.get("videoFiles") or list_media(
        request.get("videoFolder") or "", "video"
    )
    pairs = pair_movie_mode(video_files, audio_file) if audio_file else []
    emit(
        {
            "type": "pairs",
            "pairs": [pair.to_dict() for pair in pairs],
            "method": "movie mode",
            "unmatchedPrimary": [] if audio_file else video_files,
            "unmatchedSecondary": [],
            "warning": None if audio_file else "No audio file selected.",
        }
    )


def handle_probe(request: dict) -> None:
    path = request.get("path")
    if not path:
        emit_error("No path supplied to probe.")
        return
    try:
        info = probe(path)
        emit(
            {
                "type": "probe",
                "path": path,
                "hasAudio": info.has_audio,
                "hasVideo": info.has_video,
                "duration": info.duration,
                "audioCodec": info.audio_codec,
                "sampleRate": info.sample_rate,
                "channels": info.channels,
                "fps": info.fps,
                "audioTracks": [t.to_dict() for t in info.audio_tracks],
            }
        )
    except MediaError as exc:
        emit({"type": "probe", "path": path, "error": str(exc)})


def handle_list_tracks(request: dict) -> None:
    """List the selectable audio streams of each requested file.

    A container often carries an original language, a dub and a commentary.
    Without this the UI cannot offer a choice and every comparison silently
    uses the first stream.
    """
    results = []
    for path in request.get("paths") or []:
        entry = {"path": path, "name": os.path.basename(path)}
        try:
            info = probe(path)
            entry["tracks"] = [t.to_dict() for t in info.audio_tracks]
            entry["fps"] = info.fps
            entry["duration"] = info.duration
        except MediaError as exc:
            entry["tracks"] = []
            entry["error"] = str(exc)
        results.append(entry)
    emit({"type": "tracks", "files": results})


def handle_preview(request: dict) -> None:
    """Render a short aligned excerpt so the user can check a result by ear.

    Written to a temp directory rather than beside the source: a preview is a
    throwaway, and cluttering the user's media folder with them is not.
    """
    video = request.get("videoPath")
    audio = request.get("audioPath")
    delay = request.get("delayMs")

    if not video or not audio or delay is None:
        emit_error("A preview needs a video, an audio track and a measured delay.")
        emit({"type": "previewDone", "path": None})
        return

    duration = float(request.get("durationSeconds", 12.0))
    position = request.get("positionSeconds")
    if position is None:
        try:
            position = choose_preview_position(probe(video).duration, duration)
        except MediaError:
            position = 0.0

    output = os.path.join(
        tempfile.gettempdir(),
        f"audiosync-preview-{abs(hash((video, audio, round(float(delay), 3))))}.mp4",
    )

    token = CancellationToken()
    _set_token(token)
    try:
        extract_preview(
            video,
            audio,
            float(delay),
            float(position),
            duration,
            output,
            token=token,
            audio_track=int(request.get("audioTrack", 0) or 0),
            drift_ms_per_s=request.get("driftMsPerS"),
        )
        emit({
            "type": "previewDone",
            "path": output,
            "positionSeconds": float(position),
            "durationSeconds": duration,
        })
    except (MediaError, OSError) as exc:
        emit_error(f"Could not render the preview: {exc}")
        emit({"type": "previewDone", "path": None})
    finally:
        _set_token(None)


def handle_apply(request: dict) -> None:
    """Write corrected files for the supplied set of measured results."""
    items = request.get("items") or []
    if not items:
        emit({"type": "applyDone", "written": [], "failed": []})
        return

    token = CancellationToken()
    _set_token(token)
    written, failed = [], []
    try:
        for index, item in enumerate(items):
            if token.cancelled:
                break
            video = item.get("videoPath")
            audio = item.get("audioPath")
            # Corrections apply from t=0, so use the start-referenced offset.
            # With drift, delayMs is the midpoint value and would over-shift.
            delay = item.get("delayAtStartMs")
            if delay is None:
                delay = item.get("delayMs")
            if not video or not audio or delay is None:
                failed.append({"video": video, "error": "Incomplete correction request"})
                continue

            plan = plan_correction(
                video, audio, float(delay),
                drift_ms_per_s=item.get("driftMsPerS"),
                output_dir=request.get("outputDir"),
                suffix=request.get("suffix", ".synced"),
            )
            emit(
                {
                    "type": "applyStart",
                    "file": os.path.basename(video),
                    "output": plan.output_path,
                    "description": plan.describe(),
                    "command": command_string(plan),
                }
            )
            try:
                output = apply_correction(
                    plan, token, overwrite=bool(request.get("overwrite"))
                )
                written.append(output)
                emit({"type": "applyProgress", "done": index + 1, "total": len(items),
                      "file": os.path.basename(video), "output": output})
            except (MediaError, OSError) as exc:
                failed.append({"video": os.path.basename(video), "error": str(exc)})
                emit_log(f"Failed to write {os.path.basename(video)}: {exc}")
    finally:
        _set_token(None)

    emit({"type": "applyDone", "written": written, "failed": failed,
          "cancelled": token.cancelled})


def handle_dubsync(request: dict) -> None:
    """Lay a cut dub onto its video and write the result.

    Plans, renders, verifies and optionally muxes in one command, reporting
    each stage. A ``plan`` in the request skips the analysis and renders
    that plan, which is how an edited plan comes back from the app.
    """
    if not has_ffmpeg():
        emit_error("FFmpeg was not found.", fatal=True)
        emit({"type": "dubsyncDone", "error": "FFmpeg was not found"})
        return

    token = CancellationToken()
    _set_token(token)
    started = {"plan": None}

    def progress(percent: int, stage: str) -> None:
        emit({"type": "dubsyncProgress", "percent": percent, "stage": stage})

    try:
        envelopes: dict = {}
        if request.get("plan"):
            plan = DubSyncPlan.from_dict(request["plan"])
        else:
            video = request.get("videoPath")
            dub = request.get("dubPath")
            if not video or not dub or not os.path.isfile(video) or not os.path.isfile(dub):
                emit_error("A video and a dub are needed.", fatal=True)
                emit({"type": "dubsyncDone", "error": "A video and a dub are needed"})
                return
            plan = plan_dubsync(
                video, dub,
                video_track=int(request.get("videoTrack", 0) or 0),
                dub_track=int(request.get("dubTrack", 0) or 0),
                search_s=float(request.get("searchSeconds", 120.0) or 120.0),
                speed=request.get("speed"),
                dub_rate=request.get("dubRate"),
                fill_gain_db=request.get("fillGainDb"),
                keep_unmatched_dub=not bool(request.get("fillUnmatched")),
                token=token, progress=progress, log=emit_log, envelopes=envelopes,
                draft=lambda sketch: emit({"type": "dubsyncDraft", "plan": sketch.to_dict()}),
            )
        started["plan"] = plan
        emit({"type": "dubsyncPlan", "plan": plan.to_dict(), "description": plan.describe()})
        if plan.error:
            emit({"type": "dubsyncDone", "plan": plan.to_dict(), "error": plan.error})
            return
        if request.get("planOnly"):
            emit({"type": "dubsyncDone", "plan": plan.to_dict(), "output": None})
            return

        codec = resolve_codec(request.get("codec") or "same", plan, token)
        output = request.get("outputPath") or default_output_path(plan, codec, request.get("outputDir"))
        if os.path.exists(output) and not request.get("overwrite"):
            raise MediaError(f"Output already exists: {os.path.basename(output)}")
        options = RenderOptions(
            codec=codec,
            bitrate=request.get("bitrate"),
            sample_rate=request.get("sampleRate"),
            channels=request.get("channels"),
            xfade_s=float(request.get("xfadeMs", 10.0) or 10.0) / 1000.0,
            stretch=request.get("stretch") or "resample",
        )
        result = render_dub(plan, output, options, token=token, progress=progress, log=emit_log)
        for warning in result.warnings:
            emit_log(warning)

        verification = None
        if request.get("verify", True):
            stage = "checking the finished track"
            progress(0, stage)
            primary = envelopes.get("primary") or build_envelope(
                plan.video_path, plan.video_track, token,
                lambda f: progress(int(50 * f), stage), plan.video_duration_s,
            )
            finished = build_envelope(
                output, token=token,
                progress=lambda f: progress(50 + int(45 * f), stage),
                expected_duration_s=plan.video_duration_s,
            )
            verification = verify_output(primary, finished, plan, token)
            progress(100, stage)
            emit_log(verification.describe())

        muxed = None
        if request.get("mux"):
            progress(0, "muxing")
            muxed = mux_dub(
                plan, output, request.get("muxPath"),
                language=request.get("language"), title=request.get("title"),
                token=token, overwrite=bool(request.get("overwrite")),
            )
        emit({
            "type": "dubsyncDone",
            "plan": plan.to_dict(),
            "output": result.to_dict(),
            "verification": verification.to_dict() if verification else None,
            "verificationText": verification.describe() if verification else None,
            "muxedPath": muxed,
            "cancelled": token.cancelled,
        })
    except Cancelled:
        emit({"type": "dubsyncDone", "plan": started["plan"].to_dict() if started["plan"] else None,
              "cancelled": True})
    except (MediaError, OSError) as exc:
        emit_error(str(exc))
        emit({"type": "dubsyncDone", "plan": started["plan"].to_dict() if started["plan"] else None,
              "error": str(exc)})
    finally:
        _set_token(None)


def _run_dub_job(index: int, job: dict, options: dict, token: CancellationToken) -> dict:
    """One pair of the batch: plan, render, verify, optionally mux.

    Runs on its own pool thread; every event carries the job index so the
    app can route it to the right queue row.
    """
    video = job.get("videoPath")
    dub = job.get("dubPath")

    def progress(percent: int, stage: str) -> None:
        emit({"type": "dubsyncJobProgress", "job": index, "percent": percent, "stage": stage})

    def log(message: str) -> None:
        emit({"type": "dubsyncJobLog", "job": index, "message": message})

    def outcome(**extra) -> dict:
        payload = {"job": index}
        payload.update(extra)
        return payload

    emit({
        "type": "dubsyncJobStart", "job": index,
        "name": os.path.basename(video), "dub": os.path.basename(dub),
    })
    if token.cancelled:
        return outcome(cancelled=True)

    envelopes: dict = {}
    try:
        plan = plan_dubsync(
            video, dub,
            video_track=int(job.get("videoTrack", 0) or 0),
            dub_track=int(job.get("dubTrack", 0) or 0),
            search_s=float(options.get("searchSeconds", 120.0) or 120.0),
            speed=options.get("speed"),
            dub_rate=options.get("dubRate"),
            fill_gain_db=options.get("fillGainDb"),
            keep_unmatched_dub=not bool(options.get("fillUnmatched")),
            token=token, progress=progress, log=log, envelopes=envelopes,
            draft=lambda sketch: emit({"type": "dubsyncJobDraft", "job": index, "plan": sketch.to_dict()}),
        )
    except Cancelled:
        return outcome(cancelled=True)
    except (MediaError, OSError) as exc:
        return outcome(error=str(exc))

    emit({"type": "dubsyncJobPlan", "job": index, "plan": plan.to_dict(),
          "description": plan.describe()})
    if plan.error:
        return outcome(plan=plan.to_dict(), error=plan.error)
    if options.get("planOnly"):
        return outcome(plan=plan.to_dict())

    try:
        codec = resolve_codec(options.get("codec") or "same", plan, token)
        output = job.get("outputPath") or default_output_path(
            plan, codec, options.get("outputDir"))
        if os.path.exists(output) and not options.get("overwrite"):
            raise MediaError(f"Output already exists: {os.path.basename(output)}")
        render_options = RenderOptions(
            codec=codec,
            bitrate=options.get("bitrate"),
            sample_rate=options.get("sampleRate"),
            channels=options.get("channels"),
            xfade_s=float(options.get("xfadeMs", 10.0) or 10.0) / 1000.0,
            stretch=options.get("stretch") or "resample",
        )
        result = render_dub(plan, output, render_options, token=token,
                            progress=progress, log=log)
        for warning in result.warnings:
            log(warning)

        verification = None
        if options.get("verify", True):
            stage = "checking the finished track"
            progress(0, stage)
            primary = envelopes.get("primary") or build_envelope(
                plan.video_path, plan.video_track, token,
                lambda f: progress(int(50 * f), stage), plan.video_duration_s,
            )
            finished = build_envelope(
                output, token=token,
                progress=lambda f: progress(50 + int(45 * f), stage),
                expected_duration_s=plan.video_duration_s,
            )
            verification = verify_output(primary, finished, plan, token)
            progress(100, stage)
            log(verification.describe())

        muxed = None
        if options.get("mux"):
            progress(0, "muxing")
            muxed = mux_dub(
                plan, output, job.get("muxPath"),
                language=options.get("language"), title=options.get("title"),
                token=token, overwrite=bool(options.get("overwrite")),
            )
        return outcome(
            plan=plan.to_dict(),
            output=result.to_dict(),
            verification=verification.to_dict() if verification else None,
            verificationText=verification.describe() if verification else None,
            muxedPath=muxed,
        )
    except Cancelled:
        return outcome(plan=plan.to_dict(), cancelled=True)
    except (MediaError, OSError) as exc:
        return outcome(plan=plan.to_dict(), error=str(exc))


def handle_dubsync_batch(request: dict) -> None:
    """Sync a queue of pairs in parallel: a season of episodes, or several
    movies at once. Each pair runs the full dub sync (plan, write, check,
    optional mux) on its own thread; the app routes events by job index."""
    if not has_ffmpeg():
        emit_error("FFmpeg was not found.", fatal=True)
        emit({"type": "dubsyncBatchDone", "outcomes": [], "error": "FFmpeg was not found"})
        return

    jobs = [j for j in (request.get("jobs") or []) if j.get("videoPath") and j.get("dubPath")]
    if not jobs:
        emit_error("No pairs to sync.", fatal=True)
        emit({"type": "dubsyncBatchDone", "outcomes": [], "error": "No pairs to sync"})
        return

    token = CancellationToken()
    _set_token(token)
    max_workers = int(request.get("maxWorkers", 3) or 3)
    max_workers = max(1, min(max_workers, 8, len(jobs)))
    outcomes: dict = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dubsync") as pool:
            futures = {
                pool.submit(_run_dub_job, index, job, request, token): index
                for index, job in enumerate(jobs)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outcomes[index] = future.result()
                except Exception as exc:  # noqa: BLE001 - one job must not sink the batch
                    outcomes[index] = {"job": index, "error": f"{type(exc).__name__}: {exc}"}
                    emit_log(f"Job {index + 1} failed: {exc}")
                emit({"type": "dubsyncJobDone", **outcomes[index]})
    finally:
        _set_token(None)

    ordered = [outcomes.get(i) or {"job": i, "error": "No outcome was produced"} for i in range(len(jobs))]
    emit({"type": "dubsyncBatchDone", "outcomes": ordered, "cancelled": token.cancelled})


def _waveform_progress(path: str, track: int, request_id) -> Callable[[float], None]:
    """Progress of a first read of a file, once per whole percent."""
    last = [-1]

    def report(fraction: float) -> None:
        percent = int(100 * max(0.0, min(1.0, fraction)))
        if percent != last[0]:
            last[0] = percent
            emit({"type": "waveformProgress", "path": path, "track": track,
                  "requestId": request_id, "percent": percent})

    return report


def handle_waveform_build(request: dict) -> None:
    """Read a track's waveform into the cache, so the views that follow are
    immediate. Reports progress; the app asks for this as soon as a pair
    is on screen."""
    path = request.get("path")
    track = int(request.get("track", 0) or 0)
    request_id = request.get("requestId")
    if not path or not os.path.isfile(path):
        emit_error("The file to draw was not found.")
        emit({"type": "waveformReady", "path": path, "track": track, "requestId": request_id,
              "error": "file not found"})
        return
    token = CancellationToken()
    _set_token(token)
    try:
        wave = waveform_load(path, track, token, _waveform_progress(path, track, request_id))
        emit({"type": "waveformReady", "path": path, "track": track, "requestId": request_id,
              "durationS": wave.duration_s, "channels": wave.channels, "sampleRate": wave.sample_rate})
    except Cancelled:
        emit({"type": "waveformReady", "path": path, "track": track, "requestId": request_id, "cancelled": True})
    except (MediaError, OSError) as exc:
        emit_error(f"Could not read the waveform: {exc}")
        emit({"type": "waveformReady", "path": path, "track": track, "requestId": request_id, "error": str(exc)})
    finally:
        _set_token(None)


def handle_waveform_peaks(request: dict) -> None:
    """Waveform peaks of one track over a span, for the dub sync view: per
    channel, the lowest and highest sample and the RMS in each bucket.

    ``startS``/``endS`` are on the timeline the plan uses for the file: the
    video's own clock, on which a dub played at ``speed`` is stretched. A
    file not read yet is read first, with progress.
    """
    path = request.get("path")
    track = int(request.get("track", 0) or 0)
    request_id = request.get("requestId")
    if not path or not os.path.isfile(path):
        emit_error("The file to draw was not found.")
        emit({"type": "waveformPeaks", "path": path, "requestId": request_id, "error": "file not found"})
        return
    token = CancellationToken()
    _set_token(token)
    try:
        progress = None if waveform_loaded(path, track) else _waveform_progress(path, track, request_id)
        result = waveform_peaks(
            path, track,
            float(request.get("startS", 0.0) or 0.0), float(request.get("endS", 0.0) or 0.0),
            int(request.get("buckets", 1000) or 1000), float(request.get("speed", 1.0) or 1.0),
            token, progress,
        )
        result.update({"type": "waveformPeaks", "path": path, "track": track, "requestId": request_id})
        emit(result)
    except Cancelled:
        emit({"type": "waveformPeaks", "path": path, "requestId": request_id, "cancelled": True})
    except (MediaError, OSError) as exc:
        emit_error(f"Could not read the waveform: {exc}")
        emit({"type": "waveformPeaks", "path": path, "requestId": request_id, "error": str(exc)})
    finally:
        _set_token(None)


def handle_dubsync_preview(request: dict) -> None:
    """Render a span of a plan -- as edited in the app -- to play back.

    ``what`` chooses the piece (see ``dubrender.excerpt``): ``both`` (the
    default; also ``video: false`` for ``audio``) for an outside player,
    ``picture`` / ``audio`` / ``original`` for the app's own player, which
    draws the picture from one file and plays the sound from another. The
    reply carries the span actually cut: a cut with a picture starts on a
    frame, and the sound for the same window must be asked for from that
    start.
    """
    if not request.get("plan"):
        emit_error("A preview needs a plan.")
        emit({"type": "dubsyncPreviewDone", "path": None})
        return
    token = CancellationToken()
    _set_token(token)
    try:
        plan = DubSyncPlan.from_dict(request["plan"])
        lo = float(request.get("startS", 0.0) or 0.0)
        hi = float(request.get("endS", lo + 12.0) or (lo + 12.0))
        what = request.get("what") or ("both" if request.get("video", True) else "audio")
        if what not in EXCERPT_KINDS:
            raise MediaError(f"Unknown excerpt {what!r}")
        key = abs(hash((plan.video_path, plan.dub_path, round(lo, 3), round(hi, 3), what, json.dumps(request["plan"], sort_keys=True))))
        ext = "mp4" if what in ("both", "picture") else "wav"
        output = os.path.join(tempfile.gettempdir(), f"audiosync-dub-preview-{key}.{ext}")
        path, start, end = excerpt(plan, lo, hi, output, what, token=token, log=emit_log)
        emit({"type": "dubsyncPreviewDone", "path": path, "what": what, "startS": start, "endS": end})
    except Cancelled:
        emit({"type": "dubsyncPreviewDone", "path": None, "cancelled": True})
    except (MediaError, OSError) as exc:
        emit_error(f"Could not render the preview: {exc}")
        emit({"type": "dubsyncPreviewDone", "path": None, "error": str(exc)})
    finally:
        _set_token(None)


def handle_cancel(request: dict) -> None:
    with _token_lock:
        token = _active_token
    if token:
        token.cancel()
        emit_log("Cancellation requested.")
    if request.get("command") == "cancel":
        emit({"type": "cancelAck"})


HANDLERS = {
    "analyze": handle_analyze,
    "previewPairs": handle_preview_pairs,
    "probe": handle_probe,
    "listTracks": handle_list_tracks,
    "preview": handle_preview,
    "apply": handle_apply,
    "dubsync": handle_dubsync,
    "dubsyncBatch": handle_dubsync_batch,
    "waveformPeaks": handle_waveform_peaks,
    "waveformBuild": handle_waveform_build,
    "dubsyncPreview": handle_dubsync_preview,
    "cancel": handle_cancel,
    "ping": lambda _r: emit({"type": "pong"}),
}


def dispatch(request: dict) -> None:
    command = request.get("command") or ("analyze" if request.get("mode") else None)
    handler = HANDLERS.get(command or "")
    if handler is None:
        emit_error(f"Unknown command: {command}")
        return
    try:
        handler(request)
    except Exception as exc:  # noqa: BLE001 - never let one command kill the bridge
        emit_error(f"{type(exc).__name__}: {exc}", fatal=False)
        traceback.print_exc(file=sys.stderr)


def _serve(requests: "queue.Queue[dict | None]") -> None:
    """Run commands one at a time, in the order they arrived."""
    while True:
        request = requests.get()
        if request is None:
            return
        dispatch(request)


def main() -> int:
    emit({"type": "ready", "ffmpeg": has_ffmpeg()})

    # Commands run on a worker thread so this loop is always reading. It
    # used to run them here, which meant a ``cancel`` written during a run
    # sat unread in the pipe until the run it was meant to stop had
    # finished on its own -- the Stop button did nothing, on every command,
    # for as long as the work took. Everything but cancel and shutdown is
    # queued, so commands still execute one at a time in order.
    requests: "queue.Queue[dict | None]" = queue.Queue()
    worker = threading.Thread(target=_serve, args=(requests,), daemon=True, name="commands")
    worker.start()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit_error(f"Invalid JSON request: {exc}")
            continue

        if request.get("command") == "shutdown":
            break

        # Handled here, on the reading thread, so it reaches the run in
        # flight rather than queueing behind it.
        if request.get("command") == "cancel":
            handle_cancel(request)
            continue

        requests.put(request)

    # Shutdown means "no more commands", not "stop": whatever was sent
    # before it still runs to completion, as it did when commands ran on
    # this thread and the shutdown line was only read afterwards. A caller
    # that pipes a request and a shutdown together -- the CI smoke test, a
    # shell script -- relies on that. To abort instead, send cancel first;
    # the host kills the process outright when its window closes.
    requests.put(None)
    worker.join()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
