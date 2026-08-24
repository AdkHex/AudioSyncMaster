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
import sys
import tempfile
import threading
import traceback

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
        pair_every_combination,
        pair_movie_mode,
        validate_pattern,
    )
    from audiosync.media import CancellationToken, MediaError, has_ffmpeg, probe
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


def handle_cancel(_request: dict) -> None:
    with _token_lock:
        token = _active_token
    if token:
        token.cancel()
        emit_log("Cancellation requested.")
    emit({"type": "cancelAck"})


HANDLERS = {
    "analyze": handle_analyze,
    "previewPairs": handle_preview_pairs,
    "probe": handle_probe,
    "listTracks": handle_list_tracks,
    "preview": handle_preview,
    "apply": handle_apply,
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


def main() -> int:
    emit({"type": "ready", "ffmpeg": has_ffmpeg()})

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

        # Cancel must be handled inline so it is not queued behind the run it
        # is trying to stop.
        if request.get("command") == "cancel":
            handle_cancel(request)
            continue

        dispatch(request)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
