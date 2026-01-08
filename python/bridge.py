import json
import os
import sys
import glob
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
else:
    BASE_DIR = ROOT_DIR

PYTHON_FILES_DIR = os.path.join(BASE_DIR, "Python Files")
sys.path.insert(0, PYTHON_FILES_DIR)

try:
    import main as series_logic
    import main2 as movie_logic
except Exception as exc:
    sys.stderr.write(f"Failed to import Python scripts: {exc}\n")
    sys.exit(1)

_emit_lock = threading.Lock()


def emit(payload):
    with _emit_lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def emit_log(message):
    emit({"type": "log", "message": message})


def normalize_result(result_tuple):
    primary_path, secondary_path, start_delay, end_delay, error = result_tuple
    return {
        "videoFile": os.path.basename(primary_path),
        "audioFile": os.path.basename(secondary_path),
        "startDelay": start_delay,
        "endDelay": end_delay,
        "error": error,
    }


def list_movie_videos(video_folder, explicit_files):
    if explicit_files:
        return explicit_files
    exts = ("*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov")
    return [f for ext in exts for f in glob.glob(os.path.join(video_folder, ext))]


def run_movie(request):
    video_folder = request.get("video_folder")
    audio_file = request.get("audio_file")
    explicit_files = request.get("video_files") or []
    segment = float(request.get("segment_duration", 300.0))

    if not audio_file:
        emit({"type": "done", "results": []})
        return

    if not video_folder and not explicit_files:
        emit({"type": "done", "results": []})
        return

    video_files = list_movie_videos(video_folder, explicit_files)
    total = len(video_files)
    results = []

    if total == 0:
        emit({"type": "done", "results": []})
        return

    emit_log(f"Movie mode: {total} video files queued.")
    emit_log(f"Audio file: {os.path.basename(audio_file)}")

    processed = 0
    with ThreadPoolExecutor() as executor:
        def worker(video_path):
            emit({"type": "file_start", "file": os.path.basename(video_path)})
            emit({"type": "file_progress", "file": os.path.basename(video_path), "percent": 0})
            start_time = time.time()
            result = movie_logic.process_pair(video_path, audio_file, segment, False)
            elapsed_ms = int((time.time() - start_time) * 1000)
            emit({"type": "file_progress", "file": os.path.basename(video_path), "percent": 100})
            emit({"type": "file_end", "file": os.path.basename(video_path), "elapsed_ms": elapsed_ms})
            return result, elapsed_ms

        futures = {executor.submit(worker, video_path): video_path for video_path in video_files}
        for future in as_completed(futures):
            processed += 1
            result, elapsed_ms = future.result()
            normalized = normalize_result(result)
            normalized["elapsed_ms"] = elapsed_ms
            results.append(normalized)
            emit(
                {
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "current": os.path.basename(futures[future]),
                }
            )
            emit({"type": "result", **normalized})

    emit({"type": "done", "results": results})


def run_series(request):
    video_folder = request.get("video_folder")
    audio_folder = request.get("audio_folder")
    match_pattern = request.get("match_pattern")
    segment = float(request.get("segment_duration", 300.0))

    if not video_folder or not audio_folder:
        emit({"type": "done", "results": []})
        return

    matched_pairs = series_logic.find_matching_files(video_folder, audio_folder, match_pattern, False)
    total = len(matched_pairs)
    results = []

    if total == 0:
        emit_log("No matching file pairs found.")
        emit({"type": "done", "results": []})
        return

    emit_log(f"Series mode: matched {total} file pairs.")
    emit_log(f"Video folder: {video_folder}")
    emit_log(f"Audio folder: {audio_folder}")
    if match_pattern:
        emit_log(f"Match pattern: {match_pattern}")

    processed = 0
    with ThreadPoolExecutor() as executor:
        def worker(primary, secondary):
            emit({"type": "file_start", "file": os.path.basename(primary)})
            emit({"type": "file_progress", "file": os.path.basename(primary), "percent": 0})
            start_time = time.time()
            result = series_logic.process_pair(primary, secondary, segment, False)
            elapsed_ms = int((time.time() - start_time) * 1000)
            emit({"type": "file_progress", "file": os.path.basename(primary), "percent": 100})
            emit({"type": "file_end", "file": os.path.basename(primary), "elapsed_ms": elapsed_ms})
            return result, elapsed_ms

        futures = {executor.submit(worker, p, s): (p, s) for p, s in matched_pairs}
        for future in as_completed(futures):
            processed += 1
            result, elapsed_ms = future.result()
            normalized = normalize_result(result)
            normalized["elapsed_ms"] = elapsed_ms
            results.append(normalized)
            emit(
                {
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "current": os.path.basename(futures[future][0]),
                }
            )
            emit({"type": "result", **normalized})

    emit({"type": "done", "results": results})


def main():
    payload = sys.stdin.read()
    if not payload.strip():
        sys.stderr.write("No input provided.\n")
        sys.exit(1)

    try:
        request = json.loads(payload)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Invalid JSON input: {exc}\n")
        sys.exit(1)

    mode = request.get("mode")
    if mode == "movie":
        run_movie(request)
    elif mode == "series":
        run_series(request)
    else:
        sys.stderr.write(f"Unknown mode: {mode}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
