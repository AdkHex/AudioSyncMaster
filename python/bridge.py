import json
import os
import sys
import glob
import threading
import time
import hashlib
import tempfile
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
    import numpy as np
    import librosa
except Exception as exc:
    sys.stderr.write(f"Failed to import Python scripts: {exc}\n")
    sys.exit(1)

_emit_lock = threading.Lock()
_ffmpeg_ready = False


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


def list_audio_files(audio_folder):
    exts = ("*.wav", "*.mp3", "*.aac", "*.flac", "*.ogg", "*.m4a", "*.eac3", "*.ac3")
    return [f for ext in exts for f in glob.glob(os.path.join(audio_folder, ext))]


def setup_ffmpeg_env():
    global _ffmpeg_ready
    if _ffmpeg_ready:
        return

    candidates = [
        os.path.join(BASE_DIR, "resources", "ffmpeg"),
        os.path.join(ROOT_DIR, "resources", "ffmpeg"),
        os.path.join(ROOT_DIR, "ffmpeg"),
    ]
    for folder in candidates:
        ffmpeg_path = os.path.join(folder, "ffmpeg.exe")
        ffprobe_path = os.path.join(folder, "ffprobe.exe")
        if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            emit_log(f"Using bundled ffmpeg from {folder}")
            _ffmpeg_ready = True
            return

    emit_log("Using ffmpeg from PATH (bundled ffmpeg not found).")
    _ffmpeg_ready = True


def get_cache_dir():
    base = os.environ.get("AUDIOSYNC_CACHE_DIR")
    if base:
        return base
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return os.path.join(local, "AudioSync", "cache")
    return os.path.join(tempfile.gettempdir(), "audiosync_cache")


def cache_key(path, sr, duration, offset, tag):
    try:
        stat = os.stat(path)
        parts = f"{path}|{stat.st_mtime}|{stat.st_size}|{sr}|{duration}|{offset}|{tag}"
    except OSError:
        parts = f"{path}|{sr}|{duration}|{offset}|{tag}"
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def load_cached_array(key):
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.npz")
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        return data["arr_0"]
    except Exception:
        return None


def save_cached_array(key, arr):
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.npz")
    try:
        np.savez_compressed(cache_path, arr)
    except Exception:
        pass


def select_high_energy_window(y, sr, window_sec):
    if y is None or len(y) == 0:
        return y
    window_len = int(max(1, window_sec * sr))
    if len(y) <= window_len:
        return y
    frame = int(0.05 * sr)
    hop = int(0.025 * sr)
    if frame <= 0 or hop <= 0:
        return y[:window_len]
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    if len(rms) == 0:
        return y[:window_len]
    window_frames = max(1, int(window_sec / (hop / sr)))
    if len(rms) < window_frames:
        idx = int(np.argmax(rms))
    else:
        energy = np.convolve(rms, np.ones(window_frames), mode="valid")
        idx = int(np.argmax(energy))
    start = int(idx * hop)
    end = min(len(y), start + window_len)
    return y[start:end]


def load_audio_segment(path, sr, duration, offset, tag, window_sec):
    key = cache_key(path, sr, duration, offset, tag)
    cached = load_cached_array(key)
    if cached is not None:
        return cached
    y = series_logic.load_audio(path, sr=sr, duration=duration, offset=offset, verbose=False)
    if y is None:
        return None
    y = select_high_energy_window(y, sr, window_sec)
    save_cached_array(key, y)
    return y


def compute_fingerprint(y, sr):
    if y is None or len(y) == 0:
        return None
    y = y.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1), contrast.mean(axis=1), contrast.std(axis=1)]
    )
    norm = np.linalg.norm(features)
    if norm == 0:
        return None
    return features / norm


def fingerprint_similarity(fp_a, fp_b):
    if fp_a is None or fp_b is None:
        return 0.0
    return float(np.dot(fp_a, fp_b))


def enhanced_process_pair(primary_path, secondary_path, segment_sec, verbose=False):
    setup_ffmpeg_env()
    sr = 8000
    window_sec = min(30.0, float(segment_sec))
    start_primary = load_audio_segment(primary_path, sr, segment_sec, 0, "start", window_sec)
    start_secondary = load_audio_segment(secondary_path, sr, segment_sec, 0, "start", window_sec)

    if start_primary is None or start_secondary is None:
        return primary_path, secondary_path, None, None, "Failed to load start segment."

    min_len = min(len(start_primary), len(start_secondary))
    if min_len <= sr:
        return primary_path, secondary_path, None, None, "Insufficient audio at start for analysis."

    start_delay = series_logic.estimate_sync_offset_crosscorr(
        start_primary[:min_len], start_secondary[:min_len], sr=sr
    )

    primary_duration = series_logic.get_audio_duration(primary_path, verbose=False)
    secondary_duration = series_logic.get_audio_duration(secondary_path, verbose=False)
    if primary_duration is None or secondary_duration is None:
        return primary_path, secondary_path, start_delay, None, "Could not get duration for analysis."

    mid_offset_primary = max(0, primary_duration / 2 - segment_sec / 2)
    mid_offset_secondary = max(0, secondary_duration / 2 - segment_sec / 2)
    mid_primary = load_audio_segment(primary_path, sr, segment_sec, mid_offset_primary, "mid", window_sec)
    mid_secondary = load_audio_segment(secondary_path, sr, segment_sec, mid_offset_secondary, "mid", window_sec)
    if mid_primary is not None and mid_secondary is not None:
        min_len_mid = min(len(mid_primary), len(mid_secondary))
        if min_len_mid > sr:
            mid_delay = series_logic.estimate_sync_offset_crosscorr(
                mid_primary[:min_len_mid], mid_secondary[:min_len_mid], sr=sr
            )
            emit_log(f"Mid delay for {os.path.basename(primary_path)}: {mid_delay:+.1f}ms")

    end_offset_primary = max(0, primary_duration - segment_sec)
    end_offset_secondary = max(0, secondary_duration - segment_sec)
    end_primary = load_audio_segment(primary_path, sr, segment_sec, end_offset_primary, "end", window_sec)
    end_secondary = load_audio_segment(secondary_path, sr, segment_sec, end_offset_secondary, "end", window_sec)
    if end_primary is None or end_secondary is None:
        return primary_path, secondary_path, start_delay, None, "Failed to load end segment."

    min_len_end = min(len(end_primary), len(end_secondary))
    if min_len_end <= sr:
        return primary_path, secondary_path, start_delay, None, "Insufficient audio at end for analysis."

    end_delay_raw = series_logic.estimate_sync_offset_crosscorr(
        end_primary[:min_len_end], end_secondary[:min_len_end], sr=sr
    )
    duration_diff_ms = (primary_duration - secondary_duration) * 1000
    end_delay = end_delay_raw + duration_diff_ms

    return primary_path, secondary_path, start_delay, end_delay, None


def build_fingerprint_cache(files, segment_sec):
    fp_cache = {}
    sr = 8000
    window_sec = min(30.0, float(segment_sec))
    for path in files:
        y = load_audio_segment(path, sr, segment_sec, 0, "fingerprint", window_sec)
        fp_cache[path] = compute_fingerprint(y, sr) if y is not None else None
    return fp_cache


def match_by_fingerprint(video_files, audio_files, segment_sec, threshold=0.7):
    sr = 8000
    window_sec = min(30.0, float(segment_sec))
    audio_fp = build_fingerprint_cache(audio_files, segment_sec)
    pairs = []
    for video_path in video_files:
        y = load_audio_segment(video_path, sr, segment_sec, 0, "fingerprint", window_sec)
        video_fp = compute_fingerprint(y, sr) if y is not None else None
        best_audio = None
        best_score = -1.0
        for audio_path, fp in audio_fp.items():
            score = fingerprint_similarity(video_fp, fp)
            if score > best_score:
                best_score = score
                best_audio = audio_path
        if best_audio and best_score >= threshold:
            emit_log(
                f"Fingerprint match: {os.path.basename(video_path)} -> {os.path.basename(best_audio)} ({best_score:.2f})"
            )
            pairs.append((video_path, best_audio))
        else:
            emit_log(f"Fingerprint match below threshold for {os.path.basename(video_path)}.")
    return pairs

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
            try:
                result = enhanced_process_pair(video_path, audio_file, segment, False)
            except Exception as exc:
                emit_log(f"Enhanced analysis failed for {os.path.basename(video_path)}: {exc}")
                result = movie_logic.process_pair(video_path, audio_file, segment, False)
            elapsed_ms = int((time.time() - start_time) * 1000)
            emit({"type": "file_progress", "file": os.path.basename(video_path), "percent": 100})
            emit({"type": "file_end", "file": os.path.basename(video_path), "elapsed_ms": elapsed_ms})
            return result

        futures = {executor.submit(worker, video_path): video_path for video_path in video_files}
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            normalized = normalize_result(result)
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
        emit_log("No matching file pairs found by name. Falling back to fingerprint matching.")
        video_files = list_movie_videos(video_folder, [])
        audio_files = list_audio_files(audio_folder)
        matched_pairs = match_by_fingerprint(video_files, audio_files, segment)
        total = len(matched_pairs)
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
            try:
                result = enhanced_process_pair(primary, secondary, segment, False)
            except Exception as exc:
                emit_log(f"Enhanced analysis failed for {os.path.basename(primary)}: {exc}")
                result = series_logic.process_pair(primary, secondary, segment, False)
            elapsed_ms = int((time.time() - start_time) * 1000)
            emit({"type": "file_progress", "file": os.path.basename(primary), "percent": 100})
            emit({"type": "file_end", "file": os.path.basename(primary), "elapsed_ms": elapsed_ms})
            return result

        futures = {executor.submit(worker, p, s): (p, s) for p, s in matched_pairs}
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            normalized = normalize_result(result)
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
