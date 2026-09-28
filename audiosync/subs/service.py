"""The Subsync commands the bridge exposes, kept out of bridge.py.

The bridge only parses a request and hands this module an ``emit`` function
and a cancellation token; everything subtitle-specific -- running a queue of
jobs in parallel, probing, reporting which engines this machine can run,
installing packs -- lives here, where it can be tested without a pipe.
"""

from __future__ import annotations

import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from ..media import Cancelled, CancellationToken, MediaError
from .model import Cue, SubtitleDoc
from .tasks import TaskContext, TaskError, TaskOutput, TaskResult, run_job

Emit = Callable[[dict], None]

#: Heavy jobs (speech recognition, video encodes) already use every core or
#: the GPU; running more of them at once only makes each slower.
HEAVY_TASKS = frozenset({"generate", "tonemap"})
MAX_WORKERS = 6


# ------------------------------------------------------------------- batch


def run_batch(request: dict, emit: Emit, token: CancellationToken) -> None:
    """Run a queue of jobs, several at a time, reporting each by its index.

    One job failing never sinks the others; the terminal ``subsBatchDone``
    lists an outcome for every job in the order they were sent.
    """
    jobs: List[dict] = [j for j in (request.get("jobs") or []) if isinstance(j, dict)]
    if not jobs:
        emit({"type": "subsBatchDone", "outcomes": [], "error": "Nothing to run"})
        return
    secrets = {k: v for k, v in (request.get("secrets") or {}).items() if isinstance(v, str) and v}

    workers = int(request.get("maxWorkers") or 2)
    if any(j.get("task") in HEAVY_TASKS for j in jobs):
        workers = 1
    workers = max(1, min(workers, MAX_WORKERS, len(jobs)))

    outcomes: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="subsync") as pool:
        futures = {
            pool.submit(_run_one, index, job, secrets, emit, token): index
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcomes[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - one job must not sink the batch
                outcomes[index] = _outcome(index, jobs[index], error=f"{type(exc).__name__}: {exc}")
            emit({"type": "subsJobDone", **outcomes[index]})

    ordered = [outcomes.get(i) or _outcome(i, jobs[i], error="No outcome was produced") for i in range(len(jobs))]
    emit({"type": "subsBatchDone", "outcomes": ordered, "cancelled": token.cancelled})


def _outcome(index: int, job: dict, result: Optional[TaskResult] = None, **extra: Any) -> dict:
    body = result.to_dict() if result else {"outputs": [], "report": {}, "summary": "", "warnings": [], "preview": None}
    return {"job": index, "id": job.get("id") or str(index), "task": job.get("task"), **body, **extra}


def _run_one(index: int, job: dict, secrets: Dict[str, str], emit: Emit, token: CancellationToken) -> dict:
    emit({"type": "subsJobStart", "job": index, "id": job.get("id") or str(index), "task": job.get("task")})
    last = {"key": None}

    def progress(percent: int, stage: str) -> None:
        percent = max(0, min(100, int(percent)))
        key = (percent, stage)
        if key != last["key"]:
            last["key"] = key
            emit({"type": "subsJobProgress", "job": index, "percent": percent, "stage": stage})

    def log(message: str) -> None:
        emit({"type": "subsJobLog", "job": index, "message": message})

    if token.cancelled:
        return _outcome(index, job, cancelled=True)
    try:
        result = run_job(job, token, progress, log, secrets)
        result = _maybe_mux(result, job, token, log)
        return _outcome(index, job, result)
    except Cancelled:
        return _outcome(index, job, cancelled=True)
    except (TaskError, MediaError, OSError, ValueError) as exc:
        log(f"Failed: {exc}")
        return _outcome(index, job, error=str(exc))


def _maybe_mux(result: TaskResult, job: dict, token: CancellationToken, log: Callable[[str], None]) -> TaskResult:
    """``output.mux``: add the subtitles a job wrote to a copy of its video.

    Only for tasks whose result is a subtitle for a known video; a tone-map
    already wrote a video, and a mux job is itself the muxing.
    """
    out = job.get("output") or {}
    video = (job.get("input") or {}).get("video") or {}
    if not out.get("mux") or job.get("task") in ("tonemap", "mux") or not video.get("path"):
        return result
    subs = [o for o in result.outputs if o.kind == "subtitle"]
    if not subs:
        return result
    from . import tracks

    stem, _ = os.path.splitext(os.path.basename(video["path"]))
    directory = out.get("dir") or os.path.dirname(os.path.abspath(video["path"]))
    container = "mp4" if video["path"].lower().endswith((".mp4", ".m4v", ".mov")) else "mkv"
    target = os.path.join(directory, f"{stem}.subbed.{container}")
    n = 2
    while os.path.exists(target) and not out.get("overwrite"):
        target = os.path.join(directory, f"{stem}.subbed ({n}).{container}")
        n += 1
    entries = [
        {
            "path": o.path,
            "language": o.language,
            "title": out.get("muxTitle") or None,
            "default": bool(out.get("muxDefault")),
            "forced": bool(out.get("muxForced")),
        }
        for o in subs
    ]
    muxed = tracks.mux(video["path"], entries, target, container=container, keep_existing=True, token=token)
    log(f"Added to a copy of the video: {os.path.basename(muxed)}")
    result.outputs.append(TaskOutput(path=muxed, kind="video", format=container, label="Video with subtitles"))
    return result


# ------------------------------------------------------------------- probe


def probe(paths: List[str], token: Optional[CancellationToken] = None) -> List[dict]:
    from . import tracks

    files = []
    for path in paths:
        try:
            files.append(tracks.probe_file(path, token))
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad file must not hide the rest
            files.append({
                "path": path, "name": os.path.basename(path), "kind": "unknown", "duration": None,
                "video": None, "audioTracks": [], "subtitleTracks": [], "subtitle": None,
                "error": str(exc),
            })
    return files


# ------------------------------------------------------------ capabilities


def _platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    if os.name == "nt":
        return "windows"
    return "linux"


def _statuses(label: str, fn: Callable[[], List[dict]]) -> List[dict]:
    """An engine module that cannot even report its status is shown as one
    unavailable entry naming the failure, instead of breaking the panel."""
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001
        return [{"id": label, "label": label, "available": False, "reason": f"Failed to load: {exc}", "pack": None}]


def capabilities(secrets: Optional[Dict[str, str]] = None) -> dict:
    secrets = secrets or {}

    def group(module: str, *args: Any) -> List[dict]:
        def call() -> List[dict]:
            mod = __import__(f"audiosync.subs.{module}", fromlist=["engine_statuses"])
            return mod.engine_statuses(*args)

        return _statuses(module, call)

    try:
        from .tonemap import ffmpeg_capabilities

        ffmpeg = ffmpeg_capabilities()
    except Exception as exc:  # noqa: BLE001
        ffmpeg = {"path": None, "version": None, "filters": {}, "encoders": {}, "error": str(exc)}

    try:
        from .packs import status as pack_status

        packs = pack_status()
    except Exception:  # noqa: BLE001
        packs = []

    return {
        "platform": _platform(),
        "arch": platform.machine().lower(),
        "ffmpeg": ffmpeg,
        "engines": {
            "sync": group("sync"),
            "ocr": group("ocr"),
            "translate": group("translate", secrets),
            "generate": group("generate"),
            "tonemap": group("tonemap"),
            "vad": group("vad"),
        },
        "packs": packs,
    }


# ------------------------------------------------------------- load / save


def load(ref: dict, limit: int = 5000) -> dict:
    """Read a subtitle (file or embedded text track) for the review table."""
    import shutil
    import tempfile

    from .tasks import load_subtitle, preview_of

    workdir = tempfile.mkdtemp(prefix="subsync-load-")
    try:
        ctx = TaskContext(token=CancellationToken(), progress=lambda *_: None, log=lambda _m: None, workdir=workdir)
        doc = load_subtitle(ref, ctx)
        return {"preview": preview_of(doc, limit)}
    except (TaskError, MediaError, OSError, ValueError) as exc:
        return {"preview": None, "error": str(exc)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def save(path: str, fmt: Optional[str], cues: List[dict], language: Optional[str] = None) -> str:
    """Write cues edited in the app back to a file."""
    from . import formats

    doc = SubtitleDoc(cues=[Cue.from_dict(c) for c in cues], language=language)
    chosen = fmt
    if not chosen and os.path.exists(path):
        chosen = formats.detect_format(path)
    if not chosen or chosen == "unknown":
        chosen = os.path.splitext(path)[1].lstrip(".").lower() or "srt"
    formats.write(doc, path, chosen)
    return path


# ------------------------------------------------------------------- packs


def pack_install(pack: str, model: Optional[str], emit: Emit, token: CancellationToken) -> None:
    from . import packs

    def progress(percent: int, stage: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
        event = {"type": "packProgress", "pack": pack, "percent": max(0, min(100, int(percent))), "stage": stage}
        if done is not None:
            event["bytes"] = done
        if total is not None:
            event["totalBytes"] = total
        emit(event)

    try:
        packs.install(pack, progress, lambda m: emit({"type": "log", "message": m}), token, model=model)
        emit({"type": "packDone", "pack": pack, "ok": True, "status": _pack_status(pack)})
    except Cancelled:
        emit({"type": "packDone", "pack": pack, "ok": False, "error": "Cancelled", "status": _pack_status(pack)})
    except Exception as exc:  # noqa: BLE001 - reported to the user, never fatal to the bridge
        emit({"type": "packDone", "pack": pack, "ok": False, "error": str(exc), "status": _pack_status(pack)})


def pack_remove(pack: str, emit: Emit, model: Optional[str] = None) -> None:
    """Remove a pack, or with ``model`` only that model's downloaded weights."""
    from . import packs

    try:
        packs.remove(pack, model=model)
        emit({"type": "packDone", "pack": pack, "ok": True, "status": _pack_status(pack)})
    except Exception as exc:  # noqa: BLE001
        emit({"type": "packDone", "pack": pack, "ok": False, "error": str(exc), "status": _pack_status(pack)})


def _pack_status(pack: str) -> Optional[dict]:
    try:
        from .packs import pack_status

        return pack_status(pack)
    except Exception:  # noqa: BLE001 - an unknown id has no status to show
        return None
