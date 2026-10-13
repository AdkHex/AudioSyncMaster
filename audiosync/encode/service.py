"""The audio queue both pages share, and the CLI drives directly
(a port of FPSConv's ``queue.py``).

Jobs arrive while others run, carry ids the caller chose, and are skipped
when an identical one is already queued or running. Up to ``workers`` jobs
run at once; the count can change mid-run and takes effect as slots free
up. Each finished job is appended to ``history.json`` (when ``history``);
when the queue drains, ``on_idle`` gets how many jobs of each task ended
in each state since it was last idle.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from . import jobs, settings, tools
from .formats import ATMOS71_LAYOUTS, DRC_PROFILES, ENCODE_CONV, TARGET_CHANNELS, TARGETS, TASK_ENCODE, TASK_FPS
from .fps import FPS_CONVERSIONS
from .job import Job

_log = logging.getLogger(__name__)

TERMINAL = ("done", "failed", "cancelled", "skipped")
MAX_WORKERS = 8
HISTORY_LOG_LINES = 20

Emit = Callable[[dict], None]


def check_spec(spec: dict) -> str:
    """Why a job request cannot be queued, or "" (FPSConv's /api/jobs checks)."""
    if not spec.get("path"):
        return "no source file"
    if not spec.get("out_dir"):
        return "choose an output folder first"
    task = spec.get("task") or TASK_FPS
    if task not in (TASK_FPS, TASK_ENCODE):
        return f"unknown task {task!r}"
    if task == TASK_ENCODE:
        encode = spec.get("encode") or {}
        if (encode.get("target") or "ddp") not in TARGETS:
            return f"unknown target {encode.get('target')!r}"
        try:
            ch = int(encode.get("channels") or 0)
        except (TypeError, ValueError):
            return "channels must be 0, 1, 2, 6 or 8"
        if ch not in TARGET_CHANNELS:
            return "channels must be 0, 1, 2, 6 or 8"
        if (encode.get("drc") or "film_light") not in DRC_PROFILES:
            return f"unknown DRC profile {encode.get('drc')!r}"
        if (encode.get("atmos71") or "flat") not in ATMOS71_LAYOUTS:
            return f"unknown 7.1 Atmos layout {encode.get('atmos71')!r}"
    elif spec.get("conv_type") not in FPS_CONVERSIONS:
        return f"unknown conversion {spec.get('conv_type')!r}"
    if not os.path.isfile(spec["path"]):
        return f"file not found: {spec['path']}"
    return ""


def job_from_spec(spec: dict) -> Job:
    """A queued job from one request: ``{id, task, path, stream_index, out_dir, overwrite,
    conv_type, bitrate}`` for an FPS change, with ``encode: {target, channels, bitrate,
    atmos, atmos71, bed_conform, drc}`` for an encode."""
    common: dict[str, Any] = {
        "source": os.path.abspath(spec["path"]),
        "out_dir": os.path.abspath(spec["out_dir"]),
        "stream_index": int(spec.get("stream_index") or 0),
        "overwrite": spec.get("overwrite") or "overwrite",
    }
    if spec.get("id"):
        common["id"] = str(spec["id"])
    if (spec.get("task") or TASK_FPS) == TASK_ENCODE:
        encode = spec.get("encode") or {}
        return Job(conv_type=ENCODE_CONV, task=TASK_ENCODE,
                   bitrate_override=int(encode.get("bitrate") or 0),
                   target=encode.get("target") or "ddp",
                   target_channels=int(encode.get("channels") or 0),
                   atmos=bool(encode.get("atmos", True)),
                   atmos71=encode.get("atmos71") or "flat",
                   bed_conform=bool(encode.get("bed_conform", True)),
                   drc=encode.get("drc") or "film_light", **common)
    return Job(conv_type=spec["conv_type"], bitrate_override=int(spec.get("bitrate") or 0), **common)


def job_event(job: Job) -> dict:
    """A job's state for the pages, without its log (that travels in audioJobLog)."""
    snap = job.to_dict()
    snap.pop("log", None)
    return snap


class AudioQueue:
    def __init__(self, emit: Optional[Emit] = None, workers: int = 2, history: bool = True,
                 tool_overrides: Optional[dict] = None, work_root: Optional[str] = None) -> None:
        self._emit = emit or (lambda _event: None)
        self._workers = self._clamp(workers)
        self._history = history
        self._tool_overrides = dict(tool_overrides or {})
        self._work_root = work_root
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._pending: collections.deque[Job] = collections.deque()
        self._running = 0
        self._summary: dict[str, dict[str, int]] = {}
        self._log_marks: dict[str, tuple[int, str]] = {}
        self._idle = threading.Condition(self._lock)

    @staticmethod
    def _clamp(n: Any) -> int:
        try:
            return max(1, min(MAX_WORKERS, int(n)))
        except (TypeError, ValueError):
            return 1

    # -- control ---------------------------------------------------------- #

    def set_tools(self, overrides: Optional[dict]) -> None:
        self._tool_overrides = dict(overrides or {})

    def set_workers(self, n: Any) -> int:
        with self._lock:
            self._workers = self._clamp(n)
            workers = self._workers
        self._pump()
        return workers

    @property
    def workers(self) -> int:
        return self._workers

    def add(self, specs: list[dict]) -> dict:
        """Queue every valid, new job. Returns ``{"added": [ids], "skipped": [{id, reason}]}``."""
        added: list[Job] = []
        skipped: list[dict] = []
        with self._lock:
            live = {self._jobs[i].key() for i in self._order if self._jobs[i].state in ("queued", "running")}
            for spec in specs:
                why = check_spec(spec)
                if why:
                    skipped.append({"id": spec.get("id"), "reason": why})
                    continue
                job = job_from_spec(spec)
                if job.key() in live:
                    skipped.append({"id": job.id, "reason": "already queued"})
                    continue
                if job.id in self._jobs:
                    skipped.append({"id": job.id, "reason": "id already used"})
                    continue
                live.add(job.key())
                self._jobs[job.id] = job
                self._order.append(job.id)
                self._pending.append(job)
                added.append(job)
        if added:
            _log.info("queued %d job(s)", len(added))
            tools.ensure_deew_config()      # deew's temp follows the Temp folder; rewrite before running
        self._pump()
        return {"added": [j.id for j in added], "skipped": skipped}

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state == "queued":
                job.cancel()
                job.state = "cancelled"
                job.error = "cancelled before start"
                job.finished_at = time.time()
                try:
                    self._pending.remove(job)
                except ValueError:
                    pass
                queued = True
            elif job.state == "running":
                queued = False
            else:
                return False
        if queued:
            _log.info("cancelled before start: %s", os.path.basename(job.source))
            self._finished(job)
        else:
            _log.info("cancelling: %s", os.path.basename(job.source))
            job.cancel()
        return True

    def cancel_all(self) -> int:
        with self._lock:
            ids = list(self._order)
        return sum(1 for i in ids if self.cancel(i))

    # -- state ------------------------------------------------------------ #

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def busy(self) -> bool:
        with self._lock:
            return self._busy_locked()

    def _busy_locked(self) -> bool:
        return self._running > 0 or bool(self._pending)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        with self._idle:
            return self._idle.wait_for(lambda: not self._busy_locked(), timeout)

    def snapshot(self) -> dict:
        with self._lock:
            snaps = [self._jobs[i].to_dict() for i in self._order]
            workers = self._workers
        counts: dict[str, int] = {}
        for j in snaps:
            counts[j["state"]] = counts.get(j["state"], 0) + 1
        return {"jobs": snaps, "counts": counts, "workers": workers}

    # -- running ---------------------------------------------------------- #

    def _pump(self) -> None:
        start: list[Job] = []
        with self._lock:
            while self._running < self._workers and self._pending:
                job = self._pending.popleft()
                self._running += 1
                start.append(job)
        for job in start:
            threading.Thread(target=self._run, args=(job,), daemon=True, name=f"audio-{job.id[:8]}").start()

    def _run(self, job: Job) -> None:
        try:
            self._emit({"type": "audioJobStart", "jobId": job.id, "job": job_event(job)})
            resolved = tools.Tools.resolve(self._tool_overrides)
            jobs.convert(job, self._notify, resolved, self._work_root)
        except Exception as exc:  # noqa: BLE001 - one job's crash must not stop the queue
            _log.exception("job %s crashed", job.id)
            job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.finished_at = time.time()
        finally:
            with self._lock:
                self._running -= 1
            self._finished(job)
            self._pump()

    def _notify(self, job: Job) -> None:
        snap = job_event(job)
        stage = next((s["name"] for s in job.stages if s["state"] == "running"), "")
        self._emit({"type": "audioJobProgress", "jobId": job.id, "percent": snap["overall"], "stage": stage,
                    "stages": job.stages, "eta": snap["eta"], "job": snap})
        self._emit_log(job)

    def _emit_log(self, job: Job) -> None:
        mark = (len(job.log), job.log[-1] if job.log else "")
        if self._log_marks.get(job.id) != mark:
            self._log_marks[job.id] = mark
            self._emit({"type": "audioJobLog", "jobId": job.id, "lines": job.log[-400:]})

    def _finished(self, job: Job) -> None:
        """A job reached a terminal state: report it, remember it, and say so when the queue drains."""
        self._emit_log(job)
        self._log_marks.pop(job.id, None)
        self._emit({"type": "audioJobDone", "jobId": job.id, "state": job.state, "output": job.out_path,
                    "size": job.out_size, "error": job.error, "job": job_event(job)})
        if self._history:
            self._remember(job)
        with self._lock:
            tally = self._summary.setdefault(job.task, {})
            tally[job.state] = tally.get(job.state, 0) + 1
            idle = not self._busy_locked()
            summary = self._summary if idle else None
            if idle:
                self._summary = {}
            self._idle.notify_all()
        if summary is not None:
            self._emit({"type": "audioIdle", "summary": summary})

    @staticmethod
    def _remember(job: Job) -> None:
        entry = job.to_dict()
        entry["log"] = entry["log"][-HISTORY_LOG_LINES:]
        try:
            history = [e for e in settings.load_history() if e.get("id") != job.id]
            settings.save_history(history + [entry])
        except OSError as exc:
            _log.warning("could not save the encode history: %s", exc)


# ───────────────────────── history (info bridge) ─────────────────────────


def history_remove(job_id: str) -> list[dict]:
    history = [e for e in settings.load_history() if e.get("id") != job_id]
    settings.save_history(history)
    return history


def history_clear() -> list[dict]:
    settings.save_history([])
    return []
