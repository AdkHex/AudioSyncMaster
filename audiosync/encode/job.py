"""One queued job: what it converts, where it is, and its stage list
(FPSConv's ``Job``, ``engine.py:947-1123``).

The job does not run anything itself; ``jobs.py`` does, and cancels through
the job's :class:`~audiosync.media.CancellationToken`, which kills the
running tool's whole process tree.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..media import CancellationToken
from .formats import TASK_ENCODE, TASK_FPS
from .progress import fmt_time, hr_size, new_stage, parse_time

_log = logging.getLogger(__name__)

ProgressFn = Callable[[dict], None]


def stamp(ts: Optional[float] = None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts or time.time()))


def next_free(path: str) -> str:
    """``out.aac`` → ``out_1.aac``, or the first ``_N`` not taken yet."""
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


@dataclass
class Job:
    source: str
    conv_type: str
    out_dir: str
    stream_index: int = 0
    bitrate_override: int = 0          # 0 = source bitrate (fps.py) / DEE default (encode)
    overwrite: str = "overwrite"       # overwrite | skip | rename
    task: str = TASK_FPS               # fps | encode
    target: str = "ddp"                # encode: ddp | dd
    target_channels: int = 0           # encode: 0 = same as source, else 1 / 2 / 6 / 8
    atmos: bool = True                 # encode: keep Atmos when the source has it
    atmos71: str = "flat"              # encode: 7.1 Atmos legacy layout — flat (Lb Rb, patched DEE) | dee (Tfl Tfr)
    bed_conform: bool = True           # encode: truehdd --bed-conform for the flat-7.1 path
    drc: str = "film_light"            # encode: DEE DRC profile
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "queued"              # queued | running | done | failed | cancelled | skipped
    step: str = ""
    percent: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0
    codec: str = ""
    pretty: str = ""                   # "TrueHD Atmos", "DTS-HD MA" … (encode task)
    label: str = ""                    # "DDP 7.1 Atmos 1536k" (encode task)
    bitrate: int = 0
    channels: int = 0
    duration: float = 0.0
    out_path: str = ""
    out_size: int = 0
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    detail: str = ""                   # one-line "what is happening now"
    stages: list[dict] = field(default_factory=list)   # the pipeline, stage by stage — see stage_begin()
    log: list[str] = field(default_factory=list)
    token: CancellationToken = field(default_factory=CancellationToken, repr=False)
    _last_progress_line: int = field(default=-1, repr=False)

    def key(self) -> tuple:
        """What makes two queued jobs "the same" for duplicate detection."""
        base = (self.source, self.conv_type, self.stream_index, self.task)
        if self.task == TASK_ENCODE:
            base += (self.target, self.target_channels, bool(self.atmos), self.bitrate_override, self.drc,
                     self.atmos71, bool(self.bed_conform))
        return base

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled

    def cancel(self) -> None:
        self.token.cancel()

    # -- stages ------------------------------------------------------------- #
    # The Progress view: which stages the job goes through, which one runs now,
    # how long each took. Declared up front with plan_stages() so the waiting
    # ones are visible from the start; runners call stage_begin() as they go.

    def plan_stages(self, names: list[str]) -> None:
        self.stages = [new_stage(n) for n in names]

    def _stage(self, name: str) -> Optional[dict]:
        return next((s for s in self.stages if s["name"] == name), None)

    def stage_begin(self, name: str) -> None:
        """``name`` runs now; whatever ran before is done. Unplanned names slot in before the waiting ones."""
        now = time.time()
        for s in self.stages:
            if s["state"] == "running":
                if s["name"] == name:
                    return
                s.update(state="done", finished=now, pct=100.0)
        st = self._stage(name)
        if st is None:
            st = new_stage(name)
            idx = next((i for i, s in enumerate(self.stages) if s["state"] == "wait"), len(self.stages))
            self.stages.insert(idx, st)
        st.update(state="running", started=now, finished=0.0, pct=0.0)

    def stage_skip(self, name: str, note: str) -> None:
        st = self._stage(name)
        if st:
            st.update(state="skip", note=note)

    def stage_progress(self, pct: float) -> None:
        for s in self.stages:
            if s["state"] == "running":
                s["pct"] = max(0.0, min(100.0, pct))
                return

    def stage_close(self, ok: bool) -> None:
        """The job is over: the running stage finished, or it is where the job failed / was cancelled."""
        now = time.time()
        for s in self.stages:
            if s["state"] == "running":
                s.update(state="done" if ok else ("cancelled" if self.cancelled else "failed"),
                         finished=now, pct=100.0 if ok else s["pct"])

    def overall(self) -> float:
        """Whole-job percent from the stage weights (``percent`` restarts at 0 for every tool)."""
        total = sum(s["weight"] for s in self.stages if s["state"] != "skip")
        if not total:
            return self.percent
        done = sum(s["weight"] for s in self.stages if s["state"] == "done")
        running = sum(s["weight"] * s["pct"] / 100.0 for s in self.stages if s["state"] == "running")
        return min(100.0, (done + running) / total * 100.0)

    # -- log helpers -------------------------------------------------------- #

    def say(self, text: str, level: str = "info") -> None:
        """Append a timestamped line to the job log (and the engine log)."""
        self.log.append(f"{stamp()}  {text}")
        self._last_progress_line = -1
        getattr(_log, level if level in ("info", "warning", "error", "debug") else "info")(
            "[%s] %s", os.path.basename(self.source)[:40], text)

    def progress_line(self, text: str) -> None:
        """A live progress line: replaces the previous one instead of piling up."""
        line = f"{stamp()}  {text}"
        if 0 <= self._last_progress_line < len(self.log):
            self.log[self._last_progress_line] = line
        else:
            self.log.append(line)
            self._last_progress_line = len(self.log) - 1

    def to_dict(self) -> dict:
        elapsed = (time.time() - self.started_at) if self.state == "running" and self.started_at else self.elapsed
        return {
            "id": self.id, "source": self.source, "name": os.path.basename(self.source),
            "conv_type": self.conv_type, "out_dir": self.out_dir, "stream_index": self.stream_index,
            "bitrate_override": self.bitrate_override, "overwrite": self.overwrite,
            "task": self.task, "target": self.target, "target_channels": self.target_channels,
            "atmos": self.atmos, "atmos71": self.atmos71, "bed_conform": self.bed_conform,
            "drc": self.drc, "label": self.label or self.conv_type,
            "state": self.state, "step": self.step, "detail": self.detail, "percent": round(self.percent, 1),
            "elapsed": fmt_time(elapsed), "eta": fmt_time(self.eta) if self.eta > 0 else "",
            "codec": self.codec, "pretty": self.pretty, "bitrate": self.bitrate, "channels": self.channels,
            "duration": fmt_time(self.duration), "out_path": self.out_path,
            "out_name": os.path.basename(self.out_path) if self.out_path else "",
            "out_size": hr_size(self.out_size) if self.out_size else "",
            "error": self.error, "finished_at": self.finished_at, "log": self.log[-400:],
            "stages": self.stages, "overall": round(self.overall(), 1), "now": time.time(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        job = cls(
            source=d.get("source", ""), conv_type=d.get("conv_type", ""), out_dir=d.get("out_dir", ""),
            stream_index=int(d.get("stream_index", 0)), bitrate_override=int(d.get("bitrate_override", 0)),
            overwrite=d.get("overwrite", "overwrite"), id=d.get("id") or uuid.uuid4().hex[:12],
            task=d.get("task") or TASK_FPS, target=d.get("target") or "ddp",
            target_channels=int(d.get("target_channels") or 0), atmos=bool(d.get("atmos", True)),
            atmos71=d.get("atmos71") or "flat", bed_conform=bool(d.get("bed_conform", True)),
            drc=d.get("drc") or "film_light",
        )
        job.state = d.get("state", "done")
        job.step = d.get("step", "")
        job.percent = 100.0 if job.state == "done" else 0.0
        job.codec, job.bitrate, job.channels = d.get("codec", ""), int(d.get("bitrate", 0)), int(d.get("channels", 0))
        job.pretty, job.label = d.get("pretty", ""), d.get("label", "")
        job.elapsed, job.duration = parse_time(d.get("elapsed")), parse_time(d.get("duration"))
        job.out_path = d.get("out_path", "")
        job.out_size = os.path.getsize(job.out_path) if job.out_path and os.path.exists(job.out_path) else 0
        job.error = d.get("error", "")
        job.finished_at = float(d.get("finished_at", 0))
        job.stages = [dict(x) for x in d.get("stages", []) if isinstance(x, dict) and x.get("name")]
        job.log = list(d.get("log", []))
        return job
