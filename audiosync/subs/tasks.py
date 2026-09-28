"""The job runner every Subsync feature plugs into.

The app sends a queue of jobs; each names a *task* (sync, ocr, translate,
fps, generate, style, hdrSubs, tonemap, convert, extract, mux) with its own
input, engine and settings. This module resolves the task to its engine
module, gives it a context (progress, log, cancellation, a scratch
directory, the API keys the host passed), and turns what comes back into the
one outcome shape the app renders.

Engines are imported lazily: a user who never generates subtitles never pays
for importing the speech-recognition front end, and one engine failing to
import cannot take the others down with it.
"""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..media import CancellationToken, MediaError
from .languages import normalize as normalize_language
from .model import SubtitleDoc

#: task name -> "module:function". Each function has the signature
#: ``run(job: dict, ctx: TaskContext) -> TaskResult``.
TASKS: Dict[str, str] = {
    "sync": "audiosync.subs.sync:run_task",
    "fps": "audiosync.subs.retime:run_task",
    "ocr": "audiosync.subs.ocr:run_task",
    "translate": "audiosync.subs.translate:run_task",
    "generate": "audiosync.subs.generate:run_task",
    "style": "audiosync.subs.style:run_task",
    "hdrSubs": "audiosync.subs.hdr_subs:run_task",
    "tonemap": "audiosync.subs.tonemap:run_task",
    "convert": "audiosync.subs.formats:run_convert_task",
    "extract": "audiosync.subs.tracks:run_extract_task",
    "mux": "audiosync.subs.tracks:run_mux_task",
}


@dataclass
class TaskContext:
    token: CancellationToken
    #: percent 0-100 of the current stage, and a short stage name.
    progress: Callable[[int, str], None]
    log: Callable[[str], None]
    #: Scratch space for this job only; removed when the job ends.
    workdir: str
    #: API keys by engine id ("anthropic", "openai", "deepl", "google").
    #: Never log these, never write them into a report.
    secrets: Dict[str, str] = field(default_factory=dict)

    def check(self) -> None:
        self.token.raise_if_cancelled()


@dataclass
class TaskOutput:
    path: str
    #: subtitle | video | folder | report
    kind: str = "subtitle"
    format: Optional[str] = None
    language: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "format": self.format,
            "language": self.language,
            "label": self.label,
        }


@dataclass
class TaskResult:
    outputs: List[TaskOutput] = field(default_factory=list)
    #: Task-specific numbers the app shows (offset, detected language...).
    report: Dict[str, Any] = field(default_factory=dict)
    #: One sentence a person can read: what was done and how sure it is.
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    #: The resulting cues for the app's review table (``doc.to_dict``).
    preview: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outputs": [o.to_dict() for o in self.outputs],
            "report": self.report,
            "summary": self.summary,
            "warnings": self.warnings,
            "preview": self.preview,
        }


class TaskError(MediaError):
    """A job that cannot run as asked; the message is shown to the user."""


def resolve(task: str) -> Callable[[dict, TaskContext], TaskResult]:
    target = TASKS.get(task)
    if not target:
        raise TaskError(f"Unknown subtitle task: {task}")
    module_name, _, func_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TaskError(f"The {task} engine could not be loaded: {exc}") from exc
    return getattr(module, func_name)


def run_job(
    job: dict,
    token: CancellationToken,
    progress: Callable[[int, str], None],
    log: Callable[[str], None],
    secrets: Optional[Dict[str, str]] = None,
) -> TaskResult:
    """Run one job in its own scratch directory."""
    runner = resolve(str(job.get("task") or ""))
    workdir = tempfile.mkdtemp(prefix="subsync-")
    try:
        ctx = TaskContext(token=token, progress=progress, log=log, workdir=workdir, secrets=dict(secrets or {}))
        return runner(job, ctx)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ------------------------------------------------------------ shared helpers
#
# Every task reads a subtitle from a *reference* -- a file, or a track inside
# a video -- and writes its result next to the source. These helpers keep
# that identical across engines.


def load_subtitle(ref: dict, ctx: TaskContext, fps: Optional[float] = None) -> SubtitleDoc:
    """Read a text subtitle from ``{"path", "track"?, "encoding"?}``.

    ``track`` is the N of ``0:s:N`` inside a video; without it ``path`` is a
    subtitle file. Image subtitles (PGS, VobSub) raise: they need OCR first.
    """
    from . import formats, tracks

    path = ref.get("path")
    if not path or not os.path.isfile(path):
        raise TaskError(f"Subtitle not found: {path}")
    if ref.get("track") is not None:
        extracted = tracks.extract(path, int(ref["track"]), ctx.workdir, ctx.token)
        if extracted.lower().endswith((".sup", ".sub", ".idx")):
            raise TaskError("That track is an image subtitle (PGS/VobSub); run OCR on it first.")
        doc = formats.read(extracted, fps=fps)
        info = next((t for t in tracks.list_tracks(path, ctx.token) if t.index == int(ref["track"])), None)
        if info and info.language and not doc.language:
            doc.language = info.language
        return doc
    if path.lower().endswith((".sup", ".sub", ".idx")) and formats.detect_format(path) in ("pgs", "vobsub"):
        raise TaskError("That file is an image subtitle (PGS/VobSub); run OCR on it first.")
    return formats.read(path, encoding=ref.get("encoding"), fps=fps)


def output_path(
    ref: dict,
    job: dict,
    suffix: str,
    ext: str,
    language: Optional[str] = None,
) -> str:
    """Where a task writes: beside the source (or ``output.dir``), named
    ``<stem>[.<track>]<suffix>[.<lang>].<ext>``, never over an existing file
    unless ``output.overwrite``."""
    out = job.get("output") or {}
    source = ref.get("path") or "subtitle"
    directory = out.get("dir") or os.path.dirname(os.path.abspath(source))
    stem = os.path.splitext(os.path.basename(source))[0]
    if language:
        # Movie.en.srt stays Movie<suffix>.en.srt rather than growing a
        # second tag: the language goes last, once.
        head, dot, tag = stem.rpartition(".")
        if dot and head and len(tag) <= 7 and normalize_language(tag):
            stem = head
    if ref.get("track") is not None:
        stem = f"{stem}.track{int(ref['track']) + 1}"
    suffix = out.get("suffix", suffix) or ""
    parts = [stem + suffix]
    if language:
        parts.append(language)
    candidate = os.path.join(directory, ".".join(parts) + "." + ext.lstrip("."))
    if out.get("overwrite") or not os.path.exists(candidate):
        return candidate
    root, extension = os.path.splitext(candidate)
    n = 2
    while os.path.exists(f"{root} ({n}){extension}"):
        n += 1
    return f"{root} ({n}){extension}"


def write_subtitle(
    doc: SubtitleDoc,
    ref: dict,
    job: dict,
    suffix: str,
    ctx: TaskContext,
    language: Optional[str] = None,
) -> TaskOutput:
    """Write ``doc`` in ``output.format`` ("same" keeps the source's format
    where it is a text format, else SRT) and return the output record."""
    from . import formats

    wanted = (job.get("output") or {}).get("format") or "same"
    fmt = formats.resolve_output_format(wanted, doc.source_format)
    path = output_path(ref, job, suffix, formats.extension_for(fmt), language or doc.language)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    formats.write(doc, path, fmt)
    ctx.log(f"Wrote {os.path.basename(path)}")
    return TaskOutput(path=path, kind="subtitle", format=fmt, language=language or doc.language)


def preview_of(doc: SubtitleDoc, limit: int = 2000) -> Dict[str, Any]:
    return doc.to_dict(limit=limit)
