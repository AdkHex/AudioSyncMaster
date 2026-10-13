"""Running one job: the tool runners and the FPS and encode pipelines
(FPSConv ``engine.py:1125-1491``, ``:1812-2124``).

The pipelines are FPSConv's step for step, with three differences:

* tools come from a :class:`~audiosync.encode.tools.Tools` resolved when the
  job starts instead of a global settings read per call;
* cancelling kills the running tool's whole process tree through the job's
  :class:`~audiosync.media.CancellationToken` (FPSConv killed only the top
  process, so DEE could outlive a cancelled deew);
* FPS progress is measured against the re-timed length, so a speed-up
  reaches 100 % (fix #2; FPSConv stopped at 95.9 % for 23.976 → 25).
"""

from __future__ import annotations

import os
import queue as _queue
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

from . import atmos, commands, flat71, probe, settings
from .formats import ENCODE_CONV, TASK_ENCODE, encode_output_name, encode_plan, resolve_encode
from .fps import DEE_CODEC_MAP, FPS_CONVERSIONS, atempo_chain, output_name
from .job import Job, ProgressFn, next_free
from .progress import (
    ANSI_RE,
    DEEW_STAGES,
    DEEZY_STAGES,
    ERR_RE,
    FFMPEG_NOISE,
    FFMPEG_PROGRESS_KEYS,
    PATCHER_STAGES,
    TRUEHDD_STAGES,
    ParseFn,
    PatcherParser,
    ffmpeg_percent,
    fmt_time,
    hr_size,
    parse_deew_line,
    parse_deezy_line,
    parse_truehdd_line,
    pick_error,
    stage_of,
)
from .tools import Tools, ffmpeg_has_soxr

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _spawn(cmd: list[str], job: Job, env: Optional[dict] = None, cwd: Optional[str] = None) -> subprocess.Popen:
    """Start a tool with its output on one pipe, registered with the job's token so cancel kills its tree."""
    kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = _NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, env=env, cwd=cwd or None, **kwargs)
    job.token.register(proc)
    return proc


def _reap(proc: subprocess.Popen, job: Job) -> None:
    proc.wait()
    job.token.unregister(proc)


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


# ───────────────────────── runners ─────────────────────────


def run_ffmpeg_progress(cmd: list[str], job: Job, step_name: str, total_dur: float, notify: ProgressFn,
                        ratio: float = 1.0) -> bool:
    """Run ffmpeg with ``-progress pipe:1``. ``total_dur`` is the source's length;
    ``ratio`` the speed change, so the output runs ``total_dur / ratio``."""
    cmd = commands.with_progress(cmd)
    out_total = total_dur / ratio if ratio > 0 else total_dur
    job.say(f"{step_name}: $ " + " ".join(cmd))
    try:
        proc = _spawn(cmd, job)
    except OSError as exc:
        job.error = f"cannot start ffmpeg: {exc}"
        job.say(job.error, "error")
        return False
    job.step = step_name
    job.stage_begin(stage_of(step_name))
    job.detail = "starting ffmpeg"
    start = time.time()
    last = 0.0
    tail: list[str] = []
    state: dict[str, str] = {}
    assert proc.stdout is not None
    for raw in proc.stdout:
        if job.cancelled:
            break
        line = raw.decode("utf-8", "ignore").strip()
        if not line:
            continue
        if "=" in line and line.split("=", 1)[0] in FFMPEG_PROGRESS_KEYS:
            key, value = line.split("=", 1)
            state[key] = value.strip()
            if key != "progress":            # "progress=" closes each block; speed/bitrate are in by then
                continue
            try:
                cur = int(state.get("out_time_us", "0")) / 1_000_000.0
            except ValueError:
                continue
            job.percent = ffmpeg_percent(cur, total_dur, ratio)
            job.stage_progress(job.percent)
            now = time.time()
            job.elapsed = now - start
            speed = cur / job.elapsed if job.elapsed > 0 else 0
            job.eta = (out_total - cur) / speed if speed > 0 else 0
            job.detail = "  ·  ".join(p for p in (f"ffmpeg {fmt_time(cur)} / {fmt_time(out_total)}",
                                                  state.get("speed", ""), state.get("bitrate", ""))
                                     if p and p != "N/A")          # ffmpeg prints N/A for raw streams
            if now - last >= 1.0:
                job.progress_line(f"{step_name}: {job.detail}  ·  {job.percent:.1f}%")
                notify(job)
                last = now
            continue
        tail.append(line)
        tail = tail[-30:]
        if FFMPEG_NOISE.match(line):
            continue                          # version banner / build config: not useful in a job log
        job.say(f"ffmpeg: {line}", "warning" if ERR_RE.search(line) else "info")
    _reap(proc, job)
    job.elapsed = time.time() - start
    if proc.returncode != 0 and not job.cancelled:
        job.error = pick_error(tail) or f"ffmpeg exited with code {proc.returncode}"
        job.say(f"ffmpeg exited with code {proc.returncode}: {job.error}", "error")
    else:
        job.say(f"{step_name}: finished in {fmt_time(job.elapsed)}")
    return proc.returncode == 0 and not job.cancelled


def run_deew_progress(cmd: list[str], job: Job, step_name: str, notify: ProgressFn) -> bool:
    """Run deew and turn its rich progress bars into stage + percent.

    deew draws its bars with *rich*, which only refreshes live on a terminal.
    ``FORCE_COLOR=1`` makes rich treat the pipe as a terminal, so the bar is
    redrawn continuously (carriage returns + ANSI codes) and we can read
    ``[ DEE: measure | file ] ━━━ 42.10%`` as it happens.
    """
    job.say(f"{step_name}: $ " + " ".join(cmd))
    env = dict(os.environ, FORCE_COLOR="1", TERM="xterm-256color", COLUMNS="120",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        proc = _spawn(cmd, job, env=env)
    except OSError as exc:
        job.error = f"cannot start deew: {exc}"
        job.say(job.error, "error")
        return False
    job.step = step_name
    job.detail = "starting deew"
    job.percent = 0.0
    start = time.time()
    last = 0.0
    buffer = b""
    tail: list[str] = []
    seen_plain: set[str] = set()
    stage = ""
    stage_started = start
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(16)            # 16-byte reader: survives \r-only output
        if not chunk:
            break
        if job.cancelled:
            break
        buffer += chunk
        # Split on CR/LF; keep the unfinished remainder for the next chunk.
        parts = re.split(rb"[\r\n]+", buffer)
        buffer = parts.pop()
        for raw in parts:
            text = ANSI_RE.sub("", raw.decode("utf-8", "ignore")).strip()
            if not text:
                continue
            parsed = parse_deew_line(text)
            if parsed:
                name, pct = parsed
                if name != stage:
                    if stage:
                        job.say(f"deew: {stage} finished in {fmt_time(time.time() - stage_started)}")
                    stage, stage_started = name, time.time()
                    job.say(f"deew: {name} started")
                    job.stage_begin(name)
                lo, hi = DEEW_STAGES.get(name, (0, 100))
                job.percent = min(100.0, lo + (hi - lo) * pct / 100.0)
                job.stage_progress(pct)
                now = time.time()
                job.elapsed = now - start
                stage_el = now - stage_started
                job.eta = (stage_el / pct) * (100 - pct) if pct > 0 else 0
                job.detail = f"{name}  ·  {pct:.1f}%"
                if now - last >= 1.0:
                    job.progress_line(f"deew: {name}  {pct:.1f}%  ·  {fmt_time(stage_el)} in this stage")
                    notify(job)
                    last = now
                continue
            if text in seen_plain:
                continue
            seen_plain.add(text)
            tail.append(text)
            tail = tail[-40:]
            job.say(f"deew: {text}", "warning" if ERR_RE.search(text) else "info")
    if buffer.strip():
        text = ANSI_RE.sub("", buffer.decode("utf-8", "ignore")).strip()
        if text and text not in seen_plain:
            tail.append(text)
            job.say(f"deew: {text}")
    _reap(proc, job)
    job.elapsed = time.time() - start
    if stage:
        job.say(f"deew: {stage} finished in {fmt_time(time.time() - stage_started)}")
    if proc.returncode != 0 and not job.cancelled:
        job.error = pick_error(tail) or f"deew exited with code {proc.returncode}"
        job.say(f"deew exited with code {proc.returncode}: {job.error}", "error")
    else:
        job.say(f"{step_name}: deew exited with code {proc.returncode} after {fmt_time(job.elapsed)}")
    return proc.returncode == 0 and not job.cancelled


#: How often a quiet tool's row is refreshed with what it has written so far.
IDLE_HEARTBEAT_S = 5.0


def run_tool_progress(cmd: list[str], job: Job, step_name: str, notify: ProgressFn, *,
                      tool_name: str, parse: ParseFn, stages: dict[str, tuple[int, int]],
                      work_dir: str = "", idle_stage: str = "working", env_extra: Optional[dict] = None,
                      cwd: Optional[str] = None) -> bool:
    """Run a tool whose stdout carries stage / percent lines and turn them into progress.

    ``parse(line)`` returns ``(stage, percent-or-None)`` for lines that mean
    something, ``None`` otherwise. ``stages`` maps a stage name to the share of
    the job it gets on the bar. While the tool is quiet the detail line shows how
    much has been written into ``work_dir`` so a long decode is visibly alive.
    """
    job.say(f"{step_name}: $ " + " ".join(cmd))
    env = dict(os.environ, NO_COLOR="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **(env_extra or {}))
    try:
        proc = _spawn(cmd, job, env=env, cwd=cwd)
    except OSError as exc:
        job.error = f"cannot start {tool_name}: {exc}"
        job.say(job.error, "error")
        return False
    job.step = step_name
    job.stage_begin(idle_stage)
    job.detail = f"starting {tool_name}"
    job.percent = min(v[0] for v in stages.values()) if stages else 0.0
    start = time.time()
    last = 0.0
    tail: list[str] = []
    stage = ""
    stage_started = start
    last_text = ""
    lines: "_queue.Queue[Optional[bytes]]" = _queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=reader, daemon=True, name=f"{tool_name}-reader").start()

    def set_stage(name: str) -> None:
        nonlocal stage, stage_started
        if name != stage:
            if stage:
                job.say(f"{tool_name}: {stage} finished in {fmt_time(time.time() - stage_started)}")
            stage, stage_started = name, time.time()
            job.say(f"{tool_name}: {name} started")
            job.stage_begin(name)

    def progress(name: str, pct: float) -> None:
        nonlocal last
        set_stage(name)
        lo, hi = stages.get(name, (0, 100))
        job.percent = min(100.0, lo + (hi - lo) * pct / 100.0)
        job.stage_progress(pct)
        now = time.time()
        job.elapsed = now - start
        stage_el = now - stage_started
        job.eta = (stage_el / pct) * (100 - pct) if pct > 0 else 0
        job.detail = f"{name}  ·  {pct:.1f}%"
        if now - last >= 1.0:
            job.progress_line(f"{tool_name}: {name}  {pct:.1f}%  ·  {fmt_time(stage_el)} in this stage")
            notify(job)
            last = now

    while True:
        if job.cancelled:
            break
        try:
            raw = lines.get(timeout=1.0)
        except _queue.Empty:
            now = time.time()
            job.elapsed = now - start
            written = _dir_size(work_dir) if work_dir and os.path.isdir(work_dir) else 0
            what = stage or idle_stage
            job.detail = f"{what}  ·  {hr_size(written)} written to temp  ·  {fmt_time(now - stage_started)} in this stage"
            if now - last >= IDLE_HEARTBEAT_S:
                job.progress_line(f"{tool_name}: {job.detail}")
                notify(job)
                last = now
            continue
        if raw is None:
            break
        text = ANSI_RE.sub("", raw.decode("utf-8", "ignore")).strip()
        if not text or text == last_text:          # debug levels repeat every info line
            continue
        last_text = text
        parsed = parse(text)
        if parsed is not None:
            name, pct = parsed
            if pct is not None:
                progress(name, pct)
                continue
            if not stage:
                set_stage(name)
        tail.append(text)
        tail = tail[-40:]
        job.say(f"{tool_name}: {text}", "warning" if ERR_RE.search(text) else "info")
    _reap(proc, job)
    job.elapsed = time.time() - start
    if stage:
        job.say(f"{tool_name}: {stage} finished in {fmt_time(time.time() - stage_started)}")
    if proc.returncode != 0 and not job.cancelled:
        job.error = pick_error(tail) or f"{tool_name} exited with code {proc.returncode}"
        job.say(f"{tool_name} exited with code {proc.returncode}: {job.error}", "error")
    else:
        job.say(f"{step_name}: {tool_name} exited with code {proc.returncode} after {fmt_time(job.elapsed)}")
    return proc.returncode == 0 and not job.cancelled


def run_deezy_progress(cmd: list[str], job: Job, step_name: str, notify: ProgressFn,
                       work_dir: str = "") -> bool:
    """DeeZy: ``truehdd (1 of 3)  42.0%``, ``DEE measure (2 of 3) …``, ``DEE encode (3 of 3) …``.

    For a raw ``.thd`` DeeZy knows no duration, so its truehdd stage prints no
    percentage; ``DEEZY_NO_PROGRESS`` and ``--no-progress-bars`` keep it from
    drawing rich bars.
    """
    return run_tool_progress(cmd, job, step_name, notify, tool_name="deezy", parse=parse_deezy_line,
                             stages=DEEZY_STAGES, work_dir=work_dir, idle_stage="TrueHD decode",
                             env_extra={"DEEZY_NO_PROGRESS": "1"})


# ───────────────────────── pipelines ─────────────────────────


def _has_dee(dee: str) -> bool:
    return bool(dee) and (os.path.isfile(dee) or bool(shutil.which(dee)))


def convert(job: Job, notify: ProgressFn, tools: Tools, work_root: Optional[str] = None) -> Job:
    """Run one job to completion (or cancellation). Mutates and returns ``job``.

    ``work_root`` (None = the Temp folder from Preferences) gets one sub-folder per job.
    """
    work_root = work_root or str(settings.temp_dir())
    if job.task == TASK_ENCODE:
        return convert_encode(job, notify, tools, work_root)

    file_path = job.source
    conv_type = job.conv_type

    def finish(state: str, error: str = "") -> Job:
        job.state, job.error = state, error
        job.finished_at = time.time()
        notify(job)
        return job

    if conv_type not in FPS_CONVERSIONS:
        return finish("failed", f"Invalid FPS conversion type: {conv_type}")
    if not os.path.exists(file_path):
        return finish("failed", f"File not found: {file_path}")

    ratio = FPS_CONVERSIONS[conv_type]
    job.codec, src_bitrate, job.channels = probe.detect_audio_info(tools.ffprobe, file_path, job.stream_index)
    job.bitrate = job.bitrate_override or src_bitrate
    job.duration = probe.get_duration(tools.ffprobe, file_path)

    codec = job.codec
    fmt, ext = "ddp", ".ec3"
    if codec != "aac":
        _, fmt, ext = DEE_CODEC_MAP.get(codec, ("-f", "ddp", ".ec3"))
        if not _has_dee(tools.dee):
            return finish("failed", f"{codec.upper()} output needs Dolby Encoding Engine: "
                                    "set the dee.exe path in Preferences → Encode first")
    out_path = os.path.join(job.out_dir, output_name(file_path, conv_type, codec, job.stream_index))

    if os.path.exists(out_path):
        if job.overwrite == "skip":
            job.out_path = out_path
            job.out_size = os.path.getsize(out_path)
            job.percent = 100.0
            return finish("skipped", "output already exists")
        if job.overwrite == "rename":
            out_path = next_free(out_path)
    job.out_path = out_path

    work_dir = os.path.join(work_root, f"fpsjob_{job.id}")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(job.out_dir, exist_ok=True)
    job.state = "running"
    job.step = "Initializing"
    job.started_at = time.time()
    job.say(f"job started: {os.path.basename(file_path)}  →  {os.path.basename(out_path)}")
    job.say(f"source: {codec.upper()} · {job.bitrate} kbps · {job.channels} ch · {fmt_time(job.duration)} · "
            f"stream #{job.stream_index} · mode {conv_type} (atempo {atempo_chain(ratio)})")
    job.plan_stages(["ffmpeg atempo + AAC", "Finishing"] if codec == "aac"
                    else ["Extracting WAV", "DEE: measure", "DEE: encode", "Finishing"])
    notify(job)

    t0 = time.time()
    success = False
    try:
        if codec == "aac":
            cmd = commands.fps_aac(tools.ffmpeg, file_path, job.stream_index, job.bitrate, job.channels, ratio, out_path)
            success = run_ffmpeg_progress(cmd, job, "Encoding AAC Engine", job.duration, notify, ratio)
        else:
            # Unique name: deew writes its own wav/xml under ONE shared temp folder
            # using this basename, so two parallel jobs called "temp_extract"
            # would delete each other's files (FileNotFoundError in deew's cleanup).
            temp_wav = os.path.join(work_dir, f"fpsconv_{job.id}.wav")
            wav_cmd = commands.fps_wav(tools.ffmpeg, file_path, job.stream_index, job.channels, ratio, temp_wav)
            wav_ok = run_ffmpeg_progress(wav_cmd, job, "Extracting WAV (1/2)", job.duration, notify, ratio)
            if wav_ok and os.path.exists(temp_wav) and not job.cancelled:
                cmd = commands.deew_fps(tools.deew, temp_wav, fmt, job.bitrate, work_dir)
                dee_ok = run_deew_progress(cmd, job, "Dolby DEE Encoding (2/2)", notify)
                base = os.path.splitext(os.path.basename(temp_wav))[0]
                deew_out = commands.deew_output(work_dir, base)
                if dee_ok and deew_out:
                    shutil.move(deew_out, out_path)
                    success = True
                elif dee_ok:
                    job.error = (f"deew reported success but {base}{ext} was not produced — "
                                 f"check the DEE path and the deew lines above")
    except Exception as exc:  # noqa: BLE001 - a job must never take the engine down
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        _settle(job, out_path, work_dir, t0, success, notify)
    return job


def convert_encode(job: Job, notify: ProgressFn, tools: Tools, work_root: Optional[str] = None) -> Job:
    """Audio-only encode: source track -> DD / DDP / DDP Atmos, no speed change."""
    work_root = work_root or str(settings.temp_dir())
    file_path = job.source

    def finish(state: str, error: str = "") -> Job:
        job.state, job.error = state, error
        job.finished_at = time.time()
        notify(job)
        return job

    job.conv_type = ENCODE_CONV
    if not os.path.exists(file_path):
        return finish("failed", f"File not found: {file_path}")
    stream = probe.find_stream(tools.ffprobe, file_path, job.stream_index, atmos.reader(tools.mediainfo, tools.pack_python))
    if stream is None:
        return finish("failed", "no audio stream found")
    if stream["index"] != job.stream_index:
        return finish("failed", f"audio stream #{job.stream_index} not found")

    enc = resolve_encode(job.target, job.target_channels, job.atmos, job.bitrate_override,
                         stream["channels"], stream["atmos"], stream["codec"], job.atmos71)
    job.codec, job.pretty = stream["codec"], stream["pretty"]
    job.bitrate, job.channels, job.label = enc.bitrate, enc.out_channels, enc.label
    job.duration = probe.get_duration(tools.ffprobe, file_path)

    if not _has_dee(tools.dee):
        return finish("failed", f"{enc.label} needs Dolby Encoding Engine: set the dee.exe path in Preferences → Encode first")
    if enc.atmos and not tools.deezy_tools()["truehdd"]:
        return finish("failed", "DDP Atmos needs truehdd: set its path in Preferences → Encode or untick Keep Atmos")
    if enc.atmos and enc.flat71:
        ok, why = tools.flat71_ready()
        if not ok:
            return finish("failed", f"flat 7.1 Atmos: {why}")

    out_path = os.path.join(job.out_dir, encode_output_name(file_path, job.stream_index, enc))
    if os.path.exists(out_path):
        if job.overwrite == "skip":
            job.out_path = out_path
            job.out_size = os.path.getsize(out_path)
            job.percent = 100.0
            return finish("skipped", "output already exists")
        if job.overwrite == "rename":
            out_path = next_free(out_path)
    job.out_path = out_path

    work_dir = os.path.join(work_root, f"fpsjob_{job.id}")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(job.out_dir, exist_ok=True)
    job.state = "running"
    job.step = "Initializing"
    job.started_at = time.time()
    job.say(f"job started: {os.path.basename(file_path)}  →  {os.path.basename(out_path)}")
    job.say(f"source: {stream['pretty']} · {stream['channels']} ch · {stream['sample_rate']} Hz"
            f"{' · ' + stream['layout'] if stream.get('layout') else ''} · {fmt_time(job.duration)} · stream #{job.stream_index}")
    job.say(f"target: {enc.label}")
    for note in enc.notes:
        job.say(f"note: {note}")
    for line in encode_plan(enc, stream):
        job.say(f"plan: {line}")
    if enc.atmos and enc.flat71:
        job.plan_stages(["Extracting TrueHD", "truehdd decode", "Patching DEE", "DEE measure", "DEE encode",
                         "Surround EX flag", "Finishing"])
        if (stream.get("format_name") or "") == "truehd":
            job.stage_skip("Extracting TrueHD", "source is already a .thd")
    elif enc.atmos:
        job.plan_stages(["TrueHD decode", "DEE measure", "DEE encode", "Finishing"])
    else:
        job.plan_stages(["Decoding to WAV", "DEE: measure", "DEE: encode", "Finishing"])
    notify(job)

    t0 = time.time()
    success = False
    try:
        if enc.atmos and enc.flat71:
            success = _encode_flat71(job, stream, enc, work_dir, out_path, notify, tools)
        elif enc.atmos:
            cmd = commands.deezy_atmos(tools.deezy, file_path, job.stream_index, enc, job.drc, work_dir, out_path,
                                       _deezy_work_dir(), tools.deezy_tools())
            ok = run_deezy_progress(cmd, job, "DeeZy Atmos (truehdd → DEE)", notify, work_dir=work_dir)
            if ok and os.path.exists(out_path):
                success = True
            elif ok:
                job.error = f"deezy reported success but {os.path.basename(out_path)} was not produced"
        else:
            # Unique name: deew writes its own wav/xml under ONE shared temp folder
            # using this basename (see convert()).
            temp_wav = os.path.join(work_dir, f"fpsconv_{job.id}.wav")
            soxr = ffmpeg_has_soxr(tools.ffmpeg)
            if int(stream["sample_rate"]) != 48000:
                resampler = "soxr" if soxr else "ffmpeg swresample (no libsoxr in this build)"
                job.say(f"note: {stream['sample_rate']} Hz source resampled to 48 kHz with {resampler}")
            wav_ok = run_ffmpeg_progress(commands.encode_wav(tools.ffmpeg, file_path, job.stream_index, stream, enc,
                                                             temp_wav, soxr=soxr),
                                         job, "Decoding to WAV (1/2)", job.duration, notify)
            if not wav_ok and soxr and not job.cancelled and int(stream["sample_rate"]) != 48000:
                job.say("note: soxr resampler failed, retrying with ffmpeg's default resampler", "warning")
                job.error = ""
                wav_ok = run_ffmpeg_progress(commands.encode_wav(tools.ffmpeg, file_path, job.stream_index, stream,
                                                                 enc, temp_wav, soxr=False),
                                             job, "Decoding to WAV (1/2)", job.duration, notify)
            if wav_ok and os.path.exists(temp_wav) and not job.cancelled:
                cmd = commands.deew_encode(tools.deew, temp_wav, enc, job.drc, work_dir)
                dee_ok = run_deew_progress(cmd, job, "Dolby DEE Encoding (2/2)", notify)
                base = os.path.splitext(os.path.basename(temp_wav))[0]
                deew_out = commands.deew_output(work_dir, base)
                if dee_ok and deew_out:
                    shutil.move(deew_out, out_path)
                    success = True
                elif dee_ok:
                    job.error = (f"deew reported success but {base}{enc.ext} was not produced — "
                                 f"check the DEE path and the deew lines above")
    except Exception as exc:  # noqa: BLE001 - a job must never take the engine down
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        _settle(job, out_path, work_dir, t0, success, notify)
        if enc.atmos and enc.flat71 and job.state == "cancelled":
            _report_dll_after_cancel(job, tools)
    return job


def _deezy_work_dir() -> str:
    path = settings.work_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(path)


def _encode_flat71(job: Job, stream: dict, enc, work_dir: str, out_path: str, notify: ProgressFn,
                   tools: Tools) -> bool:
    """TrueHD Atmos → flat DD+ 7.1 JOC: ffmpeg copy → truehdd DAMF → patched DEE via the wrapper."""
    deezy = tools.deezy_tools()
    src = job.source
    # 1/5 the TrueHD elementary stream (skip the copy when the source already is one)
    thd = src
    if (stream.get("format_name") or "") != "truehd":
        thd = os.path.join(work_dir, f"fpsconv_{job.id}.thd")
        if not run_ffmpeg_progress(commands.thd_extract(tools.ffmpeg, src, job.stream_index, thd), job,
                                   "Extracting TrueHD (1/5)", job.duration, notify):
            return False
        if job.cancelled:
            return False
    # 2/5 truehdd → DAMF master
    master = os.path.join(work_dir, "master")
    ok = run_tool_progress(commands.truehdd(deezy["truehdd"], thd, master, job.bed_conform), job,
                           "truehdd decode (2/5)", notify, tool_name="truehdd", parse=parse_truehdd_line,
                           stages=TRUEHDD_STAGES, work_dir=work_dir, idle_stage="truehdd decode")
    if not ok or job.cancelled:
        return False
    master_atmos = master + ".atmos"
    if not os.path.isfile(master_atmos):
        job.error = "truehdd finished but wrote no master.atmos — is this really a TrueHD Atmos track?"
        job.say(job.error, "error")
        return False
    if thd != src:
        _remove(thd)                       # the copy is not needed any more; the DAMF is the big one now
    # 3/5 + 4/5 the wrapper: patched DEE (measure + encode), then the Surround EX flag
    dee_temp = os.path.join(work_dir, "dee")
    os.makedirs(dee_temp, exist_ok=True)
    script = tools.patcher_script
    assert script is not None              # flat71_ready() checked it before the job started
    cmd = commands.patcher(tools.runner, script, deezy["dee"], master_atmos, out_path, enc, job.drc, dee_temp)
    ok = run_tool_progress(cmd, job, "DEE flat 7.1 (3/5–4/5)", notify, tool_name="patcher", parse=PatcherParser(),
                           stages=PATCHER_STAGES, work_dir=dee_temp, idle_stage="Patching DEE",
                           cwd=str(script.parent))
    if not ok:
        return False
    if not os.path.exists(out_path):
        job.error = f"the patcher reported success but {os.path.basename(out_path)} was not produced"
        job.say(job.error, "error")
        return False
    return True


def _report_dll_after_cancel(job: Job, tools: Tools) -> None:
    """Cancelling kills the wrapper before it restores DEE's DLL; say what state it was left in."""
    st = flat71.dee_dll_status(tools.dee)
    if st["state"] == "patched":
        job.say("DEE's DLL was left patched by the cancelled run; the next flat 7.1 run restores it "
                "from the wrapper's backup first", "warning")
    elif st["state"] != "supported":
        job.say(f"DEE's DLL after cancel: {st['state']} {st.get('sha256', '')[:12]}", "warning")


def _settle(job: Job, out_path: str, work_dir: str, t0: float, success: bool, notify: ProgressFn) -> None:
    """Common tail of a job: clean up, set the terminal state, remove partial output."""
    ok = success and not job.cancelled and os.path.exists(out_path)
    if ok:
        job.stage_begin("Finishing")           # the temp folder can be many GB; deleting it is visible work
    shutil.rmtree(work_dir, ignore_errors=True)
    job.stage_close(ok)
    job.elapsed = time.time() - t0
    job.finished_at = time.time()
    job.detail = ""
    if job.cancelled:
        job.state = "cancelled"
        job.error = "cancelled by user"
        _remove(out_path)
        job.say("cancelled — partial output removed", "warning")
    elif success and os.path.exists(out_path):
        job.state = "done"
        job.percent = 100.0
        job.eta = 0.0
        job.out_size = os.path.getsize(out_path)
        job.say(f"done in {fmt_time(job.elapsed)}: {out_path} ({hr_size(job.out_size)})")
    else:
        job.state = "failed"
        job.error = job.error or "Conversion failed — check codec / format"
        _remove(out_path)
        job.say(f"failed: {job.error}", "error")
    notify(job)


def _remove(path: str) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
