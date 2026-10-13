"""Tests for the audio queue: a real AAC FPS change end to end, and the queue
rules -- duplicates, cancelling one or all, changing the worker count while
jobs run, and the summary when the queue drains.

The real job needs only ffmpeg. The queue-rule tests swap ffmpeg for a fake
that reports progress slowly, so a running job can be watched and cancelled.
Settings, history and deew's config go to a temporary folder.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import fps, settings, tools  # noqa: E402
from audiosync.encode.service import AudioQueue, check_spec  # noqa: E402
from audiosync.media import ffmpeg_path, ffprobe_path  # noqa: E402

FAKE_FFMPEG = r'''
import os, sys, time
out = sys.argv[-1]
steps = int(os.environ.get("FAKE_FFMPEG_STEPS", "40"))
for i in range(steps):
    print(f"out_time_us={i * 100000}", flush=True)
    print("progress=continue", flush=True)
    time.sleep(0.05)
with open(out, "wb") as handle:
    handle.write(b"x")
print("progress=end", flush=True)
'''


@contextlib.contextmanager
def sandbox():
    """Settings, history, temp folders and deew's config inside one temporary folder."""
    root = Path(tempfile.mkdtemp(prefix="encode-service-"))
    try:
        with mock.patch.object(settings, "config_dir", return_value=root / "cfg"), \
             mock.patch.object(settings, "cache_dir", return_value=root / "cache"), \
             mock.patch.object(tools, "deew_config_path", return_value=root / "deew" / "config.toml"):
            yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _sine(path: Path, seconds: float = 4.0, codec: str = "aac") -> Path:
    """A 48 kHz stereo tone, so one AAC frame is 21 ms."""
    subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate=48000",
                    "-ac", "2", "-c:a", codec, "-b:a", "128k", str(path)], check=True)
    return path


def _fake_ffmpeg(root: Path) -> str:
    script = root / "fake_ffmpeg.py"
    script.write_text(FAKE_FFMPEG, encoding="utf-8")
    if os.name == "nt":
        launcher = root / "fake_ffmpeg.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher = root / "fake_ffmpeg"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return str(launcher)


class Events:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, event: dict) -> None:
        with self._lock:
            self.items.append(event)

    def of(self, kind: str, job_id: str | None = None) -> list[dict]:
        with self._lock:
            return [e for e in self.items if e["type"] == kind and (job_id is None or e.get("jobId") == job_id)]


def _spec(job_id: str, path: Path, out: Path, conv_type: str = "23.976-25", **extra) -> dict:
    return {"id": job_id, "task": "fps", "path": str(path), "out_dir": str(out), "conv_type": conv_type, **extra}


def _wait(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _duration(path: str) -> float:
    out = subprocess.run([ffprobe_path(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def _adts_duration(path: str) -> float:
    """A raw .aac's real length: ffprobe only estimates it from the bitrate (53 ms off here),
    so count the 1024-sample frames and drop the encoder's one frame of priming."""
    out = subprocess.run([ffprobe_path(), "-v", "error", "-count_packets", "-select_streams", "a:0",
                          "-show_entries", "stream=nb_read_packets,sample_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    rate, packets = (int(v) for v in out.strip().split(","))
    return (packets - 1) * 1024 / rate


def test_real_aac_fps_change_retimes_names_and_reaches_100():
    with sandbox() as root:
        source = _sine(root / "film.m4a", 6.0)
        events = Events()
        queue = AudioQueue(events, workers=1)
        assert queue.add([_spec("j1", source, root / "out")])["added"] == ["j1"]
        assert queue.wait_idle(60)
        done = events.of("audioJobDone", "j1")[0]
        assert done["state"] == "done", done["error"]
        assert os.path.basename(done["output"]) == "film_23_976-25.aac"
        ratio = fps.FPS_CONVERSIONS["23.976-25"]
        frame = 1 / 25
        assert abs(_adts_duration(done["output"]) - _duration(str(source)) / ratio) <= frame, "not re-timed by 25/23.976"
        progress = [e["percent"] for e in events.of("audioJobProgress", "j1")]
        assert progress and progress[-1] == 100.0
        assert events.of("audioJobStart", "j1")
        assert events.of("audioIdle") == [{"type": "audioIdle", "summary": {"fps": {"done": 1}}}]
        log = events.of("audioJobLog", "j1")[-1]["lines"]
        assert any("-af atempo=1.042708" in line for line in log), "the job log carries the exact command line"
        assert not any(p.name.startswith("fpsjob_") for p in (root / "cache" / "temp").glob("*")), "temp folder left"
        history = settings.load_history()
        assert [e["id"] for e in history] == ["j1"] and len(history[0]["log"]) <= 20


def test_duplicates_and_bad_requests_are_skipped_with_a_reason():
    with sandbox() as root:
        source = _sine(root / "a.m4a", 1.0)
        queue = AudioQueue(Events(), workers=1, tool_overrides={"ffmpeg": _fake_ffmpeg(root)})
        result = queue.add([_spec("a", source, root / "out"), _spec("b", source, root / "out"),
                            _spec("c", source, root / "out", conv_type="30-25"),
                            _spec("d", root / "missing.m4a", root / "out")])
        assert result["added"] == ["a"]
        reasons = {s["id"]: s["reason"] for s in result["skipped"]}
        assert reasons["b"] == "already queued"
        assert reasons["c"] == "unknown conversion '30-25'"
        assert reasons["d"].startswith("file not found")
        queue.cancel_all()
        assert queue.wait_idle(20)


def test_cancel_queued_and_running_jobs():
    with sandbox() as root:
        a, b = _sine(root / "a.m4a", 1.0), _sine(root / "b.m4a", 1.0)
        events = Events()
        queue = AudioQueue(events, workers=1, tool_overrides={"ffmpeg": _fake_ffmpeg(root)})
        queue.add([_spec("a", a, root / "out"), _spec("b", b, root / "out")])
        assert _wait(lambda: events.of("audioJobProgress", "a"))
        assert queue.cancel("b")                              # still queued
        assert events.of("audioJobDone", "b")[0]["error"] == "cancelled before start"
        assert queue.cancel("a")                              # running: its ffmpeg tree is killed
        assert queue.wait_idle(20)
        done = events.of("audioJobDone", "a")[0]
        assert done["state"] == "cancelled"
        assert not os.path.exists(root / "out" / "a_23_976-25.aac"), "partial output left behind"
        assert not any((root / "cache" / "temp").glob("fpsjob_*")), "temp folder left behind"
        assert events.of("audioIdle")[-1]["summary"] == {"fps": {"cancelled": 2}}
        assert not queue.cancel("a")                          # already over


def test_worker_count_changes_while_jobs_run():
    with sandbox() as root:
        sources = [_sine(root / f"s{i}.m4a", 1.0) for i in range(3)]
        events = Events()
        queue = AudioQueue(events, workers=1, tool_overrides={"ffmpeg": _fake_ffmpeg(root)})
        queue.add([_spec(f"j{i}", s, root / "out") for i, s in enumerate(sources)])
        assert _wait(lambda: len(events.of("audioJobStart")) == 1), "the first job never started"
        time.sleep(0.3)
        assert len(events.of("audioJobStart")) == 1, "one worker runs one job"
        assert queue.set_workers(3) == 3
        assert _wait(lambda: len(events.of("audioJobStart")) == 3), "new slots start the waiting jobs"
        assert queue.set_workers(99) == 8
        assert queue.set_workers(0) == 1
        assert queue.wait_idle(30), "the queue never drained"
        summaries = [e["summary"] for e in events.of("audioIdle")]
        assert summaries == [{"fps": {"done": 3}}], summaries


def test_a_job_added_while_running_joins_the_live_queue():
    with sandbox() as root:
        a, b = _sine(root / "a.m4a", 1.0), _sine(root / "b.m4a", 1.0)
        events = Events()
        queue = AudioQueue(events, workers=1, tool_overrides={"ffmpeg": _fake_ffmpeg(root)})
        queue.add([_spec("a", a, root / "out")])
        assert _wait(lambda: events.of("audioJobProgress", "a"))
        queue.add([_spec("b", b, root / "out")])
        assert queue.wait_idle(30)
        assert [e["jobId"] for e in events.of("audioJobDone")] == ["a", "b"]
        assert len(events.of("audioIdle")) == 1, "one drain, one summary"


def test_encode_spec_checks():
    with tempfile.NamedTemporaryFile(suffix=".mka") as handle:
        base = {"path": handle.name, "out_dir": "/o", "task": "encode"}
        assert check_spec({**base, "encode": {"target": "ddp", "channels": 6}}) == ""
        assert check_spec({**base, "encode": {"target": "dts"}}) == "unknown target 'dts'"
        assert check_spec({**base, "encode": {"channels": 4}}) == "channels must be 0, 1, 2, 6 or 8"
        assert check_spec({**base, "encode": {"drc": "loud"}}) == "unknown DRC profile 'loud'"
        assert check_spec({**base, "encode": {"atmos71": "wide"}}) == "unknown 7.1 Atmos layout 'wide'"
        assert check_spec({**base, "out_dir": ""}) == "choose an output folder first"


def test_dolby_source_without_dee_fails_with_the_fix():
    with sandbox() as root:
        source = _sine(root / "film.ac3", 1.0, codec="ac3")
        events = Events()
        queue = AudioQueue(events, workers=1)
        queue.add([_spec("j", source, root / "out", conv_type="25-24")])
        assert queue.wait_idle(30)
        done = events.of("audioJobDone", "j")[0]
        assert done["state"] == "failed"
        assert done["error"] == "AC3 output needs Dolby Encoding Engine: set the dee.exe path in Preferences → Encode first"
