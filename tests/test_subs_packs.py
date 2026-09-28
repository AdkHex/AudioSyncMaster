"""Tests for engine packs: status, staged installs, the worker protocol.

Nothing here touches the network or a real pack. Workers are small scripts
run by this interpreter; "uv" is a fake that builds a venv whose python is a
link to this interpreter. What matters most:

*   a failed install never looks installed (staging + marker);
*   a worker's stray prints never corrupt the JSON protocol;
*   Stop kills a running worker, and a crashed one reports its stderr.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audiosync.media import Cancelled, CancellationToken  # noqa: E402
from audiosync.subs import packs  # noqa: E402
from audiosync.subs.tasks import TaskContext, TaskError  # noqa: E402

REAL_WORKERS = packs.workers_dir()
ASR_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"]


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def sandbox():
    """Temp packs dir and HF cache, no pack overrides, fake workers dir."""
    root = tempfile.mkdtemp(prefix="subsync-packs-test-")
    workers = os.path.join(root, "workers")
    os.makedirs(workers)
    shutil.copy(os.path.join(REAL_WORKERS, "_protocol.py"), workers)
    overrides = {k: None for k in os.environ if k.startswith("AUDIOSYNC_PACK_PYTHON_")}
    original = packs.workers_dir
    packs.workers_dir = lambda: workers
    try:
        with env(AUDIOSYNC_PACKS_DIR=os.path.join(root, "packs"), HF_HUB_CACHE=os.path.join(root, "hf"),
                 AUDIOSYNC_FFMPEG_FULL=None, AUDIOSYNC_UV=None, **overrides):
            yield root, workers
    finally:
        packs.workers_dir = original
        shutil.rmtree(root, ignore_errors=True)


def write_worker(workers: str, name: str, body: str) -> None:
    with open(os.path.join(workers, name), "w", encoding="utf-8") as handle:
        handle.write("import os, sys, time, json\n"
                     "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
                     "import _protocol as proto\n" + body)


def make_ctx(root: str):
    events = {"progress": [], "log": []}
    ctx = TaskContext(
        token=CancellationToken(),
        progress=lambda p, s: events["progress"].append((p, s)),
        log=lambda m: events["log"].append(m),
        workdir=root,
    )
    return ctx, events


# ------------------------------------------------------------------- status


def test_status_lists_every_pack_in_packstatus_shape():
    with sandbox():
        status = packs.status()
    ids = [s["id"] for s in status]
    assert ids == ["asr-mlx", "asr-faster", "demucs", "ocr-rapidocr", "ffmpeg-full"], ids
    for item in status:
        for key in ("id", "label", "description", "installed", "supported", "sizeBytes", "downloadBytes", "version"):
            assert key in item, (item["id"], key)
        assert item["installed"] is False
    by_id = {s["id"]: s for s in status}
    for pack_id in ("asr-mlx", "asr-faster"):
        assert [m["id"] for m in by_id[pack_id]["models"]] == ASR_MODELS
        assert all(m["installed"] is False and m["downloadBytes"] > 0 for m in by_id[pack_id]["models"])
    mac_arm = packs.host() == ("macos", "arm64")
    assert by_id["asr-mlx"]["supported"] is mac_arm
    assert by_id["asr-faster"]["supported"] is True


def test_packs_dir_override():
    with env(AUDIOSYNC_PACKS_DIR="/tmp/somewhere/packs"):
        assert packs.packs_dir() == "/tmp/somewhere/packs"
        assert packs.pack_dir("asr-faster") == "/tmp/somewhere/packs/asr-faster"


def test_python_override_adopts_an_external_environment():
    with sandbox():
        assert packs.python_for("asr-faster") is None
        with env(AUDIOSYNC_PACK_PYTHON_ASR_FASTER=sys.executable):
            assert packs.python_for("asr-faster") == sys.executable
            status = packs.pack_status("asr-faster")
            assert status["installed"] and status.get("external") and status["version"] == "external"
            # Removing must never delete someone else's environment.
            packs.remove("asr-faster")
            assert os.path.exists(sys.executable)
        with env(AUDIOSYNC_PACK_PYTHON_ASR_FASTER="/no/such/python"):
            assert packs.python_for("asr-faster") is None


def test_models_are_found_in_the_hugging_face_cache_through_links():
    with sandbox() as (root, _):
        hub = os.path.join(root, "hf")
        repo = os.path.join(hub, "models--Systran--faster-whisper-tiny")
        snap = os.path.join(repo, "snapshots", "abc123")
        os.makedirs(snap)
        os.makedirs(os.path.join(repo, "blobs"))
        # huggingface_hub >= 1.3x: repo blob -> shared store blob
        os.makedirs(os.path.join(hub, "blobs", "42"))
        shared = os.path.join(hub, "blobs", "42", "4200ff")
        with open(shared, "wb") as handle:
            handle.write(b"weights")
        os.symlink(os.path.join("..", "..", "blobs", "42", "4200ff"), os.path.join(repo, "blobs", "sha1"))
        os.symlink(os.path.join("..", "..", "blobs", "sha1"), os.path.join(snap, "model.bin"))
        spec = packs.model_spec("asr-faster", "tiny")
        assert packs.model_path(spec) == snap
        tiny = next(m for m in packs.pack_status("asr-faster")["models"] if m["id"] == "tiny")
        assert tiny["installed"] is True
        # A dangling link (download interrupted) is not a model.
        os.remove(shared)
        assert packs.model_path(spec) is None


def test_removing_a_model_frees_its_shared_blob_but_not_others():
    with sandbox() as (root, _):
        hub = os.path.join(root, "hf")

        def make_repo(repo_dir: str, blob: str, weight: str) -> str:
            snap = os.path.join(hub, repo_dir, "snapshots", "s")
            os.makedirs(snap)
            os.makedirs(os.path.join(hub, repo_dir, "blobs"))
            os.makedirs(os.path.join(hub, "blobs", blob[:2]), exist_ok=True)
            target = os.path.join(hub, "blobs", blob[:2], blob)
            with open(target, "wb") as handle:
                handle.write(b"x" * 10)
            os.symlink(os.path.join("..", "..", "blobs", blob[:2], blob), os.path.join(hub, repo_dir, "blobs", "b"))
            os.symlink(os.path.join("..", "..", "blobs", "b"), os.path.join(snap, weight))
            return target

        tiny = make_repo("models--Systran--faster-whisper-tiny", "aa11", "model.bin")
        base = make_repo("models--Systran--faster-whisper-base", "bb22", "model.bin")
        packs.remove("asr-faster", model="tiny")
        assert not os.path.exists(os.path.join(hub, "models--Systran--faster-whisper-tiny"))
        assert not os.path.exists(tiny), "the shared blob of the removed model must go too"
        assert os.path.exists(base), "another model's blob must stay"


def _hf_file(hub: str, repo_dir: str, rel: str, blob: str, data: bytes = b"x") -> str:
    """One cached file the way huggingface_hub 1.3x lays it out: snapshot
    link -> repo blob link -> shared-store blob. Returns the shared blob."""
    snap = os.path.join(hub, repo_dir, "snapshots", "s")
    os.makedirs(os.path.join(snap, os.path.dirname(rel)), exist_ok=True)
    os.makedirs(os.path.join(hub, repo_dir, "blobs"), exist_ok=True)
    os.makedirs(os.path.join(hub, "blobs", blob[:2]), exist_ok=True)
    shared = os.path.join(hub, "blobs", blob[:2], blob)
    with open(shared, "wb") as handle:
        handle.write(data)
    os.symlink(os.path.join("..", "..", "blobs", blob[:2], blob), os.path.join(hub, repo_dir, "blobs", blob))
    depth = rel.count("/")
    os.symlink(os.path.join(*([".."] * (depth + 2)), "blobs", blob), os.path.join(snap, rel))
    return shared


def test_ocr_models_share_a_repo_and_are_removed_one_by_one():
    with sandbox() as (root, _):
        hub = os.path.join(root, "hf")
        repo = "models--monkt--paddleocr-onnx"
        blobs = {}
        for group, n in (("chinese", "c1"), ("korean", "k1")):
            for name in ("rec.onnx", "dict.txt", "config.json"):
                blobs[(group, name)] = _hf_file(hub, repo, f"languages/{group}/{name}", f"{n}{name[:3]}")
        models = {m["id"]: m["installed"] for m in packs.pack_status("ocr-rapidocr")["models"]}
        assert models == {"cjk": True, "korean": True, "latin": False, "cyrillic": False}, models
        found = packs.ocr_recognizer("jpn")
        assert found["model"] == "cjk" and found["recModel"].endswith(os.path.join("chinese", "rec.onnx"))
        assert found["recKeys"].endswith(os.path.join("chinese", "dict.txt"))
        assert packs.ocr_recognizer("fr") == {"needed": "latin"}
        assert packs.ocr_recognizer("en") is None and packs.ocr_recognizer(None) is None
        packs.remove("ocr-rapidocr", model="korean")
        models = {m["id"]: m["installed"] for m in packs.pack_status("ocr-rapidocr")["models"]}
        assert models["korean"] is False and models["cjk"] is True, models
        assert not os.path.exists(blobs[("korean", "rec.onnx")]), "the removed model's blob must go"
        assert os.path.exists(blobs[("chinese", "rec.onnx")]) and os.path.exists(packs.ocr_recognizer("ja")["recModel"])


def test_ffmpeg_full_override_and_absence():
    with sandbox() as (root, _):
        assert packs.ffmpeg_full() is None
        fake = os.path.join(root, "ffmpeg")
        open(fake, "w").close()
        with env(AUDIOSYNC_FFMPEG_FULL=fake):
            assert packs.ffmpeg_full() == fake
            assert packs.pack_status("ffmpeg-full")["installed"] is True


def test_progress_adapter_passes_bytes_only_to_callbacks_that_take_them():
    two, four = [], []
    packs._progress_adapter(lambda p, s: two.append((p, s)))(50.7, "x", 10, 20)
    packs._progress_adapter(lambda p, s, d, t: four.append((p, s, d, t)))(50.7, "x", 10, 20)
    packs._progress_adapter(None)(1, "x")
    assert two == [(50, "x")] and four == [(50, "x", 10, 20)]


# ------------------------------------------------------------------ workers


def test_worker_protocol_progress_log_result_and_stray_prints():
    with sandbox() as (root, workers):
        write_worker(workers, "echo_worker.py", (
            "def handle(req):\n"
            "    print('library noise on stdout')\n"
            "    os.write(1, b'C-level noise on fd 1\\n')\n"
            "    proto.progress(40, 'Working')\n"
            "    proto.log('halfway')\n"
            "    return {'echo': req['value'], 'text': '日本語'}\n"
            "proto.main(handle)\n"
        ))
        ctx, events = make_ctx(root)
        with env(AUDIOSYNC_PACK_PYTHON_ASR_FASTER=sys.executable):
            result = packs.run_worker("asr-faster", "echo_worker.py", {"value": 7}, ctx)
    assert result == {"echo": 7, "text": "日本語"}, result
    assert (40, "Working") in events["progress"], events
    assert events["log"] == ["halfway"], events


def test_worker_error_becomes_task_error():
    with sandbox() as (root, workers):
        write_worker(workers, "fail_worker.py", (
            "def handle(req):\n"
            "    raise ValueError('model file is corrupt')\n"
            "proto.main(handle)\n"
        ))
        ctx, _ = make_ctx(root)
        with env(AUDIOSYNC_PACK_PYTHON_ASR_FASTER=sys.executable):
            try:
                packs.run_worker("asr-faster", "fail_worker.py", {}, ctx)
            except TaskError as exc:
                assert "model file is corrupt" in str(exc), exc
            else:
                raise AssertionError("worker error was swallowed")


def test_worker_crash_reports_exit_code_and_stderr_tail():
    with sandbox() as (root, workers):
        with open(os.path.join(workers, "crash_worker.py"), "w") as handle:
            handle.write("import sys\nsys.stderr.write('Segmentation fault in libfoo\\n')\nsys.exit(3)\n")
        ctx, _ = make_ctx(root)
        with env(AUDIOSYNC_PACK_PYTHON_ASR_FASTER=sys.executable):
            try:
                packs.run_worker("asr-faster", "crash_worker.py", {}, ctx)
            except TaskError as exc:
                assert "exit code 3" in str(exc) and "libfoo" in str(exc), exc
            else:
                raise AssertionError("crash was not reported")


def test_stop_kills_a_running_worker():
    with sandbox() as (root, workers):
        write_worker(workers, "slow_worker.py", (
            "def handle(req):\n"
            "    proto.progress(1, 'Starting')\n"
            "    time.sleep(60)\n"
            "    return {}\n"
            "proto.main(handle)\n"
        ))
        ctx, _ = make_ctx(root)
        threading.Timer(1.0, ctx.token.cancel).start()
        started = time.monotonic()
        with env(AUDIOSYNC_PACK_PYTHON_ASR_FASTER=sys.executable):
            try:
                packs.run_worker("asr-faster", "slow_worker.py", {}, ctx)
            except Cancelled:
                pass
            else:
                raise AssertionError("cancel was ignored")
        assert time.monotonic() - started < 10, "the worker was not killed"


def test_run_worker_without_the_pack_says_what_to_install():
    with sandbox() as (root, _):
        ctx, _ = make_ctx(root)
        try:
            packs.run_worker("asr-faster", "asr_worker.py", {}, ctx)
        except TaskError as exc:
            assert "Install" in str(exc), exc
        else:
            raise AssertionError("missing pack not reported")


def test_real_workers_do_not_import_audiosync():
    """Workers run in a pack's Python where audiosync does not exist."""
    for name in os.listdir(REAL_WORKERS):
        if name.endswith(".py"):
            with open(os.path.join(REAL_WORKERS, name), encoding="utf-8") as handle:
                source = handle.read()
            assert "import audiosync" not in source and "from audiosync" not in source, name


# ------------------------------------------------------------------ install

FAKE_UV = r'''#!{python}
import os, sys
args = sys.argv[1:]
log = os.environ.get("FAKE_UV_LOG")
if log:
    with open(log, "a") as handle:
        handle.write(" ".join(args) + "\n")
if args[:1] == ["venv"]:
    venv = args[-1]
    os.makedirs(os.path.join(venv, "bin"))
    os.symlink({python!r}, os.path.join(venv, "bin", "python"))
    print("Creating virtual environment at: " + venv)
    sys.exit(0)
if args[:2] == ["pip", "install"]:
    if os.environ.get("FAKE_UV_FAIL"):
        print("error: No solution found when resolving dependencies", file=sys.stderr)
        sys.exit(1)
    print("Resolved 3 packages in 10ms")
    print("Installed 3 packages in 5ms")
    sys.exit(0)
if args[:2] == ["pip", "freeze"]:
    print("fakepkg==1.0")
    sys.exit(0)
sys.exit(2)
'''


@contextlib.contextmanager
def fake_pack():
    spec = packs.PackSpec(
        id="asr-faster", label="Fake speech", description="test", kind="python", version="7",
        steps=(packs.InstallStep(("fakepkg==1.0",)), packs.InstallStep(("other",), no_deps=True)),
        import_check="import json", download_bytes=1, size_bytes=1, models=packs.PACKS["asr-faster"].models,
    )
    original = packs.PACKS["asr-faster"]
    packs.PACKS["asr-faster"] = spec
    try:
        yield spec
    finally:
        packs.PACKS["asr-faster"] = original


def _write_fake_uv(root: str) -> str:
    path = os.path.join(root, "uv")
    with open(path, "w") as handle:
        handle.write(FAKE_UV.replace("{python}", sys.executable).replace("{python!r}", repr(sys.executable)))
    os.chmod(path, 0o755)
    return path


def test_install_is_staged_and_a_failure_never_looks_installed():
    if os.name == "nt":
        return  # the fake uv is a shebang script
    with sandbox() as (root, _), fake_pack():
        uv = _write_fake_uv(root)
        with env(AUDIOSYNC_UV=uv, FAKE_UV_FAIL="1"):
            try:
                packs.install("asr-faster", None, None, CancellationToken())
            except TaskError as exc:
                assert "No solution found" in str(exc), exc
            else:
                raise AssertionError("failed install reported success")
        assert packs.python_for("asr-faster") is None
        assert not os.path.exists(packs.pack_dir("asr-faster"))
        leftovers = [n for n in os.listdir(packs.packs_dir()) if n.startswith(".staging")]
        assert leftovers == [], leftovers


def test_install_success_writes_marker_and_status():
    if os.name == "nt":
        return
    with sandbox() as (root, _), fake_pack():
        uv = _write_fake_uv(root)
        uv_log = os.path.join(root, "uv.log")
        seen = []
        with env(AUDIOSYNC_UV=uv, FAKE_UV_LOG=uv_log):
            status = packs.install("asr-faster", lambda p, s: seen.append((p, s)), None, CancellationToken())
        assert status["installed"] and status["version"] == "7", status
        marker = json.load(open(os.path.join(packs.pack_dir("asr-faster"), "pack.json")))
        assert marker["packages"] == ["fakepkg==1.0"] and marker["installedAt"], marker
        python = packs.python_for("asr-faster")
        assert python and python.startswith(packs.pack_dir("asr-faster")), python
        calls = open(uv_log).read().splitlines()
        assert any(c.startswith("venv --python 3.11 --relocatable") for c in calls), calls
        assert any("--no-deps" in c and c.endswith("other") for c in calls), calls
        assert seen[-1] == (100, "Installed") and all(0 <= p <= 100 for p, _ in seen)
        # Reinstalling swaps the new pack in; remove deletes it.
        with env(AUDIOSYNC_UV=uv):
            packs.install("asr-faster", None, None, CancellationToken())
        assert packs.pack_status("asr-faster")["installed"]
        packs.remove("asr-faster")
        assert packs.python_for("asr-faster") is None and not os.path.exists(packs.pack_dir("asr-faster"))


def test_uv_download_is_checksum_verified():
    """ensure_uv without uv anywhere: download (served locally), verify the
    release's sha256, unpack into packs/bin. A wrong checksum is refused."""
    with sandbox() as (root, _):
        target, _ext = packs._uv_target()
        archive = os.path.join(root, "uv.tar.gz")
        payload = os.path.join(root, "payload", f"uv-{target}")
        os.makedirs(payload)
        exe = "uv.exe" if os.name == "nt" else "uv"
        with open(os.path.join(payload, exe), "w") as handle:
            handle.write("#!/bin/sh\necho uv 0.0.0\n")
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(payload, arcname=f"uv-{target}")
        good = hashlib.sha256(open(archive, "rb").read()).hexdigest()
        original_download, original_fetch = packs.download, packs._fetch_bytes
        checksum = {"value": good}
        packs.download = lambda url, dest, *a, **k: shutil.copy(archive, dest)
        packs._fetch_bytes = lambda url, token: f"{checksum['value']}  uv-{target}.tar.gz\n".encode()
        try:
            checksum["value"] = "0" * 64
            try:
                packs.ensure_uv(None, lambda m: None, CancellationToken(), allow_path=False)
            except TaskError as exc:
                assert "corrupt" in str(exc), exc
            else:
                raise AssertionError("a bad checksum was accepted")
            assert packs.find_uv(allow_path=False) is None
            checksum["value"] = good
            path = packs.ensure_uv(None, lambda m: None, CancellationToken(), allow_path=False)
            assert path == packs._packs_uv() and os.path.isfile(path)
        finally:
            packs.download, packs._fetch_bytes = original_download, original_fetch


def test_archive_members_cannot_escape():
    with sandbox() as (root, _):
        archive = os.path.join(root, "evil.tar.gz")
        with tarfile.open(archive, "w:gz") as tf:
            data = b"owned"
            info = tarfile.TarInfo("../../escaped.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        try:
            packs._extract(archive, os.path.join(root, "out"))
        except TaskError:
            pass
        else:
            raise AssertionError("path traversal was extracted")
        assert not os.path.exists(os.path.join(root, "..", "escaped.txt"))


def test_download_reports_bytes_for_a_local_file():
    with sandbox() as (root, _):
        source = os.path.join(root, "source.bin")
        with open(source, "wb") as handle:
            handle.write(os.urandom(300_000))
        seen = []
        dest = os.path.join(root, "copy.bin")
        url = "file://" + source.replace(os.sep, "/")
        packs.download(url, dest, 300_000, on_bytes=lambda d, t: seen.append((d, t)), token=CancellationToken())
        assert open(dest, "rb").read() == open(source, "rb").read()
        assert seen and seen[-1] == (300_000, 300_000), seen[-3:]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
