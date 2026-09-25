"""The voice tools: an optional, separately installed speech detector and
voice separator, and the worker process that runs them.

The engine itself needs only numpy, and stays that way: it is frozen into
a small sidecar that starts in a tenth of a second. Telling a dub's voices
from its music needs PyTorch, Demucs (voice separation, MIT) and Silero VAD
(speech detection, MIT) -- about a gigabyte -- so they live in their own
Python, installed on request into the user's data folder:

1. ``uv`` (Astral's installer, one static binary) is downloaded from its
   GitHub release, pinned to a version whose SHA-256 is checked here;
2. uv installs its own Python 3.11 and a virtual environment beside it;
3. PyTorch goes in -- the CPU build on Windows and Linux, which is all a
   server without a graphics card can use and a tenth of the size of the
   default Linux wheel -- then Demucs and Silero VAD, all pinned;
4. ``voiceworker.py`` is copied in and started once to fetch Demucs'
   weights, so the first sync afterwards does not wait on a download.

Nothing outside that folder is touched, and removing it uninstalls.

``VoiceWorker`` starts that Python on ``voiceworker.py`` and hands it audio
as ``.npy`` files. The engine does every read of a media file itself (see
``voicefix``), so the separated voices are cut from exactly the samples the
renderer writes -- which is what lets a voice be taken out of the mix again.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from typing import Callable, Optional

import numpy as np

from .media import CancellationToken, Cancelled, MediaError, _terminate

ProgressFn = Callable[[int, str], None]
LogFn = Callable[[str], None]

# Bumped whenever anything below changes, so an older install is redone.
TOOLS_VERSION = 1
UV_VERSION = "0.12.19"
UV_ASSETS = {
    ("windows", "x86_64"): ("uv-x86_64-pc-windows-msvc.zip",
                            "6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0"),
    ("windows", "arm64"): ("uv-aarch64-pc-windows-msvc.zip",
                           "115b54cb823bc48260670f5782001add6067ac8d98d18c8263a833704e287de9"),
    ("darwin", "arm64"): ("uv-aarch64-apple-darwin.tar.gz",
                          "a9a8df1eedeb192f2e47e40e2faabfb387db4b850209118786d42f89dde3e0ba"),
    ("darwin", "x86_64"): ("uv-x86_64-apple-darwin.tar.gz",
                           "cb5fa57bafe68fc0fb94b17f06bee0b0b9a7feb94ccbd110445afa0696e39273"),
    ("linux", "x86_64"): ("uv-x86_64-unknown-linux-gnu.tar.gz",
                          "23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8"),
    ("linux", "arm64"): ("uv-aarch64-unknown-linux-gnu.tar.gz",
                         "0804e9b164c64b6914182d5920c08551958a095986f10a3731056df701126436"),
}
PYTHON_VERSION = "3.11"
TORCH = "torch==2.14.0"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PACKAGES = ["demucs==4.1.0", "silero-vad==6.2.3", "numpy>=1.24,<3"]
MARKER = "installed.json"
WORKER_TIMEOUT_S = 6 * 3600


def _machine() -> tuple:
    system = platform.system().lower()
    system = {"windows": "windows", "darwin": "darwin", "linux": "linux"}.get(system, system)
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    return system, machine


def tools_dir() -> str:
    """Where the voice tools live: ``AUDIOSYNC_VOICE_TOOLS``, or the
    platform's per-user data folder."""
    override = os.environ.get("AUDIOSYNC_VOICE_TOOLS")
    if override:
        return override
    system, _ = _machine()
    if system == "windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif system == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "AudioSyncMaster", "voice-tools")


def _env_dir() -> str:
    return os.path.join(tools_dir(), "env")


def python_path() -> str:
    if os.name == "nt":
        return os.path.join(_env_dir(), "Scripts", "python.exe")
    return os.path.join(_env_dir(), "bin", "python")


def _uv_path() -> str:
    return os.path.join(tools_dir(), "uv", "uv.exe" if os.name == "nt" else "uv")


def worker_source() -> str:
    """The worker script shipped with the engine (a data file in a frozen build)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "voiceworker.py")]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "audiosync", "voiceworker.py"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise MediaError("The voice worker script is missing from this build of the engine")


def _worker_path() -> str:
    return os.path.join(tools_dir(), "voiceworker.py")


def _child_env() -> dict:
    """Environment for everything run inside the tools: caches and model
    downloads stay in the tools folder."""
    root = tools_dir()
    env = dict(os.environ)
    env.update({
        "UV_CACHE_DIR": os.path.join(root, "cache"),
        "UV_PYTHON_INSTALL_DIR": os.path.join(root, "python"),
        "UV_PYTHON_PREFERENCE": "only-managed",
        "HF_HOME": os.path.join(root, "models", "huggingface"),
        "TORCH_HOME": os.path.join(root, "models", "torch"),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
    })
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    return env


def _creation_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def status() -> dict:
    """What is installed, where, and whether it is current."""
    root = tools_dir()
    marker = os.path.join(root, MARKER)
    info: dict = {"installed": False, "dir": root, "toolsVersion": TOOLS_VERSION}
    if os.path.isfile(marker):
        try:
            with open(marker, encoding="utf-8") as handle:
                recorded = json.load(handle)
        except (OSError, ValueError):
            recorded = {}
        info.update({k: v for k, v in recorded.items() if k not in ("installed", "dir")})
        info["installed"] = (
            recorded.get("toolsVersion") == TOOLS_VERSION
            and os.path.isfile(python_path())
            and os.path.isfile(_worker_path())
        )
        if not info["installed"]:
            info["outdated"] = True
    return info


def installed() -> bool:
    return bool(status().get("installed"))


def _check(token: Optional[CancellationToken]) -> None:
    if token:
        token.raise_if_cancelled()


def _download(url: str, target: str, sha256: str, token: Optional[CancellationToken],
              on_fraction: Callable[[float], None]) -> None:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "AudioSyncMaster"})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            _check(token)
            block = response.read(1 << 16)
            if not block:
                break
            out.write(block)
            digest.update(block)
            done += len(block)
            if total:
                on_fraction(done / total)
    if digest.hexdigest() != sha256:
        raise MediaError(f"The download of {os.path.basename(url)} did not match its checksum; not installed")


def _run_logged(command: list, token: Optional[CancellationToken], log: LogFn, what: str,
                env: Optional[dict] = None, timeout: int = 3 * 3600) -> None:
    """Run an installer step, passing its output lines to the log."""
    _check(token)
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env or _child_env(),
            **_creation_kwargs(),
        )
    except OSError as exc:
        raise MediaError(f"Could not start {what}: {exc}") from exc
    if token:
        token.register(process)
    tail: list = []
    try:
        started = time.monotonic()
        assert process.stdout is not None
        for raw in iter(process.stdout.readline, b""):
            text = raw.decode("utf-8", errors="replace").rstrip()
            if text:
                tail = (tail + [text])[-20:]
                log(f"  {text}")
            if time.monotonic() - started > timeout:
                _terminate(process)
                raise MediaError(f"{what} took longer than {timeout // 60} minutes")
        process.wait()
        if token and token.cancelled:
            raise Cancelled("operation cancelled")
        if process.returncode != 0:
            raise MediaError(f"{what} failed: {tail[-1] if tail else f'exit code {process.returncode}'}")
    finally:
        if token:
            token.unregister(process)
        if process.poll() is None:
            _terminate(process)


def _extract_uv(archive: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    name = "uv.exe" if os.name == "nt" else "uv"
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            member = next(m for m in bundle.namelist() if os.path.basename(m) == name)
            with bundle.open(member) as source, open(os.path.join(target_dir, name), "wb") as out:
                shutil.copyfileobj(source, out)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            member = next(m for m in bundle.getmembers() if os.path.basename(m.name) == name)
            source = bundle.extractfile(member)
            assert source is not None
            with source, open(os.path.join(target_dir, name), "wb") as out:
                shutil.copyfileobj(source, out)
    if os.name != "nt":
        os.chmod(os.path.join(target_dir, name), 0o755)


def install(progress: Optional[ProgressFn] = None, log: Optional[LogFn] = None,
            token: Optional[CancellationToken] = None) -> dict:
    """Install (or bring up to date) the voice tools. Returns ``status()``."""
    say = log or (lambda _m: None)
    step = progress or (lambda _p, _s: None)
    system, machine = _machine()
    asset = UV_ASSETS.get((system, machine))
    if asset is None:
        raise MediaError(f"The voice tools are not available for {system} on {machine}")
    root = tools_dir()
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, MARKER)
    if os.path.isfile(marker):
        os.remove(marker)  # a half-finished reinstall must not look installed

    # 1. uv
    stage = "downloading the installer"
    step(0, stage)
    name, sha256 = asset
    if not os.path.isfile(_uv_path()):
        url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{name}"
        say(f"downloading {url}")
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            archive = os.path.join(scratch, name)
            _download(url, archive, sha256, token, lambda f: step(int(5 * f), stage))
            _extract_uv(archive, os.path.dirname(_uv_path()))
    uv = _uv_path()

    # 2. Python and the environment
    stage = "installing Python"
    step(5, stage)
    _run_logged([uv, "python", "install", PYTHON_VERSION], token, say, "installing Python")
    step(12, stage)
    if not os.path.isfile(python_path()):
        _run_logged([uv, "venv", "--python", PYTHON_VERSION, _env_dir()], token, say,
                    "creating the environment")

    # 3. PyTorch (CPU build on Windows/Linux), then Demucs and Silero VAD
    stage = "installing PyTorch (the largest download)"
    step(15, stage)
    torch_command = [uv, "pip", "install", "--python", python_path(), TORCH]
    if system in ("windows", "linux"):
        torch_command += ["--index-url", TORCH_CPU_INDEX]
    _run_logged(torch_command, token, say, "installing PyTorch")
    stage = "installing the voice separator and speech detector"
    step(60, stage)
    _run_logged([uv, "pip", "install", "--python", python_path(), *PACKAGES], token, say,
                "installing Demucs and Silero VAD")

    # 4. the worker, and the models' weights
    stage = "fetching the models"
    step(80, stage)
    shutil.copyfile(worker_source(), _worker_path())
    with VoiceWorker(token=token, log=say) as worker:
        hello = worker.request({"cmd": "hello"})
        say(f"voice tools: PyTorch {hello.get('torch')}, Demucs {hello.get('demucs')}, "
            f"running on {hello.get('device')} with {hello.get('threads')} threads")
        worker.request({"cmd": "warmup"}, timeout=1800)
    shutil.rmtree(os.path.join(root, "cache"), ignore_errors=True)
    record = {
        "toolsVersion": TOOLS_VERSION, "uv": UV_VERSION, "python": PYTHON_VERSION,
        "packages": [TORCH, *PACKAGES], "device": hello.get("device"),
        "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    step(100, "installed")
    return status()


def remove() -> dict:
    root = tools_dir()
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    return status()


def size_bytes() -> int:
    total = 0
    for folder, _dirs, files in os.walk(tools_dir()):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(folder, name))
            except OSError:
                pass
    return total


class VoiceWorker:
    """The tools' Python running ``voiceworker.py``, one request at a time.

    Only one runs at once across the whole engine (a batch syncs several
    pairs in parallel, and every worker holds a separation model in
    memory); the others wait their turn.
    """

    _gate = threading.Lock()

    def __init__(self, token: Optional[CancellationToken] = None, log: Optional[LogFn] = None) -> None:
        self.token = token
        self.log = log or (lambda _m: None)
        self.process: Optional[subprocess.Popen] = None
        self._scratch: Optional[str] = None
        self._held = False

    def __enter__(self) -> "VoiceWorker":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def start(self) -> None:
        if not os.path.isfile(python_path()) or not os.path.isfile(_worker_path()):
            raise MediaError("The voice tools are not installed")
        while not VoiceWorker._gate.acquire(timeout=0.5):
            _check(self.token)
        self._held = True
        os.makedirs(tools_dir(), exist_ok=True)
        self._scratch = tempfile.mkdtemp(prefix="work-", dir=tools_dir())
        self._stderr = open(os.path.join(tools_dir(), "worker.log"), "ab")
        self.process = subprocess.Popen(
            [python_path(), _worker_path()], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, env=_child_env(), **_creation_kwargs(),
        )
        if self.token:
            self.token.register(self.process)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            try:
                if process.poll() is None and process.stdin:
                    process.stdin.write(b'{"cmd": "quit"}\n')
                    process.stdin.flush()
                    process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                _terminate(process)
            if self.token:
                self.token.unregister(process)
        if getattr(self, "_stderr", None):
            self._stderr.close()
            self._stderr = None
        if self._scratch:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
        if self._held:
            VoiceWorker._gate.release()
            self._held = False

    def request(self, payload: dict, timeout: int = WORKER_TIMEOUT_S) -> dict:
        _check(self.token)
        process = self.process
        if process is None or process.poll() is not None:
            raise MediaError("The voice worker is not running (see worker.log in the voice tools folder)")
        assert process.stdin is not None and process.stdout is not None
        # A request that never answers (a wedged model, a full disk) must
        # not hold the engine forever: past the timeout the worker is killed
        # and the read below returns empty.
        watchdog = threading.Timer(timeout, _terminate, args=(process,))
        watchdog.daemon = True
        watchdog.start()
        try:
            process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            process.stdin.flush()
            line = process.stdout.readline()
        except OSError as exc:
            raise MediaError(f"The voice worker stopped: {exc}") from exc
        finally:
            watchdog.cancel()
        if self.token and self.token.cancelled:
            raise Cancelled("operation cancelled")
        if not line:
            raise MediaError("The voice worker stopped (see worker.log in the voice tools folder)")
        reply = json.loads(line.decode("utf-8"))
        if not reply.get("ok"):
            raise MediaError(f"The voice worker failed: {reply.get('error')}")
        return reply

    def _roundtrip(self, command: str, audio: np.ndarray, **extra) -> np.ndarray:
        assert self._scratch is not None
        source = os.path.join(self._scratch, "in.npy")
        target = os.path.join(self._scratch, "out.npy")
        np.save(source, np.ascontiguousarray(audio, dtype=np.float32))
        self.request({"cmd": command, "input": source, "output": target, **extra})
        result = np.load(target)
        for path in (source, target):
            try:
                os.remove(path)
            except OSError:
                pass
        return result

    def speech(self, audio_16k: np.ndarray) -> np.ndarray:
        """Silero's speech probability of mono 16 kHz audio (see voiceworker)."""
        return self._roundtrip("speech", audio_16k)

    def vocals(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """The voices of ``(n, channels)`` audio at ``rate``, same shape and clock."""
        return self._roundtrip("vocals", audio, rate=int(rate))
