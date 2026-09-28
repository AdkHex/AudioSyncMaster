"""Optional engine packs: install, report, remove, and run their workers.

The shipped engine is a small PyInstaller build with numpy only. Speech
recognition (mlx-whisper, faster-whisper), Demucs, RapidOCR and a
full-featured FFmpeg are far too large -- and too platform-specific -- to
ship to every user, so each is a *pack* the app installs on demand into the
user data folder:

    <packs>/
      bin/uv                 uv, when it is not on PATH
      python/                Python 3.11 builds uv manages (shared by packs)
      asr-mlx/venv, pack.json
      asr-faster/venv, pack.json
      ffmpeg-full/bin/ffmpeg, pack.json

A Python pack is a virtual environment of its own. Its code runs as a
*worker* script (``workers/*.py``) in that environment's interpreter, talking
JSON lines over stdin/stdout, so nothing heavy is ever imported into the
engine and a crash in a native library costs one job, not the engine.

Installs are staged: everything is built in ``<packs>/.staging-*`` and only
renamed into place after the pack's imports have been checked and its
``pack.json`` marker written. A failed or cancelled install therefore never
looks installed, and a reinstall keeps the old pack until the new one works.

Model weights live in the standard Hugging Face cache, so a model the user
already downloaded for another tool is found and reused, and removing a pack
does not throw away gigabytes of weights another pack or tool still uses.
"""

from __future__ import annotations

import collections
import datetime as _dt
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..media import Cancelled, CancellationToken, _terminate
from .tasks import TaskError

#: uv release used when uv is not on PATH. Pinned so an install is
#: reproducible; the archive is checked against the release's own .sha256.
UV_VERSION = "0.12.19"
PYTHON_VERSION = "3.11"

ProgressFn = Callable[..., None]
LogFn = Callable[[str], None]


# ------------------------------------------------------------------ registry


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    #: Hugging Face repo id; the weights land in the standard HF cache.
    repo: str
    #: The file whose presence means the download finished.
    weight_file: str
    download_bytes: int
    #: Repo files this model needs, when the repo holds several models
    #: (glob patterns, as huggingface_hub's allow_patterns); () = all.
    files: Tuple[str, ...] = ()


@dataclass(frozen=True)
class InstallStep:
    """One ``uv pip install`` call.

    Most packs need one. Steps exist for the exceptions: mlx-whisper declares
    torch as a dependency although it only uses it to convert checkpoints,
    so it is installed with ``--no-deps`` after its real dependencies (saving
    ~350 MB); and on Linux the default PyPI torch drags in ~3 GB of CUDA
    libraries, so the Demucs pack takes the CPU build from PyTorch's index
    unless an NVIDIA driver is present.
    """

    packages: Tuple[str, ...]
    no_deps: bool = False
    index_url: Optional[str] = None
    #: Only on these ("macos"|"windows"|"linux", arch) pairs; None = all.
    only_on: Optional[Tuple[Tuple[str, str], ...]] = None
    #: Only when an NVIDIA GPU driver is (True) / is not (False) present.
    nvidia: Optional[bool] = None


@dataclass(frozen=True)
class PackSpec:
    id: str
    label: str
    description: str
    #: "python" (a uv venv running workers) or "ffmpeg" (a portable build).
    kind: str
    #: Bump when the pins change; status shows an older marker as outdated.
    version: str
    #: Supported (platform, arch) pairs; None = every desktop platform.
    platforms: Optional[Tuple[Tuple[str, str], ...]] = None
    steps: Tuple[InstallStep, ...] = ()
    #: Python run after installing to prove the pack actually imports.
    import_check: str = ""
    #: Approximate bytes downloaded / used on disk, for the install dialog.
    #: Measured on macOS arm64 (wheels + the first managed Python); PyTorch
    #: is larger on Windows and Linux.
    download_bytes: Optional[int] = None
    size_bytes: Optional[int] = None
    models: Tuple[ModelSpec, ...] = ()


MB = 1_000_000

_MLX_MODELS = (
    ModelSpec("tiny", "Tiny", "mlx-community/whisper-tiny-mlx", "weights.npz", 74 * MB),
    ModelSpec("base", "Base", "mlx-community/whisper-base-mlx", "weights.npz", 144 * MB),
    ModelSpec("small", "Small", "mlx-community/whisper-small-mlx", "weights.npz", 481 * MB),
    ModelSpec("medium", "Medium", "mlx-community/whisper-medium-mlx", "weights.npz", 1525 * MB),
    ModelSpec("large-v2", "Large v2", "mlx-community/whisper-large-v2-mlx", "weights.npz", 3083 * MB),
    ModelSpec("large-v3", "Large v3", "mlx-community/whisper-large-v3-mlx", "weights.npz", 3084 * MB),
    ModelSpec("large-v3-turbo", "Large v3 Turbo", "mlx-community/whisper-large-v3-turbo", "weights.safetensors", 1614 * MB),
)

# The turbo conversion faster-whisper itself names (mobiuslabsgmbh/...) now
# redirects here; using the canonical id keeps one copy in the HF cache.
_CT2_MODELS = (
    ModelSpec("tiny", "Tiny", "Systran/faster-whisper-tiny", "model.bin", 78 * MB),
    ModelSpec("base", "Base", "Systran/faster-whisper-base", "model.bin", 148 * MB),
    ModelSpec("small", "Small", "Systran/faster-whisper-small", "model.bin", 486 * MB),
    ModelSpec("medium", "Medium", "Systran/faster-whisper-medium", "model.bin", 1531 * MB),
    ModelSpec("large-v2", "Large v2", "Systran/faster-whisper-large-v2", "model.bin", 3090 * MB),
    ModelSpec("large-v3", "Large v3", "Systran/faster-whisper-large-v3", "model.bin", 3091 * MB),
    ModelSpec("large-v3-turbo", "Large v3 Turbo", "dropbox-dash/faster-whisper-large-v3-turbo", "model.bin", 1622 * MB),
)

# PP-OCRv5 recognition models converted to ONNX (Apache-2.0). The built-in
# PP-OCRv4 model knows Chinese and English only: it read 今日はいい天気ですね
# as "今日" and "hôpital" as "hopital". Measured on rendered subtitle lines
# with rapidocr_onnxruntime 1.4.4: the CJK model read Japanese (kana and
# kanji), Chinese and English exactly; Latin read French, German and Spanish
# accents (it drops the ligature œ); Cyrillic read Russian.
_OCR_REPO = "monkt/paddleocr-onnx"


def _ocr_model(model_id: str, label: str, group: str, size: int) -> ModelSpec:
    return ModelSpec(model_id, label, _OCR_REPO, f"languages/{group}/rec.onnx", size,
                     files=(f"languages/{group}/rec.onnx", f"languages/{group}/dict.txt",
                            f"languages/{group}/config.json"))


_OCR_MODELS = (
    _ocr_model("cjk", "Japanese + Chinese (PP-OCRv5 server)", "chinese", 85 * MB),
    _ocr_model("korean", "Korean (PP-OCRv5)", "korean", 13 * MB),
    _ocr_model("latin", "Accented Latin: French, German, Spanish... (PP-OCRv5)", "latin", 8 * MB),
    _ocr_model("cyrillic", "Cyrillic: Russian, Ukrainian, Bulgarian (PP-OCRv5)", "eslav", 8 * MB),
)

#: Language (ISO 639-1) -> the OCR model that reads it. English and
#: simplified Chinese are read by the built-in model.
OCR_MODEL_FOR_LANGUAGE: Dict[str, str] = {
    "ja": "cjk", "zh": "cjk", "yue": "cjk",
    "ko": "korean",
    **{code: "latin" for code in (
        "fr", "de", "es", "it", "pt", "nl", "ca", "sv", "da", "no", "nn", "fi", "is", "pl", "cs", "sk",
        "sl", "hr", "bs", "ro", "hu", "et", "lv", "lt", "tr", "az", "id", "ms", "tl", "vi", "af", "sq",
        "eu", "gl", "oc", "br", "cy", "lb", "mt", "fo", "sw", "la", "uz", "haw", "mi", "ht")},
    **{code: "cyrillic" for code in ("ru", "uk", "bg", "be", "mk", "sr", "kk", "mn", "tg", "tt", "ba")},
}

_MAC_ARM = (("macos", "arm64"),)

PACKS: Dict[str, PackSpec] = {
    "asr-mlx": PackSpec(
        id="asr-mlx",
        label="Speech recognition (Apple Silicon)",
        description=(
            "Whisper on the Apple Silicon GPU through MLX -- the fastest way to "
            "make subtitles on a Mac. Models are downloaded separately."
        ),
        kind="python",
        version="1",
        platforms=_MAC_ARM,
        steps=(
            InstallStep((
                "mlx==0.32.2", "numba==0.67.0", "scipy==1.17.1", "tiktoken==0.14.0",
                "huggingface-hub==1.33.0", "tqdm", "more-itertools", "numpy<2.5",
            )),
            InstallStep(("mlx-whisper==0.4.3",), no_deps=True),
        ),
        import_check="import mlx.core, mlx_whisper, mlx_whisper.timing, huggingface_hub",
        download_bytes=160 * MB,
        size_bytes=545 * MB,
        models=_MLX_MODELS,
    ),
    "asr-faster": PackSpec(
        id="asr-faster",
        label="Speech recognition (any computer)",
        description=(
            "Whisper through CTranslate2 (faster-whisper) on the CPU, or an NVIDIA "
            "GPU where one is set up. Also provides the Silero voice-activity "
            "detector. Models are downloaded separately."
        ),
        kind="python",
        version="1",
        steps=(InstallStep((
            "faster-whisper==1.2.1", "ctranslate2==4.8.2", "onnxruntime==1.30.0",
            "tokenizers==0.23.2", "huggingface-hub==1.33.0",
        )),),
        import_check="import faster_whisper, ctranslate2, onnxruntime; from faster_whisper.vad import get_vad_model",
        download_bytes=90 * MB,
        size_bytes=210 * MB,
        models=_CT2_MODELS,
    ),
    "demucs": PackSpec(
        id="demucs",
        label="Voice isolation (Demucs)",
        description=(
            "Separates voices from music and effects before recognition, which "
            "helps with songs and loud scores. Large: it installs PyTorch "
            "(roughly 1 GB on disk) and an 84 MB model; slow without a GPU."
        ),
        kind="python",
        version="1",
        steps=(
            InstallStep(("torch==2.14.0",), index_url="https://download.pytorch.org/whl/cpu",
                        only_on=(("linux", "x86_64"), ("linux", "arm64")), nvidia=False),
            # demucs 4.1.0 imports numpy but declares it only for Intel Macs.
            InstallStep(("torch==2.14.0", "demucs==4.1.0", "numpy<2.5")),
        ),
        import_check="import torch, demucs.pretrained, demucs.apply, julius",
        download_bytes=280 * MB,
        size_bytes=900 * MB,
        # demucs 4.1 loads its models from the Hugging Face hub.
        models=(ModelSpec("htdemucs", "HTDemucs", "adefossez/HTDemucs", "955717e8.safetensors", 84 * MB),),
    ),
    "ocr-rapidocr": PackSpec(
        id="ocr-rapidocr",
        label="OCR (RapidOCR)",
        description=(
            "PaddleOCR's text recognition on ONNX Runtime, for turning Blu-ray and DVD "
            "image subtitles into text. English and simplified Chinese work out of the "
            "box; Japanese, Korean, accented Latin and Cyrillic each need their "
            "recognition model (8-84 MB)."
        ),
        kind="python",
        version="2",
        # huggingface_hub fetches the per-language recognition models.
        steps=(InstallStep(("rapidocr_onnxruntime==1.4.4", "onnxruntime==1.30.0", "huggingface-hub==1.33.0")),),
        import_check="import rapidocr_onnxruntime, onnxruntime, cv2, huggingface_hub",
        download_bytes=130 * MB,
        size_bytes=290 * MB,
        models=_OCR_MODELS,
    ),
    "ffmpeg-full": PackSpec(
        id="ffmpeg-full",
        label="FFmpeg (full)",
        description=(
            "A portable FFmpeg (Jellyfin build) with zscale and libass on every "
            "platform, and libplacebo on Windows and Linux, for HDR tone-mapping and "
            "subtitle burn-in that a typical FFmpeg lacks."
        ),
        kind="ffmpeg",
        version="1",
        download_bytes=60 * MB,
        size_bytes=150 * MB,
    ),
}

#: jellyfin-ffmpeg portable asset suffix per platform. Checked against the
#: v8.1.2-5 release; the asset is resolved from the latest release at
#: install time so a new FFmpeg does not need an app update.
_JELLYFIN_ASSETS = {
    ("macos", "arm64"): "_portable_macarm64-gpl.tar.xz",
    ("macos", "x86_64"): "_portable_mac64-gpl.tar.xz",
    ("linux", "x86_64"): "_portable_linux64-gpl.tar.xz",
    ("linux", "arm64"): "_portable_linuxarm64-gpl.tar.xz",
    ("windows", "x86_64"): "_portable_win64-clang-gpl.zip",
    ("windows", "arm64"): "_portable_winarm64-clang-gpl.zip",
}
_JELLYFIN_API = "https://api.github.com/repos/jellyfin/jellyfin-ffmpeg/releases/latest"


# ------------------------------------------------------------------- places


def host() -> Tuple[str, str]:
    """("macos"|"windows"|"linux", "arm64"|"x86_64"|...) of the *hardware*.

    An x86_64 engine running under Rosetta reports x86_64, but the packs it
    installs run natively, so on macOS the CPU is asked directly.
    """
    system = {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux")
    machine = platform.machine().lower()
    arch = {"aarch64": "arm64", "arm64": "arm64", "amd64": "x86_64", "x86_64": "x86_64"}.get(machine, machine)
    if system == "macos" and arch == "x86_64":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=5)
            if out.stdout.strip() == "1":
                arch = "arm64"
        except (OSError, subprocess.SubprocessError):
            pass
    return system, arch


def packs_dir() -> str:
    override = os.environ.get("AUDIOSYNC_PACKS_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    system, _ = host()
    if system == "macos":
        base = os.path.expanduser("~/Library/Application Support")
    elif system == "windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "AudioSyncMaster", "packs")


def pack_dir(pack_id: str) -> str:
    return os.path.join(packs_dir(), pack_id)


def workers_dir() -> str:
    """Folder of the worker scripts.

    Next to this module in a source checkout, and in the PyInstaller onedir
    build too, provided the folder is added as data at the same relative
    path (``audiosync/subs/workers``).
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers")


def _venv_python(venv: str) -> str:
    if os.name == "nt":
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python")


def _override_python(pack_id: str) -> Optional[str]:
    """``AUDIOSYNC_PACK_PYTHON_ASR_MLX=/path/to/python`` adopts an existing
    environment (a developer's, or one set up by hand) as the pack."""
    raw = os.environ.get("AUDIOSYNC_PACK_PYTHON_" + re.sub(r"[^A-Z0-9]", "_", pack_id.upper()))
    if raw:
        path = os.path.expanduser(raw)
        if os.path.isfile(path):
            return path
    return None


def _marker(pack_id: str) -> Optional[dict]:
    try:
        with open(os.path.join(pack_dir(pack_id), "pack.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def python_for(pack_id: str) -> Optional[str]:
    """The interpreter of an installed Python pack, or None."""
    override = _override_python(pack_id)
    if override:
        return override
    spec = PACKS.get(pack_id)
    if not spec or spec.kind != "python" or _marker(pack_id) is None:
        return None
    python = _venv_python(os.path.join(pack_dir(pack_id), "venv"))
    return python if os.path.isfile(python) else None


def ffmpeg_full() -> Optional[str]:
    """The ffmpeg of the ffmpeg-full pack (or ``AUDIOSYNC_FFMPEG_FULL``)."""
    override = os.environ.get("AUDIOSYNC_FFMPEG_FULL")
    if override and os.path.isfile(os.path.expanduser(override)):
        return os.path.expanduser(override)
    if _marker("ffmpeg-full") is None:
        return None
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    path = os.path.join(pack_dir("ffmpeg-full"), "bin", exe)
    return path if os.path.isfile(path) else None


def supported(spec: PackSpec) -> bool:
    return spec.platforms is None or host() in spec.platforms


def _is_installed(spec: PackSpec) -> bool:
    if spec.kind == "ffmpeg":
        return ffmpeg_full() is not None
    return python_for(spec.id) is not None


# ------------------------------------------------------- Hugging Face cache


def hf_cache_dir() -> str:
    """The Hugging Face hub cache, resolved exactly as huggingface_hub does,
    so a model downloaded by any other tool counts as installed."""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(key):
            return os.path.expanduser(os.environ[key])
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "huggingface"
    )
    return os.path.join(os.path.expanduser(home), "hub")


def _repo_cache(repo: str) -> str:
    return os.path.join(hf_cache_dir(), "models--" + repo.replace("/", "--"))


def model_path(model: ModelSpec) -> Optional[str]:
    """The snapshot folder holding ``model``'s weights, or None if absent."""
    snapshots = os.path.join(_repo_cache(model.repo), "snapshots")
    try:
        names = sorted(os.listdir(snapshots))
    except OSError:
        return None
    for name in names:
        folder = os.path.join(snapshots, name)
        # exists() follows the snapshot symlink, so a dangling link to a
        # half-downloaded blob does not count.
        if os.path.exists(os.path.join(folder, model.weight_file)):
            return folder
    return None


def ocr_recognizer(language: Optional[str]) -> Optional[Dict[str, str]]:
    """Which OCR recognition model reads ``language``.

    None: the built-in model is the right one (English, simplified Chinese
    without the CJK model, or no language). ``{"needed": id}``: the
    language needs model ``id`` of the ocr-rapidocr pack, not downloaded
    yet. Otherwise ``{"model", "recModel", "recKeys"}``: the id and the
    paths the RapidOCR worker takes.
    """
    from .languages import normalize

    code = normalize(language) if language else None
    model_id = OCR_MODEL_FOR_LANGUAGE.get(code or "")
    if not model_id:
        return None
    spec = model_spec("ocr-rapidocr", model_id)
    folder = model_path(spec)
    if not folder:
        return {"needed": model_id}
    base = os.path.dirname(os.path.join(folder, spec.weight_file))
    return {"model": model_id, "recModel": os.path.join(base, "rec.onnx"), "recKeys": os.path.join(base, "dict.txt")}


def model_spec(pack_id: str, model_id: str) -> ModelSpec:
    spec = _spec(pack_id)
    for model in spec.models:
        if model.id == model_id:
            return model
    raise TaskError(f"{spec.label} has no model named {model_id!r}.")


def _spec(pack_id: str) -> PackSpec:
    spec = PACKS.get(pack_id)
    if not spec:
        raise TaskError(f"Unknown engine pack: {pack_id}")
    return spec


# ------------------------------------------------------------------- status


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def pack_status(pack_id: str) -> Dict[str, Any]:
    spec = _spec(pack_id)
    marker = _marker(pack_id)
    external = spec.kind == "python" and _override_python(pack_id) is not None
    if spec.kind == "ffmpeg" and os.environ.get("AUDIOSYNC_FFMPEG_FULL") and ffmpeg_full():
        external = True
    installed = _is_installed(spec)
    out: Dict[str, Any] = {
        "id": spec.id,
        "label": spec.label,
        "description": spec.description,
        "installed": installed,
        "supported": supported(spec),
        "sizeBytes": (marker or {}).get("sizeBytes") if installed and not external else None,
        "downloadBytes": spec.download_bytes,
        "version": "external" if external else ((marker or {}).get("version") if installed else None),
    }
    if external:
        out["external"] = True
    if marker and not external and str(marker.get("version")) != spec.version:
        out["outdated"] = True
    if spec.models:
        out["models"] = [
            {"id": m.id, "label": m.label, "installed": model_path(m) is not None, "downloadBytes": m.download_bytes}
            for m in spec.models
        ]
    return out


def status() -> List[Dict[str, Any]]:
    """Every pack as a ``PackStatus`` dict (see types.ts)."""
    return [pack_status(pack_id) for pack_id in PACKS]


# --------------------------------------------------------------- reporting


def _progress_adapter(progress: Optional[ProgressFn]) -> Callable[[float, str, Optional[int], Optional[int]], None]:
    """Call ``progress(percent, stage)`` or, when the callback takes them,
    ``progress(percent, stage, bytes, totalBytes)`` -- the packProgress event
    carries byte counts, a job's stage progress does not."""
    if progress is None:
        return lambda *_a: None
    try:
        params = inspect.signature(progress).parameters.values()
        wants_bytes = any(p.kind == p.VAR_POSITIONAL for p in params) or len(list(params)) >= 4
    except (TypeError, ValueError):
        wants_bytes = False

    def emit(percent: float, stage: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
        pct = int(max(0, min(100, percent)))
        if wants_bytes:
            progress(pct, stage, done, total)
        else:
            progress(pct, stage)

    return emit


# ---------------------------------------------------------------- downloads


def _curl() -> Optional[str]:
    return shutil.which("curl")


def _fetch_json(url: str, token: Optional[CancellationToken]) -> Any:
    raw = _fetch_bytes(url, token)
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise TaskError(f"Unexpected reply from {url.split('/')[2]}: {exc}") from exc


def _fetch_bytes(url: str, token: Optional[CancellationToken]) -> bytes:
    with tempfile.TemporaryDirectory(prefix="audiosync-fetch-") as tmp:
        dest = os.path.join(tmp, "body")
        download(url, dest, token=token)
        with open(dest, "rb") as handle:
            return handle.read()


def download(
    url: str,
    dest: str,
    expected_bytes: Optional[int] = None,
    on_bytes: Optional[Callable[[int, Optional[int]], None]] = None,
    token: Optional[CancellationToken] = None,
) -> None:
    """Download ``url`` to ``dest`` with byte progress and cancellation.

    curl comes first because it uses the operating system's certificate
    store. The frozen engine's Python carries an OpenSSL whose default CA
    path points into the build machine's Python install, so urllib fails
    certificate checks on most users' Macs; it is only the fallback for a
    machine without curl.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    curl = _curl()
    if curl:
        _download_curl(curl, url, dest, expected_bytes, on_bytes, token)
    else:
        _download_urllib(url, dest, expected_bytes, on_bytes, token)


def _download_curl(curl, url, dest, expected_bytes, on_bytes, token) -> None:
    if token:
        token.raise_if_cancelled()
    command = [curl, "-fsSL", "--retry", "3", "--connect-timeout", "30",
               "-H", "User-Agent: AudioSyncMaster", "-o", dest, url]
    kwargs: Dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    if token:
        token.register(process)
    try:
        while process.poll() is None:
            time.sleep(0.5)
            if on_bytes:
                try:
                    on_bytes(os.path.getsize(dest), expected_bytes)
                except OSError:
                    pass
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    finally:
        if token:
            token.unregister(process)
        _terminate(process)
    if token and token.cancelled:
        raise Cancelled("operation cancelled")
    if process.returncode != 0:
        raise TaskError(f"Download failed ({url.split('/')[2]}): {stderr.strip() or f'curl exit {process.returncode}'}")
    if on_bytes:
        on_bytes(os.path.getsize(dest), expected_bytes)


def _download_urllib(url, dest, expected_bytes, on_bytes, token) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "AudioSyncMaster"})
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except (OSError, ssl.SSLError) as exc:
        raise TaskError(f"Download failed ({url.split('/')[2]}): {exc}") from exc
    total = expected_bytes or int(response.headers.get("Content-Length") or 0) or None
    done = 0
    with response, open(dest, "wb") as handle:
        while True:
            if token and token.cancelled:
                raise Cancelled("operation cancelled")
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if on_bytes:
                on_bytes(done, total)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: str, dest: str) -> None:
    """Unpack a .zip / .tar.* refusing members that escape ``dest``."""
    root = os.path.realpath(dest)

    def safe(name: str) -> str:
        target = os.path.realpath(os.path.join(dest, name))
        if target != root and not target.startswith(root + os.sep):
            raise TaskError(f"Refusing unsafe path in archive: {name}")
        return target

    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                safe(member)
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            # Links are dropped: nothing a pack needs is one, and a link is
            # the classic way an archive writes outside its folder.
            members = [m for m in tf.getmembers() if m.isfile() or m.isdir()]
            for member in members:
                safe(member.name)
            if hasattr(tarfile, "data_filter"):
                tf.extractall(dest, members=members, filter="data")
            else:
                tf.extractall(dest, members=members)


# ----------------------------------------------------------------------- uv


def _uv_target() -> Tuple[str, str]:
    system, arch = host()
    cpu = {"arm64": "aarch64", "x86_64": "x86_64"}.get(arch, arch)
    if system == "macos":
        return f"{cpu}-apple-darwin", ".tar.gz"
    if system == "windows":
        return f"{cpu}-pc-windows-msvc", ".zip"
    libc = "gnu" if platform.libc_ver()[0] == "glibc" else "musl"
    return f"{cpu}-unknown-linux-{libc}", ".tar.gz"


def _packs_uv() -> str:
    return os.path.join(packs_dir(), "bin", "uv.exe" if os.name == "nt" else "uv")


def find_uv(allow_path: bool = True) -> Optional[str]:
    override = os.environ.get("AUDIOSYNC_UV")
    if override and os.path.isfile(override):
        return override
    if allow_path:
        found = shutil.which("uv")
        if found:
            return found
    local = _packs_uv()
    return local if os.path.isfile(local) else None


def ensure_uv(progress: ProgressFn, log: LogFn, token: Optional[CancellationToken], allow_path: bool = True) -> str:
    """uv from PATH, else the copy in ``packs/bin``, else download it."""
    found = find_uv(allow_path)
    if found:
        return found
    target, ext = _uv_target()
    name = f"uv-{target}{ext}"
    base = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{name}"
    log(f"Downloading uv {UV_VERSION} ({target})")
    emit = _progress_adapter(progress)
    with tempfile.TemporaryDirectory(prefix="audiosync-uv-", dir=_ensure_dir(packs_dir())) as tmp:
        archive = os.path.join(tmp, name)
        download(base, archive, token=token,
                 on_bytes=lambda d, t: emit(min(99, 100 * d / (t or 20 * MB)), "Downloading uv", d, t))
        expected = _fetch_bytes(base + ".sha256", token).decode("utf-8", "replace").split()[0].lower()
        actual = _sha256(archive)
        if actual != expected:
            raise TaskError(f"uv download is corrupt (sha256 {actual[:12]}... expected {expected[:12]}...)")
        unpacked = os.path.join(tmp, "unpacked")
        _extract(archive, unpacked)
        exe = "uv.exe" if os.name == "nt" else "uv"
        found = next((os.path.join(r, exe) for r, _d, f in os.walk(unpacked) if exe in f), None)
        if not found:
            raise TaskError("The uv archive did not contain uv")
        dest = _packs_uv()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(found, dest)
        if os.name != "nt":
            os.chmod(dest, 0o755)
    log(f"uv installed at {dest}")
    return dest


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def _uv_env() -> Dict[str, str]:
    """uv keeps its Pythons and cache inside the packs folder.

    Only uv-managed Pythons are used, so a pack never breaks because the user
    upgraded or removed a Homebrew/system Python; and deleting the packs
    folder removes everything the packs ever put on disk.
    """
    root = packs_dir()
    env = _clean_env()
    env.update({
        "UV_PYTHON_INSTALL_DIR": os.path.join(root, "python"),
        "UV_CACHE_DIR": os.path.join(root, "cache"),
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_PYTHON_INSTALL_BIN": "0",
        "UV_NO_PROGRESS": "1",
        "NO_COLOR": "1",
    })
    return env


def _clean_env() -> Dict[str, str]:
    """The environment for pack processes, minus what the frozen engine set.

    PyInstaller points PYTHONHOME/PYTHONPATH-like variables and (on Linux)
    LD_LIBRARY_PATH at its own bundle; a pack's Python inheriting them would
    load the engine's libraries instead of its own.
    """
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("PYTHON", "_PYI", "_MEIPASS")) or key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            env.pop(key, None)
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        original = env.pop(key + "_ORIG", None)
        if getattr(sys, "frozen", False):
            env.pop(key, None)
            if original:
                env[key] = original
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    # Signed app bundles must not gain __pycache__ folders at run time.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_streaming(
    command: List[str],
    env: Dict[str, str],
    token: Optional[CancellationToken],
    on_line: Callable[[str], None],
    cwd: Optional[str] = None,
    tick: Optional[Callable[[], None]] = None,
) -> Tuple[int, List[str]]:
    """Run a command, handing each output line (stdout+stderr) to ``on_line``
    and calling ``tick`` about once a second. Returns (exit code, last lines)."""
    if token:
        token.raise_if_cancelled()
    kwargs: Dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "env": env, "cwd": cwd}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise TaskError(f"Could not start {os.path.basename(command[0])}: {exc}") from exc
    if token:
        token.register(process)
    tail: collections.deque = collections.deque(maxlen=30)

    def reader() -> None:
        assert process.stdout is not None
        for raw in iter(process.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                tail.append(line)
                on_line(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(1.0)
            if tick:
                tick()
        process.wait()
    finally:
        if token:
            token.unregister(process)
        _terminate(process)
    if token and token.cancelled:
        raise Cancelled("operation cancelled")
    return process.returncode, list(tail)


# ------------------------------------------------------------------ install


def install(
    pack_id: str,
    progress: Optional[ProgressFn],
    log: Optional[LogFn],
    token: Optional[CancellationToken],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Install a pack, or with ``model`` download one of its models.

    ``progress(percent, stage)`` -- or ``progress(percent, stage, bytes,
    totalBytes)`` when the callback accepts four arguments. Returns the
    pack's new status. Raises TaskError (shown to the user) or Cancelled.
    """
    spec = _spec(pack_id)
    log = log or (lambda _m: None)
    token = token or CancellationToken()
    if model is not None:
        _download_model(spec, model_spec(pack_id, model), progress, log, token)
        return pack_status(pack_id)
    if not supported(spec):
        system, arch = host()
        raise TaskError(f"{spec.label} is not available on {system} {arch}.")
    if spec.kind == "ffmpeg":
        _install_ffmpeg(spec, progress, log, token)
    else:
        _install_python(spec, progress, log, token)
    return pack_status(pack_id)


def _stage_dir(pack_id: str) -> str:
    root = _ensure_dir(packs_dir())
    for name in os.listdir(root):
        # Leftovers of an install that was killed with the engine.
        if name.startswith(f".staging-{pack_id}-") or name.startswith(f".old-{pack_id}-"):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    path = os.path.join(root, f".staging-{pack_id}-{uuid.uuid4().hex[:8]}")
    os.makedirs(path)
    return path


def _commit(stage: str, pack_id: str, marker: Dict[str, Any]) -> None:
    """Write the marker, then swap the staged pack into place."""
    marker = dict(marker)
    marker["sizeBytes"] = _dir_size(stage)
    with open(os.path.join(stage, "pack.json"), "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2)
    final = pack_dir(pack_id)
    old = None
    if os.path.exists(final):
        old = os.path.join(packs_dir(), f".old-{pack_id}-{uuid.uuid4().hex[:8]}")
        os.replace(final, old)
    os.replace(stage, final)
    if old:
        shutil.rmtree(old, ignore_errors=True)


def _install_python(spec: PackSpec, progress: Optional[ProgressFn], log: LogFn, token: CancellationToken) -> None:
    emit = _progress_adapter(progress)
    emit(0, "Preparing")
    uv = ensure_uv(lambda p, s, *a: emit(p * 0.05, s), log, token)
    env = _uv_env()
    stage = _stage_dir(spec.id)
    venv = os.path.join(stage, "venv")
    cache = env["UV_CACHE_DIR"]
    try:
        emit(5, "Setting up Python")
        code, tail = _run_streaming(
            [uv, "venv", "--python", PYTHON_VERSION, "--relocatable", "--no-project", venv],
            env, token, log, cwd=stage,
        )
        if code != 0:
            raise TaskError(f"Could not create the Python environment: {tail[-1] if tail else code}")
        python = _venv_python(venv)

        system, arch = host()
        steps = [
            s for s in spec.steps
            if (s.only_on is None or (system, arch) in s.only_on)
            and (s.nvidia is None or s.nvidia == _has_nvidia())
        ]
        # Coarse progress while uv downloads silently: the cache grows by
        # roughly the unpacked size of the pack, so its growth against the
        # expected size is an honest measure of how far along it is.
        base_size = _dir_size(cache) if os.path.isdir(cache) else 0
        expected = spec.size_bytes or 300 * MB
        state = {"step": 0}

        def tick() -> None:
            grown = max(0, (_dir_size(cache) if os.path.isdir(cache) else 0) - base_size)
            frac = min(0.97, grown / expected)
            emit(15 + 75 * frac, f"Installing packages ({state['step'] + 1}/{len(steps)})", grown, expected)

        for i, step in enumerate(steps):
            state["step"] = i
            command = [uv, "pip", "install", "--python", python, "--compile-bytecode"]
            if step.no_deps:
                command.append("--no-deps")
            if step.index_url:
                command += ["--index-url", step.index_url, "--extra-index-url", "https://pypi.org/simple",
                            "--index-strategy", "unsafe-best-match"]
            command += list(step.packages)
            log("$ uv pip install " + " ".join(command[5:]))
            code, tail = _run_streaming(command, env, token, log, cwd=stage, tick=tick)
            if code != 0:
                detail = next((l for l in reversed(tail) if "error" in l.lower()), tail[-1] if tail else str(code))
                raise TaskError(f"Installing {spec.label} failed: {detail}")

        emit(92, "Checking the installation")
        if spec.import_check:
            code, tail = _run_streaming([python, "-c", spec.import_check], _clean_env(), token, log, cwd=stage)
            if code != 0:
                raise TaskError(f"{spec.label} installed but does not load: {tail[-1] if tail else code}")
        frozen = subprocess.run([uv, "pip", "freeze", "--python", python], capture_output=True, text=True, env=env)
        version = subprocess.run([python, "-c", "import platform; print(platform.python_version())"],
                                 capture_output=True, text=True, env=_clean_env())
        emit(97, "Finishing")
        _commit(stage, spec.id, {
            "id": spec.id,
            "version": spec.version,
            "packages": [l for l in frozen.stdout.splitlines() if l.strip()],
            "python": version.stdout.strip() or None,
            "installedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        })
        log(f"{spec.label} installed")
        emit(100, "Installed")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        # The venv holds hard links / clones of the cache's files, so the
        # cache costs a second copy's worth of disk once the pack is removed.
        # Clearing it keeps "remove" honest about the space it frees.
        shutil.rmtree(cache, ignore_errors=True)


def _install_ffmpeg(spec: PackSpec, progress: Optional[ProgressFn], log: LogFn, token: CancellationToken) -> None:
    emit = _progress_adapter(progress)
    key = host()
    suffix = _JELLYFIN_ASSETS.get(key)
    if not suffix:
        raise TaskError(f"No portable FFmpeg build for {key[0]} {key[1]}.")
    emit(0, "Finding the latest build")
    release = _fetch_json(_JELLYFIN_API, token)
    asset = next((a for a in release.get("assets", []) if str(a.get("name", "")).endswith(suffix)), None)
    if not asset:
        raise TaskError(f"The latest jellyfin-ffmpeg release ({release.get('tag_name')}) has no {suffix} build.")
    log(f"Downloading {asset['name']}")
    stage = _stage_dir(spec.id)
    try:
        archive = os.path.join(stage, asset["name"])
        total = int(asset.get("size") or 0) or None
        download(asset["browser_download_url"], archive, total, token=token,
                 on_bytes=lambda d, t: emit(90 * d / (t or spec.download_bytes or 1), "Downloading FFmpeg", d, t))
        digest = str(asset.get("digest") or "")
        if digest.startswith("sha256:"):
            if _sha256(archive) != digest.split(":", 1)[1].lower():
                raise TaskError("The FFmpeg download is corrupt (sha256 mismatch).")
        else:
            log("The release does not publish a checksum for this build; skipping verification.")
        emit(92, "Unpacking")
        bin_dir = os.path.join(stage, "bin")
        _extract(archive, bin_dir)
        os.remove(archive)
        exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        found = next((os.path.join(r, exe) for r, _d, f in os.walk(bin_dir) if exe in f), None)
        if not found:
            raise TaskError("The FFmpeg archive did not contain ffmpeg")
        if os.path.dirname(found) != bin_dir:
            for name in os.listdir(os.path.dirname(found)):
                shutil.move(os.path.join(os.path.dirname(found), name), os.path.join(bin_dir, name))
        ffmpeg = os.path.join(bin_dir, exe)
        if os.name != "nt":
            for name in os.listdir(bin_dir):
                os.chmod(os.path.join(bin_dir, name), 0o755)
        emit(96, "Checking the build")
        caps = _ffmpeg_caps(ffmpeg)
        log("FFmpeg filters: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in caps.items()))
        _commit(stage, spec.id, {
            "id": spec.id,
            "version": spec.version,
            "packages": [asset["name"]],
            "release": release.get("tag_name"),
            "filters": caps,
            "installedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        })
        emit(100, "Installed")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _ffmpeg_caps(ffmpeg: str) -> Dict[str, bool]:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TaskError(f"The downloaded FFmpeg does not run: {exc}") from exc
    if out.returncode != 0:
        raise TaskError(f"The downloaded FFmpeg does not run: {out.stderr.strip()[-200:]}")
    names = {parts[1] for parts in (l.split() for l in out.stdout.splitlines()) if len(parts) > 2}
    return {name: name in names for name in ("zscale", "libplacebo", "subtitles", "tonemapx", "tonemap_videotoolbox")}


def _download_model(spec: PackSpec, model: ModelSpec, progress: Optional[ProgressFn], log: LogFn,
                    token: CancellationToken) -> None:
    emit = _progress_adapter(progress)
    if model_path(model):
        log(f"{model.label} is already downloaded")
        emit(100, "Downloaded")
        return
    python = python_for(spec.id)
    if not python:
        raise TaskError(f"Install {spec.label} first; it downloads its own models.")
    size = f"{model.download_bytes / 1e9:.1f} GB" if model.download_bytes >= 1e9 else f"{model.download_bytes / 1e6:.0f} MB"
    log(f"Downloading {model.label} from {model.repo} (about {size})")

    def on_event(event: Dict[str, Any]) -> None:
        if event.get("type") == "progress":
            emit(float(event.get("percent") or 0), event.get("stage") or f"Downloading {model.label}",
                 event.get("bytes"), event.get("totalBytes"))
        elif event.get("type") == "log":
            log(str(event.get("message")))

    _run_worker_process(python, "model_worker.py", {
        "repo": model.repo,
        "weightFile": model.weight_file,
        "totalBytes": model.download_bytes,
        "label": model.label,
        "allowPatterns": list(model.files) or None,
    }, token, on_event, cwd=None, what=f"{spec.label} model download")
    if not model_path(model):
        raise TaskError(f"{model.label} did not finish downloading.")
    emit(100, "Downloaded")


# ------------------------------------------------------------------- remove


def remove(pack_id: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Delete an installed pack (or, with ``model``, that model's weights
    from the Hugging Face cache). Adopted external environments are never
    touched."""
    spec = _spec(pack_id)
    if model is not None:
        chosen = model_spec(pack_id, model)
        if chosen.files:
            _remove_repo_files(chosen)
        else:
            _remove_repo_cache(chosen.repo)
        return pack_status(pack_id)
    folder = pack_dir(pack_id)
    if os.path.isdir(folder):
        # Rename first so a half-deleted pack can never pass for installed.
        doomed = os.path.join(packs_dir(), f".old-{pack_id}-{uuid.uuid4().hex[:8]}")
        try:
            os.replace(folder, doomed)
        except OSError as exc:
            raise TaskError(f"Could not remove {spec.label}; is it in use? ({exc})") from exc
        shutil.rmtree(doomed, ignore_errors=True)
    return pack_status(pack_id)


def _links_in(folder: str) -> set:
    found = set()
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                found.add(os.path.realpath(path))
    return found


def _remove_repo_files(model: ModelSpec) -> None:
    """Delete one model's files from a repo that holds several models: its
    links in every snapshot and the blobs they point to, unless another
    cached file still links to that blob."""
    import fnmatch

    target = _repo_cache(model.repo)
    snapshots = os.path.join(target, "snapshots")
    if not os.path.isdir(snapshots):
        return
    doomed = set()
    for snap in os.listdir(snapshots):
        root = os.path.join(snapshots, snap)
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                if any(fnmatch.fnmatch(rel, pattern) for pattern in model.files):
                    if os.path.islink(path):
                        doomed.add(os.path.realpath(path))
                    os.remove(path)
    hub = hf_cache_dir()
    in_use = set()
    for name in os.listdir(hub):
        if name.startswith(("models--", "datasets--", "spaces--")):
            in_use |= _links_in(os.path.join(hub, name, "snapshots"))
    for path in doomed - in_use:
        # The repo's own blob may itself link into the shared store.
        for victim in {path, os.path.realpath(path)}:
            try:
                os.remove(victim)
            except OSError:
                pass


def _remove_repo_cache(repo: str) -> None:
    """Delete a repo from the HF cache, including its files in the shared
    blob store (``hub/blobs``) that no other cached repo links to -- deleting
    the repo folder alone would free only its symlinks."""
    target = _repo_cache(repo)
    if not os.path.isdir(target):
        return
    inside = os.path.realpath(target) + os.sep
    shared = {p for p in _links_in(os.path.join(target, "blobs")) if not p.startswith(inside)}
    shutil.rmtree(target)
    if not shared:
        return
    hub = hf_cache_dir()
    in_use = set()
    for name in os.listdir(hub):
        if name.startswith(("models--", "datasets--", "spaces--")):
            in_use |= _links_in(os.path.join(hub, name, "blobs"))
    for path in shared - in_use:
        try:
            os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------------ workers


def _run_worker_process(
    python: str,
    script: str,
    request: Dict[str, Any],
    token: CancellationToken,
    on_event: Callable[[Dict[str, Any]], None],
    cwd: Optional[str],
    what: str,
) -> Dict[str, Any]:
    path = os.path.join(workers_dir(), script)
    if not os.path.isfile(path):
        raise TaskError(f"Worker script missing: {path}")
    token.raise_if_cancelled()
    # -u unbuffered, -B no __pycache__ in a signed bundle, -s no user site,
    # -X utf8 so Japanese text survives a Windows console code page.
    command = [python, "-u", "-B", "-s", "-X", "utf8", path]
    kwargs: Dict[str, Any] = {
        "stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        "env": _clean_env(), "cwd": cwd,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise TaskError(f"Could not start the {what} ({python}): {exc}") from exc
    token.register(process)
    stderr_tail: collections.deque = collections.deque(maxlen=40)

    def drain_stderr() -> None:
        assert process.stderr is not None
        for raw in iter(process.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            # tqdm redraws with \r; keep only the last state of such a line.
            line = line.split("\r")[-1].strip()
            if line:
                stderr_tail.append(line)

    err_thread = threading.Thread(target=drain_stderr, daemon=True)
    err_thread.start()
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    try:
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps(request).encode("utf-8"))
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # it died at start-up; the exit code and stderr say why
        for raw in iter(process.stdout.readline, b""):
            try:
                event = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                stderr_tail.append(raw.decode("utf-8", "replace").rstrip())
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "result":
                result = event.get("data") or {}
            elif kind == "error":
                error = str(event.get("message") or "unknown error")
            else:
                on_event(event)
        process.wait()
        err_thread.join(2.0)
    finally:
        token.unregister(process)
        _terminate(process)
    if token.cancelled:
        raise Cancelled("operation cancelled")
    if error:
        raise TaskError(f"{what[:1].upper() + what[1:]} failed: {error}")
    if result is None:
        tail = " | ".join(list(stderr_tail)[-6:]) or "no output"
        raise TaskError(f"The {what} stopped unexpectedly (exit code {process.returncode}): {tail}")
    return result


def run_worker(pack_id: str, script: str, request: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Run ``workers/<script>`` in the pack's Python and return its result.

    ``ctx`` is a TaskContext (or anything with ``progress``, ``log`` and
    ``token``). Worker progress and log lines are forwarded as they arrive;
    Stop kills the worker's whole process group through the token.
    """
    spec = _spec(pack_id)
    python = python_for(pack_id)
    if not python:
        if not supported(spec):
            raise TaskError(f"{spec.label} is not available on this computer.")
        raise TaskError(f"Install the {spec.label} pack first (Settings > Engine packs).")

    def on_event(event: Dict[str, Any]) -> None:
        if event.get("type") == "progress":
            ctx.progress(int(float(event.get("percent") or 0)), str(event.get("stage") or spec.label))
        elif event.get("type") == "log":
            ctx.log(str(event.get("message")))

    return _run_worker_process(python, script, request, ctx.token, on_event,
                               cwd=getattr(ctx, "workdir", None), what=f"{spec.label} worker")
