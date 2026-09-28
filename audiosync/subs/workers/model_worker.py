"""Model download worker: fetches a Hugging Face repo into the standard cache.

Runs in any pack that has huggingface_hub (both speech packs and Demucs), so
downloads resume, share the cache with every other Hugging Face tool on the
machine, and use the hub's own fast transfer backend.

Request: {"repo": "mlx-community/whisper-large-v3-mlx", "totalBytes": N,
          "weightFile": "weights.npz", "label": "Large v3",
          "allowPatterns": ["languages/korean/*"] | null}   # a subset of the repo
Result:  {"path": "<snapshot folder>"}

Byte progress: ``snapshot_download`` sums every file's bytes into two bars,
network bytes received ("Downloading bytes") and bytes written
("Reconstructing"), and a tqdm stand-in handed to it reads both. The Xet
backend writes a file -- and advances the second bar -- only when the file
is complete, so the network count is what moves during a large download;
the bytes on disk are kept as a floor so a resumed download starts from what
is already there. Recent
huggingface_hub versions keep large files in a store shared by all repos
(``hub/blobs/<xx>/<sha256>``) with the repo's blob a symlink into it, so
finished files are counted through their links.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as proto  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.pop("HF_HUB_OFFLINE", None)


def _size(folder: str, shared: str) -> int:
    """Bytes of the repo on disk: finished blobs (through their links, each
    target once) plus every partial download in the repo or shared store."""
    seen = set()
    total = 0
    for root in (os.path.join(folder, "blobs"), shared):
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(dirpath, name)
                partial = name.endswith(".incomplete")
                if dirpath.startswith(shared) and not partial:
                    continue  # finished shared files count via the repo's links
                try:
                    real = os.path.realpath(path)
                    if real not in seen:
                        seen.add(real)
                        total += os.path.getsize(real)
                except OSError:
                    pass
    return total


class _Bar:
    """Just enough of tqdm for snapshot_download's bars and thread map.

    Only the two aggregated byte bars are read; the file-count bar and any
    other call are accepted and ignored.
    """

    byte_bars: "list" = []
    _lock = threading.RLock()

    def __init__(self, iterable: Any = None, *_args: Any, **kwargs: Any) -> None:
        self.iterable = iterable
        self.total = kwargs.get("total")
        self.n = kwargs.get("initial") or 0
        if kwargs.get("unit") == "B":
            _Bar.byte_bars.append(self)

    def __iter__(self):
        return iter(self.iterable if self.iterable is not None else ())

    def __enter__(self) -> "_Bar":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def update(self, n: Any = 1) -> None:
        self.n += int(n or 0)

    @property
    def format_dict(self) -> Dict[str, Any]:
        return {"rate": None, "n": self.n, "total": self.total}

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock) -> None:
        cls._lock = lock

    def __getattr__(self, _name: str):
        return lambda *_a, **_k: None


def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    from huggingface_hub import constants, snapshot_download

    repo = req["repo"]
    total = int(req.get("totalBytes") or 0) or None
    label = req.get("label") or repo
    stage = f"Downloading {label}"
    folder = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
    shared = os.path.join(constants.HF_HUB_CACHE, "blobs")
    done = threading.Event()

    def watch() -> None:
        while not done.wait(1.0):
            have = max(0, _size(folder, shared) - base)
            for bar in list(_Bar.byte_bars):
                have = max(have, int(bar.n))
            percent = 100.0 * have / total if total else 0.0
            proto.progress(min(99.0, percent), stage, bytes=have, totalBytes=total)

    # Other models of the same repo may already be on disk; only this
    # download's growth counts.
    base = _size(folder, shared)
    proto.progress(0, stage, bytes=0, totalBytes=total)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        path = snapshot_download(repo_id=repo, tqdm_class=_Bar, allow_patterns=req.get("allowPatterns") or None)
    finally:
        done.set()
        thread.join(2.0)
    weight = req.get("weightFile")
    if weight and not os.path.exists(os.path.join(path, weight)):
        raise RuntimeError(f"{repo} downloaded without {weight}")
    have = max(0, _size(folder, shared) - base)
    for bar in list(_Bar.byte_bars):
        have = max(have, int(bar.n))
    proto.progress(100, stage, bytes=have, totalBytes=total or have)
    return {"path": path}


if __name__ == "__main__":
    proto.main(handle)
