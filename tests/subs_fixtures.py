"""Shared helpers for the bitmap-subtitle tests (test_subs_pgs, test_subs_ocr).

``render`` draws captions with CoreText through ``tests/subs_render.swift``
(compiled on first use) -- white fill, black outline, transparent
background, optional italics, furigana and vertical text -- so the tests get
realistic subtitle bitmaps without shipping any. It raises ``Skip`` where
there is no macOS/swiftc; tests wrapped in ``skippable`` then report a skip
and pass, the convention the other test modules follow.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from audiosync.subs import imaging  # noqa: E402


class Skip(Exception):
    """Raised by a test that needs a tool this machine does not have."""


def skippable(fn):
    """Report a missing tool as a skip and pass, so tests/run_all.py (which
    knows only pass/fail) stays green where the tool is absent."""

    @functools.wraps(fn)
    def wrapper():
        try:
            return fn()
        except Skip as exc:
            print(f"        (skipped: {exc})")

    return wrapper


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="subsync-bitmap-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def tool(name):
    found = shutil.which(name)
    if not found:
        raise Skip(f"{name} not installed")
    return found


_RENDERER = None


def render(specs):
    """RGBA arrays of captions described like ``subs_render.swift`` expects
    (without ``out``)."""
    global _RENDERER
    swiftc = shutil.which("swiftc")
    if sys.platform != "darwin" or not swiftc:
        raise Skip("needs macOS with swiftc (CoreText renderer)")
    if _RENDERER is None:
        source = os.path.join(HERE, "subs_render.swift")
        with open(source, "rb") as handle:
            digest = str(abs(hash(handle.read())) % 10**12)
        cache = os.path.join(tempfile.gettempdir(), "subsync-test-render")
        os.makedirs(cache, exist_ok=True)
        binary = os.path.join(cache, f"render-{os.path.getsize(source)}-{int(os.path.getmtime(source))}")
        if not os.path.isfile(binary):
            staging = f"{binary}.{os.getpid()}-{digest}.tmp"
            subprocess.run([swiftc, "-O", "-swift-version", "5", "-o", staging, source], check=True,
                           capture_output=True)
            os.replace(staging, binary)
        _RENDERER = binary
    out_dir = tempfile.mkdtemp(prefix="subsync-render-")
    try:
        jobs = []
        for i, spec in enumerate(specs):
            job = dict(spec)
            job["out"] = os.path.join(out_dir, f"{i}.png")
            jobs.append(job)
        subprocess.run([_RENDERER], input=json.dumps(jobs).encode("utf-8"), check=True, capture_output=True)
        return [imaging.read_png(job["out"]) for job in jobs]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def run_module(namespace):
    """The ``__main__`` runner: every ``test_*`` in ``namespace``."""
    tests = [v for k, v in sorted(namespace.items()) if k.startswith("test_") and callable(v)]
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
