"""Run every Python test module and report a combined result.

Usage:  python tests/run_all.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODULES = [
    "test_correlate",
    "test_media",
    "test_analyze",
    "test_matching",
    "test_framerate",
    "test_tracks",
    "test_compare",
    "test_mux",
    "test_bridge",
]


def ensure_fixtures() -> None:
    manifest = os.path.join(HERE, "fixtures", "manifest.json")
    if os.path.exists(manifest):
        return
    print("Fixtures missing; generating them first...\n")
    spec = importlib.util.spec_from_file_location(
        "make_fixtures", os.path.join(HERE, "make_fixtures.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.build()
    print()


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ensure_fixtures()

    total = failed = 0
    started = time.monotonic()

    for name in MODULES:
        print(f"{name}")
        module = load(name)
        tests = [
            value
            for key, value in sorted(vars(module).items())
            if key.startswith("test_") and callable(value)
        ]
        for test in tests:
            total += 1
            try:
                test()
                print(f"  PASS  {test.__name__}")
            except AssertionError as exc:
                failed += 1
                print(f"  FAIL  {test.__name__}\n        {exc}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        print()

    elapsed = time.monotonic() - started
    print(f"{total - failed}/{total} passed in {elapsed:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
