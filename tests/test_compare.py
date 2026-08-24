"""Tests for comparing every video against every audio track.

The question this mode answers is "which release was this dub timed for?".
A dub belongs to exactly one source; against the others it either fails to
match or drifts. The test that matters is whether the right one stands out.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync.batch import BatchOptions, run_batch, summarize  # noqa: E402
from audiosync.matching import (  # noqa: E402
    MAX_COMPARE_INPUTS,
    pair_every_combination,
)

SR = 16000


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="audiosync-compare-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def _speech(seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    voiced = np.convolve(rng.standard_normal(n), np.ones(24) / 24, mode="same")
    envelope = np.zeros(n)
    pos = 0
    while pos < n:
        burst = int(rng.uniform(0.25, 0.9) * SR)
        gap = int(rng.uniform(0.1, 0.4) * SR)
        envelope[pos : min(n, pos + burst)] = rng.uniform(0.4, 1.0)
        pos = pos + burst + gap
    envelope = np.convolve(envelope, np.ones(128) / 128, mode="same")
    signal = voiced * envelope
    return (signal / np.max(np.abs(signal)) * 0.7).astype(np.float32)


def test_every_combination_is_produced():
    pairs = pair_every_combination(["/v/a.mkv", "/v/b.mkv"], ["/a/1.ac3", "/a/2.ac3"])
    assert len(pairs) == 4
    combinations = {(p.primary_name, p.secondary_name) for p in pairs}
    assert combinations == {
        ("a.mkv", "1.ac3"), ("a.mkv", "2.ac3"),
        ("b.mkv", "1.ac3"), ("b.mkv", "2.ac3"),
    }


def test_pair_keys_are_unique():
    """Results are merged by identity, so every combination needs its own key."""
    pairs = pair_every_combination(["/v/a.mkv", "/v/b.mkv"], ["/a/1.ac3", "/a/2.ac3"])
    assert len({p.key for p in pairs}) == len(pairs), "duplicate keys would merge rows"


def test_track_selection_reaches_every_pair():
    pairs = pair_every_combination(
        ["/v/a.mkv"], ["/a/1.ac3"], primary_track=2, secondary_track=1
    )
    assert all(p.primary_track == 2 and p.secondary_track == 1 for p in pairs)


def test_empty_side_produces_nothing():
    assert pair_every_combination([], ["/a/1.ac3"]) == []
    assert pair_every_combination(["/v/a.mkv"], []) == []


def test_the_matching_source_is_distinguishable():
    """The decisive case: one dub, two candidate sources, only one is right."""
    with Workspace() as workspace:
        correct = _speech(30, seed=1)
        other = _speech(30, seed=999)  # a different title entirely

        source_a = workspace.path("release_a.wav")
        source_b = workspace.path("release_b.wav")
        sf.write(source_a, correct, SR)
        sf.write(source_b, other, SR)

        # A dub timed against release A, 250ms late.
        delayed = np.concatenate([np.zeros(int(0.25 * SR), dtype=np.float32), correct])
        dub = workspace.path("dub.wav")
        sf.write(dub, delayed, SR)

        pairs = pair_every_combination([source_a, source_b], [dub])
        results = run_batch(pairs, BatchOptions(window_s=8.0, window_count=3))

        assert len(results) == 2
        by_source = {r.primary_name: r for r in results}

        matched = by_source["release_a.wav"]
        assert matched.delay_ms is not None, f"correct source rejected: {matched.error}"
        assert abs(matched.delay_ms - 250.0) < 15.0, (
            f"expected +250ms against release A, got {matched.delay_ms:+.1f}ms"
        )

        unrelated = by_source["release_b.wav"]
        assert unrelated.delay_ms is None, (
            f"unrelated source produced a confident {unrelated.delay_ms:+.1f}ms; "
            "compare mode cannot tell the releases apart"
        )


def test_summary_counts_every_combination():
    with Workspace() as workspace:
        base = _speech(20, seed=3)
        source = workspace.path("v.wav")
        sf.write(source, base, SR)
        dub = workspace.path("d.wav")
        sf.write(dub, base, SR)

        pairs = pair_every_combination([source], [dub])
        results = run_batch(pairs, BatchOptions(window_s=8.0, window_count=3))
        summary = summarize(results)
        assert summary["total"] == 1
        assert summary["matched"] == 1


def test_input_cap_is_a_sane_size():
    """The work is the product of both sides, so the cap has to be modest."""
    assert 2 <= MAX_COMPARE_INPUTS <= 10
    assert MAX_COMPARE_INPUTS**2 <= 100, "worst-case combination count is too large"


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
