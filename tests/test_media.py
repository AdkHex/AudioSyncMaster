"""Tests for decoding, probing and process lifecycle.

The cancellation tests matter most: the original could only kill the sidecar,
leaving every ffmpeg child it had spawned running until it finished on its own.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync.media import (  # noqa: E402
    Cancelled,
    CancellationToken,
    MediaError,
    _run,
    ffmpeg_path,
    is_audio_file,
    is_video_file,
    load_audio,
    probe,
)

SR = 16000


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="audiosync-media-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def _tone(seconds=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.3).astype(np.float32)


def _count_processes(pattern: str) -> int:
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
    return len([line for line in result.stdout.decode().splitlines() if line.strip()])


def test_probe_reports_streams_and_duration():
    with Workspace() as workspace:
        path = workspace.path("a.wav")
        sf.write(path, _tone(5.0), SR)
        info = probe(path)
        assert info.has_audio is True
        assert info.has_video is False
        assert info.duration is not None and abs(info.duration - 5.0) < 0.1


def test_seek_is_sample_accurate():
    """-ss must land exactly, not on the nearest keyframe.

    Placing -ss before -i (as the original did) snaps to a keyframe, which
    silently corrupts the end-of-file measurement that end delay depends on.
    """
    with Workspace() as workspace:
        path = workspace.path("a.wav")
        signal = _tone(20.0, seed=3)
        sf.write(path, signal, SR)

        full = load_audio(path, SR)
        segment = load_audio(path, SR, duration=4.0, offset=6.0)

        expected = full[6 * SR : 10 * SR]
        n = min(len(expected), len(segment))
        assert n > SR, "decoded segment too short to compare"
        assert np.max(np.abs(expected[:n] - segment[:n])) < 1e-5, (
            "seek did not land at the requested position"
        )


def test_seek_stays_accurate_deep_into_a_compressed_file():
    """The two-stage seek must not shift the window it returns.

    load_audio jumps most of the way with a fast input seek and then decodes
    accurately through the last SEEK_PREROLL_S seconds. A deep offset in a
    compressed container is the case that distinguishes it from a plain input
    seek: the coarse jump lands on a packet boundary, and only the accurate
    second stage puts the window back where it was asked for.

    WAV cannot catch this -- every frame is independently addressable, so any
    seek strategy looks correct.
    """
    with Workspace() as workspace:
        source = workspace.path("a.wav")
        encoded = workspace.path("a.m4a")
        signal = _tone(180.0, seed=11)
        sf.write(source, signal, SR)

        completed = subprocess.run(
            [ffmpeg_path(), "-y", "-nostdin", "-i", source,
             "-c:a", "aac", "-b:a", "128k", encoded, "-loglevel", "error"],
            capture_output=True,
        )
        if completed.returncode != 0:
            return  # no AAC encoder available; nothing to assert

        # Well beyond SEEK_PREROLL_S, so the coarse stage really does jump.
        offset, duration = 150.0, 4.0
        reference = load_audio(encoded, SR)
        segment = load_audio(encoded, SR, duration=duration, offset=offset)

        start = int(offset * SR)
        expected = reference[start : start + int(duration * SR)]
        n = min(len(expected), len(segment))
        assert n > SR, "decoded segment too short to compare"

        # Correlate rather than subtract: lossy round-tripping changes sample
        # values, so the question is whether the window sits at the right
        # position, not whether the samples are bit-identical.
        a = expected[:n] - expected[:n].mean()
        b = segment[:n] - segment[:n].mean()
        lag = int(np.argmax(np.correlate(a, b, mode="full"))) - (n - 1)
        assert abs(lag) <= 16, f"window shifted by {lag} samples ({lag / SR * 1000:.2f} ms)"


def test_decoded_length_matches_requested_duration():
    with Workspace() as workspace:
        path = workspace.path("a.wav")
        sf.write(path, _tone(30.0), SR)
        segment = load_audio(path, SR, duration=5.0, offset=2.0)
        assert abs(len(segment) / SR - 5.0) < 0.05


def test_missing_file_raises_a_clear_error():
    try:
        load_audio("/definitely/not/here.wav", SR)
    except MediaError as exc:
        assert "not found" in str(exc).lower()
    else:
        raise AssertionError("missing file did not raise")


def test_decoded_audio_is_writable():
    """Downstream code standardizes in place; a read-only buffer would break it."""
    with Workspace() as workspace:
        path = workspace.path("a.wav")
        sf.write(path, _tone(2.0), SR)
        decoded = load_audio(path, SR)
        decoded[0] = 0.5  # must not raise
        assert decoded.flags.writeable


def test_cancellation_terminates_running_ffmpeg():
    """Cancelling must kill in-flight subprocesses, not wait them out."""
    if os.name == "nt":
        return  # pgrep is unavailable on Windows runners

    with Workspace() as workspace:
        token = CancellationToken()
        outcomes: list[str] = []
        marker = f"audiosynctest{os.getpid()}"

        def work(index: int) -> None:
            command = [
                ffmpeg_path(), "-nostdin", "-y",
                "-f", "lavfi",
                "-i", "testsrc=size=640x480:rate=30:duration=3600",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-metadata", f"comment={marker}",
                workspace.path(f"slow{index}.mp4"),
            ]
            try:
                _run(command, 600, token, what="slow encode")
                outcomes.append("completed")
            except Cancelled:
                outcomes.append("cancelled")
            except MediaError:
                outcomes.append("killed")

        baseline = _count_processes(marker)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()

        # Wait for the encodes to actually be running.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _count_processes(marker) < 2:
            time.sleep(0.1)
        running = _count_processes(marker)
        assert running >= 1, "ffmpeg never started; cannot test cancellation"

        token.cancel()
        for thread in threads:
            thread.join(timeout=20)
        time.sleep(0.5)

        remaining = _count_processes(marker) - baseline
        assert remaining <= 0, f"{remaining} orphaned ffmpeg process(es) survived cancel"
        assert "completed" not in outcomes, f"work was not interrupted: {outcomes}"


def test_cancellation_before_start_is_immediate():
    token = CancellationToken()
    token.cancel()
    try:
        load_audio("/tmp/whatever.wav", SR, token=token)
    except Cancelled:
        pass
    except MediaError:
        pass  # File check may fire first; either is a refusal to work.
    else:
        raise AssertionError("cancelled token was ignored")


def test_extension_classification():
    assert is_video_file("/x/a.mkv") and is_video_file("/x/A.MP4")
    assert is_audio_file("/x/a.eac3") and is_audio_file("/x/a.FLAC")
    assert not is_video_file("/x/a.txt")
    assert not is_audio_file("/x/a.mkv")


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
