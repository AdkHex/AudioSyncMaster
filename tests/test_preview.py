"""Tests for the aligned preview excerpt.

The point of a preview is that playing it settles the question. So the test
that matters is not "did ffmpeg produce a file" but "is the audio in that file
actually lined up with the picture" -- which is checked by measuring the
excerpt and expecting ~0 residual offset.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync.analyze import analyze_pair  # noqa: E402
from audiosync.media import MediaError, ffmpeg_path, probe  # noqa: E402
from audiosync.mux import (  # noqa: E402
    build_preview_command,
    choose_preview_position,
    extract_preview,
)

SR = 48000


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="audiosync-preview-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def _speech(seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    voiced = np.convolve(rng.standard_normal(n), np.ones(72) / 72, mode="same")
    envelope = np.zeros(n)
    pos = 0
    while pos < n:
        burst = int(rng.uniform(0.25, 0.9) * SR)
        gap = int(rng.uniform(0.1, 0.4) * SR)
        envelope[pos : min(n, pos + burst)] = rng.uniform(0.4, 1.0)
        pos = pos + burst + gap
    envelope = np.convolve(envelope, np.ones(384) / 384, mode="same")
    signal = voiced * envelope
    return (signal / np.max(np.abs(signal)) * 0.7).astype(np.float32)


def _make_video(workspace, audio, seconds, name="source"):
    wav = workspace.path(f"{name}.wav")
    sf.write(wav, audio, SR)
    out = workspace.path(f"{name}.mkv")
    subprocess.run(
        [
            ffmpeg_path(), "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=25:duration={seconds}",
            "-i", wav,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", out,
        ],
        capture_output=True, check=True,
    )
    return out


def test_position_avoids_the_start_and_the_end():
    """Openings and credits are the likeliest places to find silence."""
    position = choose_preview_position(3600.0, 12.0)
    assert position > 0, "a preview starting at 0 often lands on a logo sting"
    assert position + 12.0 < 3600.0, "must fit inside the file"


def test_position_handles_short_and_missing_durations():
    assert choose_preview_position(5.0, 12.0) == 0.0
    assert choose_preview_position(None, 12.0) == 0.0
    assert choose_preview_position(0.0, 12.0) == 0.0


def test_command_uses_input_seeks():
    """A preview is a quick check; decoding two hours to reach it defeats that."""
    command = build_preview_command(
        "/v.mkv", "/a.ac3", 250.0, 600.0, 12.0, "/out.mp4",
    )
    video_index = command.index("/v.mkv")
    audio_index = command.index("/a.ac3")
    seeks = [i for i, part in enumerate(command) if part == "-ss"]
    assert len(seeks) == 2, f"expected two seeks, got {len(seeks)}"
    assert seeks[0] < video_index, "video seek must come before its input"
    assert seeks[1] < audio_index, "audio seek must come before its input"


def test_command_offsets_the_audio_by_the_measured_delay():
    """The audio is taken from where the measurement says it should be."""
    command = build_preview_command(
        "/v.mkv", "/a.ac3", 500.0, 100.0, 12.0, "/out.mp4",
    )
    seeks = [float(command[i + 1]) for i, part in enumerate(command) if part == "-ss"]
    assert abs(seeks[0] - 100.0) < 1e-6, "video should start at the requested position"
    assert abs(seeks[1] - 100.5) < 1e-6, "audio should start 500ms later"


def test_command_selects_the_requested_audio_track():
    command = build_preview_command(
        "/v.mkv", "/a.mka", 0.0, 10.0, 12.0, "/out.mp4", audio_track=2,
    )
    assert "1:a:2" in command
    # The video's own audio is irrelevant; only its picture is used.
    assert "0:v:0" in command
    assert not any(part.startswith("0:a:") for part in command)


def test_command_corrects_drift_so_the_excerpt_stays_aligned():
    command = build_preview_command(
        "/v.mkv", "/a.ac3", 0.0, 10.0, 12.0, "/out.mp4", drift_ms_per_s=0.5,
    )
    assert any("atempo" in part for part in command), "drift left uncorrected"


def test_missing_input_fails_clearly():
    with Workspace() as workspace:
        try:
            extract_preview(
                "/definitely/not/here.mkv", "/nor/here.ac3",
                0.0, 0.0, 5.0, workspace.path("out.mp4"),
            )
        except MediaError as exc:
            assert "not found" in str(exc).lower()
        else:
            raise AssertionError("a missing input did not raise")


def test_preview_is_actually_in_sync():
    """The decisive test: measure the excerpt and expect ~0 residual.

    A preview that renders but is still out of sync is worse than no preview at
    all, because it would talk the user out of a correct measurement.
    """
    with Workspace() as workspace:
        base = _speech(90, seed=4)
        true_offset_ms = 400.0
        delay = int(true_offset_ms / 1000 * SR)

        video = _make_video(workspace, base, 90)
        dub = workspace.path("dub.wav")
        sf.write(dub, np.concatenate([np.zeros(delay, dtype=np.float32), base]), SR)

        measured = analyze_pair(video, dub, window_s=15, window_count=4)
        assert measured.error is None, f"measurement failed: {measured.error}"

        preview = extract_preview(
            video, dub, measured.delay_ms,
            position_s=30.0, duration_s=15.0,
            output_path=workspace.path("preview.mp4"),
        )
        assert os.path.isfile(preview) and os.path.getsize(preview) > 0

        info = probe(preview)
        assert info.has_video and info.has_audio, "preview lost a stream"

        # The excerpt's own audio against its own picture: if the delay was
        # right, the two now sit together.
        residual = analyze_pair(preview, preview, window_s=6, window_count=3)
        assert residual.error is None, f"could not re-measure the preview: {residual.error}"
        assert abs(residual.delay_ms) < 30.0, (
            f"preview is still {residual.delay_ms:+.1f}ms out of sync"
        )


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
