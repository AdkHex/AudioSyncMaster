"""Tests for selecting which audio stream to compare.

A container often carries several streams -- the original language, a dub, a
commentary. Comparing against the wrong one produces a measurement of two
tracks that were never meant to align, so the choice has to be explicit and it
has to actually reach the decoder.
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
from audiosync.media import AudioTrack, ffmpeg_path, load_audio, probe  # noqa: E402

SR = 16000


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="audiosync-tracks-")
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


def _multitrack_video(workspace, first, second, seconds, fps=25):
    """A video carrying two unrelated audio streams."""
    a = workspace.path("a.wav")
    b = workspace.path("b.wav")
    sf.write(a, first, SR)
    sf.write(b, second, SR)
    out = workspace.path("multi.mkv")
    subprocess.run(
        [
            ffmpeg_path(), "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate={fps}:duration={seconds}",
            "-i", a, "-i", b,
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=Original",
            "-metadata:s:a:1", "language=fra", "-metadata:s:a:1", "title=Commentary",
            "-shortest", out,
        ],
        capture_output=True, check=True,
    )
    return out


def test_probe_lists_every_audio_track():
    with Workspace() as workspace:
        video = _multitrack_video(workspace, _speech(10, 1), _speech(10, 2), 10)
        info = probe(video)
        assert len(info.audio_tracks) == 2, f"found {len(info.audio_tracks)} tracks, want 2"
        assert info.audio_tracks[0].index == 0
        assert info.audio_tracks[1].index == 1


def test_track_metadata_is_read():
    with Workspace() as workspace:
        video = _multitrack_video(workspace, _speech(10, 1), _speech(10, 2), 10)
        tracks = probe(video).audio_tracks
        assert tracks[0].language == "eng", f"language not read: {tracks[0].language}"
        assert tracks[1].title == "Commentary", f"title not read: {tracks[1].title}"
        assert "Commentary" in tracks[1].label


def test_probe_reads_frame_rate():
    with Workspace() as workspace:
        video = _multitrack_video(workspace, _speech(6, 1), _speech(6, 2), 6, fps=25)
        assert abs((probe(video).fps or 0) - 25.0) < 0.01


def test_decoding_returns_the_requested_track():
    """The two streams are unrelated, so decoding each must give different audio."""
    with Workspace() as workspace:
        video = _multitrack_video(workspace, _speech(10, 1), _speech(10, 2), 10)
        first = load_audio(video, SR, duration=5.0, track=0)
        second = load_audio(video, SR, duration=5.0, track=1)

        n = min(len(first), len(second))
        a = first[:n] - first[:n].mean()
        b = second[:n] - second[:n].mean()
        similarity = abs(
            float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        )
        assert similarity < 0.3, (
            f"tracks 0 and 1 decoded to near-identical audio (r={similarity:.2f}); "
            "the track argument is probably being ignored"
        )


def test_analysis_matches_the_right_track_and_rejects_the_wrong_one():
    """The decisive test: a dub timed to track 0 must match track 0 only.

    The delay is measured *relative to an undelayed baseline* rather than
    against the absolute 300ms. Encoding the video introduces a codec delay of
    its own, which varies by ffmpeg build -- on one CI runner it was 64ms. That
    delay is identical for both measurements here and for both files of any real
    pair, so it cancels in the difference. Asserting the absolute figure would
    be testing the local AAC encoder, not the track selection.
    """
    with Workspace() as workspace:
        original = _speech(30, 1)
        video = _multitrack_video(workspace, original, _speech(30, 777), 30)

        aligned = workspace.path("aligned.wav")
        sf.write(aligned, original, SR)

        delayed = np.concatenate([np.zeros(int(0.3 * SR), dtype=np.float32), original])
        dub = workspace.path("dub.wav")
        sf.write(dub, delayed, SR)

        baseline = analyze_pair(video, aligned, window_s=8, window_count=3, primary_track=0)
        assert baseline.error is None, f"baseline failed: {baseline.error}"

        right = analyze_pair(video, dub, window_s=8, window_count=3, primary_track=0)
        assert right.error is None, f"correct track failed: {right.error}"

        introduced = right.delay_ms - baseline.delay_ms
        assert abs(introduced - 300.0) < 15.0, (
            f"the 300ms delay measured as {introduced:+.1f}ms "
            f"(raw {right.delay_ms:+.1f}, baseline {baseline.delay_ms:+.1f})"
        )

        wrong = analyze_pair(video, dub, window_s=8, window_count=3, primary_track=1)
        assert wrong.delay_ms is None, (
            f"unrelated track 1 produced a confident {wrong.delay_ms:+.1f}ms"
        )


def test_requesting_a_missing_track_fails_clearly():
    with Workspace() as workspace:
        video = _multitrack_video(workspace, _speech(6, 1), _speech(6, 2), 6)
        try:
            load_audio(video, SR, duration=2.0, track=9)
        except Exception as exc:
            assert "track 10" in str(exc) or "decode" in str(exc).lower()
        else:
            raise AssertionError("a nonexistent track did not raise")


def test_track_label_is_readable_without_metadata():
    plain = AudioTrack(index=2, codec="eac3", channels=6)
    assert "Track 3" in plain.label
    assert "5.1" in plain.label
    assert "EAC3" in plain.label


def test_undefined_language_is_not_shown():
    track = AudioTrack(index=0, codec="aac", language="und")
    assert "UND" not in track.label


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
