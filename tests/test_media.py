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


def test_a_file_problem_is_said_in_the_files_terms():
    from audiosync.media import describe_failure

    said = describe_failure("/x/dub.m4a", "[mov,mp4] moov atom not found\ndub.m4a: Invalid data found when processing input")
    assert said.startswith("dub.m4a is incomplete"), said
    said = describe_failure("/x/v.mkv", "Stream map '0:a:5' matches no streams.\nError opening output files: Invalid argument", 5)
    assert said.startswith("v.mkv has no audio track 6"), said
    assert "Invalid argument" in said, "FFmpeg's own words are kept for diagnosis"
    assert describe_failure("/x/a.wav", "something new").startswith("FFmpeg could not read a.wav")


def test_folders_and_empty_files_are_refused_up_front():
    with Workspace() as ws:
        folder = ws.path("folder.mkv")
        os.makedirs(folder)
        empty = ws.path("empty.m4a")
        open(empty, "wb").close()
        for path, words in ((folder, "is a folder"), (empty, "is empty (0 bytes)"), (ws.path("gone.mkv"), "was not found")):
            try:
                probe(path)
            except MediaError as exc:
                assert words in str(exc), str(exc)
            else:
                raise AssertionError(f"{path} was probed")


def test_encoder_priming_is_not_decoded_as_sound_on_any_build():
    """AAC in Matroska carries 1024 samples of encoder priming marked to be
    skipped. FFmpeg 6.1, Ubuntu 24.04's, decodes them as sound and reports the
    track starting 64 ms before zero at 16 kHz, so every read of the file came
    out that much late. A click placed at a known time must decode there both
    when sought to and when read from the top, whichever build reads it."""
    from audiosync.media import stream_audio

    with Workspace() as ws:
        click_at = 3.0
        signal = np.zeros(SR * 8, dtype=np.float32)
        signal[int(click_at * SR)] = 0.9
        source = ws.path("click.wav")
        sf.write(source, signal, SR)
        encoded = ws.path("click.mka")
        subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", source, "-c:a", "aac", "-b:a", "320k", encoded], check=True)

        def lands(samples: np.ndarray, start_s: float) -> float:
            return (start_s + int(np.argmax(np.abs(samples))) / SR - click_at) * 1000.0

        from_top = np.concatenate(list(stream_audio(encoded, SR)))
        assert abs(lands(from_top, 0.0)) < 1.0, f"read from the top, the click is {lands(from_top, 0.0):+.2f} ms off"
        for start in (0.0, 1.5):
            window = load_audio(encoded, SR, duration=4.0, offset=start)
            assert abs(lands(window, start)) < 1.0, f"sought to {start}s, the click is {lands(window, start):+.2f} ms off"


def _late_audio_mkv(ws: Workspace, lead_s: float, seconds: float, click_at: float) -> str:
    """An MKV whose audio starts `lead_s` into the file, after its picture --
    as some remuxes do -- with a click `click_at` into the audio."""
    signal = (np.random.default_rng(9).standard_normal(int(seconds * SR)) * 0.02).astype(np.float32)
    signal[int(click_at * SR)] = 0.9
    source = ws.path("late.wav")
    sf.write(source, signal, SR)
    path = ws.path("late.mkv")
    subprocess.run(
        [
            ffmpeg_path(), "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=s=160x90:r=25:d={seconds + lead_s + 1}",
            "-itsoffset", str(lead_s), "-i", source,
            "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "flac", path,
        ],
        check=True,
    )
    return path


def test_a_read_from_zero_shares_the_clock_of_every_seek():
    """Where the audio starts after the picture, a read from the top used to
    start at the track's first sample while a seek landed on the file's
    clock: the same file read both ways was out by the gap, and a pair
    measured with both kinds of read reported a cut of exactly that size.
    Every read must put the click at the same time on the file's clock."""
    with Workspace() as ws:
        lead, click = 2.0, 6.0
        path = _late_audio_mkv(ws, lead, 20.0, click)
        on_file_clock = lead + click
        for start in (0.0, 1.0, 3.0):
            window = load_audio(path, SR, duration=10.0, offset=start)
            found = start + int(np.argmax(np.abs(window))) / SR
            assert abs(found - on_file_clock) < 0.002, f"read from {start}s, the click is at {found:.3f}s, not {on_file_clock:.3f}s"
            assert len(window) == 10 * SR, f"read from {start}s returned {len(window) / SR:.3f}s, not 10s"


def test_a_window_entirely_before_the_audio_is_silence_of_the_asked_length():
    with Workspace() as ws:
        path = _late_audio_mkv(ws, 2.0, 10.0, 4.0)
        window = load_audio(path, SR, duration=1.5, offset=0.2)
        assert len(window) == int(1.5 * SR)
        assert not np.any(window)


def test_a_stereo_decode_holds_the_mono_mix_and_the_side_difference():
    """(L + R) / sqrt(2) of the stereo decode is FFmpeg's own mono decode, so
    measuring from it changes nothing about the full mix; L - R is the rest."""
    with Workspace() as ws:
        left, right = _tone(6.0, seed=1), _tone(6.0, seed=2)
        path = ws.path("stereo.wav")
        sf.write(path, np.stack([left, right], axis=1), SR)
        stereo = load_audio(path, SR, duration=4.0, offset=1.0, channels=2)
        mono = load_audio(path, SR, duration=4.0, offset=1.0)
        assert stereo.shape == (len(mono), 2), stereo.shape
        mid = (stereo[:, 0] + stereo[:, 1]) / np.sqrt(2.0)
        assert np.max(np.abs(mid - mono)) < 1e-5
        written, _ = sf.read(path, dtype="float32")  # as stored: 16-bit, clipped
        side = stereo[:, 0] - stereo[:, 1]
        assert np.max(np.abs(side - (written[:, 0] - written[:, 1])[SR:SR + len(side)])) < 1e-4


def test_a_dts_track_is_read_from_its_core():
    """A DTS-HD MA track's core sits sample for sample under the lossless
    decode and is several times quicker to decode, so DTS is read that way."""
    import audiosync.media as media  # noqa: PLC0415

    with Workspace() as ws:
        source = ws.path("source.wav")
        sf.write(source, np.stack([_tone(4.0, seed=3), _tone(4.0, seed=4)], axis=1), SR)
        path = ws.path("core.mkv")
        made = subprocess.run(
            [ffmpeg_path(), "-v", "error", "-y", "-i", source, "-ar", "48000", "-c:a", "dca", "-strict", "-2", path],
            capture_output=True,
        )
        if made.returncode != 0:
            print("        (skipped: this FFmpeg has no DTS encoder)")
            return
        commands = []
        original = media._run

        def recording(command, *args, **kwargs):
            commands.append(list(command))
            return original(command, *args, **kwargs)

        media._run = recording
        try:
            core = load_audio(path, SR, duration=2.0, offset=1.0)
        finally:
            media._run = original
        decodes = [c for c in commands if "f32le" in c]
        assert decodes and "-core_only" in decodes[-1], decodes
        assert core.size > 0


def test_a_dts_track_without_a_core_is_decoded_in_full():
    """DTS-HD MA can be lossless with no core at all, which a core-only
    decode returns nothing for; the full decode is then the answer."""
    import audiosync.media as media  # noqa: PLC0415

    with Workspace() as ws:
        path = ws.path("plain.wav")
        sf.write(path, _tone(4.0, seed=5), SR)
        tried = []
        original_decode, original_is_dts = media._decode, media._is_dts

        def no_core(*args, core_only=False, **kwargs):
            tried.append(core_only)
            if core_only:
                raise MediaError("No audio decoded from plain.wav")
            return original_decode(*args, core_only=core_only, **kwargs)

        media._decode, media._is_dts = no_core, lambda path, track: True
        try:
            samples = load_audio(path, SR, duration=2.0, offset=1.0)
        finally:
            media._decode, media._is_dts = original_decode, original_is_dts
        assert tried == [True, False], tried
        assert len(samples) == 2 * SR
