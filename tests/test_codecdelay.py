"""Tests for codec delay compensation.

Some codecs decode their output shifted from where the source sat. The shift is
constant, invisible in metadata, and cancels only when both files use the same
family -- so a WEB-DL's AAC measured against a disc rip's E-AC3 came out 5.3ms
wrong at 48kHz, every time, with full confidence.

The end-to-end test here is the one that matters: encode identical content
through two different codecs and check the measurement is unaffected.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiosync.analyze import analyze_pair  # noqa: E402
from audiosync.codecdelay import (  # noqa: E402
    codec_delay_ms,
    is_raw_stream,
    describe,
    relative_codec_delay_ms,
)
from audiosync.media import ffmpeg_path  # noqa: E402

SR = 48000

# Enough to be realistic without making the suite slow.
CODECS = {
    "aac": (["-c:a", "aac", "-b:a", "192k"], "m4a"),
    "eac3": (["-c:a", "eac3", "-b:a", "384k"], "eac3"),
    "flac": (["-c:a", "flac"], "flac"),
}


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="audiosync-codec-")
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


def _encode(source, out, args):
    subprocess.run(
        [ffmpeg_path(), "-nostdin", "-y", "-i", source] + args + [out],
        capture_output=True, check=True,
    )
    return out


def test_delay_is_expressed_in_samples_not_milliseconds():
    """256 samples is a different number of ms at each rate the format allows."""
    at_48k = codec_delay_ms("eac3", 48000, "eac3")
    at_44k = codec_delay_ms("eac3", 44100, "eac3")
    assert abs(at_48k - 256 / 48000 * 1000) < 1e-9
    assert abs(at_44k - 256 / 44100 * 1000) < 1e-9
    assert at_48k != at_44k, "a fixed ms value would be wrong at one of these rates"


def test_codecs_without_delay_return_zero():
    for codec in ("aac", "libopus", "flac", "libmp3lame", "pcm_s16le"):
        assert codec_delay_ms(codec, 48000, "eac3") == 0.0, (
            f"{codec} should need no correction"
        )


def test_unknown_input_is_left_alone():
    """An unrecognised case must not be guessed at."""
    assert codec_delay_ms(None, 48000, "eac3") == 0.0
    assert codec_delay_ms("eac3", None, "eac3") == 0.0
    assert codec_delay_ms("eac3", 0, "eac3") == 0.0
    assert codec_delay_ms("some-future-codec", 48000, "eac3") == 0.0
    # A rate the format cannot actually use means something already resampled it.
    assert codec_delay_ms("eac3", 16000, "eac3") == 0.0


def test_matching_codecs_need_no_correction():
    """The delay is identical on both sides, so it already cancels."""
    for codec in ("eac3", "ac3", "aac"):
        assert relative_codec_delay_ms(codec, 48000, codec, 48000, "eac3", "eac3") == 0.0


def test_correction_direction_matches_measurement():
    """Signs come from measurement: aac reference vs eac3 secondary reads high.

    Measured at 48kHz over identical content, the offset is +5.318ms, so the
    correction must be positive in that direction and negative when reversed.
    """
    forward = relative_codec_delay_ms("aac", 48000, "eac3", 48000, "mov,mp4,m4a", "eac3")
    reverse = relative_codec_delay_ms("eac3", 48000, "aac", 48000, "eac3", "mov,mp4,m4a")
    assert forward > 0, f"expected a positive correction, got {forward:+.3f}"
    assert abs(forward - 5.333) < 0.05, f"expected ~+5.333ms, got {forward:+.3f}"
    assert abs(forward + reverse) < 1e-9, "reversing the pair must negate the correction"


def test_description_only_appears_when_a_correction_was_made():
    assert describe("eac3", 48000, "eac3", 48000, "eac3", "eac3") is None
    text = describe("aac", 48000, "eac3", 48000, "mov,mp4,m4a", "eac3")
    assert text and "codec delay" in text.lower()


def test_cross_codec_measurement_is_accurate_end_to_end():
    """The decisive test: the same offset must be measured through any codec pair.

    Before compensation this was out by 5.3ms at 48kHz whenever AC3 or E-AC3 sat
    on exactly one side -- with full confidence, so nothing flagged it.
    """
    with Workspace() as workspace:
        base = _speech(60, seed=1)
        true_offset_ms = 250.0
        delay = int(true_offset_ms / 1000 * SR)

        reference = workspace.path("ref.wav")
        dub = workspace.path("dub.wav")
        sf.write(reference, base, SR)
        sf.write(dub, np.concatenate([np.zeros(delay, dtype=np.float32), base]), SR)

        encoded = {}
        for role, source in (("ref", reference), ("dub", dub)):
            for name, (args, ext) in CODECS.items():
                encoded[(role, name)] = _encode(
                    source, workspace.path(f"{role}_{name}.{ext}"), args
                )

        failures = []
        for ref_codec, dub_codec in itertools.product(CODECS, CODECS):
            result = analyze_pair(
                encoded[("ref", ref_codec)],
                encoded[("dub", dub_codec)],
                window_s=12,
                window_count=4,
            )
            if result.error:
                failures.append(f"{ref_codec}->{dub_codec}: {result.error}")
                continue
            error = result.delay_ms - true_offset_ms
            if abs(error) > 1.0:
                failures.append(
                    f"{ref_codec} ref vs {dub_codec} dub: "
                    f"{result.delay_ms:+.3f}ms, out by {error:+.3f}ms"
                )

        assert not failures, "Cross-codec errors:\n  " + "\n  ".join(failures)


def test_correction_is_recorded_on_the_result():
    """The adjustment has to be visible, not silently folded into the number."""
    with Workspace() as workspace:
        base = _speech(30, seed=2)
        reference = workspace.path("ref.wav")
        sf.write(reference, base, SR)

        aac = _encode(reference, workspace.path("a.m4a"), CODECS["aac"][0])
        eac3 = _encode(reference, workspace.path("a.eac3"), CODECS["eac3"][0])

        result = analyze_pair(aac, eac3, window_s=10, window_count=3)
        assert result.error is None, f"analysis failed: {result.error}"
        assert abs(result.codec_delay_ms - 5.333) < 0.05, (
            f"expected a ~5.333ms correction, recorded {result.codec_delay_ms:+.3f}"
        )
        assert result.primary_codec == "aac"
        assert result.secondary_codec == "eac3"
        # Identical content through two codecs: the true offset is zero.
        assert abs(result.delay_ms) < 1.0, (
            f"identical content measured {result.delay_ms:+.3f}ms after correction"
        )


def test_no_correction_is_applied_inside_a_container():
    """A container must not receive the raw-stream correction.

    A bare .eac3 has no timestamps, so the decoder's 256 priming samples land
    in the measurement. Whether the same stream inside a container also carries
    them is a property of the ffmpeg build: measured on ffmpeg 9 the container
    reads +0.027ms, while Ubuntu's packaged build still shows the full
    +5.35ms. So the shift cannot be predicted from the file alone.

    What is invariant, and what this pins, is that the *correction* is scoped
    to raw streams. Applying it to a container can only ever add error on a
    build that already trims, and the correction is 5.333ms while the bug it
    was masking was tens of milliseconds.

    The measured-offset assertion is deliberately absent: it would encode one
    ffmpeg's decoding behaviour as a requirement, which is what made the
    original coverage misleading -- it verified the adjustment on a bare
    stream, the one input that needs it, and shipped it to everything else.
    """
    with Workspace() as workspace:
        base = _speech(45, seed=7)
        true_offset_ms = 250.0
        delay = int(true_offset_ms / 1000 * SR)

        reference = workspace.path("ref.wav")
        dub = workspace.path("dub.wav")
        sf.write(reference, base, SR)
        sf.write(dub, np.concatenate([np.zeros(delay, dtype=np.float32), base]), SR)

        aac = _encode(reference, workspace.path("ref.m4a"), CODECS["aac"][0])
        # Same E-AC3 stream, once bare and once wrapped.
        raw = _encode(dub, workspace.path("dub.eac3"), CODECS["eac3"][0])
        wrapped = _encode(dub, workspace.path("dub.mka"), CODECS["eac3"][0])

        in_container = analyze_pair(aac, wrapped, window_s=12, window_count=3)
        assert in_container.error is None, in_container.error
        assert in_container.codec_delay_ms == 0.0, (
            f"a container needs no correction, applied {in_container.codec_delay_ms:+.3f}ms"
        )

        # The bare stream still gets its correction, and lands on the answer.
        bare = analyze_pair(aac, raw, window_s=12, window_count=3)
        assert bare.error is None, bare.error
        assert bare.codec_delay_ms > 0, "a raw stream still needs the correction"
        assert abs(bare.delay_ms - true_offset_ms) < 1.0, (
            f"raw stream measured {bare.delay_ms:+.3f}ms, want {true_offset_ms:+.3f}"
        )


def test_raw_stream_formats_are_recognised():
    assert is_raw_stream("eac3")
    assert is_raw_stream("ac3")
    # ffprobe reports several candidates for ambiguous inputs.
    assert is_raw_stream("ac3,eac3")
    # Real containers, including ones that commonly carry AC3.
    for fmt in ("matroska,webm", "mov,mp4,m4a,3gp,3g2,mj2", "mpegts", "avi", "flac"):
        assert not is_raw_stream(fmt), f"{fmt} is a container"
    # Unknown is treated as a container: skipping a needed correction is the
    # cheaper mistake, since containers vastly outnumber bare streams.
    assert not is_raw_stream(None)
    assert not is_raw_stream("")


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

