"""Tests for applying corrections.

The critical property: muxing with a measured delay must produce a file whose
residual offset is ~0. This is verified by a real ffmpeg round-trip rather than
by inspecting the command string, because an inverted itsoffset would look
perfectly reasonable in text while doubling the desync in practice.
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
from audiosync.media import ffmpeg_path, ffprobe_path, load_audio, probe  # noqa: E402
from audiosync.mux import (  # noqa: E402
    MuxPlan,
    apply_correction,
    build_command,
    command_string,
    plan_correction,
)

SR = 16000


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="audiosync-mux-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def _speechlike(seconds, seed):
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


def _make_video(workspace, name, audio, seconds):
    """Build a small real video whose audio track is the given signal."""
    wav = workspace.path(f"{name}.wav")
    sf.write(wav, audio, SR)
    out = workspace.path(f"{name}.mp4")
    subprocess.run(
        [
            ffmpeg_path(), "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
            "-i", wav,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", out,
        ],
        capture_output=True, check=True,
    )
    return out


def test_positive_delay_trims_audio_head():
    """A positive delay (audio late) must DISCARD that much audio from the start.

    Without a rate change this is an input -ss, which keeps the audio copyable.
    """
    plan = MuxPlan("/v.mkv", "/a.ac3", "/out.mkv", delay_ms=250.0)
    command = build_command(plan)
    assert "-ss" in command, f"no trim applied: {command}"
    ss_index = command.index("-ss")
    assert ss_index < command.index("/a.ac3"), "-ss must apply to the audio input"
    assert abs(float(command[ss_index + 1]) - 0.25) < 1e-6
    assert "adelay" not in " ".join(command), "must not also pad a late track"


def test_negative_delay_pads_audio_with_silence():
    """A negative delay (audio early) must INSERT silence at the start."""
    plan = MuxPlan("/v.mkv", "/a.ac3", "/out.mkv", delay_ms=-250.0)
    rendered = " ".join(build_command(plan))
    assert "adelay=250" in rendered, f"no silence padding applied: {rendered}"
    assert "atrim" not in rendered and "-ss" not in rendered, "must not trim an early track"


def test_roundtrip_correction_removes_the_offset():
    """End-to-end: measure a desynced pair, apply the fix, re-measure ~0."""
    with Workspace() as workspace:
        base = _speechlike(20.0, seed=42)
        true_offset_ms = 400.0
        delay_samples = int(true_offset_ms / 1000.0 * SR)

        video = _make_video(workspace, "source", base, 20.0)
        late_audio = np.concatenate(
            [np.zeros(delay_samples, dtype=np.float32), base]
        )
        audio_path = workspace.path("dub.wav")
        sf.write(audio_path, late_audio, SR)

        measured = analyze_pair(video, audio_path, window_s=6.0, window_count=4)
        assert measured.error is None, f"measurement failed: {measured.error}"
        assert abs(measured.delay_ms - true_offset_ms) < 15.0, (
            f"measured {measured.delay_ms:+.1f}ms, expected {true_offset_ms:+.1f}ms"
        )

        plan = plan_correction(
            video, audio_path, measured.delay_ms,
            output_path=workspace.path("fixed.mkv"),
        )
        output = apply_correction(plan)
        assert os.path.isfile(output) and os.path.getsize(output) > 0

        info = probe(output)
        assert info.has_video and info.has_audio, "output lost a stream"

        residual = analyze_pair(video, output, window_s=6.0, window_count=4)
        assert residual.error is None, f"re-measurement failed: {residual.error}"
        assert abs(residual.delay_ms) < 30.0, (
            f"correction left {residual.delay_ms:+.1f}ms residual "
            f"(started at {true_offset_ms:+.1f}ms) -- sign may be inverted"
        )


def test_refuses_to_overwrite_source():
    with Workspace() as workspace:
        base = _speechlike(3.0, seed=1)
        video = _make_video(workspace, "src", base, 3.0)
        audio = workspace.path("a.wav")
        sf.write(audio, base, SR)
        plan = MuxPlan(video, audio, video, delay_ms=10.0)
        try:
            apply_correction(plan)
        except Exception as exc:
            assert "overwrite" in str(exc).lower() or "in place" in str(exc).lower()
        else:
            raise AssertionError("overwrote the source video")


def test_refuses_existing_output_without_flag():
    with Workspace() as workspace:
        base = _speechlike(3.0, seed=2)
        video = _make_video(workspace, "src", base, 3.0)
        audio = workspace.path("a.wav")
        sf.write(audio, base, SR)
        existing = workspace.path("out.mkv")
        open(existing, "w").close()
        try:
            apply_correction(MuxPlan(video, audio, existing, delay_ms=10.0))
        except Exception as exc:
            assert "exists" in str(exc).lower()
        else:
            raise AssertionError("clobbered an existing output file")


def test_rate_correction_precedes_shift():
    """atempo must come before the shift.

    atempo rescales every timestamp after it, so a shift applied first is itself
    scaled -- leaving a proportional residual offset even after the drift is
    removed. Found by round-tripping a drifting pair: 150ms of residual delay
    survived an otherwise successful correction.
    """
    plan = plan_correction(
        "/v.mkv", "/a.ac3", delay_ms=500.0, drift_ms_per_s=0.5, output_path="/o.mkv"
    )
    rendered = " ".join(build_command(plan))
    assert "atempo" in rendered and "atrim" in rendered
    assert rendered.index("atempo") < rendered.index("atrim"), (
        f"shift applied before rate correction: {rendered}"
    )


def test_drift_plan_sets_resample_ratio():
    plan = plan_correction(
        "/v.mkv", "/a.ac3", delay_ms=0.0, drift_ms_per_s=0.5,
        output_path="/out.mkv",
    )
    assert plan.needs_resample, "drift did not produce a resample step"
    command = build_command(plan)
    assert any("atempo" in part for part in command), "no atempo filter for drift"


def test_no_drift_copies_audio_stream():
    plan = plan_correction("/v.mkv", "/a.ac3", delay_ms=100.0, output_path="/o.mkv")
    command = build_command(plan)
    index = command.index("-c:a")
    assert command[index + 1] == "copy", "audio needlessly re-encoded"


def test_command_string_is_shell_safe():
    plan = MuxPlan("/v/My Show S01.mkv", "/a/dub track.ac3", "/o/out.mkv", 100.0)
    rendered = command_string(plan)
    assert "'/v/My Show S01.mkv'" in rendered or '"/v/My Show S01.mkv"' in rendered, (
        f"spaces not quoted: {rendered}"
    )


_SRT = "1\n00:00:00,500 --> 00:00:01,500\nHello\n\n2\n00:00:02,000 --> 00:00:03,000\nWorld\n"


def _minimal_sup(path):
    """A two-cue HDMV PGS stream, built by hand (ffmpeg cannot encode PGS)."""
    import struct

    def segment(kind, payload, pts):
        return b"PG" + struct.pack(">IIBH", int(pts * 90000), 0, kind, len(payload)) + payload

    def display_set(pts, show, number):
        pcs = struct.pack(">HHBHBBBB", 160, 120, 0x10, number, 0x80 if show else 0, 0, 0, 1 if show else 0)
        if show:
            pcs += struct.pack(">HBBHH", 0, 0, 0, 20, 90)
        out = segment(0x16, pcs, pts) + segment(0x17, struct.pack(">BBHHHH", 1, 0, 20, 90, 40, 10), pts)
        if show:
            out += segment(0x14, bytes([0, 0, 1, 235, 128, 128, 255]), pts)
            rle = (bytes([0x00, 0xC0, 40, 1]) + b"\x00\x00") * 10
            body = struct.pack(">HBB", 0, 0, 0xC0) + (len(rle) + 4).to_bytes(3, "big") + struct.pack(">HH", 40, 10) + rle
            out += segment(0x15, body, pts)
        return out + segment(0x80, b"", pts)

    with open(path, "wb") as handle:
        handle.write(display_set(0.5, True, 0) + display_set(1.5, False, 1))
    return path


def _streams(path):
    result = subprocess.run(
        [ffprobe_path(), "-v", "error", "-show_entries",
         "stream=codec_type,codec_name:stream_tags=language,filename", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    import json

    return json.loads(result.stdout).get("streams") or []


def _video_with_subtitles(workspace, audio, seconds, with_pgs=False):
    """An MKV carrying an SRT track (Japanese), optionally a PGS track, and a
    font attachment -- the shape of a real anime/film release."""
    wav = workspace.path("base.wav")
    sf.write(wav, audio, SR)
    srt = workspace.path("subs.srt")
    with open(srt, "w", encoding="utf-8") as handle:
        handle.write(_SRT)
    font = workspace.path("font.ttf")
    with open(font, "wb") as handle:
        handle.write(b"not really a font")
    inputs = ["-i", srt]
    maps = ["-map", "0:v:0", "-map", "1:a:0", "-map", "2:0"]
    if with_pgs:
        inputs += ["-i", _minimal_sup(workspace.path("subs.sup"))]
        maps += ["-map", "3:0"]
    out = workspace.path("source.mkv")
    subprocess.run(
        [
            ffmpeg_path(), "-nostdin", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
            "-i", wav, *inputs, *maps,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-c:s", "copy", "-metadata:s:s:0", "language=jpn",
            "-attach", font, "-metadata:s:t", "mimetype=font/ttf", "-shortest", out,
        ],
        capture_output=True, check=True,
    )
    return out


def test_correction_keeps_subtitles_and_attachments():
    """Audio sync used to map only 0:v:0 and 1:a:0, so every synced file
    silently lost its subtitle tracks and fonts. The video timeline does not
    change, so they are carried over untouched."""
    with Workspace() as workspace:
        base = _speechlike(4.0, seed=3)
        video = _video_with_subtitles(workspace, base, 4.0, with_pgs=True)
        audio = workspace.path("dub.wav")
        sf.write(audio, base, SR)
        plan = plan_correction(video, audio, 120.0, output_path=workspace.path("synced.mkv"))
        assert "0:s:0" in command_string(plan) and "0:t?" in command_string(plan), command_string(plan)
        output = apply_correction(plan)
        streams = _streams(output)
        subtitles = [s for s in streams if s.get("codec_type") == "subtitle"]
        attachments = [s for s in streams if s.get("codec_type") == "attachment"]
        assert [s.get("codec_name") for s in subtitles] == ["subrip", "hdmv_pgs_subtitle"], streams
        assert (subtitles[0].get("tags") or {}).get("language") == "jpn", "subtitle language lost"
        assert len(attachments) == 1, f"font attachment lost: {streams}"
        assert plan.warnings == [], plan.warnings


def test_correction_to_mp4_drops_only_what_mp4_cannot_hold():
    """PGS cannot go into MP4: it is left out with a warning, the text track
    is converted to mov_text, and the correction still succeeds."""
    with Workspace() as workspace:
        base = _speechlike(4.0, seed=4)
        video = _video_with_subtitles(workspace, base, 4.0, with_pgs=True)
        audio = workspace.path("dub.wav")
        sf.write(audio, base, SR)
        plan = plan_correction(video, audio, 0.0, output_path=workspace.path("synced.mp4"))
        logged = []
        output = apply_correction(plan, log=logged.append)
        codecs = [s.get("codec_name") for s in _streams(output) if s.get("codec_type") == "subtitle"]
        assert codecs == ["mov_text"], codecs
        assert any("hdmv_pgs_subtitle" in w for w in logged), logged
        assert any("attachment" in w for w in logged), logged


def test_unprobed_plan_carries_subtitles_into_matroska_only():
    mkv = build_command(MuxPlan("/v.mkv", "/a.ac3", "/out.mkv", delay_ms=0.0))
    assert "0:s?" in mkv and "0:t?" in mkv and "-c:s" in mkv, mkv
    mp4 = build_command(MuxPlan("/v.mkv", "/a.ac3", "/out.mp4", delay_ms=0.0))
    assert "0:s?" not in mp4 and "0:t?" not in mp4, mp4


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
