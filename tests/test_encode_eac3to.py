"""Tests for eac3to doing the decoding, re-timing and TrueHD extraction:
its command lines, picking the right track out of its listing, its
progress lines, and an FPS job run end to end through a stand-in eac3to
that re-times the way eac3to does (by resampling, so pitch moves too).
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from audiosync.encode import eac3to, fps  # noqa: E402
from audiosync.encode.service import AudioQueue  # noqa: E402
from audiosync.media import ffmpeg_path, ffprobe_path  # noqa: E402
from test_encode_service import Events, _sine, _spec, sandbox  # noqa: E402

LISTING = """MKV, 1 video track, 3 audio tracks, 1 subtitle track, 2:43:48, 24p /1.001
1: Chapters, 28 chapters
2: h264/AVC, 1080p24 /1.001 (16:9)
3: TrueHD/AC3 (Atmos), English, 7.1 channels, 48kHz
4: AC3, Hindi, 5.1 channels, 448kbps, 48kHz, dialnorm: -27dB
5: E-AC3, English, 5.1 channels, 640kbps, 48kHz
6: Subtitle (PGS), English
"""


def test_tracks_skip_chapters_video_and_subtitles():
    assert [(n, c) for n, _d, c in eac3to.audio_tracks(LISTING)] == [(3, 2), (4, 1), (5, 1)]


def test_ffprobe_stream_index_maps_to_the_eac3to_track():
    # ffprobe sees TrueHD (0), its AC-3 core (1), AC3 (2), E-AC3 (3)
    assert eac3to.track_for(LISTING, 0) == (3, [])
    assert eac3to.track_for(LISTING, 1) == (3, ["-core"])
    assert eac3to.track_for(LISTING, 2) == (4, [])
    assert eac3to.track_for(LISTING, 3) == (5, [])
    assert eac3to.track_for(LISTING, 4) is None


def test_fps_flags_name_both_rates():
    assert eac3to.fps_flags("23.976-25") == ["-23.976", "-changeTo25.000"]
    assert eac3to.fps_flags("25-24") == ["-25.000", "-changeTo24.000"]
    for key in fps.FPS_CONVERSIONS:
        assert len(eac3to.fps_flags(key)) == 2, key


def test_command_lines():
    assert eac3to.fps_wav("eac3to", "D:/m.mkv", 4, [], "25-23.976", "C:/t/x.w64") == [
        "eac3to", "D:/m.mkv", "4:", "C:/t/x.w64", "-25.000", "-changeTo23.976", "-progressnumbers"]
    assert eac3to.fps_wav("eac3to", "D:/m.ac3", None, [], "24-25", "C:/t/x.w64") == [
        "eac3to", "D:/m.ac3", "C:/t/x.w64", "-24.000", "-changeTo25.000", "-progressnumbers"]
    assert eac3to.encode_wav("eac3to", "D:/m.mkv", 3, [], 96000, "C:/t/x.w64") == [
        "eac3to", "D:/m.mkv", "3:", "C:/t/x.w64", "-resampleTo48000", "-progressnumbers"]
    assert eac3to.encode_wav("eac3to", "D:/m.mkv", 3, ["-core"], 48000, "C:/t/x.w64") == [
        "eac3to", "D:/m.mkv", "3:", "C:/t/x.w64", "-core", "-progressnumbers"]
    assert eac3to.thd_extract("eac3to", "D:/m.m2ts", 3, "C:/t/x.thd") == [
        "eac3to", "D:/m.m2ts", "3:", "C:/t/x.thd", "-progressnumbers"]


def test_what_eac3to_reads():
    for name in ("a.mkv", "a.MKA", "a.m2ts", "a.ts", "a.thd", "a.dtshd", "a.flac", "a.wav"):
        assert eac3to.can_read(name), name
    for name in ("a.mp4", "a.m4a", "a.mov", "a.webm", "a.opus", "a.ogg", "a.aac"):
        assert not eac3to.can_read(name), name
    assert eac3to.is_container("a.mkv") and not eac3to.is_container("a.ac3")


def test_progress_lines():
    parse = eac3to.parse_line("Extracting WAV")
    assert parse("process: 42%") == ("Extracting WAV", 42.0)
    assert parse("analyze: 7") == ("Extracting WAV", 7.0)
    assert parse("[a04] Decoding with libav / FFmpeg...") is None
    assert parse("3: TrueHD/AC3, English, 7.1 channels") is None


# A stand-in eac3to: lists one AAC track for a container, and for a conversion
# resamples the way eac3to does (asetrate), so the output is shorter and higher.
FAKE = r'''
import os, subprocess, sys
args = sys.argv[1:]
FFMPEG = os.environ["FAKE_EAC3TO_FFMPEG"]
if len(args) == 1:
    print("MKV, 1 audio track, 0:00:04")
    print("1: AAC, English, 2.0 channels, 48kHz")
    sys.exit(0)
src, rest = args[0], args[1:]
if rest and rest[0].endswith(":"):
    rest = rest[1:]
out, opts = rest[0], rest[1:]
rate = {"23.976": 24000 / 1001, "24.000": 24.0, "25.000": 25.0}
frm = next((rate[o[1:]] for o in opts if o[1:] in rate), None)
to = next((rate[o[9:]] for o in opts if o.startswith("-changeTo")), None)
filters = ["-af", f"asetrate=48000*{to / frm},aresample=48000"] if frm and to else []
for p in (25, 50, 75):
    print(f"process: {p}%", flush=True)
subprocess.run([FFMPEG, "-y", "-v", "error", "-i", src, *filters, "-c:a", "pcm_s24le", "-f", "w64", out], check=True)
print("process: 100%")
'''


def _fake_eac3to(root) -> str:
    script = root / "fake_eac3to.py"
    script.write_text(FAKE, encoding="utf-8")
    if os.name == "nt":
        launcher = root / "eac3to.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher = root / "eac3to"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return str(launcher)


def _seconds(path: str) -> float:
    """A raw .aac's real length. ffprobe's format duration only estimates it
    from the bitrate, and how far off depends on the FFmpeg build: 102 ms on
    Ubuntu 24.04's 6.1, enough to fail a 0.1 s check on a track of the right
    length. Count the 1024-sample frames instead and drop the encoder's one
    frame of priming, as test_encode_service does."""
    out = subprocess.run([ffprobe_path(), "-v", "error", "-count_packets", "-select_streams", "a:0",
                          "-show_entries", "stream=nb_read_packets,sample_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    rate, packets = (int(v) for v in out.strip().split(","))
    return (packets - 1) * 1024 / rate


def test_an_fps_job_runs_through_eac3to_then_ffmpeg_for_aac():
    with sandbox() as root:
        m4a = _sine(root / "film.m4a", 4.0)
        mka = root / "film.mka"
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", str(m4a), "-c", "copy", str(mka)], check=True)
        os.environ.setdefault("FAKE_EAC3TO_FFMPEG", ffmpeg_path())
        events = Events()
        queue = AudioQueue(events, workers=1, tool_overrides={"eac3to": _fake_eac3to(root)})
        queue.add([_spec("j", mka, root / "out")])
        assert queue.wait_idle(60), "never finished"
        done = events.of("audioJobDone", "j")[0]
        assert done["state"] == "done", done["error"]
        assert os.path.basename(done["output"]) == "film_23_976-25.aac"     # FPSConv's name, unchanged
        log = "\n".join(events.of("audioJobLog", "j")[-1]["lines"])
        assert "1: " in log and "-23.976 -changeTo25.000 -progressnumbers" in log, log
        assert "pitch moves with speed" in log and "atempo=" not in log, log
        expected = 4.0 / fps.FPS_CONVERSIONS["23.976-25"]
        assert abs(_seconds(done["output"]) - expected) < 0.1, _seconds(done["output"])


def test_an_mp4_source_stays_on_ffmpeg_atempo():
    with sandbox() as root:
        m4a = _sine(root / "film.m4a", 2.0)
        os.environ.setdefault("FAKE_EAC3TO_FFMPEG", ffmpeg_path())
        events = Events()
        queue = AudioQueue(events, workers=1, tool_overrides={"eac3to": _fake_eac3to(root)})
        queue.add([_spec("j", m4a, root / "out")])
        assert queue.wait_idle(60)
        done = events.of("audioJobDone", "j")[0]
        assert done["state"] == "done", done["error"]
        log = "\n".join(events.of("audioJobLog", "j")[-1]["lines"])
        assert "-af atempo=1.042708" in log and "eac3to" not in log, log
