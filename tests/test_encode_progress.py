"""Tests for reading progress and errors out of the encode tools' output.

Ported from FPSConv's ``tests/test_engine.py`` (ErrorPicking, the DeeZy line
check from CommandBuilders, and Flat71's patcher parser).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import fps, progress  # noqa: E402


# FPSConv: ErrorPicking.test_prefers_informative_line
def test_prefers_informative_line():
    lines = ["[adts @ 0x1] channelConfiguration > 7 is not supported in ADTS",
             "size=0KiB time=N/A", "Conversion failed!"]
    assert progress.pick_error(lines) == "channelConfiguration > 7 is not supported in ADTS"


# FPSConv: ErrorPicking.test_falls_back_to_last_line
def test_falls_back_to_last_line():
    assert progress.pick_error(["a", "b"]) == "b"
    assert progress.pick_error([]) == ""


# FPSConv: CommandBuilders.test_deezy_progress_line
def test_deezy_progress_line():
    m = progress.DEEZY_LINE.match("truehdd (1 of 3)       42.5%")
    assert (m.group(1), m.group(4)) == ("truehdd", "42.5")
    m = progress.DEEZY_LINE.match("W1: DEE encode (3 of 3) 80.0%")
    assert (m.group(1), m.group(4)) == ("DEE encode", "80.0")
    assert progress.DEEZY_LINE.match("Dee job: something 100% done") is None
    m = progress.TRUEHDD_PCT.match("[truehdd-err] Decoding 37.5% ...")
    assert m.group(1) == "37.5"
    assert progress.TRUEHDD_PCT.match("[ffmpeg] size=  1024kB time=00:00:10 bitrate=100%") is None


# FPSConv: Flat71.test_patcher_parser_maps_dee_progress
def test_patcher_parser_maps_dee_progress():
    p = progress.PatcherParser()
    assert p("Run directory: /x") == ("Patching DEE", None)
    assert p("[2026] INFO: encode_to_atmos_ddp: Starting measurement.") == ("DEE measure", 0.0)
    assert p("Overall progress: 20.0.") == ("DEE measure", 50.0)
    assert p("[2026] INFO: encode_to_atmos_ddp: Measurement done. Final Integrated loudness: -21.8.") == ("DEE encode", 0.0)
    assert p("Overall progress: 70.0.") == ("DEE encode", 50.0)
    assert p("patched AC-3 frames: 1250") == ("Surround EX flag", 100.0)
    assert p("INFO: CPU: 17.7 %") is None
    assert progress.parse_truehdd_line("Decoding [=====>   ] 42%") == ("truehdd decode", 42.0)


# ── beyond FPSConv's suite ──


def test_fps_progress_reaches_100_on_a_speed_up():
    """Fix #2: ffmpeg's out_time counts the shorter output, so divide by the stretched length."""
    source = 6000.0
    ratio = fps.FPS_CONVERSIONS["23.976-25"]
    end = source / ratio                                  # where ffmpeg's out_time stops
    assert progress.ffmpeg_percent(end, source, ratio) == 100.0
    assert round(end / source * 100, 1) == 95.9           # what FPSConv showed at the end
    assert progress.ffmpeg_percent(end / 2, source, ratio) == 50.0
    assert progress.ffmpeg_percent(10, 0) == 0.0


def test_deew_rich_bar_line():
    text = progress.ANSI_RE.sub("", "\x1b[1m[ DEE: measure | fpsconv_x.wav ]\x1b[0m ━━━━━━  42.10%")
    assert progress.parse_deew_line(text) == ("DEE: measure", 42.1)
    assert progress.parse_deew_line("deew 3.3.0") is None


def test_time_and_size_text():
    assert progress.fmt_time(3725.9) == "01:02:05"
    assert progress.parse_time("01:02:05") == 3725.0
    assert progress.parse_time("garbage") == 0.0
    assert progress.hr_size(1536) == "1.50 KB"
    assert progress.hr_size(None) == "0 B"
