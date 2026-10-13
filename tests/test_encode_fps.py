"""Tests for the FPS change: atempo, output names, frame-rate detection and
the Target-video suggestion.

Ported from FPSConv's ``tests/test_engine.py`` (AtempoChain, OutputName) and
its four module-level fps tests, which ``unittest`` never ran. The expected
strings are FPSConv's, so a file converted here is named and re-timed exactly
as FPSConv named and re-timed it.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import fps  # noqa: E402
from audiosync.encode.formats import AUDIO_EXTS  # noqa: E402


# FPSConv: AtempoChain.test_fps_py_ratios_are_single_atempo
def test_fps_py_ratios_are_single_atempo():
    for key, ratio in fps.FPS_CONVERSIONS.items():
        assert re.match(r"^atempo=\d\.\d{6}$", fps.atempo_chain(ratio)), key


# FPSConv: AtempoChain.test_identity_is_anull
def test_identity_is_anull():
    assert fps.atempo_chain(1.0) == "anull"


# FPSConv: AtempoChain.test_large_ratio_is_chained
def test_large_ratio_is_chained():
    assert fps.atempo_chain(4.0) == "atempo=2.0,atempo=2.000000"
    assert fps.atempo_chain(0.2) == "atempo=0.5,atempo=0.5,atempo=0.800000"


# FPSConv: AtempoChain.test_garbage_falls_back
def test_garbage_falls_back():
    assert fps.atempo_chain("x") == "atempo=1.0"


# FPSConv: OutputName.test_matches_fps_py
def test_matches_fps_py():
    assert fps.output_name("/x/movie.mka", "23.976-25", "ac3") == "movie_23_976-25.ac3"
    assert fps.output_name("/x/movie.mka", "24-25", "aac") == "movie_24-25.aac"
    assert fps.output_name("/x/movie.mka", "24-25", "truehd") == "movie_24-25.thd"


# FPSConv: OutputName.test_second_stream_is_tagged
def test_second_stream_is_tagged():
    assert fps.output_name("/x/m.mkv", "24-25", "eac3", 1) == "m_a1_24-25.ec3"


# FPSConv: test_fps_label_rounds_to_known_rates (orphan)
def test_fps_label_rounds_to_known_rates():
    assert fps.fps_label(24000 / 1001) == "23.976"
    assert fps.fps_label(24.0) == "24"
    assert fps.fps_label(30000 / 1001) == "29.97"
    assert fps.fps_label(0) is None
    assert fps.fps_label("x") is None


# FPSConv: test_detect_fps_prefers_video_track_then_tags_then_name (orphan)
def test_detect_fps_prefers_video_track_then_tags_then_name():
    video = {"streams": [{"codec_type": "audio"}, {"codec_type": "video", "avg_frame_rate": "24000/1001"}]}
    assert fps.detect_fps(video, "movie.25fps.mkv") == ("23.976", "video")
    tagged = {"format": {"tags": {"FPS": "25"}}, "streams": [{"codec_type": "audio"}]}
    assert fps.detect_fps(tagged, "a.mka") == ("25", "tag")
    assert fps.detect_fps({"streams": []}, "Movie.2019.23.976fps.DDP5.1.mka") == ("23.976", "name")
    assert fps.detect_fps({"streams": []}, "Show.S01E24.mka") == (None, "")
    # a cover-art "video" stream is not a frame rate
    art = {"streams": [{"codec_type": "video", "avg_frame_rate": "90000/1", "disposition": {"attached_pic": 1}}]}
    assert fps.detect_fps(art, "x.m4a") == (None, "")


# FPSConv: test_suggest_from_known_frame_rates (orphan)
def test_suggest_from_known_frame_rates():
    sg = fps.suggest_conversion("23.976", 0, "25", 0)
    assert sg["conv_type"] == "23.976-25"
    assert fps.suggest_conversion("25", 0, "25", 0)["conv_type"] is None


# FPSConv: test_suggest_from_durations_when_audio_has_no_fps (orphan)
def test_suggest_from_durations_when_audio_has_no_fps():
    video = 5400.0                       # 1 h 30 at 25 fps
    audio = video * 25 / 24              # the same cut at 24 fps runs longer
    sg = fps.suggest_conversion(None, audio, "25", video)
    assert sg["conv_type"] == "24-25"
    # tells 23.976 and 24 apart (they differ by 0.1 %)
    sg = fps.suggest_conversion(None, video * 25 / (24000 / 1001), "25", video)
    assert sg["conv_type"] == "23.976-25"
    # nothing sensible when the durations are unrelated
    assert fps.suggest_conversion(None, 100.0, "25", 5400.0) is None
    # same length already
    assert fps.suggest_conversion(None, 5400.5, None, 5400.0)["conv_type"] is None


# ── beyond FPSConv's suite ──


def test_ratios_are_fpsconvs_floats_bit_for_bit():
    """Built from exact fractions, the ratios still equal FPSConv's float arithmetic."""
    fpsconv = {
        "23.976-24": 24 / (24000 / 1001), "23.976-25": 25 / (24000 / 1001),
        "24-23.976": (24000 / 1001) / 24, "24-25": 25 / 24,
        "25-23.976": (24000 / 1001) / 25, "25-24": 24 / 25,
    }
    assert fps.FPS_CONVERSIONS == fpsconv
    assert list(fps.FPS_CONVERSIONS) == list(fpsconv)   # suggestion order depends on it


def test_codec_class_keeps_dolby_and_sends_the_rest_to_aac():
    assert [fps.codec_class(c) for c in ("eac3", "ac3", "truehd", "dts", "flac", "pcm_s24le", "", None)] == [
        "eac3", "ac3", "truehd", "aac", "aac", "aac", "aac", "aac"]


def test_one_extension_list_covers_what_fpsconvs_dialog_missed():
    """Fix #5: FPSConv's file dialog filter lacked these, though drops accepted them."""
    assert {".eb3", ".mlp", ".dts", ".dtshd", ".w64"} <= AUDIO_EXTS
