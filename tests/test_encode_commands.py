"""Tests for every command line an encode or FPS job runs, and deew's config.

Ported from FPSConv's ``tests/test_engine.py`` (CommandBuilders, Flat71's
command check, DeewConfig). The argv lists are FPSConv's, flag for flag: a
job here runs exactly the commands FPSConv ran.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import commands, formats, fps, settings, tools  # noqa: E402

TOOLS = {"ffmpeg": "", "ffprobe": "", "deew_python": "/venv/python",
         "deezy": r"C:\Tools\deezy.exe", "deezy_python": "", "mediainfo": "", "truehdd": ""}


# FPSConv: CommandBuilders.test_deew_argv
def test_deew_argv():
    deew = tools.deew_cmd(TOOLS)
    e = formats.resolve_encode("ddp", 6, True, 0, 8, False)
    assert commands.deew_encode(deew, "/w/t.wav", e, "film_light", "/w") == [
        "/venv/python", "-m", "deew", "-i", "/w/t.wav", "-f", "ddp", "-b", "1024",
        "-r", "film_light", "-o", "/w", "-np", "-dm", "6"]
    e = formats.resolve_encode("ddp", 8, True, 1536, 8, False)
    cmd = commands.deew_encode(deew, "/w/t.wav", e, "bogus", "/w")
    assert cmd[5:10] == ["-f", "ddp", "-b", "1536", "-r"]
    assert "film_light" in cmd                       # unknown DRC falls back


# FPSConv: CommandBuilders.test_deezy_argv
def test_deezy_argv():
    e = formats.resolve_encode("ddp", 0, True, 0, 8, True, "truehd")
    cmd = commands.deezy_atmos(tools.deezy_cmd(TOOLS), "/in/m.mkv", 1, e, "film_standard", "/w", "/o/m.ec3",
                               "/cfg/deezy-work", {"ffmpeg": "/bin/ffmpeg", "dee": r"C:\DEE\dee.exe", "truehdd": ""})
    assert cmd == [r"C:\Tools\deezy.exe", "--no-progress-bars", "--log-level", "debug", "encode", "atmos",
                   "--ffmpeg", "/bin/ffmpeg", "--dee", r"C:\DEE\dee.exe",
                   "--atmos-mode", "bluray", "--bitrate", "1536",
                   "--track-index", "a:1", "--drc-line-mode", "film_standard",
                   "--working-dir", "/cfg/deezy-work", "--temp-dir", os.path.abspath("/w"),
                   "--output", "/o/m.ec3", "--overwrite", "/in/m.mkv"]


# FPSConv: CommandBuilders.test_deezy_cmd_prefers_exe_then_python
def test_deezy_cmd_prefers_exe_then_python():
    t = dict(TOOLS)
    assert tools.deezy_cmd(t) == [r"C:\Tools\deezy.exe"]
    t["deezy"] = ""
    t["deezy_python"] = "/venv/python"
    assert tools.deezy_cmd(t) == ["/venv/python", "-m", "deezy"]


# FPSConv: CommandBuilders.test_wav_argv
def test_wav_argv():
    e = formats.resolve_encode("ddp", 0, True, 0, 6, False)
    cmd = commands.encode_wav("ffmpeg", "/in/m.mkv", 0, {"codec": "eac3", "sample_rate": 48000}, e, "/w/t.wav")
    assert cmd[:5] == ["ffmpeg", "-y", "-nostdin", "-drc_scale", "0"]   # no decoder DRC on Dolby sources
    assert "-af" not in cmd
    assert cmd[cmd.index("-ac") + 1] == "6"
    cmd = commands.encode_wav("ffmpeg", "/in/m.mkv", 0, {"codec": "truehd", "sample_rate": 96000}, e, "/w/t.wav")
    assert "-drc_scale" not in cmd
    assert "aresample=out_sample_rate=48000:resampler=soxr:precision=28" in cmd
    cmd = commands.encode_wav("ffmpeg", "/in/m.mkv", 0, {"codec": "truehd", "sample_rate": 96000}, e, "/w/t.wav",
                              soxr=False)
    assert "-af" not in cmd
    assert cmd[cmd.index("-ar") + 1] == "48000"


# FPSConv: CommandBuilders.test_deew_output_accepts_eb3
def test_deew_output_accepts_eb3():
    with tempfile.TemporaryDirectory() as tmp:
        assert commands.deew_output(tmp, "fpsconv_x") is None
        open(os.path.join(tmp, "fpsconv_x.eb3"), "w").close()
        assert commands.deew_output(tmp, "fpsconv_x") == os.path.join(tmp, "fpsconv_x.eb3")


# FPSConv: Flat71.test_commands
def test_commands():
    assert commands.truehdd("/t/truehdd", "/w/in.thd", "/w/master") == [
        "/t/truehdd", "--progress", "decode", "--output-path", "/w/master", "--warp-mode", "normal",
        "--bed-conform", "/w/in.thd"]
    assert "--bed-conform" not in commands.truehdd("/t/truehdd", "/w/in.thd", "/w/master", bed_conform=False)
    assert commands.thd_extract("ffmpeg", "/in/m.mkv", 1, "/w/x.thd") == [
        "ffmpeg", "-y", "-nostdin", "-i", "/in/m.mkv", "-map", "0:a:1", "-vn", "-sn", "-dn",
        "-c:a", "copy", "-f", "truehd", "/w/x.thd"]
    e = formats.resolve_encode("ddp", 8, True, 1536, 8, True, "truehd")
    runner = ["/pack/python"]
    script = Path("/p/dee-ddp71-atmos-wrapper.py")
    cmd = commands.patcher(runner, script, r"C:\DEE\dee.exe", "/w/master.atmos", "/o/m.ec3", e,
                           "film_standard", "/w/dee")
    assert cmd[1:] == [str(script), r"C:\DEE\dee.exe", "/w/master.atmos", "/o/m.ec3",
                       "--compatibility-layout", "flat-7.1", "--data-rate", "1536",
                       "--line-mode-drc-profile", "film_standard", "--rf-mode-drc-profile", "film_standard",
                       "--temp-dir", "/w/dee", "--overwrite"]
    assert cmd[0] == runner[0]


# FPSConv: DeewConfig.test_config_uses_our_temp_dir_and_quotes_paths
def test_config_uses_our_temp_dir_and_quotes_paths():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(settings, "config_dir", return_value=Path(tmp)), \
             mock.patch.object(settings, "cache_dir", return_value=Path(tmp) / "cache"), \
             mock.patch.object(tools, "deew_config_path", return_value=Path(tmp) / "deew" / "config.toml"):
            path = tools.write_deew_config(r"C:\Dolby\DEE\dee.exe")
            data = tools.read_deew_config()
            assert data["dee_path"] == r"C:\Dolby\DEE\dee.exe"
            assert data["temp_path"] == str(Path(tmp) / "cache" / "temp" / "deew")
            assert (Path(tmp) / "cache" / "temp" / "deew").is_dir()
            # a Temp folder chosen in Preferences moves deew's temp with it
            settings.save_settings({"dirs": {"temp": str(Path(tmp) / "fast")}})
            tools.write_deew_config(r"C:\Dolby\DEE\dee.exe")
            assert tools.read_deew_config()["temp_path"] == str(Path(tmp) / "fast" / "deew")
            assert data["logo"] == 0
            assert path.exists()


# ── beyond FPSConv's suite ──


def test_fps_commands_match_fpsconvs_convert():
    """The argv FPSConv's convert() built inline, for an AAC and a Dolby source."""
    ratio = fps.FPS_CONVERSIONS["23.976-25"]
    assert commands.fps_aac("ffmpeg", "/in/a.mka", 0, 192, 2, ratio, "/o/a_23_976-25.aac") == [
        "ffmpeg", "-y", "-nostdin", "-i", "/in/a.mka", "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-map_metadata", "-1", "-map_chapters", "-1", "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-af", "atempo=1.042708", "/o/a_23_976-25.aac"]
    assert commands.fps_wav("ffmpeg", "/in/a.mka", 1, 6, fps.FPS_CONVERSIONS["25-23.976"], "/w/fpsconv_x.wav") == [
        "ffmpeg", "-y", "-nostdin", "-i", "/in/a.mka", "-map", "0:a:1", "-vn", "-sn", "-dn",
        "-map_metadata", "-1", "-map_chapters", "-1", "-c:a", "pcm_s24le", "-ar", "48000", "-ac", "6",
        "-af", "atempo=0.959041", "-rf64", "auto", "/w/fpsconv_x.wav"]
    assert commands.deew_fps(["/venv/python", "-m", "deew"], "/w/fpsconv_x.wav", "dd", 640, "/w") == [
        "/venv/python", "-m", "deew", "-i", "/w/fpsconv_x.wav", "-f", "dd", "-b", "640", "-o", "/w", "-np"]


def test_progress_flag_goes_before_the_output():
    cmd = commands.with_progress(["ffmpeg", "-i", "a", "out.aac"])
    assert cmd == ["ffmpeg", "-i", "a", "-progress", "pipe:1", "out.aac"]
    assert commands.with_progress(cmd) == cmd


def test_deezy_falls_back_to_path_before_this_interpreter():
    with mock.patch.object(shutil, "which", return_value="/usr/bin/deezy"):
        assert tools.deezy_cmd({}) == ["deezy"]
