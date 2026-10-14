"""Tests for carrying an FPSConv user's setup over: what is taken from its
settings.json and deew's config.toml, what is deliberately left behind,
and that FPSConv's own files are only read.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import fpsconv_import, settings, tools  # noqa: E402

FPSCONV_SETTINGS = {
    "out_dir": r"D:\Encodes",
    "conv_type": "25-23.976",
    "mode": "batch",
    "workers": 3,
    "bitrate": 0,
    "overwrite": "rename",
    "auto_update": True,
    "native_window": True,
    "task": "encode",
    "encode": {"target": "dd", "channels": 6, "bitrate": 448, "atmos": False, "atmos71": "dee",
               "bed_conform": False, "drc": "speech"},
    "tools": {"ffmpeg": "", "ffprobe": "", "deew_python": "", "mediainfo": r"C:\Tools\MediaInfo.exe",
              "deezy": r"C:\Tools\deezy.exe", "deezy_python": "", "truehdd": r"C:\Tools\truehdd.exe",
              "patcher": r"C:\Users\me\AppData\Local\FPSConv\tools\dee-ddp71-atmos-wrapper-v1.1-stable"},
    "dirs": {"temp": r"E:\fast", "work": "", "logs": r"C:\logs"},
}
DEEW_CONFIG = {"dee_path": r"C:\Dolby\DEE\dee.exe", "temp_path": r"C:\x\temp", "logo": 0}


def test_tool_paths_encode_defaults_output_folder_and_dee_come_over():
    with mock.patch.object(fpsconv_import, "fpsconv_cache_dir", return_value=Path(r"C:\Users\me\AppData\Local\FPSConv")):
        out = fpsconv_import.mapped(FPSCONV_SETTINGS, DEEW_CONFIG)
    assert out["out_dir"] == r"D:\Encodes"
    assert (out["conv_type"], out["workers"], out["overwrite"], out["task"]) == ("25-23.976", 3, "rename", "encode")
    assert out["encode"] == FPSCONV_SETTINGS["encode"]
    assert out["tools"] == {"mediainfo": r"C:\Tools\MediaInfo.exe", "deezy": r"C:\Tools\deezy.exe",
                            "truehdd": r"C:\Tools\truehdd.exe"}
    assert out["dirs"] == {"temp": r"E:\fast"}
    assert out["dee_path"] == r"C:\Dolby\DEE\dee.exe"
    for left_behind in ("auto_update", "native_window", "mode"):     # FPSConv's window, updater and UI mode
        assert left_behind not in out, left_behind


def test_fpsconvs_own_downloads_are_left_behind():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "FPSConv" / "settings.json"
        with mock.patch.object(fpsconv_import, "fpsconv_settings_path", return_value=cfg), \
             mock.patch.object(fpsconv_import, "fpsconv_cache_dir", return_value=Path(tmp) / "cache" / "FPSConv"):
            own_ffmpeg = str(Path(tmp) / "FPSConv" / "bin" / "ffmpeg.exe")
            own_patcher = str(Path(tmp) / "cache" / "FPSConv" / "tools" / "x")
            users = str(Path(tmp) / "Tools" / "FPSConv-builds" / "ffprobe.exe")
            out = fpsconv_import.mapped({"tools": {"ffmpeg": own_ffmpeg, "truehdd": own_patcher, "ffprobe": users}}, {})
    assert out["tools"] == {"ffprobe": users}


def test_import_saves_once_and_never_changes_fpsconvs_files():
    root = Path(tempfile.mkdtemp(prefix="encode-import-"))
    try:
        fpsconv = root / "roaming" / "FPSConv" / "settings.json"
        fpsconv.parent.mkdir(parents=True)
        fpsconv.write_text(json.dumps(FPSCONV_SETTINGS), encoding="utf-8")
        before = fpsconv.read_bytes()
        with mock.patch.object(fpsconv_import, "fpsconv_settings_path", return_value=fpsconv), \
             mock.patch.object(settings, "config_dir", return_value=root / "cfg"), \
             mock.patch.object(settings, "cache_dir", return_value=root / "cache"), \
             mock.patch.object(tools, "deew_config_path", return_value=root / "deew" / "config.toml"):
            assert settings.load_settings()["fpsconv_import_offered"] is False
            found = fpsconv_import.find()
            assert found["found"] and found["settings"]["out_dir"] == r"D:\Encodes"
            assert not (root / "cfg" / "settings.json").exists(), "find() only looks"
            fpsconv_import.apply()
            saved = settings.load_settings()
            assert saved["out_dir"] == r"D:\Encodes" and saved["encode"]["drc"] == "speech"
            assert saved["tools"]["truehdd"] == r"C:\Tools\truehdd.exe"
            assert saved["fpsconv_import_offered"] is True
        assert fpsconv.read_bytes() == before
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_nothing_to_import_still_records_the_offer():
    root = Path(tempfile.mkdtemp(prefix="encode-import-"))
    try:
        with mock.patch.object(fpsconv_import, "fpsconv_settings_path", return_value=root / "none.json"), \
             mock.patch.object(settings, "config_dir", return_value=root / "cfg"), \
             mock.patch.object(tools, "deew_config_path", return_value=root / "deew" / "config.toml"):
            assert fpsconv_import.find()["found"] is False
            assert fpsconv_import.apply()["found"] is False
            assert settings.load_settings()["fpsconv_import_offered"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)
