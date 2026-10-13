"""Tests for the encode settings file the app and the CLI share.

Ported from FPSConv's ``tests/test_engine.py`` (Settings, Folders, and the
nested-settings check from JobModel).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import settings, tools  # noqa: E402


# FPSConv: Settings.test_roundtrip_in_temp_dir
def test_roundtrip_in_temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(settings, "config_dir", return_value=Path(tmp)):
            assert settings.load_settings()["conv_type"] == "23.976-25"
            saved = settings.save_settings({"conv_type": "24-25", "tools": {"ffmpeg": "/x/ffmpeg"}, "junk": 1})
            assert saved["conv_type"] == "24-25"
            assert saved["tools"]["ffmpeg"] == "/x/ffmpeg"
            assert "junk" not in saved
            assert settings.load_settings()["tools"]["ffmpeg"] == "/x/ffmpeg"
            settings.save_history([{"id": "a"}] * 300)
            assert len(settings.load_history()) == 200


# FPSConv: JobModel.test_nested_encode_settings_roundtrip
def test_nested_encode_settings_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(settings, "config_dir", return_value=Path(tmp)):
            saved = settings.save_settings({"task": "encode",
                                            "encode": {"target": "dd", "channels": "6", "atmos": 0, "junk": 1},
                                            "tools": {"truehdd": "/x/truehdd"}})
            assert saved["encode"] == {"target": "dd", "channels": 6, "bitrate": 0, "atmos": False,
                                       "atmos71": "flat", "bed_conform": True, "drc": "film_light"}
            assert settings.load_settings()["tools"]["truehdd"] == "/x/truehdd"
            assert settings.load_settings()["task"] == "encode"


# FPSConv: Folders.test_defaults_and_overrides
def test_defaults_and_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(settings, "config_dir", return_value=Path(tmp) / "cfg"), \
             mock.patch.object(settings, "cache_dir", return_value=Path(tmp) / "cache"):
            assert settings.temp_dir() == Path(tmp) / "cache" / "temp"
            assert settings.work_dir() == Path(tmp) / "cache" / "work"
            assert settings.logs_dir() == Path(tmp) / "cfg" / "logs"
            saved = settings.save_settings({"dirs": {"temp": "D:/fast/tmp", "work": "", "junk": "x"}})
            assert saved["dirs"] == {"temp": "D:/fast/tmp", "work": "", "logs": ""}
            assert settings.temp_dir() == Path("D:/fast/tmp")
            assert settings.work_dir() == Path(tmp) / "cache" / "work"   # empty = default
            assert tools.deezy_work_dir() == str(Path(tmp) / "cache" / "work")


# ── beyond FPSConv's suite ──


def test_settings_live_under_the_app_identifier():
    assert settings.config_dir().parts[-2:] == ("com.audiosyncmaster.app", "encode")
    assert settings.cache_dir().parts[-2:] == ("com.audiosyncmaster.app", "encode")


def test_a_write_keeps_keys_another_writer_saved():
    """The app and the CLI each save a partial; neither loses the other's change."""
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(settings, "config_dir", return_value=Path(tmp)):
            settings.save_settings({"tools": {"truehdd": "/cli/truehdd"}})
            settings.save_settings({"tools": {"deezy": "/app/deezy.exe"}, "workers": 4})
            loaded = settings.load_settings()
            assert loaded["tools"]["truehdd"] == "/cli/truehdd"
            assert loaded["tools"]["deezy"] == "/app/deezy.exe"
            assert loaded["workers"] == 4
            assert not (Path(tmp) / "settings.json.tmp").exists()


def test_a_corrupt_file_reads_as_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "settings.json").write_text("{not json", encoding="utf-8")
        with mock.patch.object(settings, "config_dir", return_value=Path(tmp)):
            assert settings.load_settings() == settings.DEFAULTS
            assert settings.load_history() == []
