"""Encode settings and job history, shared by the app and the CLI
(a port of FPSConv's ``config.py``).

Stored per user, outside the app folder, so an upgrade never wipes them:

* config (settings.json, history.json):
  ``%APPDATA%\\com.audiosyncmaster.app\\encode\\``
* cache (job temp folders, DeeZy's working folder):
  ``%LOCALAPPDATA%\\com.audiosyncmaster.app\\encode\\``

The pages read and write these through the engine rather than localStorage,
because the CLI has to see the same tool paths and folders.

Same keys, defaults and coercion as FPSConv, minus ``auto_update`` and
``native_window``, which belonged to FPSConv's own updater and window.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

APP_ID = "com.audiosyncmaster.app"

DEFAULTS: dict[str, Any] = {
    "out_dir": "",
    "conv_type": "23.976-25",
    "mode": "single",          # single | batch
    "workers": 2,
    "bitrate": 0,              # 0 = use the source bitrate (fps.py behaviour)
    "overwrite": "overwrite",  # overwrite | skip | rename
    "task": "fps",             # fps | encode
    "encode": {                # the audio-only encode task
        "target": "ddp",       # ddp | dd
        "channels": 0,         # 0 = same as source, else 1 / 2 / 6 / 8
        "bitrate": 0,          # 0 = DEE default for the layout
        "atmos": True,         # keep Atmos when the source has it (TrueHD Atmos -> DDP Atmos)
        "atmos71": "flat",     # 7.1 Atmos legacy layout: flat (Lb Rb, patched DEE) | dee (DEE's 5.1+2 heights)
        "bed_conform": True,   # truehdd --bed-conform on the flat-7.1 path
        "drc": "film_light",
    },
    "tools": {"ffmpeg": "", "ffprobe": "", "deew_python": "",
              "mediainfo": "", "deezy": "", "deezy_python": "", "truehdd": "",
              "patcher": ""},   # folder of the DD+ 7.1 Atmos patcher ("" = the Dolby tools pack's)
    "dirs": {                  # "" = the default shown by dir_defaults()
        "temp": "",            # job intermediates (WAVs, truehdd masters) + deew's own temp
        "work": "",            # DeeZy's logs and batch results
        "logs": "",            # kept for FPSConv's settings shape; the engine logs to the Output dock
    },
}

_NESTED = ("tools", "encode", "dirs")
HISTORY_KEEP = 200

_lock = threading.Lock()


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_ID / "encode"


def cache_dir() -> Path:
    """Big, disposable files: per-user local data, never the roaming profile or the program folder."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / APP_ID / "encode"


def dir_defaults() -> dict[str, str]:
    return {"temp": str(cache_dir() / "temp"), "work": str(cache_dir() / "work"), "logs": str(config_dir() / "logs")}


def _dir(key: str) -> Path:
    chosen = (load_settings().get("dirs") or {}).get(key, "")
    return Path(chosen) if chosen else Path(dir_defaults()[key])


def temp_dir() -> Path:
    """Where jobs put their intermediates (a sub-folder per job) and deew its own temp."""
    return _dir("temp")


def work_dir() -> Path:
    """DeeZy's working folder (logs, batch results)."""
    return _dir("work")


def logs_dir() -> Path:
    return _dir("logs")


def _read(name: str, default: Any) -> Any:
    path = config_dir() / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write(name: str, payload: Any) -> None:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f"{name}.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, directory / name)


def load_settings() -> dict[str, Any]:
    with _lock:
        data = _read("settings.json", {})
    merged = json.loads(json.dumps(DEFAULTS))
    if isinstance(data, dict):
        for key, value in data.items():
            if key in _NESTED and isinstance(value, dict):
                merged[key].update(_clean(key, value))
            elif key in merged and key not in _NESTED:
                merged[key] = value
    return merged


def _clean(section: str, value: dict) -> dict:
    """Only known keys of a nested section, coerced to the type of the default."""
    out = {}
    for k, v in value.items():
        if k not in DEFAULTS[section]:
            continue
        default = DEFAULTS[section][k]
        if isinstance(default, bool):
            out[k] = bool(v)
        elif isinstance(default, int):
            try:
                out[k] = int(v or 0)
            except (TypeError, ValueError):
                out[k] = default
        else:
            out[k] = str(v or "").strip()
    return out


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    """Merge ``update`` into what is on disk now and write it atomically.

    Re-reading first means a change from the CLI and one from the app never
    overwrite each other's other keys.
    """
    current = load_settings()
    for key, value in update.items():
        if key in _NESTED and isinstance(value, dict):
            current[key].update(_clean(key, value))
        elif key in DEFAULTS and key not in _NESTED:
            current[key] = value
    with _lock:
        _write("settings.json", current)
    return current


def load_history() -> list[dict[str, Any]]:
    with _lock:
        data = _read("history.json", [])
    return data if isinstance(data, list) else []


def save_history(entries: list[dict[str, Any]], keep: int = HISTORY_KEEP) -> None:
    with _lock:
        _write("history.json", entries[-keep:])
