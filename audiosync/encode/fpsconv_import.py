"""Carry an FPSConv user's setup over: tool paths, encode defaults, output
folder, chosen folders and the DEE path.

FPSConv kept its settings in ``%APPDATA%\\FPSConv\\settings.json`` (the same
keys as ours) and the DEE path only in deew's ``config.toml``. Its own
ffmpeg download and patcher download are not carried over: the ffmpeg was
never hash-checked, and the Dolby tools pack brings a verified patcher.
FPSConv's files are only read, never changed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from . import settings, tools

#: Tool paths FPSConv users set by hand; the rest came from FPSConv's own downloads.
_TOOL_KEYS = ("ffmpeg", "ffprobe", "deew_python", "mediainfo", "deezy", "deezy_python", "truehdd")
_PLAIN_KEYS = ("out_dir", "conv_type", "workers", "bitrate", "overwrite", "task")


def fpsconv_settings_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "FPSConv" / "settings.json"


def fpsconv_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "FPSConv"


def _fpsconv_download(path: str) -> bool:
    """A path inside FPSConv's own folders: its unverified ffmpeg, or its patcher download."""
    target = os.path.normcase(os.path.abspath(path))
    for folder in (fpsconv_settings_path().parent, fpsconv_cache_dir()):
        root = os.path.normcase(os.path.abspath(folder))
        if target == root or target.startswith(root + os.sep):
            return True
    return False


def mapped(fpsconv: dict, deew_config: dict) -> dict[str, Any]:
    """The settings partial to save from FPSConv's settings and deew's config. Pure."""
    out: dict[str, Any] = {}
    for key in _PLAIN_KEYS:
        if key in fpsconv and fpsconv[key] not in (None, ""):
            out[key] = fpsconv[key]
    if isinstance(fpsconv.get("encode"), dict):
        out["encode"] = dict(fpsconv["encode"])
    tool_paths = {k: str(v) for k, v in (fpsconv.get("tools") or {}).items()
                  if k in _TOOL_KEYS and v and not _fpsconv_download(str(v))}
    if tool_paths:
        out["tools"] = tool_paths
    dirs = {k: str(v) for k, v in (fpsconv.get("dirs") or {}).items() if k in ("temp", "work") and v}
    if dirs:
        out["dirs"] = dirs
    dee = str(deew_config.get("dee_path") or "")
    if dee:
        out["dee_path"] = dee
    return out


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find() -> dict[str, Any]:
    """``{"found": bool, "settings": partial, "path": str}`` without changing anything."""
    path = fpsconv_settings_path()
    data = _read_json(path)
    if data is None:
        return {"found": False, "settings": {}, "path": str(path)}
    return {"found": True, "settings": mapped(data, tools.read_deew_config()), "path": str(path)}


def apply(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Save what :func:`find` found; with a DEE path, also point deew's config at our temp folder."""
    found = find()
    if found["found"]:
        settings.save_settings(found["settings"])
        dee = found["settings"].get("dee_path")
        if dee and tools.resolved(dee):
            tools.write_deew_config(dee, ffmpeg, ffprobe)
    settings.save_settings({"fpsconv_import_offered": True})
    return found
