"""Finding the encode tools and keeping deew's ``config.toml`` right
(FPSConv ``engine.py:245-411``).

deew reads DEE's path from its own config file, so the app writes that file
for the user from the DEE path in Preferences, and rewrites it before every
queue start if an older writer left it pointing at the wrong temp folder.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from . import settings

_log = logging.getLogger(__name__)

FROZEN = bool(getattr(sys, "frozen", False))


# ───────────────────────── deew and DeeZy ─────────────────────────


def deew_cmd(tools: dict, pack_python: str = "") -> list[str]:
    """How to run deew.

    * a Python set in Preferences (``deew_python``) → ``python -m deew``
    * the Dolby tools pack's Python → ``python -m deew``
    * from source → this interpreter's ``-m deew``
    """
    python = (tools or {}).get("deew_python", "") or pack_python
    if python:
        return [python, "-m", "deew"]
    return [sys.executable, "-m", "deew"]


def deezy_cmd(tools: dict) -> list[str]:
    """How to run DeeZy (DDP Atmos).

    DeeZy cannot share an environment with deew (it needs rich >= 14, deew
    rich < 14), so it is an external tool like truehdd:

    * ``deezy`` path from Preferences (its standalone exe) → that
    * a Python set in Preferences (``deezy_python``) → ``python -m deezy``
    * ``deezy`` on PATH → that
    * from source → this interpreter's ``-m deezy``
    """
    tools = tools or {}
    exe = tools.get("deezy", "")
    if exe:
        return [exe]
    python = tools.get("deezy_python", "")
    if python:
        return [python, "-m", "deezy"]
    if shutil.which("deezy") or FROZEN:
        # The frozen engine is not a Python that can run -m; a missing deezy
        # then fails as "cannot start deezy", which says what to fix.
        return ["deezy"]
    return [sys.executable, "-m", "deezy"]


def deezy_work_dir() -> str:
    """A writable folder for DeeZy's logs and batch results (the Working folder)."""
    path = settings.work_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(path)


# ───────────────────────── deew's config.toml ─────────────────────────


def deew_config_path() -> Path:
    """deew reads ``config.toml`` from platformdirs' user_config_dir('deew')."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "deew" / "config.toml"


def read_deew_config() -> dict:
    path = deew_config_path()
    if not path.exists():
        return {}
    try:
        import tomllib
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable is the same as absent
        return {}


def deew_temp_dir() -> Path:
    """deew's own temp folder, inside the Temp folder from Preferences."""
    return settings.temp_dir() / "deew"


def write_deew_config(dee_path: str, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> Path:
    """Create / update deew's config.toml so the user never edits it by hand.

    Keeps everything deew's own generator writes, points it at our ffmpeg /
    ffprobe, turns the logo off (it is not a terminal) and sets the DEE path.
    """
    path = deew_config_path()
    current = read_deew_config()
    bitrates = current.get("default_bitrates") or {
        "dd_1_0": 128, "dd_2_0": 256, "dd_5_1": 640,
        "ddp_1_0": 128, "ddp_2_0": 256, "ddp_5_1": 1024, "ddp_7_1": 1536,
    }
    ffmpeg = shutil.which(ffmpeg) or ffmpeg
    ffprobe = shutil.which(ffprobe) or ffprobe
    # A temp folder we own (under the Temp folder from Preferences). Left empty,
    # deew would use "temp" next to its own launcher, which is the wrong place.
    temp_dir = deew_temp_dir()
    temp_dir.mkdir(parents=True, exist_ok=True)

    def q(v: str) -> str:
        return json.dumps(str(v))  # a TOML basic string; escapes backslashes and quotes

    text = f"""# Written by AudioSyncMaster. Edit the DEE path in Preferences → Encode instead of here.
ffmpeg_path = {q(ffmpeg)}
ffprobe_path = {q(ffprobe)}
dee_path = {q(dee_path)}
temp_path = {q(str(temp_dir))}
logo = 0
max_instances = {q(current.get('max_instances', '50%'))}

[default_bitrates]
""" + "".join(f"    {k} = {v}\n" for k, v in bitrates.items()) + """
[summary_sections]
    deew_info = false
    binaries = false
    input_info = false
    output_info = false
    other = false
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def ensure_deew_config(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
    """Rewrite a deew config another writer left behind (no temp_path, logo on, another temp folder)."""
    current = read_deew_config()
    dee = str(current.get("dee_path") or "")
    if not dee:
        return
    wanted = str(deew_temp_dir())
    if current.get("temp_path") != wanted or current.get("logo") != 0:
        try:
            write_deew_config(dee, ffmpeg, ffprobe)
            _log.info("updated deew config: temp folder %s", wanted)
        except OSError as exc:
            _log.warning("could not update deew config: %s", exc)


def configured_dee() -> str:
    """DEE's path from deew's config when that file exists and DEE is there, else ""."""
    dee = str(read_deew_config().get("dee_path") or "")
    return dee if dee and (os.path.isfile(dee) or shutil.which(dee)) else ""


def resolved(path: Optional[str]) -> str:
    """An executable's full path when it can be found, else ""."""
    if not path:
        return ""
    return shutil.which(path) or (path if os.path.isfile(path) else "")
