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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import media
from ..subs import packs
from . import flat71, settings

_log = logging.getLogger(__name__)

FROZEN = bool(getattr(sys, "frozen", False))


# ───────────────────────── deew and DeeZy ─────────────────────────


def deew_cmd(tools: dict, pack_python: str = "") -> list[str]:
    """How to run deew.

    * a Python set in Preferences (``deew_python``) → ``python -m deew``
    * the Dolby tools pack's Python → ``python -m deew``
    * ``deew`` on PATH → that
    * from source → this interpreter's ``-m deew``
    """
    python = (tools or {}).get("deew_python", "") or pack_python
    if python:
        return [python, "-m", "deew"]
    if shutil.which("deew") or FROZEN:
        # The frozen engine is not a Python that can run -m (it would read
        # "-m" as its own argument); a missing deew fails as "cannot start deew".
        return ["deew"]
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


# ───────────────────────── discovery ─────────────────────────

DOLBY_PACK = "dolby-tools"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def tool_settings(overrides: Optional[dict] = None) -> dict:
    """The tool paths from the settings file, with a request's own paths on top."""
    tools = dict(settings.load_settings().get("tools") or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, str) and value.strip():
            tools[key] = value.strip()
    return tools


def dolby_pack_python() -> str:
    """The Dolby tools pack's interpreter (deew, pymediainfo, the patcher), or ""."""
    return packs.python_for(DOLBY_PACK) or ""


def patcher_folder(tools: dict) -> Path:
    """The 7.1 patcher folder: the one set in Preferences, else the Dolby tools pack's."""
    chosen = (tools or {}).get("patcher", "")
    return Path(chosen) if chosen else Path(packs.pack_dir(DOLBY_PACK)) / "patcher"


def script_runner(pack_python: str) -> list[str]:
    """The Python that runs the patcher's stdlib-only wrapper script, or [] when there is none."""
    if pack_python:
        return [pack_python]
    return [] if FROZEN else [sys.executable]


def find_ffmpeg(tools: dict) -> str:
    """ffmpeg: the path set in Preferences, else the ffmpeg-full pack, else the app's own discovery."""
    return (tools or {}).get("ffmpeg") or packs.ffmpeg_full() or media.ffmpeg_path()


def find_ffprobe(tools: dict) -> str:
    chosen = (tools or {}).get("ffprobe")
    if chosen:
        return chosen
    full = packs.ffmpeg_full()
    if full:
        sibling = os.path.join(os.path.dirname(full), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(sibling):
            return sibling
    return media.ffprobe_path()


@dataclass
class Tools:
    """Every tool a job may run, resolved once when the job starts."""

    ffmpeg: str
    ffprobe: str
    mediainfo: str
    truehdd: str
    dee: str                    # from deew's config; "" = not configured
    deew: list[str]
    deezy: list[str]
    pack_python: str
    patcher_dir: Path
    patcher_script: Optional[Path]
    runner: list[str]

    @classmethod
    def resolve(cls, overrides: Optional[dict] = None) -> "Tools":
        tools = tool_settings(overrides)
        pack_python = dolby_pack_python()
        folder = patcher_folder(tools)
        return cls(
            ffmpeg=find_ffmpeg(tools),
            ffprobe=find_ffprobe(tools),
            mediainfo=tools.get("mediainfo") or "mediainfo",
            truehdd=tools.get("truehdd") or "truehdd",
            dee=configured_dee(),
            deew=deew_cmd(tools, pack_python),
            deezy=deezy_cmd(tools),
            pack_python=pack_python,
            patcher_dir=folder,
            patcher_script=flat71.patcher_script(folder),
            runner=script_runner(pack_python),
        )

    def deezy_tools(self) -> dict:
        """Explicit tool paths for DeeZy: our ffmpeg, DEE from deew's config, truehdd."""
        return {"ffmpeg": resolved(self.ffmpeg), "dee": self.dee, "truehdd": resolved(self.truehdd)}

    def flat71_ready(self) -> tuple[bool, str]:
        script = self.patcher_script if self.runner else None
        return flat71.flat71_ready(script, self.deezy_tools())


_SOXR_CACHE: dict[str, bool] = {}


def ffmpeg_has_soxr(ffmpeg: str) -> bool:
    """Whether this ffmpeg build links libsoxr (``--enable-libsoxr`` in ``-version``)."""
    if ffmpeg not in _SOXR_CACHE:
        try:
            out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=20,
                                 creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL).stdout
            _SOXR_CACHE[ffmpeg] = "--enable-libsoxr" in out
        except Exception:  # noqa: BLE001
            _SOXR_CACHE[ffmpeg] = False
    return _SOXR_CACHE[ffmpeg]


def _runs(argv: list[str]) -> Optional[bool]:
    """True when ``argv --version`` exits 0, None when it cannot run."""
    try:
        return subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=30,
                              creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL).returncode == 0 or None
    except Exception:  # noqa: BLE001
        return None


def doctor(overrides: Optional[dict] = None) -> dict:
    """What is installed: FPSConv's /api/doctor fields. DEE is only visible through deew's config."""
    t = Tools.resolve(overrides)
    current = settings.load_settings()
    out = {
        "python": sys.version.split()[0],
        "frozen": FROZEN,
        "ffmpeg": resolved(t.ffmpeg) or None,
        "ffprobe": resolved(t.ffprobe) or None,
        "ffmpeg_soxr": False,
        "deew": _runs(t.deew),
        "deew_via": t.deew[0],
        "deew_config": None,
        "dee": None,
        "dee_path": "",
        "mediainfo": resolved(t.mediainfo) or None,
        "pymediainfo": bool(t.pack_python),
        "truehdd": resolved(t.truehdd) or None,
        "deezy": _runs(t.deezy),
        "deezy_via": t.deezy[0],
        "dolby_pack": bool(t.pack_python),
        "config_dir": str(settings.config_dir()),
        "dirs": {"temp": str(settings.temp_dir()), "work": str(settings.work_dir()), "logs": str(settings.logs_dir()),
                 "defaults": settings.dir_defaults()},
        "settings": current,
        "patcher": str(t.patcher_script) if t.patcher_script else None,
        "patcher_dir": str(t.patcher_dir),
    }
    if out["ffmpeg"]:
        out["ffmpeg_soxr"] = ffmpeg_has_soxr(out["ffmpeg"])
    cfg = deew_config_path()
    if cfg.exists():
        out["deew_config"] = str(cfg)
        dee = str(read_deew_config().get("dee_path") or "")
        out["dee_path"] = dee
        out["dee"] = t.dee or None
    out["dee_dll"] = flat71.dee_dll_status(out["dee_path"])
    ok, why = t.flat71_ready()
    out["flat71"] = {"ok": ok, "reason": why}
    return out
