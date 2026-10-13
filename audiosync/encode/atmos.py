"""Atmos detection and MediaInfo reports (FPSConv ``engine.py:622-741``).

ffprobe cannot see Atmos, so MediaInfo decides it. When MediaInfo is not
available the answer is ``None`` -- "unknown" -- never ``False``.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def parse_mediainfo_audio(data: Optional[dict]) -> Optional[list[dict]]:
    """Per audio track (in container order): ``{"atmos": bool, "commercial": str}``.

    ``None`` when MediaInfo was not available — ffprobe cannot see Atmos, so
    "unknown" must stay distinguishable from "no".
    """
    if not data:
        return None
    tracks = ((data.get("media") or {}).get("track")) or []
    out = []
    for t in tracks:
        if t.get("@type") != "Audio":
            continue
        commercial = str(t.get("Format_Commercial_IfAny") or "")
        extra = str(t.get("Format_AdditionalFeatures") or "")
        atmos = "atmos" in commercial.lower() or "joc" in extra.lower() or "16-ch" in extra.lower()
        out.append({"atmos": atmos, "commercial": commercial})
    return out


def pretty_codec(stream: dict) -> str:
    """``TrueHD Atmos``, ``DTS-HD MA``, ``E-AC-3``, ``AAC`` … for the queue row."""
    name = (stream.get("codec_name") or "?").lower()
    profile = stream.get("profile") or ""
    base = {"truehd": "TrueHD", "eac3": "E-AC-3", "ac3": "AC-3", "aac": "AAC", "dts": "DTS",
            "flac": "FLAC", "opus": "Opus", "vorbis": "Vorbis", "mlp": "MLP"}.get(name)
    if base is None:
        base = "PCM" if name.startswith("pcm") else name.upper()
    if name == "dts" and profile:
        base = profile                     # ffprobe says "DTS-HD MA", "DTS-HD HRA", "DTS:X" …
    if stream.get("atmos") is True:
        base += " Atmos"
    return base


def report_from_json(data: dict) -> str:
    """A readable MediaInfo-style report from its JSON, for when only JSON is available."""
    out = []
    for t in ((data.get("media") or {}).get("track")) or []:
        out.append(str(t.get("@type", "")))
        for k, v in t.items():
            if k.startswith("@") or k == "extra" or v in (None, ""):
                continue
            out.append(f"{k:<40}: {v}")
        out.append("")
    return "\n".join(out).strip()


def mediainfo_json(path: str, mediainfo: str = "mediainfo") -> Optional[dict]:
    """MediaInfo's JSON for a file from the ``mediainfo`` CLI; ``None`` when it does not run.

    The pymediainfo fallback runs in the Dolby tools pack's worker, not here.
    """
    try:
        return json.loads(subprocess.check_output(
            [mediainfo, "--Output=JSON", path],
            text=True, creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL, timeout=120,
        ))
    except Exception:  # noqa: BLE001 - missing, crashed or not JSON: all mean "no answer"
        return None


def mediainfo_atmos(path: str, mediainfo: str = "mediainfo") -> Optional[list[dict]]:
    return parse_mediainfo_audio(mediainfo_json(path, mediainfo))


def mediainfo_report(path: str, mediainfo: str = "mediainfo") -> str:
    """MediaInfo's text report for a file: the ``mediainfo`` CLI, else its JSON rendered."""
    if not os.path.isfile(path):
        return ""
    try:
        text = subprocess.check_output([mediainfo, path], text=True, creationflags=_NO_WINDOW,
                                       stdin=subprocess.DEVNULL, timeout=120)
        if text.strip():
            return text.strip()
    except Exception:  # noqa: BLE001
        pass
    data = mediainfo_json(path, mediainfo)
    return report_from_json(data) if data else ""


def find_output(path: str, folder: str = "", stem: str = "") -> tuple[str, str]:
    """Locate a file for the MediaInfo report: ``(found path, diagnosis)``.

    Tries the exact path (also normalised), then any file in ``folder`` whose
    name starts with ``stem`` (case-insensitive, newest first — the tool may
    have written a slightly different name), and otherwise explains what the
    folder does contain so a mismatch can be seen instead of guessed.
    """
    for cand in (path, os.path.normpath(path) if path else ""):
        if cand and os.path.isfile(cand):
            return cand, ""
    folder = folder or (os.path.dirname(path) if path else "")
    stem = (stem or os.path.splitext(os.path.basename(path))[0]).lower()
    if folder and os.path.isdir(folder):
        try:
            names = os.listdir(folder)
        except OSError as exc:
            return "", f"cannot read {folder}: {exc}"
        hits = [n for n in names if stem and n.lower().startswith(stem)]
        if hits:
            hits.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=True)
            return os.path.join(folder, hits[0]), ""
        # nothing with that stem: show the closest names so the difference is visible
        first = stem.split("_")[0][:24]
        similar = [n for n in names if first and n.lower().startswith(first)][:6]
        detail = (f"folder {folder} has no file starting with {stem!r}; similar names there: {similar}"
                  if similar else f"folder {folder} has {len(names)} file(s), none starting with {stem!r}")
        return "", f"not found: {path}\n{detail}"
    return "", f"not found: {path}\nfolder {folder or '?'} does not exist or is not reachable from this app"
