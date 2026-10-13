"""Reading a file's audio streams, frame rate and duration with ffprobe, and
Atmos with MediaInfo (FPSConv ``engine.py:482-495``, ``:602-609``,
``:744-807`` and the ``/api/probe`` row in ``server.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Callable, Optional

from . import atmos
from .fps import DEE_CODEC_MAP, codec_class, detect_fps
from .progress import fmt_time, hr_size

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: Per audio track Atmos facts for a file, or None when MediaInfo is unavailable.
AtmosFn = Callable[[str], Optional[list[dict]]]


def ffprobe_json(ffprobe: str, path: str) -> dict:
    return json.loads(subprocess.check_output(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
        text=True, creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL,
    ))


def get_duration(ffprobe: str, path: str) -> float:
    try:
        return float(ffprobe_json(ffprobe, path)["format"].get("duration", 0))
    except Exception:  # noqa: BLE001
        return 0.0


def probe_video(ffprobe: str, file_path: str) -> dict:
    """Frame rate + duration of a reference video (no audio needed)."""
    data = ffprobe_json(ffprobe, file_path)
    fps, source = detect_fps(data, file_path)
    duration = float((data.get("format") or {}).get("duration") or 0)
    return {"path": file_path, "name": os.path.basename(file_path), "fps": fps,
            "fps_source": source, "duration": fmt_time(duration), "duration_s": duration}


def probe_streams(ffprobe: str, file_path: str, atmos_for: Optional[AtmosFn] = None) -> dict:
    """Every audio stream of a file, plus container duration.

    ``codec`` is the fps.py family (aac / ac3 / eac3 / truehd) — anything that
    is not Dolby is treated as AAC and goes through ffmpeg, exactly as fps.py did.
    ``atmos`` is True / False from MediaInfo, or None when it is not available
    (or when ``atmos_for`` is not given).
    """
    streams: list[dict] = []
    duration = 0.0
    fps, fps_source = None, ""
    format_name = ""
    try:
        data = ffprobe_json(ffprobe, file_path)
        duration = float((data.get("format") or {}).get("duration") or 0)
        format_name = str((data.get("format") or {}).get("format_name") or "")
        fps, fps_source = detect_fps(data, file_path)
        fmt_bitrate = int((data.get("format") or {}).get("bit_rate") or 0) // 1000
        for s in data.get("streams", []):
            if s.get("codec_type") != "audio":
                continue
            tags = s.get("tags") or {}
            bitrate = int(s["bit_rate"]) // 1000 if s.get("bit_rate") else fmt_bitrate or 640
            streams.append({
                "index": len(streams),                     # position among audio streams (0:a:N)
                "codec": codec_class(s.get("codec_name")),
                "codec_name": s.get("codec_name", "?"),
                "profile": s.get("profile", ""),
                "bitrate": bitrate,
                "channels": int(s.get("channels", 2)),
                "layout": s.get("channel_layout", ""),
                "sample_rate": int(s.get("sample_rate") or 48000),
                "language": tags.get("language", ""),
                "title": tags.get("title", ""),
                "atmos": None,
                "commercial": "",
                "format_name": format_name,
            })
    except Exception:  # noqa: BLE001 - an unreadable file is "no audio stream found"
        pass
    if streams and atmos_for is not None:
        info = atmos_for(file_path)
        if info is not None:
            for s in streams:
                mi = info[s["index"]] if s["index"] < len(info) else {"atmos": False, "commercial": ""}
                s["atmos"], s["commercial"] = mi["atmos"], mi["commercial"]
    for s in streams:
        s["pretty"] = atmos.pretty_codec(s)
    return {"streams": streams, "duration": duration, "fps": fps, "fps_source": fps_source}


def find_stream(ffprobe: str, file_path: str, stream_index: int,
                atmos_for: Optional[AtmosFn] = None) -> Optional[dict]:
    info = probe_streams(ffprobe, file_path, atmos_for)
    for s in info["streams"]:
        if s["index"] == stream_index:
            return s
    return info["streams"][0] if info["streams"] else None


def detect_audio_info(ffprobe: str, file_path: str, stream_index: int = 0) -> tuple[str, int, int]:
    """(codec, bitrate_kbps, channels) of one audio stream — fps.py defaults if unknown."""
    s = find_stream(ffprobe, file_path, stream_index)
    if s:
        return s["codec"], s["bitrate"], s["channels"]
    return "aac", 640, 2


def probe_row(ffprobe: str, path: str, atmos_for: Optional[AtmosFn] = None) -> dict:
    """One row of the queue table: the file, its frame rate and every stream with the path it would take."""
    if not os.path.isfile(path):
        return {"path": path, "name": os.path.basename(path), "error": "file not found"}
    info = probe_streams(ffprobe, path, atmos_for)
    if not info["streams"]:
        return {"path": path, "name": os.path.basename(path), "error": "no audio stream found"}
    for s in info["streams"]:
        s["engine"] = ("ffmpeg aac" if s["codec"] == "aac"
                       else f"ffmpeg wav → deew {DEE_CODEC_MAP.get(s['codec'], ('', 'ddp', ''))[1]}")
        s["engine_encode"] = "truehdd → DEE Atmos (deezy)" if s["atmos"] else "ffmpeg wav → deew"
    return {
        "path": path, "name": os.path.basename(path),
        "duration": fmt_time(info["duration"]), "duration_s": info["duration"],
        "fps": info["fps"], "fps_source": info["fps_source"],
        "size": hr_size(os.path.getsize(path)),
        "streams": info["streams"],
    }
