"""The FPS change: conversions, ``atempo``, codec classes, frame-rate
detection and output names (FPSConv ``engine.py:53-66``, ``:497-619``,
``:810-832``).

Pure: works on ffprobe's JSON and plain values, runs nothing.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction
from typing import Optional

from .progress import fmt_time

DEE_CODEC_MAP = {
    "eac3":   ("-f", "ddp", ".ec3"),
    "ac3":    ("-f", "dd",  ".ac3"),
    "truehd": ("-f", "thd", ".thd"),
}

# The exact rates from framerate.COMMON_RATES. Their ratios are the same
# floats FPSConv computed from 24000/1001, bit for bit.
_RATES = {"23.976": Fraction(24000, 1001), "24": Fraction(24), "25": Fraction(25)}
CONVERSION_KEYS = ("23.976-24", "23.976-25", "24-23.976", "24-25", "25-23.976", "25-24")
FPS_CONVERSIONS = {
    key: float(_RATES[key.split("-")[1]] / _RATES[key.split("-")[0]]) for key in CONVERSION_KEYS
}


def atempo_chain(ratio) -> str:
    parts = []
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return "atempo=1.0"
    if abs(r - 1.0) < 0.0001:
        return "anull"
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r *= 2.0
    parts.append(f"atempo={r:.6f}")
    return ",".join(parts)


def codec_class(codec_name: str) -> str:
    """The FPS path a stream takes: eac3, ac3 and truehd keep their codec, the rest become AAC."""
    c = (codec_name or "aac").lower()
    if "eac3" in c:
        return "eac3"
    if "ac3" in c:
        return "ac3"
    if "truehd" in c:
        return "truehd"
    return "aac"


def output_name(file_path: str, conv_type: str, codec: str, stream_index: int = 0) -> str:
    ext = ".aac" if codec == "aac" else DEE_CODEC_MAP.get(codec, ("-f", "ddp", ".ec3"))[2]
    stem = os.path.splitext(os.path.basename(file_path))[0]
    track = f"_a{stream_index}" if stream_index else ""
    return f"{stem}{track}_{conv_type.replace('.', '_')}{ext}"


# ───────────────────────── frame-rate detection ─────────────────────────

# Frame rates the app knows how to convert between, plus the common ones
# worth naming when they show up in a video track.
_FPS_LABELS = (
    (23.976, "23.976"), (24.0, "24"), (25.0, "25"), (29.97, "29.97"),
    (30.0, "30"), (50.0, "50"), (59.94, "59.94"), (60.0, "60"),
)
# "…23.976fps…", "…25 fps…", "…29.97…" in a file name.  A bare 24/25/30
# is not trusted (S01E24) unless "fps" follows it.
_FPS_IN_NAME = re.compile(r"(?<!\d)(23\.976|23\.98|29\.97|59\.94|(?:24|25|30|50|60)(?=[\s._-]?fps))[\s._-]?(?:fps)?(?!\d)", re.I)


def fps_label(value: float) -> Optional[str]:
    """24.0 → "24", 23.976023… → "23.976"; None for 0 / nonsense."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not v or v != v or v > 1000:
        return None
    for ref, label in _FPS_LABELS:
        if abs(v - ref) < 0.015:
            return label
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _parse_rate(text: str) -> float:
    """ffprobe rates are "24000/1001" or "25/1"."""
    try:
        num, _, den = str(text).partition("/")
        return float(num) / (float(den) if den else 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _fps_from_tags(*tag_dicts: dict) -> Optional[str]:
    for tags in tag_dicts:
        for k, v in (tags or {}).items():
            if k.lower().replace("_", "").replace("-", "") in ("fps", "framerate", "originalfps", "sourcefps", "videofps"):
                label = fps_label(_parse_rate(v))
                if label:
                    return label
    return None


def detect_fps(data: dict, file_path: str = "") -> tuple[Optional[str], str]:
    """(fps label, where it came from) for a probed file.

    Audio has no frame rate of its own: it is only "23.976 fps audio" because
    it was cut to a 23.976 fps video.  So we look, in order, at the video track
    it is muxed with, at fps-like container / stream tags, and finally at the
    file name.  (None, "") when nothing tells us.
    """
    for s in data.get("streams", []):
        if s.get("codec_type") != "video" or s.get("disposition", {}).get("attached_pic"):
            continue
        for key in ("avg_frame_rate", "r_frame_rate"):
            label = fps_label(_parse_rate(s.get(key, "")))
            if label:
                return label, "video"
    fmt = data.get("format") or {}
    tag_fps = _fps_from_tags(fmt.get("tags"), *[s.get("tags") for s in data.get("streams", [])])
    if tag_fps:
        return tag_fps, "tag"
    m = _FPS_IN_NAME.search(os.path.basename(file_path))
    if m:
        return fps_label(float(m.group(1))) or m.group(1), "name"
    return None, ""


def suggest_conversion(src_fps: Optional[str], src_duration: float,
                       ref_fps: Optional[str], ref_duration: float,
                       tolerance: float = 0.0005) -> Optional[dict]:
    """Which FPS_CONVERSIONS key turns this audio into one that fits the reference video.

    * both frame rates known → "<src>-<ref>" if the app has it
    * otherwise compare durations: audio cut at 24 fps played against a 25 fps
      video is 25/24 longer, so the conversion whose speed ratio matches
      ``src_duration / ref_duration`` (within ``tolerance``, 0.05 %) is the one.
    Returns {"conv_type", "reason", "delta"} or None.
    """
    if src_fps and ref_fps:
        key = f"{src_fps}-{ref_fps}"
        if key in FPS_CONVERSIONS:
            return {"conv_type": key, "reason": f"audio is {src_fps} fps, video is {ref_fps} fps", "delta": 0.0}
        if src_fps == ref_fps:
            return {"conv_type": None, "reason": f"already {ref_fps} fps – no conversion needed", "delta": 0.0}
    if src_duration > 0 and ref_duration > 0:
        want = src_duration / ref_duration
        best, best_err = None, tolerance
        for key, ratio in FPS_CONVERSIONS.items():
            if ref_fps and not key.endswith("-" + ref_fps):
                continue
            err = abs(ratio - want) / want
            if err < best_err:
                best, best_err = key, err
        if best:
            delta = abs(src_duration - ref_duration * FPS_CONVERSIONS[best])
            return {"conv_type": best,
                    "reason": f"audio runs {fmt_time(src_duration)} against a {fmt_time(ref_duration)} video (×{want:.5f})",
                    "delta": round(delta, 2)}
        if abs(want - 1) < tolerance:
            return {"conv_type": None, "reason": "durations already match – no conversion needed",
                    "delta": round(abs(src_duration - ref_duration), 2)}
    return None
