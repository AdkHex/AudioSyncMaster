"""Reading progress out of ffmpeg, deew, DeeZy, truehdd and the 7.1 patcher,
and the stage list a job shows (FPSConv ``engine.py:449-480``, ``:1125-1449``,
``:1644-1682``).

Pure: every parser takes one line of tool output and returns what it means.
The runners that feed them live in ``jobs.py``.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# ───────────────────────── text ─────────────────────────


def hr_size(size) -> str:
    try:
        size = float(size)
    except (TypeError, ValueError):
        return "0 B"
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {u}"
        size /= 1024
    return f"{size:.2f} PB"


def parse_time(text) -> float:
    """``HH:MM:SS`` (what fmt_time writes) back to seconds; anything else is 0."""
    try:
        parts = [float(p) for p in str(text or "0").split(":")]
    except ValueError:
        return 0.0
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def fmt_time(sec) -> str:
    try:
        sec = max(0.0, float(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    except (TypeError, ValueError):
        return "00:00:00"


# ───────────────────────── stages ─────────────────────────

#: Share of a job each stage usually takes — only for the overall percent, so rough is fine.
STAGE_WEIGHT = {
    "Extracting TrueHD": 5, "truehdd decode": 45, "Patching DEE": 1, "DEE measure": 14, "DEE encode": 31,
    "Surround EX flag": 1, "TrueHD decode": 40, "Decoding to WAV": 35, "Extracting WAV": 35,
    "ffmpeg": 5, "DEE: measure": 20, "DEE: encode": 37, "ffmpeg atempo + AAC": 97, "Finishing": 3,
}
_STEP_SUFFIX = re.compile(r"\s*\(\d+/\d+(?:[–-]\d+/\d+)?\)$")   # "Extracting TrueHD (1/5)" → "Extracting TrueHD"
_STAGE_ALIAS = {"Encoding AAC Engine": "ffmpeg atempo + AAC"}


def new_stage(name: str) -> dict:
    return {"name": name, "state": "wait", "weight": STAGE_WEIGHT.get(name, 10),
            "started": 0.0, "finished": 0.0, "pct": 0.0, "note": ""}


def stage_of(step_name: str) -> str:
    """The stage a single-tool step shows as in the Progress view."""
    return _STAGE_ALIAS.get(step_name) or _STEP_SUFFIX.sub("", step_name)


# ───────────────────────── errors ─────────────────────────

ERR_RE = re.compile(r"not supported|no such|not found|no module|invalid|error|denied", re.IGNORECASE)


def pick_error(lines: list[str]) -> str:
    """The most informative line of a failed tool's output, not just the last one."""
    for line in lines:
        if ERR_RE.search(line) and "Conversion failed" not in line:
            return re.sub(r"^\[[^\]]*\]\s*", "", line).strip()
    return lines[-1].strip() if lines else ""


# ───────────────────────── ffmpeg ─────────────────────────

FFMPEG_NOISE = re.compile(r"^(ffmpeg version|built with|configuration:|lib(avutil|avcodec|avformat|avdevice|avfilter|swscale|swresample|postproc)\s)")
#: The ``-progress pipe:1`` keys; ``progress=`` closes each block.
FFMPEG_PROGRESS_KEYS = ("out_time_us", "out_time_ms", "out_time", "speed", "bitrate", "total_size", "frame",
                        "fps", "progress", "drop_frames", "dup_frames", "stream_0_0_q")


def ffmpeg_percent(out_time_s: float, source_duration: float, ratio: float = 1.0) -> float:
    """How far ffmpeg is, from the output time it reports.

    ``out_time`` counts the *output*, which a speed change makes shorter or
    longer than the source: a 23.976 → 25 conversion ends at source ÷ 1.0427.
    FPSConv divided by the source duration, so a speed-up stopped at 95.9 %.
    """
    total = source_duration / ratio if ratio > 0 else source_duration
    return min(100.0, out_time_s / total * 100) if total > 0 else 0.0


# ───────────────────────── deew ─────────────────────────

PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
STAGE_RE = re.compile(r"\[\s*([^\]|]+?)\s*\|")
#: deew's stages and the share of the job each one gets on the progress bar.
DEEW_STAGES = {"ffmpeg": (0, 10), "DEE: measure": (10, 40), "DEE: encode": (40, 100)}


def parse_deew_line(text: str) -> Optional[tuple[str, float]]:
    """``[ DEE: measure | file ] ━━━ 42.10%`` → ("DEE: measure", 42.1); None for other lines."""
    m = PCT_RE.search(text)
    st = STAGE_RE.search(text)
    if m and st:
        return st.group(1).strip(), float(m.group(1))
    return None


# ───────────────────────── DeeZy ─────────────────────────

DEEZY_LINE = re.compile(r"^(?:[\w-]+:\s*)?(.+?)\s*\((\d+) of (\d+)\)\s+(\d+(?:\.\d+)?)%")
#: truehdd's own ``--progress`` output, relayed by DeeZy at debug level as ``[truehdd-err] …``.
TRUEHDD_PCT = re.compile(r"^\[truehdd[^\]]*\].*?(\d+(?:\.\d+)?)\s*%")
#: DeeZy's Atmos stages and the share of the job each one gets on the progress bar.
DEEZY_STAGES = {"TrueHD decode": (0, 35), "DEE measure": (35, 55), "DEE encode": (55, 100), "ffmpeg": (0, 20)}
_DEEZY_ALIAS = {"truehdd": "TrueHD decode", "TrueHD extract & decode": "TrueHD decode", "FFMPEG": "ffmpeg"}

ParseFn = Callable[[str], Optional[tuple[str, Optional[float]]]]


def parse_deezy_line(text: str) -> Optional[tuple[str, Optional[float]]]:
    m = DEEZY_LINE.match(text)
    if m:
        name = m.group(1).strip()
        return _DEEZY_ALIAS.get(name, name), float(m.group(4))
    t = TRUEHDD_PCT.match(text)
    if t:
        return "TrueHD decode", float(t.group(1))
    if text.startswith("[truehdd"):
        return "TrueHD decode", None
    return None


# ───────────────────────── truehdd and the 7.1 patcher ─────────────────────────

_TRUEHDD_ANY_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DEE_OVERALL = re.compile(r"Overall progress:\s*(\d+(?:\.\d+)?)")
TRUEHDD_STAGES = {"truehdd decode": (0, 100)}
PATCHER_STAGES = {"Patching DEE": (0, 2), "DEE measure": (2, 40), "DEE encode": (40, 97), "Surround EX flag": (97, 100)}


def parse_truehdd_line(text: str) -> Optional[tuple[str, Optional[float]]]:
    m = _TRUEHDD_ANY_PCT.search(text)
    return ("truehdd decode", float(m.group(1))) if m else None


class PatcherParser:
    """DEE prints 'Overall progress: N.N.' for the whole job (measure ≈ first 40 %, encode the
    rest); the wrapper adds its own lines before and after."""

    def __init__(self) -> None:
        self.stage = "Patching DEE"

    def __call__(self, text: str) -> Optional[tuple[str, Optional[float]]]:
        if "Starting measurement" in text:
            self.stage = "DEE measure"
            return self.stage, 0.0
        if "Measurement done" in text:
            self.stage = "DEE encode"
            return self.stage, 0.0
        m = _DEE_OVERALL.search(text)
        if m:
            pct = float(m.group(1))
            if self.stage == "DEE measure":
                return self.stage, min(100.0, pct / 40.0 * 100.0)
            if self.stage == "DEE encode":
                return self.stage, max(0.0, min(100.0, (pct - 40.0) / 60.0 * 100.0))
            return self.stage, pct
        if "patched AC-3 frames" in text or "dsur" in text.lower() and "wrote" in text.lower():
            self.stage = "Surround EX flag"
            return self.stage, 100.0
        if text.startswith(("Run directory", "Verified backup", "Generated ")):
            return "Patching DEE", None
        return None
