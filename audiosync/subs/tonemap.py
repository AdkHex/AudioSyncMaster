"""HDR10 / HLG / Dolby Vision video -> SDR BT.709 (task "tonemap").

Five engines, because no single FFmpeg feature is both everywhere and right:

``lut``
    Works with any FFmpeg. ``mergeplanes`` relabels the yuv444p10 frame as
    gbrp10 without touching a bit; a 2-point LUT does the BT.2020 Y'CbCr ->
    R'G'B' matrix exactly (a LUT of an affine map is the map); a 65-point
    LUT does PQ/HLG to light, the tone curve, 2020 -> 709 primaries and the
    BT.1886 encode, and writes BT.709 Y'CbCr codes; ``mergeplanes`` relabels
    them back. FFmpeg's scaler is kept away from the colour maths because,
    measured, it converts Y'CbCr <-> RGB 3-4 10-bit codes off at white (see
    ``colorimetry.ycbcr_to_rgb_lut``).
``videotoolbox``
    macOS hardware decode + scaling. With the Full FFmpeg pack (Jellyfin
    FFmpeg) the tone-mapping is its Metal ``tonemap_videotoolbox``: BT.2390
    from MaxCLL / mastering / the DV RPU per frame, with Dolby Vision
    reshaping -- measured monotonic, 10000-nit master: 203 nits -> Y' 159,
    1000 -> 205, 4000 -> 230. With a plain FFmpeg it is ``scale_vt``'s
    colour conversion, which *does* tone-map but with one fixed curve that
    ignores the metadata and every option: 100 nits -> 49%, 203 -> 78.5%,
    and everything from about 800 nits up clips to white.
``zscale``
    zimg linearisation + FFmpeg's ``tonemap`` filter (hable, mobius,
    reinhard, clip). ``tonemap`` has no BT.2390, so for that curve zscale
    only resizes and the LUT does the mapping.
``libplacebo``
    Per-scene peak detection and Dolby Vision reshaping
    (``apply_dolbyvision``). In the Full FFmpeg pack on Windows and Linux.
``tonemapx``
    Jellyfin FFmpeg's CPU (SIMD) tone-mapper, with Dolby Vision reshaping
    for the RPU type profile 5 files use. Measured: it ignores MaxCLL and
    mastering metadata (assumes 1000 nits) and darkens anything above its
    peak, so it is always given the highest bound the metadata allows.

Dolby Vision profile 5 is only converted by an engine that reshapes it:
libplacebo, then the Metal VideoToolbox filter, then tonemapx.

Every output is tagged BT.709 / BT.709 / BT.709 limited range, and the HDR
side data (mastering display, MaxCLL, HDR10+, DV RPU) is deleted in the
filter graph: measured, FFmpeg otherwise copies the input's mastering
metadata into an SDR x264 *and* x265 output (stream-level and SEI), and a
TV that sees it switches to HDR mode and shows the SDR picture wrong.
"""

from __future__ import annotations

import collections
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..media import Cancelled, MediaError, _popen_kwargs, _run, _terminate, ffmpeg_path, ffprobe_path
from . import colorimetry
from .tasks import TaskContext, TaskError, TaskOutput, TaskResult, output_path

ENGINES = ("lut", "videotoolbox", "zscale", "libplacebo", "tonemapx")
ENGINE_LABELS = {
    "lut": "3D LUT (any FFmpeg)",
    "videotoolbox": "VideoToolbox (macOS)",
    "zscale": "zscale + tonemap",
    "libplacebo": "libplacebo",
    "tonemapx": "tonemapx (Jellyfin FFmpeg)",
}
#: The pack that brings an FFmpeg with zscale, libplacebo and libass.
FULL_FFMPEG_PACK = "ffmpeg-full"

DEFAULTS: Dict[str, Any] = {
    "engine": "auto",
    "algorithm": "bt2390",
    "resolution": "1080p",
    "sourcePeak": "auto",
    "targetPeak": 100,
    "desaturation": 0.5,
    "encoder": "x264",
    "crf": 18,
    "preset": "medium",
    "bitDepth": 8,
    "audio": "copy",
    "subtitles": "copy",
    "burnTrack": None,
    "container": "mkv",
    "dolbyVision": "auto",
}

RESOLUTION_BOXES = {
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
}

ENCODERS = {
    "x264": "libx264",
    "x265": "libx265",
    "h264_videotoolbox": "h264_videotoolbox",
    "hevc_videotoolbox": "hevc_videotoolbox",
}

FILTERS_OF_INTEREST = (
    "zscale", "libplacebo", "tonemap", "tonemapx", "tonemap_videotoolbox", "tonemap_opencl", "lut3d",
    "mergeplanes", "scale_vt",
    "colorspace", "overlay", "subtitles", "ass", "scdet", "sidedata", "setparams", "scale",
)
ENCODERS_OF_INTEREST = (
    "libx264", "libx265", "h264_videotoolbox", "hevc_videotoolbox", "aac", "aac_at", "mov_text",
)

#: Frame side data that only makes sense on HDR pictures. Deleted from every
#: output (only the names this FFmpeg knows are used).
HDR_SIDE_DATA = (
    "MASTERING_DISPLAY_METADATA", "CONTENT_LIGHT_LEVEL", "DYNAMIC_HDR_PLUS",
    "DOVI_RPU_BUFFER", "DOVI_METADATA", "DYNAMIC_HDR_VIVID", "AMBIENT_VIEWING_ENVIRONMENT",
)

#: Audio codecs an MP4 can carry as they are.
MP4_AUDIO_COPY = {"aac", "ac3", "eac3", "mp3", "alac", "flac", "opus"}

#: Silence between progress events is capped well below the host's
#: 30-minute watchdog even when a percent takes minutes on a long film.
HEARTBEAT_S = 5.0


# ------------------------------------------------------------ capabilities

_CAPS_CACHE: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
_HELP_CACHE: Dict[Tuple[str, str], str] = {}
_CACHE_LOCK = threading.Lock()


def _full_ffmpeg() -> Optional[str]:
    try:
        from . import packs  # noqa: PLC0415 - lazy: the packs module is optional here
    except Exception:  # noqa: BLE001 - no packs module means no pack
        return None
    try:
        path = packs.ffmpeg_full()
    except Exception:  # noqa: BLE001
        return None
    return path if path and os.path.isfile(path) else None


def ffmpeg_candidates() -> List[str]:
    """The FFmpeg binaries worth trying, best first: the full pack, then the
    app's own (bundled or PATH) FFmpeg."""
    out: List[str] = []
    for path in (_full_ffmpeg(), ffmpeg_path()):
        if path and path not in out:
            out.append(path)
    return out


def _cache_key(path: str) -> Tuple[str, float, int]:
    try:
        st = os.stat(path)
        return (path, st.st_mtime, st.st_size)
    except OSError:
        return (path, 0.0, 0)


def _quiet(command: List[str], timeout: int = 30) -> str:
    try:
        return _run(command, timeout, None, what=" ".join(command[1:3])).decode("utf-8", "replace")
    except MediaError:
        return ""


def _parse_listing(text: str, flag_pattern: str) -> List[str]:
    names = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(flag_pattern, parts[0]) and parts[1] != "=":
            names.append(parts[1])
    return names


def ffmpeg_capabilities(ffmpeg: Optional[str] = None) -> Dict[str, Any]:
    """What an FFmpeg binary can do, cached per binary (path + mtime).

    Shape matches ``Capabilities.ffmpeg`` in types.ts, plus ``hwaccels``.
    Without an argument, describes the preferred binary (the full pack when
    installed).
    """
    path = ffmpeg or ffmpeg_candidates()[0]
    key = _cache_key(path)
    with _CACHE_LOCK:
        cached = _CAPS_CACHE.get(key)
    if cached is not None:
        return cached
    version_text = _quiet([path, "-hide_banner", "-version"])
    caps: Dict[str, Any] = {"path": path, "version": None, "filters": {}, "encoders": {}, "hwaccels": []}
    if version_text:
        first = version_text.splitlines()[0].split()
        caps["version"] = first[2] if len(first) > 2 else None
        filters = set(_parse_listing(_quiet([path, "-hide_banner", "-filters"]), r"[TSC.|]{2,3}"))
        encoders = set(_parse_listing(_quiet([path, "-hide_banner", "-encoders"]), r"[VASFXBD.]{6}"))
        hw_text = _quiet([path, "-hide_banner", "-hwaccels"])
        hwaccels = [ln.strip() for ln in hw_text.splitlines()[1:] if ln.strip()]
        caps["filters"] = {name: name in filters for name in FILTERS_OF_INTEREST}
        caps["encoders"] = {name: name in encoders for name in ENCODERS_OF_INTEREST}
        caps["hwaccels"] = hwaccels
    else:
        caps["path"] = None if not os.path.isfile(path) else path
        caps["filters"] = {name: False for name in FILTERS_OF_INTEREST}
        caps["encoders"] = {name: False for name in ENCODERS_OF_INTEREST}
    with _CACHE_LOCK:
        _CAPS_CACHE[key] = caps
    return caps


def _help_text(ffmpeg: str, what: str) -> str:
    """``ffmpeg -h filter=X`` / ``encoder=X``, cached: option names changed
    across versions (mergeplanes' map options, libx265's -dolbyvision)."""
    key = (ffmpeg, what)
    with _CACHE_LOCK:
        if key in _HELP_CACHE:
            return _HELP_CACHE[key]
    text = _quiet([ffmpeg, "-hide_banner", "-h", what])
    with _CACHE_LOCK:
        _HELP_CACHE[key] = text
    return text


def _engine_problem(engine: str, caps: Dict[str, Any]) -> Optional[str]:
    """Why ``engine`` cannot run on this binary, or None when it can."""
    filters = caps.get("filters") or {}
    if not caps.get("version"):
        return "FFmpeg was not found."
    if engine == "lut":
        missing = [f for f in ("lut3d", "mergeplanes") if not filters.get(f)]
        return f"This FFmpeg lacks the {', '.join(missing)} filter." if missing else None
    if engine == "videotoolbox":
        if sys.platform != "darwin":
            return "VideoToolbox is only on macOS."
        if not filters.get("scale_vt") or "videotoolbox" not in (caps.get("hwaccels") or []):
            return "This FFmpeg was built without VideoToolbox (needs scale_vt, FFmpeg 7+)."
        return None
    if engine == "zscale":
        if not (filters.get("zscale") and filters.get("tonemap")):
            return "Needs an FFmpeg with zscale (zimg). Install the Full FFmpeg pack."
        return None
    if engine == "libplacebo":
        if not filters.get("libplacebo"):
            return ("Needs an FFmpeg with libplacebo: the Full FFmpeg pack on Windows and Linux "
                    "(its macOS build has no libplacebo).")
        return None
    if engine == "tonemapx":
        if not filters.get("tonemapx"):
            return "Needs the Jellyfin FFmpeg of the Full FFmpeg pack."
        return None
    return f"Unknown engine: {engine}"


def can_reshape_dolby_vision(engine: str, caps: Dict[str, Any]) -> bool:
    """Whether ``engine`` on this binary applies Dolby Vision RPU reshaping.

    libplacebo: ``apply_dolbyvision``. Jellyfin FFmpeg's
    ``tonemap_videotoolbox`` (Metal) and ``tonemapx`` (CPU) read the RPU
    FFmpeg's HEVC decoder exports (AV_FRAME_DATA_DOVI_METADATA) and apply
    its polynomial/MMR reshaping and IPT-PQ-c2 -> LMS -> RGB matrices when
    the RPU needs no enhancement layer (jellyfin-ffmpeg debian/patches 0046
    and 0053). tonemapx additionally requires ``vdr_rpu_profile == 0``,
    which its authors note is a guess from samples, so it ranks last.
    """
    filters = caps.get("filters") or {}
    if engine == "libplacebo":
        return bool(filters.get("libplacebo"))
    if engine == "videotoolbox":
        return bool(filters.get("tonemap_videotoolbox"))
    if engine == "tonemapx":
        return bool(filters.get("tonemapx"))
    return False


def binary_for(engine: str, dolby_vision: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """(ffmpeg path, None) for the first binary that can run ``engine`` --
    and reshape Dolby Vision, when asked -- or (None, reason)."""
    reason = None
    for path in ffmpeg_candidates():
        caps = ffmpeg_capabilities(path)
        problem = _engine_problem(engine, caps)
        if problem is None and dolby_vision and not can_reshape_dolby_vision(engine, caps):
            problem = f"The {ENGINE_LABELS.get(engine, engine)} engine of this FFmpeg cannot apply Dolby Vision."
        if problem is None:
            return path, None
        reason = reason or problem
    return None, reason or "FFmpeg was not found."


def engine_statuses() -> List[Dict[str, Any]]:
    out = []
    for engine in ENGINES:
        path, reason = binary_for(engine)
        out.append({
            "id": engine,
            "label": ENGINE_LABELS[engine],
            "available": path is not None,
            "reason": reason,
            "pack": FULL_FFMPEG_PACK if engine in ("zscale", "libplacebo", "tonemapx") and path is None else None,
        })
    return out


# ------------------------------------------------------------------ source


@dataclass
class Source:
    """What the tone-mapper needs to know about the input video."""

    path: str
    width: int
    height: int
    hdr: str
    #: "pq", "hlg", "ipt" (DV profile 5), or "sdr".
    transfer: str
    dv_profile: Optional[int] = None
    dv_compat: Optional[int] = None
    max_cll: Optional[float] = None
    mastering_peak: Optional[float] = None
    duration: Optional[float] = None
    full_range: bool = False
    audio: List[Dict[str, Any]] = field(default_factory=list)
    subtitles: List[Dict[str, Any]] = field(default_factory=list)


def base_layer_transfer(info: Dict[str, Any]) -> str:
    """The transfer of the picture a decoder outputs, before any DV RPU.

    Dolby Vision's base layer depends on the profile and its compatibility
    id: 8.1 and 7 carry HDR10 (PQ), 8.4 carries HLG, 8.2 carries SDR, and
    profile 5 carries IPT-PQ-c2, which no display shows correctly.
    """
    hdr = (info.get("hdr") or "sdr").lower()
    transfer = (info.get("transfer") or "").lower()
    profile = info.get("dvProfile")
    compat = info.get("dvCompatibility")
    if hdr == "dv" or profile:
        if profile == 5:
            return "ipt"
        if compat == 4:
            return "hlg"
        if compat == 2:
            return "sdr"
        if compat in (1, 6) or profile == 7:
            return "pq"
    if hdr == "hlg" or transfer == "arib-std-b67":
        return "hlg"
    if hdr in ("hdr10", "hdr10plus", "dv") or transfer == "smpte2084":
        return "pq"
    return "sdr"


def resolve_source_peak(option: Any, source: Source) -> Tuple[float, str]:
    """The peak the tone curve compresses from, and where it came from.

    MaxCLL is the brightest pixel actually in the film, the mastering peak
    the monitor it was graded on; the lower of the two is the tighter true
    bound. Values outside 100-10000 are treated as missing: authoring tools
    write 0 or 65535 when they did not measure.
    """
    if option not in (None, "", "auto"):
        try:
            value = float(option)
        except (TypeError, ValueError):
            raise TaskError(f"Source peak must be a number of nits or 'auto', not {option!r}.") from None
        if not 100 <= value <= 10000:
            raise TaskError("Source peak must be between 100 and 10000 nits.")
        return value, "set"
    if source.transfer == "hlg":
        return colorimetry.DEFAULT_SOURCE_PEAK_NITS, "HLG nominal peak (BT.2100)"

    def usable(v: Any) -> bool:
        return isinstance(v, (int, float)) and 100 < float(v) <= 10000

    cll, mastering = source.max_cll, source.mastering_peak
    if usable(cll) and usable(mastering):
        return (float(cll), "MaxCLL") if cll <= mastering else (float(mastering), "mastering display")
    if usable(cll):
        return float(cll), "MaxCLL"
    if usable(mastering):
        return float(mastering), "mastering display"
    return colorimetry.DEFAULT_SOURCE_PEAK_NITS, "default (no metadata)"


def fit_resolution(width: int, height: int, resolution: str) -> Tuple[int, int]:
    """Fit inside the named box keeping aspect; never upscale; even sizes
    (4:2:0 chroma needs them)."""
    width, height = int(width), int(height)
    if resolution in RESOLUTION_BOXES:
        box_w, box_h = RESOLUTION_BOXES[resolution]
        scale = min(box_w / width, box_h / height, 1.0)
    else:
        scale = 1.0
    w = max(2, int(math.floor(width * scale / 2.0 + 1e-6)) * 2)
    h = max(2, int(math.floor(height * scale / 2.0 + 1e-6)) * 2)
    if scale < 1.0:
        # Rounding each side down can nudge the aspect; prefer the exact
        # box edge on the side that fills it.
        if abs(width * scale - box_w) < 1:
            w = box_w - box_w % 2
        if abs(height * scale - box_h) < 1:
            h = box_h - box_h % 2
    return w, h


def resolution_label(width: int, height: int) -> str:
    """"2160p"/"1440p"/"1080p"/"720p" by the box a frame fills, else "<h>p".

    Judged by width too: a 3840x1600 scope film is "2160p" to everyone.
    """
    for label, (box_w, box_h) in RESOLUTION_BOXES.items():
        if width >= box_w * 0.9 or height >= box_h * 0.95:
            if width <= box_w * 1.05 and height <= box_h * 1.05:
                return label
    return f"{height}p"


def _probe_json(path: str, token) -> Dict[str, Any]:
    command = [ffprobe_path(), "-v", "error", "-show_entries",
               "format=duration:stream=index,codec_type,codec_name,pix_fmt,color_range,channels,"
               "width,height:stream_tags=language,title",
               "-of", "json", path]
    try:
        return json.loads(_run(command, 60, token, what=f"probe {os.path.basename(path)}").decode("utf-8", "replace"))
    except (MediaError, json.JSONDecodeError) as exc:
        raise TaskError(f"Could not read {os.path.basename(path)}: {exc}") from exc


def inspect_source(path: str, token=None) -> Source:
    """HDR facts from ``tracks.video_info`` (the one place that parses DV
    configuration records and mastering metadata), plus stream layout."""
    from . import tracks  # noqa: PLC0415 - lazy by design

    info = tracks.video_info(path, token)
    if not info:
        raise TaskError(f"No video stream in {os.path.basename(path)}.")
    probe = _probe_json(path, token)
    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = [
        {"index": n, "codec": s.get("codec_name"), "channels": s.get("channels")}
        for n, s in enumerate(s for s in streams if s.get("codec_type") == "audio")
    ]
    try:
        subs = [t.to_dict() for t in tracks.list_tracks(path, token)]
    except Exception:  # noqa: BLE001 - subtitles are optional for tone-mapping
        subs = []
    duration = None
    try:
        duration = float((probe.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return Source(
        path=path,
        width=int(info.get("width") or video.get("width") or 0),
        height=int(info.get("height") or video.get("height") or 0),
        hdr=str(info.get("hdr") or "sdr"),
        transfer=base_layer_transfer(info),
        dv_profile=info.get("dvProfile"),
        dv_compat=info.get("dvCompatibility"),
        max_cll=info.get("maxCll"),
        mastering_peak=info.get("masteringPeak"),
        duration=duration,
        full_range=(video.get("color_range") == "pc"),
        audio=audio,
        subtitles=subs,
    )


# -------------------------------------------------------------------- plan


@dataclass
class Plan:
    """Everything that decides the FFmpeg command, resolved and validated."""

    source: Source
    ffmpeg: str
    engine: str
    algorithm: str
    source_peak: float
    peak_source: str
    target_peak: float
    desaturation: float
    width: int
    height: int
    label: str
    encoder: str
    pix_fmt: str
    crf: int
    preset: str
    container: str
    audio: str
    subtitles: str
    burn_track: Optional[Dict[str, Any]] = None
    apply_dv: bool = False
    #: VideoToolbox through Jellyfin's Metal tone-mapper rather than scale_vt.
    vt_metal: bool = False
    lut_size: int = 65
    warnings: List[str] = field(default_factory=list)

    @property
    def resize(self) -> bool:
        return (self.width, self.height) != (self.source.width, self.source.height)

    @property
    def uses_lut(self) -> bool:
        return self.engine == "lut" or (self.engine == "zscale" and self.algorithm == "bt2390")


def _choose_engine(requested: str, source: Source, dolby_vision: str, algorithm: str) -> Tuple[str, str]:
    """Pick the engine (and its binary) or raise with what to do instead."""
    dv5 = source.transfer == "ipt"
    if dv5:
        explanation = (
            "This is Dolby Vision profile 5. Its base layer is IPT-PQ-c2, which only shows correct "
            "colours after the Dolby Vision reshaping carried in the RPU is applied; tone-mapping it "
            "directly gives a green and purple picture."
        )
        if dolby_vision == "baseLayer":
            raise TaskError(explanation + " Set Dolby Vision to auto so an engine that applies it is used.")
        if requested != "auto" and requested not in ENGINES:
            raise TaskError(f"Unknown tone-mapping engine: {requested}")
        order = ["libplacebo", "videotoolbox", "tonemapx"] if requested == "auto" else [requested]
        reason = None
        for engine in order:
            path, reason = binary_for(engine, dolby_vision=True)
            if path:
                return engine, path
        where = ("libplacebo (the Full FFmpeg pack on Windows and Linux) or, on macOS, VideoToolbox "
                 f"or tonemapx from the Full FFmpeg pack ({FULL_FFMPEG_PACK})")
        if requested == "auto":
            raise TaskError(explanation + f" The engines that apply it are {where}; none is available here.")
        raise TaskError(explanation + f" The {ENGINE_LABELS[requested]} engine cannot apply it here "
                        f"({reason}); use {where}.")
    if requested == "auto":
        # libplacebo first (per-scene peak detection, DV). Then the LUT: it
        # honours every setting and was measured to match its formula. zscale
        # next (zimg resizing, FFmpeg's tonemap curves); VideoToolbox after
        # (plain scale_vt clips above ~800 nits; the Jellyfin Metal filter is
        # sound but ignores target peak); tonemapx last (see _jellyfin_peak).
        order = ["libplacebo", "lut", "zscale", "videotoolbox", "tonemapx"]
        last_reason = None
        for engine in order:
            path, reason = binary_for(engine)
            if path:
                return engine, path
            last_reason = last_reason or reason
        raise TaskError(f"No tone-mapping engine can run: {last_reason}")
    if requested not in ENGINES:
        raise TaskError(f"Unknown tone-mapping engine: {requested}")
    path, reason = binary_for(requested)
    if not path:
        raise TaskError(f"The {ENGINE_LABELS[requested]} engine is not available: {reason}")
    return requested, path


def build_plan(options: Dict[str, Any], source: Source) -> Plan:
    opts = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None or k == "burnTrack"}}
    warnings: List[str] = []
    if source.transfer == "sdr":
        if source.hdr == "dv":
            raise TaskError("This Dolby Vision video has an SDR (BT.709) base layer (profile 8.2); "
                            "there is nothing to tone-map.")
        raise TaskError("This video is already SDR; there is nothing to tone-map.")
    if not source.width or not source.height:
        raise TaskError("Could not read the video's frame size.")

    algorithm = str(opts["algorithm"]).lower()
    if algorithm not in colorimetry.TONE_CURVES:
        raise TaskError(f"Unknown tone curve: {algorithm}")
    dolby_vision = str(opts.get("dolbyVision") or "auto")
    engine, ffmpeg = _choose_engine(str(opts["engine"]), source, dolby_vision, algorithm)

    source_peak, peak_source = resolve_source_peak(opts.get("sourcePeak"), source)
    try:
        target_peak = float(opts.get("targetPeak") or 100)
    except (TypeError, ValueError):
        raise TaskError("Target peak must be a number of nits.") from None
    if not 48 <= target_peak <= 1000:
        raise TaskError("Target peak must be between 48 and 1000 nits.")
    if source_peak <= target_peak:
        warnings.append(f"The source peak ({source_peak:g} nits) is not above the target "
                        f"({target_peak:g} nits), so highlights are not compressed.")
    desaturation = float(min(max(float(opts.get("desaturation") or 0.0), 0.0), 1.0))

    resolution = str(opts.get("resolution") or "source")
    width, height = fit_resolution(source.width, source.height, resolution)
    box = RESOLUTION_BOXES.get(resolution)
    if box and source.width < box[0] and source.height < box[1]:
        warnings.append(f"The video is smaller than {resolution}; kept at {width}x{height} (never upscaled).")
    label = resolution_label(width, height)

    encoder_key = str(opts.get("encoder") or "x264")
    encoder = ENCODERS.get(encoder_key)
    if not encoder:
        raise TaskError(f"Unknown encoder: {encoder_key}")
    caps = ffmpeg_capabilities(ffmpeg)
    if not (caps.get("encoders") or {}).get(encoder):
        raise TaskError(f"This FFmpeg has no {encoder} encoder.")
    bit_depth = int(opts.get("bitDepth") or 8)
    if encoder in ("libx265", "hevc_videotoolbox") and bit_depth == 10:
        pix_fmt = "yuv420p10le" if encoder == "libx265" else "p010le"
    else:
        if bit_depth == 10:
            warnings.append("10-bit output needs x265 or HEVC VideoToolbox; H.264 is written as 8-bit.")
        pix_fmt = "nv12" if encoder.endswith("videotoolbox") else "yuv420p"

    container = str(opts.get("container") or "mkv").lower()
    if container not in ("mkv", "mp4"):
        raise TaskError(f"Unknown container: {container}")
    audio = str(opts.get("audio") or "copy")
    subtitles = str(opts.get("subtitles") or "copy")
    burn_track = None
    if subtitles == "burn":
        burn_track = _pick_burn_track(opts.get("burnTrack"), source)
        if burn_track.get("kind") != "image" and not (caps.get("filters") or {}).get("subtitles"):
            raise TaskError(
                "Burning a text subtitle needs FFmpeg's subtitles filter (libass), which this FFmpeg "
                f"lacks. Install the Full FFmpeg pack ({FULL_FFMPEG_PACK}), or copy the subtitles instead."
            )

    vt_metal = engine == "videotoolbox" and bool((caps.get("filters") or {}).get("tonemap_videotoolbox"))
    apply_dv = (dolby_vision == "auto" and bool(source.dv_profile)
                and can_reshape_dolby_vision(engine, caps))
    if source.dv_profile and not apply_dv and source.transfer != "ipt":
        warnings.append(f"Dolby Vision metadata ignored: the profile {source.dv_profile} base layer "
                        f"({'HLG' if source.transfer == 'hlg' else 'HDR10'}) was tone-mapped.")
    if engine == "videotoolbox" and not vt_metal:
        warnings.append("VideoToolbox uses its own fixed tone curve: source peak, target peak, curve and "
                        "desaturation settings do not apply, and highlights above ~800 nits clip.")
    if engine == "tonemapx" or vt_metal:
        warnings.append("This engine puts SDR white at the 203-nit HDR reference white (BT.2408) and "
                        "takes the source peak from the metadata unless you set one; the target peak "
                        "and desaturation settings do not apply.")
    if engine == "tonemapx" and source.transfer == "ipt":
        warnings.append("tonemapx reshapes Dolby Vision only for the RPU type most profile 5 files use; "
                        "if colours come out green or purple, use VideoToolbox from the same pack.")
    if engine == "zscale" and algorithm == "bt2390":
        warnings.append("zscale's tonemap filter has no BT.2390; zscale resizes and the 3D LUT maps.")
    elif engine == "zscale" and desaturation > 0:
        warnings.append("The zscale engine's tonemap filter runs without highlight desaturation.")
    if engine == "libplacebo":
        warnings.append("libplacebo takes the peaks from the metadata and its own per-scene peak "
                        "detection; the source peak, target peak and desaturation settings do not apply.")

    return Plan(
        source=source, ffmpeg=ffmpeg, engine=engine, algorithm=algorithm,
        source_peak=source_peak, peak_source=peak_source, target_peak=target_peak,
        desaturation=desaturation, width=width, height=height, label=label,
        encoder=encoder, pix_fmt=pix_fmt, crf=int(opts.get("crf") if opts.get("crf") is not None else 18),
        preset=str(opts.get("preset") or "medium"), container=container, audio=audio,
        subtitles=subtitles, burn_track=burn_track, apply_dv=apply_dv, vt_metal=vt_metal, warnings=warnings,
    )


def _pick_burn_track(requested: Any, source: Source) -> Dict[str, Any]:
    tracks_ = source.subtitles
    if not tracks_:
        raise TaskError("There is no subtitle track to burn in.")
    if requested is not None:
        found = next((t for t in tracks_ if int(t.get("index", -1)) == int(requested)), None)
        if not found:
            raise TaskError(f"Subtitle track {int(requested) + 1} does not exist.")
        return found
    return next((t for t in tracks_ if t.get("default")), tracks_[0])


# ----------------------------------------------------------- filter graph


def escape_filter_value(value: str) -> str:
    """Escape a string for use as a filter option value inside a graph.

    Two levels, per ffmpeg-filters "Notes on filtergraph escaping": first
    the option parser (``\\ ' :``), then the graph parser (``\\ ' [ ] , ;``).
    """
    first = value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    out = []
    for ch in first:
        if ch in "\\'[],;":
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _mergeplanes(ffmpeg: str, fmt: str) -> str:
    """Relabel planes 0/1/2 as-is under another pixel format (no conversion)."""
    if "map0s" in _help_text(ffmpeg, "filter=mergeplanes"):
        return f"mergeplanes=map0s=0:map0p=0:map1s=0:map1p=1:map2s=0:map2p=2:format={fmt}"
    return f"mergeplanes=0x000102:{fmt}"


def _strip_hdr_side_data(ffmpeg: str) -> List[str]:
    known = _help_text(ffmpeg, "filter=sidedata")
    return [f"sidedata=mode=delete:type={name}" for name in HDR_SIDE_DATA if re.search(rf"\b{name}\b", known)]


#: LUT files, written into the job's scratch directory (FFmpeg's cwd).
MATRIX_LUT = "ycbcr2rgb.cube"
TONEMAP_LUT = "tonemap.cube"

#: Input index of the second copy of the source used for burning bitmap
#: subtitles (see ``filter_graph``).
BURN_INPUT = 1

SDR_TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"

_PLACEBO_CURVES = {"bt2390": "bt.2390", "hable": "hable", "mobius": "mobius", "reinhard": "reinhard", "clip": "clip"}


def video_chain(plan: Plan) -> List[str]:
    """The tone-mapping filters from the decoded frame to SDR ``plan.pix_fmt``."""
    w, h = plan.width, plan.height
    out_fmt = "yuv420p10le" if plan.pix_fmt in ("yuv420p10le", "p010le") else "yuv420p"
    chain: List[str] = []
    lut_steps = [
        # Relabel, not convert: Y'/Cb/Cr planes become G/B/R untouched.
        _mergeplanes(plan.ffmpeg, "gbrp10le"),
        f"lut3d=file={MATRIX_LUT}:interp=tetrahedral",
        f"lut3d=file={TONEMAP_LUT}:interp=tetrahedral",
        _mergeplanes(plan.ffmpeg, "yuv444p10le"),
    ]
    if plan.engine == "lut":
        if plan.resize:
            chain.append(f"scale=w={w}:h={h}:flags=lanczos")
        chain.append("format=yuv444p10le")
        chain += lut_steps
        chain.append(f"format={out_fmt}")
    elif plan.engine == "zscale":
        if plan.algorithm == "bt2390":
            if plan.resize:
                chain.append(f"zscale=w={w}:h={h}:filter=lanczos")
            chain.append("format=yuv444p10le")
            chain += lut_steps
            chain.append(f"format={out_fmt}")
        else:
            peak = max(plan.source_peak / plan.target_peak, 1.0)
            size = f"w={w}:h={h}:filter=lanczos:" if plan.resize else ""
            chain += [
                f"zscale={size}t=linear:npl={plan.target_peak:g}",
                "format=gbrpf32le",
                "zscale=p=bt709",
                f"tonemap=tonemap={plan.algorithm}:peak={peak:.4f}:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
                f"format={out_fmt}",
                # tonemap's curves pass 1.0 for light above ``peak`` and zscale
                # writes that as super-white (measured Y' 254-255); clamp to
                # the legal range the output is tagged with.
                "lutyuv=y=clipval:u=clipval:v=clipval",
            ]
    elif plan.engine == "videotoolbox" and plan.vt_metal:
        vt_fmt = "p010le" if out_fmt == "yuv420p10le" else "nv12"
        chain += [
            f"scale_vt=w={w}:h={h}",
            f"tonemap_videotoolbox=tonemap={plan.algorithm}:t=bt709:m=bt709:p=bt709:r=tv:format={vt_fmt}:"
            f"apply_dovi={1 if plan.apply_dv else 0}{_jellyfin_peak(plan)}",
            "hwdownload",
            f"format={vt_fmt}",
            f"format={out_fmt}",
        ]
    elif plan.engine == "tonemapx":
        if plan.resize:
            chain.append(f"scale=w={w}:h={h}:flags=lanczos")
        chain += [
            # tonemapx takes yuv420p10/p010 only (and planar for DV reshaping).
            "format=yuv420p10le",
            f"tonemapx=tonemap={plan.algorithm}:t=bt709:m=bt709:p=bt709:r=tv:format={out_fmt}:"
            f"apply_dovi={1 if plan.apply_dv else 0}:desat=0{_jellyfin_peak(plan)}",
        ]
    elif plan.engine == "videotoolbox":
        chain += [
            f"scale_vt=w={w}:h={h}:color_matrix=bt709:color_primaries=bt709:color_transfer=bt709",
            "hwdownload",
            "format=p010le",
            f"format={out_fmt}",
        ]
    elif plan.engine == "libplacebo":
        chain.append(
            f"libplacebo=w={w}:h={h}:format={out_fmt}:colorspace=bt709:color_primaries=bt709:"
            f"color_trc=bt709:range=tv:tonemapping={_PLACEBO_CURVES[plan.algorithm]}:"
            f"apply_dolbyvision={1 if plan.apply_dv else 0}"
        )
    else:
        raise TaskError(f"Unknown engine: {plan.engine}")
    return chain


def main_filter(plan: Plan) -> str:
    """The filter that does the tone-mapping, for the report."""
    if plan.engine == "videotoolbox":
        return "tonemap_videotoolbox" if plan.vt_metal else "scale_vt"
    if plan.uses_lut:
        return "lut3d"
    return {"zscale": "zscale+tonemap", "libplacebo": "libplacebo", "tonemapx": "tonemapx"}[plan.engine]


def _jellyfin_peak(plan: Plan) -> str:
    """``peak`` for tonemapx / tonemap_videotoolbox, in their unit of 10
    nits (patch 0053: src_peak = peak / 10 * 100 / 203, in 203-nit units).

    Left out, tonemap_videotoolbox reads MaxCLL / mastering / the DV RPU per
    frame -- measured correct. tonemapx does not: measured, it assumed 1000
    nits whatever the metadata said, and anything brighter than its peak
    came out *darker* (4000 nits -> Y' 208, 10000 -> 168, below 1000's
    235). So it always gets the highest bound the metadata allows, unless
    the Dolby Vision RPU supplies the peak.
    """
    if plan.peak_source == "set":
        return f":peak={plan.source_peak / 10.0:g}"
    if plan.engine == "tonemapx" and not plan.apply_dv:
        bounds = [plan.source_peak] + [float(v) for v in (plan.source.max_cll, plan.source.mastering_peak)
                                       if isinstance(v, (int, float)) and 0 < v <= 10000]
        return f":peak={max(bounds) / 10.0:g}"
    return ""


def filter_graph(plan: Plan) -> str:
    """The complete ``-filter_complex`` graph ending in ``[vout]``."""
    chain = video_chain(plan)
    tail = [SDR_TAGS] + _strip_hdr_side_data(plan.ffmpeg)
    burn = plan.burn_track if plan.subtitles == "burn" else None
    if burn and burn.get("kind") == "image":
        # Bitmap subtitles are drawn on the finished SDR picture, scaled from
        # the source canvas to the output size, converted with the BT.709
        # matrix the picture now uses (the auto-inserted scaler would pick
        # BT.601 and shift coloured subtitles).
        #
        # Timing, measured with a PGS event at 0.5-1.5 s on 24 fps video:
        # read from the video's own input, FFmpeg's sub2video heartbeat runs
        # ahead of B-frame reordering and the subtitle appeared 5 frames
        # late. From a second input of the same file it is 1 frame late,
        # because sub2video emits a blank frame and the caption with the
        # same timestamp and overlay takes the blank one; pulling the
        # subtitle stream 1 ms earlier makes it frame-exact (12-35).
        n = int(burn["index"])
        return (
            f"[0:v:0]{','.join(chain)}[main];"
            f"[{BURN_INPUT}:s:{n}]setpts=PTS-0.001/TB,"
            f"scale=w={plan.width}:h={plan.height}:out_color_matrix=bt709:out_range=tv,"
            f"format=yuva420p[subs];"
            f"[main][subs]overlay=eof_action=pass:format=auto,format={chain_out_format(plan)},"
            f"{','.join(tail)}[vout]"
        )
    if burn:
        n = int(burn["index"])
        chain.append(
            f"subtitles=filename={escape_filter_value(plan.source.path)}:si={n}:"
            f"original_size={plan.source.width}x{plan.source.height}"
        )
    return f"[0:v:0]{','.join(chain + tail)}[vout]"


def chain_out_format(plan: Plan) -> str:
    return "yuv420p10le" if plan.pix_fmt in ("yuv420p10le", "p010le") else "yuv420p"


def _vt_quality(crf: int) -> int:
    # One "lower is better" CRF control drives every encoder; VideoToolbox's
    # -q:v is 1-100 higher-is-better. CRF 18 -> 64, 23 -> 54, 28 -> 44.
    return int(min(95, max(20, round(100 - 2 * crf))))


def _vt_bitrate(width: int, height: int) -> str:
    pixels = width * height
    return "35M" if pixels > 2560 * 1440 else "12M" if pixels > 1280 * 720 else "6M"


def encoder_args(plan: Plan) -> List[str]:
    args = ["-c:v", plan.encoder]
    if plan.encoder in ("libx264", "libx265"):
        args += ["-preset", plan.preset, "-crf", str(plan.crf), "-pix_fmt", plan.pix_fmt]
        if plan.encoder == "libx265":
            args += ["-x265-params", "log-level=error"]
            # FFmpeg 7.1+ libx265 writes Dolby Vision RPUs by default when the
            # input stream carries a DV configuration: an SDR file would then
            # claim to be Dolby Vision.
            if "dolbyvision" in _help_text(plan.ffmpeg, "encoder=libx265"):
                args += ["-dolbyvision", "0"]
    else:
        args += ["-pix_fmt", plan.pix_fmt]
        if platform.machine().lower() in ("arm64", "aarch64"):
            args += ["-q:v", str(_vt_quality(plan.crf))]
        else:
            # Constant-quality VideoToolbox needs Apple Silicon.
            args += ["-b:v", _vt_bitrate(plan.width, plan.height)]
        if plan.encoder == "hevc_videotoolbox" and plan.pix_fmt == "p010le":
            args += ["-profile:v", "main10"]
    if plan.encoder in ("libx265", "hevc_videotoolbox"):
        args += ["-tag:v", "hvc1"]
    args += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    return args


def stream_args(plan: Plan) -> Tuple[List[str], List[str]]:
    """(-map/-c args for audio and subtitles, warnings)."""
    args: List[str] = []
    warnings: List[str] = []
    for track in plan.source.audio:
        n = track["index"]
        args += ["-map", f"0:a:{n}"]
        codec = (track.get("codec") or "").lower()
        reencode = plan.audio == "aac" or (plan.container == "mp4" and codec not in MP4_AUDIO_COPY)
        if reencode:
            channels = int(track.get("channels") or 2)
            args += [f"-c:a:{n}", "aac", f"-b:a:{n}", f"{min(640, max(128, 64 * channels))}k"]
            if plan.audio == "copy":
                warnings.append(f"Audio track {n + 1} ({codec}) cannot go in MP4 as is; encoded to AAC.")
        else:
            args += [f"-c:a:{n}", "copy"]
    if plan.subtitles == "copy":
        if plan.container == "mkv":
            args += ["-map", "0:s?", "-c:s", "copy", "-map", "0:t?"]
        else:
            out_n = 0
            for track in plan.source.subtitles:
                if track.get("kind") == "image":
                    warnings.append(f"Subtitle track {int(track['index']) + 1} ({track.get('codec')}) is an "
                                    "image subtitle, which MP4 cannot hold; left out.")
                    continue
                args += ["-map", f"0:s:{int(track['index'])}", f"-c:s:{out_n}", "mov_text"]
                out_n += 1
    return args, warnings


def ffmpeg_command(plan: Plan, out_path: str) -> List[str]:
    command = [plan.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if plan.engine == "videotoolbox":
        command += ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
    command += ["-i", plan.source.path]
    if plan.subtitles == "burn" and plan.burn_track and plan.burn_track.get("kind") == "image":
        # The same file again, read only for its subtitles (see filter_graph).
        command += ["-i", plan.source.path]
    command += ["-filter_complex", filter_graph(plan), "-map", "[vout]"]
    streams, _ = stream_args(plan)
    command += encoder_args(plan) + streams
    command += ["-map_metadata", "0", "-map_chapters", "0"]
    if plan.container == "mp4":
        command += ["-movflags", "+faststart"]
    command += ["-progress", "pipe:1", "-nostats", out_path]
    return command


def write_lut(plan: Plan, workdir: str) -> List[str]:
    """Build the plan's two LUTs into ``workdir``. FFmpeg runs there, so the
    filters name them relatively and no path escaping is involved.

    ``MATRIX_LUT`` turns the relabelled Y'CbCr codes into BT.2020 R'G'B'
    (exactly: it is affine); ``TONEMAP_LUT`` does everything else and hands
    back BT.709 Y'CbCr codes. See ``colorimetry`` for why it is split so.
    """
    transfer = "hlg" if plan.source.transfer == "hlg" else "pq"
    colorimetry.write_cube(
        os.path.join(workdir, MATRIX_LUT),
        colorimetry.ycbcr_to_rgb_lut(bits=10, full_range=plan.source.full_range),
        title="BT.2020 Y'CbCr (Cr,Y,Cb) -> R'G'B'", clip=False,
    )
    lut = colorimetry.build_tonemap_lut(
        plan.lut_size, transfer, plan.algorithm, plan.source_peak, plan.target_peak, plan.desaturation,
    )
    colorimetry.write_cube(
        os.path.join(workdir, TONEMAP_LUT), lut,
        title=f"{transfer.upper()} {plan.source_peak:g} nits -> SDR {plan.target_peak:g} nits ({plan.algorithm})",
    )
    return [MATRIX_LUT, TONEMAP_LUT]


# ------------------------------------------------------------------ runner


def run_ffmpeg(
    command: List[str],
    ctx: TaskContext,
    duration: Optional[float],
    stage: str = "Tone-mapping",
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Run FFmpeg with ``-progress pipe:1``, reporting percent as it goes.

    The process is registered with the job's cancellation token, so Cancel
    kills it (and its process group) at once. stderr is drained on its own
    thread; the last lines become the error message on failure.
    """
    ctx.check()
    try:
        process = subprocess.Popen(command, cwd=cwd, **_popen_kwargs())
    except FileNotFoundError as exc:
        raise TaskError(f"{os.path.basename(command[0])} not found.") from exc
    except OSError as exc:
        raise TaskError(f"Could not start FFmpeg: {exc}") from exc

    tail: collections.deque = collections.deque(maxlen=40)

    def drain() -> None:
        assert process.stderr is not None
        for raw in process.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                tail.append(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    ctx.token.register(process)
    stats: Dict[str, Any] = {"outTimeS": 0.0, "speed": None, "frames": 0}
    last_pct, last_emit = -1, 0.0
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            key, _, value = raw.decode("utf-8", "replace").strip().partition("=")
            if key in ("out_time_us", "out_time_ms") and value not in ("", "N/A"):
                try:
                    # Both are microseconds (out_time_ms is misnamed in FFmpeg).
                    stats["outTimeS"] = max(0.0, int(value) / 1e6)
                except ValueError:
                    pass
            elif key == "speed" and value.endswith("x"):
                try:
                    stats["speed"] = float(value[:-1])
                except ValueError:
                    pass
            elif key == "frame":
                try:
                    stats["frames"] = int(value)
                except ValueError:
                    pass
            elif key == "progress":
                pct = 0
                if duration and duration > 0:
                    pct = int(min(99.0, 100.0 * stats["outTimeS"] / duration))
                now = time.monotonic()
                if pct != last_pct or now - last_emit >= HEARTBEAT_S:
                    speed = f", {stats['speed']:.2f}x" if stats["speed"] else ""
                    ctx.progress(pct, f"{stage} ({_clock(stats['outTimeS'])}{speed})")
                    last_pct, last_emit = pct, now
        process.wait()
        reader.join(timeout=5)
    finally:
        ctx.token.unregister(process)
        if process.poll() is None:
            _terminate(process)
    if ctx.token.cancelled:
        raise Cancelled("operation cancelled")
    if process.returncode != 0:
        detail = [ln for ln in tail if ln.strip()]
        raise TaskError("FFmpeg failed: " + (" | ".join(detail[-3:]) if detail else f"exit code {process.returncode}"))
    return stats


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def verify_output(path: str, token=None) -> Dict[str, Any]:
    """Read back the output's colour tags and HDR side data."""
    command = [ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
               "-show_entries",
               "stream=color_transfer,color_primaries,color_space,color_range,width,height,pix_fmt:"
               "stream_side_data=side_data_type:frame_side_data=side_data_type",
               "-show_frames", "-of", "json", path]
    data = json.loads(_run(command, 60, token, what="check output").decode("utf-8", "replace"))
    stream = (data.get("streams") or [{}])[0]
    frames = data.get("frames") or [{}]
    side = [d.get("side_data_type", "") for d in stream.get("side_data_list") or []]
    side += [d.get("side_data_type", "") for d in frames[0].get("side_data_list") or []]
    hdr_side = [s for s in side if re.search(r"mastering|light level|dolby|dovi|hdr", s, re.IGNORECASE)]
    return {
        "transfer": stream.get("color_transfer"),
        "primaries": stream.get("color_primaries"),
        "matrix": stream.get("color_space"),
        "range": stream.get("color_range"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pixFmt": stream.get("pix_fmt"),
        "hdrSideData": sorted(set(hdr_side)),
    }


# -------------------------------------------------------------------- task


def _hdr_label(source: Source) -> str:
    if source.dv_profile:
        compat = f".{source.dv_compat}" if source.dv_compat is not None and source.dv_profile == 8 else ""
        return f"Dolby Vision {source.dv_profile}{compat}"
    return {"hdr10": "HDR10", "hdr10plus": "HDR10+", "hlg": "HLG"}.get(source.hdr, source.hdr.upper())


def run_task(job: dict, ctx: TaskContext) -> TaskResult:
    started = time.monotonic()
    inputs = job.get("input") or {}
    ref = inputs.get("video") or inputs.get("subtitle") or {}
    path = ref.get("path")
    if not path or not os.path.isfile(path):
        raise TaskError(f"Video not found: {path}")
    options = {**DEFAULTS, **(job.get("options") or {})}

    ctx.progress(0, "Reading HDR metadata")
    source = inspect_source(path, ctx.token)
    plan = build_plan(options, source)
    ctx.log(
        f"{_hdr_label(source)} {source.width}x{source.height}, peak {plan.source_peak:g} nits "
        f"({plan.peak_source}) -> SDR {plan.target_peak:g} nits with {ENGINE_LABELS[plan.engine]}"
    )
    if plan.uses_lut:
        ctx.progress(0, "Building the 3D LUT")
        write_lut(plan, ctx.workdir)

    ext = plan.container
    out_ref = {"path": path}
    out_path = output_path(out_ref, job, f".sdr.{plan.label}", ext)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    command = ffmpeg_command(plan, out_path)
    _, stream_warnings = stream_args(plan)
    warnings = plan.warnings + stream_warnings

    try:
        stats = run_ffmpeg(command, ctx, source.duration, cwd=ctx.workdir)
    except BaseException:
        # A cancelled or failed run leaves a truncated file that would look
        # like a finished conversion in a folder listing.
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise

    ctx.progress(99, "Checking the output")
    check = verify_output(out_path, ctx.token)
    if check["hdrSideData"]:
        warnings.append("The output still carries HDR metadata: " + ", ".join(check["hdrSideData"]))
    if check["transfer"] != "bt709" or check["primaries"] != "bt709":
        warnings.append(f"The output is tagged {check['primaries']}/{check['transfer']}, not BT.709.")
    elapsed = time.monotonic() - started
    speed = stats.get("speed")
    if speed is None and source.duration:
        speed = source.duration / max(elapsed, 1e-6)

    engine_detail = ENGINE_LABELS[plan.engine]
    curve = "VideoToolbox's own curve" if plan.engine == "videotoolbox" and not plan.vt_metal else {
        "bt2390": "BT.2390", "hable": "Hable", "mobius": "Mobius", "reinhard": "Reinhard", "clip": "clip",
    }[plan.algorithm]
    quality = f"CRF {plan.crf}" if plan.encoder.startswith("lib") else f"quality {_vt_quality(plan.crf)}"
    summary = (
        f"Tone-mapped {_hdr_label(source)} ({plan.source_peak:g} nits, {plan.peak_source}) to SDR BT.709 "
        f"at {plan.width}x{plan.height} with {engine_detail} ({curve}), {plan.encoder} {quality}"
        + (f", {speed:.2f}x realtime." if speed else ".")
    )
    report = {
        "source": {
            "hdr": source.hdr,
            "dvProfile": source.dv_profile,
            "peakNits": plan.source_peak,
            "peakSource": plan.peak_source,
            "resolution": f"{source.width}x{source.height}",
        },
        "engine": plan.engine,
        "filter": main_filter(plan),
        "ffmpeg": ffmpeg_capabilities(plan.ffmpeg).get("version"),
        "algorithm": plan.algorithm if plan.engine != "videotoolbox" or plan.vt_metal else "videotoolbox",
        "targetPeak": plan.target_peak,
        "dolbyVisionApplied": plan.apply_dv,
        "outputResolution": f"{plan.width}x{plan.height}",
        "encoder": plan.encoder,
        "elapsedS": round(elapsed, 2),
        "speed": round(speed, 3) if speed else None,
        "output": check,
    }
    return TaskResult(
        outputs=[TaskOutput(path=out_path, kind="video", format=plan.container, label=f"SDR {plan.label}")],
        report=report,
        summary=summary,
        warnings=warnings,
    )
