"""Subtitle streams inside a video: list them, pull them out, add them in.

Also answers "what is this file?" for the Subsync file list (``probe_file``)
and "what kind of picture is this?" (``video_info``): HDR10, HDR10+, HLG and
Dolby Vision decide how bright subtitles must be and whether a tone-map is
needed, so the HDR details are read here with the rest of the probe.

MKVToolNix is preferred where it is present: ``mkvmerge`` writes the
Matroska flags players actually honour (forced, hearing-impaired) and is
the only way to get DVD VobSub out as ``.idx/.sub``. FFmpeg is the
fallback so nothing here *requires* MKVToolNix.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from ..media import (
    CancellationToken,
    MediaError,
    PROBE_TIMEOUT_S,
    _parse_frame_rate,
    _run,
    ffmpeg_path,
    ffprobe_path,
    is_audio_file,
)
from . import formats, languages
from .tasks import TaskError, TaskOutput, TaskResult, output_path

EXTRACT_TIMEOUT_S = 1800
MUX_TIMEOUT_S = 3600

IMAGE_CODECS = frozenset({"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"})

# What ffmpeg can stream-copy straight into a standalone file.
_COPY_EXTENSIONS = {"subrip": "srt", "ass": "ass", "ssa": "ass", "webvtt": "vtt"}

# Text formats mkvmerge and ffmpeg read directly; the rest go through SRT.
_MUXABLE_TEXT = frozenset({"srt", "ass", "ssa", "vtt"})

# Codecs Matroska takes as they are, and what MP4 can hold (mov_text only).
_MKV_SUBTITLE_CODECS = frozenset({"subrip", "ass", "ssa", "webvtt", "hdmv_pgs_subtitle", "dvd_subtitle",
                                  "dvb_subtitle", "text"})


@dataclass
class SubtitleTrack:
    #: N of ``0:s:N``.
    index: int
    #: Absolute stream index in the container.
    stream_index: int
    codec: str
    #: "text" or "image"
    kind: str
    language: Optional[str] = None
    title: Optional[str] = None
    forced: bool = False
    default: bool = False
    hearing_impaired: bool = False
    #: Number of subtitle events, when the muxer recorded statistics.
    events: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "codec": self.codec,
            "kind": self.kind,
            "language": self.language,
            "title": self.title,
            "forced": self.forced,
            "default": self.default,
            "hearingImpaired": self.hearing_impaired,
            "events": self.events,
        }


# ------------------------------------------------------------------ probing


def _probe_json(path: str, token: Optional[CancellationToken], extra: Optional[List[str]] = None) -> dict:
    if not os.path.isfile(path):
        raise MediaError(f"File not found: {path}")
    command = [ffprobe_path(), "-v", "error", *(extra or ["-show_streams", "-show_format"]), "-of", "json", path]
    stdout = _run(command, PROBE_TIMEOUT_S, token, what=f"probe {os.path.basename(path)}")
    try:
        return json.loads(stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"Could not parse ffprobe output for {os.path.basename(path)}") from exc


def _tag(tags: Dict[str, Any], *names: str) -> Optional[str]:
    """Tag lookup ignoring case and Matroska's ``-eng`` statistics suffix."""
    lowered = {str(k).lower(): v for k, v in (tags or {}).items()}
    for name in names:
        for key in (name.lower(), f"{name.lower()}-eng"):
            value = lowered.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _language(raw: Optional[str]) -> Optional[str]:
    if not raw or raw.lower() in ("und", "mul", "zxx", "unknown"):
        return None
    return languages.normalize(raw) or raw


def _tracks_from_streams(streams: List[dict]) -> List[SubtitleTrack]:
    tracks: List[SubtitleTrack] = []
    for stream in streams:
        if stream.get("codec_type") != "subtitle":
            continue
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        codec = stream.get("codec_name") or stream.get("codec_tag_string") or "unknown"
        title = _tag(tags, "title")
        lowered_title = (title or "").lower()
        events = _tag(tags, "NUMBER_OF_FRAMES")
        tracks.append(SubtitleTrack(
            index=len(tracks),
            stream_index=int(stream.get("index", len(tracks))),
            codec=codec,
            kind="image" if codec in IMAGE_CODECS else "text",
            language=_language(_tag(tags, "language")),
            title=title,
            # Many releases only say "Forced"/"SDH" in the title.
            forced=bool(disposition.get("forced")) or "forced" in lowered_title,
            default=bool(disposition.get("default")),
            hearing_impaired=bool(disposition.get("hearing_impaired")) or "sdh" in lowered_title.split(),
            events=int(events) if events and events.isdigit() else None,
        ))
    return tracks


def list_tracks(path: str, token: Optional[CancellationToken] = None) -> List[SubtitleTrack]:
    """Every subtitle stream in a video, in ``0:s:N`` order."""
    payload = _probe_json(path, token, ["-show_streams"])
    return _tracks_from_streams(payload.get("streams") or [])


def _main_video(streams: List[dict]) -> Optional[dict]:
    for stream in streams:
        if stream.get("codec_type") == "video" and not (stream.get("disposition") or {}).get("attached_pic"):
            return stream
    return None


def _rational(value: Any) -> Optional[float]:
    """``"10000000/10000"`` -> 1000.0 (no range limit, unlike frame rates)."""
    if value in (None, "", "N/A"):
        return None
    try:
        if isinstance(value, str) and "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(value)
    except (TypeError, ValueError):
        return None


def _side_data(entries: List[dict], kind: str) -> Optional[dict]:
    return next((sd for sd in entries or [] if sd.get("side_data_type") == kind), None)


def _video_info_from(path: str, stream: dict, token: Optional[CancellationToken]) -> Dict[str, Any]:
    transfer = stream.get("color_transfer")
    primaries = stream.get("color_primaries")
    side = list(stream.get("side_data_list") or [])
    dovi = _side_data(side, "DOVI configuration record")

    # HDR10 metadata is often only in the bitstream's SEI, which ffprobe
    # reports on the first decoded frame rather than on the stream -- and
    # HDR10+ dynamic metadata only ever lives there. Some muxes (ffmpeg's own
    # x265 MKVs among them) leave even the transfer "unknown" on the stream
    # while every frame says PQ, so a high-bit-depth stream without a known
    # transfer is checked too. One frame is enough.
    unknown_transfer = transfer in (None, "unknown", "unspecified")
    deep = any(bits in str(stream.get("pix_fmt") or "") for bits in ("10", "12"))
    frame_side: List[dict] = []
    if dovi or transfer in ("smpte2084", "arib-std-b67") or (unknown_transfer and deep):
        try:
            frames = _probe_json(path, token, [
                "-select_streams", str(stream.get("index", 0)), "-read_intervals", "%+#1", "-show_frames",
            ]).get("frames") or []
            if frames:
                frame_side = list(frames[0].get("side_data_list") or [])
                if unknown_transfer:
                    transfer = frames[0].get("color_transfer") or transfer
                if primaries in (None, "unknown", "unspecified"):
                    primaries = frames[0].get("color_primaries") or primaries
        except MediaError:
            frame_side = []
    every = side + frame_side

    mastering = _side_data(every, "Mastering display metadata")
    light = _side_data(every, "Content light level metadata")
    hdr10plus = any("2094-40" in str(sd.get("side_data_type")) for sd in frame_side)
    dv_frames = any("dolby vision" in str(sd.get("side_data_type")).lower() for sd in frame_side)

    if dovi or dv_frames:
        hdr = "dv"
    elif hdr10plus:
        hdr = "hdr10plus"
    elif transfer == "smpte2084":
        hdr = "hdr10"
    elif transfer == "arib-std-b67":
        hdr = "hlg"
    else:
        hdr = "sdr"

    max_cll = light.get("max_content") if light else None
    peak = _rational(mastering.get("max_luminance")) if mastering else None
    fps = _parse_frame_rate(stream.get("r_frame_rate")) or _parse_frame_rate(stream.get("avg_frame_rate"))
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "codec": stream.get("codec_name") or "unknown",
        "hdr": hdr,
        "dvProfile": int(dovi["dv_profile"]) if dovi and dovi.get("dv_profile") is not None else None,
        "dvCompatibility": int(dovi["dv_bl_signal_compatibility_id"])
        if dovi and dovi.get("dv_bl_signal_compatibility_id") is not None else None,
        "maxCll": int(max_cll) if max_cll else None,
        "masteringPeak": round(peak, 3) if peak else None,
        "transfer": transfer if transfer not in (None, "unknown") else None,
        "primaries": primaries if primaries not in (None, "unknown") else None,
    }


def video_info(path: str, token: Optional[CancellationToken] = None) -> Optional[Dict[str, Any]]:
    """``VideoInfo`` for the file's main video stream, or None without one."""
    payload = _probe_json(path, token, ["-show_streams"])
    stream = _main_video(payload.get("streams") or [])
    return _video_info_from(path, stream, token) if stream else None


def _count_pgs(path: str) -> Optional[int]:
    """Display sets that show something (a PCS with objects): the cue count,
    read from segment headers alone."""
    count = 0
    try:
        with open(path, "rb") as handle:
            while True:
                header = handle.read(13)
                if len(header) < 13:
                    break
                if header[:2] != b"PG":
                    return count or None
                size = int.from_bytes(header[11:13], "big")
                if header[10] == 0x16 and size >= 11:
                    payload = handle.read(size)
                    if payload[10] > 0:
                        count += 1
                else:
                    handle.seek(size, os.SEEK_CUR)
    except OSError:
        return None
    return count


def _count_vobsub(idx_path: str) -> Optional[int]:
    try:
        with open(idx_path, "r", encoding="latin-1") as handle:
            return sum(1 for line in handle if line.startswith("timestamp:"))
    except OSError:
        return None


def _probe_subtitle_file(path: str, fmt: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"format": fmt, "kind": "text", "cues": None,
                            "language": formats.language_from_filename(path), "encoding": None}
    if fmt == "pgs":
        info.update(kind="image", cues=_count_pgs(path))
        return info
    if fmt == "vobsub":
        idx = path if path.lower().endswith(".idx") else os.path.splitext(path)[0] + ".idx"
        info.update(kind="image", cues=_count_vobsub(idx) if os.path.isfile(idx) else None)
        return info
    with open(path, "rb") as handle:
        data = handle.read()
    encoding = formats.detect_encoding(data, info["language"])
    text = data.decode(encoding, errors="replace")
    info["encoding"] = encoding
    # A frame-based file without a header still has a countable number of
    # cues; the rate only matters once the times are used.
    doc = formats.parse(text, fmt, fps=25.0 if fmt == "microdvd" else None)
    info["cues"] = len(doc.cues)
    info["language"] = info["language"] or doc.language
    return info


def probe_file(path: str, token: Optional[CancellationToken] = None) -> Dict[str, Any]:
    """``ProbedFile`` for anything dropped on Subsync. Never raises: a bad
    file comes back with ``error`` set, so one broken file cannot fail the
    whole list."""
    out: Dict[str, Any] = {
        "path": path, "name": os.path.basename(path), "kind": "unknown", "duration": None,
        "video": None, "audioTracks": [], "subtitleTracks": [], "subtitle": None, "error": None,
    }
    try:
        if not os.path.isfile(path):
            raise MediaError(f"File not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        fmt = formats.detect_format(path) if ext in formats.SUBTITLE_EXTENSIONS else "unknown"
        if fmt != "unknown":
            out["kind"] = "subtitle"
            out["subtitle"] = _probe_subtitle_file(path, fmt)
            return out
        payload = _probe_json(path, token)
        streams = payload.get("streams") or []
        raw_duration = (payload.get("format") or {}).get("duration")
        out["duration"] = _rational(raw_duration)
        video = _main_video(streams)
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        out["audioTracks"] = [
            {
                "index": n,
                "codec": s.get("codec_name"),
                "language": _language(_tag(s.get("tags") or {}, "language")),
                "title": _tag(s.get("tags") or {}, "title"),
                "channels": s.get("channels"),
            }
            for n, s in enumerate(audio)
        ]
        out["subtitleTracks"] = [t.to_dict() for t in _tracks_from_streams(streams)]
        if video:
            out["kind"] = "video"
            out["video"] = _video_info_from(path, video, token)
        elif audio or is_audio_file(path):
            out["kind"] = "audio"
        elif out["subtitleTracks"]:
            # .mks and friends: a container with nothing but subtitles.
            out["kind"] = "video"
    except Exception as exc:  # noqa: BLE001 - reported per file, never raised
        out["error"] = str(exc) or type(exc).__name__
    return out


# --------------------------------------------------------------- extraction


def _mkvtool(name: str) -> Optional[str]:
    override = os.environ.get(f"AUDIOSYNC_{name.upper()}")
    if override and os.path.isfile(override):
        return override
    return shutil.which(name)


def _discard(*paths: str) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _resolve_track(path: str, track: Union[int, SubtitleTrack], token: Optional[CancellationToken]) -> SubtitleTrack:
    if isinstance(track, SubtitleTrack):
        return track
    tracks = list_tracks(path, token)
    for candidate in tracks:
        if candidate.index == int(track):
            return candidate
    raise MediaError(
        f"{os.path.basename(path)} has no subtitle track {int(track) + 1} "
        f"(it has {len(tracks)})."
    )


def _mkvmerge_ids(path: str, token: Optional[CancellationToken]) -> List[dict]:
    """mkvmerge's view of the tracks (its track IDs are what mkvextract wants)."""
    mkvmerge = _mkvtool("mkvmerge")
    if not mkvmerge:
        raise MediaError("MKVToolNix (mkvmerge) is not installed.")
    stdout = _run([mkvmerge, "-J", path], PROBE_TIMEOUT_S, token, what=f"identify {os.path.basename(path)}")
    return (json.loads(stdout.decode("utf-8", errors="replace") or "{}").get("tracks")) or []


def extract(
    path: str,
    track: Union[int, SubtitleTrack],
    out_dir: str,
    token: Optional[CancellationToken] = None,
) -> str:
    """Write subtitle track ``0:s:N`` of ``path`` to its own file in
    ``out_dir`` and return the path: SRT/ASS/WebVTT as they are, other text
    codecs (mov_text...) as SRT, PGS as ``.sup``, VobSub as ``.idx`` + ``.sub``."""
    info = _resolve_track(path, track, token)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    base = os.path.join(out_dir, f"{stem}.track{info.index + 1}")
    if info.language:
        base += f".{info.language}"

    if info.codec == "dvd_subtitle":
        return _extract_vobsub(path, info, base, token)
    if info.kind == "image" and info.codec != "hdmv_pgs_subtitle":
        raise MediaError(
            f"Track {info.index + 1} is {info.codec}, an image subtitle that cannot be saved as a "
            "standalone file. Remux it to PGS or VobSub with MKVToolNix first."
        )
    if info.codec == "hdmv_pgs_subtitle":
        out, codec, muxer = base + ".sup", "copy", ["-f", "sup"]
    elif info.codec in _COPY_EXTENSIONS:
        out, codec, muxer = f"{base}.{_COPY_EXTENSIONS[info.codec]}", "copy", []
    else:
        out, codec, muxer = base + ".srt", "srt", ["-f", "srt"]
    # -copyts: ffmpeg otherwise shifts each input to start at zero, which
    # moves every subtitle by the container's start time.
    command = [ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-copyts", "-i", path,
               "-map", f"0:s:{info.index}", "-c:s", codec, *muxer, out]
    try:
        _run(command, EXTRACT_TIMEOUT_S, token, what=f"extract subtitle track {info.index + 1}")
    except BaseException:
        _discard(out)
        raise
    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        _discard(out)
        raise MediaError(f"Subtitle track {info.index + 1} of {os.path.basename(path)} is empty.")
    return out


def _extract_vobsub(path: str, info: SubtitleTrack, base: str, token: Optional[CancellationToken]) -> str:
    mkvextract = _mkvtool("mkvextract")
    if not mkvextract or not _mkvtool("mkvmerge"):
        raise MediaError("Extracting a DVD (VobSub) subtitle needs MKVToolNix (mkvmerge and mkvextract).")
    subtitle_ids = [t.get("id") for t in _mkvmerge_ids(path, token) if t.get("type") == "subtitles"]
    if info.index >= len(subtitle_ids):
        raise MediaError(
            f"VobSub track {info.index + 1} could not be matched to a Matroska track; "
            "only VobSub inside MKV can be extracted."
        )
    idx = base + ".idx"
    try:
        _run([mkvextract, path, "tracks", f"{subtitle_ids[info.index]}:{idx}"], EXTRACT_TIMEOUT_S, token,
             what=f"extract VobSub track {info.index + 1}")
    except MediaError as exc:
        if not (os.path.isfile(idx) and os.path.isfile(base + ".sub")):
            _discard(idx, base + ".sub")
            raise MediaError(f"mkvextract could not extract VobSub track {info.index + 1}: {exc}") from exc
    if not os.path.isfile(idx) or not os.path.isfile(base + ".sub"):
        raise MediaError(f"mkvextract did not write {os.path.basename(idx)} and its .sub.")
    return idx


# ---------------------------------------------------------------------- mux


def _prepare_subtitle(sub: Dict[str, Any], workdir: str, token: Optional[CancellationToken]) -> Dict[str, Any]:
    """Resolve one subtitle to a file both muxers read, with its charset."""
    path = sub.get("path")
    if not path or not os.path.isfile(path):
        raise MediaError(f"Subtitle not found: {path}")
    if sub.get("track") is not None:
        path = extract(path, int(sub["track"]), workdir, token)
    fmt = formats.detect_format(path)
    prepared = dict(sub, path=path, format=fmt, kind="image" if fmt in ("pgs", "vobsub") else "text", charset=None)
    if fmt == "vobsub" and not path.lower().endswith(".idx"):
        idx = os.path.splitext(path)[0] + ".idx"
        if not os.path.isfile(idx):
            raise MediaError(f"{os.path.basename(path)} needs its .idx file beside it.")
        prepared["path"] = idx
    if prepared["kind"] == "text":
        if fmt not in _MUXABLE_TEXT:
            doc = formats.read(path, encoding=sub.get("encoding"))
            converted = os.path.join(workdir, os.path.splitext(os.path.basename(path))[0] + ".srt")
            formats.write(doc, converted, "srt")
            prepared.update(path=converted, format="srt")
            prepared["language"] = prepared.get("language") or doc.language
        else:
            with open(path, "rb") as handle:
                data = handle.read()
            encoding = sub.get("encoding") or formats.detect_encoding(data, sub.get("language"))
            if encoding not in ("utf-8", "utf-8-sig", "utf-16", "utf-32"):
                prepared["charset"] = encoding.upper()
    if not prepared.get("language"):
        prepared["language"] = formats.language_from_filename(path)
    return prepared


def _mux_language(code: Optional[str]) -> str:
    return languages.to_639_2(code) or (code if code and len(code) == 3 else "und")


def _run_mkvmerge(args: List[str], token: Optional[CancellationToken], what: str,
                  log: Callable[[str], None]) -> None:
    """mkvmerge exits 1 for warnings with a good file and 2 for errors; its
    messages go to stdout. They are captured to a file so the error line can
    be reported, and warnings are passed on instead of failing the job."""
    mkvmerge = _mkvtool("mkvmerge")
    assert mkvmerge
    handle, report = tempfile.mkstemp(prefix="mkvmerge-", suffix=".log")
    os.close(handle)
    try:
        try:
            _run([mkvmerge, "--redirect-output", report, *args], MUX_TIMEOUT_S, token, what=what)
            failed = None
        except MediaError as exc:
            failed = exc
        with open(report, "r", encoding="utf-8", errors="replace") as source:
            lines = [line.strip() for line in source if line.strip()]
        errors = [line for line in lines if line.startswith("Error:")]
        for line in lines:
            if line.startswith("Warning:"):
                log(f"mkvmerge: {line}")
        if errors or (failed and "exit code 1" not in str(failed)):
            raise MediaError(f"Failed to {what}: {errors[-1] if errors else failed}")
    finally:
        _discard(report)


def mux(
    video: str,
    subtitles: List[Dict[str, Any]],
    out_path: str,
    container: str = "mkv",
    keep_existing: bool = True,
    token: Optional[CancellationToken] = None,
    log: Optional[Callable[[str], None]] = None,
    engine: Optional[str] = None,
) -> str:
    """Write a copy of ``video`` with ``subtitles`` added as tracks.

    Each subtitle is ``{path, track?, language, title, default, forced,
    hearingImpaired, encoding?}``. MKV goes through mkvmerge when installed
    (``engine`` "mkvmerge"/"ffmpeg" forces one), else ffmpeg. MP4 holds only
    text subtitles (as mov_text); image subtitles are refused there, and an
    existing image track is left out with a warning.

    The copy is written under a temporary name and renamed at the end, so a
    stopped or failed mux never leaves a truncated video at ``out_path``.
    """
    log = log or (lambda _msg: None)
    container = (container or "mkv").lower()
    if container not in ("mkv", "mp4"):
        raise MediaError(f"Unsupported container: {container}")
    if not os.path.isfile(video):
        raise MediaError(f"Video not found: {video}")
    if os.path.abspath(out_path) == os.path.abspath(video):
        raise MediaError("Refusing to overwrite the source video in place.")
    use_mkvmerge = container == "mkv" and engine != "ffmpeg" and _mkvtool("mkvmerge") is not None
    if engine == "mkvmerge" and not use_mkvmerge:
        raise MediaError("mkvmerge is not available for this mux.")

    workdir = tempfile.mkdtemp(prefix="subsync-mux-")
    stem, ext = os.path.splitext(out_path)
    staging = f"{stem}.part{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    try:
        prepared = [_prepare_subtitle(sub, workdir, token) for sub in subtitles]
        if container == "mp4":
            refused = [os.path.basename(p["path"]) for p in prepared if p["kind"] == "image"]
            if refused:
                raise MediaError(
                    "MP4 cannot hold image subtitles (PGS/VobSub): " + ", ".join(refused)
                    + ". Use MKV, or OCR them to text first."
                )
        existing = list_tracks(video, token) if keep_existing else []
        if use_mkvmerge:
            _mux_mkvmerge(video, prepared, staging, keep_existing, existing, token, log)
        else:
            _mux_ffmpeg(video, prepared, staging, container, existing, token, log)
        if not os.path.isfile(staging) or os.path.getsize(staging) == 0:
            raise MediaError(f"The mux produced no output for {os.path.basename(out_path)}")
        os.replace(staging, out_path)
    except BaseException:
        _discard(staging)
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return out_path


def _mux_mkvmerge(video: str, prepared: List[Dict[str, Any]], staging: str, keep_existing: bool,
                  existing: List[SubtitleTrack], token: Optional[CancellationToken],
                  log: Callable[[str], None]) -> None:
    args = ["-o", staging]
    if not keep_existing:
        args.append("--no-subtitles")
    elif any(p.get("default") for p in prepared) and any(t.default for t in existing):
        # One default subtitle per file: the new one takes over.
        ids = [t.get("id") for t in _mkvmerge_ids(video, token) if t.get("type") == "subtitles"]
        for track in existing:
            if track.default and track.index < len(ids):
                args += ["--default-track-flag", f"{ids[track.index]}:0"]
    args.append(video)
    for sub in prepared:
        flag = lambda value: "1" if value else "0"  # noqa: E731
        args += ["--language", f"0:{_mux_language(sub.get('language'))}"]
        if sub.get("title"):
            args += ["--track-name", f"0:{sub['title']}"]
        args += ["--default-track-flag", f"0:{flag(sub.get('default'))}",
                 "--forced-display-flag", f"0:{flag(sub.get('forced'))}",
                 "--hearing-impaired-flag", f"0:{flag(sub.get('hearingImpaired'))}"]
        if sub.get("charset"):
            args += ["--sub-charset", f"0:{sub['charset']}"]
        args.append(sub["path"])
    _run_mkvmerge(args, token, f"mux {os.path.basename(staging)}", log)


def _mux_ffmpeg(video: str, prepared: List[Dict[str, Any]], staging: str, container: str,
                existing: List[SubtitleTrack], token: Optional[CancellationToken],
                log: Callable[[str], None]) -> None:
    # -copyts keeps every input on its own timeline. Without it ffmpeg shifts
    # each input to start at zero -- harmless for SRT, but a .sup whose first
    # caption is at 0:45 had the whole track pulled 45 s early.
    command = [ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-copyts", "-i", video]
    for sub in prepared:
        if sub.get("charset"):
            command += ["-sub_charenc", sub["charset"]]
        command += ["-i", sub["path"]]
    # Video, audio and (for MKV) attachments; no data streams, which neither
    # container reliably takes (an MP4 timecode track breaks a Matroska mux).
    command += ["-map", "0:v?", "-map", "0:a?"]
    codecs: List[str] = []
    kept: List[SubtitleTrack] = []
    for track in existing:
        if container == "mp4" and track.kind == "image":
            log(f"Left out subtitle track {track.index + 1} ({track.codec}): MP4 cannot hold image subtitles.")
            continue
        kept.append(track)
        command += ["-map", f"0:s:{track.index}"]
        if container == "mp4":
            codecs.append("copy" if track.codec == "mov_text" else "mov_text")
        else:
            codecs.append("copy" if track.codec in _MKV_SUBTITLE_CODECS else "srt")
    if container == "mkv":
        command += ["-map", "0:t?"]
    for n, sub in enumerate(prepared, start=1):
        command += ["-map", f"{n}:0"]
        codecs.append("mov_text" if container == "mp4" else "copy")
    command += ["-c", "copy"]
    for k, codec in enumerate(codecs):
        command += [f"-c:s:{k}", codec]
    new_default = any(p.get("default") for p in prepared)
    for k, track in enumerate(kept):
        if new_default and track.default:
            command += [f"-disposition:s:{k}", "-default"]
    for n, sub in enumerate(prepared):
        k = len(kept) + n
        command += [f"-metadata:s:s:{k}", f"language={_mux_language(sub.get('language'))}"]
        if sub.get("title"):
            command += [f"-metadata:s:s:{k}", f"title={sub['title']}"]
        flags = [name for name, key in (("default", "default"), ("forced", "forced"),
                                        ("hearing_impaired", "hearingImpaired")) if sub.get(key)]
        command += [f"-disposition:s:{k}", "+".join(flags) if flags else "0"]
    if container == "mp4":
        command += ["-movflags", "+faststart"]
    command += ["-f", "matroska" if container == "mkv" else "mp4", staging]
    _run(command, MUX_TIMEOUT_S, token, what=f"mux {os.path.basename(staging)}")


# ------------------------------------------------------------------- tasks


def run_extract_task(job: dict, ctx) -> TaskResult:
    """Task "extract": subtitle tracks of a video to files."""
    inputs = job.get("input") or {}
    ref = inputs.get("video") or inputs.get("subtitle")
    if not ref or not ref.get("path"):
        raise TaskError("Choose a video to extract subtitles from.")
    source = ref["path"]
    tracks = list_tracks(source, ctx.token)
    if not tracks:
        raise TaskError(f"{os.path.basename(source)} has no subtitle tracks.")
    wanted = (job.get("options") or {}).get("tracks", "all")
    if wanted in (None, "all"):
        chosen = tracks
    else:
        indexes = {int(i) for i in wanted}
        chosen = [t for t in tracks if t.index in indexes]
        missing = sorted(indexes - {t.index for t in chosen})
        if missing:
            raise TaskError(f"{os.path.basename(source)} has no subtitle track {missing[0] + 1}.")
    target = ((job.get("output") or {}).get("format") or "same")

    outputs = []
    warnings: List[str] = []
    for n, track in enumerate(chosen):
        ctx.check()
        ctx.progress(int(100 * n / len(chosen)), f"Track {track.index + 1}")
        try:
            extracted = extract(source, track, ctx.workdir, ctx.token)
        except MediaError as exc:
            warnings.append(str(exc))
            continue
        ext = os.path.splitext(extracted)[1].lstrip(".")
        fmt = {"srt": "srt", "ass": "ass", "vtt": "vtt", "sup": "pgs", "idx": "vobsub"}.get(ext, ext)
        track_ref = {"path": source, "track": track.index}
        if track.kind == "text" and target not in ("same", None):
            fmt = formats.resolve_output_format(target, fmt)
            if fmt == "pgs":
                warnings.append(f"Track {track.index + 1} is text; kept as {ext.upper()} instead of SUP.")
                fmt = {"srt": "srt", "ass": "ass", "vtt": "vtt"}.get(ext, "srt")
            doc = formats.read(extracted)
            destination = output_path(track_ref, job, "", formats.extension_for(fmt), track.language)
            formats.write(doc, destination, fmt)
        else:
            if track.kind == "image" and target not in ("same", None, "sup"):
                warnings.append(f"Track {track.index + 1} is an image subtitle; run OCR to get text.")
            destination = output_path(track_ref, job, "", ext, track.language)
            os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
            shutil.move(extracted, destination)
            if fmt == "vobsub":
                shutil.move(os.path.splitext(extracted)[0] + ".sub", os.path.splitext(destination)[0] + ".sub")
        ctx.log(f"Wrote {os.path.basename(destination)}")
        outputs.append(TaskOutput(path=destination, kind="subtitle", format=fmt, language=track.language,
                                  label=track.title or f"Track {track.index + 1}"))
    if not outputs:
        raise TaskError("No track could be extracted: " + "; ".join(warnings))
    ctx.progress(100, "Done")
    return TaskResult(
        outputs=outputs,
        report={"tracks": [t.to_dict() for t in chosen], "extracted": len(outputs)},
        summary=f"Extracted {len(outputs)} of {len(chosen)} subtitle track{'s' if len(chosen) != 1 else ''} "
                f"from {os.path.basename(source)}.",
        warnings=warnings,
    )


def run_mux_task(job: dict, ctx) -> TaskResult:
    """Task "mux": add subtitles to a copy of a video."""
    inputs = job.get("input") or {}
    video = inputs.get("video") or {}
    subtitles = inputs.get("subtitles") or []
    if not video.get("path") or not os.path.isfile(video["path"]):
        raise TaskError(f"Video not found: {video.get('path')}")
    if not subtitles:
        raise TaskError("Choose at least one subtitle to add.")
    options = job.get("options") or {}
    container = options.get("container") or "mkv"
    destination = output_path({"path": video["path"]}, job, ".muxed", container)
    warnings: List[str] = []

    def log(message: str) -> None:
        ctx.log(message)
        warnings.append(message)

    ctx.progress(5, "Muxing")
    mux(video["path"], subtitles, destination, container=container,
        keep_existing=bool(options.get("keepExisting", True)), token=ctx.token, log=log)
    ctx.progress(100, "Done")
    return TaskResult(
        outputs=[TaskOutput(path=destination, kind="video", format=container)],
        report={"added": len(subtitles), "container": container},
        summary=f"Added {len(subtitles)} subtitle track{'s' if len(subtitles) != 1 else ''} to "
                f"{os.path.basename(destination)}.",
        warnings=warnings,
    )
