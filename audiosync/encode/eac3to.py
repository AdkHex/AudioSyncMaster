"""eac3to: decoding, frame-rate changes and TrueHD extraction (Windows only).

When eac3to is set in Preferences it does everything it can, and ffmpeg
only what it cannot:

* FPS change -- eac3to decodes and re-times to a W64
  (``-23.976 -changeTo25.000``); DEE encodes Dolby sources from it as
  before, ffmpeg encodes AAC from it. eac3to's frame-rate change resamples,
  so the pitch moves with the speed (about 4 % for 23.976 → 25), as a PAL
  speed-up does. ffmpeg's ``atempo``, used without eac3to, keeps the pitch.
* Encode -- eac3to decodes to a W64 and resamples to
  48 kHz with ``-resampleTo48000``; DEE encodes. Layouts DEE cannot take
  (3, 5 or 7 channels) are padded by ffmpeg instead.
* Flat 7.1 -- eac3to extracts the TrueHD stream (without the AC-3 core).

eac3to reads Matroska, M2TS/TS, VOB/EVO and raw elementary streams, not
MP4, MOV, WebM or Ogg; those stay on ffmpeg. W64 rather than WAV because a
two-hour 7.1 track is past WAV's 4 GB; deew reads it through its own
ffmpeg step.

Pure apart from :func:`list_tracks`, which runs eac3to to read a container.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

#: Containers eac3to lists numbered tracks for; their tracks are picked with "N:".
CONTAINERS = {".mkv", ".mka", ".m2ts", ".mts", ".ts", ".vob", ".evo"}
#: Elementary streams and plain audio files eac3to reads directly.
RAW = {".ac3", ".ec3", ".eac3", ".eb3", ".thd", ".truehd", ".mlp", ".dts", ".dtshd", ".wav", ".w64", ".flac"}

#: eac3to's names for the frame rates the FPS change offers.
FPS_FLAG = {"23.976": "23.976", "24": "24.000", "25": "25.000"}

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_TRACK = re.compile(r"^\s*(\d+):\s*(.+?)\s*$")
_NOT_AUDIO = re.compile(r"^(chapters|subtitle|h26[45]|avc|hevc|vc-?1|mpeg-?[124]\b|mpeg[124]?\s*video|mjpeg|attachment|vp[89]|av1)", re.I)
# "process: 42%", or with -progressnumbers a bare "process: 42" per line.
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%|^\s*[a-z][\w ]*:\s*(\d+(?:\.\d+)?)\s*$", re.I)


def can_read(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in CONTAINERS | RAW


def is_container(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in CONTAINERS


def fps_flags(conv_type: str) -> list[str]:
    """``23.976-25`` → ``["-23.976", "-changeTo25.000"]``."""
    src, dst = conv_type.split("-")
    return [f"-{FPS_FLAG[src]}", f"-changeTo{FPS_FLAG[dst]}"]


def audio_tracks(listing: str) -> list[tuple[int, str, int]]:
    """eac3to's audio tracks as ``(track number, description, ffprobe streams it covers)``.

    A Blu-ray "TrueHD/AC3" track is one track to eac3to and two audio streams
    (TrueHD, then its AC-3 core) to ffprobe, so it covers two stream indices.
    """
    out = []
    for line in listing.splitlines():
        m = _TRACK.match(line)
        if not m:
            continue
        number, desc = int(m.group(1)), m.group(2)
        if _NOT_AUDIO.match(desc):
            continue
        out.append((number, desc, 2 if desc.lower().startswith("truehd/ac3") else 1))
    return out


def track_for(listing: str, stream_index: int) -> Optional[tuple[int, list[str]]]:
    """The eac3to track for ffprobe's ``0:a:<stream_index>``, and the options picking it.

    The AC-3 core of a TrueHD/AC3 track is the same track with ``-core``.
    None when the listing has fewer audio tracks.
    """
    seen = 0
    for number, _desc, covers in audio_tracks(listing):
        if stream_index < seen + covers:
            return number, (["-core"] if stream_index - seen == 1 else [])
        seen += covers
    return None


def list_tracks(eac3to: str, src: str, timeout: float = 120) -> str:
    """eac3to's own listing of a container's tracks."""
    proc = subprocess.run([eac3to, src], capture_output=True, text=True, errors="replace", timeout=timeout,
                          creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL)
    return proc.stdout + proc.stderr


def _io(eac3to: str, src: str, track: Optional[int], out: str) -> list[str]:
    return [eac3to, src, *([f"{track}:"] if track is not None else []), out]


def fps_wav(eac3to: str, src: str, track: Optional[int], extra: list[str], conv_type: str, out_w64: str) -> list[str]:
    """Decode and re-time to a W64: ``eac3to src [N:] out.w64 -23.976 -changeTo25.000``."""
    return _io(eac3to, src, track, out_w64) + extra + fps_flags(conv_type) + ["-progressnumbers"]


def encode_wav(eac3to: str, src: str, track: Optional[int], extra: list[str], sample_rate: int, out_w64: str) -> list[str]:
    """Decode to a W64 at 48 kHz for DEE, resampling when the source is not 48 kHz."""
    resample = ["-resampleTo48000"] if int(sample_rate or 48000) != 48000 else []
    return _io(eac3to, src, track, out_w64) + extra + resample + ["-progressnumbers"]


def thd_extract(eac3to: str, src: str, track: Optional[int], out_thd: str) -> list[str]:
    """Extract the TrueHD stream for truehdd, untouched and without the AC-3 core."""
    return _io(eac3to, src, track, out_thd) + ["-progressnumbers"]


def parse_line(stage: str):
    """A progress parser for one eac3to step: every ``42%`` it prints is that step's percent."""
    def parse(text: str) -> Optional[tuple[str, Optional[float]]]:
        m = _PCT.search(text)
        if not m:
            return None
        pct = float(m.group(1) or m.group(2))
        return (stage, pct) if pct <= 100 else None
    return parse
