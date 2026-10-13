"""Every command line a job runs (FPSConv ``engine.py:1616-1757``,
``:1880-1918``).

Pure: tool paths come in as arguments, so the same inputs always give the
same argv. The strings are FPSConv's, flag for flag, and the tests pin them.
"""

from __future__ import annotations

import os
from typing import Optional

from .formats import DRC_PROFILES, Encode
from .fps import atempo_chain


def _drc(drc: str) -> str:
    return drc if drc in DRC_PROFILES else "film_light"


def with_progress(cmd: list[str]) -> list[str]:
    """ffmpeg reports progress as ``key=value`` blocks on stdout, before the output path."""
    if "-progress" in cmd:
        return cmd
    return cmd[:-1] + ["-progress", "pipe:1", cmd[-1]]


# ───────────────────────── FPS change ─────────────────────────


def fps_aac(ffmpeg: str, src: str, stream_index: int, bitrate: int, channels: int,
            ratio: float, out_path: str) -> list[str]:
    """AAC source: one ffmpeg pass, re-timed and re-encoded to AAC."""
    return [
        ffmpeg, "-y", "-nostdin", "-i", src,
        "-map", f"0:a:{stream_index}", "-vn", "-sn", "-dn",
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-c:a", "aac", "-b:a", f"{bitrate}k",
        "-ac", str(channels),
        "-af", atempo_chain(ratio),
        out_path,
    ]


def fps_wav(ffmpeg: str, src: str, stream_index: int, channels: int, ratio: float,
            temp_wav: str) -> list[str]:
    """AC-3 / E-AC-3 / TrueHD source: re-timed to a 24-bit 48 kHz WAV for DEE."""
    return [
        ffmpeg, "-y", "-nostdin", "-i", src,
        "-map", f"0:a:{stream_index}", "-vn", "-sn", "-dn",
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-c:a", "pcm_s24le", "-ar", "48000", "-ac", str(channels),
        "-af", atempo_chain(ratio), "-rf64", "auto",
        temp_wav,
    ]


def deew_fps(deew: list[str], temp_wav: str, fmt: str, bitrate: int, out_dir: str) -> list[str]:
    """``deew`` back to the source's codec at its bitrate (``fmt`` is dd, ddp or thd)."""
    return deew + ["-i", temp_wav, "-f", fmt, "-b", str(bitrate), "-o", out_dir, "-np"]


# ───────────────────────── encode ─────────────────────────


def encode_wav(ffmpeg: str, src: str, stream_index: int, stream: dict, enc: Encode,
               temp_wav: str, soxr: bool = True) -> list[str]:
    """Lossless decode to 24-bit 48 kHz WAV, no speed change.

    * ``-drc_scale 0`` for AC-3 / E-AC-3 sources — ffmpeg applies the stream's
      DRC metadata by default, which is wrong for a transcode.
    * a source that is not 48 kHz (96 kHz TrueHD) is resampled with soxr.
    """
    cmd = [ffmpeg, "-y", "-nostdin"]
    if stream.get("codec") in ("ac3", "eac3"):
        cmd += ["-drc_scale", "0"]
    cmd += ["-i", src, "-map", f"0:a:{stream_index}", "-vn", "-sn", "-dn",
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-c:a", "pcm_s24le", "-ac", str(enc.wav_channels)]
    if int(stream.get("sample_rate") or 48000) != 48000 and soxr:
        cmd += ["-af", "aresample=out_sample_rate=48000:resampler=soxr:precision=28"]
    cmd += ["-ar", "48000", "-rf64", "auto", temp_wav]
    return cmd


def deew_encode(deew: list[str], temp_wav: str, enc: Encode, drc: str, out_dir: str) -> list[str]:
    cmd = deew + ["-i", temp_wav, "-f", enc.fmt, "-b", str(enc.bitrate),
                  "-r", _drc(drc), "-o", out_dir, "-np"]
    if enc.out_channels < enc.wav_channels:
        cmd += ["-dm", str(enc.out_channels)]
    return cmd


def deezy_atmos(deezy: list[str], src: str, stream_index: int, enc: Encode, drc: str, work_dir: str,
                out_path: str, deezy_work_dir: str, tools: Optional[dict] = None) -> list[str]:
    """``deezy encode atmos …``; tool paths are passed explicitly so nobody has to
    write DeeZy's own config (``tools`` = {"ffmpeg", "dee", "truehdd"}, empty = its default).

    Only ``--no-progress-bars`` is a top-level DeeZy option; ``--ffmpeg`` /
    ``--dee`` / ``--truehdd`` belong to the ``encode atmos`` sub-parser.

    ``--working-dir`` (DeeZy's logs / batch-results) is the Working folder:
    left alone, DeeZy creates ``deezy_work`` next to its own exe, which is
    Program Files for a normal install and not writable.
    """
    cmd = deezy + ["--no-progress-bars", "--log-level", "debug", "encode", "atmos"]
    for flag in ("ffmpeg", "dee", "truehdd"):
        value = (tools or {}).get(flag) or ""
        if value:
            cmd += [f"--{flag}", value]
    return cmd + [
        "--atmos-mode", enc.atmos_mode, "--bitrate", str(enc.bitrate),
        "--track-index", f"a:{stream_index}",
        "--drc-line-mode", _drc(drc),
        "--working-dir", deezy_work_dir, "--temp-dir", os.path.abspath(work_dir),
        "--output", out_path, "--overwrite", src,
    ]


# ───────────────────────── flat 7.1 Atmos ─────────────────────────


def thd_extract(ffmpeg: str, src: str, stream_index: int, out_thd: str) -> list[str]:
    """Copy the TrueHD elementary stream out of a container, untouched."""
    return [ffmpeg, "-y", "-nostdin", "-i", src, "-map", f"0:a:{stream_index}", "-vn", "-sn", "-dn",
            "-c:a", "copy", "-f", "truehd", out_thd]


def truehdd(truehdd_exe: str, thd: str, out_base: str, bed_conform: bool = True) -> list[str]:
    """Decode the TrueHD Atmos presentation to a DAMF master (``out_base.atmos`` + siblings)."""
    cmd = [truehdd_exe, "--progress", "decode", "--output-path", out_base, "--warp-mode", "normal"]
    if bed_conform:
        cmd.append("--bed-conform")
    return cmd + [thd]


def patcher(runner: list[str], script: str, dee_exe: str, master_atmos: str, out_path: str,
            enc: Encode, drc: str, temp_dir: str) -> list[str]:
    """``dee-ddp71-atmos-wrapper.py <dee> <master> <out> --compatibility-layout flat-7.1 …``

    ``runner`` is the Python that runs the wrapper: the Dolby tools pack's.
    """
    profile = _drc(drc)
    return runner + [
        str(script),
        dee_exe, master_atmos, out_path,
        "--compatibility-layout", "flat-7.1",
        "--data-rate", str(enc.bitrate),
        "--line-mode-drc-profile", profile, "--rf-mode-drc-profile", profile,
        "--temp-dir", temp_dir, "--overwrite",
    ]


def deew_output(work_dir: str, base: str) -> Optional[str]:
    """deew names DDP 7.1 Blu-ray-profile output ``.eb3``; accept every extension it can write."""
    for ext in (".ec3", ".eb3", ".ac3", ".thd"):
        cand = os.path.join(work_dir, f"{base}{ext}")
        if os.path.exists(cand):
            return cand
    return None
