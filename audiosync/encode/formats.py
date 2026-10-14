"""What an encode job produces: targets, layouts, DEE bitrates and the notes
that explain every change from what was asked (FPSConv ``engine.py:84-242``).

Pure: no tools are run and no settings are read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TASK_FPS = "fps"
TASK_ENCODE = "encode"
ENCODE_CONV = "encode"          # conv_type of an encode job: speed ratio 1.0, no atempo

TARGETS = ("ddp", "dd")
TARGET_CHANNELS = (0, 1, 2, 6, 8)   # 0 = same as the source
DRC_PROFILES = ("film_light", "film_standard", "music_light", "music_standard", "speech")
LAYOUT_NAMES = {1: "1.0", 2: "2.0", 6: "5.1", 8: "7.1"}
FORMAT_NAMES = {"ddp": "DDP", "dd": "DD"}

# One list for the file dialog, drops and folder expansion. FPSConv's dialog
# filter was a second, shorter list that missed .eb3 .mlp .dts .dtshd .w64.
AUDIO_EXTS = {
    ".mka", ".mkv", ".mp4", ".m4a", ".mov", ".ts", ".m2ts", ".webm",
    ".ac3", ".ec3", ".eac3", ".eb3", ".thd", ".truehd", ".mlp", ".dts", ".dtshd",
    ".aac", ".wav", ".w64", ".flac", ".ogg", ".opus",
}
#: What the "Target video" picker shows: anything ffprobe can read a frame rate from.
VIDEO_EXTS = AUDIO_EXTS | {".avi", ".m4v", ".wmv", ".mpg", ".mpeg", ".flv", ".vob"}

# What DEE accepts, copied from deew/bitrates.py. 7.1 is deew's "combined" list:
# up to 1024 kbps DEE runs in the standard 7.1 mode, above it in Blu-ray mode.
BITRATES = {
    ("dd", 1): [96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640],
    ("dd", 2): [96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640],
    ("dd", 6): [224, 256, 320, 384, 448, 512, 576, 640],
    ("ddp", 1): [32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 200,
                 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400,
                 448, 512, 576, 640, 704, 768, 832, 896, 960, 1008, 1024],
    ("ddp", 2): [96, 104, 112, 120, 128, 144, 160, 176, 192, 200, 208, 216, 224, 232, 240, 248,
                 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 448, 512, 576, 640, 704, 768,
                 832, 896, 960, 1008, 1024],
    ("ddp", 6): [192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368,
                 384, 400, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1008, 1024],
    ("ddp", 8): [384, 448, 576, 640, 704, 768, 832, 896, 960, 1008, 1024, 1280, 1536, 1664],
}
# DD+ JOC rates DeeZy passes to DEE (deezy/enums/atmos.py).
ATMOS_BITRATES = {
    "streaming": [384, 448, 512, 576, 640, 768, 1024],      # DDP 5.1 + JOC
    "bluray": [1152, 1280, 1408, 1512, 1536, 1664],         # DDP 7.1 + JOC
}
#: The 7.1 patcher wrapper's --data-rate list.
FLAT71_BITRATES = [1152, 1280, 1408, 1512, 1536, 1664]
ATMOS71_LAYOUTS = ("flat", "dee")       # flat = Lb Rb via the patcher · dee = DEE default 5.1+2 heights
DEFAULT_BITRATE = {
    ("dd", 1): 192, ("dd", 2): 256, ("dd", 6): 640,
    ("ddp", 1): 128, ("ddp", 2): 256, ("ddp", 6): 1024, ("ddp", 8): 1536,
    ("atmos", 6): 768, ("atmos", 8): 1536,
}


def allowed_bitrates(fmt: str, channels: int, atmos: bool = False) -> list[int]:
    if atmos:
        return ATMOS_BITRATES["bluray" if channels == 8 else "streaming"]
    return BITRATES[(fmt, channels)]


def snap_bitrate(fmt: str, channels: int, kbps: int, atmos: bool = False) -> int:
    """The closest bitrate DEE can actually do for this layout.

    0 / None picks the layout's default. So does a request outside the layout's
    whole range (1536 kbps on a 2.0 file): that number belongs to another
    layout, and the layout's default is more sensible than its maximum.
    """
    allowed = allowed_bitrates(fmt, channels, atmos)
    default = DEFAULT_BITRATE[("atmos", 8 if channels == 8 else 6) if atmos else (fmt, channels)]
    kbps = int(kbps or 0)
    if not kbps or kbps < allowed[0] or kbps > allowed[-1]:
        return default
    return min(allowed, key=lambda b: abs(b - kbps))


def pad_layout(channels: int) -> int:
    """deew/DEE only take 1, 2, 6 or 8 channel WAVs; odd layouts get padded up."""
    channels = int(channels or 2)
    if channels <= 1:
        return 1
    if channels == 2:
        return 2
    if channels <= 6:
        return 6
    return 8


@dataclass
class Encode:
    fmt: str                  # ddp | dd
    src_channels: int
    wav_channels: int         # what ffmpeg writes (padded layout)
    out_channels: int         # what DEE writes
    atmos: bool
    atmos_mode: str           # streaming | bluray | ""
    bitrate: int
    notes: list[str] = field(default_factory=list)
    flat71: bool = False      # bluray Atmos through the patched DEE: legacy 7.1 = Lb Rb, not Tfl Tfr

    @property
    def ext(self) -> str:
        return ".ac3" if self.fmt == "dd" else ".ec3"

    @property
    def label(self) -> str:
        return (f"{FORMAT_NAMES[self.fmt]} {LAYOUT_NAMES[self.out_channels]}"
                f"{' Atmos' if self.atmos else ''}{' flat' if self.flat71 else ''} {self.bitrate}k")

    def to_dict(self) -> dict:
        return {"fmt": self.fmt, "src_channels": self.src_channels, "wav_channels": self.wav_channels,
                "out_channels": self.out_channels, "atmos": self.atmos, "atmos_mode": self.atmos_mode,
                "flat71": self.flat71, "bitrate": self.bitrate, "label": self.label, "notes": self.notes}


def resolve_encode(target: str, target_channels: int, keep_atmos: bool, bitrate: int,
                   src_channels: int, src_atmos: Optional[bool], src_codec: str = "",
                   atmos71: str = "flat") -> Encode:
    """Decide what one encode job will really produce, and why.

    Never upmixes: a 7.1 request on a 2.0 source yields 2.0. Atmos survives only
    for a DDP 5.1 / 7.1 target from a source that MediaInfo confirmed as Atmos.
    ``atmos71`` picks the 7.1 Atmos legacy layout: ``flat`` (Lb Rb, patched DEE)
    or ``dee`` (DEE's own 5.1 + 2 heights, Tfl Tfr).
    """
    fmt = target if target in TARGETS else "ddp"
    notes: list[str] = []
    wav = pad_layout(src_channels)
    if wav != src_channels:
        notes.append(f"{src_channels}-channel source padded to {LAYOUT_NAMES[wav]} with silent channels")
    want = int(target_channels or 0) or wav
    out = min(want, wav)
    if want > wav:
        notes.append(f"no upmix: {LAYOUT_NAMES.get(want, want)} requested, source is {LAYOUT_NAMES[wav]}")
    if fmt == "dd" and out > 6:
        out = 6
        notes.append("DD has no 7.1: downmixed to 5.1")
    atmos = False
    if keep_atmos:
        if src_atmos is True and fmt == "ddp" and out >= 6:
            atmos = True
        elif src_atmos is True and fmt == "dd":
            notes.append("Atmos dropped: DD cannot carry Atmos")
        elif src_atmos is True:
            notes.append("Atmos dropped: needs a 5.1 or 7.1 target")
        elif src_atmos is None and src_codec == "truehd":
            notes.append("Atmos unknown (mediainfo not available): encoding the bed only")
    mode = ("bluray" if out == 8 else "streaming") if atmos else ""
    flat71 = bool(atmos and mode == "bluray" and atmos71 != "dee")
    if flat71:
        kbps = int(bitrate or 0)
        kbps = (min(FLAT71_BITRATES, key=lambda b: abs(b - kbps))
                if FLAT71_BITRATES[0] <= kbps <= FLAT71_BITRATES[-1] else DEFAULT_BITRATE[("atmos", 8)])
    else:
        kbps = snap_bitrate(fmt, out, bitrate, atmos)
    if bitrate and kbps != int(bitrate):
        lo, hi = allowed_bitrates(fmt, out, atmos)[0], allowed_bitrates(fmt, out, atmos)[-1]
        what = f"{FORMAT_NAMES[fmt]} {LAYOUT_NAMES[out]}{' Atmos' if atmos else ''}"
        if int(bitrate) < lo or int(bitrate) > hi:
            notes.append(f"{bitrate} kbps is outside {what} ({lo}–{hi}): using the default {kbps}")
        else:
            notes.append(f"{bitrate} kbps is not a DEE {what} rate: using {kbps}")
    if fmt == "ddp" and out == 8 and not atmos and kbps > 1024:
        notes.append("DDP 7.1 above 1024 kbps: Blu-ray profile")
    if atmos and mode == "bluray":
        notes.append("flat 7.1: Lb Rb via the patched DEE (truehdd → DEE 5.2.1 patch → Surround EX flag)" if flat71
                     else "DEE default: legacy 7.1 is 5.1 + 2 heights — MediaInfo shows Tfl Tfr, not Lb Rb")
    return Encode(fmt, int(src_channels or 2), wav, out, atmos, mode, kbps, notes, flat71)


def encode_output_name(file_path: str, stream_index: int, enc: Encode) -> str:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    track = f"_a{stream_index}" if stream_index else ""
    tag = f"{FORMAT_NAMES[enc.fmt]}{LAYOUT_NAMES[enc.out_channels]}{'Atmos' if enc.atmos else ''}"
    return f"{stem}{track}_{tag}_{enc.bitrate}k{enc.ext}"


def encode_plan(enc: Encode, stream: dict) -> list[str]:
    """What the job is about to do, step by step, for the job log."""
    src = stream.get("pretty") or stream.get("codec", "").upper()
    if enc.atmos and enc.flat71:
        return [
            f"1/5 ffmpeg copies the {src} stream unchanged into a .thd in the temp folder (no decode)",
            "2/5 truehdd decodes the bed and the Atmos objects into a Dolby Atmos master (DAMF) — the slow step; "
            "for a raw .thd no percentage is available, so the row shows how much has been written",
            "3/5 the DD+ 7.1 patcher backs up DEE's dee_audio_filter_ddp_atmos.dll, applies the validated flat-7.1 "
            "patch (two bytes, 19 → 21), and runs DEE: loudness measurement, then DD+ JOC Blu-ray encode at "
            f"{enc.bitrate} kbps with a real 7.1 render and a Pro Logic IIx 5.1 core; the original DLL is restored",
            "4/5 the Surround EX flag patcher sets dsurexmod=2 in the AC-3 core (metadata + CRC only)",
            "5/5 the .ec3 (MediaInfo: L R C LFE Ls Rs Lb Rb) is moved to the output folder, temp deleted",
        ]
    if enc.atmos:
        mode = "7.1 (Blu-ray mode)" if enc.atmos_mode == "bluray" else "5.1 (streaming mode)"
        return [
            f"1/4 ffmpeg copies the {src} stream unchanged into truehdd (no decode by ffmpeg)",
            "2/4 truehdd decodes the bed and the Atmos objects into a Dolby Atmos master (DAMF: "
            ".atmos / .atmos.audio / .atmos.metadata) in the job's temp folder — the slow step; "
            "for a raw .thd no percentage is available, so the row shows how much has been written",
            f"3/4 DEE measures loudness (dialnorm) on that master, then encodes DD+ JOC {mode} at "
            f"{enc.bitrate} kbps, DRC profile as chosen",
            "4/4 the .ec3 is moved to the output folder and the temp folder is deleted",
        ]
    codec_name = stream.get("codec_name") or ""
    lossless = stream.get("codec") in ("truehd",) or codec_name in ("dts", "flac", "mlp") or codec_name.startswith("pcm")
    decode = "lossless decode" if lossless else "decode (lossy source: quality can only stay the same)"
    resample = ("" if int(stream.get("sample_rate") or 48000) == 48000
                else f", resampled from {stream.get('sample_rate')} Hz to 48 kHz")
    dm = ""
    if enc.out_channels < enc.wav_channels:
        dm = f" with DEE's own {LAYOUT_NAMES[enc.wav_channels]} → {LAYOUT_NAMES[enc.out_channels]} downmix"
    profile = " (Blu-ray profile)" if enc.fmt == "ddp" and enc.out_channels == 8 and enc.bitrate > 1024 else ""
    return [
        f"1/3 ffmpeg: {decode} of the {src} stream to 24-bit 48 kHz {LAYOUT_NAMES[enc.wav_channels]} WAV{resample}",
        f"2/3 deew writes DEE's XML job; DEE measures loudness (dialnorm), then encodes "
        f"{FORMAT_NAMES[enc.fmt]} {LAYOUT_NAMES[enc.out_channels]} at {enc.bitrate} kbps{profile}{dm}",
        f"3/3 the {enc.ext} is moved to the output folder and the temp folder is deleted",
    ]


def list_audio_files(folder: str, recursive: bool = False) -> list[str]:
    """The audio files directly in ``folder`` (sub-folders are not searched, as in FPSConv)."""
    root = Path(folder)
    it = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        str(p) for p in it
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith(".")
    )
