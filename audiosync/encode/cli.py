"""``audiosync-cli convert | encode | probe | doctor | patcher | dee``
(a port of FPSConv's ``__main__.py``).

The flags are FPSConv's, so a script written for ``fpsconv-cli.exe`` needs
only the exe name changed. The engine exe is a console program already; with
no arguments it is still the app's engine (``python/bridge.py``), and it only
comes here when its first argument is one of :data:`COMMANDS`.

Exit codes: 0 every job done or skipped, 1 a job ended otherwise (or a tool
check failed), 2 not Windows, 130 Ctrl-C (every job cancelled first).
No history is written: the app's history shows the app's jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from . import WINDOWS_ONLY, is_windows
from .formats import ATMOS71_LAYOUTS, DRC_PROFILES, TARGET_CHANNELS, TARGETS, TASK_ENCODE, list_audio_files
from .fps import FPS_CONVERSIONS, suggest_conversion
from .progress import fmt_time

APP_NAME = "AudioSyncMaster"
COMMANDS = ("convert", "encode", "probe", "doctor", "patcher", "dee", "--version", "-h", "--help")


def app_version() -> str:
    """The release version CI stamps into the engine, else the checkout's package.json."""
    try:
        from .._version import __version__  # type: ignore[import-not-found]
        return __version__
    except ImportError:
        pass
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        with open(os.path.join(root, "package.json"), encoding="utf-8") as handle:
            return str(json.load(handle).get("version") or "unknown")
    except (OSError, ValueError):
        return "unknown"


def is_cli(argv: list[str]) -> bool:
    """Whether the engine was started as the command line rather than the app's engine."""
    return bool(argv) and argv[0] in COMMANDS


def _cli_convert(args: argparse.Namespace) -> int:
    from .service import AudioQueue

    sources: list[str] = []
    for p in args.inputs:
        if os.path.isdir(p):
            sources.extend(list_audio_files(p, recursive=args.recursive))
        else:
            sources.append(p)
    if not sources:
        print("No input files.", file=sys.stderr)
        return 1

    queue = AudioQueue(workers=args.jobs, history=False)
    if args.cmd == "encode":
        encode = {"target": args.format, "channels": args.channels, "bitrate": args.bitrate,
                  "atmos": not args.no_atmos, "drc": args.drc, "atmos71": args.atmos_71,
                  "bed_conform": not args.no_bed_conform}
        specs = [{"task": TASK_ENCODE, "path": s, "stream_index": args.stream, "out_dir": args.output,
                  "overwrite": args.overwrite, "encode": encode} for s in sources]
    else:
        specs = [{"task": "fps", "path": s, "stream_index": args.stream, "out_dir": args.output,
                  "overwrite": args.overwrite, "conv_type": args.mode, "bitrate": args.bitrate} for s in sources]
    result = queue.add(specs)
    # The same file twice is dropped quietly, as FPSConv did; anything else is an error.
    rejected = [s for s in result["skipped"] if s["reason"] != "already queued"]
    for skipped in rejected:
        print(f"skipped: {skipped['reason']}", file=sys.stderr)
    last = ""
    try:
        while True:
            snap = queue.snapshot()
            line = " | ".join(
                f"{j['name'][:28]} {j['step']} {j['percent']:.0f}%" for j in snap["jobs"]
                if j["state"] == "running"
            )
            if line and line != last:
                print("\r" + line[:120].ljust(120), end="", flush=True)
                last = line
            if all(j["state"] not in ("queued", "running") for j in snap["jobs"]):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        queue.cancel_all()
        queue.wait_idle(30)
        print("\ncancelled")
        return 130
    print()
    rc = 1 if rejected else 0
    for j in queue.snapshot()["jobs"]:
        mark = {"done": "done   ", "failed": "FAILED ", "cancelled": "cancel ", "skipped": "skipped"}[j["state"]]
        what = f" [{j['pretty']} -> {j['label']}]" if j.get("task") == TASK_ENCODE and j.get("label") else ""
        print(f"{mark} {j['name']}{what} -> {j['out_name'] or '-'} {j['out_size']} {j['error']}")
        if j["state"] not in ("done", "skipped"):
            rc = 1
    return rc


def _cli_probe(args: argparse.Namespace) -> int:
    """Show every audio stream and the frame rate each file is tied to.

    With ``--target VIDEO`` it also says which conversion makes the audio fit
    that video (from the frame rates, or from the durations if the audio-only
    file carries no fps).
    """
    from . import atmos, probe
    from .tools import Tools

    t = Tools.resolve()
    ref = None
    if args.target:
        ref = probe.probe_video(t.ffprobe, args.target)
        print(f"target : {ref['name']}  {ref['fps'] or '?'} fps ({ref['fps_source'] or 'unknown'})  {ref['duration']}")
        print()
    rc = 0
    for path in args.inputs:
        info = probe.probe_streams(t.ffprobe, path, atmos.reader(t.mediainfo, t.pack_python))
        if not info["streams"]:
            print(f"{os.path.basename(path)}: no audio stream found")
            rc = 1
            continue
        fps = f"{info['fps']} fps ({info['fps_source']})" if info["fps"] else "fps unknown (audio only – use --target)"
        print(f"{os.path.basename(path)}  {fmt_time(info['duration'])}  {fps}")
        for st in info["streams"]:
            unknown = st["atmos"] is None and st["codec"] == "truehd"
            note = "  (Atmos: unknown, mediainfo missing)" if unknown else ""
            print(f"    #{st['index']} {st['pretty']} {st['bitrate']} kbps {st['channels']} ch {st['sample_rate']} Hz"
                  f"{' ' + st['language'] if st['language'] else ''}{' – ' + st['title'] if st['title'] else ''}{note}")
        if ref:
            sg = suggest_conversion(info["fps"], info["duration"], ref["fps"], ref["duration_s"])
            if sg is None:
                print("    suggestion: none – durations do not match any known conversion")
            elif sg["conv_type"]:
                off = f", off by {sg['delta']} s" if sg["delta"] else ""
                print(f"    suggestion: --mode {sg['conv_type']}   ({sg['reason']}{off})")
            else:
                print(f"    suggestion: {sg['reason']}")
    return rc


_DLL_TEXT = {
    "supported": "DEE 5.2.1 build supported by the flat-7.1 patch",
    "patched": "DLL currently patched (the patcher restores it from its backup on the next run)",
    "unsupported": "this DEE build is NOT supported by the flat-7.1 patch (needs 5.2.1-5994839)",
    "no-dll": "dee_audio_filter_ddp_atmos.dll is not next to dee.exe",
    "no-dee": "DEE not configured",
    "unreadable": "DLL unreadable",
}


def _cli_doctor() -> int:
    from .tools import doctor

    d = doctor()
    print(f"{APP_NAME} {app_version()}  ({'installed build' if d['frozen'] else 'from source'})")
    print(f"Python      : {d['python']} ({sys.executable})")
    print(f"ffmpeg      : {d['ffmpeg'] or 'NOT FOUND'}")
    print(f"ffprobe     : {d['ffprobe'] or 'NOT FOUND'}")
    print(f"Dolby pack  : {'installed' if d['dolby_pack'] else 'not installed (Preferences → Encode → Dolby tools pack, or: patcher)'}")
    print(f"deew        : {'ok (' + d['deew_via'] + ')' if d['deew'] else 'NOT AVAILABLE (install the Dolby tools pack)'}")
    print(f"deew config : {d['deew_config'] or 'not written yet (set the DEE path in Preferences → Encode, or: dee <path>)'}")
    print(f"DEE         : {d['dee'] or 'not found' + (' at ' + d['dee_path'] if d['dee_path'] else '')}")
    print(f"mediainfo   : {d['mediainfo'] or ('pymediainfo (Dolby tools pack)' if d['pymediainfo'] else 'NOT FOUND (needed to detect Atmos)')}")
    print(f"deezy       : {'ok (' + d['deezy_via'] + ')' if d['deezy'] else 'NOT FOUND — needed for DDP Atmos: DeeZy standalone exe (github.com/jessielw/DeeZy), set its path in Preferences → Encode'}")
    print(f"truehdd     : {d['truehdd'] or 'NOT FOUND — needed for DDP Atmos (github.com/truehdd/truehdd)'}")
    print(f"7.1 patcher : {d['patcher'] or 'not installed (Preferences → Encode → Dolby tools pack, or: patcher)'}")
    print(f"DEE for 7.1 : {_DLL_TEXT[d['dee_dll']['state']]}")
    print(f"settings    : {d['config_dir']}")
    ok = bool(d["ffmpeg"] and d["ffprobe"])
    print()
    print("AAC sources need ffmpeg only." if ok else "ffmpeg/ffprobe missing: nothing will work.")
    print("AC-3 / E-AC-3 / TrueHD sources additionally need deew + Dolby Encoding Engine.")
    print("Audio encode to DD / DDP needs DEE; DDP Atmos additionally needs truehdd + deezy + DEE 5.2.x.")
    return 0 if ok else 1


def _cli_patcher() -> int:
    from ..media import CancellationToken
    from ..subs import packs
    from .tools import DOLBY_PACK

    def progress(percent: float = 0, stage: str = "", *_args, **_kwargs) -> None:
        print(f"\r{stage} {float(percent or 0):.0f}%".ljust(60), end="", flush=True)

    packs.install(DOLBY_PACK, progress, lambda message: print(f"\n{message}"), CancellationToken())
    print(f"\ninstalled: {packs.pack_dir(DOLBY_PACK)}")
    return _cli_doctor()


def _cli_dee(path: str) -> int:
    from . import settings
    from .tools import Tools, write_deew_config

    t = Tools.resolve()
    print(f"wrote {write_deew_config(path, t.ffmpeg, t.ffprobe)}")
    settings.save_settings({"dee_path": path})
    return _cli_doctor()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiosync-cli", description=f"{APP_NAME} {app_version()}")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {app_version()}")
    sub = p.add_subparsers(dest="cmd")

    conv = sub.add_parser("convert", help="convert files or folders from the command line")
    conv.add_argument("inputs", nargs="+", help="files and/or folders")
    conv.add_argument("--mode", "-m", required=True, choices=list(FPS_CONVERSIONS))
    conv.add_argument("--output", "-o", default="output")
    conv.add_argument("--jobs", "-j", type=int, default=1)
    conv.add_argument("--recursive", "-r", action="store_true")
    conv.add_argument("--stream", "-s", type=int, default=0, help="audio stream index (0 = first)")
    conv.add_argument("--bitrate", "-b", type=int, default=0, help="kbps; 0 = keep the source bitrate")
    conv.add_argument("--overwrite", choices=["overwrite", "skip", "rename"], default="overwrite")

    enc = sub.add_parser("encode", help="audio-only encode to DD / DDP / DDP Atmos (no fps change)")
    enc.add_argument("inputs", nargs="+", help="files and/or folders")
    enc.add_argument("--format", "-f", choices=list(TARGETS), default="ddp")
    enc.add_argument("--channels", "-c", type=int, choices=list(TARGET_CHANNELS), default=0,
                     help="0 = same as source, else 1 / 2 / 6 (5.1) / 8 (7.1); never upmixes")
    enc.add_argument("--bitrate", "-b", type=int, default=0, help="kbps; 0 = DEE default for the layout")
    enc.add_argument("--no-atmos", action="store_true", help="encode the bed only, even for TrueHD Atmos sources")
    enc.add_argument("--atmos-71", choices=list(ATMOS71_LAYOUTS), default="flat",
                     help="7.1 Atmos legacy layout: flat = Lb Rb through the patched DEE (default), "
                          "dee = DEE's own 5.1+2 heights (Tfl Tfr)")
    enc.add_argument("--no-bed-conform", action="store_true", help="flat 7.1: do not pass --bed-conform to truehdd")
    enc.add_argument("--drc", choices=list(DRC_PROFILES), default="film_light")
    enc.add_argument("--output", "-o", default="output")
    enc.add_argument("--jobs", "-j", type=int, default=1)
    enc.add_argument("--recursive", "-r", action="store_true")
    enc.add_argument("--stream", "-s", type=int, default=0, help="audio stream index (0 = first)")
    enc.add_argument("--overwrite", choices=["overwrite", "skip", "rename"], default="overwrite")

    probe = sub.add_parser("probe", help="show audio streams and the frame rate a file is tied to")
    probe.add_argument("inputs", nargs="+", help="audio / video files")
    probe.add_argument("--target", "-t", help="video the audio must fit; suggests the conversion")

    sub.add_parser("doctor", help="show what is installed")
    sub.add_parser("patcher", help="install the Dolby tools pack: deew, pymediainfo and the DD+ 7.1 Atmos "
                                   "patcher (LumaVistaLab, GPL-3.0)")

    dee = sub.add_parser("dee", help="write deew's config.toml for a Dolby Encoding Engine path")
    dee.add_argument("path", help=r"path to dee.exe, e.g. C:\Dolby\DEE\dee.exe")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    if not args.cmd:
        parser().print_help()
        return 2
    if not is_windows():
        print(f"{WINDOWS_ONLY}.", file=sys.stderr)
        return 2
    if args.cmd in ("convert", "encode"):
        return _cli_convert(args)
    if args.cmd == "probe":
        return _cli_probe(args)
    if args.cmd == "doctor":
        return _cli_doctor()
    if args.cmd == "patcher":
        return _cli_patcher()
    return _cli_dee(args.path)
