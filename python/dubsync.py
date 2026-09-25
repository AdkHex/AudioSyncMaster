#!/usr/bin/env python3
"""Lay a cut dub onto its video, filling what it lacks with the original.

    python python/dubsync.py MOVIE.mkv MOVIE.hindi.eac3

reads both, works out which stretch of the dub belongs at every moment of
the video, writes MOVIE.hindi.dubsynced.flac beside the video -- the dub
wherever it exists, the video's own audio wherever it does not, the length
of the video exactly -- and then measures the result against the video.

Everything it decides is printed as a table, one line per piece of the
output, and saved as JSON beside the output. That JSON can be edited and
rendered again with --from-plan, which skips the analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audiosync.dubrender import (  # noqa: E402
    CODECS,
    RenderOptions,
    default_output_path,
    mux,
    render,
    resolve_codec,
)
from audiosync.dubsync import (  # noqa: E402
    DEFAULT_SEARCH_S,
    DubSyncPlan,
    build_envelope,
    plan_dubsync,
    verify_output,
)
from audiosync import voicetools  # noqa: E402
from audiosync.linecheck import line_check  # noqa: E402
from audiosync.voicefix import check_voices  # noqa: E402
from audiosync.media import Cancelled, MediaError, has_ffmpeg  # noqa: E402


def _speed(value: str):
    if value == "auto":
        return None
    try:
        if "/" in value:
            return float(Fraction(value))
        return float(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"speed must be 'auto', a number or a ratio like 25/23.976: {value}") from exc


def _gain(value: str):
    if value == "auto":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"fill gain must be 'auto' or a number of dB: {value}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dubsync",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  dubsync movie.mkv movie.hin.eac3\n"
            "  dubsync movie.mkv movie.hin.eac3 --codec eac3 --mux --lang hin\n"
            "  dubsync movie.mkv movie.hin.eac3 --plan-only\n"
            "  dubsync --from-plan movie.hin.dubsync.json -o fixed.flac\n"
        ),
    )
    parser.add_argument("video", nargs="?", help="the video (or its original-language audio)")
    parser.add_argument("dub", nargs="?", help="the dub to lay onto it")
    parser.add_argument("-o", "--output", help="where to write the synced track (default: beside the video)")
    parser.add_argument("--video-track", type=int, default=0, help="audio stream of the video to compare against (0-based)")
    parser.add_argument("--dub-track", type=int, default=0, help="audio stream of the dub (0-based)")
    parser.add_argument("--codec", choices=["same", *sorted(CODECS)], default="same",
                        help="output codec: the dub's own where it can be written back, else flac (default: same)")
    parser.add_argument("--bitrate", help="bitrate for lossy codecs, e.g. 640k (default: by channel count)")
    parser.add_argument("--sample-rate", type=int, help="output sample rate (default: the dub's)")
    parser.add_argument("--channels", type=int, help="output channel count (default: the dub's)")
    parser.add_argument("--search", type=float, default=DEFAULT_SEARCH_S,
                        help=f"seconds beyond the duration difference to search for the dub (default {DEFAULT_SEARCH_S:.0f})")
    parser.add_argument("--speed", type=_speed, default="auto",
                        help="playback speed of the dub relative to the video: auto, a number, or a ratio like 25/23.976")
    parser.add_argument("--dub-rate", type=float, default=None, metavar="FPS",
                        help="the frame rate the dub was mastered at (e.g. 23.976); with the video's own rate this fixes the speed")
    parser.add_argument("--fill-gain", type=_gain, default="auto",
                        help="dB applied to the original where it fills a gap: auto (match levels) or a number")
    parser.add_argument("--fill-unmatched", action="store_true",
                        help="also fill stretches where the dub did not correlate even though its offset is unchanged")
    parser.add_argument("--stretch", choices=("resample", "atempo"), default="resample",
                        help="how a dub at another speed is brought onto the video's clock (default: resample)")
    parser.add_argument("--xfade", type=float, default=10.0, help="crossfade at each seam, in ms (default 10)")
    parser.add_argument("--plan-only", action="store_true", help="analyse and print the plan; write nothing but the plan JSON")
    parser.add_argument("--plan", help="where to save the plan JSON (default: beside the output)")
    parser.add_argument("--from-plan", help="render this plan JSON instead of analysing")
    parser.add_argument("--mux", nargs="?", const=True, default=False, metavar="OUT.mkv",
                        help="also write a copy of the video with the synced track added")
    parser.add_argument("--lang", help="language tag for the muxed track, e.g. hin")
    parser.add_argument("--title", help="title for the muxed track")
    parser.add_argument("--no-verify", action="store_true", help="skip measuring the finished track against the video")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--quiet", action="store_true", help="only print the plan and the verification")
    parser.add_argument("--fix-voices", dest="fix_voices", action="store_true", default=True,
                        help="check the dub's voices against the lips and move them where they were cut apart "
                             "from the music (default, when the voice tools are installed)")
    parser.add_argument("--no-fix-voices", dest="fix_voices", action="store_false",
                        help="skip the voice check, and write a plan's voice moves as if there were none")
    parser.add_argument("--voice-tools", choices=("status", "install", "remove"),
                        help="show, install (about 1 GB) or remove the voice tools, then exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not has_ffmpeg() and not args.voice_tools:
        print("ffmpeg was not found. Install it and put it on PATH.", file=sys.stderr)
        return 2

    started = time.monotonic()
    say = (lambda _m: None) if args.quiet else (lambda m: print(f"  {m}", flush=True))

    if args.voice_tools:
        if args.voice_tools == "install":
            print(f"--- installing the voice tools into {voicetools.tools_dir()} ---")
            state = voicetools.install(
                progress=lambda p, s: print(f"  [{p:3d}%] {s}", flush=True), log=say)
        elif args.voice_tools == "remove":
            state = voicetools.remove()
        else:
            state = voicetools.status()
        print(json.dumps(state, indent=2))
        return 0

    def progress(percent: int, stage: str) -> None:
        if not args.quiet and sys.stdout.isatty():
            print(f"\r  {percent:3d}% {stage:<40}", end="", flush=True)

    envelopes: dict = {}
    try:
        if args.from_plan:
            with open(args.from_plan, "r", encoding="utf-8") as handle:
                plan = DubSyncPlan.from_dict(json.load(handle))
            if args.video:
                plan.video_path = args.video
            if args.dub:
                plan.dub_path = args.dub
            print(f"--- plan from {os.path.basename(args.from_plan)} ---")
        else:
            if not args.video or not args.dub:
                build_parser().error("a video and a dub are needed (or --from-plan)")
            for path in (args.video, args.dub):
                if not os.path.isfile(path):
                    print(f"not found: {path}", file=sys.stderr)
                    return 2
            print(f"--- reading {os.path.basename(args.video)} and {os.path.basename(args.dub)} ---")
            plan = plan_dubsync(
                args.video, args.dub,
                video_track=args.video_track, dub_track=args.dub_track,
                search_s=args.search, speed=args.speed, dub_rate=args.dub_rate, fill_gain_db=args.fill_gain,
                keep_unmatched_dub=not args.fill_unmatched,
                progress=progress, log=say, envelopes=envelopes,
            )
            if not args.quiet and sys.stdout.isatty():
                print("\r" + " " * 60 + "\r", end="")
            if plan.error:
                print(f"could not plan: {plan.error}", file=sys.stderr)
                return 1
            if args.fix_voices:
                print("--- checking the dub's voices against the lips ---")
                plan.voice_pieces = check_voices(plan, progress=progress, log=say)
                if not args.quiet and sys.stdout.isatty():
                    print("\r" + " " * 60 + "\r", end="")
            print(f"--- plan ({time.monotonic() - started:.0f}s) ---")

        print(plan.describe())

        codec = resolve_codec(args.codec, plan)
        output = args.output or default_output_path(plan, codec)
        plan_path = args.plan or (
            os.path.splitext(output)[0] + ".dubsync.json"
            if not args.from_plan else None
        )
        if plan_path:
            with open(plan_path, "w", encoding="utf-8") as handle:
                json.dump(plan.to_dict(), handle, indent=2)
            say(f"plan saved to {plan_path}")
        if args.plan_only:
            return 0

        if os.path.exists(output) and not args.overwrite:
            print(f"output already exists: {output} (use --overwrite)", file=sys.stderr)
            return 1
        print(f"--- writing {os.path.basename(output)} ---")
        options = RenderOptions(
            codec=codec, bitrate=args.bitrate, sample_rate=args.sample_rate,
            channels=args.channels, xfade_s=args.xfade / 1000.0, stretch=args.stretch,
            fix_voices=args.fix_voices,
        )
        result = render(plan, output, options, progress=progress, log=say)
        if not args.quiet and sys.stdout.isatty():
            print("\r" + " " * 60 + "\r", end="")
        say(f"wrote {result.seconds:.1f}s, {result.channels}ch {result.sample_rate}Hz")
        for warning in result.warnings:
            print(f"  ! {warning}")

        if not args.no_verify:
            print("--- checking the finished track against the original ---")
            stage = "reading the finished track"
            primary = envelopes.get("primary") or build_envelope(
                plan.video_path, plan.video_track,
                progress=lambda f: progress(int(50 * f), "reading the original"),
                expected_duration_s=plan.video_duration_s,
            )
            finished = build_envelope(
                output, progress=lambda f: progress(int(100 * f), stage),
                expected_duration_s=plan.video_duration_s,
            )
            if not args.quiet and sys.stdout.isatty():
                print("\r" + " " * 60 + "\r", end="")
            verification = verify_output(primary, finished, plan)
            print("--- checking the dub's lines against the original's ---")
            lines = line_check(plan, output)
            verification.lines, verification.lines_text = lines.to_dict(), lines.describe()
            print(verification.describe())

        if args.mux:
            target = args.mux if isinstance(args.mux, str) else None
            print("--- muxing ---")
            written = mux(plan, output, target, language=args.lang, title=args.title, overwrite=args.overwrite)
            print(f"  wrote {written}")
        print(f"--- done in {time.monotonic() - started:.0f}s ---")
        return 0
    except Cancelled:
        print("cancelled", file=sys.stderr)
        return 130
    except MediaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
