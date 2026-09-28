"""Run engines over bench cases and compare them against the ground truth.

    python tests/bench/run_bench.py OUT_DIR CASES_DIR [CASES_DIR ...] \
        --engine head=. --engine v211=/path/to/old/checkout [--jobs 4] [--only NAME ...]

Each engine is a checkout whose ``audiosync`` package is imported as is, so
an old release can be held against the working tree on the same cases.
Plans are kept in OUT_DIR/<engine>/<case>.json and reused on a later run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from score import score  # noqa: E402

RUNNER = """
import json, os, sys, time
engine, video, vt, dub, dt, out = sys.argv[1:7]
sys.path.insert(0, engine)
from audiosync.dubsync import plan_dubsync
started = time.monotonic()
lines = []
plan = plan_dubsync(video, dub, video_track=int(vt), dub_track=int(dt), log=lines.append)
data = plan.to_dict()
data["_seconds"] = time.monotonic() - started
data["_log"] = lines
json.dump(data, open(out + ".part", "w"), indent=1)
os.replace(out + ".part", out)
"""


def plan_case(engine: str, path: str, case: dict, out: str) -> str:
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", RUNNER, path, case["video"], str(case.get("videoTrack", 0)),
         case["dub"], str(case.get("dubTrack", 0)), out],
        capture_output=True, text=True, env=env, timeout=4 * 3600,
    )
    if completed.returncode != 0:
        json.dump({"error": completed.stderr[-2000:], "segments": []}, open(out, "w"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--engine", action="append", required=True, help="name=checkout")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    engines = [tuple(e.split("=", 1)) for e in args.engine]
    cases = []
    for root in args.cases:
        for path in sorted(glob.glob(os.path.join(root, "*", "case.json"))):
            case = json.load(open(path))
            if args.only and case["name"] not in args.only:
                continue
            case["_id"] = f"{os.path.basename(os.path.normpath(root))}/{case['name']}"
            cases.append(case)
    work = [(name, os.path.abspath(path), case) for case in cases for name, path in engines]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(plan_case, path, path, case, os.path.join(args.out, name, case["_id"].replace("/", "__") + ".json")): (name, case)
            for name, path, case in work
        }
        for future in futures:
            name, case = futures[future]
            future.result()
            print(f"planned {case['_id']} with {name}", flush=True)

    rows = []
    header = f"{'case':24s}" + "".join(f" | {name:>36s}" for name, _ in engines)
    print("\n" + header)
    print("-" * len(header))
    totals = {name: {"dubS": 0.0, "correctS": 0.0, "wrongS": 0.0, "missedS": 0.0, "falseS": 0.0} for name, _ in engines}
    for case in cases:
        line = f"{case['_id']:24s}"
        for name, _ in engines:
            plan_path = os.path.join(args.out, name, case["_id"].replace("/", "__") + ".json")
            plan = json.load(open(plan_path))
            if plan.get("error") and not plan.get("segments"):
                line += f" | {'FAILED: ' + str(plan['error']).splitlines()[-1][:28]:>36s}"
                rows.append({"case": case["_id"], "engine": name, "failed": plan["error"]})
                totals[name]["missedS"] += sum(p["videoEnd"] - p["videoStart"] for p in case["truth"])
                continue
            s = score(case, plan)
            rows.append({"case": case["_id"], "engine": name, **s})
            for key in totals[name]:
                totals[name][key] += s[key]
            line += (f" | {s['correctPct']:5.1f}% w{s['wrongS']:6.1f} m{s['missedS']:6.1f} "
                     f"f{s['falseS']:5.1f} {s['seconds']:5.0f}s")
        print(line, flush=True)
    print("-" * len(header))
    line = f"{'TOTAL':24s}"
    for name, _ in engines:
        t = totals[name]
        line += (f" | {100 * t['correctS'] / max(t['dubS'], 1e-9):5.1f}% w{t['wrongS']:6.0f} "
                 f"m{t['missedS']:6.0f} f{t['falseS']:5.0f}      ")
    print(line)
    json.dump(rows, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
