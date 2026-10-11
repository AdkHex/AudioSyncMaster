"""What each of the planner's rules is worth: switch one off, plan every
bench case again, and hold the result against the plans with it on.

    python tests/bench/ablate.py OUT_DIR CASES_DIR [CASES_DIR ...] --baseline RUN_DIR/head \
        [--rules NAME ...] [--jobs 4]

``--baseline``: the plans with every rule on (``run_bench.py``'s plans of the
same checkout). Each rule's plans go to OUT_DIR/<rule>/ and are reused on a
later run. A rule whose removal changes nothing on any case earns its place
only by a test that shows what it is for; one whose removal helps is wrong.
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
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from score import score, score_voices  # noqa: E402

# How to switch each rule off, as code run against ``d`` (the planner's
# module) before planning. Each leaves the rest of the planner as it is.
RULES = {
    "foreign_scenes": "d._split_foreign = lambda aligner, blocks, notes=None: blocks\n"
                      "d._gap_is_foreign = lambda *a, **k: False",
    "verify_cuts": "d._verify_cuts = lambda *a, **k: None",
    "join_overlapping": "d._join_overlapping = lambda blocks: blocks",
    "gap_unused_dub": "d.GAP_UNUSED_MAX_S = 0.0\nd._consistent_in_gap = lambda pieces, narrow, neighbours: pieces",
    "recover_from_gaps": "d._recover_from_gaps = lambda *a, **k: []",
    "wide_pass": "d._wide_blocks = lambda *a, **k: []",
    "follow_drift": "d._follow_drift = lambda aligner, blocks, notes=None: blocks",
    "resolve_reuse": "d._resolve_reuse = lambda aligner, blocks, p, s: blocks",
    "follow_effects": "d._follow_effects = lambda aligner, blocks: None",
    "split_levels": "d._split_levels = lambda aligner, blocks: blocks",
    "trace_gaps": "d._trace_gaps = lambda aligner, blocks, s, notes, warnings: blocks",
    "settle": "d._settle = lambda aligner, block: False",
    "sharpen": "d._sharpen = lambda *a, **k: None",
    "snap_to_picture": "d._snap_to_picture = lambda *a, **k: None",
    "speech": "d._Aligner.speech_choice = lambda self, *a, **k: None\nd._speech_step = lambda *a, **k: None",
    "split_silences": "d._split_silences = lambda blocks, primary, secondary, warnings: blocks",
    "trim_uncertain_edges": "d._trim_uncertain_edges = lambda blocks, trims=None: blocks",
    "bridge_exposure_5s": "d.BRIDGE_EXPOSURE_S = 5.0",
    "short_levels": "d._split_short_levels = lambda aligner, blocks, p, s, notes=None: blocks",
    "ramp_guard": "d._reads_as_ramp = lambda *a, **k: False",
    "effects_rule": "d._Aligner._effects_rule = lambda self, found, runs, offsets_ms: (runs, offsets_ms)",
}

RUNNER = """
import json, os, sys, time
engine, video, vt, dub, dt, out, patch = sys.argv[1:8]
sys.path.insert(0, engine)
import audiosync.dubsync as d
exec(patch, {"d": d})
started = time.monotonic()
lines = []
plan = d.plan_dubsync(video, dub, video_track=int(vt), dub_track=int(dt), log=lines.append)
data = plan.to_dict()
data["_seconds"] = time.monotonic() - started
data["_log"] = lines
json.dump(data, open(out + ".part", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
os.replace(out + ".part", out)
"""


def plan_case(engine: str, case: dict, out: str, patch: str) -> str:
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        [sys.executable, "-c", RUNNER, engine, case["video"], str(case.get("videoTrack", 0)),
         case["dub"], str(case.get("dubTrack", 0)), out, patch],
        capture_output=True, text=True, env=env, timeout=4 * 3600,
    )
    if done.returncode != 0:
        json.dump({"error": done.stderr[-2000:], "segments": []}, open(out, "w"))
    return out


def totals(cases, plan_dir):
    out = {"correctS": 0.0, "wrongS": 0.0, "missedS": 0.0, "falseS": 0.0, "voiceOnLipsS": 0.0, "failed": 0}
    per_case = {}
    for case in cases:
        path = os.path.join(plan_dir, case["_id"].replace("/", "__") + ".json")
        plan = json.load(open(path))
        if plan.get("error") and not plan.get("segments"):
            out["failed"] += 1
            per_case[case["_id"]] = None
            continue
        s = score(case, plan)
        v = score_voices(case, plan) or {}
        for key in ("correctS", "wrongS", "missedS", "falseS"):
            out[key] += s[key]
        out["voiceOnLipsS"] += v.get("voiceOnLipsS", 0.0)
        per_case[case["_id"]] = s["correctS"]
    return out, per_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--engine", default=ROOT)
    parser.add_argument("--rules", nargs="*", default=sorted(RULES))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    cases = []
    for root in args.cases:
        for path in sorted(glob.glob(os.path.join(root, "*", "case.json"))):
            case = json.load(open(path))
            case["_id"] = f"{os.path.basename(os.path.normpath(root))}/{case['name']}"
            cases.append(case)
    work = [(rule, case) for rule in args.rules for case in cases]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(plan_case, os.path.abspath(args.engine), case,
                               os.path.join(args.out, rule, case["_id"].replace("/", "__") + ".json"), RULES[rule])
                   for rule, case in work]
        for future in futures:
            future.result()

    base, base_cases = totals(cases, args.baseline)
    print(f"with every rule on: correct {base['correctS']:.0f} s, wrong {base['wrongS']:.0f} s, "
          f"missed {base['missedS']:.0f} s, false {base['falseS']:.0f} s, voices on the lips {base['voiceOnLipsS']:.0f} s")
    print(f"\n{'switched off':22s} {'correct':>9s} {'wrong':>8s} {'missed':>8s} {'false':>8s} {'voices':>8s} "
          f"{'failed':>6s}  cases changed")
    rows = []
    for rule in args.rules:
        got, got_cases = totals(cases, os.path.join(args.out, rule))
        changed = [cid for cid in base_cases if got_cases.get(cid) != base_cases[cid]
                   and not (got_cases.get(cid) is not None and base_cases[cid] is not None
                            and abs(got_cases[cid] - base_cases[cid]) < 0.05)]
        delta = {k: got[k] - base[k] for k in ("correctS", "wrongS", "missedS", "falseS", "voiceOnLipsS")}
        rows.append({"rule": rule, **delta, "failed": got["failed"], "changed": changed})
        print(f"{rule:22s} {delta['correctS']:+9.1f} {delta['wrongS']:+8.1f} {delta['missedS']:+8.1f} "
              f"{delta['falseS']:+8.1f} {delta['voiceOnLipsS']:+8.1f} {got['failed']:6d}  {len(changed)}")
    json.dump(rows, open(os.path.join(args.out, "ablation.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
