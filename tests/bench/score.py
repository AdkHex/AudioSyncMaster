"""Hold a dub-sync plan against a bench case's ground truth.

Every 50 ms of the video is one of:
    correct  the plan reads the dub within 20 ms of where it belongs
    close    within 45 ms, the edge of what can be seen on lips
    wrong    further off than that: a visible (or audible) error
    missed   the dub has this moment but the plan fills it from the original
    false    the dub has nothing for this moment but the plan plays some of it
    none     neither has anything, and the plan fills: right
Moments within EDGE_S of a true cut are left out: where exactly a cut lands
within a frame is not an error worth counting, and a cut placed further
off than that shows up as wrong or missed time next to it.

    python tests/bench/score.py CASE_DIR/case.json PLAN.json
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Optional

import numpy as np

STEP_S = 0.05
EDGE_S = 0.1
CORRECT_MS = 20.0
CLOSE_MS = 45.0


def truth_at(case: dict, times: np.ndarray) -> np.ndarray:
    """Dub file time for each video time, NaN where the dub has nothing."""
    out = np.full(len(times), np.nan)
    speed = float(case.get("speed", 1.0))
    for piece in case["truth"]:
        inside = (times >= piece["videoStart"]) & (times < piece["videoEnd"])
        out[inside] = piece["dubStart"] + (times[inside] - piece["videoStart"]) / speed
    return out


def near_edges(case: dict, times: np.ndarray) -> np.ndarray:
    edges: List[float] = []
    for piece in case["truth"]:
        edges += [piece["videoStart"], piece["videoEnd"]]
    mask = np.zeros(len(times), dtype=bool)
    for edge in edges:
        mask |= np.abs(times - edge) < EDGE_S
    return mask


def plan_at(plan: dict, times: np.ndarray) -> np.ndarray:
    """Dub file time the plan plays at each video time, NaN where it fills."""
    out = np.full(len(times), np.nan)
    speed = float(plan.get("speed") or 1.0)
    for seg in plan.get("segments", []):
        if seg.get("kind") != "dub":
            continue
        inside = (times >= seg["startS"]) & (times < seg["endS"])
        decoded = seg["sourceStartS"] + (times[inside] - seg["startS"])
        out[inside] = decoded / speed
    return out


def score(case: dict, plan: dict, duration: Optional[float] = None) -> Dict[str, float]:
    if duration is None:
        duration = float(plan.get("videoDurationS") or max(p["videoEnd"] for p in case["truth"]))
    times = np.arange(0.0, duration, STEP_S)
    want = truth_at(case, times)
    got = plan_at(plan, times)
    keep = ~near_edges(case, times)
    speed = float(case.get("speed", 1.0))
    has_want, has_got = ~np.isnan(want), ~np.isnan(got)
    both = keep & has_want & has_got
    error_ms = np.abs(got - want) * speed * 1000.0
    result = {
        "dubS": float(np.sum(keep & has_want) * STEP_S),
        "correctS": float(np.sum(both & (error_ms <= CORRECT_MS)) * STEP_S),
        "closeS": float(np.sum(both & (error_ms > CORRECT_MS) & (error_ms <= CLOSE_MS)) * STEP_S),
        "wrongS": float(np.sum(both & (error_ms > CLOSE_MS)) * STEP_S),
        "missedS": float(np.sum(keep & has_want & ~has_got) * STEP_S),
        "falseS": float(np.sum(keep & ~has_want & has_got) * STEP_S),
        "noneS": float(np.sum(keep & ~has_want & ~has_got) * STEP_S),
        "medianErrMs": float(np.median(error_ms[both])) if both.any() else float("nan"),
        "p95ErrMs": float(np.percentile(error_ms[both & (error_ms <= CLOSE_MS)], 95))
        if (both & (error_ms <= CLOSE_MS)).any() else float("nan"),
        "speedPlan": float(plan.get("speed") or 1.0),
        "speedTrue": speed,
        "dubSegments": sum(1 for s in plan.get("segments", []) if s.get("kind") == "dub"),
        "seconds": float(plan.get("_seconds", float("nan"))),
    }
    result["correctPct"] = 100.0 * result["correctS"] / max(result["dubS"], 1e-9)
    result["badS"] = result["wrongS"] + result["missedS"] + result["falseS"]
    result["error"] = plan.get("error")
    result["delayOnly"] = plan.get("delayOnly")
    return result


def main() -> int:
    case = json.load(open(sys.argv[1]))
    plan = json.load(open(sys.argv[2]))
    for key, value in score(case, plan).items():
        print(f"{key:12s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
