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


def voices_at(plan: dict, times: np.ndarray) -> np.ndarray:
    """Dub file time of the voices the written track plays at each video
    time, NaN where it plays none: the plan's stretch carries them unless a
    voice piece (see ``voicefix``) took them elsewhere, and a piece plays
    its own at its own level."""
    out = plan_at(plan, times)
    speed = float(plan.get("speed") or 1.0)
    pieces = plan.get("voicePieces") or []
    for piece in pieces:
        decoded = out * speed
        moved = (decoded >= piece["dubStartS"]) & (decoded < piece["dubEndS"])
        out[moved] = np.nan
    for piece in pieces:
        inside = (times >= piece["videoStartS"]) & (times < piece["videoEndS"])
        out[inside] = (times[inside] + piece["levelS"]) / speed
    return out


def score_voices(case: dict, plan: dict) -> Optional[Dict[str, float]]:
    """Where the dub's lines play against where they belong, judged only
    while the dub speaks: on the lips (within 45 ms), off them, or gone."""
    if "voiceTruth" not in case:
        return None
    duration = float(plan.get("videoDurationS") or max(p["videoEnd"] for p in case["voiceTruth"]))
    times = np.arange(0.0, duration, STEP_S)
    voiced = {"truth": case["voiceTruth"], "speed": case.get("speed", 1.0)}
    want = truth_at(voiced, times)
    got = voices_at(plan, times)
    speaking = np.zeros(len(times), dtype=bool)
    for a, b in case.get("speech", []):
        speaking |= (times >= a) & (times < b)
    judged = speaking & ~np.isnan(want) & ~near_edges(voiced, times)
    speed = float(case.get("speed", 1.0))
    error_ms = np.abs(got - want) * speed * 1000.0
    placed = judged & ~np.isnan(got)
    result = {
        "voiceS": float(judged.sum() * STEP_S),
        "voiceOnLipsS": float(np.sum(placed & (error_ms <= CLOSE_MS)) * STEP_S),
        "voiceOffS": float(np.sum(placed & (error_ms > CLOSE_MS)) * STEP_S),
        "voiceGoneS": float(np.sum(judged & np.isnan(got)) * STEP_S),
        "voicePieces": len(plan.get("voicePieces") or []),
    }
    result["voicePct"] = 100.0 * result["voiceOnLipsS"] / max(result["voiceS"], 1e-9)
    return result


def main() -> int:
    case = json.load(open(sys.argv[1]))
    plan = json.load(open(sys.argv[2]))
    for key, value in score(case, plan).items():
        print(f"{key:12s} {value}")
    for key, value in (score_voices(case, plan) or {}).items():
        print(f"{key:12s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def problems(case: dict, plan: dict, min_s: float = 0.3) -> List[str]:
    """The runs of wrong, missed and false time, in words."""
    duration = float(plan.get("videoDurationS") or max(p["videoEnd"] for p in case["truth"]))
    times = np.arange(0.0, duration, STEP_S)
    want, got = truth_at(case, times), plan_at(plan, times)
    keep = ~near_edges(case, times)
    speed = float(case.get("speed", 1.0))
    err = (got - want) * speed * 1000.0
    label = np.full(len(times), "", dtype=object)
    both = ~np.isnan(want) & ~np.isnan(got)
    label[both & (np.abs(err) > CLOSE_MS)] = "wrong"
    label[~np.isnan(want) & np.isnan(got)] = "missed"
    label[np.isnan(want) & ~np.isnan(got)] = "false"
    label[~keep] = ""
    out, i = [], 0
    while i < len(times):
        if not label[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(times) and label[j + 1] == label[i]:
            j += 1
        if (j - i + 1) * STEP_S >= min_s:
            extra = f" by {np.nanmedian(err[i:j + 1]):+.0f} ms" if label[i] == "wrong" else ""
            out.append(f"{label[i]:6s} {times[i]:8.2f}-{times[j] + STEP_S:8.2f} ({(j - i + 1) * STEP_S:5.1f}s){extra}")
        i = j + 1
    return out
