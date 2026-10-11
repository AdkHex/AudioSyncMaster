"""Check a plan for a real pair, where no ground truth exists.

The dub and the original share their music and effects, so wherever the
plan lays the dub, the two tracks' onsets must agree there -- measured
window by window around the plan's own offset -- and wherever the plan
fills from the original, the dub should have nothing for that moment:
a strong match anywhere in the dub is dub the plan left out.

A dub whose background holds two copies of itself a tenth of a second
apart (see the Goblin notes) reads "off" here where its dialogue follows
the other copy; the lips decide those, not this check.

    python tests/bench/real_check.py PLAN.json [--cache DIR] [--list]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from audiosync.dubsync import ENVELOPE_RATE as R, _fast_fft_size, build_envelope  # noqa: E402

WINDOW_S = 10.0
STEP_S = 5.0
REACH_S = 1.0
STRONG_Z = 8.0
MISSED_Z = 12.0


def envelope(path: str, track: int, cache: Optional[str]):
    key = os.path.join(cache, f"env_{os.path.basename(path)}_{track}.pkl") if cache else None
    if key and os.path.exists(key):
        return pickle.load(open(key, "rb"))
    env = build_envelope(path, track)
    if key:
        os.makedirs(cache, exist_ok=True)
        pickle.dump(env, open(key, "wb"))
    return env


def ncc(template: np.ndarray, signal: np.ndarray) -> np.ndarray:
    t = template.astype(np.float64)
    s = signal.astype(np.float64)
    w = len(t)
    if len(s) < w:
        return np.zeros(0)
    c = t - t.mean()
    norm = np.sqrt((c * c).sum())
    if norm < 1e-9:
        return np.zeros(len(s) - w + 1)
    size = _fast_fft_size(len(s))
    num = np.fft.irfft(np.fft.rfft(s, size) * np.conj(np.fft.rfft(c, size)), size)[: len(s) - w + 1]
    o = np.concatenate([[0.0], np.cumsum(s)])
    q = np.concatenate([[0.0], np.cumsum(s * s)])
    var = np.maximum((q[w:] - q[:-w]) - (o[w:] - o[:-w]) ** 2 / w, 0.0)
    den = norm * np.sqrt(var)
    out = np.zeros(len(num))
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out


def peak(row: np.ndarray) -> Tuple[int, float]:
    k = int(np.argmax(row))
    away = np.abs(np.arange(len(row)) - k) > max(3, int(0.25 * R))
    if away.sum() < 8:
        return k, 0.0
    return k, float((row[k] - np.median(row[away])) / max(float(np.std(row[away])), 1e-9))


def check(plan: dict, cache: Optional[str]) -> dict:
    video = envelope(plan["videoPath"], int(plan.get("videoTrack", 0)), cache)
    dub = envelope(plan["dubPath"], int(plan.get("dubTrack", 0)), cache).at_speed(float(plan.get("speed") or 1.0))
    bands = [b for b in video.bands if b in dub.bands]
    w, m = int(WINDOW_S * R), int(REACH_S * R)
    rows = []
    segments = plan["segments"]
    end = float(plan.get("videoDurationS") or video.duration_s)
    t = 0.0
    while t + WINDOW_S <= end:
        covering = [s for s in segments if s["startS"] < t + WINDOW_S and s["endS"] > t]
        dubs = [s for s in covering if s["kind"] == "dub"]
        fills = [s for s in covering if s["kind"] != "dub"]
        row = {"t": t}
        s0 = int(t * R)
        if dubs and not fills and max(d["offsetS"] for d in dubs) - min(d["offsetS"] for d in dubs) < 0.002:
            o = int(round(dubs[0]["offsetS"] * R))
            total = None
            for b in bands:
                lo = s0 + o - m
                if lo < 0 or lo + w + 2 * m > len(dub.onsets(b)):
                    continue
                r = ncc(video.onsets(b)[s0:s0 + w], dub.onsets(b)[lo: lo + w + 2 * m])
                if r.size:
                    u = (r - np.median(r)) / (np.std(r) + 1e-9)
                    total = u if total is None else total + u
            if total is not None:
                k, z = peak(total)
                row.update(kind="dub", residualMs=(k - m) / R * 1000.0, z=z)
            else:
                row.update(kind="dub", residualMs=None, z=0.0)
        elif fills and not dubs:
            r = ncc(video.onset[s0:s0 + w], dub.onset)
            k, z = peak(r) if r.size else (0, 0.0)
            row.update(kind="fill", bestOffsetS=(k - s0) / R, z=z)
        else:
            row.update(kind="seam")
        rows.append(row)
        t += STEP_S
    dub_rows = [r for r in rows if r["kind"] == "dub" and r.get("residualMs") is not None and r["z"] >= STRONG_Z]
    res = np.array([abs(r["residualMs"]) for r in dub_rows])
    summary = {
        "windows": len(rows),
        "dubMeasured": len(dub_rows),
        "within20ms": int(np.sum(res <= 20)),
        "over45ms": int(np.sum(res > 45)),
        "over100ms": int(np.sum(res > 100)),
        "fillWindows": sum(1 for r in rows if r["kind"] == "fill"),
        "fillsTheDubHas": sum(1 for r in rows if r["kind"] == "fill" and r["z"] >= MISSED_Z),
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--cache")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    result = check(json.load(open(args.plan)), args.cache)
    print(json.dumps(result["summary"]))
    if args.list:
        h = lambda x: f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{int(x % 60):02d}"
        for r in result["rows"]:
            if r["kind"] == "dub" and r.get("residualMs") is not None and r["z"] >= STRONG_Z and abs(r["residualMs"]) > 45:
                print(f"  {h(r['t'])}  dub off by {r['residualMs']:+.0f} ms (z {r['z']:.0f})")
            if r["kind"] == "fill" and r["z"] >= MISSED_Z:
                print(f"  {h(r['t'])}  filled, but the dub has it (offset {r['bestOffsetS']:+.3f}s, z {r['z']:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
