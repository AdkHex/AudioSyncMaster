"""Run an engine checkout over the measurement bench the way MkvBatchMux does.

    FFMPEG_DIR=... python run_bench.py ENGINE_ROOT CASES_DIR OUT.json [--jobs 4] [--only a,b] [--timeline]

ENGINE_ROOT is any directory holding an ``audiosync`` package -- this checkout
(``.``), or another revision extracted with ``git archive`` -- so two engines
can be compared case for case.

Pass 1: survey, 6 x 45 s windows, +-60 s, timeline off (the app's defaults).
Pass 2: for a pair the app would search again (error, no delay, confidence < 0.5):
        the fast route over +-300 s.
Scored on the value the app fills in: delayAtStartMs, else delayMs.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

AC3_COMP = 256 / 48000 * 1000


def comp(path):
    return AC3_COMP if path.endswith((".ac3", ".eac3")) else 0.0


def expected(t):
    base = t["delay_ms"]
    if t["kind"] in ("rate", "drift"):
        base = base / t["speed"]
    return base + t["decode_shift_ms"] + t.get("latency_ms", 0.0) - comp(t["secondary"])


def run_case(engine_root, name, t, timeline):
    sys.path.insert(0, engine_root)
    from audiosync.analyze import analyze_pair
    t0 = time.time()
    r = analyze_pair(t["primary"], t["secondary"], window_s=45.0, window_count=6, max_offset_ms=60000.0,
                     prefer_fast=False, timeline=timeline).to_dict()
    survey_s = time.time() - t0
    passes = "survey"
    d = r["delayAtStartMs"] if r["delayAtStartMs"] is not None else r["delayMs"]
    conf = r["confidence"]
    if r["error"] or d is None or (conf is not None and conf < 0.5):
        t1 = time.time()
        r2 = analyze_pair(t["primary"], t["secondary"], window_s=45.0, window_count=6, max_offset_ms=300000.0,
                          prefer_fast=True, timeline=False).to_dict()
        r2["_survey"] = {k: r[k] for k in ("delayMs", "delayAtStartMs", "confidence", "error")}
        r = r2
        passes = "survey+wide"
        survey_s += time.time() - t1
    r["_elapsed_s"] = round(survey_s, 2)
    r["_passes"] = passes
    return name, r


def autofillable(r):
    d = r["delayAtStartMs"] if r["delayAtStartMs"] is not None else r["delayMs"]
    if r["error"] or d is None:
        return False, "error" if r["error"] else "no delay"
    if r["isLikelyCut"] and not r["isMinorSlip"]:
        return False, "cut"
    if abs(d) > 300000:
        return False, "implausible"
    if r["confidence"] is not None and r["confidence"] < 0.5:
        return False, f"weak {r['confidence']:.2f}"
    return True, ""


def score(t, r):
    d = r["delayAtStartMs"] if r["delayAtStartMs"] is not None else r["delayMs"]
    exp = expected(t)
    err = None if d is None else d - exp
    fill, why = autofillable(r)
    if t["kind"] == "none":
        # Unrelated audio: the only right answer is not to fill a delay in.
        return ("refused" if not fill else "FALSE-MATCH"), err
    if t["kind"] == "constant":
        if not fill:
            verdict = f"WITHHELD({why})"
        elif abs(err) <= 2:
            verdict = "exact"
        elif abs(err) <= 10:
            verdict = "good"
        elif abs(err) <= 45:
            verdict = "close"
        else:
            verdict = "WRONG"
    elif t["kind"] == "cut":
        ok = r["isLikelyCut"] and not r["isMinorSlip"] and err is not None and abs(err) <= 10
        verdict = "cut-ok" if ok else ("CUT-MISSED" if fill else f"cut?({why})")
    elif t["kind"] == "drift":
        drift_exp = (1 / t["speed"] - 1) * 1000
        drift = r["driftMsPerS"]
        ok_drift = drift is not None and abs(drift - drift_exp) <= abs(drift_exp) * 0.03
        ok_delay = err is not None and abs(err) <= 20
        verdict = "drift-ok" if ok_drift and ok_delay else ("drift-only" if ok_drift else "DRIFT-MISSED")
    else:
        drift_exp = (1 / t["speed"] - 1) * 1000
        drift = r["driftMsPerS"]
        ok_rate = r["isRateMismatch"] and drift is not None and abs(drift - drift_exp) <= abs(drift_exp) * 0.03
        ok_delay = err is not None and abs(err) <= 20
        verdict = "rate-ok" if ok_rate and ok_delay else ("rate-only" if ok_rate else ("RATE-MISSED" if fill else f"rate?({why})"))
    return verdict, err


def main():
    engine_root, cases_dir, out = sys.argv[1:4]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    timeline = "--timeline" in sys.argv
    os.environ["PATH"] = os.environ["FFMPEG_DIR"] + os.pathsep + os.environ["PATH"]
    truth = json.load(open(os.path.join(cases_dir, "truth.json")))
    names = [n for n in truth if only is None or n in only]
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(run_case, engine_root, n, truth[n], timeline) for n in names]
        for f in as_completed(futs):
            name, r = f.result()
            results[name] = r
            v, err = score(truth[name], r)
            print(f"{name:26s} {v:18s} err={'-' if err is None else f'{err:+.2f}'} conf={r['confidence'] or 0:.3f} "
                  f"method={r['method']} {r['_passes']} {r['_elapsed_s']}s", flush=True)
    total = time.time() - t0
    rows = []
    for n in names:
        v, err = score(truth[n], results[n])
        rows.append((n, v, err, results[n]))
    json.dump({n: {"verdict": v, "err": err, "result": r} for n, v, err, r in rows}, open(out, "w"), indent=1, default=str)
    print("\nSUMMARY", engine_root)
    tally = {}
    for n, v, err, r in rows:
        key = v.split("(")[0]
        tally[key] = tally.get(key, 0) + 1
    print(" ".join(f"{k}={c}" for k, c in sorted(tally.items())), f"wall={total:.0f}s",
          f"sum_elapsed={sum(r['_elapsed_s'] for *_, r in rows):.0f}s")


if __name__ == "__main__":
    main()
