"""End-to-end check of Subsync through the real bridge, on real films.

Drives python/bridge.py over stdin/stdout exactly as the desktop host does
(JSON request lines in, JSON event lines out) and checks what comes back.
Everything is written under OUT (default /tmp/subsync-e2e/out); the source
films are only read.

    python/.venv/bin/python tests/manual/subsync_e2e.py

Films used when present (skipped otherwise):
  DEMON  ~/Movies/...Demon.Slayer...mkv  Japanese audio + an English SRT track
  HERE   ~/Downloads/Here Now ...mkv     a Blu-ray PGS track (Italian SDH)
  GOBLIN ~/Downloads/Video.mkv           Korean audio, no subtitles
Speech recognition needs AUDIOSYNC_PACK_PYTHON_ASR_MLX (or an installed pack).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from audiosync.subs import formats, retime  # noqa: E402

OUT = os.environ.get("SUBSYNC_E2E_OUT", "/tmp/subsync-e2e/out")
DEMON = next(iter(glob.glob(os.path.expanduser("~/Movies/*Demon.Slayer*.mkv"))), None)
HERE = next(iter(glob.glob(os.path.expanduser("~/Downloads/Here Now*.mkv"))), None)
GOBLIN = os.path.expanduser("~/Downloads/Video.mkv")


class Bridge:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "python", "bridge.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(OUT, "bridge.stderr"), "w"),
            text=True, cwd=ROOT,
        )
        ready = json.loads(self.proc.stdout.readline())
        assert ready.get("type") == "ready", ready

    def ask(self, request: dict, terminal: str, echo=None) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            event = json.loads(line)
            if echo:
                echo(event)
            if event.get("type") == terminal:
                return event
        raise RuntimeError("bridge closed")

    def close(self) -> None:
        self.proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=60)


def progress_printer():
    last = {}

    def echo(event: dict) -> None:
        kind = event.get("type")
        if kind == "subsJobProgress":
            key = (event["job"], event["stage"])
            if last.get(event["job"]) != key and event["percent"] in (0, 100) or last.get(event["job"], (None, None))[1] != event["stage"]:
                last[event["job"]] = key
                print(f"    job {event['job'] + 1}: {event['stage']} {event['percent']}%", flush=True)
        elif kind == "subsJobDone":
            state = "ERROR " + str(event.get("error")) if event.get("error") else event.get("summary")
            print(f"  done job {event['job'] + 1} [{event['task']}]: {state}", flush=True)

    return echo


def batch(bridge: Bridge, jobs: list, workers: int = 2) -> list:
    for job in jobs:
        job.setdefault("output", {}).setdefault("dir", OUT)
    started = time.monotonic()
    done = bridge.ask({"command": "subsBatch", "jobs": jobs, "maxWorkers": workers}, "subsBatchDone", progress_printer())
    print(f"  batch took {time.monotonic() - started:.1f}s")
    return done["outcomes"]


def cue_errors(result_path: str, truth_path: str) -> tuple:
    got = formats.read(result_path).cues
    want = formats.read(truth_path).cues
    n = min(len(got), len(want))
    errs = sorted(abs(g.start - w.start) for g, w in zip(got[:n], want[:n]))
    return len(got), len(want), errs[len(errs) // 2] if errs else None, errs[int(len(errs) * 0.9)] if errs else None


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    bridge = Bridge()
    failures = []

    def check(ok: bool, what: str) -> None:
        print(("  PASS  " if ok else "  FAIL  ") + what, flush=True)
        if not ok:
            failures.append(what)

    films = [p for p in (DEMON, HERE, GOBLIN) if p and os.path.isfile(p)]
    print("== probe")
    probed = bridge.ask({"command": "subsProbe", "paths": films}, "subsProbeResult")["files"]
    for f in probed:
        tracks = ", ".join(f"{t['index']}:{t['codec']}/{t['language']}/{t['kind']}" for t in f["subtitleTracks"])
        video = f.get("video") or {}
        print(f"  {f['name'][:60]}  {f['kind']} {video.get('width')}x{video.get('height')} "
              f"{video.get('hdr')} fps={video.get('fps')}  subs=[{tracks}]  err={f.get('error')}")
    check(all(not f.get("error") for f in probed), "every film probes without error")

    print("== capabilities")
    caps = bridge.ask({"command": "subsCaps", "secrets": {}}, "subsCapsResult")["caps"]
    for group, engines in caps["engines"].items():
        print(f"  {group}: " + ", ".join(f"{e['id']}{'' if e['available'] else ' (no)'}" for e in engines))
    print("  packs: " + ", ".join(f"{p['id']}{' installed' if p['installed'] else ''}" for p in caps["packs"]))

    if DEMON:
        print("== Demon Slayer: extract the English SRT, style check, fps, then damage it and sync it back")
        out = batch(bridge, [{"id": "x", "task": "extract", "input": {"video": {"path": DEMON}}, "options": {"tracks": [0]}}])
        truth = out[0]["outputs"][0]["path"] if out[0]["outputs"] else None
        check(bool(truth) and truth.endswith(".srt"), f"extracted English SRT -> {truth}")
        if truth:
            doc = formats.read(truth)
            print(f"  truth: {len(doc.cues)} cues, first '{doc.cues[0].plain[:40]}' at {doc.cues[0].start:.3f}s")
            # 25 fps timing + 2.5 s late: what a PAL release's subtitle looks like here.
            damaged = retime.convert_fps(doc, 24000 / 1001, 25.0).shift(2.5)
            damaged_path = os.path.join(OUT, "demon.damaged.en.srt")
            formats.write(damaged, damaged_path, "srt")

            out = batch(bridge, [
                {"id": "sync", "task": "sync", "input": {"subtitle": {"path": damaged_path}, "video": {"path": DEMON, "audioTrack": 1}},
                 "options": {"engine": "audio", "maxOffsetS": 60, "detectFramerate": True, "allowSplits": True, "splitPenalty": 7, "vad": "energy"},
                 "output": {"suffix": ".synced"}},
                {"id": "style", "task": "style", "input": {"subtitle": {"path": truth}, "video": {"path": DEMON}},
                 "options": {"preset": "netflix", "language": "en", "fix": True, "snapToShotChanges": False,
                             "cps": None, "cpl": None, "maxLines": None, "minDurationS": None, "maxDurationS": None,
                             "minGapFrames": None, "children": False, "sdh": False}},
                {"id": "fps", "task": "fps", "input": {"subtitle": {"path": truth}},
                 "options": {"from": 24000 / 1001, "to": 25, "mode": "time", "offsetMs": 0}},
                {"id": "tr", "task": "translate", "input": {"subtitle": {"path": truth}},
                 "options": {"engine": "claude", "source": "en", "target": "fr", "model": "claude-opus-5-5", "quality": "accurate",
                             "glossary": "", "context": "", "honorifics": "keep", "formality": "default", "keepSdh": True,
                             "fitReadingSpeed": True, "bilingual": False, "batchSize": 60}},
            ])
            sync, style, fps, tr = out
            print(f"  sync report: {json.dumps({k: sync['report'].get(k) for k in ('offsetMs', 'framerate', 'splits', 'confidence', 'score')})}")
            if sync["outputs"]:
                n_got, n_want, med, p90 = cue_errors(sync["outputs"][0]["path"], truth)
                print(f"  synced {n_got}/{n_want} cues; start error median {med * 1000:.0f} ms, p90 {p90 * 1000:.0f} ms")
                check(med is not None and med < 0.1, "sync puts a 25 fps, +2.5 s subtitle back within 100 ms (median)")
            else:
                check(False, f"sync produced output ({sync.get('error')})")
            print(f"  style: {style['summary']}")
            check(not style.get("error") and style["report"].get("issuesAfter", 1e9) <= style["report"].get("issuesBefore", 0),
                  "style check/fix runs and does not add issues")
            # Timed for 23.976, played at 25: the film runs faster, so every
            # time shrinks by 23.976/25.
            check(not fps.get("error") and abs(fps["report"].get("ratio", 0) - 24000 / 1001 / 25) < 1e-9, "fps 23.976 -> 25 ratio exact")
            check(bool(tr.get("error")) and "key" in tr["error"].lower(), f"translate without a key fails clearly: {tr.get('error')}")

        print("== Demon Slayer: generate Japanese subtitles for the first 4 minutes, then mux them")
        clip = os.path.join(OUT, "demon.first4min.mkv")
        if not os.path.exists(clip):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", "240", "-i", DEMON, "-map", "0:v:0", "-map", "0:a:1",
                            "-c", "copy", clip], check=True)
        out = batch(bridge, [{"id": "gen", "task": "generate", "input": {"video": {"path": clip, "audioTrack": 0}},
                              "options": {"engine": "mlx-whisper", "model": "large-v3", "language": "auto", "task": "transcribe",
                                          "vocalIsolation": False, "vad": True, "beamSize": 5, "initialPrompt": "",
                                          "suppressHallucinations": True, "stylePreset": "netflix", "sdh": False},
                              "output": {"format": "srt", "mux": True}}], workers=1)
        gen = out[0]
        print(f"  generate: {gen.get('error') or gen['summary']}")
        for cue in (gen.get("preview") or {}).get("cues", [])[:8]:
            print(f"    {cue['start']:7.2f}-{cue['end']:7.2f}  {cue['text']!r}")
        check(not gen.get("error") and gen["report"].get("language") == "ja", "generate detects Japanese and writes cues")
        check(any(o["kind"] == "video" for o in gen.get("outputs", [])), "generate + mux wrote a video with the new track")

    if HERE:
        print("== Here Now: OCR the Blu-ray PGS track with Apple Vision, and dim it for HDR")
        out = batch(bridge, [
            {"id": "ocr", "task": "ocr", "input": {"subtitle": {"path": HERE, "track": 0}},
             "options": {"engine": "vision", "language": "it", "fallbackEngine": "none", "minConfidence": 0.6,
                         "removeFurigana": False, "detectItalics": True, "keepPositions": True, "forced": "all",
                         "fixCommonErrors": True, "upscale": 2, "claudeModel": "claude-sonnet-5", "saveFlaggedImages": False},
             "output": {"format": "srt"}},
            {"id": "hdr", "task": "hdrSubs", "input": {"subtitle": {"path": HERE, "track": 0}},
             "options": {"mode": "nits", "targetNits": 203, "assumedPeakNits": 1000, "brightnessPercent": 60,
                         "color": "keep", "outputFormat": "same"}},
        ])
        ocr, hdr = out
        print(f"  ocr: {ocr.get('error') or ocr['summary']}")
        cues = (ocr.get("preview") or {}).get("cues", [])
        for cue in cues[100:110]:
            print(f"    {cue['start']:8.2f}  conf={cue.get('confidence')}  {cue['text']!r}")
        check(not ocr.get("error") and len(cues) > 500, "OCR reads the whole PGS track")
        print(f"  hdr: {hdr.get('error') or hdr['summary']}")
        check(not hdr.get("error") and any(o["path"].endswith(".sup") for o in hdr.get("outputs", [])), "HDR dimming writes a .sup")

    print("== synthetic HDR10 clip: tone-map to 1080p SDR with the LUT engine")
    hdr_clip = os.path.join(OUT, "hdr10_test.mkv")
    if not os.path.exists(hdr_clip):
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=3840x2160:d=3:r=24000/1001",
            "-pix_fmt", "yuv420p10le", "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
            "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400:hdr10-opt=1:repeat-headers=1:log-level=error",
            hdr_clip], check=True)
    out = batch(bridge, [{"id": "tm", "task": "tonemap", "input": {"video": {"path": hdr_clip}},
                          "options": {"engine": "lut", "algorithm": "bt2390", "resolution": "1080p", "sourcePeak": "auto",
                                      "targetPeak": 100, "desaturation": 0.5, "encoder": "x264", "crf": 18, "preset": "veryfast",
                                      "bitDepth": 8, "audio": "copy", "subtitles": "copy", "burnTrack": None, "container": "mkv",
                                      "dolbyVision": "auto"}}], workers=1)
    tm = out[0]
    print(f"  tonemap: {tm.get('error') or tm['summary']}")
    if tm.get("outputs"):
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=width,height,pix_fmt,color_transfer,color_primaries,color_space", "-of", "compact",
                               tm["outputs"][0]["path"]], capture_output=True, text=True).stdout.strip()
        print(f"  output: {info}")
        check("width=1920" in info and "color_transfer=bt709" in info, "tone-mapped output is 1920 wide and tagged BT.709")
    else:
        check(False, f"tonemap produced output ({tm.get('error')})")

    bridge.close()
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
