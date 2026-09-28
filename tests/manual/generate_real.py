"""Real speech-recognition runs for Subsync's Generate tool (not part of CI).

Needs packs and real models, so it lives here rather than in tests/:

    AUDIOSYNC_PACK_PYTHON_ASR_MLX=~/.cache/aidub/asr/bin/python \\
    AUDIOSYNC_PACKS_DIR=/tmp/subsync-packs \\
        python/.venv/bin/python tests/manual/generate_real.py [say|korean|faster|all]

say     Japanese and French clips spoken by macOS ``say`` with known text and
        known timing (a long silence in the middle and at the end, where
        Whisper likes to invent lines): character/word error rate and the
        error of each sentence's start and end.
korean  The first 3 minutes of ~/Downloads/Video.mkv (a Korean drama without
        subtitles): language detection, speed, sample cues.
faster  The asr-faster pack, installed with uv into AUDIOSYNC_PACKS_DIR when
        missing, with the "small" model (downloaded when missing) on the
        Japanese clip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from audiosync.media import CancellationToken, ffmpeg_path  # noqa: E402
from audiosync.subs import generate, packs  # noqa: E402
from audiosync.subs.tasks import TaskContext  # noqa: E402

SR = 16000
MEDIA = os.environ.get("SUBSYNC_MEDIA", "/tmp/subsync-media")

JA = [
    "今日はとても良い天気ですね。",
    "駅までの道を教えていただけますか。",
    "私は毎朝コーヒーを飲みながら新聞を読みます。",
    "明日の会議は午後三時から始まります。",
    "この本はとても面白かったので、友達にも勧めました。",
    "東京タワーの近くに新しいレストランができました。",
]
FR = [
    "Bonjour, je m'appelle Thomas et j'habite à Paris.",
    "Le train pour Lyon part à huit heures du matin.",
    "Nous avons visité le musée du Louvre la semaine dernière.",
    "Pourriez-vous me dire où se trouve la gare, s'il vous plaît ?",
    "Il fait très beau aujourd'hui, allons nous promener au parc.",
    "Ma sœur travaille dans un hôpital depuis cinq ans.",
]
#: Silence before each sentence; the 8 s one tests hallucination in silence.
GAPS = [2.0, 1.2, 1.5, 8.0, 1.0, 1.4]
TAIL_S = 5.0


# ------------------------------------------------------------------ clips


def _decode(path: str) -> np.ndarray:
    out = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", path, "-f", "f32le",
                          "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _speech_bounds(x: np.ndarray) -> tuple:
    """First and last sample above -40 dBFS (10 ms frames)."""
    hop = SR // 100
    n = len(x) // hop
    db = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loud = np.flatnonzero(db > -40)
    return (int(loud[0]) * hop, int(loud[-1] + 1) * hop) if len(loud) else (0, len(x))


def build_clip(sentences, voice: str, name: str) -> tuple:
    """Speak each sentence with ``say``, lay them out with GAPS of silence,
    encode as AAC in .m4a, and return (path, [(start, end, text)])."""
    os.makedirs(MEDIA, exist_ok=True)
    out = os.path.join(MEDIA, f"{name}.m4a")
    truth_path = out + ".json"
    if os.path.exists(out) and os.path.exists(truth_path):
        with open(truth_path, encoding="utf-8") as handle:
            return out, [tuple(t) for t in json.load(handle)]
    pieces, truth, t = [], [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (text, gap) in enumerate(zip(sentences, GAPS)):
            aiff = os.path.join(tmp, f"{i}.aiff")
            subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
            x = _decode(aiff)
            s, e = _speech_bounds(x)
            x = x[s:e]
            pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
            t += gap
            truth.append((round(t, 3), round(t + len(x) / SR, 3), text))
            pieces.append(x)
            t += len(x) / SR
        pieces.append(np.zeros(int(TAIL_S * SR), dtype=np.float32))
        wav = os.path.join(tmp, "clip.wav")
        with wave.open(wav, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SR)
            handle.writeframes((np.clip(np.concatenate(pieces), -1, 1) * 32767).astype("<i2").tobytes())
        subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", wav, "-c:a", "aac", "-b:a", "128k", out],
                       check=True)
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(truth, handle, ensure_ascii=False)
    return out, truth


# ---------------------------------------------------------------- metrics


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch if unicodedata.category(ch)[0] in "LN" else " " for ch in text)


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(reference: str, hypothesis: str, by_chars: bool) -> float:
    ref, hyp = _norm(reference), _norm(hypothesis)
    if by_chars:
        ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
        return _edit_distance(list(ref), list(hyp)) / max(1, len(ref))
    return _edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))


def timing(truth, words) -> list:
    """(start error, end error) per sentence: first/last recognised word
    whose middle falls within the sentence (+-0.5 s)."""
    rows = []
    for start, end, _text in truth:
        inside = [w for w in words if start - 0.5 <= (w.start + w.end) / 2 <= end + 0.5]
        if inside:
            rows.append((inside[0].start - start, inside[-1].end - end))
        else:
            rows.append((None, None))
    return rows


# --------------------------------------------------------------------- runs


def _ctx() -> TaskContext:
    workdir = tempfile.mkdtemp(prefix="subsync-real-")
    state = {"stage": None, "pct": -10}

    def progress(pct: int, stage: str) -> None:
        if stage != state["stage"] or pct >= state["pct"] + 25 or pct == 100:
            print(f"    [{pct:3d}%] {stage}", flush=True)
            state.update(stage=stage, pct=pct)

    return TaskContext(token=CancellationToken(), progress=progress,
                       log=lambda m: print(f"    log: {m}", flush=True), workdir=workdir)


def run_job(path: str, options: dict, out_dir: str) -> dict:
    ctx = _ctx()
    job = {"id": "real", "task": "generate", "input": {"video": {"path": path, "audioTrack": 0}},
           "options": options, "output": {"dir": out_dir, "format": "srt", "overwrite": True}}
    try:
        result = generate.run_task(job, ctx)
    finally:
        shutil.rmtree(ctx.workdir, ignore_errors=True)
    return result.to_dict()


def check_clip(name: str, sentences, voice: str, language_hint: str, by_chars: bool, options: dict) -> dict:
    path, truth = build_clip(sentences, voice, name)
    ctx = _ctx()
    try:
        started = time.monotonic()
        words, language, info = generate.transcribe({"path": path}, options, ctx)
        elapsed = time.monotonic() - started
    finally:
        shutil.rmtree(ctx.workdir, ignore_errors=True)
    hypothesis = "".join(w.text for w in words)
    reference = " ".join(t[2] for t in truth)
    rate = error_rate(reference, hypothesis, by_chars)
    rows = timing(truth, words)
    starts = [abs(s) for s, _e in rows if s is not None]
    ends = [abs(e) for _s, e in rows if e is not None]
    in_silence = [w for w in words if not any(s - 0.5 <= (w.start + w.end) / 2 <= e + 0.5 for s, e, _ in truth)]
    print(f"  {name}: language {language} ({info.get('languageProbability')}), expected {language_hint}")
    print(f"  {'CER' if by_chars else 'WER'} {rate:.1%}   elapsed {elapsed:.1f} s for {info['durationS']:.1f} s "
          f"(RTF {elapsed / info['durationS']:.3f})   removed {info['removedHallucinations']}")
    print(f"  timing |start| mean {np.mean(starts) * 1000:.0f} ms max {np.max(starts) * 1000:.0f} ms; "
          f"|end| mean {np.mean(ends) * 1000:.0f} ms max {np.max(ends) * 1000:.0f} ms; "
          f"words in silence: {len(in_silence)}")
    print(f"  heard: {hypothesis.strip()}")
    for (s, e), (ts, te, text) in zip(rows, truth):
        print(f"    {ts:6.2f}-{te:6.2f}  start {s * 1000 if s is not None else float('nan'):+5.0f} ms  "
              f"end {e * 1000 if e is not None else float('nan'):+5.0f} ms  {text}")
    return {"name": name, "language": language, "rate": rate, "elapsed": elapsed, "info": info,
            "startMeanMs": float(np.mean(starts) * 1000), "endMeanMs": float(np.mean(ends) * 1000),
            "wordsInSilence": len(in_silence)}


def run_say(engine: str = "mlx-whisper", model: str = "large-v3") -> None:
    print(f"== say clips, {engine} {model}")
    options = {"engine": engine, "model": model, "language": "auto", "vad": True}
    check_clip("say_ja", JA, "Kyoko", "ja", True, options)
    check_clip("say_fr", FR, "Thomas", "fr", False, options)


def run_korean(model: str = "large-v3") -> None:
    source = os.path.expanduser("~/Downloads/Video.mkv")
    clip = os.path.join(MEDIA, "korean3min.mkv")
    if not os.path.exists(clip):
        os.makedirs(MEDIA, exist_ok=True)
        subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-t", "180", "-i", source,
                        "-map", "0:v:0", "-map", "0:a", "-c", "copy", clip], check=True)
    print(f"== Korean drama, first 3 minutes, mlx-whisper {model}")
    out_dir = os.path.join(MEDIA, "out")
    os.makedirs(out_dir, exist_ok=True)
    result = run_job(clip, {"engine": "mlx-whisper", "model": model, "language": "auto", "vad": True}, out_dir)
    print("  summary:", result["summary"])
    print("  report:", json.dumps(result["report"], ensure_ascii=False))
    print("  output:", result["outputs"][0]["path"])
    cues = result["preview"]["cues"]
    for cue in cues[: min(len(cues), 12)]:
        print(f"    {cue['start']:7.2f} --> {cue['end']:7.2f}  {cue['text']!r}")


def run_faster() -> None:
    print(f"== asr-faster pack in {packs.packs_dir()}")
    status = packs.pack_status("asr-faster")
    if not status["installed"]:
        started = time.monotonic()
        status = packs.install("asr-faster", lambda p, s, *a: None, lambda m: None, CancellationToken())
        print(f"  installed with uv in {time.monotonic() - started:.1f} s")
    print(f"  installed={status['installed']} sizeBytes={status['sizeBytes']} version={status['version']}")
    small = next(m for m in status["models"] if m["id"] == "small")
    if not small["installed"]:
        started = time.monotonic()
        packs.install("asr-faster", lambda p, s, *a: None, print, CancellationToken(), model="small")
        print(f"  downloaded small in {time.monotonic() - started:.1f} s")
    check_clip("say_ja", JA, "Kyoko", "ja", True, {"engine": "faster-whisper", "model": "small", "language": "auto"})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("say", "all"):
        run_say()
    if which in ("korean", "all"):
        run_korean()
    if which in ("faster", "all"):
        run_faster()
