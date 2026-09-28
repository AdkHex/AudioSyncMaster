"""Dub-sync bench: dub variants with exactly known edits.

The source is a video whose file carries two audio tracks that are in sync
with each other -- an original and a dub from the same release, e.g. a
dual-audio Blu-ray. Every case keeps the video and its original track and
builds a new dub from the other track by cutting, inserting, reordering,
replacing, silencing, speeding up and re-encoding it, recording as it goes
exactly which moment of the dub belongs at every moment of the video. That
record is the ground truth ``score.py`` holds a plan against.

    python tests/bench/make_cases.py SOURCE.mkv OUT_DIR --foreign OTHER.mkv [--only NAME ...]

SOURCE.mkv: video, original audio (audio track 0) and the in-sync dub
(audio track 1). OTHER.mkv: anything else with a dub-language track 1, for
material that does not belong to this video (recaps, other scenes).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from audiosync.media import audio_lead_s, probe, stream_audio  # noqa: E402
from audiosync.shots import detect_cuts, scene_scores  # noqa: E402

SR = 48000
CHECK_SR = 16000
PAL = 25.0 / (24000.0 / 1001.0)  # a 23.976 master played at 25 fps
NTSC = 1001.0 / 1000.0            # a dub from a 24.000 master against a 23.976 video
NTSC_SLOW = 1000.0 / 1001.0       # a dub from a 23.976 master against a 24.000 video

CODECS = {
    "ac3": (["-c:a", "ac3", "-b:a", "384k"], "ac3"),
    "eac3": (["-c:a", "eac3", "-b:a", "224k"], "eac3"),
    "aac": (["-c:a", "aac", "-b:a", "192k"], "m4a"),
    "aac_low": (["-c:a", "aac", "-b:a", "48k"], "m4a"),
    "flac": (["-c:a", "flac"], "flac"),
    "mp3": (["-c:a", "libmp3lame", "-b:a", "192k"], "mp3"),
}


@dataclass
class Case:
    name: str
    description: str
    ops: List[tuple]
    """("src", a, b): the dub's own material for video [a, b);
    ("foreign", a, b): material from elsewhere (no place in this video);
    ("silence", seconds)."""
    speed: float = 1.0
    speed_mode: str = "pitch"  # "pitch": resampled, like a PAL transfer; "tempo": pitch kept
    codec: str = "ac3"
    channels: int = 2
    audio_filter: str = ""
    kinds: List[str] = field(default_factory=list)


def decode(path: str, track: int, channels: int, sr: int = SR) -> np.ndarray:
    """A whole track on its file's clock, the way the engine reads it."""
    info = probe(path)
    blocks = list(stream_audio(path, sr, track=track, channels=channels, block_s=30.0,
                               lead_s=audio_lead_s(info, track)))
    audio = np.concatenate(blocks) if blocks else np.zeros((0, channels), np.float32)
    return audio.reshape(-1, channels)


def source_offset(path: str, cache: str) -> float:
    """How far the source's own dub track lags its original, in seconds.

    Two tracks of one release are not always laid to the sample: a Blu-ray
    measured here carries its English 62.6 ms behind its Japanese. The
    ground truth is the dub's own relation to the picture, so that lag is
    measured -- waveform against waveform at 48 kHz, window by window --
    and kept in every case.
    """
    if os.path.exists(cache):
        return float(json.load(open(cache))["offsetS"])
    a = decode(path, 0, 1)[:, 0]
    b = decode(path, 1, 1)[:, 0]
    width, reach = 10 * SR, SR // 4
    lags = []
    for start in range(SR * 10, len(a) - width - SR, SR * 20):
        t = a[start:start + width].astype(np.float64)
        s = b[start - reach:start + width + reach].astype(np.float64)
        c = t - t.mean()
        size = 1 << int(np.ceil(np.log2(len(s) + width)))
        num = np.fft.irfft(np.fft.rfft(s, size) * np.conj(np.fft.rfft(c, size)), size)[: 2 * reach + 1]
        o = np.concatenate([[0.0], np.cumsum(s)])
        q = np.concatenate([[0.0], np.cumsum(s * s)])
        n = 2 * reach + 1
        var = np.maximum((q[width:width + n] - q[:n]) - (o[width:width + n] - o[:n]) ** 2 / width, 1e-12)
        r = num / (np.sqrt(var) * np.sqrt((c * c).sum()) + 1e-12)
        k = int(np.argmax(r))
        if r[k] < 0.3 or not 0 < k < n - 1:
            continue
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        den = y0 - 2 * y1 + y2
        lags.append((k - reach + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)) / SR)
    if len(lags) < 5:
        raise SystemExit("the source's two tracks do not correlate: not a usable bench source")
    offset = float(np.median(lags))
    spread = float(np.percentile(np.abs(np.array(lags) - offset), 90))
    if spread > 0.002:
        raise SystemExit(f"the source's two tracks are not in step (spread {spread * 1000:.1f} ms)")
    json.dump({"offsetS": offset, "windows": len(lags), "spreadS": spread}, open(cache, "w"))
    return offset


def shot_cuts(path: str, duration: float, cache: str) -> List[float]:
    if os.path.exists(cache):
        return json.load(open(cache))
    times, scores = scene_scores(path, 0.0, duration)
    cuts = detect_cuts(times, scores)
    json.dump(cuts, open(cache, "w"))
    return cuts


# -- scenarios ---------------------------------------------------------------


def _pick_cuts(rng, shots: List[float], count: int, lo: float, hi: float, duration: float,
               at_shots: bool = True, margin: float = 20.0) -> List[Tuple[float, float]]:
    """Non-overlapping spans [a, b) to remove or replace, lengths in [lo, hi],
    starting (and, where one lies close enough, ending) on shot changes."""
    spans: List[Tuple[float, float]] = []
    pool = [s for s in shots if margin < s < duration - margin] if at_shots else []
    for _ in range(count * 50):
        if len(spans) >= count:
            break
        length = float(rng.uniform(lo, hi))
        if pool:
            a = float(rng.choice(pool))
            ends = [s for s in pool if lo <= s - a <= hi]
            b = float(rng.choice(ends)) if ends and rng.random() < 0.7 else a + length
        else:
            a = float(rng.uniform(margin, duration - margin - length))
            b = a + length
        if b > duration - margin:
            continue
        if any(not (b + 3.0 < x or a > y + 3.0) for x, y in spans):
            continue
        spans.append((a, b))
    return sorted(spans)


def _ops_without(spans: List[Tuple[float, float]], duration: float,
                 replace: Optional[Callable[[int, float, float], List[tuple]]] = None) -> List[tuple]:
    ops: List[tuple] = []
    cursor = 0.0
    for index, (a, b) in enumerate(spans):
        if a > cursor:
            ops.append(("src", cursor, a))
        if replace:
            ops.extend(replace(index, a, b))
        cursor = b
    if cursor < duration:
        ops.append(("src", cursor, duration))
    return ops


def _inserts(rng, shots: List[float], duration: float, count: int, lo: float, hi: float) -> List[tuple]:
    points = sorted(float(x) for x in rng.choice([s for s in shots if 20 < s < duration - 20], count, replace=False))
    ops: List[tuple] = []
    cursor = 0.0
    for p in points:
        ops.append(("src", cursor, p))
        start = float(rng.uniform(0, 250))
        ops.append(("foreign", start, start + float(rng.uniform(lo, hi))))
        cursor = p
    ops.append(("src", cursor, duration))
    return ops


def scenarios(duration: float, shots: List[float], seed: int) -> List[Case]:
    rng = np.random.default_rng(seed)
    d = duration
    whole = [("src", 0.0, d)]
    out: List[Case] = []

    def add(name, desc, ops, kinds, **kw):
        out.append(Case(name, desc, ops, kinds=kinds, **kw))

    add("clean", "same edit, re-encoded", whole, ["delay-only"])
    add("delay_late", "dub carries 4.37 s of other material first", [("foreign", 40.0, 44.37)] + whole,
        ["delay-only"])
    add("delay_early", "dub lacks the first 7.2 s", [("src", 7.2, d)], ["delay-only"])
    add("delay_ntsc", "24 vs 23.976 fps and a 2.5 s delay, nothing cut", [("silence", 2.5)] + whole,
        ["delay-only", "speed"], speed=NTSC, codec="aac")
    add("head_tail", "first 30 s missing, 20 s of something else at the end",
        [("src", 30.0, d), ("foreign", 100.0, 120.0)], ["edges"])
    add("cuts_few", "4 scenes cut (3-40 s) at shot changes",
        _ops_without(_pick_cuts(rng, shots, 4, 3, 40, d), d), ["cuts"])
    add("cuts_many", "30 short cuts (0.2-4 s) at shot changes, a TV edit",
        _ops_without(_pick_cuts(rng, shots, 30, 0.2, 4, d, margin=8.0), d), ["cuts", "many"])
    add("cuts_midshot", "10 cuts (0.5-10 s) anywhere, not on shot changes",
        _ops_without(_pick_cuts(rng, shots, 10, 0.5, 10, d, at_shots=False), d), ["cuts"])
    add("frame_trims", "25 trims of 1-3 frames at shot changes",
        _ops_without(_pick_cuts(rng, shots, 25, 0.04, 0.13, d, margin=8.0), d), ["cuts", "frames"])
    add("inserts", "4 pieces of other material (5-60 s) inserted at shot changes",
        _inserts(rng, shots, d, 4, 5, 60), ["inserts"])
    add("recap", "90 s recap before the episode and 45 s of other material mid-way",
        [("foreign", 300.0, 390.0), ("src", 0.0, d / 2), ("foreign", 10.0, 55.0), ("src", d / 2, d)],
        ["inserts", "edges"])
    a1 = float(rng.uniform(120, d / 2 - 100)); l1 = float(rng.uniform(40, 90))
    a2 = float(rng.uniform(d / 2, d - 150)); l2 = float(rng.uniform(40, 90))
    add("reorder", "two scenes swapped",
        [("src", 0, a1), ("src", a2, a2 + l2), ("src", a1 + l1, a2), ("src", a1, a1 + l1), ("src", a2 + l2, d)],
        ["reorder"])
    add("replace", "a 60 s scene replaced by another of the same length",
        _ops_without(_pick_cuts(rng, shots, 1, 55, 65, d), d, lambda i, a, b: [("foreign", 200.0, 200.0 + (b - a))]),
        ["replace"])
    add("dropouts", "3 dropouts to digital silence (3-15 s)",
        _ops_without(_pick_cuts(rng, shots, 3, 3, 15, d, at_shots=False), d, lambda i, a, b: [("silence", b - a)]),
        ["dropouts"])
    add("pal_pitch", "PAL speed-up (resampled) and 3 cuts",
        _ops_without(_pick_cuts(rng, shots, 3, 2, 30, d), d), ["speed", "cuts"], speed=PAL)
    add("pal_tempo", "PAL speed-up with the pitch kept and 3 cuts",
        _ops_without(_pick_cuts(rng, shots, 3, 2, 30, d), d), ["speed", "cuts"], speed=PAL, speed_mode="tempo")
    ops = _ops_without(_pick_cuts(rng, shots, 6, 1, 25, d), d)
    ops = ops[:2] + [("foreign", 150.0, 162.5)] + ops[2:5] + [("foreign", 420.0, 447.0)] + ops[5:]
    add("ntsc_edit", "dub at 23.976 against a 24 fps video (longer), 6 cuts and 2 inserts", ops,
        ["speed", "cuts", "inserts"], speed=NTSC_SLOW, codec="aac")
    add("drift", "slow clock drift (0.3 per mille), nothing cut", whole, ["speed"], speed=1.0003)
    add("mix_diff", "different mix (low-passed, compressed, quieter) and 5 cuts",
        _ops_without(_pick_cuts(rng, shots, 5, 2, 20, d), d), ["mix", "cuts"],
        audio_filter="lowpass=f=6000,acompressor=threshold=-24dB:ratio=6,volume=-8dB")
    add("low_mono", "48 kb/s AAC mono and 5 cuts",
        _ops_without(_pick_cuts(rng, shots, 5, 2, 20, d), d), ["codec", "cuts"], codec="aac_low", channels=1)
    add("surround", "EAC3 5.1 and 5 cuts",
        _ops_without(_pick_cuts(rng, shots, 5, 2, 20, d), d), ["codec", "cuts"], codec="eac3", channels=6)
    spans = _pick_cuts(rng, shots, 20, 0.3, 12, d, margin=10.0)
    ops = _ops_without(spans, d)
    k = len(ops) // 2
    ops = ops[:k] + [("foreign", 500.0, 531.0)] + ops[k:]
    add("combo", "24 vs 23.976 fps, 20 cuts, an insert, as MP3", ops, ["speed", "cuts", "inserts", "many"],
        speed=NTSC, codec="mp3")
    return out


# -- building ----------------------------------------------------------------


def build(case: Case, dub: np.ndarray, foreign: np.ndarray, video: str, out_dir: str, lag: float = 0.0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    channels = case.channels
    parts: List[np.ndarray] = []
    truth: List[dict] = []
    position = 0.0  # seconds into the dub, before any speed change

    def take(source: np.ndarray, a: float, b: float) -> np.ndarray:
        i, j = int(round(a * SR)), int(round(b * SR))
        piece = source[i:j]
        if len(piece) < j - i:
            piece = np.concatenate([piece, np.zeros((j - i - len(piece), source.shape[1]), np.float32)])
        return piece

    for op in case.ops:
        if op[0] == "src":
            piece = take(dub, op[1], op[2])
            # The dub's own track at time x carries the picture's x - lag.
            truth.append({"videoStart": op[1] - lag, "videoEnd": op[2] - lag, "dubStart": position})
        elif op[0] == "foreign":
            piece = take(foreign, op[1], op[2])
        else:
            piece = np.zeros((int(round(op[1] * SR)), dub.shape[1]), np.float32)
        parts.append(piece)
        position += len(piece) / SR
    audio = np.concatenate(parts).astype(np.float32)
    if channels == 1:
        audio = audio.mean(axis=1, keepdims=True)
    raw = os.path.join(out_dir, "dub.f32")
    audio.tofile(raw)

    args, ext = CODECS[case.codec]
    target = os.path.join(out_dir, f"dub.{ext}")
    filters = []
    speed = case.speed
    rate_in = SR
    if abs(case.speed - 1.0) > 1e-9:
        if case.speed_mode == "tempo":
            filters.append(f"atempo={case.speed:.9f}")
        else:
            # Read at a faster (slower) rate and resampled back: a speed-up
            # (slow-down) with the pitch, like a PAL transfer. The raw
            # reader takes a whole number of samples a second, so the speed
            # is what that makes it.
            rate_in = int(round(SR * case.speed))
            speed = rate_in / SR
            filters.append(f"aresample={SR}")
    if case.audio_filter:
        filters.append(case.audio_filter)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate_in),
               "-ac", str(audio.shape[1]), "-i", raw]
    if filters:
        command += ["-af", ",".join(filters)]
    command += ["-ar", str(SR)] + args + [target]
    subprocess.run(command, check=True)
    os.remove(raw)

    # Where the written file really starts against what was meant: codec
    # priming, a resampler's latency. Measured, not assumed.
    delay = _measured_delay(target, audio, speed)
    for piece in truth:
        piece["dubStart"] = piece["dubStart"] / speed + delay
    info = {
        "name": case.name,
        "description": case.description,
        "kinds": case.kinds,
        "video": os.path.abspath(video),
        "videoTrack": 0,
        "dub": os.path.abspath(target),
        "dubTrack": 0,
        "speed": speed,
        "measuredDelayS": delay,
        "sourceLagS": lag,
        "truth": truth,
    }
    json.dump(info, open(os.path.join(out_dir, "case.json"), "w"), indent=1)
    return info



def _measured_delay(path: str, meant: np.ndarray, speed: float) -> float:
    """Seconds the written file lags what was meant, over its first minute."""
    got = decode(path, 0, 1, CHECK_SR)[:, 0][: CHECK_SR * 70]
    want = meant.mean(axis=1)
    n = int(len(want) / speed / (SR / CHECK_SR))
    want = np.interp(np.arange(n) * speed * (SR / CHECK_SR), np.arange(len(want)), want)[: CHECK_SR * 70]
    a, b = want[CHECK_SR * 5: CHECK_SR * 65], got
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a - a.mean(), size)), size)
    reach = CHECK_SR  # +-1 s
    lags = np.arange(-reach, reach + 1)
    values = corr[(CHECK_SR * 5 + lags) % size]
    k = int(np.argmax(values))
    lag = lags[k]
    if 0 < k < len(values) - 1:
        y0, y1, y2 = values[k - 1], values[k], values[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            lag = lag + 0.5 * (y0 - y2) / den
    return float(lag) / CHECK_SR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--foreign", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    duration = float(probe(args.source).duration or 0.0)
    shots = shot_cuts(args.source, duration, os.path.join(args.out, "shots.json"))
    print(f"{len(shots)} shot changes in {duration:.0f} s", flush=True)
    lag = source_offset(args.source, os.path.join(args.out, "source_offset.json"))
    print(f"the source's dub track lags its original by {lag * 1000:.3f} ms", flush=True)
    stereo = decode(args.source, 1, 2)
    foreign = decode(args.foreign, 1, 2)
    surround: Optional[np.ndarray] = None
    for case in scenarios(duration, shots, args.seed):
        if args.only and case.name not in args.only:
            continue
        source, other = stereo, foreign
        if case.channels > 2:
            surround = decode(args.source, 1, case.channels) if surround is None else surround
            source, other = surround, np.repeat(foreign[:, :1], case.channels, axis=1)
        info = build(case, source, other, args.source, os.path.join(args.out, case.name), lag)
        print(f"{case.name:14s} {case.description}  (measured start {1000 * info['measuredDelayS']:+.2f} ms)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
