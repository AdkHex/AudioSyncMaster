"""Build a bench source from a film and its music-and-effects stem.

``make_cases.py`` needs a video whose file carries an original and a dub in
step with each other. A dual-audio release is the natural source, but any
film whose music-and-effects stem is published will do -- Blender's open
movies publish theirs -- and builds a dub the way a studio does: the same
stem, with lines in another language laid where the original's are.

    python tests/bench/make_source.py VIDEO STEM OUT.mkv --lines LINES.json [--original-track N]
    python tests/bench/make_source.py VIDEO STEM OUT.mkv --dense SEED

``--lines``: the film's own lines, ``[{"start", "end", "en", "hi"}]`` (its
subtitles, translated). The original is the film's own soundtrack (audio
track N); each Hindi line is laid where the English one starts, found on
the dialogue left when the stem is taken out of the film's mix.

``--dense``: a talkative film. The original is the stem with English lines
spoken over it, the dub the stem with the same lines in Hindi, from the
phrases below, a few seconds apart for the whole runtime.

OUT.mkv carries the video (360p), a:0 the original, a:1 the dub (stereo),
a:2 the dub's background alone and a:3 its voices alone (a:1 = a:2 + a:3),
so a case can cut a dub's voices apart from its music, as a TV dub's
studio does. ``lines.json`` beside it lists every line where it was laid.
The voices are macOS's (``say``): Lekha for Hindi.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np

SR = 48000
FRAME = SR // 100  # 10 ms

ENGLISH_VOICES = ["Daniel", "Samantha (English (US))", "Aman", "Flo (English (UK))", "Eddy (English (US))", "Tara"]
HINDI_VOICE = "Lekha"
# Speaking rates (words a minute) that tell Lekha's characters apart.
HINDI_RATES = [165, 185, 150, 175, 195, 160]

PHRASES: List[Tuple[str, str]] = [
    ("Where have you been all this time?", "तुम इतने समय से कहाँ थे?"),
    ("We have to leave before it gets dark.", "अंधेरा होने से पहले हमें निकलना होगा।"),
    ("I told you not to come here.", "मैंने तुमसे कहा था यहाँ मत आना।"),
    ("Listen to me, just this once.", "बस एक बार मेरी बात सुनो।"),
    ("Nobody knows the way through the mountains.", "पहाड़ों के पार का रास्ता कोई नहीं जानता।"),
    ("Are you hurt?", "क्या तुम्हें चोट लगी है?"),
    ("It was never about the money.", "बात कभी पैसों की थी ही नहीं।"),
    ("Keep your voice down, they can hear us.", "धीरे बोलो, वे हमें सुन सकते हैं।"),
    ("I remember this place.", "मुझे यह जगह याद है।"),
    ("You should have trusted me.", "तुम्हें मुझ पर भरोसा करना चाहिए था।"),
    ("How long until the storm reaches us?", "तूफ़ान हम तक कितनी देर में पहुँचेगा?"),
    ("Give me your hand.", "अपना हाथ दो।"),
    ("That is not what I asked you.", "मैंने तुमसे यह नहीं पूछा था।"),
    ("We are running out of time.", "हमारे पास समय कम है।"),
    ("Did you see where he went?", "क्या तुमने देखा वह कहाँ गया?"),
    ("Everything changed after that night.", "उस रात के बाद सब कुछ बदल गया।"),
    ("I am not afraid of you.", "मैं तुमसे नहीं डरता।"),
    ("Close the door behind you.", "अपने पीछे दरवाज़ा बंद कर दो।"),
    ("The river is too deep to cross here.", "यहाँ नदी पार करने के लिए बहुत गहरी है।"),
    ("Tell me the truth, for once.", "कम से कम एक बार सच बताओ।"),
    ("We will rest here tonight.", "आज रात हम यहीं आराम करेंगे।"),
    ("What are you looking at?", "तुम क्या देख रहे हो?"),
    ("She left without saying a word.", "वह बिना कुछ कहे चली गई।"),
    ("This is the last chance we have.", "यह हमारा आख़िरी मौका है।"),
    ("Do not look back.", "पीछे मत देखना।"),
    ("I have been waiting for you.", "मैं तुम्हारा इंतज़ार कर रहा था।"),
    ("Who sent you here?", "तुम्हें यहाँ किसने भेजा?"),
    ("The fire will keep them away.", "आग उन्हें दूर रखेगी।"),
    ("You cannot do this alone.", "तुम यह अकेले नहीं कर सकते।"),
    ("Something is wrong, I can feel it.", "कुछ गड़बड़ है, मुझे महसूस हो रहा है।"),
    ("Let him go.", "उसे जाने दो।"),
    ("We were friends once.", "हम कभी दोस्त थे।"),
    ("Why did you lie to me?", "तुमने मुझसे झूठ क्यों बोला?"),
    ("The gate opens only at dawn.", "दरवाज़ा सिर्फ़ सुबह खुलता है।"),
    ("I will find a way, I promise.", "मैं कोई रास्ता ढूँढ लूँगा, वादा करता हूँ।"),
    ("Stay close to me.", "मेरे पास रहो।"),
    ("Is anyone there?", "क्या वहाँ कोई है?"),
    ("You were right about everything.", "तुम हर बात में सही थे।"),
    ("Put the sword down, slowly.", "तलवार धीरे से नीचे रखो।"),
    ("There is no going back now.", "अब वापस जाने का कोई रास्ता नहीं है।"),
    ("Thank you for coming.", "आने के लिए शुक्रिया।"),
    ("I never wanted any of this.", "मैं यह सब कभी नहीं चाहता था।"),
    ("Wake up, we have to move.", "उठो, हमें चलना होगा।"),
    ("Where is the map?", "नक्शा कहाँ है?"),
    ("He will not survive the night.", "वह यह रात नहीं काट पाएगा।"),
    ("Look at me. Look at me!", "मेरी तरफ़ देखो। मेरी तरफ़ देखो!"),
    ("We should not have come back.", "हमें वापस नहीं आना चाहिए था।"),
    ("Can you hear that?", "क्या तुम्हें वह सुनाई दे रहा है?"),
    ("It is colder than I expected.", "यह मेरी सोच से ज़्यादा ठंडा है।"),
    ("Promise me you will come back.", "मुझसे वादा करो कि तुम लौटोगे।"),
]


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def decode(path: str, stream: str, channels: Optional[int] = None) -> np.ndarray:
    """A whole stream at SR as float32 (frames, channels)."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", stream, "-ar", str(SR)]
    if channels:
        command += ["-ac", str(channels)]
    command += ["-f", "f32le", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    if channels is None:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream.split(":", 1)[1], "-show_entries", "stream=channels",
             "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()
        channels = int(probe[0])
    return np.frombuffer(raw, np.float32).reshape(-1, channels).copy()


def speak(text: str, voice: str, rate: Optional[int], work: str) -> np.ndarray:
    """One line spoken, mono at SR, leading and trailing silence trimmed."""
    aiff = os.path.join(work, "line.aiff")
    command = ["say", "-v", voice, "-o", aiff]
    if rate:
        command += ["-r", str(rate)]
    subprocess.run(command + [text], check=True)
    pcm = decode(aiff, "0:a:0", 1)[:, 0]
    level = np.sqrt(np.convolve(pcm ** 2, np.ones(FRAME) / FRAME, mode="same"))
    loud = np.flatnonzero(level > 0.01 * level.max())
    return pcm[loud[0]:loud[-1] + 1] if loud.size else pcm


def fit(pcm: np.ndarray, seconds: float, work: str) -> np.ndarray:
    """A line brought towards ``seconds`` long, as a dubbing actor fits a
    line to the mouth: by at most a quarter either way, pitch kept."""
    natural = len(pcm) / SR
    factor = float(np.clip(natural / max(seconds, 0.3), 0.8, 1.25))
    if abs(factor - 1.0) < 0.02:
        return pcm
    raw, out = os.path.join(work, "fit.f32"), os.path.join(work, "fit.wav")
    pcm.astype(np.float32).tofile(raw)
    ffmpeg("-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", raw, "-af", f"atempo={factor:.6f}", out)
    return decode(out, "0:a:0", 1)[:, 0]


def active_rms(pcm: np.ndarray) -> float:
    frames = pcm[: len(pcm) // FRAME * FRAME].reshape(-1, FRAME)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = rms[rms > 0.1 * rms.max()] if rms.size else rms
    return float(np.sqrt((loud ** 2).mean())) if loud.size else 0.0


def phat_lag(a: np.ndarray, b: np.ndarray, reach: int) -> int:
    """Samples ``a`` lags ``b`` by (GCC-PHAT)."""
    size = 1 << (len(a) + len(b) - 1).bit_length()
    cross = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    c = np.fft.irfft(cross / (np.abs(cross) + 1e-12), size)
    c = np.concatenate([c[-reach:], c[:reach + 1]])
    return int(np.argmax(c)) - reach


def stem_lag(mix: np.ndarray, stem: np.ndarray) -> int:
    """Where the stem sits in the film's mix: a decoder's delay (AC3's 256
    samples) or an edit list. One lag for the whole film, or no source."""
    lags = []
    for start in np.linspace(30 * SR, len(mix) - 60 * SR, 7).astype(int):
        lags.append(phat_lag(mix[start:start + 20 * SR], stem[start:start + 20 * SR], SR // 10))
    lag = int(np.median(lags))
    if sum(abs(x - lag) <= 1 for x in lags) < 5:
        raise SystemExit(f"the stem does not sit at one lag in the film's mix: {lags}")
    return lag


def shift(audio: np.ndarray, samples: int, length: int) -> np.ndarray:
    """``audio`` delayed by ``samples`` (advanced when negative), ``length`` long."""
    out = np.zeros((length, audio.shape[1]), np.float32)
    if samples >= 0:
        piece = audio[: max(0, length - samples)]
        out[samples:samples + len(piece)] = piece
    else:
        piece = audio[-samples:-samples + length]
        out[: len(piece)] = piece
    return out


def onset_near(dialogue_level: np.ndarray, t: float, reach: float = 0.6) -> Optional[float]:
    """Where a line really starts near its subtitle's time ``t``: the first
    10 ms frame of the film's dialogue rising clear of its own floor."""
    lo, hi = max(0, int((t - reach) * 100)), min(len(dialogue_level), int((t + reach) * 100))
    if hi - lo < 10:
        return None
    floor = float(np.percentile(dialogue_level[max(0, lo - 300):hi], 20))
    peak = float(dialogue_level[lo:hi].max())
    if peak < 4 * floor:
        return None
    above = np.flatnonzero(dialogue_level[lo:hi] > floor + 0.25 * (peak - floor))
    return (lo + int(above[0])) / 100.0 if above.size else None


def dense_lines(duration: float, seed: int, work: str) -> List[dict]:
    """A talkative film's lines: two or three characters trading lines a
    second or few apart, with now and then a passage of music alone."""
    rng = np.random.default_rng(seed)
    lines, t = [], 4.0
    while t < duration - 8.0:
        en, hi = PHRASES[int(rng.integers(len(PHRASES)))]
        who = int(rng.integers(len(ENGLISH_VOICES)))
        pcm = speak(en, ENGLISH_VOICES[who], None, work)
        length = len(pcm) / SR
        if t + length > duration - 4.0:
            break
        lines.append({"start": round(t, 3), "end": round(t + length, 3), "en": en, "hi": hi, "who": who})
        gap = float(rng.uniform(0.5, 3.5)) if rng.random() > 0.08 else float(rng.uniform(12.0, 30.0))
        t += length + gap
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video")
    parser.add_argument("stem", help="the film's music-and-effects stem")
    parser.add_argument("out")
    parser.add_argument("--lines", help="the film's own lines, timed and translated")
    parser.add_argument("--original-track", type=int, default=0, help="audio track of VIDEO that is the original")
    parser.add_argument("--dense", type=int, metavar="SEED", help="speak a talkative script over the stem instead")
    parser.add_argument("--dub-gain-db", type=float, default=-2.0, help="the dub's background against the stem")
    args = parser.parse_args()
    if bool(args.lines) == (args.dense is not None):
        parser.error("give exactly one of --lines and --dense")

    work = tempfile.mkdtemp(prefix="asm-source-")
    stem = decode(args.stem, "0:a:0", 2)
    stem_mono = stem.mean(axis=1)

    if args.lines:
        mix = decode(args.video, f"0:a:{args.original_track}")
        mix_mono = decode(args.video, f"0:a:{args.original_track}", 2).mean(axis=1)
        length = len(mix)
        lag = stem_lag(mix_mono, stem_mono)
        print(f"the stem sits {lag:+d} samples ({1000 * lag / SR:+.3f} ms) into the film's mix", flush=True)
        stem = shift(stem, lag, length)
        stem_mono = stem.mean(axis=1)
        gain = float(np.dot(mix_mono, stem_mono) / (np.dot(stem_mono, stem_mono) + 1e-12))
        dialogue = mix_mono[:length] - gain * stem_mono[:length]
        frames = dialogue[: len(dialogue) // FRAME * FRAME].reshape(-1, FRAME)
        dialogue_level = np.sqrt((frames ** 2).mean(axis=1))
        lines = json.load(open(args.lines))
        voice_rms = active_rms(dialogue[: len(dialogue) // FRAME * FRAME]) or 0.05
        original = mix
        bed_gain = gain * 10 ** (args.dub_gain_db / 20.0)
        for line in lines:
            said = onset_near(dialogue_level, float(line["start"]))
            line["laid"] = round(said if said is not None else float(line["start"]), 3)
    else:
        length = len(stem)
        lines = dense_lines(length / SR, args.dense, work)
        bed_gain = 10 ** (args.dub_gain_db / 20.0)
        voice_rms = 2.0 * float(np.sqrt((stem_mono ** 2).mean()))
        english = np.zeros(length, np.float32)
        for line in lines:
            pcm = speak(line["en"], ENGLISH_VOICES[line["who"]], None, work)
            pcm = pcm * (voice_rms / (active_rms(pcm) + 1e-12))
            i = int(round(line["start"] * SR))
            english[i:i + len(pcm)] += pcm[: max(0, length - i)]
            line["laid"] = line["start"]
        original = stem[:length] + english[:, None]

    voices = np.zeros(length, np.float32)
    for index, line in enumerate(lines):
        who = int(line.get("who", index % len(HINDI_RATES)))
        pcm = speak(line["hi"], HINDI_VOICE, HINDI_RATES[who % len(HINDI_RATES)], work)
        pcm = fit(pcm, float(line["end"]) - float(line["start"]), work)
        pcm = pcm * (voice_rms / (active_rms(pcm) + 1e-12))
        i = int(round(float(line["laid"]) * SR))
        voices[i:i + len(pcm)] += pcm[: max(0, length - i)]
        line["hiEnd"] = round(float(line["laid"]) + len(pcm) / SR, 3)
    background = (stem[:length] * bed_gain).astype(np.float32)
    voices2 = np.repeat(voices[:, None], 2, axis=1)
    dub = background + voices2
    peak = max(float(np.abs(dub).max()), float(np.abs(original).max()), 1e-9)
    scale = min(1.0, 0.98 / peak)

    paths = []
    for name, audio in (("original", original), ("dub", dub), ("background", background), ("voices", voices2)):
        raw = os.path.join(work, f"{name}.f32")
        flac = os.path.join(work, f"{name}.flac")
        (audio * scale).astype(np.float32).tofile(raw)
        ffmpeg("-f", "f32le", "-ar", str(SR), "-ac", str(audio.shape[1]), "-i", raw, "-c:a", "flac",
               "-sample_fmt", "s32", flac)
        os.remove(raw)
        paths.append(flac)
    command = ["-i", args.video]
    for path in paths:
        command += ["-i", path]
    command += ["-map", "0:v:0"] + sum((["-map", f"{k + 1}:a:0"] for k in range(4)), [])
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-vf", "scale=-2:360", "-c:a", "copy",
                "-metadata:s:a:0", "title=original", "-metadata:s:a:0", "language=eng",
                "-metadata:s:a:1", "title=dub", "-metadata:s:a:1", "language=hin",
                "-metadata:s:a:2", "title=dub background", "-metadata:s:a:3", "title=dub voices",
                "-t", f"{length / SR:.6f}", args.out]
    ffmpeg(*command)
    json.dump(lines, open(os.path.splitext(args.out)[0] + ".lines.json", "w"), ensure_ascii=False, indent=1)
    print(f"{args.out}: {length / SR:.1f} s, {len(lines)} lines", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
