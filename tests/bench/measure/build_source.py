"""Build the measurement bench's source material from Sintel.

Writes into OUT (FLAC, 48 kHz, all on the film's own clock):
  orig51.flac      the film's English AC-3 5.1 mix, decoded from the top
  me.flac          the music-and-effects stem, moved onto the film's clock
  es_sparse.flac   Spanish voices at the film's own subtitle times (stereo, centre)
  en_dense.flac    English voices every few seconds for the whole runtime
  es_dense.flac    Spanish voices at the same starts as en_dense
  meta.json        the stem's measured lag in the film mix
"""
import json, os, random, re, subprocess, sys, tempfile
import numpy as np

FF = os.environ["FFMPEG"]
SR = 48000


def decode(path, ch, extra=()):
    cmd = [FF, "-v", "error", "-i", path, *extra, "-f", "f32le", "-ac", str(ch), "-ar", str(SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).reshape(-1, ch).copy()


def write_flac(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", str(arr.shape[1]), "-i", "-",
                    "-c:a", "flac", "-sample_fmt", "s32", path], input=arr.tobytes(), check=True)


def phat_lag(a, b, max_lag):
    """Lag (samples, fractional) at which b best matches a: a[n] ~ b[n - lag]."""
    n = len(a) + len(b)
    size = 1 << (n - 1).bit_length()
    X = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    X /= np.abs(X) + 1e-12
    c = np.fft.irfft(X, size)
    c = np.concatenate([c[-max_lag:], c[: max_lag + 1]])
    i = int(np.argmax(c))
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return i - max_lag + frac, float(c[i] / np.median(np.abs(c)))


def srt(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        out.append((start, end, " ".join(lines[2:])))
    return out


def tts(text, voice, rate, tmp):
    path = os.path.join(tmp, "line.wav")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", path, "--data-format=LEF32@48000", text], check=True)
    return decode(path, 1)[:, 0]


def lay(track, start_s, line, gain):
    i = int(round(start_s * SR))
    j = min(len(track), i + len(line))
    if j > i:
        track[i:j, 0] += line[: j - i] * gain
        track[i:j, 1] += line[: j - i] * gain


def main():
    film, stem, subs_dir, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    # The film mix, 6 channels, from the top (sample-exact, no seek).
    cmd = [FF, "-v", "error", "-i", film, "-map", "0:a:0", "-f", "f32le", "-"]
    orig51 = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 6).copy()
    film2 = decode(film, 2, ["-map", "0:a:0"])
    me = decode(stem, 2)
    n = len(orig51)
    print("film", n / SR, "stem", len(me) / SR)

    # Where the stem sits in the film's mix: correlate its side channel with the film's
    # (the dialogue cancels there) over the middle ten minutes.
    a = (film2[60 * SR: 660 * SR, 0] - film2[60 * SR: 660 * SR, 1]).astype(np.float64)
    b = (me[60 * SR: 660 * SR, 0] - me[60 * SR: 660 * SR, 1]).astype(np.float64)
    lag, ratio = phat_lag(a, b, 4800)
    print(f"stem lag in film: {lag:.3f} samples ({lag / SR * 1000:.4f} ms), PHAT ratio {ratio:.0f}")
    whole = int(round(lag))
    aligned = np.zeros((n, 2), np.float32)
    if whole >= 0:
        take = min(n - whole, len(me))
        aligned[whole: whole + take] = me[:take]
    else:
        take = min(n, len(me) + whole)
        aligned[:take] = me[-whole: -whole + take]
    lag2, _ = phat_lag(a, (aligned[60 * SR: 660 * SR, 0] - aligned[60 * SR: 660 * SR, 1]).astype(np.float64), 4800)
    print(f"residual after integer shift: {lag2:.3f} samples")
    write_flac(os.path.join(out, "orig51.flac"), orig51)
    write_flac(os.path.join(out, "me.flac"), aligned)

    rng = random.Random(7)
    es_voices = ["Eddy (Spanish (Spain))", "Flo (Spanish (Spain))", "Grandma (Spanish (Spain))",
                 "Eddy (Spanish (Mexico))", "Flo (Spanish (Mexico))", "Monica", "Paulina", "Jorge"]
    en_voices = ["Daniel (English (UK))", "Eddy (English (US))", "Flo (English (US))", "Grandma (English (UK))",
                 "Samantha", "Fred", "Albert", "Karen"]
    avail = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    es_voices = [v for v in es_voices if v.split(" (")[0] in avail and (v in avail or "(" not in v)] or ["Eddy (Spanish (Spain))"]
    en_voices = [v for v in en_voices if v.split(" (")[0] in avail and (v in avail or "(" not in v)] or ["Eddy (English (US))"]
    print("voices", es_voices, en_voices)

    es_lines = srt(os.path.join(subs_dir, "sub4.srt"))
    en_lines = srt(os.path.join(subs_dir, "sub3.srt"))
    with tempfile.TemporaryDirectory() as tmp:
        sparse = np.zeros((n, 2), np.float32)
        for k, (start, end, text) in enumerate(es_lines):
            lay(sparse, start, tts(text, es_voices[k % len(es_voices)], 175, tmp), 0.25)
        write_flac(os.path.join(out, "es_sparse.flac"), sparse)

        en = np.zeros((n, 2), np.float32)
        es = np.zeros((n, 2), np.float32)
        t, k, placed = 4.0, 0, []
        while t < n / SR - 6:
            idx = k % len(en_lines)
            rate = rng.choice([150, 165, 180, 195, 210])
            le = tts(en_lines[idx][2], rng.choice(en_voices), rate, tmp)
            ls = tts(es_lines[idx][2], rng.choice(es_voices), rng.choice([150, 165, 180, 195, 210]), tmp)
            lay(en, t, le, 0.3)
            lay(es, t, ls, 0.3)
            placed.append(t)
            t += max(len(le), len(ls)) / SR + rng.uniform(0.3, 2.0)
            k += 1
        write_flac(os.path.join(out, "en_dense.flac"), en)
        write_flac(os.path.join(out, "es_dense.flac"), es)
    json.dump({"stem_lag_samples": lag, "dense_lines": len(placed), "duration_s": n / SR},
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("dense lines", len(placed))


if __name__ == "__main__":
    main()
