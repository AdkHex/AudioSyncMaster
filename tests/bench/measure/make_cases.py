"""Measurement bench cases: real film audio, dubs with planted delays.

    FFMPEG=... MKVMERGE=... python make_cases.py SRC_DIR SINTEL.mkv OUT_DIR [--only name,name]

truth.json lists every case: primary, secondary, the planted delay at the start
(engine convention: positive = the dub starts later), what it is expected to
report (constant / cut / rate / drift / none), and two calibrations:

- ``decode_shift_ms``: how much later than what was encoded the written file
  decodes from the top (raw AC-3/E-AC-3 256 samples, ADTS AAC 1024), found by
  correlating the decode with the array that was encoded;
- ``latency_ms``: how far a processing step itself moved the content. A
  limiter's lookahead delays it by its attack; a time-stretch that keeps the
  pitch (atempo) moves real film content by about +11 ms -- measured on the
  film's own transients, which is where a person checking by ear would put
  it. (Isolated clicks through the same stretch move -20 ms, which is not what
  dense material does.)
"""
import json, os, subprocess, sys
import numpy as np

FF = os.environ["FFMPEG"]
SR = 48000
PAL = 25 / (24000 / 1001)
NTSC24 = 24 / (24000 / 1001)
PAL24 = 25 / 24
SLOW = (24000 / 1001) / 24


def load(path, ch):
    out = subprocess.run([FF, "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).reshape(-1, ch).copy()


def ffilter(arr, chain):
    """Run an array through an FFmpeg audio filter chain."""
    ch = arr.shape[1]
    out = subprocess.run([FF, "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", str(ch), "-i", "-",
                          "-af", chain, "-f", "f32le", "-ar", str(SR), "-ac", str(ch), "-"],
                         input=np.ascontiguousarray(arr, np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).reshape(-1, ch).copy()


LAYOUT = {1: "mono", 2: "stereo", 6: "5.1(side)"}


def encode(arr, path, args, video=None):
    ch = arr.shape[1]
    cmd = [FF, "-v", "error", "-y"]
    if video:
        cmd += ["-i", video]
    cmd += ["-f", "f32le", "-ar", str(SR), "-ch_layout", LAYOUT[ch], "-i", "-"]
    if video:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    cmd += [*args, path]
    subprocess.run(cmd, input=np.ascontiguousarray(arr, np.float32).tobytes(), check=True)


def shift(arr, delay_s):
    """Positive: the content starts later (silence in front). Negative: the head is cut."""
    k = int(round(delay_s * SR))
    if k >= 0:
        return np.concatenate([np.zeros((k, arr.shape[1]), np.float32), arr])
    return arr[-k:].copy()


def upmix51(me, voices_mono):
    """5.1(side): M&E on the fronts and (quieter, delayed) surrounds, voices in the centre."""
    n = len(me)
    out = np.zeros((n, 6), np.float32)
    out[:, 0] = me[:, 0]
    out[:, 1] = me[:, 1]
    out[:, 2] = voices_mono
    lfe = ffilter(me.mean(axis=1, keepdims=True), "lowpass=f=120")[:n, 0]
    out[: len(lfe), 3] = lfe * 0.5
    d = int(0.012 * SR)
    out[d:, 4] = me[:-d, 0] * 0.45
    out[d:, 5] = me[:-d, 1] * 0.45
    return out


def phat_lag(a, b, max_lag):
    n = len(a) + len(b)
    size = 1 << (n - 1).bit_length()
    X = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    X /= np.abs(X) + 1e-12
    c = np.fft.irfft(X, size)
    c = np.concatenate([c[-max_lag:], c[: max_lag + 1]])
    i = int(np.argmax(np.abs(c)))
    y0, y1, y2 = np.abs(c[i - 1]), np.abs(c[i]), np.abs(c[i + 1])
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return i - max_lag + frac


def decode_shift_ms(path, pristine, start_s=200.0, length_s=120.0):
    """How much later than what was encoded the file decodes from the top, in ms."""
    dec = load(path, 2)
    a = dec[int(start_s * SR): int((start_s + length_s) * SR)].mean(axis=1).astype(np.float64)
    p = pristine if pristine.ndim == 1 else pristine[:, : min(2, pristine.shape[1])].mean(axis=1)
    b = p[int(start_s * SR): int((start_s + length_s) * SR)].astype(np.float64)
    m = min(len(a), len(b))
    return phat_lag(a[:m], b[:m], int(0.3 * SR)) / SR * 1000.0


def filter_latency_ms(arr, chain, start_s=200.0, length_s=120.0):
    """How far a filter chain that keeps the speed delays its input, in ms."""
    a = arr[int(start_s * SR): int((start_s + length_s) * SR)].mean(axis=1, keepdims=True)
    b = ffilter(a, chain)[: len(a), 0].astype(np.float64)
    return phat_lag(b, a[:, 0].astype(np.float64), int(0.05 * SR)) / SR * 1000.0


_STRETCH = {}


def stretch_displacement_ms(me, ratio):
    """Where a pitch-kept time-stretch puts the film's own transients: the
    median displacement of its strongest attacks from where the speed alone
    would put them."""
    if ratio in _STRETCH:
        return _STRETCH[ratio]
    x = me[: min(len(me), 880 * SR)].mean(axis=1).astype(np.float32)
    y = ffilter(x[:, None], f"atempo={ratio:.6f}")[:, 0]
    hop = SR // 1000

    def env(a):
        k = len(a) // hop
        return np.log(np.sqrt((a[: k * hop].reshape(k, hop).astype(np.float64) ** 2).mean(1) + 1e-12))

    ex, ey = env(x), env(y)
    rise = np.maximum(0, np.diff(ex))
    picked = []
    for i in np.argsort(rise)[::-1][:5000]:
        if i < 60 or i > len(rise) - 60 or any(abs(i - p) < 400 for p in picked):
            continue
        if ex[i - 50: i - 5].max() > ex[i + 3] - 1.2:
            continue
        picked.append(i)
        if len(picked) >= 80:
            break
    moved = []
    for i in picked:
        ideal = i / ratio
        lo = int(ideal - 60)
        above = np.nonzero(ey[lo: int(ideal + 60)] > ex[i + 3] - 0.6)[0]
        if len(above):
            moved.append(lo + above[0] - ideal)
    _STRETCH[ratio] = float(np.median(moved)) if moved else 0.0
    return _STRETCH[ratio]


def jingle(seconds):
    """A studio logo that belongs to no part of the film."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(3)
    tone = 0.2 * np.sin(2 * np.pi * (220 + 180 * t) * t) * np.exp(-((t % 2.5) * 1.2))
    noise = rng.standard_normal(len(t)) * 0.02
    mono = (tone + noise).astype(np.float32)
    return np.stack([mono, mono * 0.9], axis=1)


def loopify(me, regions, seg_s=6.0):
    """Replace each region with a short segment of itself tiled: music that repeats exactly."""
    out = me.copy()
    for a, b in regions:
        seg = me[int(a * SR): int((a + seg_s) * SR)]
        reps = int(np.ceil((b - a) / seg_s))
        tiled = np.tile(seg, (reps, 1))[: int((b - a) * SR)]
        out[int(a * SR): int(a * SR) + len(tiled)] = tiled
    return out


def main():
    src, sintel, out = sys.argv[1:4]
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    os.makedirs(out, exist_ok=True)
    me = load(os.path.join(src, "me.flac"), 2)
    n = len(me)
    es_sparse = load(os.path.join(src, "es_sparse.flac"), 2)[:n]
    en_dense = load(os.path.join(src, "en_dense.flac"), 2)[:n]
    es_dense = load(os.path.join(src, "es_dense.flac"), 2)[:n]
    truth_path = os.path.join(out, "truth.json")
    truth = json.load(open(truth_path)) if os.path.exists(truth_path) else {}

    def want(name):
        return only is None or name in only

    # Primaries.
    prim_film = sintel  # the real film: H.264 + AC-3 5.1 640k in MKV
    prim_talky = os.path.join(out, "P_talky.mkv")
    if want("P_talky") and not os.path.exists(prim_talky):
        orig = upmix51(me, en_dense[:, 0] * 1.4142)
        encode(orig, prim_talky, ["-c:a", "eac3", "-b:a", "640k"], video=sintel)
    prim_talky2398 = os.path.join(out, "P_talky2398.mkv")
    if want("P_talky2398") and not os.path.exists(prim_talky2398):
        subprocess.run([os.environ["MKVMERGE"], "-q", "-o", prim_talky2398, "--default-duration", "0:24000/1001p",
                        prim_talky], check=True)
    loop_regions = [(100, 250), (430, 600), (690, 800)]
    prim_loop = os.path.join(out, "P_loop.mkv")
    me_loop = None
    if any(want(x) for x in ("P_loop", "loop_aac", "loop_ac3")):
        me_loop = loopify(me, loop_regions)
        if not os.path.exists(prim_loop):
            encode(upmix51(me_loop, en_dense[:, 0] * 1.4142), prim_loop, ["-c:a", "eac3", "-b:a", "640k"], video=sintel)
    # A one-hour feature: four passes of the talky mix, dub as raw elementary streams.
    prim_long = os.path.join(out, "P_long.mkv")
    if any(want(x) for x in ("P_long", "long_adts", "long_eac3", "long_mp3")) and not os.path.exists(prim_long):
        orig = upmix51(me, en_dense[:, 0] * 1.4142)
        encode(np.concatenate([orig] * 4), prim_long, ["-c:a", "eac3", "-b:a", "640k"])

    film_dub = me + es_sparse
    talky_dub = me + es_dense

    def remix(arr_me, voices, me_db, v_db, width):
        m = ffilter(arr_me, "highpass=f=100,lowpass=f=9000,acompressor=threshold=-24dB:ratio=4:attack=5:release=120")
        mid = (m[:, 0] + m[:, 1]) / 2
        side = (m[:, 0] - m[:, 1]) / 2 * width
        m2 = np.stack([mid + side, mid - side], axis=1) * 10 ** (me_db / 20)
        return (m2[:n] + voices[: len(m2[:n])] * 10 ** (v_db / 20)).astype(np.float32)

    cases = []

    def case(name, primary, kind, delay_ms, build, fmt, ext, **extra):
        cases.append(dict(name=name, primary=primary, kind=kind, delay_ms=delay_ms, build=build, fmt=fmt, ext=ext, **extra))

    AAC = ["-c:a", "aac", "-b:a", "192k"]
    # Film: the real English mix against the stem with Spanish lines.
    case("film_aac_mka", prim_film, "constant", 1312, lambda: shift(film_dub, 1.312), AAC, "mka")
    case("film_ac3_51_raw", prim_film, "constant", -840,
         lambda: shift(upmix51(me, es_sparse[:, 0] * 1.4142), -0.840), ["-c:a", "ac3", "-b:a", "448k", "-f", "ac3"], "ac3")
    case("film_eac3_raw", prim_film, "constant", 4321,
         lambda: shift(upmix51(me, es_sparse[:, 0] * 1.4142), 4.321), ["-c:a", "eac3", "-b:a", "256k", "-f", "eac3"], "eac3")
    case("film_adts_mono64", prim_film, "constant", 250,
         lambda: shift(film_dub.mean(axis=1, keepdims=True), 0.250), ["-c:a", "aac", "-b:a", "64k", "-f", "adts"], "aac")
    case("film_m4a_remix", prim_film, "constant", -2500,
         lambda: shift(remix(me, es_sparse, -4, 4, 0.4), -2.5), ["-c:a", "aac", "-b:a", "128k"], "m4a")
    case("film_opus", prim_film, "constant", 37, lambda: shift(film_dub, 0.037), ["-c:a", "libopus", "-b:a", "96k"], "mka")
    case("film_lrswap", prim_film, "constant", 700, lambda: shift(film_dub[:, ::-1], 0.7), AAC, "mka")
    case("film_polarity", prim_film, "constant", -120, lambda: shift(-film_dub, -0.12), AAC, "mka")
    case("film_quiet", prim_film, "constant", 3000, lambda: shift(film_dub * 0.03, 3.0), AAC, "mka")
    case("film_dialog_heavy", prim_film, "constant", 900, lambda: shift(remix(me, es_sparse, -12, 8, 1.0), 0.9), AAC, "mka")
    case("film_mp3_vbr", prim_film, "constant", -64, lambda: shift(film_dub, -0.064), ["-c:a", "libmp3lame", "-q:a", "4", "-f", "mp3"], "mp3")
    case("film_logo_intro", prim_film, "constant", 12800,
         lambda: np.concatenate([jingle(12.5), shift(film_dub, 0.3)]), AAC, "mka")
    case("film_recap_missing", prim_film, "constant", -75000, lambda: shift(film_dub, -75.0), AAC, "mka")
    case("film_censor_cut", prim_film, "cut", 1000,
         lambda: shift(np.concatenate([film_dub[: 400 * SR], film_dub[420 * SR:]]), 1.0), AAC, "mka",
         cut_at_s=400.0, after_ms=1000 - 20000)
    case("film_pal25_resample", prim_film, "rate", 500,
         lambda: ffilter(shift(film_dub, 0.5), f"asetrate={SR * PAL24:.6f},aresample={SR}"),
         ["-c:a", "ac3", "-b:a", "384k", "-f", "ac3"], "ac3", speed=PAL24)
    case("film_pal25_pitchkeep", prim_film, "rate", 500,
         lambda: ffilter(shift(film_dub, 0.5), f"atempo={PAL24:.6f}"), AAC, "mka", speed=PAL24,
         latency=lambda: stretch_displacement_ms(me, PAL24))
    case("film_ntsc_slow", prim_film, "rate", 250,
         lambda: ffilter(shift(film_dub, 0.25), f"asetrate={SR * SLOW:.6f},aresample={SR}"), AAC, "mka", speed=SLOW)
    case("talky_pal_resample", prim_talky2398, "rate", 800,
         lambda: ffilter(shift(talky_dub, 0.8), f"asetrate={SR * PAL:.6f},aresample={SR}"),
         ["-c:a", "ac3", "-b:a", "448k", "-f", "ac3"], "ac3", speed=PAL)
    case("talky_24", prim_talky2398, "rate", -300,
         lambda: ffilter(shift(talky_dub, -0.3), f"asetrate={SR * NTSC24:.6f},aresample={SR}"), AAC, "mka", speed=NTSC24)
    case("film_pal_resample", prim_film, "drift", 500,
         lambda: ffilter(shift(film_dub, 0.5), f"asetrate={SR * PAL:.6f},aresample={SR}"),
         ["-c:a", "ac3", "-b:a", "384k", "-f", "ac3"], "ac3", speed=PAL)
    case("film_pal_pitchkeep", prim_film, "drift", 500,
         lambda: ffilter(shift(film_dub, 0.5), f"atempo={PAL:.6f}"), AAC, "mka", speed=PAL,
         latency=lambda: stretch_displacement_ms(me, PAL))
    case("film_ntsc24", prim_film, "drift", 100,
         lambda: ffilter(shift(film_dub, 0.1), f"asetrate={SR * NTSC24:.6f},aresample={SR}"), AAC, "mka", speed=NTSC24)
    # Talky: dense, loud, different dialogue in both, shared M&E.
    case("talky_aac", prim_talky, "constant", 1312, lambda: shift(talky_dub, 1.312), ["-c:a", "aac", "-b:a", "160k"], "mka")
    case("talky_ac3_51_raw", prim_talky, "constant", -3333,
         lambda: shift(upmix51(me, es_dense[:, 0] * 1.4142), -3.333), ["-c:a", "ac3", "-b:a", "448k", "-f", "ac3"], "ac3")
    case("talky_mono_adts", prim_talky, "constant", 42,
         lambda: shift(talky_dub.mean(axis=1, keepdims=True), 0.042), ["-c:a", "aac", "-b:a", "64k", "-f", "adts"], "aac")
    case("talky_remix", prim_talky, "constant", 15000, lambda: shift(remix(me, es_dense, -8, 3, 0.35), 15.0),
         ["-c:a", "aac", "-b:a", "128k"], "m4a")
    case("talky_mono_remix_lowbit", prim_talky, "constant", -1777,
         lambda: shift(remix(me, es_dense, -10, 2, 0.0).mean(axis=1, keepdims=True), -1.777),
         ["-c:a", "aac", "-b:a", "48k", "-ar", "24000"], "m4a")
    case("talky_pal_pitchkeep", prim_talky2398, "rate", -1200,
         lambda: ffilter(shift(talky_dub, -1.2), f"atempo={PAL:.6f}"), ["-c:a", "eac3", "-b:a", "384k"], "mka", speed=PAL,
         latency=lambda: stretch_displacement_ms(me, PAL))
    # Music that repeats sample for sample over 40% of the runtime.
    case("loop_aac", prim_loop, "constant", 2700,
         lambda: shift(loopify(me, loop_regions) + es_dense, 2.7), AAC, "mka")
    case("loop_ac3", prim_loop, "constant", -450,
         lambda: shift(upmix51(loopify(me, loop_regions), es_dense[:, 0] * 1.4142), -0.45),
         ["-c:a", "ac3", "-b:a", "448k"], "mka")
    # Harder dubs.
    def far(arr, seconds):
        """The same bed from far away in the film: audio no search range can match."""
        k = int(seconds * SR)
        return np.concatenate([arr[k:], arr[:k]])
    case("hard_tv_rip", prim_film, "constant", 2222,
         lambda: shift(ffilter(film_dub, "aecho=0.8:0.6:60|130:0.35|0.25,lowpass=f=3800,highpass=f=150")[:n]
                       + (np.random.default_rng(9).standard_normal((n, 2)) * 0.01).astype(np.float32), 2.222),
         ["-c:a", "aac", "-b:a", "48k", "-ac", "1"], "m4a")
    case("hard_other_music", prim_talky, "constant", -5150,
         lambda: shift(me * 0.6 + far(me, 400) * 0.6 + es_dense, -5.15), AAC, "mka")
    case("hard_buried_bed", prim_talky, "constant", 640,
         lambda: shift(me * 0.1 + es_dense * 1.2, 0.64), ["-c:a", "aac", "-b:a", "96k"], "mka")
    case("hard_phase_remix", prim_film, "constant", -333,
         lambda: shift(np.stack([ffilter(film_dub[:, :1], "allpass=f=400:t=q:w=0.7,allpass=f=2500:t=q:w=0.5")[:n, 0],
                                 ffilter(film_dub[:, 1:], "allpass=f=900:t=q:w=0.6")[:n, 0]], axis=1), -0.333), AAC, "mka")
    case("hard_offcentre_voices", prim_talky, "constant", 4000,
         lambda: shift(me + np.stack([es_dense[:, 0] * 1.3, es_dense[:, 1] * 0.5], axis=1), 4.0), AAC, "mka")
    case("hard_44k", prim_film, "constant", -910,
         lambda: shift(film_dub, -0.91), ["-c:a", "aac", "-b:a", "160k", "-ar", "44100"], "m4a")
    loud = "acompressor=threshold=-30dB:ratio=8:attack=2:release=60:makeup=6,alimiter=limit=0.5"
    case("hard_loud_ott", prim_talky, "constant", 1800,
         lambda: shift(ffilter(talky_dub, loud)[:n], 1.8),
         ["-c:a", "eac3", "-b:a", "192k"], "mka", latency=lambda: filter_latency_ms(talky_dub, loud))
    case("hard_tv_edits", prim_film, "cut", 700,
         lambda: shift(np.concatenate([film_dub[: 150 * SR], film_dub[154 * SR: 380 * SR], film_dub[388 * SR: 610 * SR],
                                       film_dub[613 * SR:]]), 0.7), AAC, "mka", cut_at_s=150.0, after_ms=700 - 4000)
    case("neg_unrelated", prim_film, "none", 0,
         lambda: (me[::-1] + es_sparse).astype(np.float32), AAC, "mka")
    case("neg_unrelated_talky", prim_talky, "none", 0,
         lambda: (far(me, 444)[::-1] * 0.8 + far(es_dense, 300)).astype(np.float32), AAC, "mka")
    # One hour, dubs as bare elementary streams (no index: seeks are estimates).
    long_dub = lambda: np.concatenate([talky_dub] * 4)
    case("long_adts", prim_long, "constant", 1500, lambda: shift(long_dub(), 1.5),
         ["-c:a", "aac", "-b:a", "128k", "-f", "adts"], "aac")
    case("long_eac3", prim_long, "constant", -600,
         lambda: shift(np.concatenate([upmix51(me, es_dense[:, 0] * 1.4142)] * 4), -0.6),
         ["-c:a", "eac3", "-b:a", "384k", "-f", "eac3"], "eac3")
    case("long_mp3", prim_long, "constant", 800, lambda: shift(long_dub(), 0.8),
         ["-c:a", "libmp3lame", "-q:a", "5", "-f", "mp3"], "mp3")

    for c in cases:
        if not want(c["name"]):
            continue
        path = os.path.join(out, f"{c['name']}.{c['ext']}")
        print("building", c["name"], flush=True)
        arr = c["build"]()
        encode(arr, path, c["fmt"])
        cal_start = 30.0 if c["name"] == "film_recap_missing" else 200.0
        sh = 0.0 if c["kind"] == "none" else decode_shift_ms(path, arr, start_s=cal_start)
        entry = {k: v for k, v in c.items() if k not in ("build", "fmt", "latency")}
        latency = c["latency"]() if "latency" in c else 0.0
        entry.update(secondary=path, decode_shift_ms=round(sh, 4), latency_ms=round(latency, 3))
        truth[c["name"]] = entry
        print(f"  {c['name']}: decode shift {sh:+.3f} ms, processing latency {latency:+.3f} ms", flush=True)
        json.dump(truth, open(truth_path, "w"), indent=1)


if __name__ == "__main__":
    main()
