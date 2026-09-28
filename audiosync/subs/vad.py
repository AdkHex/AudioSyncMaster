"""Speech activity: where in a film someone is talking.

Subtitle sync by audio compares two on/off signals -- "a cue is showing" and
"someone is speaking" -- so what matters here is not a perfect speech detector
but one whose runs start where the talking starts, whatever sits underneath
it. Films and series rarely give dialogue a quiet room: an anime scene runs its
score under every line, a drama pads its silences with rain and room tone. A
detector that fires on loudness alone reports the music, and every cue then
correlates with the soundtrack instead of the dialogue.

So each 10 ms frame is judged on what separates a voice from a bed:

*   **syllabic modulation** of the speech band (300-3400 Hz): speech energy
    rises and falls four or five times a second, one syllable at a time,
    and drops away between words; a pad, a string section or wind is steady
    at that rate. Only movement above
    the local noise floor counts -- a running low percentile, so a music bed
    or an air conditioner becomes the floor within seconds.
*   **voicing** (cepstral peak prominence): vowels are harmonic with a pitch in
    the voice's range; explosions, rain and applause are not.
*   **spectral balance**: a voice puts most of its energy in the speech
    band; bass-heavy score and hissy effects put it elsewhere.

These are combined by a logistic model whose weights were fitted on real
films against their own subtitles (an anime with a constant score under the
dialogue, and a live-action drama), then smoothed with hysteresis and a
hangover so a run does not break at every stop consonant. Spectral flatness
and flux were measured too and left out: on those films each alone scored a
frame AUC of 0.40-0.68, and adding them did not improve the fitted model.
Loudness itself is not an input -- a score is loud too -- but the floor and
the level place each run's edges (``snap_edges``).

The audio is decoded once, from the top, in blocks (``media.stream_audio``):
a two-hour film at 16 kHz is half a gigabyte of samples, but only three
numbers per 10 ms frame -- under 10 MB -- are kept. Measured on real films:
two hours of AAC or E-AC-3 in 16-20 s, the whole process peaking near 200 MB.

The ``silero`` engine hands the same question to the Silero VAD network,
which ships in the faster-whisper speech recognition pack.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..media import CancellationToken, MediaError, probe, stream_audio

SR = 16000
HOP_S = 0.01
HOP = int(SR * HOP_S)  # 160 samples
WIN = 400  # 25 ms analysis window
NFFT = 512
_BIN_HZ = SR / NFFT  # 31.25 Hz

# Speech band. Telephone-band speech is fully intelligible, and the range
# leaves out the sub-bass of score and effects and the hiss of ambience.
SPEECH_LO_HZ, SPEECH_HI_HZ = 300.0, 3400.0
FULL_LO_HZ = 60.0
_SP_LO = int(round(SPEECH_LO_HZ / _BIN_HZ))
_SP_HI = int(round(SPEECH_HI_HZ / _BIN_HZ)) + 1
_FULL_LO = int(round(FULL_LO_HZ / _BIN_HZ))

# Pitch range for the voicing measure, as cepstral quefrency in samples:
# 70 Hz (a low male voice) to 400 Hz (a child, a shout).
_Q_LO = int(SR / 400.0)
_Q_HI = int(SR / 70.0)

# The adaptive floor: the quietest stretch of each half second, then a low
# percentile of those over half a minute either side. Long enough that a
# scene of continuous dialogue does not pull the floor up to the dialogue,
# short enough that a music cue becomes the floor within its first bars.
FLOOR_BLOCK_S = 0.5
FLOOR_SPAN_S = 15.0
FLOOR_BLOCK_Q = 20.0
FLOOR_SPAN_Q = 10.0
FLOOR_GATE_DB = 3.0

# Decoding block. Larger blocks mean fewer numpy calls; the memory held is
# the block plus the per-frame features.
BLOCK_S = 30.0

# Hysteresis: a run is a stretch above OFF that reaches ON somewhere, and
# runs closer than HANGOVER_S are one run. Stop consonants and the pauses
# between words are shorter than the hangover, so one sentence is one run.
ON_THRESHOLD = 0.6
OFF_THRESHOLD = 0.4
HANGOVER_S = 0.2
# Shorter runs are clicks, a door, a single drum hit.
MIN_RUN_S = 0.12


@dataclass
class Features:
    """Per-frame measurements of a whole track, 10 ms apart."""

    speech_db: np.ndarray
    """Energy in the speech band, dB."""
    full_db: np.ndarray
    """Energy from 60 Hz up, dB."""
    cpp: np.ndarray
    """Cepstral peak prominence in the voice's pitch range: voicing."""

    def __len__(self) -> int:
        return len(self.speech_db)


class _FeatureStream:
    """Turns blocks of samples into per-frame features, carrying the overlap.

    Frames straddle block boundaries, so the last ``WIN - HOP`` samples of a
    block are kept and prepended to the next; the frame grid therefore stays
    exactly on multiples of HOP from the first sample of the file.
    """

    def __init__(self) -> None:
        self._window = np.hanning(WIN).astype(np.float32)
        self._parts: Dict[str, List[np.ndarray]] = {k: [] for k in ("speech_db", "full_db", "cpp")}
        # The first frame is centred on t=0: half a window of silence ahead
        # of the audio puts frame k's centre at k * HOP_S.
        self._tail = np.zeros(WIN // 2, dtype=np.float32)

    def push(self, block: np.ndarray) -> None:
        buf = np.concatenate([self._tail, np.asarray(block, dtype=np.float32)])
        if len(buf) < WIN:
            self._tail = buf
            return
        n = 1 + (len(buf) - WIN) // HOP
        frames = np.lib.stride_tricks.sliding_window_view(buf, WIN)[: n * HOP : HOP]
        self._tail = buf[n * HOP:]
        self._measure(frames)

    def finish(self) -> Features:
        if len(self._tail) > WIN // 2:
            # Pad the end so the last partial frame is measured too.
            self.push(np.zeros(WIN, dtype=np.float32))
        return Features(**{k: (np.concatenate(v) if v else np.zeros(0, np.float32)) for k, v in self._parts.items()})

    def _measure(self, frames: np.ndarray) -> None:
        spec = np.fft.rfft(frames * self._window, n=NFFT, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        # A floor far below any real recording keeps digital silence finite.
        power += 1e-10
        e_speech = power[:, _SP_LO:_SP_HI].sum(axis=1)
        e_full = power[:, _FULL_LO:].sum(axis=1)

        # Cepstrum of the whole log spectrum; the peak in the pitch range,
        # measured against its own neighbourhood, is how harmonic the frame is.
        cep = np.fft.irfft(np.log(power), n=NFFT, axis=1)[:, _Q_LO:_Q_HI + 1]
        cpp = cep.max(axis=1) - np.median(cep, axis=1)

        self._parts["speech_db"].append((10.0 * np.log10(e_speech)).astype(np.float32))
        self._parts["full_db"].append((10.0 * np.log10(e_full)).astype(np.float32))
        self._parts["cpp"].append(cpp.astype(np.float32))


def features_from_samples(samples: np.ndarray) -> Features:
    """Features of mono 16 kHz samples already in memory (tests, short clips)."""
    stream = _FeatureStream()
    step = int(BLOCK_S * SR)
    for i in range(0, len(samples), step):
        stream.push(samples[i:i + step])
    return stream.finish()


def stream_features(
    path: str,
    audio_track: int = 0,
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> Features:
    """Decode one audio stream from the top and measure it frame by frame."""
    duration = None
    try:
        duration = probe(path, token).duration
    except MediaError:
        duration = None
    stream = _FeatureStream()
    done = 0
    last = -1
    for block in stream_audio(path, SR, track=audio_track, token=token, block_s=BLOCK_S):
        stream.push(block)
        done += len(block)
        if progress and duration:
            percent = min(99, int(100 * done / (duration * SR)))
            if percent != last:
                progress(percent)
                last = percent
    features = stream.finish()
    if progress:
        progress(100)
    return features


# ------------------------------------------------------------ the decision


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) == 0:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(width, dtype=np.float64) / width
    pad = width // 2
    padded = np.pad(x.astype(np.float64), (pad, width - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _running_floor(level_db: np.ndarray) -> np.ndarray:
    """Local noise floor of a dB curve: see FLOOR_* above."""
    n = len(level_db)
    block = max(1, int(FLOOR_BLOCK_S / HOP_S))
    count = max(1, -(-n // block))
    padded = np.pad(level_db, (0, count * block - n), mode="edge").reshape(count, block)
    lows = np.percentile(padded, FLOOR_BLOCK_Q, axis=1)
    span = max(1, int(FLOOR_SPAN_S / FLOOR_BLOCK_S))
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(lows, span, mode="edge"), 2 * span + 1)
    floor_blocks = np.percentile(windows, FLOOR_SPAN_Q, axis=1)
    centres = (np.arange(count) + 0.5) * block
    return np.interp(np.arange(n), centres, floor_blocks).astype(np.float32)


def feature_matrix(features: Features) -> np.ndarray:
    """The model's inputs, one row per frame (columns: FEATURE_NAMES).

    Every column is in dB differences or ratios, so none depends on how loud
    the film was mixed.
    """
    level = features.speech_db
    if len(level) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    # Syllabic modulation: the speech-band level band-passed to roughly
    # 2-10 Hz (a 30 ms smoothing minus a 310 ms one), then its RMS over half a
    # second and over a second. On real films this one measure separates
    # dialogue from score better than all the others together. The level is
    # held up at 3 dB over the local floor first, so the flicker of hiss and
    # room tone below it does not count as syllables.
    gated = np.maximum(level, _running_floor(level) + FLOOR_GATE_DB)
    band = _moving_average(gated, 3) - _moving_average(gated, 31)
    power = band * band
    modulation = np.sqrt(_moving_average(power, 50))
    modulation_long = np.sqrt(_moving_average(power, 100))

    # Share of the last second spent well below its own average: the gaps
    # between syllables and words, which sustained music does not have.
    dips = _moving_average((level < _moving_average(level, 100) - 6.0).astype(np.float32), 100)

    # How much the voicing strength moves: a voice's harmonics come and go
    # with every syllable, a held note's stay put.
    cpp_mean = _moving_average(features.cpp, 30)
    cpp_spread = np.sqrt(np.maximum(_moving_average(features.cpp * features.cpp, 30) - cpp_mean * cpp_mean, 0.0))
    voiced = _moving_average((features.cpp > VOICED_CPP).astype(np.float32), 100)

    balance = _moving_average(features.speech_db - features.full_db, 50)

    columns = [
        np.clip(modulation, 0.0, 20.0),
        np.clip(modulation_long, 0.0, 20.0),
        dips,
        cpp_spread * 5.0,
        voiced,
        np.clip(balance, -30.0, 0.0) / 5.0,
    ]
    return np.stack(columns, axis=1).astype(np.float32)


FEATURE_NAMES = ("modulation", "modulation_long", "dips", "cpp_spread", "voiced", "balance")

# A frame whose cepstral peak stands this far out counts as voiced.
VOICED_CPP = 0.25

# Logistic weights for FEATURE_NAMES, and the bias. Fitted by logistic
# regression on 45 minutes each of two real films, every 10 ms frame labelled
# by whether a cue of the film's own subtitle was showing: an anime whose
# score runs under nearly every line (Japanese audio, English subtitles) and a
# live-action drama (Italian audio, Romanian subtitles). Trained on either one
# and tested on the other, frame-level AUC is 0.80-0.87 -- against labels that
# themselves run past the speech, since a cue stays up after its line ends.
WEIGHTS = np.array([0.065, 0.512, 0.916, 1.753, 0.089, 0.273], dtype=np.float32)
BIAS = -2.084


def speech_probability(features: Features) -> np.ndarray:
    """Per-frame probability of speech before smoothing."""
    x = feature_matrix(features)
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    z = x @ WEIGHTS + BIAS
    return (1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))).astype(np.float32)


def hysteresis(probs: np.ndarray, hop_s: float = HOP_S) -> np.ndarray:
    """Speech runs (bool per frame) from probabilities.

    A run is a stretch above OFF_THRESHOLD that reaches ON_THRESHOLD
    somewhere; runs separated by less than HANGOVER_S are joined, and runs
    shorter than MIN_RUN_S dropped. The hangover only bridges dips inside a
    sentence: it never extends a run past its last frame above OFF. Extending
    ends but not starts would shift every run's centre later, and the sync
    engines would read that as an offset.
    """
    above_off = probs >= OFF_THRESHOLD
    out = np.zeros(len(probs), dtype=bool)
    if not above_off.any():
        return out
    above_on = probs >= ON_THRESHOLD
    # Hits of ON per run of OFF, from a cumulative count.
    hits = np.concatenate([[0], np.cumsum(above_on)])
    runs = [(s, e) for s, e in _runs(above_off) if hits[e] > hits[s]]
    hang = int(round(HANGOVER_S / hop_s))
    joined: List[List[int]] = []
    for start, end in runs:
        if joined and start - joined[-1][1] <= hang:
            joined[-1][1] = end
        else:
            joined.append([start, end])
    min_run = max(1, int(round(MIN_RUN_S / hop_s)))
    for start, end in joined:
        if end - start >= min_run:
            out[start:end] = True
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index pairs of the True stretches."""
    if len(mask) == 0:
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def snap_edges(decided: np.ndarray, level_db: np.ndarray, hop_s: float = HOP_S) -> np.ndarray:
    """Move each run's edges to where the speech-band level actually rises
    and falls.

    The model looks at half a second to a second around every frame, which
    is what lets it tell a voice from a score -- and also what blurs its
    edges: it starts calling speech a few hundred milliseconds before the
    first syllable, because the silence before a line is itself a dip that
    speech has. Measured against the cue starts of two real films, the
    blurred runs began 100-400 ms early, and a sync against them came out
    early by the same amount. The level at 30 ms resolution has no such
    blur: within reach of each edge, the run starts where the level first
    climbs half-way (in dB) from the local floor to the run's own speech
    level and stays there, and ends where it last does.
    """
    out = np.zeros_like(decided)
    n = len(decided)
    if n == 0 or not decided.any():
        return out
    level = _moving_average(level_db, 3)
    floor = _running_floor(level_db)
    reach = int(round(SNAP_REACH_S / hop_s))
    hold = max(1, int(round(SNAP_HOLD_S / hop_s)))
    for start, end in _runs(decided):
        peak = float(np.percentile(level[start:end], 90))
        base = float(floor[min(n - 1, (start + end) // 2)])
        if peak - base < SNAP_MIN_RANGE_DB:
            out[start:end] = True
            continue
        threshold = base + 0.5 * (peak - base)
        above = level >= threshold
        lo, hi = max(0, start - reach), min(n, end + reach)
        # Held above for `hold` frames: one loud frame is a click.
        held = np.convolve(above[lo:hi].astype(np.int32), np.ones(hold, dtype=np.int32), mode="full")[hold - 1:hold - 1 + hi - lo] >= hold
        # held[k] says frames lo+k-hold+1 .. lo+k are all above.
        firsts = np.flatnonzero(held[: min(hi - lo, (start - lo) + 2 * reach)])
        lasts = np.flatnonzero(held[max(0, (end - lo) - 2 * reach):]) + max(0, (end - lo) - 2 * reach)
        new_start = lo + int(firsts[0]) - hold + 1 if len(firsts) else start
        new_end = lo + int(lasts[-1]) + 1 if len(lasts) else end
        if new_end - new_start < max(1, int(round(MIN_RUN_S / hop_s))):
            new_start, new_end = start, end
        out[max(0, new_start):min(n, new_end)] = True
    return out


# How far an edge may move, how long the level must hold, and the least
# difference between speech and floor that gives an edge to find.
SNAP_REACH_S = 0.5
SNAP_HOLD_S = 0.05
SNAP_MIN_RANGE_DB = 6.0


def smooth_probability(raw: np.ndarray, level_db: Optional[np.ndarray] = None, hop_s: float = HOP_S) -> np.ndarray:
    """Smoothed probabilities that agree with the run decision.

    The sync engines use the probability itself, not only the decision, so
    frames the detector was unsure of weigh less. It is folded so that it sits
    above 0.5 exactly inside the runs: a caller thresholding at 0.5 gets the
    runs, and one correlating against it gets the confidence. With the
    speech-band level, the runs' edges are snapped to it (see snap_edges).
    """
    if len(raw) == 0:
        return raw.astype(np.float32)
    soft = _moving_average(raw, max(1, int(round(0.08 / hop_s))))
    decided = hysteresis(soft, hop_s)
    if level_db is not None and len(level_db) == len(decided):
        decided = snap_edges(decided, level_db, hop_s)
    out = np.where(decided, 0.5 + 0.5 * soft, OUTSIDE_WEIGHT * soft)
    return out.astype(np.float32)


# Outside a run the probability is scaled down to this: still a hint of
# speech the runs missed, but not enough to put the blurred edge back.
OUTSIDE_WEIGHT = 0.5


def activity_from_features(features: Features) -> np.ndarray:
    return smooth_probability(speech_probability(features), features.speech_db)


def activity_from_samples(samples: np.ndarray) -> Tuple[np.ndarray, float]:
    """Speech activity of mono 16 kHz samples in memory."""
    return activity_from_features(features_from_samples(samples)), HOP_S


def speech_runs(probs: np.ndarray, hop_s: float = HOP_S, threshold: float = 0.5) -> List[Tuple[float, float]]:
    """(start, end) seconds of each stretch above ``threshold``."""
    return [(s * hop_s, e * hop_s) for s, e in _runs(probs > threshold)]


# ------------------------------------------------------------- engines


def speech_activity(
    path: str,
    audio_track: int = 0,
    engine: str = "energy",
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, float]:
    """Probability of speech every 10 ms along one audio stream of ``path``.

    Returns ``(probs, hop_s)``; frame ``k`` is centred on ``k * hop_s``.
    ``progress`` is called with 0-100.
    """
    if engine == "silero":
        return _silero(path, audio_track, token, progress)
    if engine not in ("energy", None, ""):
        from .tasks import TaskError

        raise TaskError(f"Unknown speech detector: {engine}")
    features = stream_features(path, audio_track, token, progress)
    if len(features) == 0:
        raise MediaError(f"No audio decoded from {os.path.basename(path)}")
    return activity_from_features(features), HOP_S


# Pack and worker the Silero engine runs in (owned by packs.py). Silero
# ships inside the faster-whisper pack, so there is no pack of its own.
SILERO_PACK = "asr-faster"
SILERO_WORKER = "vad_worker.py"
SILERO_MISSING = "Install the Speech recognition (any computer) pack"


def _silero(
    path: str,
    audio_track: int,
    token: Optional[CancellationToken],
    progress: Optional[Callable[[int], None]],
) -> Tuple[np.ndarray, float]:
    """Silero VAD from the speech pack, resampled onto the 10 ms grid.

    The worker answers at Silero's own hop (512 samples, 32 ms) with a raw
    float32 file (``probsPath``) when given ``outDir``, or a list
    (``probs``); either is interpolated onto the 10 ms grid. Each value
    scores one window, so it is placed at the window's centre.
    """
    from .tasks import TaskContext, TaskError

    try:
        from . import packs
    except ImportError as exc:
        raise TaskError(f"The Silero speech detector is not available: {SILERO_MISSING.lower()}, or use the built-in detector.") from exc
    if not _pack_installed(packs):
        raise TaskError(f"The Silero speech detector is not installed: {SILERO_MISSING.lower()}, or use the built-in detector.")

    workdir = tempfile.mkdtemp(prefix="subsync-vad-")
    ctx = TaskContext(
        token=token or CancellationToken(),
        progress=lambda percent, stage="": progress(int(percent)) if progress else None,
        log=lambda message: None,
        workdir=workdir,
    )
    try:
        result = packs.run_worker(
            SILERO_PACK, SILERO_WORKER,
            {"audio": path, "audioTrack": int(audio_track), "hopS": HOP_S, "outDir": workdir},
            ctx,
        )
        hop = float(result.get("hopS") or HOP_S)
        if result.get("probsPath"):
            probs = np.fromfile(result["probsPath"], dtype=np.float32)
        else:
            probs = np.asarray(result.get("probs") or [], dtype=np.float32)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if len(probs) == 0:
        raise TaskError("The speech pack's detector returned no speech data.")
    grid = np.arange(int(len(probs) * hop / HOP_S)) * HOP_S
    probs = np.interp(grid, (np.arange(len(probs)) + 0.5) * hop, probs).astype(np.float32)
    return smooth_probability(np.clip(probs, 0.0, 1.0)), HOP_S


def _pack_installed(packs) -> bool:
    try:
        return bool(packs.python_for(SILERO_PACK))
    except Exception:  # noqa: BLE001 - a broken pack is a missing pack here
        return False


def engine_statuses() -> List[dict]:
    """The speech detectors and whether each can run here."""
    statuses = [{
        "id": "energy",
        "label": "Built-in (speech band, modulation, voicing)",
        "available": True,
        "reason": None,
        "pack": None,
    }]
    available = False
    reason = SILERO_MISSING
    try:
        from . import packs

        available = _pack_installed(packs)
    except ImportError:
        available = False
    statuses.append({
        "id": "silero",
        "label": "Silero VAD",
        "available": available,
        "reason": None if available else reason,
        "pack": SILERO_PACK,
    })
    return statuses
