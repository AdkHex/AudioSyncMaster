"""Offset estimation between two audio signals.

Sign convention (verified by tests/test_correlate.py against known fixtures):

    positive delay  =>  the secondary track starts LATER than the primary.
                        To align, the secondary must be moved EARLIER.

The previous implementation returned the negation of this and documented the
opposite of what it returned, so every reported delay had the wrong sign.

Estimation runs on onset-strength envelopes rather than raw PCM. Raw waveform
correlation assumes both tracks are near-identical recordings; a re-encoded or
re-mastered dub breaks that assumption while preserving transient timing, which
is what the envelope captures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Envelope frame rate. The envelope is decimated relative to the audio, so the
# coarse search runs on a much smaller array; precision is recovered afterwards
# by refining against the full-rate signal.
ENVELOPE_HOP = 32

# Below this peak-prominence ratio a correlation is indistinguishable from the
# noise floor. Calibrated in tests: unrelated audio scores ~1-8, true matches
# score >>20 even under heavy level mismatch and codec differences.
MIN_PEAK_RATIO = 12.0

# A segment with less energy than this contributes nothing but numerical noise.
MIN_RMS = 1e-4

# Why a silent segment was refused, by side.
SILENT_PRIMARY = "the reference track is silent or near-silent here"
SILENT_SECONDARY = "the compared track is silent or near-silent here"

# Why a match was refused: no correlation peak stood out from the rest.
NO_PEAK = "no distinct correlation peak; tracks appear unrelated"

# How far the waveform refinement may move the envelope's answer. Its only job
# is to undo envelope quantisation, which is ENVELOPE_HOP frames wide, so a
# couple of frames either side is all it can legitimately need. The original
# 60ms let it relocate the answer entirely.
REFINE_WINDOW_MS = 3.0 * ENVELOPE_HOP / 16000 * 1000.0

# Prominence the waveform peak must clear before it is trusted, on the phase
# transform (GCC-PHAT) the refinement correlates with.
#
# PHAT whitens the cross-spectrum, so every frequency votes with its phase
# alone: audio that is the same recording underneath -- a re-encode, or a dub
# built on the same music and effects as the original -- lines up at one lag
# across the whole band and forms a single spike, whatever EQ, level or
# codec sits between the two; audio that is not leaves only noise. Measured
# on 16 kHz windows: an English 5.1 master against a Hindi AC-3 dub on its
# M&E scores 770-3350, with or without the dialogue in it; unrelated audio
# scores a median of 5.6 and at most 20.8 over 180 windows. The bar sits well
# clear of that tail and an order of magnitude below any real match.
#
# The plain correlation this replaced needed a bar of 200 for the same job,
# which a dub never cleared: its dialogue swamps the shared M&E unless the
# spectrum is whitened first. So a dub kept the envelope's answer, up to
# half a millisecond out, where the M&E it shares could have placed it to a
# few microseconds.
MIN_REFINE_PEAK_RATIO = 50.0


@dataclass
class OffsetEstimate:
    """Result of a single offset measurement."""

    delay_ms: Optional[float]
    """Positive => secondary starts later than primary. None if no match."""

    confidence: float
    """0.0-1.0 match quality, derived from correlation peak prominence."""

    peak_ratio: float
    """Raw peak-to-noise-floor ratio, before being squashed into confidence."""

    reason: Optional[str] = None
    """Why the estimate was rejected, when delay_ms is None."""

    confirmed: bool = False
    """The raw waveforms agree at this offset (GCC-PHAT, see
    MIN_REFINE_PEAK_RATIO): the two tracks share actual audio here, not only
    the timing of it, and delay_ms is placed to a fraction of a sample."""

    waveform_ratio: float = 0.0
    """The waveform peak's prominence, when the waveforms agreed."""

    mix: str = "full"
    """What was correlated: ``full``, the whole mix, or ``me``, the music and
    effects with the centre-panned dialogue cancelled out (see
    ``analyze._measure_window``)."""

    @property
    def matched(self) -> bool:
        return self.delay_ms is not None


def _standardize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance copy.

    Returns a new array; the previous implementation mutated its argument in
    place, which corrupted caller buffers because NumPy slices are views.
    """
    out = signal.astype(np.float64, copy=True)
    out -= out.mean()
    std = out.std()
    if std > 1e-12:
        out /= std
    return out


def onset_envelope(signal: np.ndarray, hop: int = ENVELOPE_HOP) -> np.ndarray:
    """Half-wave-rectified energy difference: a lightweight onset strength curve.

    Tracks where energy *rises*, which is what a listener perceives as the
    attack of a word or a sound effect. Robust to level and codec differences
    because it responds to change, not absolute amplitude.
    """
    if len(signal) < hop * 2:
        return np.zeros(0, dtype=np.float64)

    n_frames = len(signal) // hop
    frames = signal[: n_frames * hop].reshape(n_frames, hop).astype(np.float64)
    energy = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    log_energy = np.log1p(energy * 1000.0)
    # Rising edges only; falling energy is not an onset.
    return np.maximum(0.0, np.diff(log_energy))


def _peak_ratio(corr: np.ndarray, peak_index: int, sr_hint: int) -> float:
    """Prominence of the winning peak against the rest of the correlation.

    A true alignment produces one sharp spike far above the surrounding surface.
    Unrelated audio produces a diffuse surface whose maximum is barely above the
    rest. Comparing the peak to the median of everything outside its immediate
    neighbourhood separates the two cleanly.
    """
    peak = float(corr[peak_index])
    if peak <= 0:
        return 0.0

    # Exclude a window around the peak so the peak's own shoulders don't inflate
    # the baseline. Width scales with the array so it holds at any resolution.
    exclude = max(4, len(corr) // 200)
    lo = max(0, peak_index - exclude)
    hi = min(len(corr), peak_index + exclude + 1)
    masked = np.concatenate([corr[:lo], corr[hi:]])
    if masked.size == 0:
        return 0.0

    baseline = float(np.median(np.abs(masked)))
    if baseline < 1e-12:
        return 0.0
    return peak / baseline


def _parabolic_vertex(corr: np.ndarray, index: int) -> float:
    """Sub-sample peak position via parabolic interpolation of the three points
    around the maximum. Recovers precision finer than one sample period."""
    if index <= 0 or index >= len(corr) - 1:
        return float(index)
    left, centre, right = float(corr[index - 1]), float(corr[index]), float(corr[index + 1])
    denom = left - 2.0 * centre + right
    if abs(denom) < 1e-12:
        return float(index)
    offset = 0.5 * (left - right) / denom
    if not math.isfinite(offset) or abs(offset) > 1.0:
        return float(index)
    return index + offset


def _fast_fft_size(n: int) -> int:
    """Smallest 5-smooth size >= n.

    FFT cost depends sharply on the factorisation of the transform length. A
    plain next-power-of-two can nearly double the work when n sits just above a
    power of two; allowing factors of 3 and 5 finds a much closer fit.
    """
    if n <= 16:
        return 16

    best = 1 << (n - 1).bit_length()  # a power of two always works
    # Every 5-smooth number is (3^a * 5^b) shifted up by powers of two, so
    # walking the odd 3/5 combinations below `best` covers the whole space.
    power_of_three = 1
    while power_of_three < best:
        odd = power_of_three
        while odd < best:
            candidate = odd
            while candidate < n:
                candidate *= 2
            best = min(best, candidate)
            odd *= 5
        power_of_three *= 3
    return best


def _correlate(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Cross-correlation of two standardized signals, full overlap.

    Uses numpy's real-input FFT directly rather than scipy.signal.fftconvolve.
    scipy was the single heaviest import in the frozen sidecar (~0.3s of a
    ~0.35s cold start, and a large share of the bundle) for exactly one
    function. rfft also halves the transform work versus a complex FFT, since
    both inputs are real.
    """
    n = len(primary) + len(secondary) - 1
    size = _fast_fft_size(n)
    spectrum = np.fft.rfft(primary, size) * np.fft.rfft(secondary[::-1], size)
    return np.fft.irfft(spectrum, size)[:n]


def _correlate_phat(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Cross-correlation with the phase transform (GCC-PHAT), full overlap.

    Same lag layout as ``_correlate``. Every frequency bin is scaled to unit
    magnitude before the inverse transform, so the peak reflects how
    consistently the phases agree, not which band is loudest.
    """
    n = len(primary) + len(secondary) - 1
    size = _fast_fft_size(n)
    spectrum = np.fft.rfft(primary, size) * np.fft.rfft(secondary[::-1], size)
    spectrum /= np.abs(spectrum) + 1e-12
    return np.fft.irfft(spectrum, size)[:n]


def estimate_offset(
    primary: np.ndarray,
    secondary: np.ndarray,
    sr: int,
    max_offset_ms: Optional[float] = None,
    min_peak_ratio: float = MIN_PEAK_RATIO,
    search_ms: Optional[Tuple[float, float]] = None,
) -> OffsetEstimate:
    """Estimate how far the secondary track lags the primary.

    Args:
        primary: reference signal (typically the video's own audio).
        secondary: signal to align against the primary.
        sr: sample rate shared by both signals.
        max_offset_ms: reject alignments beyond this magnitude. Bounding the
            search suppresses distant spurious peaks; None searches everything.
        min_peak_ratio: prominence below which the match is rejected outright.
        search_ms: (lowest, highest) delay to accept, for a search centred
            somewhere other than zero. Applied on top of max_offset_ms.

    Returns:
        OffsetEstimate. delay_ms is positive when the secondary starts later.
    """
    if len(primary) < sr // 4 or len(secondary) < sr // 4:
        return OffsetEstimate(None, 0.0, 0.0, "segment too short to analyse")

    # Reject segments that carry no usable signal before spending effort on them.
    primary_rms = float(np.sqrt(np.mean(primary.astype(np.float64) ** 2)))
    secondary_rms = float(np.sqrt(np.mean(secondary.astype(np.float64) ** 2)))
    if primary_rms < MIN_RMS or secondary_rms < MIN_RMS:
        # Which side, since "silent" is only actionable once it names a track.
        return OffsetEstimate(None, 0.0, 0.0, SILENT_PRIMARY if primary_rms < MIN_RMS else SILENT_SECONDARY)

    env_primary = onset_envelope(primary)
    env_secondary = onset_envelope(secondary)
    if env_primary.size < 8 or env_secondary.size < 8:
        return OffsetEstimate(None, 0.0, 0.0, "segment too short for onset analysis")

    coarse_p = _standardize(env_primary)
    coarse_s = _standardize(env_secondary)
    if coarse_p.std() < 1e-9 or coarse_s.std() < 1e-9:
        return OffsetEstimate(None, 0.0, 0.0, "segment has no detectable onsets")

    corr = _correlate(coarse_p, coarse_s)
    zero_lag_index = len(coarse_s) - 1
    env_sr = sr / ENVELOPE_HOP

    # Restrict the search window before choosing a winner, so an out-of-range
    # spurious peak can never beat a plausible in-range one.
    search = corr
    search_base = 0
    if max_offset_ms is not None:
        max_lag = int(math.ceil(abs(max_offset_ms) / 1000.0 * env_sr)) + 2
        lo = max(0, zero_lag_index - max_lag)
        hi = min(len(corr), zero_lag_index + max_lag + 1)
        if hi - lo >= 3:
            search = corr[lo:hi]
            search_base = lo
    if search_ms is not None:
        # delay = -(index - zero_lag) / env_sr, so the highest delay is the
        # lowest index.
        lo = max(search_base, zero_lag_index - int(math.ceil(search_ms[1] / 1000.0 * env_sr)) - 2)
        hi = min(search_base + len(search), zero_lag_index - int(math.floor(search_ms[0] / 1000.0 * env_sr)) + 3)
        if hi - lo < 3:
            return OffsetEstimate(None, 0.0, 0.0, "search range lies outside the decoded audio")
        search = corr[lo:hi]
        search_base = lo

    local_peak = int(np.argmax(search))
    peak_index = search_base + local_peak
    ratio = _peak_ratio(corr, peak_index, sr)

    if ratio < min_peak_ratio:
        return OffsetEstimate(
            None,
            _ratio_to_confidence(ratio, min_peak_ratio),
            ratio,
            NO_PEAK,
        )

    refined_index = _parabolic_vertex(corr, peak_index)

    # argmax > zero_lag means the primary had to slide forward to line up, i.e.
    # the primary starts later, i.e. the secondary starts EARLIER (negative).
    lag_frames = refined_index - zero_lag_index
    delay_ms = -(lag_frames / env_sr) * 1000.0

    confidence = _ratio_to_confidence(ratio, min_peak_ratio)
    refined = _refine_against_waveform(primary, secondary, sr, delay_ms)
    if refined is None:
        return OffsetEstimate(delay_ms=float(delay_ms), confidence=confidence, peak_ratio=float(ratio))

    # The waveforms agreeing is the stronger evidence of the two: noise
    # cannot produce it (see MIN_REFINE_PEAK_RATIO), so it decides how sure
    # this window is when the envelope's peak was the weaker one.
    refined_ms, waveform_ratio = refined
    return OffsetEstimate(
        delay_ms=float(refined_ms),
        confidence=max(confidence, _ratio_to_confidence(waveform_ratio, MIN_REFINE_PEAK_RATIO)),
        peak_ratio=float(ratio),
        confirmed=True,
        waveform_ratio=float(waveform_ratio),
    )


def _refine_against_waveform(
    primary: np.ndarray,
    secondary: np.ndarray,
    sr: int,
    coarse_delay_ms: float,
    window_ms: float = REFINE_WINDOW_MS,
) -> Optional[Tuple[float, float]]:
    """Sharpen a coarse envelope-derived estimate using the full-rate signal.

    The envelope search is decimated by ENVELOPE_HOP, so its resolution is
    coarse. Correlating the raw waveform across a narrow window around the
    coarse answer recovers sample-level precision.

    Only valid where the two tracks share actual audio. A different
    performance -- different voices, different room -- has raw waveforms
    that do not correlate at all, even though the onsets line up exactly, and
    refining one means correlating noise against noise. The plain correlation
    this once used overwrote correct envelope answers that way. Two guards
    keep it to audio that is shared: a search window narrow enough to only
    correct envelope quantisation, and a prominence bar noise cannot clear.

    The correlation is GCC-PHAT: the cross-spectrum is whitened before it is
    transformed back, so a dub's shared music and effects line up across the
    whole band even with its own dialogue, EQ and codec on top.

    Returns:
        (refined delay in ms, waveform peak prominence), or None when the
        waveforms do not agree and the envelope's answer stands.
    """
    coarse_shift = int(round(-coarse_delay_ms / 1000.0 * sr))
    search_radius = int(window_ms / 1000.0 * sr)

    # Overlap the two signals at the coarse alignment.
    if coarse_shift >= 0:
        a = primary[coarse_shift:]
        b = secondary[: len(a)]
    else:
        b = secondary[-coarse_shift:]
        a = primary[: len(b)]

    n = min(len(a), len(b))
    if n < sr // 2:
        return None
    a = _standardize(a[:n])
    b = _standardize(b[:n])
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None

    corr = _correlate_phat(a, b)
    zero_lag = len(b) - 1
    lo = max(0, zero_lag - search_radius)
    hi = min(len(corr), zero_lag + search_radius + 1)
    if hi - lo < 3:
        return None

    window = corr[lo:hi]
    local_peak = int(np.argmax(window))
    peak_index = lo + local_peak
    waveform_ratio = _peak_ratio(corr, peak_index, sr)
    if waveform_ratio < MIN_REFINE_PEAK_RATIO:
        # No real waveform agreement: these are different recordings, and the
        # envelope's answer is the better one. Leaving it alone is the fix.
        return None

    refined_index = _parabolic_vertex(corr, peak_index)
    residual_frames = refined_index - zero_lag
    residual_ms = -(residual_frames / sr) * 1000.0
    return coarse_delay_ms + residual_ms, waveform_ratio


def _ratio_to_confidence(ratio: float, threshold: float) -> float:
    """Map peak prominence onto 0-1.

    The threshold sits at 0.5 so the number reads intuitively: below 0.5 the
    match was rejected, above it the score rises towards 1 as the peak sharpens.
    """
    if ratio <= 0:
        return 0.0
    scaled = 0.5 * (ratio / threshold) if ratio < threshold else 0.5 + 0.5 * (
        1.0 - math.exp(-(ratio - threshold) / threshold)
    )
    return float(max(0.0, min(1.0, scaled)))
