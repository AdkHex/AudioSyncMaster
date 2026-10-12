"""Each track reduced to onset envelopes, full band and per band.

Moved out of ``dubsync`` (see its docstring for the method as a whole);
imported from there by everything else."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..correlate import ENVELOPE_HOP
from ..media import Cancelled, CancellationToken, MediaError, audio_lead_s, probe, stream_audio

from .tuning import ANALYSIS_SR, BANDS, BAND_KNEE, BAND_WINDOW, ENVELOPE_RATE, SIDE_FILTER, SIDE_MIN_RATIO


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


@dataclass
class TrackEnvelope:
    """One track reduced to a 500 Hz frame-energy curve and its onsets.

    The energy is kept as well as the onsets because the onsets are derived
    from it and the derivation is cheap: a trial speed is a resample of the
    energy curve followed by that derivation, which is how speeds are tried
    without decoding the file again.
    """

    path: str
    track: int
    energy: np.ndarray
    """RMS of each ENVELOPE_HOP-sample frame, float32."""
    onset: np.ndarray
    """Half-wave-rectified rise in log energy, one shorter than ``energy``."""
    speed: float = 1.0
    """Playback-speed factor applied when decoding: the timeline these curves
    are on is the file's own time multiplied by this."""
    rate: int = ENVELOPE_RATE
    band_energy: Dict[str, np.ndarray] = field(default_factory=dict)
    """Frame energy of each band in BANDS, same frames as ``energy``."""
    band_onset: Dict[str, np.ndarray] = field(default_factory=dict)
    """Onsets of each band, derived like ``onset``."""

    @property
    def duration_s(self) -> float:
        return len(self.energy) / self.rate

    def onsets(self, band: str) -> np.ndarray:
        """The onset curve of one band; ``full`` is the whole spectrum."""
        return self.onset if band == "full" else self.band_onset[band]

    @property
    def bands(self) -> List[str]:
        """``full`` first, then every band this track carries."""
        return ["full"] + sorted(self.band_onset)

    @classmethod
    def from_energy(
        cls, path: str, track: int, energy: np.ndarray, speed: float = 1.0,
        band_energy: Optional[Dict[str, np.ndarray]] = None,
    ):
        bands = {k: v.astype(np.float32) for k, v in (band_energy or {}).items()}
        return cls(
            path, track, energy.astype(np.float32), _onsets(energy), speed,
            band_energy=bands, band_onset={k: _band_onsets(v) for k, v in bands.items()},
        )

    def at_speed(self, speed: float) -> "TrackEnvelope":
        """The same track as it would decode at another playback speed."""
        if abs(speed - self.speed) < 1e-9:
            return self
        factor = speed / self.speed
        n = int(len(self.energy) * factor)
        source = np.arange(n, dtype=np.float64) / factor
        energy = np.interp(source, np.arange(len(self.energy)), self.energy)
        bands = {
            k: np.interp(source, np.arange(len(v)), v) for k, v in self.band_energy.items()
        }
        return TrackEnvelope.from_energy(self.path, self.track, energy, speed, bands)


def _onsets(energy: np.ndarray) -> np.ndarray:
    """``correlate.onset_envelope``, from frame energies already computed."""
    if len(energy) < 2:
        return np.zeros(0, dtype=np.float32)
    log_energy = np.log1p(energy.astype(np.float64) * 1000.0)
    return np.maximum(0.0, np.diff(log_energy)).astype(np.float32)


def _band_onsets(energy: np.ndarray) -> np.ndarray:
    """``_onsets`` of a band, with the band's own level put on the knee.

    ``_onsets`` compresses with log1p(1000 * energy), a knee set for the
    full-band RMS of a film. A band carries far less -- the 4-8 kHz band a
    hundredth -- and below the knee the curve is not the log rise it is
    above it but the raw difference, a few tall spikes where the band is
    loudest and nothing between: correlated, that gives heavy-tailed noise
    and chance peaks of eight or nine floors. Scaling the band so its
    median frame sits at the knee makes its curve as dense as the full
    band's, and the noise floors comparable.
    """
    if len(energy) == 0:
        return np.zeros(0, dtype=np.float32)
    typical = float(np.median(energy))
    scale = BAND_KNEE / (1000.0 * max(typical, 1e-9))
    return _onsets(energy.astype(np.float64) * scale)


def _frame_energies(blocks, hop: int, on_samples: Optional[Callable[[int], None]] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """RMS of every ``hop``-sample frame of a stream of mono blocks, and the
    whole-frame samples themselves, block by block, for the band pass."""
    energies: List[np.ndarray] = []
    kept: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    seen = 0
    for block in blocks:
        if carry.size:
            block = np.concatenate([carry, block])
        frames = len(block) // hop
        if frames:
            shaped = block[: frames * hop].reshape(frames, hop).astype(np.float64)
            energies.append(np.sqrt(np.mean(shaped * shaped, axis=1) + 1e-12).astype(np.float32))
        kept.append(block[: frames * hop])
        carry = block[frames * hop:]
        seen += frames * hop
        if on_samples:
            on_samples(seen)
    return (np.concatenate(energies) if energies else np.zeros(0, dtype=np.float32)), kept


def _side_energy(
    path: str, track: int, rate: int, token: Optional[CancellationToken], lead_s: float,
) -> np.ndarray:
    """Frame energy of the track's stereo difference (see SIDE_FILTER)."""
    blocks = stream_audio(path, rate, track=track, token=token, block_s=30.0, audio_filter=SIDE_FILTER, lead_s=lead_s)
    energy, _ = _frame_energies(blocks, ENVELOPE_HOP)
    return energy


def build_envelope(
    path: str,
    track: int = 0,
    token: Optional[CancellationToken] = None,
    progress: Optional[Callable[[float], None]] = None,
    expected_duration_s: Optional[float] = None,
    speed: float = 1.0,
    lead_s: Optional[float] = None,
    channels: Optional[int] = None,
    issues: Optional[List[str]] = None,
) -> TrackEnvelope:
    """Decode a whole track once, keeping only its 500 Hz energy curve.

    The curve is on the file's own clock (see ``audio_lead_s``): time zero
    is the file's, not the audio stream's first sample, which is the time
    every seek, picture frame and mux uses. A track whose audio starts
    after its picture is padded at the front by the difference.

    Args:
        speed: decode at this playback speed. Asking ffmpeg for
            ANALYSIS_SR * speed and reading the samples as ANALYSIS_SR is a
            time-stretch by that factor, done by the decoder's resampler.
        progress: called with 0.0-1.0 as the file is read, when the duration
            is known.
        lead_s: where the stream's first sample sits on the file's clock;
            probed when not given.
        channels: the track's channel count; probed when not given. A
            track of two or more channels also gets the ``side`` band (see
            SIDE_FILTER), decoded alongside the mono one.
    """
    if lead_s is None or channels is None:
        info = probe(path, token)
        if lead_s is None:
            lead_s = audio_lead_s(info, track)
        if channels is None:
            stream = info.audio_tracks[track] if track < len(info.audio_tracks) else None
            channels = int((stream.channels if stream else None) or info.channels or 1)
    rate = max(1, int(round(ANALYSIS_SR * speed)))
    hop = ENVELOPE_HOP

    side: dict = {}
    helper: Optional[threading.Thread] = None
    if channels >= 2:
        def read_side() -> None:
            try:
                side["energy"] = _side_energy(path, track, rate, token, lead_s)
            except (MediaError, Cancelled) as exc:
                side["error"] = exc
        helper = threading.Thread(target=read_side, name="side-band", daemon=True)
        helper.start()

    bands = _BandFrames()

    def mono_blocks():
        for block in stream_audio(path, rate, track=track, token=token, block_s=30.0, lead_s=lead_s, issues=issues):
            yield block

    def on_samples(seen: int) -> None:
        if progress and expected_duration_s:
            progress(min(1.0, seen / rate / expected_duration_s))

    energies: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    seen = 0
    for block in mono_blocks():
        if carry.size:
            block = np.concatenate([carry, block])
        frames = len(block) // hop
        if frames:
            shaped = block[: frames * hop].reshape(frames, hop).astype(np.float64)
            energies.append(np.sqrt(np.mean(shaped * shaped, axis=1) + 1e-12).astype(np.float32))
        bands.push(block[: frames * hop])
        carry = block[frames * hop:]
        seen += len(block) - len(carry)
        on_samples(seen)
    if helper is not None:
        helper.join()
    if token is not None and token.cancelled:
        raise Cancelled("operation cancelled")
    if not energies:
        raise MediaError(f"No audio decoded from {os.path.basename(path)}")
    energy = np.concatenate(energies)
    band_energy = bands.finish(len(energy))
    difference = side.get("energy")
    if difference is not None and len(difference):
        if len(difference) < len(energy):
            difference = np.concatenate([difference, np.full(len(energy) - len(difference), difference[-1], dtype=np.float32)])
        difference = difference[: len(energy)]
        level = float(np.sqrt(np.mean(energy.astype(np.float64) ** 2)))
        if float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))) >= SIDE_MIN_RATIO * level:
            band_energy["side"] = difference
    return TrackEnvelope.from_energy(path, track, energy, speed, band_energy)


class _BandFrames:
    """The band-limited frame energies of a stream, built block by block.

    A Hann window of BAND_WINDOW samples is centred on every ENVELOPE_HOP
    frame of the full-band envelope, so frame ``i`` of a band and frame
    ``i`` of the full band describe the same instant; the samples the
    window needs beyond the block are kept back until the next block
    arrives. Energy is the RMS of the band's part of the windowed frame,
    on the same scale as the full-band RMS.
    """

    def __init__(self) -> None:
        self.window = np.hanning(BAND_WINDOW).astype(np.float32)
        freqs = np.fft.rfftfreq(BAND_WINDOW, 1.0 / ANALYSIS_SR)
        self.masks = {name: (freqs >= lo) & (freqs < hi) for name, (lo, hi) in BANDS.items()}
        # Parseval for a one-sided spectrum of a Hann-windowed frame.
        self.scale = 2.0 / (BAND_WINDOW * float(np.sum(self.window ** 2)))
        self.lead = BAND_WINDOW // 2 - ENVELOPE_HOP // 2
        self.buffer = np.zeros(self.lead, dtype=np.float32)
        self.done = 0
        self.frames: Dict[str, List[np.ndarray]] = {name: [] for name in BANDS}

    def push(self, samples: np.ndarray) -> None:
        self.buffer = np.concatenate([self.buffer, samples.astype(np.float32)])
        count = (len(self.buffer) - BAND_WINDOW) // ENVELOPE_HOP + 1
        if count <= 0:
            return
        shaped = np.lib.stride_tricks.as_strided(
            self.buffer, shape=(count, BAND_WINDOW),
            strides=(self.buffer.strides[0] * ENVELOPE_HOP, self.buffer.strides[0]),
        )
        power = np.abs(np.fft.rfft(shaped * self.window, axis=1)) ** 2
        for name, mask in self.masks.items():
            self.frames[name].append(np.sqrt(power[:, mask].sum(axis=1) * self.scale + 1e-12).astype(np.float32))
        self.buffer = self.buffer[count * ENVELOPE_HOP:].copy()
        self.done += count

    def finish(self, frames: int) -> Dict[str, np.ndarray]:
        """The band energies, padded or cut to ``frames`` frames."""
        self.push(np.zeros(BAND_WINDOW, dtype=np.float32))
        out = {}
        for name, parts in self.frames.items():
            curve = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if len(curve) < frames:
                curve = np.concatenate([curve, np.full(frames - len(curve), curve[-1] if len(curve) else 0.0, dtype=np.float32)])
            out[name] = curve[:frames]
        return out


def _pool(values: np.ndarray, factor: int) -> np.ndarray:
    """Max over consecutive groups, so a transient keeps its height."""
    if factor <= 1:
        return values.astype(np.float64)
    n = len(values) // factor
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return values[: n * factor].reshape(n, factor).max(axis=1).astype(np.float64)
