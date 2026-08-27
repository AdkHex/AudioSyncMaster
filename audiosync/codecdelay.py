"""Compensate the AC3 decoder delay that survives into a raw stream.

AC3 and E-AC3 decode their output shifted by 256 samples -- 5.333ms at 48kHz.
A bare .ac3/.eac3 stream carries no timestamps, so nothing can correct that
shift and it lands directly in a measurement.

Inside a container the answer depends on the ffmpeg build. Measured with
identical source content, AAC reference against the same E-AC3 stream:

    raw .eac3                     +5.38 ms   every build
    .mka / .mkv, ffmpeg 9         +0.03 ms   priming trimmed from timestamps
    .mka / .mkv, Ubuntu packaged  +5.35 ms   priming still present

So the shift cannot be predicted from the file alone, and this module does not
try. The correction is scoped to raw streams, where it is unconditionally
right. A container is left alone: on a build that trims, correcting would
introduce the very 5.333ms error the correction exists to remove, and on a
build that does not, the residue is 5.333ms -- an order of magnitude below the
drift and refinement errors that actually matter, and it does not vary within
a run, so it cannot masquerade as drift.

Erring toward doing nothing is deliberate. A wrong correction is invisible and
lands with full confidence; an absent one leaves a small constant bias that
shows up identically at the start and the end of a file.

Expressed in samples rather than milliseconds because the shift is a fixed
sample count: a hard-coded ms value would be right at 48kHz and wrong at 44.1.
"""

from __future__ import annotations

from typing import Optional

# Delay in samples at the stream's own rate, keyed by ffprobe's codec_name.
# Positive means a measurement against a zero-delay codec reads that much high.
CODEC_DELAY_SAMPLES = {
    "ac3": 256,
    "eac3": 256,
}

# Sample rates AC3 and E-AC3 actually support. A file claiming anything else has
# been resampled somewhere, and the delay no longer applies cleanly.
AC3_SAMPLE_RATES = (32000, 44100, 48000)

# ffprobe format names for raw elementary streams -- a bare sequence of frames
# with no timestamps. Everything else is a real container whose timestamps
# ffmpeg already uses to trim decoder priming.
RAW_STREAM_FORMATS = frozenset({"ac3", "eac3"})


def is_raw_stream(container_format: Optional[str]) -> bool:
    """Whether this is a bare elementary stream rather than a container.

    ffprobe reports comma-separated candidates for ambiguous inputs, so any
    listed name matching is enough. An unknown format is treated as a container:
    that skips the correction, and skipping one that was needed costs 5.333ms,
    while applying one that was not costs the same in the other direction on
    far more files.
    """
    if not container_format:
        return False
    names = {part.strip().lower() for part in container_format.split(",")}
    return bool(names & RAW_STREAM_FORMATS)


def codec_delay_ms(
    codec: Optional[str],
    sample_rate: Optional[int],
    container_format: Optional[str] = None,
) -> float:
    """Delay this codec adds to its decoded output, in milliseconds.

    Returns 0.0 when the codec is unknown, adds no delay, sits in a container
    that already compensates it, or runs at a rate where the correction cannot
    be trusted -- an unrecognised case must leave the measurement untouched
    rather than guess.
    """
    if not codec or not sample_rate or sample_rate <= 0:
        return 0.0

    samples = CODEC_DELAY_SAMPLES.get(codec.lower())
    if samples is None:
        return 0.0

    # The decisive check. In a container ffmpeg trims the priming samples using
    # the stream's timestamps, so no shift ever reaches the measurement and
    # correcting for one introduces the very error it was meant to remove.
    if not is_raw_stream(container_format):
        return 0.0

    if codec.lower() in ("ac3", "eac3") and sample_rate not in AC3_SAMPLE_RATES:
        # Not a rate the format defines; something has already resampled it and
        # the fixed sample count no longer describes what happened.
        return 0.0

    return samples / sample_rate * 1000.0


def relative_codec_delay_ms(
    primary_codec: Optional[str],
    primary_rate: Optional[int],
    secondary_codec: Optional[str],
    secondary_rate: Optional[int],
    primary_format: Optional[str] = None,
    secondary_format: Optional[str] = None,
) -> float:
    """How much of a measured offset is pure codec delay rather than real sync.

    Subtract this from a measurement to recover the true offset. Returns 0.0
    for two files using the same codec at the same rate, because the delay is
    identical on both sides and already cancels -- and for anything in a real
    container, where there is no delay to remove.
    """
    primary = codec_delay_ms(primary_codec, primary_rate, primary_format)
    secondary = codec_delay_ms(secondary_codec, secondary_rate, secondary_format)
    # A measurement says how far the secondary sits behind the primary, so a
    # delay on the secondary inflates it and one on the primary reduces it.
    return secondary - primary


def describe(
    primary_codec: Optional[str],
    primary_rate: Optional[int],
    secondary_codec: Optional[str],
    secondary_rate: Optional[int],
    primary_format: Optional[str] = None,
    secondary_format: Optional[str] = None,
) -> Optional[str]:
    """Explain a non-zero correction, for the console and the result detail."""
    correction = relative_codec_delay_ms(
        primary_codec, primary_rate, secondary_codec, secondary_rate,
        primary_format, secondary_format,
    )
    if abs(correction) < 0.001:
        return None
    return (
        f"Compensated {correction:+.3f}ms of codec delay "
        f"({primary_codec or 'unknown'} vs {secondary_codec or 'unknown'}); "
        "the two formats decode with different alignment."
    )
