"""Compensate the constant delay some codecs add to their own output.

Several codecs emit audio shifted from where the source sat, as a property of
their transform rather than a property of the file. The shift is invisible in
metadata -- the container reports ``start_time`` of 0 -- so nothing downstream
can see it, yet it lands directly in a measured offset.

It only matters when the two files use *different* codecs. Both sides in AC3
cancel exactly; a WEB-DL's AAC against a disc rip's E-AC3 does not, and that is
the common case for dubbed material.

Measured, not assumed. Encoding one signal through two codecs and correlating
the results against each other gives the difference directly -- with no real
offset present, whatever is measured *is* the codec difference:

    codec            delay
    aac              0 samples
    libmp3lame       0 samples   (encoder delay is carried in start_time)
    libopus          0 samples
    flac             0 samples
    ac3           +256 samples
    eac3          +256 samples

The AC3 figure held at 192k/448k/640k, mono/stereo/5.1, and at every sample
rate the format supports -- always exactly 256 samples, which is 5.333ms at
48kHz and 5.805ms at 44.1kHz. Expressing it in samples rather than milliseconds
is what makes the correction exact at any rate.

The sign is fixed by measurement, not by reasoning about which way a decoder
shifts its output: with AAC as reference and E-AC3 as secondary over identical
source content, the measured offset is +5.318ms at 48kHz, so that is what has
to come back out of a real measurement.
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


def codec_delay_ms(codec: Optional[str], sample_rate: Optional[int]) -> float:
    """Delay this codec adds to its decoded output, in milliseconds.

    Returns 0.0 when the codec is unknown, adds no delay, or is running at a
    rate where the correction cannot be trusted -- an unrecognised case must
    leave the measurement untouched rather than guess.
    """
    if not codec or not sample_rate or sample_rate <= 0:
        return 0.0

    samples = CODEC_DELAY_SAMPLES.get(codec.lower())
    if samples is None:
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
) -> float:
    """How much of a measured offset is pure codec delay rather than real sync.

    Subtract this from a measurement to recover the true offset. Two files using
    the same codec at the same rate return 0.0, because the delay is identical
    on both sides and already cancels.
    """
    primary = codec_delay_ms(primary_codec, primary_rate)
    secondary = codec_delay_ms(secondary_codec, secondary_rate)
    # A measurement says how far the secondary sits behind the primary, so a
    # delay on the secondary inflates it and one on the primary reduces it.
    return secondary - primary


def describe(
    primary_codec: Optional[str],
    primary_rate: Optional[int],
    secondary_codec: Optional[str],
    secondary_rate: Optional[int],
) -> Optional[str]:
    """Explain a non-zero correction, for the console and the result detail."""
    correction = relative_codec_delay_ms(
        primary_codec, primary_rate, secondary_codec, secondary_rate
    )
    if abs(correction) < 0.001:
        return None
    return (
        f"Compensated {correction:+.3f}ms of codec delay "
        f"({primary_codec or 'unknown'} vs {secondary_codec or 'unknown'}); "
        "the two formats decode with different alignment."
    )
