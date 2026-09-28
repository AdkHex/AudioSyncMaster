"""Colour science for HDR -> SDR: transfer functions, primaries, tone curves.

Everything the tone-mapper and the HDR subtitle tool need, in plain numpy so
the shipped engine can build a 3D LUT that works with *any* FFmpeg -- the
Homebrew and Windows builds most users have lack zscale and libplacebo, but
every build has ``lut3d``.

The pipeline a LUT entry goes through (``hdr_to_sdr``), and why:

1.  Decode the BT.2020 R'G'B' code value to display light in cd/m2 (nits):
    the PQ EOTF (SMPTE ST 2084 / ITU-R BT.2100 Table 4) or, for HLG, the
    inverse OETF followed by the BT.2100 OOTF for the nominal display peak.
2.  Desaturate highlights above HDR reference white. BT.2408 puts graphics
    and diffuse white at 203 cd/m2; what is brighter than that is a specular
    or emissive highlight, and squeezing it into 100 nits while keeping full
    chroma turns fire and neon into flat saturated blobs. Mixing toward the
    pixel's own luminance keeps luminance exact while trading colour for
    brightness, which is what a camera does as it approaches clipping.
3.  Tone-map max(R,G,B) with the chosen curve and scale all three channels
    by the same ratio. Mapping the largest channel (as FFmpeg's ``tonemap``
    filter does) guarantees no channel exceeds white, and a common ratio
    keeps hue, where per-channel curves shift skin toward yellow.
    BT.2390's EETF (ITU-R BT.2390-x section 5.4) is the default: it leaves
    everything below the knee untouched and rolls the rest off in the PQ
    domain, which is perceptually uniform, so the roll-off looks even.
4.  Convert BT.2020 primaries to BT.709 (the RGB-to-RGB matrix of ITU-R
    BT.2087 Annex; computed here from the primaries, not copied). Colours
    outside BT.709 come out with negative components; they are compressed
    softly toward the achromatic axis with the ACES Reference Gamut
    Compression curve, so a saturated 2020 green fades into the 709 gamut
    instead of clipping into a flat band.
5.  Encode for an SDR display with the inverse BT.1886 EOTF (gamma 2.4 with
    zero black; ITU-R BT.1886 Annex 1). The output is display-referred, so
    inverting the *display* curve is what reproduces the tone-mapped light;
    the BT.709 camera OETF would brighten shadows.

Code values here are normalised: R'G'B' 0..1 means full-range, and callers
convert to and from limited-range Y'CbCr themselves.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------ PQ (ST 2084)
#
# Constants exactly as ITU-R BT.2100-2 Table 4 gives them.

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0
PQ_PEAK_NITS = 10000.0

#: ITU-R BT.2408: HDR reference white for graphics and diffuse white. It is
#: 58% PQ and 75% HLG, and it is where subtitles belong on an HDR picture.
REFERENCE_WHITE_NITS = 203.0

#: A PQ source that declares no peak is assumed graded on a 1000-nit display,
#: the commonest mastering monitor (and the HLG nominal peak in BT.2100).
DEFAULT_SOURCE_PEAK_NITS = 1000.0


def pq_eotf(signal) -> np.ndarray:
    """PQ code value E' in 0..1 -> display luminance in cd/m2 (0..10000)."""
    e = np.clip(np.asarray(signal, dtype=np.float64), 0.0, 1.0)
    p = e ** (1.0 / PQ_M2)
    return PQ_PEAK_NITS * (np.maximum(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p)) ** (1.0 / PQ_M1)


def pq_inverse_eotf(nits) -> np.ndarray:
    """Display luminance in cd/m2 -> PQ code value E' in 0..1."""
    y = np.clip(np.asarray(nits, dtype=np.float64) / PQ_PEAK_NITS, 0.0, 1.0)
    ym = y ** PQ_M1
    return ((PQ_C1 + PQ_C2 * ym) / (1.0 + PQ_C3 * ym)) ** PQ_M2


# ------------------------------------------------------------------- HLG
#
# ITU-R BT.2100-2 Table 5. HLG is scene-referred: the signal describes scene
# light, and the display applies an OOTF whose system gamma depends on its
# own peak -- so the same HLG picture is brighter, but not linearly so, on a
# brighter display.

HLG_A = 0.17883277
HLG_B = 1.0 - 4.0 * HLG_A
HLG_C = 0.5 - HLG_A * math.log(4.0 * HLG_A)


def hlg_oetf(scene) -> np.ndarray:
    """Normalised scene light E (0..1) -> HLG signal E' (0..1)."""
    e = np.clip(np.asarray(scene, dtype=np.float64), 0.0, 1.0)
    low = np.sqrt(3.0 * e)
    high = HLG_A * np.log(np.maximum(12.0 * e - HLG_B, 1e-12)) + HLG_C
    return np.where(e <= 1.0 / 12.0, low, high)


def hlg_inverse_oetf(signal) -> np.ndarray:
    """HLG signal E' (0..1) -> normalised scene light E (0..1)."""
    e = np.clip(np.asarray(signal, dtype=np.float64), 0.0, 1.0)
    low = e * e / 3.0
    high = (np.exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0
    return np.where(e <= 0.5, low, high)


def hlg_system_gamma(peak_nits: float) -> float:
    """System gamma of the HLG OOTF for a display of ``peak_nits``.

    BT.2100 Table 5 note 5f gives 1.2 + 0.42 log10(Lw/1000) for 400-2000
    cd/m2; ITU-R BT.2390 section 6.2 gives the extended form
    1.2 * 1.111^log2(Lw/1000) for displays outside that range.
    """
    lw = max(float(peak_nits), 1.0)
    if 400.0 <= lw <= 2000.0:
        return 1.2 + 0.42 * math.log10(lw / 1000.0)
    return 1.2 * 1.111 ** math.log2(lw / 1000.0)


def hlg_ootf(scene_rgb, peak_nits: float = DEFAULT_SOURCE_PEAK_NITS) -> np.ndarray:
    """Scene-linear BT.2020 RGB (0..1) -> display light in cd/m2.

    BT.2100: F_D = alpha * Y_S^(gamma - 1) * E, with Y_S the BT.2020
    luminance of the scene light and alpha the display's peak (black level
    taken as zero, so beta = 0).
    """
    rgb = np.asarray(scene_rgb, dtype=np.float64)
    ys = np.maximum(rgb @ BT2020_LUMA, 0.0)
    gamma = hlg_system_gamma(peak_nits)
    gain = np.where(ys > 0, np.power(np.maximum(ys, 1e-12), gamma - 1.0), 0.0)
    return float(peak_nits) * rgb * gain[..., None]


def hlg_eotf(signal_rgb, peak_nits: float = DEFAULT_SOURCE_PEAK_NITS) -> np.ndarray:
    """HLG R'G'B' (0..1) -> display light in cd/m2, per BT.2100."""
    return hlg_ootf(hlg_inverse_oetf(signal_rgb), peak_nits)


# ----------------------------------------------------------- SDR BT.1886

SDR_GAMMA = 2.4


def bt1886_eotf(signal, gamma: float = SDR_GAMMA) -> np.ndarray:
    """SDR R'G'B' (0..1) -> relative display light (0..1), zero black."""
    return np.clip(np.asarray(signal, dtype=np.float64), 0.0, 1.0) ** gamma


def bt1886_inverse_eotf(light, gamma: float = SDR_GAMMA) -> np.ndarray:
    """Relative display light (0..1) -> SDR R'G'B' (0..1)."""
    return np.clip(np.asarray(light, dtype=np.float64), 0.0, 1.0) ** (1.0 / gamma)


# -------------------------------------------------------------- primaries

D65 = (0.3127, 0.3290)
BT709_PRIMARIES = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060))
BT2020_PRIMARIES = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))

#: Luma coefficients (Kr, Kg, Kb): BT.709 Table 3 item 3.2; BT.2020 Table 4.
BT709_LUMA = np.array([0.2126, 0.7152, 0.0722])
BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593])


def rgb_to_xyz_matrix(primaries: Sequence[Tuple[float, float]], white: Tuple[float, float] = D65) -> np.ndarray:
    """The normalised primary matrix (SMPTE RP 177) for a set of primaries."""
    xyz = np.array([[x / y, 1.0, (1.0 - x - y) / y] for x, y in primaries]).T
    wx, wy = white
    white_xyz = np.array([wx / wy, 1.0, (1.0 - wx - wy) / wy])
    scale = np.linalg.solve(xyz, white_xyz)
    return xyz * scale


#: Linear BT.2020 RGB -> linear BT.709 RGB. Matches ITU-R BT.2087-0 Annex
#: equation (the 1.6605 / -0.5876 / -0.0728 ... matrix) to four decimals.
BT2020_TO_BT709 = np.linalg.inv(rgb_to_xyz_matrix(BT709_PRIMARIES)) @ rgb_to_xyz_matrix(BT2020_PRIMARIES)
BT709_TO_BT2020 = np.linalg.inv(BT2020_TO_BT709)


# ------------------------------------------------------------ tone curves

TONE_CURVES = ("bt2390", "hable", "mobius", "reinhard", "clip")


def bt2390_eetf(
    nits,
    source_peak: float,
    target_peak: float,
    source_black: float = 0.0,
    target_black: float = 0.0,
) -> np.ndarray:
    """ITU-R BT.2390-x section 5.4.1 EETF, returning display light in cd/m2.

    Works on PQ code values normalised to the source's black..peak range.
    Below the knee KS = 1.5 * maxLum - 0.5 the signal passes unchanged (so a
    100-nit source shown on a 100-nit target is untouched); above it a
    Hermite spline bends the rest so the source peak lands exactly on the
    target peak. The (1 - E2)^4 term lifts black to the target's black.
    """
    src_hi = float(pq_inverse_eotf(max(source_peak, 1e-6)))
    src_lo = float(pq_inverse_eotf(max(source_black, 0.0)))
    span = max(src_hi - src_lo, 1e-9)
    e1 = (pq_inverse_eotf(nits) - src_lo) / span
    e1 = np.clip(e1, 0.0, 1.0)
    min_lum = (float(pq_inverse_eotf(max(target_black, 0.0))) - src_lo) / span
    max_lum = (float(pq_inverse_eotf(target_peak)) - src_lo) / span
    if max_lum >= 1.0:
        # The display is at least as bright as the master: nothing to roll off.
        e2 = e1
    else:
        ks = 1.5 * max_lum - 0.5
        t = (e1 - ks) / max(1.0 - ks, 1e-9)
        t2, t3 = t * t, t * t * t
        spline = (2 * t3 - 3 * t2 + 1) * ks + (t3 - 2 * t2 + t) * (1.0 - ks) + (-2 * t3 + 3 * t2) * max_lum
        e2 = np.where(e1 < ks, e1, spline)
    if min_lum > 0:
        e2 = e2 + min_lum * (1.0 - e2) ** 4
    e4 = e2 * span + src_lo
    return pq_eotf(e4)


def _hable(x):
    # John Hable's Uncharted 2 filmic curve, with the constants FFmpeg's
    # ``tonemap`` filter and mpv use.
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return (x * (x * a + c * b) + d * e) / (x * (x * a + b) + d * f) - e / f


def tone_curve(nits, algorithm: str, source_peak: float, target_peak: float) -> np.ndarray:
    """Map display light (cd/m2) to 0..1 of the target display's peak.

    hable / mobius / reinhard follow FFmpeg's vf_tonemap.c (mobius knee 0.3,
    reinhard offset 1.0) on light normalised to the target peak, and divide
    by the curve's value at the source peak so the source peak lands on
    target white, which is how the filter behaves with ``peak`` set.
    """
    algorithm = (algorithm or "bt2390").lower()
    target_peak = max(float(target_peak), 1e-6)
    source_peak = max(float(source_peak), target_peak)
    light = np.maximum(np.asarray(nits, dtype=np.float64), 0.0)
    x = light / target_peak
    peak = source_peak / target_peak
    if algorithm == "bt2390":
        return np.clip(bt2390_eetf(light, source_peak, target_peak) / target_peak, 0.0, 1.0)
    if algorithm == "clip" or peak <= 1.0 + 1e-9:
        return np.clip(x, 0.0, 1.0)
    x = np.minimum(x, peak)
    if algorithm == "hable":
        return np.clip(_hable(x) / _hable(peak), 0.0, 1.0)
    if algorithm == "mobius":
        j = 0.3
        a = -j * j * (peak - 1.0) / (j * j - 2.0 * j + peak)
        b = (j * j - 2.0 * j * peak + peak) / max(peak - 1.0, 1e-6)
        rolled = (b * b + 2.0 * b * j + j * j) / (b - a) * (x + a) / (x + b)
        return np.clip(np.where(x <= j, x, rolled), 0.0, 1.0)
    if algorithm == "reinhard":
        offset = 1.0
        return np.clip(x / (x + offset) * (peak + offset) / peak, 0.0, 1.0)
    raise ValueError(f"Unknown tone curve: {algorithm}")


# --------------------------------------------------------- gamut handling

def desaturate_highlights(rgb_nits: np.ndarray, strength: float, luma=BT2020_LUMA,
                          threshold_nits: float = REFERENCE_WHITE_NITS) -> np.ndarray:
    """Pull colours brighter than reference white toward their own grey.

    ``strength`` 0..1. The mix amount is ``strength * (1 - threshold / Y)``:
    zero at reference white, half the strength at twice it, approaching the
    full strength for the brightest speculars. Mixing toward Y keeps the
    pixel's luminance exactly, so the tone curve sees the same brightness.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    rgb = np.asarray(rgb_nits, dtype=np.float64)
    if strength <= 0:
        return rgb
    y = np.maximum(rgb @ luma, 0.0)
    amount = strength * np.clip(1.0 - threshold_nits / np.maximum(y, 1e-9), 0.0, 1.0)
    return rgb + (y[..., None] - rgb) * amount[..., None]


# ACES Reference Gamut Compression (Academy, ACES 1.3, 2021): distances from
# the achromatic axis beyond THRESHOLD are compressed so that LIMIT lands on
# the gamut boundary. The limits are measured here for BT.2020 -> BT.709, so
# the whole 2020 gamut maps inside 709 and nothing is hard-clipped.
GAMUT_THRESHOLD = 0.9
GAMUT_POWER = 1.2


def _max_distances(matrix: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 33)
    u, v = np.meshgrid(grid, grid)
    u, v = u.ravel(), v.ravel()
    faces = []
    for axis in range(3):
        for value in (0.0, 1.0):
            face = np.zeros((u.size, 3))
            others = [a for a in range(3) if a != axis]
            face[:, axis] = value
            face[:, others[0]] = u
            face[:, others[1]] = v
            faces.append(face)
    samples = np.concatenate(faces) @ matrix.T
    ach = samples.max(axis=1, keepdims=True)
    keep = ach[:, 0] > 1e-6
    dist = (ach[keep] - samples[keep]) / ach[keep]
    return dist.max(axis=0)


GAMUT_LIMITS = np.maximum(_max_distances(BT2020_TO_BT709) * 1.0001, GAMUT_THRESHOLD + 0.05)


def compress_gamut(rgb: np.ndarray, limits=GAMUT_LIMITS, threshold: float = GAMUT_THRESHOLD,
                   power: float = GAMUT_POWER) -> np.ndarray:
    """Softly bring negative (out-of-gamut) components back to >= 0.

    Distance from the achromatic axis is ``(max - c) / max`` per channel;
    1.0 is the gamut edge. Distances below ``threshold`` are untouched, and
    the range threshold..limit is squeezed into threshold..1 with the ACES
    RGC power curve, which is smooth at the threshold (no visible band).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    ach = rgb.max(axis=-1, keepdims=True)
    safe = np.where(ach > 1e-9, ach, 1.0)
    dist = np.where(ach > 1e-9, (ach - rgb) / safe, 0.0)
    lim = np.asarray(limits, dtype=np.float64)
    scale = (lim - threshold) / np.power(np.power((1.0 - threshold) / (lim - threshold), -power) - 1.0, 1.0 / power)
    nd = np.maximum(dist - threshold, 0.0) / scale
    compressed = threshold + scale * nd / np.power(1.0 + np.power(nd, power), 1.0 / power)
    dist = np.where(dist < threshold, dist, compressed)
    out = ach - dist * safe
    return np.where(ach > 1e-9, out, rgb)


# ------------------------------------------------------------- HDR -> SDR

def hdr_to_display_light(signal_rgb, transfer: str, source_peak: float) -> np.ndarray:
    """BT.2020 R'G'B' (0..1) -> BT.2020 display light in cd/m2."""
    transfer = (transfer or "pq").lower()
    if transfer in ("hlg", "arib-std-b67"):
        return hlg_eotf(signal_rgb, source_peak)
    rgb = np.asarray(signal_rgb, dtype=np.float64)
    return pq_eotf(rgb)


def hdr_to_sdr(
    signal_rgb,
    transfer: str = "pq",
    algorithm: str = "bt2390",
    source_peak: float = DEFAULT_SOURCE_PEAK_NITS,
    target_peak: float = 100.0,
    desaturation: float = 0.5,
    sdr_gamma: float = SDR_GAMMA,
) -> np.ndarray:
    """BT.2020 PQ/HLG R'G'B' (0..1, shape (..., 3)) -> BT.709 SDR R'G'B'.

    See the module docstring for the five steps. ``source_peak`` is the
    mastering peak for PQ, the nominal display peak for HLG.
    """
    light = hdr_to_display_light(signal_rgb, transfer, source_peak)
    light = desaturate_highlights(light, desaturation)
    sig = np.maximum(light.max(axis=-1), 1e-9)
    mapped = tone_curve(sig, algorithm, source_peak, target_peak)
    rel = light / target_peak * (mapped / (sig / target_peak))[..., None]
    rel = np.maximum(rel, 0.0) @ BT2020_TO_BT709.T
    rel = compress_gamut(rel)
    # The matrix can push the largest channel past white (a 2020 red is
    # 1.66 in 709); scaling the pixel keeps its hue where clipping would not.
    top = rel.max(axis=-1, keepdims=True)
    rel = np.where(top > 1.0, rel / np.maximum(top, 1e-9), rel)
    return bt1886_inverse_eotf(np.clip(rel, 0.0, 1.0), sdr_gamma)


# ---------------------------------------------------- Y'CbCr code values
#
# Integer code values per ITU-R BT.709 / BT.2020 / BT.2100: narrow ("TV",
# limited) range puts black at 16 * 2^(n-8) and white at 235 * 2^(n-8);
# full range spans 0..2^n - 1 (BT.2100 Table 9).


def ycbcr_to_rgb(y, cb, cr, luma=BT709_LUMA, bits: int = 8, full_range: bool = False) -> np.ndarray:
    """Y'CbCr code values -> normalised R'G'B' (0..1 nominal, unclipped)."""
    y = np.asarray(y, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    cr = np.asarray(cr, dtype=np.float64)
    mid = float(1 << (bits - 1))
    if full_range:
        top = float((1 << bits) - 1)
        yn, cbn, crn = y / top, (cb - mid) / top, (cr - mid) / top
    else:
        scale = float(1 << (bits - 8))
        yn = (y - 16.0 * scale) / (219.0 * scale)
        cbn = (cb - mid) / (224.0 * scale)
        crn = (cr - mid) / (224.0 * scale)
    kr, kg, kb = luma
    r = yn + 2.0 * (1.0 - kr) * crn
    b = yn + 2.0 * (1.0 - kb) * cbn
    g = (yn - kr * r - kb * b) / kg
    return np.stack([r, g, b], axis=-1)


def rgb_to_ycbcr_codes(rgb, luma=BT709_LUMA, bits: int = 8, full_range: bool = False):
    """Normalised R'G'B' -> *unrounded* Y'CbCr code values (floats)."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    kr, _kg, kb = luma
    yn = rgb @ np.asarray(luma)
    cbn = (rgb[..., 2] - yn) / (2.0 * (1.0 - kb))
    crn = (rgb[..., 0] - yn) / (2.0 * (1.0 - kr))
    mid = float(1 << (bits - 1))
    if full_range:
        top = float((1 << bits) - 1)
        return yn * top, cbn * top + mid, crn * top + mid
    scale = float(1 << (bits - 8))
    return yn * 219.0 * scale + 16.0 * scale, cbn * 224.0 * scale + mid, crn * 224.0 * scale + mid


def rgb_to_ycbcr(rgb, luma=BT709_LUMA, bits: int = 8, full_range: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised R'G'B' -> rounded integer Y'CbCr, clamped to the range."""
    y, cb, cr = rgb_to_ycbcr_codes(rgb, luma, bits, full_range)
    scale = float(1 << (bits - 8))
    if full_range:
        lo_y, hi_y, lo_c, hi_c = 0.0, float((1 << bits) - 1), 0.0, float((1 << bits) - 1)
    else:
        lo_y, hi_y, lo_c, hi_c = 16.0 * scale, 235.0 * scale, 16.0 * scale, 240.0 * scale
    return (
        np.clip(np.rint(y), lo_y, hi_y).astype(np.int64),
        np.clip(np.rint(cb), lo_c, hi_c).astype(np.int64),
        np.clip(np.rint(cr), lo_c, hi_c).astype(np.int64),
    )


# ---------------------------------------------------------------- 3D LUTs

LUT_SIZES = (17, 33, 65)


def build_rgb_lut(
    size: int = 65,
    transfer: str = "pq",
    algorithm: str = "bt2390",
    source_peak: float = DEFAULT_SOURCE_PEAK_NITS,
    target_peak: float = 100.0,
    desaturation: float = 0.5,
) -> np.ndarray:
    """A ``(size, size, size, 3)`` table, indexed ``[r, g, b]``, taking
    BT.2020 PQ/HLG R'G'B' (0..1) to SDR BT.709 R'G'B' (0..1).

    For grading tools and players that feed a LUT RGB; the FFmpeg engine
    uses the same grid through ``build_tonemap_lut``.
    """
    if size not in LUT_SIZES:
        raise ValueError(f"LUT size must be one of {LUT_SIZES}")
    axis = np.linspace(0.0, 1.0, size)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([r, g, b], axis=-1)
    return hdr_to_sdr(grid, transfer, algorithm, source_peak, target_peak, desaturation)


def ycbcr_to_rgb_lut(bits: int = 10, full_range: bool = False, luma=BT2020_LUMA, output_bias: float = 0.5) -> np.ndarray:
    """A 2-point LUT that converts raw Y'CbCr code values to R'G'B' codes.

    Channels are arranged as FFmpeg's ``mergeplanes`` delivers a yuv444p10
    frame relabelled gbrp10 -- R holds Cr, G holds Y', B holds Cb -- and the
    output is ordinary R'G'B'. Tetrahedral interpolation reproduces an
    affine map exactly, so two nodes per axis are the whole matrix, with
    nothing to cache-miss on. Node values fall outside 0..1 (a Y'CbCr cube
    corner is no colour at all); lut3d clips its *result*, which is the
    clipping to legal R'G'B' that a matrix conversion does anyway.

    Why not FFmpeg's scaler: measured on FFmpeg 9.0.1, swscale converts
    10-bit Y'=940 to 3-4 codes short of full scale in every RGB format
    (gbrp10/12/16, gbrpf32) and R'G'B' white back to Y'=943, not 940.
    """
    top = float((1 << bits) - 1)
    corners = np.array([0.0, top])
    cr, y, cb = np.meshgrid(corners, corners, corners, indexing="ij")
    rgb = ycbcr_to_rgb(y, cb, cr, luma, bits, full_range)
    # lut3d truncates to integer (vf_lut3d.c: av_clip_uintp2(v * max, depth)),
    # so half a code makes it round.
    return rgb + output_bias / top


def build_tonemap_lut(
    size: int = 65,
    transfer: str = "pq",
    algorithm: str = "bt2390",
    source_peak: float = DEFAULT_SOURCE_PEAK_NITS,
    target_peak: float = 100.0,
    desaturation: float = 0.5,
    bits: int = 10,
    output_bias: float = 0.5,
) -> np.ndarray:
    """BT.2020 PQ/HLG R'G'B' (0..1) -> BT.709 SDR as limited-range Y'CbCr
    code values, arranged (Cr, Y', Cb) for ``mergeplanes`` back to yuv444.

    The grid is in R'G'B' because each channel goes through the steep PQ
    curve on its own: a grid over R'G'B' follows that, and the max(R,G,B)
    kinks of the tone curve lie exactly on the planes (r = g, g = b, r = b)
    that tetrahedral interpolation splits cubes along. Measured against the
    direct formula over 20 000 colours at 65 points: luma error median 0.08
    and p90 0.37 10-bit codes, p99 ~4 (all of it above 100 nits, in the
    compressed highlights) -- about a third of a grid over Y'CbCr. Writing
    Y'CbCr straight out is free: interpolation commutes with the linear
    R'G'B' -> Y'CbCr step, and it spares another conversion.
    """
    if size not in LUT_SIZES:
        raise ValueError(f"LUT size must be one of {LUT_SIZES}")
    top = float((1 << bits) - 1)
    sdr = build_rgb_lut(size, transfer, algorithm, source_peak, target_peak, desaturation)
    y, cb, cr = rgb_to_ycbcr_codes(sdr, BT709_LUMA, bits, full_range=False)
    return np.clip((np.stack([cr, y, cb], axis=-1) + output_bias) / top, 0.0, 1.0)


def apply_lut(lut: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Tetrahedral interpolation as FFmpeg's lut3d does it (for tests and
    for predicting what a LUT will output). ``coords`` are 0..1."""
    size = lut.shape[0] - 1
    p = np.clip(np.asarray(coords, dtype=np.float64), 0.0, 1.0) * size
    i = np.minimum(np.floor(p).astype(int), size - 1)
    f = p - i
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    a, b, c = i[..., 0], i[..., 1], i[..., 2]

    def node(x: int, y: int, z: int) -> np.ndarray:
        return lut[a + x, b + y, c + z]

    c000, c111 = node(0, 0, 0), node(1, 1, 1)
    rg, gb, rb = fr > fg, fg > fb, fr > fb
    cases = [
        (rg & gb, lambda: (1 - fr) * c000 + (fr - fg) * node(1, 0, 0) + (fg - fb) * node(1, 1, 0) + fb * c111),
        (rg & ~gb & rb, lambda: (1 - fr) * c000 + (fr - fb) * node(1, 0, 0) + (fb - fg) * node(1, 0, 1) + fg * c111),
        (rg & ~gb & ~rb, lambda: (1 - fb) * c000 + (fb - fr) * node(0, 0, 1) + (fr - fg) * node(1, 0, 1) + fg * c111),
        (~rg & ~gb, lambda: (1 - fb) * c000 + (fb - fg) * node(0, 0, 1) + (fg - fr) * node(0, 1, 1) + fr * c111),
        (~rg & gb & ~rb, lambda: (1 - fg) * c000 + (fg - fb) * node(0, 1, 0) + (fb - fr) * node(0, 1, 1) + fr * c111),
        (~rg & gb & rb, lambda: (1 - fg) * c000 + (fg - fr) * node(0, 1, 0) + (fr - fb) * node(1, 1, 0) + fb * c111),
    ]
    out = np.empty(p.shape, dtype=np.float64)
    for mask, value in cases:
        m = mask[..., 0]
        out[m] = value()[m]
    return out


def write_cube(path: str, lut: np.ndarray, title: Optional[str] = None, clip: bool = True) -> str:
    """Write an Adobe/Resolve ``.cube`` file (red index varies fastest).

    FFmpeg's lut3d reads this order (vf_lut3d.c parse_cube). ``clip=False``
    keeps node values outside 0..1, which lut3d accepts and the affine
    Y'CbCr LUT needs. Values are written with 9 decimals.
    """
    size = lut.shape[0]
    # [r, g, b] -> rows ordered b-major, r fastest.
    rows = lut.transpose(2, 1, 0, 3).reshape(-1, 3)
    if clip:
        rows = np.clip(rows, 0.0, 1.0)
    header = []
    if title:
        header.append(f'TITLE "{title}"')
    header += [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(header) + "\n")
        np.savetxt(handle, rows, fmt="%.9f")
    return path


def read_cube(path: str) -> np.ndarray:
    """Read a ``.cube`` back to ``[r, g, b]`` order (ignores the domain)."""
    size = None
    values = []
    with open(path, "r", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if line[0].isalpha():
                continue
            values.append([float(v) for v in line.split()])
    if size is None:
        raise ValueError("Not a 3D .cube file")
    return np.asarray(values).reshape(size, size, size, 3).transpose(2, 1, 0, 3)
