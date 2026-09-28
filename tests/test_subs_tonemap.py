"""Tests for HDR -> SDR tone-mapping: the colour maths, the plan, and real
FFmpeg runs on synthetic HDR clips whose patch luminances are known.

The end-to-end checks decode the output and measure the patches, because a
filter graph that "looks right" can still be off: that is how the swscale
gain error, the lut3d truncation and the mastering metadata leaking into
SDR outputs were found (see tonemap.py / colorimetry.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from audiosync.media import Cancelled, CancellationToken, ffmpeg_path, ffprobe_path  # noqa: E402
from audiosync.subs import colorimetry as cm  # noqa: E402
from audiosync.subs import tasks, tonemap  # noqa: E402
from audiosync.subs.tasks import TaskContext, TaskError  # noqa: E402

#: Patch luminances of the synthetic HDR10 clip, in nits.
PQ_NITS = [0.0, 0.1, 10.0, 100.0, 203.0, 1000.0, 4000.0, 10000.0]
#: HLG signal levels of the synthetic HLG clip (75% = BT.2408 reference white).
HLG_LEVELS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
PATCH_W, CLIP_H = 48, 64

#: Filled in by the end-to-end tests and printed by the runner.
MEASURED: dict = {}


class Workspace:
    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="subsync-tonemap-")
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.root, name)


def _have_x265() -> bool:
    try:
        return bool(tonemap.ffmpeg_capabilities(ffmpeg_path())["encoders"].get("libx265"))
    except Exception:  # noqa: BLE001
        return False


def _make_clip(path, codes, transfer, max_cll=None, mastering_nits=None, frames=6):
    """Grey patches of 10-bit Y' ``codes`` (Cb = Cr = 512), lossless HEVC."""
    width = PATCH_W * len(codes)
    y = np.zeros((CLIP_H, width), np.uint16)
    for i, code in enumerate(codes):
        y[:, i * PATCH_W:(i + 1) * PATCH_W] = int(code)
    chroma = np.full((CLIP_H // 2, width // 2), 512, np.uint16)
    frame = y.tobytes() + chroma.tobytes() + chroma.tobytes()
    trc = "smpte2084" if transfer == "pq" else "arib-std-b67"
    params = [f"log-level=error", "lossless=1", "colorprim=bt2020", f"transfer={trc}",
              "colormatrix=bt2020nc", "range=limited"]
    if mastering_nits:
        params.append("master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"
                      f"L({int(mastering_nits * 10000)},50)")
    if max_cll:
        params += [f"max-cll={int(max_cll)},400", "hdr10=1"]
    command = [ffmpeg_path(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
               "-s", f"{width}x{CLIP_H}", "-r", "24", "-i", "-",
               # The raw input has no colour tags; setparams gives the frames
               # (and so the container) the right ones.
               "-vf", f"setparams=color_primaries=bt2020:color_trc={trc}:colorspace=bt2020nc:range=tv",
               "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", ":".join(params), path]
    subprocess.run(command, input=frame * frames, check=True, capture_output=True)
    return path


def _pq_codes(nits):
    return [int(round(64 + 876 * float(cm.pq_inverse_eotf(n)))) for n in nits]


def _hlg_codes(levels):
    return [int(round(64 + 876 * v)) for v in levels]


def _measure(path, n_patches):
    """Mean luma of each patch's centre, in 8-bit code values."""
    info = json.loads(subprocess.run(
        [ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt,width,height", "-of", "json", path], capture_output=True, check=True).stdout)["streams"][0]
    w, h = info["width"], info["height"]
    ten = "10" in info["pix_fmt"]
    raw = subprocess.run([ffmpeg_path(), "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "yuv420p10le" if ten else "yuv420p", "-"], capture_output=True, check=True).stdout
    plane = np.frombuffer(raw, np.uint16 if ten else np.uint8)[: w * h].reshape(h, w).astype(float)
    plane /= 4.0 if ten else 1.0
    pw = w / n_patches
    return [float(plane[h // 4: h - h // 4, int(i * pw + pw / 4): int((i + 1) * pw - pw / 4)].mean())
            for i in range(n_patches)]


def _run_job(clip, out_dir, **options):
    job = {"id": "t", "task": "tonemap", "input": {"video": {"path": clip}},
           "options": {"resolution": "source", "crf": 0, "preset": "ultrafast", **options},
           "output": {"dir": out_dir}}
    events = []
    result = tasks.run_job(job, CancellationToken(), lambda p, s: events.append((p, s)), lambda m: None)
    return result, events


# ------------------------------------------------------------ colour maths


def test_pq_round_trip_and_reference_points():
    nits = np.array([0.0, 0.1, 1.0, 100.0, 203.0, 1000.0, 10000.0])
    assert np.allclose(cm.pq_eotf(cm.pq_inverse_eotf(nits)), nits, rtol=1e-9, atol=1e-9)
    # BT.2408: 203 nits is 58% PQ; 100 nits is 50.8%; 10000 is 100%.
    assert abs(float(cm.pq_inverse_eotf(203)) - 0.5807) < 5e-4
    assert abs(float(cm.pq_inverse_eotf(100)) - 0.5081) < 5e-4
    assert abs(float(cm.pq_inverse_eotf(10000)) - 1.0) < 1e-12


def test_hlg_reference_white_is_203_nits():
    # BT.2408: 75% HLG on a 1000-nit display is the 203-nit reference white.
    nits = cm.hlg_eotf(np.array([0.75, 0.75, 0.75]), 1000.0)
    assert abs(float(nits[0]) - 203.0) < 1.0, nits
    assert abs(cm.hlg_system_gamma(1000) - 1.2) < 1e-12
    signal = np.linspace(0, 1, 11)
    assert np.allclose(cm.hlg_oetf(cm.hlg_inverse_oetf(signal)), signal, atol=1e-9)


def test_bt2020_to_bt709_matrix_matches_bt2087():
    expected = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])
    assert np.allclose(cm.BT2020_TO_BT709, expected, atol=6e-5), cm.BT2020_TO_BT709
    assert np.allclose(cm.BT2020_TO_BT709.sum(axis=1), 1.0), "white must stay white"


def test_bt2390_eetf_properties():
    nits = np.geomspace(0.01, 1000, 200)
    # Target as bright as the source: identity.
    assert np.allclose(cm.bt2390_eetf(nits, 1000, 1000), nits, rtol=1e-9)
    out = cm.bt2390_eetf(nits, 1000, 100)
    assert np.all(np.diff(out) >= -1e-9), "must be monotonic"
    assert abs(float(out[-1]) - 100.0) < 1e-6, "source peak must land on target peak"
    # Below the knee (KS = 1.5 * maxLum - 0.5 in PQ) the signal is untouched.
    assert abs(float(cm.bt2390_eetf(10.0, 1000, 100)) - 10.0) < 1e-9
    for curve in cm.TONE_CURVES:
        mapped = cm.tone_curve(nits, curve, 1000, 100)
        assert np.all(np.diff(mapped) >= -1e-9), curve
        assert 0.0 <= mapped.min() and mapped.max() <= 1.0 + 1e-12, curve


def test_gamut_compression_keeps_every_2020_colour_in_709():
    rng = np.random.default_rng(1)
    samples = np.concatenate([np.eye(3), 1 - np.eye(3), rng.random((2000, 3))])
    out = cm.compress_gamut(samples @ cm.BT2020_TO_BT709.T)
    assert out.min() >= -1e-9, out.min()
    # Colours well inside 709 are left alone.
    inside = np.array([[0.5, 0.45, 0.4], [0.2, 0.2, 0.2]])
    assert np.allclose(cm.compress_gamut(inside), inside)


def test_matrix_lut_is_exact_and_tonemap_lut_is_accurate():
    # The 2-point Y'CbCr -> R'G'B' LUT reproduces the BT.2020 matrix exactly.
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1024, (2000, 3)).astype(float)  # (Cr, Y, Cb)
    matrix = cm.ycbcr_to_rgb_lut(output_bias=0.0)
    exact = cm.ycbcr_to_rgb(codes[:, 1], codes[:, 2], codes[:, 0], cm.BT2020_LUMA, 10)
    assert np.abs(cm.apply_lut(matrix, codes / 1023) - exact).max() < 1e-9

    # The tone-map LUT against the direct formula, on greys and on colours,
    # after the 10-bit rounding the real chain has between the two LUTs.
    lut = cm.build_tonemap_lut(65, source_peak=1000, output_bias=0.0)
    rgb = rng.random((20000, 3))
    rgb[:8000] = rgb[:8000] * 0.2 + rgb[:8000, :1] * 0.8  # near-neutral
    rgb = np.rint(rgb * 1023) / 1023
    got = cm.apply_lut(lut, rgb) * 1023
    y, cb, cr = cm.rgb_to_ycbcr_codes(cm.hdr_to_sdr(rgb, source_peak=1000), cm.BT709_LUMA, 10)
    error = np.abs(got[:, 1] - y)
    # Measured: median 0.08, p90 0.37, p99 ~4 10-bit codes; the tail is all
    # above 100 nits, where curve knee, desaturation and gamut compression
    # bend hardest -- about one 8-bit code in compressed highlights.
    assert np.median(error) < 0.2 and np.percentile(error, 90) < 0.6, np.percentile(error, [50, 90])
    assert np.percentile(error, 99) < 5.0, np.percentile(error, 99)
    chroma = np.maximum(np.abs(got[:, 0] - cr), np.abs(got[:, 2] - cb))
    assert np.percentile(chroma, 99) < 3.0, np.percentile(chroma, 99)
    greys = np.repeat(np.linspace(0, 1, 1024)[:, None], 3, axis=1)
    grey_out = cm.apply_lut(lut, greys) * 1023
    expected = 64 + 876 * cm.tone_curve(cm.pq_eotf(greys[:, 0]), "bt2390", 1000, 100) ** (1 / 2.4)
    # The SDR encode x^(1/2.4) is near-vertical at black, so the first grid
    # step (below 0.01 nit) is off by up to 1.6 10-bit codes -- under half an
    # 8-bit code. From 0.1 nit up it is well under a code.
    grey_error = np.abs(grey_out[:, 1] - expected)
    assert grey_error.max() < 2.0, grey_error.max()
    assert grey_error[greys[:, 0] >= float(cm.pq_inverse_eotf(0.1))].max() < 0.6
    assert np.abs(grey_out[:, 0] - 512).max() < 0.01 and np.abs(grey_out[:, 2] - 512).max() < 0.01


def test_rgb_lut_for_grading_tools():
    lut = cm.build_rgb_lut(33, source_peak=1000)
    diagonal = lut[np.arange(33), np.arange(33), np.arange(33)]
    signal = np.linspace(0, 1, 33)
    expected = cm.tone_curve(cm.pq_eotf(signal), "bt2390", 1000, 100) ** (1 / 2.4)
    assert np.allclose(diagonal[:, 0], expected, atol=1e-9) and np.allclose(diagonal, diagonal[:, :1])


def test_cube_file_round_trip():
    with Workspace() as ws:
        lut = cm.build_tonemap_lut(17)
        path = cm.write_cube(ws.path("t.cube"), lut, title="t")
        with open(path) as handle:
            head = handle.read(200)
        assert "LUT_3D_SIZE 17" in head and "DOMAIN_MAX 1.0 1.0 1.0" in head, head
        assert np.abs(cm.read_cube(path) - lut).max() < 1e-6


# ----------------------------------------------------------------- the plan


def _source(**kw):
    base = dict(path="/x/film.mkv", width=3840, height=2160, hdr="hdr10", transfer="pq", duration=60.0)
    base.update(kw)
    return tonemap.Source(**base)


def test_fit_resolution_and_labels():
    assert tonemap.fit_resolution(3840, 2160, "1080p") == (1920, 1080)
    assert tonemap.fit_resolution(3840, 1600, "1080p") == (1920, 800)
    assert tonemap.fit_resolution(3840, 1608, "720p") == (1280, 536)
    assert tonemap.fit_resolution(1920, 1080, "2160p") == (1920, 1080), "never upscale"
    assert tonemap.fit_resolution(1919, 1037, "source") == (1918, 1036), "even sizes"
    assert tonemap.resolution_label(3840, 1600) == "2160p"
    assert tonemap.resolution_label(1920, 800) == "1080p"
    assert tonemap.resolution_label(1280, 536) == "720p"


def test_source_peak_from_metadata():
    peak = tonemap.resolve_source_peak
    assert peak("auto", _source(max_cll=1400, mastering_peak=4000)) == (1400.0, "MaxCLL")
    assert peak("auto", _source(max_cll=6000, mastering_peak=4000)) == (4000.0, "mastering display")
    assert peak("auto", _source(max_cll=0, mastering_peak=1000)) == (1000.0, "mastering display")
    assert peak("auto", _source())[0] == 1000.0
    assert peak(600, _source(max_cll=1400)) == (600.0, "set")
    assert peak("auto", _source(transfer="hlg", hdr="hlg"))[0] == 1000.0


def test_dolby_vision_base_layers():
    assert tonemap.base_layer_transfer({"hdr": "dv", "dvProfile": 5, "dvCompatibility": 0}) == "ipt"
    assert tonemap.base_layer_transfer({"hdr": "dv", "dvProfile": 8, "dvCompatibility": 1}) == "pq"
    assert tonemap.base_layer_transfer({"hdr": "dv", "dvProfile": 8, "dvCompatibility": 4}) == "hlg"
    assert tonemap.base_layer_transfer({"hdr": "dv", "dvProfile": 8, "dvCompatibility": 2}) == "sdr"
    assert tonemap.base_layer_transfer({"hdr": "dv", "dvProfile": 7, "dvCompatibility": 6}) == "pq"
    assert tonemap.base_layer_transfer({"hdr": "hlg", "transfer": "arib-std-b67"}) == "hlg"
    assert tonemap.base_layer_transfer({"hdr": "sdr", "transfer": "bt709"}) == "sdr"


def _with_engines(available, fn, dolby_vision=()):
    """Run ``fn`` with only the ``available`` engines reported as usable, and
    only those in ``dolby_vision`` able to reshape Dolby Vision."""
    originals = tonemap.binary_for, tonemap.can_reshape_dolby_vision

    def binary_for(engine, dolby_vision=False):
        if engine in available and (not dolby_vision or engine in dv):
            return ffmpeg_path(), None
        return None, "not here"

    dv = set(dolby_vision)
    tonemap.binary_for = binary_for
    tonemap.can_reshape_dolby_vision = lambda engine, caps: engine in dv
    try:
        return fn()
    finally:
        tonemap.binary_for, tonemap.can_reshape_dolby_vision = originals


def test_profile_5_needs_a_reshaping_engine_and_says_why():
    dv5 = _source(hdr="dv", transfer="ipt", dv_profile=5, dv_compat=0)
    for engine in ("auto", "lut", "videotoolbox"):
        try:
            _with_engines({"lut", "videotoolbox"}, lambda: tonemap.build_plan({"engine": engine}, dv5))
        except TaskError as exc:
            assert "IPT-PQ-c2" in str(exc) and "libplacebo" in str(exc), str(exc)
        else:
            raise AssertionError(f"profile 5 accepted by {engine} without reshaping")
    try:
        _with_engines({"libplacebo"}, lambda: tonemap.build_plan({"dolbyVision": "baseLayer"}, dv5), {"libplacebo"})
    except TaskError as exc:
        assert "IPT-PQ-c2" in str(exc)
    else:
        raise AssertionError("profile 5 base layer accepted")
    plan = _with_engines({"libplacebo", "lut"}, lambda: tonemap.build_plan({}, dv5), {"libplacebo"})
    assert plan.engine == "libplacebo" and plan.apply_dv
    assert "apply_dolbyvision=1" in tonemap.filter_graph(plan)
    # macOS with the Jellyfin pack: VideoToolbox (Metal) before tonemapx.
    both = {"videotoolbox", "tonemapx"}
    plan = _with_engines(both | {"lut"}, lambda: tonemap.build_plan({}, dv5), both)
    assert plan.engine == "videotoolbox" and plan.apply_dv
    plan = _with_engines({"tonemapx", "lut"}, lambda: tonemap.build_plan({}, dv5), {"tonemapx"})
    assert plan.engine == "tonemapx" and plan.apply_dv
    graph = tonemap.filter_graph(plan)
    assert "tonemapx=tonemap=bt2390" in graph and "apply_dovi=1" in graph and ":peak=" not in graph, graph


def test_auto_engine_order_and_sdr_refusal():
    src = _source()
    assert _with_engines({"lut", "videotoolbox"}, lambda: tonemap.build_plan({}, src)).engine == "lut"
    assert _with_engines({"videotoolbox"}, lambda: tonemap.build_plan({}, src)).engine == "videotoolbox"
    assert _with_engines({"zscale", "lut"}, lambda: tonemap.build_plan({"algorithm": "hable"}, src)).engine == "lut"
    assert _with_engines({"zscale", "tonemapx"}, lambda: tonemap.build_plan({}, src)).engine == "zscale"
    plan = _with_engines({"tonemapx"}, lambda: tonemap.build_plan({}, _source(max_cll=1400, mastering_peak=4000)))
    assert ":peak=400" in tonemap.filter_graph(plan), "tonemapx must get the highest metadata bound"
    try:
        _with_engines({"lut"}, lambda: tonemap.build_plan({}, _source(hdr="sdr", transfer="sdr")))
    except TaskError as exc:
        assert "already SDR" in str(exc)
    else:
        raise AssertionError("SDR input accepted")
    try:
        _with_engines({"lut"}, lambda: tonemap.build_plan({"engine": "zscale"}, src))
    except TaskError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("unavailable engine accepted")


def test_filter_graph_tags_sdr_and_strips_hdr_metadata():
    src = _source()
    plan = _with_engines({"lut"}, lambda: tonemap.build_plan({"engine": "lut"}, src))
    graph = tonemap.filter_graph(plan)
    assert graph.startswith("[0:v:0]scale=w=1920:h=1080") and graph.endswith("[vout]"), graph
    assert "lut3d=file=ycbcr2rgb.cube:interp=tetrahedral,lut3d=file=tonemap.cube:interp=tetrahedral" in graph
    assert "mergeplanes" in graph and "format=gbrp10le" in graph
    assert "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv" in graph
    assert "sidedata=mode=delete:type=MASTERING_DISPLAY_METADATA" in graph
    assert "sidedata=mode=delete:type=CONTENT_LIGHT_LEVEL" in graph
    command = tonemap.ffmpeg_command(plan, "/o/out.mkv")
    assert command[command.index("-color_trc") + 1] == "bt709"
    assert "-progress" in command and command[-1] == "/o/out.mkv"


def test_burn_and_mp4_stream_mapping():
    subs = [
        {"index": 0, "codec": "hdmv_pgs_subtitle", "kind": "image", "default": True},
        {"index": 1, "codec": "subrip", "kind": "text", "default": False},
    ]
    audio = [{"index": 0, "codec": "truehd", "channels": 8}, {"index": 1, "codec": "ac3", "channels": 6}]
    src = _source(subtitles=subs, audio=audio)
    plan = _with_engines({"lut"}, lambda: tonemap.build_plan({"container": "mp4"}, src))
    args, warnings = tonemap.stream_args(plan)
    assert "0:s:1" in args and "mov_text" in args and "0:s:0" not in args, args
    assert args[args.index("-c:a:0") + 1] == "aac" and args[args.index("-c:a:1") + 1] == "copy", args
    assert any("image subtitle" in w for w in warnings) and any("truehd" in w for w in warnings), warnings

    burn = _with_engines({"lut"}, lambda: tonemap.build_plan({"subtitles": "burn"}, src))
    graph = tonemap.filter_graph(burn)
    assert "[1:s:0]setpts=PTS-0.001/TB,scale=w=1920:h=1080:out_color_matrix=bt709" in graph, graph
    assert graph.index("overlay") < graph.index("setparams"), "subtitles go on the SDR picture"
    command = tonemap.ffmpeg_command(burn, "/o/out.mkv")
    assert command.count("-i") == 2, "bitmap subtitles are read from a second input"
    args, _ = tonemap.stream_args(burn)
    assert "0:s?" not in args

    if not tonemap.ffmpeg_capabilities(ffmpeg_path())["filters"].get("subtitles"):
        try:
            _with_engines({"lut"}, lambda: tonemap.build_plan({"subtitles": "burn", "burnTrack": 1}, src))
        except TaskError as exc:
            assert "libass" in str(exc) and "ffmpeg-full" in str(exc)
        else:
            raise AssertionError("text burn accepted without the subtitles filter")


def test_filter_value_escaping():
    escaped = tonemap.escape_filter_value("/Films/It's: a, [test];.mkv")
    # Level 1 turns ' into \' and : into \:, level 2 escapes those backslashes
    # and the graph's own specials.
    assert escaped == "/Films/It\\\\\\'s\\\\: a\\, \\[test\\]\\;.mkv", escaped


def test_ffmpeg_capabilities_shape_and_cache():
    caps = tonemap.ffmpeg_capabilities(ffmpeg_path())
    assert set(caps) >= {"path", "version", "filters", "encoders"}
    for name in ("zscale", "libplacebo", "tonemap", "lut3d", "scale_vt", "overlay", "subtitles", "scdet"):
        assert isinstance(caps["filters"][name], bool), name
    for name in ("libx264", "libx265", "h264_videotoolbox", "hevc_videotoolbox"):
        assert isinstance(caps["encoders"][name], bool), name
    assert tonemap.ffmpeg_capabilities(ffmpeg_path()) is caps, "not cached"
    statuses = tonemap.engine_statuses()
    assert [s["id"] for s in statuses] == ["lut", "videotoolbox", "zscale", "libplacebo", "tonemapx"]
    assert all(set(s) == {"id", "label", "available", "reason", "pack"} for s in statuses)


def test_progress_and_cancellation_kill_ffmpeg():
    token = CancellationToken()
    events = []

    def progress(pct, stage):
        events.append((pct, stage))
        token.cancel()

    ctx = TaskContext(token=token, progress=progress, log=lambda m: None, workdir=tempfile.gettempdir())
    command = [ffmpeg_path(), "-nostdin", "-v", "error", "-re", "-f", "lavfi", "-i",
               "testsrc=size=160x120:rate=25", "-f", "null", "-progress", "pipe:1", "-nostats", "-"]
    started = time.monotonic()
    try:
        tonemap.run_ffmpeg(command, ctx, duration=3600)
    except Cancelled:
        pass
    else:
        raise AssertionError("cancel did not stop FFmpeg")
    assert time.monotonic() - started < 10, "cancel was slow"
    assert events and events[0][1].startswith("Tone-mapping"), events


# A stand-in for a Chocolatey/Scoop ffmpeg shim or a venv python.exe
# launcher: the process we start runs the real worker as a child that shares
# our stdout pipe, then waits for it.
_LAUNCHER = r"""
import subprocess, sys
child = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])
child.wait()
"""
_CHILD = r"""
import os, sys, time
with open(sys.argv[1], "w") as handle:
    handle.write(str(os.getpid()))
while True:
    sys.stdout.write("frame=1\nprogress=continue\n")
    sys.stdout.flush()
    time.sleep(0.1)
"""


def _alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_cancel_kills_a_launcher_and_the_child_holding_the_pipe():
    """Stop must end the whole tree. Killing only the launcher left the real
    worker writing into our pipe, and the read loop waited for it forever --
    the Windows CI job hung here for an hour."""
    token = CancellationToken()

    def progress(pct, stage):
        token.cancel()

    with Workspace() as ws:
        pid_file = ws.path("child.pid")
        ctx = TaskContext(token=token, progress=progress, log=lambda m: None, workdir=ws.root)
        started = time.monotonic()
        try:
            tonemap.run_ffmpeg([sys.executable, "-c", _LAUNCHER, _CHILD, pid_file], ctx, duration=None)
        except Cancelled:
            pass
        else:
            raise AssertionError("cancel did not stop the launcher")
        assert time.monotonic() - started < 15, "cancel waited on the child's pipe"
        with open(pid_file) as handle:
            child = int(handle.read())
        deadline = time.monotonic() + 5
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(child), f"the launcher's child {child} outlived the cancel"


# ------------------------------------------------------- end-to-end, measured


def test_lut_engine_measured_on_hdr10_patches():
    if not _have_x265():
        print("        (skipped: no libx265)")
        return
    with Workspace() as ws:
        clip = _make_clip(ws.path("hdr10.mkv"), _pq_codes(PQ_NITS), "pq", max_cll=10000, mastering_nits=10000)
        result, events = _run_job(clip, ws.root, engine="lut")
        out = result.outputs[0].path
        assert os.path.basename(out) == "hdr10.sdr.64p.mkv", out
        y = _measure(out, len(PQ_NITS))
        MEASURED["lut, source peak 10000 (auto, MaxCLL)"] = y
        assert result.report["source"]["peakNits"] == 10000.0, result.report
        assert all(b >= a - 0.3 for a, b in zip(y, y[1:])), f"not monotonic: {y}"
        assert abs(y[0] - 16) <= 0.5, f"black lifted: {y[0]}"
        assert y[3] < 234, f"100 nits clipped: {y[3]}"
        assert y[6] < 235, f"4000 nits reached full white with BT.2390 from a 10000-nit peak: {y[6]}"
        expected = 16 + 219 * cm.tone_curve(np.array(PQ_NITS), "bt2390", 10000, 100) ** (1 / 2.4)
        assert np.abs(np.array(y) - expected).max() < 1.0, f"measured {y} vs predicted {expected}"
        check = result.report["output"]
        assert (check["transfer"], check["primaries"], check["matrix"]) == ("bt709", "bt709", "bt709"), check
        assert check["hdrSideData"] == [], check
        assert result.warnings == [], result.warnings
        assert any(stage.startswith("Tone-mapping") for _, stage in events), events

        # The common case: a 1000-nit master.
        result, _ = _run_job(clip, ws.root, engine="lut", sourcePeak=1000)
        y = _measure(result.outputs[0].path, len(PQ_NITS))
        MEASURED["lut, source peak 1000"] = y
        assert 180 <= y[4] <= 225, f"203-nit reference white at {y[4]}"
        assert y[5] >= 234.5, f"1000 nits should reach white: {y[5]}"


def test_lut_engine_measured_on_hlg_patches():
    if not _have_x265():
        print("        (skipped: no libx265)")
        return
    with Workspace() as ws:
        clip = _make_clip(ws.path("hlg.mkv"), _hlg_codes(HLG_LEVELS), "hlg")
        result, _ = _run_job(clip, ws.root, engine="lut")
        assert result.report["source"]["hdr"] == "hlg", result.report
        y = _measure(result.outputs[0].path, len(HLG_LEVELS))
        MEASURED["lut, HLG 1000-nit nominal"] = y
        assert all(b >= a - 0.3 for a, b in zip(y, y[1:])), y
        rgb = np.repeat(np.array(HLG_LEVELS)[:, None], 3, axis=1)
        expected = 16 + 219 * cm.hdr_to_sdr(rgb, transfer="hlg", source_peak=1000)[:, 0]
        assert np.abs(np.array(y) - expected).max() < 1.0, f"{y} vs {expected}"
        assert 180 <= y[3] <= 225, f"75% HLG (reference white) at {y[3]}"


def test_videotoolbox_engine_measured():
    if sys.platform != "darwin" or not _have_x265():
        print("        (skipped: macOS with libx265 only)")
        return
    path, reason = tonemap.binary_for("videotoolbox")
    if not path:
        print(f"        (skipped: {reason})")
        return
    with Workspace() as ws:
        clip = _make_clip(ws.path("hdr10.mkv"), _pq_codes(PQ_NITS), "pq", max_cll=10000, mastering_nits=10000)
        result, _ = _run_job(clip, ws.root, engine="videotoolbox")
        y = _measure(result.outputs[0].path, len(PQ_NITS))
        MEASURED[f"videotoolbox ({result.report['filter']})"] = y
        assert all(b >= a - 1.0 for a, b in zip(y, y[1:])), f"not monotonic: {y}"
        assert y[0] <= 17 and 150 <= y[4] <= 225, y
        assert y[3] < 234, f"100 nits clipped: {y[3]}"
        check = result.report["output"]
        assert (check["transfer"], check["primaries"], check["matrix"]) == ("bt709", "bt709", "bt709"), check
        assert check["hdrSideData"] == [], check


def test_jellyfin_engines_measured():
    """tonemap_videotoolbox (Metal) and tonemapx from the Full FFmpeg pack,
    when installed (or pointed at with AUDIOSYNC_FFMPEG_FULL)."""
    if not _have_x265():
        print("        (skipped: no libx265)")
        return
    ran = []
    with Workspace() as ws:
        clip = _make_clip(ws.path("hdr10.mkv"), _pq_codes(PQ_NITS), "pq", max_cll=10000, mastering_nits=10000)
        for engine in ("videotoolbox", "tonemapx", "zscale"):
            path, _ = tonemap.binary_for(engine)
            if not path or (engine == "videotoolbox"
                            and not tonemap.ffmpeg_capabilities(path)["filters"].get("tonemap_videotoolbox")):
                continue
            result, _ = _run_job(clip, ws.root, engine=engine, algorithm="hable" if engine == "zscale" else "bt2390",
                                 sourcePeak=1000 if engine == "zscale" else "auto")
            y = _measure(result.outputs[0].path, len(PQ_NITS))
            label = "zscale hable, peak 1000" if engine == "zscale" else f"{result.report['filter']} (Jellyfin)"
            MEASURED[label] = y
            assert all(b >= a - 0.5 for a, b in zip(y, y[1:])), f"{engine} not monotonic: {y}"
            assert y[0] <= 17 and 233 <= y[-1] <= 235.5 and y[3] < 200, f"{engine}: {y}"
            assert result.report["output"]["hdrSideData"] == [], result.report["output"]
            ran.append(engine)
    if not ran:
        print("        (skipped: no Jellyfin FFmpeg; set AUDIOSYNC_FFMPEG_FULL to run)")


def test_text_burn_with_libass_and_an_awkward_path():
    if not _have_x265():
        print("        (skipped: no libx265)")
        return
    path, _ = tonemap.binary_for("lut")
    if not tonemap.ffmpeg_capabilities(path)["filters"].get("subtitles"):
        print("        (skipped: this FFmpeg has no libass; the Full FFmpeg pack does)")
        return
    w, h, frames = 320, 180, 48
    with Workspace() as ws:
        # Every character libass's filter syntax treats specially. Windows
        # forbids ':' in a file name; there the drive letter ("C:") is the
        # colon that must survive escaping.
        folder = ws.path("It's a [weird], dir; x" if os.name == "nt" else "It's a [weird], dir; x:y")
        os.makedirs(folder)
        grey = np.full((h, w), _pq_codes([10])[0], np.uint16)
        chroma = np.full((h // 2, w // 2), 512, np.uint16)
        video = os.path.join(folder, "v.mkv")
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
                        "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-vf",
                        "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv",
                        "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error", video],
                       input=(grey.tobytes() + chroma.tobytes() * 2) * frames, check=True, capture_output=True)
        srt = os.path.join(folder, "s.srt")
        with open(srt, "w") as handle:
            handle.write("1\n00:00:00,500 --> 00:00:01,500\nHELLO WORLD\n")
        film = os.path.join(folder, "fi'lm, [x].mkv")
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", video, "-i", srt, "-map", "0", "-map", "1",
                        "-c", "copy", film], check=True, capture_output=True)
        result, _ = _run_job(film, ws.root, engine="lut", subtitles="burn")
        raw = subprocess.run([ffmpeg_path(), "-v", "error", "-i", result.outputs[0].path, "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        planes = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
        shown = [k for k in range(len(planes)) if planes[k, 120:175].max() > 200]
        assert shown and (shown[0], shown[-1]) == (12, 35), f"text on frames {shown[:1]}-{shown[-1:]}"


def test_image_burn_is_frame_exact():
    """A PGS caption at 0.5-1.5 s must cover frames 12-35 at 24 fps, on a
    B-frame HEVC source (where FFmpeg's same-input sub2video ran 5 frames late)."""
    if not _have_x265():
        print("        (skipped: no libx265)")
        return
    from audiosync.subs import pgs

    w, h, frames = 320, 180, 48
    with Workspace() as ws:
        grey = np.full((h, w), _pq_codes([100])[0], np.uint16)
        chroma = np.full((h // 2, w // 2), 512, np.uint16)
        video = ws.path("v.mkv")
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
                        "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-vf",
                        "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv",
                        "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error:bframes=3",
                        video], input=(grey.tobytes() + chroma.tobytes() * 2) * frames, check=True, capture_output=True)
        box = np.full((40, 120, 4), 255, np.uint8)
        sup = pgs.write_sup([pgs.BitmapEvent(0.5, 1.5, box, 100, 120, video_size=(w, h))], ws.path("s.sup"), (w, h))
        film = ws.path("film.mkv")
        # -copyts keeps the caption at 0.5 s (FFmpeg would otherwise move the
        # .sup's first timestamp to zero).
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-copyts", "-i", video, "-i", sup, "-map", "0",
                        "-map", "1", "-c", "copy", film], check=True, capture_output=True)
        result, _ = _run_job(film, ws.root, engine="lut", subtitles="burn")
        raw = subprocess.run([ffmpeg_path(), "-v", "error", "-i", result.outputs[0].path, "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        planes = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
        shown = [k for k in range(len(planes)) if planes[k, 140, 160] > 240]
        assert shown and (shown[0], shown[-1]) == (12, 35), f"caption on frames {shown[:1]}-{shown[-1:]}"
        assert planes[20, 60, 160] < 240, "picture outside the caption changed"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    if MEASURED:
        print("\n  Measured output luma (8-bit Y', 16 = black, 235 = SDR white):")
        for name, values in MEASURED.items():
            labels = PQ_NITS if len(values) == len(PQ_NITS) else [f"{v:.0%}" for v in HLG_LEVELS]
            cells = ", ".join(f"{lab if isinstance(lab, str) else f'{lab:g}'}->{v:.1f}" for lab, v in zip(labels, values))
            print(f"    {name}: {cells}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
