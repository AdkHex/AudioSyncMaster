"""Tests for what an encode job produces: DEE bitrates, layouts, Atmos rules,
the notes that explain them, output names and the job-log plan.

Ported from FPSConv's ``tests/test_engine.py`` (BitrateSnapping,
ResolveEncode, EncodeNaming, and the plan and flat-vs-DEE checks from
CommandBuilders and Flat71). The expected values are FPSConv's.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import commands, formats  # noqa: E402

DEEW = ["/venv/python", "-m", "deew"]


# FPSConv: BitrateSnapping.test_tables_match_dee
def test_tables_match_dee():
    assert formats.snap_bitrate("ddp", 6, 1536) == 1024      # DDP 5.1 has no 1536: its default (= max)
    assert formats.snap_bitrate("ddp", 8, 1503) == 1536      # "1503 kbps" is 1536 nominal
    assert formats.snap_bitrate("ddp", 6, 300) == 304        # in range: nearest
    assert formats.snap_bitrate("dd", 6, 1024) == 640
    assert formats.snap_bitrate("dd", 2, 0) == 256
    assert formats.snap_bitrate("ddp", 2, 0) == 256
    assert formats.snap_bitrate("ddp", 2, 1536) == 256       # a 7.1 rate on a 2.0 file: the 2.0 default


# FPSConv: BitrateSnapping.test_atmos_tables
def test_atmos_tables():
    assert formats.snap_bitrate("ddp", 6, 1536, atmos=True) == 768    # 7.1 rate on 5.1 Atmos: default
    assert formats.snap_bitrate("ddp", 6, 1000, atmos=True) == 1024   # in range: nearest
    assert formats.snap_bitrate("ddp", 8, 0, atmos=True) == 1536
    assert formats.snap_bitrate("ddp", 8, 1503, atmos=True) == 1512   # DeeZy's Blu-ray JOC list
    assert formats.snap_bitrate("ddp", 6, 0, atmos=True) == 768


# FPSConv: BitrateSnapping.test_pad_layout
def test_pad_layout():
    assert [formats.pad_layout(c) for c in (1, 2, 3, 5, 6, 7, 8, 10)] == [1, 2, 6, 6, 6, 8, 8, 8]


# FPSConv: ResolveEncode.test_downmix_71_to_51
def test_downmix_71_to_51():
    e = formats.resolve_encode("ddp", 6, True, 0, 8, False)
    assert (e.wav_channels, e.out_channels, e.bitrate, e.atmos) == (8, 6, 1024, False)
    assert e.label == "DDP 5.1 1024k"
    assert "-dm" in commands.deew_encode(DEEW, "/t.wav", e, "film_light", "/w")


# FPSConv: ResolveEncode.test_never_upmixes
def test_never_upmixes():
    e = formats.resolve_encode("ddp", 8, True, 1536, 2, False)
    assert (e.out_channels, e.bitrate) == (2, 256)
    assert any("no upmix" in n for n in e.notes)
    assert any("outside DDP 2.0" in n for n in e.notes)
    assert "-dm" not in commands.deew_encode(DEEW, "/t.wav", e, "film_light", "/w")


# FPSConv: ResolveEncode.test_odd_layout_is_padded
def test_odd_layout_is_padded():
    e = formats.resolve_encode("ddp", 0, True, 0, 5, False)
    assert (e.wav_channels, e.out_channels) == (6, 6)
    assert any("padded" in n for n in e.notes)


# FPSConv: ResolveEncode.test_dd_has_no_71
def test_dd_has_no_71():
    e = formats.resolve_encode("dd", 0, True, 0, 8, True, "truehd")
    assert (e.out_channels, e.bitrate, e.atmos, e.ext) == (6, 640, False, ".ac3")
    assert any("DD cannot carry Atmos" in n for n in e.notes)


# FPSConv: ResolveEncode.test_atmos_only_when_confirmed
def test_atmos_only_when_confirmed():
    yes = formats.resolve_encode("ddp", 0, True, 0, 8, True, "truehd")
    assert (yes.atmos, yes.atmos_mode, yes.bitrate, yes.label) == (True, "bluray", 1536, "DDP 7.1 Atmos flat 1536k")
    streaming = formats.resolve_encode("ddp", 6, True, 0, 8, True, "truehd")
    assert (streaming.atmos, streaming.atmos_mode, streaming.bitrate) == (True, "streaming", 768)
    stereo = formats.resolve_encode("ddp", 2, True, 0, 8, True, "truehd")
    assert not stereo.atmos
    unknown = formats.resolve_encode("ddp", 0, True, 0, 8, None, "truehd")
    assert not unknown.atmos
    assert any("mediainfo" in n for n in unknown.notes)
    off = formats.resolve_encode("ddp", 0, False, 0, 8, True, "truehd")
    assert (off.atmos, off.label) == (False, "DDP 7.1 1536k")


# FPSConv: ResolveEncode.test_bluray_profile_note
def test_bluray_profile_note():
    assert any("Blu-ray" in n for n in formats.resolve_encode("ddp", 8, False, 1536, 8, False).notes)
    assert not any("Blu-ray" in n for n in formats.resolve_encode("ddp", 8, False, 1024, 8, False).notes)


# FPSConv: EncodeNaming.test_names
def test_names():
    e = formats.resolve_encode("ddp", 0, True, 0, 8, True, "truehd")
    assert formats.encode_output_name("/x/movie.mkv", 0, e) == "movie_DDP7.1Atmos_1536k.ec3"
    e = formats.resolve_encode("dd", 0, True, 0, 6, False)
    assert formats.encode_output_name("/x/movie.mkv", 1, e) == "movie_a1_DD5.1_640k.ac3"
    e = formats.resolve_encode("ddp", 0, True, 0, 2, False)
    assert formats.encode_output_name("/x/song.m4a", 0, e) == "song_DDP2.0_256k.ec3"


# FPSConv: CommandBuilders.test_encode_plan_mentions_the_real_steps
def test_encode_plan_mentions_the_real_steps():
    e = formats.resolve_encode("ddp", 8, True, 1536, 8, True, "truehd")
    plan = formats.encode_plan(e, {"pretty": "TrueHD Atmos", "codec": "truehd", "sample_rate": 48000})
    assert len(plan) == 5                                   # flat 7.1: the patched-DEE chain
    assert "truehdd decodes" in plan[1] and "flat-7.1" in plan[2] and "1536 kbps" in plan[2] and "Lb Rb" in plan[4]
    e = formats.resolve_encode("ddp", 8, True, 1536, 8, True, "truehd", atmos71="dee")
    plan = formats.encode_plan(e, {"pretty": "TrueHD Atmos", "codec": "truehd", "sample_rate": 48000})
    assert len(plan) == 4
    assert "truehdd decodes" in plan[1] and "Blu-ray mode" in plan[2] and "1536 kbps" in plan[2]
    e = formats.resolve_encode("ddp", 6, True, 0, 8, False)
    plan = formats.encode_plan(e, {"pretty": "DTS-HD MA", "codec": "aac", "codec_name": "dts", "sample_rate": 96000})
    assert "lossless decode" in plan[0] and "96000 Hz to 48 kHz" in plan[0] and "7.1 → 5.1 downmix" in plan[1]


# FPSConv: Flat71.test_resolve_flat_vs_dee
def test_resolve_flat_vs_dee():
    flat = formats.resolve_encode("ddp", 8, True, 1536, 8, True, "truehd")
    assert flat.flat71 and flat.label == "DDP 7.1 Atmos flat 1536k"
    assert any("Lb Rb" in n for n in flat.notes)
    dee = formats.resolve_encode("ddp", 8, True, 1536, 8, True, "truehd", atmos71="dee")
    assert not dee.flat71 and any("Tfl Tfr" in n for n in dee.notes)
    # flat 7.1 uses the wrapper's data-rate list; out-of-range falls back to the default
    assert formats.resolve_encode("ddp", 8, True, 1500, 8, True, "truehd").bitrate == 1512
    assert formats.resolve_encode("ddp", 8, True, 640, 8, True, "truehd").bitrate == 1536
    # 5.1 Atmos and non-Atmos never take the flat path
    assert not formats.resolve_encode("ddp", 6, True, 0, 8, True, "truehd").flat71
    assert not formats.resolve_encode("ddp", 8, True, 0, 8, False, "truehd").flat71
    assert formats.encode_output_name("/x/m.thd", 0, flat) == "m_DDP7.1Atmos_1536k.ec3"
