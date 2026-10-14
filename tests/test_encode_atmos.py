"""Tests for Atmos detection from MediaInfo, codec names and the MediaInfo report.

Ported from FPSConv's ``tests/test_engine.py`` (MediaInfoAtmos, MediaInfoReport).
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import atmos  # noqa: E402

FIXTURE = {"media": {"track": [
    {"@type": "General"},
    {"@type": "Video"},
    {"@type": "Audio", "Format": "MLP FBA", "Format_Commercial_IfAny": "Dolby TrueHD with Dolby Atmos",
     "Format_AdditionalFeatures": "16-ch"},
    {"@type": "Audio", "Format": "E-AC-3", "Format_Commercial_IfAny": "Dolby Digital Plus with Dolby Atmos",
     "Format_AdditionalFeatures": "JOC"},
    {"@type": "Audio", "Format": "DTS", "Format_Commercial_IfAny": "DTS-HD Master Audio"},
    {"@type": "Text"},
]}}


# FPSConv: MediaInfoAtmos.test_parses_per_audio_track
def test_parses_per_audio_track():
    info = atmos.parse_mediainfo_audio(FIXTURE)
    assert [t["atmos"] for t in info] == [True, True, False]
    assert info[2]["commercial"] == "DTS-HD Master Audio"


# FPSConv: MediaInfoAtmos.test_missing_mediainfo_is_unknown
def test_missing_mediainfo_is_unknown():
    assert atmos.parse_mediainfo_audio(None) is None
    with mock.patch.object(atmos, "mediainfo_json", return_value=None):
        assert atmos.mediainfo_atmos("/x.mkv") is None


# FPSConv: MediaInfoAtmos.test_pretty_names
def test_pretty_names():
    assert atmos.pretty_codec({"codec_name": "truehd", "atmos": True}) == "TrueHD Atmos"
    assert atmos.pretty_codec({"codec_name": "dts", "profile": "DTS-HD MA", "atmos": False}) == "DTS-HD MA"
    assert atmos.pretty_codec({"codec_name": "eac3", "atmos": None}) == "E-AC-3"
    assert atmos.pretty_codec({"codec_name": "pcm_s24le"}) == "PCM"


# FPSConv: MediaInfoReport.test_json_fallback_renders_tracks
def test_json_fallback_renders_tracks():
    text = atmos.report_from_json({"media": {"track": [
        {"@type": "General", "Format": "E-AC-3", "extra": {"x": 1}},
        {"@type": "Audio", "Format": "E-AC-3", "Channels": "8", "Empty": ""}]}})
    assert "General" in text and "Channels" in text
    assert "extra" not in text and "Empty" not in text


# FPSConv: MediaInfoReport.test_missing_file_is_empty
def test_missing_file_is_empty():
    assert atmos.mediainfo_report("/no/such/file.ec3") == ""


# FPSConv: MediaInfoReport.test_find_output_exact_stem_and_diagnosis
def test_find_output_exact_stem_and_diagnosis():
    with tempfile.TemporaryDirectory() as d:
        real = os.path.join(d, "Movie_DDP7.1Atmos_1536k.ec3")
        open(real, "w").close()
        assert atmos.find_output(real) == (real, "")
        # the recorded path is stale but a file with that stem is in the folder
        found, why = atmos.find_output(os.path.join(d, "gone", "Movie_DDP7.1Atmos_1536k.ec3"), d,
                                       "Movie_DDP7.1Atmos_1536k")
        assert (found, why) == (real, "")
        # nothing matches: the error names the folder and the closest names
        found, why = atmos.find_output(os.path.join(d, "Movie_DD5.1_640k.ac3"), d, "Movie_DD5.1_640k")
        assert found == ""
        assert "similar names there: ['Movie_DDP7.1Atmos_1536k.ec3']" in why
        found, why = atmos.find_output("/nope/x.ec3")
        assert "does not exist or is not reachable" in why
