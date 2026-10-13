"""Tests for the flat 7.1 checks: the DEE DLL verdict and what blocks a run.

Ported from FPSConv's ``tests/test_engine.py`` (Flat71). A DEE build the
patcher is not validated for must refuse the job, never fall back.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import flat71  # noqa: E402

SCRIPT = Path("/p/w.py")


# FPSConv: Flat71.test_dll_status
def test_dll_status():
    with tempfile.TemporaryDirectory() as tmp:
        dee = Path(tmp) / "dee.exe"
        dee.write_bytes(b"x")
        assert flat71.dee_dll_status(str(dee))["state"] == "no-dll"
        dll = Path(tmp) / flat71.PATCHER_DLL
        dll.write_bytes(b"not the real one")
        st = flat71.dee_dll_status(str(dee))
        assert st["state"] == "unsupported" and len(st["sha256"]) == 64
        with mock.patch.object(flat71, "PATCHER_DLL_SHA256", st["sha256"]):
            assert flat71.dee_dll_status(str(dee))["state"] == "supported"
        assert flat71.dee_dll_status("/nope/dee.exe")["state"] == "no-dee"


# FPSConv: Flat71.test_flat71_ready_reasons
def test_flat71_ready_reasons():
    assert "not installed" in flat71.flat71_ready(None, {"dee": "/d/dee.exe", "truehdd": "t"})[1]
    assert "Dolby Encoding Engine" in flat71.flat71_ready(SCRIPT, {"dee": "", "truehdd": "t"})[1]
    unsupported = {"state": "unsupported", "sha256": "ab" * 32, "dll": "/d/x.dll"}
    reason = flat71.flat71_ready(SCRIPT, {"dee": "/d/dee.exe", "truehdd": "t"}, unsupported)[1]
    assert "not the one the 7.1 patcher is validated for" in reason


# FPSConv: Flat71.test_flat71_ready_accepts_a_dll_left_patched_and_doctor_reports_the_verdict
def test_flat71_ready_accepts_a_dll_left_patched_and_doctor_reports_the_verdict():
    # an interrupted run leaves the patched bytes in place; the wrapper restores from backup, so that is fine
    patched = {"state": "patched", "sha256": "cd" * 32, "dll": "/d/x.dll"}
    assert flat71.flat71_ready(SCRIPT, {"dee": "/d/dee.exe", "truehdd": "t"}, patched) == (True, "")


# ── beyond FPSConv's suite ──


def test_hashes_are_fpsconvs_verbatim():
    assert flat71.PATCHER_ZIP_SHA256 == "b0dd0d73e37d1fb20131ff6845f5e340a09d3669a6b3a0fd20e822f7df1328e9"
    assert flat71.PATCHER_DLL_SHA256 == "3d66bcec36031fd48e6565d15f05fea656642377ca4f8c98cdce1cce8b7e95d2"
    assert flat71.PATCHER_DLL_PATCHED_SHA256 == "fd49c7b9b19bba5f7ec0b862a9811a7b822b2efe200bc82a033fb9b7f54c1588"


def test_truehdd_missing_blocks_the_run():
    assert "truehdd" in flat71.flat71_ready(SCRIPT, {"dee": "/d/dee.exe", "truehdd": ""})[1]


def test_patcher_script_found_in_the_release_folder():
    with tempfile.TemporaryDirectory() as tmp:
        assert flat71.patcher_script(Path(tmp)) is None
        nested = Path(tmp) / flat71.PATCHER_ZIP_TOP
        nested.mkdir()
        (nested / flat71.PATCHER_SCRIPT).write_text("")
        assert flat71.patcher_script(Path(tmp)) == nested / flat71.PATCHER_SCRIPT
        assert flat71.patcher_script(None) is None


def test_doctor_reports_the_flat71_verdict():
    """The second half of FPSConv's Flat71 test: doctor() carries flat71_ready()'s answer."""
    from audiosync.encode import tools

    with mock.patch.object(tools.Tools, "flat71_ready", return_value=(False, "why not")), \
         mock.patch.object(tools, "_runs", return_value=None):
        assert tools.doctor()["flat71"] == {"ok": False, "reason": "why not"}
