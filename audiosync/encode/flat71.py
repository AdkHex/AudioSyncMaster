"""Flat 7.1 Atmos through the patched DEE (FPSConv ``engine.py:1492-1613``).

LumaVistaLab/DEE_DDPlusJOC_7.1_Patcher makes DEE 5.2.1 render the Blu-ray
DD+ JOC legacy presentation as flat 7.1 (Lb Rb) instead of 5.1 + 2 heights
(Tfl Tfr). It patches two bytes of one DEE DLL, so it is validated for
exactly one DEE build: a DLL with any other hash makes the job refuse, and it
never falls back to the DEE-default layout on its own.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from .formats import ATMOS71_LAYOUTS, FLAT71_BITRATES  # noqa: F401 - re-exported for callers

#: Release v1.1-stable of the patcher. GPL-3.0, stdlib-only Python, run as a separate process.
PATCHER_URL = ("https://github.com/LumaVistaLab/DEE_DDPlusJOC_7.1_Patcher/raw/main/release/"
               "dee-ddp71-atmos-wrapper-v1.1-stable.zip")
PATCHER_ZIP_SHA256 = "b0dd0d73e37d1fb20131ff6845f5e340a09d3669a6b3a0fd20e822f7df1328e9"
PATCHER_ZIP_TOP = "dee-ddp71-atmos-wrapper-v1.1-stable"
PATCHER_SCRIPT = "dee-ddp71-atmos-wrapper.py"
PATCHER_DLL = "dee_audio_filter_ddp_atmos.dll"
#: The one DEE 5.2.1 build the patch is validated for (original and patched DLL hashes).
PATCHER_DLL_SHA256 = "3d66bcec36031fd48e6565d15f05fea656642377ca4f8c98cdce1cce8b7e95d2"
PATCHER_DLL_PATCHED_SHA256 = "fd49c7b9b19bba5f7ec0b862a9811a7b822b2efe200bc82a033fb9b7f54c1588"


def patcher_script(folder: Optional[Path]) -> Optional[Path]:
    """The wrapper script in ``folder`` or in the release's top folder inside it."""
    if not folder:
        return None
    d = Path(folder)
    for cand in (d / PATCHER_SCRIPT, d / PATCHER_ZIP_TOP / PATCHER_SCRIPT):
        if cand.is_file():
            return cand
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dee_dll_status(dee_path: str) -> dict:
    """Whether the user's DEE carries the exact DLL build the flat-7.1 patch is validated for."""
    dee = dee_path or ""
    if not dee or not os.path.isfile(dee):
        return {"state": "no-dee", "dll": "", "sha256": ""}
    dll = Path(dee).parent / PATCHER_DLL
    if not dll.is_file():
        return {"state": "no-dll", "dll": str(dll), "sha256": ""}
    try:
        digest = sha256(dll)
    except OSError as exc:
        return {"state": "unreadable", "dll": str(dll), "sha256": "", "error": str(exc)}
    if digest == PATCHER_DLL_SHA256:
        state = "supported"
    elif digest == PATCHER_DLL_PATCHED_SHA256:
        state = "patched"          # left patched by an interrupted run; the wrapper restores it from its backup
    else:
        state = "unsupported"
    return {"state": state, "dll": str(dll), "sha256": digest}


def flat71_ready(script: Optional[Path], tools: dict, dll: Optional[dict] = None) -> tuple[bool, str]:
    """(ok, reason) — everything the flat-7.1 path needs, checked up front.

    ``tools`` holds the resolved ``dee`` and ``truehdd`` paths ("" = not found);
    ``dll`` is :func:`dee_dll_status` for that DEE, worked out here when not given.
    """
    if not script:
        return False, "the DD+ 7.1 patcher is not installed: Preferences → Encode → Dolby tools pack → Install"
    if not tools.get("dee"):
        return False, "Dolby Encoding Engine is not configured: set the dee.exe path in Preferences → Encode"
    if not tools.get("truehdd"):
        return False, "truehdd is not found: set its path in Preferences → Encode → Tools"
    st = dll if dll is not None else dee_dll_status(tools["dee"])
    if st["state"] == "no-dll":
        return False, f"{PATCHER_DLL} is not next to dee.exe — the patcher needs a full DEE 5.2.1 install"
    if st["state"] == "unsupported":
        return False, (f"this DEE build is not the one the 7.1 patcher is validated for "
                       f"({PATCHER_DLL} SHA-256 {st['sha256'][:12]}…, expected {PATCHER_DLL_SHA256[:12]}…); "
                       "use DEE 5.2.1-5994839 or choose the DEE-default layout")
    if st["state"] == "unreadable":
        return False, f"cannot read {st['dll']}: {st.get('error', '')}"
    return True, ""
