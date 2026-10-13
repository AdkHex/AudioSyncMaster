"""Audio encode and FPS change, ported from FPSConv v1.1.24. See ENCODE.md.

Two tasks share one queue:

* **FPS change** -- an audio track timed to one frame rate re-timed to
  another with ``atempo``: AAC in one ffmpeg pass, AC-3 / E-AC-3 / TrueHD
  through a 24-bit WAV and DEE so the codec stays the same.
* **Encode** -- any source to DD / DDP / DDP Atmos through DEE, no speed
  change; TrueHD Atmos to DDP JOC through truehdd and DeeZy, or to flat 7.1
  (Lb Rb) through the patched DEE.

DEE runs on Windows only, so every entry point that runs a job calls
:func:`require_windows` first. The pure modules (formats, fps, commands,
progress, atmos) import and test on every OS.

Behaviour matches FPSConv exactly -- the same command lines, output names,
bitrate tables and notes -- apart from the fixes listed in
FPSCONV-MIGRATION.md section 8.
"""

from __future__ import annotations

import os
import sys

WINDOWS_ONLY = "Encode is available on Windows only"


class WindowsOnly(RuntimeError):
    """Raised by an encode entry point on macOS and Linux."""


def is_windows() -> bool:
    """True on Windows, or anywhere ``AUDIOSYNC_ENCODE_ANY_OS=1`` is set.

    The override exists for the test suite: the AAC FPS path needs only
    ffmpeg, so the queue, the bridge commands and the CLI can be exercised
    end to end on the macOS and Linux CI runners too.
    """
    return sys.platform == "win32" or os.environ.get("AUDIOSYNC_ENCODE_ANY_OS") == "1"


def require_windows() -> None:
    if not is_windows():
        raise WindowsOnly(WINDOWS_ONLY)
