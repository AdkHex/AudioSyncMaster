"""MediaInfo JSON for a batch of files, in the Dolby tools pack (pymediainfo).

Encode needs MediaInfo to tell TrueHD Atmos from plain TrueHD (ffprobe
cannot), and most machines have no MediaInfo CLI. pymediainfo ships the
library itself on Windows, so the pack is enough.

Request: ``{"paths": [...]}``. Result: ``{"files": {path: json-or-null}}``.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _protocol  # noqa: E402


def handle(request: dict) -> dict:
    from pymediainfo import MediaInfo

    files = {}
    for path in request.get("paths") or []:
        try:
            files[path] = json.loads(MediaInfo.parse(path, output="JSON"))
        except Exception as exc:  # noqa: BLE001 - one unreadable file is "unknown", not a failed batch
            _protocol.log(f"mediainfo could not read {os.path.basename(path)}: {exc}")
            files[path] = None
    return {"files": files}


if __name__ == "__main__":
    _protocol.main(handle)
