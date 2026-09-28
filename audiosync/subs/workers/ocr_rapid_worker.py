"""RapidOCR worker (runs in the ocr-rapidocr pack).

PaddleOCR's detection and recognition models on ONNX Runtime, via the
``rapidocr_onnxruntime`` package (the newer ``rapidocr`` package works too).
The engine sends prepared line images -- one text line each, black on white
-- so each is read whole by the recogniser; detection runs only when that
read is unsure, to split off anything the engine's line splitter left
attached (a second speaker far to the right) and give each piece a box.

Request::

    {"images": ["/tmp/l0001.png", ...], "language": "ja",
     "recModel": "<rec.onnx>" | null, "recKeys": "<dict.txt>" | null}

``recModel``/``recKeys`` are the recognition model and character list for
the language (``packs.ocr_recognizer`` picks them: PP-OCRv5 CJK, Korean,
Latin, Cyrillic); without them the package's built-in Chinese+English
model is used.

Result::

    {"results": [{"index": 0, "observations": [
        {"text": "...", "confidence": 0.97, "box": [x, y, w, h]}]}, ...],
     "model": "default" | "<path>"}

Boxes are normalised to the image size with a top-left origin, like the
Vision helper's, so ``ocr.py`` treats every engine alike.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as proto  # noqa: E402

STAGE = "Reading text (RapidOCR)"


def _find_model(request: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    rec, keys = request.get("recModel"), request.get("recKeys")
    if rec and os.path.isfile(rec):
        return rec, keys if keys and os.path.isfile(keys) else None
    if rec:
        proto.log(f"Recognition model not found ({rec}); using the built-in model")
    return None, None


def _make_engine(rec: Optional[str], keys: Optional[str]):
    """(callable, flavour). rapidocr_onnxruntime 1.x takes model paths as
    keyword arguments; rapidocr 2+/3 takes a params dict."""
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        kwargs: Dict[str, Any] = {}
        if rec:
            kwargs["rec_model_path"] = rec
            if keys:
                kwargs["rec_keys_path"] = keys
        return RapidOCR(**kwargs), "onnxruntime"
    except ImportError:
        from rapidocr import RapidOCR  # type: ignore

        params: Dict[str, Any] = {}
        if rec:
            params["Rec.model_path"] = rec
            if keys:
                params["Rec.rec_keys_path"] = keys
        return RapidOCR(params=params) if params else RapidOCR(), "rapidocr"


def _image_size(path: str) -> Tuple[int, int]:
    import struct

    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", head[16:24])
        return int(width), int(height)
    return 0, 0


def _box(points: Any, size: Tuple[int, int]) -> Optional[List[float]]:
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    width, height = size
    if not width or not height:
        return None
    return [min(xs) / width, min(ys) / height, (max(xs) - min(xs)) / width, (max(ys) - min(ys)) / height]


def _mean_confidence(observations: List[Dict[str, Any]]) -> float:
    texts = [o for o in observations if o["text"].strip()]
    return sum(o["confidence"] for o in texts) / len(texts) if texts else 0.0


def _read(engine: Any, flavour: str, path: str) -> List[Dict[str, Any]]:
    size = _image_size(path)
    if flavour == "onnxruntime":
        # The engine sends one text line per image, which is what the
        # recogniser expects; detection on such a tight crop tends to cut
        # thin glyphs at the ends ("I don't" -> "don't"). So recognise the
        # whole line first and run detection only when that is unsure.
        whole: List[Dict[str, Any]] = []
        result, _elapsed = engine(path, use_det=False, use_cls=False, use_rec=True)
        for item in result or []:
            if len(item) >= 2:
                whole.append({"text": str(item[-2]).strip(), "confidence": float(item[-1]), "box": None})
        if whole and _mean_confidence(whole) >= 0.8:
            return whole
        pieces: List[Dict[str, Any]] = []
        result, _elapsed = engine(path)
        for item in result or []:
            if len(item) == 3:  # [box, text, score]
                pieces.append({"text": str(item[1]).strip(), "confidence": float(item[2]), "box": _box(item[0], size)})
        return pieces if _mean_confidence(pieces) > _mean_confidence(whole) else whole
    out: List[Dict[str, Any]] = []
    result = engine(path)
    texts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [])
    boxes = getattr(result, "boxes", None)
    for i, text in enumerate(texts):
        box = _box(boxes[i], size) if boxes is not None and i < len(boxes) else None
        out.append({"text": str(text), "confidence": float(scores[i]) if i < len(scores) else 0.0, "box": box})
    return out


def handle(request: Dict[str, Any]) -> Dict[str, Any]:
    images = list(request.get("images") or [])
    rec, keys = _find_model(request)
    proto.progress(0, "Loading RapidOCR")
    engine, flavour = _make_engine(rec, keys)
    if rec:
        proto.log(f"RapidOCR recognition model: {os.path.basename(rec)}")
    results = []
    for i, path in enumerate(images):
        try:
            results.append({"index": i, "observations": _read(engine, flavour, path)})
        except Exception as exc:  # noqa: BLE001 - one bad image must not end the job
            results.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
        if i % 10 == 9 or i + 1 == len(images):
            proto.progress(100.0 * (i + 1) / max(1, len(images)), STAGE)
    return {"results": results, "model": rec or "default"}


if __name__ == "__main__":
    proto.main(handle)
