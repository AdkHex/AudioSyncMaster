"""Tests for Subsync OCR: image subtitles to text.

Most tests are pure Python and run anywhere: separating the fill from the
outline, line splitting, furigana, italics, vertical text, text fixes,
merging, and every engine's plumbing against a stand-in (a fake
``tesseract`` script, a stub RapidOCR module run through the real pack
worker protocol, a local server speaking the Anthropic Messages API).

On macOS the end-to-end tests render real captions with CoreText (white
fill, black outline), encode them as PGS, and read them back with Apple
Vision through the whole task, asserting the text and reporting the
character accuracy.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import stat
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from audiosync.media import CancellationToken  # noqa: E402
from audiosync.subs import imaging, ocr, ocr_engines, pgs  # noqa: E402
from audiosync.subs.model import Cue  # noqa: E402
from audiosync.subs.tasks import TaskContext  # noqa: E402
from subs_fixtures import Skip, Workspace, render, run_module, skippable, tool  # noqa: E402


def context(workdir, secrets=None, logs=None):
    return TaskContext(
        token=CancellationToken(),
        progress=lambda percent, stage: None,
        log=(logs.append if logs is not None else (lambda message: None)),
        workdir=workdir,
        secrets=secrets or {},
    )


def outlined(mask, fill=(255, 255, 255), outline=(0, 0, 0), width=2):
    """RGBA of a boolean glyph mask drawn as a subtitle: fill + outline."""
    ring = imaging.dilate(mask, width) & ~mask
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[ring] = outline + (255,)
    rgba[mask] = fill + (255,)
    return rgba


def bars(h, w, pitch=10, stroke=4, top=0, shear=0.0):
    """Vertical strokes (a stand-in for letters), optionally slanted."""
    mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        offset = int(round(shear * (h - 1 - y)))
        for x in range(4, w - stroke - 4 - int(shear * h), pitch):
            mask[y, x + offset : x + offset + stroke] = True
    mask[:top] = False
    return mask


def accuracy(got, want):
    """Character accuracy: 1 - edit distance / length of the truth."""
    a, b = got, want
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(1, len(b)))


# ------------------------------------------------------------------ ink


def test_ink_separates_white_fill_from_black_outline():
    mask = bars(40, 200)
    ink = ocr.ink_map(outlined(mask))
    ring = imaging.dilate(mask, 2) & ~mask
    assert ink[mask].min() > 0.95
    assert ink[ring].max() < 0.05
    assert ink[~imaging.dilate(mask, 2)].max() == 0


def test_ink_handles_yellow_fill_dark_fill_and_no_outline():
    mask = bars(40, 200)
    yellow = ocr.ink_map(outlined(mask, fill=(255, 230, 0)))
    assert yellow[mask].min() > 0.95
    # Dark text with a light outline: the fill is the dark tone.
    dark = ocr.ink_map(outlined(mask, fill=(20, 20, 20), outline=(240, 240, 240)))
    assert dark[mask].min() > 0.95 and dark[imaging.dilate(mask, 2) & ~mask].max() < 0.05
    # Plain text, no outline: the glyphs are the opaque pixels.
    plain = np.zeros(mask.shape + (4,), dtype=np.uint8)
    plain[mask] = (250, 250, 250, 255)
    assert np.array_equal(ocr.ink_map(plain) > 0.5, mask)


def test_ink_of_a_fading_caption_matches_the_full_one():
    rgba = outlined(bars(40, 200))
    faded = rgba.copy()
    faded[..., 3] = (faded[..., 3].astype(int) * 90 // 255).astype(np.uint8)
    full_image = ocr.line_image(ocr.ink_map(rgba), 2)
    fade_image = ocr.line_image(ocr.ink_map(faded), 2)
    assert ocr._image_key(full_image) == ocr._image_key(fade_image)


# ---------------------------------------------------------------- lines


def test_lines_split_accents_join_and_furigana_drop():
    mask = np.zeros((200, 300), dtype=bool)
    mask[20:24, 100:110] = True  # an accent above line 1
    mask[27:67, 20:280] = True  # line 1
    mask[90:110, 110:170] = True  # small band right above line 2: ruby
    mask[114:154, 20:280] = True  # line 2
    bands, removed = ocr.split_lines(mask, cjk=True, remove_furigana=True)
    assert bands == [[20, 67], [114, 154]] and removed == 1
    # Latin text keeps a small line (it is not ruby).
    bands, removed = ocr.split_lines(mask, cjk=False, remove_furigana=True)
    assert len(bands) == 3 and removed == 0
    bands, removed = ocr.split_lines(mask, cjk=True, remove_furigana=False)
    assert len(bands) == 3


def test_italic_shear_is_detected():
    upright, gain = ocr.slant(bars(40, 400))
    assert abs(upright) < 0.05
    shear, gain = ocr.slant(bars(40, 400, shear=0.22))
    assert 0.15 <= shear <= 0.3 and gain >= ocr.ITALIC_MIN_GAIN


def test_vertical_text_is_relaid_as_horizontal_lines():
    event_mask = np.zeros((200, 130), dtype=bool)
    # Two columns (right one first), three square "characters" each.
    for col, x in enumerate((80, 10)):
        for row in range(3):
            y = 10 + row * 60
            event_mask[y : y + 45, x : x + 40] = True
            event_mask[y + 20 : y + 25, x : x + 40] = False  # a gap inside a glyph (like 三)
    event = pgs.BitmapEvent(0, 1, outlined(event_mask), 1700, 100, video_size=(1920, 1080))
    assert ocr.is_vertical(event, event_mask)
    strips = ocr.vertical_lines(event_mask.astype(np.float32), event_mask)
    assert len(strips) == 2
    for strip in strips:
        # Three glyph cells side by side, each as tall as the column is wide.
        cols = imaging.runs(strip.sum(axis=0))
        assert len(cols) == 3 and strip.shape[0] == 45
    # A two-line horizontal caption of the same size is not vertical.
    horizontal = np.zeros((130, 200), dtype=bool)
    horizontal[10:50, 10:190] = True
    horizontal[80:120, 10:190] = True
    assert not ocr.is_vertical(pgs.BitmapEvent(0, 1, outlined(horizontal), 800, 900), horizontal)


# ---------------------------------------------------------------- fixes


def test_fix_common_errors_latin():
    cases = {
        "|t's fine, | promise.": "It's fine, I promise.",
        "don 't and it' s and I 'm": "don't and it's and I'm",
        "l'm sure l saw it.": "I'm sure I saw it.",
        "Wait , what ?": "Wait, what?",
        "I don't know,I left.": "I don't know, I left.",
        "It costs 1,000 dollars.": "It costs 1,000 dollars.",
        "Hello . . .": "Hello...",
        "All  right": "All right",
        "Paul liked lemons.": "Paul liked lemons.",
    }
    for raw, want in cases.items():
        assert ocr.fix_common_errors(raw, "en") == want, (raw, ocr.fix_common_errors(raw, "en"))
    # French keeps its space before ? and !; l/I rules are English only.
    assert ocr.fix_common_errors("Quoi ? l'homme", "fr") == "Quoi ? l'homme"


def test_fix_common_errors_cjk():
    cases = {
        "今日は 天気が いい": "今日は天気がいい",
        "本当?うそ!": "本当？うそ！",
        "そうか,じゃあ.": "そうか、じゃあ。",
        "待って...": "待って…",
        "コ一ヒ一": "コーヒー",
        "コ-ヒ-を": "コーヒーを",
        "統ー": "統一",
        "ー人で": "一人で",
        "チーム一丸": "チーム一丸",
        "もう一度": "もう一度",
        "Version 1.5 です": "Version 1.5 です",
    }
    for raw, want in cases.items():
        assert ocr.fix_common_errors(raw, "ja") == want, (raw, ocr.fix_common_errors(raw, "ja"))
    # A typed space the image really had is kept.
    assert ocr.fix_common_errors("そうか じゃあ", "ja", wide_gaps=1) == "そうか じゃあ"
    assert ocr.fix_common_errors("你好,世界", "zh") == "你好，世界"
    assert ocr.fix_vertical_forms("コ｜ヒ｜﹁行こう﹂") == "コーヒー「行こう」"


def test_combine_observations_orders_and_drops_ruby():
    obs = [
        {"text": "ですね", "confidence": 1.0, "box": [0.6, 0.30, 0.3, 0.5]},
        {"text": "てんき", "confidence": 0.5, "box": [0.35, 0.02, 0.12, 0.15]},
        {"text": "今日はいい天気", "confidence": 0.5, "box": [0.05, 0.30, 0.5, 0.5]},
    ]
    text, conf, dropped = ocr.combine_observations(obs, cjk=True, remove_furigana=True)
    assert (text, dropped) == ("今日はいい天気 ですね", 1)
    assert abs(conf - (0.5 * 7 + 1.0 * 3) / 10) < 1e-9
    assert ocr.fix_common_errors(text, "ja") == "今日はいい天気ですね"
    assert ocr.combine_observations([], True, True) == ("", 0.0, 0)


def test_merge_touching_joins_fade_steps_only():
    cues = [
        Cue(1.0, 1.2, "Hello"), Cue(1.2, 1.4, "Hello"), Cue(1.4, 3.0, "Hello"),
        Cue(3.5, 4.0, "Hello"),  # a gap: a new caption
        Cue(1.0, 3.0, "Sign", align=8),  # a simultaneous top caption stays separate
    ]
    merged, count = ocr.merge_touching(cues)
    assert count == 2
    assert [(c.start, c.end, c.text) for c in merged] == [(1.0, 3.0, "Hello"), (1.0, 3.0, "Sign"), (3.5, 4.0, "Hello")]


# -------------------------------------------------------------- engines


def test_vision_language_mapping():
    supported = ["en-US", "fr-FR", "zh-Hans", "zh-Hant", "ko-KR", "ja-JP", "ru-RU", "vi-VT", "pt-BR"]
    assert ocr_engines.vision_languages("ja", supported) == ["ja-JP", "en-US"]
    assert ocr_engines.vision_languages("jpn", supported) == ["ja-JP", "en-US"]
    assert ocr_engines.vision_languages("zh", supported) == ["zh-Hans", "zh-Hant", "en-US"]
    assert ocr_engines.vision_languages("zh-Hant", supported) == ["zh-Hant", "zh-Hans", "en-US"]
    assert ocr_engines.vision_languages("ko", supported) == ["ko-KR", "en-US"]
    assert ocr_engines.vision_languages("vi", supported) == ["vi-VT", "en-US"]
    assert ocr_engines.vision_languages("en", supported) == ["en-US"]
    assert ocr_engines.vision_languages("hi", supported) == []  # unsupported: let Vision detect
    assert ocr_engines.vision_languages(None, supported) == []


def test_engine_statuses_shape():
    statuses = ocr.engine_statuses()
    assert [s["id"] for s in statuses] == ["vision", "tesseract", "rapidocr", "claude"]
    for s in statuses:
        assert set(s) == {"id", "label", "available", "reason", "pack"}
        assert s["available"] or s["reason"]
    assert statuses[2]["pack"] == "ocr-rapidocr"
    claude = ocr.engine_statuses({"openai": "x"})[3]
    assert not claude["available"] and "Anthropic" in claude["reason"]
    if sys.platform != "darwin":
        assert not statuses[0]["available"]


def _executable(path, text):
    with open(path, "w") as handle:
        handle.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_tesseract_engine_with_a_stand_in_binary():
    if os.name == "nt":
        return
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "1\t1\t0\t0\t0\t0\t0\t0\t400\t60\t-1\t\n"
        "5\t1\t1\t1\t1\t1\t10\t10\t80\t40\t96.5\t今日は\n"
        "5\t1\t1\t1\t1\t2\t95\t10\t80\t40\t88.5\t天気\n"
    )
    with Workspace() as ws:
        script = _executable(ws.path("tesseract"), f"""#!/bin/sh
if [ "$1" = "--list-langs" ]; then printf 'List of available languages (2):\\neng\\njpn\\n'; exit 0; fi
echo "$@" >> "{ws.path('calls')}"
cat <<'TSV'
{tsv}TSV
""")
        png = imaging.write_png(ws.path("l.png"), np.full((60, 400), 255, np.uint8))
        old = os.environ.get("AUDIOSYNC_TESSERACT")
        os.environ["AUDIOSYNC_TESSERACT"] = script
        try:
            reads = ocr_engines.run_tesseract([png, png], "ja", {}, context(ws.root))
            with open(ws.path("calls")) as handle:
                calls = handle.read().split("\n")
            assert "-l jpn+eng --psm 7 tsv" in calls[0]
            assert reads[0].observations[0]["text"] == "今日は天気"
            assert abs(reads[0].observations[0]["confidence"] - 0.925) < 1e-9
            try:
                ocr_engines.run_tesseract([png], "ko", {}, context(ws.root))
                raise AssertionError("missing Korean data should be an error")
            except ocr_engines.TaskError as exc:
                assert "kor" in str(exc) and "tesseract-lang" in str(exc)
        finally:
            if old is None:
                os.environ.pop("AUDIOSYNC_TESSERACT", None)
            else:
                os.environ["AUDIOSYNC_TESSERACT"] = old


_STUB_RAPIDOCR = '''
import struct
class RapidOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, path, use_det=True, use_cls=True, use_rec=True):
        with open(path, "rb") as handle:
            w, h = struct.unpack(">II", handle.read(24)[16:24])
        if use_det:
            return [[[[0, 0], [w, 0], [w, h], [0, h]], "detected", 0.99]], 0.01
        score = 0.95 if w > 100 else 0.5
        return [[" line %dx%d" % (w, h), score]], 0.01
'''


def test_rapidocr_worker_protocol_with_a_stub_library():
    """The real worker, through packs.run_worker and the JSON-lines
    protocol, with a stand-in for rapidocr_onnxruntime."""
    if os.name == "nt":
        return
    with Workspace() as ws:
        stub = ws.path("stub")
        os.makedirs(os.path.join(stub, "rapidocr_onnxruntime"))
        with open(os.path.join(stub, "rapidocr_onnxruntime", "__init__.py"), "w") as handle:
            handle.write(_STUB_RAPIDOCR)
        python = _executable(ws.path("python"), f'#!/bin/sh\nPYTHONPATH="{stub}" exec "{sys.executable}" "$@"\n')
        wide = imaging.write_png(ws.path("a.png"), np.full((50, 300), 255, np.uint8))
        narrow = imaging.write_png(ws.path("b.png"), np.full((50, 60), 255, np.uint8))
        key = "AUDIOSYNC_PACK_PYTHON_OCR_RAPIDOCR"
        old = os.environ.get(key)
        os.environ[key] = python
        try:
            reads = ocr_engines.run_rapidocr([wide, narrow], "en", {}, context(ws.root))
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        # Confident whole-line read kept; an unsure one falls back to detection.
        assert reads[0].observations == [{"text": "line 300x50", "confidence": 0.95, "box": None}]
        assert reads[1].observations[0]["text"] == "detected"
        assert reads[1].observations[0]["box"] == [0.0, 0.0, 1.0, 1.0]


class _MockAnthropic(http.server.BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        _MockAnthropic.requests.append((dict(self.headers), body))
        content = body["messages"][0]["content"]
        ids = [re.match(r"Image (\S+):", b["text"]).group(1) for b in content
               if b["type"] == "text" and b["text"].startswith("Image ")]
        lines = [{"id": i, "text": f"text of {i}", "uncertain": i == "L2"} for i in ids]
        reply = {"content": [{"type": "text", "text": json.dumps({"lines": lines})}], "stop_reason": "end_turn",
                 "usage": {"input_tokens": 10, "output_tokens": 5}}
        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_claude_engine_request_shape_against_a_local_server():
    from audiosync.subs import translate_engines

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockAnthropic)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    old = translate_engines.ENDPOINTS["anthropic"]
    translate_engines.ENDPOINTS["anthropic"] = f"http://127.0.0.1:{server.server_address[1]}"
    _MockAnthropic.requests = []
    logs: list = []
    try:
        with Workspace() as ws:
            paths = [imaging.write_png(ws.path(f"{i}.png"), np.full((40, 200), 255, np.uint8)) for i in range(30)]
            reads = ocr_engines.run_claude(paths, "ja", {"claudeModel": "claude-sonnet-5"},
                                           context(ws.root, {"anthropic": "sk-test-SECRET"}, logs))
    finally:
        translate_engines.ENDPOINTS["anthropic"] = old
        server.shutdown()
    assert len(_MockAnthropic.requests) == 2  # 24 + 6 lines
    headers, body = _MockAnthropic.requests[0]
    lowered = {k.lower(): v for k, v in headers.items()}
    assert lowered["x-api-key"] == "sk-test-SECRET" and lowered["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-5"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "low"
    images = [b for b in body["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == ocr_engines.CLAUDE_BATCH
    assert images[0]["source"]["type"] == "base64" and images[0]["source"]["media_type"] == "image/png"
    assert "Japanese" in body["messages"][0]["content"][-1]["text"]
    assert reads[0].observations[0] == {"text": "text of L1", "confidence": ocr_engines.CLAUDE_CONFIDENCE, "box": None}
    assert reads[1].observations[0]["confidence"] == ocr_engines.CLAUDE_UNCERTAIN
    assert reads[29].observations[0]["text"] == "text of L30"
    assert not any("SECRET" in line for line in logs)
    with Workspace() as ws:
        try:
            ocr_engines.run_claude(paths[:1], "ja", {}, context(ws.root))
            raise AssertionError("no key must be an error")
        except ocr_engines.TaskError as exc:
            assert "Anthropic API key" in str(exc)


# ------------------------------------------------------------ the task


def _fake_engine(texts):
    """An engine that answers from ``texts`` = [(text, confidence)]: the
    widest line image gets the first answer, narrower ones the next."""

    def run(paths, language, options, ctx, progress=None):
        widths = [imaging.read_png(p).shape[1] for p in paths]
        order = sorted(set(widths), reverse=True)
        reads = []
        for w in widths:
            text, conf = texts[min(order.index(w), len(texts) - 1)]
            reads.append(ocr_engines.LineRead(observations=[{"text": text, "confidence": conf, "box": None}]))
        return reads

    return run


def test_run_task_outputs_forced_split_positions_and_review_images():
    wide, narrow = outlined(bars(40, 400)), outlined(bars(40, 200, pitch=14))
    events = [
        pgs.BitmapEvent(1.0, 3.0, wide, 700, 950),
        pgs.BitmapEvent(1.5, 2.5, narrow, 800, 40, forced=True),
        pgs.BitmapEvent(4.0, 5.0, narrow, 800, 950),
    ]
    saved = ocr_engines.RUNNERS["vision"]
    ocr_engines.RUNNERS["vision"] = _fake_engine([("First line", 0.9), ("Sign", 0.3)])
    try:
        with Workspace() as ws:
            sup = pgs.write_sup(events, ws.path("Show.fr.sup"), (1920, 1080))
            job = {"id": "j", "task": "ocr", "input": {"subtitle": {"path": sup}},
                   "options": {"engine": "vision", "language": "auto", "forced": "split", "saveFlaggedImages": True},
                   "output": {}}
            result = ocr.run_task(job, context(ws.root))
            names = sorted(os.path.basename(o.path) for o in result.outputs)
            assert names == ["Show.forced.fr.srt", "Show.fr.flagged", "Show.fr.srt"], names
            from audiosync.subs import formats

            main = formats.read(ws.path("Show.fr.srt"))
            assert [c.text for c in main.cues] == ["First line", "Sign", "Sign"]
            assert main.cues[1].align == 8 and main.cues[0].align in (None, 2)
            forced = formats.read(ws.path("Show.forced.fr.srt"))
            assert [(c.start, c.text) for c in forced.cues] == [(1.5, "Sign")]
            flagged = sorted(os.listdir(ws.path("Show.fr.flagged")))
            assert flagged == ["0001_00-00-01.500.png", "0002_00-00-04.000.png", "flagged.tsv"], flagged
            review = imaging.read_png(ws.path("Show.fr.flagged/0001_00-00-01.500.png"))
            assert review.shape[:2] == narrow.shape[:2]
            report = result.report
            assert (report["events"], report["cues"], report["forced"], report["lowConfidence"]) == (3, 3, 1, 2)
            assert report["language"] == "fr" and result.preview["count"] == 3
            # forcedOnly writes just the forced caption.
            ocr_engines.RUNNERS["vision"] = _fake_engine([("Sign", 0.9)])
            job["options"]["forced"] = "forcedOnly"
            job["output"] = {"dir": ws.path("only")}
            only = ocr.run_task(job, context(ws.root))
            assert [(c.start, c.text) for c in formats.read(only.outputs[0].path).cues] == [(1.5, "Sign")]
            assert only.report["events"] == 1
    finally:
        ocr_engines.RUNNERS["vision"] = saved


def test_fallback_engine_rereads_unsure_lines():
    events = [pgs.BitmapEvent(1.0, 2.0, outlined(bars(40, 300)), 700, 950),
              pgs.BitmapEvent(3.0, 4.0, outlined(bars(40, 200, pitch=14)), 700, 950)]
    saved = dict(ocr_engines.RUNNERS)
    ocr_engines.RUNNERS["vision"] = _fake_engine([("sure", 0.9), ("unsure", 0.2)])
    ocr_engines.RUNNERS["tesseract"] = _fake_engine([("better", 0.8)])
    try:
        with Workspace() as ws:
            outcome = ocr.ocr_events(events, {"engine": "vision", "fallbackEngine": "tesseract", "language": "en"},
                                     context(ws.root), "en")
    finally:
        ocr_engines.RUNNERS.update(saved)
    assert [c.text for c in outcome.doc.cues] == ["sure", "better"]
    assert outcome.report["fallbackUsed"] == 1 and outcome.report["lowConfidence"] == 0


# ------------------------------------------------- Apple Vision, end to end

ENGLISH = [
    {"lines": ["Where did you put the keys?", "I don't know, I left them here."]},
    {"lines": ["Somewhere far away, a bell rang."], "italic": True},
    {"lines": ["It's 7:45 and we're late again!"], "fill": [255, 230, 0]},
    {"lines": ["NORTH GATE: STAFF ONLY"], "top": True, "forced": True},
    {"lines": ["Can you hear me, Jonathan?"]},
]
JAPANESE = [
    {"lines": ["今日はいい天気ですね", "散歩に行きましょう"], "ruby": [[5, 7, "てんき"]]},
    {"lines": ["コーヒーを飲みたい"]},
    {"lines": ["東京駅はどこですか？"]},
    {"lines": ["第一章"], "top": True, "forced": True},
    {"lines": ["東京駅", "三番線"], "vertical": True},
]


def _render_events(captions, font, size):
    specs = []
    for cap in captions:
        lines = [{"text": t} for t in cap["lines"]]
        if cap.get("ruby"):
            lines[0]["ruby"] = cap["ruby"]
        specs.append({"lines": lines, "font": font, "size": size, "outline": 4, "italic": cap.get("italic", False),
                      "fill": cap.get("fill", [255, 255, 255]), "vertical": cap.get("vertical", False)})
    images = render(specs)
    events = []
    for i, (cap, rgba) in enumerate(zip(captions, images)):
        h, w = rgba.shape[:2]
        if cap.get("vertical"):
            x, y = 1920 - 120 - w, 120
        else:
            x, y = (1920 - w) // 2, 60 if cap.get("top") else 1080 - 70 - h
        events.append(pgs.BitmapEvent(2.0 + 3 * i, 4.5 + 3 * i, rgba, x, y, cap.get("forced", False)))
    return events


def _vision_available():
    if sys.platform != "darwin":
        raise Skip("Apple Vision is macOS only")
    if ocr_engines.vision_helper(build=False) is None and not (ocr_engines.vision_source() and tool("swiftc")):
        raise Skip("no Vision helper and no swiftc to build it")


def _ocr_task(ws, events, language, name, extra=None):
    sup = pgs.write_sup(events, ws.path(f"{name}.sup"), (1920, 1080))
    options = {"engine": "vision", "language": language}
    options.update(extra or {})
    job = {"id": name, "task": "ocr", "input": {"subtitle": {"path": sup}}, "options": options, "output": {}}
    return ocr.run_task(job, context(ws.root))


def _score(cues, captions, language):
    """Per-caption character accuracy (tags, line breaks and -- for CJK --
    spaces ignored); prints them, returns the overall accuracy."""
    total_err = total_len = 0
    for cue, cap in zip(cues, captions):
        want = " ".join(cap["lines"])
        got = re.sub(r"</?i>|\{\\an\d\}", "", cue.text).replace("\n", " ")
        if language == "ja":
            want, got = want.replace(" ", ""), got.replace(" ", "")
        acc = accuracy(got, want)
        total_err += (1 - acc) * len(want)
        total_len += len(want)
        print(f"        {acc:6.1%}  {got!r}")
    return 1 - total_err / max(1, total_len)


@skippable
def test_vision_reads_english_captions_end_to_end():
    _vision_available()
    from audiosync.subs import formats

    events = _render_events(ENGLISH, "HelveticaNeue-Medium", 52)
    with Workspace() as ws:
        result = _ocr_task(ws, events, "en", "Movie")
        doc = formats.read(result.outputs[0].path)
        assert os.path.basename(result.outputs[0].path) == "Movie.en.srt"
        cues = sorted(doc.cues, key=lambda c: c.start)
        assert len(cues) == len(ENGLISH)
        overall = _score(cues, ENGLISH, "en")
        print(f"        English character accuracy {overall:.1%}")
        assert overall >= 0.95, overall
        assert cues[1].text.startswith("<i>") and "<i>" not in cues[0].text + cues[2].text + cues[4].text
        assert cues[3].align == 8
        assert result.report["italics"] == 1 and result.report["forced"] == 1


@skippable
def test_vision_reads_japanese_with_furigana_and_vertical_text():
    _vision_available()
    from audiosync.subs import formats

    events = _render_events(JAPANESE, "HiraginoSans-W6", 56)
    with Workspace() as ws:
        result = _ocr_task(ws, events, "ja", "Anime", {"forced": "split"})
        doc = formats.read(result.outputs[0].path)
        cues = sorted(doc.cues, key=lambda c: c.start)
        assert len(cues) == len(JAPANESE)
        overall = _score(cues, JAPANESE, "ja")
        print(f"        Japanese character accuracy {overall:.1%}")
        assert overall >= 0.95, overall
        assert "てんき" not in cues[0].text and result.report["furiganaRemoved"] >= 1
        assert result.report["vertical"] == 1 and cues[4].align == 9
        assert cues[3].align == 8
        forced = formats.read(result.outputs[1].path)
        assert [c.text for c in forced.cues] == ["{\\an8}第一章"] or [c.plain for c in forced.cues] == ["第一章"]


@skippable
def test_vision_reads_a_pgs_track_inside_a_video():
    """input.subtitle = {path: video, track: 0}: extracted with tracks.extract."""
    _vision_available()
    ffmpeg, mkvmerge = tool("ffmpeg"), tool("mkvmerge")
    events = _render_events(ENGLISH[:1], "HelveticaNeue-Medium", 52)
    with Workspace() as ws:
        sup = pgs.write_sup(events, ws.path("subs.sup"), (1920, 1080))
        video = ws.path("video.mkv")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=320x180:d=6:r=25",
                        "-c:v", "mpeg4", video], check=True)
        clip = ws.path("Clip.mkv")
        subprocess.run([mkvmerge, "-q", "-o", clip, video, "--language", "0:eng", sup], check=True)
        job = {"id": "v", "task": "ocr", "input": {"subtitle": {"path": clip, "track": 0}},
               "options": {"engine": "vision", "language": "auto"}, "output": {}}
        result = ocr.run_task(job, context(ws.root))
        assert os.path.basename(result.outputs[0].path) == "Clip.track1.en.srt"
        from audiosync.subs import formats

        (cue,) = formats.read(result.outputs[0].path).cues
        assert accuracy(cue.text.replace("\n", " "), " ".join(ENGLISH[0]["lines"])) >= 0.95
        assert abs(cue.start - 2.0) < 0.002


@skippable
def test_vision_reads_vobsub_made_by_ffmpeg():
    """DVD path: our .sup -> FFmpeg dvdsub in MKV -> mkvextract .idx/.sub -> OCR."""
    _vision_available()
    ffmpeg, mkvextract = tool("ffmpeg"), tool("mkvextract")
    events = _render_events(ENGLISH[:2], "HelveticaNeue-Medium", 52)
    with Workspace() as ws:
        sup = pgs.write_sup(events, ws.path("in.sup"), (1920, 1080))
        mkv = ws.path("dvd.mkv")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-copyts", "-fix_sub_duration", "-i", sup, "-map", "0:s",
                        "-c:s", "dvdsub", "-f", "matroska", mkv], check=True)
        subprocess.run([mkvextract, mkv, "tracks", f"0:{ws.path('Film.idx')}"], check=True, capture_output=True)
        job = {"id": "d", "task": "ocr", "input": {"subtitle": {"path": ws.path("Film.idx")}},
               "options": {"engine": "vision", "language": "en"}, "output": {}}
        result = ocr.run_task(job, context(ws.root))
        from audiosync.subs import formats

        cues = formats.read(result.outputs[0].path).cues
        assert len(cues) == 2 and result.report["source"] == "vobsub"
        overall = _score(cues, ENGLISH[:2], "en")
        print(f"        VobSub character accuracy {overall:.1%}")
        assert overall >= 0.95


if __name__ == "__main__":
    sys.exit(1 if run_module(dict(globals())) else 0)
