"""Tests for subtitle generation around the recogniser.

The recogniser itself needs a pack and a model, so here it is replaced by a
fake worker that answers with canned segments (tests/manual/generate_real.py
runs the real thing). What is tested is everything the engine adds: the
speech regions it hands Whisper, the timeline mapping, snapping word edges
onto the speech, removing what Whisper invents, and the task's output.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from audiosync.media import CancellationToken, ffmpeg_path  # noqa: E402
from audiosync.subs import generate, packs  # noqa: E402
from audiosync.subs.model import Word  # noqa: E402
from audiosync.subs.tasks import TaskContext, TaskError  # noqa: E402

SR = 16000
REAL_WORKERS = packs.workers_dir()


def _load_worker(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", os.path.join(REAL_WORKERS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def seg(start, end, text, words=None, **extra):
    base = {"start": start, "end": end, "text": text, "avgLogprob": -0.2, "noSpeechProb": 0.05,
            "compressionRatio": 1.2, "temperature": 0.0}
    base.update(extra)
    if words is None:
        pieces = text.split(" ") if " " in text.strip() else [text]
        step = (end - start) / max(1, len(pieces))
        words = [{"start": start + i * step, "end": start + (i + 1) * step,
                  "text": (" " if i else "") + p, "probability": 0.9} for i, p in enumerate(pieces)]
    base["words"] = words
    return base


# ------------------------------------------------------------------ options


def test_options_defaults_and_validation():
    opts = generate._options({"language": "jpn", "beamSize": 99})
    assert opts["engine"] == "mlx-whisper" and opts["model"] == "large-v3"
    assert opts["language"] == "ja" and opts["beamSize"] == 10 and opts["vad"] is True
    assert generate._options({"language": "auto"})["language"] == "auto"
    for bad in ({"engine": "whisper.cpp"}, {"language": "Klingon"}, {"task": "summarise"}):
        try:
            generate._options(bad)
        except TaskError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_engine_statuses_name_their_pack():
    with env(AUDIOSYNC_PACKS_DIR=tempfile.mkdtemp(), AUDIOSYNC_PACK_PYTHON_ASR_MLX=None,
             AUDIOSYNC_PACK_PYTHON_ASR_FASTER=None):
        statuses = {s["id"]: s for s in generate.engine_statuses()}
        assert set(statuses) == {"mlx-whisper", "faster-whisper"}
        assert statuses["faster-whisper"]["pack"] == "asr-faster" and not statuses["faster-whisper"]["available"]
        assert "Install" in statuses["faster-whisper"]["reason"]
        with env(AUDIOSYNC_PACK_PYTHON_ASR_FASTER=sys.executable):
            assert {s["id"]: s for s in generate.engine_statuses()}["faster-whisper"]["available"]


# ----------------------------------------------------------- hallucinations


def test_known_hallucinations_in_several_languages():
    invented = [
        ("Thanks for watching!", "en"),
        ("Thank you so much for watching, see you next time!", "en"),
        ("ご視聴ありがとうございました", "ja"),
        ("チャンネル登録と高評価をお願いします", "ja"),
        ("Sous-titres réalisés par la communauté d'Amara.org", "fr"),
        ("Subtítulos realizados por la comunidad de Amara.org", "es"),
        ("Sottotitoli creati dalla comunità Amara.org", "it"),
        ("Untertitel im Auftrag des ZDF, 2021", "de"),
        ("시청해 주셔서 감사합니다.", "ko"),
        ("MBC 뉴스 이덕영입니다.", "ko"),
        ("请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "zh"),
        ("Thanks for watching!", "ja"),  # English sign-offs appear over any language
    ]
    for text, lang in invented:
        assert generate._known_hallucination(text, lang), (text, lang)
    real = [
        ("Thank you for watching the kids tonight.", "en"),
        ("今日はいい天気ですね。", "ja"),
        ("Merci beaucoup, à demain.", "fr"),
        ("감사합니다.", "ko"),
    ]
    for text, lang in real:
        assert generate._known_hallucination(text, lang) is None, (text, lang)


def test_repetition_loops_are_cut_to_two():
    words = [{"text": t} for t in ["I", " said", " no", " no", " no", " no", " no", " no", " okay"]]
    kept, loops = generate._collapse_loops(words)
    assert [w["text"] for w in kept] == ["I", " said", " no", " no", " okay"] and loops == 1
    phrase = [{"text": t} for t in ["ありがとう", "ございます"] * 5]
    kept, loops = generate._collapse_loops(phrase)
    assert len(kept) == 4 and loops == 1
    normal = [{"text": t} for t in [" yes", " yes", " I", " know"]]
    assert generate._collapse_loops(normal) == (normal, 0)


def test_clean_segments_removes_what_whisper_invents():
    regions = [(0.0, 10.0), (20.0, 30.0)]
    segments = [
        seg(1.0, 3.0, " Hello there."),
        seg(3.5, 5.0, " Subtitles by the Amara.org community"),
        seg(5.5, 7.0, " Maybe not.", noSpeechProb=0.9, avgLogprob=-1.1),
        seg(12.0, 14.0, " Words in the silence"),  # outside both regions
        seg(21.0, 22.0, " [Music]"),
        seg(22.0, 23.0, " Pinocchio Geppetto"),  # the prompt echoed back
        seg(23.0, 24.0, " Again and again."),
        seg(24.0, 25.0, " Again and again."),
        seg(25.0, 26.0, " Again and again."),
    ]
    kept, removed = generate.clean_segments(segments, "en", regions, prompt="Pinocchio, Geppetto")
    texts = [s["text"].strip() for s in kept]
    assert texts == ["Hello there.", "Again and again.", "Again and again."], texts
    reasons = sorted(r["reason"] for r in removed)
    assert reasons == ["outside detected speech", "repeated line", "repeated the prompt",
                       "scored as not speech", "subtitle credit"], reasons
    # SDH keeps Whisper's own sound tags; suppression off keeps everything else.
    kept, _ = generate.clean_segments(segments, "en", regions, sdh=True)
    assert any(s.get("sound") and s["text"].strip() == "[Music]" for s in kept)
    kept, removed = generate.clean_segments(segments, "en", regions, suppress=False)
    assert len(kept) == 8 and removed == []


def chars(start, texts, probs, step=0.3, gap=0.0):
    """Japanese words as Whisper splits them: one token each, no spaces."""
    out, t = [], start
    for text, p in zip(texts, probs):
        out.append({"start": round(t, 3), "end": round(t + step, 3), "text": text, "probability": p})
        t += step + gap
    return out


def test_short_genuine_lines_survive_the_filter():
    # Mugen Train, 50-68 s: whispered names, one per breath, then shouts at
    # 214-230 s. Figures are Whisper's own (large-v3-turbo): one window's
    # avgLogprob on every segment of it, a doubted first character.
    names = [(50.24, "マサオ", [0.077, 0.985, 0.975]), (54.29, "ユウタ", [0.82, 0.68, 0.93]),
             (56.55, "ヤイチ", [0.56, 0.99, 0.998]), (58.43, "トヨガキ", [0.54, 0.998, 0.97, 0.998]),
             (60.59, "イサオ", [0.73, 0.985, 0.988]), (62.51, "リョウスケ", [0.72, 0.8, 0.985, 0.98, 0.98]),
             (64.53, "ユキオ", [0.956, 0.996, 0.543]), (66.47, "タカヒロ", [0.855, 0.99, 0.99, 0.99])]
    segments = []
    for start, text, probs in names:
        words = chars(start, list(text), probs, step=0.25)
        segments.append(seg(start, words[-1]["end"], text, words, avgLogprob=-0.51, noSpeechProb=0.49))
    shouts = [(214.02, ["う", "る", "し", "な!"], [0.19, 0.87, 0.4, 0.36]), (215.18, ["バ", "カ!"], [0.45, 0.81]),
              (216.28, ["俺", "た", "ち", "も", "行", "こう!"], [0.74, 0.84, 1.0, 0.998, 0.99, 0.99]),
              (228.80, ["イ", "ェ", "ー", "イ!"], [0.10, 0.30, 0.67, 0.96]), (229.78, ["デ", "イ", "ズ!"], [0.40, 0.87, 0.49])]
    for start, pieces, probs in shouts:
        words = chars(start, pieces, probs, step=0.15)
        segments.append(seg(start, words[-1]["end"], "".join(pieces), words, avgLogprob=-0.81))
    # A real line decoded on a retry (temperature 0.6) that passed with room.
    words = chars(79.7, list("お館様"), [0.6, 0.9, 0.8])
    segments.append(seg(79.7, 80.6, "お館様", words, avgLogprob=-0.44, temperature=0.6))
    regions = [(49.8, 70.3), (79.1, 84.6), (213.6, 231.9)]
    kept, removed = generate.clean_segments(segments, "ja", regions)
    assert removed == [], removed
    assert [s["text"] for s in kept] == [s["text"] for s in segments]


def test_latin_loop_glued_onto_a_japanese_transcript_is_removed():
    # What reached the subtitles at 208.7-217.3 s before:
    # 「いても行くしかないよProblem」 then 「Problem Problem Problem」.
    first = chars(208.0, ["いて", "も", "行", "く", "しか", "ないよ"], [0.05, 0.99, 0.85, 0.999, 0.997, 0.99])
    first.append({"start": 209.8, "end": 212.0, "text": "Problem", "probability": 0.62})
    loop = [{"start": 214.4 + i, "end": 215.2 + i, "text": (" " if i else "") + "Problem", "probability": 0.7}
            for i in range(3)]
    segments = [seg(208.0, 212.0, "いても行くしかないよProblem", first),
                seg(214.4, 217.3, "Problem Problem Problem", loop)]
    kept, removed = generate.clean_segments(segments, "ja", [(206.0, 220.0)])
    assert [s["text"] for s in kept] == ["いても行くしかないよ"], kept
    assert abs(kept[0]["end"] - first[-2]["end"]) < 1e-9, kept[0]  # ends with its last real word
    assert [(r["text"], r["reason"]) for r in removed] == [
        ("Problem", "repeated Latin word"), ("Problem Problem Problem", "repetition loop")], removed
    assert all(set(r) == {"start", "end", "text", "reason"} for r in removed)

    # Once, glued on and doubted: appended, not heard.
    words = chars(1.0, ["行", "こう"], [0.9, 0.95]) + [{"start": 1.6, "end": 2.0, "text": "Yeah", "probability": 0.2}]
    kept, removed = generate.clean_segments([seg(1.0, 2.0, "行こうYeah", words)], "ja", None)
    assert kept[0]["text"] == "行こう" and removed[0]["reason"] == "stray Latin word", (kept, removed)

    # English really said in Japanese dialogue stays: a confident word, an
    # acronym said twice, and any Latin in an English transcript.
    keep = [
        seg(1.0, 2.0, "新しいiPhone", chars(1.0, ["新しい"], [0.9]) + [{"start": 1.3, "end": 2.0, "text": "iPhone", "probability": 0.93}]),
        seg(3.0, 4.0, "OK、行こう", [{"start": 3.0, "end": 3.3, "text": "OK", "probability": 0.8}] + chars(3.3, ["、行", "こう"], [0.9, 0.9])),
        seg(5.0, 6.0, "OK", [{"start": 5.0, "end": 5.4, "text": "OK", "probability": 0.8}]),
    ]
    kept, removed = generate.clean_segments(keep, "ja", None)
    assert removed == [] and len(kept) == 3, removed
    kept, removed = generate.clean_segments([seg(1.0, 3.0, " Problem solved, no problem.")], "en", None)
    assert removed == [] and len(kept) == 1, removed


def test_single_token_loops_and_failed_decodes_are_removed():
    loop = [{"start": 16.0 + i * 0.5, "end": 16.4 + i * 0.5, "text": "55", "probability": p}
            for i, p in enumerate([0.0002, 0.1, 0.4])]
    segments = [
        # All temperatures failed: logprob NaN, compression ratio 9.6.
        seg(15.95, 17.07, "5555", loop[:2], avgLogprob=float("nan"), compressionRatio=9.62, temperature=1.0),
        # One token three times, doubted: a loop whatever the script.
        seg(18.0, 19.5, "555555", loop, avgLogprob=-0.5),
        # Sampled retry that barely passed, over the opening score.
        seg(21.0, 22.0, "エンディボロックス", chars(21.0, ["エ", "ンデ", "ィ", "ボ", "ロ", "ック", "ス"],
                                             [0.019, 0.6, 0.91, 0.31, 0.17, 0.72, 0.99]),
            avgLogprob=-0.97, temperature=0.2),
        # Half-scored as not speech and doubted word by word (large-v3 over
        # the same music).
        seg(24.0, 25.0, "エヴィトレックス", chars(24.0, ["エ", "ヴィ", "ト", "レ", "ッ", "クス"],
                                           [0.20, 0.52, 0.82, 0.20, 0.26, 0.79]),
            avgLogprob=-0.87, noSpeechProb=0.57),
        # 「はいはいはい」 said for real is heard with confidence.
        seg(30.0, 31.0, "はいはいはい", chars(30.0, ["はい", "はい", "はい"], [0.9, 0.95, 0.97])),
    ]
    kept, removed = generate.clean_segments(segments, "ja", None)
    assert [s["text"] for s in kept] == ["はいはいはい"], kept
    assert [r["reason"] for r in removed] == ["failed decoding", "repetition loop", "low confidence",
                                              "scored as not speech"], removed


def test_report_lists_what_was_removed_capped():
    removed = [{"start": float(i), "end": i + 0.5, "text": f"x{i}", "reason": "known phrase"} for i in range(80)]
    removed[0] = dict(removed[0], text="ビ" * 600, reason="failed decoding")
    info = {"language": "ja", "spokenLanguage": "ja", "languageProbability": 0.9, "durationS": 100.0,
            "elapsedS": 1.0, "realtimeFactor": 0.01, "engine": "mlx-whisper", "model": "large-v3",
            "decoding": "", "device": "", "vad": "energy", "removed": removed,
            "removedHallucinations": len(removed), "soundCues": []}
    root = tempfile.mkdtemp()
    media = os.path.join(root, "clip.mkv")
    open(media, "wb").close()
    original = generate.transcribe
    generate.transcribe = lambda media, opts, ctx: ([Word(1.0, 1.5, "今日")], "ja", info)
    try:
        ctx = TaskContext(token=CancellationToken(), progress=lambda p, s: None, log=lambda m: None, workdir=root)
        result = generate.run_task({"id": "g", "task": "generate", "input": {"video": {"path": media}},
                                    "options": {}, "output": {"format": "srt"}}, ctx)
    finally:
        generate.transcribe = original
        shutil.rmtree(root, ignore_errors=True)
    assert result.report["removedHallucinations"] == 80
    listed = result.report["removed"]
    assert listed[1:] == removed[1:generate.REPORT_REMOVED_MAX] and len(listed) == 50
    assert listed[0]["text"] == "ビ" * generate.REPORT_TEXT_MAX + "…" and listed[0]["reason"] == "failed decoding"
    assert generate.REPORT_REMOVED_MAX == 50 and "removed 80 likely hallucinations" in result.summary


# ------------------------------------------------------------------ speech


def test_speech_regions_are_merged_padded_and_filtered():
    hop = 0.01
    probs = np.zeros(3000, dtype=np.float32)
    probs[100:300] = 0.9    # 1.0-3.0
    probs[400:500] = 0.9    # 4.0-5.0: 1 s gap -> merged
    probs[1000:1005] = 0.9  # 10.0-10.05: too short
    probs[2000:2200] = 0.9  # 20.0-22.0
    regions = generate.speech_regions(probs, hop, 30.0)
    assert [(round(s, 2), round(e, 2)) for s, e in regions] == [(0.6, 5.4), (19.6, 22.4)], regions
    # Hysteresis: a run that never reaches ON is not speech.
    probs = np.full(1000, 0.4, dtype=np.float32)
    assert generate.speech_regions(probs, hop, 10.0, on=0.5, off=0.35) == []
    # A 2.4 s pause shrinks to 1.6 s once both sides are padded: still a
    # pause, kept whole rather than swapped for the worker's 1 s spacer
    # (Mugen Train lost a whispered "Susumu" in exactly such a gap).
    probs = np.zeros(1000, dtype=np.float32)
    probs[100:200] = 0.9  # 1.0-2.0
    probs[440:540] = 0.9  # 4.4-5.4
    assert [(round(s, 2), round(e, 2)) for s, e in generate.speech_regions(probs, hop, 10.0)] == [(0.6, 5.8)]


def test_loud_gaps_between_speech_are_joined_quiet_ones_are_not():
    # dBFS of each stretch: voices at -30, a train and music at -30 between
    # two of them (the detector heard no speech under it), true pauses at
    # -68, and a loud stretch that is too long to be a gap in a scene.
    loud = {(214.0, 220.9): -31.9, (220.9, 228.8): -29.9, (228.8, 231.9): -29.6,
            (70.3, 79.1): -68.4, (79.1, 84.6): -47.3, (50.0, 70.3): -53.1,
            (231.9, 250.0): -22.0, (250.0, 255.0): -25.0}
    regions = [(50.0, 70.3), (79.1, 84.6), (214.0, 220.9), (228.8, 231.9), (250.0, 255.0)]
    joined = generate.join_loud_gaps(regions, lambda a, b: loud.get((a, b), -90.0))
    assert joined == [(50.0, 70.3), (79.1, 84.6), (214.0, 231.9), (250.0, 255.0)], joined
    assert generate.join_loud_gaps(regions[:1], lambda a, b: 0.0) == regions[:1]


def test_level_db_reads_the_wav():
    root = tempfile.mkdtemp()
    try:
        path = os.path.join(root, "tone.wav")
        _tone_wav(path, 2.0)  # 0.2 amplitude sine: -17 dBFS
        assert abs(generate._level_db(path, 0.5, 1.5) - 20 * np.log10(0.2 / np.sqrt(2))) < 0.1
        assert generate._level_db(path, 5.0, 6.0) == -120.0  # past the end
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_worker_decodes_speech_in_chunks_that_start_at_long_pauses():
    asr = _load_worker("asr_worker")
    audio = np.zeros(300 * SR, dtype=np.float32)
    assert asr._chunks(audio, None)[0][1] is None  # no detection: one piece, no map
    # Opening music (13.6-34.6 s), a 14.6 s pause, the names (49.2-70.3 s):
    # the names must start a window of their own. Regions in one chunk never
    # exceed a window; a region longer than a window is a chunk by itself.
    speech = [[13.6, 18.9], [21.3, 30.8], [32.6, 34.6], [49.2, 70.3], [72.0, 79.8],
              [82.0, 150.0], [151.0, 152.0], [153.0, 154.0], [158.5, 159.0]]
    pieces = asr._chunks(audio, speech)
    firsts = [round(t.orig_start[0] / SR, 1) for _a, t in pieces]
    assert firsts == [13.6, 49.2, 82.0, 151.0, 158.5], firsts
    assert all(len(a) <= asr.CHUNK_S * SR or len(t.lengths) == 1 for a, t in pieces)
    # Each chunk's own map puts its words back on the original timeline.
    audio_names, tmap = pieces[1]
    second = 70.3 - 49.2 + asr.SPACER_S
    segments = [{"start": 1.0, "end": second + 0.5, "text": "x",
                 "words": [{"start": 1.0, "end": 1.8, "text": "マサオ"},
                           {"start": second + 0.1, "end": second + 0.5, "text": "ふっ"}]}]
    asr._restore(segments, tmap)
    assert [(w["start"], w["end"]) for w in segments[0]["words"]] == [(50.2, 51.0), (72.1, 72.5)], segments


def test_snap_words_moves_edges_onto_the_speech():
    hop = 0.01
    frames = np.zeros(1000, dtype=bool)
    frames[200:400] = True  # speech 2.0-4.0
    frames[600:800] = True  # speech 6.0-8.0
    words = [
        Word(1.65, 2.3, " early"),   # starts 350 ms before the voice
        Word(2.3, 3.6, " middle"),   # ends 400 ms before the voice stops
        Word(5.5, 5.95, " squeezed"),  # wholly in the silence before 6.0
        Word(5.95, 6.5, " next"),
    ]
    moved = generate.snap_words(words, generate.Speech([], frames, hop, "test"))
    assert moved >= 3
    assert abs(words[0].start - 2.0) < 1e-6, words[0]
    assert abs(words[1].end - 4.0) < 1e-6, words[1]
    assert abs(words[2].start - 6.0) < 1e-6 and words[2].end <= words[3].end, words[2]


def test_time_map_restores_the_original_timeline_across_the_spacer():
    asr = _load_worker("asr_worker")
    audio = np.zeros(20 * SR, dtype=np.float32)
    work, tmap = asr._cut_speech(audio, [[1.0, 3.0], [10.0, 12.0]])
    spacer = asr.SPACER_S
    assert len(work) == int((4.0 + spacer) * SR), len(work)
    second = 2.0 + spacer  # where the second stretch starts in the concatenation
    segments = [{
        "start": 0.5, "end": second + 1.0, "text": "x",
        "words": [
            {"start": 0.5, "end": 1.0, "text": "a"},                       # stretch 1
            {"start": 1.8, "end": 2.0, "text": "b"},                       # end of stretch 1
            {"start": second - 0.4, "end": second + 0.2, "text": "c"},    # aligned early into the spacer
            {"start": second + 0.5, "end": second + 1.0, "text": "d"},    # stretch 2
        ],
    }]
    asr._restore(segments, tmap)
    got = [(w["text"], w["start"], w["end"]) for w in segments[0]["words"]]
    assert got == [("a", 1.5, 2.0), ("b", 2.8, 3.0), ("c", 10.0, 10.2), ("d", 10.5, 11.0)], got
    assert (segments[0]["start"], segments[0]["end"]) == (1.5, 11.0)


# --------------------------------------------------------------- the task


FAKE_ASR = '''
def handle(req):
    with open("request.json", "w") as handle:
        json.dump(req, handle)
    proto.progress(50, "Recognising speech")
    def words(start, texts, step=0.4):
        return [{"start": start + i * step, "end": start + (i + 1) * step, "text": t, "probability": 0.9}
                for i, t in enumerate(texts)]
    segs = [
        {"start": 1.0, "end": 2.6, "text": "今日はいい天気ですね。", "avgLogprob": -0.2, "noSpeechProb": 0.01,
         "compressionRatio": 1.0, "temperature": 0.0, "words": words(1.0, ["今日", "は", "いい", "天気ですね。"])},
        {"start": 5.0, "end": 6.2, "text": "ご視聴ありがとうございました", "avgLogprob": -0.3, "noSpeechProb": 0.5,
         "compressionRatio": 1.0, "temperature": 0.0, "words": words(5.0, ["ご視聴", "ありがとう", "ございました"])},
    ]
    return {"language": "ja", "languageProbability": 0.97, "segments": segs, "engine": "faster-whisper",
            "decoding": "beam search, 5 beams", "device": "cpu (int8)", "loadS": 0.1, "recogniseS": 0.2,
            "duration": 8.0, "speechDuration": 3.0}
proto.main(handle)
'''

FAKE_VAD = '''
import numpy as np
def handle(req):
    probs = np.zeros(250, dtype="<f4")      # 8 s of 32 ms windows
    probs[31:82] = 0.95                      # speech 1.0-2.6 s
    path = os.path.join(req["outDir"], "silero_probs.f32")
    probs.tofile(path)
    return {"hopS": 0.032, "probsPath": path, "count": 250, "duration": 8.0}
proto.main(handle)
'''


def _tone_wav(path: str, seconds: float = 8.0) -> None:
    t = np.arange(int(seconds * SR)) / SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((x * 32767).astype("<i2").tobytes())


@contextlib.contextmanager
def fake_engine():
    root = tempfile.mkdtemp(prefix="subsync-generate-test-")
    workers = os.path.join(root, "workers")
    os.makedirs(workers)
    shutil.copy(os.path.join(REAL_WORKERS, "_protocol.py"), workers)
    header = "import os, sys, json\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport _protocol as proto\n"
    for name, body in (("asr_worker.py", FAKE_ASR), ("vad_worker.py", FAKE_VAD)):
        with open(os.path.join(workers, name), "w", encoding="utf-8") as handle:
            handle.write(header + body)
    hub = os.path.join(root, "hf")
    snap = os.path.join(hub, "models--Systran--faster-whisper-tiny", "snapshots", "abc")
    os.makedirs(snap)
    open(os.path.join(snap, "model.bin"), "wb").close()
    original = packs.workers_dir
    packs.workers_dir = lambda: workers
    try:
        with env(AUDIOSYNC_PACKS_DIR=os.path.join(root, "packs"), HF_HUB_CACHE=hub,
                 AUDIOSYNC_PACK_PYTHON_ASR_FASTER=sys.executable, AUDIOSYNC_PACK_PYTHON_ASR_MLX=None):
            yield root
    finally:
        packs.workers_dir = original
        shutil.rmtree(root, ignore_errors=True)


def _have_ffmpeg() -> bool:
    try:
        return subprocess.run([ffmpeg_path(), "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


def test_generate_task_end_to_end_with_a_fake_recogniser():
    if not _have_ffmpeg():
        print("        (skipped: no ffmpeg)")
        return
    with fake_engine() as root:
        media = os.path.join(root, "Episode 1.wav")
        _tone_wav(media)
        workdir = os.path.join(root, "work")
        os.makedirs(workdir)
        progress, logs = [], []
        ctx = TaskContext(token=CancellationToken(), progress=lambda p, s: progress.append((p, s)),
                          log=logs.append, workdir=workdir)
        job = {"id": "g1", "task": "generate", "input": {"video": {"path": media, "audioTrack": 0}},
               "options": {"engine": "faster-whisper", "model": "tiny", "language": "auto", "vad": True},
               "output": {"format": "srt"}}
        result = generate.run_task(job, ctx)
        request = json.load(open(os.path.join(workdir, "request.json")))

        assert request["engine"] == "faster" and request["language"] is None and request["beamSize"] == 5
        assert request["modelPath"].endswith(os.path.join("snapshots", "abc"))
        assert request["speech"] == [[0.592, 3.024]], request["speech"]  # 0.992-2.624 padded by 0.4 s
        assert request["hallucinationSilenceThreshold"] == generate.HALLUCINATION_SILENCE_S

        out = result.outputs[0].path
        assert os.path.basename(out) == "Episode 1.ja.srt", out
        text = open(out, encoding="utf-8").read()
        assert "今日はいい天気ですね。" in text and "ご視聴" not in text, text
        report = result.report
        assert report["language"] == "ja" and report["languageName"] == "Japanese"
        assert report["languageProbability"] == 0.97 and report["removedHallucinations"] == 1
        assert report["cues"] == 1 and report["words"] == 4 and report["vad"] == "silero"
        assert report["removed"] == [{"start": 5.0, "end": 6.2, "text": "ご視聴ありがとうございました",
                                      "reason": "known phrase"}], report["removed"]
        assert abs(report["durationS"] - 8.0) < 0.05 and report["realtimeFactor"] > 0
        assert "Japanese" in result.summary and "1 likely hallucination" in result.summary, result.summary
        assert result.preview["cues"][0]["start"] >= 0.99, result.preview
        assert any(s == "Extracting audio" for _p, s in progress)
        assert any("ご視聴" in m for m in logs), logs


def test_missing_audio_track_is_a_clear_error():
    if not _have_ffmpeg():
        return
    with fake_engine() as root:
        media = os.path.join(root, "clip.wav")
        _tone_wav(media, 1.0)
        ctx = TaskContext(token=CancellationToken(), progress=lambda p, s: None, log=lambda m: None, workdir=root)
        try:
            generate.extract_audio(media, 1, os.path.join(root, "x.wav"), ctx)
        except TaskError as exc:
            assert "no audio track 2" in str(exc), exc
        else:
            raise AssertionError("missing track not reported")


def test_generate_without_the_pack_says_what_to_install():
    with env(AUDIOSYNC_PACKS_DIR=tempfile.mkdtemp(), AUDIOSYNC_PACK_PYTHON_ASR_FASTER=None):
        ctx = TaskContext(token=CancellationToken(), progress=lambda p, s: None, log=lambda m: None,
                          workdir=tempfile.mkdtemp())
        try:
            generate.transcribe({"path": "/x.mkv"}, {"engine": "faster-whisper"}, ctx)
        except TaskError as exc:
            assert "Install" in str(exc), exc
        else:
            raise AssertionError("missing pack not reported")


def test_fallback_segmentation_breaks_at_pauses_and_sentences():
    words = [Word(0.0, 0.4, "Hello"), Word(0.4, 0.8, " there."), Word(0.9, 1.3, " How"),
             Word(1.3, 1.6, " are"), Word(1.6, 2.0, " you?"), Word(5.0, 5.5, " Fine")]
    cues = generate._fallback_cues(words, "en")
    assert [c.text for c in cues] == ["Hello there.", "How are you?", "Fine"], [c.text for c in cues]
    assert all(c.end - c.start >= 0.833 - 1e-9 for c in cues)
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
