"""Tests for the timed-text style rules: line breaking, segmentation, check, fix.

The rules are the Netflix Timed Text Style Guide's, per language, and the
tests pin both the numbers (so a refactor cannot quietly change a limit) and
the behaviour a viewer sees: lines that break at natural points, subtitles
cut where people pause, gaps of exactly two frames, cues that start on the
cut. Everything runs offline; the shot-change test builds its own video
with lavfi.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.subs import style  # noqa: E402
from audiosync.subs.model import Cue, SubtitleDoc, Word, strip_tags  # noqa: E402

FPS = 24000 / 1001
FRAME = 1 / FPS


# ------------------------------------------------------------------ helpers


def _words(spec: Sequence[Tuple[str, float, float]], leading_space: bool = False) -> List[Word]:
    return [Word(start=s, end=e, text=(" " + t) if leading_space else t, probability=0.9) for t, s, e in spec]


def _timed(tokens: Sequence[str], start: float = 0.0, per_char: float = 0.06, gap: float = 0.08,
           pauses: dict = None) -> List[Tuple[str, float, float]]:
    """Give each token a duration from its length, with extra pauses after
    the indexes in ``pauses`` -- the shape a recogniser returns."""
    out = []
    t = start
    for i, tok in enumerate(tokens):
        length = max(0.12, len(tok) * per_char)
        out.append((tok, round(t, 3), round(t + length, 3)))
        t += length + gap + (pauses or {}).get(i, 0.0)
    return out


def _widths(text: str, rules) -> List[float]:
    return [style.text_width(line, mode=rules.width) for line in strip_tags(text).split("\n")]


def _fits(text: str, rules) -> bool:
    widths = _widths(text, rules)
    return len(widths) <= rules.max_lines and max(widths) <= rules.cpl


def _assert_timing(cues: List[Cue], rules) -> None:
    for a, b in zip(cues, cues[1:]):
        assert b.start - a.end >= rules.min_gap_s - 1e-6, f"gap {b.start - a.end:.3f} between {a.text!r} / {b.text!r}"
        frames = round((b.start - a.end) * FPS)
        assert frames == 2 or frames >= 12, f"{frames}-frame gap left open between {a.text!r} / {b.text!r}"
    for cue in cues:
        assert cue.duration >= rules.min_duration_s - 1e-6, f"{cue.text!r} only {cue.duration:.3f}s"
        assert cue.duration <= rules.max_duration_s + 1e-6, f"{cue.text!r} {cue.duration:.3f}s"


def _types(issues) -> set:
    return {i.type for i in issues}


# -------------------------------------------------------------------- rules


def test_netflix_numbers_per_language():
    en = style.rules_for("en", {})
    assert (en.cpl, en.cps, en.max_lines, en.max_duration_s) == (42, 20, 2, 7)
    assert abs(en.min_duration_s - 5 / 6) < 1e-9
    assert style.rules_for("eng", {"children": True}).cps == 17
    ja = style.rules_for("ja", {})
    assert (ja.cpl, ja.cps, ja.min_duration_s, ja.width) == (13, 4, 0.5, "cjk")
    assert (style.rules_for("jpn", {"sdh": True}).cpl, style.rules_for("ja", {"sdh": True}).cps) == (16, 7)
    ko = style.rules_for("ko", {})
    assert (ko.cpl, ko.cps, style.rules_for("ko", {"children": True}).cps) == (16, 12, 9)
    zh = style.rules_for("zh-Hans", {})
    assert (zh.cpl, zh.cps, zh.max_lines) == (16, 9, 2)
    assert style.rules_for("chi", {"sdh": True}).max_lines == 3
    fr = style.rules_for("fr", {})
    assert (fr.cpl, fr.cps, fr.dash) == (42, 17, "- ")
    assert style.rules_for("de", {}).dash == "-"
    assert style.rules_for("hi", {}).cps == 22
    assert style.rules_for("th", {}).cpl == 35
    fallback = style.rules_for("sw", {})
    assert (fallback.cpl, fallback.cps, fallback.source) == (42, 17, style.P.GENERAL_SOURCE)
    assert style.rules_for(None, {}).cpl == 42


def test_overrides_and_frame_rates():
    rules = style.rules_for("en", {"cps": 15, "cpl": 37, "maxLines": None, "minGapFrames": 3, "maxDurationS": 6})
    assert (rules.cps, rules.cpl, rules.max_lines, rules.min_gap_frames, rules.max_duration_s) == (15, 37, 2, 3, 6)
    assert abs(rules.min_gap_s - 3 / FPS) < 1e-9
    assert style.rules_for("en", {}).half_second_frames == 12  # 23.976 default
    assert style.rules_for("en", {}, 25).half_second_frames == 12
    assert style.rules_for("en", {}, 30000 / 1001).half_second_frames == 15
    assert style.rules_for("en", {}, 60).half_second_frames == 30
    d = style.rules_for("ja", {"preset": "custom"}).to_dict()
    assert d["preset"] == "custom" and d["cpl"] == 13 and d["maxLines"] == 2 and "minGapFrames" in d


def test_width_counts_like_the_guides():
    assert style.text_width("カメラ", "ja") == 3
    assert style.text_width("iPhoneで", "ja") == 3 + 1  # six half-width = 3
    assert style.text_width("<i>안녕 OK</i>", "ko") == 2 + 0.5 + 1  # Hangul 1; space, Latin 0.5
    # Thai: the tone mark and upper vowel of "น้ำ" are not counted.
    assert style.text_width("น้ำ", "th") == 2
    assert style.text_width("Hello, world", "en") == 12


# ---------------------------------------------------------- line breaking


def test_english_breaks_bottom_heavy_at_natural_points():
    rules = style.rules_for("en", {})
    assert style.break_lines("Where are you going?", rules, "en") == "Where are you going?"
    out = style.break_lines("I told you we should have left the party before the police arrived.", rules, "en")
    top, bottom = out.split("\n")
    assert _fits(out, rules)
    assert not top.endswith((" the", " a", " to")), out
    out = style.break_lines("We walked for hours, but nobody had seen him.", rules, "en")
    assert out.split("\n")[0].endswith("hours,"), out
    out = style.break_lines("Tell me everything you know about Harry Potter and his friends at school.", rules, "en")
    assert "Harry\nPotter" not in out, out
    assert _fits(out, rules)


def test_break_keeps_italics_balanced_and_words_intact():
    rules = style.rules_for("en", {})
    text = "<i>I never thought I would see this place again,</i> but here we are at last."
    out = style.break_lines(text, rules, "en")
    assert _fits(out, rules)
    assert strip_tags(out).split() == strip_tags(text).split()
    assert out.count("<i>") == 1 and out.count("</i>") == 1
    assert out.index("<i>") < out.index("</i>")
    # A tag spanning the break stays one pair and still wraps the same words.
    out = style.break_lines("<i>Somewhere over the rainbow, way up high, there's a land that I heard of.</i>", rules, "en")
    assert out.startswith("<i>") and out.endswith("</i>") and "\n" in out


def test_dual_speakers_one_per_line():
    en = style.rules_for("en", {})
    assert style.break_lines("-Are you coming? -In a minute.", en, "en") == "-Are you coming?\n-In a minute."
    fr = style.rules_for("fr", {})
    assert style.break_lines("- Tu viens ?\n- J'arrive.", fr, "fr") == "- Tu viens ?\n- J'arrive."
    assert style.break_lines("- Tu viens ? - J'arrive.", fr, "fr") == "- Tu viens ?\n- J'arrive."
    # A dash inside a sentence is not a speaker change.
    assert "\n" not in style.break_lines("It was well-known.", en, "en")


def test_japanese_breaks_after_punctuation_and_particles():
    rules = style.rules_for("ja", {})
    out = style.break_lines("今日はとても楽しかったね、また一緒に行こう", rules, "ja")
    assert out == "今日はとても楽しかったね、\nまた一緒に行こう", out
    out = style.break_lines("これはiPhoneのカメラで撮った写真です", rules, "ja")
    assert "iPh" in out.split("\n")[0] and "iPhone" in out.replace("\n", "")  # never inside a Latin word
    assert _fits(out, rules)
    for line in style.break_lines("ちょっと待ってください。すぐに戻りますから", rules, "ja").split("\n"):
        assert line[:1] not in "。、っゃゅょー", line


def _no_forbidden_ja_break(out: str) -> None:
    """The breaks Japanese never allows, checked on every line boundary."""
    lines = out.split("\n")
    for top, bottom in zip(lines, lines[1:]):
        pair = top[-1:] + bottom[:1]
        # Two particles in a row, or a particle cut from its word.
        assert bottom[:1] not in "はがをにでともへの" or bottom.startswith(("もう", "また")), out
        assert not top.endswith(("この", "その", "あの", "どの", "こんな", "そんな")), out
        assert pair not in ("ち上", "ちひ", "ち切"), out
        assert bottom[:1] not in "、。！？」』）ーぁぃぅぇぉっゃゅょァィゥェォッャュョ", out
        assert top[-1:] not in "「『（", out


# The breaks a real Demon Slayer run produced (segment_words then
# break_lines, 13 full-width characters per line) and what they should be.
JA_REAL_BREAKS = [
    ("誰にも断ち切ることができない", ["誰にも\n断ち切ることができない"]),
    ("人はまた立ち上がって戦うんだ", ["人は\nまた立ち上がって戦うんだ", "人はまた\n立ち上がって戦うんだ"]),
    ("死んでしまったこの子たちの無念は", ["死んでしまった\nこの子たちの無念は"]),
    ("いつまでここに足を運べるものだろうか", ["いつまでここに足を\n運べるものだろうか"]),
]


def test_japanese_never_splits_particles_prenominals_or_compound_verbs():
    rules = style.rules_for("ja", {})
    for text, good in JA_REAL_BREAKS:
        out = style.break_lines(text, rules, "ja")
        assert out in good, (text, out)
        _no_forbidden_ja_break(out)
    # Each rule on its own.
    for text in ("東京にはもう何年も行っていない", "学校ではみんな静かにしていた",
                 "あの店からの手紙がまだ届かないんだ", "明日までに全部終わらせないと困る"):
        out = style.break_lines(text, rules, "ja")
        _no_forbidden_ja_break(out)
        assert _fits(out, rules), out
        for pair in ("に\nは", "で\nは", "から\nの", "まで\nに"):
            assert pair not in out, out
    out = style.break_lines("それでも彼はまた立ち上がって歩き出した", rules, "ja")
    assert "立ち\n" not in out and "\n上が" not in out, out
    out = style.break_lines("どんなに打ちひしがれても前を向いて進むんだ", rules, "ja")
    assert "打ち\n" not in out and "\nひし" not in out, out
    # A number keeps its counter; a line never starts with a small kana or ー.
    out = style.break_lines("鬼による被害の報告を7件受けている", rules, "ja")
    assert "7\n件" not in out and _fits(out, rules), out
    # A quotative particle stays with the quote it closes.
    out = style.break_lines("「待って」と彼女は小さな声で言った", rules, "ja")
    assert "」\nと" not in out and _fits(out, rules), out
    out = style.break_lines("スーパーマーケットでアイスクリームを買った", rules, "ja")
    assert out == "スーパーマーケットで\nアイスクリームを買った", out
    # Balance is only a tie-breaker: an uneven break between words beats an
    # even one inside ございました.
    out = style.break_lines("ありがとうございました本当に", rules, "ja")
    assert out == "ありがとうございました\n本当に", out
    # Whatever fits on one line stays on one line.
    assert style.break_lines("どれほど打ちひしがれようと", rules, "ja") == "どれほど打ちひしがれようと"


def test_japanese_full_width_space_is_a_sentence_break():
    rules = style.rules_for("ja", {})
    out = style.break_lines("とりあえず刀は背中に隠そう　やばもう出発だ", rules, "ja")
    assert out == "とりあえず刀は背中に隠そう\nやばもう出発だ", out
    # Kept where the text fits on one line; never doubled or turned half-width.
    assert style.break_lines("そうか　行こう", rules, "ja") == "そうか　行こう"


def test_segment_japanese_real_run_breaks():
    rules = style.rules_for("ja", {})
    # Whisper returns Japanese as sub-word pieces: the boundaries it knows
    # are carried into line breaking, and the language rules still apply.
    samples = [
        (["誰", "に", "も", "断ち", "切る", "ことが", "できない"], JA_REAL_BREAKS[0][1]),
        (["人", "は", "また", "立ち", "上が", "って", "戦う", "んだ"], JA_REAL_BREAKS[1][1]),
        (["死んで", "しまった", "この", "子", "たち", "の", "無念", "は"], JA_REAL_BREAKS[2][1]),
        (["いつ", "まで", "ここに", "足", "を", "運べる", "もの", "だろう", "か"], JA_REAL_BREAKS[3][1]),
    ]
    for tokens, good in samples:
        cues = style.segment_words(_words(_timed(tokens, per_char=0.12)), rules, "ja")
        assert len(cues) == 1 and cues[0].text in good, (tokens, [c.text for c in cues])


def test_segment_japanese_pause_ends_unpunctuated_sentence():
    rules = style.rules_for("ja", {})
    tokens = ["とりあえず", "刀", "は", "背中", "に", "隠そう", "やば", "もう", "出発", "だ"]
    cues = style.segment_words(_words(_timed(tokens, per_char=0.12, pauses={5: 0.4})), rules, "ja")
    texts = [c.text for c in cues]
    assert not any("隠そうやば" in t.replace("\n", "") for t in texts), texts
    assert texts == ["とりあえず刀は背中に隠そう", "やばもう出発だ"], texts
    for cue in cues:
        _no_forbidden_ja_break(cue.text)
    _assert_timing(cues, rules)
    # A short first sentence stays in the subtitle, set off by a full-width space.
    cues = style.segment_words(_words(_timed(["刀", "は", "隠そう", "やば", "もう", "出発", "だ"],
                                             per_char=0.12, pauses={2: 0.4})), rules, "ja")
    assert [c.text for c in cues] == ["刀は隠そう　やばもう出発だ"], [c.text for c in cues]
    # A pause after a topic particle or a te-form is not a sentence end.
    cues = style.segment_words(_words(_timed(tokens, per_char=0.12, pauses={1: 0.4, 2: 0.4})), rules, "ja")
    assert "　" not in "".join(c.text for c in cues), [c.text for c in cues]
    cues = style.segment_words(_words(_timed(["立ち", "上がって", "戦う", "んだ"], per_char=0.12, pauses={1: 0.4})),
                               rules, "ja")
    assert [c.text for c in cues] == ["立ち上がって戦うんだ"], [c.text for c in cues]


def test_chinese_breaks_at_phrases_never_inside_numbers_or_latin():
    zh = style.rules_for("zh", {})
    out = style.break_lines("他花了3.5个小时才做完这些作业，真的太累了", zh, "zh")
    assert out == "他花了3.5个小时才做完这些作业，\n真的太累了", out
    for text, run in (("我昨天买了一部iPhone 15手机非常好用", "iPhone"),
                      ("会议在10:30开始请大家准时到达会议室", "10:30"),
                      ("他说这个问题的答案其实就是COVID-19疫苗", "COVID-19")):
        out = style.break_lines(text, zh, "zh")
        assert _fits(out, zh) and run in out, out
    # A number keeps its measure word; a particle never starts a line.
    out = style.break_lines("我们一共买了12个苹果和三斤香蕉还有牛奶", zh, "zh")
    assert "12\n个" not in out and out.split("\n")[1][:1] not in "的了吗呢吧", out
    # Unpunctuated speech breaks before a phrase-starting word, not inside 学校.
    out = style.break_lines("我们明天早上八点在学校门口见面吧别迟到了", zh, "zh")
    assert out == "我们明天早上八点\n在学校门口见面吧别迟到了", out


def test_korean_keeps_determiners_and_bound_nouns_with_their_words():
    ko = style.rules_for("ko", {})
    for text in ("그 사람은 내가 할 수 있는 일이 아니라고 했어요", "우리는 그 사람을 절대로 용서할 수 없을 거예요",
                 "어제 이런 이상한 일이 생겨서 정말 놀랐어요"):
        out = style.break_lines(text, ko, "ko")
        assert _fits(out, ko) and out.replace("\n", " ") == text, out
        top, bottom = out.split("\n")
        assert top.split()[-1] not in ("그", "이런", "수"), out
        assert bottom.split()[0] not in ("수", "것"), out


def test_chinese_and_korean_breaks():
    zh = style.rules_for("zh", {})
    out = style.break_lines("我昨天和朋友一起去看电影 电影非常好看", zh, "zh")
    assert out == "我昨天和朋友一起去看电影\n电影非常好看", out
    ko = style.rules_for("ko", {})
    out = style.break_lines("어제 친구들이랑 영화를 보러 갔는데 정말 재미있었어요", ko, "ko")
    assert _fits(out, ko), out
    assert out.replace("\n", " ") == "어제 친구들이랑 영화를 보러 갔는데 정말 재미있었어요"


# ------------------------------------------------------------ segmentation


def test_segment_english_sentences_and_pauses():
    rules = style.rules_for("en", {})
    tokens = ("So I went back to the house, and the door was wide open. "
              "Nobody was there. I waited for almost an hour before I called the police "
              "and told them everything that I had seen that night.").split()
    words = _words(_timed(tokens, start=1.0, pauses={12: 0.9, 15: 0.35}), leading_space=True)
    cues = style.segment_words(words, rules, "en")
    assert " ".join(strip_tags(c.text).replace("\n", " ") for c in cues).split() == tokens
    for cue in cues:
        assert _fits(cue.text, rules), cue.text
    texts = [strip_tags(c.text).replace("\n", " ") for c in cues]
    assert texts[0].endswith("wide open."), texts
    assert texts[1] == "Nobody was there.", texts
    assert cues[0].start == 1.0
    _assert_timing(cues, rules)
    assert not any(c.text.split("\n")[0].endswith((" the", " a")) for c in cues)


def test_segment_french_punctuation_spacing():
    rules = style.rules_for("fr", {})
    tokens = ["Tu", "viens", "?", "Je", "t'attends", "depuis", "une", "heure", "devant", "la", "gare",
              ",", "et", "il", "commence", "à", "pleuvoir", "!"]
    cues = style.segment_words(_words(_timed(tokens, pauses={2: 0.4})), rules, "fr")
    text = " / ".join(strip_tags(c.text).replace("\n", " ") for c in cues)
    assert text.startswith("Tu viens ? / Je t'attends"), text
    assert "gare, et il" in text and text.endswith("pleuvoir !"), text
    _assert_timing(cues, rules)


def test_segment_japanese():
    rules = style.rules_for("ja", {})
    tokens = ["今日", "は", "本当に", "ありがとう", "ございました", "。", "また", "明日", "会い", "ましょう", "。",
              "駅", "まで", "一緒に", "歩いて", "帰り", "ませんか"]
    cues = style.segment_words(_words(_timed(tokens, per_char=0.12, pauses={5: 0.3, 10: 0.7})), rules, "ja")
    joined = "".join(strip_tags(c.text).replace("\n", "") for c in cues)
    assert joined == "".join(tokens)  # no spaces invented between Japanese words
    texts = [strip_tags(c.text).replace("\n", "") for c in cues]
    assert texts[0] == "今日は本当にありがとうございました。", texts
    assert texts[-1].startswith("駅まで"), texts
    for cue in cues:
        assert _fits(cue.text, rules), cue.text
    _assert_timing(cues, rules)


def test_segment_chinese_without_punctuation():
    rules = style.rules_for("zh", {})
    # Whisper often returns Chinese with no punctuation at all: pauses are
    # the only sentence boundaries.
    tokens = ["我们", "明天", "早上", "八点", "在", "学校", "门口", "见面", "吧",
              "别", "迟到", "了", "老师", "说", "要", "点名"]
    cues = style.segment_words(_words(_timed(tokens, per_char=0.2, pauses={8: 0.8})), rules, "zh")
    texts = [strip_tags(c.text).replace("\n", "") for c in cues]
    assert texts[0] == "我们明天早上八点在学校门口见面吧", texts
    assert "".join(texts) == "".join(tokens)
    for cue in cues:
        assert _fits(cue.text, rules), cue.text
    _assert_timing(cues, rules)


def test_segment_korean_uses_spaces():
    rules = style.rules_for("ko", {})
    tokens = ["저는", "어제", "친구들이랑", "영화를", "보러", "갔는데", "정말", "재미있었어요.",
              "다음에", "같이", "갈래요?"]
    cues = style.segment_words(_words(_timed(tokens, per_char=0.15, pauses={7: 0.5}), leading_space=True), rules, "ko")
    texts = [strip_tags(c.text).replace("\n", " ") for c in cues]
    assert " ".join(texts).split() == tokens
    assert texts[-1] == "다음에 같이 갈래요?", texts
    for cue in cues:
        assert _fits(cue.text, rules), cue.text
    _assert_timing(cues, rules)


def test_segment_speaker_change_and_long_monologue():
    rules = style.rules_for("en", {})
    tokens = ("I have been thinking about what you said yesterday and I believe you were right about "
              "everything including the money and the car and the house by the lake").split()
    words = _words(_timed(tokens, per_char=0.05, gap=0.05))
    words.append(Word(start=words[-1].end + 0.2, end=words[-1].end + 0.6, text="Really?", speaker="B"))
    for w in words[:-1]:
        w.speaker = "A"
    cues = style.segment_words(words, rules, "en")
    assert strip_tags(cues[-1].text) == "Really?" and cues[-1].speaker == "B"
    for cue in cues:
        assert _fits(cue.text, rules) and cue.duration <= rules.max_duration_s + 1e-6
    _assert_timing(cues, rules)


def test_segmented_output_passes_check():
    # Reading speed is the one rule a transcript cannot always meet: the
    # Japanese guide's 4 cps assumes condensed translation, not verbatim speech.
    samples = {
        "en": ("So I went back to the house, and the door was wide open. Nobody was there. I waited for "
               "almost an hour before I called the police and told them everything that I had seen.").split(),
        "fr": ["Tu", "viens", "?", "Je", "t'attends", "depuis", "une", "heure", "devant", "la", "gare", ",",
               "et", "il", "commence", "à", "pleuvoir", "!"],
        "ja": ["実は", "昨日", "の", "夜", "遅く", "まで", "仕事", "を", "して", "いた", "ので", "今朝", "は", "とても",
               "眠くて", "何", "も", "できなかった", "ん", "だ", "けど", "午後", "から", "少し", "元気", "に", "なった"],
        "zh": ["我们", "明天", "早上", "八点", "在", "学校", "门口", "见面", "吧", "别", "迟到", "了", "老师", "说", "要", "点名"],
        "ko": ["저는", "어제", "친구들이랑", "영화를", "보러", "갔는데", "정말", "재미있었어요.", "다음에", "같이", "갈래요?"],
    }
    for lang, tokens in samples.items():
        rules = style.rules_for(lang, {})
        spaced = lang in ("en", "ko")
        cues = style.segment_words(_words(_timed(tokens, per_char=0.1, pauses={2: 0.4, 11: 0.7}), spaced), rules, lang)
        issues = style.check(SubtitleDoc(cues=cues, language=lang), rules)
        assert _types(issues) <= {"cps_exceeded"}, (lang, [i.to_dict() for i in issues])
        # No word ever split across lines: every line boundary is a token boundary.
        if lang in ("ja", "zh"):
            for cue in cues:
                for line in strip_tags(cue.text).split("\n")[:-1]:
                    joined = ""
                    for tok in tokens:
                        joined += tok
                        if joined.endswith(line):
                            break
                    else:
                        raise AssertionError((lang, cue.text))


# ------------------------------------------------------------ check / fix


def _doc(*cues: Tuple[float, float, str], language: str = "en") -> SubtitleDoc:
    return SubtitleDoc(cues=[Cue(s, e, t) for s, e, t in cues], language=language, source_format="srt")


def test_check_reports_every_rule():
    rules = style.rules_for("en", {})
    doc = _doc(
        (1.0, 1.4, "Hi."),                                                     # too short
        (1.45, 3.0, "Hello there."),                                           # 1 frame gap before
        (3.2, 4.0, "This line has far too many characters to be read in time here."),  # cpl, cps
        (3.9, 5.0, "Overlap."),                                                # overlaps previous
        (6.0, 14.0, "Line one\nline two\nline three"),                         # too long, too many lines
        (15.0, 17.0, ""),                                                      # empty
        (18.0, 20.0, "I went to the\nshop yesterday with my brother."),       # bad break
        (21.0, 23.0, "Short\nlines"),                                          # fits on one line
    )
    types = _types(style.check(doc, rules))
    for expected in ("too_short", "gap_too_small", "cpl_exceeded", "cps_exceeded", "overlap", "too_long",
                     "too_many_lines", "empty", "bad_line_break"):
        assert expected in types, (expected, types)
    issue = style.check(doc, rules)[0].to_dict()
    assert set(issue) >= {"index", "type", "message", "severity"}


def test_check_gap_between_three_and_eleven_frames():
    rules = style.rules_for("en", {})
    doc = _doc((1.0, 2.0, "One."), (2.0 + 6 * FRAME, 3.0, "Two."), (3.0 + 2 * FRAME, 4.0, "Three."),
               (4.0 + 12 * FRAME, 5.0, "Four."))
    issues = [i for i in style.check(doc, rules) if i.type == "gap_too_small"]
    assert [i.index for i in issues] == [0], issues


def test_fix_timing_never_changes_words():
    rules = style.rules_for("en", {})
    doc = _doc(
        (1.0, 1.3, "Hi."),
        (2.0 + 5 * FRAME, 3.5, "<i>Where have you been all this time? We looked everywhere for you.</i>"),
        (3.4, 5.0, "Nowhere."),
        (5.2, 6.0, "I was at the station waiting for the last train home all night."),
    )
    fixed, repairs = style.fix(doc, rules)
    assert [strip_tags(c.text).split() for c in fixed.cues] == [strip_tags(c.text).split() for c in doc.cues]
    assert repairs and {"too_short", "overlap"} <= _types(repairs)
    _assert_timing(fixed.cues, rules)
    after = _types(style.check(fixed, rules))
    assert not after & {"too_short", "overlap", "gap_too_small", "cpl_exceeded", "bad_line_break"}, after
    assert fixed.cues[1].text.count("<i>") == 1 and fixed.cues[1].text.count("</i>") == 1
    assert doc.cues[0].end == 1.3  # the input is not modified


def test_fix_splits_long_cue_at_sentence():
    rules = style.rules_for("en", {})
    text = ("I don't want to talk about it anymore. You never listen to anything I say, "
            "and I'm tired of repeating myself over and over.")
    doc = _doc((10.0, 20.0, text))
    fixed, repairs = style.fix(doc, rules)
    assert len(fixed.cues) >= 2
    assert strip_tags(fixed.cues[0].text).replace("\n", " ") == "I don't want to talk about it anymore."
    assert " ".join(strip_tags(c.text).replace("\n", " ") for c in fixed.cues).split() == text.split()
    assert fixed.cues[0].start == 10.0 and abs(fixed.cues[-1].end - 20.0) < 1e-6
    _assert_timing(fixed.cues, rules)
    for cue in fixed.cues:
        assert _fits(cue.text, rules)
    assert "too_long" in _types(repairs)


def test_split_keeps_italics_on_both_halves():
    rules = style.rules_for("en", {})
    doc = _doc((0.0, 9.0, "<i>We had nowhere else to go that winter. So we stayed in the old house by the sea.</i>"))
    fixed, _ = style.fix(doc, rules)
    assert len(fixed.cues) == 2
    for cue in fixed.cues:
        assert cue.text.startswith("<i>") and cue.text.endswith("</i>"), cue.text


def test_shot_change_rules():
    rules = style.rules_for("en", {})
    cut = 10.0
    doc = _doc(
        (cut + 5 * FRAME, cut + 2.0, "Starts just after the cut."),
        (20.0, 22.0 - 6 * FRAME, "Ends just before the cut."),
        (30.0, 31.5 + 3 * FRAME, "Ends just after the cut."),
    )
    cuts = [cut, 22.0, 31.5]
    assert _types(style.check(doc, rules, cuts)) == {"shot_change_proximity"}
    fixed, repairs = style.fix(doc, rules, cuts)
    assert abs(fixed.cues[0].start - cut) < 1e-9
    assert round((22.0 - fixed.cues[1].end) * FPS) == 2
    assert round((31.5 - fixed.cues[2].end) * FPS) == 2
    assert "shot_change_proximity" not in _types(style.check(fixed, rules, cuts))
    # A cue that crosses a cut but starts 10 frames before it starts half a
    # second before the cut instead.
    doc = _doc((40.0 - 10 * FRAME, 42.0, "Across the cut."))
    fixed, _ = style.fix(doc, rules, [40.0])
    assert round((40.0 - fixed.cues[0].start) * FPS) == 12


# ------------------------------------------------------------ shot changes


def _ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None or bool(os.environ.get("AUDIOSYNC_FFMPEG"))


def test_shot_changes_found_within_one_frame():
    if not _ffmpeg_ok():
        print("        skipped: no ffmpeg")
        return
    from audiosync.media import ffmpeg_path

    root = tempfile.mkdtemp(prefix="subsync-style-")
    try:
        path = os.path.join(root, "cuts.mp4")
        rate = "24000/1001"
        # Solid colours and a moving test pattern with hard cuts at 2.0 s and 3.5 s.
        subprocess.run([
            ffmpeg_path(), "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=red:s=320x240:r={rate}:d=2",
            "-f", "lavfi", "-i", f"testsrc=s=320x240:r={rate}:d=1.5",
            "-f", "lavfi", "-i", f"color=c=blue:s=320x240:r={rate}:d=2",
            "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", path,
        ], check=True)
        seen = []
        cuts = style.shot_changes(path, progress=seen.append)
        assert len(cuts) == 2, cuts
        for found, expected in zip(cuts, (2.0, 3.5)):
            assert abs(found - expected) <= FRAME + 1e-6, (found, expected)
        assert seen and seen[-1] == 1.0
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------- task


def test_run_task_writes_netflix_file():
    try:
        from audiosync.subs import formats  # noqa: F401
        from audiosync.subs.tasks import TaskContext
        from audiosync.media import CancellationToken
    except ImportError as exc:
        print(f"        skipped: {exc}")
        return
    stubbed = False
    try:
        from audiosync.subs import tracks  # noqa: F401
    except ImportError:
        # tracks.py is only consulted for tracks inside a video; a file path
        # never reaches it, so a stand-in keeps this test independent of it.
        import types

        sys.modules["audiosync.subs.tracks"] = types.ModuleType("audiosync.subs.tracks")
        stubbed = True
    root = tempfile.mkdtemp(prefix="subsync-style-")
    try:
        src = os.path.join(root, "episode.srt")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write("1\n00:00:01,000 --> 00:00:01,300\nHi.\n\n"
                     "2\n00:00:01,500 --> 00:00:04,000\nWhere have you been? We looked\neverywhere.\n\n")
        logs = []
        ctx = TaskContext(token=CancellationToken(), progress=lambda p, s: None, log=logs.append, workdir=root)
        job = {"id": "1", "task": "style", "input": {"subtitle": {"path": src}},
               "options": {"preset": "netflix", "language": "en", "fix": True, "snapToShotChanges": True},
               "output": {}}
        result = style.run_task(job, ctx)
        report = result.report
        assert report["rules"]["cpl"] == 42 and report["language"] == "en"
        assert report["issuesBefore"] > report["issuesAfter"]
        assert "too_short" in report["byTypeBefore"]
        assert result.outputs and result.outputs[0].path.endswith("episode.netflix.srt"), result.outputs[0].path
        assert os.path.isfile(result.outputs[0].path)
        assert result.summary
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if stubbed:
            sys.modules.pop("audiosync.subs.tracks", None)


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
