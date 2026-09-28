"""Per-language numbers from the Netflix Timed Text Style Guides.

Data only; ``style.py`` turns it into ``StyleRules``. Every figure below was
read from the Netflix Partner Help Center page cited next to its language
(fetched 2026-09-26), so when Netflix revises a guide the one line to change
is next to its source.

What the General Requirements and Subtitle Timing Guidelines say for every
language (``GENERAL``):

* at most 2 lines, 7 s maximum and 5/6 s minimum duration ("20 frames for
  24fps"; the timing guideline rounds this to "20 frames (or 4/5 sec)");
* at least 2 frames between subtitles, at any frame rate;
* "half a second" is 12 frames at 24 fps, 15 at 30, 30 at 60: gaps of 3-11
  frames are closed to 2 frames ("2 frames or half a second or more");
* in-times within half a second after a shot change move onto it; an in-time
  that crosses a shot change must be at least half a second before it;
* out-times within half a second before a shot change extend to 2 frames
  before it; an out-time after a shot change sits 2 frames before it or at
  least half a second (12 frames) after it;
* the out-time ideally lingers half a second after the audio ends when no
  subtitle follows.

Languages with no guide listed here fall back to ``GENERAL`` (42 CPL,
17/13 CPS, 20/17 SDH), which is what the large majority of Latin, Cyrillic
and Greek guides specify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

HELP = "https://partnerhelp.netflixstudios.com/hc/en-us/articles/"

GENERAL_SOURCE = HELP + "215758617-Timed-Text-Style-Guide-General-Requirements"
TIMING_SOURCE = HELP + "360051554394-Timed-Text-Style-Guide-Subtitle-Timing-Guidelines"

#: Timing figures shared by every language (General Requirements + Timing
#: Guidelines). Frame values are at 24 fps and are rescaled by style.py.
GENERAL_TIMING = {
    "min_duration_s": 5 / 6,
    "max_duration_s": 7.0,
    "min_gap_frames": 2,
    #: "half a second" -- the chaining threshold and every shot-change window.
    "half_second_s": 0.5,
    #: Linger after the audio when nothing follows (timing guideline 1).
    "tail_s": 0.5,
    "max_lines": 2,
}


@dataclass(frozen=True)
class LanguagePreset:
    name: str
    source: str
    cpl: float = 42
    cps: float = 17
    cps_children: float = 13
    cps_sdh: float = 20
    cps_sdh_children: float = 17
    #: SDH line length when the guide allows a longer one.
    cpl_sdh: Optional[float] = None
    max_lines: int = 2
    max_lines_sdh: Optional[int] = None
    min_duration_s: Optional[float] = None
    #: "-" (hyphen without a space) or "- " (hyphen followed by a space).
    dash: str = "- "
    #: Dutch/Finnish: only the second speaker gets a dash.
    dash_second_only: bool = False
    italics: bool = True
    #: How characters are counted: "latin" (every code point), "cjk"
    #: (full-width 1, half-width 0.5), "korean" (Hangul 1, Latin/space/
    #: punctuation 0.5), "thai" (tone marks and upper/lower vowels are free).
    width: str = "latin"
    note: str = ""


PRESETS: Dict[str, LanguagePreset] = {
    "en": LanguagePreset(
        "English (USA)", HELP + "217350977-English-USA-Timed-Text-Style-Guide",
        cpl=42, cps=20, cps_children=17, cps_sdh=20, cps_sdh_children=17, dash="-",
        note="English (UK), " + HELP + "30806198616339-English-UK-Timed-Text-Style-Guide, has the same numbers.",
    ),
    "ja": LanguagePreset(
        "Japanese", HELP + "215767517-Japanese-Timed-Text-Style-Guide",
        cpl=13, cpl_sdh=16, cps=4, cps_children=4, cps_sdh=7, cps_sdh_children=7,
        min_duration_s=0.5, dash="-", width="cjk",
        note="13 full-width chars horizontal (11 vertical), half-width counts 0.5; the guide gives no separate children's speed.",
    ),
    "ko": LanguagePreset(
        "Korean", HELP + "216001127-Korean-Timed-Text-Style-Guide",
        cpl=16, cps=12, cps_children=9, cps_sdh=14, cps_sdh_children=11,
        dash="- ", italics=False, width="korean",
    ),
    "zh": LanguagePreset(
        "Chinese (Simplified)", HELP + "215986007-Chinese-Simplified-Timed-Text-Style-Guide",
        cpl=16, cpl_sdh=18, cps=9, cps_children=7, cps_sdh=11, cps_sdh_children=9,
        max_lines_sdh=3, dash="-", italics=False, width="cjk",
        note="Traditional, " + HELP + "215994807-Chinese-Traditional-Timed-Text-Style-Guide, has the same numbers. "
        "No commas or periods: a space separates phrases.",
    ),
    "yue": LanguagePreset(
        "Chinese (Traditional)", HELP + "215994807-Chinese-Traditional-Timed-Text-Style-Guide",
        cpl=16, cpl_sdh=18, cps=9, cps_children=7, cps_sdh=11, cps_sdh_children=9,
        max_lines_sdh=3, dash="-", italics=False, width="cjk",
    ),
    "fr": LanguagePreset("French (France)", HELP + "217351577-French-France-Timed-Text-Style-Guide", dash="- "),
    "es": LanguagePreset(
        "Spanish (Latin America & Spain)", HELP + "217349997-Spanish-Latin-America-Spain-Timed-Text-Style-Guide",
        dash="- ", note="One guide covers Castilian and Latin American Spanish.",
    ),
    "it": LanguagePreset("Italian", HELP + "215349898-Italian-Timed-Text-Style-Guide", dash="- "),
    "de": LanguagePreset("German", HELP + "217351587-German-Timed-Text-Style-Guide", dash="-"),
    "pt": LanguagePreset(
        "Portuguese (Brazil)", HELP + "215600497-Portuguese-Brazil-Timed-Text-Style-Guide", dash="- ",
        note="Portuguese (EMEA), " + HELP + "216787938-Portuguese-EMEA-Timed-Text-Style-Guide, has the same numbers.",
    ),
    "nl": LanguagePreset(
        "Dutch", HELP + "215350158-Dutch-Timed-Text-Style-Guide", dash="-", dash_second_only=True,
    ),
    "hi": LanguagePreset(
        "Hindi", HELP + "115003196707-Hindi-Timed-Text-Style-Guide",
        cps=22, cps_children=18, cps_sdh=25, cps_sdh_children=20, dash="-",
    ),
    "ta": LanguagePreset(
        "Tamil", HELP + "4481912003987-Tamil-Timed-Text-Style-Guide",
        cps=22, cps_children=18, cps_sdh=22, cps_sdh_children=18, dash="-",
    ),
    "te": LanguagePreset(
        "Telugu", HELP + "4482320288787-Telugu-Timed-Text-Style-Guide",
        cps=22, cps_children=18, cps_sdh=22, cps_sdh_children=18, dash="-",
    ),
    "bn": LanguagePreset(
        "Bangla", HELP + "4483351778963-Bangla-Timed-Text-Style-Guide",
        cps=22, cps_children=18, cps_sdh=22, cps_sdh_children=18, dash="-",
    ),
    "ar": LanguagePreset(
        "Arabic", HELP + "215517947-Arabic-Timed-Text-Style-Guide",
        cps=20, cps_children=17, cps_sdh=23, cps_sdh_children=20, dash="- ",
    ),
    "ru": LanguagePreset("Russian", HELP + "215346638-Russian-Timed-Text-Style-Guide", dash="- "),
    "tr": LanguagePreset(
        "Turkish", HELP + "215342858-Turkish-Timed-Text-Style-Guide", dash="-",
        note="Encourages extending the minimum duration to one second where possible.",
    ),
    "th": LanguagePreset(
        "Thai", HELP + "220448308-Thai-Timed-Text-Style-Guide", cpl=35, dash="- ", width="thai",
        note="35 characters excluding tone marks and upper/lower vowels.",
    ),
    "pl": LanguagePreset("Polish", HELP + "216787928-Polish-Timed-Text-Style-Guide", dash="- "),
    "id": LanguagePreset("Indonesian", HELP + "216009727-Indonesian-Timed-Text-Style-Guide", dash="- "),
    "ms": LanguagePreset("Malay", HELP + "115002675707-Malay-Timed-Text-Style-Guide", dash="- "),
    "vi": LanguagePreset("Vietnamese", HELP + "220447048-Vietnamese-Timed-Text-Style-Guide", dash="- "),
    "tl": LanguagePreset("Filipino", HELP + "4480978897939-Filipino-Timed-Text-Style-Guide", dash="-"),
    "sv": LanguagePreset("Swedish", HELP + "216014517-Swedish-Timed-Text-Style-Guide", dash="-"),
    "da": LanguagePreset("Danish", HELP + "216014347-Danish-Timed-Text-Style-Guide", dash="-"),
    "no": LanguagePreset("Norwegian", HELP + "216015647-Norwegian-Timed-Text-Style-Guide", dash="-"),
    "nn": LanguagePreset("Norwegian", HELP + "216015647-Norwegian-Timed-Text-Style-Guide", dash="-"),
    "fi": LanguagePreset(
        "Finnish", HELP + "215087558-Finnish-Timed-Text-Style-Guide", dash="-", dash_second_only=True,
    ),
    "el": LanguagePreset("Greek", HELP + "235511047-Greek-Timed-Text-Style-Guide", dash="-"),
    "he": LanguagePreset("Hebrew", HELP + "220636427-Hebrew-Timed-Text-Style-Guide", dash="-"),
    "cs": LanguagePreset("Czech", HELP + "115002884887-Czech-Timed-Text-Style-Guide", dash="- "),
    "hu": LanguagePreset("Hungarian", HELP + "115003064248-Hungarian-Timed-Text-Style-Guide", dash="- "),
    "ro": LanguagePreset("Romanian", HELP + "220294068-Romanian-Timed-Text-Style-Guide", dash="- "),
    "uk": LanguagePreset("Ukrainian", HELP + "115002229068-Ukrainian-Timed-Text-Style-Guide", dash="-"),
    "hr": LanguagePreset("Croatian", HELP + "115002790368-Croatian-Timed-Text-Style-Guide", dash="-"),
}

#: Languages without a guide above: the common Netflix figures.
DEFAULT = LanguagePreset(
    "General (no language guide)", GENERAL_SOURCE,
    note="42 CPL and 17/13 CPS (20/17 SDH): the figures most Netflix language guides share.",
)


# ---------------------------------------------------------------- line breaks
#
# Netflix (English USA, and the same wording in the French, Spanish, Italian,
# German, Portuguese and Dutch guides): break after punctuation, before
# conjunctions and before prepositions; do not separate an article or
# adjective from its noun, a first name from a last name, or a subject pronoun
# from its verb. BREAK_BEFORE are words a new line may start with; KEEP_WITH_NEXT
# are words that must not end a line (articles, possessives, subject pronouns,
# prepositions -- the latter both begin a phrase and bind to its object).

BREAK_BEFORE: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        "and but or nor so yet because although though while when whenever where whereas if unless "
        "until since after before as than that which who whom whose what how why "
        "to of in on at by for with from about into onto over under between through without "
        "during against among toward towards like".split()
    ),
    "fr": frozenset(
        "et mais ou donc or ni car que qui quand lorsque puisque comme si parce pour sans avec "
        "dans sur sous chez vers entre depuis pendant avant après contre par de du des à au aux "
        "dont où quoique bien alors".split()
    ),
    "es": frozenset(
        "y e o u pero sino ni que porque aunque cuando mientras donde como si pues "
        "a ante bajo con contra de desde durante en entre hacia hasta para por según sin sobre tras "
        "del al cual quien".split()
    ),
    "it": frozenset(
        "e ed o oppure ma però perché che quando mentre dove come se anche né "
        "di a da in con su per tra fra del dello della dei degli delle al allo alla ai agli alle "
        "dal dalla dai nel nella nei sul sulla sui senza verso dopo prima".split()
    ),
    "de": frozenset(
        "und aber oder denn sondern doch dass weil wenn als ob obwohl während bevor nachdem damit "
        "bis seit wie wo was wer mit von zu bei nach aus für über unter vor hinter neben zwischen "
        "durch gegen ohne um an auf in".split()
    ),
    "pt": frozenset(
        "e ou mas porém que porque quando enquanto onde como se nem pois embora "
        "a de em para por com sem sobre entre até desde contra do da dos das no na nos nas ao aos "
        "pelo pela pelos pelas num numa".split()
    ),
    "nl": frozenset(
        "en maar of want dus omdat als toen terwijl dat die wat wie waar hoe voordat nadat zodat "
        "tenzij hoewel met van voor naar in op bij uit over onder door tegen zonder tussen tot "
        "sinds om aan achter".split()
    ),
}

KEEP_WITH_NEXT: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        "a an the this that these those my your his her its our their some any no every each "
        "i you he she we they mr. mrs. ms. dr. to of in on at by for with from very".split()
    ),
    "fr": frozenset(
        "le la les l' un une des du de d' au aux ce cet cette ces mon ma mes ton ta tes son sa ses "
        "notre nos votre vos leur leurs je tu il elle on nous vous ils elles ne n' à en par pour "
        "m. mme très".split()
    ),
    "es": frozenset(
        "el la los las un una unos unas lo al del este esta estos estas ese esa esos esas mi mis "
        "tu tus su sus nuestro nuestra yo tú él ella nosotros vosotros ellos ellas no a de en con "
        "por para sr. sra. muy".split()
    ),
    "it": frozenset(
        "il lo la i gli le l' un uno una un' del della dei al alla questo questa quel quella mio "
        "mia tuo tua suo sua nostro vostro io tu lui lei noi voi loro non di a da in con su per "
        "molto".split()
    ),
    "de": frozenset(
        "der die das den dem des ein eine einen einem einer eines kein keine mein meine dein deine "
        "sein seine ihr ihre unser euer ich du er sie es wir zu von mit bei nach aus für in an auf "
        "sehr herr frau".split()
    ),
    "pt": frozenset(
        "o a os as um uma uns umas do da dos das no na ao este esta esse essa aquele aquela meu "
        "minha teu tua seu sua nosso nossa eu tu ele ela nós vós eles elas não de em com por para "
        "sr. sra. muito".split()
    ),
    "nl": frozenset(
        "de het een deze dit die dat mijn jouw zijn haar ons onze hun ik jij je hij zij wij we "
        "jullie niet van in op met voor naar bij te heel meneer mevrouw".split()
    ),
    "ko": frozenset(
        # Determiners and numerals that modify the next word: 그 | 사람 splits
        # a noun phrase exactly like "the | man" does.
        "이 그 저 이런 그런 저런 어떤 무슨 모든 몇 한 두 세 네 새 각 온 아무".split()
        # A bare bound noun ends a phrase that continues: 할 수 | 있어.
        + "수 줄 적 듯".split()
    ),
}

#: Words a line must not start with: they attach to the word before them.
KEEP_WITH_PREVIOUS: Dict[str, FrozenSet[str]] = {
    # Korean bound nouns are written as separate words but cannot stand
    # alone: 할 | 수 있어 and 먹을 | 것 break a phrase in the middle.
    "ko": frozenset("수 것 거 게 걸 건 줄 적 데 뿐 듯 만큼 대로 때문에 때문이야 따름".split()),
}

# Japanese: the guide's own examples break after 、 and 。 or after a
# particle ending a phrase; a line never starts with closing punctuation or a
# small kana (kinsoku shori), never ends with an opening bracket. There is no
# dictionary here, so the word lists below are the few closed classes that
# decide most breaks: particles, prenominals, and the handful of hiragana
# words that commonly begin a phrase.

#: Particles that end a phrase, longest first so にも wins over も. Compound
#: particles are listed whole because the single-kana rule in style.py only
#: trusts a kana after kanji or katakana, and に|も must read as one unit.
JA_PARTICLES = (
    "からの", "からは", "までに", "までは", "までの", "よりも",
    "には", "にも", "では", "でも", "とは", "とも", "へは", "へも", "への", "との", "での",
    "から", "まで", "より", "けど", "ので", "のに", "だけ", "ほど", "など",
    "は", "が", "を", "に", "で", "と", "も", "へ", "の", "ね", "よ", "か",
)
#: Kana that, at the start of a line, would be a particle or ending cut off
#: from the word before it (誰に | も, 足 | を, 代 | で).
JA_ATTACHING = "はがをにでともへのやかねよ"
#: Prenominals: they modify the noun after them and never end a line.
JA_PRENOMINALS = (
    "この", "その", "あの", "どの", "こんな", "そんな", "あんな", "どんな",
    "こういう", "そういう", "ああいう", "どういう", "こうした", "そうした", "我が",
)
#: Hiragana words that usually begin a phrase: a break before one is a
#: break between words, not inside one (死んでしまった | この子たち).
JA_WORD_STARTS = JA_PRENOMINALS + (
    "また", "もう", "まだ", "もっと", "もし", "もちろん", "だから", "だけど", "しかし", "そして",
    "それでも", "それに", "とりあえず", "やっぱり", "やはり", "ちょっと", "きっと", "ずっと",
    "みんな", "どうして", "なぜ", "いつも", "すぐ", "ほら", "さあ",
)
#: Endings after which a pause is still mid-sentence (a particle, a te-form
#: or a conjunctive ending), so ``segment_words`` must not treat it as the
#: end of one.
JA_CONTINUES = JA_PARTICLES[:-3] + ("って", "て", "や", "し", "ば", "たら", "ながら", "けれど")

# Chinese: after ，。！？、 is the break; the particles below close the word
# or clause before them and never start a line.
ZH_ATTACHING = "的了吗呢吧啊呀嘛们着过得地么啦"
#: Sentence-final particles: in unpunctuated speech they are the best clue
#: that a clause has ended (见面吧 | 别迟到了).
ZH_FINAL = "吧呢吗啊呀嘛啦了"
#: Words that begin a phrase -- prepositions, conjunctions, common adverbs --
#: so a break before one falls between words, not inside one
#: (早上八点 | 在学校). Without a dictionary every other pair of hanzi could
#: be one word.
ZH_BREAK_BEFORE = (
    "但是", "可是", "因为", "所以", "如果", "虽然", "然后", "而且", "或者", "还是", "不过", "就是",
    "请", "在", "和", "跟", "把", "被", "给", "对", "从", "到", "就", "都", "也", "还", "又", "才", "却", "而",
)

CJK_BREAK_AFTER = "、。，．！？!?…‥・：；:;」』）)】》〉］｝”’"
NO_LINE_START = (
    "、。，．！？!?…‥・：；:;」』）)】》〉］｝”’〜～ーゝゞヽヾ"
    "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ々"
)
NO_LINE_END = "「『（(【《〈［｛“‘"
