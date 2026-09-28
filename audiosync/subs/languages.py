"""Language codes shared by every subtitle engine.

Speech recognition speaks Whisper's ISO 639-1 codes, containers tag tracks
with ISO 639-2 (Matroska uses the bibliographic "B" forms: ger, fre, chi),
and translation services want BCP-47. One table here keeps a track tagged
``jpn`` in an MKV, Whisper's ``ja`` and a translator's ``ja`` the same
language.
"""

from __future__ import annotations

from typing import Dict, Optional

#: ISO 639-1 (Whisper's set, large-v3) -> English name.
LANGUAGES: Dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian", "mk": "Macedonian",
    "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", "ne": "Nepali",
    "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer",
    "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan",
    "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati",
    "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Norwegian Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Burmese", "bo": "Tibetan",
    "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian",
    "ln": "Lingala", "ha": "Hausa", "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese",
    "yue": "Cantonese",
}

#: ISO 639-1 -> ISO 639-2/B, the form Matroska and most muxers write.
TO_639_2: Dict[str, str] = {
    "en": "eng", "zh": "chi", "de": "ger", "es": "spa", "ru": "rus", "ko": "kor", "fr": "fre",
    "ja": "jpn", "pt": "por", "tr": "tur", "pl": "pol", "ca": "cat", "nl": "dut", "ar": "ara",
    "sv": "swe", "it": "ita", "id": "ind", "hi": "hin", "fi": "fin", "vi": "vie", "he": "heb",
    "uk": "ukr", "el": "gre", "ms": "may", "cs": "cze", "ro": "rum", "da": "dan", "hu": "hun",
    "ta": "tam", "no": "nor", "th": "tha", "ur": "urd", "hr": "hrv", "bg": "bul", "lt": "lit",
    "la": "lat", "mi": "mao", "ml": "mal", "cy": "wel", "sk": "slo", "te": "tel", "fa": "per",
    "lv": "lav", "bn": "ben", "sr": "srp", "az": "aze", "sl": "slv", "kn": "kan", "et": "est",
    "mk": "mac", "br": "bre", "eu": "baq", "is": "ice", "hy": "arm", "ne": "nep", "mn": "mon",
    "bs": "bos", "kk": "kaz", "sq": "alb", "sw": "swa", "gl": "glg", "mr": "mar", "pa": "pan",
    "si": "sin", "km": "khm", "sn": "sna", "yo": "yor", "so": "som", "af": "afr", "oc": "oci",
    "ka": "geo", "be": "bel", "tg": "tgk", "sd": "snd", "gu": "guj", "am": "amh", "yi": "yid",
    "lo": "lao", "uz": "uzb", "fo": "fao", "ht": "hat", "ps": "pus", "tk": "tuk", "nn": "nno",
    "mt": "mlt", "sa": "san", "lb": "ltz", "my": "bur", "bo": "tib", "tl": "tgl", "mg": "mlg",
    "as": "asm", "tt": "tat", "haw": "haw", "ln": "lin", "ha": "hau", "ba": "bak", "jw": "jav",
    "su": "sun", "yue": "chi",
}

#: ISO 639-2/T forms that differ from /B, so either spelling resolves.
_TERMINOLOGIC = {
    "zho": "zh", "deu": "de", "fra": "fr", "nld": "nl", "ell": "el", "msa": "ms", "ces": "cs",
    "ron": "ro", "cym": "cy", "slk": "sk", "fas": "fa", "mkd": "mk", "eus": "eu", "isl": "is",
    "hye": "hy", "sqi": "sq", "kat": "ka", "mri": "mi", "mya": "my", "bod": "bo",
}

_FROM_639_2: Dict[str, str] = {three: two for two, three in TO_639_2.items() if two != "yue"}
_FROM_639_2.update(_TERMINOLOGIC)

#: Scripts that are measured in full-width characters and have no spaces
#: between words; line-length rules and word joining differ for these.
CJK = frozenset({"zh", "ja", "ko", "yue"})


def normalize(code: Optional[str]) -> Optional[str]:
    """Any of ``ja``, ``jpn``, ``ja-JP``, ``Japanese`` -> ``ja``; unknown -> None.

    ``und``/``mul``/empty mean "not known" and return None.
    """
    if not code:
        return None
    raw = code.strip()
    lowered = raw.lower().replace("_", "-")
    if lowered in ("und", "mul", "zxx", "mis", ""):
        return None
    base = lowered.split("-")[0]
    if base in LANGUAGES:
        return base
    if base in _FROM_639_2:
        return _FROM_639_2[base]
    if lowered in ("zh-hant", "zh-hans", "zh-tw", "zh-cn", "zh-hk"):
        return "zh"
    for two, name in LANGUAGES.items():
        if name.lower() == lowered:
            return two
    return None


def to_639_2(code: Optional[str]) -> Optional[str]:
    two = normalize(code)
    return TO_639_2.get(two) if two else None


def name_of(code: Optional[str]) -> str:
    two = normalize(code)
    return LANGUAGES.get(two, code or "Unknown") if two else (code or "Unknown")


def is_cjk(code: Optional[str]) -> bool:
    return normalize(code) in CJK
