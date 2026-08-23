"""Pair up video and audio files across two folders.

Series matching originally tried a fixed chain of regexes and, on failure, fell
back to "every number in the filename" -- which matches resolution, year and
release-group digits as readily as episode numbers, producing confident but
wrong pairings. Here matching is explicit about *how* a pair was found and how
much to trust it, so the UI can show its reasoning and let the user intervene.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from .media import is_audio_file, is_media_file, is_video_file

# Ordered by specificity: the first pattern that pairs anything wins.
EPISODE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"[Ss](\d{1,2})[\s._-]*[Ee](\d{1,3})", "S01E01"),
    (r"(\d{1,2})x(\d{1,3})", "1x01"),
    (r"[Ss]eason[\s._-]*(\d{1,2}).*?[Ee]pisode[\s._-]*(\d{1,3})", "Season 1 Episode 1"),
    (r"[\s._-](\d{1,2})(\d{2})[\s._-]", "101 (season+episode)"),
)

# Tokens that look like episode numbers but never are.
NOISE_TOKENS = re.compile(
    r"\b(?:\d{3,4}p|x26[45]|h\.?26[45]|hevc|avc|aac|ac3|eac3|dts|ddp?5\.?1|"
    r"web-?dl|web-?rip|blu-?ray|bd-?rip|hdtv|remux|10bit|8bit|hdr|sdr|"
    r"19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)


@dataclass
class MatchPair:
    """One proposed video/audio pairing."""

    primary_path: str
    secondary_path: str
    key: str
    method: str
    score: float

    @property
    def primary_name(self) -> str:
        return os.path.basename(self.primary_path)

    @property
    def secondary_name(self) -> str:
        return os.path.basename(self.secondary_path)

    def to_dict(self) -> dict:
        return {
            "primaryPath": self.primary_path,
            "secondaryPath": self.secondary_path,
            "primaryName": self.primary_name,
            "secondaryName": self.secondary_name,
            "key": self.key,
            "method": self.method,
            "score": self.score,
        }


@dataclass
class MatchReport:
    """The full outcome of matching two folders."""

    pairs: List[MatchPair]
    unmatched_primary: List[str]
    unmatched_secondary: List[str]
    method: str
    pattern_used: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "pairs": [pair.to_dict() for pair in self.pairs],
            "unmatchedPrimary": [os.path.basename(p) for p in self.unmatched_primary],
            "unmatchedSecondary": [os.path.basename(p) for p in self.unmatched_secondary],
            "method": self.method,
            "patternUsed": self.pattern_used,
            "warning": self.warning,
        }


def validate_pattern(pattern: str) -> Tuple[bool, Optional[str]]:
    """Check a user-supplied regex before a run starts.

    The original UI rendered a warning for a bad pattern but started the run
    anyway, which then matched nothing after a long wait.
    """
    if not pattern or not pattern.strip():
        return False, "Pattern is empty"
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return False, f"Invalid regex: {exc}"
    if compiled.groups < 1:
        return False, "Pattern must capture at least one group, e.g. S(\\d+)E(\\d+)"
    return True, None


def list_media(folder: str, kind: str = "any") -> List[str]:
    """List media files in a folder, sorted, non-recursive."""
    if not os.path.isdir(folder):
        return []
    predicate = {
        "video": is_video_file,
        "audio": is_audio_file,
    }.get(kind, is_media_file)

    entries = []
    for name in os.listdir(folder):
        if name.startswith("."):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) and predicate(path):
            entries.append(path)
    return sorted(entries)


def _strip_noise(name: str) -> str:
    return NOISE_TOKENS.sub(" ", name)


def _episode_key(filename: str, pattern: str) -> Optional[str]:
    """Extract a normalized season/episode key, ignoring release-metadata noise."""
    stem = _strip_noise(os.path.splitext(os.path.basename(filename))[0])
    match = re.search(pattern, stem, re.IGNORECASE)
    if not match:
        return None
    groups = [g for g in match.groups() if g is not None]
    if not groups:
        return None
    if len(groups) == 1:
        return f"e{int(groups[0]):03d}"
    return f"s{int(groups[0]):02d}e{int(groups[1]):03d}"


def _normalize_title(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = _strip_noise(stem)
    return re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()


def match_folders(
    primary_folder: str,
    secondary_folder: str,
    custom_pattern: Optional[str] = None,
    fuzzy_threshold: float = 0.72,
) -> MatchReport:
    """Pair files across two folders, preferring the most reliable method."""
    primaries = list_media(primary_folder, "video") or list_media(primary_folder)
    secondaries = list_media(secondary_folder)

    if not primaries:
        return MatchReport([], [], secondaries, "none", None, "No media files in the video folder")
    if not secondaries:
        return MatchReport([], primaries, [], "none", None, "No media files in the audio folder")

    if custom_pattern:
        valid, problem = validate_pattern(custom_pattern)
        if not valid:
            return MatchReport([], primaries, secondaries, "none", custom_pattern, problem)
        report = _match_by_pattern(primaries, secondaries, custom_pattern, "custom pattern")
        if report.pairs:
            return report
        return MatchReport(
            [], primaries, secondaries, "none", custom_pattern,
            "Custom pattern matched no pairs",
        )

    # Try each pattern on both sides first, so a folder that consistently uses
    # one convention pairs with a strong, unambiguous key.
    for pattern, label in EPISODE_PATTERNS:
        report = _match_by_pattern(primaries, secondaries, pattern, f"episode ({label})")
        if report.pairs:
            return report

    # Neither side matched the same single pattern. Episode numbering may still
    # be present in different notations on each side (S02E05 vs 2x05), so key
    # each file by whichever pattern it does match and pair on that key.
    report = _match_by_any_pattern(primaries, secondaries)
    if report.pairs:
        return report

    # Nothing episodic matched; fall back to filename similarity rather than
    # the original "any number" rule, which paired on resolution and year.
    return _match_by_similarity(primaries, secondaries, fuzzy_threshold)


def _match_by_pattern(
    primaries: Sequence[str], secondaries: Sequence[str], pattern: str, method: str
) -> MatchReport:
    primary_map: Dict[str, List[str]] = {}
    for path in primaries:
        key = _episode_key(path, pattern)
        if key:
            primary_map.setdefault(key, []).append(path)

    secondary_map: Dict[str, List[str]] = {}
    for path in secondaries:
        key = _episode_key(path, pattern)
        if key:
            secondary_map.setdefault(key, []).append(path)

    pairs: List[MatchPair] = []
    used_secondary = set()
    warning = None

    for key in sorted(set(primary_map) & set(secondary_map)):
        primary_group = sorted(primary_map[key])
        secondary_group = sorted(secondary_map[key])
        if len(primary_group) > 1 or len(secondary_group) > 1:
            warning = (
                f"Multiple files share key {key}; paired in name order. "
                "Check the pairing preview."
            )
        for primary_path, secondary_path in zip(primary_group, secondary_group):
            pairs.append(MatchPair(primary_path, secondary_path, key, method, 1.0))
            used_secondary.add(secondary_path)

    matched_primary = {pair.primary_path for pair in pairs}
    return MatchReport(
        pairs=pairs,
        unmatched_primary=[p for p in primaries if p not in matched_primary],
        unmatched_secondary=[s for s in secondaries if s not in used_secondary],
        method=method,
        pattern_used=pattern,
        warning=warning,
    )


def _episode_key_any(filename: str) -> Optional[str]:
    """Key a file by the first episode pattern it matches, whichever that is.

    Lets a folder named S02E05 pair with one named 2x05: both normalize to the
    same s02e005 key even though no single pattern matches both spellings.
    """
    for pattern, _label in EPISODE_PATTERNS:
        key = _episode_key(filename, pattern)
        if key:
            return key
    return None


def _match_by_any_pattern(
    primaries: Sequence[str], secondaries: Sequence[str]
) -> MatchReport:
    primary_map: Dict[str, List[str]] = {}
    for path in primaries:
        key = _episode_key_any(path)
        if key:
            primary_map.setdefault(key, []).append(path)

    secondary_map: Dict[str, List[str]] = {}
    for path in secondaries:
        key = _episode_key_any(path)
        if key:
            secondary_map.setdefault(key, []).append(path)

    pairs: List[MatchPair] = []
    used_secondary = set()
    warning = None
    for key in sorted(set(primary_map) & set(secondary_map)):
        primary_group = sorted(primary_map[key])
        secondary_group = sorted(secondary_map[key])
        if len(primary_group) > 1 or len(secondary_group) > 1:
            warning = (
                f"Multiple files share key {key}; paired in name order. "
                "Check the pairing preview."
            )
        for primary_path, secondary_path in zip(primary_group, secondary_group):
            pairs.append(
                MatchPair(primary_path, secondary_path, key, "episode (mixed notation)", 0.95)
            )
            used_secondary.add(secondary_path)

    matched_primary = {pair.primary_path for pair in pairs}
    return MatchReport(
        pairs=pairs,
        unmatched_primary=[p for p in primaries if p not in matched_primary],
        unmatched_secondary=[s for s in secondaries if s not in used_secondary],
        method="episode (mixed notation)",
        pattern_used=None,
        warning=warning,
    )


def _match_by_similarity(
    primaries: Sequence[str], secondaries: Sequence[str], threshold: float
) -> MatchReport:
    """Greedy best-first pairing on normalized filename similarity."""
    candidates = []
    for primary_path in primaries:
        primary_title = _normalize_title(primary_path)
        for secondary_path in secondaries:
            score = SequenceMatcher(
                None, primary_title, _normalize_title(secondary_path)
            ).ratio()
            if score >= threshold:
                candidates.append((score, primary_path, secondary_path))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    pairs: List[MatchPair] = []
    used_primary, used_secondary = set(), set()
    for score, primary_path, secondary_path in candidates:
        if primary_path in used_primary or secondary_path in used_secondary:
            continue
        used_primary.add(primary_path)
        used_secondary.add(secondary_path)
        pairs.append(
            MatchPair(primary_path, secondary_path, _normalize_title(primary_path),
                      "filename similarity", round(score, 3))
        )

    pairs.sort(key=lambda pair: pair.primary_path)
    warning = (
        "Paired by filename similarity rather than episode numbers. "
        "Review the pairing preview before running."
        if pairs
        else "No episode numbers found and filenames are not similar enough to pair."
    )
    return MatchReport(
        pairs=pairs,
        unmatched_primary=[p for p in primaries if p not in used_primary],
        unmatched_secondary=[s for s in secondaries if s not in used_secondary],
        method="filename similarity",
        pattern_used=None,
        warning=warning,
    )


def pair_movie_mode(video_paths: Sequence[str], audio_path: str) -> List[MatchPair]:
    """Every video against a single audio file."""
    return [
        MatchPair(video_path, audio_path, os.path.basename(video_path), "movie mode", 1.0)
        for video_path in video_paths
    ]
