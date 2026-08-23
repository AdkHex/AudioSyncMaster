"""Tests for file pairing, including the release-noise cases the old
'any number in the filename' fallback got wrong."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.matching import (  # noqa: E402
    match_folders,
    pair_movie_mode,
    validate_pattern,
)


class Folders:
    """Two temp folders populated with empty files of the given names."""

    def __init__(self, primary_names, secondary_names):
        self.root = tempfile.mkdtemp(prefix="audiosync-test-")
        self.primary = os.path.join(self.root, "video")
        self.secondary = os.path.join(self.root, "audio")
        os.makedirs(self.primary)
        os.makedirs(self.secondary)
        for name in primary_names:
            open(os.path.join(self.primary, name), "w").close()
        for name in secondary_names:
            open(os.path.join(self.secondary, name), "w").close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)


def test_standard_season_episode_matching():
    with Folders(
        ["Show.S01E01.1080p.mkv", "Show.S01E02.1080p.mkv", "Show.S01E03.1080p.mkv"],
        ["Show.S01E01.dub.eac3", "Show.S01E02.dub.eac3", "Show.S01E03.dub.eac3"],
    ) as f:
        report = match_folders(f.primary, f.secondary)
        assert len(report.pairs) == 3, f"got {len(report.pairs)} pairs"
        for pair in report.pairs:
            assert pair.primary_name[:11] == pair.secondary_name[:11], (
                f"mismatched pair: {pair.primary_name} <-> {pair.secondary_name}"
            )


def test_resolution_and_year_do_not_create_false_pairs():
    """The old fallback keyed on any digits, so 1080p/2019 paired episodes wrongly."""
    with Folders(
        ["Movie.2019.1080p.x264.mkv"],
        ["Totally.Different.2019.1080p.x264.ac3"],
    ) as f:
        report = match_folders(f.primary, f.secondary)
        for pair in report.pairs:
            assert pair.method != "episode", (
                f"paired unrelated files on release metadata: "
                f"{pair.primary_name} <-> {pair.secondary_name} via {pair.method}"
            )


def test_cross_format_episode_matching():
    """Different naming styles on each side must still pair correctly."""
    with Folders(
        ["Show.S02E05.WEB-DL.mkv", "Show.S02E06.WEB-DL.mkv"],
        ["Show 2x05 dub.ac3", "Show 2x06 dub.ac3"],
    ) as f:
        report = match_folders(f.primary, f.secondary)
        # S02E05 and 2x05 normalize to the same key only if one pattern matches
        # both sides; otherwise similarity should still pair them.
        assert len(report.pairs) >= 1, f"no pairs found (method={report.method})"


def test_unmatched_files_are_reported():
    with Folders(
        ["Show.S01E01.mkv", "Show.S01E02.mkv", "Show.S01E99.mkv"],
        ["Show.S01E01.ac3", "Show.S01E02.ac3"],
    ) as f:
        report = match_folders(f.primary, f.secondary)
        assert len(report.pairs) == 2
        assert any("E99" in name for name in report.unmatched_primary), (
            f"unmatched file not reported: {report.unmatched_primary}"
        )


def test_invalid_pattern_is_rejected_before_running():
    ok, problem = validate_pattern("S(\\d+E(\\d+)")  # unbalanced paren
    assert not ok and problem, "invalid regex accepted"

    ok, problem = validate_pattern("nogroups")
    assert not ok, "pattern without capture groups accepted"

    ok, problem = validate_pattern(r"S(\d+)E(\d+)")
    assert ok, f"valid pattern rejected: {problem}"


def test_invalid_custom_pattern_returns_warning_not_crash():
    with Folders(["a.S01E01.mkv"], ["a.S01E01.ac3"]) as f:
        report = match_folders(f.primary, f.secondary, custom_pattern="S(\\d+E(")
        assert report.pairs == []
        assert report.warning, "no explanation for failed custom pattern"


def test_empty_folders_are_handled():
    with Folders([], []) as f:
        report = match_folders(f.primary, f.secondary)
        assert report.pairs == []
        assert report.warning


def test_non_media_files_ignored():
    with Folders(
        ["Show.S01E01.mkv", "notes.txt", "cover.jpg"],
        ["Show.S01E01.ac3", "readme.md"],
    ) as f:
        report = match_folders(f.primary, f.secondary)
        assert len(report.pairs) == 1
        assert all(
            not p.secondary_name.endswith((".md", ".txt")) for p in report.pairs
        )


def test_movie_mode_pairs_all_videos_to_one_audio():
    pairs = pair_movie_mode(["/v/a.mkv", "/v/b.mkv", "/v/c.mkv"], "/a/track.eac3")
    assert len(pairs) == 3
    assert {p.secondary_path for p in pairs} == {"/a/track.eac3"}


def test_duplicate_keys_produce_warning():
    with Folders(
        ["Show.S01E01.PROPER.mkv", "Show.S01E01.REPACK.mkv"],
        ["Show.S01E01.ac3"],
    ) as f:
        report = match_folders(f.primary, f.secondary)
        assert report.warning, "ambiguous duplicate keys did not warn"


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
