"""Tests for file pairing, including the release-noise cases the old
'any number in the filename' fallback got wrong."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.matching import (  # noqa: E402
    match_auto,
    match_folders,
    match_lists,
    match_movies,
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


def test_match_lists_pairs_each_episode_with_its_own_dub():
    """The dub queue pairs explicit lists the same way folders pair."""
    with Folders(
        ["Show.S01E01.1080p.mkv", "Show.S01E02.1080p.mkv", "Show.S01E03.1080p.mkv"],
        ["Show.S01E03.dub.eac3", "Show.S01E01.dub.eac3", "Show.S01E02.dub.eac3"],
    ) as f:
        videos = [os.path.join(f.primary, n) for n in os.listdir(f.primary)]
        dubs = [os.path.join(f.secondary, n) for n in os.listdir(f.secondary)]
        report = match_lists(videos, dubs)
        assert len(report.pairs) == 3, f"got {len(report.pairs)} pairs"
        keys = sorted(p.key for p in report.pairs)
        assert keys == ["s01e001", "s01e002", "s01e003"], keys
        # Order in the lists does not matter: each video lands on its own dub.
        for pair in report.pairs:
            assert pair.primary_name[:11] == pair.secondary_name[:11], (
                f"mismatched: {pair.primary_name} <-> {pair.secondary_name}"
            )


def test_match_lists_reports_unmatched_on_either_side():
    with Folders(
        ["Movie.A.mkv", "Movie.B.mkv", "Movie.C.mkv"],
        ["Movie.A.dub.ac3"],
    ) as f:
        videos = [os.path.join(f.primary, n) for n in os.listdir(f.primary)]
        dubs = [os.path.join(f.secondary, n) for n in os.listdir(f.secondary)]
        report = match_lists(videos, dubs)
        assert len(report.pairs) == 1
        assert len(report.unmatched_primary) == 2, report.unmatched_primary
        assert report.unmatched_secondary == []


def test_match_movies_pairs_each_movie_with_its_own_dub():
    """Movies pair by filename alone, even when release noise and language
    words push the names apart."""
    with Folders(
        ["Interstellar (2014) 1080p BluRay.mkv", "Dune.mkv"],
        ["Dune Hindi.ac3", "Interstellar Hindi DD5.1 eac3"],
    ) as f:
        videos = [os.path.join(f.primary, n) for n in os.listdir(f.primary)]
        dubs = [os.path.join(f.secondary, n) for n in os.listdir(f.secondary)]
        report = match_movies(videos, dubs)
        assert len(report.pairs) == 2, f"got {len(report.pairs)} pairs: {report.warning}"
        by_video = {p.primary_name: p for p in report.pairs}
        assert by_video["Dune.mkv"].secondary_name == "Dune Hindi.ac3", (
            f"mismatched: {by_video['Dune.mkv']}"
        )
        assert by_video["Interstellar (2014) 1080p BluRay.mkv"].secondary_name == (
            "Interstellar Hindi DD5.1 eac3"
        )
        assert all(p.method == "filename similarity" for p in report.pairs)
        # Similarity is the only name-based method movies have, so a clean
        # pairing does not nag.
        assert report.warning is None, report.warning


def test_match_movies_pairs_unrelated_names_in_the_order_added():
    """A dub named after its language, not its film ("Video" against
    "Hindi"), is the normal case; the movies pair by list position rather
    than refusing to pair."""
    with Folders(
        ["Inception.mkv", "Dune.mkv"],
        ["Insidious Hindi.ac3", "Dune Hindi.ac3"],
    ) as f:
        videos = [os.path.join(f.primary, n) for n in os.listdir(f.primary)]
        dubs = [os.path.join(f.secondary, n) for n in os.listdir(f.secondary)]
        report = match_movies(videos, dubs)
        # Dune pairs by name; Inception, whose name no dub resembles, takes
        # the dub its name could not place, in the order added.
        by_video = {p.primary_name: p for p in report.pairs}
        assert len(report.pairs) == 2, report
        assert by_video["Dune.mkv"].secondary_name == "Dune Hindi.ac3"
        assert by_video["Dune.mkv"].method == "filename similarity"
        assert by_video["Inception.mkv"].secondary_name == "Insidious Hindi.ac3"
        assert by_video["Inception.mkv"].method == "list order"
        assert by_video["Inception.mkv"].score == 0.0
        assert report.method == "filename similarity + list order"
        # The pairs themselves say how they were made; no banner for
        # movies, whose pairing is a guess in the normal run of things.
        assert report.warning is None


def test_match_movies_orders_entirely_unrelated_lists_and_reports_leftovers():
    with Folders(
        ["Alpha.mkv", "Bravo.mkv", "Charlie.mkv"],
        ["One.ac3", "Two.ac3"],
    ) as f:
        # The app passes files in the order they were added; os.listdir's
        # order is arbitrary, so build the lists explicitly.
        videos = [os.path.join(f.primary, n) for n in ["Alpha.mkv", "Bravo.mkv", "Charlie.mkv"]]
        dubs = [os.path.join(f.secondary, n) for n in ["One.ac3", "Two.ac3"]]
        report = match_movies(videos, dubs)
        by_video = {p.primary_name: p for p in report.pairs}
        assert len(report.pairs) == 2, report
        assert by_video["Alpha.mkv"].secondary_name == "One.ac3"
        assert by_video["Bravo.mkv"].secondary_name == "Two.ac3"
        assert all(p.method == "list order" for p in report.pairs)
        assert report.method == "list order"
        assert len(report.unmatched_primary) == 1
        assert any(path.endswith("Charlie.mkv") for path in report.unmatched_primary)
        assert report.warning is None


def _paths(folder, names):
    return [os.path.join(folder, n) for n in names]


def test_match_auto_pairs_a_season_by_episode():
    """A season of episodes with their dubs takes the episode matcher."""
    with Folders(
        [f"Show.S01E0{n}.1080p.WEB-DL.mkv" for n in range(1, 6)],
        [f"Show.S01E0{n} Hindi.eac3" for n in (4, 2, 5, 1, 3)],
    ) as f:
        videos = _paths(f.primary, sorted(os.listdir(f.primary)))
        dubs = _paths(f.secondary, sorted(os.listdir(f.secondary)))
        report = match_auto(videos, dubs)
        assert report.method.startswith("episode"), report.method
        assert len(report.pairs) == 5 and not report.unmatched_primary, report
        for pair in report.pairs:
            assert pair.primary_name[:10] == pair.secondary_name[:10], (
                f"mismatched: {pair.primary_name} <-> {pair.secondary_name}"
            )
        assert report.to_dict() == {
            **match_lists(videos, dubs).to_dict()
        }, "auto on a clean season must equal the series result"


def test_match_auto_pairs_movies_by_filename_despite_resolution_tokens():
    """"1920x1080" reads as 20x108 and "1080" as a number to an episode
    pattern; the movies must still pair by name, never as episodes."""
    with Folders(
        ["Movie.Title.2019.1080p.BluRay.x264.mkv", "Other.Film.1920x1080.mkv"],
        ["Other Film hin.eac3", "Movie Title hindi.ac3"],
    ) as f:
        videos = _paths(f.primary, [
            "Movie.Title.2019.1080p.BluRay.x264.mkv", "Other.Film.1920x1080.mkv",
        ])
        dubs = _paths(f.secondary, ["Movie Title hindi.ac3", "Other Film hin.eac3"])
        report = match_auto(videos, dubs)
        assert not report.method.startswith("episode"), report.method
        assert report.method == "filename similarity", report.method
        by_video = {p.primary_name: p for p in report.pairs}
        assert len(report.pairs) == 2 and not report.unmatched_primary, report
        assert by_video["Movie.Title.2019.1080p.BluRay.x264.mkv"].secondary_name == (
            "Movie Title hindi.ac3"
        )
        assert by_video["Other.Film.1920x1080.mkv"].secondary_name == "Other Film hin.eac3"
        assert all(p.method == "filename similarity" for p in report.pairs)


def test_match_auto_rejects_episode_keys_that_appear_on_both_sides_by_accident():
    """Resolution tokens on both a movie and its dub give the episode
    matcher a shared key ("1920x1080" -> s19e...), but it pairs only
    part of the list, so the movies fall back to filename pairing."""
    with Folders(
        ["Alpha.Quest.1920x1080.mkv", "Beta.Run.mkv"],
        ["Alpha Quest 1920x1080 hin.eac3", "Beta Run hin.eac3"],
    ) as f:
        videos = _paths(f.primary, ["Alpha.Quest.1920x1080.mkv", "Beta.Run.mkv"])
        dubs = _paths(f.secondary, ["Alpha Quest 1920x1080 hin.eac3", "Beta Run hin.eac3"])
        assert match_lists(videos, dubs).unmatched_primary, "premise: episode path leaves one out"
        report = match_auto(videos, dubs)
        assert report.method == "filename similarity", report.method
        assert len(report.pairs) == 2 and not report.unmatched_primary
        by_video = {p.primary_name: p.secondary_name for p in report.pairs}
        assert by_video == {
            "Alpha.Quest.1920x1080.mkv": "Alpha Quest 1920x1080 hin.eac3",
            "Beta.Run.mkv": "Beta Run hin.eac3",
        }, by_video


def test_match_auto_falls_back_to_movies_when_episodes_do_not_cover_every_video():
    """Two episodes plus a film: the episode matcher pairs two of three,
    so the whole selection goes to the filename matcher instead."""
    with Folders(
        ["Show.S01E01.mkv", "Show.S01E02.mkv", "Some.Film.mkv"],
        ["Show.S01E01 Hindi.ac3", "Show.S01E02 Hindi.ac3", "Some Film Hindi.ac3"],
    ) as f:
        videos = _paths(f.primary, ["Show.S01E01.mkv", "Show.S01E02.mkv", "Some.Film.mkv"])
        dubs = _paths(f.secondary, [
            "Show.S01E01 Hindi.ac3", "Show.S01E02 Hindi.ac3", "Some Film Hindi.ac3",
        ])
        by_episode = match_lists(videos, dubs)
        assert by_episode.method.startswith("episode") and by_episode.unmatched_primary, (
            "premise: episode matcher pairs some, not all"
        )
        report = match_auto(videos, dubs)
        assert report.method == match_movies(videos, dubs).method
        assert not report.method.startswith("episode"), report.method
        assert len(report.pairs) == 3 and not report.unmatched_primary, report
        by_video = {p.primary_name: p.secondary_name for p in report.pairs}
        assert by_video["Some.Film.mkv"] == "Some Film Hindi.ac3", by_video


def test_match_auto_keeps_episode_pairing_for_a_season_with_a_missing_dub():
    """16 episodes, 15 episode dubs and a Special: E08 has no dub and the
    Special belongs to no episode. Both stay unpaired; the movie matcher
    would have paired them with each other by list order."""
    episodes = [f"Goblin.S01E{n:02d}.1080p.mkv" for n in range(1, 17)]
    dubs = [f"Goblin.S01E{n:02d}.hin.eac3" for n in range(1, 17) if n != 8]
    special = "Goblin.S01.Special.hin.eac3"
    with Folders(episodes, [*dubs, special]) as f:
        videos = _paths(f.primary, episodes)
        audio = _paths(f.secondary, [*dubs, special])
        report = match_auto(videos, audio)
        assert report.method.startswith("episode"), report.method
        assert len(report.pairs) == 15, len(report.pairs)
        assert [os.path.basename(p) for p in report.unmatched_primary] == [
            "Goblin.S01E08.1080p.mkv"
        ], report.unmatched_primary
        assert [os.path.basename(p) for p in report.unmatched_secondary] == [special], (
            report.unmatched_secondary
        )
        for pair in report.pairs:
            assert pair.primary_name[:14] == pair.secondary_name[:14], pair
        assert report.to_dict() == match_lists(videos, audio).to_dict()
        # The premise: the movie matcher would have got this one wrong.
        wrong = {p.primary_name: p.secondary_name for p in match_movies(videos, audio).pairs}
        assert wrong.get("Goblin.S01E08.1080p.mkv") == special, wrong


def test_match_auto_with_nothing_to_pair_reports_like_the_other_matchers():
    assert match_auto([], []).pairs == []
    report = match_auto(["/v/a.mkv"], [])
    assert report.pairs == [] and report.unmatched_primary == ["/v/a.mkv"]


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
