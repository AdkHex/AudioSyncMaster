"""Tests for ``audiosync-cli``: FPSConv's subcommands and flags, the exit
codes scripts rely on, and the engine still being the engine with no arguments.

The flag tables below are FPSConv v1.1.24's ``__main__.py``, so a script
written for ``fpsconv-cli.exe`` keeps working with only the exe renamed.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from audiosync.encode import cli  # noqa: E402
from test_bridge import BRIDGE, PYTHON  # noqa: E402
from test_encode_service import _fake_ffmpeg, _sine, sandbox  # noqa: E402

ANY_OS = {"AUDIOSYNC_ENCODE_ANY_OS": "1"}


def _options(sub: str) -> dict[str, tuple]:
    """Every option of a subcommand: its flags, default and choices."""
    action = next(a for a in cli.parser()._actions if a.dest == "cmd")
    out = {}
    for a in action.choices[sub]._actions:
        if a.dest != "help":
            out[a.dest] = (tuple(a.option_strings), a.default, tuple(a.choices) if a.choices else None)
    return out


def test_subcommands_are_fpsconvs_minus_gui():
    action = next(a for a in cli.parser()._actions if a.dest == "cmd")
    assert sorted(action.choices) == ["convert", "dee", "doctor", "encode", "patcher", "probe"]


def test_convert_flags_match_fpsconv():
    assert _options("convert") == {
        "inputs": ((), None, None),
        "mode": (("--mode", "-m"), None, ("23.976-24", "23.976-25", "24-23.976", "24-25", "25-23.976", "25-24")),
        "output": (("--output", "-o"), "output", None),
        "jobs": (("--jobs", "-j"), 1, None),
        "recursive": (("--recursive", "-r"), False, None),
        "stream": (("--stream", "-s"), 0, None),
        "bitrate": (("--bitrate", "-b"), 0, None),
        "overwrite": (("--overwrite",), "overwrite", ("overwrite", "skip", "rename")),
    }


def test_encode_flags_match_fpsconv():
    assert _options("encode") == {
        "inputs": ((), None, None),
        "format": (("--format", "-f"), "ddp", ("ddp", "dd")),
        "channels": (("--channels", "-c"), 0, (0, 1, 2, 6, 8)),
        "bitrate": (("--bitrate", "-b"), 0, None),
        "no_atmos": (("--no-atmos",), False, None),
        "atmos_71": (("--atmos-71",), "flat", ("flat", "dee")),
        "no_bed_conform": (("--no-bed-conform",), False, None),
        "drc": (("--drc",), "film_light", ("film_light", "film_standard", "music_light", "music_standard", "speech")),
        "output": (("--output", "-o"), "output", None),
        "jobs": (("--jobs", "-j"), 1, None),
        "recursive": (("--recursive", "-r"), False, None),
        "stream": (("--stream", "-s"), 0, None),
        "overwrite": (("--overwrite",), "overwrite", ("overwrite", "skip", "rename")),
    }


def test_probe_and_dee_flags_match_fpsconv():
    assert _options("probe") == {"inputs": ((), None, None), "target": (("--target", "-t"), None, None)}
    assert _options("dee") == {"path": ((), None, None)}


def test_only_known_subcommands_reach_the_cli():
    assert cli.is_cli(["convert", "a.mka", "-m", "24-25"])
    assert cli.is_cli(["--version"])
    assert not cli.is_cli([])
    assert not cli.is_cli(["-m", "deew"])


def _run(argv: list[str], env: dict | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, env or {}, clear=False), \
         contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
    return code, out.getvalue(), err.getvalue()


def test_every_subcommand_refuses_off_windows_with_exit_2():
    if sys.platform == "win32":
        return
    with mock.patch.dict(os.environ, {"AUDIOSYNC_ENCODE_ANY_OS": ""}):
        for argv in (["convert", "a.mka", "-m", "24-25"], ["encode", "a.mka"], ["probe", "a.mka"],
                     ["doctor"], ["patcher"], ["dee", "C:/dee.exe"]):
            code, out, err = _run(argv)
            assert code == 2, argv
            assert err.strip() == "Encode is available on Windows only.", err


def test_version_prints_the_app_version_everywhere():
    code, out, _ = _run(["--version"])
    assert code == 0
    with open(os.path.join(ROOT, "package.json"), encoding="utf-8") as handle:
        assert out.strip() == f"AudioSyncMaster {json.load(handle)['version']}"


def test_convert_exit_0_then_1_for_a_failed_job():
    with sandbox() as root:
        source = _sine(root / "film.m4a", 1.0)
        code, out, _ = _run(["convert", str(source), "-m", "23.976-25", "-o", str(root / "out")], ANY_OS)
        assert code == 0
        assert "done    film.m4a -> film_23_976-25.aac" in out
        # an AC-3 source without DEE fails: exit 1
        ac3 = _sine(root / "film.ac3", 1.0, codec="ac3")
        code, out, _ = _run(["convert", str(ac3), "-m", "25-24", "-o", str(root / "out")], ANY_OS)
        assert code == 1
        assert "FAILED  film.ac3 -> -" in out


def test_same_file_twice_is_one_job_and_still_exit_0():
    with sandbox() as root:
        source = _sine(root / "film.m4a", 1.0)
        code, out, err = _run(["convert", str(source), str(source), "-m", "24-25", "-o", str(root / "out")], ANY_OS)
        assert code == 0, err
        assert out.count("film_24-25.aac") == 1


def test_ctrl_c_cancels_everything_and_exits_130():
    with sandbox() as root:
        source = _sine(root / "film.m4a", 1.0)
        fake = _fake_ffmpeg(root)
        from audiosync.encode import settings
        settings.save_settings({"tools": {"ffmpeg": fake}})
        calls = {"n": 0}
        real_sleep = cli.time.sleep

        def interrupt(seconds):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            real_sleep(seconds)

        with mock.patch.object(cli.time, "sleep", side_effect=interrupt):
            code, out, _ = _run(["convert", str(source), "-m", "24-25", "-o", str(root / "out")], ANY_OS)
        assert code == 130
        assert out.rstrip().endswith("cancelled")
        assert not (root / "out" / "film_24-25.aac").exists(), "partial output left behind"
        assert not list((root / "cache" / "temp").glob("fpsjob_*")), "temp folder left behind"


def test_no_arguments_is_still_the_engine():
    process = subprocess.run([PYTHON, BRIDGE], input=b'{"command":"ping"}\n{"command":"shutdown"}\n',
                             capture_output=True, timeout=60, cwd=ROOT)
    events = [json.loads(line) for line in process.stdout.decode().splitlines() if line.strip()]
    assert [e["type"] for e in events] == ["ready", "pong"]


def test_the_engine_exe_runs_the_cli():
    process = subprocess.run([PYTHON, BRIDGE, "--version"], capture_output=True, timeout=60, cwd=ROOT)
    assert process.returncode == 0
    assert process.stdout.decode().startswith("AudioSyncMaster ")
