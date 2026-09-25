"""Protocol tests for the stdin/stdout bridge.

These check the exact wire contract the Rust host depends on. The original code
had Python emitting snake_case `elapsed_ms` while Rust deserialized camelCase
`elapsedMs` with no rename, so every elapsed time silently became null on the
completion path -- a class of bug only a round-trip test catches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRIDGE = os.path.join(ROOT, "python", "bridge.py")
PYTHON = os.path.join(ROOT, "python", ".venv", "bin", "python")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable

# The dub batch test builds its pairs with test_dubsync's fixture writer.
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MANIFEST = os.path.join(HERE, "fixtures", "manifest.json")


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as handle:
        return json.load(handle)


def _case(name):
    for case in load_manifest()["cases"]:
        if case["name"] == name:
            return case
    raise KeyError(name)


def run_bridge(commands, timeout=300):
    """Send commands to the bridge and collect the emitted events."""
    payload = "".join(json.dumps(c) + "\n" for c in commands)
    payload += json.dumps({"command": "shutdown"}) + "\n"

    process = subprocess.run(
        [PYTHON, BRIDGE],
        input=payload.encode(),
        capture_output=True,
        timeout=timeout,
        cwd=ROOT,
    )
    events = []
    for line in process.stdout.decode().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"bridge emitted non-JSON line: {line!r} ({exc})")
    return events, process


def test_bridge_starts_and_reports_ready():
    events, process = run_bridge([{"command": "ping"}])
    assert process.returncode == 0, f"bridge exited {process.returncode}: {process.stderr.decode()[:400]}"
    kinds = [e["type"] for e in events]
    assert "ready" in kinds, f"no ready event: {kinds}"
    assert "pong" in kinds, f"ping not answered: {kinds}"


def test_every_event_has_a_type():
    events, _ = run_bridge([{"command": "ping"}])
    for event in events:
        assert "type" in event, f"event without type: {event}"


def test_probe_returns_camelcase_stream_info():
    case = _case("offset_0ms")
    events, _ = run_bridge([{"command": "probe", "path": case["primary"]}])
    probes = [e for e in events if e["type"] == "probe"]
    assert probes, "no probe event"
    probe = probes[0]
    assert probe.get("hasAudio") is True, f"audio not detected: {probe}"
    assert "duration" in probe and probe["duration"] > 0


def test_analyze_emits_results_with_camelcase_keys():
    """Field names must match the Rust structs exactly."""
    case = _case("offset_500ms")
    events, process = run_bridge([
        {
            "command": "analyze",
            "mode": "movie",
            "videoFiles": [case["primary"]],
            "audioFile": case["secondary"],
            "windowSeconds": 8.0,
            "windowCount": 3,
        }
    ])
    assert process.returncode == 0, process.stderr.decode()[:500]

    results = [e for e in events if e["type"] == "result"]
    assert results, f"no result events: {[e['type'] for e in events]}"

    result = results[0]
    for key in ("videoFile", "audioFile", "delayMs", "confidence", "elapsedMs"):
        assert key in result, f"missing {key} in result payload: {sorted(result)}"

    assert result["elapsedMs"] is not None, "elapsedMs was null (the old serde bug)"
    assert result["delayMs"] is not None


def test_done_event_preserves_elapsed_and_summary():
    """The completion payload must carry the same fields as streamed results."""
    case = _case("offset_50ms")
    events, _ = run_bridge([
        {
            "command": "analyze",
            "mode": "movie",
            "videoFiles": [case["primary"]],
            "audioFile": case["secondary"],
            "windowSeconds": 8.0,
            "windowCount": 3,
        }
    ])
    done = [e for e in events if e["type"] == "done"]
    assert done, "no done event"
    payload = done[-1]
    assert payload["results"], "done carried no results"
    assert payload["results"][0]["elapsedMs"] is not None, (
        "elapsedMs lost on the done path -- this was the original serde mismatch"
    )
    assert "summary" in payload and payload["summary"]["total"] == 1


def test_analyze_recovers_correct_offset_through_the_bridge():
    case = _case("offset_500ms")
    events, _ = run_bridge([
        {
            "command": "analyze",
            "mode": "movie",
            "videoFiles": [case["primary"]],
            "audioFile": case["secondary"],
            "windowSeconds": 8.0,
            "windowCount": 4,
        }
    ])
    result = [e for e in events if e["type"] == "result"][0]
    assert abs(result["delayMs"] - 500.0) < 10.0, (
        f"bridge reported {result['delayMs']:+.1f}ms, expected +500ms"
    )


def test_unrelated_audio_reports_error_not_a_number():
    case = _case("unrelated")
    events, _ = run_bridge([
        {
            "command": "analyze",
            "mode": "movie",
            "videoFiles": [case["primary"]],
            "audioFile": case["secondary"],
            "windowSeconds": 8.0,
            "windowCount": 3,
        }
    ])
    result = [e for e in events if e["type"] == "result"][0]
    assert result["delayMs"] is None, (
        f"unrelated audio reported {result['delayMs']}ms instead of an error"
    )
    assert result["error"], "no error explanation for unrelated audio"


def test_invalid_pattern_rejected_before_analysis():
    events, _ = run_bridge([
        {
            "command": "analyze",
            "mode": "series",
            "videoFolder": os.path.join(HERE, "fixtures"),
            "audioFolder": os.path.join(HERE, "fixtures"),
            "matchPattern": "S(\\d+E(",
        }
    ])
    errors = [e for e in events if e["type"] == "error"]
    assert errors, "invalid regex was not rejected"
    assert any("pattern" in e["message"].lower() for e in errors)


def test_malformed_json_does_not_kill_the_bridge():
    """A bad line must be reported and the bridge must keep serving."""
    payload = b'{"command": "ping"}\nnot json at all\n{"command": "ping"}\n{"command":"shutdown"}\n'
    process = subprocess.run(
        [PYTHON, BRIDGE], input=payload, capture_output=True, timeout=120, cwd=ROOT
    )
    events = [json.loads(l) for l in process.stdout.decode().splitlines() if l.strip()]
    kinds = [e["type"] for e in events]
    assert kinds.count("pong") == 2, f"bridge stopped serving after bad input: {kinds}"
    assert any(e["type"] == "error" for e in events), "malformed line not reported"


def test_unknown_command_is_reported():
    events, process = run_bridge([{"command": "definitely-not-a-command"}])
    assert process.returncode == 0
    assert any(e["type"] == "error" for e in events)


def test_cancel_reaches_a_run_in_flight():
    """Stop has to work while the work is happening.

    Commands used to run on the thread that reads stdin, so a cancel written
    during a run sat unread in the pipe until the run finished on its own.
    The Stop button did nothing, on every command, for as long as the work
    took. Here the cancel goes in while a dub sync is decoding, and the
    bridge must acknowledge it at once, end the run as cancelled, and still
    answer the next command.
    """
    import time

    case = _case("offset_0ms")
    process = subprocess.Popen(
        [PYTHON, BRIDGE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        text=True,
    )
    try:
        assert json.loads(process.stdout.readline())["type"] == "ready"
        # A dub sync of a 30s fixture against itself: a second or two of work,
        # long enough to cancel inside.
        process.stdin.write(json.dumps({
            "command": "dubsync",
            "videoPath": case["primary"],
            "dubPath": case["secondary"],
            "planOnly": True,
        }) + "\n")
        process.stdin.flush()
        # Give the run time to start decoding before pulling the plug.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            event = json.loads(process.stdout.readline())
            if event["type"] == "dubsyncProgress":
                break
            assert event["type"] != "dubsyncDone", "the run finished before it could be cancelled"

        sent_at = time.monotonic()
        process.stdin.write(json.dumps({"command": "cancel"}) + "\n")
        process.stdin.flush()

        acknowledged = finished = None
        while acknowledged is None or finished is None:
            event = json.loads(process.stdout.readline())
            if event["type"] == "cancelAck":
                acknowledged = time.monotonic() - sent_at
            elif event["type"] == "dubsyncDone":
                finished = event
        assert acknowledged < 2.0, f"cancel took {acknowledged:.1f}s to be acknowledged"
        assert finished.get("cancelled") is True, f"the run did not end as cancelled: {finished}"

        process.stdin.write(json.dumps({"command": "ping"}) + "\n")
        process.stdin.flush()
        assert json.loads(process.stdout.readline())["type"] == "pong", "bridge stopped serving after a cancel"
    finally:
        process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
        process.stdin.flush()
        process.wait(timeout=30)


def test_shutdown_lets_queued_work_finish():
    """A request followed at once by shutdown still completes: that is how a
    script -- and the CI smoke test -- talks to the engine."""
    case = _case("offset_500ms")
    events, process = run_bridge([{
        "command": "analyze",
        "mode": "movie",
        "videoFiles": [case["primary"]],
        "audioFile": case["secondary"],
        "windowSeconds": 8,
        "windowCount": 3,
    }])
    assert process.returncode == 0
    done = [e for e in events if e["type"] == "done"]
    assert done and not done[0].get("cancelled"), "the run was cut short by the shutdown"
    assert done[0]["results"] and done[0]["results"][0]["delayMs"] is not None


def test_dubsync_batch_plans_every_pair_and_reports_by_job():
    """A queue of pairs is planned in parallel, each under its own job index,
    and the batch ends with one outcome per job, in queue order."""
    import shutil
    import tempfile

    from test_dubsync import Workspace, build_pair

    jobs = []
    workspaces = []
    try:
        # One clean pair and one with a cut the engine must fill. The engine
        # needs ~2 minutes of material to pin an offset.
        pieces = [
            [("org", 0, 120)],
            [("org", 0, 60), ("extra", 3.0), ("org", 60, 120)],
        ]
        for piece_set in pieces:
            workspace = Workspace()
            workspace.__enter__()
            workspaces.append(workspace)
            build_pair(workspace, 120, piece_set)
            jobs.append({"videoPath": workspace.path("org.wav"),
                         "dubPath": workspace.path("dub.wav")})

        events, process = run_bridge(
            [{"command": "dubsyncBatch", "jobs": jobs, "planOnly": True, "maxWorkers": 2}],
            timeout=600,
        )
        assert process.returncode == 0, process.stderr.decode()[:600]

        starts = sorted(e["job"] for e in events if e["type"] == "dubsyncJobStart")
        plans = sorted(e["job"] for e in events if e["type"] == "dubsyncJobPlan")
        assert starts == [0, 1], f"jobs not started individually: {starts}"
        assert plans == [0, 1], f"jobs not planned individually: {plans}"

        done = [e for e in events if e["type"] == "dubsyncBatchDone"]
        assert len(done) == 1, "the batch did not end with one dubsyncBatchDone"
        outcomes = done[0]["outcomes"]
        assert [o["job"] for o in outcomes] == [0, 1], "outcomes are not in queue order"
        for outcome in outcomes:
            assert outcome.get("plan") and not outcome.get("error"), (
                f"job {outcome['job']} did not produce a clean plan: {outcome}"
            )
            assert outcome["plan"]["segments"], f"job {outcome['job']} planned nothing"
            assert outcome["plan"]["videoPath"] != outcome["plan"]["dubPath"]
        assert not done[0].get("cancelled")
    finally:
        for workspace in workspaces:
            workspace.__exit__(None, None, None)
        shutil.rmtree(os.path.dirname(jobs[0]["videoPath"]), ignore_errors=True)


def test_preview_pairs_dub_scope_chooses_the_matcher():
    """The dub tab's Movies scope pairs by filename; Series by episode."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="audiosync-bridge-pairs-")
    try:
        movies = ["Dune.mkv", "Interstellar.mkv"]
        dubs = ["Interstellar Hindi.ac3", "Dune Hindi.ac3"]
        movie_paths = [os.path.join(root, n) for n in movies]
        dub_paths = [os.path.join(root, n) for n in dubs]
        for path in [*movie_paths, *dub_paths]:
            open(path, "w").close()

        events, process = run_bridge([{
            "command": "previewPairs",
            "mode": "dubsync",
            "dubKind": "movies",
            "videoFiles": movie_paths,
            "audioFiles": dub_paths,
        }])
        pairs = [e for e in events if e["type"] == "pairs"][0]
        assert len(pairs["pairs"]) == 2, f"movies not paired: {pairs}"
        assert pairs["method"] == "filename similarity", pairs["method"]

        episodes = ["Show.S01E01.mkv", "Show.S01E02.mkv"]
        ep_dubs = ["Show.S01E02 Hindi.ac3", "Show.S01E01 Hindi.ac3"]
        ep_paths = [os.path.join(root, n) for n in episodes]
        ep_dub_paths = [os.path.join(root, n) for n in ep_dubs]
        for path in [*ep_paths, *ep_dub_paths]:
            open(path, "w").close()

        events, process = run_bridge([{
            "command": "previewPairs",
            "mode": "dubsync",
            "dubKind": "series",
            "videoFiles": ep_paths,
            "audioFiles": ep_dub_paths,
        }])
        pairs = [e for e in events if e["type"] == "pairs"][0]
        assert len(pairs["pairs"]) == 2, f"episodes not paired: {pairs}"
        assert pairs["method"].startswith("episode"), pairs["method"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_shot_cuts_reports_the_picture_cuts_of_a_span():
    """The cut editor's ruler: the shot changes of a span on the video's
    clock, to the frame, with the frame's length; a span too long is
    clamped and says so; a file with no picture has no cuts to give."""
    import shutil
    import tempfile

    from audiosync.media import ffmpeg_path

    root = tempfile.mkdtemp(prefix="audiosync-shots-")
    try:
        # Four solid shots of 5 s at 24 fps: cuts at exactly 5, 10 and 15 s.
        video = os.path.join(root, "cuts.mkv")
        shots = ["red", "blue", "green", "white"]
        inputs = []
        for colour in shots:
            inputs += ["-f", "lavfi", "-i", f"color=c={colour}:s=160x90:r=24:d=5"]
        graph = "".join(f"[{i}:v]" for i in range(len(shots))) + f"concat=n={len(shots)}:v=1:a=0[v]"
        subprocess.run(
            [ffmpeg_path(), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]",
             "-c:v", "libx264", "-preset", "ultrafast", video],
            check=True, capture_output=True,
        )
        sound = _case("offset_0ms")["primary"]
        events, process = run_bridge([
            {"command": "shotCuts", "path": video, "startS": 0, "endS": 20, "requestId": "a"},
            {"command": "shotCuts", "path": video, "startS": 6, "endS": 12, "requestId": "b"},
            {"command": "shotCuts", "path": video, "startS": 0, "endS": 5000, "requestId": "c"},
            {"command": "shotCuts", "path": sound, "startS": 0, "endS": 10, "requestId": "d"},
            {"command": "shotCuts", "path": os.path.join(root, "missing.mkv"), "startS": 0, "endS": 10, "requestId": "e"},
        ])
        assert process.returncode == 0, process.stderr.decode()[:600]
        replies = {e["requestId"]: e for e in events if e["type"] == "shotCuts"}
        assert sorted(replies) == ["a", "b", "c", "d", "e"], replies

        whole = replies["a"]
        assert [round(c, 2) for c in whole["cuts"]] == [5.0, 10.0, 15.0], whole
        assert abs(whole["frameS"] - 1 / 24) < 1e-6, whole
        assert (whole["startS"], whole["endS"]) == (0.0, 20.0), whole
        # A span inside one already read comes from what was decoded.
        assert [round(c, 2) for c in replies["b"]["cuts"]] == [10.0], replies["b"]
        # Clamped to ten minutes from its start.
        assert replies["c"]["endS"] == 600.0, replies["c"]
        assert [round(c, 2) for c in replies["c"]["cuts"]] == [5.0, 10.0, 15.0], replies["c"]
        # A sound file has no picture: no cuts, and no error either.
        assert replies["d"]["cuts"] is None and "error" not in replies["d"], replies["d"]
        assert replies["e"]["error"] == "file not found", replies["e"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
