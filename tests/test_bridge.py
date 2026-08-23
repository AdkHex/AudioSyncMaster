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
