"""Tests for the job model: duplicate keys, history round trips, the stage
list behind the Progress tab, and the numbered-copy output policy.

Ported from FPSConv's ``tests/test_stages.py`` (StageList) and
``tests/test_engine.py`` (JobModel, OverwritePolicy).
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.encode import progress  # noqa: E402
from audiosync.encode.job import Job, next_free  # noqa: E402


def _states(job: Job) -> list[tuple[str, str]]:
    return [(s["name"], s["state"]) for s in job.stages]


# FPSConv: JobModel.test_old_history_defaults_to_fps
def test_old_history_defaults_to_fps():
    job = Job.from_dict({"source": "/a.mkv", "conv_type": "24-25", "out_dir": "/o", "state": "done"})
    assert (job.task, job.target, job.target_channels, job.atmos, job.drc) == ("fps", "ddp", 0, True, "film_light")
    assert job.to_dict()["label"] == "24-25"


# FPSConv: JobModel.test_encode_fields_roundtrip
def test_encode_fields_roundtrip():
    job = Job(source="/a.mkv", conv_type="encode", out_dir="/o", task="encode", target="dd",
              target_channels=6, atmos=False, drc="speech", bitrate_override=448, atmos71="dee", bed_conform=False)
    job.label, job.pretty = "DD 5.1 448k", "TrueHD"
    back = Job.from_dict(job.to_dict())
    assert back.key() == job.key()
    assert (back.label, back.pretty) == ("DD 5.1 448k", "TrueHD")


# FPSConv: JobModel.test_key_distinguishes_encode_settings
def test_key_distinguishes_encode_settings():
    a = Job(source="/a.mkv", conv_type="encode", out_dir="/o", task="encode", target_channels=6)
    b = Job(source="/a.mkv", conv_type="encode", out_dir="/o", task="encode", target_channels=8)
    assert a.key() != b.key()
    fps1 = Job(source="/a.mkv", conv_type="24-25", out_dir="/o", bitrate_override=0)
    fps2 = Job(source="/a.mkv", conv_type="24-25", out_dir="/o", bitrate_override=640)
    assert fps1.key() == fps2.key()      # fps duplicate rule unchanged: source + mode + stream


# FPSConv: StageList.test_plan_then_run_through
def test_plan_then_run_through():
    job = Job(source="a.thd", conv_type="encode", out_dir="/out")
    job.plan_stages(["Extracting TrueHD", "truehdd decode", "DEE encode", "Finishing"])
    job.stage_skip("Extracting TrueHD", "source is already a .thd")
    assert job.overall() == 0.0

    job.stage_begin("truehdd decode")
    job.stage_progress(50)
    assert _states(job) == [("Extracting TrueHD", "skip"), ("truehdd decode", "running"),
                            ("DEE encode", "wait"), ("Finishing", "wait")]
    w = progress.STAGE_WEIGHT
    total = w["truehdd decode"] + w["DEE encode"] + w["Finishing"]      # the skipped one does not count
    assert abs(job.overall() - w["truehdd decode"] * 0.5 / total * 100) < 1e-3

    job.stage_begin("DEE encode")                                       # the previous one closes itself
    assert job.stages[1]["state"] == "done"
    assert job.stages[1]["pct"] == 100.0
    assert job.stages[1]["finished"] > 0
    job.stage_begin("DEE encode")                                       # same stage again: no-op
    assert job.stages[2]["state"] == "running"

    job.stage_begin("Finishing")
    job.stage_close(True)
    assert all(s["state"] in ("done", "skip") for s in job.stages)
    assert job.overall() == 100.0


# FPSConv: StageList.test_unplanned_stage_slots_in_before_the_waiting_ones
def test_unplanned_stage_slots_in_before_the_waiting_ones():
    job = Job(source="a.wav", conv_type="encode", out_dir="/out")
    job.plan_stages(["Decoding to WAV", "DEE: measure", "DEE: encode", "Finishing"])
    job.stage_begin("Decoding to WAV")
    job.stage_begin("ffmpeg")                     # deew's own pre-step the plan did not list
    assert [s["name"] for s in job.stages] == ["Decoding to WAV", "ffmpeg", "DEE: measure", "DEE: encode", "Finishing"]
    assert job.stages[1]["state"] == "running"


# FPSConv: StageList.test_failure_and_cancel_mark_the_running_stage
def test_failure_and_cancel_mark_the_running_stage():
    job = Job(source="a.thd", conv_type="encode", out_dir="/out")
    job.plan_stages(["truehdd decode", "DEE encode"])
    job.stage_begin("truehdd decode")
    job.stage_progress(31)
    job.stage_close(False)
    assert job.stages[0]["state"] == "failed"
    assert job.stages[0]["pct"] == 31                          # keeps where it stopped
    assert job.stages[1]["state"] == "wait"

    job2 = Job(source="b.thd", conv_type="encode", out_dir="/out")
    job2.plan_stages(["truehdd decode"])
    job2.stage_begin("truehdd decode")
    job2.cancel()
    job2.stage_close(False)
    assert job2.stages[0]["state"] == "cancelled"


# FPSConv: StageList.test_no_plan_falls_back_to_percent
def test_no_plan_falls_back_to_percent():
    job = Job(source="a.wav", conv_type="23.976-25", out_dir="/out")
    job.percent = 42.0
    assert job.overall() == 42.0
    job.stage_begin("ffmpeg atempo + AAC")         # runners still create stages on the fly
    assert _states(job) == [("ffmpeg atempo + AAC", "running")]


# FPSConv: StageList.test_history_roundtrip_and_snapshot_fields
def test_history_roundtrip_and_snapshot_fields():
    job = Job(source="a.thd", conv_type="encode", out_dir="/out")
    job.plan_stages(["truehdd decode", "Finishing"])
    job.stage_begin("truehdd decode")
    d = job.to_dict()
    assert "stages" in d and "overall" in d and "now" in d
    back = Job.from_dict(d)
    assert _states(back) == _states(job)
    assert Job.from_dict({"source": "old.mka", "state": "done"}).stages == []   # pre-stages history


# FPSConv: StageList.test_stage_of_strips_step_numbering
def test_stage_of_strips_step_numbering():
    assert progress.stage_of("Extracting TrueHD (1/5)") == "Extracting TrueHD"
    assert progress.stage_of("DEE flat 7.1 (3/5–4/5)") == "DEE flat 7.1"
    assert progress.stage_of("Encoding AAC Engine") == "ffmpeg atempo + AAC"


# FPSConv: OverwritePolicy.test_next_free_numbering
def test_next_free_numbering():
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "out.aac")
        open(base, "w").close()
        open(os.path.join(tmp, "out_1.aac"), "w").close()
        assert next_free(base) == os.path.join(tmp, "out_2.aac")


# ── beyond FPSConv's suite ──


def test_progress_lines_replace_each_other_until_something_is_said():
    job = Job(source="a.wav", conv_type="24-25", out_dir="/out")
    job.say("started")
    job.progress_line("ffmpeg 10%")
    job.progress_line("ffmpeg 20%")
    assert len(job.log) == 2 and job.log[1].endswith("ffmpeg 20%")
    job.say("finished")
    job.progress_line("deew 5%")
    assert len(job.log) == 4
