import { describe, expect, it } from "vitest";

import { initialRunState, packsReducer, runProgress, runReducer, type RunState } from "./reducer";
import type { SubJobOutcome } from "./types";

function started(): RunState {
  return runReducer(initialRunState, {
    type: "started",
    task: "sync",
    at: 1000,
    jobs: [
      { id: "a", label: "Movie.en.srt" },
      { id: "b", label: "Other.en.srt" },
    ],
  });
}

function outcome(job: number, id: string, patch: Partial<SubJobOutcome> = {}): SubJobOutcome {
  return {
    job,
    id,
    task: "sync",
    outputs: [{ path: `/out/${id}.srt`, kind: "subtitle", format: "srt", language: "en", label: null }],
    report: { offsetMs: 3250 },
    summary: "Shifted +3.250 s",
    warnings: [],
    preview: null,
    ...patch,
  };
}

describe("runReducer", () => {
  it("queues every job", () => {
    const state = started();
    expect(state.status).toBe("running");
    expect(state.jobs.map((job) => job.status)).toEqual(["queued", "queued"]);
    expect(runProgress(state)).toEqual({ done: 0, total: 2, percent: 0 });
  });

  it("routes start, progress and logs to the job by index", () => {
    let state = started();
    state = runReducer(state, { type: "event", event: { type: "subsJobStart", job: 1, id: "b", task: "sync" } });
    state = runReducer(state, { type: "event", event: { type: "subsJobProgress", job: 1, percent: 42.4, stage: "reading the audio" } });
    state = runReducer(state, { type: "event", event: { type: "subsJobLog", job: 1, message: "speech found" } });
    expect(state.jobs[0].status).toBe("queued");
    expect(state.jobs[1]).toMatchObject({ status: "running", percent: 42, stage: "reading the audio", logs: ["speech found"] });
    expect(runProgress(state).percent).toBeCloseTo(21, 0);
  });

  it("marks outcomes done or failed, and settles the batch", () => {
    let state = started();
    state = runReducer(state, { type: "event", event: { type: "subsJobDone", ...outcome(0, "a") } });
    expect(state.jobs[0].status).toBe("done");
    state = runReducer(state, {
      type: "event",
      event: { type: "subsBatchDone", outcomes: [outcome(0, "a"), outcome(1, "b", { error: "No speech found" })] },
    });
    expect(state.status).toBe("complete");
    expect(state.jobs[1]).toMatchObject({ status: "failed", outcome: { error: "No speech found" } });
  });

  it("the promise result after the done event changes nothing", () => {
    let state = started();
    state = runReducer(state, { type: "finished", outcomes: [outcome(0, "a"), outcome(1, "b")] });
    const again = runReducer(state, { type: "finished", outcomes: [] });
    expect(again).toBe(state);
  });

  it("a cancelled batch marks the jobs that never ran as stopped", () => {
    let state = started();
    state = runReducer(state, { type: "event", event: { type: "subsJobStart", job: 0, id: "a", task: "sync" } });
    state = runReducer(state, { type: "finished", outcomes: [outcome(0, "a", { cancelled: true })], cancelled: true });
    expect(state.status).toBe("cancelled");
    expect(state.jobs.map((job) => job.status)).toEqual(["cancelled", "cancelled"]);
  });

  it("a failed invoke keeps the message", () => {
    const state = runReducer(started(), { type: "failed", message: "The engine stopped" });
    expect(state).toMatchObject({ status: "failed", error: "The engine stopped" });
  });

  it("ignores events outside a run", () => {
    const state = runReducer(initialRunState, { type: "event", event: { type: "subsJobLog", job: 0, message: "x" } });
    expect(state).toBe(initialRunState);
  });

  it("keeps the log of a chatty job bounded", () => {
    let state = started();
    for (let i = 0; i < 400; i += 1) {
      state = runReducer(state, { type: "event", event: { type: "subsJobLog", job: 0, message: `line ${i}` } });
    }
    expect(state.jobs[0].logs).toHaveLength(300);
    expect(state.jobs[0].logs.at(-1)).toBe("line 399");
  });
});

describe("packsReducer", () => {
  it("follows an install from progress to done", () => {
    let state = packsReducer({}, { type: "begin", pack: "speech", action: "install" });
    state = packsReducer(state, {
      type: "event",
      event: { type: "packProgress", pack: "speech", percent: 50, stage: "downloading", bytes: 100, totalBytes: 200 },
    });
    expect(state.speech).toMatchObject({ busy: true, percent: 50, bytes: 100, totalBytes: 200 });
    state = packsReducer(state, { type: "event", event: { type: "packDone", pack: "speech", ok: true } });
    expect(state.speech).toMatchObject({ busy: false, error: null });
  });

  it("keeps the error of a failed install", () => {
    const state = packsReducer({}, { type: "done", pack: "ocr", ok: false, error: "No network" });
    expect(state.ocr.error).toBe("No network");
  });
});
