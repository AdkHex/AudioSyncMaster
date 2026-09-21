import { describe, expect, it } from "vitest";

import { describePlan, dubQueueReducer, initialDubQueueState, type DubQueueState } from "./dubQueueReducer";
import type { DubJobOutcome, DubSyncPlan } from "./types";

/** Two jobs queued: the shape the engine sees after a pairing. */
function queued(): DubQueueState {
  return dubQueueReducer(initialDubQueueState, {
    type: "queueStarted",
    jobs: [
      { name: "Show.S01E01.mkv", dubName: "Show.S01E01.dub.eac3" },
      { name: "Show.S01E02.mkv", dubName: "Show.S01E02.dub.eac3" },
    ],
  });
}

const plan: DubSyncPlan = {
  videoPath: "/v/Show.S01E01.mkv",
  dubPath: "/a/Show.S01E01.dub.eac3",
  videoTrack: 0,
  dubTrack: 0,
  speed: 1,
  videoFps: null,
  dubRate: null,
  fillGainDb: 0,
  videoDurationS: 2700,
  dubDurationS: 2600,
  segments: [],
  warnings: [],
  error: null,
  filledS: 0,
};

const outcome: DubJobOutcome = {
  job: 0,
  plan,
  output: {
    outputPath: "/v/Show.S01E01.dub.dubsynced.flac",
    sampleRate: 48000,
    channels: 6,
    seconds: 2700,
    clippedSamples: 0,
    warnings: [],
  },
  verification: null,
  muxedPath: null,
};

describe("dubQueueReducer", () => {
  it("starts a queue of jobs, all queued", () => {
    const state = queued();
    expect(state.status).toBe("running");
    expect(state.jobs).toHaveLength(2);
    expect(state.jobs.every((job) => job.status === "queued")).toBe(true);
    expect(state.done).toBe(0);
  });

  it("routes progress and the plan to the right job by index", () => {
    let state = dubQueueReducer(queued(), { type: "jobStart", job: 0 });
    state = dubQueueReducer(state, { type: "jobProgress", job: 0, percent: 40, stage: "finding the offsets" });
    state = dubQueueReducer(state, { type: "jobProgress", job: 1, percent: 10, stage: "reading the dub" });
    state = dubQueueReducer(state, { type: "jobPlan", job: 0, plan });
    expect(state.jobs[0].status).toBe("running");
    expect(state.jobs[0].percent).toBe(40);
    expect(state.jobs[0].plan).toEqual(plan);
    expect(state.jobs[1].percent).toBe(10);
    expect(state.jobs[1].plan).toBeNull();
  });

  it("closes a job as done when its outcome streams in", () => {
    const state = dubQueueReducer(queued(), { type: "jobDone", outcome });
    expect(state.jobs[0].status).toBe("done");
    expect(state.jobs[0].output?.outputPath).toBe("/v/Show.S01E01.dub.dubsynced.flac");
    expect(state.jobs[1].status).toBe("queued");
    expect(state.done).toBe(1);
    // The batch is still running while other jobs remain.
    expect(state.status).toBe("running");
  });

  it("counts a duplicate completion once", () => {
    let state = dubQueueReducer(queued(), { type: "jobDone", outcome });
    state = dubQueueReducer(state, { type: "jobDone", outcome });
    expect(state.done).toBe(1);
  });

  it("marks a job failed from its outcome error", () => {
    const state = dubQueueReducer(queued(), {
      type: "jobDone",
      outcome: { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "No part of the dub could be matched" },
    });
    expect(state.jobs[1].status).toBe("failed");
    expect(state.jobs[1].error).toBe("No part of the dub could be matched");
    expect(state.done).toBe(1);
  });

  it("ends the batch when every job has an outcome, engine list winning", () => {
    const running = dubQueueReducer(queued(), { type: "jobStart", job: 0 });
    const state = dubQueueReducer(running, {
      type: "batchDone",
      outcomes: [
        outcome,
        { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "cut too deep" },
      ],
      cancelled: false,
    });
    expect(state.status).toBe("complete");
    expect(state.jobs[0].status).toBe("done");
    expect(state.jobs[1].status).toBe("failed");
    expect(state.done).toBe(2);
  });

  it("ends a cancelled batch as cancelled, keeping streamed work", () => {
    let state = dubQueueReducer(queued(), { type: "jobStart", job: 0 });
    state = dubQueueReducer(state, { type: "jobPlan", job: 0, plan });
    state = dubQueueReducer(state, { type: "batchDone", outcomes: [], cancelled: true });
    expect(state.status).toBe("cancelled");
    expect(state.jobs[0].plan).toEqual(plan);
    expect(state.jobs[0].status).toBe("cancelled");
    expect(state.done).toBe(2);
  });

  it("records a batch-level failure", () => {
    const state = dubQueueReducer(queued(), { type: "batchFailed", message: "Engine crashed" });
    expect(state.status).toBe("failed");
    expect(state.error).toBe("Engine crashed");
  });

  it("resets to an empty idle queue", () => {
    const state = dubQueueReducer(queued(), { type: "reset" });
    expect(state).toEqual(initialDubQueueState);
  });
});

describe("describePlan", () => {
  const base = {
    ...plan,
    videoFps: null as number | null,
    dubRate: null as number | null,
    speed: 1,
  };

  it("names a rate mismatch, read from the video's metadata", () => {
    const text = describePlan({
      ...base,
      videoFps: 24000 / 1001,
      dubRate: 25,
      speed: 25 / (24000 / 1001),
    });
    expect(text).toContain("video 23.976 fps, dub mastered at 25 fps");
    expect(text).toContain("played at 1.042708×");
  });

  it("confirms a matched rate instead of leaving it implied", () => {
    const text = describePlan({ ...base, videoFps: 24000 / 1001, dubRate: 24000 / 1001 });
    expect(text).toContain("video 23.976 fps, dub at the same rate");
    expect(text).not.toContain("played at");
  });

  it("falls back to the bare speed when the video has no frame rate", () => {
    const text = describePlan({ ...base, videoFps: null, dubRate: null, speed: 0.999001 });
    expect(text).toContain("dub played at 0.999001×");
    expect(text).not.toContain("mastered");
  });
});
