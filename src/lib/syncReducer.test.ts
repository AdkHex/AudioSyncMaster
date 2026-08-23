import { describe, expect, it } from "vitest";

import {
  estimateRemainingMs,
  initialSyncState,
  syncReducer,
  validateSelection,
  type SyncState,
} from "./syncReducer";
import type { FileItem, SyncResult } from "./types";

function file(name: string, kind: "video" | "audio" = "video", folder = "/media"): FileItem {
  return { id: `${kind}-${name}`, name, path: `${folder}/${name}`, type: kind, size: 1024 };
}

function result(video: string, delayMs: number | null = 100): SyncResult {
  return {
    videoFile: video,
    audioFile: "dub.ac3",
    primaryPath: `/media/${video}`,
    secondaryPath: "/audio/dub.ac3",
    delayMs,
    delayAtStartMs: delayMs,
    confidence: delayMs === null ? 0 : 0.9,
    driftMsPerS: null,
    totalDriftMs: null,
    hasSignificantDrift: false,
    startDelayMs: delayMs,
    endDelayMs: delayMs,
    windowsUsed: 6,
    windowsTotal: 6,
    error: delayMs === null ? "no match" : null,
    elapsedMs: 1200,
  };
}

function reduce(state: SyncState, ...actions: Parameters<typeof syncReducer>[1][]): SyncState {
  return actions.reduce(syncReducer, state);
}

describe("file selection", () => {
  it("adds files and records the folder", () => {
    const state = syncReducer(initialSyncState, {
      type: "addFiles",
      kind: "video",
      files: [file("a.mkv"), file("b.mkv")],
      folder: "/media",
    });
    expect(state.videoFiles).toHaveLength(2);
    expect(state.videoFolder).toBe("/media");
  });

  it("ignores files that are already selected", () => {
    const state = reduce(
      initialSyncState,
      { type: "addFiles", kind: "video", files: [file("a.mkv")] },
      { type: "addFiles", kind: "video", files: [file("a.mkv"), file("b.mkv")] },
    );
    expect(state.videoFiles.map((f) => f.name)).toEqual(["a.mkv", "b.mkv"]);
  });

  it("clears the stale pairing preview whenever the selection changes", () => {
    const withPairing: SyncState = {
      ...initialSyncState,
      pairing: { pairs: [], unmatchedPrimary: [], unmatchedSecondary: [], method: "x", patternUsed: null, warning: null },
    };
    const state = syncReducer(withPairing, {
      type: "addFiles",
      kind: "video",
      files: [file("a.mkv")],
    });
    expect(state.pairing).toBeNull();
  });

  it("drops selections when the mode changes", () => {
    const state = reduce(
      initialSyncState,
      { type: "addFiles", kind: "video", files: [file("a.mkv")] },
      { type: "addFiles", kind: "audio", files: [file("a.ac3", "audio")] },
      { type: "setMode", mode: "series" },
    );
    expect(state.mode).toBe("series");
    expect(state.videoFiles).toHaveLength(0);
    expect(state.audioFiles).toHaveLength(0);
  });

  it("keeps state identical when re-selecting the current mode", () => {
    const before = syncReducer(initialSyncState, {
      type: "addFiles",
      kind: "video",
      files: [file("a.mkv")],
    });
    expect(syncReducer(before, { type: "setMode", mode: "movie" })).toBe(before);
  });
});

describe("results", () => {
  it("merges a repeated result for the same pair instead of duplicating it", () => {
    const state = reduce(
      initialSyncState,
      { type: "runStarted", total: 1 },
      { type: "result", result: result("a.mkv", 100) },
      { type: "result", result: result("a.mkv", 250) },
    );
    expect(state.results).toHaveLength(1);
    expect(state.results[0].delayMs).toBe(250);
  });

  it("keeps streamed results when the final payload is empty", () => {
    // This is the partial-failure case the original lost entirely.
    const state = reduce(
      initialSyncState,
      { type: "runStarted", total: 2 },
      { type: "result", result: result("a.mkv") },
      { type: "runFinished", results: [], summary: null, cancelled: true },
    );
    expect(state.results).toHaveLength(1);
    expect(state.status).toBe("cancelled");
  });

  it("prefers the engine's final list when it has content", () => {
    const state = reduce(
      initialSyncState,
      { type: "runStarted", total: 2 },
      { type: "result", result: result("a.mkv") },
      {
        type: "runFinished",
        results: [result("a.mkv"), result("b.mkv")],
        summary: null,
        cancelled: false,
      },
    );
    expect(state.results).toHaveLength(2);
    expect(state.status).toBe("complete");
  });

  it("clears previous results when a new run starts", () => {
    const state = reduce(
      initialSyncState,
      { type: "result", result: result("old.mkv") },
      { type: "runStarted", total: 1 },
    );
    expect(state.results).toHaveLength(0);
    expect(state.status).toBe("processing");
  });
});

describe("progress", () => {
  it("ignores progress from a file that is no longer active", () => {
    const state = reduce(
      initialSyncState,
      { type: "fileStart", file: "b.mkv" },
      { type: "fileProgress", file: "b.mkv", percent: 60 },
      { type: "fileProgress", file: "a.mkv", percent: 10 },
    );
    expect(state.fileProgress).toBe(60);
  });

  it("estimates remaining time from observed throughput", () => {
    const state: SyncState = {
      ...initialSyncState,
      status: "processing",
      startedAt: Date.now() - 10_000,
      progress: { processed: 2, total: 10 },
    };
    const remaining = estimateRemainingMs(state);
    expect(remaining).not.toBeNull();
    // 5s per item, 8 remaining => ~40s.
    expect(remaining! / 1000).toBeGreaterThan(30);
    expect(remaining! / 1000).toBeLessThan(50);
  });

  it("has no estimate before anything completes", () => {
    expect(estimateRemainingMs({ ...initialSyncState, status: "processing", startedAt: Date.now() }))
      .toBeNull();
  });
});

describe("logs", () => {
  it("caps retained lines so a long run cannot grow without bound", () => {
    let state = initialSyncState;
    for (let i = 0; i < 900; i += 1) {
      state = syncReducer(state, { type: "log", message: `line ${i}` });
    }
    expect(state.logs.length).toBeLessThanOrEqual(500);
    expect(state.logs.at(-1)).toBe("line 899");
  });
});

describe("validation", () => {
  it("requires video files", () => {
    expect(validateSelection(initialSyncState).ok).toBe(false);
  });

  it("requires audio in movie mode", () => {
    const state = syncReducer(initialSyncState, {
      type: "addFiles", kind: "video", files: [file("a.mkv")],
    });
    const check = validateSelection(state);
    expect(check.ok).toBe(false);
    expect(check.reason).toMatch(/audio file/i);
  });

  it("requires both folders in series mode", () => {
    const state = reduce(
      { ...initialSyncState, mode: "series" },
      { type: "addFiles", kind: "video", files: [file("a.mkv")], folder: "/v" },
      { type: "addFiles", kind: "audio", files: [file("a.ac3", "audio")] },
    );
    expect(validateSelection(state).ok).toBe(false);
  });

  it("accepts a complete movie-mode selection", () => {
    const state = reduce(
      initialSyncState,
      { type: "addFiles", kind: "video", files: [file("a.mkv")], folder: "/v" },
      { type: "addFiles", kind: "audio", files: [file("a.ac3", "audio")], folder: "/a" },
    );
    expect(validateSelection(state).ok).toBe(true);
  });
});

describe("lifecycle", () => {
  it("records the reason a run failed", () => {
    const state = reduce(
      initialSyncState,
      { type: "runStarted", total: 1 },
      { type: "runFailed", message: "engine exited" },
    );
    expect(state.status).toBe("idle");
    expect(state.error).toBe("engine exited");
    expect(state.logs.some((l) => l.includes("engine exited"))).toBe(true);
  });

  it("keeps the selected mode when clearing", () => {
    const state = reduce(
      initialSyncState,
      { type: "setMode", mode: "series" },
      { type: "addFiles", kind: "video", files: [file("a.mkv")] },
      { type: "clearAll" },
    );
    expect(state.mode).toBe("series");
    expect(state.videoFiles).toHaveLength(0);
  });

  it("restores a historical run without touching file selection", () => {
    const state = reduce(
      initialSyncState,
      { type: "addFiles", kind: "video", files: [file("a.mkv")] },
      { type: "loadResults", results: [result("old.mkv")], summary: null, mode: "movie" },
    );
    expect(state.status).toBe("complete");
    expect(state.results).toHaveLength(1);
    expect(state.videoFiles).toHaveLength(1);
  });
});
