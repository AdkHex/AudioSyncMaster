import { describe, expect, it } from "vitest";

import {
  announceApplyFinished,
  announceProgress,
  announceRunFailed,
  announceRunFinished,
  announceRunStarted,
} from "./announce";
import type { RunSummary, SyncResult } from "./types";

function result(overrides: Partial<SyncResult> = {}): SyncResult {
  return {
    videoFile: "ep01.mkv",
    audioFile: "ep01.ac3",
    delayMs: 120,
    delayAtStartMs: 120,
    confidence: 0.9,
    driftMsPerS: null,
    totalDriftMs: null,
    hasSignificantDrift: false,
    startDelayMs: 120,
    endDelayMs: 120,
    windowsUsed: 6,
    windowsTotal: 6,
    error: null,
    elapsedMs: 1000,
    ...overrides,
  };
}

function summary(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    total: 1, matched: 1, failed: 0, drifting: 0,
    cuts: 0, rateMismatches: 0, high: 1, medium: 0, low: 0,
    ...overrides,
  };
}

describe("run start", () => {
  it("says how much work is queued", () => {
    expect(announceRunStarted(12, "series").message).toContain("12 pairs");
  });

  it("calls them combinations in compare mode", () => {
    expect(announceRunStarted(6, "compare").message).toContain("6 combinations");
  });

  it("uses the singular for one item", () => {
    const { message } = announceRunStarted(1, "movie");
    expect(message).toContain("1 pair");
    expect(message).not.toContain("pairs");
  });
});

describe("run completion", () => {
  it("reports a clean run politely", () => {
    const announcement = announceRunFinished([result()], summary(), false);
    expect(announcement.politeness).toBe("polite");
    expect(announcement.message).toContain("1 file matched");
  });

  it("interrupts when files failed", () => {
    // A failure needs a decision, so it should not wait behind other speech.
    const announcement = announceRunFinished(
      [result()], summary({ matched: 4, failed: 2 }), false,
    );
    expect(announcement.politeness).toBe("assertive");
    expect(announcement.message).toContain("2 failed");
  });

  it("interrupts when a different cut was found", () => {
    const announcement = announceRunFinished([result()], summary({ cuts: 1 }), false);
    expect(announcement.politeness).toBe("assertive");
    expect(announcement.message).toContain("1 different cut");
  });

  it("mentions drift and frame-rate mismatches", () => {
    const { message } = announceRunFinished(
      [result()], summary({ drifting: 3, rateMismatches: 2 }), false,
    );
    expect(message).toContain("3 files drifting");
    expect(message).toContain("2 frame rate mismatches");
  });

  it("says results were kept when cancelled", () => {
    const { message } = announceRunFinished([result(), result()], null, true);
    expect(message).toContain("stopped");
    expect(message).toContain("2 results kept");
  });

  it("interrupts when nothing at all came back", () => {
    const announcement = announceRunFinished([], summary({ total: 0, matched: 0 }), false);
    expect(announcement.politeness).toBe("assertive");
    expect(announcement.message).toContain("no results");
  });
});

describe("failures", () => {
  it("carries the reason and interrupts", () => {
    const announcement = announceRunFailed("FFmpeg was not found.");
    expect(announcement.politeness).toBe("assertive");
    expect(announcement.message).toContain("FFmpeg was not found.");
  });
});

describe("applying corrections", () => {
  it("reports what was written", () => {
    const announcement = announceApplyFinished(5, 0);
    expect(announcement.message).toContain("5 corrected files written");
    expect(announcement.politeness).toBe("polite");
  });

  it("interrupts when some failed", () => {
    expect(announceApplyFinished(3, 2).politeness).toBe("assertive");
  });

  it("interrupts when nothing could be written", () => {
    const announcement = announceApplyFinished(0, 4);
    expect(announcement.politeness).toBe("assertive");
    expect(announcement.message).toContain("No files were written");
  });
});

describe("progress throttling", () => {
  it("speaks only at quarter marks", () => {
    // Announcing every file would flood a screen reader on a long season.
    const spoken: number[] = [];
    for (let i = 1; i <= 20; i += 1) {
      if (announceProgress(i, 20)) spoken.push(i);
    }
    expect(spoken).toEqual([5, 10, 15]);
  });

  it("says nothing at the start or the end", () => {
    expect(announceProgress(0, 10)).toBeNull();
    expect(announceProgress(10, 10)).toBeNull();
  });

  it("handles a single-file run without dividing by zero", () => {
    expect(announceProgress(1, 1)).toBeNull();
    expect(announceProgress(1, 0)).toBeNull();
  });

  it("includes the counts it is reporting", () => {
    const announcement = announceProgress(5, 20);
    expect(announcement?.message).toContain("25 percent");
    expect(announcement?.message).toContain("5 of 20");
  });
});
