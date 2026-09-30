import { describe, expect, it } from "vitest";

import type { HistoryEntry, SyncResult } from "@/lib/types";

import { historyName, historyOutcome, severityOf, whenText } from "./dock";

const result = (videoFile: string): SyncResult => ({
  videoFile,
  audioFile: "dub.ac3",
  delayMs: 100,
  delayAtStartMs: 100,
  confidence: 0.9,
  driftMsPerS: null,
  totalDriftMs: null,
  hasSignificantDrift: false,
  startDelayMs: 100,
  endDelayMs: 100,
  windowsUsed: 6,
  windowsTotal: 6,
  error: null,
  elapsedMs: 1000,
});

const entry = (patch: Partial<HistoryEntry>): HistoryEntry => ({
  id: "run",
  date: "2026-09-29T10:00:00.000Z",
  mode: "movie",
  results: [],
  summary: null,
  fileCount: 0,
  ...patch,
});

describe("history rows", () => {
  it("uses a summary entry's own name and outcome", () => {
    const e = entry({ mode: "dubsync", name: "Skyline Heist", outcome: { tone: "ok", text: "1 synced" }, fileCount: 1 });
    expect(historyName(e)).toBe("Skyline Heist");
    expect(historyOutcome(e)).toEqual({ tone: "ok", text: "1 synced" });
  });

  it("names an analysis run after its first file", () => {
    expect(historyName(entry({ results: [result("Blade Runner 2049.mkv"), result("b.mkv")], fileCount: 2 }))).toBe("Blade Runner 2049 and 1 more");
  });

  it("counts each result of an analysis run once, as its row says", () => {
    const e = entry({
      results: [
        result("a.mkv"),
        result("b.mkv"),
        { ...result("pal.mkv"), hasSignificantDrift: true, isRateMismatch: true },
        { ...result("web.mkv"), hasSignificantDrift: true },
        { ...result("edit.mkv"), confidence: 0.31, isLikelyCut: true },
        { ...result("sample.mkv"), delayMs: null, error: "No audio track" },
      ],
      fileCount: 6,
      // The engine's totals overlap: drifting and rate changes are also matched.
      summary: { total: 6, matched: 5, failed: 1, drifting: 2, cuts: 1, rateMismatches: 1, high: 3, medium: 1, low: 1 },
    });
    expect(historyOutcome(e)).toEqual({ tone: "warn", text: "3 matched · 1 drifting · 1 different cut · 1 failed" });
  });

  it("falls back to the totals for a Find match run", () => {
    const e = entry({
      mode: "compare",
      results: [result("a.mkv")],
      summary: { total: 1, matched: 1, failed: 0, drifting: 0, cuts: 0, rateMismatches: 0, high: 1, medium: 0, low: 0 },
    });
    expect(historyOutcome(e)).toEqual({ tone: "ok", text: "1 matched" });
  });
});

describe("output colours", () => {
  it("reads severity from the words", () => {
    expect(severityOf("Error: the dub ends early")).toBe("bad");
    expect(severityOf("Skipped a file")).toBe("warn");
    expect(severityOf("Wrote Movie.synced.mkv")).toBe("ok");
    expect(severityOf("Reading Movie.mkv")).toBe("");
  });
});

describe("when a run finished", () => {
  const now = new Date(2026, 8, 30, 15, 0);
  it("says today and yesterday, then the date", () => {
    expect(whenText(new Date(2026, 8, 30, 11, 5).toISOString(), now)).toBe("Today 11:05");
    expect(whenText(new Date(2026, 8, 29, 22, 40).toISOString(), now)).toBe("Yesterday 22:40");
    expect(whenText(new Date(2026, 8, 26, 9, 30).toISOString(), now)).toBe("Sep 26 09:30");
    expect(whenText(new Date(2025, 8, 26, 9, 30).toISOString(), now)).toBe("Sep 26 2025");
    expect(whenText("not a date", now)).toBe("Unknown date");
  });
});
