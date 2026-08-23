import { describe, expect, it } from "vitest";

import {
  confidenceLevel,
  ffmpegCommandFor,
  formatDelay,
  formatDrift,
  formatDuration,
  formatElapsed,
  formatSize,
  type SyncResult,
} from "./types";

function make(overrides: Partial<SyncResult> = {}): SyncResult {
  return {
    videoFile: "ep01.mkv",
    audioFile: "ep01.ac3",
    primaryPath: "/media/ep01.mkv",
    secondaryPath: "/audio/ep01.ac3",
    delayMs: 250,
    delayAtStartMs: 250,
    confidence: 0.9,
    driftMsPerS: null,
    totalDriftMs: null,
    hasSignificantDrift: false,
    startDelayMs: 250,
    endDelayMs: 250,
    windowsUsed: 6,
    windowsTotal: 6,
    error: null,
    elapsedMs: 3400,
    ...overrides,
  };
}

describe("confidence banding", () => {
  it("maps engine scores onto bands", () => {
    expect(confidenceLevel(make({ confidence: 0.95 }))).toBe("high");
    expect(confidenceLevel(make({ confidence: 0.6 }))).toBe("medium");
    expect(confidenceLevel(make({ confidence: 0.2 }))).toBe("low");
  });

  it("treats an errored or unmatched result as low", () => {
    expect(confidenceLevel(make({ error: "no match", delayMs: null }))).toBe("low");
    expect(confidenceLevel(make({ delayMs: null, confidence: null }))).toBe("low");
  });
});

describe("formatting", () => {
  it("signs delays explicitly so direction is never ambiguous", () => {
    expect(formatDelay(250)).toBe("+250.0 ms");
    expect(formatDelay(-250)).toBe("-250.0 ms");
    expect(formatDelay(0)).toBe("0.0 ms");
    expect(formatDelay(null)).toBe("--");
  });

  it("formats drift and duration", () => {
    expect(formatDrift(0.25)).toBe("+0.250 ms/s");
    expect(formatDrift(null)).toBe("--");
    expect(formatDuration(3725)).toBe("1:02:05");
    expect(formatDuration(125)).toBe("2:05");
    expect(formatDuration(null)).toBe("--");
  });

  it("formats sizes and elapsed times", () => {
    expect(formatSize(1024)).toBe("1.0 KB");
    expect(formatSize(0)).toBe("--");
    expect(formatSize(null)).toBe("--");
    expect(formatElapsed(450)).toBe("450 ms");
    expect(formatElapsed(4500)).toBe("4.5 s");
    expect(formatElapsed(null)).toBe("--");
  });

  it("handles non-finite values without producing NaN text", () => {
    expect(formatDelay(Number.NaN)).toBe("--");
    expect(formatDrift(Number.POSITIVE_INFINITY)).toBe("--");
  });
});

describe("ffmpeg command", () => {
  it("trims the head of a late audio track", () => {
    // Positive delay = audio starts later = discard that much from its start.
    const command = ffmpegCommandFor(make({ delayMs: 250 }))!;
    expect(command).toContain("-ss 0.250000");
    expect(command).not.toContain("adelay");
  });

  it("pads an early audio track with silence", () => {
    const command = ffmpegCommandFor(make({ delayMs: -250 }))!;
    expect(command).toContain("adelay=250.000");
    expect(command).not.toContain("-ss");
  });

  it("copies the audio stream when only trimming", () => {
    expect(ffmpegCommandFor(make({ delayMs: 250 }))!).toContain("-c:a copy");
  });

  it("returns nothing when there is no measurement to apply", () => {
    expect(ffmpegCommandFor(make({ delayMs: null }))).toBeNull();
    expect(ffmpegCommandFor(make({ primaryPath: null }))).toBeNull();
  });

  it("quotes paths containing spaces", () => {
    const command = ffmpegCommandFor(
      make({ primaryPath: "/media/My Show S01E01.mkv" }),
    )!;
    expect(command).toContain('"/media/My Show S01E01.mkv"');
  });
});
