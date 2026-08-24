import { describe, expect, it } from "vitest";

import {
  confidenceLevel,
  frameOffset,
  ffmpegCommandFor,
  formatDelay,
  formatDrift,
  formatDuration,
  formatElapsed,
  formatSize,
  streamSummary,
  type AudioTrackInfo,
  type SyncResult,
  type TrackListing,
} from "./types";

function track(overrides: Partial<AudioTrackInfo> = {}): AudioTrackInfo {
  return {
    index: 0,
    codec: "ac3",
    language: "eng",
    title: null,
    channels: 6,
    sampleRate: 48000,
    bitRate: 448000,
    isDefault: true,
    label: "Track 1",
    ...overrides,
  };
}

function listing(overrides: Partial<TrackListing> = {}): TrackListing {
  return {
    path: "/media/ep01.mkv",
    name: "ep01.mkv",
    tracks: [track()],
    fps: 23.976,
    duration: 3600,
    ...overrides,
  };
}

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

describe("frame offset", () => {
  it("converts a delay into frames at the file's rate", () => {
    // 8 frames at 23.976fps is ~333ms; this is the number an editor thinks in.
    expect(frameOffset(333.7, 23.976)).toBe(8);
    expect(frameOffset(1000, 25)).toBe(25);
  });

  it("signs the result so direction survives", () => {
    expect(frameOffset(-333.7, 23.976)).toBe(-8);
  });

  it("says nothing when the offset rounds to under a frame", () => {
    expect(frameOffset(5, 23.976)).toBeNull();
  });

  it("says nothing without a frame rate", () => {
    expect(frameOffset(500, null)).toBeNull();
    expect(frameOffset(500, undefined)).toBeNull();
    expect(frameOffset(null, 25)).toBeNull();
  });

  it("ignores non-finite input", () => {
    expect(frameOffset(Number.NaN, 25)).toBeNull();
    expect(frameOffset(500, Number.POSITIVE_INFINITY)).toBeNull();
  });
});

describe("stream summary", () => {
  it("reports frame rate for video, which is what explains steady drift", () => {
    const summary = streamSummary(undefined, listing(), "video");
    expect(summary).toContain("23.976 fps");
    expect(summary).toContain("AC3");
    expect(summary).toContain("5.1");
  });

  it("never reports frame rate for audio, which has no frames", () => {
    // An audio-only file probes with fps null; claiming one would be inventing
    // a property the format does not have.
    const summary = streamSummary(undefined, listing({ fps: null }), "audio");
    expect(summary).not.toContain("fps");
    expect(summary).toContain("AC3");
  });

  it("drops the sample rate when it is the universal 48 kHz", () => {
    // Space in the sidebar is scarce; a value that is the same on every film
    // release earns none of it.
    expect(streamSummary(undefined, listing(), "audio")).not.toContain("kHz");
    expect(
      streamSummary(undefined, listing({ tracks: [track({ sampleRate: 44100 })] }), "audio"),
    ).toContain("44.1 kHz");
  });

  it("counts tracks only when there is a choice to make", () => {
    expect(streamSummary(undefined, listing(), "audio")).not.toContain("tracks");

    const multi = listing({
      tracks: [track(), track({ index: 1 }), track({ index: 2 })],
    });
    expect(streamSummary(undefined, multi, "audio")).toContain("3 tracks");
  });

  it("omits a bitrate the container never declared", () => {
    const summary = streamSummary(
      undefined,
      listing({ tracks: [track({ bitRate: null })] }),
      "audio",
    );
    expect(summary).not.toContain("k ");
    expect(summary).toContain("AC3");
  });

  it("says nothing at all when there is nothing to say", () => {
    expect(streamSummary(undefined, undefined, "audio")).toBeNull();
  });
});
