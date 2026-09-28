import { describe, expect, it } from "vitest";

import {
  cueCps,
  formatBytes,
  formatCueTime,
  formatReportValue,
  hdrBadge,
  isLowConfidence,
  reportLabel,
} from "./format";
import type { VideoInfo } from "./types";

const base: VideoInfo = {
  width: 3840, height: 2160, fps: 23.976, codec: "hevc", hdr: "sdr", dvProfile: null, dvCompatibility: null,
  maxCll: null, masteringPeak: null, transfer: null, primaries: null,
};

describe("subsync formatting", () => {
  it("formats cue times", () => {
    expect(formatCueTime(0)).toBe("00:00.000");
    expect(formatCueTime(125.5)).toBe("02:05.500");
    expect(formatCueTime(3725.004)).toBe("1:02:05.004");
    expect(formatCueTime(-1.25)).toBe("-00:01.250");
  });

  it("counts characters per second without markup or line breaks", () => {
    expect(cueCps({ start: 0, end: 2, text: "{\\an8}<i>Hello</i>\\Nthere" })).toBe(5);
    expect(cueCps({ start: 1, end: 1, text: "x" })).toBeNull();
  });

  it("flags cues below 0.6 confidence only", () => {
    expect(isLowConfidence({ start: 0, end: 1, text: "", confidence: 0.59 })).toBe(true);
    expect(isLowConfidence({ start: 0, end: 1, text: "", confidence: 0.6 })).toBe(false);
    expect(isLowConfidence({ start: 0, end: 1, text: "" })).toBe(false);
  });

  it("names the HDR format", () => {
    expect(hdrBadge({ ...base, hdr: "dv", dvProfile: 8, dvCompatibility: 1 })).toBe("DV P8.1");
    expect(hdrBadge({ ...base, hdr: "dv", dvProfile: 5, dvCompatibility: 0 })).toBe("DV P5");
    expect(hdrBadge({ ...base, hdr: "hdr10" })).toBe("HDR10");
    expect(hdrBadge(base)).toBe("SDR");
    expect(hdrBadge(null)).toBeNull();
  });

  it("makes report values readable", () => {
    expect(formatReportValue("offsetMs", 3250)).toBe("+3.250 s");
    expect(formatReportValue("offsetMs", -40)).toBe("-0.040 s");
    expect(formatReportValue("durationS", 12.5)).toBe("12.500 s");
    expect(formatReportValue("ratio", 1.0427083)).toBe("×1.042708");
    expect(formatReportValue("fixed", true)).toBe("Yes");
    expect(formatReportValue("splits", [])).toBe("none");
    expect(formatReportValue("method", "audio")).toBe("audio");
    expect(formatReportValue("x", null)).toBe("—");
  });

  it("labels report keys", () => {
    expect(reportLabel("offsetMs")).toBe("Offset");
    expect(reportLabel("maxCll")).toBe("Max CLL");
    expect(reportLabel("linesFixed")).toBe("Lines fixed");
  });

  it("formats byte counts", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.0 GB");
    expect(formatBytes(null)).toBe("");
  });
});
