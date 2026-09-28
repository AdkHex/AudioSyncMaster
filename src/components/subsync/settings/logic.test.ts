import { describe, expect, it } from "vitest";

import { COMMON_LANGUAGES, DEFAULT_OPTIONS, LANGUAGES } from "@/lib/subsync/types";

import {
  driftOver,
  formatBytes,
  formatTimestamp,
  fpsPreset,
  fpsTimeRatio,
  languageGroups,
  netflixPreview,
  parseIndexList,
  parseTimestamp,
  syncEnginePatch,
  tonemapEncoderPatch,
  translateEnginePatch,
  twoPointFit,
} from "./logic";

describe("timestamps", () => {
  it("parses the forms people type", () => {
    expect(parseTimestamp("00:01:02.500")).toBe(62.5);
    expect(parseTimestamp("1:02:03,25")).toBeCloseTo(3723.25, 6);
    expect(parseTimestamp("41:40.250")).toBeCloseTo(2500.25, 6);
    expect(parseTimestamp("75.5")).toBe(75.5);
    expect(parseTimestamp("-0:00:01.000")).toBe(-1);
  });

  it("rejects what is not a time", () => {
    for (const bad of ["", "abc", "1:60:00", "00:00:61", "1:2:3:4", "00:01:02.5000"]) {
      expect(parseTimestamp(bad), bad).toBeNull();
    }
  });

  it("formats back to hh:mm:ss.mmm and round-trips", () => {
    expect(formatTimestamp(3723.25)).toBe("01:02:03.250");
    expect(formatTimestamp(0)).toBe("00:00:00.000");
    expect(parseTimestamp(formatTimestamp(2500.123))).toBeCloseTo(2500.123, 6);
  });
});

describe("frame rate", () => {
  it("PAL speed-up: 25 -> 23.976 stretches by 1.0427, 115 s over 45 minutes", () => {
    const ratio = fpsTimeRatio(25, 24000 / 1001)!;
    expect(ratio).toBeCloseTo(1.042708, 6);
    expect(driftOver(ratio)).toBeCloseTo(115.3, 1);
  });

  it("matches presets exactly, so a typed 23.976 stays custom", () => {
    expect(fpsPreset(24000 / 1001)?.label).toBe("23.976");
    expect(fpsPreset(23.976)).toBeNull();
  });

  it("two-point fit gives ratio and offset, and refuses equal points", () => {
    const fit = twoPointFit([0, 1], [2400, 2501]);
    expect(fit.ok && fit.fit.ratio).toBeCloseTo(2500 / 2400, 9);
    expect(fit.ok && fit.fit.offsetS).toBeCloseTo(1, 9);
    expect(twoPointFit([10, 10], [10, 20]).ok).toBe(false);
    expect(twoPointFit([10, 100], [20, 50]).ok).toBe(false);
  });
});

describe("languages", () => {
  it("common first in their order, then all alphabetically", () => {
    const { common, all } = languageGroups();
    expect(common.map(([c]) => c)).toEqual(COMMON_LANGUAGES);
    expect(all).toHaveLength(Object.keys(LANGUAGES).length);
    const names = all.map(([, n]) => n);
    expect(names).toEqual([...names].sort((a, b) => a.localeCompare(b)));
  });
});

describe("option patches", () => {
  it("sync: only switching between reference engines drops the reference", () => {
    const ref = { ...DEFAULT_OPTIONS.sync, engine: "subtitle" as const, reference: { path: "/a.srt" } };
    expect(syncEnginePatch(ref, "reference")).toEqual({ engine: "reference", reference: null });
    expect(syncEnginePatch(ref, "audio")).toEqual({ engine: "audio" });
  });

  it("translate: keeps a model the new service knows, replaces one it does not", () => {
    const opts = DEFAULT_OPTIONS.translate;
    expect(translateEnginePatch(opts, "openai")).toEqual({ engine: "openai", model: "gpt-5" });
    expect(translateEnginePatch({ ...opts, engine: "ollama", model: "x" }, "deepl")).toEqual({ engine: "deepl" });
  });

  it("tonemap: 10-bit falls back to 8 on H.264", () => {
    const opts = { ...DEFAULT_OPTIONS.tonemap, encoder: "x265" as const, bitDepth: 10 as const };
    expect(tonemapEncoderPatch(opts, "x264")).toEqual({ encoder: "x264", bitDepth: 8 });
    expect(tonemapEncoderPatch(opts, "hevc_videotoolbox")).toEqual({ encoder: "hevc_videotoolbox" });
  });
});

describe("misc", () => {
  it("netflix preview per language, children and SDH", () => {
    expect(netflixPreview("en")).toMatchObject({ cpl: 42, cps: 20, exact: true });
    expect(netflixPreview("ja", false, true)).toMatchObject({ cpl: 16, cps: 7 });
    expect(netflixPreview("zh", false, true)).toMatchObject({ cpl: 18, maxLines: 3 });
    expect(netflixPreview("fr", true)).toMatchObject({ cpl: 42, cps: 13, exact: false });
  });

  it("parses track lists and sizes", () => {
    expect(parseIndexList("1, 3 3;5")).toEqual([1, 3, 5]);
    expect(parseIndexList("1, x")).toBeNull();
    expect(formatBytes(3.1e9)).toBe("3.1 GB");
    expect(formatBytes(480e6)).toBe("480 MB");
    expect(formatBytes(null)).toBeNull();
  });
});
