import { describe, expect, it } from "vitest";

import type { SyncResult } from "@/lib/types";

import { applyFilter, commonTitle, delayText, dubLabel, episodeCode, releaseName, resultState, resultsOutcome, tookText, trackText } from "./views";

const base: SyncResult = {
  videoFile: "a.mkv",
  audioFile: "dub.ac3",
  delayMs: 317,
  delayAtStartMs: 317,
  confidence: 0.94,
  driftMsPerS: null,
  totalDriftMs: null,
  hasSignificantDrift: false,
  startDelayMs: 317,
  endDelayMs: 317,
  windowsUsed: 6,
  windowsTotal: 6,
  error: null,
  elapsedMs: 41200,
};
const r = (patch: Partial<SyncResult>): SyncResult => ({ ...base, ...patch });

describe("result status", () => {
  it("says what needs a look, in the order it matters", () => {
    expect(resultState(base)).toEqual(["ok", "Matched"]);
    expect(resultState(r({ error: "The video has no audio track.", delayMs: null }))).toEqual(["bad", "No audio"]);
    expect(resultState(r({ isLikelyCut: true }))).toEqual(["warn", "Different cut"]);
    expect(resultState(r({ isRateMismatch: true, hasSignificantDrift: true }))).toEqual(["ok", "Matched"]);
    expect(resultState(r({ hasSignificantDrift: true }))).toEqual(["warn", "Drifting"]);
    expect(resultState(r({ confidence: 0.31 }))).toEqual(["warn", "Low confidence"]);
    expect(resultState(r({ confidence: 0.63 }))).toEqual(["warn", "Low confidence"]);
    expect(resultState(r({ confidence: 0.71, isRateMismatch: true, hasSignificantDrift: true }))).toEqual(["warn", "Rate change"]);
  });

  it("sums a run up the way its rows read", () => {
    const run = [base, r({ confidence: 0.63 }), r({ confidence: 0.71, isRateMismatch: true }), r({ error: "x", delayMs: null })];
    expect(resultsOutcome(run).text).toBe("1 matched · 1 low confidence · 1 rate change · 1 failed");
  });
});

describe("figures", () => {
  it("writes delays with a true minus and thousands separators, in the player's sign", () => {
    expect(delayText(317)).toBe("\u2212317.0 ms");
    expect(delayText(-1208)).toBe("+1,208.0 ms");
    expect(delayText(0)).toBe("0.0 ms");
    expect(delayText(-0.01)).toBe("0.0 ms");
    expect(delayText(null)).toBe("—");
    expect(delayText(-5258.3, false)).toBe("+5,258.3");
  });

  it("gives run times in seconds, then minutes and seconds", () => {
    expect(tookText(41200)).toBe("41.2 s");
    expect(tookText(118000)).toBe("1m 58s");
    expect(tookText(null)).toBe("—");
  });

  it("names audio tracks by number, language and format", () => {
    const track = { index: 0, codec: "truehd", language: "eng", title: null, channels: 8, sampleRate: 48000, bitRate: null, isDefault: true, label: "Track 1" };
    expect(trackText(track)).toBe("1 · English · TrueHD 7.1");
    expect(trackText({ ...track, index: 1, codec: "eac3", language: "und", channels: 6 })).toBe("2 · E-AC3 5.1");
  });
});

describe("names", () => {
  it("finds the shared title, with a trailing year in brackets", () => {
    expect(commonTitle(["Parasite.2019.1080p.BluRay.x264.mkv", "Parasite.2019.1080p.WEB-DL.DDP5.1.mkv"])).toBe("Parasite (2019)");
    expect(commonTitle(["Blade Runner 2049 (2017) 1080p.mkv", "Blade Runner 2049 (2017) PAL DVD.mkv"])).toBe("Blade Runner 2049 (2017)");
  });

  it("tells releases and dubs apart by what differs", () => {
    expect(releaseName("Parasite.2019.1080p.WEB-DL.DDP5.1.mkv", "Parasite (2019)")).toBe("WEB-DL");
    expect(releaseName("Parasite.2019.1080p.BluRay.x264.mkv", "Parasite (2019)")).toBe("BluRay");
    const dubs = ["Parasite.2019.Hindi.AC3.ac3", "Parasite.2019.Tamil.AAC.aac", "Parasite.2019.Telugu.E-AC3.eac3"];
    expect(dubs.map((d) => dubLabel(d, dubs))).toEqual(["Hindi AC3", "Tamil AAC", "Telugu E-AC3"]);
  });
});

describe("filters", () => {
  const results = [base, r({ videoFile: "b", confidence: 0.6 }), r({ videoFile: "c", hasSignificantDrift: true }), r({ videoFile: "d", isLikelyCut: true, hasSignificantDrift: true }), r({ videoFile: "e", error: "x", delayMs: null })];
  it("counts each band", () => {
    expect(applyFilter(results, "all")).toHaveLength(5);
    expect(applyFilter(results, "medium").map((x) => x.videoFile)).toEqual(["b"]);
    expect(applyFilter(results, "drift").map((x) => x.videoFile)).toEqual(["c"]);
    expect(applyFilter(results, "cut").map((x) => x.videoFile)).toEqual(["d"]);
    expect(applyFilter(results, "failed").map((x) => x.videoFile)).toEqual(["e"]);
  });
});

describe("episode codes", () => {
  it("reads the engine's key, then the name", () => {
    expect(episodeCode("s01e003")).toBe("S01E03");
    expect(episodeCode(null, "Goblin.S01E11.1080p.mkv")).toBe("S01E11");
    expect(episodeCode(undefined, "Show 2x05.mkv")).toBe("S02E05");
    expect(episodeCode(null, "Movie.1920x1080.mkv")).toBe("—");
    expect(episodeCode(null, "Movie.mkv")).toBe("—");
  });
});
