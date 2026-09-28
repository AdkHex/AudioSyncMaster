import { describe, expect, it } from "vitest";

import { DEFAULT_OPTIONS } from "./types";
import { inputShape, needsVideo, toolOf, useAsTargets, workersFor } from "./tools";

describe("subsync tools", () => {
  it("groups convert, extract and mux under one rail entry", () => {
    expect(toolOf("extract")).toBe("formats");
    expect(toolOf("sync")).toBe("sync");
    expect(inputShape("mux")).toBe("mux");
    expect(inputShape("tonemap")).toBe("video");
    expect(inputShape("style")).toBe("subtitle");
  });

  it("runs heavy tasks one at a time", () => {
    expect(workersFor("generate")).toBe(1);
    expect(workersFor("tonemap")).toBe(1);
    expect(workersFor("sync")).toBe(2);
  });

  it("needs a video for audio sync only", () => {
    expect(needsVideo("sync", DEFAULT_OPTIONS.sync)).toBe(true);
    expect(needsVideo("sync", { ...DEFAULT_OPTIONS.sync, engine: "subtitle" })).toBe(false);
    expect(needsVideo("translate", DEFAULT_OPTIONS.translate)).toBe(false);
  });

  it("offers each result to the tools that can read it", () => {
    expect(useAsTargets({ kind: "subtitle", format: "srt", path: "/a.srt" })).toContain("translate");
    expect(useAsTargets({ kind: "subtitle", format: null, path: "/a.sup" })).toEqual(["ocr", "hdrSubs", "mux"]);
    expect(useAsTargets({ kind: "video", format: "mkv", path: "/a.mkv" })).toContain("tonemap");
    expect(useAsTargets({ kind: "report", format: "json", path: "/a.json" })).toEqual([]);
  });
});
