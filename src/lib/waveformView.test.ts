import { describe, expect, it } from "vitest";

import type { DubSegment, DubSyncPlan } from "./types";
import { clampView, followCursor, MIN_VIEW_S, peaksKey, tickStep, visibleRequests } from "./waveformView";

function dub(startS: number, endS: number, offsetS: number): DubSegment {
  return { kind: "dub", startS, endS, sourceStartS: startS + offsetS, offsetS, match: 0.5, note: "", uncertaintyS: 0 };
}
function fill(startS: number, endS: number): DubSegment {
  return { kind: "fill", startS, endS, sourceStartS: startS, offsetS: null, match: null, note: "dub is cut here", uncertaintyS: 0 };
}
const plan: DubSyncPlan = {
  videoPath: "/v.mkv",
  dubPath: "/d.mp4",
  videoTrack: 1,
  dubTrack: 2,
  speed: 1.001,
  videoFps: 24,
  dubRate: 23.976,
  fillGainDb: 0,
  videoDurationS: 100,
  dubDurationS: 120,
  // The dub lacks video 40-50: the second stretch is read ten seconds earlier.
  segments: [dub(0, 40, 0.5), fill(40, 50), dub(50, 100, -9.5)],
  warnings: [],
  error: null,
  filledS: 10,
};

describe("tickStep", () => {
  it("spaces the ruler about ninety pixels apart, on round figures", () => {
    expect(tickStep(100, 900)).toBe(10);
    expect(tickStep(5, 1000)).toBe(0.5);
    expect(tickStep(7200, 800)).toBe(900);
  });
});

describe("clampView", () => {
  it("keeps the view inside the film and never narrower than the minimum", () => {
    expect(clampView({ startS: -5, endS: 15 }, 100)).toEqual({ startS: 0, endS: 20 });
    expect(clampView({ startS: 90, endS: 110 }, 100)).toEqual({ startS: 80, endS: 100 });
    expect(clampView({ startS: 10, endS: 10.1 }, 100)).toEqual({ startS: 10, endS: 10 + MIN_VIEW_S });
    expect(clampView({ startS: -50, endS: 500 }, 100)).toEqual({ startS: 0, endS: 100 });
  });
});

describe("visibleRequests", () => {
  it("reads the original across the view and each stretch of dub from where the plan reads it", () => {
    const requests = visibleRequests(plan, { startS: 30, endS: 60 }, 300);
    expect(requests).toHaveLength(3);
    expect(requests[0]).toEqual({ path: "/v.mkv", track: 1, startS: 30, endS: 60, buckets: 300, speed: 1 });
    // Video 30-40 of the first stretch is dub 30.5-40.5, ten of the thirty seconds shown.
    expect(requests[1]).toEqual({ path: "/d.mp4", track: 2, startS: 30.5, endS: 40.5, buckets: 100, speed: 1.001 });
    // Video 50-60 of the second stretch is dub 40.5-50.5.
    expect(requests[2]).toEqual({ path: "/d.mp4", track: 2, startS: 40.5, endS: 50.5, buckets: 100, speed: 1.001 });
  });

  it("reads nothing from the dub for a fill, and nothing for stretches out of view", () => {
    const requests = visibleRequests(plan, { startS: 41, endS: 49 }, 800);
    expect(requests).toHaveLength(1);
    expect(requests[0].path).toBe("/v.mkv");
  });

  it("asks for at least one bucket for a sliver of a stretch", () => {
    const requests = visibleRequests(plan, { startS: 39.999, endS: 49.999 }, 100);
    expect(requests[1].buckets).toBe(1);
  });

  it("returns nothing for an empty view or canvas", () => {
    expect(visibleRequests(plan, { startS: 10, endS: 10 }, 800)).toEqual([]);
    expect(visibleRequests(plan, { startS: 10, endS: 20 }, 0)).toEqual([]);
  });
});

describe("followCursor", () => {
  it("leaves a view alone while the cursor is inside it", () => {
    expect(followCursor({ startS: 90, endS: 110 }, 100, 0.2)).toBeNull();
    expect(followCursor({ startS: 90, endS: 110 }, 90, 0.2)).toBeNull();
  });

  it("jumps so the cursor lands at the given share of the width, same length", () => {
    expect(followCursor({ startS: 90, endS: 110 }, 140, 0.2)).toEqual({ startS: 136, endS: 156 });
    expect(followCursor({ startS: 90, endS: 110 }, 20, 0.2)).toEqual({ startS: 16, endS: 36 });
  });
});

describe("peaksKey", () => {
  it("tells the same span at two speeds apart, and rounds away float noise", () => {
    const a = { path: "/d.mp4", track: 0, startS: 1.0000001, endS: 2, buckets: 10, speed: 1 };
    expect(peaksKey({ ...a, startS: 1 })).toBe(peaksKey(a));
    expect(peaksKey({ ...a, speed: 1.001 })).not.toBe(peaksKey(a));
    expect(peaksKey({ ...a, speed: undefined })).toBe(peaksKey(a));
  });
});
