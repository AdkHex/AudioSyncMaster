import { describe, expect, it } from "vitest";

import type { DubSegment, DubSyncPlan } from "./types";
import {
  centreViewAt,
  clampView,
  DRAG_THRESHOLD_PX,
  followCursor,
  isDrag,
  MIN_VIEW_S,
  OVERVIEW_MIN_BOX_PX,
  overviewBox,
  overviewDragView,
  overviewHit,
  overviewT,
  overviewX,
  panView,
  peaksKey,
  SHOT_BLOCK_S,
  shotBlocks,
  shotCutsIn,
  shotSpanToFetch,
  slipOffsetDelta,
  snapToCuts,
  storeShotCuts,
  tickStep,
  visibleRequests,
  wheelDeltaPx,
  wheelZoomFactor,
  zoomAround,
} from "./waveformView";

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

describe("isDrag", () => {
  it("treats a wobble of a few pixels as a click and anything further as a drag", () => {
    expect(isDrag(0, 0)).toBe(false);
    expect(isDrag(DRAG_THRESHOLD_PX, 0)).toBe(false);
    expect(isDrag(3, 2)).toBe(false);
    expect(isDrag(-DRAG_THRESHOLD_PX - 1, 0)).toBe(true);
    expect(isDrag(3, 3)).toBe(true);
  });
});

describe("panView", () => {
  it("keeps what was under the pointer under it: dragging right shows earlier time", () => {
    // 20 s on 200 px: a pixel is a tenth of a second.
    expect(panView({ startS: 100, endS: 120 }, 50, 200)).toEqual({ startS: 95, endS: 115 });
    expect(panView({ startS: 100, endS: 120 }, -50, 200)).toEqual({ startS: 105, endS: 125 });
  });

  it("leaves clamping to the view", () => {
    expect(panView({ startS: 0, endS: 20 }, 100, 200)).toEqual({ startS: -10, endS: 10 });
  });
});

describe("zoomAround", () => {
  it("keeps the instant under the pointer where it is on screen", () => {
    // 10 s at a quarter of the view; halving the view keeps it at a quarter.
    const next = zoomAround({ startS: 0, endS: 40 }, 10, 0.5, 100);
    expect(next).toEqual({ startS: 5, endS: 25 });
    expect((10 - next.startS) / (next.endS - next.startS)).toBeCloseTo(0.25);
  });

  it("never zooms past the minimum or out past the film, and stays inside it", () => {
    expect(zoomAround({ startS: 10, endS: 11 }, 10.5, 0.01, 100)).toEqual({ startS: 10.25, endS: 10.25 + MIN_VIEW_S });
    expect(zoomAround({ startS: 10, endS: 60 }, 30, 10, 100)).toEqual({ startS: 0, endS: 100 });
    expect(zoomAround({ startS: 80, endS: 100 }, 99, 2, 100)).toEqual({ startS: 60, endS: 100 });
  });
});

describe("wheel", () => {
  it("reads line and page deltas as pixels", () => {
    expect(wheelDeltaPx(3, 0)).toBe(3);
    expect(wheelDeltaPx(3, 1)).toBe(48);
    expect(wheelDeltaPx(1, 2)).toBe(400);
  });

  it("zooms a mouse notch by about a quarter either way, and a pinch smoothly", () => {
    expect(wheelZoomFactor(100)).toBeCloseTo(Math.exp(0.25));
    expect(wheelZoomFactor(-100)).toBeCloseTo(Math.exp(-0.25));
    // A notch reported in lines is capped the same.
    expect(wheelZoomFactor(3, 1)).toBeCloseTo(Math.exp(0.25));
    // Small pinch deltas are small steps, and out and back is where it began.
    expect(wheelZoomFactor(2)).toBeCloseTo(1.0202, 4);
    expect(wheelZoomFactor(4) * wheelZoomFactor(-4)).toBeCloseTo(1);
    expect(wheelZoomFactor(0)).toBe(1);
  });
});

describe("slipOffsetDelta", () => {
  it("moves the offset down by the distance the sound is dragged right, to the millisecond", () => {
    // 10 s on 1000 px: a pixel is 10 ms.
    expect(slipOffsetDelta(12, { startS: 0, endS: 10 }, 1000)).toBe(-0.12);
    expect(slipOffsetDelta(-3, { startS: 50, endS: 60 }, 1000)).toBe(0.03);
    // 1 s on 700 px: a pixel is 1.43 ms, rounded to a whole millisecond.
    expect(slipOffsetDelta(1, { startS: 0, endS: 1 }, 700)).toBe(-0.001);
    expect(Object.is(slipOffsetDelta(0, { startS: 0, endS: 1 }, 700), 0)).toBe(true);
  });
});

describe("overview", () => {
  it("maps the whole film onto the bar and back, inside the film", () => {
    expect(overviewX(50, 100, 400)).toBe(200);
    expect(overviewT(200, 100, 400)).toBe(50);
    expect(overviewT(-20, 100, 400)).toBe(0);
    expect(overviewT(500, 100, 400)).toBe(100);
  });

  it("draws the view as a box, never thinner than the minimum and never off the bar", () => {
    expect(overviewBox({ startS: 25, endS: 50 }, 100, 400)).toEqual({ x0: 100, x1: 200 });
    // A second of a 1.5-hour film is a fraction of a pixel; it is widened about its middle.
    const tiny = overviewBox({ startS: 2700, endS: 2701 }, 5400, 1000);
    expect(tiny.x1 - tiny.x0).toBe(OVERVIEW_MIN_BOX_PX);
    expect((tiny.x0 + tiny.x1) / 2).toBeCloseTo(500.09, 1);
    expect(overviewBox({ startS: 0, endS: 1 }, 5400, 1000)).toEqual({ x0: 0, x1: OVERVIEW_MIN_BOX_PX });
    expect(overviewBox({ startS: 5399, endS: 5400 }, 5400, 1000)).toEqual({ x0: 1000 - OVERVIEW_MIN_BOX_PX, x1: 1000 });
  });

  it("grabs the box within a little slack of it", () => {
    const box = { x0: 100, x1: 200 };
    expect(overviewHit(150, box)).toBe(true);
    expect(overviewHit(98, box)).toBe(true);
    expect(overviewHit(203, box)).toBe(true);
    expect(overviewHit(96, box)).toBe(false);
    expect(overviewHit(204, box)).toBe(false);
  });

  it("centres the view on a click, keeping its length, where the film allows", () => {
    expect(centreViewAt({ startS: 0, endS: 20 }, 50, 100)).toEqual({ startS: 40, endS: 60 });
    expect(centreViewAt({ startS: 0, endS: 20 }, 95, 100)).toEqual({ startS: 80, endS: 100 });
    expect(centreViewAt({ startS: 50, endS: 70 }, 2, 100)).toEqual({ startS: 0, endS: 20 });
  });

  it("moves the box with the pointer at the bar's scale, not the view's", () => {
    // 100 s on 400 px: a pixel of the bar is a quarter of a second.
    expect(overviewDragView({ startS: 10, endS: 20 }, 40, 100, 400)).toEqual({ startS: 20, endS: 30 });
    expect(overviewDragView({ startS: 10, endS: 20 }, -400, 100, 400)).toEqual({ startS: 0, endS: 10 });
    expect(overviewDragView({ startS: 10, endS: 20 }, 400, 100, 400)).toEqual({ startS: 90, endS: 100 });
  });
});

describe("snapToCuts", () => {
  const cuts = [10, 27.215, 30];
  it("lands on the nearest picture cut within the threshold on screen", () => {
    // 10 px a second: 0.5 s is 5 px, inside the default 6.
    expect(snapToCuts(27.7, cuts, 10)).toBe(27.215);
    expect(snapToCuts(9.5, cuts, 10)).toBe(10);
    // Between two cuts in reach (at 4 px a second both are), the nearer wins.
    expect(snapToCuts(28.7, cuts, 4)).toBe(30);
    expect(snapToCuts(28.5, cuts, 4)).toBe(27.215);
  });

  it("leaves the time alone out of reach, and measures reach in pixels", () => {
    expect(snapToCuts(20, cuts, 10)).toBe(20);
    // The same 0.5 s is 50 px zoomed in: too far.
    expect(snapToCuts(27.7, cuts, 100)).toBe(27.7);
    expect(snapToCuts(27.7, cuts, 100, 60)).toBe(27.215);
    expect(snapToCuts(27.7, [], 10)).toBe(27.7);
    expect(snapToCuts(27.7, cuts, 0)).toBe(27.7);
  });

  it("reaches the first and the last cut from outside them", () => {
    expect(snapToCuts(9.6, cuts, 10)).toBe(10);
    expect(snapToCuts(30.4, cuts, 10)).toBe(30);
  });
});

describe("picture cut blocks", () => {
  const duration = 200;

  it("covers the view with round blocks, inside the film", () => {
    expect(SHOT_BLOCK_S).toBe(60);
    expect(shotBlocks({ startS: 70, endS: 130 }, duration)).toEqual([60, 120]);
    expect(shotBlocks({ startS: 150, endS: 260 }, duration)).toEqual([120, 180]);
    expect(shotBlocks({ startS: 0, endS: 60 }, duration)).toEqual([0]);
  });

  it("asks for the blocks not read yet, as one span, and nothing once they are", () => {
    let known = new Map<number, number[] | null>();
    const view = { startS: 70, endS: 190 };
    expect(shotSpanToFetch(known, view, duration)).toEqual({ startS: 60, endS: 200 });
    known = storeShotCuts(known, { cuts: [65, 119.9, 120, 185], startS: 60, endS: 200 }, duration);
    expect(known.get(60)).toEqual([65, 119.9]);
    expect(known.get(120)).toEqual([120]);
    expect(known.get(180)).toEqual([185]);
    expect(shotSpanToFetch(known, view, duration)).toBeNull();
    // Panning back needs only the block brought into sight.
    expect(shotSpanToFetch(known, { startS: 30, endS: 100 }, duration)).toEqual({ startS: 0, endS: 60 });
  });

  it("files a picture that cannot be read as known, so it is not asked again", () => {
    const known = storeShotCuts(new Map(), { cuts: null, startS: 0, endS: 120 }, duration);
    expect(known.get(0)).toBeNull();
    expect(shotSpanToFetch(known, { startS: 0, endS: 120 }, duration)).toBeNull();
    expect(shotCutsIn(known, { startS: 0, endS: 120 }, duration)).toEqual([]);
  });

  it("files only the blocks an answer covers whole", () => {
    const known = storeShotCuts(new Map(), { cuts: [5, 70], startS: 0, endS: 90 }, duration);
    expect([...known.keys()]).toEqual([0]);
  });

  it("gives the cuts in the view, in order", () => {
    const known = storeShotCuts(new Map(), { cuts: [5, 65, 119, 130], startS: 0, endS: 180 }, duration);
    expect(shotCutsIn(known, { startS: 60, endS: 125 }, duration)).toEqual([65, 119]);
  });
});
