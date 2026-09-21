import { describe, expect, it } from "vitest";

import {
  DRIFT_S,
  frameIndex,
  inWindow,
  LOOP_S,
  loopAround,
  needsResync,
  stepFrame,
  timecode,
  WINDOW_AFTER_S,
  WINDOW_BEFORE_S,
  windowAround,
} from "./previewClock";

describe("windowAround", () => {
  it("takes ten seconds before the cursor and twenty after", () => {
    expect(windowAround(1000, 7200)).toEqual({
      startS: 990,
      endS: 1020,
    });
  });

  it("clamps against the film's start", () => {
    expect(windowAround(5, 7200)).toEqual({ startS: 0, endS: 30 });
  });

  it("clamps against the film's end", () => {
    expect(windowAround(7198, 7200)).toEqual({ startS: 7170, endS: 7200 });
  });

  it("covers a film shorter than the window in full", () => {
    expect(windowAround(10, 12)).toEqual({ startS: 0, endS: 12 });
  });
});

describe("loopAround", () => {
  it("is four seconds centred on the point where it was switched on", () => {
    expect(loopAround(1000, 990, 1020)).toEqual({ a: 998, b: 1002 });
  });

  it("clamps to the window at both ends", () => {
    expect(loopAround(991, 990, 1020)).toEqual({ a: 990, b: 994 });
    expect(loopAround(1019, 990, 1020)).toEqual({ a: 1016, b: 1020 });
  });

  it("loops the whole of a window shorter than the loop", () => {
    expect(loopAround(10, 10, 12)).toEqual({ a: 10, b: 12 });
  });
});

describe("frameIndex", () => {
  it("counts frames from the film's start", () => {
    expect(frameIndex(1.5, 1 / 24)).toBe(36);
    expect(frameIndex(0, 1 / 24)).toBe(0);
  });

  it("does not round a frame's exact position onto the next frame", () => {
    expect(frameIndex(1.0 - 1e-9, 1 / 24)).toBe(23);
    expect(frameIndex(1.0, 1 / 24)).toBe(24);
  });
});

describe("timecode", () => {
  it("prints h:mm:ss:ff with the frame zero-padded", () => {
    expect(timecode(0, 1 / 24)).toBe("0:00:00:00");
    expect(timecode(1.5, 1 / 24)).toBe("0:00:01:12");
    expect(timecode(3723.25, 1 / 25)).toBe("1:02:03:06");
  });

  it("stays on frame 23 at the last instant of a second at 24 fps", () => {
    expect(timecode(1.0 - 1e-9, 1 / 24)).toBe("0:00:01:23");
  });

  it("counts 0..29 at 29.97 fps", () => {
    expect(timecode(1.0 - 1e-9, 1 / 29.97)).toBe("0:00:01:29");
    expect(timecode(0.5, 1 / 29.97)).toBe("0:00:00:14");
  });
});

describe("stepFrame", () => {
  it("steps one frame in either direction", () => {
    expect(stepFrame(10, 1 / 24, 1, 0, 30)).toBeCloseTo(10 + 1 / 24, 12);
    expect(stepFrame(10, 1 / 24, -1, 0, 30)).toBeCloseTo(10 - 1 / 24, 12);
  });

  it("stays put past the window's ends", () => {
    expect(stepFrame(0, 1 / 24, -1, 0, 30)).toBe(0);
    expect(stepFrame(30, 1 / 24, 1, 0, 30)).toBe(30);
  });

  it("lands on exact frames even from a mid-frame time", () => {
    expect(stepFrame(10.0001, 1 / 24, 1, 0, 30)).toBeCloseTo(10 + 1 / 24, 12);
  });
});

describe("needsResync", () => {
  it("is quiet inside 20 ms and loud beyond it", () => {
    expect(needsResync(10, 10.019)).toBe(false);
    expect(needsResync(10, 10.021)).toBe(true);
    expect(needsResync(10, 10)).toBe(false);
  });

  it("uses the named threshold", () => {
    expect(needsResync(10, 10 + DRIFT_S + 1e-6)).toBe(true);
  });
});

describe("inWindow", () => {
  it("includes both ends", () => {
    expect(inWindow(990, 990, 1020)).toBe(true);
    expect(inWindow(1020, 990, 1020)).toBe(true);
    expect(inWindow(989.999, 990, 1020)).toBe(false);
  });

  it("exposes the window constants", () => {
    expect(WINDOW_BEFORE_S).toBe(10);
    expect(WINDOW_AFTER_S).toBe(20);
    expect(LOOP_S).toBe(4);
  });
});
