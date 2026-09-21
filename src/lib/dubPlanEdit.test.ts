import { describe, expect, it } from "vitest";

import {
  HAND_NOTE,
  MIN_PIECE_S,
  asLoadedPlan,
  canMerge,
  codecOfPath,
  convertToDub,
  convertToFill,
  dubSpanS,
  frameSeconds,
  mergeWithNext,
  moveBoundary,
  nudgeOffset,
  samePlan,
  segmentAt,
  setOffset,
  splitAt,
  suggestedOffset,
  validatePlan,
} from "./dubPlanEdit";
import type { DubSegment, DubSyncPlan } from "./types";

function dub(startS: number, endS: number, offsetS: number, note = ""): DubSegment {
  return { kind: "dub", startS, endS, sourceStartS: startS + offsetS, offsetS, match: 0.5, note, uncertaintyS: 0 };
}
function fill(startS: number, endS: number, note = "dub is cut here"): DubSegment {
  return { kind: "fill", startS, endS, sourceStartS: startS, offsetS: null, match: null, note, uncertaintyS: 0 };
}
function plan(segments: DubSegment[], extra: Partial<DubSyncPlan> = {}): DubSyncPlan {
  return {
    videoPath: "/v.mkv",
    dubPath: "/d.mp4",
    videoTrack: 0,
    dubTrack: 0,
    speed: 1,
    videoFps: 24,
    dubRate: 24,
    fillGainDb: 0,
    videoDurationS: 100,
    dubDurationS: 120,
    segments,
    warnings: [],
    notes: [],
    error: null,
    filledS: segments.filter((s) => s.kind === "fill").reduce((n, s) => n + s.endS - s.startS, 0),
    ...extra,
  };
}

// A plan with a cut: the dub lacks video 40-50, so the second stretch sits ten seconds earlier.
const base = plan([dub(0, 40, 2), fill(40, 50), dub(50, 100, -8)]);

describe("moveBoundary", () => {
  it("moves the cut and keeps both pieces' offsets", () => {
    const moved = moveBoundary(base, 0, 38);
    expect(moved.segments[0].endS).toBe(38);
    expect(moved.segments[1].startS).toBe(38);
    expect(moved.segments[1].sourceStartS).toBe(38); // a fill reads the original in place
    expect(moved.segments[0].offsetS).toBe(2);
    expect(moved.filledS).toBeCloseTo(12);
    const dubBoundary = moveBoundary(base, 1, 52);
    expect(dubBoundary.segments[2].startS).toBe(52);
    expect(dubBoundary.segments[2].sourceStartS).toBe(44); // 52 + (-8): the offset is kept
    expect(dubBoundary.segments[2].offsetS).toBe(-8);
  });

  it("marks both pieces as set by hand", () => {
    const moved = moveBoundary(base, 0, 38);
    expect(moved.segments[0].note).toBe(HAND_NOTE);
    expect(moved.segments[1].note).toBe(HAND_NOTE);
    expect(moved.segments[0].match).toBeNull();
  });

  it("clamps so neither piece vanishes", () => {
    expect(moveBoundary(base, 0, 0).segments[0].endS).toBe(MIN_PIECE_S);
    expect(moveBoundary(base, 0, 49.99).segments[1].startS).toBe(50 - MIN_PIECE_S);
  });

  it("cannot move the end of the plan, and ignores a boundary that does not exist", () => {
    expect(moveBoundary(base, 2, 90)).toBe(base);
    expect(moveBoundary(base, -1, 10)).toBe(base);
  });

  it("leaves the plan untouched when nothing changes", () => {
    expect(moveBoundary(base, 0, 40)).toBe(base);
  });
});

describe("offsets", () => {
  it("setOffset moves where a dub piece reads from and nothing else", () => {
    const changed = setOffset(base, 2, -8.042);
    expect(changed.segments[2].offsetS).toBe(-8.042);
    expect(changed.segments[2].sourceStartS).toBe(41.958);
    expect(changed.segments[2].startS).toBe(50);
    expect(changed.segments[2].endS).toBe(100);
    expect(changed.segments[2].note).toBe(HAND_NOTE);
  });

  it("nudgeOffset adds to the offset", () => {
    const frame = frameSeconds(base);
    expect(frame).toBeCloseTo(1 / 24);
    const nudged = nudgeOffset(base, 0, frame);
    expect(nudged.segments[0].offsetS).toBeCloseTo(2 + 1 / 24, 3);
  });

  it("refuses to read the dub before its start or past its end", () => {
    expect(setOffset(base, 0, -5).segments[0].offsetS).toBe(0); // startS 0: cannot read before dub 0
    const span = dubSpanS(base);
    expect(span).toBe(120);
    expect(setOffset(base, 2, 500).segments[2].offsetS).toBe(20); // endS 100 + 20 = 120, the dub's end
  });

  it("does nothing to a fill", () => {
    expect(setOffset(base, 1, 3)).toBe(base);
    expect(nudgeOffset(base, 1, 3)).toBe(base);
  });

  it("dubSpanS stretches the dub's length by the speed it is played at", () => {
    expect(dubSpanS({ dubDurationS: 1001, speed: 1000 / 1001 })).toBeCloseTo(1000);
  });
});

describe("split and merge", () => {
  it("splits the piece under the cursor into two continuous halves", () => {
    const split = splitAt(base, 60);
    expect(split.segments).toHaveLength(4);
    expect(split.segments[2]).toMatchObject({ startS: 50, endS: 60, offsetS: -8, sourceStartS: 42 });
    expect(split.segments[3]).toMatchObject({ startS: 60, endS: 100, offsetS: -8, sourceStartS: 52 });
    expect(canMerge(split.segments[2], split.segments[3])).toBe(true);
    expect(samePlan(mergeWithNext(split, 2), base)).toBe(true);
  });

  it("will not leave a sliver", () => {
    expect(splitAt(base, 50.05)).toBe(base);
    expect(splitAt(base, 99.95)).toBe(base);
  });

  it("only merges continuous pieces", () => {
    expect(canMerge(base.segments[0], base.segments[1])).toBe(false);
    const twoOffsets = plan([dub(0, 50, 2), dub(50, 100, 2.5)]);
    expect(mergeWithNext(twoOffsets, 0)).toBe(twoOffsets);
    const fills = plan([fill(0, 50), fill(50, 100)]);
    expect(mergeWithNext(fills, 0).segments).toHaveLength(1);
  });

  it("segmentAt finds the piece, including at the very end", () => {
    expect(segmentAt(base, 0)).toBe(0);
    expect(segmentAt(base, 45)).toBe(1);
    expect(segmentAt(base, 100)).toBe(2);
    expect(segmentAt(base, 101)).toBeNull();
  });
});

describe("kind changes", () => {
  it("a fill becomes dub at the neighbour's offset, and can go back", () => {
    expect(suggestedOffset(base, 1)).toBe(2);
    const asDub = convertToDub(base, 1);
    expect(asDub.segments[1]).toMatchObject({ kind: "dub", offsetS: 2, sourceStartS: 42, note: HAND_NOTE });
    expect(asDub.filledS).toBe(0);
    const back = convertToFill(asDub, 1);
    expect(back.segments[1]).toMatchObject({ kind: "fill", sourceStartS: 40, offsetS: null });
    expect(back.filledS).toBe(10);
  });

  it("takes an explicit offset for a new dub piece", () => {
    expect(convertToDub(base, 1, -8).segments[1].sourceStartS).toBe(32);
  });

  it("is a no-op on a piece already of that kind", () => {
    expect(convertToDub(base, 0)).toBe(base);
    expect(convertToFill(base, 1)).toBe(base);
  });
});

describe("validatePlan", () => {
  it("accepts a plan the engine made and everything the editor makes of it", () => {
    expect(validatePlan(base)).toEqual([]);
    let edited = moveBoundary(base, 0, 38.5);
    edited = splitAt(edited, 70);
    edited = nudgeOffset(edited, 3, 0.05);
    edited = convertToDub(edited, 1);
    expect(validatePlan(edited)).toEqual([]);
  });

  it("names what is wrong", () => {
    const gap = plan([dub(0, 40, 2), dub(41, 100, -8)]);
    expect(validatePlan(gap)).toEqual(["Piece 2 does not start where piece 1 ends."]);
    const short = plan([dub(0, 100, 2)], { videoDurationS: 120 });
    expect(validatePlan(short)).toEqual(["The last piece does not end where the video ends."]);
    const beyond = plan([dub(0, 100, 30)]);
    expect(validatePlan(beyond)).toEqual(["Piece 1 reads the dub past its end."]);
    expect(validatePlan(plan([]))).toEqual(["The plan has no pieces."]);
  });
});

describe("codecOfPath", () => {
  it("reads the codec a track was written in off its extension", () => {
    expect(codecOfPath("/v/film.dubsynced.flac")).toBe("flac");
    expect(codecOfPath("/v/film.dubsynced.M4A")).toBe("aac");
    expect(codecOfPath("/v/film.dubsynced.ec3")).toBe("eac3");
    expect(codecOfPath("/v/film.dubsynced.ac3")).toBe("ac3");
    expect(codecOfPath("/v/film.dubsynced.opus")).toBe("opus");
    expect(codecOfPath("/v/film.dubsynced.wav")).toBe("wav");
    expect(codecOfPath("/v/film.dubsynced.mka")).toBeNull();
    expect(codecOfPath("/v/film")).toBeNull();
  });
});

describe("asLoadedPlan", () => {
  it("lays the dub at the video's start and fills the tail it does not reach", () => {
    const loaded = asLoadedPlan({ videoPath: "/v.mkv", dubPath: "/d.mp4", videoTrack: 0, dubTrack: 1, videoDurationS: 100, dubDurationS: 90 });
    expect(loaded.segments).toEqual([
      { kind: "dub", startS: 0, endS: 90, sourceStartS: 0, offsetS: 0, match: null, note: "as loaded", uncertaintyS: 0 },
      { kind: "fill", startS: 90, endS: 100, sourceStartS: 90, offsetS: null, match: null, note: "past dub end", uncertaintyS: 0 },
    ]);
    expect(loaded.speed).toBe(1);
    expect(loaded.filledS).toBe(10);
    expect(loaded.dubTrack).toBe(1);
  });

  it("covers the whole video when the dub is longer", () => {
    const loaded = asLoadedPlan({ videoPath: "/v.mkv", dubPath: "/d.mp4", videoTrack: 0, dubTrack: 0, videoDurationS: 100, dubDurationS: 120 });
    expect(loaded.segments).toHaveLength(1);
    expect(loaded.segments[0]).toMatchObject({ kind: "dub", startS: 0, endS: 100 });
    expect(validatePlan(loaded)).toEqual([]);
  });
});
