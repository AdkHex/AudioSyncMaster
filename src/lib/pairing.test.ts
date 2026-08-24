import { describe, expect, it } from "vitest";

import {
  applyOverrides,
  countExcluded,
  countManualPairs,
  pruneOverrides,
  type PairOverrides,
} from "./pairing";
import type { MatchPair, PairingReport } from "./types";

function pair(video: string, audio: string): MatchPair {
  return {
    primaryPath: `/v/${video}`,
    secondaryPath: `/a/${audio}`,
    primaryName: video,
    secondaryName: audio,
    key: video,
    method: "episode (S01E01)",
    score: 1,
    primaryTrack: 0,
    secondaryTrack: 0,
  };
}

function report(pairs: MatchPair[], overrides: Partial<PairingReport> = {}): PairingReport {
  return {
    pairs,
    unmatchedPrimary: [],
    unmatchedSecondary: [],
    method: "episode (S01E01)",
    patternUsed: null,
    warning: null,
    ...overrides,
  };
}

const AUDIO = [
  { path: "/a/ep01.ac3", name: "ep01.ac3" },
  { path: "/a/ep02.ac3", name: "ep02.ac3" },
  { path: "/a/ep03.ac3", name: "ep03.ac3" },
];

describe("no overrides", () => {
  it("returns the engine's report untouched", () => {
    const original = report([pair("ep01.mkv", "ep01.ac3")]);
    expect(applyOverrides(original, {}, AUDIO)).toBe(original);
  });
});

describe("correcting a pairing", () => {
  it("repoints a video at the audio the user chose", () => {
    // The classic case: the matcher paired ep02 with the wrong dub.
    const original = report([
      pair("ep01.mkv", "ep01.ac3"),
      pair("ep02.mkv", "ep03.ac3"),
    ]);
    const fixed = applyOverrides(
      original,
      { "/v/ep02.mkv": "/a/ep02.ac3" },
      AUDIO,
    );

    const ep02 = fixed.pairs.find((p) => p.primaryName === "ep02.mkv");
    expect(ep02?.secondaryPath).toBe("/a/ep02.ac3");
    expect(ep02?.secondaryName).toBe("ep02.ac3");
    expect(ep02?.method).toBe("chosen by hand");
  });

  it("leaves the other pairs alone", () => {
    const fixed = applyOverrides(
      report([pair("ep01.mkv", "ep01.ac3"), pair("ep02.mkv", "ep03.ac3")]),
      { "/v/ep02.mkv": "/a/ep02.ac3" },
      AUDIO,
    );
    const ep01 = fixed.pairs.find((p) => p.primaryName === "ep01.mkv");
    expect(ep01?.secondaryPath).toBe("/a/ep01.ac3");
    expect(ep01?.method).toBe("episode (S01E01)");
  });

  it("marks the report so the user can see a hand edit was made", () => {
    const fixed = applyOverrides(
      report([pair("ep01.mkv", "ep02.ac3")]),
      { "/v/ep01.mkv": "/a/ep01.ac3" },
      AUDIO,
    );
    expect(fixed.method).toContain("manual");
  });
});

describe("excluding a video", () => {
  it("drops it from the run entirely", () => {
    const fixed = applyOverrides(
      report([pair("ep01.mkv", "ep01.ac3"), pair("ep02.mkv", "ep02.ac3")]),
      { "/v/ep02.mkv": null },
      AUDIO,
    );
    expect(fixed.pairs).toHaveLength(1);
    expect(fixed.pairs[0].primaryName).toBe("ep01.mkv");
  });

  it("warns when everything has been excluded", () => {
    const fixed = applyOverrides(
      report([pair("ep01.mkv", "ep01.ac3")]),
      { "/v/ep01.mkv": null },
      AUDIO,
    );
    expect(fixed.pairs).toHaveLength(0);
    expect(fixed.warning).toMatch(/excluded/i);
  });
});

describe("pairing something the matcher missed", () => {
  it("adds a pair for a video that had none", () => {
    const fixed = applyOverrides(
      report([pair("ep01.mkv", "ep01.ac3")], { unmatchedPrimary: ["ep99.mkv"] }),
      { "/v/ep99.mkv": "/a/ep03.ac3" },
      AUDIO,
    );
    expect(fixed.pairs).toHaveLength(2);
    const added = fixed.pairs.find((p) => p.primaryName === "ep99.mkv");
    expect(added?.secondaryName).toBe("ep03.ac3");
  });

  it("removes it from the unmatched list once paired", () => {
    const fixed = applyOverrides(
      report([], { unmatchedPrimary: ["ep99.mkv"] }),
      { "/v/ep99.mkv": "/a/ep01.ac3" },
      AUDIO,
    );
    expect(fixed.unmatchedPrimary).not.toContain("ep99.mkv");
  });
});

describe("unmatched audio is recomputed", () => {
  it("lists only audio nothing is using", () => {
    const fixed = applyOverrides(
      report([pair("ep01.mkv", "ep01.ac3")]),
      { "/v/ep01.mkv": "/a/ep02.ac3" },
      AUDIO,
    );
    // ep02 is now in use; ep01 and ep03 are not.
    expect(fixed.unmatchedSecondary).toEqual(["ep01.ac3", "ep03.ac3"]);
  });
});

describe("pruning stale overrides", () => {
  it("drops a correction whose video is no longer selected", () => {
    const overrides: PairOverrides = { "/v/gone.mkv": "/a/ep01.ac3" };
    expect(pruneOverrides(overrides, ["/v/ep01.mkv"], ["/a/ep01.ac3"])).toEqual({});
  });

  it("drops a correction pointing at audio that was removed", () => {
    const overrides: PairOverrides = { "/v/ep01.mkv": "/a/gone.ac3" };
    expect(pruneOverrides(overrides, ["/v/ep01.mkv"], ["/a/ep01.ac3"])).toEqual({});
  });

  it("keeps an exclusion as long as its video is still selected", () => {
    const overrides: PairOverrides = { "/v/ep01.mkv": null };
    expect(pruneOverrides(overrides, ["/v/ep01.mkv"], [])).toEqual({
      "/v/ep01.mkv": null,
    });
  });
});

describe("counts", () => {
  it("separates hand-paired from excluded", () => {
    const overrides: PairOverrides = {
      "/v/a.mkv": "/a/1.ac3",
      "/v/b.mkv": null,
      "/v/c.mkv": "/a/2.ac3",
    };
    expect(countManualPairs(overrides)).toBe(2);
    expect(countExcluded(overrides)).toBe(1);
  });
});
