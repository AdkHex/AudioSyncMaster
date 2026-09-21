import { createRequire } from "node:module";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const { PLATFORMS, decide } = require("./decide-version.cjs") as {
  PLATFORMS: string[];
  decide: (facts: {
    repo: string;
    released: string[];
    platforms: (tag: string) => string[];
    changes: (tag: string) => string[] | null;
  }) => { version: string; release: boolean; reason: string };
};

const complete = () => PLATFORMS;
const noWindows = () => PLATFORMS.filter((p) => !p.startsWith("windows"));
const shipped = () => ["src/pages/Index.tsx"];
const nothing = () => [] as string[];
const unknown = () => null;

describe("decide", () => {
  it("releases the repo's version when nothing has been released yet", () => {
    expect(decide({ repo: "1.0.0", released: [], platforms: complete, changes: nothing })).toMatchObject({
      version: "1.0.0",
      release: true,
    });
  });

  it("honours a repo version newer than the latest release", () => {
    const decision = decide({ repo: "3.0.0", released: ["2.10.0", "2.11.0"], platforms: complete, changes: nothing });
    expect(decision).toMatchObject({ version: "3.0.0", release: true });
    expect(decision.reason).toContain("newer than the latest release v2.11.0");
  });

  it("releases the next minor when something shipped has changed", () => {
    const decision = decide({ repo: "2.10.0", released: ["2.9.0", "2.10.0"], platforms: complete, changes: shipped });
    expect(decision).toMatchObject({ version: "2.11.0", release: true });
    expect(decision.reason).toContain("1 shipped file(s) changed");
  });

  it("releases nothing when only unshipped files changed", () => {
    const decision = decide({ repo: "2.10.0", released: ["2.10.0"], platforms: complete, changes: nothing });
    expect(decision).toMatchObject({ version: "2.11.0", release: false });
  });

  it("releases rather than skips when the latest tag cannot be compared against", () => {
    expect(decide({ repo: "2.10.0", released: ["2.10.0"], platforms: complete, changes: unknown })).toMatchObject({
      version: "2.11.0",
      release: true,
    });
  });

  it("orders releases as versions, not as strings", () => {
    // 2.9.0 < 2.10.0: a string sort would call 2.9.0 the latest.
    expect(decide({ repo: "2.10.0", released: ["2.10.0", "2.9.0"], platforms: complete, changes: shipped }).version).toBe(
      "2.11.0",
    );
  });

  // The case that made this rule: v2.11.0 was published by the macOS and
  // Linux builds while the Windows build failed, so Windows apps were never
  // offered it, and the push that fixed the build touched nothing shipped.
  it("completes a release that a platform never got, while nothing shipped has changed", () => {
    const decision = decide({ repo: "2.10.0", released: ["2.10.0", "2.11.0"], platforms: noWindows, changes: nothing });
    expect(decision).toMatchObject({ version: "2.11.0", release: true });
    expect(decision.reason).toContain("no build for windows-x86_64");
    expect(decision.reason).toContain("released as 2.11.0 again");
  });

  it("supersedes an incomplete release once something shipped has changed", () => {
    const decision = decide({ repo: "2.10.0", released: ["2.10.0", "2.11.0"], platforms: noWindows, changes: shipped });
    expect(decision).toMatchObject({ version: "2.12.0", release: true });
    expect(decision.reason).toContain("v2.11.0 has no build for windows-x86_64; it is superseded");
  });

  it("treats a release without a manifest as offered to nobody", () => {
    const decision = decide({ repo: "2.10.0", released: ["2.11.0"], platforms: () => [], changes: nothing });
    expect(decision).toMatchObject({ version: "2.11.0", release: true });
    expect(decision.reason).toContain(`no build for ${PLATFORMS.join(", ")}`);
  });

  it("does not complete a release when it cannot tell what changed since it", () => {
    // Unknown changes: the next minor, as before -- never the same number blind.
    expect(decide({ repo: "2.10.0", released: ["2.11.0"], platforms: noWindows, changes: unknown })).toMatchObject({
      version: "2.12.0",
      release: true,
    });
  });
});
