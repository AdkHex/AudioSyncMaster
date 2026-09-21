#!/usr/bin/env node
// Confirm a release offers an update to every platform the workflow builds.
//
//   node scripts/check-release.cjs v2.11.0
//
// The three platform builds publish to one release, each merging its own
// entry into latest.json. A build that failed, or a merge that lost an
// entry, leaves a release that looks published and that one platform's
// apps are never offered -- and nothing else in the pipeline would say so.
// This runs once every build has finished and names what is missing, and
// the way out: the next push that changes nothing shipped completes the
// release under the same number (scripts/decide-version.cjs).
const { PLATFORMS, releasePlatforms } = require("./decide-version.cjs");

const tag = process.argv[2];
if (!/^v\d+\.\d+\.\d+$/.test(tag || "")) {
  console.error("usage: node scripts/check-release.cjs vMAJOR.MINOR.PATCH");
  process.exit(2);
}

const offered = releasePlatforms(tag);
const missing = PLATFORMS.filter((p) => !offered.includes(p));
if (missing.length === 0) {
  console.log(`${tag} offers an update to every platform: ${PLATFORMS.join(", ")}`);
  process.exit(0);
}
console.log(
  `::error::${tag} has no build for ${missing.join(", ")}: apps on ${missing.length === 1 ? "that platform are" : "those platforms are"} ` +
    `not offered it. Fix the failed build and push; while nothing that ships has changed, the next push ` +
    `is released as ${tag.slice(1)} again and completes it.`,
);
process.exit(1);
