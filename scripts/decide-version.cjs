#!/usr/bin/env node
// Decide which version a push to main is released as, and whether it is
// released at all. Prints GitHub Actions outputs; see RELEASING.md.
//
// The rule: the repo's version when it is newer than the latest release (you
// chose the number); otherwise the next minor after the latest release, and
// then only if something that ships has changed since it. So an ordinary
// push is released as the next version without anyone touching the number,
// a push that only touches docs, tests or the workflow is not released, and
// a number you set by hand is used as it is.
const cp = require("child_process");
const fs = require("fs");

const repo = JSON.parse(fs.readFileSync("package.json", "utf8")).version;
const parse = (v) => v.split(".").map(Number);
const compare = (a, b) => {
  const x = parse(a), y = parse(b);
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return 0;
};
const released = JSON.parse(cp.execSync("gh release list --limit 200 --json tagName", { encoding: "utf8" }))
  .map((r) => r.tagName)
  .filter((t) => /^v\d+\.\d+\.\d+$/.test(t))
  .map((t) => t.slice(1))
  .sort(compare);
const latest = released[released.length - 1];

// Paths that do not ship: a change there alone is not a release.
const unshipped = ["*.md", "tests", ".github", "scripts", "dev.sh", "dev.bat", "eslint.config.js", "nimbalyst-local"];
function tagExists(tag) {
  return cp.spawnSync("git", ["rev-parse", "-q", "--verify", `${tag}^{commit}`]).status === 0;
}
function shippedChanges(sinceTag) {
  if (!tagExists(sinceTag)) cp.spawnSync("git", ["fetch", "--tags", "--quiet", "origin"]);
  if (!tagExists(sinceTag)) return null; // unknown: released rather than silently skipped
  const args = ["diff", "--name-only", sinceTag, "HEAD", "--", ".", ...unshipped.map((p) => `:(exclude)${p}`)];
  return cp.execFileSync("git", args, { encoding: "utf8" }).trim().split("\n").filter(Boolean);
}

let version, release, reason;
if (!latest) {
  version = repo; release = true;
  reason = `no release exists yet, so the repo's ${repo} is released`;
} else if (compare(repo, latest) > 0) {
  version = repo; release = true;
  reason = `the repo says ${repo}, newer than the latest release v${latest}, so that number is released`;
} else {
  const [major, minor] = parse(latest);
  version = `${major}.${minor + 1}.0`;
  const changed = shippedChanges(`v${latest}`);
  release = changed === null || changed.length > 0;
  reason = changed === null
    ? `the repo's ${repo} is already released (latest v${latest}); the tag could not be fetched to compare, so this push is released as ${version}`
    : release
      ? `the repo's ${repo} is already released (latest v${latest}); ${changed.length} shipped file(s) changed since, so this push is released as ${version}`
      : `the repo's ${repo} is already released (latest v${latest}) and nothing that ships has changed since, so there is nothing to release`;
}

const out = process.env.GITHUB_OUTPUT;
const lines = [`value=${version}`, `release=${release}`, `reason=${reason}`];
if (out) fs.appendFileSync(out, lines.join("\n") + "\n");
console.log(lines.join("\n"));
