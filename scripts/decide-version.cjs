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
//
// One more case: the three platform builds publish to one release, so a
// build that fails leaves a release the other platforms got and that one
// never will -- its apps are simply never offered it. Such a release is
// completed rather than moved past: while nothing that ships has changed
// since it, the next push is released under the same number again, and
// the missing build joins it (tauri-action adds to an existing release).
// Once something shipped has changed, the next minor supersedes it.
const cp = require("child_process");
const fs = require("fs");

// Every release must offer an update to each of these: one per build in
// the workflow's matrix (build-release.yml), as tauri-action names them
// in latest.json.
const PLATFORMS = ["darwin-aarch64", "linux-x86_64", "windows-x86_64"];

// Paths that do not ship: a change there alone is not a release.
const unshipped = ["*.md", "tests", ".github", "scripts", "dev.sh", "dev.bat", "eslint.config.js", "nimbalyst-local"];

const parse = (v) => v.split(".").map(Number);
const compare = (a, b) => {
  const x = parse(a), y = parse(b);
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return 0;
};

/** Released versions, from the repository's releases. */
function releasedVersions() {
  return JSON.parse(cp.execSync("gh release list --limit 200 --json tagName", { encoding: "utf8" }))
    .map((r) => r.tagName)
    .filter((t) => /^v\d+\.\d+\.\d+$/.test(t))
    .map((t) => t.slice(1));
}

/** The platforms a release offers an update for: the keys of its
 *  latest.json. None when it has no manifest at all -- then nobody is
 *  offered it. */
function releasePlatforms(tag) {
  const result = cp.spawnSync("gh", ["release", "download", tag, "--pattern", "latest.json", "-O", "-"], {
    encoding: "utf8",
  });
  if (result.status !== 0) return [];
  try {
    return Object.keys(JSON.parse(result.stdout).platforms || {});
  } catch {
    return [];
  }
}

function tagExists(tag) {
  return cp.spawnSync("git", ["rev-parse", "-q", "--verify", `${tag}^{commit}`]).status === 0;
}

/** Shipped files changed since a tag, or null when the tag cannot be found
 *  (then the push is released rather than silently skipped). */
function shippedChanges(sinceTag) {
  if (!tagExists(sinceTag)) cp.spawnSync("git", ["fetch", "--tags", "--quiet", "origin"]);
  if (!tagExists(sinceTag)) return null;
  const args = ["diff", "--name-only", sinceTag, "HEAD", "--", ".", ...unshipped.map((p) => `:(exclude)${p}`)];
  return cp.execFileSync("git", args, { encoding: "utf8" }).trim().split("\n").filter(Boolean);
}

/** The decision itself, on facts gathered by the callers above (or a test):
 *  repo is package.json's version, released the versions released so far,
 *  platforms(tag) what a release offers, changes(tag) the shipped files
 *  changed since it (null when unknown). */
function decide({ repo, released, platforms, changes }) {
  // As versions, not as strings: 2.9.0 comes before 2.10.0.
  const latest = [...released].sort(compare).pop();
  if (!latest) {
    return { version: repo, release: true, reason: `no release exists yet, so the repo's ${repo} is released` };
  }
  if (compare(repo, latest) > 0) {
    return {
      version: repo,
      release: true,
      reason: `the repo says ${repo}, newer than the latest release v${latest}, so that number is released`,
    };
  }
  const tag = `v${latest}`;
  const changed = changes(tag);
  const offered = platforms(tag);
  const missing = PLATFORMS.filter((p) => !offered.includes(p));
  if (missing.length > 0 && changed !== null && changed.length === 0) {
    return {
      version: latest,
      release: true,
      reason:
        `the latest release ${tag} has no build for ${missing.join(", ")} and nothing that ships has changed ` +
        `since, so this push is released as ${latest} again to complete it`,
    };
  }
  const [major, minor] = parse(latest);
  const version = `${major}.${minor + 1}.0`;
  const incomplete = missing.length > 0 ? ` (${tag} has no build for ${missing.join(", ")}; it is superseded)` : "";
  if (changed === null) {
    return {
      version,
      release: true,
      reason:
        `the repo's ${repo} is already released (latest ${tag}); the tag could not be fetched to compare, ` +
        `so this push is released as ${version}${incomplete}`,
    };
  }
  if (changed.length > 0) {
    return {
      version,
      release: true,
      reason:
        `the repo's ${repo} is already released (latest ${tag}); ${changed.length} shipped file(s) changed since, ` +
        `so this push is released as ${version}${incomplete}`,
    };
  }
  return {
    version,
    release: false,
    reason: `the repo's ${repo} is already released (latest ${tag}) and nothing that ships has changed since, so there is nothing to release`,
  };
}

module.exports = { PLATFORMS, decide, releasePlatforms, releasedVersions };

if (require.main === module) {
  const repo = JSON.parse(fs.readFileSync("package.json", "utf8")).version;
  const { version, release, reason } = decide({
    repo,
    released: releasedVersions(),
    platforms: releasePlatforms,
    changes: shippedChanges,
  });
  const out = process.env.GITHUB_OUTPUT;
  const lines = [`value=${version}`, `release=${release}`, `reason=${reason}`];
  if (out) fs.appendFileSync(out, lines.join("\n") + "\n");
  console.log(lines.join("\n"));
}
