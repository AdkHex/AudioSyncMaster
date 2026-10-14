#!/usr/bin/env node
// The Windows installer installs for the user who runs it, never for every
// account (RELEASING.md, "Per-user Windows install"). One check per stage:
//
//   node scripts/windows-installer.cjs config                 the repo's Tauri config, before anything is built
//   node scripts/windows-installer.cjs bundle <release dir>   a Windows build's output (src-tauri/target/release)
//   node scripts/windows-installer.cjs release vX.Y.Z         what the published latest.json offers Windows
//
// Tauri's MSI always installs per machine (Program Files, every account), so
// on the shared RDP server one install reached every user. Windows builds
// only the NSIS installer, in currentUser mode, and the updater must offer it.
const fs = require("node:fs");
const path = require("node:path");
const cp = require("node:child_process");

const ROOT = path.join(__dirname, "..");

/** RFC 7386 JSON merge patch: how Tauri lays tauri.windows.conf.json over tauri.conf.json. */
function mergePatch(target, patch) {
  if (patch === null || typeof patch !== "object" || Array.isArray(patch)) return patch;
  const out = target && typeof target === "object" && !Array.isArray(target) ? { ...target } : {};
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) delete out[key];
    else out[key] = mergePatch(out[key], value);
  }
  return out;
}

/** Problems with the config a Windows build uses; none when it is per-user NSIS only. */
function checkConfig(base, windows) {
  const config = mergePatch(base, windows);
  const bundle = config.bundle || {};
  const problems = [];
  const targets = bundle.targets;
  if (!Array.isArray(targets) || targets.length !== 1 || targets[0] !== "nsis") {
    problems.push(`Windows bundle targets are ${JSON.stringify(targets)}, not ["nsis"]: an MSI installs for every account`);
  }
  const mode = bundle.windows?.nsis?.installMode;
  if (mode !== "currentUser") {
    problems.push(`bundle.windows.nsis.installMode is ${JSON.stringify(mode)}, not "currentUser"`);
  }
  const updaterMode = config.plugins?.updater?.windows?.installMode;
  if (updaterMode !== "passive") {
    problems.push(`plugins.updater.windows.installMode is ${JSON.stringify(updaterMode)}, not "passive"`);
  }
  return problems;
}

/** Problems with what a Windows build produced in `releaseDir` (…/target[/<triple>]/release). */
function checkBundle(releaseDir) {
  const problems = [];
  const files = (dir) => (fs.existsSync(dir) ? fs.readdirSync(dir) : []);
  const msis = files(path.join(releaseDir, "bundle", "msi")).filter((f) => f.toLowerCase().endsWith(".msi"));
  if (msis.length) problems.push(`an MSI was built (${msis.join(", ")}); it installs for every account`);
  const setups = files(path.join(releaseDir, "bundle", "nsis")).filter((f) => f.endsWith("-setup.exe"));
  if (!setups.length) problems.push("no NSIS -setup.exe was built");
  const script = path.join(releaseDir, "nsis", "x64", "installer.nsi");
  if (!fs.existsSync(script)) {
    problems.push(`no generated installer script at ${script}`);
  } else {
    const nsi = fs.readFileSync(script, "utf8");
    if (!/^\s*!define INSTALLMODE "currentUser"/m.test(nsi)) problems.push('installer.nsi does not define INSTALLMODE "currentUser"');
    if (!/RequestExecutionLevel user/.test(nsi)) problems.push("installer.nsi does not request user (non-admin) execution");
  }
  return { problems, setups, msis };
}

/** Problems with a release's latest.json as installed copies read it. */
function checkManifest(latest) {
  const platforms = latest?.platforms || {};
  const problems = [];
  const windows = platforms["windows-x86_64"];
  if (!windows) problems.push("latest.json has no windows-x86_64 entry");
  else {
    if (!/-setup\.exe$/.test(windows.url || "")) problems.push(`windows-x86_64 points at ${windows.url}, not the NSIS -setup.exe`);
    if (!windows.signature) problems.push("windows-x86_64 has no signature");
  }
  const msi = Object.keys(platforms).filter((k) => k.startsWith("windows-") && k.endsWith("-msi"));
  if (msi.length) problems.push(`latest.json still offers an MSI (${msi.join(", ")})`);
  return problems;
}

function report(what, problems) {
  if (problems.length === 0) {
    console.log(`${what}: per-user NSIS only`);
    return 0;
  }
  for (const p of problems) console.log(`::error::${what}: ${p}`);
  return 1;
}

if (require.main === module) {
  const [mode, arg] = process.argv.slice(2);
  const read = (file) => JSON.parse(fs.readFileSync(path.join(ROOT, file), "utf8"));
  if (mode === "config") {
    process.exit(report("Windows config", checkConfig(read("src-tauri/tauri.conf.json"), read("src-tauri/tauri.windows.conf.json"))));
  } else if (mode === "bundle" && arg) {
    const { problems, setups } = checkBundle(path.resolve(arg));
    if (setups.length) console.log(`built: ${setups.join(", ")}`);
    process.exit(report("Windows bundle", problems));
  } else if (mode === "release" && /^v\d+\.\d+\.\d+$/.test(arg || "")) {
    const result = cp.spawnSync("gh", ["release", "download", arg, "--pattern", "latest.json", "-O", "-"], { encoding: "utf8" });
    if (result.status !== 0) {
      console.log(`::error::${arg} has no latest.json to check`);
      process.exit(1);
    }
    process.exit(report(`${arg} latest.json`, checkManifest(JSON.parse(result.stdout))));
  } else {
    console.error("usage: node scripts/windows-installer.cjs config | bundle <release dir> | release vMAJOR.MINOR.PATCH");
    process.exit(2);
  }
}

module.exports = { checkBundle, checkConfig, checkManifest, mergePatch };
