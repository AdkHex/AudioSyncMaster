#!/usr/bin/env node
// Write one version into every file that carries it.
//
//   node scripts/set-version.cjs 2.11.0
//
// CI runs this before every release build, so the number in the repo need
// not be bumped by hand (see RELEASING.md). Run it yourself only to choose a
// specific number -- a major bump, say -- which CI then honours because it is
// newer than the latest release.
//
// The five places, and why each one: package.json is what CI tags from and
// what the frontend shows in Settings > About; tauri.conf.json is what the
// app compares against latest.json when it looks for updates; Cargo.toml is
// the crate; and both lockfiles record the root package's version, so `npm
// ci` and a locked cargo build refuse a mismatch.
const fs = require("fs");
const path = require("path");

const version = process.argv[2];
if (!/^\d+\.\d+\.\d+$/.test(version || "")) {
  console.error("usage: node scripts/set-version.cjs MAJOR.MINOR.PATCH");
  process.exit(2);
}
const root = path.resolve(__dirname, "..");
const file = (name) => path.join(root, name);

function json(name, edit) {
  const text = fs.readFileSync(file(name), "utf8");
  const data = JSON.parse(text);
  edit(data);
  // Keep the file's own indentation and trailing newline.
  const indent = (text.match(/^(\s+)"/m) || [, "  "])[1];
  fs.writeFileSync(file(name), JSON.stringify(data, null, indent) + (text.endsWith("\n") ? "\n" : ""));
}

json("package.json", (p) => { p.version = version; });
json("package-lock.json", (l) => {
  l.version = version;
  if (l.packages && l.packages[""]) l.packages[""].version = version;
});
json("src-tauri/tauri.conf.json", (c) => { c.version = version; });

const cargo = fs.readFileSync(file("src-tauri/Cargo.toml"), "utf8");
const crate = (cargo.match(/^\[package\][^[]*?^name\s*=\s*"([^"]+)"/ms) || [])[1];
if (!crate) throw new Error("src-tauri/Cargo.toml: no [package] name");
fs.writeFileSync(
  file("src-tauri/Cargo.toml"),
  cargo.replace(/^(\[package\][^[]*?^version\s*=\s*")[^"]+(")/ms, `$1${version}$2`),
);
const lock = fs.readFileSync(file("src-tauri/Cargo.lock"), "utf8");
const entry = new RegExp(`(\\[\\[package\\]\\]\\nname = "${crate}"\\nversion = ")[^"]+(")`);
if (!entry.test(lock)) throw new Error(`src-tauri/Cargo.lock: no entry for ${crate}`);
fs.writeFileSync(file("src-tauri/Cargo.lock"), lock.replace(entry, `$1${version}$2`));

console.log(`version ${version} written to package.json, package-lock.json, tauri.conf.json, Cargo.toml, Cargo.lock`);
