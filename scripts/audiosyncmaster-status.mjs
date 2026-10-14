#!/usr/bin/env node
// Remote switch and access password for installed copies of AudioSyncMaster
// (docs/REMOTE-STATUS.md).
//
//   node scripts/audiosyncmaster-status.mjs status                print what the published document says
//   node scripts/audiosyncmaster-status.mjs maintenance ["message"] [--title t] [--check-minutes n]
//   node scripts/audiosyncmaster-status.mjs ok [--min-version x] [--update-url u]
//   node scripts/audiosyncmaster-status.mjs password             set or change the access password (asked twice, hidden)
//   node scripts/audiosyncmaster-status.mjs password --off       no password
//   node scripts/audiosyncmaster-status.mjs serve --serve 8799   serve the local file for a test run
//
// Every change starts from the document as it is and changes only what it is
// asked to: maintenance and ok keep the password, password keeps the status.
// It is written to --out (default ./remote-status/audiosyncmaster-status.json,
// gitignored) and published to the status gist, which every installed copy
// reads within checkMinutes plus up to 5 min of GitHub caching.
//
// Options:
//   --title t             maintenance screen title (default "Work is in progress")
//   --check-minutes n     how often the apps check, 1..60 (default: as it is, else 5)
//   --min-version x       ok / maintenance: builds older than x show "Update required"
//   --update-url u        the "Download the new version" link
//   --off                 password: remove it, so nobody is asked
//   --out <file>          the local copy (default ./remote-status/audiosyncmaster-status.json)
//   --no-publish          local only: start from and write the --out file, never touch the gist
//   --serve <port>        serve the --out file on http://127.0.0.1:<port>/status.json for a debug
//                         build (AUDIOSYNC_STATUS_URL); implies --no-publish unless --publish is given
//   --publish             publish even with --serve
//
// password reads the password from the terminal without echo, or, piped, the
// first line of stdin (printf '%s\n' "$PW" | node ... password). It is hashed
// with PBKDF2-SHA256 (600 000 rounds, random 16-byte salt) and never printed,
// written or sent anywhere: only the salt, the hash and its id are published.
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { execFileSync } from "node:child_process";

// ---- where the published document lives (the URL baked into src-tauri/remote-status.json)
const GIST_ID = "1cf1fb34cc1a7fcabc97a1cef5f7bd7a";
const GIST_FILE = "audiosyncmaster-status.json";
const GIST_OWNER = "AdkHex";
const RAW_URL = `https://gist.githubusercontent.com/${GIST_OWNER}/${GIST_ID}/raw/${GIST_FILE}`;

const ITERATIONS = 600000;
const MIN_PASSWORD = 8;

const argv = process.argv.slice(2);
const SWITCHES = ["--no-publish", "--publish", "--off", "--help", "-h"];
const VALUE_FLAGS = ["--title", "--min-version", "--update-url", "--check-minutes", "--out", "--serve"];
for (const a of argv) {
  if (a.startsWith("-") && !SWITCHES.includes(a) && !VALUE_FLAGS.includes(a)) die(`unknown option ${a} (--help for the list)`);
}
const flags = new Set(argv.filter((a) => SWITCHES.includes(a)));
const opt = (k, d) => {
  const i = argv.indexOf(k);
  if (i < 0) return d;
  const v = argv[i + 1];
  if (v === undefined || (v.startsWith("--") && v !== "--")) die(`${k} needs a value`);
  return v;
};
const positional = argv.filter((a, i) => !a.startsWith("-") && !VALUE_FLAGS.includes(argv[i - 1]));
const [mode, message] = positional;

function die(msg) {
  console.error(`audiosyncmaster-status: ${msg}`);
  process.exit(1);
}

function usage() {
  const lines = fs.readFileSync(new URL(import.meta.url), "utf8").split("\n");
  const text = lines.slice(1).filter((l) => l.startsWith("//")).map((l) => l.slice(3));
  console.log(text.join("\n"));
}

function gh(args) {
  try {
    return execFileSync("gh", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  } catch (e) {
    die(`gh ${args.join(" ")} failed: ${(e.stderr || e.message || "").toString().trim()}\n(is the GitHub CLI installed and logged in as ${GIST_OWNER}?)`);
  }
}

function readPublished() {
  return gh(["gist", "view", GIST_ID, "--raw", "--filename", GIST_FILE]);
}

function parseDoc(text, where) {
  try {
    const doc = JSON.parse(text);
    if (doc && typeof doc === "object" && !Array.isArray(doc)) return doc;
  } catch {
    /* fall through */
  }
  die(`${where} is not a JSON object; fix or replace it by hand first (installed copies ignore it meanwhile)`);
}

/** The document every change starts from: the gist when publishing, the local file otherwise. */
function baseDoc(publishing, out) {
  if (publishing) return parseDoc(readPublished(), `the gist's ${GIST_FILE}`);
  if (fs.existsSync(out)) return parseDoc(fs.readFileSync(out, "utf8"), out);
  return { status: "ok", checkMinutes: 5 };
}

function checkMinutes(base) {
  const given = opt("--check-minutes");
  if (given === undefined) return Number.isFinite(base.checkMinutes) ? base.checkMinutes : 5;
  const n = Number(given);
  if (!Number.isFinite(n) || n < 1 || n > 60) die("--check-minutes must be 1..60");
  return Math.round(n);
}

function versionOpt() {
  const v = opt("--min-version");
  if (v !== undefined && !/^v?\d+(\.\d+)*([-+].*)?$/.test(v)) die(`--min-version ${v} is not a version like 2.22.0`);
  return v;
}

function urlOpt() {
  const u = opt("--update-url");
  if (u !== undefined && !/^https?:\/\//i.test(u)) die("--update-url must be an http(s) address");
  return u;
}

/** Keep the key order readable: status first, the password last. */
function ordered(doc) {
  const keys = ["status", "title", "message", "minVersion", "updateUrl", "checkMinutes"];
  const out = {};
  for (const k of keys) if (doc[k] !== undefined) out[k] = doc[k];
  for (const k of Object.keys(doc)) if (!keys.includes(k) && k !== "access") out[k] = doc[k];
  if (doc.access !== undefined) out.access = doc.access;
  return out;
}

function withStatus(base, status) {
  const doc = { ...base, status };
  const title = opt("--title");
  if (status === "maintenance") {
    // The screen's text is this pause's, never one left from the last.
    delete doc.title;
    delete doc.message;
    if (title) doc.title = title;
    if (message) doc.message = message;
  } else {
    if (message || title) die("ok takes no message or title");
    delete doc.title;
    delete doc.message;
    // "ok" releases everyone: an old minVersion goes unless given again.
    delete doc.minVersion;
    delete doc.updateUrl;
  }
  const minVersion = versionOpt();
  const updateUrl = urlOpt();
  if (minVersion) doc.minVersion = minVersion;
  if (updateUrl) doc.updateUrl = updateUrl;
  doc.checkMinutes = checkMinutes(base);
  return ordered(doc);
}

async function readPassword(prompt) {
  if (!process.stdin.isTTY) {
    const chunks = [];
    for await (const chunk of process.stdin) chunks.push(chunk);
    return Buffer.concat(chunks).toString("utf8").split(/\r?\n/)[0];
  }
  return new Promise((resolve) => {
    const stdin = process.stdin;
    process.stdout.write(prompt);
    stdin.setRawMode(true);
    stdin.setEncoding("utf8");
    stdin.resume();
    let value = "";
    const cleanup = () => {
      stdin.setRawMode(false);
      stdin.pause();
      stdin.removeListener("data", onData);
      process.stdout.write("\n");
    };
    const onData = (text) => {
      for (const c of text) {
        if (c === "\r" || c === "\n") {
          cleanup();
          resolve(value);
          return;
        }
        if (c === "\u0003") {
          cleanup();
          process.exit(130);
        }
        if (c === "\u007f" || c === "\b") value = value.slice(0, -1);
        else if (c >= " ") value += c;
      }
    };
    stdin.on("data", onData);
  });
}

/** The access block for a password: what src-tauri/src/remote_status.rs checks against. */
function accessFor(password) {
  const salt = crypto.randomBytes(16);
  const hash = crypto.pbkdf2Sync(password, salt, ITERATIONS, 32, "sha256");
  const id = crypto.createHash("sha256").update(Buffer.concat([salt, hash])).digest("hex").slice(0, 16);
  return { alg: "pbkdf2-sha256", iterations: ITERATIONS, salt: salt.toString("base64"), hash: hash.toString("base64"), id };
}

async function withPassword(base) {
  if (message || opt("--title")) die("password takes no message or title");
  const doc = { ...base };
  if (flags.has("--off")) {
    delete doc.access;
    return ordered(doc);
  }
  const first = await readPassword("New access password: ");
  if (first.length < MIN_PASSWORD) die(`the password must have at least ${MIN_PASSWORD} characters (a long passphrase is best: the hash is public)`);
  if (process.stdin.isTTY) {
    const second = await readPassword("Once more: ");
    if (second !== first) die("the two passwords differ; nothing changed");
  }
  doc.access = accessFor(first);
  return ordered(doc);
}

function writeLocal(file, doc) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(doc, null, 2) + "\n");
  console.log(`wrote ${file}`);
}

/** Publish step: replace the gist file with the local document. Swap this out to move hosting. */
function publish(file, doc) {
  gh(["gist", "edit", GIST_ID, "--filename", GIST_FILE, file]);
  const back = JSON.parse(readPublished());
  if (JSON.stringify(back) !== JSON.stringify(doc)) die("the gist does not read back as what was published; check it with `status`");
  console.log(`published to gist ${GIST_ID} (${RAW_URL})`);
  console.log("installed copies see it within their checkMinutes plus up to 5 min of GitHub caching; Check again is immediate");
}

function describe(doc) {
  const lock =
    doc.status === "maintenance"
      ? "LOCKED (maintenance)"
      : doc.status === "ok"
        ? doc.minVersion
          ? `ok, builds older than ${doc.minVersion} locked`
          : "ok (running normally)"
        : `not a valid status (${JSON.stringify(doc.status)}): installed copies ignore the document`;
  const access = doc.access ? `password ON (id ${doc.access.id}): every OS user enters it once` : "password off";
  return `${lock}\n${access}`;
}

function show(text, where) {
  console.log(text.trim());
  try {
    console.log(`\n${describe(JSON.parse(text))}`);
  } catch {
    console.log(`\n(${where} is not valid JSON: installed copies ignore it and keep their last known state)`);
  }
}

function serve(file, port) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");
    if (url.pathname !== "/status.json") {
      res.writeHead(404).end("try /status.json");
      return;
    }
    let body;
    try {
      body = fs.readFileSync(file);
    } catch {
      res.writeHead(404).end("no status file yet");
      return;
    }
    res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
    res.end(body);
    let state = "?";
    try {
      const doc = JSON.parse(body);
      state = `${doc.status}${doc.access ? `, password ${doc.access.id}` : ""}`;
    } catch {
      state = "not JSON";
    }
    console.log(`${new Date().toLocaleTimeString()} GET ${req.url} -> ${state}`);
  });
  server.listen(port, "127.0.0.1", () => console.log(`serving ${file} on http://127.0.0.1:${port}/status.json`));
}

// ---- main
if (flags.has("--help") || flags.has("-h") || !mode) {
  usage();
  process.exit(mode ? 0 : 1);
}
const out = path.resolve(opt("--out", "remote-status/audiosyncmaster-status.json"));
const port = opt("--serve") !== undefined ? Number(opt("--serve")) : null;
if (port != null && !(Number.isInteger(port) && port > 0 && port < 65536)) die("--serve needs a port number");
const publishing = flags.has("--publish") || (!flags.has("--no-publish") && port == null);

switch (mode) {
  case "status":
    if (publishing) show(readPublished(), "the gist");
    else if (fs.existsSync(out)) show(fs.readFileSync(out, "utf8"), out);
    else die(`no local file at ${out}`);
    break;
  case "serve":
    if (port == null) die("serve needs --serve <port>");
    if (!fs.existsSync(out)) writeLocal(out, { status: "ok", checkMinutes: 5 });
    serve(out, port);
    break;
  case "ok":
  case "maintenance":
  case "password": {
    const base = baseDoc(publishing, out);
    const doc = mode === "password" ? await withPassword(base) : withStatus(base, mode);
    writeLocal(out, doc);
    console.log(describe(doc));
    if (publishing) publish(out, doc);
    else console.log("not published (local only)");
    if (port != null) serve(out, port);
    break;
  }
  default:
    die(`unknown mode "${mode}" (use status, maintenance, ok, password or serve; --help for details)`);
}
