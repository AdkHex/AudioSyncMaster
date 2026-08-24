# Releasing

Installed apps update themselves from GitHub Releases. This document covers how
that works and what you have to do.

## Shipping an update

1. Bump the version in **all four** of:
   - `package.json` — CI tags the release from this one
   - `src-tauri/tauri.conf.json` — the app compares its own version from here
   - `src-tauri/Cargo.toml` — the crate version
   - `RELEASING.md` is not versioned, but the lockfiles are: run
     `npm install --package-lock-only` and update the `audiosyncmaster` entry
     in `src-tauri/Cargo.lock`, or `npm ci` and the Rust build will fail on a
     lockfile mismatch.

   The frontend reads its displayed version from `package.json` at build time
   (see `define` in `vite.config.ts`), so Settings → About needs no edit.
2. Push to `main`.

That is the whole process. CI then:

- runs the full test suite on Linux, Windows and macOS, plus Rust checks
- builds installers for all three platforms
- signs the updater artifacts with the private key held in GitHub Secrets
- publishes a release tagged `v<version>` with a `latest.json` manifest
- fails loudly if `latest.json` is missing, since that would silently strand
  every installed app

Nothing is published unless the tests pass first.

## What users see

On launch the app waits three seconds, then asks GitHub whether a newer version
exists. If one does, a dialog shows the version, the release notes, and three
choices: **Install and restart**, **Later**, or **Skip this version**.

Downloads show real progress. The app restarts into the new version when the
install finishes.

Checks are throttled to once every six hours, and failures are silent — a user
who is offline or behind a proxy that blocks GitHub still gets a working app.
There is also a **Check now** button under Settings → Updates, which bypasses
both the throttle and any skipped version.

## Version numbers

The updater compares semantic versions, so `2.0.1` supersedes `2.0.0`. Never
reuse or lower a version: an app on a higher version than the release will
simply never update.

## The signing key

Updates are signed with a minisign key. The app embeds the matching public key
and **refuses any update not signed by it**, so a compromised release host
cannot push a malicious build.

Two GitHub Actions secrets drive this:

| Secret | Value |
| --- | --- |
| `TAURI_SIGNING_PRIVATE_KEY` | contents of the private key file |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | the key's password (empty if none) |

The public key lives in `src-tauri/tauri.conf.json` under `plugins.updater.pubkey`.

### If you lose the private key

Every installed app will reject all future updates permanently. There is no
recovery: you would have to publish a new version with a new public key and get
every user to reinstall manually. **Keep an offline backup.**

### Rotating the key

Only worth doing if the private key leaks. Generate a new pair, update the
`pubkey` in `tauri.conf.json`, replace the secrets, and release. Users must
manually install that release once — their existing app cannot verify it — but
updates work normally from then on.

```sh
npx tauri signer generate -w ~/.tauri/audiosync.key
```

## Platform notes

- **Windows** — the NSIS installer is used for updates (`updaterJsonPreferNsis`).
  It runs in passive mode: a progress bar, no prompts.
- **macOS** — updates replace the `.app` bundle. The build is unsigned by Apple,
  so first-time users still need to right-click → Open. Updates themselves are
  unaffected.
- **Linux** — only AppImage supports self-updating. `.deb` users must install
  new versions through their package manager.

## Verifying a release worked

```sh
gh release view v<version> --json assets --jq '.assets[].name'
```

You should see installers for each platform plus `latest.json`. To inspect what
the app will actually fetch:

```sh
curl -sL https://github.com/AdkHex/AudioSyncMaster/releases/latest/download/latest.json | jq
```

Each platform entry needs a `signature` and a `url`. A platform missing from
that file will not receive the update.
