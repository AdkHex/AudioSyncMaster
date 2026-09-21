# Releasing

Installed apps update themselves from GitHub Releases. This document covers how
that works and what you have to do.

## Shipping an update

Push to `main`. That is the whole process.

CI decides the version number itself (`scripts/decide-version.cjs`):

- an ordinary push is released as the **next minor version** after the latest
  release -- v2.10.0 released, push, v2.11.0 released -- with nothing in the
  repo edited by hand;
- a push that changes nothing that ships (only `*.md`, `tests/`, `.github/`,
  `scripts/`, the dev helpers or the mockups) is tested and **not** released,
  and the run says so in a notice; installed apps are not bothered for a
  README change;
- if you want a particular number -- a major bump, a patch -- set it yourself
  with `node scripts/set-version.cjs 3.0.0` and push: a repo version newer
  than the latest release is used as it is.

The build jobs write the decided version into every file that carries one
(`package.json`, both lockfiles, `src-tauri/tauri.conf.json`,
`src-tauri/Cargo.toml`) before they install or build, so the app's own
version, the tag and `latest.json` agree. The files in the repo are not
changed by CI: `package.json` may say 2.10.0 while the latest release is
v2.13.0, and that is expected -- the releases page is the record of what
shipped, and Settings → About in the app shows the built version.

CI then:

- runs the full test suite on Linux, Windows and macOS, plus Rust checks
- builds installers for all three platforms
- signs the updater artifacts with the private key held in GitHub Secrets
- publishes a release tagged `v<version>` with a `latest.json` manifest
- fails loudly if `latest.json` is missing, since that would silently strand
  every installed app

Nothing is published unless the tests pass first. Two pushes close together
do not race for the same number: the second run waits for the first.

### Why the number is not in the commit

The old process was to bump five files by hand before pushing, and the
failure mode was forgetting: the run went green, the commit looked shipped,
and nothing reached users (`tauri-action` will not overwrite a tag that
exists). A later version skipped the build with a notice instead, which was
honest but still shipped nothing. Deciding the number in CI removes the step
that was being forgotten. Committing the bump back from CI would work too,
but every release would then move `main` under you and the next push would
be rejected until you pulled; leaving the repo's number alone avoids that.

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

## The packaged engine

The Python engine ships as a PyInstaller **directory** build under
`src-tauri/resources/engine`, not as a single-file executable and not as a
Tauri `externalBin`.

This is deliberate. A `--onefile` build re-extracts its entire payload to a
temp directory on every launch; measured cold start was 32-63 seconds per run
on macOS once Gatekeeper rescanned the unpacked copy. The directory build
starts in ~0.12s because nothing is unpacked. A directory cannot be an
`externalBin`, hence the resource folder.

CI verifies the packaged engine both starts *and* measures a known fixture
correctly, since a build that launches but computes garbage would otherwise
ship unnoticed.

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
