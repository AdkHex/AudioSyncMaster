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
- confirms, once every build has finished, that the release offers an
  update to all three platforms (`scripts/check-release.cjs`)

Nothing is published unless the tests pass first. Two pushes close together
do not race for the same number: the second run waits for the first.

### When one platform's build fails

The three builds publish to the same release, so a build that fails leaves
a release the other two platforms got and that one never will -- its apps
are simply never offered it (the updater reads `latest.json`, which then
has no entry for them). The run is red and the last job names the missing
platform. To recover, fix whatever failed and push: while **nothing that
ships** has changed since the incomplete release, the push is released as
the **same version again** and the missing build joins it (`tauri-action`
adds to an existing release and merges the platform into `latest.json`).
Once something shipped has changed, the next minor is released instead and
supersedes it, which reaches the stranded platform just the same.

This happened on v2.11.0: the Windows build failed in `set-version.cjs`,
which matched `Cargo.lock` with a bare LF while Git for Windows checks out
with CRLF. The script now accepts either and writes each file back with
its own line endings.

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

- **Windows** — only the NSIS installer is built, and it installs for the user
  who runs it (see below). Updates use the same installer (`updaterJsonPreferNsis`)
  in passive mode: a progress bar, no prompts, no administrator.
- **macOS** — updates replace the `.app` bundle. The build is unsigned by Apple,
  so first-time users still need to right-click → Open. Updates themselves are
  unaffected.
- **Linux** — only AppImage supports self-updating. `.deb` users must install
  new versions through their package manager.

## Per-user Windows install

On a shared Windows machine (the RDP server) an install must reach only the
account that ran it. Tauri's MSI always installs **per machine** -- Program
Files, every account -- and `bundle.targets: "all"` used to build one beside
the NSIS installer, so whoever ran the MSI installed the app for everyone.

From the first release built from this change on (v2.22.0, if CI picks the
next minor after v2.21.0 as usual):

- Windows builds **only** the NSIS installer, in `currentUser` mode
  (`src-tauri/tauri.windows.conf.json`: `bundle.targets: ["nsis"]`,
  `bundle.windows.nsis.installMode: "currentUser"`). It installs into
  `%LOCALAPPDATA%\AudioSyncMaster`, with the Start menu entry and uninstall
  entry under that user's HKCU, and asks for no administrator. macOS and Linux
  still build everything (`targets: "all"` in `tauri.conf.json`).
- The release has no `.msi` and `latest.json` has no `windows-x86_64-msi`
  entry: `windows-x86_64` is the `-setup.exe` and its signature. Installed
  copies that came from an old MSI look for the `-msi` entry first, find none,
  and fall back to that `-setup.exe` -- so their next update installs a
  per-user copy for whoever runs it.
- Three checks keep it that way (`scripts/windows-installer.cjs`): the config
  is tested before anything is built (`npm test`, in every test job), the
  Windows build job checks its own output before trusting it (no `.msi`, a
  `-setup.exe`, `RequestExecutionLevel user` in the generated `installer.nsi`),
  and the last job checks the published `latest.json`.

### Migrating an all-users copy (administrator, once per machine)

A per-user installer cannot remove a copy installed for all users: that takes
administrator rights. Until an administrator removes it, every account still
sees the old copy, and the new per-user copy shows its users a one-time notice
("An all-users copy is still installed … Ask an administrator to uninstall it
from Settings > Apps").

On the RDP server, an administrator:

1. Opens **Settings > Apps > Installed apps** (or Control Panel > Programs and
   Features) and uninstalls **AudioSyncMaster** -- the entry installed for all
   users, in `C:\Program Files\AudioSyncMaster`. Uninstalling it does not touch
   any user's settings, history or per-user copy.
2. From then on each user who should have the app installs the new
   `AudioSyncMaster_<version>_x64-setup.exe` from their own account (or keeps
   the per-user copy the updater already installed for them).

Say this in the release notes of the first per-user release (the workflow's
release body carries it).

## Remote switch and access password

Every copy reads a status document from a secret gist: it can pause every copy
("Work is in progress"), force an update, and require an access password that
each OS user enters once. Operating it is a command, not a release:

```sh
node scripts/audiosyncmaster-status.mjs maintenance "Updating, back soon"
node scripts/audiosyncmaster-status.mjs ok
node scripts/audiosyncmaster-status.mjs password          # set or change it
node scripts/audiosyncmaster-status.mjs password --off
node scripts/audiosyncmaster-status.mjs status
```

Everything else -- the document, what stops while paused, testing against a
local server, limits -- is in [docs/REMOTE-STATUS.md](docs/REMOTE-STATUS.md).
Never delete or rename the gist, and raise the app version before using
`--min-version`.

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
