# Remote status: the "Work is in progress" switch and the access password

Every installed copy of AudioSyncMaster reads one small JSON document from a secret GitHub Gist,
at start and every few minutes. The document does two things:

- **The switch.** When it says `maintenance`, every copy stops its work and shows a full-window
  "Work is in progress" screen. When it says `ok` again, the screen goes and the work carries on.
  A `minVersion` locks out older builds with "Update required".
- **The access password.** When it holds an `access` block, every OS user (Windows and macOS) must
  enter the password once. After that the app remembers it for that user, permanently, in the
  OS credential store. Changing the password asks everyone again; removing it asks nobody.

The password screen says: "This app is strictly prohibited to be used by others. If you are the
trusted one, enter the password."

## Operating it

Everything goes through `scripts/audiosyncmaster-status.mjs`, which needs the GitHub CLI `gh`
logged in as `AdkHex` (the gist's owner).

| Goal | Command |
|---|---|
| Stop every copy | `node scripts/audiosyncmaster-status.mjs maintenance "Updating, back soon"` |
| … with a title and faster checks | `node scripts/audiosyncmaster-status.mjs maintenance "msg" --title "Scheduled maintenance" --check-minutes 2` |
| Release them | `node scripts/audiosyncmaster-status.mjs ok` |
| Require a password / change it (everyone re-enters) | `node scripts/audiosyncmaster-status.mjs password` |
| No password | `node scripts/audiosyncmaster-status.mjs password --off` |
| Force an update | `node scripts/audiosyncmaster-status.mjs ok --min-version 2.23.0 --update-url https://github.com/AdkHex/AudioSyncMaster/releases/latest` |
| See the current state | `node scripts/audiosyncmaster-status.mjs status` |

- Every change starts from the document as it is and changes only what it was asked to:
  `maintenance` and `ok` keep the password, and `password` keeps the status. `ok` clears the
  maintenance text and any `minVersion` (give `--min-version` again to keep forcing an update).
- `password` asks twice without echo. It hashes the password with PBKDF2-SHA256 (600 000
  rounds, a random 16-byte salt) and publishes only the salt, the hash and its id. The password
  itself is never printed, written or sent. Use a long passphrase (at least 8 characters
  are required): anyone with the gist link can read the hash and try to guess it offline.
- Raise the app version before using `--min-version`, or the new build locks itself out.
- Changes reach every copy within `checkMinutes` (default 5) plus up to 5 minutes of GitHub
  caching. **Check again** on the lock screen is immediate.

## What happens while paused

The lock is enforced in the Rust core, not only in the UI. Every command checks one guard first
(`service::guard`, and `BridgeHandle::with` for anything that reaches an engine) and answers
`AudioSyncMaster is paused: work is in progress` (maintenance, update required) or
`AudioSyncMaster is locked` (password, first check pending).

| Area | Behaviour |
|---|---|
| Running work | The moment the lock starts, every engine is told to cancel: Movies/Series/Find match analysis, Dub sync (batch and single re-render), Subsync jobs, engine-pack and voice-tool installs, waveform and picture reads. The cancel is repeated every 2 s while any registered work is still running, so a run that was just starting cannot slip past it. |
| Conversions (Encode / FPSConv queue) | Every queued and running job is cancelled (`audioCancelAll`) and remembered. When the lock lifts, the same jobs are queued again under their own ids, once each cancelled job has reported back. |
| New work | Refused in the core: runs, previews, probes, pickers, drops, exports, opening and revealing files, API-key changes, Encode commands. Stopping (`cancel_sync`, cancelling queue jobs) always works, and so do the updater and the lock screen's own commands. |
| Resuming | The lock's event says what it stopped (`stopped`), and the event that lifts it says what to resume (`resume`). The page whose run was stopped starts it again: Dub sync only the jobs the pause cut short, Movies/Series/Find match and Subsync the run as it was set up. A toast says "AudioSyncMaster is back. … carries on." Installs, writes of corrected files and single re-renders are not restarted; press their button again. The lists live in memory, so after a restart there is nothing to resume. |
| Windows | The lock screen covers the whole window. The pages underneath are inert and their shortcuts and menus do nothing. The window stays movable and (Windows) keeps its minimize, maximize and close buttons. The main window comes to the front when a lock starts. |
| Startup | A copy that starts paused or asking for the password shows only the lock screen: the pages mount once the app has been open, so nothing on them asks the core for work it would refuse. |

The screen shows the logo, the title (`title`, default "Work is in progress"), the `message`,
"Your work is paused and will continue when AudioSyncMaster is back." and **Check again**. For
`minVersion` the title is "Update required", with **Install version X** when the in-app updater
already sees the new release, or **Download the new version** (the `updateUrl`) otherwise. The
updater keeps working while locked; it is the way out of "Update required".

## The access password

- The app knows a user may use it when that user's OS credential store holds the password's `id`:
  Windows Credential Manager or the macOS login Keychain, service `com.audiosyncmaster.app`,
  account `access`. The store belongs to the OS user, so on a shared RDP server every account is
  asked separately, and only the id is kept, never the password.
- The password is checked in Rust: PBKDF2-HMAC-SHA256 with the document's salt and rounds, compared
  with the hash in constant time. Five wrong passwords in a row make the next attempt wait 30 s.
- A new password means a new salt, hash and id, so everyone is asked once more. Removing `access`
  (`password --off`) asks nobody.
- The build has `"requireAccess": true` baked in (`src-tauri/remote-status.json`). A copy that has
  **never** read the document (fresh install, offline) shows "Connect to the internet once to verify
  access" with **Try again** instead of opening. After one successful check the usual sticky rules
  apply, offline included.
- Maintenance and "Update required" win over the password screen when both apply.

## The document

```json
{
  "status": "ok",
  "title": "Work is in progress",
  "message": "We're updating AudioSyncMaster. Back in a few minutes.",
  "minVersion": "2.23.0",
  "updateUrl": "https://github.com/AdkHex/AudioSyncMaster/releases/latest",
  "checkMinutes": 5,
  "access": {
    "alg": "pbkdf2-sha256",
    "iterations": 600000,
    "salt": "<base64, 16 random bytes>",
    "hash": "<base64, 32 bytes>",
    "id": "<first 16 hex digits of sha256(salt || hash)>"
  }
}
```

| Field | Meaning |
|---|---|
| `status` | `"ok"` or `"maintenance"` (required; anything else = no information) |
| `title`, `message` | The maintenance screen's text (cut at 200 / 2000 characters) |
| `minVersion` | Builds older than this show "Update required" (numeric compare, `2.9.0 < 2.10.0`) |
| `updateUrl` | The "Download the new version" link (http/https only) |
| `checkMinutes` | How often the apps check, 1..60, default 5 |
| `access` | Present = password required, absent or `null` = no password |

Unknown fields are ignored and invalid optional fields dropped. A document that is not JSON, has no
valid `status`, is larger than 64 KB, or has an `access` block that cannot be checked (wrong `alg`,
rounds outside 10 000..10 000 000, bad base64, an `id` that does not match) counts as **no
information**: the app keeps its last known state. It never locks or unlocks because of garbage.

## Where it is hosted

A secret GitHub Gist on `AdkHex`:

- gist id `1cf1fb34cc1a7fcabc97a1cef5f7bd7a`, file `audiosyncmaster-status.json`
- raw URL, baked into the build:
  `https://gist.githubusercontent.com/AdkHex/1cf1fb34cc1a7fcabc97a1cef5f7bd7a/raw/audiosyncmaster-status.json`

The URL and `requireAccess` come from `src-tauri/remote-status.json`, read at compile time
(`include_str!`), so testers cannot point an installed copy elsewhere. The baked URL must be
`https://`. An empty `url` turns the switch off in that build, and then it also forgets any lock an
earlier build left behind. A secret gist is unlisted, not private: anyone with the link can read it,
which is why it only ever holds a status and a salted hash. The link is kept out of a release
build's log and out of the error shown on the lock screen.

**Never delete or rename the gist.** Copies that cannot reach it keep their last known state forever.

## Testing locally (never against the real gist)

Debug builds read two environment variables. Release builds ignore both:

| Variable | Meaning |
|---|---|
| `AUDIOSYNC_STATUS_URL` | Status URL instead of the baked one: `https://`, or `http://` for 127.0.0.1 / localhost only |
| `AUDIOSYNC_STATUS_POLL_SECS` | Poll interval in seconds instead of `checkMinutes` |

```sh
# terminal 1: a local document and a server for it (--serve implies --no-publish)
node scripts/audiosyncmaster-status.mjs ok --no-publish --out /tmp/asm/status.json
node scripts/audiosyncmaster-status.mjs serve --serve 8799 --out /tmp/asm/status.json

# terminal 2: the dev server and the debug app against it
npm run dev
cargo build --manifest-path src-tauri/Cargo.toml
AUDIOSYNC_STATUS_URL=http://127.0.0.1:8799/status.json AUDIOSYNC_STATUS_POLL_SECS=3 \
  ./src-tauri/target/debug/audiosyncmaster

# terminal 3: flip it -- ALWAYS with --no-publish and the same --out
node scripts/audiosyncmaster-status.mjs maintenance "Testing the lock" --no-publish --out /tmp/asm/status.json
node scripts/audiosyncmaster-status.mjs ok --no-publish --out /tmp/asm/status.json
printf '%s\n' 'a test passphrase' | node scripts/audiosyncmaster-status.mjs password --no-publish --out /tmp/asm/status.json
node scripts/audiosyncmaster-status.mjs password --off --no-publish --out /tmp/asm/status.json
```

Without `--no-publish` a command publishes to the real gist (only `--serve` implies it). The debug
app shares its data folder and Keychain entry with an installed AudioSyncMaster, so clean up after
a test:

```sh
rm -f ~/Library/Application\ Support/com.audiosyncmaster.app/remote-status.json
security delete-generic-password -s com.audiosyncmaster.app -a access   # macOS
```

On Windows the same files are `%APPDATA%\com.audiosyncmaster.app\remote-status.json` and the
Credential Manager entry `access` under `com.audiosyncmaster.app`.

Automated tests:

```sh
cargo test --manifest-path src-tauri/Cargo.toml remote_status   # parsing, versions, PBKDF2 vector from Node, id, throttling, gate
cargo test --manifest-path src-tauri/Cargo.toml service          # the work registry and the conversion tracker
cargo test --manifest-path src-tauri/Cargo.toml --test remote_status   # against a local HTTP server: lock/unlock, sticky across restarts, password set/changed/off, connect once, garbage never locks
AUDIOSYNC_E2E=1 cargo test --manifest-path src-tauri/Cargo.toml a_remote_pause   # the real engine: a dub sync in flight stopped, new work refused, resume list
python/.venv/bin/python tests/run_all.py test_encode_service      # a cancelled conversion queued again under its id
```

## Offline behaviour: sticky and fail-open

- Every successful check is kept in `remote-status.json` in the app data folder. On start the app
  applies it **before** anything can run, so a lock survives restarts and offline periods until a
  good `ok` arrives.
- A failed check (no network, DNS, timeout, HTTP error, redirect to http, unusable document) changes
  nothing; the reason shows on the lock screen and in the log ("remote status: …").
- A copy that has never reached the server runs normally only if the build does not require access.
  This build does, so it shows "Connect to the internet once" until its first successful check.
- `minVersion` is compared with the running build, so installing a newer build lifts "Update
  required" at once, even offline.

## Security notes and limits

- Requests are GET only, with a cache-busting `?_hs=` query and `Cache-Control: no-cache`, no cookies,
  a 10 s timeout and a 64 KB cap, through a proxy set in the environment (`HTTPS_PROXY`), like the
  updater. Redirects are followed only to https (or a loopback test server).
- The document can only lock, unlock and show text; it cannot run anything. Text is rendered as text.
- Anyone who can edit the gist can lock every copy: keep the GitHub account secure.
- This keeps honest users and other accounts out; it is not DRM. Someone with admin rights and
  enough effort (editing the kept document while blocking GitHub, patching the binary) can get round
  any client-side check.
- macOS: the build is not signed by Apple, so after an update macOS may ask once whether the new
  version may read its Keychain entry. **Always Allow** keeps it quiet; **Deny** only means the app
  asks for the password again.

## Code map

| Piece | Where |
|---|---|
| Parsing, versions, the lock decision, PBKDF2 check, throttling, fetch, kept document | `src-tauri/src/remote_status.rs` |
| Guard, work registry, stopping and resuming, conversion tracker, poll loop, commands `service_lock` / `check_service_status` / `unlock_access` / `legacy_install`, Keychain store | `src-tauri/src/service.rs` |
| Baked URL and `requireAccess` | `src-tauri/remote-status.json` |
| Guard calls | every command in `lib.rs`, `subsync.rs`, `encode.rs`; `BridgeHandle::with` in `bridge.rs` |
| Re-queueing a cancelled conversion | `AudioQueue.add` in `audiosync/encode/service.py` |
| Lock screen, password form, "all users" notice | `src/app/ServiceLock.tsx`, `src/app/useServiceLock.ts`, `src/lib/service.ts`, styles at the end of `src/ui/ui.css` |
| Mounting and resuming | `src/App.tsx`, `src/pages/Index.tsx` (remote lock section), `resume` in `src/app/shell.tsx` / `DubPage.tsx` |
| Developer tool | `scripts/audiosyncmaster-status.mjs` |
| Tests | `src-tauri/tests/remote_status.rs`, unit tests in both Rust modules, `a_remote_pause_stops_the_engine_and_hands_its_work_back` in `lib.rs`, `tests/test_encode_service.py` |
