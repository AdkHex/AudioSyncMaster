# FPSConv → AudioSyncMaster: master plan

**Status:** plan only, written 2026-10-12. Nothing is built yet.

| | |
|---|---|
| **Source** | FPSConv **v1.1.24**: the `fpsconv/` package in `~/Documents/FPSConv` (repo `AdkHex/FPSConv`). It is Windows-only and ships as an Inno installer with its own updater. |
| **Target** | AudioSyncMaster **v2.19.0** (this repo): Tauri, React and a Python engine. It ships for Windows, macOS (arm64) and Linux. |
| **Goal** | Every FPSConv feature, including its CLI, lives in AudioSyncMaster as two **Windows-only** pages, **Encode** and **FPSConv**, right after Subsync. A final FPSConv release turns installed copies into a **pure black screen**, the old installers are deleted, and the FPSConv repo is **archived**. |

Unless a path says otherwise, refs like `engine.py:1880` point into FPSConv's `fpsconv/` package and refs like `Index.tsx:50` point into this repo.

---

## 0. Verdict

**It can be done.** Each FPSConv piece already has a counterpart in AudioSyncMaster:

| FPSConv piece | AudioSyncMaster counterpart |
|---|---|
| Job queue with per-job progress and cancel | The batch pattern in `python/bridge.py` (`handle_dubsync_batch`, `subs/service.py`) |
| Killing child processes | `CancellationToken` already kills whole process trees (`media.py:183-247`) |
| ffprobe stream info and frame-rate detection | `media.probe`, `choose_frame_rate`, and the exact rates in `framerate.py` |
| On-demand tool downloads with hash checks | The engine-pack system (`subs/packs.py`) |
| Pages, dock, history, toasts, drag-drop, updater | The workstation shell (`src/pages/Index.tsx`, `src/app/shell.tsx`) |

These parts do **not** exist yet and must be built:

1. **The Dolby stack:** DEE, deew, DeeZy, truehdd, MediaInfo and the 7.1 patcher. A grep finds no trace of any of them in AudioSyncMaster today.
2. **Tool-path settings:** AudioSyncMaster finds tools only through environment variables, the bundled folder or PATH, and has no screen for setting paths.
3. **A separate engine process for the audio queue.** The main engine runs **one command at a time** (`bridge.py:1159-1179`). If the queue used it, a 40-minute Atmos encode would lock Movies, Dub sync and Subsync.
4. **Reach of the FPSConv kill.** v1.1.x has no kill switch and no forced update. The black-screen release only reaches copies with auto-update on, which is the default (§10.1).

---

## 1. Decisions this plan is built on

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Platform | **The Encode and FPSConv pages and the ported CLI are Windows-only** (your call, 2026-10-12, because DEE is Windows-only). The pages are hidden on macOS and Linux, and the engine handlers and CLI refuse there (§4). | DEE does not run on macOS or Linux. |
| D2 | Pages | Two pages **right after Subsync**: **`encode` "Encode"** (Ctrl+6) and **`fpsconv` "FPSConv"** (Ctrl+7). The FPSConv page does the old app's FPS change. | Your choice (2026-10-12). Users of the old app find the FPS tool under its familiar name. |
| D3 | Engine process | A dedicated **EncodeBridge** process runs the audio queue. Quick lookups (probe, resolve, doctor, MediaInfo) go to the existing info bridge. | Long encodes then never block the other pages, which matches today's two-app setup. |
| D4 | Queue | **One shared audio queue**, as in FPSConv. Each page shows only the jobs for its own task. Jobs can be added, cancelled one at a time, and the worker count changed while the queue runs. | Keeps FPSConv's behaviour (`queue.py`). |
| D5 | deew, pymediainfo, 7.1 patcher | An on-demand **"Dolby tools" engine pack**, Windows x64 only. It is a uv venv with pinned versions plus the patcher zip, whose SHA-256 is pinned. | deew pins `rich<14`, and keeping it out of the main engine avoids that conflict. The pack also provides a real Python for the patcher wrapper script, replacing the `fpsconv-cli.exe <script.py>` workaround. AudioSyncMaster already installs isolated dependencies this way. |
| D6 | DEE, truehdd, DeeZy | **Paths you set** in Preferences → Encode, as in FPSConv. | DEE is proprietary. DeeZy needs `rich>=14`. |
| D7 | ffmpeg | AudioSyncMaster's own discovery plus its **ffmpeg-full pack** (SHA-checked), with optional path overrides. | Replaces FPSConv's unverified gyan.dev download (`engine.py:414-446`). |
| D8 | Behaviour | **Match FPSConv exactly first.** Same command lines, output names, bitrate tables and notes, pinned by the ported tests. Only the fixes in §8 change behaviour. | Users switching from FPSConv get identical files. |
| D9 | Settings | The engine owns a settings file: **`%APPDATA%\com.audiosyncmaster.app\encode\settings.json`**, a port of FPSConv's `config.py` (same keys and defaults, atomic writes, key coercion). Finished jobs go in `history.json` next to it. The pages read and write both through the engine, and the CLI reads the same files. Settings are **imported once** from FPSConv's `settings.json` and deew's `config.toml`. | The CLI is ported (D11), and a CLI can't read the webview's localStorage. Users don't have to set up the tools again. |
| D10 | Rollout | All work happens on branch `feature/encode-fps`. **Nothing reaches `main` until the Windows acceptance run (§7) passes.** | A push to `main` ships to every installed AudioSyncMaster. |
| D11 | CLI | **FPSConv's CLI is ported** into the engine exe as `audiosync-cli.exe convert \| encode \| probe \| doctor \| patcher \| dee`, with the same flags (§2.4). With no arguments the exe is still the app's engine, which is how Rust starts it (`bridge.rs:255`, `:278-279`). A Start-menu shortcut "AudioSyncMaster command line" replaces FPSConv's. | Your choice (2026-10-12). The engine is already a console exe, so no second program is needed. |
| D12 | eac3to | **eac3to does all it can** when its path is set in Preferences → Encode: it decodes and re-times FPS changes to W64 (`-23.976 -changeTo25.000`), decodes Encode sources (`-resampleTo48000` when not 48 kHz) and extracts TrueHD for flat 7.1. DEE still encodes every Dolby output, ffmpeg the AAC output. ffmpeg keeps FPSConv's steps when eac3to is not set, for containers eac3to cannot read (MP4, M4A, MOV, WebM, Ogg) and for 3/5/7-channel layouts that need padding. **This overrides D8 for those steps:** eac3to's frame-rate change resamples, so the pitch moves with the speed (≈4 % for 23.976 → 25) where FPSConv's `atempo` kept it, and FPS-change outputs no longer match the Phase 0 baseline byte for byte. Output names and durations do not change. | Your choice (2026-10-13). eac3to is freeware, Windows only, and not redistributable as far as its terms say, so it is a user-set path like DEE. |

---

## 2. Feature map: FPSConv → AudioSyncMaster

### 2.1 FPS change → **FPSConv** page

| FPSConv feature (ref) | AudioSyncMaster home |
|---|---|
| Six conversions between 23.976, 24 and 25 (`engine.py:59-66`) | `audiosync/encode/fps.py`. The ratios are rebuilt from `framerate.py` fractions, which give the same values FPSConv uses (FPSConv already uses 24000/1001). |
| Ratio shown next to the dropdown, e.g. `×1.0427 · atempo=1.042709` | A readout in the page strip |
| Speed change with `atempo`, chained when the ratio is outside 0.5–2, or `anull` at ratio 1 (`:810-825`) | `fps.atempo_chain()`. Same strings. |
| Codec classes: eac3, ac3 and truehd keep their codec; everything else becomes AAC (`:611-619`) | `fps.codec_class()`, carried over unchanged (§9.2) |
| AAC path: one ffmpeg pass (`:1880-1891`) | `commands.fps_aac()` |
| AC-3, E-AC-3 and TrueHD path: ffmpeg → 24-bit WAV → deew (`:1893-1918`) | `commands.fps_wav()` + `commands.deew()` |
| Bitrate: same as source, or custom 32–6144 kbps | Strip overflow menu |
| Parallel jobs (1–8) and the if-output-exists policy (overwrite, skip, numbered copy) | Strip overflow menu, shared with Encode |
| **Target video:** reads its fps and duration, suggests a conversion per row and switches the dropdown (`:566-599`, `index.html:784-793`) | A "Target video" picker in the strip, backed by `audioReference` and `audioSuggest` |
| fps detection: video track, then tags, then filename (`:505`, `:541-563`) | `fps.detect_fps()`, built on `media.probe` and `choose_frame_rate`. The tag and filename fallbacks are ported. |
| Output name `<stem>[_a<N>]_23_976-25<ext>` (`:828-832`) | `fps.output_name()`. Same names. |

### 2.2 Audio encode → **Encode** page

| FPSConv feature (ref) | AudioSyncMaster home |
|---|---|
| Formats DDP and DD. Channels: as source, 7.1, 5.1, 2.0, 1.0. Bitrate "Auto" or a list. Atmos switch. Layout: Flat 7.1 or DEE default. Five DRC profiles. (`index.html:394-409`) | Encode page strip |
| Bitrate tables, snapping to allowed values, odd layouts padded, `resolve_encode` rules with user-visible notes (`engine.py:90-235`) | `audiosync/encode/formats.py`. The rules stay the same, plus fix #1 in §8. |
| Per-row preview, e.g. "DDP 7.1 Atmos flat 1536k" (`/api/resolve`) | `encodeResolve`, shown in a queue-table column |
| Plain DD/DDP: WAV with `-drc_scale 0`, soxr resampling with fallback, then `deew -r <drc> [-dm N]` (`:1696-1721`, `:2021-2026`) | `commands.encode_wav()` + `commands.deew()` |
| Atmos via DeeZy: streaming 5.1 or Blu-ray "DEE default" (`:1724-1757`) | `commands.deezy()`. The DeeZy CLI quirks are kept: flag order, `--working-dir`, `DEEZY_NO_PROGRESS`. |
| **Flat 7.1 Atmos:** copy the TrueHD stream, truehdd decode, then the LumaVistaLab wrapper (`:2045-2086`, `:1616-1641`) | `audiosync/encode/flat71.py`. The wrapper runs on the Dolby pack's Python. |
| Patcher download (zip SHA pinned, zip-slip guard) and DEE DLL hash check with six verdicts (`:1497-1582`) | `flat71.py`. **Copy both full hashes verbatim** from `engine.py:1497-1505`. A DLL that doesn't match must make the job refuse, never fall back. |
| Atmos detection: MediaInfo JSON from the CLI, falling back to pymediainfo (`:622-636`, `:705-722`) | `audiosync/encode/atmos.py` for parsing, plus a `mediainfo_worker.py` run in the Dolby pack (§3.1) |
| Output name `<stem>[_a<N>]_DDP7.1Atmos_1536k.ec3` (`:238-242`) | `formats.output_name()`. Same names. |

### 2.3 Shared FPSConv behaviour

| FPSConv | AudioSyncMaster |
|---|---|
| Queue sections Running, Queued, Ready, Finished, with counts and an overall bar | `src/app/pages/audio/AudioQueue.tsx`, adapted from `DubQueue.tsx` and `FileTable.tsx`, plus the page's Lcd |
| Per row: stream picker, metadata chips, MediaInfo, remove, cancel, retry, open folder | Same, in table row actions |
| Skip a job that is already queued or running (`Job.key()`, `engine.py:991-997`) | `service.py`. Same key. |
| Stage list with weighted percent, elapsed and ETA (`:1004-1051`, `:1137-1141`) | `encode/progress.py`. The Progress tab is in the page dock. |
| Progress parsers for ffmpeg, deew, DeeZy, truehdd and the patcher, plus the 5-second idle heartbeat (`:1167-1449`, `:1644-1682`) | `encode/progress.py`, ported line for line |
| Picks the first informative error line from tool output (`:1156-1161`) | `progress.pick_error()` |
| Job log with the exact command lines | Job log tab in the page dock |
| App log | The **Output** dock tab that already exists |
| History: the last 200 finished jobs survive a restart, with Retry, Open, MediaInfo and Remove (`queue.py:31-54`) | The engine's `history.json` (200 jobs, 20 log lines each), as in FPSConv. Each time the queue drains, a summary also goes to the shared **History** dock (`createSummaryEntry`). |
| Settings saved as you change them (`config.py`) | The engine's `settings.json` (D9). Every change is saved, and the CLI sees it too. |
| Doctor: status rows, readiness callout, DLL verdict, status bar | Preferences → Encode tab, plus an `InfoBar` on the page when something blocks a run |
| deew `config.toml` management, including migrating old configs (`:329-411`) | `encode/tools.py`, run before every queue start and on Apply |
| Temp, Working and Logs folders, plus per-job temp folders that are deleted afterwards (`:1862`, `:2089-2116`) | Temp and Working folders in Preferences. Defaults move under AudioSyncMaster's cache folder. Per-job cleanup is ported. Logs go to AudioSyncMaster's own logging. |
| Keyboard: Ctrl+O to add, Ctrl+Enter to start, Delete to remove, Esc | Shell commands through `usePageCommands` (add, start with Enter, stop with Esc, remove) |
| Drag and drop of files and folders (folders are not searched recursively) | The window-wide shell drop calls the page's `drop` command. **One** list of extensions (`AUDIO_EXTS`, `engine.py:68-72`) serves the file dialog, drops and folder expansion. |

### 2.4 FPSConv CLI → `audiosync-cli.exe`

The flags stay exactly as in FPSConv (`__main__.py:208-249`), so existing scripts only need the exe name changed.

| FPSConv | AudioSyncMaster |
|---|---|
| `fpsconv-cli.exe convert INPUTS… -m <mode> [-o -j -r -s -b --overwrite]` | `audiosync-cli.exe convert …`, same flags |
| `fpsconv-cli.exe encode INPUTS… [-f -c -b --no-atmos --atmos-71 --no-bed-conform --drc -o -j -r -s --overwrite]` | `audiosync-cli.exe encode …`, same flags |
| `probe [--target VIDEO]` | `audiosync-cli.exe probe …` |
| `doctor` | `audiosync-cli.exe doctor` |
| `patcher` (downloads the 7.1 patcher) | `audiosync-cli.exe patcher`, which installs the **whole Dolby tools pack** (deew, pymediainfo, patcher) |
| `dee <path>` | `audiosync-cli.exe dee <path>`, which writes deew's `config.toml` and the settings file |
| `--version` | `audiosync-cli.exe --version` prints the AudioSyncMaster version |
| `gui` | Dropped: the app is the GUI |
| Hidden `deew` / `deezy` / `<script.py>` runners | Not needed: the Dolby pack's Python runs them |
| Start-menu "FPSConv command line" (`fpsconv-cli.exe doctor` in a console) | Start-menu "AudioSyncMaster command line", added through a Tauri NSIS installer hook. MSI installs get no shortcut; `ENCODE.md` gives the exe path. |

**Behaviour kept from `__main__.py:80-129`:**
- a progress line every 0.5 s
- Ctrl-C cancels everything and exits 130
- exit 1 if any job ends other than done or skipped
- no history written (`history=False`)

**Changed:** on macOS and Linux every subcommand prints "Windows only" and exits 2.

---

## 3. Architecture in AudioSyncMaster

### 3.1 Python: new package `audiosync/encode/`

Split FPSConv's 95 KB `engine.py` by concern. **Don't** copy it as one file.

| Module | Contents (FPSConv source) | Pure? |
|---|---|---|
| `formats.py` | Targets, channel counts, DRC profiles, bitrate tables, `snap_bitrate`, padding, `resolve_encode` and its notes, encode naming (`:90-242`) | yes |
| `fps.py` | Conversions as `framerate` fractions, `atempo_chain`, `codec_class`, `detect_fps` fallbacks, `suggest_conversion`, FPS naming (`:59-66`, `:499-619`, `:810-832`) | yes |
| `commands.py` | Every command builder: fps_aac, fps_wav, encode_wav, deew, deezy, thd_copy, truehdd, flat71 wrapper (`:1616-1757`, `:1880-1918`) | yes |
| `progress.py` | Stage model, weights, all parsers, heartbeat, `pick_error` (`:1004-1051`, `:1134-1449`, `:1644-1682`) | yes |
| `atmos.py` | MediaInfo JSON parsing, `is_atmos`, pretty codec names, report text fallback (`:622-722`, `:783-788`) | yes |
| `tools.py` | Tool discovery from request paths, then the pack, then `media._find_binary`, then PATH; `doctor()`; deew `config.toml` read/write/migrate (`:247-411`, `:886-944`) | no |
| `flat71.py` | Patcher download, unpack and verify; DLL verdict; `flat71_ready` (`:1497-1613`) | no |
| `jobs.py` | `run_fps_job`, `run_encode_job` with a `CancellationToken`, per-job temp folders, output policy (overwrite, skip, rename `_1`…), partial output deleted on cancel (`:1814-2116`) | no |
| `service.py` | The long-lived queue: a pool of up to 8 workers, enqueue, cancel one job, cancel all, change worker count, duplicate-key skip, drained summaries | no |
| `settings.py` | Port of FPSConv's `config.py`: keys, defaults, coercion of nested sections, atomic `.tmp` + `os.replace` writes, the 200-job history cap (`config.py:19-165`). Paths: config `%APPDATA%\com.audiosyncmaster.app\encode\`, cache (temp, work) `%LOCALAPPDATA%\com.audiosyncmaster.app\encode\`. | yes, given a folder |
| `fpsconv_import.py` | Reads `%APPDATA%\FPSConv\settings.json` and `%LOCALAPPDATA%\deew\config.toml` and maps them into `settings.py` | yes, given the file contents |
| `cli.py` | Port of FPSConv's `__main__.py` subcommands (§2.4), driving `service.py` directly, with no bridge | no |
| `workers/mediainfo_worker.py` | Runs inside the Dolby pack (pymediainfo). Takes a list of paths and returns MediaInfo JSON for each. Uses the `_protocol.py` rules: stdout carries only protocol lines, and it must not import `audiosync`. | no |

**Platform guard.** Every handler entry point and every CLI subcommand calls `encode.require_windows()` first. The pure modules import fine everywhere, so tests run on all three CI operating systems.

**CLI dispatch.** At the start of `main()` in `python/bridge.py` (`:1213`), check the arguments. If the first one is a known subcommand or `--version`, return `audiosync.encode.cli.main(argv)`. Otherwise run the stdin engine loop as today. Rust always starts the engine with no arguments, so the app never reaches the CLI path.

### 3.2 Bridge protocol (`python/bridge.py`)

Add these to `HANDLERS` (`bridge.py:1120-1143`) and to a new protocol table in `ENCODE.md`.

| Command | Bridge | Events | Terminal |
|---|---|---|---|
| `audioProbe {paths, tools}` | info | none | `audioProbeResult {rows[{path, duration_s, fps, fps_source, size, streams[{index, codec, channels, layout, sample_rate, bitrate, language, title, atmos, commercial, pretty, engine, engine_encode}]}]}` |
| `audioReference {path}` | info | none | `audioReferenceResult {fps, fps_source, duration_s}` |
| `audioSuggest {reference, items}` | info | none | `audioSuggestResult {suggestions[]}` |
| `encodeResolve {streams, encode}` | info | none | `encodeResolveResult {plans[]}` |
| `encodeDoctor {tools, dirs}` | info | none | `encodeDoctorResult {…same fields as FPSConv /api/doctor}` |
| `encodeSetDee {dee_path, tools, dirs}` | info | none | `encodeSetDeeResult {config, doctor}` |
| `mediainfoReport {path, id?}` | info | none | `mediainfoReportResult {report, resolved, error}` |
| `fpsconvImport {}` | info | none | `fpsconvImportResult {found, settings}` |
| `encodeSettingsLoad {}` / `encodeSettingsSave {partial}` | info | none | `encodeSettingsResult {settings, defaults}` |
| `audioHistoryLoad {}` / `audioHistoryRemove {jobId}` / `audioHistoryClear {}` | info | none | `audioHistoryResult {jobs}` |
| `audioQueue {jobs, workers, tools, dirs}` | **EncodeBridge** | `audioJobStart`, `audioJobProgress {jobId, percent, stage, stages, eta}`, `audioJobLog`, `audioJobDone {jobId, state, output, size, error}`, `audioIdle {summary}`, `audioHeartbeat` | `audioQueueDone` (app exit or crash only) |
| `audioEnqueue {jobs}` | EncodeBridge, **immediate** | `audioEnqueueAck` | none |
| `audioCancelJob {jobId}` / `audioCancelAll` / `audioSetWorkers {n}` | EncodeBridge, **immediate** | `…Ack` | none |
| `packInstall {id: "dolby-tools"}` / `packRemove` | main (existing) | `packProgress` | `packDone` |

- **Immediate commands** run on the stdin reader thread, the same way `cancel` does (`bridge.py:1194-1198`). That way they reach a queue that is already running.
- **The queue session stays open.** It opens on the first Start and stays open while the app runs, sending an idle heartbeat every 60 s. This keeps Rust's 1800 s `EVENT_TIMEOUT` (`lib.rs:19`) from firing during long idle stretches or silent DEE phases. Jobs added mid-run therefore always join the live queue, so there is no race at a batch boundary.
- **Job ids** are UUIDs created in the frontend, not list positions, because jobs arrive while the queue runs.
- **When the queue drains**, `audioIdle` reports how many jobs finished for each task. The frontend turns that into one History entry per page.

### 3.3 Rust: new `src-tauri/src/encode.rs`

Model it on `subsync.rs`.

- **`EncodeBridge(BridgeHandle)`** is managed state, shut down on window close next to the others (`lib.rs:1388-1445`).
- **Commands:**
  - `start_audio_queue` holds the EncodeBridge and forwards every event unchanged as **`encode-event`**, like `subsync-event` (`subsync.rs:173-219`).
  - `audio_send(msg)` uses `send_now` with an allow-list: `audioEnqueue`, `audioCancelJob`, `audioCancelAll`, `audioSetWorkers`.
  - `audio_probe`, `audio_reference`, `audio_suggest`, `encode_resolve`, `encode_doctor`, `encode_set_dee`, `audio_mediainfo`, `fpsconv_import`, `encode_settings_load`, `encode_settings_save` and `audio_history` (load, remove, clear) use the info bridge.
- **Reused as they are:** `pick_media_files`, `subs_pick_files(accept)`, `resolve_dropped_paths`, `open_path`, `reveal_path`, `subs_pack`.
- **Platform guard:** every new command starts with `ensure_windows()?`, which returns `Err("Encode is available on Windows only")` elsewhere. The commands stay registered on every OS, which avoids `cfg` gymnastics inside `generate_handler!`.
- **Crash handling:** if the EncodeBridge process dies, `start_audio_queue` returns `Err`. The page then marks queued and running jobs as failed ("engine stopped"), and the next Start launches a new process.
- **Rust tests:** check the allow-list and that the guard refuses on non-Windows hosts.

### 3.4 Frontend

**New files**
- `src/app/pages/EncodePage.tsx` and `src/app/pages/FpsConvPage.tsx`
- `src/app/pages/audio/`, shared by both pages: `AudioQueue.tsx`, `StreamPicker.tsx`, `ProgressPanel.tsx` (the stage list), `JobLog.tsx`, `MediaInfoModal.tsx`, `audio.css`
- `src/lib/audio/`:
  - `api.ts` (invoke wrappers and the `encode-event` subscription)
  - `types.ts`
  - `store.ts`: one queue store shared by both pages, with a subscription, like `dub/cursorStore.ts`
  - `reducer.ts`: start, event, enqueue, cancel and retry, modelled on `subsync/reducer.ts`
  - `settings.ts`: loads and saves the engine settings file through `encodeSettingsLoad` / `encodeSettingsSave`. This is not localStorage (D9).
  - `format.ts`: chips, the plan label, ETA
  - each with a `*.test.ts`

**Places to edit** (refs from the architecture map)

| File | Change |
|---|---|
| `src/lib/types.ts:6` | Add `"encode" \| "fpsconv"` to `SyncMode` |
| `src/app/dock.tsx:137-143`, `:209` | `PAGE_LABEL` (the compiler enforces it) and the hard-coded page list in the History filter |
| `src/app/pages/analyse/AnalysePage.tsx:47`, `views.tsx:65` | Add the new ids to the analysis `Exclude<…>` |
| `src/ui/frame.tsx:44` | `export const IS_WINDOWS = OS === "windows";` |
| `src/pages/Index.tsx:50-56` | `PAGES` gains **Encode** then **FPSConv**, directly after Subsync, **only when `IS_WINDOWS`** |
| `Index.tsx:565-569` | Mount both pages only when `IS_WINDOWS` |
| `Index.tsx:369` | Derive the Ctrl+number regex from `PAGES.length` instead of the hard-coded `/^[1-5]$/` |
| `Index.tsx:386` | Decide whether Enter starts these pages; the default is yes, through the shell |
| `Index.tsx:70-90` | `dropLabel` for audio extensions |
| `Index.tsx` | If the stored active page is missing from `PAGES`, fall back to `movie` |
| `Index.tsx:441-480` | `buildMenus` and `menuActions`: the Windows menu is drawn in React, so check whether View items come from `PAGES` automatically |
| `src-tauri/src/lib.rs:1324-1339` | **No change.** The native menu is macOS-only and the pages never exist there. |
| `src/app/Preferences.tsx:22`, `:24-30`, `:154-246`, and `shell.tsx:81` | New **Encode** tab, shown only when `IS_WINDOWS` |
| `src/dev/demo.ts:661-762`, `:7-17` | Mock the new invokes and add a `?demo=encode` scenario with fake progress |

**Engine claims.** The audio pages do **not** call `shell.claimEngine`, because they have their own process. Both pages share the queue store, so only one audio queue runs at a time, as in FPSConv.

### 3.5 Preferences → Encode tab (Windows only)

| Section | Contents |
|---|---|
| **Dolby Engine** | `dee.exe` path, Browse, **Apply** (`encodeSetDee` writes deew's `config.toml`), and the DEE DLL verdict row |
| **Dolby tools pack** | Install, Remove and progress through `subs_pack("dolby-tools")`, plus the installed versions of deew, pymediainfo and the patcher |
| **Tools** | Paths for truehdd, `deezy.exe`, MediaInfo CLI (optional), ffmpeg and ffprobe (optional, otherwise AudioSyncMaster discovery or the ffmpeg-full pack), and the patcher folder override. Each has a status dot from `encodeDoctor`. |
| **Flat 7.1** | The `truehdd --bed-conform` checkbox (`encode.bed_conform`) and "flat 7.1 ready / why not" |
| **Folders** | Temp and Working, each with Browse and Open, defaults shown as placeholders |
| **Import** | "Import FPSConv settings" (offered automatically once, the first time the Encode or FPSConv page opens, if FPSConv's `settings.json` exists) |

Everything in this tab is saved to the engine's `settings.json` (D9), so the CLI uses the same tools and folders.

The DESIGN.md rules apply (`design/mockup/DESIGN.md:58-69`): no helper text, one accent colour. FPSConv's long patcher explanation moves to `ENCODE.md` and is linked from the tab.

---

## 4. Windows-only gating, layer by layer

| Layer | On Windows | On macOS and Linux |
|---|---|---|
| Page list, mounts, Ctrl+6/7 | Encode and FPSConv after Subsync | Absent. A stored `encode`/`fpsconv` page falls back to `movie`. |
| CLI subcommands (`convert`, `encode`, …) | Run | Print "Windows only" and exit 2. With no arguments the engine runs normally on every OS. |
| Start-menu "AudioSyncMaster command line" | NSIS install only | None |
| Preferences → Encode tab | Present | Absent |
| Rust commands | Run | `Err("…Windows only")` |
| Python handlers | Run | `error` event with the same text |
| Dolby tools pack | `platforms = Windows x64` | Not offered (the pack system already filters by platform) |
| Engine build | `audiosync.encode` included | Also included: pure Python, a few KB, unreachable. **One build recipe, no conditional builds.** |
| CI tests | All, plus the Windows end-to-end smoke test (§6) | The pure-logic tests and a guard test that checks the refusal |
| Demo / mockup | `?demo=encode` | `?os=windows` forces the Windows layout so the mockup can be reviewed on a Mac |

---

## 5. Phases

Sizes are relative: S = small, M = medium, L = large. Each phase ends with something you can see working.

### Phase 0: Prepare (S)
- **Land or shelve the 18 uncommitted UI files on AudioSyncMaster `main` first.** They change `PBar`, `PageView` and `ui.css`, which the new pages are built on.
- Create branch `feature/encode-fps` from `main`. Nothing merges until Phase 8.
- On the Windows machine, build a sample set:
  - AAC 2.0, AC-3 5.1, E-AC-3 5.1, E-AC-3 JOC, TrueHD 7.1, TrueHD Atmos, DTS 5.1, FLAC 2.0, 7.1 PCM WAV
  - one 23.976 video and one 25 video to use as targets
- **Record a baseline with FPSConv v1.1.24** for every acceptance case in §7: the output file, the job log (the exact command lines) and the MediaInfo text.
- **Done when:** the branch exists and `baseline/` holds outputs, logs and MediaInfo for every case.

### Phase 1: Port the pure engine logic (M)
- Port `formats.py`, `fps.py`, `commands.py`, `progress.py` and `atmos.py`.
- Port all **69** FPSConv unittest methods **plus the 4 orphaned pytest-style FPS tests** (`tests/test_engine.py:428-458`, which `unittest` never runs) into AudioSyncMaster's plain `test_*` style. Add them to `MODULES` in `tests/run_all.py:19-50`.
- **Done when:** `npm run test:py` passes on macOS, and the command-line tests show strings identical to FPSConv's.

### Phase 2: Job runner, queue service, settings, bridge commands and CLI (L)
- Port `settings.py`, `tools.py`, `flat71.py`, `jobs.py` and `service.py`, and add the bridge handlers from §3.2, including the immediate commands and the session that stays open.
- Port the CLI as `cli.py` (§2.4) and add the argument dispatch in `bridge.py`.
- Port the idle heartbeat. **Check that cancelling an encode in flat 7.1 mode leaves the DEE DLL in a known state.** FPSConv recognises an interrupted run by the hash `fd49c7b9…`. AudioSyncMaster kills the whole process tree (`taskkill /T`, `media.py:222-247`), while FPSConv only killed the top process.
- Add bridge contract tests to `tests/test_bridge.py`:
  - every operating system: the guard refuses
  - Windows: probe, resolve, and a real **AAC FPS job** through `audioQueue` (needs ffmpeg only)
- **Done when:** on Windows, both of these work:
  - `python python/bridge.py` runs an AAC 23.976→25 job from stdin, and the output duration equals the source duration ÷ 1.0427083 within one frame
  - `python python/bridge.py convert sample.aac -m 23.976-25 -o out` writes the same file and exits 0

### Phase 3: Dolby tools pack, MediaInfo worker, settings import (M)
- Add a `dolby-tools` entry to `PACKS` (`subs/packs.py:187`):
  - `kind: "python"`, Windows x64 only
  - **pinned** `deew==<the version FPSConv 1.1.24 shipped>` (read it from that release's build log, not `>=2.9`) and `pymediainfo==<pinned>`
  - a new download step for the patcher zip, with the URL and SHA-256 copied from `engine.py:1497-1501`
- Add `mediainfo_worker.py`, called once per batch of added files.
- Add `fpsconv_import.py`.
- **Check first that a `kind:"python"` pack installs on a real Windows machine** (uv download, venv). This is the riskiest step in the port.
- **Done when:** a clean Windows user can go from no tools to a passing doctor with only DEE installed, through Install pack → Apply DEE path.

### Phase 4: Rust (M)
- `encode.rs` as in §3.3, registered in `lib.rs:1392-1429`, plus Rust tests.
- **Done when:** `npm run check:rust` passes, and in `npm run tauri:dev` on Windows a hand-written `invoke("start_audio_queue")` streams `encode-event`s.

### Phase 5: Mockups, then your approval (M) — checkpoint
- Add `design/mockup/screens/encode.tsx`, `fpsconv.tsx` and the Preferences Encode tab, with both pages placed after Subsync in the page bar. Register them in `design/mockup/main.tsx:18-62` and in `shell.tsx` `Page`/`PAGES`.
- **You approve the mockups before any page code is written**, the same way v3 was approved (`DESIGN.md:5-9`).
- **Done when:** you approve the mockups.

### Phase 6: Pages, Preferences tab, store and demo (L)
- Build everything in §3.4 and §3.5 and the demo scenario.
- Add vitest tests for the reducer, store, storage, format helpers, and static render tests for the components.
- **Done when:**
  - `npm run test:all` passes
  - `/?demo=encode` matches the approved mockup
  - on Windows, `tauri:dev` runs an AAC FPS job and a plain DDP 5.1 encode end to end

### Phase 7: Packaging, CI and docs (S–M)
- **Engine build:** add `--collect-submodules audiosync.encode` and `--add-data` for its `workers/` folder in **all three** places: `build-release.yml:236-283`, `dev.sh:41-63` and `dev.bat:49-73`. The two dev scripts already lack `voicefix`, `voicetools` and `voiceworker.py`; fix that in the same change.
- **Windows CI smoke test** after the engine build, run on the **built** `audiosync-cli.exe`:
  - `doctor` exits 0 and reports DEE as missing
  - `convert <generated AAC fixture> -m 23.976-25 -o out` exits 0 and writes the expected file
  - with no arguments, the existing `ping` and analysis smoke tests still pass (`build-release.yml:293-337`)
- **Start-menu shortcut:** add `src-tauri/windows/hooks.nsh` (`NSIS_HOOK_POSTINSTALL` creates "AudioSyncMaster command line", `NSIS_HOOK_PREUNINSTALL` removes it) and point `bundle.windows.nsis.installerHooks` at it in `tauri.windows.conf.json`.
- **Docs:**
  - a new **`ENCODE.md`** in the `SUBSYNC.md` format: tool table, layout, contracts, protocol table, pack, tests, and the flat 7.1 explanation
  - README Modes list (`README.md:37-48`) and Layout tree (`:346-368`)
  - a line in `RELEASING.md` that these pages and the CLI are Windows-only
  - a CLI section in `ENCODE.md`: every subcommand, its flags, exit codes and the exe path
- **Done when:** CI is green on all three operating systems on the branch.

### Phase 8: Windows acceptance, merge, AudioSyncMaster release (M)
- Run the full matrix in §7 on a real Windows machine with **DEE 5.2.1-5994839**, against the Phase 0 baseline.
- Merge to `main`. CI decides the next minor version and releases it to all installed copies (macOS and Linux users see no new pages).
- Install that release on Windows from the **published installer**, not a dev build, and repeat a smoke subset: one FPS AC-3 job, one plain DDP, one flat 7.1, and the CLI cases D1 and D6.
- **Done when:** every row in §7 passes on the released build.

### Phase 9: The FPSConv black-screen release (S)
See §10.2. **Only after Phase 8**, because the black screen sends users to AudioSyncMaster.

### Phase 10: Delete old installers, archive, clean up (S)
See §10.3–10.5. **The order matters:** an archived repo is read-only, so the old releases must be deleted **before** archiving.

---

## 6. Tests

| Suite | Where | Runs on |
|---|---|---|
| Ported FPSConv logic: 69 methods + 4 orphans | `tests/test_encode_formats.py`, `test_encode_commands.py`, `test_encode_progress.py`, `test_encode_atmos.py`, `test_encode_flat71.py`, `test_encode_fps.py` | all 3 OSes |
| Settings file: defaults, coercion, atomic write, 200-job history cap | `tests/test_encode_settings.py` | all 3 OSes |
| CLI: every subcommand's flags parse as in FPSConv; exit codes 0/1/2/130; `--version`; no arguments → engine mode; "Windows only" refusal | `tests/test_encode_cli.py` | all 3 OSes |
| Queue service: enqueue, cancel one, cancel all, live worker change, duplicate skip, drain summary (using fake tool processes) | `tests/test_encode_service.py` | all 3 OSes |
| FPSConv settings import (fixture JSON and TOML) | `tests/test_encode_import.py` | all 3 OSes |
| Bridge contract: guard refusal everywhere; probe and a real AAC FPS job on Windows | `tests/test_bridge.py` | all, plus Windows |
| Rust: allow-list and guard | `encode.rs` `#[cfg(test)]` | Ubuntu (existing job) |
| Reducer, store, storage, format, component render | `src/lib/audio/*.test.ts`, `src/app/pages/audio/*.test.tsx` | vitest |
| Built-engine smoke test | `build-release.yml`, Windows job | Windows |
| **DEE, DeeZy and truehdd** | **Manual only (§7).** DEE is proprietary and can't run in CI. | your Windows machine |

FPSConv's updater, picker and window tests (`test_updater.py`, `test_pick.py`, `test_window.py`) are **not** ported, because AudioSyncMaster has its own updater, pickers and window.

---

## 7. Manual acceptance matrix (Windows, DEE 5.2.1)

Every row is compared with the Phase 0 FPSConv baseline. The job-log command lines must match apart from paths.

| # | Case | Pass when |
|---|---|---|
| A1 | FPS: AAC 2.0, 23.976→25 | AAC output; duration = source × 24000/25025 within 1 frame; progress reaches 100 %. With eac3to set (D12) and an MKV/MKA source the job log shows eac3to's `-23.976 -changeTo25.000` and the pitch is raised; an M4A source stays on `atempo` |
| A2 | FPS: AC-3 5.1, 25→23.976 | `.ac3` 5.1 at the source bitrate |
| A3 | FPS: E-AC-3 5.1, 24→25 | `.ec3` 5.1 |
| A4 | FPS: TrueHD 7.1, 23.976→24 | `.thd` 7.1 |
| A5 | FPS: DTS 5.1 and FLAC 2.0 | AAC output, as in FPSConv (§9.2) |
| A6 | FPS: Target video (25 fps) with mixed rows | Each row's suggestion matches FPSConv's; "no change needed" where the durations match |
| B1 | Encode: DD 5.1, Auto bitrate | `.ac3` at 640 kbps |
| B2 | Encode: DDP 2.0 / 5.1 / 7.1 from PCM | 256 / 1024 / 1536 kbps |
| B3 | Encode: 7.1 source with Channels 2.0 | DEE downmix (`-dm`) and its note |
| B4 | Encode: 2.0 source with Channels 7.1 | Stays 2.0 with a "no upmix" note |
| B5 | Atmos streaming 5.1 from TrueHD Atmos | DDP JOC at 768 kbps (DeeZy) |
| B6 | Atmos Blu-ray, layout "DEE default" | MediaInfo shows `Tfl Tfr`. This is expected: DEE uses that layout by design. |
| B7 | **Flat 7.1** at 1536 kbps | MediaInfo shows **`Lb Rb`** at about **1503 kbps**, and the DEE DLL hash is the original afterwards |
| B8 | Flat 7.1 with a different DEE build (DLL hash differs) | The job **refuses** with the DLL verdict and doesn't fall back |
| B9 | E-AC-3 JOC source with Atmos on | Behaviour from fix #1 in §8 |
| C1 | Two parallel jobs, then add a third while running | The third starts when a slot frees up |
| C2 | Cancel a running flat 7.1 job | Partial output deleted; temp folder gone; DLL verdict is "supported", or "patched (interrupted)" and the next run recovers |
| C3 | Retry a failed job; Overwrite / Skip / Numbered copy on an existing output | Same as FPSConv |
| C4 | MediaInfo on the source and on the output; Open folder | Works |
| C5 | Fresh Windows user: Import FPSConv settings | Tool paths, encode defaults, output folder and DEE path are carried over |
| C6 | Movies analysis running during a long encode | Both make progress; neither blocks the other |
| C7 | macOS build | No Encode or FPSConv page, no Encode tab, Ctrl+6/7 do nothing; `audiosync-cli convert …` prints "Windows only" and exits 2 |
| C8 | Page bar on Windows | Order is … Subsync, **Encode**, **FPSConv**; Ctrl+6 opens Encode and Ctrl+7 opens FPSConv |
| D1 | CLI: `audiosync-cli.exe convert <AC-3 5.1> -m 25-23.976 -o out` | Output and job-log commands identical to case A2 run from the page |
| D2 | CLI: `audiosync-cli.exe encode <TrueHD Atmos> -f ddp --atmos-71 flat -b 1536` | Same result as case B7 |
| D3 | CLI: `doctor`, `probe --target <video>`, `dee <path>`, `patcher`, `--version` | Same information as FPSConv's; `dee` and `patcher` changes show up in the app's Preferences |
| D4 | CLI: Ctrl-C during an encode | Exit code 130; partial output and temp folder removed |
| D5 | CLI: a job fails | Exit code 1 |
| D6 | Start menu → "AudioSyncMaster command line" | A console opens and runs `doctor` |

---

## 8. Bugs found in FPSConv, fixed during the port

| # | Bug | FPSConv today | In AudioSyncMaster |
|---|---|---|---|
| 1 | **E-AC-3 JOC routed into the TrueHD Atmos path** | `is_atmos` is true for DD+ JOC (`tests/test_engine.py:313-322`), and `resolve_encode` doesn't check the codec. A DD+ Atmos source goes to truehdd/DeeZy, which only take TrueHD. The README says the opposite (`README.md:89-90`). | The Atmos path accepts **TrueHD Atmos only**. E-AC-3 JOC is encoded as plain DDP from its bed, with a note. Confirm with a real sample (case B9) before locking it with a test. |
| 2 | **FPS progress stops short of 100 %** on a speed-up | Percent = ffmpeg `out_time` ÷ source duration, but the output is shorter | Divide by the stretched duration (source ÷ ratio) |
| 3 | **Four FPS-detection tests never run** | Module-level pytest functions after `__main__`, skipped by `unittest` | Ported as normal tests |
| 4 | **ffmpeg download not verified** | gyan.dev zip with no hash check (`engine.py:414-446`) | ffmpeg-full pack, SHA-256 checked |
| 5 | **File dialog misses some formats** | `AUDIO_FILTER` lacks `.eb3 .mlp .dts .dtshd .w64` (`window.py:17-18`) | One extension list for the dialog, drops and folders |
| 6 | **Settings changes lost** | The 250 ms debounce sends only the last change in each window (`index.html:621`) | Each change is sent as its own partial save to the engine's settings file |
| 7 | **Local API open to any web page** | HTTP on 127.0.0.1 with no token; any Content-Type parsed as JSON (`server.py:48-55`) | Gone: Tauri IPC, no HTTP server |

---

## 9. Changes you will notice

### 9.1 Not ported, because AudioSyncMaster already covers it or it no longer applies

| FPSConv | Replaced by |
|---|---|
| Its own updater and update banner | AudioSyncMaster's signed Tauri updater |
| Theme toggle, Quit button, desktop-window / browser-mode switch | The AudioSyncMaster shell |
| App log tab | The Output dock |
| In-page fallback file browser | Tauri native dialogs (always available) |
| Desktop notification "Finished: n done, m failed" | A toast plus a screen-reader announcement. AudioSyncMaster has no notification plugin; adding `tauri-plugin-notification` is a small follow-up if you want it. |
| `fpsconv-cli.exe deew / deezy / <script.py>` runners | The Dolby pack's Python |
| The CLI's `gui` subcommand | Not needed. **Every other CLI subcommand is ported** (§2.4). |
| `history.json` (old finished jobs) | Not imported. The output files on disk are untouched. |

### 9.2 Carried over unchanged (candidates for later, not part of this port)
- **FPS change turns DTS, FLAC, PCM and Opus sources into AAC**, so lossless sources come out lossy. AudioSyncMaster's `dubrender` already has a FLAC encoder that a follow-up could use.
- FPS change of TrueHD goes through DEE as `thd` and **drops the Atmos objects**.
- FPS change uses `atempo` (keeps pitch). It offers only 23.976, 24 and 25; there are no 29.97, 30, 50 or 59.94 conversions.
- Outputs are raw elementary streams, with no muxing.
- Folders are added without searching sub-folders.

---

## 10. Retiring FPSConv: the black-screen release

### 10.1 What can and can't be reached

FPSConv v1.1.x checks `https://github.com/AdkHex/FPSConv/releases/latest/download/latest.json` 5 s after launch and every 6 h after that (`updater.py:37-38`). If the version there is higher, it downloads the installer from `url`, checks it against `sha256`, and runs it with `/VERYSILENT … /RESTARTAPPLICATIONS` (`updater.py:93-102`).

| Installed copy | Gets the black screen? |
|---|---|
| Auto-update **on** (the default) | **Yes.** It installs on its own once no job is queued or running, then relaunches into the black screen. |
| Auto-update **off** | **Only if the user clicks** "Install v…". The banner stays, but the update **can't be forced**: v1.1.x has no kill switch or minimum version. |
| Never goes online, firewalled, or the update fails (e.g. antivirus) | No |
| Run from source (`run.sh`), forks, or an old installer someone kept | No |

### 10.2 The tombstone build: FPSConv `VERSION` 1.1 → **1.2**

The release workflow names it `1.2.<run>`, which compares higher than 1.1.24.

| File | Change |
|---|---|
| `fpsconv/__main__.py` | **Any** launch opens a 1320×860 window whose content is **pure black (`#000`) and nothing else**: no text, no buttons, no links (your choice, O1). The window keeps its normal frame, so it can still be closed. Windows needs a title for the taskbar, so it stays "FPSConv". Every CLI subcommand, plus the hidden `deew` / `deezy` / `<script.py>` runners, **prints nothing and exits with code 1**. |
| `fpsconv/server.py`, `queue.py`, `engine.py` | No longer reachable: no HTTP server, no jobs, no downloads. Delete them from the build. |
| `fpsconv/updater.py` | **Keep it running in the background**, with auto-install on. It shows nothing. If the tombstone has a mistake, you can still push a fixed one: unarchive, push, re-archive. |
| `packaging/installer.iss` | Keep the **same AppId** `{7284D80D-…}` so it upgrades in place. Remove the "FPSConv command line" Start-menu entry. Keep the `[Run]` relaunch so users see the screen right after the silent update. Leave `%APPDATA%\FPSConv` alone, because AudioSyncMaster's settings import reads it. |
| `packaging/FPSConv.spec` | Drop deew, pymediainfo and the other runtime dependencies. Keep pywebview. |
| `tests/` | Replace with tests that check the window HTML is an empty black page, every CLI entry exits 1 with no output, and the updater still works. |
| `README.md` | Replace with a short retirement notice. This is the repo page on GitHub, not the app, so the pure-black rule doesn't apply here. |

**Release:** push to FPSConv `main`; `release.yml` builds it and publishes it as **Latest**. Then check that:
- it is a normal release, not a draft or prerelease, because `releases/latest` skips those
- `latest.json` has a non-empty `sha256` (otherwise the hash check is skipped, `updater.py:69-90`)
- installing v1.1.24 by hand on a test machine, with auto-update on, turns it into the black screen within about a minute of launch
- `fpsconv-cli.exe doctor` on that machine prints nothing and exits 1

### 10.3 Delete the old releases (your choice, O4), before archiving
1. **Record the reach baseline first:** save the download counts of every old installer, e.g. `gh release view v1.1.24 -R AdkHex/FPSConv --json assets`. Deleting the releases loses these numbers, and they are the only way to measure reach later (there is no telemetry).
2. **Delete every release except the tombstone** (`gh release delete <tag> -R AdkHex/FPSConv --yes` for each v1.1.x tag). Keep the git tags; they don't offer installers. This removes every working installer from GitHub, so nobody can reinstall a working copy and switch auto-update off. **It can't be undone.**
3. Check that `https://github.com/AdkHex/FPSConv/releases/latest/download/latest.json` still returns the **tombstone's** version.

Copies that already updated are unaffected. Stragglers fetch `releases/latest`, which is still the tombstone. Someone who kept an old installer file can still install it; that can't be prevented.

### 10.4 Archive the repo, for now (your choice, O2)
- **Archive** `AdkHex/FPSConv` (Settings → Archive) **after** step 10.3, because archived repos are read-only and their releases can't be deleted.
- Releases stay downloadable, so `latest.json` keeps resolving and **every straggler that comes online later still gets the black screen**. Download counts stay visible, so compare the tombstone installer's count with the 10.3 baseline now and then.
- **If you ever delete the repo or make it private,** that URL breaks. Every copy that hasn't updated by then gets a 404, shows an update error, and **keeps working forever**. Before doing that, compare the tombstone's download count with the old baseline to judge how many stragglers are left.

### 10.5 Local cleanup (after archiving)
1. `git -C ~/Documents/FPSConv bundle create ~/FPSConv-final.bundle --all`. This keeps `main`, the local-only `master` branch and the old `fpsdee/` and `fpsaudio/` snapshots.
2. Copy the Phase 0 baseline set into AudioSyncMaster's `tests/manual/` notes, or another safe place.
3. Delete `~/Documents/FPSConv` and the `.claude/projects/-Users-santosh-Documents-FPSConv` memory folder.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| A push to AudioSyncMaster `main` ships to every user | All work on `feature/encode-fps`; merge only after §7 passes (D10) |
| `python` packs may never have been installed on a real Windows machine (they aren't restricted to macOS, but there's no proof they've been run there) | Prove it first in Phase 3. If it fails, the fallback is to bundle deew and pymediainfo into the Windows engine build only and add an argv runner to `bridge.py`, as FPSConv's `__main__` did. |
| A silent DEE phase longer than 30 min trips `EVENT_TIMEOUT` | Port the idle heartbeat, plus the 60 s queue heartbeat |
| Cancelling during flat 7.1 leaves the DEE DLL patched | Case C2. The DLL verdict already recognises the patched hash. |
| A new deew release changes its CLI | Pin the exact deew version in the pack |
| FPSConv and AudioSyncMaster both write deew's `config.toml` during the overlap | Low risk. The tombstone ends it. AudioSyncMaster rewrites the config before every queue start. |
| The 18 uncommitted UI files conflict with the new pages | Phase 0 lands or shelves them first |
| Stragglers keep a working FPSConv | Accepted (§10.1). Archiving rather than deleting keeps the door open for them to flip. |
| A pure black window with no text looks like a crash. Users may report bugs or hunt for an old installer. | Accepted (O1). The only explanation is the GitHub README notice. Deleting the old releases (§10.3) removes the easy way back. |
| The app and the CLI write `settings.json` at the same time | Atomic writes, re-reading and merging before every write (the same model as FPSConv's `config.py`) |
| Old releases are deleted by mistake, or the tombstone is deleted with them | §10.3: record download counts, delete only v1.1.x tags, then check that `releases/latest/download/latest.json` still returns the tombstone |

---

## 12. Decisions (answered 2026-10-12)

| # | Question | Answer | Where it's applied |
|---|---|---|---|
| O1 | Tombstone screen | **Pure black, nothing else.** The CLI prints nothing and exits 1. | §10.2 |
| O2 | FPSConv repo | **Archive, for now** | §10.4 |
| O3 | FPSConv's CLI | **Ported** into `audiosync-cli.exe` with the same subcommands and flags | D11, §2.4, Phases 2 and 7, cases D1–D6 |
| O4 | Old v1.1.x installers on GitHub | **Delete** once the tombstone is verified, and before archiving | §10.3 |
| O5 | Page names and place | **"Encode" (Ctrl+6) and "FPSConv" (Ctrl+7), right after Subsync** | D2, §3.4, Phase 5, case C8 |
