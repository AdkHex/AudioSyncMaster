# Handoff: the in-app video player for the dub sync waveform view

This document is a complete brief for building the next feature of
AudioSyncMaster: a video player inside the waveform view, so lip-sync can be
checked frame by frame ("wave match") and the cuts adjusted while watching.
It is written for an engineer (or a coding model) who has not seen this
codebase before. Read it top to bottom before touching code. Everything it
says about the existing code was verified on 2026-09-21 against the working
tree; the tests named here all pass.

The prompt to give the coding model is at the very end (section 9).

---

## 1. Orientation

**What the app is.** A desktop app (Tauri v2: Rust host, React/TypeScript
front end, Python analysis engine) that syncs foreign-language dubs onto
videos. The *Dub sync* tab pairs each video with its dub, analyses where each
stretch of the dub belongs (cuts, offsets, fills from the original), writes
the synced track, and now shows both tracks as waveforms and lets the cuts be
edited by hand.

**Three layers, one wire format.**

| Layer | Where | Talks to |
|---|---|---|
| Engine (Python 3, numpy, ffmpeg) | `audiosync/*.py`, CLI in `python/dubsync.py` | nothing; pure library + ffmpeg subprocesses |
| Bridge (Python) | `python/bridge.py` | reads JSON commands on stdin, one per line; emits JSON events on stdout, one per line. Commands run one at a time on a worker thread; `cancel` and `shutdown` are handled on the reading thread |
| Host (Rust, Tauri) | `src-tauri/src/lib.rs` (commands), `src-tauri/src/bridge.rs` (process handling) | spawns the bridge, forwards commands, turns streamed events into Tauri events for the UI |
| UI (React 18, TS, Tailwind, vitest) | `src/` | calls Rust commands through `src/lib/api.ts`; subscribes to events there too |

All three layers use **camelCase** field names so payloads cross unchanged.

**Two engine processes.** The host runs two bridge processes: the main one
(`BridgeHandle`) for syncs, and `WaveformBridge` for everything the waveform
view asks (peaks, excerpts). The engine takes one command at a time per
process, so this keeps the picture responsive during a two-hour sync. The
player's excerpts go through the waveform process (`render_dub_preview`
already does).

**Commands you will run.**

```sh
npm run typecheck                          # tsc -b
npm run lint                               # eslint (one pre-existing warning in ThemeProvider.tsx is fine)
npx vitest run                             # front-end tests (163 pass today)
npm run check:rust                         # cargo fmt --check + clippy -D warnings (needs ~/.cargo/bin on PATH)
python/.venv/bin/python tests/run_all.py   # every Python module (209 pass today, ~3.5 min)
python/.venv/bin/python tests/run_all.py test_editor          # one module
python/.venv/bin/python tests/run_all.py test_editor frame    # tests whose name contains "frame"
AUDIOSYNC_E2E=1 cargo test --manifest-path src-tauri/Cargo.toml --lib   # Rust tests incl. real-engine E2E (12 pass)
npm run dev                                # vite on http://localhost:8081 (front end only, no engine)
npm run tauri:dev                          # the real desktop app
```

`npm run test:all` runs typecheck + vitest + Rust checks + Python.

**House rules (from the owner's CLAUDE.md; follow them exactly).**

- Never commit. The owner commits ("Update project files") and pushes to
  `main` themselves. CI decides the version; **do not bump versions** in
  package.json / tauri.conf.json / Cargo.toml.
- `.env`, `config.py`, `*.conf` hold live credentials: never read them unless
  the task needs it, never paste their values anywhere.
- Verification means showing the command and its real output. Say when
  something was not run. Never describe untested code as working.
- Report every bug fixed as a table: `| # | Bug | Before | After |`.
- Code style: match the surrounding code — comments explain *why*, prose
  docstrings, no `type(scope):` prefixes anywhere, no emoji.

---

## 2. What already exists (do not rebuild these)

### 2.1 The waveform drawing — `src/components/DubWaveformView.tsx`

A `forwardRef` canvas component. Two lanes on the video's clock: the original
(green) on top, the dub *as the plan would write it* (orange) below; fills
drawn as the hatched original; per-channel min/max/RMS shapes; per-lane auto
gain; ruler; cuts as lines (with grips when `editable`); an optional status
label + progress bar over the ruler.

Props (all exist today):

```ts
plan: DubSyncPlan
fetchPeaks: (request: WaveformRequest) => Promise<WaveformPeaks>
editable?: boolean
selected?: number | null;      onSelect?: (index: number | null) => void
cursorS?: number | null;       onCursor?: (timeS: number) => void
onBoundaryDragStart?/onBoundaryDrag?/onBoundaryDragEnd?/onSplitAt?
onViewChange?: (view: View) => void
laneHeight?: number            // 84 in the strip, 110 in the editor
status?: { label: string; percent: number | null } | null
reading?: Record<string, number>
className?: string
```

Imperative handle (`DubWaveformViewHandle`): `zoomBy(factor)`, `fit()`,
`show(startS, endS)`, `view()`.

The cursor is drawn as a red vertical line at `cursorS`. **The player's
playhead should reuse the cursor** (the parent sets `cursorS` from the
player's clock) — do not add a second line. What is missing is auto-scroll:
a `keepCursorInView?: boolean` prop (see step 5).

Pure helpers live in `src/lib/waveformView.ts` (`View`, `clampView`,
`tickStep`, `peaksKey`, `visibleRequests`) with tests.

### 2.2 The cut editor — `src/components/DubWaveformEditor.tsx`

Full-window overlay (`fixed inset-0 z-50`). Owns the plan (`useState`), undo/
redo, selection, `cursorS`, the toolbar, the piece list and the footer. Uses
`DubWaveformView` with `editable`. Keys (window `keydown` in capture phase):
`Ctrl/⌘+Z` undo, `Shift` redo, `Escape` close (not while `busy`), `← →`
nudge the selected stretch a frame (`Shift`: 10 ms), `,` `.` nudge 1 ms, `S`
split at the cursor. Inputs are ignored by the key handler.

Props: `plan`, `enginePlan?`, `fetchPeaks`, `onPreview(plan, startS, endS)`,
`onApply(plan)`, `onStop?`, `onClose`, `busy?`, `reading?`.

Today's "Play 12 s at the cursor" button calls `onPreview`, which in
`src/pages/Index.tsx` (`handleEditPreview`) renders a `both` excerpt and hands
the file to the OS player. **Keep it** as the fallback, relabelled
"Open in player".

All plan edits are pure functions in `src/lib/dubPlanEdit.ts`
(`moveBoundary`, `nudgeOffset`, `splitAt`, `mergeWithNext`, `convertToFill`,
`convertToDub`, `validatePlan`, `samePlan`, `frameSeconds(plan)`,
`asLoadedPlan`, `codecOfPath`), all tested.

### 2.3 The strip — `src/components/DubWaveformStrip.tsx`

The waveforms at the top of the Dub sync tab. Shows the pair *as loaded*
before a sync (`asLoadedPlan`), the engine's live drafts during one
(`job.draft`), and the final plan after (`job.plan`). Has a pair/job
selector and an "Edit the cuts" button (present when `onEdit` is passed —
`Index.tsx` passes it only for a finished job while the queue is not
running). Props: `pair`, `plan`, `status`, `choices`, `index`, `onChoose`,
`onEdit?`, `fetchPeaks`, `buildWaveform`, `reading`. It renders
`DubWaveformView` read-only with `laneHeight={84}` and no cursor today.

### 2.4 The excerpt service — engine, bridge, host, API

`audiosync/dubrender.py`:

```python
EXCERPT_KINDS = ("both", "audio", "picture", "original")

def excerpt(plan, lo_s, hi_s, output_path, what="both", token=None, log=None) -> (path, start_s, end_s)
def first_frame_offset(video_path, lo_s, token=None) -> float     # 0..1 frame
def frame_aligned_start(video_path, lo_s, token=None) -> float
def preview_span(plan, lo_s, hi_s, output_path, with_video=True, ...) -> path   # old wrapper
```

- `audio`: the synced track over the span as **16-bit WAV** (rendered by the
  same code that writes the final file: crossfades, fill gain, resample —
  sample-exact). Measured: 0.2–0.4 s for 12–30 s of a real film.
- `original`: the video's own sound over the span, 16-bit WAV, at the dub's
  rate/channels, no gain.
- `picture`: 480p H.264 MP4, **no audio stream**, no B-frames, keyframe every
  12 frames (so `<video>` seeks and frame steps are instant). Measured: ~1.3 s
  for 12 s, ~2.9 s for 30 s.
- `both`: picture + synced AAC, for the OS player (the old behaviour).

**Frame alignment (important, already solved — rely on it).** ffmpeg starts
a cut's clock at the first frame it keeps, not at the seek point, so a cut
that starts between frames comes out up to one frame early against sound cut
at the same instant. `excerpt()` therefore moves the start of any cut with a
picture onto the first frame at or after `lo_s` (probed with one tiny ffmpeg
run, ~0.3 s on a 6 GB MKV) and **returns the start it actually used**. The
player must:

1. request the `picture` first,
2. take `startS` from the reply,
3. request `audio` (and `original`) with **that** `startS` and the same `endS`.

`tests/test_editor.py::test_a_picture_excerpt_is_frame_accurate_against_its_sound`
proves flash and click land within 3 ms; `test_the_bridge_cuts_the_players_pieces_on_a_frame`
proves the same through the bridge.

Bridge command (`python/bridge.py`, `handle_dubsync_preview`):

```json
{"command": "dubsyncPreview", "plan": {...}, "startS": 1500.3, "endS": 1530.3, "what": "picture"}
→ {"type": "dubsyncPreviewDone", "path": "/tmp/audiosync-dub-preview-<hash>.mp4", "what": "picture", "startS": 1500.333, "endS": 1530.3}
```

Files are written to the OS temp dir as `audiosync-dub-preview-<hash>.<mp4|wav>`;
the hash covers path, span, kind and the whole plan, so an edited plan gets a
new file. On failure the reply has `path: null` and `error`.

Rust (`src-tauri/src/lib.rs`): `render_dub_preview(request) -> Option<Value>`
on the **waveform** bridge; returns the reply event or `None`.

TypeScript (`src/lib/api.ts`):

```ts
export type DubExcerptKind = "both" | "audio" | "picture" | "original";
export interface DubPreviewRequest { plan; startS; endS; what?: DubExcerptKind; video?: boolean }
export interface DubExcerpt { path: string; what: DubExcerptKind; startS: number; endS: number }
export async function renderDubPreview(request): Promise<DubExcerpt | null>
```

### 2.5 Other facts you will need

- `frameSeconds(plan)` (`src/lib/dubPlanEdit.ts`) gives the frame period from
  `plan.videoFps` (defaults sensibly when null). Use it for stepping and the
  timecode.
- Plans are on the **video's clock**; a dub played at `plan.speed` is already
  accounted for inside the engine's renderer. The player never needs `speed`.
- `Index.tsx` holds `editingJob` (the job open in the editor), `shownPair`
  (the strip's pair/job), `reading` (waveform read progress), the `strip`
  memo, `handleEditPreview`, `handleEditApply`, `handleCancel`.
- The app's CSP is in `src-tauri/tauri.conf.json` (`app.security.csp`). It
  has no `media-src`, so `blob:` media is blocked until step 1 adds it.
- The Tauri asset protocol is **not** enabled. Do not enable it; load excerpt
  bytes through a Rust command (step 1) and make Blob URLs.
- Webviews: WKWebView on macOS, WebView2 on Windows. Both play H.264/AAC MP4
  in `<video>`, both have Web Audio, both have `requestVideoFrameCallback`
  (WebKit since Safari 15.4). Neither plays MKV/AC3/DTS — that is why
  excerpts exist.

---

## 3. The feature, exactly

Decisions already taken with the owner: build **(1) the player in the cut
editor, (2) the same player in the strip, (3) A/B original / dub / both**.
"Instant client-side offset nudges" is **out of scope**.

### 3.1 Behaviour

- **Hidden by default.** Nothing plays and no player is shown until the user
  presses **Play video** (picture + sound) or **Play sample** (sound only,
  no picture wait). Pressing either opens the player panel; **Close** (or
  Escape when nothing else claims it) hides it again and stops playback.
- **What plays is the plan as it stands.** In the editor that is the edited
  plan (undo/redo included); in the strip, the finished plan. The sound is
  the `audio` excerpt — bit-for-bit what the written file will contain.
- **Window.** The player loads a window of **30 s** around the editor's
  cursor: `[cursor − 10 s, cursor + 20 s]`, clamped to the film. The
  picture's actual start comes back from the engine (frame-aligned) and is
  the window's time 0 for everything.
- **Playhead.** While playing (and when stepping), the waveform cursor
  follows the picture's clock (`startS + video.currentTime`, read from
  `requestVideoFrameCallback` when available, else `timeupdate`/rAF). The
  view auto-scrolls to keep the cursor visible (jump so the cursor lands at
  ~20 % of the view width when it leaves the view).
- **Seeking.** Clicking the waveform while the player is open moves the
  cursor; if the time is inside the loaded window the player seeks there;
  if outside, the player reloads a new window around it (showing "Loading…"
  meanwhile) and stays paused/playing as it was.
- **Edits re-render only the sound.** Any plan change while the player is
  open triggers, after a 300 ms debounce, a new `audio` excerpt for the same
  window; when it arrives the sound buffer is swapped at the current
  position without stopping the picture. The picture is never re-rendered
  for an edit. A stale reply (an older request finishing after a newer one)
  is ignored.
- **A/B.** A three-way switch in the transport: **Dub** (default),
  **Original**, **Both** (original in the left ear, dub in the right). The
  `original` excerpt is fetched the first time it is needed and kept for the
  window.
- **Loop.** A toggle that loops a **4 s** region centred on the point where
  the loop was switched on (clamped to the window); when playback passes the
  end it jumps back to the start and keeps going.
- **Frame stepping and timecode.** `[` and `]` step one frame back/forward
  while paused (frame period from `frameSeconds(plan)`); the transport shows
  the current position as `h:mm:ss:ff` plus the frame number, updated from
  the actual presented frame. **Space** plays/pauses. These keys act only
  while the player is open; the editor's existing keys keep working
  (`← →` nudge, `,` `.` 1 ms, `S` split — a split at the paused frame is the
  point).
- **Volume** slider (0–100 %), remembered in the session.
- **Errors.** If an excerpt fails, the player shows the message inline and
  the rest keeps working (a failed `original` only disables that mix; a
  failed `picture` falls back to sound-only playback with a notice).
- **Never blocks the engine.** All excerpts go through the waveform bridge;
  the player must remain usable while a sync runs on the main one.

### 3.2 Placement

- **Editor:** the player docks in the lower area, to the **left of the piece
  list** (the list keeps the remaining width). Picture at most 360 px tall,
  16:9, letterboxed; transport bar under it. When closed, the piece list
  takes the full width as today. Toolbar: replace the single "Play 12 s at
  the cursor" button with three: **Play video**, **Play sample**, and a
  ghost **Open in player** (the old behaviour).
- **Strip:** a **Play video** button next to "Edit the cuts" (only when a
  final plan exists and the queue is not running); the player appears
  **below** the waveform, same component, `laneHeight` unchanged. The strip
  needs a cursor: add `const [cursorS, setCursorS] = useState<number|null>(null)`
  and pass `cursorS`/`onCursor` to `DubWaveformView`.

### 3.3 Not in scope

Client-side offset shifting; playing the raw source files; a full NLE
timeline; keyboard-mappable shortcuts; anything outside the Dub sync tab.

---

## 4. Architecture of the solution

```
 excerpt(plan, lo, hi, "picture") ─┐   Rust read_preview_bytes(path) ──► ArrayBuffer ──► Blob URL ──► <video muted>
 excerpt(plan, lo', hi, "audio")  ─┼─► same ──► AudioBuffer (decodeAudioData) ──┐
 excerpt(plan, lo', hi, "original")┘                                            ├─► GainNodes ─► StereoPanner ─► destination
                                                                                 ┘
   master clock = <video>.currentTime (rVFC mediaTime)   audio source nodes (re)started at the video's time
```

- The picture is muted; the sound is Web Audio. One clock: the video's.
- Audio nodes: for each of `dub` and `original` a `GainNode` (mix + volume)
  feeding a `StereoPannerNode` (0 for Dub/Original alone, −1/+1 in Both),
  into `ctx.destination`. `AudioBufferSourceNode`s are disposable: create,
  `start(when=0, offset=video.currentTime + slack)`, stop on pause/seek/swap.
- Drift check: on each frame callback, `expected = ctx.currentTime −
  startedAtCtx + startedAtMedia`; if `|video.currentTime − expected| > 0.02`,
  restart the sources at the video's time. Log nothing; just correct.
- Blob URLs are revoked when replaced or on unmount. `decodeAudioData`
  detaches the `ArrayBuffer` — copy if you need it twice.

---

## 5. Implementation steps (do them in order; each ends green)

### Step 1 — Bytes to the webview, and the CSP

**Rust** (`src-tauri/src/lib.rs`): add

```rust
/// Bytes of one of the engine's preview excerpts, for the player. Only
/// files the engine wrote for that purpose are served: inside the OS temp
/// dir, named `audiosync-dub-preview-*`.
#[tauri::command]
fn read_preview_bytes(path: String) -> Result<tauri::ipc::Response, String> { ... }
```

Rules: canonicalise both the path and `std::env::temp_dir()` (macOS temp is
`/var/folders/...` → `/private/var/...`), require the file to be inside the
temp dir and its name to start with `audiosync-dub-preview-`, then
`fs::read` and return `tauri::ipc::Response::new(bytes)`. Register it in
`generate_handler!`. Unit-test the guard (a temp file with the prefix is
served; one without the prefix, or outside the temp dir, is refused) — no
engine needed.

**CSP** (`src-tauri/tauri.conf.json`, `app.security.csp`): add
`media-src 'self' blob:;`. Keep everything else.

**TS** (`src/lib/api.ts`):

```ts
export async function readPreviewBytes(path: string): Promise<ArrayBuffer> {
  requireDesktop("Loading a preview");
  return invoke<ArrayBuffer>("read_preview_bytes", { path });
}
```

Done when: `npm run check:rust` clean, the Rust unit test passes, typecheck
clean.

### Step 2 — Pure clock helpers: `src/lib/previewClock.ts` (+ tests)

```ts
export const WINDOW_BEFORE_S = 10, WINDOW_AFTER_S = 20, LOOP_S = 4, DRIFT_S = 0.02;
export function windowAround(cursorS: number, durationS: number): { startS: number; endS: number }
export function loopAround(atS: number, startS: number, endS: number): { a: number; b: number }
export function timecode(t: number, frameS: number): string      // "h:mm:ss:ff", frames zero-padded to 2
export function frameIndex(t: number, frameS: number): number     // floor with 1e-6 tolerance
export function stepFrame(t: number, frameS: number, direction: -1 | 1, startS: number, endS: number): number
export function needsResync(videoTime: number, audioTime: number): boolean
export function inWindow(t: number, startS: number, endS: number): boolean
```

`vitest` tests: window clamping at both ends; loop clamping; timecode at
24, 25, 29.97 fps including the rounding edge (`t = 1.0 − 1e-9` is frame 23,
not 24); stepping from the last frame stays put; drift threshold.

### Step 3 — Loading: `src/lib/useExcerpt.ts` (+ tests)

A hook that owns the loaded window for a plan:

```ts
export interface ExcerptSources {
  startS: number; endS: number;             // the window actually cut (from the picture, or the audio when picture-less)
  pictureUrl: string | null;                 // Blob URL, or null in sample mode / on failure
  dub: AudioBuffer | null;
  original: AudioBuffer | null;              // fetched lazily
  loading: "picture" | "audio" | "original" | null;
  error: string | null;
}
export function useExcerpt(args: {
  plan: DubSyncPlan;
  wantedStartS: number | null;   // null = nothing loaded
  wantedEndS: number | null;
  withPicture: boolean;
  wantOriginal: boolean;
  audioContext: AudioContext | null;
  render: (r: DubPreviewRequest) => Promise<DubExcerpt | null>;   // api.renderDubPreview
  read: (path: string) => Promise<ArrayBuffer>;                   // api.readPreviewBytes
}): ExcerptSources
```

Logic:
1. When `wantedStartS/EndS/withPicture` change: request `picture` (if
   `withPicture`), take its `startS`, then `audio` for `[startS, endS]`;
   otherwise request `audio` for the wanted span directly. Make the Blob
   URL for the picture (`new Blob([bytes], { type: "video/mp4" })`), decode
   the audio (`audioContext.decodeAudioData(bytes)`).
2. When `plan` changes (any edit): after a 300 ms debounce, re-request
   `audio` only, for the **same** `[startS, endS]`, and replace `dub`.
3. When `wantOriginal` first becomes true: request `original` for the same
   span once and keep it.
4. Every request carries a sequence number; a reply whose number is not the
   latest for its kind is dropped. Unmount revokes Blob URLs.
5. The `original` is invalidated when the window changes, not when the plan
   changes (the original does not depend on the plan).

Tests (`vitest`, with fake `render`/`read` and a fake `AudioContext` whose
`decodeAudioData` resolves an object): the picture's returned `startS` is
used for the audio; an edit re-requests only audio for the same span; a
stale audio reply is ignored; `original` is requested once and only when
wanted; debounce coalesces three quick edits into one request.

### Step 4 — The player: `src/components/DubPreviewPlayer.tsx` (+ tests)

Props:

```ts
plan: DubSyncPlan;
frameS: number;                       // frameSeconds(plan)
cursorS: number;                      // the editor's / strip's cursor
onCursor: (t: number) => void;        // playhead → cursor (call at most once per frame)
withPicture: boolean;                 // Play video vs Play sample
sources: ExcerptSources;              // from useExcerpt
onNeedWindow: (aroundS: number) => void;   // ask the parent to move the window (cursor outside)
onClose: () => void;
audioContext: AudioContext;           // created by the parent on the button press (user gesture!)
```

Behaviour: everything in 3.1. Structure:

- `<video muted playsInline>` with `src={sources.pictureUrl}`; when
  `withPicture` is false, keep a hidden `<video>`? **No** — in sample mode
  there is no picture; use the AudioContext clock as master instead
  (`ctx.currentTime − startedAtCtx + startedAtMedia`), ticking with rAF.
  Write the transport so the "clock" is one small object with
  `now()`, `seek(t)`, `play()`, `pause()`, implemented twice (video-backed,
  context-backed).
- Web Audio graph as in section 4. Mix switch sets gains: Dub → dub 1 /
  original 0 / pans 0; Original → 0 / 1 / 0; Both → 1 / 1 with dub pan +1,
  original pan −1.
- Sources restart on play, seek, buffer swap, mix change is gain-only (no
  restart).
- Frame callback: `video.requestVideoFrameCallback` if present, else
  `requestAnimationFrame` reading `video.currentTime`. It updates the
  timecode, calls `onCursor(startS + mediaTime)`, checks the loop end and
  the drift.
- Keys are handled by the **parent** (editor/strip) and forwarded through a
  small imperative handle: `togglePlay()`, `step(±1)`, `toggleLoop()`; or
  the player registers its own window listener in capture phase and stops
  propagation for Space/`[`/`]` only. Either is fine; document the choice.
- Seeking from the parent: when `cursorS` changes and was **not** produced
  by this player's last `onCursor` call (keep the last emitted value in a
  ref), then if `inWindow` → `clock.seek(cursorS − startS)`, else
  `onNeedWindow(cursorS)`.
- Transport bar contents: Play/Pause, `[ ]` step buttons, loop toggle,
  timecode `h:mm:ss:ff` + frame number, mix switch (Dub / Original / Both),
  volume slider, "Loading…" / error text, Close. Keep it one row, 30 px
  controls, the app's `Button`/`Tag` primitives from `src/components/ui.tsx`.

Tests: SSR (`renderToStaticMarkup`, like the other component tests) for
the transport's presence and the sample-mode layout; pure-function tests
for the mix table (gains/pans for the three modes). Do not try to unit-test
Web Audio scheduling; that is covered by the browser check in step 8.

### Step 5 — Auto-scroll in `DubWaveformView`

Add `keepCursorInView?: boolean`. In an effect on `cursorS`: if the prop is
true and `cursorS` is outside `[view.startS, view.endS]`, call `setView`
so the cursor sits at 20 % of the view width (same length). Do nothing
while the user is dragging or panning (`drag !== null`). Test the maths in
`waveformView.ts` as a pure helper (`followCursor(view, cursorS, share)`).

### Step 6 — Editor integration

In `DubWaveformEditor.tsx`:

- State: `player: { mode: "video" | "sample" } | null`, `window: { startS, endS } | null`,
  `audioContext` (created in the button handler: `new AudioContext()`; call
  `resume()` there too — it is the user gesture).
- Toolbar: **Play video** → `window = windowAround(cursorS, duration)`,
  `player = { mode: "video" }`; **Play sample** likewise with `"sample"`;
  **Open in player** = the old `onPreview` behaviour.
- Render `DubPreviewPlayer` left of the piece list when `player` is set,
  with `sources = useExcerpt({ plan, wantedStartS: window?.startS ?? null, ... })`.
  `plan` here is the **edited** plan, so edits re-render the sound.
- `onCursor` from the player → `setCursorS`; pass `keepCursorInView={player !== null}`
  to the view. `onNeedWindow(t)` → `setWindow(windowAround(t, duration))`.
- Keys: Space / `[` / `]` only when `player` is set. Escape closes the
  player first; a second Escape closes the editor.
- "Write the track with these cuts" while the player is open: allowed;
  close the player first (stop audio) to keep things simple.

### Step 7 — Strip integration

In `DubWaveformStrip.tsx`: cursor state; a **Play video** button (shown when
`plan` is set, `status` is null and an `onPlayable` condition from the
parent is true — `Index.tsx` should pass a boolean `playable` = job done and
queue not running); the player below the waveform with the same hook and
component; `withPicture` always true here (sample mode is an editor thing).
`Index.tsx` passes `fetchPeaks`, `renderDubPreview` and `readPreviewBytes`
from `api` (do not import `api` inside the components — keep them injectable
like the existing ones, so tests render without Tauri).

### Step 8 — Verification in a real browser (required)

The components can be exercised without Tauri, as was done for the waveform
view. Create a throwaway harness (delete it before finishing):

- `dev-editor.html` at the repo root + `src/dev/editor.tsx` that mounts
  `DubWaveformEditor` (and the strip) with the fixture plan
  `src/lib/__fixtures__/dubsync-outcome.json`, synthetic peaks, and a fake
  `renderDubPreview`/`readPreviewBytes` that serve **real** files: generate a
  30 s picture MP4 and a WAV with ffmpeg into `public/dev/` (e.g. a colour
  bar clip with a burnt-in timecode via `drawtext`, and a tone with a click
  each second), and fetch them with `fetch('/dev/…')`. The burnt-in
  timecode lets you verify frame stepping and the displayed timecode agree.
- `npm run dev`, open `http://localhost:8081/dev-editor.html`, and check:
  play/pause, playhead following with auto-scroll, seek by clicking, frame
  stepping matches the burnt-in frame counter, loop, mix switch (Both = one
  side each), an edit re-renders sound without a picture reload, cursor
  outside the window reloads the window, close stops the sound, no console
  errors. Take screenshots for the report.
- Then the real app: `npm run tauri:dev` with a real pair, open the editor,
  Play video, confirm sound and picture together, confirm the Stop of a
  running sync still works while the player is open.

### Step 9 — Docs and report

- README: add a short "Playing it back" paragraph under *Editing the cuts*
  (what the buttons do, the keys, that what you hear is the written file's
  sound, the 30 s window, the A/B switch).
- Final report: what was built (per step), every command run with its
  output, the bug table if anything pre-existing was fixed, what was not
  verified and why.

---

## 6. Pitfalls (each of these has bitten before)

- **User gesture.** Create/resume the `AudioContext` and call `video.play()`
  inside the click handler's synchronous path; WKWebView refuses otherwise.
- **`blob:` and the CSP.** Without `media-src blob:` the video shows nothing
  and the console says CSP. Do step 1 first.
- **Do not re-snap the sound.** The picture's returned `startS` is already on
  a frame. Ask for `audio`/`original` with exactly that start; never call the
  picture path for them.
- **`decodeAudioData` detaches the buffer** you pass it. Do not reuse it.
- **`requestVideoFrameCallback` must be re-requested** inside its own
  callback each frame, and cancelled on unmount (`cancelVideoFrameCallback`).
- **Keep clocks in refs, not state.** State updates per frame re-render the
  whole editor. Only the timecode text and the cursor need updating, and the
  cursor goes through `onCursor` once per frame.
- **Escape** is used by the editor to close; the player must claim it first
  while open (capture-phase listener, `stopPropagation`).
- **The editor's key handler ignores inputs** (`INPUT`, `TEXTAREA`,
  contentEditable) — keep that; the volume slider is an input.
- **Alt+Arrow** is a browser back gesture on some platforms — do not bind it.
- **Two processes.** Excerpts go through `WaveformBridge`; if you add a Rust
  command for the player, use `State<'_, WaveformBridge>`, not `BridgeHandle`.
- **`render_dub_preview` returns `Option<Value>`**, not a path string (this
  changed on 2026-09-21).
- **Temp files accumulate** (a few MB each). Acceptable for now; do not add
  a cleanup that could delete a file still loading. If you add one, delete
  only files older than an hour, at app start.
- **Do not touch** `scripts/decide-version.cjs`, `scripts/set-version.cjs`,
  the CI workflow, or version numbers.

---

## 7. Definition of done

- Steps 1–7 implemented; step 8 performed with screenshots; step 9 written.
- `npm run typecheck`, `npm run lint` (no new warnings), `npx vitest run`,
  `npm run check:rust`, `python/.venv/bin/python tests/run_all.py`,
  `AUDIOSYNC_E2E=1 cargo test --manifest-path src-tauri/Cargo.toml --lib`
  all green, outputs pasted in the report.
- No commits made; no version bumps; no harness files left behind
  (`dev-editor.html`, `src/dev/`, `public/dev/`).
- The old behaviours still work: "Open in player" (OS player), editing and
  writing the track, the strip's live drafts, Stop during a sync.

---

## 8. Quick reference: the existing types you will touch

```ts
// src/lib/types.ts
interface DubSegment { kind: "dub" | "fill"; startS; endS; sourceStartS; offsetS: number | null; match; note; uncertaintyS }
interface DubSyncPlan { videoPath; dubPath; videoTrack; dubTrack; speed; videoFps: number | null; dubRate; fillGainDb;
                        videoDurationS; dubDurationS; segments: DubSegment[]; warnings; notes?; error; filledS; rateConfirmed? }
// src/lib/api.ts
renderDubPreview(request: DubPreviewRequest): Promise<DubExcerpt | null>
// src/lib/dubPlanEdit.ts
frameSeconds(plan): number
```

Python entry points: `audiosync/dubrender.py: excerpt, first_frame_offset,
frame_aligned_start`; `python/bridge.py: handle_dubsync_preview`.

Tests to read first for style: `src/components/DubWaveformEditor.test.tsx`,
`src/lib/waveformView.test.ts`, `tests/test_editor.py`.

---

## 9. The prompt (give this to the coding model verbatim)

```
You are working in the AudioSyncMaster repository (Tauri v2 desktop app: Rust host in src-tauri/, React + TypeScript front end in src/, Python engine in audiosync/ and python/bridge.py). Read HANDOFF-video-player.md at the repository root in full before doing anything else; it is the specification, the plan and the rules for this task. Then read the files it names in sections 2 and 8.

Build the in-app video player for the dub sync waveform view exactly as sections 3–7 of that document describe: hidden by default; "Play video" and "Play sample" buttons in the cut editor (and "Play video" in the strip) open a player that shows a muted 480p picture excerpt of the 30 s window around the cursor and plays the synced dub's sound through the Web Audio API locked to the picture's clock; the waveform cursor follows the playhead and auto-scrolls; clicking the waveform seeks; edits re-render only the sound after a 300 ms debounce and swap it in at the current position; a Dub / Original / Both switch (Both = original left ear, dub right); a 4 s loop; frame stepping with [ and ] and a h:mm:ss:ff timecode from the presented frame; Space plays and pauses; Close stops everything. The engine, bridge and API for the excerpts already exist and are frame-accurate — use them as documented (picture first, then the sound for the start the picture reply returns). Do not implement client-side offset shifting.

Work through the steps of section 5 in order and keep the tree green after each one (typecheck, lint, vitest, check:rust, the Python suite). Write the tests the plan lists. Do step 8 in a real browser with the throwaway harness and take screenshots; then delete the harness. Follow the house rules in section 1: never commit, never bump a version, never read or print credential files, show every verification command with its actual output, say plainly what was not run, and report any pre-existing bug you fix as a | # | Bug | Before | After | table. Match the surrounding code's style (comments explain why; no type(scope): prefixes; no emoji). When you are done, write the final report described in section 5, step 9, and check every line of section 7.

If something in the document turns out to be wrong about the code, say so in the report and follow the code, not the document.
```
