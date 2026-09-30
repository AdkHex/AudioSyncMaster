# AudioSyncMaster redesign — mockup v3

A clickable prototype of every page, every step, Preferences and History.

**Status:** approved 2026-09-29, and implemented in `src/` on branch
`redesign/workstation-ui`. `src/ui/ui.css` is this folder's `styles.css`, ported
verbatim with three token renames (`--ac`, `--sunk`, `--inp`); change the two
together. Compare the app (`/?demo=movies&os=windows`, see `src/dev/demo.ts`)
against a screen here (`/design/mockup/#s=<id>`).

- v1 (cards, gradients, helper text) read as AI-generated.
- v2 had a left sidebar, which reads as SaaS.
- v3 is built like desktop media software (DaVinci Resolve, Media Encoder,
  Premiere): a menu bar, a toolbar with a status display, docked panels, and a page
  bar along the bottom. There is no sidebar and no web-style tab strip.

## Open it

- `npm run dev`, then <http://localhost:8081/design/mockup/>
- One file, no server: `node design/mockup/export.mjs` →
  `design/mockup/dist/AudioSyncMaster-mockup.html` (double-click it)
- The left list is the mockup's own screen picker, not part of the app. It covers
  46 screens. *All screens* shows them side by side; Dark/Light and Windows/macOS
  switch live. `←` `→` step, `G` all screens, `T` theme.

## The frame

```
┌ ■ File Edit View Tools Help        Movies — AudioSyncMaster        – □ × ┐  title bar + menu bar
│ + Add videos  ♫ Change dub  🗑  ↦  ≡   [ ⚠ 4 matched · 1 drifting …  ]  ⟳  [Fix 3] │  toolbar + status display
│ ┌ Videos 6 ─────────────── ♫ Dub … ┐ ┌ selected file ──────┐                     │
│ │ list: Ready → progress → result  │ │ −317.0 ms, chart,   │                     │  workspace
│ └──────────────────────────────────┘ └ properties, actions ┘                     │
│ ┌ Timeline | Output | History ───────────────────────────── tools ┐              │  bottom dock (when needed)
│ FFmpeg 6.1 · GPU     ▣ Movies  Series  Find match  Dub sync  Subsync     ⟲ ▤ ⚙   │  page bar
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Menu bar** (Windows): File, Edit, View, Tools, Help, with shortcuts. On macOS the
  menus live in the system bar, so the window shows only the traffic lights.
- **Status display:** the one place that always says what the app is doing.
  - Ready: "6 videos · 1 dub".
  - Running: "Placing the cuts", with a progress line and time left.
  - Done: "4 matched · 1 drifting · 1 different cut · 1 failed".
- **Page bar** (bottom, like Resolve's pages): Movies · Series · Find match · Dub sync
  · Subsync, Ctrl+1–5. A running page shows a small spinner. On the right sit History,
  Output and Preferences.
- **Subsync's tools** sit in a tool strip under the toolbar (Sync, OCR, Translate,
  Frame rate, Generate, Style, HDR subtitles, Tone-map, Convert & mux), the way
  editors show tool modes.
- **Workspace:** docked panels on a darker base.
  - The list shows every file and its live status.
  - The details panel shows only the selection.
  - The bottom dock holds Timeline (Dub sync), Cues (Subsync), Output and History.
- **Preferences** is its own window, with icon tabs: General, Analysis, Dub sync,
  Subtitles (engines and API keys), Updates.

## Rules

- **One accent** for the next action, the selection and progress. Green, amber and
  red appear only as 16 px status icons.
- **No cards, gradients, badges or helper text.** Panels, dividers and spacing do the
  structure.
- **One status vocabulary:** Ready, Waiting, a progress bar, Matched/Done, a warning
  word, Failed.
- **File names keep their end.** The middle is ellipsized, so the release and
  extension stay visible.
- **Windows 11 metrics and Fluent System Icons** (Microsoft's own set); 13 px data
  text, 32 px controls, 36 px rows.

## Building it

1. **`tauri.conf.json`:**
   - `decorations: false` + `transparent: true`;
   - `windowEffects: { effects: ["mica"] }` on Windows;
   - `titleBarStyle: "Overlay"` + `hiddenTitle` on macOS (needs `macOSPrivateApi`);
   - native menus on macOS through Tauri's menu API.
2. **Tokens and icons:** the tokens in `styles.css` replace `src/index.css`.
   `@fluentui/react-icons` replaces lucide.
3. **Shell:** title bar/menu, toolbar + status display, workspace grid, dock and page
   bar become shared components. Each page keeps its state when you switch.
4. **Pages, one at a time:** Movies → Series → Find match → Dub sync → Subsync →
   Preferences/History. The engine and IPC are unchanged.

## Assumptions to confirm

- Dub sync pairs by episode number when names have one, and by file name otherwise
  (no Movies/Series switch).
- History keeps every page (today it keeps Movies and Series only).
- macOS-only engines (Apple Vision, mlx-whisper) show as unavailable on Windows.
- Titles and numbers are invented, in the app's real formats and labels.
