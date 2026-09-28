# Subsync

Subsync is the subtitle side of AudioSyncMaster, a mode beside Dub sync. It
has one tool per job, and each tool has its own engines and settings:

| Tool | What it does | Engines |
|---|---|---|
| **Sync** | Re-times a subtitle to a video: constant offset, frame-rate drift, and different offsets per scene for cut or extended edits | Speech activity of the video's audio · a reference subtitle · the reference release's audio · transcript alignment |
| **Frame rate** | Converts timing between 23.976 / 24 / 25 / 29.97 / 30 / 48 / 50 / 59.94 / 60 / 120, or between two known points | Built in |
| **OCR** | Turns image subtitles (Blu-ray PGS `.sup`, DVD VobSub `.idx/.sub`) into text | Apple Vision · Tesseract · RapidOCR (PaddleOCR models) · Claude vision |
| **Translate** | Translates a subtitle by scene, with a glossary and an optional review pass | Claude · OpenAI-compatible · DeepL · Google · Ollama (local) |
| **Generate** | Makes subtitles for videos that have none (about 100 languages) | mlx-whisper (Apple Silicon) · faster-whisper (any OS) |
| **Style** | Checks and fixes against the Netflix Timed Text Style Guide for the language: line length, reading speed, durations, gaps, shot changes, line breaks | Built in |
| **HDR subtitles** | Keeps subtitles from glaring on HDR / Dolby Vision video: PGS palettes and ASS/SRT colours tone-mapped to a target brightness in nits | Built in |
| **Tone-map video** | Converts HDR10 / HLG / Dolby Vision video to SDR at 4K, 1080p or 720p | 3D LUT (any FFmpeg) · VideoToolbox (macOS) · zscale · libplacebo · tonemapx |
| Convert / Extract / Mux | Converts between formats, pulls tracks out of a video, and adds tracks back in | Built in (mkvmerge when present) |

## Layout

```
audiosync/subs/
  model.py        Cue, SubtitleDoc, Word -- the one shape every engine uses
  languages.py    ISO 639-1/-2 codes, names, CJK set
  tasks.py        job runner: task -> engine module, TaskContext, outputs
  formats.py      read/write SRT, ASS/SSA, WebVTT, TTML, MicroDVD, SBV
  tracks.py       subtitle streams in a video: list, extract, mux; probe_file
  retime.py       offsets, frame-rate conversion, two-point, piecewise maps
  vad.py          speech activity from audio
  sync.py         the sync engines
  pgs.py          HDMV PGS (.sup): parse, render, rewrite palettes, encode
  vobsub.py       DVD VobSub (.idx/.sub): parse, render
  ocr.py          OCR task: preprocessing, engines, furigana, italics, fixes
  style.py        per-language timed-text rules, check, fix, segment words
  translate.py    translation engines and the scene-by-scene pipeline
  generate.py     speech recognition front end (runs inside an engine pack)
  packs.py        optional engine packs: install, status, run workers
  workers/        scripts that run inside a pack's own Python
  hdr_subs.py     subtitle brightness for HDR
  tonemap.py      HDR/DV -> SDR video
native/vision-ocr/  the Apple Vision OCR helper (Swift)
src/lib/subsync/    types, API wrapper, reducer, persisted settings
src/components/subsync/  the Subsync workspace UI
```

## Contracts

These signatures are what the modules call on each other. A module may add
more, but must not change these without updating every caller.

### tasks.py (written first; all engines plug into it)

- `run(job: dict, ctx: TaskContext) -> TaskResult` is the entry point of
  every task module, registered in `tasks.TASKS`.
- `job = {"id", "task", "input": {"subtitle"?, "video"?, "subtitles"?},
  "options": {...}, "output": {...}}`. The key names are exactly those in
  `src/lib/subsync/types.ts` (camelCase).
- `ctx.progress(percent, stage)`, `ctx.log(msg)`, `ctx.check()` (raises
  `Cancelled`), `ctx.token`, `ctx.workdir`, `ctx.secrets`.
- Helpers: `load_subtitle(ref, ctx, fps=None)`,
  `write_subtitle(doc, ref, job, suffix, ctx, language=None)`,
  `output_path(ref, job, suffix, ext, language=None)`, `preview_of(doc)`.
- Raise `TaskError(message)` for anything the user must fix. Its message is
  shown as the job's error.

### formats.py
- `TEXT_FORMATS = ("srt","ass","ssa","vtt","ttml","microdvd","sbv")`
- `detect_format(path) -> str` (one of the above, or "pgs", "vobsub", "unknown")
- `detect_encoding(data: bytes, language: str|None = None) -> str`
- `read(path, encoding=None, fps=None) -> SubtitleDoc`
- `parse(text, fmt, fps=None) -> SubtitleDoc`
- `render(doc, fmt, fps=None) -> str`, `write(doc, path, fmt=None)`
- `resolve_output_format(wanted, source_format) -> str` ("same" means the
  source's format when that is text, otherwise "srt"; "sub" means MicroDVD)
- `extension_for(fmt) -> str`
- `run_convert_task(job, ctx)`

### tracks.py
- `SubtitleTrack` dataclass; `.to_dict()` matches `SubtitleTrackInfo` in types.ts
- `list_tracks(path, token=None) -> list[SubtitleTrack]`
- `extract(path, track, out_dir, token=None) -> str`: text tracks in their own
  format (.srt/.ass/.vtt), PGS as .sup, VobSub as .idx (+ .sub)
- `video_info(path, token=None) -> dict | None` matches `VideoInfo` (HDR kind,
  DV profile, MaxCLL, mastering peak)
- `probe_file(path, token=None) -> dict` matches `ProbedFile`
- `mux(video, subtitles: list[dict], out_path, container="mkv", keep_existing=True, token=None) -> str`
- `run_extract_task`, `run_mux_task`

### retime.py
- `fps_ratio(from_fps, to_fps) -> float` (exact for the 1001 rates)
- `convert_fps(doc, from_fps, to_fps, mode="time") -> SubtitleDoc`
- `two_point(doc, a: (src, dst), b: (src, dst)) -> SubtitleDoc`
- `piecewise(doc, pieces: list[(src_start, src_end, offset_s)]) -> SubtitleDoc`
- `run_task` (task "fps")

### vad.py / sync.py
- `vad.speech_activity(path, audio_track=0, engine="energy", token=None, progress=None) -> (probs: np.ndarray, hop_s: float)`
- `vad.engine_statuses() -> list[dict]`
- `sync.sync_doc(doc, options, video: MediaRef|None, reference, ctx) -> SyncOutcome`
  (`.doc`, `.offset_s`, `.ratio`, `.splits`, `.score`, `.method`, `.describe()`)
- `sync.engine_statuses() -> list[dict]`, `sync.run_task`

### pgs.py / vobsub.py / ocr.py
- `pgs.read_events(path) -> list[BitmapEvent]`, where `BitmapEvent(start, end,
  rgba: np.ndarray HxWx4 uint8, x, y, forced, video_size)` is shared with
  vobsub and defined in pgs.py
- `pgs.adjust_palette(in_path, out_path, fn)` where `fn(y, cb, cr, a) -> (y, cb, cr, a)`
  maps each palette entry (0-255 ints, the PGS YCbCr). Every other byte of
  the stream is kept.
- `pgs.write_sup(events, path, video_size)`, used by tests and burn-in
- `vobsub.read_events(idx_path) -> list[BitmapEvent]`
- `ocr.engine_statuses() -> list[dict]`, `ocr.run_task`

### style.py
- `StyleRules` dataclass; `rules_for(language, options: dict, fps=None) -> StyleRules`
- `segment_words(words: list[Word], rules, language) -> list[Cue]`
- `break_lines(text, rules, language) -> str`
- `check(doc, rules, shot_changes=None) -> list[Issue]` (`Issue.to_dict()`)
- `fix(doc, rules, shot_changes=None) -> (SubtitleDoc, list[Issue])`
- `shot_changes(video_path, token=None, progress=None) -> list[float]`
- `run_task`

### translate.py
- `translate_doc(doc, options: dict, ctx) -> (SubtitleDoc, report: dict)`
- `engine_statuses(secrets: dict) -> list[dict]`, `run_task`

### generate.py / packs.py
- `generate.transcribe(media: MediaRef, options: dict, ctx) -> (words: list[Word], language: str, info: dict)`
- `generate.engine_statuses() -> list[dict]`, `generate.run_task`
- `packs.status() -> list[dict]` (matches `PackStatus`)
- `packs.install(pack_id, progress, log, token, model=None)`, `packs.remove(pack_id, model=None)`
  (with `model`, only that model's weights are deleted from the Hugging Face cache)
- `packs.pack_status(pack_id) -> dict`; statuses may carry `external` (a development
  interpreter from `AUDIOSYNC_PACK_PYTHON_<ID>`) and `outdated`
- Pack ids: `asr-mlx`, `asr-faster` (also provides Silero VAD), `demucs`, `ocr-rapidocr`, `ffmpeg-full`
- `packs.python_for(pack_id) -> str | None`
- `packs.run_worker(pack_id, script, request: dict, ctx) -> dict`
- `packs.ffmpeg_full() -> str | None`: an FFmpeg with zscale/libplacebo, if installed

### hdr_subs.py / tonemap.py
- `hdr_subs.run_task`
- `tonemap.ffmpeg_capabilities(ffmpeg=None) -> dict` (path, version, filters, encoders)
- `tonemap.engine_statuses() -> list[dict]`, `tonemap.run_task`

Every `engine_statuses()` returns dicts shaped like `EngineStatus` in types.ts:
`{"id", "label", "available", "reason", "pack"}`.

## Bridge protocol

The requests are JSON lines on the engine's stdin (see `python/bridge.py`).
Every event below goes to the app as the Tauri event `subsync-event`.

| Request | Events | Terminal |
|---|---|---|
| `{"command":"subsBatch","jobs":[SubJob],"maxWorkers":2,"secrets":{...}}` | `subsJobStart`, `subsJobProgress`, `subsJobLog`, `subsJobDone` | `subsBatchDone` |
| `{"command":"subsProbe","paths":[...]}` | | `subsProbeResult {files: ProbedFile[]}` |
| `{"command":"subsCaps","secrets":{...}}` | | `subsCapsResult {caps: Capabilities}` |
| `{"command":"subsLoad","ref":MediaRef,"limit":5000}` | | `subsLoadResult {preview: CuePreview, error?}` |
| `{"command":"subsSave","path","format","cues":[CueData],"language"}` | | `subsSaveResult {path, error?}` |
| `{"command":"packInstall","pack","model"?}` | `packProgress` | `packDone` |
| `{"command":"packRemove","pack"}` | | `packDone` |

`secrets` travel with the request only. The engine never logs them, never
writes them to disk, and never puts them into a report.

## Engine packs

The shipped engine stays small (numpy only). Speech recognition, Demucs,
RapidOCR and a full-featured FFmpeg are **packs** that the app installs on
demand into the user data folder:

- macOS `~/Library/Application Support/AudioSyncMaster/packs`
- Windows `%LOCALAPPDATA%\AudioSyncMaster\packs`
- Linux `~/.local/share/AudioSyncMaster/packs`
- `AUDIOSYNC_PACKS_DIR` overrides all three

Python packs are separate virtual environments made by `uv` (taken from PATH,
or downloaded into `packs/bin`), each with its own Python 3.11. Engine code
runs there as a worker script (`audiosync/subs/workers/*.py`). The worker
reads one JSON request on stdin and writes JSON lines on stdout
(`progress`, `log`, `result`, `error`). Model weights use the standard
Hugging Face cache, so models already downloaded elsewhere are reused.

## Tests

- `tests/test_subs_*.py` contain plain `test_*` functions, run by
  `tests/run_all.py` and `python/.venv/bin/python tests/test_subs_x.py`.
- They must pass without network, packs or API keys. A test that needs a
  platform tool (`say`, `swiftc`, Apple Vision) skips elsewhere.
- Real-media checks that need packs or large files live in
  `tests/manual/` and are not part of CI.
