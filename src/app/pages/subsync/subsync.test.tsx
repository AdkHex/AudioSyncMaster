import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";

import { ShellContext, type Shell } from "@/app/shell";
import { DEFAULT_SETTINGS } from "@/lib/types";
import { buildItems, buildJobs, emptyPools, poolsReducer, type InputPools } from "@/lib/subsync/inputs";
import { initialRunState, runReducer, type RunState } from "@/lib/subsync/reducer";
import { DEFAULT_OPTIONS, type Capabilities, type CueData, type ProbedFile, type SubJobOutcome, type SubTask } from "@/lib/subsync/types";

import { SubsyncPage } from "../SubsyncPage";
import { CueTable } from "./CuesDock";
import { resetEngines } from "./engines";
import { EnginesSection } from "./EnginesSection";
import { FilesTable, MuxTable } from "./FileTable";
import { ResultView } from "./ResultView";
import { doneLcd, historyLine, needsCheck, readyLcd, rowStatus, runningLcd, shortResult, splitsText } from "./rows";

const noop = () => undefined;

// ------------------------------------------------------------------ fixtures

function video(path: string, audio = 1): ProbedFile {
  return {
    path,
    name: path.replace(/^.*\//, ""),
    kind: "video",
    duration: 9828,
    video: {
      width: 1920, height: 1080, fps: 24000 / 1001, codec: "h264", hdr: "sdr", dvProfile: null, dvCompatibility: null,
      maxCll: null, masteringPeak: null, transfer: null, primaries: null,
    },
    audioTracks: Array.from({ length: audio }, (_, index) => ({ index, codec: "eac3", language: "eng", title: null, channels: 6 })),
    subtitleTracks: [],
    subtitle: null,
  };
}

function subtitle(path: string, kind: "text" | "image" = "text"): ProbedFile {
  return {
    path,
    name: path.replace(/^.*\//, ""),
    kind: "subtitle",
    duration: null,
    video: null,
    audioTracks: [],
    subtitleTracks: [],
    subtitle: { format: kind === "text" ? "srt" : "pgs", kind, cues: 1204, language: "en", encoding: "utf-8" },
  };
}

function poolsWith(task: SubTask, files: ProbedFile[]): InputPools {
  let pools = poolsReducer(emptyPools(), {
    type: "add",
    task,
    files: files.map((file) => ({ name: file.name, path: file.path, type: file.kind === "unknown" ? "video" : file.kind, size: 1 })),
  });
  pools = poolsReducer(pools, { type: "probed", files });
  return pools;
}

/** The mockup's Sync page: two subtitles of the film, and a Hindi one the
 *  user has taken off the video ("No video"). */
const SYNC_FILES = [
  subtitle("/m/Blade.Runner.2049.2017.1080p.BluRay.en.srt"),
  subtitle("/m/Blade.Runner.2049.2017.1080p.BluRay.en.forced.srt"),
  subtitle("/m/Blade.Runner.2049.hi.srt"),
  video("/m/Blade.Runner.2049.2017.1080p.BluRay.mkv", 2),
];

function syncPools(): InputPools {
  return poolsReducer(poolsWith("sync", SYNC_FILES), { type: "setPair", task: "sync", subtitle: "/m/Blade.Runner.2049.hi.srt", video: null });
}

const outcome = (job: number, id: string, patch: Partial<SubJobOutcome> = {}): SubJobOutcome => ({
  job,
  id,
  task: "sync",
  outputs: [{ path: `/m/out-${job}.synced.en.srt`, kind: "subtitle", format: "srt", language: "en", label: null }],
  report: { offsetMs: 3250, method: "audio", splits: [] },
  summary: "Matched against the video's speech: shifted by +3.250 s.",
  warnings: [],
  preview: null,
  ...patch,
});

function syncRun(): { run: RunState; keys: string[]; skipped: number } {
  const pools = syncPools();
  const items = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
  const built = buildJobs("sync", items, DEFAULT_OPTIONS.sync, {}, (i) => `j${i}`);
  let run = runReducer(initialRunState, {
    type: "started",
    task: "sync",
    at: 0,
    jobs: built.jobs.map((job, i) => ({ id: job.id, label: built.labels[i] })),
  });
  run = runReducer(run, {
    type: "finished",
    outcomes: [
      outcome(0, "j0", { report: { offsetMs: 3250, splits: [{ atS: 1230, offsetMs: -127400 }, { atS: 3070, offsetMs: -238000 }] } }),
      outcome(1, "j1"),
    ],
  });
  return { run, keys: built.keys, skipped: items.length - built.jobs.length };
}

// ------------------------------------------------------------------- rows

describe("the status display", () => {
  it("says what is missing before a run", () => {
    const pools = syncPools();
    const items = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
    expect(readyLcd("Sync", "subtitle", items, pools.sync.files)).toEqual({
      icon: "warn",
      l1: "2 of 3 ready",
      l2: "Blade.Runner.2049.hi.srt needs a video",
    });
    expect(readyLcd("Sync", "subtitle", [], [])).toEqual({ l1: "Sync: drop subtitles and their videos" });
    const translate = poolsWith("translate", [subtitle("/k/Goblin.S01E01.ko.srt"), subtitle("/k/Goblin.S01E02.ko.srt")]);
    expect(readyLcd("Translate", "subtitle", buildItems("translate", translate.translate, DEFAULT_OPTIONS.translate), translate.translate.files)).toEqual({
      l1: "Translate · 2 files",
      l2: "Ready",
    });
  });

  it("counts the mux inputs", () => {
    const pools = poolsWith("mux", [video("/m/Film.mkv"), subtitle("/m/Film.en.srt"), subtitle("/m/Film.hi.srt"), subtitle("/m/Film.en.forced.srt")]);
    const items = buildItems("mux", pools.mux, DEFAULT_OPTIONS.mux);
    expect(readyLcd("Convert & mux", "mux", items, pools.mux.files)).toEqual({ l1: "Convert & mux · 1 video, 3 subtitles", l2: "Ready" });
  });

  it("follows the running job, then sums up the batch", () => {
    let run = runReducer(initialRunState, { type: "started", task: "sync", at: 0, jobs: [{ id: "a", label: "Movie.en.srt" }, { id: "b", label: "Movie.fr.srt" }] });
    run = runReducer(run, { type: "event", event: { type: "subsJobProgress", job: 0, percent: 50, stage: "Listening for speech" } });
    expect(runningLcd("Sync", run)).toEqual({ icon: "run", l1: "Sync: listening for speech", l2: "Movie.en.srt", pct: 25, time: "25%" });

    const { run: done, skipped } = syncRun();
    expect(skipped).toBe(1);
    expect(doneLcd("sync", done, skipped)).toMatchObject({ icon: "warn", l1: "2 synced · 1 skipped" });
    expect(historyLine("sync", done, skipped)).toEqual({
      name: "Blade.Runner.2049.2017.1080p.BluRay.en.srt and 1 more",
      tone: "warn",
      text: "2 synced · 1 skipped",
    });
  });

  it("names a single file's result, and a failed batch", () => {
    let run = runReducer(initialRunState, { type: "started", task: "sync", at: 0, jobs: [{ id: "a", label: "Movie.en.srt" }] });
    run = runReducer(run, { type: "finished", outcomes: [outcome(0, "a")] });
    expect(doneLcd("sync", run, 0)).toEqual({ icon: "ok", l1: "1 synced", l2: "Movie.en.srt → out-0.synced.en.srt" });
    expect(historyLine("sync", run, 0)).toEqual({ name: "Movie.en.srt", tone: "ok", text: "Synced +3.250 s" });

    const failed = runReducer(runReducer(initialRunState, { type: "started", task: "ocr", at: 0, jobs: [{ id: "a", label: "x.sup" }] }), {
      type: "failed",
      message: "The engine stopped.",
    });
    expect(doneLcd("ocr", failed, 0)).toEqual({ icon: "bad", l1: "The run did not finish", l2: "The engine stopped." });
  });

  it("says how many OCR lines were read and are left to check", () => {
    let run = runReducer(initialRunState, { type: "started", task: "ocr", at: 0, jobs: [{ id: "a", label: "Your.Name.2016.jpn.sup" }] });
    run = runReducer(run, {
      type: "finished",
      outcomes: [
        outcome(0, "a", {
          task: "ocr",
          report: { cues: 1204, lowConfidence: 17 },
          outputs: [{ path: "/n/Your.Name.2016.ja.srt", kind: "subtitle", format: "srt", language: "ja", label: null }],
        }),
      ],
    });
    expect(doneLcd("ocr", run, 0)).toEqual({ icon: "warn", l1: "1,204 lines read · 17 to check", l2: "Your.Name.2016.jpn.sup → Your.Name.2016.ja.srt" });
    expect(historyLine("ocr", run, 0).text).toBe("1,204 lines read · 17 to check");
  });
});

describe("row status and result", () => {
  it("reads short results from each task's report", () => {
    expect(shortResult(outcome(0, "a"))).toBe("+3.250 s");
    expect(shortResult(outcome(0, "a", { task: "ocr", report: { cues: 1204 } }))).toBe("1,204 lines");
    expect(shortResult(outcome(0, "a", { task: "fps", report: { ratio: 25 / (24000 / 1001) } }))).toBe("×1.042708");
    expect(shortResult(outcome(0, "a", { task: "tonemap", report: { outputResolution: "1920x1080" } }))).toBe("1080p");
    expect(shortResult(outcome(0, "a", { error: "no audio" }))).toBeNull();
  });

  it("says where a synced subtitle's offset changes", () => {
    expect(splitsText({ offsetMs: 3250, splits: [{ atS: 1230, offsetMs: -127400 }, { atS: 3070, offsetMs: -238000 }] })).toBe("2 scenes cut");
    expect(splitsText({ offsetMs: 0, splits: [{ atS: 600, offsetMs: 42000 }] })).toBe("1 scene added");
    expect(splitsText({ offsetMs: 0, splits: [{ atS: 600, offsetMs: 42000 }, { atS: 900, offsetMs: 1000 }] })).toBe("2 scene changes");
    expect(splitsText({ splits: 3 })).toBe("3 scene changes");
    expect(splitsText({ offsetMs: 3250, splits: [] })).toBeNull();
  });

  it("marks what a run skipped, and why", () => {
    const pools = syncPools();
    const items = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
    const hindi = items.find((item) => item.subtitleName === "Blade.Runner.2049.hi.srt")!;
    expect(rowStatus(hindi, undefined, null, false, false)).toMatchObject({ s: "warn", text: "Needs a video", title: "Choose the video to sync it to." });
    expect(rowStatus(hindi, undefined, null, true, true)).toMatchObject({ s: "wait", text: "Skipped · no video" });
    expect(rowStatus(hindi, undefined, null, true, false)).toMatchObject({ s: "warn", text: "Skipped · no video" });
  });
});

// ------------------------------------------------------------- files list

describe("FilesTable", () => {
  const table = (run: RunState | null, keys: string[] = []) => {
    const pools = syncPools();
    const items = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
    const jobs = new Map(run ? run.jobs.map((job, i) => [keys[i], job] as const) : []);
    return renderToStaticMarkup(
      <FilesTable
        task="sync"
        pool={pools.sync}
        items={items}
        options={DEFAULT_OPTIONS.sync}
        jobs={jobs}
        ranHere={!!run}
        running={false}
        current={items[0].key}
        onSelect={noop}
        disabled={false}
        dispatch={noop}
      />,
    );
  };

  it("pairs each subtitle with its video, and asks for the one that has none", () => {
    const html = table(null);
    expect(html).toContain("Blade.Runner.2049.2017.1080p.BluRay.mkv");
    expect(html).toContain("Choose a video");
    // The video has two audio tracks: each paired row lets you pick one.
    expect(html.match(/aria-label="Audio track of Blade.Runner.2049.2017.1080p.BluRay.mkv"/g)).toHaveLength(2);
    expect(html).toContain("Needs a video");
    expect(html.match(/>Ready</g)).toHaveLength(2);
  });

  it("shows each job's result and status after a run", () => {
    const { run, keys } = syncRun();
    const html = table(run, keys);
    expect(html.match(/>\+3\.250 s</g)).toHaveLength(2);
    expect(html).toContain("Done · 2 scenes cut");
    expect(html).toContain(">Done<");
    expect(html).toContain("Skipped · no video");
  });

  it("lists mux tracks with their language, title and flags", () => {
    const pools = poolsWith("mux", [video("/m/Film.mkv"), subtitle("/m/Film.en.srt"), subtitle("/m/Film.fr.forced.srt")]);
    const items = buildItems("mux", pools.mux, DEFAULT_OPTIONS.mux);
    const html = renderToStaticMarkup(
      <MuxTable task="mux" pool={pools.mux} item={items[0]} current="/m/Film.mkv" onSelect={noop} disabled={false} dispatch={noop} />,
    );
    expect(html).toContain("Video, 1 audio");
    expect(html).toContain('value="en"');
    expect(html).toContain('value="fr"');
    expect(html.match(/role="checkbox" aria-checked="true"/g)).toHaveLength(1);
  });
});

// ------------------------------------------------------------------ cues

describe("CueTable", () => {
  it("draws only the rows in view of a long file", () => {
    const cues: CueData[] = Array.from({ length: 5000 }, (_, i) => ({ start: i * 2, end: i * 2 + 1.5, text: `Line ${i}` }));
    const html = renderToStaticMarkup(<CueTable cues={cues} mode="cps" minConfidence={0.6} onlyCheck={false} />);
    const rows = html.match(/role="row"/g) ?? [];
    expect(rows.length).toBeLessThan(30);
    expect(html).toContain("Line 0");
    expect(html).not.toContain("Line 4999");
  });

  it("flags lines read too fast, and lines an engine was unsure of", () => {
    const cues: CueData[] = [
      { start: 0, end: 2, text: "Where are you going?", confidence: 0.95 },
      { start: 3, end: 3.9, text: "You always say that like it's easy to do.", confidence: 0.95 },
      { start: 5, end: 8, text: "unsure", confidence: 0.4 },
    ];
    expect(cues.map((cue) => needsCheck(cue, "cps", 0.6))).toEqual([false, true, true]);
    expect(cues.map((cue) => needsCheck(cue, "ocr", 0.6))).toEqual([false, false, true]);
    const html = renderToStaticMarkup(<CueTable cues={cues} mode="cps" minConfidence={0.6} onlyCheck />);
    expect(html).not.toContain("Where are you going?");
    expect(html).toContain("45.6");
  });

  it("lists OCR lines with how sure the engine was", () => {
    const cues: CueData[] = [
      { start: 760.12, end: 762, text: "どこへ行くの？", confidence: 0.97 },
      { start: 766.02, end: 768, text: "そんな場所、ないよ。", confidence: 0.43 },
    ];
    const html = renderToStaticMarkup(
      <CueTable cues={cues} mode="ocr" minConfidence={0.6} onlyCheck={false} selected={1} onSelect={noop} edits={{ 0: "どこ行くの？" }} />,
    );
    expect(html).toContain(">Sure<");
    expect(html).toContain("97%");
    expect(html).toContain('class="r num warn" style="display:flex">43%');
    expect(html).toContain("どこ行くの？");
    expect(html).toContain('aria-selected="true" aria-label="Line 2"');
  });
});

// ---------------------------------------------------------------- result

describe("ResultView", () => {
  it("shows a finished job's summary, report and outputs", () => {
    const { run } = syncRun();
    const job = { ...run.jobs[1], outcome: outcome(1, "j1", { warnings: ["Two scenes were cut"] }) };
    const html = renderToStaticMarkup(<ResultView job={job} actions={{ onReveal: noop, onOpen: noop, onUseAs: noop, onOpenConsole: noop }} />);
    expect(html).toContain("shifted by +3.250 s");
    expect(html).toContain("+3.250 s");
    expect(html).toContain("Two scenes were cut");
    expect(html).toContain("out-1.synced.en.srt");
    expect(html).toContain("Use as input for…");
    expect(html).toContain("Reveal");
  });

  it("says why a job failed and offers the output", () => {
    const html = renderToStaticMarkup(
      <ResultView
        job={{ index: 0, id: "a", task: "sync", label: "x", status: "failed", percent: 100, stage: null, logs: [], outcome: outcome(0, "a", { error: "No audio track" }) }}
        actions={{ onReveal: noop, onOpen: noop, onUseAs: noop, onOpenConsole: noop }}
      />,
    );
    expect(html).toContain("No audio track");
    expect(html).toContain("Open the output");
  });
});

// ------------------------------------------------------------ the page

const shell = (): Shell => ({
  active: "subsync",
  show: noop,
  desktop: false,
  settings: DEFAULT_SETTINGS,
  updateSettings: noop,
  enginePage: null,
  claimEngine: () => true,
  releaseEngine: noop,
  setEngineListeners: noop,
  log: noop,
  clearLogs: noop,
  openOutput: noop,
  dock: { tab: null, setTab: noop, output: null, history: null },
  history: [],
  addHistory: noop,
  announce: noop,
  setCommands: noop,
  openPreferences: noop,
  copy: noop,
});

describe("SubsyncPage", () => {
  it("draws the tool strip and says, in a browser, that it needs the desktop app", () => {
    const html = renderToStaticMarkup(
      <ShellContext.Provider value={shell()}>
        <SubsyncPage hidden={false} />
      </ShellContext.Provider>,
    );
    for (const tool of ["Sync", "OCR", "Translate", "Frame rate", "Generate", "Style", "HDR subtitles", "Tone-map", "Convert &amp; mux"]) {
      expect(html).toContain(`</span>${tool}</button>`);
    }
    expect(html).toContain("Sync: drop subtitles and their videos");
    expect(html).toContain("Subtitle tools need the desktop app");
  });
});

// ----------------------------------------------------- Preferences › Subtitles

const CAPS: Capabilities = {
  platform: "windows",
  arch: "x64",
  ffmpeg: { path: "ffmpeg", version: "6.1", filters: {}, encoders: {} },
  engines: { sync: [], vad: [], ocr: [], translate: [], generate: [], tonemap: [] },
  packs: [
    {
      id: "asr-faster", label: "Speech recognition", description: "faster-whisper", installed: true, supported: true,
      sizeBytes: 1.6e9, downloadBytes: null, version: "1.1",
      models: [{ id: "large-v3", label: "Large v3", installed: false, downloadBytes: 3.1e9 }],
    },
    { id: "asr-mlx", label: "mlx-whisper", description: "Apple Silicon", installed: false, supported: false, sizeBytes: null, downloadBytes: 1e9, version: null },
    { id: "ffmpeg-full", label: "Tone-mapping", description: "FFmpeg with libplacebo", installed: false, supported: true, sizeBytes: null, downloadBytes: 140e6, version: null },
  ],
};

describe("EnginesSection", () => {
  const win = globalThis as { window?: unknown };
  afterEach(() => {
    delete win.window;
    resetEngines();
  });

  it("lists the packs and the API keys from the shared store", () => {
    win.window = { __TAURI_INTERNALS__: {} };
    resetEngines({ caps: CAPS, secrets: { anthropic: true, openai: false, deepl: false, google: false } });
    const html = renderToStaticMarkup(<EnginesSection />);
    expect(html).toContain('<div class="group-h">Engines</div>');
    // Its size under its name; what it is and its version are the tooltip.
    expect(html).toContain('title="faster-whisper · Version 1.1"');
    expect(html).toContain('<div class="d">1.6 GB</div>');
    expect(html).toContain('aria-label="Remove Speech recognition"');
    // A pack this platform cannot run is not offered.
    expect(html).not.toContain("Not available on Windows");
    expect(html).toContain('aria-label="Install Tone-mapping"');
    expect(html).toContain("about 140 MB");
    expect(html).toContain('aria-label="Download the Large v3 model"');
    expect(html).toContain("API keys");
    expect(html).toContain("•••••••••••• saved");
    expect(html).toContain('aria-label="Remove the Anthropic key"');
    expect(html.match(/placeholder="Paste a key"/g)).toHaveLength(3);
  });

  it("shows an install in progress", () => {
    win.window = { __TAURI_INTERNALS__: {} };
    resetEngines({
      caps: CAPS,
      packs: { "ffmpeg-full": { busy: true, action: "install", percent: 40, stage: "Downloading", bytes: 56e6, totalBytes: 140e6, error: null } },
    });
    const html = renderToStaticMarkup(<EnginesSection />);
    expect(html).toContain("56 MB of 140 MB");
    expect(html).toContain(">Stop<");
    // Nothing else can be installed meanwhile.
    expect(html).toMatch(/disabled=""[^>]*aria-label="Remove Speech recognition"/);
  });

  it("points a browser at the desktop app", () => {
    expect(renderToStaticMarkup(<EnginesSection />)).toContain("Installed and removed from the desktop app.");
  });
});
