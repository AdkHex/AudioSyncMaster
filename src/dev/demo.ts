/** Demo harness: the real app in a plain browser, with the desktop shell
 *  mocked.
 *
 *  Loaded only by `src/main.tsx`, only in `vite dev`, only when the URL has
 *  `?demo`. Nothing here reaches a production build (see the check in the
 *  task notes: the marker string below is absent from dist/).
 *
 *    /?demo=movies       6 Blade Runner 2049 releases against one Hindi dub
 *    /?demo=series       Goblin S01, 16 episodes, 15 dubs
 *    /?demo=match        Parasite, 3 videos x 3 dubs (Find match)
 *    /?demo=dub          Skyline Heist, one pair, a cut dub
 *    /?demo=dub-season   five Goblin pairs, E04 fails
 *    /?demo=subs         Subsync: Blade Runner subtitles, Your Name OCR
 *
 *  Options: `&update=1` offers an update (2.15.0), `&speed=0.25` runs
 *  streamed commands four times faster, `&voice=0` starts with the voice
 *  tools not installed, `&pick=ocr` makes the Subsync pickers return the
 *  image subtitle. `window.__demo.drop([...paths])` fakes a file drop. */

import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { InvokeArgs } from "@tauri-apps/api/core";
import { emit } from "@tauri-apps/api/event";

import type { WaveformBuildRequest, WaveformRequest } from "@/lib/api";
import type {
  AnalyzeRequest,
  ApplyResult,
  CorrectionItem,
  DubCodec,
  DubJobOutcome,
  DubSyncBatchRequest,
  DubSyncBatchResult,
  DubSyncOutcome,
  DubSyncPlan,
  DubSyncRequest,
  FileItem,
  PairingReport,
  SyncResult,
  SyncRun,
  VoiceToolsResult,
  VoiceToolsStatus,
} from "@/lib/types";
import type { SubJob, SubJobOutcome, SubsyncEvent } from "@/lib/subsync/types";

import { DUB_STAGES, DUB_STAGE_MS, DUB_WRITE_MS, describePlan, draftPlan, generatedPlan, muxPathFor, outputFor, skylinePlan, verification, verificationText } from "./dubfx";
import { Run, beginRun, cancelAll, clamp, log, nap, pool, send, setTimeScale } from "./engine";
import {
  MATCH_DUBS,
  MATCH_VIDEOS,
  MOVIE_DIR,
  MOVIE_DUB,
  MOVIE_DUB_DIR,
  MOVIE_VIDEOS,
  SEASON_DUBS,
  SEASON_VIDEOS,
  SERIES_DIR,
  SERIES_DUBS,
  SERIES_DUB_DIR,
  SERIES_VIDEOS,
  SKYLINE_DUB,
  SKYLINE_VIDEO,
  DUB_DIR,
  MATCH_DIR,
  baseName,
  dirName,
  episodeKey,
  extOf,
  fileForPath,
  joinPath,
  listing,
  pairing,
  probeOf,
  resultFor,
  stemOf,
  summarize,
  type FileFx,
} from "./fixtures";
import {
  BR_DIR,
  PACKS,
  YOUR_NAME_SUP,
  BR_SUBS,
  BR_VIDEO,
  capabilities,
  cuePreview,
  expandFolder,
  outcomeFor,
  pickedFromPath,
  probeOf as subsProbeOf,
  stagesFor,
} from "./subsfx";
import { silentWav, synthCuts, synthPeaks } from "./synth";

/** Unique string the production build must not contain. */
const MARKER = "__AUDIOSYNC_DEMO__";

type Args = Record<string, unknown>;

interface Scenario {
  name: string;
  videos: FileFx[];
  dubs: FileFx[];
  videoFolder: string;
  audioFolder: string;
}

const SCENARIOS: Record<string, Scenario> = {
  movies: { name: "movies", videos: MOVIE_VIDEOS, dubs: [MOVIE_DUB], videoFolder: MOVIE_DIR, audioFolder: MOVIE_DUB_DIR },
  series: { name: "series", videos: SERIES_VIDEOS, dubs: SERIES_DUBS, videoFolder: SERIES_DIR, audioFolder: SERIES_DUB_DIR },
  match: { name: "match", videos: MATCH_VIDEOS, dubs: MATCH_DUBS, videoFolder: MATCH_DIR, audioFolder: `${MATCH_DIR}\\Dubs` },
  dub: { name: "dub", videos: [SKYLINE_VIDEO], dubs: [SKYLINE_DUB], videoFolder: DUB_DIR, audioFolder: DUB_DIR },
  "dub-season": { name: "dub-season", videos: SEASON_VIDEOS, dubs: SEASON_DUBS, videoFolder: SERIES_DIR, audioFolder: SERIES_DUB_DIR },
  subs: { name: "subs", videos: MOVIE_VIDEOS, dubs: [MOVIE_DUB], videoFolder: BR_DIR, audioFolder: MOVIE_DUB_DIR },
};

const VERSION = "2.14.0";
const NEW_VERSION = "2.15.0";

interface DemoState {
  scenario: Scenario;
  options: URLSearchParams;
  voiceInstalled: boolean;
  built: Set<string>;
  secrets: Record<string, boolean>;
}

let state: DemoState;

// ----------------------------------------------------------------- helpers

const args = (value: InvokeArgs | undefined): Args =>
  value && typeof value === "object" && !Array.isArray(value) && !(value instanceof ArrayBuffer) && !(value instanceof Uint8Array)
    ? (value as Args)
    : {};

const item = (file: FileFx, type: "video" | "audio" = file.kind) => ({ name: file.name, path: file.path, type, size: file.size });

const FILE_EXT = new Set([
  "mkv", "mp4", "avi", "mov", "m4v", "ts", "webm", "ac3", "eac3", "aac", "dts", "flac", "wav", "mka", "mp3", "m4a", "opus", "ogg", "thd",
  "srt", "ass", "ssa", "vtt", "sub", "sup", "ttml", "sbv",
]);
/** A file, as against a dropped folder. */
const hasExtension = (path: string) => FILE_EXT.has(extOf(baseName(path)));

function pickResponse(files: FileFx[], folder: string, type?: "video" | "audio") {
  return { folder, files: files.map((file) => item(file, type)) };
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

// ------------------------------------------------------------- file picking

function resolveDropped(request: Args): Omit<FileItem, "id">[] {
  const kind = request.kind === "audio" ? "audio" : "video";
  const accept = (request.accept as string | undefined) ?? kind;
  const wanted = (file: FileFx) => accept === "media" || file.kind === accept;
  const out: FileFx[] = [];
  stringList(request.paths).forEach((path) => {
    if (hasExtension(path)) {
      const file = fileForPath(path);
      if (wanted(file)) out.push({ ...file, path });
      return;
    }
    // A folder: the scenario's files of the kind it is dropped for.
    const { videos, dubs } = state.scenario;
    if (accept === "media") out.push(...videos, ...dubs);
    else if (accept === "audio") out.push(...dubs);
    else out.push(...videos);
  });
  return out.map((file) => ({ name: file.name, path: file.path, type: file.kind, size: file.size }));
}

// ------------------------------------------------------------------ pairing

function pairingRequest(request: Partial<AnalyzeRequest>): PairingReport {
  const { videos, dubs } = state.scenario;
  const videoPaths = request.videoFiles?.length ? request.videoFiles : videos.map((f) => f.path);
  const dubPaths =
    request.mode === "movie" && request.audioFile
      ? [request.audioFile]
      : request.audioFiles?.length
        ? request.audioFiles
        : dubs.map((f) => f.path);
  return pairing({
    mode: request.mode ?? "movie",
    dubKind: request.dubKind,
    videos: videoPaths,
    dubs: dubPaths,
    explicit: request.pairs ?? null,
  });
}

// --------------------------------------------------------------- start_sync

async function startSync(request: AnalyzeRequest): Promise<SyncRun> {
  const run = beginRun();
  try {
    const report = pairingRequest(request);
    send("sync-pairs", report);
    const pairs = report.pairs;
    const total = pairs.length;
    // Several pairs at a time, as the engine's worker pool does.
    const workers = Math.max(1, Math.min(request.maxWorkers || 3, total));
    const each = clamp((6000 * workers) / Math.max(1, total), 600, 2500);
    log(`Analysing ${total} pair(s) with ${workers} worker(s).`);
    const measured: (SyncResult | undefined)[] = [];
    let cancelled = false;
    let next = 0;
    let processed = 0;
    const measure = async (index: number) => {
      const pair = pairs[index];
      // Uneven lengths, so the workers do not move in lockstep.
      const length = each * (0.7 + ((index * 37) % 60) / 100);
      send("sync-file-start", { file: pair.primaryName });
      for (const percent of [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) {
        if (!(await run.sleep(length / 10))) return false;
        send("sync-file-progress", { file: pair.primaryName, percent });
      }
      const result = resultFor(fileForPath(pair.primaryPath), fileForPath(pair.secondaryPath));
      measured[index] = result;
      processed += 1;
      send("sync-result", result);
      send("sync-progress", { processed, total, current: pair.primaryName });
      log(
        result.error
          ? `${pair.primaryName}: ${result.error}`
          : `${pair.primaryName}: delay ${result.delayMs?.toFixed(1)} ms, confidence ${Math.round((result.confidence ?? 0) * 100)}%`,
      );
      return true;
    };
    const worker = async () => {
      while (!cancelled && next < total) {
        const index = next++;
        if (!(await measure(index))) cancelled = true;
      }
    };
    await Promise.all(Array.from({ length: workers }, worker));
    const results = measured.filter((r): r is SyncResult => r !== undefined);
    if (cancelled) log("Analysis cancelled.");
    return { results, summary: summarize(results), cancelled };
  } finally {
    run.end();
  }
}

// ------------------------------------------------------------- apply_corrections

async function applyCorrections(request: Args): Promise<ApplyResult> {
  const run = beginRun();
  try {
    const items = (request.items as CorrectionItem[] | undefined) ?? [];
    const suffix = (request.suffix as string | undefined) ?? ".synced";
    const written: string[] = [];
    for (const [index, entry] of items.entries()) {
      const file = baseName(entry.videoPath);
      const audio = baseName(entry.audioPath);
      // As audiosync/mux.py plan_correction: the video's name, beside it or in outputDir.
      const output = joinPath((request.outputDir as string | null | undefined) ?? dirName(entry.videoPath), `${stemOf(file)}${suffix}.${extOf(file) || "mkv"}`);
      const delay = entry.delayAtStartMs ?? entry.delayMs;
      const description =
        delay >= 0 ? `Trim ${delay.toFixed(1)} ms from the start of ${audio}` : `Add ${(-delay).toFixed(1)} ms of silence to the start of ${audio}`;
      send("sync-apply-progress", { file, output, description });
      log(`Writing ${baseName(output)}`);
      if (!(await run.sleep(3000 / Math.max(1, items.length)))) return { written, failed: [], cancelled: true };
      written.push(output);
      send("sync-apply-progress", { file, output, done: index + 1, total: items.length });
    }
    return { written, failed: [], cancelled: false };
  } finally {
    run.end();
  }
}

// ----------------------------------------------------------------- dub sync

interface Sink {
  progress(percent: number, stage: string): void;
  draft(plan: DubSyncPlan): void;
  plan(plan: DubSyncPlan, description: string): void;
}

interface PipelineInput {
  video: FileFx;
  dub: FileFx;
  codec: DubCodec;
  mux: boolean;
  fixVoices: boolean;
  /** An edited plan to write as it is, skipping the analysis. */
  plan?: DubSyncPlan;
  outputPath?: string | null;
  muxPath?: string | null;
  /** A batch's Save to folder; null writes beside the dub. */
  outputDir?: string | null;
}

const isSeasonFailure = (video: FileFx) => state.scenario.name === "dub-season" && episodeKey(video.name) === "S01E04";

const TYPICAL_MS: Record<string, number> = { S01E01: 0.2, S01E02: 0.3, S01E03: 0.4, S01E05: 0.2 };

function planFor(input: PipelineInput): DubSyncPlan {
  const { video, dub } = input;
  const plan =
    video.path === SKYLINE_VIDEO.path || /skyline/i.test(video.name) ? skylinePlan(video, dub) : generatedPlan(video, dub, episodeKey(video.name) === "S01E03");
  return input.fixVoices ? plan : { ...plan, voicePieces: undefined };
}

async function runStage(run: Run, sink: Sink, stage: string, ms: number, onTick?: (fraction: number) => void): Promise<boolean> {
  const ticks = Math.max(2, Math.round(ms / 170));
  sink.progress(0, stage);
  for (let tick = 1; tick <= ticks; tick += 1) {
    if (!(await run.sleep(ms / ticks))) return false;
    sink.progress(Math.round((100 * tick) / ticks), stage);
    onTick?.(tick / ticks);
  }
  return true;
}

/** The whole engine walk for one pair, streamed through `sink`. */
async function pipeline(run: Run, input: PipelineInput, sink: Sink): Promise<DubSyncOutcome> {
  const { video, dub } = input;
  const stopped: DubSyncOutcome = { plan: null, output: null, verification: null, verificationText: null, muxedPath: null, cancelled: true, error: null };
  const edited = input.plan !== undefined;
  const plan = input.plan ?? planFor(input);
  const count = plan.segments.length;

  if (!edited) {
    log(`Reading ${video.name} · ${(video.tracks[0]?.codec ?? "audio").toUpperCase()}, 48 kHz, ${new Date(video.durationS * 1000).toISOString().slice(11, 19)}`);
    if (!(await runStage(run, sink, DUB_STAGES[0], DUB_STAGE_MS[0]))) return stopped;
    if (!(await runStage(run, sink, DUB_STAGES[1], DUB_STAGE_MS[1]))) return stopped;
    log(`Frame rate: video ${(video.fps ?? 23.976).toFixed(3)} fps, dub mastered at ${(plan.dubRate ?? 23.976).toFixed(3)} fps`);
    if (!(await runStage(run, sink, DUB_STAGES[2], DUB_STAGE_MS[2]))) return stopped;
    if (isSeasonFailure(video)) {
      const error = "The dub ends 38 minutes before the video. It is probably a different edit.";
      log(`${video.name.replace(/\.mkv$/, "")}: the dub ends 38 minutes before the video (a different edit?)`);
      log(`${video.name.replace(/\.mkv$/, "")}: source files left unchanged`);
      return { ...stopped, cancelled: false, error };
    }
    const first = plan.segments.find((s) => s.kind === "dub");
    if (first) log(`Dub belongs at ${first.offsetS !== null && first.offsetS >= 0 ? "+" : "−"}${Math.abs(first.offsetS ?? 0).toFixed(3)} s from ${new Date(first.startS * 1000).toISOString().slice(11, 23)}`);
    sink.draft(draftPlan(plan, Math.min(2, count)));

    let shown = Math.min(2, count);
    const cuts = plan.segments.filter((s) => s.reason === "cut");
    if (
      !(await runStage(run, sink, DUB_STAGES[3], DUB_STAGE_MS[3], (fraction) => {
        const placed = Math.max(shown, Math.round(2 + fraction * (count - 2)));
        if (placed !== shown) {
          shown = placed;
          sink.draft(draftPlan(plan, placed));
        }
      }))
    ) {
      return stopped;
    }
    cuts.forEach((cut) => log(`Scene cut from the dub at ${new Date(cut.startS * 1000).toISOString().slice(11, 23)} (${(cut.endS - cut.startS).toFixed(1)}s) — filled from the original`));
    if (!(await runStage(run, sink, DUB_STAGES[4], DUB_STAGE_MS[4]))) return stopped;
    const typical = TYPICAL_MS[episodeKey(video.name) ?? ""] ?? 0.2;
    log(`Measured ${plan.segments.filter((s) => s.kind === "dub").length} stretches — ${typical.toFixed(2)} ms typical`);
    if (!(await runStage(run, sink, DUB_STAGES[5], DUB_STAGE_MS[5]))) return stopped;
    if (!(await runStage(run, sink, DUB_STAGES[6], DUB_STAGE_MS[6]))) return stopped;
    sink.plan(plan, describePlan(plan));
  }

  const typical = TYPICAL_MS[episodeKey(video.name) ?? ""] ?? 0.2;
  const checked = verification(plan, typical, Math.round(typical * 2 * 100) / 100);
  const output = outputFor(plan, dub, input.codec, plan.videoDurationS);
  const outputPath = input.outputPath ?? (input.outputDir ? joinPath(input.outputDir, baseName(output.outputPath)) : output.outputPath);
  // As python/bridge.py: the track is written, then the finished track is checked.
  if (!(await runStage(run, sink, `writing ${baseName(outputPath)}`, edited ? 1400 : DUB_WRITE_MS))) return stopped;
  if (!(await runStage(run, sink, DUB_STAGES[7], edited ? 1400 : DUB_STAGE_MS[7]))) return stopped;
  log(`Wrote ${baseName(outputPath)} — ${checked.typicalMs?.toFixed(1)} ms typical, ${checked.worstMs?.toFixed(1)} ms at worst`);
  return {
    plan,
    output: { ...output, outputPath },
    verification: checked,
    verificationText: verificationText(checked),
    muxedPath: input.mux ? (input.muxPath ?? muxPathFor(plan)) : null,
    cancelled: false,
    error: null,
  };
}

async function startDubBatch(request: DubSyncBatchRequest): Promise<DubSyncBatchResult> {
  const run = beginRun();
  try {
    const jobs = request.jobs;
    const workers = clamp(request.maxWorkers || 3, 1, 8);
    log(`Dub sync: ${jobs.length} pair${jobs.length === 1 ? "" : "s"}, ${workers} at a time`);
    const outcomes: DubJobOutcome[] = jobs.map((_, job) => ({ job, plan: null, output: null, verification: null, muxedPath: null, cancelled: true, error: null }));
    await pool(jobs.length, workers, async (job) => {
      if (run.cancelled) return;
      const spec = jobs[job];
      const video = fileForPath(spec.videoPath);
      const dub = fileForPath(spec.dubPath);
      send("dubsync-job-start", { job, name: video.name, dub: dub.name });
      const outcome = await pipeline(
        run,
        { video, dub, codec: request.codec, mux: request.mux, fixVoices: request.fixVoices !== false, outputDir: request.outputDir },
        {
          progress: (percent, stage) => send("dubsync-job-progress", { job, percent, stage }),
          draft: (plan) => send("dubsync-job-draft", { job, plan }),
          plan: (plan, description) => send("dubsync-job-plan", { job, plan, description }),
        },
      );
      const done: DubJobOutcome = {
        job,
        plan: outcome.plan,
        output: outcome.output,
        verification: outcome.verification,
        verificationText: outcome.verificationText,
        muxedPath: outcome.muxedPath,
        cancelled: outcome.cancelled,
        error: outcome.error ?? null,
      };
      outcomes[job] = done;
      send("dubsync-job-done", done);
    });
    return { outcomes, cancelled: run.cancelled };
  } finally {
    run.end();
  }
}

/** One pair on its own: the edited plan written again, or a fresh analysis. */
async function startDubSync(request: DubSyncRequest): Promise<DubSyncOutcome> {
  const run = beginRun();
  try {
    const video = fileForPath(request.videoPath);
    const dub = fileForPath(request.dubPath);
    return await pipeline(
      run,
      {
        video,
        dub,
        codec: request.codec,
        mux: request.mux,
        fixVoices: request.fixVoices !== false,
        plan: request.plan,
        outputPath: request.outputPath,
        muxPath: request.muxPath,
      },
      {
        progress: (percent, stage) => send("dubsync-progress", { percent, stage }),
        draft: (plan) => send("dubsync-draft", { plan }),
        plan: (plan, description) => send("dubsync-plan", { plan, description }),
      },
    );
  } finally {
    run.end();
  }
}

// -------------------------------------------------------------- voice tools

function voiceStatus(): VoiceToolsStatus {
  return state.voiceInstalled
    ? {
        installed: true,
        dir: "C:\\Users\\Santosh\\AppData\\Roaming\\AudioSyncMaster\\voice-tools",
        toolsVersion: 3,
        uv: "0.4.18",
        python: "3.11.9",
        packages: ["torch 2.4.1", "demucs 4.0.1", "silero-vad 5.1"],
        device: "cuda",
        installedAt: "2026-09-26T09:12:00.000Z",
        outdated: false,
        sizeBytes: 1.1 * 1024 ** 3,
      }
    : { installed: false, dir: "C:\\Users\\Santosh\\AppData\\Roaming\\AudioSyncMaster\\voice-tools", toolsVersion: 3 };
}

async function voiceTools(action: "status" | "install" | "remove"): Promise<VoiceToolsResult> {
  if (action === "install") {
    const run = beginRun();
    try {
      const stages = ["downloading PyTorch", "downloading Demucs", "downloading the speech detector", "checking the install"];
      for (const [i, stage] of stages.entries()) {
        for (let step = 1; step <= 4; step += 1) {
          if (!(await run.sleep(250))) {
            log("Voice tools install cancelled.");
            return { type: "voiceTools", action, status: voiceStatus(), error: "cancelled" };
          }
          send("voice-tools-progress", { percent: Math.round(((i * 4 + step) / (stages.length * 4)) * 100), stage });
        }
        log(`Voice tools: ${stage}`);
      }
      state.voiceInstalled = true;
    } finally {
      run.end();
    }
  } else if (action === "remove") {
    await nap(600);
    state.voiceInstalled = false;
  }
  return { type: "voiceTools", action, status: voiceStatus(), error: null };
}

// ----------------------------------------------------------------- waveform

async function waveformPeaks(request: WaveformRequest) {
  const key = `${request.path}|${request.track}`;
  if (!state.built.has(key)) await waveformBuild({ path: request.path, track: request.track, requestId: request.requestId });
  return synthPeaks(request);
}

async function waveformBuild(request: WaveformBuildRequest) {
  const file = fileForPath(request.path);
  const key = `${request.path}|${request.track}`;
  if (!state.built.has(key)) {
    for (const percent of [0, 25, 55, 85]) {
      send("waveform-progress", { path: request.path, track: request.track, percent, requestId: request.requestId });
      await nap(180);
    }
    state.built.add(key);
  }
  send("waveform-progress", { path: request.path, track: request.track, percent: 100, requestId: request.requestId });
  return { path: request.path, track: request.track, durationS: file.durationS, channels: 2, sampleRate: 48000, requestId: request.requestId };
}

// ------------------------------------------------------------------ subsync

function subsPicks(accept: string): { folder: string | null; files: ReturnType<typeof pickedFromPath>[] } {
  if (accept === "video") return { folder: BR_DIR, files: [BR_VIDEO] };
  if (state.options.get("pick") === "ocr") return { folder: dirName(YOUR_NAME_SUP.path), files: [YOUR_NAME_SUP] };
  return { folder: BR_DIR, files: accept === "subtitle" ? BR_SUBS : [...BR_SUBS, BR_VIDEO] };
}

function subsEvent(event: SubsyncEvent): void {
  send("subsync-event", event);
}

async function startSubsBatch(request: Args): Promise<{ outcomes: SubJobOutcome[]; cancelled: boolean }> {
  const run = beginRun();
  try {
    const { jobs, maxWorkers } = (args(request.request as InvokeArgs) as { jobs?: SubJob[]; maxWorkers?: number });
    const list = jobs ?? [];
    const outcomes: SubJobOutcome[] = list.map((job, index) => ({
      job: index,
      id: job.id,
      task: job.task,
      outputs: [],
      report: {},
      summary: "",
      warnings: [],
      preview: null,
      cancelled: true,
    }));
    // Four seconds in all: each job's share of the workers' time.
    const workers = clamp(maxWorkers ?? 2, 1, 4);
    const each = (4000 * workers) / Math.max(1, list.length);
    await pool(list.length, workers, async (index) => {
      if (run.cancelled) return;
      const job = list[index];
      subsEvent({ type: "subsJobStart", job: index, id: job.id, task: job.task });
      const stages = stagesFor(job.task);
      for (const [i, stage] of stages.entries()) {
        for (const step of [1, 2, 3]) {
          if (!(await run.sleep(each / (stages.length * 3)))) return;
          subsEvent({ type: "subsJobProgress", job: index, percent: Math.round(((i + step / 3) / stages.length) * 100), stage });
        }
        subsEvent({ type: "subsJobLog", job: index, message: stage });
      }
      const outcome = outcomeFor(job, index);
      outcomes[index] = outcome;
      subsEvent({ type: "subsJobDone", ...outcome });
    });
    subsEvent({ type: "subsBatchDone", outcomes, cancelled: run.cancelled });
    return { outcomes, cancelled: run.cancelled };
  } finally {
    run.end();
  }
}

async function subsPack(action: string, packId: string): Promise<unknown> {
  const pack = PACKS.find((entry) => entry.id === packId);
  if (!pack) return { type: "packDone", pack: packId, ok: false, error: "Unknown pack" };
  if (action === "install") {
    const run = beginRun();
    try {
      for (const percent of [10, 30, 55, 80, 100]) {
        if (!(await run.sleep(500))) return { type: "packDone", pack: packId, ok: false, error: "cancelled", status: pack };
        subsEvent({ type: "packProgress", pack: packId, percent, stage: percent < 100 ? "downloading" : "unpacking", bytes: (pack.downloadBytes ?? 0) * (percent / 100), totalBytes: pack.downloadBytes ?? 0 });
      }
    } finally {
      run.end();
    }
    pack.installed = true;
    pack.sizeBytes = pack.downloadBytes;
    pack.version = pack.version ?? "1.0.0";
  } else {
    await nap(400);
    pack.installed = false;
    pack.sizeBytes = null;
  }
  const done = { type: "packDone" as const, pack: packId, ok: true, error: null, status: pack };
  subsEvent(done);
  return done;
}

// ------------------------------------------------------------------ plugins

function updaterMetadata() {
  return {
    rid: 4242,
    currentVersion: VERSION,
    version: NEW_VERSION,
    date: "2026-09-29T08:00:00.000Z",
    body: "- Dub sync keeps your waveform edits when you write again\n- Subsync reads Japanese image subtitles faster\n- Fixes a crash when a dropped folder held only subtitles",
    rawJson: {},
  };
}

/** Feed a plugin Channel the way the shell does: ordered `{index, message}`. */
async function streamUpdate(channel: { id: number }): Promise<void> {
  const internals = (window as unknown as { __TAURI_INTERNALS__: { runCallback(id: number, data: unknown): void } }).__TAURI_INTERNALS__;
  const total = 71 * 1024 * 1024;
  let index = 0;
  const push = (message: unknown) => internals.runCallback(channel.id, { index: index++, message });
  push({ event: "Started", data: { contentLength: total } });
  const chunk = Math.round(total / 20);
  for (let i = 0; i < 20; i += 1) {
    await nap(150);
    push({ event: "Progress", data: { chunkLength: chunk } });
  }
  push({ event: "Finished" });
  internals.runCallback(channel.id, { index: index++, end: true });
}

function dialogOpen(options: Args): unknown {
  const { videos, dubs, videoFolder } = state.scenario;
  if (options.directory) return videoFolder;
  const files = ((options.title as string | undefined) ?? "").toLowerCase().includes("audio") ? dubs : videos;
  return options.multiple ? files.map((file) => file.path) : files[0]?.path ?? null;
}

// ------------------------------------------------------------------ handler

const NO_OP_COMMANDS = new Set([
  "reveal_path",
  "open_path",
  "plugin:opener|reveal_item_in_dir",
  "plugin:opener|open_path",
  "plugin:opener|open_url",
  "plugin:log|log",
  "plugin:process|restart",
  "plugin:process|exit",
  "plugin:resources|close",
  "plugin:updater|install",
  "plugin:dialog|message",
]);

function handler(cmd: string, raw?: InvokeArgs): unknown {
  const a = args(raw);
  const { scenario } = state;
  switch (cmd) {
    // -- pickers and probes (src/lib/api.ts)
    case "pick_video_folder":
      return pickResponse(scenario.videos, scenario.videoFolder, "video");
    case "pick_audio_folder":
      return pickResponse(scenario.dubs, scenario.audioFolder, "audio");
    case "pick_audio_file":
      return pickResponse(scenario.dubs.slice(0, 1), scenario.audioFolder, "audio");
    case "pick_media_file":
      return pickResponse((a.kind === "audio" ? scenario.dubs : scenario.videos).slice(0, 1), a.kind === "audio" ? scenario.audioFolder : scenario.videoFolder);
    case "pick_media_files":
      return pickResponse(a.kind === "audio" ? scenario.dubs : scenario.videos, a.kind === "audio" ? scenario.audioFolder : scenario.videoFolder);
    case "resolve_dropped_paths":
      return resolveDropped(a);
    case "probe_media":
      return probeOf(fileForPath(String(a.path ?? "")));
    case "list_audio_tracks":
      return stringList(a.paths).map((path) => listing(fileForPath(path)));
    case "preview_pairs": {
      const report = pairingRequest((a.request ?? {}) as Partial<AnalyzeRequest>);
        return report;
    }

    // -- streamed runs
    case "start_sync":
      return startSync(a.request as AnalyzeRequest);
    case "start_dubsync":
      return startDubSync(a.request as DubSyncRequest);
    case "start_dubsync_batch":
      return startDubBatch(a.request as DubSyncBatchRequest);
    case "apply_corrections":
      return applyCorrections((a.request ?? {}) as Args);
    case "voice_tools":
      return voiceTools(a.action as "status" | "install" | "remove");
    case "cancel_sync":
      cancelAll();
      return null;

    // -- previews, waveform, exports
    case "render_preview":
      // A path to hand to the OS player (open_path is a no-op here).
      return "C:\\Temp\\preview.mkv";
    case "render_dub_preview": {
      // Sound excerpts play (read_preview_bytes serves a silent WAV); the
      // demo has no picture to cut, so the player falls back to sound only.
      const request = (a.request ?? {}) as { startS?: number; endS?: number; what?: string; video?: boolean };
      const what = request.what ?? (request.video ? "both" : "audio");
      if (what !== "audio" && what !== "original") return null;
      return { path: "C:/Temp/preview.wav", what, startS: request.startS ?? 0, endS: request.endS ?? 0 };
    }
    case "read_preview_bytes":
      return silentWav();
    case "waveform_peaks":
      return waveformPeaks(a.request as WaveformRequest);
    case "waveform_build":
      return waveformBuild(a.request as WaveformBuildRequest);
    case "shot_cuts": {
      const r = a.request as { path: string; startS: number; endS: number };
      return synthCuts(r.path, r.startS, r.endS);
    }
    case "export_csv":
      return "C:\\Users\\Santosh\\Documents\\AudioSyncMaster\\exports\\audiosync-results.csv";
    case "export_json":
      return "C:\\Users\\Santosh\\Documents\\AudioSyncMaster\\exports\\audiosync-results.json";

    // -- Subsync (src/lib/subsync/api.ts)
    case "subs_pick_files":
      return subsPicks(String(a.accept ?? "any"));
    case "subs_pick_folder":
      return BR_DIR;
    case "subs_resolve_dropped":
      return stringList(a.paths).flatMap((path) => (hasExtension(path) ? [pickedFromPath(path)] : expandFolder(path)));
    case "subs_probe":
      return { files: stringList(a.paths).map(subsProbeOf) };
    case "subs_caps":
      return { caps: capabilities() };
    case "subs_load": {
      const ref = a.reference as { path: string };
      return { preview: cuePreview(ref.path) };
    }
    case "subs_save":
      return { path: String(a.path ?? "") };
    case "start_subs_batch":
      return startSubsBatch(a);
    case "subs_pack":
      return subsPack(String(a.action), String(a.pack));
    case "subs_secret_status":
      return { ...state.secrets };
    case "subs_secret_set":
      state.secrets[String(a.name)] = String(a.value ?? "") !== "";
      return null;

    // -- plugins
    case "plugin:updater|check":
      return state.options.get("update") === "1" ? updaterMetadata() : null;
    case "plugin:updater|download":
      return streamUpdate(a.onEvent as { id: number }).then(() => 4243);
    case "plugin:updater|download_and_install":
      return streamUpdate(a.onEvent as { id: number });
    case "plugin:dialog|open":
      return dialogOpen((a.options ?? a) as Args);
    case "plugin:dialog|save":
      return joinPath(scenario.videoFolder, "export.txt");
    case "plugin:dialog|ask":
    case "plugin:dialog|confirm":
      return true;
    case "plugin:app|version":
      return VERSION;
    case "plugin:app|name":
      return "AudioSyncMaster";
    case "plugin:app|tauri_version":
      return "2.11.1";
    case "plugin:app|identifier":
      return "com.audiosyncmaster.app";
    case "plugin:window|is_maximized":
    case "plugin:window|is_fullscreen":
    case "plugin:window|is_minimized":
      return false;
    case "plugin:window|is_focused":
    case "plugin:window|is_visible":
      return true;
    case "plugin:window|scale_factor":
      return 1;
    case "plugin:window|inner_size":
    case "plugin:window|outer_size":
      return { width: 1440, height: 900 };
    default:
      if (!NO_OP_COMMANDS.has(cmd)) console.debug(`[demo] unmocked command: ${cmd}`, raw);
      return null;
  }
}

// --------------------------------------------------------------- drag and drop

export interface DemoApi {
  scenario: string;
  /** Fake the whole gesture: enter, over, drop. */
  drop(paths: string[]): Promise<void>;
  /** Enter and hover, without dropping (for a screenshot of the drop state). */
  hover(paths: string[]): Promise<void>;
  /** Leave without dropping. */
  leave(): void;
}

declare global {
  interface Window {
    __demo?: DemoApi;
  }
}

const POSITION = { x: 720, y: 420 };

function dragApi(scenario: string): DemoApi {
  const hover = async (paths: string[]) => {
    await emit("tauri://drag-enter", { paths, position: POSITION });
    await nap(60);
    await emit("tauri://drag-over", { position: POSITION });
  };
  return {
    scenario,
    hover,
    async drop(paths) {
      await hover(paths);
      await nap(120);
      await emit("tauri://drag-drop", { paths, position: POSITION });
    },
    leave() {
      void emit("tauri://drag-leave");
    },
  };
}

// --------------------------------------------------------------------- install

export function installDemo(name: string): void {
  const internals = (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  // A real shell is left alone; a page that already ran the demo may switch scenario.
  if (internals !== undefined && window.__demo === undefined) {
    console.warn(`[${MARKER}] a Tauri shell is already present; the demo mocks are not installed.`);
    return;
  }
  const options = new URLSearchParams(window.location.search);
  const scenario = SCENARIOS[name] ?? SCENARIOS.movies;
  if (!(name in SCENARIOS)) console.warn(`[${MARKER}] unknown scenario "${name}"; using "movies". Known: ${Object.keys(SCENARIOS).join(", ")}`);
  state = {
    scenario,
    options,
    voiceInstalled: options.get("voice") !== "0",
    built: new Set(),
    secrets: { anthropic: false, openai: false, deepl: false, google: false },
  };
  setTimeScale(Number(options.get("speed") ?? 1));

  // The event mock drops a listener's callback on unlisten but keeps its id,
  // so the next emit warns about it. Harmless; keep the console readable.
  const warn = console.warn.bind(console);
  console.warn = (...parts: unknown[]) => {
    if (typeof parts[0] === "string" && parts[0].startsWith("[TAURI] Couldn't find callback id")) return;
    warn(...parts);
  };

  // An update offered on every load, not once per six hours.
  if (options.get("update") === "1") {
    try {
      localStorage.removeItem("audiosync.update.lastCheck");
      localStorage.removeItem("audiosync.update.skipped");
    } catch {
      /* storage unavailable */
    }
  }

  mockWindows("main");
  mockIPC(handler, { shouldMockEvents: true });
  window.__demo = dragApi(scenario.name);
  console.info(`[${MARKER}] mocked desktop shell, scenario "${scenario.name}". window.__demo.drop([paths]) fakes a file drop.`);
}
