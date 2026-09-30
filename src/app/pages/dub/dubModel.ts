/** The Dub sync page's wording and arithmetic, kept free of React so it can
 *  be tested: what each row, the status display and the history say about
 *  the queue, and which stage of the engine a job is in. */

import {
  describeStage,
  dubQueueReducer,
  type DubQueueAction,
  type DubQueueJob,
  type DubQueueState,
} from "@/lib/dubQueueReducer";
import type { PairOverrides } from "@/lib/pairing";
import {
  AUDIBLE_MS,
  formatClock,
  formatFps,
  formatMs,
  formatSpan,
  type DubJobOutcome,
  type DubSegment,
  type DubSyncPlan,
  type DubVerification,
  type FileItem,
  type MatchPair,
} from "@/lib/types";
import type { St } from "@/ui/kit";

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/* ------------------------------------------------------------------ files */

/** Containers that hold a picture. Everything else a drop brings is a dub. */
export const VIDEO_EXT = /\.(mkv|mp4|m4v|mov|avi|ts|m2ts|mts|webm|wmv|flv|mpg|mpeg|vob|ogv|3gp)$/i;
export const isVideoLike = (name: string) => VIDEO_EXT.test(name);

/** Where a file name stops being the title and starts being the release. */
const RELEASE =
  /[ ._-]+[([]?(?:\d{3,4}p|4k|uhd|hdr\d*|blu-?ray|bdrip|brrip|bdremux|remux|web-?dl|web-?rip|hdtv|dvd(?:rip|5|9)?|pal|ntsc|x26[45]|h\.?26[45]|hevc|avc|10-?bit|dts(?:-hd)?|truehd|atmos|ac3|e-?ac-?3|ddp?\d?(?:\.\d)?|aac|flac|hin(?:di)?|tv[ .]?rip)\b.*$/i;

/** A file's title without its extension and release tags:
 *  "Skyline Heist (2023) 1080p BluRay.mkv" is "Skyline Heist (2023)". */
export function shortTitle(name: string): string {
  const stem = name.replace(/^.*[\\/]/, "").replace(/\.[^.]{1,5}$/, "");
  const cut = stem.replace(RELEASE, "").replace(/[ ._-]+$/, "");
  return cut || stem;
}

const EPISODE = /[ ._-]*(?:S(\d{1,2})[ ._-]?E\d{1,3}|\d{1,2}x\d{2}).*$/i;

/** A series' title from one episode's name, or null when the name carries
 *  no episode number. */
function seriesOf(name: string): { title: string; season: number | null } | null {
  const title = shortTitle(name);
  const match = EPISODE.exec(title);
  if (!match) return null;
  const series = title.slice(0, match.index).replace(/[._]+/g, " ").trim();
  return series ? { title: series, season: match[1] ? Number(match[1]) : null } : null;
}

/** What to call several pairs at once: "Goblin · Season 1" for a season,
 *  "Skyline Heist (2023) and 2 more" otherwise. */
export function collectionTitle(names: string[]): string {
  if (names.length === 0) return "";
  if (names.length === 1) return shortTitle(names[0]);
  const series = names.map(seriesOf);
  const first = series[0];
  if (first && series.every((s) => s && s.title.toLowerCase() === first.title.toLowerCase())) {
    const seasons = new Set(series.map((s) => s!.season));
    return seasons.size === 1 && first.season !== null ? `${first.title} · Season ${first.season}` : first.title;
  }
  return `${shortTitle(names[0])} and ${names.length - 1} more`;
}

/* ----------------------------------------------------------------- stages */

export interface StageDef {
  label: string;
  /** The engine's stage names that belong to it. */
  keys: string[];
}

/** The checklist a running job shows, in the order the engine goes through
 *  it, named as the status display names them. */
export const DUB_STAGES: StageDef[] = [
  { label: "Reading the files", keys: ["starting", "probing", "reading the original", "reading the dub"] },
  { label: "Checking the frame rate", keys: ["checking the frame rate"] },
  { label: "Finding where the dub belongs", keys: ["finding the offsets"] },
  { label: "Measuring each stretch", keys: ["measuring the offsets"] },
  { label: "Placing the cuts", keys: ["placing the cuts"] },
  { label: "Looking for dub inside the gaps", keys: ["looking for dub inside the gaps"] },
  { label: "Assembling the plan", keys: ["assembling the plan"] },
  { label: "Writing the track", keys: ["writing"] },
  { label: "Checking against the video", keys: ["checking the finished track"] },
];

/** Only when the track also goes into a copy of the video. */
export const MUX_STAGE: StageDef = { label: "Adding the track to a copy of the video", keys: ["muxing"] };

export const stagesFor = (mux: boolean): StageDef[] => (mux ? [...DUB_STAGES, MUX_STAGE] : DUB_STAGES);

/** Where an engine stage falls in the checklist; -1 when it is not one. */
export function stageIndex(stage: string | null, stages: StageDef[] = stagesFor(true)): number {
  if (!stage) return -1;
  const s = stage.toLowerCase();
  if (s === "done") return stages.length;
  return stages.findIndex((d) => d.keys.some((k) => s === k || (k === "writing" && s.startsWith("writing "))));
}

/** A stage as the status display and the queue say it. */
export function stageLabel(stage: string | null): string {
  const all = stagesFor(true);
  const i = stageIndex(stage, all);
  return i >= 0 && i < all.length ? all[i].label : describeStage(stage);
}

/* ------------------------------------------------------------------ plans */

/** A fill that stands in for a scene the dub does not have. */
function isCut(segment: DubSegment): boolean {
  if (segment.kind !== "fill") return false;
  return segment.reason ? segment.reason === "cut" : /\bcut\b/i.test(segment.note);
}

export interface PlanFacts {
  dubs: number;
  fills: number;
  cuts: number;
  stretches: string;
  fromOriginal: string;
  fillLevel: string;
  frameRate: string;
  voices: string;
}

/** The plan's numbers for the details pane. */
export function planFacts(plan: DubSyncPlan): PlanFacts {
  const dubs = plan.segments.filter((s) => s.kind === "dub").length;
  const fills = plan.segments.filter((s) => s.kind === "fill").length;
  const cuts = plan.segments.filter(isCut).length;
  let frameRate = "—";
  if (plan.videoFps) {
    const same = plan.dubRate === null || Math.abs(plan.dubRate - plan.videoFps) <= 1e-6;
    frameRate = same ? `${formatFps(plan.videoFps)} fps, both` : `${formatFps(plan.videoFps)} fps, dub ${formatFps(plan.dubRate!)}`;
    if (plan.rateConfirmed === false) frameRate += " (assumed)";
  } else if (Math.abs(plan.speed - 1) > 1e-9) {
    frameRate = `Dub played at ${plan.speed.toFixed(6)}×`;
  }
  const voices = plan.voicePieces?.length ?? 0;
  return {
    dubs,
    fills,
    cuts,
    stretches: `${dubs} of dub`,
    fromOriginal:
      plan.filledS > 0 ? (cuts > 0 ? `${formatSpan(plan.filledS)} · ${plural(cuts, "cut")}` : formatSpan(plan.filledS)) : "None",
    fillLevel: fills > 0 ? `${plan.fillGainDb >= 0 ? "+" : ""}${plan.fillGainDb.toFixed(1)} dB` : "—",
    frameRate,
    voices: voices > 0 ? plural(voices, "scene") : "None",
  };
}

/** One line on what a plan does: "6 stretches · 5 cuts filled from the original". */
export function planLine(plan: DubSyncPlan): string {
  const { dubs, cuts } = planFacts(plan);
  const head = plural(dubs, "stretch", "stretches");
  if (cuts > 0) return `${head} · ${plural(cuts, "cut")} filled from the original`;
  if (plan.filledS > 0) return `${head} · ${formatSpan(plan.filledS)} from the original`;
  return `${head} · the dub throughout`;
}

/** "Dub stretch 4 of 6": a piece counted among its own kind. */
export function stretchName(plan: DubSyncPlan, index: number): string {
  const segment = plan.segments[index];
  if (!segment) return "";
  const same = plan.segments.filter((s) => s.kind === segment.kind);
  const n = plan.segments.slice(0, index + 1).filter((s) => s.kind === segment.kind).length;
  return `${segment.kind === "dub" ? "Dub" : "Original"} stretch ${n} of ${same.length}`;
}

/** A stretch's span, as the status display says it. */
export const stretchSpan = (segment: DubSegment) => `${formatClock(segment.startS)} – ${formatClock(segment.endS)}`;

/** An offset as the offset box shows it: signed, to the millisecond. */
export function formatOffset(seconds: number): string {
  const rounded = Math.round(seconds * 1000) / 1000 || 0;
  return `${rounded < 0 ? "−" : "+"}${Math.abs(rounded).toFixed(3)}`;
}

/** Whatever was typed into the offset box, as seconds; null when it is not a number. */
export function parseOffset(text: string): number | null {
  const cleaned = text
    .replace(/[−–—]/g, "-")
    .replace(/,/g, ".")
    .replace(/\s+/g, "")
    .replace(/s$/i, "");
  if (!/^[+-]?(\d+\.?\d*|\.\d+)$/.test(cleaned)) return null;
  const value = Number(cleaned);
  return Number.isFinite(value) ? value : null;
}

/* ------------------------------------------------------------ verification */

export interface Verdict {
  tone: "ok" | "warn" | "bad";
  /** Share of the sweep within the audible bound, percent. */
  share: number | null;
  /** Stretches the check found audibly out. */
  audible: number;
}

/** How the written track measured. The sweep counts as much as the spots:
 *  a dozen spots can all land on fills or on the one stretch that is right,
 *  and the sweep is what covers the whole runtime. */
export function verdict(v: DubVerification): Verdict {
  const share = v.sweepMeasured > 0 ? Math.round((100 * v.sweepWithinAudible) / v.sweepMeasured) : null;
  const audible = v.stretches.length;
  const tone: Verdict["tone"] =
    audible > 0 || (share !== null && share < 80)
      ? "bad"
      : v.worstMs === null
        ? "warn"
        : v.worstMs <= AUDIBLE_MS / 2 && (share === null || share >= 95)
          ? "ok"
          : "warn";
  return { tone, share, audible };
}

/** The verdict in one line. */
export function verdictLine(v: DubVerification): string {
  const { share, audible } = verdict(v);
  if (audible > 0) {
    return `${plural(audible, "stretch", "stretches")} audibly out` + (share !== null ? ` · ${share}% within ${AUDIBLE_MS} ms` : "");
  }
  const measured = v.spots.some((s) => s.residualMs !== null);
  if (!measured || v.worstMs === null) return "Could not be measured";
  return (
    `${v.typicalMs !== null ? `${formatMs(v.typicalMs)} ms typical, ` : ""}${formatMs(v.worstMs)} ms at worst` +
    (share !== null && share < 95 ? ` · ${share}% of the runtime within ${AUDIBLE_MS} ms` : "")
  );
}

/** Stretches the check says to go and look at. */
export const toCheck = (job: DubQueueJob) => job.verification?.stretches.length ?? 0;

/** The Sync error column: the typical error, measured. */
export function syncError(job: DubQueueJob | null): string | null {
  const typical = job?.status === "done" ? job.verification?.typicalMs : null;
  return typical === null || typical === undefined ? null : `${formatMs(typical)} ms`;
}

/** An engine error, short enough for a status cell. */
export function shortReason(error: string | null): string {
  if (!error) return "Failed";
  const first = error.split(/(?<=[a-z0-9)])[.:;](?:\s|$)|\s—\s|\n/i)[0].trim();
  const text = first.length > 64 ? `${first.slice(0, 63)}…` : first;
  return text ? text[0].toUpperCase() + text.slice(1) : "Failed";
}

/** Why a pair failed, in a word or two for its status cell, for the
 *  failures the engine names; anything else as its first clause. */
export function failWord(error: string | null): string {
  const e = (error ?? "").toLowerCase();
  if (/different (edit|cut)/.test(e)) return "Different edit";
  if (/no part of the dub could be (matched|placed)/.test(e)) return "No match";
  if (/is silent from start to end/.test(e)) return "Silent audio";
  if (/has no audio/.test(e)) return "No audio";
  if (/is only [\d.]+ s long/.test(e)) return "Too short";
  return shortReason(error);
}

/** Why a pair failed, for the status display's second line. */
function failLine(error: string | null): string {
  if (/different (edit|cut)/i.test(error ?? "")) return "the dub is a different edit";
  return lowerFirst(error ?? "could not be synced");
}

const lowerFirst = (text: string) => (text && /^[A-Z][a-z]/.test(text) ? text[0].toLowerCase() + text.slice(1) : text);

/** An engine message as a sentence on its own. */
export const sentence = (text: string) => (text ? text[0].toUpperCase() + text.slice(1) : text);

/** A track's length: "5m 00.0s", or "2:20:30" once it runs to hours. */
export function formatLength(seconds: number): string {
  return seconds >= 3600 ? formatClock(seconds).replace(/\.\d+$/, "") : formatSpan(seconds);
}

/* ------------------------------------------------------------------ rows */

export interface QueueRow {
  /** The video's path; a dub's path for a dub without its movie. */
  key: string;
  videoPath: string | null;
  videoName: string | null;
  dubPath: string | null;
  dubName: string | null;
  videoTrack: number;
  dubTrack: number;
  /** The job of the last run for this very pair, if any. */
  job: DubQueueJob | null;
  paired: boolean;
  /** Paired by hand, or taken out by hand. */
  manual: boolean;
  skipped: boolean;
  /** The engine has not said yet what goes with what. */
  pending: boolean;
}

/** The queue's rows. During a run, its jobs as they were queued; otherwise
 *  the movies as paired now, each with the result of the last run for the
 *  same movie and dub. */
export function buildRows({
  running,
  jobs,
  pairs,
  videos,
  dubs,
  overrides,
  trackChoices,
}: {
  running: boolean;
  jobs: DubQueueJob[];
  pairs: MatchPair[] | null;
  videos: FileItem[];
  dubs: FileItem[];
  overrides: PairOverrides;
  trackChoices: Record<string, number>;
}): QueueRow[] {
  if (running && jobs.length > 0) {
    return jobs.map((job) => ({
      key: job.videoPath || `job-${job.id}`,
      videoPath: job.videoPath,
      videoName: job.name,
      dubPath: job.dubPath,
      dubName: job.dubName,
      videoTrack: job.videoTrack,
      dubTrack: job.dubTrack,
      job,
      paired: true,
      manual: false,
      skipped: false,
      pending: false,
    }));
  }
  if (videos.length === 0) {
    return dubs.map((dub) => ({
      key: dub.path,
      videoPath: null,
      videoName: null,
      dubPath: dub.path,
      dubName: dub.name,
      videoTrack: 0,
      dubTrack: trackChoices[dub.path] ?? 0,
      job: null,
      paired: false,
      manual: false,
      skipped: false,
      pending: false,
    }));
  }
  const byVideo = new Map((pairs ?? []).map((pair) => [pair.primaryPath, pair]));
  const rows: QueueRow[] = [];
  const seen = new Set<string>();
  const add = (videoPath: string, videoName: string, pair: MatchPair | undefined) => {
    seen.add(videoPath);
    const videoTrack = trackChoices[videoPath] ?? pair?.primaryTrack ?? 0;
    const dubTrack = pair ? (trackChoices[pair.secondaryPath] ?? pair.secondaryTrack ?? 0) : 0;
    rows.push({
      key: videoPath,
      videoPath,
      videoName,
      dubPath: pair?.secondaryPath ?? null,
      dubName: pair?.secondaryName ?? null,
      videoTrack,
      dubTrack,
      job: pair ? (jobs.find((j) => j.videoPath === videoPath && j.dubPath === pair.secondaryPath) ?? null) : null,
      paired: !!pair,
      manual: videoPath in overrides,
      skipped: overrides[videoPath] === null,
      pending: pairs === null && dubs.length > 0,
    });
  };
  for (const video of videos) add(video.path, video.name, byVideo.get(video.path));
  for (const pair of pairs ?? []) if (!seen.has(pair.primaryPath)) add(pair.primaryPath, pair.primaryName, pair);
  return rows;
}

export interface RowStatus {
  s: St;
  text?: string;
  pct?: number | null;
  /** The whole message, when the status only has room for a word. */
  title?: string;
}

/** A job's state in the one status vocabulary. */
export function jobStatus(job: DubQueueJob): RowStatus {
  switch (job.status) {
    case "queued":
      return { s: "wait" };
    case "running":
      return { s: "run", pct: job.percent, text: stageLabel(job.stage) };
    case "done": {
      const n = toCheck(job);
      return { s: "ok", text: n > 0 ? `Done · ${plural(n, "stretch", "stretches")} to check` : "Done" };
    }
    case "failed":
      return { s: "bad", text: failWord(job.error), title: job.error ? sentence(job.error) : undefined };
    case "cancelled":
      return { s: "wait", text: "Stopped" };
  }
}

/** A row's state, run or not. */
export function rowStatus(row: QueueRow, dubsAdded: number): RowStatus {
  if (row.job) return jobStatus(row.job);
  if (!row.videoPath) return { s: "warn", text: "Needs its movie" };
  if (row.pending) return { s: "wait", text: "Pairing…" };
  if (row.paired) return { s: "ready" };
  if (row.skipped) return { s: "wait", text: "Skipped" };
  return dubsAdded > 0 ? { s: "bad", text: "Not paired" } : { s: "wait", text: "Needs a dub" };
}

/* ----------------------------------------------------------- status display */

export interface PageStatus {
  icon?: "run" | "ok" | "warn" | "bad" | "idle";
  l1: string;
  l2?: string;
  pct?: number | null;
  time?: string;
}

/** How long a run has left, from how far it has come. */
export function timeLeft(startedAt: number | null, fraction: number, now: number): string | undefined {
  if (!startedAt || fraction < 0.02 || fraction >= 1) return undefined;
  const remaining = ((now - startedAt) * (1 - fraction)) / fraction;
  const minutes = remaining / 60_000;
  if (minutes < 1) return "Under a minute left";
  if (minutes < 60) return `${Math.ceil(minutes)} min left`;
  return `${Math.floor(minutes / 60)} h ${Math.round(minutes % 60)} min left`;
}

/** A length difference as people say it: "10m 20s". */
export function formatGap(seconds: number): string {
  const s = Math.round(Math.abs(seconds));
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${s}s`;
}

/** How the dub's length compares with the video's, before a run. */
export function gapNote(videoS: number, dubS: number): string {
  const diff = dubS - videoS;
  if (Math.abs(diff) < 0.5) return "The dub and the video are the same length";
  return `The dub is ${formatGap(diff)} ${diff < 0 ? "shorter" : "longer"} than the video`;
}

/** How the engine paired a list, in words. */
function pairedBy(method: string | null): string | undefined {
  if (!method) return undefined;
  const m = method.toLowerCase();
  const base = m.includes("episode") ? "Paired by episode number" : m.includes("filename") || m.includes("name") ? "Paired by name" : m.includes("order") ? "Paired in list order" : undefined;
  return base;
}

export interface StatusInput {
  rows: QueueRow[];
  queue: DubQueueState;
  /** The row shown in the timeline and the details. */
  shown: QueueRow | null;
  videos: number;
  dubs: number;
  pairingLoading: boolean;
  /** Lengths of the only pair, when there is one and both are known. */
  lengths: { videoS: number; dubS: number } | null;
  method: string | null;
  manual: number;
  /** Edits to the shown job's cuts not written yet. */
  edit: { changes: number; stretch: DubSegment | null; stretchName: string | null } | null;
  workers: number;
  now: number;
  /** The queue's jobs in the run under way, when it is not all of them (a retry). */
  runJobs?: number[] | null;
}

/** What the status display says: the one place that always says what the
 *  page is doing. */
export function pageStatus(i: StatusInput): PageStatus {
  const { queue, rows } = i;
  const jobs = queue.jobs;
  if (i.videos + i.dubs === 0) return { l1: "Drop movies and their dubs" };

  if (queue.rerender) {
    const job = jobs[queue.rerender.job];
    return { icon: "run", l1: "Writing the track with your cuts", l2: job ? shortTitle(job.name) : undefined, pct: job?.percent ?? null, time: job ? `${Math.round(job.percent)}%` : undefined };
  }

  if (queue.status === "running") {
    const run = i.runJobs ? i.runJobs.map((k) => jobs[k]).filter((j): j is DubQueueJob => !!j) : jobs;
    if (run.length === 1) {
      const job = run[0];
      const pct = job.status === "running" ? job.percent : 0;
      return { icon: "run", l1: job.status === "queued" ? "Starting…" : stageLabel(job.stage), l2: shortTitle(job.name), pct, time: `${Math.round(pct)}%` };
    }
    const running = run.filter((j) => j.status === "running");
    const closed = run.filter((j) => j.status !== "queued" && j.status !== "running").length;
    const focus = i.shown?.job?.status === "running" ? i.shown.job : running[0];
    const fraction = (closed + running.reduce((sum, j) => sum + j.percent / 100, 0)) / run.length;
    return {
      icon: "run",
      // Through the queue as far as the jobs started: two done and one
      // running is the third of five.
      l1: `Syncing ${Math.min(run.length, Math.max(1, closed + running.length))} of ${run.length}` + (focus ? ` · ${stageLabel(focus.stage)}` : ""),
      l2: `${collectionTitle(run.map((j) => j.name))} · ${Math.min(i.workers, run.length)} at a time`,
      pct: fraction * 100,
      time: timeLeft(queue.startedAt, fraction, i.now),
    };
  }

  if (i.edit && i.edit.changes > 0) {
    return {
      icon: "warn",
      l1: `${plural(i.edit.changes, "change")} not applied`,
      l2: i.edit.stretch && i.edit.stretchName ? `${i.edit.stretchName} · ${stretchSpan(i.edit.stretch)}` : i.shown?.videoName ? shortTitle(i.shown.videoName) : undefined,
    };
  }

  if (queue.status === "failed") return { icon: "bad", l1: "The sync did not finish", l2: queue.error ?? undefined };

  const ready = rows.filter((r) => r.paired && !r.job);
  if (ready.length > 0) {
    const pairs = rows.filter((r) => r.paired).length;
    const unpaired = rows.filter((r) => r.videoPath && !r.paired && !r.skipped).length;
    let l2: string | undefined;
    if (pairs === 1 && rows.length === 1 && i.lengths) l2 = gapNote(i.lengths.videoS, i.lengths.dubS);
    else if (unpaired > 0) l2 = `${plural(unpaired, "movie")} without a dub`;
    else {
      const by = pairedBy(i.method);
      l2 = by && i.manual > 0 ? `${by} · ${i.manual} by hand` : by;
    }
    return { l1: `${plural(ready.length, "pair")} ready`, l2 };
  }

  const finished = rows.map((r) => r.job).filter((j): j is DubQueueJob => !!j && j.status !== "queued" && j.status !== "running");
  if (finished.length > 0) {
    const done = finished.filter((j) => j.status === "done");
    const failed = finished.filter((j) => j.status === "failed");
    const stopped = finished.filter((j) => j.status === "cancelled");
    if (failed.length > 0) {
      const first = failed[0];
      return {
        icon: "bad",
        l1: `${done.length} synced · ${failed.length} failed`,
        l2: `${shortTitle(first.name)}: ${failLine(first.error)}`,
      };
    }
    if (stopped.length > 0) {
      return { icon: "warn", l1: done.length > 0 ? `Stopped · ${done.length} synced` : "Stopped", l2: `${plural(stopped.length, "pair")} not synced` };
    }
    const shownJob = i.shown?.job?.status === "done" ? i.shown.job : done[0];
    if (done.length === 1 || finished.length === 1) {
      const job = shownJob;
      const v = job.verification;
      const line = job.plan ? planLine(job.plan) : undefined;
      if (!v) return { icon: "ok", l1: "Synced", l2: line };
      const { tone } = verdict(v);
      if (tone === "ok" && v.typicalMs !== null && v.worstMs !== null) {
        return { icon: "ok", l1: `On the lips — ${formatMs(v.typicalMs)} ms typical, ${formatMs(v.worstMs)} ms at worst`, l2: line };
      }
      return { icon: tone, l1: verdictLine(v), l2: line };
    }
    const check = done.filter((j) => (j.verification ? verdict(j.verification).tone !== "ok" : false)).length;
    return {
      icon: check > 0 ? "warn" : "ok",
      l1: `${done.length} synced` + (check > 0 ? ` · ${check} to check` : ""),
      l2: shownJob?.verification ? `${shortTitle(shownJob.name)}: ${verdictLine(shownJob.verification)}` : undefined,
    };
  }

  if (i.pairingLoading) return { l1: "Pairing…" };
  if (i.dubs === 0) return { l1: plural(i.videos, "movie"), l2: "Add their dubs" };
  if (i.videos === 0) return { l1: plural(i.dubs, "dub"), l2: "Add the movies they belong to" };
  return { icon: "warn", l1: "Nothing paired", l2: "Choose a dub for each movie" };
}

/* ------------------------------------------------------------------ history */

/** What History keeps of a finished queue. */
export function batchSummary(
  jobs: { name: string }[],
  outcomes: DubJobOutcome[],
  cancelled: boolean,
): { name: string; tone: "ok" | "warn" | "bad"; text: string } {
  const name = collectionTitle(jobs.map((j) => j.name));
  const ok = outcomes.filter((o) => !o.cancelled && !o.error && !o.plan?.error && o.output);
  const failed = outcomes.filter((o) => !o.cancelled && (o.error || o.plan?.error || !o.output));
  const check = ok.filter((o) => (o.verification?.stretches.length ?? 0) > 0).length;
  if (ok.length === 0 && failed.length === 1 && !cancelled) {
    const reason = failed[0].error ?? failed[0].plan?.error ?? "could not be synced";
    return { name, tone: "bad", text: `Failed — ${lowerFirst(failWord(reason))}` };
  }
  const parts = [`${ok.length} synced`];
  if (failed.length) parts.push(`${failed.length} failed`);
  if (check) parts.push(`${check} to check`);
  const text = (cancelled ? "Stopped, " : "") + parts.join(", ");
  const tone = failed.length > 0 ? (ok.length > 0 ? "warn" : "bad") : cancelled || check > 0 ? "warn" : "ok";
  return { name, tone, text };
}

/** The channel layout of a track, as a release names it. */
export function channelName(channels: number | null | undefined): string | null {
  if (!channels) return null;
  return ({ 1: "1.0", 2: "2.0", 6: "5.1", 8: "7.1" } as Record<number, string>)[channels] ?? `${channels}ch`;
}

/* ------------------------------------------------------------------ ruler */

/** Round-figure spacings for the ruler, finest first. */
const RULER_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];

/** The ruler's ticks over a view: at most ten, and never closer than 70 px,
 *  labelled h:mm over a whole film, h:mm:ss zoomed in, and to the tenth
 *  below a second. None within 40% of a step of the right edge, where a
 *  label would be cut off. */
export function rulerTicks(startS: number, endS: number, width: number): { t: number; label: string }[] {
  const span = endS - startS;
  if (!(span > 0) || width <= 0) return [];
  const most = Math.max(1, Math.min(10, width / 70));
  const step = RULER_STEPS.find((s) => span / s <= most) ?? 3600;
  const zoomed = span < 2000;
  const ticks: { t: number; label: string }[] = [];
  for (let k = Math.ceil(startS / step - 1e-9); k * step <= endS - step * 0.4; k++) {
    const t = Math.round(k * step * 1000) / 1000;
    const clock = formatClock(t);
    ticks.push({ t, label: step < 1 ? clock.slice(0, -2) : zoomed ? clock.slice(0, -4) : clock.slice(0, -7) });
  }
  return ticks;
}

/* ------------------------------------------------------------------ queue */

export type PageQueueAction =
  | DubQueueAction
  /** Run some finished jobs again -- a retry, or a retry with another dub --
   *  keeping every other job's result. */
  | { type: "retryStarted"; jobs: { index: number; dubPath?: string; dubName?: string; dubTrack?: number }[] };

const closed = (jobs: DubQueueJob[]) => jobs.filter((job) => job.status !== "queued" && job.status !== "running").length;

/** The shared queue reducer, plus what the page adds: retrying some jobs,
 *  and a failed batch closing the jobs it never finished. */
export function pageQueueReducer(state: DubQueueState, action: PageQueueAction): DubQueueState {
  if (action.type === "retryStarted") {
    if (state.status === "running") return state;
    const jobs = state.jobs.slice();
    for (const retry of action.jobs) {
      const job = jobs[retry.index];
      if (!job) continue;
      jobs[retry.index] = {
        ...job,
        dubPath: retry.dubPath ?? job.dubPath,
        dubName: retry.dubName ?? job.dubName,
        dubTrack: retry.dubTrack ?? job.dubTrack,
        status: "queued",
        percent: 0,
        stage: null,
        plan: null,
        enginePlan: null,
        draft: null,
        output: null,
        verification: null,
        muxedPath: null,
        error: null,
      };
    }
    return { ...state, status: "running", jobs, done: closed(jobs), startedAt: Date.now(), error: null, rerender: null };
  }
  const next = dubQueueReducer(state, action);
  if (action.type === "batchFailed" && next !== state) {
    const jobs = next.jobs.map((job) =>
      job.status === "queued" || job.status === "running" ? { ...job, status: "failed" as const, error: action.message, percent: 0, stage: null } : job,
    );
    return { ...next, jobs, done: closed(jobs) };
  }
  return next;
}
