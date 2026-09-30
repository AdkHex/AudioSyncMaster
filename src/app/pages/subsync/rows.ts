/** What the Subsync page shows about its files and its run, in words: the
 *  status of each row, its short result, the status display (the toolbar's
 *  LCD) and the line History keeps. Pure, so it is tested on its own. */

import { cueCps, formatReportValue, hdrBadge, plainText, reportLabel } from "@/lib/subsync/format";
import type { InputItem, PoolFile } from "@/lib/subsync/inputs";
import type { RunJob, RunState } from "@/lib/subsync/reducer";
import { runProgress } from "@/lib/subsync/reducer";
import type { InputShape, ToolId } from "@/lib/subsync/tools";
import { LANGUAGES, type CueData, type SubJobOutcome, type SubTask, type SubtitleTrackInfo, type VideoInfo } from "@/lib/subsync/types";
import { formatSize } from "@/lib/types";
import type { LcdProps } from "@/ui/frame";
import type { St } from "@/ui/kit";

/** The tool strip, in the mockup's order and words. */
export const STRIP: { id: ToolId; label: string }[] = [
  { id: "sync", label: "Sync" },
  { id: "ocr", label: "OCR" },
  { id: "translate", label: "Translate" },
  { id: "fps", label: "Frame rate" },
  { id: "generate", label: "Generate" },
  { id: "style", label: "Style" },
  { id: "hdrSubs", label: "HDR subtitles" },
  { id: "tonemap", label: "Tone-map" },
  { id: "formats", label: "Convert & mux" },
];

export const stripLabel = (tool: ToolId) => STRIP.find((entry) => entry.id === tool)?.label ?? tool;

const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

/** "Reading the subtitle" -> "reading the subtitle"; "OCR…" stays. */
export function lowerFirst(text: string): string {
  if (text.length > 1 && text[1] === text[1].toUpperCase() && /[A-Z]/.test(text[1])) return text;
  return text.charAt(0).toLowerCase() + text.slice(1);
}

// ---------------------------------------------------------------- problems

/** The engine-independent problems inputs.ts reports, in three lengths: the
 *  row's status word, what "Skipped · …" says, and the LCD's sentence. */
const PROBLEMS: Record<string, { label: string; short: string; needs: string }> = {
  "Choose the video to sync it to.": { label: "Needs a video", short: "no video", needs: "needs a video" },
  "An image subtitle: run OCR on it first.": {
    label: "Image subtitle",
    short: "image subtitle",
    needs: "is an image subtitle: run OCR on it first",
  },
  "Needs an image subtitle (PGS or VobSub).": {
    label: "Not an image subtitle",
    short: "not an image subtitle",
    needs: "is text, not an image subtitle",
  },
  "No image subtitle tracks in this video.": { label: "No image tracks", short: "no image tracks", needs: "has no image subtitle tracks" },
  "No text subtitle tracks, and no subtitle file matched it.": {
    label: "No subtitle",
    short: "no subtitle",
    needs: "has no subtitle track, and no subtitle file matched it",
  },
  "No subtitle tracks in this video.": { label: "No subtitle tracks", short: "no subtitle tracks", needs: "has no subtitle tracks" },
  "Not a video.": { label: "Not a video", short: "not a video", needs: "is not a video" },
  "Add the video the subtitles go into.": { label: "Needs a video", short: "no video", needs: "needs a video" },
  "Add the subtitle files to put into it.": { label: "Needs subtitles", short: "no subtitles", needs: "needs subtitles" },
};

export const problemLabel = (problem: string) => PROBLEMS[problem]?.label ?? problem.replace(/\.$/, "");
export const problemShort = (problem: string) => PROBLEMS[problem]?.short ?? lowerFirst(problem.replace(/\.$/, ""));
export const problemSentence = (name: string, problem: string) =>
  PROBLEMS[problem] ? `${name} ${PROBLEMS[problem].needs}` : `${name}: ${lowerFirst(problem)}`;

// ------------------------------------------------------------------- rows

export interface RowStatus {
  s: St;
  text?: string;
  pct?: number | null;
  /** The whole message, when the status only has room for a word. */
  title?: string;
}

/** A row's name: the subtitle, or the video when the subtitle is its track. */
export function itemName(item: InputItem): string {
  if (item.embedded || !item.subtitleName) return item.videoName ?? "Untitled";
  return item.subtitleName;
}

/** Where a synced subtitle's offset changes, in words. The engine lists
 *  each change with the offset from there on: an earlier offset means the
 *  video lacks a scene the subtitle was timed with (cut), a later one that
 *  it has one more (added). A count alone is a scene change. */
export function splitsText(report: Record<string, unknown>): string | null {
  const splits = report.splits;
  if (typeof splits === "number") return splits > 0 ? plural(splits, "scene change") : null;
  if (!Array.isArray(splits) || splits.length === 0) return null;
  let before = typeof report.offsetMs === "number" ? report.offsetMs : null;
  let cut = 0;
  let added = 0;
  for (const split of splits as { offsetMs?: unknown }[]) {
    const after = typeof split?.offsetMs === "number" ? split.offsetMs : null;
    if (before !== null && after !== null) {
      if (after < before) cut += 1;
      else if (after > before) added += 1;
    }
    before = after;
  }
  if (cut === splits.length) return `${plural(cut, "scene")} cut`;
  if (added === splits.length) return `${plural(added, "scene")} added`;
  return plural(splits.length, "scene change");
}

/** What a finished job says in the Status column. */
export function doneStatus(outcome: SubJobOutcome): RowStatus {
  const r = outcome.report ?? {};
  const num = (key: string) => (typeof r[key] === "number" ? (r[key] as number) : null);
  const splits = outcome.task === "sync" ? splitsText(r) : null;
  if (splits) return { s: "warn", text: `Done · ${splits}`, title: outcome.summary };
  const check = outcome.task === "ocr" ? num("lowConfidence") : outcome.task === "style" ? num("issuesAfter") : null;
  if (check) return { s: "warn", text: `Done · ${check.toLocaleString()} to check`, title: outcome.summary };
  if (outcome.warnings.length > 0) {
    return { s: "warn", text: `Done · ${plural(outcome.warnings.length, "warning")}`, title: outcome.warnings.join("\n") };
  }
  return { s: "ok", text: "Done", title: outcome.summary || undefined };
}

export function jobStatus(job: RunJob): RowStatus {
  switch (job.status) {
    case "queued":
      return { s: "wait", text: "Waiting" };
    case "running":
      // The percent, as the mockup has it; the stage is in the tooltip and the status display.
      return { s: "run", pct: job.percent, title: job.stage ?? undefined };
    case "done":
      return job.outcome ? doneStatus(job.outcome) : { s: "ok" };
    case "failed":
      return { s: "bad", text: "Failed", title: job.outcome?.error ?? undefined };
    case "cancelled":
      return { s: "warn", text: "Stopped" };
  }
}

/** A row's status: its job's when it is part of the run, otherwise whether
 *  it can run. Rows a run left out say they were skipped, and why. */
export function rowStatus(item: InputItem, file: PoolFile | undefined, job: RunJob | null, ranHere: boolean, running: boolean): RowStatus {
  if (job) return jobStatus(job);
  if (file?.pending) return { s: "wait", text: "Reading…" };
  if (item.problem) {
    if (file?.error && item.problem === file.error) return { s: "bad", text: "Can't read", title: item.problem };
    if (ranHere) return { s: running ? "wait" : "warn", text: `Skipped · ${problemShort(item.problem)}`, title: item.problem };
    return { s: "warn", text: problemLabel(item.problem), title: item.problem };
  }
  return { s: "ready" };
}

/** "1920x1080" / a video's size -> "1080p". */
export function resolutionLabel(width: number, height: number): string {
  const standard: Record<number, string> = { 3840: "2160p", 4096: "2160p", 2560: "1440p", 1920: "1080p", 1280: "720p", 720: "480p" };
  return standard[width] ?? `${width}×${height}`;
}

/** The short result of a finished job: "+3.250 s", "1,204 lines". */
export function shortResult(outcome: SubJobOutcome | null | undefined): string | null {
  if (!outcome || outcome.error || outcome.cancelled) return null;
  const r = outcome.report ?? {};
  const num = (key: string) => (typeof r[key] === "number" ? (r[key] as number) : null);
  switch (outcome.task) {
    case "sync":
      return num("offsetMs") !== null ? formatReportValue("offsetMs", r.offsetMs) : null;
    case "fps":
      return num("ratio") !== null ? `×${num("ratio")!.toFixed(6)}` : null;
    case "style":
      if (num("fixes")) return `${num("fixes")!.toLocaleString()} fixed`;
      return num("issuesAfter") !== null ? plural(num("issuesAfter")!, "issue") : null;
    case "extract":
      return num("extracted") !== null ? plural(num("extracted")!, "track") : null;
    case "mux":
      return num("added") !== null ? plural(num("added")!, "track") : null;
    case "tonemap": {
      const size = typeof r.outputResolution === "string" ? /^(\d+)x(\d+)$/.exec(r.outputResolution) : null;
      return size ? resolutionLabel(Number(size[1]), Number(size[2])) : null;
    }
    default: {
      const cues = num("cues") ?? outcome.preview?.count ?? null;
      return cues !== null ? plural(cues, "line") : null;
    }
  }
}

// ------------------------------------------------------------ file facts

/** "Dolby Vision 8.1", "HDR10", …; null for SDR. */
export function hdrLabel(video: VideoInfo | null | undefined): string | null {
  if (!video || video.hdr === "sdr") return null;
  if (video.hdr === "dv" && video.dvProfile !== null) {
    return `Dolby Vision ${video.dvProfile}${video.dvCompatibility ? `.${video.dvCompatibility}` : ""}`;
  }
  return video.hdr === "dv" ? "Dolby Vision" : hdrBadge(video);
}

/** One line of what a video is: "2160p · Dolby Vision 8.1 · 71.4 GB". */
export function videoFacts(file: PoolFile): string {
  const parts: string[] = [];
  if (file.video) parts.push(resolutionLabel(file.video.width, file.video.height));
  const hdr = hdrLabel(file.video);
  if (hdr) parts.push(hdr);
  if (file.size) parts.push(formatSize(file.size));
  return parts.join(" · ");
}

// ------------------------------------------------------------------- LCD

/** What each task did, for "2 synced · 1 skipped". */
export const DONE_VERB: Record<SubTask, string> = {
  sync: "synced",
  ocr: "read",
  translate: "translated",
  fps: "converted",
  generate: "generated",
  style: "checked",
  hdrSubs: "adjusted",
  tonemap: "tone-mapped",
  convert: "converted",
  extract: "extracted",
  mux: "muxed",
};

const EMPTY_PROMPT: Record<InputShape, string> = {
  subtitle: "drop subtitles and their videos",
  video: "drop videos",
  mux: "drop a video and its subtitles",
};

export function emptyTitle(shape: InputShape): string {
  const text = EMPTY_PROMPT[shape];
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Before a run: what is there, and what stops it running. */
export function readyLcd(label: string, shape: InputShape, items: InputItem[], files: PoolFile[]): LcdProps {
  if (files.length === 0) return { l1: `${label}: ${EMPTY_PROMPT[shape]}` };
  if (shape === "mux") {
    const videos = files.filter((file) => file.kind === "video").length;
    const subtitles = files.filter((file) => file.kind === "subtitle").length;
    const l1 = `${label} · ${plural(videos, "video")}, ${plural(subtitles, "subtitle")}`;
    if (files.some((file) => file.pending)) return { l1, l2: "Reading the files…" };
    const problem = items[0]?.problem;
    return problem ? { icon: "warn", l1, l2: problem } : { l1, l2: "Ready" };
  }
  const l1 = `${label} · ${plural(items.length, "file")}`;
  if (files.some((file) => file.pending)) return { l1, l2: "Reading the files…" };
  const blocked = items.filter((item) => item.problem);
  if (blocked.length === 0) return { l1, l2: "Ready" };
  const first = blocked[0];
  return {
    icon: "warn",
    l1: `${items.length - blocked.length} of ${items.length} ready`,
    l2: problemSentence(itemName(first), first.problem!),
  };
}

/** During a run: the tool, what the first running job is doing, overall %. */
export function runningLcd(label: string, run: RunState): LcdProps {
  const progress = runProgress(run);
  const active = run.jobs.find((job) => job.status === "running");
  const stage = active?.stage ? lowerFirst(active.stage) : active ? "starting" : progress.done > 0 ? `${progress.done} of ${progress.total} done` : "starting";
  return {
    icon: "run",
    l1: `${label}: ${stage}`,
    l2: active?.label ?? (run.jobs.length > 1 ? plural(run.jobs.length, "file") : run.jobs[0]?.label),
    pct: progress.percent,
    time: `${Math.round(progress.percent)}%`,
  };
}

const baseName = (path: string) => path.replace(/^.*[\\/]/, "");

/** After a run: how many did what, and the first thing worth a look. */
export function doneLcd(task: SubTask, run: RunState, skipped: number): LcdProps {
  if (run.error) return { icon: "bad", l1: "The run did not finish", l2: run.error };
  const done = run.jobs.filter((job) => job.status === "done");
  const failed = run.jobs.filter((job) => job.status === "failed");
  const stopped = run.jobs.filter((job) => job.status === "cancelled");
  const warned = done.find((job) => (job.outcome?.warnings.length ?? 0) > 0);
  const check = done.some((job) => job.outcome && doneStatus(job.outcome).s === "warn");

  const parts: string[] = [];
  const single = run.jobs.length === 1 && done.length === 1 ? done[0].outcome : null;
  if (single && task === "ocr" && typeof single.report?.cues === "number") {
    parts.push(`${plural(single.report.cues as number, "line")} read`);
    const low = typeof single.report.lowConfidence === "number" ? (single.report.lowConfidence as number) : 0;
    if (low) parts.push(`${low.toLocaleString()} to check`);
  } else if (done.length > 0 || (failed.length === 0 && stopped.length === 0)) {
    parts.push(`${done.length.toLocaleString()} ${DONE_VERB[task]}`);
  }
  if (failed.length) parts.push(`${failed.length.toLocaleString()} failed`);
  if (stopped.length) parts.push(`${stopped.length.toLocaleString()} stopped`);
  if (skipped) parts.push(`${skipped.toLocaleString()} skipped`);

  const icon: LcdProps["icon"] = failed.length ? (done.length ? "warn" : "bad") : stopped.length || skipped || check ? "warn" : "ok";

  // A single OCR run already says how many lines to check; its second line
  // is where they went.
  const ocrSingle = !!single && task === "ocr";
  let l2: string | undefined;
  if (warned && !ocrSingle) l2 = `${warned.label}: ${lowerFirst(warned.outcome!.warnings[0])}`;
  else if (failed.length) l2 = `${failed[0].label}: ${failed[0].outcome?.error ?? "failed"}`;
  else if (run.status === "cancelled") l2 = "Stopped before every file was done.";
  else if (single) {
    const written = single.outputs.find((output) => output.kind !== "folder" && output.kind !== "report");
    l2 = written ? `${done[0].label} → ${baseName(written.path)}` : single.summary;
  } else if (check) {
    const first = done.find((job) => job.outcome && doneStatus(job.outcome).s === "warn")!;
    l2 = `${first.label}: ${lowerFirst(doneStatus(first.outcome!).text!.replace(/^Done · /, ""))}`;
  }
  return { icon, l1: parts.join(" · "), l2 };
}

/** The History row of a finished batch. */
export function historyLine(task: SubTask, run: RunState, skipped: number): { name: string; tone: "ok" | "warn" | "bad"; text: string } {
  const first = run.jobs[0]?.label ?? "Subtitles";
  const name = run.jobs.length > 1 ? `${first} and ${run.jobs.length - 1} more` : first;
  const lcd = doneLcd(task, run, skipped);
  const tone = lcd.icon === "bad" ? "bad" : lcd.icon === "warn" ? "warn" : "ok";
  const single = run.jobs.length === 1 ? run.jobs[0].outcome : null;
  const result = single ? shortResult(single) : null;
  if (single && !run.error && result && run.jobs[0].status === "done") {
    const verb = DONE_VERB[task];
    const text = task === "ocr" ? String(lcd.l1) : `${verb.charAt(0).toUpperCase()}${verb.slice(1)} ${result}`;
    return { name, tone, text };
  }
  return { name, tone, text: String(run.error ?? lcd.l1) };
}

// ------------------------------------------------------------ file labels

/** The file a row stands for: what Remove takes out. */
export const keyPath = (key: string) => key.replace(/^(sub|emb|vid):/, "");

const ISO_639_2: Record<string, string> = {
  eng: "English", jpn: "Japanese", kor: "Korean", zho: "Chinese", chi: "Chinese", fre: "French", fra: "French",
  spa: "Spanish", ita: "Italian", ger: "German", deu: "German", por: "Portuguese", hin: "Hindi", rus: "Russian",
  ara: "Arabic", nld: "Dutch", dut: "Dutch", swe: "Swedish", pol: "Polish", tur: "Turkish", tha: "Thai",
};

/** "eng" / "en" -> "English"; null when unknown or undetermined. */
export function languageLabel(code: string | null | undefined): string | null {
  if (!code || code === "und") return null;
  const key = code.toLowerCase();
  return LANGUAGES[key] ?? ISO_639_2[key] ?? code;
}

const CHANNELS: Record<number, string> = { 1: "mono", 2: "2.0", 6: "5.1", 8: "7.1" };

/** "Audio 1 · English · E-AC3 5.1" */
export function audioText(track: PoolFile["audioTracks"][number]): string {
  const codec = [track.codec?.toUpperCase().replace(/^EAC3$/, "E-AC3"), track.channels ? (CHANNELS[track.channels] ?? `${track.channels} ch`) : null]
    .filter(Boolean)
    .join(" ");
  // A title that only repeats the codec ("E-AC3 5.1") says nothing more.
  const title = track.title && track.title.toLowerCase() !== codec.toLowerCase() ? track.title : null;
  return [`Audio ${track.index + 1}`, languageLabel(track.language), codec || null, title].filter(Boolean).join(" · ");
}

/** "Subtitle 2 · English · PGS · forced · 1,204 events" */
export function subtitleTrackText(track: SubtitleTrackInfo): string {
  return [
    `Subtitle ${track.index + 1}`,
    languageLabel(track.language),
    track.title,
    formatName(track.codec),
    track.forced ? "forced" : null,
    track.hearingImpaired ? "SDH" : null,
    track.events !== null ? plural(track.events, "event") : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

/** "hdmv_pgs_subtitle" / "sup" -> "PGS", "subrip" -> "SRT". */
export function formatName(format: string | null | undefined): string {
  const f = (format ?? "").toLowerCase();
  if (/pgs|sup/.test(f)) return "PGS";
  if (/vobsub|dvd_sub|idx/.test(f)) return "VobSub";
  if (f === "subrip") return "SRT";
  if (f === "mov_text") return "MP4 text";
  if (f === "webvtt") return "VTT";
  return f.toUpperCase();
}

// ------------------------------------------------------------------- cues

/** Above this a line is hard to read for most audiences. */
export const FAST_CPS = 20;

export type CueMode = "cps" | "ocr";

/** Whether a line is worth a second look: an engine unsure of it, or (for
 *  timing) faster than most people read. */
export function needsCheck(cue: CueData, mode: CueMode, minConfidence: number): boolean {
  if (typeof cue.confidence === "number" && cue.confidence < minConfidence) return true;
  if (mode === "ocr") return false;
  const cps = cueCps(cue);
  return cps !== null && cps > FAST_CPS;
}

/** One line of a cue's visible text. */
export const cueLine = (text: string) => plainText(text).replace(/\n+/g, " / ");

/** The report's plain values as label / value rows; nested ones (split
 *  lists, per-issue details) stay in the log. */
export function reportRows(outcome: SubJobOutcome): [string, string][] {
  return Object.entries(outcome.report ?? {})
    .filter(([, value]) => typeof value !== "object" || value === null || (Array.isArray(value) && value.every((item) => typeof item !== "object")))
    .map(([key, value]) => [reportLabel(key), formatReportValue(key, value)]);
}
