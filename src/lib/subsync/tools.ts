/** The tools of the Subsync workspace: what each takes, what it writes and
 *  how it is described. Pure data, shared by the UI and the job builder. */

import type { OutputFormat, OutputOptions, SubTask, TaskOptions } from "./types";

/** An entry in the tool rail. "formats" groups three small tasks. */
export type ToolId =
  | "sync"
  | "ocr"
  | "translate"
  | "fps"
  | "generate"
  | "style"
  | "hdrSubs"
  | "tonemap"
  | "formats";

/** What one queue item is built from. */
export type InputShape =
  /** a subtitle (file or embedded track), optionally with its video */
  | "subtitle"
  /** a video */
  | "video"
  /** one video and several subtitles, all into one job */
  | "mux";

export interface ToolInfo {
  id: ToolId;
  label: string;
  description: string;
  tasks: SubTask[];
}

export const TOOLS: ToolInfo[] = [
  {
    id: "sync",
    label: "Sync",
    description: "Re-times a subtitle to a video: a constant offset, frame-rate drift, or a different offset per scene for cut or extended edits.",
    tasks: ["sync"],
  },
  {
    id: "ocr",
    label: "OCR",
    description: "Turns image subtitles (Blu-ray PGS, DVD VobSub) into text.",
    tasks: ["ocr"],
  },
  {
    id: "translate",
    label: "Translate",
    description: "Translates a subtitle scene by scene, with a glossary and an optional review pass.",
    tasks: ["translate"],
  },
  {
    id: "fps",
    label: "Frame rate",
    description: "Converts timing between frame rates (25 ↔ 23.976 and the rest), or between two known points.",
    tasks: ["fps"],
  },
  {
    id: "generate",
    label: "Generate",
    description: "Makes subtitles for a video that has none, by speech recognition in about 100 languages.",
    tasks: ["generate"],
  },
  {
    id: "style",
    label: "Style (Netflix)",
    description: "Checks and fixes line length, reading speed, durations, gaps and line breaks against the Netflix style guide.",
    tasks: ["style"],
  },
  {
    id: "hdrSubs",
    label: "HDR subtitles",
    description: "Tones subtitle brightness down so it does not glare on HDR and Dolby Vision video.",
    tasks: ["hdrSubs"],
  },
  {
    id: "tonemap",
    label: "Tone-map video",
    description: "Converts HDR10, HLG or Dolby Vision video to SDR at 4K, 1080p or 720p.",
    tasks: ["tonemap"],
  },
  {
    id: "formats",
    label: "Convert / Extract / Mux",
    description: "Converts between subtitle formats, pulls subtitle tracks out of a video, and adds them back in.",
    tasks: ["convert", "extract", "mux"],
  },
];

export const TASK_LABELS: Record<SubTask, string> = {
  sync: "Sync",
  ocr: "OCR",
  translate: "Translate",
  fps: "Frame rate",
  generate: "Generate",
  style: "Style",
  hdrSubs: "HDR subtitles",
  tonemap: "Tone-map video",
  convert: "Convert",
  extract: "Extract",
  mux: "Mux",
};

export function toolOf(task: SubTask): ToolId {
  return TOOLS.find((tool) => tool.tasks.includes(task))?.id ?? "sync";
}

export function inputShape(task: SubTask): InputShape {
  if (task === "generate" || task === "tonemap" || task === "extract") return "video";
  if (task === "mux") return "mux";
  return "subtitle";
}

/** Which kind of subtitle a task reads. */
export function subtitleKindFor(task: SubTask): "text" | "image" | "any" {
  if (task === "ocr") return "image";
  if (task === "hdrSubs") return "any";
  return "text";
}

/** Whether each item needs a video beside its subtitle. Sync needs one
 *  unless the timing comes from a reference given in its settings. */
export function needsVideo(task: SubTask, options: TaskOptions[SubTask]): boolean {
  if (inputShape(task) !== "subtitle") return inputShape(task) === "video";
  if (task === "sync") {
    const engine = (options as TaskOptions["sync"]).engine;
    return engine === "audio" || engine === "transcript";
  }
  return false;
}

/** Heavy tasks run one at a time; the rest two at a time. */
export function workersFor(task: SubTask): number {
  return task === "generate" || task === "tonemap" ? 1 : 2;
}

export const DEFAULT_OUTPUT: Record<SubTask, OutputOptions> = {
  sync: { dir: null, format: "same", suffix: ".synced", overwrite: false, mux: false },
  ocr: { dir: null, format: "srt", suffix: "", overwrite: false, mux: false },
  translate: { dir: null, format: "same", suffix: "", overwrite: false, mux: false },
  fps: { dir: null, format: "same", suffix: ".retimed", overwrite: false, mux: false },
  generate: { dir: null, format: "srt", suffix: "", overwrite: false, mux: false },
  style: { dir: null, format: "same", suffix: ".styled", overwrite: false, mux: false },
  hdrSubs: { dir: null, suffix: ".hdr", overwrite: false, mux: false },
  tonemap: { dir: null, suffix: ".sdr", overwrite: false },
  convert: { dir: null, format: "srt", suffix: "", overwrite: false },
  extract: { dir: null, suffix: "", overwrite: false },
  mux: { dir: null, suffix: ".muxed", overwrite: false },
};

/** The output fields each task offers. */
export interface OutputFields {
  formats: OutputFormat[] | null;
  mux: boolean;
}

const TEXT_FORMATS: OutputFormat[] = ["srt", "ass", "vtt", "ttml", "sub", "sbv"];

export function outputFields(task: SubTask): OutputFields {
  switch (task) {
    case "ocr":
    case "generate":
      return { formats: TEXT_FORMATS, mux: true };
    case "convert":
      return { formats: TEXT_FORMATS, mux: false };
    case "sync":
    case "translate":
    case "fps":
    case "style":
      return { formats: ["same", ...TEXT_FORMATS], mux: true };
    case "hdrSubs":
      // The format is a setting of its own (PGS can stay PGS).
      return { formats: null, mux: true };
    default:
      return { formats: null, mux: false };
  }
}

export const FORMAT_LABELS: Record<OutputFormat, string> = {
  same: "Same as the source",
  srt: "SubRip (.srt)",
  ass: "Advanced SubStation (.ass)",
  vtt: "WebVTT (.vtt)",
  ttml: "TTML (.ttml)",
  sub: "MicroDVD (.sub)",
  sbv: "YouTube (.sbv)",
  sup: "Blu-ray PGS (.sup)",
};

/** Tasks a result can be handed to with "Use as input for…". */
export function useAsTargets(output: { kind: string; format: string | null; path: string }): SubTask[] {
  if (output.kind === "video") return ["generate", "tonemap", "extract", "mux", "sync"];
  if (output.kind !== "subtitle") return [];
  const format = (output.format ?? output.path.replace(/^.*\./, "")).toLowerCase();
  if (["sup", "pgs", "vobsub", "idx"].includes(format)) return ["ocr", "hdrSubs", "mux"];
  return ["sync", "translate", "fps", "style", "hdrSubs", "convert", "mux"];
}
