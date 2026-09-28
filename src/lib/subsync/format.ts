/** Display helpers for the Subsync workspace: times, badges and the
 *  key/value reports tasks return. */

import type { CueData, ProbedFile, VideoInfo } from "./types";

/** 3725.5 -> "1:02:05.500"; under an hour "02:05.500". */
export function formatCueTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "--:--.---";
  const sign = seconds < 0 ? "-" : "";
  const ms = Math.round(Math.abs(seconds) * 1000);
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  const rest = ms % 1000;
  const mmss = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(rest).padStart(3, "0")}`;
  return h > 0 ? `${sign}${h}:${mmss}` : `${sign}${mmss}`;
}

/** Visible text of a cue: no ASS override blocks, no markup, one line. */
export function plainText(text: string): string {
  return text
    .replace(/\{[^}]*\}/g, "")
    .replace(/<[^>]+>/g, "")
    .replace(/\\[Nn]/g, "\n");
}

/** Characters per second, the reading-speed measure style guides use.
 *  Line breaks do not count. */
export function cueCps(cue: CueData): number | null {
  const duration = cue.end - cue.start;
  if (duration <= 0) return null;
  const chars = plainText(cue.text).replace(/\n/g, "").length;
  return chars / duration;
}

export const LOW_CONFIDENCE = 0.6;

export function isLowConfidence(cue: CueData): boolean {
  return typeof cue.confidence === "number" && cue.confidence < LOW_CONFIDENCE;
}

/** "DV P8.1", "HDR10", "HLG", "SDR"; null without video. */
export function hdrBadge(video: VideoInfo | null | undefined): string | null {
  if (!video) return null;
  switch (video.hdr) {
    case "dv": {
      if (video.dvProfile === null) return "Dolby Vision";
      const compat = video.dvCompatibility ? `.${video.dvCompatibility}` : "";
      return `DV P${video.dvProfile}${compat}`;
    }
    case "hdr10":
      return "HDR10";
    case "hdr10plus":
      return "HDR10+";
    case "hlg":
      return "HLG";
    default:
      return "SDR";
  }
}

export function formatFps(fps: number | null | undefined): string | null {
  if (!fps) return null;
  const rounded = Math.round(fps * 1000) / 1000;
  return `${rounded} fps`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

// ---------------------------------------------------------------- reports

const SIGNED = /offset|shift|delay|drift|change|delta/i;

/** "offsetMs" -> "Offset", "maxCll" -> "Max CLL", "linesFixed" -> "Lines fixed". */
export function reportLabel(key: string): string {
  const base = key.replace(/(Ms|S|Percent|Pct)$/, "");
  const words = base
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b(cll|fall|cps|cpl|fps|vad|asr|ocr|hdr|sdr|dv|id)\b/g, (word) => word.toUpperCase());
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function seconds(value: number, signed: boolean): string {
  const text = `${Math.abs(value).toFixed(3)} s`;
  if (!signed) return value < 0 ? `-${text}` : text;
  return `${value < 0 ? "-" : "+"}${text}`;
}

/** A report value made readable: offsetMs 3250 -> "+3.250 s". */
export function formatReportValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (/Ms$/.test(key)) return seconds(value / 1000, SIGNED.test(key));
    if (/[a-z]S$/.test(key)) return seconds(value, SIGNED.test(key));
    if (/(Percent|Pct)$/.test(key)) return `${Math.round(value * 10) / 10}%`;
    if (/ratio/i.test(key)) return `×${value.toFixed(6)}`;
    if (/fps/i.test(key)) return `${Math.round(value * 1000) / 1000}`;
    if (/nits|peak|cll|fall/i.test(key)) return `${Math.round(value)} nits`;
    if (Number.isInteger(value)) return value.toLocaleString();
    return String(Math.round(value * 1000) / 1000);
  }
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    if (value.length === 0) return "none";
    if (value.every((item) => typeof item !== "object") && value.length <= 6) return value.join(", ");
    return `${value.length} items`;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** "#2 · jpn · truehd · 8 ch · Commentary" */
export function audioTrackLabel(track: ProbedFile["audioTracks"][number]): string {
  const parts = [`#${track.index + 1}`, track.language ?? "und"];
  if (track.codec) parts.push(track.codec);
  if (track.channels) parts.push(`${track.channels} ch`);
  if (track.title) parts.push(track.title);
  return parts.join(" · ");
}
