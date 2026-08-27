/** Shared types. Field names match the Rust structs and the Python engine
 *  exactly -- all three layers speak camelCase across the wire. */

export type SyncMode = "movie" | "series" | "compare";

/** Upper bound per side in compare mode. The work is the product of both
 *  sides, so five against five is already 25 analyses. */
export const MAX_COMPARE_INPUTS = 5;

export type ProcessingStatus = "idle" | "processing" | "complete" | "cancelled";

export type ConfidenceLevel = "high" | "medium" | "low";

export interface FileItem {
  id: string;
  name: string;
  path: string;
  type: "video" | "audio";
  size?: number | null;
}

export interface PickResponse {
  folder: string | null;
  files: Omit<FileItem, "id">[];
}

export interface AudioTrackInfo {
  index: number;
  codec: string | null;
  language: string | null;
  title: string | null;
  channels: number | null;
  sampleRate: number | null;
  bitRate: number | null;
  isDefault: boolean;
  label: string;
}

/** Why a file drifts: a frame-rate conversion, or a different cut. */
export interface RateDiagnosis {
  driftMsPerS: number;
  speedRatio: number;
  sourceFps: number | null;
  targetFps: number | null;
  isRateMismatch: boolean;
  isLikelyCut: boolean;
  explanation: string;
  correctionRatio: number | null;
}

export interface TrackListing {
  path: string;
  name: string;
  tracks: AudioTrackInfo[];
  fps: number | null;
  duration: number | null;
  error?: string | null;
}

export interface MediaProbe {
  hasAudio: boolean;
  hasVideo: boolean;
  duration: number | null;
  audioCodec: string | null;
  fps?: number | null;
  audioTracks?: AudioTrackInfo[];
  error?: string | null;
}

/** One measured pair, as returned by the engine. */
export interface SyncResult {
  videoFile: string;
  audioFile: string;
  primaryPath?: string | null;
  secondaryPath?: string | null;
  delayMs: number | null;
  /** Offset at t=0. With drift, delayMs is the midpoint value; a correction is
   *  applied from the start of the file and must use this instead. */
  delayAtStartMs: number | null;
  confidence: number | null;
  driftMsPerS: number | null;
  totalDriftMs: number | null;
  hasSignificantDrift: boolean | null;
  startDelayMs: number | null;
  endDelayMs: number | null;
  windowsUsed: number | null;
  windowsTotal: number | null;
  error: string | null;
  elapsedMs: number | null;
  primaryDurationS?: number | null;
  secondaryDurationS?: number | null;
  primaryTrack?: number | null;
  secondaryTrack?: number | null;
  primaryFps?: number | null;
  secondaryFps?: number | null;
  isLikelyCut?: boolean | null;
  isRateMismatch?: boolean | null;
  /** Codec delay already removed from delayMs, so the figure is not silent. */
  codecDelayMs?: number | null;
  primaryCodec?: string | null;
  secondaryCodec?: string | null;
  rateDiagnosis?: RateDiagnosis | null;
}

export interface RunSummary {
  total: number;
  matched: number;
  failed: number;
  drifting: number;
  cuts: number;
  rateMismatches: number;
  high: number;
  medium: number;
  low: number;
}

export interface SyncRun {
  results: SyncResult[];
  summary: RunSummary | null;
  cancelled: boolean;
}

export interface MatchPair {
  primaryPath: string;
  secondaryPath: string;
  primaryName: string;
  secondaryName: string;
  key: string;
  method: string;
  score: number;
  primaryTrack?: number;
  secondaryTrack?: number;
}

export interface PairingReport {
  pairs: MatchPair[];
  unmatchedPrimary: string[];
  unmatchedSecondary: string[];
  method: string;
  patternUsed: string | null;
  warning: string | null;
}

export interface AnalyzeRequest {
  mode: SyncMode;
  videoFolder: string | null;
  audioFolder: string | null;
  audioFile: string | null;
  videoFiles: string[] | null;
  audioFiles: string[] | null;
  matchPattern: string | null;
  videoTrack: number;
  audioTrack: number;
  /** Explicit pairs, sent when the user has corrected the matching by hand.
   *  The engine uses these verbatim instead of re-matching, which would
   *  silently undo the edit. */
  pairs?: MatchPair[] | null;
  windowSeconds: number;
  windowCount: number;
  maxOffsetMs: number;
  maxWorkers: number;
}

export interface CorrectionItem {
  videoPath: string;
  audioPath: string;
  delayMs: number;
  delayAtStartMs?: number | null;
  driftMsPerS?: number | null;
}

export interface ApplyResult {
  written: string[];
  failed: { video: string; error: string }[];
  cancelled?: boolean;
}

export interface HistoryEntry {
  id: string;
  /** ISO 8601. Stored as a string because JSON has no Date type -- the original
   *  typed this as Date and silently got a string back after a reload. */
  date: string;
  mode: SyncMode;
  results: SyncResult[];
  summary: RunSummary | null;
  fileCount: number;
}

export interface AppSettings {
  windowSeconds: number;
  windowCount: number;
  maxOffsetMs: number;
  maxWorkers: number;
  matchPattern: string;
  outputSuffix: string;
  theme: "light" | "dark" | "system";
}

export const DEFAULT_SETTINGS: AppSettings = {
  windowSeconds: 45,
  windowCount: 6,
  maxOffsetMs: 60000,
  maxWorkers: 3,
  matchPattern: "",
  outputSuffix: ".synced",
  theme: "dark",
};

export type ResultStatus = "ok" | "drift" | "rate-mismatch" | "cut" | "failed";

/** Classify a result for display.
 *
 *  A cut is reported separately from drift because it is the one outcome no
 *  correction can fix: the two files contain different material, so there is no
 *  single delay or speed ratio that aligns them.
 */
export function resultStatus(result: SyncResult): ResultStatus {
  if (result.error || result.delayMs === null) return "failed";
  if (result.isLikelyCut) return "cut";
  if (result.isRateMismatch) return "rate-mismatch";
  if (result.hasSignificantDrift) return "drift";
  return "ok";
}

export const STATUS_LABELS: Record<ResultStatus, string> = {
  ok: "OK",
  drift: "Drift",
  "rate-mismatch": "Frame rate",
  cut: "Different cut",
  failed: "Failed",
};

/** Map a 0-1 engine confidence onto the three bands the UI displays. */
export function confidenceLevel(result: SyncResult): ConfidenceLevel {
  if (result.error || result.delayMs === null || result.confidence === null) {
    return "low";
  }
  if (result.confidence >= 0.75) return "high";
  if (result.confidence >= 0.5) return "medium";
  return "low";
}

export function formatDelay(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "--";
  const sign = ms > 0 ? "+" : "";
  return `${sign}${ms.toFixed(1)} ms`;
}

/** A measured offset, expressed the way every player and muxer expects it.
 *
 *  The engine measures where the audio sits: negative means the dub starts
 *  before the picture. MKVToolNix, VLC and an Audacity-style manual workflow
 *  all ask the opposite question -- how much delay do I *add* to fix this --
 *  so the same situation carries the opposite sign there.
 *
 *  Showing the measurement under the label "Delay" invited exactly that
 *  confusion, so the displayed number is the one you can type straight into
 *  those tools. The engine's own value is untouched: it drives the correction,
 *  and flipping it there would break every write path. */
export function playerDelayMs(ms: number | null): number | null {
  if (ms === null || !Number.isFinite(ms)) return null;
  // -0 formats as "-0.0 ms", which reads as a real negative offset.
  return ms === 0 ? 0 : -ms;
}

/** Format an offset in the player convention. */
export function formatPlayerDelay(ms: number | null): string {
  return formatDelay(playerDelayMs(ms));
}

export function formatDrift(msPerS: number | null): string {
  if (msPerS === null || !Number.isFinite(msPerS)) return "--";
  const sign = msPerS > 0 ? "+" : "";
  return `${sign}${msPerS.toFixed(3)} ms/s`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || !Number.isFinite(seconds)) return "--";
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

export function formatSize(bytes?: number | null): string {
  if (!bytes || bytes <= 0) return "--";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

const CHANNEL_NAMES: Record<number, string> = {
  1: "mono",
  2: "stereo",
  6: "5.1",
  8: "7.1",
};

/** One line describing what a file actually contains.
 *
 *  All of this was probed already and then thrown away, leaving rows that
 *  showed only a duration and a size. Seeing the codec and channel layout is
 *  how you notice you have loaded a commentary track or a stereo downmix
 *  rather than the feature audio.
 *
 *  Frame rate appears only for video: an audio file has no frames, so there is
 *  no fps to report. For video it is worth the space, because a 25 fps PAL
 *  master against a 23.976 fps source is the usual cause of steady drift. */
export function streamSummary(
  probe: MediaProbe | undefined,
  listing: TrackListing | undefined,
  kind: "video" | "audio",
): string | null {
  const tracks = listing?.tracks ?? probe?.audioTracks ?? [];
  const first = tracks[0];
  const parts: string[] = [];

  if (kind === "video") {
    const fps = listing?.fps ?? probe?.fps;
    if (fps) parts.push(`${Number(fps.toFixed(3))} fps`);
  }

  const codec = first?.codec ?? probe?.audioCodec;
  if (codec) parts.push(codec.toUpperCase());

  if (first?.channels) {
    parts.push(CHANNEL_NAMES[first.channels] ?? `${first.channels}ch`);
  }

  // Sample rate is the least useful of these -- it is 48 kHz on essentially
  // every film release -- so it is dropped when it would push the line past
  // the sidebar's width and truncate something that does vary.
  if (first?.sampleRate && first.sampleRate !== 48000) {
    const khz = first.sampleRate / 1000;
    parts.push(`${Number(khz.toFixed(1))} kHz`);
  }

  if (first?.bitRate) {
    parts.push(`${Math.round(first.bitRate / 1000)}k`);
  }

  // Only worth saying when there is a choice to make.
  if (tracks.length > 1) {
    parts.push(`${tracks.length} tracks`);
  }

  return parts.length > 0 ? parts.join(" · ") : null;
}

export function formatElapsed(ms: number | null): string {
  if (!ms || ms <= 0) return "--";
  if (ms < 1000) return `${ms} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.round(seconds % 60)).padStart(2, "0")}`;
}

/** A delay expressed in video frames, which is how editors think about sync.
 *
 *  "+317.5 ms" says nothing about whether that is a lot. "8 frames" is
 *  immediately meaningful to anyone who works with video, and the frame rate is
 *  already known.
 */
export function frameOffset(
  delayMs: number | null,
  fps: number | null | undefined,
): number | null {
  if (delayMs === null || !fps || !Number.isFinite(delayMs) || !Number.isFinite(fps)) {
    return null;
  }
  const frames = Math.round((delayMs / 1000) * fps);
  // Below half a frame there is nothing useful to say.
  return frames === 0 ? null : frames;
}

/** The ffmpeg command a user would run by hand for this result. */
export function ffmpegCommandFor(result: SyncResult): string | null {
  if (result.delayMs === null || !result.primaryPath || !result.secondaryPath) {
    return null;
  }
  const quote = (value: string) => `"${value.replace(/"/g, '\\"')}"`;
  const video = quote(result.primaryPath);
  const audio = quote(result.secondaryPath);
  const output = quote(result.primaryPath.replace(/(\.[^.]+)$/, ".synced$1"));

  const parts = ["ffmpeg", "-i", video];
  if (result.delayMs > 0) {
    parts.push("-ss", (result.delayMs / 1000).toFixed(6));
  }
  parts.push("-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy");
  if (result.delayMs < 0) {
    parts.push("-filter:a", `adelay=${(-result.delayMs).toFixed(3)}:all=1`, "-c:a", "aac");
  } else {
    parts.push("-c:a", "copy");
  }
  parts.push(output);
  return parts.join(" ");
}

/** Stable identity for a measured pair, used for selection and de-duplication. */
export function resultKey(result: SyncResult): string {
  return `${result.primaryPath ?? result.videoFile}::${result.secondaryPath ?? result.audioFile}`;
}
