/** Shared types. Field names match the Rust structs and the Python engine
 *  exactly -- all three layers speak camelCase across the wire. */

export type SyncMode = "movie" | "series" | "compare" | "dubsync";

/** What the Dub sync tab is pairing: a batch of movies by filename, or a
 *  season of episodes by season/episode number. */
export type DubScope = "movies" | "series";

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
  /** Null when a cut left too few windows on one side to fit a line through. */
  driftMsPerS: number | null;
  speedRatio: number;
  sourceFps: number | null;
  targetFps: number | null;
  isRateMismatch: boolean;
  isLikelyCut: boolean;
  /** Where the offset jumps and by how much, once a splice has been located. */
  cutPositionS: number | null;
  cutMagnitudeMs: number | null;
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
  /** Where the cut is in the video's timeline, how tightly that was pinned
   *  down, and how far the offset jumps there. Present only when isLikelyCut. */
  cutPositionS?: number | null;
  cutUncertaintyS?: number | null;
  cutMagnitudeMs?: number | null;
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
  /** Which pairing the dub tab wants: movies by filename, series by episode. */
  dubKind?: DubScope;
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

/** Output codecs the dub sync renderer can write. "same" follows the dub's
 *  own codec where an encoder exists for it, and falls back to FLAC. */
export type DubCodec = "same" | "flac" | "eac3" | "ac3" | "aac" | "opus" | "wav";

export const DUB_CODECS: { id: DubCodec; label: string }[] = [
  { id: "same", label: "Same as the dub" },
  { id: "flac", label: "FLAC (lossless)" },
  { id: "eac3", label: "E-AC3" },
  { id: "ac3", label: "AC3" },
  { id: "aac", label: "AAC" },
  { id: "opus", label: "Opus" },
  { id: "wav", label: "WAV (24-bit)" },
];

export interface AppSettings {
  windowSeconds: number;
  windowCount: number;
  maxOffsetMs: number;
  maxWorkers: number;
  matchPattern: string;
  outputSuffix: string;
  theme: "light" | "dark" | "system";
  /** Dub sync output. */
  dubCodec: DubCodec;
  /** Also write a copy of the video with the synced track added. */
  dubMux: boolean;
  /** ISO 639-2 tag for the muxed track, e.g. "hin". Empty leaves it untagged. */
  dubLanguage: string;
  /** Replace stretches the dub did not correlate on even when its offset is
   *  unchanged either side. Off by default: a weakly correlating scene is
   *  still the dub, and swapping it for the original puts the wrong language
   *  over a scene that had the right one. */
  dubFillUnmatched: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  windowSeconds: 45,
  windowCount: 6,
  maxOffsetMs: 60000,
  maxWorkers: 3,
  matchPattern: "",
  outputSuffix: ".synced",
  theme: "dark",
  dubCodec: "same",
  dubMux: false,
  dubLanguage: "",
  dubFillUnmatched: false,
};

// ------------------------------------------------------------------ dub sync

/** One piece of the synced track, on the video's timeline. */
export interface DubSegment {
  kind: "dub" | "fill";
  startS: number;
  endS: number;
  /** Where it is read from: the dub's timeline for "dub", the video's own
   *  audio for "fill". */
  sourceStartS: number;
  /** dub time minus video time, for "dub" pieces. */
  offsetS: number | null;
  /** Envelope correlation across the stretch, 0-1, for "dub" pieces. */
  match: number | null;
  note: string;
  /** How far either edge might really sit from where it was placed. */
  uncertaintyS: number;
}

export interface DubSyncPlan {
  videoPath: string;
  dubPath: string;
  videoTrack: number;
  dubTrack: number;
  /** Playback-speed factor the dub was decoded at; 1 when it matched as is. */
  speed: number;
  /** Gain applied to the original where it fills a gap. */
  fillGainDb: number;
  videoDurationS: number;
  dubDurationS: number;
  segments: DubSegment[];
  /** Things to check: a fill, a drift, a stretch replaced. */
  warnings: string[];
  /** Things worth knowing that need no checking: the dub kept across a
   *  span it did not correlate in. Absent from plans made before it was
   *  added. */
  notes?: string[];
  error: string | null;
  /** Seconds of the output taken from the original. */
  filledS: number;
}

export interface DubSpotCheck {
  positionS: number;
  /** Null where the spot could not be measured (silence, or a fill). */
  residualMs: number | null;
  match: number;
  note: string;
}

export interface DubStretch {
  startS: number;
  endS: number;
  residualMs: number;
  windows: number;
}

/** How the finished track sits against the video, measured after writing. */
export interface DubVerification {
  spots: DubSpotCheck[];
  typicalMs: number | null;
  worstMs: number | null;
  sweepWindows: number;
  sweepMeasured: number;
  sweepWithinAudible: number;
  sweepTypicalMs: number | null;
  sweepWorstMs: number | null;
  stretches: DubStretch[];
}

export interface DubOutput {
  outputPath: string;
  sampleRate: number;
  channels: number;
  seconds: number;
  clippedSamples: number;
  warnings: string[];
}

export interface DubSyncRequest {
  videoPath: string;
  dubPath: string;
  videoTrack: number;
  dubTrack: number;
  codec: DubCodec;
  mux: boolean;
  language: string | null;
  fillUnmatched: boolean;
  overwrite: boolean;
}

/** One pair of the dub sync queue: an episode or a movie and its own dub. */
export interface DubSyncJob {
  videoPath: string;
  dubPath: string;
  videoTrack: number;
  dubTrack: number;
}

/** A queue of pairs, synced in parallel by the engine. Output options are
 *  shared: a season of episodes is written the same way throughout. */
export interface DubSyncBatchRequest {
  jobs: DubSyncJob[];
  codec: DubCodec;
  mux: boolean;
  language: string | null;
  fillUnmatched: boolean;
  overwrite: boolean;
  maxWorkers: number;
}

/** The engine's final word on one job of a batch. */
export interface DubJobOutcome {
  job: number;
  plan: DubSyncPlan | null;
  output: DubOutput | null;
  verification: DubVerification | null;
  verificationText?: string | null;
  muxedPath: string | null;
  cancelled?: boolean;
  error?: string | null;
}

export interface DubSyncBatchResult {
  outcomes: DubJobOutcome[];
  cancelled: boolean;
}

/** The engine's final word on a dub sync. */
export interface DubSyncOutcome {
  plan: DubSyncPlan | null;
  output: DubOutput | null;
  verification: DubVerification | null;
  verificationText: string | null;
  muxedPath: string | null;
  cancelled?: boolean;
  error?: string | null;
}

/** h:mm:ss.mmm, as the engine prints positions. Milliseconds matter here:
 *  a cut placed at 1:23:45.317 is a different claim from one at 1:23:45. */
export function formatClock(seconds: number): string {
  const ms = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(ms / 3_600_000);
  const minutes = Math.floor((ms % 3_600_000) / 60_000);
  const rest = (ms % 60_000) / 1000;
  return `${hours}:${String(minutes).padStart(2, "0")}:${rest.toFixed(3).padStart(6, "0")}`;
}

/** A span's length as people say it: "2m 30.0s", "0.7s". */
export function formatSpan(seconds: number): string {
  if (seconds >= 60) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${(seconds - minutes * 60).toFixed(1).padStart(4, "0")}s`;
  }
  return `${seconds.toFixed(1)}s`;
}

/** Lip-sync error becomes visible around here. */
export const AUDIBLE_MS = 100;

/** The note the engine puts on a fill that replaced dub which was audible but
 *  could not be matched -- the "Replace stretches that did not correlate"
 *  setting at work. A different claim from "dub is cut here": nothing showed
 *  the dub lacks the scene. One string, shared with the engine. */
export const UNMATCHED_FILL_NOTE = "dub audible but did not correlate; replaced";

export function isUnmatchedFill(segment: DubSegment): boolean {
  return segment.kind === "fill" && segment.note === UNMATCHED_FILL_NOTE;
}

/** A frame rate as people write it, from the exact rational the engine reports.
 *
 *  The engine holds 23.976 as 24000/1001, because a correction ratio built from
 *  the decimal is wrong by 1e-6. Printed raw that arrives as
 *  "23.976023976023978 fps", so the shorthand is restored at the last moment --
 *  for display only, never for arithmetic.
 */
export function formatFps(fps: number): string {
  return fps.toFixed(3).replace(/\.?0+$/, "");
}

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
