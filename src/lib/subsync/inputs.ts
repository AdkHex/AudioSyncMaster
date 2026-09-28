/** Inputs of the Subsync tools: the files added to each tool, how they pair
 *  up into queue items, and the jobs those items become.
 *
 *  Everything here is pure: the workspace keeps an `InputPools` value in a
 *  reducer, derives the items from it on every render, and builds the jobs
 *  from the items when Run is pressed. */

import type { PickedFile } from "./api";
import { inputShape, needsVideo, subtitleKindFor } from "./tools";
import {
  LANGUAGES,
  type MediaRef,
  type MuxTrackMeta,
  type OutputOptions,
  type ProbedFile,
  type SubJob,
  type SubTask,
  type SubtitleTrackInfo,
  type TaskOptions,
} from "./types";

// --------------------------------------------------------------- the pool

export interface PoolFile extends ProbedFile {
  size: number | null;
  /** Added, not probed yet. */
  pending: boolean;
}

/** The files added to one task, and the choices made about them. */
export interface InputPool {
  files: PoolFile[];
  /** Subtitle path -> the video it goes with; null = none. Absent = automatic. */
  pairs: Record<string, string | null>;
  /** Video path -> embedded subtitle stream used as the source (0:s:N). */
  tracks: Record<string, number>;
  /** Video path -> audio stream to listen to (0:a:N). */
  audio: Record<string, number>;
  /** Mux: the video the subtitles go into; null = the first video. */
  muxVideo: string | null;
  /** Mux: per subtitle path, what its track is tagged with. */
  muxMeta: Record<string, MuxTrackMeta>;
}

export type InputPools = Record<SubTask, InputPool>;

export const EMPTY_POOL: InputPool = {
  files: [],
  pairs: {},
  tracks: {},
  audio: {},
  muxVideo: null,
  muxMeta: {},
};

const ALL_TASKS: SubTask[] = [
  "sync", "ocr", "translate", "fps", "generate", "style", "hdrSubs", "tonemap", "convert", "extract", "mux",
];

export function emptyPools(): InputPools {
  return Object.fromEntries(ALL_TASKS.map((task) => [task, EMPTY_POOL])) as InputPools;
}

export type PoolAction =
  | { type: "add"; task: SubTask; files: PickedFile[] }
  | { type: "probed"; files: ProbedFile[] }
  | { type: "remove"; task: SubTask; path: string }
  | { type: "clear"; task: SubTask }
  | { type: "setPair"; task: SubTask; subtitle: string; video: string | null }
  | { type: "setTrack"; task: SubTask; video: string; track: number }
  | { type: "setAudio"; task: SubTask; video: string; track: number }
  | { type: "setMuxVideo"; task: SubTask; video: string }
  | { type: "setMuxMeta"; task: SubTask; path: string; patch: MuxTrackMeta };

export function pendingFile(file: PickedFile): PoolFile {
  return {
    path: file.path,
    name: file.name,
    kind: file.type,
    duration: null,
    video: null,
    audioTracks: [],
    subtitleTracks: [],
    subtitle: null,
    size: file.size ?? null,
    pending: true,
  };
}

function withoutKey<T>(record: Record<string, T>, key: string): Record<string, T> {
  if (!(key in record)) return record;
  const next = { ...record };
  delete next[key];
  return next;
}

function updatePool(pools: InputPools, task: SubTask, fn: (pool: InputPool) => InputPool): InputPools {
  return { ...pools, [task]: fn(pools[task]) };
}

export function poolsReducer(pools: InputPools, action: PoolAction): InputPools {
  switch (action.type) {
    case "add":
      return updatePool(pools, action.task, (pool) => {
        const known = new Set(pool.files.map((file) => file.path));
        const added = action.files.filter((file) => !known.has(file.path)).map(pendingFile);
        if (added.length === 0) return pool;
        return { ...pool, files: [...pool.files, ...added] };
      });
    case "probed": {
      const byPath = new Map(action.files.map((file) => [file.path, file]));
      const next = { ...pools };
      for (const task of ALL_TASKS) {
        const pool = pools[task];
        if (!pool.files.some((file) => byPath.has(file.path))) continue;
        next[task] = {
          ...pool,
          files: pool.files.map((file) => {
            const probed = byPath.get(file.path);
            return probed ? { ...probed, size: file.size, pending: false } : file;
          }),
        };
      }
      return next;
    }
    case "remove":
      return updatePool(pools, action.task, (pool) => ({
        files: pool.files.filter((file) => file.path !== action.path),
        pairs: Object.fromEntries(
          Object.entries(withoutKey(pool.pairs, action.path)).filter(([, video]) => video !== action.path),
        ),
        tracks: withoutKey(pool.tracks, action.path),
        audio: withoutKey(pool.audio, action.path),
        muxVideo: pool.muxVideo === action.path ? null : pool.muxVideo,
        muxMeta: withoutKey(pool.muxMeta, action.path),
      }));
    case "clear":
      return updatePool(pools, action.task, () => EMPTY_POOL);
    case "setPair":
      return updatePool(pools, action.task, (pool) => ({
        ...pool,
        pairs: { ...pool.pairs, [action.subtitle]: action.video },
      }));
    case "setTrack":
      return updatePool(pools, action.task, (pool) => ({
        ...pool,
        tracks: { ...pool.tracks, [action.video]: action.track },
      }));
    case "setAudio":
      return updatePool(pools, action.task, (pool) => ({
        ...pool,
        audio: { ...pool.audio, [action.video]: action.track },
      }));
    case "setMuxVideo":
      return updatePool(pools, action.task, (pool) => ({ ...pool, muxVideo: action.video }));
    case "setMuxMeta":
      return updatePool(pools, action.task, (pool) => ({
        ...pool,
        muxMeta: { ...pool.muxMeta, [action.path]: { ...pool.muxMeta[action.path], ...action.patch } },
      }));
  }
}

// ------------------------------------------------------------ file kinds

const IMAGE_SUB_EXT = /\.(sup|idx)$/i;

export function isSubtitleFile(file: ProbedFile): boolean {
  return file.kind === "subtitle";
}

export function isVideoFile(file: ProbedFile): boolean {
  return file.kind === "video";
}

export function subtitleFileKind(file: ProbedFile): "text" | "image" {
  if (file.subtitle?.kind) return file.subtitle.kind;
  return IMAGE_SUB_EXT.test(file.path) ? "image" : "text";
}

function kindAccepted(kind: "text" | "image", wanted: "text" | "image" | "any"): boolean {
  return wanted === "any" || wanted === kind;
}

export function usableTracks(file: ProbedFile, task: SubTask): SubtitleTrackInfo[] {
  const wanted = subtitleKindFor(task);
  return file.subtitleTracks.filter((track) => kindAccepted(track.kind, wanted));
}

export function trackLabel(track: SubtitleTrackInfo): string {
  const parts = [`#${track.index + 1}`, track.language ?? "und"];
  if (track.title) parts.push(track.title);
  parts.push(track.codec);
  if (track.forced) parts.push("forced");
  if (track.hearingImpaired) parts.push("SDH");
  if (track.events !== null) parts.push(`${track.events} events`);
  return parts.join(" · ");
}

// ---------------------------------------------------------------- pairing

const TAG_TOKENS = new Set(["forced", "sdh", "cc", "hi", "default", "full", "signs", "songs", "dialogue"]);

/** ISO 639-2 codes common in release names; a bare three-letter word is
 *  not taken as a language unless it is one of these. */
const ISO_639_2 = new Set(
  ("eng jpn kor zho chi fre fra spa ita ger deu por hin rus ara und nld dut swe nor dan fin pol tur heb " +
    "tha vie ind msa may ces cze hun ron rum ell gre ukr bul hrv srp slk slv est lav lit tam tel ben mal " +
    "kan mar urd per fas cat glg eus isl fil tgl").split(" "),
);

function stripExtension(name: string): string {
  return name.replace(/\.[^.\\/]+$/, "");
}

/** The names a subtitle file might share with its video, longest first:
 *  "Movie.en.forced.srt" -> ["Movie.en.forced", "Movie.en", "Movie"].
 *  Only trailing language codes and tags are dropped, so a title whose last
 *  word merely looks like one still matches on its full name first. */
export function stemCandidates(name: string): string[] {
  const parts = stripExtension(name).split(".");
  const out = [parts.join(".")];
  while (parts.length > 1) {
    const last = parts[parts.length - 1].toLowerCase();
    if (!TAG_TOKENS.has(last) && !/^[a-z]{2,3}(-[a-z0-9]{2,4})?$/.test(last)) break;
    parts.pop();
    out.push(parts.join("."));
  }
  return out;
}

/** What a subtitle file's name says about its track: "Movie.pt-br.forced.srt"
 *  -> language "pt", forced. */
export function metaFromName(name: string): MuxTrackMeta {
  const tokens = stripExtension(name).split(".").slice(1).map((token) => token.toLowerCase());
  const meta: MuxTrackMeta = { default: false, forced: false, hearingImpaired: false };
  for (const token of tokens.reverse()) {
    if (TAG_TOKENS.has(token)) {
      if (token === "forced") meta.forced = true;
      if (token === "sdh" || token === "cc" || token === "hi") meta.hearingImpaired = true;
      continue;
    }
    const code = token.split("-")[0];
    if (!meta.language && (code in LANGUAGES || ISO_639_2.has(code))) {
      meta.language = code;
      continue;
    }
    break;
  }
  return meta;
}

/** Pair each subtitle with a video by filename stem (Movie.en.srt ->
 *  Movie.mkv). With a single video, every subtitle goes with it. */
export function autoPair(subtitles: ProbedFile[], videos: ProbedFile[]): Record<string, string | null> {
  const byStem = new Map<string, string>();
  for (const video of videos) byStem.set(stripExtension(video.name).toLowerCase(), video.path);
  const pairs: Record<string, string | null> = {};
  for (const subtitle of subtitles) {
    const match = stemCandidates(subtitle.name)
      .map((stem) => byStem.get(stem.toLowerCase()))
      .find((path) => path !== undefined);
    pairs[subtitle.path] = match ?? (videos.length === 1 ? videos[0].path : null);
  }
  return pairs;
}

// ------------------------------------------------------------------ items

/** One row of a tool's queue: becomes one job. */
export interface InputItem {
  key: string;
  subtitle: MediaRef | null;
  subtitleName: string | null;
  /** The subtitle is a stream inside the video rather than a file. */
  embedded: boolean;
  video: MediaRef | null;
  videoName: string | null;
  /** Mux only. */
  subtitles?: (MediaRef & MuxTrackMeta & { name: string })[];
  /** Why this row cannot run as it stands; null when it can. */
  problem: string | null;
}

function nameOf(files: ProbedFile[], path: string | null | undefined): string | null {
  if (!path) return null;
  return files.find((file) => file.path === path)?.name ?? path.replace(/^.*[\\/]/, "");
}

function subtitleItems(task: SubTask, pool: InputPool, options: TaskOptions[SubTask]): InputItem[] {
  const subtitles = pool.files.filter(isSubtitleFile);
  const videos = pool.files.filter(isVideoFile);
  const auto = autoPair(subtitles, videos);
  const wanted = subtitleKindFor(task);
  const videoRequired = needsVideo(task, options);
  const items: InputItem[] = [];
  const pairedVideos = new Set<string>();

  for (const file of subtitles) {
    const videoPath = file.path in pool.pairs ? pool.pairs[file.path] : auto[file.path];
    if (videoPath) pairedVideos.add(videoPath);
    const kind = subtitleFileKind(file);
    let problem: string | null = null;
    if (file.error) problem = file.error;
    else if (!kindAccepted(kind, wanted))
      problem = kind === "image" ? "An image subtitle: run OCR on it first." : "Needs an image subtitle (PGS or VobSub).";
    else if (videoRequired && !videoPath) problem = "Choose the video to sync it to.";
    items.push({
      key: `sub:${file.path}`,
      subtitle: { path: file.path },
      subtitleName: file.name,
      embedded: false,
      video: videoPath ? { path: videoPath, audioTrack: pool.audio[videoPath] ?? 0 } : null,
      videoName: nameOf(pool.files, videoPath),
      problem,
    });
  }

  // A video no subtitle file went with is its own source: one of its tracks.
  for (const video of videos) {
    if (pairedVideos.has(video.path)) continue;
    const tracks = usableTracks(video, task);
    const chosen = tracks.find((track) => track.index === pool.tracks[video.path]) ?? tracks[0];
    items.push({
      key: `emb:${video.path}`,
      subtitle: chosen ? { path: video.path, track: chosen.index } : null,
      subtitleName: chosen ? trackLabel(chosen) : null,
      embedded: true,
      video: { path: video.path, audioTrack: pool.audio[video.path] ?? 0 },
      videoName: video.name,
      problem: video.error
        ? video.error
        : video.pending
          ? null
          : chosen
            ? null
            : wanted === "image"
              ? "No image subtitle tracks in this video."
              : "No text subtitle tracks, and no subtitle file matched it.",
    });
  }
  return items;
}

function videoItems(task: SubTask, pool: InputPool): InputItem[] {
  return pool.files.map((file) => ({
    key: `vid:${file.path}`,
    subtitle: null,
    subtitleName: null,
    embedded: false,
    video: isVideoFile(file) ? { path: file.path, audioTrack: pool.audio[file.path] ?? 0 } : null,
    videoName: file.name,
    problem: file.error
      ? file.error
      : !isVideoFile(file)
        ? "Not a video."
        : task === "extract" && !file.pending && file.subtitleTracks.length === 0
          ? "No subtitle tracks in this video."
          : null,
  }));
}

function muxItem(pool: InputPool): InputItem[] {
  const videos = pool.files.filter(isVideoFile);
  const subtitles = pool.files.filter(isSubtitleFile);
  if (videos.length === 0 && subtitles.length === 0) return [];
  const video = videos.find((file) => file.path === pool.muxVideo) ?? videos[0] ?? null;
  return [
    {
      key: "mux",
      subtitle: null,
      subtitleName: null,
      embedded: false,
      video: video ? { path: video.path } : null,
      videoName: video?.name ?? null,
      subtitles: subtitles.map((file) => ({
        path: file.path,
        name: file.name,
        ...metaFromName(file.name),
        ...pool.muxMeta[file.path],
      })),
      problem: !video
        ? "Add the video the subtitles go into."
        : subtitles.length === 0
          ? "Add the subtitle files to put into it."
          : null,
    },
  ];
}

export function buildItems(task: SubTask, pool: InputPool, options: TaskOptions[SubTask]): InputItem[] {
  switch (inputShape(task)) {
    case "subtitle":
      return subtitleItems(task, pool, options);
    case "video":
      return videoItems(task, pool);
    case "mux":
      return muxItem(pool);
  }
}

// ------------------------------------------------------------------- jobs

/** A job's display name: what the row shows while it runs. */
export function itemLabel(item: InputItem): string {
  if (item.embedded) return `${item.videoName ?? "video"} (${item.subtitleName ?? "track"})`;
  return item.subtitleName ?? item.videoName ?? "input";
}

/** The trimmed output options: a blank folder means beside the source. */
export function cleanOutput(output: OutputOptions): OutputOptions {
  const dir = output.dir?.trim() ? output.dir.trim() : null;
  return { ...output, dir, suffix: output.suffix ?? "" };
}

/** One job per runnable item, in queue order. */
export function buildJobs(
  task: SubTask,
  items: InputItem[],
  options: TaskOptions[SubTask],
  output: OutputOptions,
  newId: (index: number) => string,
): { jobs: SubJob[]; labels: string[] } {
  const jobs: SubJob[] = [];
  const labels: string[] = [];
  const out = cleanOutput(output);
  for (const item of items) {
    if (item.problem) continue;
    const input: SubJob["input"] = {};
    const shape = inputShape(task);
    if (shape === "mux") {
      if (!item.video || !item.subtitles?.length) continue;
      input.video = { path: item.video.path };
      input.subtitles = item.subtitles.map((sub) => {
        const { name: _name, ...rest } = sub;
        return rest;
      });
    } else if (shape === "video") {
      if (!item.video) continue;
      input.video =
        task === "generate" ? { path: item.video.path, audioTrack: item.video.audioTrack ?? 0 } : { path: item.video.path };
    } else {
      if (!item.subtitle) continue;
      input.subtitle = { ...item.subtitle };
      if (item.video) input.video = { ...item.video };
    }
    jobs.push({ id: newId(jobs.length), task, input, options, output: { ...out } });
    labels.push(itemLabel(item));
  }
  return { jobs, labels };
}

/** The frame rate of the first video among the items, for fps defaults. */
export function firstVideoFps(pool: InputPool): number | null {
  for (const file of pool.files) {
    if (file.video?.fps) return file.video.fps;
  }
  return null;
}
