/** Files, pairings and analysis results for the demo scenarios. Dev only.
 *
 *  The numbers are the design mockup's (design/mockup/data.ts). The mockup
 *  shows delays in the player convention, which is the NEGATION of the
 *  engine's `delayMs`, so a mockup "−317.0 ms" is stored here as +317. */

import type {
  AudioTrackInfo,
  MatchPair,
  MediaProbe,
  PairingReport,
  RateDiagnosis,
  RunSummary,
  SyncResult,
  TrackListing,
} from "@/lib/types";

// ------------------------------------------------------------------- files

export type Kind = "video" | "audio";

export interface FileFx {
  name: string;
  dir: string;
  path: string;
  size: number;
  durationS: number;
  fps: number | null;
  kind: Kind;
  tracks: AudioTrackInfo[];
}

export const GB = 1024 ** 3;
export const MB = 1024 ** 2;
const NTSC_FILM = 24000 / 1001;

const CHANNEL_NAMES: Record<number, string> = { 1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1" };

/** Same wording as `AudioTrack.label` in audiosync/media.py. */
function track(
  index: number,
  codec: string,
  channels: number,
  language: string | null,
  title: string | null,
  bitRate: number | null,
): AudioTrackInfo {
  const parts = [`Track ${index + 1}`];
  if (language) parts.push(language.toUpperCase());
  if (title) parts.push(title);
  parts.push(`(${codec.toUpperCase()}, ${CHANNEL_NAMES[channels] ?? `${channels}ch`})`);
  return {
    index,
    codec,
    language,
    title,
    channels,
    sampleRate: 48000,
    bitRate,
    isDefault: index === 0,
    label: parts.join(" · "),
  };
}

export const joinPath = (dir: string, name: string) => `${dir}\\${name}`;
export const baseName = (path: string) => path.replace(/^.*[\\/]/, "");
export const dirName = (path: string) => path.replace(/[\\/][^\\/]*$/, "");
export const stemOf = (name: string) => name.replace(/\.[^.]+$/, "");
export const extOf = (name: string) => (/\.([^.]+)$/.exec(name)?.[1] ?? "").toLowerCase();

const registry = new Map<string, FileFx>();

function make(
  kind: Kind,
  dir: string,
  name: string,
  size: number,
  durationS: number,
  fps: number | null,
  tracks: AudioTrackInfo[],
): FileFx {
  const file: FileFx = { name, dir, path: joinPath(dir, name), size, durationS, fps, kind, tracks };
  registry.set(file.path.toLowerCase(), file);
  registry.set(`name:${name.toLowerCase()}`, file);
  return file;
}

const hash = (text: string): number => {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};
export const unit = (text: string) => (hash(text) % 100000) / 100000;

/** A file the scenario does not know: still probes, still pairs. */
export function fileForPath(path: string): FileFx {
  const known = registry.get(path.toLowerCase()) ?? registry.get(`name:${baseName(path).toLowerCase()}`);
  if (known) return known;
  const name = baseName(path);
  const ext = extOf(name);
  const isVideo = ["mkv", "mp4", "avi", "mov", "m4v", "ts", "webm"].includes(ext);
  const seed = unit(name);
  const durationS = Math.round(2400 + seed * 6000);
  return {
    name,
    dir: dirName(path),
    path,
    size: isVideo ? Math.round((2 + seed * 10) * GB) : Math.round((0.1 + seed * 0.5) * GB),
    durationS,
    fps: isVideo ? NTSC_FILM : null,
    kind: isVideo ? "video" : "audio",
    tracks: [track(0, isVideo ? "eac3" : ext === "aac" ? "aac" : ext || "ac3", 6, "eng", null, 448000)],
  };
}

// ------------------------------------------------------------------ movies

export const MOVIE_DIR = "D:\\Media\\Blade Runner 2049 (2017)";
export const MOVIE_DUB_DIR = "D:\\Media\\Blade Runner 2049 (2017)\\Hindi";
const BR = "Blade Runner 2049 (2017)";

export const MOVIE_VIDEOS: FileFx[] = [
  make("video", MOVIE_DIR, `${BR} 2160p UHD BluRay REMUX.mkv`, 58.2 * GB, 9828, NTSC_FILM, [
    track(0, "truehd", 8, "eng", "TrueHD 7.1 · English", 4_200_000),
    track(1, "ac3", 6, "fre", "Dolby Digital 5.1 · French", 640_000),
    track(2, "ac3", 6, "spa", "Dolby Digital 5.1 · Spanish", 640_000),
    track(3, "ac3", 2, "eng", "Commentary", 192_000),
  ]),
  make("video", MOVIE_DIR, `${BR} 1080p BluRay x264.mkv`, 14.1 * GB, 9828, NTSC_FILM, [
    track(0, "dts", 8, "eng", "DTS-HD MA 7.1", 3_800_000),
  ]),
  make("video", MOVIE_DIR, `${BR} 1080p WEB-DL DDP5.1.mkv`, 7.9 * GB, 9824, NTSC_FILM, [
    track(0, "eac3", 6, "eng", "E-AC3 5.1", 640_000),
  ]),
  make("video", MOVIE_DIR, `${BR} PAL DVD.mkv`, 7.4 * GB, 9433, 25, [track(0, "ac3", 6, "eng", null, 448_000)]),
  make("video", MOVIE_DIR, `${BR} Fan Edit.mkv`, 4.2 * GB, 9062, NTSC_FILM, [track(0, "aac", 2, "eng", null, 160_000)]),
  make("video", MOVIE_DIR, `${BR} Sample.mkv`, 88 * MB, 60, NTSC_FILM, []),
];

export const MOVIE_DUB = make("audio", MOVIE_DUB_DIR, `${BR} Hindi DD5.1.ac3`, 550 * MB, 9828, null, [
  track(0, "ac3", 6, "hin", null, 448_000),
]);

// ------------------------------------------------------------------ series

export const SERIES_DIR = "D:\\Media\\Goblin (2016)";
export const SERIES_DUB_DIR = "D:\\Media\\Goblin (2016)\\Hindi";
const EP_SECONDS = [4423, 4390, 4471, 4402, 4445, 4380, 4468, 4415, 4436, 4459, 4398, 4427, 4444, 4409, 4481, 5210];
const epName = (n: number) => String(n).padStart(2, "0");

export const SERIES_VIDEOS: FileFx[] = EP_SECONDS.map((seconds, i) =>
  make("video", SERIES_DIR, `Goblin.S01E${epName(i + 1)}.1080p.BluRay.x264.mkv`, (3.1 + unit(`v${i}`) * 0.5) * GB, seconds, NTSC_FILM, [
    track(0, "eac3", 6, "kor", "E-AC3 5.1", 640_000),
  ]),
);

/** Fifteen dubs: episode 7's is named without a number, episode 8 has none,
 *  and a Special that belongs to no episode. */
export const SERIES_DUBS: FileFx[] = [
  ...EP_SECONDS.flatMap((seconds, i) => {
    if (i === 7) return [];
    const name = i === 6 ? "Goblin ep7 final (hin).eac3" : `Goblin.S01E${epName(i + 1)}.hin.eac3`;
    return [make("audio", SERIES_DUB_DIR, name, (330 + unit(`d${i}`) * 40) * MB, seconds - 3 - (i % 4), null, [track(0, "eac3", 6, "hin", null, 384_000)])];
  }),
  make("audio", SERIES_DUB_DIR, "Goblin.S01.Special.hin.eac3", 96 * MB, 1320, null, [track(0, "eac3", 6, "hin", null, 384_000)]),
];

/** The five-episode season of the Dub sync mockup: same episodes, shorter names. */
export const SEASON_VIDEOS: FileFx[] = [1, 2, 3, 4, 5].map((n) =>
  make("video", SERIES_DIR, `Goblin.S01E${epName(n)}.1080p.BluRay.mkv`, (3.1 + unit(`s${n}`) * 0.5) * GB, EP_SECONDS[n - 1], NTSC_FILM, [
    track(0, "eac3", 6, "kor", "E-AC3 5.1", 640_000),
  ]),
);
export const SEASON_DUBS: FileFx[] = SERIES_DUBS.filter((file) => /S01E0[1-5]\./.test(file.name));

// ------------------------------------------------------------------- match

export const MATCH_DIR = "D:\\Media\\Parasite (2019)";
export const MATCH_VIDEOS: FileFx[] = [
  ["Parasite.2019.1080p.BluRay.x264.mkv", 13.6 * GB, 7920],
  ["Parasite.2019.1080p.WEB-DL.DDP5.1.mkv", 6.8 * GB, 7916],
  ["Parasite.2019.Black.and.White.1080p.mkv", 11.2 * GB, 7920],
].map(([name, size, seconds]) =>
  make("video", MATCH_DIR, name as string, size as number, seconds as number, NTSC_FILM, [track(0, "eac3", 6, "eng", null, 640_000)]),
);
export const MATCH_DUBS: FileFx[] = [
  ["Parasite.2019.Hindi.AC3.ac3", "ac3"],
  ["Parasite.2019.Tamil.AAC.aac", "aac"],
  ["Parasite.2019.Telugu.E-AC3.eac3", "eac3"],
].map(([name, codec], i) =>
  make("audio", `${MATCH_DIR}\\Dubs`, name, (240 + i * 60) * MB, 7900 + i * 9, null, [
    track(0, codec, codec === "aac" ? 2 : 6, ["hin", "tam", "tel"][i], null, codec === "aac" ? 192_000 : 448_000),
  ]),
);

// -------------------------------------------------------------------- dub

export const DUB_DIR = "D:\\Media\\Skyline Heist (2023)";
export const SKYLINE_VIDEO = make("video", DUB_DIR, "Skyline Heist (2023) 1080p BluRay.mkv", 12.4 * GB, 8430, NTSC_FILM, [
  track(0, "eac3", 6, "eng", "E-AC3 5.1", 640_000),
]);
export const SKYLINE_DUB = make("audio", DUB_DIR, "Skyline Heist (2023) Hindi TV rip.ac3", 437 * MB, 7810, null, [
  track(0, "ac3", 6, "hin", null, 448_000),
]);

// ------------------------------------------------------ probing and listing

export function listing(file: FileFx): TrackListing {
  return { path: file.path, name: file.name, tracks: file.tracks, fps: file.fps, duration: file.durationS, error: null };
}

export function probeOf(file: FileFx): MediaProbe {
  return {
    hasAudio: file.tracks.length > 0,
    hasVideo: file.kind === "video",
    duration: file.durationS,
    audioCodec: file.tracks[0]?.codec ?? null,
    fps: file.fps,
    audioTracks: file.tracks,
    error: null,
  };
}

// ----------------------------------------------------------------- pairing

const RELEASE_TOKENS = new Set([
  "480p", "576p", "720p", "1080p", "2160p", "4k", "uhd", "bluray", "brrip", "bdrip", "web", "dl", "webdl", "webrip", "hdrip", "dvd", "pal",
  "remux", "x264", "x265", "h264", "h265", "hevc", "ac3", "eac3", "aac", "dd5", "ddp5", "dts", "hindi", "tamil", "telugu", "tv", "rip", "tvrip",
  "mkv", "mp4", "1", "fan", "edit", "sample", "final", "hin", "eng", "dub",
]);

function titleTokens(name: string): string[] {
  return stemOf(name)
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token && !RELEASE_TOKENS.has(token));
}

function similarity(a: string, b: string): number {
  const left = new Set(titleTokens(a));
  const right = new Set(titleTokens(b));
  if (left.size === 0 || right.size === 0) return 0;
  const shared = [...left].filter((token) => right.has(token)).length;
  return shared / new Set([...left, ...right]).size;
}

/** "S01E07" from a file name, or null. */
export function episodeKey(name: string): string | null {
  const match = /S(\d{1,2})[ ._-]?E(\d{1,3})/i.exec(name) ?? /\b(\d{1,2})x(\d{2,3})\b/i.exec(name);
  return match ? `S${match[1].padStart(2, "0")}E${match[2].padStart(2, "0")}` : null;
}

function pairOf(video: string, dub: string, method: string, score: number, key?: string): MatchPair {
  return {
    primaryPath: video,
    secondaryPath: dub,
    primaryName: baseName(video),
    secondaryName: baseName(dub),
    key: key ?? titleTokens(baseName(video)).join(" "),
    method,
    score,
    primaryTrack: 0,
    secondaryTrack: 0,
  };
}

function byEpisode(videos: string[], dubs: string[]): PairingReport {
  const pairs: MatchPair[] = [];
  const usedDubs = new Set<string>();
  const dubsByKey = new Map<string, string>();
  dubs.forEach((dub) => {
    const key = episodeKey(baseName(dub));
    if (key && !dubsByKey.has(key)) dubsByKey.set(key, dub);
  });
  const leftVideos: string[] = [];
  videos.forEach((video) => {
    const key = episodeKey(baseName(video));
    const dub = key ? dubsByKey.get(key) : undefined;
    if (key && dub) {
      pairs.push(pairOf(video, dub, `episode (${key})`, 1, key));
      usedDubs.add(dub);
    } else leftVideos.push(video);
  });
  // A dub that names its episode loosely ("ep7") is placed by list order,
  // into the slot of the video that has no dub of its own.
  const stillVideos: string[] = [];
  leftVideos.forEach((video) => {
    const number = /S\d{1,2}E(\d{1,3})/i.exec(baseName(video))?.[1];
    const loose = dubs.find((dub) => !usedDubs.has(dub) && number !== undefined && new RegExp(`\\bep\\s?0?${Number(number)}\\b`, "i").test(baseName(dub)));
    if (loose) {
      pairs.push(pairOf(video, loose, "list order", 0, episodeKey(baseName(video)) ?? undefined));
      usedDubs.add(loose);
    } else stillVideos.push(video);
  });
  pairs.sort((a, b) => a.primaryPath.localeCompare(b.primaryPath));
  return {
    pairs,
    unmatchedPrimary: stillVideos,
    unmatchedSecondary: dubs.filter((dub) => !usedDubs.has(dub)),
    method: pairs[0]?.method ?? "none",
    patternUsed: null,
    warning: null,
  };
}

function byTitle(videos: string[], dubs: string[]): PairingReport {
  const candidates: { score: number; video: string; dub: string }[] = [];
  videos.forEach((video) =>
    dubs.forEach((dub) => {
      const score = similarity(baseName(video), baseName(dub));
      if (score >= 0.6) candidates.push({ score, video, dub });
    }),
  );
  candidates.sort((a, b) => b.score - a.score);
  const usedV = new Set<string>();
  const usedD = new Set<string>();
  const pairs: MatchPair[] = [];
  candidates.forEach(({ score, video, dub }) => {
    if (usedV.has(video) || usedD.has(dub)) return;
    usedV.add(video);
    usedD.add(dub);
    pairs.push(pairOf(video, dub, "filename similarity", Math.round(score * 1000) / 1000));
  });
  const leftV = videos.filter((video) => !usedV.has(video));
  const leftD = dubs.filter((dub) => !usedD.has(dub));
  leftV.slice(0, leftD.length).forEach((video, i) => {
    pairs.push(pairOf(video, leftD[i], "list order", 0));
    usedV.add(video);
    usedD.add(leftD[i]);
  });
  pairs.sort((a, b) => a.primaryPath.localeCompare(b.primaryPath));
  const ordered = pairs.some((pair) => pair.method === "list order");
  return {
    pairs,
    unmatchedPrimary: videos.filter((video) => !usedV.has(video)),
    unmatchedSecondary: dubs.filter((dub) => !usedD.has(dub)),
    method: ordered ? (pairs.some((pair) => pair.method === "filename similarity") ? "filename similarity + list order" : "list order") : "filename similarity",
    patternUsed: null,
    warning: null,
  };
}

export interface PairingInput {
  mode: string;
  dubKind?: string;
  videos: string[];
  dubs: string[];
  explicit?: MatchPair[] | null;
}

export function pairing(input: PairingInput): PairingReport {
  const { mode, videos, dubs } = input;
  if (input.explicit && input.explicit.length > 0) {
    return { pairs: input.explicit, unmatchedPrimary: [], unmatchedSecondary: [], method: "chosen by hand", patternUsed: null, warning: null };
  }
  if (mode === "movie") {
    const dub = dubs[0];
    return {
      pairs: dub ? videos.map((video) => pairOf(video, dub, "movie mode", 1)) : [],
      unmatchedPrimary: [],
      unmatchedSecondary: [],
      method: "movie mode",
      patternUsed: null,
      warning: null,
    };
  }
  if (mode === "compare") {
    return {
      pairs: videos.flatMap((video) => dubs.map((dub) => pairOf(video, dub, "every combination", 1))),
      unmatchedPrimary: [],
      unmatchedSecondary: [],
      method: "every combination",
      patternUsed: null,
      warning: null,
    };
  }
  if (mode === "series") return byEpisode(videos, dubs);
  // Dub sync: movies by title, seasons by episode, "auto" decides by result.
  if (input.dubKind === "movies") return byTitle(videos, dubs);
  const episodes = byEpisode(videos, dubs);
  if (input.dubKind === "series") return episodes;
  const paired = episodes.pairs.length;
  return paired > 0 && (episodes.unmatchedPrimary.length === 0 || (paired >= 3 && paired >= 0.75 * videos.length))
    ? episodes
    : byTitle(videos, dubs);
}

// ---------------------------------------------------------------- results

const NO_DRIFT_TEXT = "No meaningful drift; a single delay aligns the whole file.";

function diagnosis(over: Partial<RateDiagnosis> = {}): RateDiagnosis {
  return {
    driftMsPerS: 0,
    speedRatio: 1,
    sourceFps: null,
    targetFps: null,
    isRateMismatch: false,
    isLikelyCut: false,
    cutPositionS: null,
    cutMagnitudeMs: null,
    explanation: NO_DRIFT_TEXT,
    correctionRatio: null,
    ...over,
  };
}

/** A measured pair. `delayMs` is the engine's (negated for display). */
export function result(video: FileFx, dub: FileFx, over: Partial<SyncResult> & { delayMs: number | null }): SyncResult {
  const delay = over.delayMs;
  return {
    videoFile: video.name,
    audioFile: dub.name,
    primaryPath: video.path,
    secondaryPath: dub.path,
    delayAtStartMs: delay,
    confidence: 0.9,
    driftMsPerS: 0,
    totalDriftMs: 0,
    hasSignificantDrift: false,
    startDelayMs: delay,
    endDelayMs: delay,
    windowsUsed: 6,
    windowsTotal: 6,
    error: null,
    elapsedMs: 5000,
    primaryDurationS: video.durationS,
    secondaryDurationS: dub.durationS,
    primaryTrack: 0,
    secondaryTrack: 0,
    primaryFps: video.fps,
    secondaryFps: dub.fps,
    isLikelyCut: false,
    cutPositionS: null,
    cutUncertaintyS: null,
    cutMagnitudeMs: null,
    isRateMismatch: false,
    codecDelayMs: 0,
    primaryCodec: video.tracks[0]?.codec ?? null,
    secondaryCodec: dub.tracks[0]?.codec ?? null,
    rateDiagnosis: diagnosis(),
    ...over,
  };
}

const failure = (video: FileFx, dub: FileFx, error: string, elapsedMs: number): SyncResult =>
  result(video, dub, {
    delayMs: null,
    delayAtStartMs: null,
    confidence: null,
    driftMsPerS: null,
    totalDriftMs: null,
    hasSignificantDrift: null,
    startDelayMs: null,
    endDelayMs: null,
    windowsUsed: null,
    windowsTotal: null,
    error,
    elapsedMs,
    rateDiagnosis: null,
    codecDelayMs: null,
  });

/** Mockup MOVIES, in order. */
function movieResult(index: number, video: FileFx, dub: FileFx): SyncResult {
  switch (index) {
    case 0:
      return result(video, dub, { delayMs: 317.0, startDelayMs: 317.0, endDelayMs: 317.1, confidence: 0.94, driftMsPerS: 0.00001, totalDriftMs: 0.1, elapsedMs: 41200 });
    case 1:
      return result(video, dub, { delayMs: 317.0, startDelayMs: 317.0, endDelayMs: 316.9, confidence: 0.93, driftMsPerS: -0.00001, totalDriftMs: -0.1, elapsedMs: 38900 });
    case 2:
      // Drifting: displays "+1,208.0 ms" (midpoint), runs from +1,206.9 to
      // +5,258.3 ms; the correction applies delayAtStartMs.
      return result(video, dub, {
        delayMs: -1208.0,
        delayAtStartMs: -1206.9,
        startDelayMs: -1206.9,
        endDelayMs: -5258.3,
        confidence: 0.68,
        driftMsPerS: 0.412,
        totalDriftMs: 4051.4,
        hasSignificantDrift: true,
        windowsUsed: 5,
        elapsedMs: 44000,
        rateDiagnosis: diagnosis({
          driftMsPerS: 0.412,
          speedRatio: 1.000412,
          explanation: "The offset drifts steadily but does not match a standard frame-rate conversion. Resampling still corrects it, but check the result.",
        }),
      });
    case 3:
      return result(video, dub, {
        delayMs: 84.0,
        startDelayMs: 84.0,
        endDelayMs: 84.2,
        confidence: 0.88,
        driftMsPerS: -0.00002,
        totalDriftMs: -0.2,
        isRateMismatch: true,
        elapsedMs: 52700,
        rateDiagnosis: diagnosis({
          driftMsPerS: 42.708,
          speedRatio: 1.042708,
          sourceFps: NTSC_FILM,
          targetFps: 25,
          isRateMismatch: true,
          explanation: "The audio was timed against a 23.976fps source, but this video is 25fps. Resampling the audio corrects it exactly.",
          correctionRatio: 1 / 1.042708,
        }),
      });
    case 4:
      // Different cut: the offset jumps near 1:02:10 (3730 s) by 462 s.
      return result(video, dub, {
        delayMs: 300.0,
        startDelayMs: 300.0,
        endDelayMs: -461700.0,
        confidence: 0.31,
        driftMsPerS: null,
        totalDriftMs: null,
        windowsUsed: 2,
        isLikelyCut: true,
        cutPositionS: 3730,
        cutUncertaintyS: 6,
        cutMagnitudeMs: 462000,
        elapsedMs: 72000,
        rateDiagnosis: diagnosis({
          driftMsPerS: null,
          isLikelyCut: true,
          cutPositionS: 3730,
          cutMagnitudeMs: 462000,
          explanation:
            "The offset jumps by 7m 42.0s about 1:02:10 in (give or take 6s) and stays there. These are different cuts of the same title: the delay below aligns everything before the jump, and nothing after it.",
        }),
      });
    default:
      return failure(video, dub, "The video has no audio track to compare against.", 400);
  }
}

/** Mockup EPISODES: confidence, delay (engine sign) and a note per episode. */
const EP_CONF = [94, 92, 95, 91, 63, 90, 88, 0, 93, 96, 71, 92, 94, 89, 91, 90];
const EP_DELAY = [317.0, 317.0, 320.2, 317.0, -42.0, 317.0, 316.8, 0, 317.0, 317.1, 1540.0, 317.0, 317.0, 318.4, 317.0, 317.0];

function episodeResult(episode: number, video: FileFx, dub: FileFx): SyncResult {
  const i = episode - 1;
  const base = { delayMs: EP_DELAY[i], confidence: EP_CONF[i] / 100, elapsedMs: 5200 + ((i * 313) % 2100) };
  if (episode === 5) return result(video, dub, { ...base, windowsUsed: 3 });
  if (episode === 11) {
    return result(video, dub, {
      ...base,
      isRateMismatch: true,
      rateDiagnosis: diagnosis({
        driftMsPerS: -40.9,
        speedRatio: 25 / NTSC_FILM,
        sourceFps: 25,
        targetFps: NTSC_FILM,
        isRateMismatch: true,
        explanation: "The audio was timed against a 25fps source, but this video is 23.976fps. Resampling the audio corrects it exactly.",
        correctionRatio: NTSC_FILM / 25,
      }),
    });
  }
  return result(video, dub, base);
}

/** Mockup MATCH: confidences per [video][dub]; delays only where it matched. */
const MATCH_CONF = [
  [94, 38, 22],
  [41, 91, 88],
  [12, 18, 27],
];
const MATCH_DELAY = [
  [317.0, 8123.4, -3320.1],
  [-15422.0, -42.0, -41.8],
  [2210.7, -9038.2, 5561.9],
];

function matchResult(row: number, col: number, video: FileFx, dub: FileFx): SyncResult {
  const conf = MATCH_CONF[row][col];
  return result(video, dub, {
    delayMs: MATCH_DELAY[row][col],
    confidence: conf / 100,
    windowsUsed: conf >= 75 ? 6 : conf >= 40 ? 4 : 2,
    elapsedMs: 3400 + row * 410 + col * 230,
  });
}

/** The result for one pair: the scenario's fixture when the names are its
 *  own, a deterministic made-up one for any other file. */
export function resultFor(video: FileFx, dub: FileFx): SyncResult {
  const movie = MOVIE_VIDEOS.indexOf(video);
  if (movie >= 0 && dub === MOVIE_DUB) return movieResult(movie, video, dub);
  const episode = episodeKey(video.name);
  if (episode && /^Goblin/i.test(video.name)) return episodeResult(Number(episode.slice(4)), video, dub);
  const row = MATCH_VIDEOS.indexOf(video);
  const col = MATCH_DUBS.indexOf(dub);
  if (row >= 0 && col >= 0) return matchResult(row, col, video, dub);
  const seed = unit(video.name + dub.name);
  return result(video, dub, {
    delayMs: Math.round((seed * 1600 - 800) * 10) / 10,
    confidence: 0.55 + seed * 0.4,
    elapsedMs: 3000 + Math.round(seed * 4000),
  });
}

export function summarize(results: SyncResult[]): RunSummary {
  const matched = results.filter((r) => r.error === null && r.delayMs !== null);
  const conf = (r: SyncResult) => r.confidence ?? 0;
  return {
    total: results.length,
    matched: matched.length,
    failed: results.filter((r) => r.error !== null).length,
    drifting: matched.filter((r) => r.hasSignificantDrift).length,
    cuts: results.filter((r) => r.isLikelyCut).length,
    rateMismatches: matched.filter((r) => r.isRateMismatch).length,
    high: matched.filter((r) => conf(r) >= 0.75).length,
    medium: matched.filter((r) => conf(r) >= 0.5 && conf(r) < 0.75).length,
    low: matched.filter((r) => conf(r) < 0.5).length,
  };
}
