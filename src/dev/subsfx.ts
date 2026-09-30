/** Subsync fixtures: subtitle files, probes, engines and cues. Dev only.
 *  Cue text and timings are design/mockup/data.ts CUES and OCR. */

import type {
  Capabilities,
  CueData,
  CuePreview,
  EngineStatus,
  PackStatus,
  ProbedFile,
  SubJob,
  SubJobOutcome,
  SubTask,
} from "@/lib/subsync/types";

import { GB, MB, baseName, dirName, extOf, stemOf, unit } from "./fixtures";

export interface PickedFx {
  name: string;
  path: string;
  type: "video" | "audio" | "subtitle";
  size: number | null;
}

export const BR_DIR = "D:\\Media\\Blade Runner 2049 (2017)";
export const YN_DIR = "D:\\Media\\Your Name (2016)";
export const BR_VIDEO_NAME = "Blade.Runner.2049.2017.1080p.BluRay.mkv";

const file = (dir: string, name: string, type: PickedFx["type"], size: number): PickedFx => ({
  name,
  path: `${dir}\\${name}`,
  type,
  size,
});

export const BR_VIDEO = file(BR_DIR, BR_VIDEO_NAME, "video", 14.1 * GB);
export const BR_SUBS: PickedFx[] = [
  file(BR_DIR, "Blade.Runner.2049.en.srt", "subtitle", 96 * 1024),
  file(BR_DIR, "Blade.Runner.2049.en.forced.srt", "subtitle", 7 * 1024),
  file(BR_DIR, "Blade.Runner.2049.hi.srt", "subtitle", 104 * 1024),
];
export const YOUR_NAME_SUP = file(YN_DIR, "Your.Name.2016.jpn.sup", "subtitle", 14.7 * MB);

const SUB_EXT = ["srt", "ass", "ssa", "vtt", "sub", "sup", "ttml", "sbv"];
const VIDEO_EXT = ["mkv", "mp4", "avi", "mov", "m4v", "ts", "webm"];

/** Any path as a picked file, by its extension. */
export function pickedFromPath(path: string): PickedFx {
  const name = baseName(path);
  const ext = extOf(name);
  const type = SUB_EXT.includes(ext) ? "subtitle" : VIDEO_EXT.includes(ext) ? "video" : "audio";
  const known = [BR_VIDEO, ...BR_SUBS, YOUR_NAME_SUP].find((f) => f.name.toLowerCase() === name.toLowerCase());
  if (known) return { ...known, path };
  return { name, path, type, size: Math.round((type === "video" ? 4 * GB : 90 * 1024) * (0.6 + unit(name))) };
}

/** A dropped folder: what it holds. */
export function expandFolder(path: string): PickedFx[] {
  if (/your name/i.test(path)) return [YOUR_NAME_SUP];
  return [...BR_SUBS, BR_VIDEO];
}

const LANG_OF: Record<string, string> = { en: "en", hi: "hi", jpn: "ja", ja: "ja", ko: "ko", fr: "fr" };

export function probeOf(path: string): ProbedFile {
  const picked = pickedFromPath(path);
  const base = { path, name: picked.name, duration: null, video: null, audioTracks: [], subtitleTracks: [], error: null };
  if (picked.type === "video") {
    return {
      ...base,
      kind: "video",
      duration: 9828,
      video: {
        width: 1920,
        height: 1080,
        fps: 24000 / 1001,
        codec: "h264",
        hdr: "sdr",
        dvProfile: null,
        dvCompatibility: null,
        maxCll: null,
        masteringPeak: null,
        transfer: "bt709",
        primaries: "bt709",
      },
      audioTracks: [{ index: 0, codec: "eac3", language: "eng", title: "E-AC3 5.1", channels: 6 }],
      subtitleTracks: [
        { index: 0, codec: "subrip", kind: "text", language: "eng", title: null, forced: false, default: true, hearingImpaired: false, events: 1412 },
        { index: 1, codec: "hdmv_pgs_subtitle", kind: "image", language: "hin", title: null, forced: false, default: false, hearingImpaired: false, events: 1388 },
      ],
    };
  }
  if (picked.type === "subtitle") {
    const ext = extOf(picked.name);
    const tag = /\.(en|hi|jpn|ja|ko|fr)(\.forced)?\.[^.]+$/i.exec(picked.name)?.[1]?.toLowerCase() ?? "en";
    const image = ext === "sup";
    return {
      ...base,
      kind: "subtitle",
      subtitle: {
        format: image ? "pgs" : ext,
        kind: image ? "image" : "text",
        cues: image ? 1204 : /forced/i.test(picked.name) ? 74 : 1412,
        language: LANG_OF[tag] ?? tag,
        encoding: image ? null : "utf-8",
      },
    };
  }
  return { ...base, kind: "audio", duration: 9828, audioTracks: [{ index: 0, codec: "ac3", language: "hin", title: null, channels: 6 }] };
}

// ------------------------------------------------------------ capabilities

const engine = (id: string, label: string, available: boolean, reason?: string, pack?: string): EngineStatus => ({
  id,
  label,
  available,
  reason: reason ?? null,
  pack: pack ?? null,
});

/** Pack state; `subs_pack` flips `installed`. */
export const PACKS: PackStatus[] = [
  {
    id: "ocr-rapidocr",
    label: "OCR engine (RapidOCR)",
    description: "Reads image subtitles (PGS, VobSub) with PaddleOCR models.",
    installed: true,
    supported: true,
    sizeBytes: 212 * MB,
    downloadBytes: 190 * MB,
    version: "1.3.0",
    // The engine's model ids (audiosync/subs/packs.py).
    models: [
      { id: "cjk", label: "Japanese + Chinese", installed: true, downloadBytes: 85 * MB },
      { id: "latin", label: "Latin script", installed: true, downloadBytes: 8 * MB },
      { id: "korean", label: "Korean", installed: false, downloadBytes: 13 * MB },
    ],
  },
  {
    id: "asr-faster",
    label: "Speech recognition (Whisper)",
    description: "faster-whisper on CPU or NVIDIA GPUs, for generating and matching subtitles by speech.",
    installed: false,
    supported: true,
    sizeBytes: null,
    // Decimal, as download sizes are quoted: "Whisper (1.6 GB)".
    downloadBytes: 1.6e9,
    version: null,
    models: [
      { id: "large-v3-turbo", label: "large-v3-turbo", installed: false, downloadBytes: 1.6e9 },
      { id: "small", label: "small", installed: false, downloadBytes: 480 * MB },
    ],
  },
  {
    id: "asr-mlx",
    label: "Speech recognition (Apple Silicon)",
    description: "mlx-whisper on the Apple GPU.",
    installed: false,
    supported: false,
    sizeBytes: null,
    downloadBytes: null,
    version: null,
  },
  {
    id: "demucs",
    label: "Vocal isolation (Demucs)",
    description: "Separates voices from music before recognition.",
    installed: false,
    supported: true,
    sizeBytes: null,
    downloadBytes: 1.1 * GB,
    version: null,
  },
  {
    id: "ffmpeg-full",
    label: "Full FFmpeg",
    description: "An FFmpeg build with zscale, libplacebo and tonemapx.",
    installed: true,
    supported: true,
    sizeBytes: 148 * MB,
    downloadBytes: 62 * MB,
    version: "7.1",
  },
];

export function capabilities(): Capabilities {
  // The speech engines run once their pack is installed (`subs_pack`).
  const hasWhisper = PACKS.some((p) => p.id === "asr-faster" && p.installed);
  const whisper = hasWhisper ? undefined : "Install the Whisper pack";
  return {
    platform: "windows",
    arch: "x86_64",
    ffmpeg: {
      path: "C:\\Program Files\\AudioSyncMaster\\ffmpeg\\ffmpeg.exe",
      version: "7.1",
      filters: { zscale: true, libplacebo: true, tonemapx: true, tonemap: true },
      encoders: { libx264: true, libx265: true, h264_nvenc: true, hevc_nvenc: true, aac: true },
    },
    engines: {
      sync: [
        engine("audio", "Speech in the video", true),
        engine("subtitle", "A correctly timed subtitle", true),
        engine("reference", "The audio of another release", true),
        engine("transcript", "Speech recognition, line by line", hasWhisper, whisper, "asr-faster"),
      ],
      ocr: [
        engine("vision", "Apple Vision", false, "Only on macOS"),
        engine("tesseract", "Tesseract", false, "Tesseract is not installed"),
        engine("rapidocr", "RapidOCR", true, undefined, "ocr-rapidocr"),
        engine("claude", "Claude vision", true),
      ],
      translate: [
        engine("claude", "Claude", true),
        engine("openai", "OpenAI", true),
        engine("deepl", "DeepL", false, "Add a DeepL key"),
        engine("google", "Google Translate", false, "Add a Google key"),
        engine("ollama", "Ollama (local)", false, "Ollama is not running"),
      ],
      generate: [
        engine("mlx-whisper", "MLX Whisper", false, "Apple Silicon only", "asr-mlx"),
        engine("faster-whisper", "faster-whisper", hasWhisper, whisper, "asr-faster"),
      ],
      tonemap: [
        engine("lut", "3D LUT", true),
        engine("zscale", "zscale", true, undefined, "ffmpeg-full"),
        engine("libplacebo", "libplacebo", true, undefined, "ffmpeg-full"),
        engine("tonemapx", "tonemapx", true, undefined, "ffmpeg-full"),
      ],
      vad: [engine("energy", "Energy", true), engine("silero", "Silero", hasWhisper, whisper, "asr-faster")],
    },
    packs: PACKS,
  };
}

// -------------------------------------------------------------------- cues

const clock = (text: string): number => {
  const [h, m, s] = text.split(":");
  return Number(h) * 3600 + Number(m) * 60 + Number(s);
};

const CUE_ROWS: [string, string, string][] = [
  ["00:31:02.480", "00:31:04.902", "Where are you going?"],
  ["00:31:05.120", "00:31:07.300", "Somewhere the snow doesn't reach."],
  ["00:31:07.640", "00:31:08.850", "That's everywhere, K."],
  ["00:31:09.010", "00:31:12.760", "Then I'll find the place that isn't."],
  ["00:31:13.200", "00:31:14.100", "You always say that like it's easy to do."],
  ["00:31:15.400", "00:31:18.020", "[distant thunder]"],
  ["00:31:18.300", "00:31:21.100", "I kept the wooden horse."],
];

const OCR_ROWS: [string, string, number][] = [
  ["00:12:40.120", "どこへ行くの？", 97],
  ["00:12:43.100", "雪が届かないところへ。", 94],
  ["00:12:46.020", "そんな場所、ないよ。", 43],
  ["00:12:48.900", "じゃあ、探しに行こう。", 91],
  ["00:12:52.010", "三葉？ 三葉なの？", 52],
  ["00:12:55.300", "君の名前は…", 96],
];

/** Every line of a file, as the engine sends them (up to 2,000): the
 *  mockup's lines at the mockup's numbers (412-418, OCR 88-93), and lines
 *  made from them around those, at an easy reading speed. */
function fileCues(count: number, rows: { start: number; end: number; text: string; confidence?: number }[], at: number, seed: string): CueData[] {
  const cues: CueData[] = [];
  const first = rows[0].start;
  const last = rows[rows.length - 1].end;
  for (let i = 0; i < count; i += 1) {
    const own = rows[i - at];
    if (own) {
      cues.push({ ...own });
      continue;
    }
    const before = i < at;
    // Spread evenly before the mockup's lines and after them.
    const start = before ? 30 + ((first - 40) * i) / at : last + 1 + (i - at - rows.length) * 4.4;
    const text = rows[i % rows.length].text;
    const cue: CueData = { start, end: start + 3.2, text };
    if (rows[0].confidence !== undefined) {
      // Every 80th below 60%: with the mockup's two, its 17 to check.
      const low = i % 80 === 40;
      cue.confidence = low ? 0.35 + unit(`${seed}c${i}`) * 0.2 : 0.85 + unit(`${seed}c${i}`) * 0.14;
    }
    cues.push(cue);
  }
  return cues;
}

export function cuePreview(path: string, shiftS = 0): CuePreview {
  const image = extOf(path) === "sup";
  if (image) {
    const rows = OCR_ROWS.map(([start, text, confidence], i) => {
      const from = clock(start);
      const next = OCR_ROWS[i + 1] ? clock(OCR_ROWS[i + 1][0]) - 0.15 : from + 2.6;
      return { start: from, end: next, text, confidence: confidence / 100 };
    });
    return { language: "ja", format: "pgs", count: 1204, cues: fileCues(1204, rows, 87, path) };
  }
  const count = /forced/i.test(path) ? 74 : 1412;
  const rows = CUE_ROWS.map(([start, end, text]) => ({ start: clock(start), end: clock(end), text }));
  const cues = fileCues(count, rows, Math.min(411, count - rows.length), path).map((cue) => ({ ...cue, start: cue.start + shiftS, end: cue.end + shiftS }));
  return { language: /\.hi\./i.test(path) ? "hi" : "en", format: "srt", count, cues };
}

// ---------------------------------------------------------------- outcomes

const STAGES: Record<string, string[]> = {
  sync: ["listening for speech", "matching the subtitle to the speech", "looking for cuts", "writing"],
  ocr: ["reading the images", "recognising text", "re-reading unsure lines", "writing"],
  translate: ["reading the subtitle", "translating", "reviewing the lines", "writing"],
  fps: ["reading the subtitle", "changing the rate", "writing"],
};
export const stagesFor = (task: SubTask): string[] => STAGES[task] ?? ["working", "writing"];

function outputPath(job: SubJob, language: string | null, ext?: string): string {
  const source = job.input.subtitle?.path ?? job.input.video?.path ?? job.id;
  const dir = job.output.dir ?? dirName(source);
  const suffix = job.output.suffix ?? ".synced";
  const format = job.output.format && job.output.format !== "same" ? job.output.format : ext ?? (extOf(source) === "sup" ? "srt" : extOf(source) || "srt");
  const stem = stemOf(baseName(source)).replace(/\.(jpn|ja)$/i, language ? `.${language}` : "");
  return `${dir}\\${stem}${suffix}.${format}`;
}

/** The engine's final word on one job. */
export function outcomeFor(job: SubJob, index: number): SubJobOutcome {
  const source = job.input.subtitle?.path ?? job.input.video?.path ?? "";
  const done = { job: index, id: job.id, task: job.task, error: null, cancelled: false };
  if (job.task === "sync") {
    const first = index === 0;
    const path = outputPath(job, null);
    return {
      ...done,
      outputs: [{ path, kind: "subtitle", format: extOf(path), language: "en", label: null }],
      // The engine lists where the offset changes (audiosync/subs/sync.py).
      report: {
        offsetMs: 3250,
        method: "audio",
        framerate: "23.976",
        splits: first ? [{ atS: 1230, offsetMs: -127400 }, { atS: 3070.25, offsetMs: -238000 }] : [],
        cuesMoved: 1412,
      },
      summary: "Shifted by +3.250 s",
      warnings: first ? ["Two scenes were cut"] : [],
      preview: cuePreview(source, 3.25),
    };
  }
  if (job.task === "ocr") {
    const path = outputPath(job, "ja", "srt");
    return {
      ...done,
      outputs: [{ path, kind: "subtitle", format: "srt", language: "ja", label: null }],
      // The engine's own keys (audiosync/subs/ocr.py).
      report: { events: 1204, cues: 1204, lowConfidence: 17, engine: "rapidocr" },
      summary: "1,204 lines read · 17 to check",
      warnings: ["17 lines are below 60% confidence"],
      preview: cuePreview(source),
    };
  }
  const path = outputPath(job, null);
  return {
    ...done,
    outputs: [{ path, kind: "subtitle", format: extOf(path), language: null, label: null }],
    report: { task: job.task },
    summary: `${job.task} finished`,
    warnings: [],
    preview: null,
  };
}
