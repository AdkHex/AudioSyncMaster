/** Subsync: every subtitle tool, its engines and settings, and the protocol
 *  the app speaks with the engine about them.
 *
 *  Field names here are the wire format: the Python side reads exactly these
 *  camelCase keys from `job.options` (see SUBSYNC.md), so renaming one here
 *  means renaming it in the engine too. */

// ------------------------------------------------------------------ tasks

export type SubTask =
  | "sync"
  | "ocr"
  | "translate"
  | "fps"
  | "generate"
  | "style"
  | "hdrSubs"
  | "tonemap"
  | "convert"
  | "extract"
  | "mux";

/** A subtitle or media input: a file, or stream N of a kind inside a file.
 *  `track` is the N of `0:s:N` (subtitle streams only); `audioTrack` the N
 *  of `0:a:N` when the audio of a video is what matters. */
export interface MediaRef {
  path: string;
  track?: number | null;
  audioTrack?: number;
  /** Text encoding override for subtitle files; omitted = detect. */
  encoding?: string;
}

export type SubFormat = "srt" | "ass" | "vtt" | "ttml" | "sub" | "sbv";
export type OutputFormat = SubFormat | "same" | "sup";

export interface OutputOptions {
  /** Folder for results; null = beside the source. */
  dir?: string | null;
  format?: OutputFormat;
  /** Appended to the file stem, e.g. ".synced" -> Movie.synced.en.srt */
  suffix?: string;
  overwrite?: boolean;
  /** Also write a copy of the video with the result added as a track. */
  mux?: boolean;
  muxDefault?: boolean;
  muxForced?: boolean;
  muxTitle?: string;
}

export interface SubJob {
  id: string;
  task: SubTask;
  input: {
    subtitle?: MediaRef;
    video?: MediaRef;
    /** mux: every subtitle to add. */
    subtitles?: (MediaRef & MuxTrackMeta)[];
  };
  options: TaskOptions[SubTask];
  output: OutputOptions;
}

// -------------------------------------------------------------- languages

/** ISO 639-1 codes Whisper recognises, with English names. Mirrors
 *  audiosync/subs/languages.py. */
export const LANGUAGES: Record<string, string> = {
  en: "English", zh: "Chinese", de: "German", es: "Spanish", ru: "Russian",
  ko: "Korean", fr: "French", ja: "Japanese", pt: "Portuguese", tr: "Turkish",
  pl: "Polish", ca: "Catalan", nl: "Dutch", ar: "Arabic", sv: "Swedish",
  it: "Italian", id: "Indonesian", hi: "Hindi", fi: "Finnish", vi: "Vietnamese",
  he: "Hebrew", uk: "Ukrainian", el: "Greek", ms: "Malay", cs: "Czech",
  ro: "Romanian", da: "Danish", hu: "Hungarian", ta: "Tamil", no: "Norwegian",
  th: "Thai", ur: "Urdu", hr: "Croatian", bg: "Bulgarian", lt: "Lithuanian",
  la: "Latin", mi: "Maori", ml: "Malayalam", cy: "Welsh", sk: "Slovak",
  te: "Telugu", fa: "Persian", lv: "Latvian", bn: "Bengali", sr: "Serbian",
  az: "Azerbaijani", sl: "Slovenian", kn: "Kannada", et: "Estonian", mk: "Macedonian",
  br: "Breton", eu: "Basque", is: "Icelandic", hy: "Armenian", ne: "Nepali",
  mn: "Mongolian", bs: "Bosnian", kk: "Kazakh", sq: "Albanian", sw: "Swahili",
  gl: "Galician", mr: "Marathi", pa: "Punjabi", si: "Sinhala", km: "Khmer",
  sn: "Shona", yo: "Yoruba", so: "Somali", af: "Afrikaans", oc: "Occitan",
  ka: "Georgian", be: "Belarusian", tg: "Tajik", sd: "Sindhi", gu: "Gujarati",
  am: "Amharic", yi: "Yiddish", lo: "Lao", uz: "Uzbek", fo: "Faroese",
  ht: "Haitian Creole", ps: "Pashto", tk: "Turkmen", nn: "Norwegian Nynorsk",
  mt: "Maltese", sa: "Sanskrit", lb: "Luxembourgish", my: "Burmese", bo: "Tibetan",
  tl: "Tagalog", mg: "Malagasy", as: "Assamese", tt: "Tatar", haw: "Hawaiian",
  ln: "Lingala", ha: "Hausa", ba: "Bashkir", jw: "Javanese", su: "Sundanese",
  yue: "Cantonese",
};

/** Shown first in every language picker. */
export const COMMON_LANGUAGES = ["en", "ja", "ko", "zh", "fr", "es", "it", "de", "pt", "hi", "ru", "ar"];

// ------------------------------------------------------------- per task

export type SyncEngine = "audio" | "subtitle" | "reference" | "transcript";
export interface SyncOptions {
  /** audio: speech activity of the video's own audio (any language).
   *  subtitle: a correctly timed subtitle (any language) as the reference.
   *  reference: the audio of the release the subtitle was timed for.
   *  transcript: speech recognition, line by line (same-language subtitles). */
  engine: SyncEngine;
  /** Largest shift searched, in seconds. */
  maxOffsetS: number;
  /** Also try 23.976/24/25/29.97/30 conversions and keep the best. */
  detectFramerate: boolean;
  /** Allow different offsets for different scenes (cut/extended edits). */
  allowSplits: boolean;
  /** Cost of each extra split; higher = fewer, only very clear splits. */
  splitPenalty: number;
  /** Speech detector for the audio engine. silero needs the speech pack. */
  vad: "energy" | "silero";
  reference?: MediaRef | null;
  asrEngine?: AsrEngine;
  asrModel?: AsrModel;
}

export const FPS_PRESETS: { value: number; label: string }[] = [
  { value: 24000 / 1001, label: "23.976" },
  { value: 24, label: "24" },
  { value: 25, label: "25 (PAL)" },
  { value: 30000 / 1001, label: "29.97 (NTSC)" },
  { value: 30, label: "30" },
  { value: 48000 / 1001, label: "47.952" },
  { value: 48, label: "48" },
  { value: 50, label: "50" },
  { value: 60000 / 1001, label: "59.94" },
  { value: 60, label: "60" },
  { value: 120000 / 1001, label: "119.88" },
  { value: 120, label: "120" },
];

export interface FpsOptions {
  /** Rate the subtitle was timed for; "auto" = from the subtitle's source
   *  (MicroDVD header, or the reference video). */
  from: number | "auto";
  /** Rate of the video it should play on; "auto" = read from input.video. */
  to: number | "auto";
  /** time: speed change (PAL speed-up/slow-down, what a 25<->23.976
   *  conversion needs). frames: keep frame numbers (frame-based sources
   *  whose rate was mislabelled). */
  mode: "time" | "frames";
  /** Shift applied after the rate change. */
  offsetMs: number;
  /** Two known points instead of rates: subtitle time -> correct time. */
  twoPoint?: { a: [number, number]; b: [number, number] } | null;
}

export type OcrEngine = "vision" | "tesseract" | "rapidocr" | "claude";
export interface OcrOptions {
  /** vision: Apple Vision (macOS, built in). tesseract: installed CLI.
   *  rapidocr: PaddleOCR models via ONNX (OCR pack). claude: Claude vision. */
  engine: OcrEngine;
  /** Language of the text in the images. */
  language: string;
  /** Second engine for lines the first is unsure of. */
  fallbackEngine: OcrEngine | "none";
  /** Below this the line is re-read by the fallback and flagged for review. */
  minConfidence: number;
  /** Drop ruby/furigana above kanji (Japanese). */
  removeFurigana: boolean;
  detectItalics: boolean;
  /** Keep top-of-screen placement as {\an8}. */
  keepPositions: boolean;
  /** all: one file. forcedOnly: just forced captions. split: two files. */
  forced: "all" | "forcedOnly" | "split";
  fixCommonErrors: boolean;
  /** Image enlargement before OCR. */
  upscale: number;
  claudeModel: string;
  /** Keep the images of flagged lines beside the output for review. */
  saveFlaggedImages: boolean;
}

export type TranslateEngine = "claude" | "openai" | "deepl" | "google" | "ollama";
export interface TranslateOptions {
  engine: TranslateEngine;
  source: string | "auto";
  target: string;
  /** Model id for claude/openai/ollama. */
  model: string;
  /** openai-compatible/ollama endpoint. */
  baseUrl?: string;
  /** accurate: a second pass reviews every line against the source. */
  quality: "fast" | "accurate";
  /** One per line: "源氏 = Genji" -- names and terms kept consistent. */
  glossary: string;
  /** What the show is, who the characters are: guides tone and pronouns. */
  context: string;
  honorifics: "keep" | "localize";
  formality: "default" | "more" | "less";
  /** Keep [sound] descriptions and speaker labels. */
  keepSdh: boolean;
  /** Condense lines that would read faster than the style preset allows. */
  fitReadingSpeed: boolean;
  /** Also write a file with source and translation stacked. */
  bilingual: boolean;
  /** Lines sent per request (the scene window). */
  batchSize: number;
}

export type AsrEngine = "mlx-whisper" | "faster-whisper";
export type AsrModel =
  | "tiny"
  | "base"
  | "small"
  | "medium"
  | "large-v2"
  | "large-v3"
  | "large-v3-turbo";
export interface GenerateOptions {
  /** mlx-whisper: Apple Silicon GPU. faster-whisper: CPU/NVIDIA, any OS. */
  engine: AsrEngine;
  model: AsrModel;
  language: string | "auto";
  /** translate: Whisper's own speech -> English. */
  task: "transcribe" | "translate";
  /** Separate the voices from music first (Demucs; needs its pack). */
  vocalIsolation: boolean;
  /** Skip non-speech before recognising (stops invented lines). */
  vad: boolean;
  beamSize: number;
  /** Names and terms to spell right, in the spoken language. */
  initialPrompt: string;
  suppressHallucinations: boolean;
  /** Style rules used to cut the words into subtitles. */
  stylePreset: string;
  /** Mark [music] / [laughter] where no speech is found. */
  sdh: boolean;
}

export interface StyleOptions {
  /** netflix (per-language Netflix Timed Text Style Guide) or custom. */
  preset: "netflix" | "custom";
  language: string | "auto";
  /** false = report only. */
  fix: boolean;
  snapToShotChanges: boolean;
  /** Overrides; null = the preset's value for the language. */
  cps: number | null;
  cpl: number | null;
  maxLines: number | null;
  minDurationS: number | null;
  maxDurationS: number | null;
  minGapFrames: number | null;
  /** Children's programming reading speeds. */
  children: boolean;
  /** Subtitles for the deaf/hard of hearing rules. */
  sdh: boolean;
}

export interface HdrSubsOptions {
  /** nits: aim subtitle white at a luminance. percent: scale brightness. */
  mode: "nits" | "percent";
  /** 203 nits is the BT.2408 reference white for graphics in HDR. */
  targetNits: number;
  /** What a player shows full-white subtitles at on this setup. */
  assumedPeakNits: number;
  brightnessPercent: number;
  color: "keep" | "white" | "gray" | "yellow" | string;
  /** Output: same format (PGS stays PGS), ASS, or SUP. */
  outputFormat: "same" | "ass" | "sup";
}

export type TonemapEngine = "auto" | "lut" | "videotoolbox" | "zscale" | "libplacebo" | "tonemapx";
export interface TonemapOptions {
  /** lut: portable 3D LUT (any FFmpeg). videotoolbox: macOS hardware.
   *  zscale/libplacebo/tonemapx: when the FFmpeg build has them (the
   *  engine-pack FFmpeg). libplacebo, and on macOS the pack's Metal
   *  VideoToolbox filter and tonemapx, apply Dolby Vision profile 5
   *  reshaping. */
  engine: TonemapEngine;
  algorithm: "bt2390" | "hable" | "mobius" | "reinhard" | "clip";
  resolution: "source" | "2160p" | "1440p" | "1080p" | "720p";
  /** Mastering peak in nits; "auto" = from the file's metadata. */
  sourcePeak: number | "auto";
  targetPeak: number;
  /** 0..1, how much bright highlights lose saturation. */
  desaturation: number;
  encoder: "x264" | "x265" | "h264_videotoolbox" | "hevc_videotoolbox";
  crf: number;
  preset: "ultrafast" | "veryfast" | "fast" | "medium" | "slow" | "slower";
  bitDepth: 8 | 10;
  audio: "copy" | "aac";
  subtitles: "copy" | "none" | "burn";
  burnTrack?: number | null;
  container: "mkv" | "mp4";
  /** auto: use the DV metadata when the engine can, otherwise the base layer. */
  dolbyVision: "auto" | "baseLayer";
}

export interface ConvertOptions {
  /** Frame rate for frame-based formats (MicroDVD) in or out. */
  fps?: number | null;
}

export interface ExtractOptions {
  /** Subtitle stream indexes (0:s:N) or "all". */
  tracks: number[] | "all";
}

export interface MuxTrackMeta {
  language?: string;
  title?: string;
  default?: boolean;
  forced?: boolean;
  hearingImpaired?: boolean;
}

export interface MuxOptions {
  container: "mkv" | "mp4";
  keepExisting: boolean;
}

export interface TaskOptions {
  sync: SyncOptions;
  ocr: OcrOptions;
  translate: TranslateOptions;
  fps: FpsOptions;
  generate: GenerateOptions;
  style: StyleOptions;
  hdrSubs: HdrSubsOptions;
  tonemap: TonemapOptions;
  convert: ConvertOptions;
  extract: ExtractOptions;
  mux: MuxOptions;
}

export const DEFAULT_OPTIONS: TaskOptions = {
  sync: {
    engine: "audio",
    maxOffsetS: 60,
    detectFramerate: true,
    allowSplits: true,
    splitPenalty: 7,
    vad: "energy",
    reference: null,
    asrEngine: "mlx-whisper",
    asrModel: "large-v3-turbo",
  },
  ocr: {
    engine: "vision",
    language: "ja",
    fallbackEngine: "none",
    minConfidence: 0.6,
    removeFurigana: true,
    detectItalics: true,
    keepPositions: true,
    forced: "all",
    fixCommonErrors: true,
    upscale: 2,
    claudeModel: "claude-sonnet-5",
    saveFlaggedImages: false,
  },
  translate: {
    engine: "claude",
    source: "auto",
    target: "en",
    model: "claude-opus-5-5",
    quality: "accurate",
    glossary: "",
    context: "",
    honorifics: "keep",
    formality: "default",
    keepSdh: true,
    fitReadingSpeed: true,
    bilingual: false,
    batchSize: 60,
  },
  fps: { from: 25, to: "auto", mode: "time", offsetMs: 0, twoPoint: null },
  generate: {
    engine: "mlx-whisper",
    model: "large-v3",
    language: "auto",
    task: "transcribe",
    vocalIsolation: false,
    vad: true,
    beamSize: 5,
    initialPrompt: "",
    suppressHallucinations: true,
    stylePreset: "netflix",
    sdh: false,
  },
  style: {
    preset: "netflix",
    language: "auto",
    fix: true,
    snapToShotChanges: true,
    cps: null,
    cpl: null,
    maxLines: null,
    minDurationS: null,
    maxDurationS: null,
    minGapFrames: null,
    children: false,
    sdh: false,
  },
  hdrSubs: {
    mode: "nits",
    targetNits: 203,
    assumedPeakNits: 1000,
    brightnessPercent: 60,
    color: "keep",
    outputFormat: "same",
  },
  tonemap: {
    engine: "auto",
    algorithm: "bt2390",
    resolution: "1080p",
    sourcePeak: "auto",
    targetPeak: 100,
    desaturation: 0.5,
    encoder: "x264",
    crf: 18,
    preset: "medium",
    bitDepth: 8,
    audio: "copy",
    subtitles: "copy",
    burnTrack: null,
    container: "mkv",
    dolbyVision: "auto",
  },
  convert: { fps: null },
  extract: { tracks: "all" },
  mux: { container: "mkv", keepExisting: true },
};

// ------------------------------------------------------------- probing

export type HdrKind = "sdr" | "hdr10" | "hdr10plus" | "hlg" | "dv";

export interface SubtitleTrackInfo {
  /** N of 0:s:N */
  index: number;
  codec: string;
  /** text (SRT/ASS/WebVTT/mov_text) or image (PGS/VobSub/DVB). */
  kind: "text" | "image";
  language: string | null;
  title: string | null;
  forced: boolean;
  default: boolean;
  hearingImpaired: boolean;
  /** Number of subtitle events, where the container records it. */
  events: number | null;
}

export interface VideoInfo {
  width: number;
  height: number;
  fps: number | null;
  codec: string;
  hdr: HdrKind;
  /** Dolby Vision profile (5, 7, 8) and base-layer compatibility id. */
  dvProfile: number | null;
  dvCompatibility: number | null;
  maxCll: number | null;
  masteringPeak: number | null;
  transfer: string | null;
  primaries: string | null;
}

export interface ProbedFile {
  path: string;
  name: string;
  kind: "video" | "audio" | "subtitle" | "unknown";
  duration: number | null;
  video: VideoInfo | null;
  audioTracks: { index: number; codec: string | null; language: string | null; title: string | null; channels: number | null }[];
  subtitleTracks: SubtitleTrackInfo[];
  /** For subtitle files. */
  subtitle?: {
    format: string;
    kind: "text" | "image";
    cues: number | null;
    language: string | null;
    encoding: string | null;
  } | null;
  error?: string | null;
}

// -------------------------------------------------------- capabilities

export interface EngineStatus {
  id: string;
  label: string;
  available: boolean;
  /** Why not, and what would make it available ("Install the OCR pack"). */
  reason?: string | null;
  /** Pack that provides it, if any. */
  pack?: string | null;
}

export interface PackStatus {
  id: string;
  label: string;
  description: string;
  installed: boolean;
  /** Can this platform run it at all? */
  supported: boolean;
  sizeBytes: number | null;
  /** Approximate download size before installing. */
  downloadBytes: number | null;
  version: string | null;
  /** Provided by an interpreter set through AUDIOSYNC_PACK_PYTHON_<ID>
   *  (development), not installed by the app. */
  external?: boolean;
  /** Installed from an older pack definition; reinstalling updates it. */
  outdated?: boolean;
  /** Models downloaded inside the pack (ASR sizes, OCR languages). */
  models?: { id: string; label: string; installed: boolean; downloadBytes: number | null }[];
}

export interface Capabilities {
  platform: "macos" | "windows" | "linux";
  arch: string;
  ffmpeg: {
    path: string | null;
    version: string | null;
    filters: Record<string, boolean>;
    encoders: Record<string, boolean>;
  };
  engines: Record<"sync" | "ocr" | "translate" | "generate" | "tonemap" | "vad", EngineStatus[]>;
  packs: PackStatus[];
}

// -------------------------------------------------------------- outcomes

export interface CueData {
  start: number;
  end: number;
  text: string;
  style?: string;
  align?: number;
  forced?: boolean;
  speaker?: string;
  confidence?: number;
}

export interface CuePreview {
  language: string | null;
  format: string | null;
  count: number;
  cues: CueData[];
}

export interface TaskOutputFile {
  path: string;
  kind: "subtitle" | "video" | "folder" | "report";
  format: string | null;
  language: string | null;
  label: string | null;
}

export interface SubJobOutcome {
  job: number;
  id: string;
  task: SubTask;
  outputs: TaskOutputFile[];
  report: Record<string, unknown>;
  summary: string;
  warnings: string[];
  preview: CuePreview | null;
  error?: string | null;
  cancelled?: boolean;
}

// ------------------------------------------------------------- events
// Every engine event is forwarded as the Tauri event "subsync-event" with
// the engine's JSON as its payload.

export type SubsyncEvent =
  | { type: "subsJobStart"; job: number; id: string; task: SubTask }
  | { type: "subsJobProgress"; job: number; percent: number; stage: string }
  | { type: "subsJobLog"; job: number; message: string }
  | ({ type: "subsJobDone" } & SubJobOutcome)
  | { type: "subsBatchDone"; outcomes: SubJobOutcome[]; cancelled?: boolean; error?: string }
  | { type: "packProgress"; pack: string; percent: number; stage: string; bytes?: number; totalBytes?: number }
  | { type: "packDone"; pack: string; ok: boolean; error?: string | null; status?: PackStatus }
  | { type: "log"; message: string };

/** Keys the host passes to the engine per run, by service. Stored by the
 *  desktop shell, never in localStorage, never logged. */
export type SecretName = "anthropic" | "openai" | "deepl" | "google";
