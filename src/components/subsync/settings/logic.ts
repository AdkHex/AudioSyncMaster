/** Pure helpers behind the Subsync settings panels: engine lookup, language
 *  lists, time parsing, frame-rate maths and the style preview. Kept free of
 *  React so the panels stay thin and these can be tested on their own. */

import {
  COMMON_LANGUAGES,
  FPS_PRESETS,
  LANGUAGES,
  type AsrModel,
  type Capabilities,
  type EngineStatus,
  type FpsOptions,
  type PackStatus,
  type SyncEngine,
  type SyncOptions,
  type TonemapOptions,
  type TranslateEngine,
  type TranslateOptions,
} from "@/lib/subsync/types";

export type EngineGroup = keyof Capabilities["engines"];

// ---------------------------------------------------------------- engines

/** The engine's status from caps, or null while caps load / when the engine
 *  did not report it (treated as "unknown", never as unavailable). */
export function engineStatus(
  caps: Capabilities | null,
  group: EngineGroup,
  id: string,
): EngineStatus | null {
  return caps?.engines[group]?.find((status) => status.id === id) ?? null;
}

export function isUnavailable(status: EngineStatus | null): boolean {
  return status !== null && !status.available;
}

export function findPack(caps: Capabilities | null, id: string | null | undefined): PackStatus | null {
  if (!caps || !id) return null;
  return caps.packs.find((pack) => pack.id === id) ?? null;
}

/** The pack that provides Demucs vocal isolation. The pack id is matched
 *  loosely because it is owned by the engine side. */
export function demucsPack(caps: Capabilities | null): PackStatus | null {
  return caps?.packs.find((pack) => pack.id === "demucs") ?? caps?.packs.find((pack) => /demucs/i.test(pack.id)) ?? null;
}

/** Engines whose only requirement is an API key typed into the Engines panel. */
export const KEYED_TRANSLATE_ENGINES: TranslateEngine[] = ["claude", "openai", "deepl", "google"];

export interface EngineAction {
  label: string;
  target: string;
}

/** What the "Install" button next to an unavailable engine should open. */
export function installAction(status: EngineStatus | null, keyed = false): EngineAction | null {
  if (!status || status.available) return null;
  if (status.pack) return { label: "Install", target: status.pack };
  if (keyed) return { label: "Add API key", target: "keys" };
  return null;
}

// -------------------------------------------------------------- languages

export function languageName(code: string): string {
  return LANGUAGES[code] ?? code;
}

/** Common languages first (in their curated order), then every language
 *  alphabetically by English name. */
export function languageGroups(): { common: [string, string][]; all: [string, string][] } {
  const common = COMMON_LANGUAGES.filter((code) => code in LANGUAGES).map(
    (code) => [code, LANGUAGES[code]] as [string, string],
  );
  const all = Object.entries(LANGUAGES).sort((a, b) => a[1].localeCompare(b[1]));
  return { common, all };
}

export const CJK_HONORIFIC_LANGUAGES = ["ja", "ko", "zh", "yue"];

// ----------------------------------------------------------------- numbers

export function clamp(value: number, min?: number, max?: number): number {
  let out = value;
  if (min !== undefined) out = Math.max(min, out);
  if (max !== undefined) out = Math.min(max, out);
  return out;
}

export function formatBytes(bytes: number | null | undefined): string | null {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes <= 0) return null;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

/** "1,234" / "1 234" -> 1234; the list form used by Extract's track input. */
export function parseIndexList(text: string): number[] | null {
  const parts = text
    .split(/[\s,;]+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) return null;
  const out: number[] = [];
  for (const part of parts) {
    if (!/^\d+$/.test(part)) return null;
    const n = Number(part);
    if (!out.includes(n)) out.push(n);
  }
  return out;
}

// -------------------------------------------------------------- timestamps

/** Parse "h:mm:ss.mmm", "mm:ss.mmm" or "ss.mmm" (comma or dot decimals,
 *  optional leading "-") into seconds. Returns null when it is not a time. */
export function parseTimestamp(text: string): number | null {
  const trimmed = text.trim().replace(",", ".");
  const match = /^(-)?(?:(?:(\d+):)?(\d{1,2}):)?(\d{1,2}(?:\.\d{1,3})?|\d+(?:\.\d{1,3})?)$/.exec(trimmed);
  if (!match) return null;
  const [, neg, h, m, s] = match;
  const seconds = Number(s);
  if (m !== undefined && seconds >= 60) return null;
  if (h !== undefined && Number(m) >= 60) return null;
  const total = Number(h ?? 0) * 3600 + Number(m ?? 0) * 60 + seconds;
  if (!Number.isFinite(total)) return null;
  return neg ? -total : total;
}

/** Seconds -> "hh:mm:ss.mmm" */
export function formatTimestamp(seconds: number): string {
  const sign = seconds < 0 ? "-" : "";
  const ms = Math.round(Math.abs(seconds) * 1000);
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  const rest = ms % 1000;
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${sign}${pad(h)}:${pad(m)}:${pad(s)}.${pad(rest, 3)}`;
}

// -------------------------------------------------------------- frame rate

/** A preset matching `value`, or null for a custom rate. Tight tolerance so
 *  23.976 typed as a decimal stays "custom" rather than 24000/1001. */
export function fpsPreset(value: number): { value: number; label: string } | null {
  return FPS_PRESETS.find((preset) => Math.abs(preset.value - value) < 1e-9) ?? null;
}

export function formatFps(value: number): string {
  return fpsPreset(value)?.label.split(" ")[0] ?? String(Math.round(value * 1000) / 1000);
}

/** Factor subtitle times are multiplied by: a subtitle timed for `from`
 *  played on `to`. The same for both modes; `mode` changes what the engine
 *  does with frame-based sources, not the resulting speed. */
export function fpsTimeRatio(from: number, to: number): number | null {
  if (!(from > 0) || !(to > 0)) return null;
  return from / to;
}

export const EPISODE_S = 45 * 60;

/** How far the last line of a 45-minute episode moves, in seconds. */
export function driftOver(ratio: number, seconds = EPISODE_S): number {
  return seconds * (ratio - 1);
}

export interface TwoPointFit {
  ratio: number;
  offsetS: number;
}

export type TwoPointResult = { ok: true; fit: TwoPointFit; warning: string | null } | { ok: false; error: string };

export function twoPointFit(a: [number, number], b: [number, number]): TwoPointResult {
  const span = b[0] - a[0];
  if (Math.abs(span) < 0.001) return { ok: false, error: "The two subtitle times must be different." };
  const ratio = (b[1] - a[1]) / span;
  if (!(ratio > 0)) return { ok: false, error: "The correct times must run in the same order as the subtitle times." };
  const offsetS = a[1] - a[0] * ratio;
  const warning =
    ratio < 0.8 || ratio > 1.25
      ? "That is a large speed change; check the two points are the same lines."
      : Math.abs(span) < 60
        ? "Points less than a minute apart give an imprecise speed; pick lines far apart."
        : null;
  return { ok: true, fit: { ratio, offsetS }, warning };
}

export function formatSigned(value: number, digits = 1): string {
  const rounded = value.toFixed(digits);
  return value > 0 ? `+${rounded}` : rounded;
}

// -------------------------------------------------------------- whisper

export const ASR_MODELS: { id: AsrModel; label: string; note: string }[] = [
  { id: "tiny", label: "Tiny", note: "fastest, rough" },
  { id: "base", label: "Base", note: "fast, rough" },
  { id: "small", label: "Small", note: "fast" },
  { id: "medium", label: "Medium", note: "good" },
  { id: "large-v2", label: "Large v2", note: "accurate" },
  { id: "large-v3", label: "Large v3", note: "most accurate" },
  { id: "large-v3-turbo", label: "Large v3 Turbo", note: "near large-v3, much faster" },
];

/** "Large v3 · installed" / "Large v3 · 3.1 GB download" from the pack's models. */
export function asrModelLabel(model: (typeof ASR_MODELS)[number], pack: PackStatus | null): string {
  const info = pack?.models?.find((m) => m.id === model.id);
  if (!info) return `${model.label} (${model.note})`;
  if (info.installed) return `${model.label} (${model.note}) · installed`;
  const size = formatBytes(info.downloadBytes);
  return `${model.label} (${model.note}) · ${size ? `${size} download` : "not downloaded"}`;
}

// --------------------------------------------------------------- models

export const MODEL_SUGGESTIONS: Record<"claude" | "openai" | "ollama", string[]> = {
  claude: ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5-1"],
  openai: ["gpt-5", "gpt-5-mini", "gpt-4.1"],
  ollama: ["qwen3:14b", "gemma3:12b", "llama3.1:8b"],
};

export const CLAUDE_VISION_MODELS = ["claude-sonnet-5", "claude-opus-5-5", "claude-opus-5", "claude-haiku-4-5"];

export function isLlmEngine(engine: TranslateEngine): engine is "claude" | "openai" | "ollama" {
  return engine === "claude" || engine === "openai" || engine === "ollama";
}

// ------------------------------------------------------------- style preview

export interface StylePreviewRow {
  cpl: number;
  cps: number;
  cpsChildren: number;
  cpsSdh: number;
  cpsSdhChildren: number;
  maxLines: number;
  minDurationS: number;
  maxDurationS: number;
  minGapFrames: number;
  /** SDH overrides of the line limits, where the guide has them. */
  cplSdh?: number;
  maxLinesSdh?: number;
}

const LATIN_DEFAULT: StylePreviewRow = {
  cpl: 42,
  cps: 17,
  cpsChildren: 13,
  cpsSdh: 20,
  cpsSdhChildren: 17,
  maxLines: 2,
  minDurationS: 5 / 6,
  maxDurationS: 7,
  minGapFrames: 2,
};

const INDIC = { ...LATIN_DEFAULT, cps: 22, cpsChildren: 18, cpsSdh: 22, cpsSdhChildren: 18 };
const CHINESE = { ...LATIN_DEFAULT, cpl: 16, cplSdh: 18, maxLinesSdh: 3, cps: 9, cpsChildren: 7, cpsSdh: 11, cpsSdhChildren: 9 };

/** A preview of the Netflix Timed Text Style Guide numbers, mirroring
 *  audiosync/subs/style.py rules_for(). The engine is authoritative; this
 *  only shows what "empty = preset" means for the language. */
export const NETFLIX_PREVIEW: Record<string, StylePreviewRow> = {
  en: { ...LATIN_DEFAULT, cps: 20, cpsChildren: 17 },
  ja: { ...LATIN_DEFAULT, cpl: 13, cplSdh: 16, cps: 4, cpsChildren: 4, cpsSdh: 7, cpsSdhChildren: 7, minDurationS: 0.5 },
  ko: { ...LATIN_DEFAULT, cpl: 16, cps: 12, cpsChildren: 9, cpsSdh: 14, cpsSdhChildren: 11 },
  zh: CHINESE,
  yue: CHINESE,
  ar: { ...LATIN_DEFAULT, cps: 20, cpsChildren: 17 },
  hi: INDIC,
  ta: INDIC,
  te: INDIC,
  bn: INDIC,
  th: { ...LATIN_DEFAULT, cpl: 35 },
};

export interface EffectiveStyle {
  cpl: number;
  cps: number;
  maxLines: number;
  minDurationS: number;
  maxDurationS: number;
  minGapFrames: number;
  /** false when the language has no guide of its own (the default applies). */
  exact: boolean;
}

export function netflixPreview(language: string, children = false, sdh = false): EffectiveStyle {
  const row = NETFLIX_PREVIEW[language] ?? LATIN_DEFAULT;
  const cps = sdh ? (children ? row.cpsSdhChildren : row.cpsSdh) : children ? row.cpsChildren : row.cps;
  return {
    cpl: sdh && row.cplSdh ? row.cplSdh : row.cpl,
    cps,
    maxLines: sdh && row.maxLinesSdh ? row.maxLinesSdh : row.maxLines,
    minDurationS: row.minDurationS,
    maxDurationS: row.maxDurationS,
    minGapFrames: row.minGapFrames,
    exact: language in NETFLIX_PREVIEW,
  };
}

// --------------------------------------------------------------- colour

export const HEX_COLOR = /^#[0-9a-f]{6}$/i;

export function isNamedColor(color: string): color is "keep" | "white" | "gray" | "yellow" {
  return color === "keep" || color === "white" || color === "gray" || color === "yellow";
}

// ------------------------------------------------------------ option patches
// What changing one option also changes, so the panels never leave a
// combination the engine would reject or silently ignore.

/** Switching between the two engines that take a reference file drops the
 *  file: a subtitle is not a usable audio reference, and the other way round. */
export function syncEnginePatch(options: SyncOptions, engine: SyncEngine): Partial<SyncOptions> {
  const refEngines: SyncEngine[] = ["subtitle", "reference"];
  if (options.engine !== engine && refEngines.includes(options.engine) && refEngines.includes(engine)) {
    return { engine, reference: null };
  }
  return { engine };
}

/** A model id only means something to its own service, so switching between
 *  LLM engines picks the new service's first suggestion unless the current
 *  model is already one of its suggestions. */
export function translateEnginePatch(options: TranslateOptions, engine: TranslateEngine): Partial<TranslateOptions> {
  if (!isLlmEngine(engine) || engine === options.engine) return { engine };
  const suggestions = MODEL_SUGGESTIONS[engine];
  return suggestions.includes(options.model) ? { engine } : { engine, model: suggestions[0] };
}

export const TEN_BIT_ENCODERS: TonemapOptions["encoder"][] = ["x265", "hevc_videotoolbox"];

/** 10-bit output only exists for the HEVC encoders. */
export function tonemapEncoderPatch(options: TonemapOptions, encoder: TonemapOptions["encoder"]): Partial<TonemapOptions> {
  if (options.bitDepth === 10 && !TEN_BIT_ENCODERS.includes(encoder)) return { encoder, bitDepth: 8 };
  return { encoder };
}

/** FFmpeg's name for an encoder, as caps.ffmpeg.encoders keys it. */
export const FFMPEG_ENCODER: Record<TonemapOptions["encoder"], string> = {
  x264: "libx264",
  x265: "libx265",
  h264_videotoolbox: "h264_videotoolbox",
  hevc_videotoolbox: "hevc_videotoolbox",
};

export const DEFAULT_TWO_POINT: NonNullable<FpsOptions["twoPoint"]> = { a: [60, 60], b: [2400, 2400] };

// --------------------------------------------------------------- rapidocr

const CYRILLIC = ["ru", "uk", "bg", "sr", "mk", "be", "kk", "mn", "tg", "tt", "ba"];
const LATIN_SCRIPT = [
  "en", "fr", "es", "it", "de", "pt", "nl", "pl", "cs", "sk", "sl", "hr", "bs", "ro", "hu", "sv", "da", "no", "nn",
  "fi", "et", "lv", "lt", "is", "fo", "ca", "gl", "eu", "oc", "br", "cy", "ga", "mt", "lb", "tr", "az", "uz", "tk",
  "id", "ms", "tl", "vi", "sq", "af", "sw", "so", "ha", "yo", "sn", "ln", "mg", "mi", "haw", "ht", "la", "jw", "su",
];

/** The RapidOCR pack model that reads `language`, or null when none does. */
export function rapidOcrModelFor(language: string): "cjk" | "korean" | "latin" | "cyrillic" | null {
  if (language === "ja" || language === "zh" || language === "yue") return "cjk";
  if (language === "ko") return "korean";
  if (CYRILLIC.includes(language)) return "cyrillic";
  if (LATIN_SCRIPT.includes(language)) return "latin";
  return null;
}

/** PackStatus.outdated is reported by the engine but not yet in types.ts. */
export function packOutdated(pack: PackStatus | null): boolean {
  return Boolean(pack && (pack as PackStatus & { outdated?: boolean }).outdated);
}
