import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_OPTIONS,
  type Capabilities,
  type EngineStatus,
  type SubTask,
  type TaskOptions,
} from "@/lib/subsync/types";

import { SETTINGS_PANELS, type SettingsPanelProps } from "./index";
import { byLabel, change, checkbox, click, control, expand, findAll, radio, textOf, type HostNode } from "./testTree";

const ok = (id: string, label = id, pack: string | null = null): EngineStatus => ({
  id,
  label,
  available: true,
  reason: null,
  pack,
});
const missing = (id: string, reason: string, pack: string | null = null): EngineStatus => ({
  id,
  label: id,
  available: false,
  reason,
  pack,
});

/** What the engine reports on an Apple Silicon Mac with no packs and no keys. */
const CAPS: Capabilities = {
  platform: "macos",
  arch: "arm64",
  ffmpeg: {
    path: "/opt/homebrew/bin/ffmpeg",
    version: "7.1",
    filters: { zscale: false, libplacebo: false },
    encoders: { libx264: true, libx265: true, h264_videotoolbox: true, hevc_videotoolbox: true },
  },
  engines: {
    sync: [ok("audio"), ok("subtitle"), ok("reference"), missing("transcript", "Install a speech pack", "asr-mlx")],
    vad: [ok("energy"), missing("silero", "Install the faster-whisper speech pack", "asr-faster")],
    ocr: [
      ok("vision"),
      missing("tesseract", "Tesseract is not installed (brew install tesseract)"),
      missing("rapidocr", "Install the OCR pack", "ocr-rapidocr"),
      missing("claude", "Add an Anthropic API key"),
    ],
    translate: [
      missing("claude", "Add an Anthropic API key"),
      missing("openai", "Add an OpenAI API key"),
      missing("deepl", "Add a DeepL API key"),
      missing("google", "Add a Google API key"),
      ok("ollama"),
    ],
    generate: [missing("mlx-whisper", "Install the mlx-whisper pack", "asr-mlx"), ok("faster-whisper", "faster-whisper", "asr-faster")],
    tonemap: [ok("lut"), ok("videotoolbox"), missing("zscale", "Needs the full FFmpeg", "ffmpeg-full"), missing("libplacebo", "Needs the full FFmpeg", "ffmpeg-full")],
  },
  packs: [
    {
      id: "asr-faster",
      label: "faster-whisper",
      description: "",
      installed: true,
      supported: true,
      sizeBytes: 900e6,
      downloadBytes: null,
      version: "1.1",
      models: [
        { id: "large-v3", label: "large-v3", installed: false, downloadBytes: 3.1e9 },
        { id: "small", label: "small", installed: true, downloadBytes: 480e6 },
      ],
    },
    {
      id: "demucs",
      label: "Demucs",
      description: "",
      installed: false,
      supported: true,
      sizeBytes: null,
      downloadBytes: 1.2e9,
      version: null,
    },
  ],
};

/** Mount a panel as a host tree, recording every patch and install request. */
function mount<K extends SubTask>(
  task: K,
  options: Partial<TaskOptions[K]> = {},
  extra: Partial<SettingsPanelProps<K>> = {},
) {
  const patches: Partial<TaskOptions[K]>[] = [];
  const installs: string[] = [];
  const Panel = SETTINGS_PANELS[task] as unknown as (p: SettingsPanelProps<K>) => React.ReactElement;
  const props: SettingsPanelProps<K> = {
    options: { ...DEFAULT_OPTIONS[task], ...options },
    onChange: (patch) => patches.push(patch),
    caps: CAPS,
    disabled: false,
    onInstallPack: (id) => installs.push(id),
    pickFile: async () => null,
    videoFps: null,
    ...extra,
  };
  const tree: HostNode[] = expand(<Panel {...props} />);
  const html = renderToStaticMarkup(<Panel {...props} />);
  return { tree, html, patches, installs };
}

describe("SETTINGS_PANELS", () => {
  it("has a panel for every task, and each renders its defaults", () => {
    for (const task of Object.keys(DEFAULT_OPTIONS) as SubTask[]) {
      expect(SETTINGS_PANELS[task], task).toBeTypeOf("function");
      const { html } = mount(task);
      expect(html.length, task).toBeGreaterThan(100);
    }
  });

  it("renders while caps are still loading, without disabling any engine", () => {
    const { tree, html } = mount("ocr", {}, { caps: null });
    expect(html).toContain("Checking…");
    const radios = findAll([byLabel(tree, "OCR engine")], (n) => n.props.role === "radio");
    expect(radios).toHaveLength(4);
    expect(radios.every((r) => !r.props.disabled)).toBe(true);
  });

  it("disables every control while a run is in progress", () => {
    const { tree } = mount("translate", {}, { disabled: true });
    const controls = findAll(tree, (n) => ["input", "select", "textarea"].includes(n.type) || n.props.role === "radio");
    expect(controls.length).toBeGreaterThan(10);
    expect(controls.filter((n) => !n.props.disabled)).toEqual([]);
  });
});

describe("Sync", () => {
  it("shows the reference picker only for the reference engines", () => {
    expect(mount("sync", { engine: "audio" }).html).not.toContain("Reference subtitle");
    expect(mount("sync", { engine: "subtitle" }).tree.some((n) => findAll([n], (x) => x.component === "ReferencePicker").length)).toBe(true);
    expect(mount("sync", { engine: "reference" }).html).toContain("Reference release");
    const chosen = mount("sync", { engine: "subtitle", reference: { path: "/films/ref/Movie.en.srt", track: null } }).html;
    expect(chosen).toContain("Movie.en.srt");
    expect(chosen).not.toContain("Choose a reference before running");
  });

  it("switching between the two reference engines drops the chosen file", () => {
    const { tree, patches } = mount("sync", { engine: "subtitle", reference: { path: "/x/ref.srt" } });
    click(radio(tree, "Sync engine", "Audio of the original release"));
    expect(patches).toEqual([{ engine: "reference", reference: null }]);
  });

  it("disables speech recognition without the pack, and Install opens that pack", () => {
    const { tree, installs, html } = mount("sync");
    const transcript = radio(tree, "Sync engine", "Speech recognition");
    expect(transcript.props.disabled).toBe(true);
    expect(html).toContain("Install a speech pack");
    const install = findAll(tree, (n) => n.type === "button" && textOf(n) === "Install")[0];
    click(install);
    expect(installs).toEqual(["asr-mlx"]);
  });

  it("hides the split penalty when splits are off, and explains it when on", () => {
    expect(mount("sync", { allowSplits: false }).html).not.toContain("Split penalty");
    const { html, tree, patches } = mount("sync", { allowSplits: true });
    expect(html).toContain("Higher means fewer splits");
    change(byLabel(tree, "Split penalty"), "12");
    expect(patches).toEqual([{ splitPenalty: 12 }]);
  });

  it("shows ASR engine and model only for the transcript engine", () => {
    expect(mount("sync", { engine: "audio" }).html).not.toContain("Recogniser");
    const { html } = mount("sync", { engine: "transcript" });
    expect(html).toContain("Recogniser");
    expect(html).toContain("Large v3 Turbo");
  });

  it("offers Silero only with its reason when the pack is missing", () => {
    const { html, tree } = mount("sync", { engine: "audio" });
    expect(html).toContain("Silero: Install the faster-whisper speech pack");
    expect(radio(tree, "Speech detector", "Silero").props.disabled).toBe(true);
  });
});

describe("Frame rate", () => {
  it("previews the speed factor and the drift over a 45-minute episode", () => {
    const { html } = mount("fps", { from: 25, to: 24000 / 1001 });
    // 25 / 23.976 = 1.042708; 2700 s * 0.042708 = 115.3 s
    expect(html).toContain("×1.042708");
    expect(html).toContain("moves by +115.3 s");
  });

  it("uses the queued video's fps for 'auto' and names it", () => {
    const { html } = mount("fps", { from: 25, to: "auto" }, { videoFps: 24 });
    expect(html).toContain("From the video (24 fps)");
    expect(html).toContain("×1.041667");
  });

  it("choosing Custom keeps a non-preset rate and shows the number box", () => {
    const { tree, patches } = mount("fps", { from: 25 });
    change(control(tree, "Subtitle was timed for"), "custom");
    expect(patches).toEqual([{ from: 23.976 }]);
    const custom = mount("fps", { from: 23.976 });
    expect(byLabel(custom.tree, "Subtitle was timed for (custom)").props.value).toBe(23.976);
  });

  it("two-point sync parses times, patches the pair and validates", () => {
    const { tree, patches } = mount("fps", { twoPoint: { a: [60, 61], b: [2400, 2500] } });
    change(byLabel(tree, "Late line, correct time"), "0:41:40.250");
    expect(patches).toEqual([{ twoPoint: { a: [60, 61], b: [2400, 2500.25] } }]);
    change(byLabel(tree, "Late line, correct time"), "41:99");
    expect(patches).toHaveLength(1);

    expect(mount("fps", { twoPoint: { a: [60, 60], b: [60, 90] } }).html).toContain("must be different");
    const fitted = mount("fps", { twoPoint: { a: [0, 0], b: [2400, 2500] } }).html;
    expect(fitted).toContain("×1.041667");
  });

  it("switching to two points seeds a pair and back clears it", () => {
    const { tree, patches } = mount("fps");
    click(radio(tree, "Convert using", "Two Points"));
    expect(patches[0].twoPoint).toEqual({ a: [60, 60], b: [2400, 2400] });
    const back = mount("fps", { twoPoint: { a: [0, 0], b: [10, 10] } });
    click(radio(back.tree, "Convert using", "Frame Rates"));
    expect(back.patches).toEqual([{ twoPoint: null }]);
  });
});

describe("OCR", () => {
  it("offers furigana removal only for Japanese", () => {
    expect(mount("ocr", { language: "ja" }).html).toContain("Remove furigana");
    expect(mount("ocr", { language: "en" }).html).not.toContain("Remove furigana");
  });

  it("asks for a key for Claude and opens the keys section", () => {
    const { tree, installs } = mount("ocr");
    const add = findAll(tree, (n) => n.type === "button" && textOf(n) === "Add API key")[0];
    click(add);
    expect(installs).toEqual(["keys"]);
  });

  it("shows the Claude model only when Claude reads or re-reads", () => {
    expect(mount("ocr", { engine: "vision", fallbackEngine: "none" }).html).not.toContain("Claude model");
    expect(mount("ocr", { engine: "vision", fallbackEngine: "claude" }).html).toContain("Claude model");
  });

  it("choosing the fallback engine as the main engine clears the fallback", () => {
    const { tree, patches } = mount("ocr", { engine: "vision", fallbackEngine: "claude" }, {
      caps: { ...CAPS, engines: { ...CAPS.engines, ocr: [ok("vision"), ok("claude"), ok("tesseract"), ok("rapidocr")] } },
    });
    click(radio(tree, "OCR engine", "Claude vision"));
    expect(patches).toEqual([{ engine: "claude", fallbackEngine: "none" }]);
  });
});

describe("OCR models", () => {
  const rapid = (installed: boolean, outdated = false): Capabilities => ({
    ...CAPS,
    engines: { ...CAPS.engines, ocr: [ok("vision"), ok("rapidocr", "RapidOCR", "ocr-rapidocr")] },
    packs: [
      ...CAPS.packs,
      Object.assign(
        {
          id: "ocr-rapidocr",
          label: "OCR",
          description: "",
          installed: true,
          supported: true,
          sizeBytes: null,
          downloadBytes: null,
          version: "2",
          models: [
            { id: "cjk", label: "Japanese + Chinese", installed, downloadBytes: 85e6 },
            { id: "korean", label: "Korean", installed: false, downloadBytes: 13e6 },
          ],
        },
        outdated ? { outdated: true } : {},
      ),
    ],
  });

  it("says whether the language's RapidOCR model is installed, and opens Engines to get it", () => {
    expect(mount("ocr", { engine: "rapidocr", language: "ja" }, { caps: rapid(true) }).html).toContain(
      "Japanese + Chinese model (installed)",
    );
    const { html, tree, installs } = mount("ocr", { engine: "rapidocr", language: "ko" }, { caps: rapid(true) });
    expect(html).toContain("needs its Korean model (13 MB)");
    click(findAll(tree, (n) => n.type === "button" && textOf(n) === "Engines")[0]);
    expect(installs).toEqual(["ocr-rapidocr"]);
    expect(mount("ocr", { engine: "rapidocr", language: "th" }, { caps: rapid(true) }).html).toContain("no model for Thai");
    expect(mount("ocr", { engine: "rapidocr", language: "ja" }, { caps: rapid(true, true) }).html).toContain("out of date");
    expect(mount("ocr", { engine: "vision", language: "ko" }, { caps: rapid(true) }).html).not.toContain("RapidOCR needs");
  });
});

describe("Translate", () => {
  it("says which API key is missing and offers to add it", () => {
    const { html, tree, installs } = mount("translate");
    expect(html).toContain("Add a DeepL API key");
    expect(radio(tree, "Translation engine", "DeepL").props.disabled).toBe(true);
    // The selected engine stays selectable even when it is unavailable.
    expect(radio(tree, "Translation engine", "Claude").props.disabled).toBe(false);
    click(findAll(tree, (n) => n.type === "button" && textOf(n) === "Add API key")[0]);
    expect(installs).toEqual(["keys"]);
  });

  it("switching LLM engines swaps in a model that service knows", () => {
    const { tree, patches } = mount("translate", { engine: "claude", model: "claude-opus-5-5" });
    click(radio(tree, "Translation engine", "Ollama"));
    expect(patches).toEqual([{ engine: "ollama", model: "qwen3:14b" }]);
  });

  it("shows model, server, quality and context only where they apply", () => {
    const deepl = mount("translate", { engine: "deepl" }).html;
    expect(deepl).not.toContain("Server address");
    expect(deepl).not.toContain("About the show");
    expect(deepl).not.toContain(">Quality<");
    expect(deepl).toContain("Formality");
    const ollama = mount("translate", { engine: "ollama" }).html;
    expect(ollama).toContain("Server address");
    expect(ollama).toContain("http://localhost:11434");
    expect(ollama).toContain("A second pass checks every line");
    expect(mount("translate", { engine: "google" }).html).not.toContain("Formality");
  });

  it("shows the honorifics choice for Japanese, Korean and Chinese sources", () => {
    expect(mount("translate", { source: "ko" }).html).toContain("Honorifics");
    expect(mount("translate", { source: "fr" }).html).not.toContain("Honorifics");
  });

  it("puts common languages first and has the glossary example", () => {
    const { html } = mount("translate");
    expect(html.indexOf('label="Common"')).toBeLessThan(html.indexOf('label="All languages"'));
    expect(html).toContain("源氏 = Genji");
  });
});

describe("Generate", () => {
  it("lists model install state and size from the engine's pack", () => {
    const { html } = mount("generate", { engine: "faster-whisper", model: "large-v3" });
    expect(html).toContain("Large v3 (most accurate) · 3.1 GB download");
    expect(html).toContain("Small (fast) · installed");
    expect(html).toContain("downloaded on first use");
  });

  it("vocal isolation needs the Demucs pack", () => {
    const { tree, installs } = mount("generate");
    expect(checkbox(tree, "Separate voices from music first").props.disabled).toBe(true);
    click(findAll(tree, (n) => n.type === "button" && textOf(n) === "Install" && n.props.className === "underline")[0]);
    expect(installs).toEqual(["demucs"]);
  });

  it("patches the task and the prompt", () => {
    const { tree, patches } = mount("generate");
    click(radio(tree, "Output", "English"));
    change(byLabel(tree, "Names and terms"), "Kim Shin");
    change(checkbox(tree, "Mark sounds"), true);
    expect(patches).toEqual([{ task: "translate" }, { initialPrompt: "Kim Shin" }, { sdh: true }]);
  });
});

describe("Style", () => {
  it("previews the Netflix numbers for the language", () => {
    const ja = mount("style", { language: "ja" }).html;
    expect(ja).toMatch(/data-rule="cpl">13 chars/);
    expect(ja).toMatch(/data-rule="cps">4 chars\/s/);
    const en = mount("style", { language: "en", children: true }).html;
    expect(en).toMatch(/data-rule="cps">17 chars\/s/);
  });

  it("custom shows override boxes with the preset as placeholder; empty = null", () => {
    const { tree, patches } = mount("style", { preset: "custom", language: "ko", cps: 15 });
    const cpl = byLabel(tree, "Characters per line");
    expect(cpl.props.placeholder).toBe("16");
    change(byLabel(tree, "Reading speed"), "");
    change(cpl, "20");
    expect(patches).toEqual([{ cps: null }, { cpl: 20 }]);
  });

  it("fix vs report only", () => {
    const { tree, patches } = mount("style");
    click(radio(tree, "When a rule is broken", "Report Only"));
    expect(patches).toEqual([{ fix: false }]);
  });
});

describe("HDR subtitles", () => {
  it("explains 203 nits and switches to percent", () => {
    const { html, tree, patches } = mount("hdrSubs");
    expect(html).toContain("203 nits is the BT.2408 reference white");
    click(radio(tree, "Set brightness by", "Percentage"));
    expect(patches).toEqual([{ mode: "percent" }]);
    expect(mount("hdrSubs", { mode: "percent" }).html).not.toContain("Subtitle white");
  });

  it("custom colour takes a hex value and shows a swatch", () => {
    const { tree, patches } = mount("hdrSubs");
    change(control(tree, "Colour"), "custom");
    expect(patches).toEqual([{ color: "#e6e6e6" }]);
    const custom = mount("hdrSubs", { color: "#336699" });
    expect(custom.html).toContain("background:#336699");
    expect(byLabel(custom.tree, "Custom colour").props.value).toBe("#336699");
    expect(mount("hdrSubs", { color: "#33" }).html).toContain("Use a hex colour");
  });
});

describe("Tone-map video", () => {
  it("hides VideoToolbox encoders unless this FFmpeg has them", () => {
    const withVt = mount("tonemap").html;
    expect(withVt).toContain("HEVC (VideoToolbox hardware)");
    const noVt = mount("tonemap", {}, { caps: { ...CAPS, ffmpeg: { ...CAPS.ffmpeg, encoders: { libx264: true, libx265: true } } } }).html;
    expect(noVt).not.toContain("VideoToolbox hardware");
    expect(mount("tonemap", {}, { caps: null }).html).not.toContain("VideoToolbox hardware");
  });

  it("10-bit only with HEVC, and switching to H.264 drops back to 8-bit", () => {
    const { tree } = mount("tonemap", { encoder: "x264" });
    expect(radio(tree, "Bit depth", "10-bit").props.disabled).toBe(true);
    const hevc = mount("tonemap", { encoder: "x265", bitDepth: 10 });
    change(control(hevc.tree, "Encoder"), "h264_videotoolbox");
    expect(hevc.patches).toEqual([{ encoder: "h264_videotoolbox", bitDepth: 8 }]);
  });

  it("libplacebo and tonemapx are disabled with the full-FFmpeg install, and DV profile 5 is explained", () => {
    const withTonemapx = { ...CAPS, engines: { ...CAPS.engines, tonemap: [...CAPS.engines.tonemap, missing("tonemapx", "Needs the full FFmpeg", "ffmpeg-full")] } };
    const { tree, installs, html } = mount("tonemap", {}, { caps: withTonemapx });
    expect(radio(tree, "Tone-mapping engine", "libplacebo").props.disabled).toBe(true);
    expect(radio(tree, "Tone-mapping engine", "tonemapx").props.disabled).toBe(true);
    expect(html).toContain("Applies Dolby Vision profile 5");
    expect(html).toContain("needs libplacebo, or VideoToolbox with the Full FFmpeg pack");
    click(findAll(tree, (n) => n.type === "button" && textOf(n) === "Install")[0]);
    expect(installs).toEqual(["ffmpeg-full"]);
  });

  it("engines that fix SDR white at 203 nits disable target peak and desaturation", () => {
    const lut = mount("tonemap", { engine: "lut" });
    expect(byLabel(lut.tree, "Target peak").props.disabled).toBe(false);
    const placebo = mount("tonemap", { engine: "libplacebo" });
    expect(placebo.html).toContain("fixes SDR white at 203 nits");
    expect(byLabel(placebo.tree, "Target peak").props.disabled).toBe(true);
    expect(byLabel(placebo.tree, "Highlight desaturation").props.disabled).toBe(true);
    // VideoToolbox: fixed curve without the pack, fixed white with it.
    const vt = mount("tonemap", { engine: "videotoolbox" });
    expect(vt.html).toContain("uses its own fixed curve");
    expect(byLabel(vt.tree, "Target peak").props.disabled).toBe(false);
    const full = { ...CAPS, packs: [...CAPS.packs, { id: "ffmpeg-full", label: "Full FFmpeg", description: "", installed: true, supported: true, sizeBytes: null, downloadBytes: null, version: "1" }] };
    expect(byLabel(mount("tonemap", { engine: "videotoolbox" }, { caps: full }).tree, "Target peak").props.disabled).toBe(true);
  });

  it("asks for the burn-in track only when burning, counting from 1", () => {
    expect(mount("tonemap").html).not.toContain("Track to burn in");
    const { tree, patches } = mount("tonemap", { subtitles: "burn", burnTrack: null });
    change(byLabel(tree, "Track to burn in"), "2");
    expect(patches).toEqual([{ burnTrack: 1 }]);
  });
});

describe("Convert, Extract, Mux", () => {
  it("convert picks a MicroDVD frame rate", () => {
    const { tree, patches } = mount("convert");
    change(control(tree, "Frame rate for MicroDVD"), "2");
    expect(patches).toEqual([{ fps: 25 }]);
  });

  it("extract switches to selected tracks and parses the list", () => {
    const { tree, patches } = mount("extract");
    click(radio(tree, "Tracks", "Selected"));
    expect(patches).toEqual([{ tracks: [0] }]);
    const sel = mount("extract", { tracks: [0] });
    const input = byLabel(sel.tree, "Track numbers");
    (input.props.onBlur as (e: unknown) => void)({ target: { value: "1, 3", setCustomValidity: () => undefined } });
    expect(sel.patches).toEqual([{ tracks: [0, 2] }]);
  });

  it("mux container and keep existing", () => {
    const { tree, patches } = mount("mux");
    click(radio(tree, "Container", "MP4"));
    change(checkbox(tree, "Keep the video's existing subtitles"), false);
    expect(patches).toEqual([{ container: "mp4" }, { keepExisting: false }]);
  });
});
