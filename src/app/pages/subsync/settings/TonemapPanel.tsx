import type { TonemapEngine, TonemapOptions } from "@/lib/subsync/types";

import {
  Advanced,
  EnginePicker,
  Field,
  Hint,
  NumberField,
  Pair,
  Panel,
  Segmented,
  Select,
  Slider,
  type Choice,
  type EngineOption,
} from "./fields";
import { FFMPEG_ENCODER, TEN_BIT_ENCODERS, engineStatus, findPack, tonemapEncoderPatch } from "./logic";
import type { SettingsPanelProps } from "./props";

/** "tonemapx" is reported by the engine but not yet in types.ts's
 *  TonemapEngine; widened here until it is. */
type PanelTonemapEngine = TonemapEngine | "tonemapx";

/** Engines that fix SDR white at 203 nits, so targetPeak and desaturation
 *  are not used (VideoToolbox only with the Full FFmpeg pack's Metal path). */
function fixedWhite(engine: PanelTonemapEngine, fullFfmpeg: boolean): boolean {
  return engine === "libplacebo" || engine === "tonemapx" || (engine === "videotoolbox" && fullFfmpeg);
}

const TONEMAP_ENGINES: EngineOption<PanelTonemapEngine>[] = [
  {
    id: "auto",
    label: "Automatic",
    description: "Picks the best engine available for each video.",
  },
  {
    id: "lut",
    label: "3D LUT",
    description: "A portable colour lookup table. Works with any FFmpeg; good quality, moderate speed.",
  },
  {
    id: "videotoolbox",
    label: "VideoToolbox",
    description:
      "macOS hardware, the fastest on a Mac. With the Full FFmpeg pack it follows the curve and source peak and handles Dolby Vision profile 5; without it the curve is fixed.",
  },
  {
    id: "zscale",
    label: "zscale",
    description: "Precise software tone mapping on the CPU.",
    needs: "an FFmpeg build with zscale",
  },
  {
    id: "libplacebo",
    label: "libplacebo",
    description: "GPU tone mapping with the best highlights. Applies Dolby Vision profile 5.",
    needs: "an FFmpeg build with libplacebo",
  },
  {
    id: "tonemapx",
    label: "tonemapx",
    description: "Jellyfin FFmpeg's fast CPU tone mapper.",
    needs: "the Full FFmpeg pack",
  },
];

/** What Automatic tries, in order (audiosync/subs/tonemap.py). */
const AUTO_ORDER: PanelTonemapEngine[] = ["libplacebo", "lut", "zscale", "videotoolbox", "tonemapx"];

const ALGORITHMS: Choice<TonemapOptions["algorithm"]>[] = [
  { value: "bt2390", label: "BT.2390", title: "The ITU reference curve (recommended): natural contrast, soft highlight roll-off" },
  { value: "hable", label: "Hable", title: "Filmic curve; punchy, can crush shadows" },
  { value: "mobius", label: "Mobius", title: "Keeps mid-tones exact; highlights compress late" },
  { value: "reinhard", label: "Reinhard", title: "Simple and flat" },
  { value: "clip", label: "Clip", title: "Cuts off everything above the target; only for testing" },
];

const ENCODER_LABEL: Record<TonemapOptions["encoder"], string> = {
  x264: "H.264",
  x265: "HEVC",
  h264_videotoolbox: "H.264 (VideoToolbox hardware)",
  hevc_videotoolbox: "HEVC (VideoToolbox hardware)",
};

export function TonemapPanel({ options, onChange, caps, disabled, onInstallPack }: SettingsPanelProps<"tonemap">) {
  const set = (patch: Partial<TonemapOptions>) => onChange(patch);
  const statuses = Object.fromEntries(TONEMAP_ENGINES.map((e) => [e.id, engineStatus(caps, "tonemap", e.id)]));
  const encoders = caps?.ffmpeg.encoders ?? null;
  const hardware = options.encoder === "h264_videotoolbox" || options.encoder === "hevc_videotoolbox";
  const encoderChoices = (Object.keys(ENCODER_LABEL) as TonemapOptions["encoder"][])
    .filter((e) => {
      const isVt = e.endsWith("videotoolbox");
      // VideoToolbox only when this FFmpeg has it; keep a chosen one visible.
      return !isVt || encoders?.[FFMPEG_ENCODER[e]] === true || e === options.encoder;
    })
    .map((e) => {
      const missing = encoders !== null && encoders[FFMPEG_ENCODER[e]] === false;
      return {
        value: e,
        label: missing ? `${ENCODER_LABEL[e]} (not in this FFmpeg)` : ENCODER_LABEL[e],
        disabled: missing && e !== options.encoder,
      };
    });
  const tenBitOk = TEN_BIT_ENCODERS.includes(options.encoder);
  const engine = options.engine as PanelTonemapEngine;
  const fullFfmpeg = findPack(caps, "ffmpeg-full")?.installed === true;
  const whiteFixed = fixedWhite(engine, fullFfmpeg);
  // "Automatic (libplacebo)": the engine's own order for a plain video.
  const autoPick = AUTO_ORDER.find((id) => statuses[id]?.available);
  const engines = TONEMAP_ENGINES.map((e) =>
    e.id === "auto" && autoPick ? { ...e, label: `Automatic (${TONEMAP_ENGINES.find((x) => x.id === autoPick)!.label})` } : e,
  );

  return (
    <Panel>
      <EnginePicker
        packs={caps?.packs}
        label="Tone-mapping engine"
        value={engine}
        options={engines}
        statuses={statuses}
        loading={caps === null}
        disabled={disabled}
        onChange={(next) => set({ engine: next as TonemapEngine })}
        onInstallPack={onInstallPack}
      />
      <Field
        group
        label="Dolby Vision"
        hint={
          options.dolbyVision === "auto"
            ? "Uses the Dolby Vision metadata when the engine can, otherwise the HDR10 base layer. Profile 5 has no usable base layer, so it needs libplacebo, or VideoToolbox with the Full FFmpeg pack."
            : "Ignores the Dolby Vision metadata and converts the HDR10 base layer. Profile 5 files come out with wrong colours this way."
        }
      >
        <Segmented<TonemapOptions["dolbyVision"]>
          label="Dolby Vision"
          value={options.dolbyVision}
          disabled={disabled}
          onChange={(dolbyVision) => set({ dolbyVision })}
          choices={[
            { value: "auto", label: "Use metadata", title: "Apply Dolby Vision reshaping when possible" },
            { value: "baseLayer", label: "Base layer", title: "Use the HDR10 base layer only" },
          ]}
        />
      </Field>

      <Pair>
        <Field label="Resolution" hint="Scales the picture down; the source size is kept when it is already smaller.">
          <Select<TonemapOptions["resolution"]>
            value={options.resolution}
            disabled={disabled}
            onChange={(resolution) => set({ resolution })}
            choices={[
              { value: "source", label: "Same as source" },
              { value: "2160p", label: "2160p" },
              { value: "1440p", label: "1440p" },
              { value: "1080p", label: "1080p" },
              { value: "720p", label: "720p" },
            ]}
          />
        </Field>
        <Field
          label="Curve"
          hint={
            engine === "videotoolbox" && !fullFfmpeg
              ? "VideoToolbox without the Full FFmpeg pack uses its own fixed curve; this applies to the other engines."
              : "How HDR highlights are squeezed into SDR. BT.2390 looks most natural on almost everything."
          }
        >
          <Select<TonemapOptions["algorithm"]>
            value={options.algorithm}
            disabled={disabled}
            onChange={(algorithm) => set({ algorithm })}
            choices={ALGORITHMS}
          />
        </Field>
      </Pair>
      <Pair>
        <Field
          label="Encoder"
          hint={hardware ? "Hardware encoding is much faster; files are larger for the same quality." : "Software encoding: slower, the best quality for the size."}
        >
          <Select<TonemapOptions["encoder"]>
            value={options.encoder}
            disabled={disabled}
            onChange={(encoder) => set(tonemapEncoderPatch(options, encoder))}
            choices={encoderChoices}
          />
        </Field>
        <Field
          label="Quality"
          hint={
            hardware
              ? "Lower is better and larger; 18 looks like the source. Converted to VideoToolbox's own quality scale."
              : "Lower is better and larger; 18 looks like the source, 23 is a good smaller file."
          }
        >
          <NumberField
            ariaLabel="Quality (CRF)"
            value={options.crf}
            min={0}
            max={51}
            step={1}
            unit="CRF"
            width="100%"
            disabled={disabled}
            onChange={(v) => v !== null && set({ crf: Math.round(v) })}
          />
        </Field>
      </Pair>
      <Field
        group
        label="Subtitles"
        hint={
          options.subtitles === "burn"
            ? "Draws one subtitle track into the picture, so it always shows and cannot be turned off."
            : options.subtitles === "copy"
              ? "Keeps the subtitle tracks as selectable tracks."
              : "Leaves the subtitles out."
        }
      >
        <Segmented<TonemapOptions["subtitles"]>
          label="Subtitles"
          value={options.subtitles}
          disabled={disabled}
          onChange={(subtitles) => set({ subtitles })}
          choices={[
            { value: "copy", label: "Copy" },
            { value: "none", label: "None" },
            { value: "burn", label: "Burn in" },
          ]}
        />
      </Field>
      {options.subtitles === "burn" && (
        <Field inline label="Track to burn in" hint="The subtitle stream number in the video, counting from 1. Empty uses the first.">
          <NumberField
            ariaLabel="Track to burn in"
            value={options.burnTrack == null ? null : options.burnTrack + 1}
            nullable
            min={1}
            max={99}
            step={1}
            placeholder="1"
            width={80}
            disabled={disabled}
            onChange={(v) => set({ burnTrack: v === null ? null : Math.max(0, Math.round(v) - 1) })}
          />
        </Field>
      )}

      <Advanced>
        {!hardware && (
          <Field label="Speed" hint="Slower presets make smaller files at the same quality.">
            <Select<TonemapOptions["preset"]>
              value={options.preset}
              disabled={disabled}
              onChange={(preset) => set({ preset })}
              choices={(["ultrafast", "veryfast", "fast", "medium", "slow", "slower"] as const).map((p) => ({
                value: p,
                label: p === "medium" ? "medium (default)" : p,
              }))}
            />
          </Field>
        )}
        <Field
          group
          label="Bit depth"
          hint={tenBitOk ? "10-bit avoids banding in skies and gradients; some older players cannot play it." : "10-bit needs an HEVC encoder."}
        >
          <Segmented<"8" | "10">
            label="Bit depth"
            value={String(options.bitDepth) as "8" | "10"}
            disabled={disabled}
            onChange={(v) => set({ bitDepth: v === "10" ? 10 : 8 })}
            choices={[
              { value: "8", label: "8-bit" },
              { value: "10", label: "10-bit", disabled: !tenBitOk, title: tenBitOk ? undefined : "Needs x265 or HEVC VideoToolbox" },
            ]}
          />
        </Field>
        <Field group label="Audio" hint="Copy keeps the original tracks untouched. AAC re-encodes for players that cannot play them.">
          <Segmented<TonemapOptions["audio"]>
            label="Audio"
            value={options.audio}
            disabled={disabled}
            onChange={(audio) => set({ audio })}
            choices={[
              { value: "copy", label: "Copy" },
              { value: "aac", label: "AAC" },
            ]}
          />
        </Field>
        <Field
          group
          label="Container"
          hint={options.container === "mp4" ? "MP4 plays nearly everywhere; some subtitle and audio formats cannot go in it." : "MKV holds every audio and subtitle format."}
        >
          <Segmented<TonemapOptions["container"]>
            label="Container"
            value={options.container}
            disabled={disabled}
            onChange={(container) => set({ container })}
            choices={[
              { value: "mkv", label: "MKV" },
              { value: "mp4", label: "MP4" },
            ]}
          />
        </Field>
        <Field label="Source peak" hint="The brightest the video was mastered for. Automatic reads it from the file; set it when the metadata is missing or wrong.">
          <Select<"auto" | "set">
            value={options.sourcePeak === "auto" ? "auto" : "set"}
            disabled={disabled}
            onChange={(v) => set({ sourcePeak: v === "auto" ? "auto" : 1000 })}
            choices={[
              { value: "auto", label: "From the file's metadata" },
              { value: "set", label: "Set by hand" },
            ]}
          />
        </Field>
        {options.sourcePeak !== "auto" && (
          <NumberField
            ariaLabel="Source peak"
            value={options.sourcePeak}
            min={100}
            max={10000}
            step={100}
            unit="nits"
            width={120}
            disabled={disabled}
            onChange={(v) => v !== null && set({ sourcePeak: v })}
          />
        )}
        {whiteFixed && (
          <Hint>This engine fixes SDR white at 203 nits, so target peak and highlight desaturation are not used.</Hint>
        )}
        <Field inline label="Target peak" hint="The brightness of SDR white. 100 nits is the standard; raise it a little for a brighter picture.">
          <NumberField
            ariaLabel="Target peak"
            value={options.targetPeak}
            min={80}
            max={400}
            step={10}
            unit="nits"
            width={110}
            disabled={disabled || whiteFixed}
            onChange={(v) => v !== null && set({ targetPeak: v })}
          />
        </Field>
        <Field label="Highlight desaturation" hint="How much very bright colours fade towards white. Higher avoids garish highlights; lower keeps them vivid.">
          <Slider
            ariaLabel="Highlight desaturation"
            min={0}
            max={1}
            step={0.05}
            value={options.desaturation}
            display={options.desaturation.toFixed(2)}
            disabled={disabled || whiteFixed}
            onChange={(desaturation) => set({ desaturation })}
          />
        </Field>
      </Advanced>
    </Panel>
  );
}
