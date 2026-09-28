import type { AsrEngine, AsrModel, GenerateOptions } from "@/lib/subsync/types";

import {
  Advanced,
  EnginePicker,
  Field,
  Hint,
  LanguageSelect,
  NumberField,
  Section,
  Segmented,
  Select,
  TextArea,
  Toggle,
  type EngineOption,
} from "./fields";
import { ASR_MODELS, asrModelLabel, demucsPack, engineStatus, findPack } from "./logic";
import type { SettingsPanelProps } from "./props";

const ASR_ENGINES: EngineOption<AsrEngine>[] = [
  {
    id: "mlx-whisper",
    label: "mlx-whisper",
    description: "Whisper on the Apple Silicon GPU. The fastest choice on an M-series Mac.",
    needs: "the mlx-whisper speech pack (Apple Silicon only)",
  },
  {
    id: "faster-whisper",
    label: "faster-whisper",
    description: "Whisper on the CPU, or an NVIDIA GPU. Runs on any system.",
    needs: "the faster-whisper speech pack",
  },
];

export function GeneratePanel({ options, onChange, caps, disabled, onInstallPack }: SettingsPanelProps<"generate">) {
  const statuses = Object.fromEntries(ASR_ENGINES.map((e) => [e.id, engineStatus(caps, "generate", e.id)]));
  const pack = findPack(caps, statuses[options.engine]?.pack);
  const modelInfo = pack?.models?.find((m) => m.id === options.model) ?? null;
  const demucs = demucsPack(caps);
  const demucsMissing = demucs !== null && !demucs.installed;
  const set = (patch: Partial<GenerateOptions>) => onChange(patch);
  const knownPreset = options.stylePreset === "netflix" || options.stylePreset === "custom";

  return (
    <div>
      <Section title="Engine">
        <EnginePicker
          label="Speech recognition engine"
          value={options.engine}
          options={ASR_ENGINES}
          statuses={statuses}
          loading={caps === null}
          disabled={disabled}
          onChange={(engine) => set({ engine })}
          onInstallPack={onInstallPack}
        />
        <Field
          label="Model"
          hint="Larger models get more words right, especially names and quiet speech, but are slower and a bigger download."
          extra={
            pack &&
            modelInfo &&
            !modelInfo.installed && (
              <Hint>
                This model is downloaded on first use, or now from{" "}
                <button type="button" className="underline" onClick={() => onInstallPack(pack.id)}>
                  Engines
                </button>
                .
              </Hint>
            )
          }
        >
          <Select<AsrModel>
            value={options.model}
            disabled={disabled}
            onChange={(model) => set({ model })}
            choices={ASR_MODELS.map((m) => ({ value: m.id, label: asrModelLabel(m, pack) }))}
          />
        </Field>
      </Section>

      <Section title="Speech">
        <Field label="Spoken language" hint="Auto-detect listens to the first half-minute. Set it when the video opens with music or a different language.">
          <LanguageSelect
            value={options.language}
            auto="Auto-detect"
            disabled={disabled}
            onChange={(language) => set({ language })}
          />
        </Field>
        <Field
          group
          label="Output"
          hint={
            options.task === "transcribe"
              ? "Subtitles in the language that is spoken."
              : "Whisper translates the speech straight into English subtitles. For other target languages, use Translate afterwards."
          }
        >
          <Segmented<GenerateOptions["task"]>
            label="Output"
            value={options.task}
            disabled={disabled}
            onChange={(task) => set({ task })}
            choices={[
              { value: "transcribe", label: "Transcribe", title: "Same language as spoken" },
              { value: "translate", label: "English", title: "Translate speech to English" },
            ]}
          />
        </Field>
        <Toggle
          label="Separate voices from music first"
          hint="Isolates the dialogue with Demucs before recognising. Much better under music and effects; slower."
          checked={options.vocalIsolation}
          disabled={disabled || (demucsMissing && !options.vocalIsolation)}
          onChange={(vocalIsolation) => set({ vocalIsolation })}
          extra={
            demucsMissing && (
              <Hint tone="warning">
                Needs the Demucs pack.{" "}
                <button type="button" className="underline" onClick={() => onInstallPack(demucs.id)}>
                  Install
                </button>
              </Hint>
            )
          }
        />
        <Toggle
          label="Skip non-speech"
          hint="Only recognises stretches with speech in them. Stops Whisper inventing lines over music and silence."
          checked={options.vad}
          disabled={disabled}
          onChange={(vad) => set({ vad })}
        />
        <Field label="Names and terms" hint="Spelled as they should appear, in the spoken language. Whisper favours these spellings.">
          <TextArea
            ariaLabel="Names and terms"
            value={options.initialPrompt}
            rows={2}
            placeholder="Kim Shin, Wang Yeo, Ji Eun-tak, Goblin"
            disabled={disabled}
            onChange={(initialPrompt) => set({ initialPrompt })}
          />
        </Field>
      </Section>

      <Section title="Subtitles">
        <Field label="Line rules" hint="How the words are cut into subtitles: line length, reading speed and duration.">
          <Select<string>
            value={options.stylePreset}
            disabled={disabled}
            onChange={(stylePreset) => set({ stylePreset })}
            choices={[
              { value: "netflix", label: "Netflix style guide for the language" },
              { value: "custom", label: "Custom (the Style tool's overrides)" },
              ...(knownPreset ? [] : [{ value: options.stylePreset, label: options.stylePreset }]),
            ]}
          />
        </Field>
        <Toggle
          label="Mark sounds"
          hint="Adds [music] and [laughter] where there is sound but no speech, for viewers who cannot hear it."
          checked={options.sdh}
          disabled={disabled}
          onChange={(sdh) => set({ sdh })}
        />
      </Section>

      <Advanced>
        <Toggle
          label="Suppress hallucinations"
          hint="Drops lines Whisper tends to invent, such as repeated phrases and “Thanks for watching”."
          checked={options.suppressHallucinations}
          disabled={disabled}
          onChange={(suppressHallucinations) => set({ suppressHallucinations })}
        />
        <Field label="Beam size" hint="How many candidate readings are compared. Higher is slightly more accurate and slower; 5 is standard.">
          <NumberField
            ariaLabel="Beam size"
            value={options.beamSize}
            min={1}
            max={10}
            step={1}
            disabled={disabled}
            onChange={(v) => v !== null && set({ beamSize: Math.round(v) })}
          />
        </Field>
      </Advanced>
    </div>
  );
}
