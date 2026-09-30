import type { OcrEngine, OcrOptions } from "@/lib/subsync/types";

import {
  Advanced,
  ComboInput,
  EnginePicker,
  Field,
  Hint,
  HintLink,
  LanguageSelect,
  Panel,
  Section,
  Segmented,
  Select,
  Slider,
  Toggle,
  type EngineOption,
} from "./fields";
import { CLAUDE_VISION_MODELS, engineStatus, findPack, formatBytes, languageName, packOutdated, rapidOcrModelFor } from "./logic";
import type { SettingsPanelProps } from "./props";

const OCR_ENGINES: EngineOption<OcrEngine>[] = [
  {
    id: "vision",
    label: "Apple Vision",
    description: "Built into macOS. Fast and strong on Japanese, Chinese and Korean.",
  },
  {
    id: "tesseract",
    label: "Tesseract",
    description: "The classic open-source reader. Good on clean Latin-script text.",
    needs: "Tesseract installed, with the language's data",
  },
  {
    id: "rapidocr",
    label: "RapidOCR",
    description: "PaddleOCR models, local. Strong on CJK and busy backgrounds.",
    needs: "the OCR pack",
  },
  {
    id: "claude",
    label: "Claude vision",
    description: "Reads each image with Claude. Most accurate on hard fonts; costs API usage.",
    needs: "an Anthropic API key",
  },
];

export function OcrPanel({ options, onChange, caps, disabled, onInstallPack }: SettingsPanelProps<"ocr">) {
  const statuses = Object.fromEntries(OCR_ENGINES.map((e) => [e.id, engineStatus(caps, "ocr", e.id)]));
  const usesClaude = options.engine === "claude" || options.fallbackEngine === "claude";
  const set = (patch: Partial<OcrOptions>) => onChange(patch);
  const rapidPack = findPack(caps, statuses.rapidocr?.pack ?? "ocr-rapidocr");
  const rapidModelId = rapidOcrModelFor(options.language);
  const rapidModel = rapidPack?.models?.find((m) => m.id === rapidModelId) ?? null;
  const usesRapid = options.engine === "rapidocr" || options.fallbackEngine === "rapidocr";

  return (
    <Panel>
      <EnginePicker
        packs={caps?.packs}
        label="OCR engine"
        value={options.engine}
        options={OCR_ENGINES}
        statuses={statuses}
        loading={caps === null}
        disabled={disabled}
        keyed={["claude"]}
        onChange={(engine) => set(engine === options.fallbackEngine ? { engine, fallbackEngine: "none" } : { engine })}
        onInstallPack={onInstallPack}
      />

      <Field
        label="Language"
        hint="The engine reads this script; picking the right one matters most."
        extra={
          // Only when there is something to do: the model that reads it is installed otherwise.
          usesRapid &&
          rapidPack?.installed &&
          (!rapidModelId || !rapidModel?.installed || packOutdated(rapidPack)) && (
            <Hint tone={!rapidModelId || !rapidModel?.installed || packOutdated(rapidPack) ? "warning" : undefined}>
              {!rapidModelId
                ? `RapidOCR has no model for ${languageName(options.language)}; choose another engine.`
                : packOutdated(rapidPack)
                  ? "The OCR pack is out of date; reinstall it to get the per-language models."
                  : rapidModel?.installed
                    ? `RapidOCR reads this with its ${rapidModel.label} model (installed).`
                    : `RapidOCR needs its ${rapidModel?.label ?? rapidModelId} model${
                        rapidModel && formatBytes(rapidModel.downloadBytes) ? ` (${formatBytes(rapidModel.downloadBytes)})` : ""
                      }; it downloads on first use, or now from `}
              {rapidModelId && (packOutdated(rapidPack) || !rapidModel?.installed) && (
                <HintLink onClick={() => onInstallPack(rapidPack.id)}>Engines</HintLink>
              )}
            </Hint>
          )
        }
      >
        <LanguageSelect value={options.language} disabled={disabled} onChange={(language) => set({ language })} />
      </Field>

      <Section title="Text">
        {options.language === "ja" && (
          <Toggle
            label="Remove furigana"
            hint="Drops the small reading aids printed above kanji, keeping only the main line."
            checked={options.removeFurigana}
            disabled={disabled}
            onChange={(removeFurigana) => set({ removeFurigana })}
          />
        )}
        <Toggle
          label="Detect italics"
          hint="Marks slanted text as italic, as used for off-screen voices and songs."
          checked={options.detectItalics}
          disabled={disabled}
          onChange={(detectItalics) => set({ detectItalics })}
        />
        <Toggle
          label="Fix common OCR errors"
          hint="Corrects typical misreads such as l/I, 0/O and broken punctuation."
          checked={options.fixCommonErrors}
          disabled={disabled}
          onChange={(fixCommonErrors) => set({ fixCommonErrors })}
        />
      </Section>

      <Field
        label="Flag lines below"
        hint="Lines read with less confidence are re-read by the second engine and flagged for review. Higher flags more."
      >
        <Slider
          ariaLabel="Flag lines below"
          min={0}
          max={1}
          step={0.05}
          value={options.minConfidence}
          display={`${Math.round(options.minConfidence * 100)}%`}
          disabled={disabled}
          onChange={(minConfidence) => set({ minConfidence })}
        />
      </Field>

      <Advanced>
        <Field label="Second engine" hint="Re-reads the lines the first engine was unsure of.">
          <Select<OcrEngine | "none">
            value={options.fallbackEngine}
            disabled={disabled}
            onChange={(fallbackEngine) => set({ fallbackEngine })}
            choices={[
              { value: "none", label: "None" },
              ...OCR_ENGINES.filter((e) => e.id !== options.engine).map((e) => {
                const status = statuses[e.id];
                const off = status !== null && !status.available;
                return {
                  value: e.id,
                  label: off ? `${e.label} (not available)` : e.label,
                  disabled: off && options.fallbackEngine !== e.id,
                  title: status?.reason ?? undefined,
                };
              }),
            ]}
          />
        </Field>
        {usesClaude && (
          <Field label="Claude model" hint="The model that reads the images. Any model id is accepted.">
            <ComboInput
              id="subsync-ocr-claude-models"
              ariaLabel="Claude model"
              value={options.claudeModel}
              suggestions={CLAUDE_VISION_MODELS}
              disabled={disabled}
              onChange={(claudeModel) => set({ claudeModel })}
            />
          </Field>
        )}
        <Field
          group
          label="Forced captions"
          hint={
            options.forced === "all"
              ? "Every caption goes into one file."
              : options.forced === "forcedOnly"
                ? "Only forced captions (signs, foreign dialogue) are kept."
                : "Writes two files: everything, and the forced captions alone."
          }
        >
          <Segmented<OcrOptions["forced"]>
            label="Forced captions"
            value={options.forced}
            disabled={disabled}
            onChange={(forced) => set({ forced })}
            choices={[
              { value: "all", label: "All", title: "One file with every caption" },
              { value: "forcedOnly", label: "Forced only", title: "Just the forced captions" },
              { value: "split", label: "Both", title: "Two files: all, and forced only" },
            ]}
          />
        </Field>
        <Toggle
          label="Keep positions"
          hint="Captions shown at the top of the screen stay at the top ({\an8})."
          checked={options.keepPositions}
          disabled={disabled}
          onChange={(keepPositions) => set({ keepPositions })}
        />
        <Field label="Enlarge images" hint="Scales the images up before reading. Helps small or low-resolution text; slower.">
          <Select<string>
            value={String(options.upscale)}
            disabled={disabled}
            onChange={(v) => set({ upscale: Number(v) })}
            choices={[...new Set([1, 1.5, 2, 3, 4, options.upscale])]
              .sort((a, b) => a - b)
              .map((n) => ({ value: String(n), label: n === 1 ? "Off (1×)" : `${n}×` }))}
          />
        </Field>
        <Toggle
          label="Save images of flagged lines"
          hint="Writes the image of each uncertain line beside the output so you can check it."
          checked={options.saveFlaggedImages}
          disabled={disabled}
          onChange={(saveFlaggedImages) => set({ saveFlaggedImages })}
        />
        {options.engine === "tesseract" && <Hint>Tesseract needs its data for the chosen language installed.</Hint>}
      </Advanced>
    </Panel>
  );
}
