import type { TranslateEngine, TranslateOptions } from "@/lib/subsync/types";

import {
  Advanced,
  ComboInput,
  EnginePicker,
  Field,
  LanguageSelect,
  NumberField,
  Section,
  Segmented,
  Select,
  TextArea,
  TextInput,
  Toggle,
  type EngineOption,
} from "./fields";
import {
  CJK_HONORIFIC_LANGUAGES,
  KEYED_TRANSLATE_ENGINES,
  MODEL_SUGGESTIONS,
  engineStatus,
  isLlmEngine,
  translateEnginePatch,
} from "./logic";
import type { SettingsPanelProps } from "./props";

const TRANSLATE_ENGINES: EngineOption<TranslateEngine>[] = [
  {
    id: "claude",
    label: "Claude",
    description: "Translates whole scenes with context, glossary and tone. The most natural result.",
    needs: "an Anthropic API key",
  },
  {
    id: "openai",
    label: "OpenAI-compatible",
    description: "OpenAI or any server with the same API (LM Studio, OpenRouter, vLLM…).",
    needs: "an API key, and the server address if it is not OpenAI",
  },
  {
    id: "deepl",
    label: "DeepL",
    description: "Fast machine translation, line by line. Supports formality and glossaries.",
    needs: "a DeepL API key",
  },
  {
    id: "google",
    label: "Google Translate",
    description: "Fast machine translation for the most languages, line by line.",
    needs: "a Google Cloud API key",
  },
  {
    id: "ollama",
    label: "Ollama (local)",
    description: "A model running on this computer. Private and free; slower, and quality depends on the model.",
    needs: "Ollama running, with the model pulled",
  },
];

const BASE_URL_PLACEHOLDER: Partial<Record<TranslateEngine, string>> = {
  openai: "https://api.openai.com/v1",
  ollama: "http://localhost:11434",
};

export function TranslatePanel({ options, onChange, caps, disabled, onInstallPack }: SettingsPanelProps<"translate">) {
  const statuses = Object.fromEntries(TRANSLATE_ENGINES.map((e) => [e.id, engineStatus(caps, "translate", e.id)]));
  const llm = isLlmEngine(options.engine);
  const set = (patch: Partial<TranslateOptions>) => onChange(patch);
  const showHonorifics = options.source === "auto" || CJK_HONORIFIC_LANGUAGES.includes(options.source);
  const showFormality = options.engine !== "google";

  return (
    <div>
      <Section title="Engine">
        <EnginePicker
          label="Translation engine"
          value={options.engine}
          options={TRANSLATE_ENGINES}
          statuses={statuses}
          loading={caps === null}
          disabled={disabled}
          keyed={KEYED_TRANSLATE_ENGINES}
          onChange={(engine) => set(translateEnginePatch(options, engine))}
          onInstallPack={onInstallPack}
        />
        {llm && (
          <Field label="Model" hint="Suggestions are listed; any model id the service offers can be typed.">
            <ComboInput
              id={`subsync-translate-models-${options.engine}`}
              ariaLabel="Model"
              value={options.model}
              suggestions={MODEL_SUGGESTIONS[options.engine as "claude" | "openai" | "ollama"]}
              disabled={disabled}
              onChange={(model) => set({ model })}
            />
          </Field>
        )}
        {(options.engine === "openai" || options.engine === "ollama") && (
          <Field
            label="Server address"
            hint={
              options.engine === "openai"
                ? "Leave empty for OpenAI itself; set it for another OpenAI-compatible server."
                : "Leave empty for Ollama on this computer."
            }
          >
            <TextInput
              ariaLabel="Server address"
              value={options.baseUrl ?? ""}
              mono
              placeholder={BASE_URL_PLACEHOLDER[options.engine]}
              disabled={disabled}
              onChange={(baseUrl) => set({ baseUrl: baseUrl.trim() === "" ? undefined : baseUrl })}
            />
          </Field>
        )}
      </Section>

      <Section title="Languages">
        <Field label="From">
          <LanguageSelect
            value={options.source}
            auto="Detect from the subtitle"
            disabled={disabled}
            onChange={(source) => set({ source })}
          />
        </Field>
        <Field label="To">
          <LanguageSelect value={options.target} disabled={disabled} onChange={(target) => set({ target })} />
        </Field>
      </Section>

      <Section title="Translation">
        {llm && (
          <Field
            group
            label="Quality"
            hint={
              options.quality === "accurate"
                ? "A second pass checks every line against the original and fixes mistranslations. About twice the time and cost."
                : "One pass per scene. Quicker and cheaper; small slips are not caught."
            }
          >
            <Segmented<TranslateOptions["quality"]>
              label="Quality"
              value={options.quality}
              disabled={disabled}
              onChange={(quality) => set({ quality })}
              choices={[
                { value: "fast", label: "Fast", title: "One pass" },
                { value: "accurate", label: "Accurate", title: "Translate, then review every line" },
              ]}
            />
          </Field>
        )}
        <Field label="Glossary" hint="One term per line, source = translation. Names and terms stay the same in every line.">
          <TextArea
            ariaLabel="Glossary"
            value={options.glossary}
            rows={3}
            mono
            placeholder={"源氏 = Genji\n先輩 = senpai"}
            disabled={disabled}
            onChange={(glossary) => set({ glossary })}
          />
        </Field>
        {llm && (
          <Field
            label="About the show"
            hint="What it is and who the characters are. Guides tone, pronouns and how people address each other."
          >
            <TextArea
              ariaLabel="About the show"
              value={options.context}
              rows={3}
              placeholder="A period drama at the Heian court. Genji is a prince; Murasaki is his young ward."
              disabled={disabled}
              onChange={(context) => set({ context })}
            />
          </Field>
        )}
        {showHonorifics && (
          <Field
            group
            label="Honorifics"
            hint={
              options.honorifics === "keep"
                ? "Keeps -san, -nim, senpai and the like as they are. Applies to Japanese, Korean and Chinese sources."
                : "Replaces honorifics with natural phrasing in the target language. Applies to Japanese, Korean and Chinese sources."
            }
          >
            <Segmented<TranslateOptions["honorifics"]>
              label="Honorifics"
              value={options.honorifics}
              disabled={disabled}
              onChange={(honorifics) => set({ honorifics })}
              choices={[
                { value: "keep", label: "Keep", title: "Keep honorifics such as -san" },
                { value: "localize", label: "Localize", title: "Translate them into natural phrasing" },
              ]}
            />
          </Field>
        )}
        {showFormality && (
          <Field label="Formality" hint="How polite the translation sounds, where the target language makes the difference.">
            <Select<TranslateOptions["formality"]>
              value={options.formality}
              disabled={disabled}
              onChange={(formality) => set({ formality })}
              choices={[
                { value: "default", label: "Automatic" },
                { value: "more", label: "More formal" },
                { value: "less", label: "Less formal" },
              ]}
            />
          </Field>
        )}
      </Section>

      <Section title="Output">
        <Toggle
          label="Keep sound descriptions"
          hint="Keeps [music], [door slams] and speaker labels, translated. Off removes them."
          checked={options.keepSdh}
          disabled={disabled}
          onChange={(keepSdh) => set({ keepSdh })}
        />
        <Toggle
          label="Fit reading speed"
          hint="Condenses lines that would be too long to read in the time they are on screen."
          checked={options.fitReadingSpeed}
          disabled={disabled}
          onChange={(fitReadingSpeed) => set({ fitReadingSpeed })}
        />
        <Toggle
          label="Also write a bilingual file"
          hint="A second file with the original and the translation stacked in each line."
          checked={options.bilingual}
          disabled={disabled}
          onChange={(bilingual) => set({ bilingual })}
        />
      </Section>

      <Advanced>
        <Field
          label="Lines per request"
          hint="How much of the scene is sent at once. More gives better context but longer waits and bigger retries."
        >
          <NumberField
            ariaLabel="Lines per request"
            value={options.batchSize}
            min={1}
            max={500}
            step={10}
            unit="lines"
            disabled={disabled}
            onChange={(v) => v !== null && set({ batchSize: Math.round(v) })}
          />
        </Field>
      </Advanced>
    </div>
  );
}
