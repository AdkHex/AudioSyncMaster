import type { StyleOptions } from "@/lib/subsync/types";

import { Field, Hint, LanguageSelect, NumberField, Section, Segmented, Toggle } from "./fields";
import { languageName, netflixPreview, type EffectiveStyle } from "./logic";
import type { SettingsPanelProps } from "./props";

type OverrideKey = "cps" | "cpl" | "maxLines" | "minDurationS" | "maxDurationS" | "minGapFrames";

const OVERRIDES: { key: OverrideKey; label: string; unit: string; step: number; min: number; max: number; hint: string }[] = [
  { key: "cpl", label: "Characters per line", unit: "chars", step: 1, min: 5, max: 80, hint: "Longer lines are broken or shortened." },
  { key: "cps", label: "Reading speed", unit: "chars/s", step: 1, min: 1, max: 40, hint: "Faster lines are extended or flagged." },
  { key: "maxLines", label: "Lines per subtitle", unit: "lines", step: 1, min: 1, max: 4, hint: "More lines are split into two subtitles." },
  { key: "minDurationS", label: "Shortest subtitle", unit: "s", step: 0.1, min: 0.1, max: 5, hint: "Shorter ones are extended." },
  { key: "maxDurationS", label: "Longest subtitle", unit: "s", step: 0.5, min: 1, max: 20, hint: "Longer ones are split or trimmed." },
  { key: "minGapFrames", label: "Gap between subtitles", unit: "frames", step: 1, min: 0, max: 24, hint: "Smaller gaps are closed or widened to this." },
];

function formatRule(key: OverrideKey, style: EffectiveStyle): string {
  const value = style[key];
  return key === "minDurationS" || key === "maxDurationS" ? String(Math.round(value * 1000) / 1000) : String(value);
}

export function StylePanel({ options, onChange, disabled }: SettingsPanelProps<"style">) {
  const set = (patch: Partial<StyleOptions>) => onChange(patch);
  const previewLanguage = options.language === "auto" ? "en" : options.language;
  const preset = netflixPreview(previewLanguage, options.children, options.sdh);
  const custom = options.preset === "custom";

  return (
    <div>
      <Section>
        <Field
          group
          label="Rules"
          hint={
            custom
              ? "Starts from the Netflix rules for the language and applies the values you set below."
              : "The Netflix Timed Text Style Guide for the subtitle's language."
          }
        >
          <Segmented<StyleOptions["preset"]>
            label="Rules"
            value={options.preset}
            disabled={disabled}
            onChange={(p) => set({ preset: p })}
            choices={[
              { value: "netflix", label: "Netflix", title: "Netflix Timed Text Style Guide" },
              { value: "custom", label: "Custom", title: "Netflix rules with your overrides" },
            ]}
          />
        </Field>
        <Field label="Subtitle language" hint="Each language has its own line length and reading speed.">
          <LanguageSelect
            value={options.language}
            auto="From the subtitle"
            disabled={disabled}
            onChange={(language) => set({ language })}
          />
        </Field>
        <Field
          group
          label="When a rule is broken"
          hint={options.fix ? "Fixes what it safely can and lists the rest." : "Changes nothing; writes a report of every problem."}
        >
          <Segmented<"fix" | "report">
            label="When a rule is broken"
            value={options.fix ? "fix" : "report"}
            disabled={disabled}
            onChange={(v) => set({ fix: v === "fix" })}
            choices={[
              { value: "fix", label: "Fix", title: "Fix and report" },
              { value: "report", label: "Report Only", title: "Only report" },
            ]}
          />
        </Field>
      </Section>

      <Section title="Audience">
        <Toggle
          label="Snap to shot changes"
          hint="Moves line starts and ends onto nearby cuts in the picture. Needs the video; skipped for a subtitle on its own."
          checked={options.snapToShotChanges}
          disabled={disabled}
          onChange={(snapToShotChanges) => set({ snapToShotChanges })}
        />
        <Toggle
          label="Children's programme"
          hint="Slower reading speeds for young viewers."
          checked={options.children}
          disabled={disabled}
          onChange={(children) => set({ children })}
        />
        <Toggle
          label="For deaf and hard-of-hearing viewers (SDH)"
          hint="SDH rules: sound descriptions and speaker labels allowed, with their own reading speeds."
          checked={options.sdh}
          disabled={disabled}
          onChange={(sdh) => set({ sdh })}
        />
      </Section>

      <Section title={custom ? "Your values" : "Netflix values (preview)"}>
        <Hint>
          {custom ? "Empty uses the Netflix value, shown in grey, " : "A preview of the rules "}for{" "}
          {options.language === "auto" ? "English (the language is read from the subtitle when it runs)" : languageName(previewLanguage)}
          {preset.exact ? "" : ", which uses the general rules"}. The engine has the final say.
        </Hint>
        {custom ? (
          <div className="grid grid-cols-1 gap-2.5">
            {OVERRIDES.map((o) => (
              <Field key={o.key} label={o.label} hint={o.hint}>
                <NumberField
                  ariaLabel={o.label}
                  value={options[o.key]}
                  nullable
                  placeholder={formatRule(o.key, preset)}
                  min={o.min}
                  max={o.max}
                  step={o.step}
                  unit={o.unit}
                  disabled={disabled}
                  onChange={(v) => set({ [o.key]: v } as Partial<StyleOptions>)}
                />
              </Field>
            ))}
          </div>
        ) : (
          <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 rounded-lg bg-elevated px-2.5 py-2 text-[11.5px]">
            {OVERRIDES.map((o) => (
              <div key={o.key} className="contents">
                <dt className="text-muted-foreground">{o.label}</dt>
                <dd className="tabular text-right font-mono" data-rule={o.key}>
                  {formatRule(o.key, preset)} {o.unit}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </Section>
    </div>
  );
}
