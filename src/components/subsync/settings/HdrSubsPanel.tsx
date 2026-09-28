import type { HdrSubsOptions } from "@/lib/subsync/types";

import { Field, Hint, NumberField, Section, Segmented, Select, Slider, TextInput } from "./fields";
import { HEX_COLOR, isNamedColor } from "./logic";
import type { SettingsPanelProps } from "./props";

type ColorChoice = "keep" | "white" | "gray" | "yellow" | "custom";

const SWATCH: Record<Exclude<ColorChoice, "keep" | "custom">, string> = {
  white: "#ffffff",
  gray: "#c8c8c8",
  yellow: "#f5e25b",
};

/** A starting colour for "Custom…" that is not one of the named ones. */
const CUSTOM_SEED = "#e6e6e6";

export function HdrSubsPanel({ options, onChange, disabled }: SettingsPanelProps<"hdrSubs">) {
  const set = (patch: Partial<HdrSubsOptions>) => onChange(patch);
  const colorChoice: ColorChoice = isNamedColor(options.color) ? options.color : "custom";
  const hex = colorChoice === "custom" ? options.color : colorChoice === "keep" ? null : SWATCH[colorChoice];
  const validHex = HEX_COLOR.test(options.color);

  return (
    <div>
      <Section title="Brightness">
        <Field
          group
          label="Set brightness by"
          hint={
            options.mode === "nits"
              ? "Aims subtitle white at a fixed luminance, the same on every video."
              : "Scales the subtitles' own brightness down by a percentage."
          }
        >
          <Segmented<HdrSubsOptions["mode"]>
            label="Set brightness by"
            value={options.mode}
            disabled={disabled}
            onChange={(mode) => set({ mode })}
            choices={[
              { value: "nits", label: "Target Nits", title: "A luminance in cd/m²" },
              { value: "percent", label: "Percentage", title: "A fraction of the current brightness" },
            ]}
          />
        </Field>
        {options.mode === "nits" ? (
          <>
            <Field
              label="Subtitle white"
              hint="203 nits is the BT.2408 reference white: where HDR graphics are meant to sit, about as bright as paper white. Lower is dimmer; 100–150 suits a dark room."
            >
              <NumberField
                ariaLabel="Subtitle white"
                value={options.targetNits}
                min={20}
                max={1000}
                step={1}
                unit="nits"
                disabled={disabled}
                onChange={(v) => v !== null && set({ targetNits: v })}
              />
            </Field>
            <Field
              label="Player shows full white at"
              hint="How bright your player and TV show full-white subtitles. Used to work out how much to dim them; 1000 nits is typical."
            >
              <NumberField
                ariaLabel="Player shows full white at"
                value={options.assumedPeakNits}
                min={100}
                max={10000}
                step={100}
                unit="nits"
                disabled={disabled}
                onChange={(v) => v !== null && set({ assumedPeakNits: v })}
              />
            </Field>
          </>
        ) : (
          <Field label="Brightness" hint="Percentage of the subtitles' current brightness. 50–70% stops most glare.">
            <Slider
              ariaLabel="Brightness"
              min={10}
              max={100}
              step={5}
              value={options.brightnessPercent}
              display={`${options.brightnessPercent}%`}
              disabled={disabled}
              onChange={(brightnessPercent) => set({ brightnessPercent })}
            />
          </Field>
        )}
      </Section>

      <Section title="Colour and format">
        <Field label="Colour" hint="Keep leaves each subtitle's colour and only changes brightness. Grey glares least.">
          <div className="flex items-center gap-2">
            <span
              aria-hidden
              className="h-[22px] w-[22px] shrink-0 rounded-md border border-border-strong"
              style={
                hex && HEX_COLOR.test(hex)
                  ? { background: hex }
                  : { background: "linear-gradient(135deg, #fff 0 45%, #f5e25b 55% 100%)" }
              }
            />
            <Select<ColorChoice>
              value={colorChoice}
              disabled={disabled}
              onChange={(c) => set({ color: c === "custom" ? (validHex ? options.color : CUSTOM_SEED) : c })}
              choices={[
                { value: "keep", label: "Keep the original colours" },
                { value: "white", label: "White" },
                { value: "gray", label: "Grey" },
                { value: "yellow", label: "Yellow" },
                { value: "custom", label: "Custom…" },
              ]}
            />
          </div>
        </Field>
        {colorChoice === "custom" && (
          <div className="flex items-center gap-2">
            <input
              type="color"
              aria-label="Custom colour"
              value={validHex ? options.color.toLowerCase() : CUSTOM_SEED}
              disabled={disabled}
              onChange={(event) => set({ color: event.target.value })}
              className="h-[26px] w-[34px] shrink-0 cursor-pointer rounded border border-border-strong bg-transparent p-0.5"
            />
            <TextInput
              ariaLabel="Custom colour hex"
              value={options.color}
              mono
              placeholder="#e6e6e6"
              disabled={disabled}
              onChange={(color) => set({ color: color.trim() })}
            />
          </div>
        )}
        {colorChoice === "custom" && !validHex && <Hint tone="destructive">Use a hex colour like #e6e6e6.</Hint>}
        <Field label="Write as" hint="Same keeps the input's format: an image (PGS) subtitle stays an image, text stays text.">
          <Select<HdrSubsOptions["outputFormat"]>
            value={options.outputFormat}
            disabled={disabled}
            onChange={(outputFormat) => set({ outputFormat })}
            choices={[
              { value: "same", label: "Same format as the input" },
              { value: "ass", label: "ASS (styled text)" },
              { value: "sup", label: "SUP (Blu-ray image subtitles)" },
            ]}
          />
        </Field>
      </Section>
    </div>
  );
}
