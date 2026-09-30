import type { HdrSubsOptions } from "@/lib/subsync/types";

import { Field, Hint, NumberField, Panel, Segmented, Select, Slider, TextInput } from "./fields";
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
    <Panel>
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
            { value: "nits", label: "Nits", title: "A luminance in cd/m²" },
            { value: "percent", label: "Percentage", title: "A fraction of the current brightness" },
          ]}
        />
      </Field>
      {options.mode === "nits" ? (
        <>
          <Field
            inline
            plain
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
              width={110}
              disabled={disabled}
              onChange={(v) => v !== null && set({ targetNits: v })}
            />
          </Field>
          <Field
            inline
            plain
            label="Display peak"
            hint="How bright your player and TV show full-white subtitles. Used to work out how much to dim them; 1000 nits is typical."
          >
            <NumberField
              ariaLabel="Display peak"
              value={options.assumedPeakNits}
              min={100}
              max={10000}
              step={100}
              unit="nits"
              width={110}
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

      <Field label="Colour" hint="Keep leaves each subtitle's colour and only changes brightness. Grey glares least.">
        <span className="row" style={{ gap: 8 }}>
          {hex && HEX_COLOR.test(hex) && <span aria-hidden className="ss-swatch" style={{ background: hex }} />}
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
        </span>
      </Field>
      {colorChoice === "custom" && (
        <div className="row" style={{ gap: 8 }}>
          <input
            type="color"
            aria-label="Custom colour"
            className="ss-color"
            value={validHex ? options.color.toLowerCase() : CUSTOM_SEED}
            disabled={disabled}
            onChange={(event) => set({ color: event.target.value })}
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
    </Panel>
  );
}
