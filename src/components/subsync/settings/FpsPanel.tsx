import { FPS_PRESETS } from "@/lib/subsync/types";
import { cx } from "@/lib/cx";

import { Field, Hint, NumberField, Section, Segmented, Select } from "./fields";
import {
  DEFAULT_TWO_POINT,
  driftOver,
  formatFps,
  formatSigned,
  formatTimestamp,
  fpsPreset,
  fpsTimeRatio,
  parseTimestamp,
  twoPointFit,
} from "./logic";
import type { SettingsPanelProps } from "./props";

/** A decimal that is deliberately not one of the exact presets, so choosing
 *  "Custom…" stays custom and shows the number box. */
const CUSTOM_SEED = 23.976;

type RateChoice = "auto" | "custom" | `${number}`;

function rateChoice(value: number | "auto"): RateChoice {
  if (value === "auto") return "auto";
  const index = FPS_PRESETS.findIndex((p) => p === fpsPreset(value));
  return index >= 0 ? (`${index}` as RateChoice) : "custom";
}

function RatePicker({
  label,
  hint,
  value,
  autoLabel,
  disabled,
  onChange,
}: {
  label: string;
  hint: string;
  value: number | "auto";
  autoLabel: string;
  disabled: boolean;
  onChange: (value: number | "auto") => void;
}) {
  const choice = rateChoice(value);
  return (
    <Field label={label} hint={hint}>
      <Select<RateChoice>
        value={choice}
        disabled={disabled}
        onChange={(v) => {
          if (v === "auto") onChange("auto");
          else if (v === "custom") onChange(value === "auto" || fpsPreset(value) ? CUSTOM_SEED : value);
          else onChange(FPS_PRESETS[Number(v)].value);
        }}
        choices={[
          { value: "auto", label: autoLabel },
          ...FPS_PRESETS.map((p, i) => ({ value: `${i}` as RateChoice, label: `${p.label} fps` })),
          { value: "custom", label: "Custom…" },
        ]}
      />
      {choice === "custom" && typeof value === "number" && (
        <div className="mt-1.5">
          <NumberField
            ariaLabel={`${label} (custom)`}
            value={value}
            min={1}
            max={1000}
            step={0.001}
            unit="fps"
            width="w-24"
            disabled={disabled}
            onChange={(v) => v !== null && v > 0 && onChange(v)}
          />
        </div>
      )}
    </Field>
  );
}

/** hh:mm:ss.mmm entry. Uncontrolled so half-typed times are not overwritten;
 *  a valid time is committed as it is typed, an invalid one marks the box. */
function TimeInput({
  seconds,
  label,
  disabled,
  onCommit,
}: {
  seconds: number;
  label: string;
  disabled: boolean;
  onCommit: (seconds: number) => void;
}) {
  return (
    <input
      type="text"
      defaultValue={formatTimestamp(seconds)}
      aria-label={label}
      placeholder="00:00:00.000"
      disabled={disabled}
      spellCheck={false}
      onChange={(event) => {
        const parsed = parseTimestamp(event.target.value);
        event.target.setCustomValidity(parsed === null || parsed < 0 ? "Use h:mm:ss.mmm, e.g. 00:01:02.500" : "");
        if (parsed !== null && parsed >= 0) onCommit(parsed);
      }}
      className={cx(
        "tabular w-full min-w-0 rounded-md border border-border-strong bg-input px-2 py-1 font-mono text-xs",
        "focus:outline-none focus:ring-2 focus:ring-ring/40 disabled:opacity-50 invalid:border-destructive",
      )}
    />
  );
}

function Preview({ ratio, offsetS }: { ratio: number; offsetS?: number }) {
  const drift = driftOver(ratio);
  const same = Math.abs(ratio - 1) < 1e-9;
  return (
    <div className="rounded-lg bg-elevated px-2.5 py-2 text-[11.5px]" aria-live="polite">
      <div className="flex justify-between gap-2">
        <span className="text-muted-foreground">Speed factor</span>
        <span className="tabular font-mono">×{ratio.toFixed(6)}</span>
      </div>
      {offsetS !== undefined && (
        <div className="flex justify-between gap-2">
          <span className="text-muted-foreground">Then shifted by</span>
          <span className="tabular font-mono">{formatSigned(offsetS, 3)} s</span>
        </div>
      )}
      <p className="mt-1 text-[10.5px] leading-snug text-muted-foreground">
        {same
          ? "No speed change: every line keeps its time."
          : `A line at the end of a 45-min episode moves by ${formatSigned(drift, 1)} s.`}
      </p>
    </div>
  );
}

export function FpsPanel({ options, onChange, disabled, videoFps }: SettingsPanelProps<"fps">) {
  const twoPoint = options.twoPoint ?? null;
  const from = options.from === "auto" ? null : options.from;
  const to = options.to === "auto" ? videoFps : options.to;
  const ratio = from !== null && to !== null ? fpsTimeRatio(from, to) : null;
  const fit = twoPoint ? twoPointFit(twoPoint.a, twoPoint.b) : null;

  const setPoint = (which: "a" | "b", index: 0 | 1, seconds: number) => {
    const base = twoPoint ?? DEFAULT_TWO_POINT;
    const pair: [number, number] = [...base[which]];
    pair[index] = seconds;
    onChange({ twoPoint: { ...base, [which]: pair } });
  };

  return (
    <div>
      <Section>
        <Field group label="Convert using">
          <Segmented<"rates" | "points">
            label="Convert using"
            value={twoPoint ? "points" : "rates"}
            disabled={disabled}
            onChange={(v) => onChange({ twoPoint: v === "points" ? (twoPoint ?? DEFAULT_TWO_POINT) : null })}
            choices={[
              { value: "rates", label: "Frame Rates", title: "Convert between two known frame rates" },
              { value: "points", label: "Two Points", title: "Match two lines to their correct times" },
            ]}
          />
        </Field>
      </Section>

      {!twoPoint && (
        <>
          <Section title="Rates">
            <RatePicker
              label="Subtitle was timed for"
              hint="The frame rate of the release the subtitle came from."
              value={options.from}
              autoLabel="From the subtitle (MicroDVD header or reference)"
              disabled={disabled}
              onChange={(from) => onChange({ from })}
            />
            <RatePicker
              label="Video plays at"
              hint="The frame rate of the video it should match."
              value={options.to}
              autoLabel={videoFps ? `From the video (${formatFps(videoFps)} fps)` : "From the video"}
              disabled={disabled}
              onChange={(to) => onChange({ to })}
            />
            <Field
              group
              label="Conversion"
              hint={
                options.mode === "time"
                  ? "Speed change: lines stretch or shrink with the video, as a PAL 25 fps speed-up needs. Use this in almost every case."
                  : "Keep frame numbers: for frame-based subtitles whose rate was labelled wrong. Line N stays on frame N."
              }
            >
              <Segmented<"time" | "frames">
                label="Conversion"
                value={options.mode}
                disabled={disabled}
                onChange={(mode) => onChange({ mode })}
                choices={[
                  { value: "time", label: "Speed Change", title: "Scale times by the ratio of the rates" },
                  { value: "frames", label: "Keep Frames", title: "Re-read frame numbers at the new rate" },
                ]}
              />
            </Field>
            {ratio !== null ? (
              <Preview ratio={ratio} />
            ) : (
              <Hint>
                The speed factor is shown once both rates are known
                {options.to === "auto" && !videoFps ? " (the video's rate is read when the job runs)" : ""}.
              </Hint>
            )}
          </Section>
          <Section>
            <Field label="Then shift by" hint="Moves every line after the rate change. Negative is earlier.">
              <NumberField
                ariaLabel="Then shift by"
                value={options.offsetMs}
                step={10}
                unit="ms"
                width="w-24"
                disabled={disabled}
                onChange={(v) => v !== null && onChange({ offsetMs: v })}
              />
            </Field>
          </Section>
        </>
      )}

      {twoPoint && (
        <Section title="Two known points">
          <Hint>
            Pick an early and a late line. Type when it appears in the subtitle and when it should appear in the video.
            Everything between is stretched to fit.
          </Hint>
          <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-x-2 gap-y-1.5 text-[11.5px]">
            <span />
            <span className="text-muted-foreground">Subtitle says</span>
            <span className="text-muted-foreground">Should be</span>
            {(["a", "b"] as const).map((which) => (
              <div key={which} className="contents">
                <span className="text-muted-foreground">{which === "a" ? "Early line" : "Late line"}</span>
                <TimeInput
                  label={`${which === "a" ? "Early" : "Late"} line, subtitle time`}
                  seconds={twoPoint[which][0]}
                  disabled={disabled}
                  onCommit={(s) => setPoint(which, 0, s)}
                />
                <TimeInput
                  label={`${which === "a" ? "Early" : "Late"} line, correct time`}
                  seconds={twoPoint[which][1]}
                  disabled={disabled}
                  onCommit={(s) => setPoint(which, 1, s)}
                />
              </div>
            ))}
          </div>
          {fit && !fit.ok && <Hint tone="destructive">{fit.error}</Hint>}
          {fit && fit.ok && (
            <>
              <Preview ratio={fit.fit.ratio} offsetS={fit.fit.offsetS} />
              {fit.warning && <Hint tone="warning">{fit.warning}</Hint>}
            </>
          )}
        </Section>
      )}
    </div>
  );
}
