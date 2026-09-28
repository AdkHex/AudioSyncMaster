/** The form vocabulary of the Subsync settings panels.
 *
 *  Every field is a label, its control, and one sentence saying what the
 *  setting does. Controls follow the macOS conventions the app already uses:
 *  checkboxes for single on/off settings, a segmented control for two to five
 *  closely related choices (equal segments, noun labels, a tooltip each),
 *  pop-up menus for longer lists, radio cards for engines because each needs
 *  a description and an availability line.
 *
 *  These components hold no React state and use no hooks: labels wrap their
 *  controls instead of pointing at generated ids, and "Advanced" is a native
 *  <details>. That keeps them usable in more than one panel at once and
 *  lets the tests expand the element tree without a DOM. */

import { cx } from "@/lib/cx";
import type { EngineStatus } from "@/lib/subsync/types";

import { clamp, installAction, languageGroups } from "./logic";

export const SELECT_ARROW =
  "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23969696' stroke-width='2.5'><path d='m6 9 6 6 6-6'/></svg>\")";

const CONTROL =
  "rounded-md bg-elevated px-2 py-1.5 text-[12px] text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50";
const TEXT_INPUT =
  "rounded-md border border-border-strong bg-input px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-ring/40 disabled:opacity-50 invalid:border-destructive";

// ------------------------------------------------------------------ layout

export function Section({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3 border-b border-border py-3.5 first:pt-0 last:border-0">
      {title && <h3 className="text-[11px] font-semibold text-muted-foreground">{title}</h3>}
      {children}
    </section>
  );
}

export function Hint({ children, tone }: { children: React.ReactNode; tone?: "warning" | "destructive" }) {
  return (
    <p
      className={cx(
        "mt-1 text-[10.5px] leading-snug",
        tone === "warning"
          ? "text-warning"
          : tone === "destructive"
            ? "text-destructive"
            : "text-muted-foreground/80",
      )}
    >
      {children}
    </p>
  );
}

/** Label over control, hint under it. `group` renders a <div role=group>
 *  for controls that are button sets (a <label> would forward clicks on its
 *  text to the first button). */
export function Field({
  label,
  hint,
  group,
  children,
  extra,
}: {
  label: string;
  hint?: React.ReactNode;
  group?: boolean;
  children: React.ReactNode;
  extra?: React.ReactNode;
}) {
  const heading = <span className="mb-1 block text-[11.5px] text-muted-foreground">{label}</span>;
  return (
    <div>
      {group ? (
        <div role="group" aria-label={label}>
          {heading}
          {children}
        </div>
      ) : (
        <label className="block">
          {heading}
          {children}
        </label>
      )}
      {hint && <Hint>{hint}</Hint>}
      {extra}
    </div>
  );
}

/** Rarely needed settings, closed by default. */
export function Advanced({ children, title = "Advanced" }: { children: React.ReactNode; title?: string }) {
  return (
    <details className="group py-3.5">
      <summary className="flex cursor-pointer select-none list-none items-center gap-1.5 text-[11px] font-semibold text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
        <span aria-hidden className="inline-block transition-transform group-open:rotate-90">
          ›
        </span>
        {title}
      </summary>
      <div className="mt-3 space-y-3">{children}</div>
    </details>
  );
}

// ---------------------------------------------------------------- controls

export interface Choice<T extends string> {
  value: T;
  label: string;
  /** Tooltip; macOS shows one per segment. */
  title?: string;
  disabled?: boolean;
}

export function Segmented<T extends string>({
  label,
  value,
  choices,
  disabled,
  onChange,
}: {
  label: string;
  value: T;
  choices: Choice<T>[];
  disabled?: boolean;
  onChange: (value: T) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="grid gap-1 rounded-[9px] bg-elevated p-1"
      style={{ gridTemplateColumns: `repeat(${choices.length}, minmax(0, 1fr))` }}
    >
      {choices.map((choice) => (
        <button
          key={choice.value}
          type="button"
          role="radio"
          aria-checked={value === choice.value}
          title={choice.title}
          disabled={disabled || choice.disabled}
          onClick={() => onChange(choice.value)}
          className={cx(
            "truncate rounded-[7px] px-2 py-1 text-[12px] font-medium transition-colors",
            value === choice.value
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
            "disabled:pointer-events-none disabled:opacity-50",
          )}
        >
          {choice.label}
        </button>
      ))}
    </div>
  );
}

export function Select<T extends string>({
  value,
  choices,
  disabled,
  onChange,
  groups,
  ariaLabel,
}: {
  value: T;
  choices: Choice<T>[];
  disabled?: boolean;
  onChange: (value: T) => void;
  /** Optional <optgroup>s rendered after `choices`. */
  groups?: { label: string; choices: Choice<T>[] }[];
  ariaLabel?: string;
}) {
  const option = (choice: Choice<T>) => (
    <option key={choice.value} value={choice.value} disabled={choice.disabled} title={choice.title}>
      {choice.label}
    </option>
  );
  return (
    <select
      value={value}
      disabled={disabled}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value as T)}
      className={cx(CONTROL, "w-full appearance-none pr-6")}
      style={{ backgroundImage: SELECT_ARROW, backgroundRepeat: "no-repeat", backgroundPosition: "right 7px center" }}
    >
      {choices.map(option)}
      {groups?.map((group) => (
        <optgroup key={group.label} label={group.label}>
          {group.choices.map(option)}
        </optgroup>
      ))}
    </select>
  );
}

export function LanguageSelect({
  value,
  onChange,
  disabled,
  auto,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  /** Label for an "auto" first option, e.g. "Detect from the audio". */
  auto?: string;
  ariaLabel?: string;
}) {
  const { common, all } = languageGroups();
  const known = value === "auto" || all.some(([code]) => code === value);
  return (
    <Select
      value={value}
      disabled={disabled}
      ariaLabel={ariaLabel}
      onChange={onChange}
      choices={[
        ...(auto ? [{ value: "auto", label: auto }] : []),
        // Keep an unknown code visible rather than silently showing another language.
        ...(known ? [] : [{ value, label: value }]),
      ]}
      groups={[
        { label: "Common", choices: common.map(([code, name]) => ({ value: code, label: name })) },
        { label: "All languages", choices: all.map(([code, name]) => ({ value: code, label: name })) },
      ]}
    />
  );
}

/** A number with a unit. Out-of-range values are allowed while typing and
 *  pulled into range when the field loses focus. `nullable` fields treat an
 *  empty box as null ("use the preset"). */
export function NumberField({
  value,
  onChange,
  min,
  max,
  step = 1,
  unit,
  disabled,
  placeholder,
  nullable,
  ariaLabel,
  width = "w-20",
}: {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  disabled?: boolean;
  placeholder?: string;
  nullable?: boolean;
  ariaLabel?: string;
  width?: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <input
        type="number"
        inputMode="decimal"
        value={value === null ? "" : value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        placeholder={placeholder}
        aria-label={ariaLabel}
        onChange={(event) => {
          const raw = event.target.value;
          if (raw === "") {
            if (nullable) onChange(null);
            return;
          }
          const n = Number(raw);
          if (Number.isFinite(n)) onChange(n);
        }}
        onBlur={() => {
          if (value === null) return;
          const fixed = clamp(value, min, max);
          if (fixed !== value) onChange(fixed);
        }}
        className={cx(TEXT_INPUT, "tabular font-mono", width)}
      />
      {unit && <span className="text-[11.5px] text-muted-foreground">{unit}</span>}
    </span>
  );
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  display,
  disabled,
  ariaLabel,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  display: string;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  return (
    <span className="flex items-center gap-2.5">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        aria-valuetext={display}
        onChange={(event) => onChange(Number(event.target.value))}
        className="min-w-0 flex-1 cursor-pointer disabled:cursor-default"
      />
      <span className="tabular w-11 shrink-0 text-right font-mono text-xs font-semibold">{display}</span>
    </span>
  );
}

/** A single on/off setting: a checkbox with its title and one-line hint. */
export function Toggle({
  label,
  hint,
  checked,
  onChange,
  disabled,
  extra,
}: {
  label: string;
  hint?: React.ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  extra?: React.ReactNode;
}) {
  return (
    <div>
      <label className={cx("flex items-start gap-2 text-[12px]", disabled ? "opacity-60" : "cursor-pointer")}>
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
          className="mt-[2px] h-[15px] w-[15px] shrink-0 accent-primary"
        />
        <span className="min-w-0">
          {label}
          {hint && <span className="block text-[10.5px] leading-snug text-muted-foreground/80">{hint}</span>}
        </span>
      </label>
      {extra && <div className="ml-[23px]">{extra}</div>}
    </div>
  );
}

export function TextInput({
  value,
  onChange,
  placeholder,
  disabled,
  list,
  mono,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  list?: string;
  mono?: boolean;
  ariaLabel?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      list={list}
      aria-label={ariaLabel}
      spellCheck={false}
      onChange={(event) => onChange(event.target.value)}
      className={cx(TEXT_INPUT, "w-full", mono && "font-mono")}
    />
  );
}

/** A text input with suggestions that still accepts anything typed. */
export function ComboInput({
  id,
  value,
  onChange,
  suggestions,
  placeholder,
  disabled,
  ariaLabel,
}: {
  /** Page-unique id for the <datalist>. */
  id: string;
  value: string;
  onChange: (value: string) => void;
  suggestions: string[];
  placeholder?: string;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  return (
    <>
      <TextInput
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        list={id}
        mono
        ariaLabel={ariaLabel}
      />
      <datalist id={id}>
        {suggestions.map((s) => (
          <option key={s} value={s} />
        ))}
      </datalist>
    </>
  );
}

export function TextArea({
  value,
  onChange,
  placeholder,
  disabled,
  rows = 3,
  mono,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  rows?: number;
  mono?: boolean;
  ariaLabel?: string;
}) {
  return (
    <textarea
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      rows={rows}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value)}
      className={cx(TEXT_INPUT, "block w-full resize-y py-1.5 leading-relaxed", mono && "font-mono")}
    />
  );
}

// ---------------------------------------------------------------- engines

export interface EngineOption<T extends string> {
  id: T;
  label: string;
  /** One line: what it is good at. */
  description: string;
  /** What it needs from the user, shown when selected. */
  needs?: string;
}

/** Radio cards, one per engine. An engine caps reports as unavailable is
 *  disabled, says why, and offers an Install (or Add API key) button that
 *  opens the Engines panel at the right place. While caps load, every engine
 *  is selectable and shows "Checking…". */
export function EnginePicker<T extends string>({
  label,
  value,
  options,
  statuses,
  loading,
  disabled,
  keyed,
  onChange,
  onInstallPack,
}: {
  label: string;
  value: T;
  options: EngineOption<T>[];
  /** Status per option id (null = not reported). */
  statuses: Record<string, EngineStatus | null>;
  loading: boolean;
  disabled: boolean;
  /** Ids whose missing requirement is an API key. */
  keyed?: readonly string[];
  onChange: (value: T) => void;
  onInstallPack: (packId: string) => void;
}) {
  const selected = statuses[value] ?? null;
  return (
    <div>
      <div role="radiogroup" aria-label={label} className="space-y-1.5">
        {options.map((option) => {
          const status = statuses[option.id] ?? null;
          const unavailable = status !== null && !status.available;
          const checked = value === option.id;
          const action = installAction(status, keyed?.includes(option.id));
          return (
            <div
              key={option.id}
              className={cx(
                "rounded-lg border px-2.5 py-2 transition-colors",
                checked ? "border-primary/50 bg-accent/60" : "border-border bg-card",
                unavailable && !checked && "bg-sunken/60",
              )}
            >
              <button
                type="button"
                role="radio"
                aria-checked={checked}
                aria-disabled={unavailable || undefined}
                disabled={disabled || (unavailable && !checked)}
                onClick={() => onChange(option.id)}
                data-engine={option.id}
                className="flex w-full items-start gap-2 text-left disabled:cursor-default"
              >
                <span
                  aria-hidden
                  className={cx(
                    "mt-[3px] grid h-[13px] w-[13px] shrink-0 place-items-center rounded-full border",
                    checked ? "border-primary bg-primary" : "border-border-strong bg-background",
                  )}
                >
                  {checked && <span className="h-[5px] w-[5px] rounded-full bg-primary-foreground" />}
                </span>
                <span className={cx("min-w-0 flex-1", unavailable && "opacity-70")}>
                  <span className="block text-[12px] font-medium">{option.label}</span>
                  <span className="block text-[10.5px] leading-snug text-muted-foreground">
                    {option.description}
                  </span>
                  {checked && option.needs && (
                    <span className="mt-0.5 block text-[10.5px] leading-snug text-muted-foreground/80">
                      Needs: {option.needs}
                    </span>
                  )}
                </span>
                {loading && <span className="shrink-0 text-[10.5px] text-muted-foreground">Checking…</span>}
              </button>
              {unavailable && (
                <div className="ml-[21px] mt-1 flex items-center gap-2">
                  <span className="min-w-0 flex-1 text-[10.5px] leading-snug text-warning">
                    {status.reason || "Not available on this system."}
                  </span>
                  {action && (
                    <button
                      type="button"
                      onClick={() => onInstallPack(action.target)}
                      className="shrink-0 rounded-md border border-border-strong bg-card px-2 py-0.5 text-[11px] font-medium hover:bg-secondary"
                    >
                      {action.label}
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
      {selected && !selected.available && (
        <Hint tone="warning">The selected engine is not available yet; the run will fail until it is.</Hint>
      )}
    </div>
  );
}
