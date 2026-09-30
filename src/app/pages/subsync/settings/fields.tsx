/** The form vocabulary of the Subsync settings panels, in the look of the
 *  approved mockup (design/mockup/screens/subs.tsx): a label in the tertiary
 *  colour over its control, a small hint under it, switches for on/off
 *  settings, segmented controls for a few choices, combo boxes for lists,
 *  and a "More options" expander for what is rarely needed.
 *
 *  The controls are the native elements (select, input, textarea, and
 *  buttons with role="radio"), restyled with the UI kit's classes, so they
 *  keep the platform's keyboard and lists.
 *
 *  These components hold no React state and use no hooks: labels wrap their
 *  controls instead of pointing at generated ids, and "More options" is a
 *  native <details>. That keeps them usable in more than one panel at once
 *  and lets the tests expand the element tree without a DOM. */

import { ChevronDownRegular, ChevronRightRegular } from "@fluentui/react-icons";
import type { ReactNode } from "react";

import type { EngineStatus, PackStatus } from "@/lib/subsync/types";
import { cx } from "@/ui/kit";

import { clamp, formatBytes, installAction, languageGroups } from "./logic";

// ------------------------------------------------------------------ layout

/** A panel's column of fields, 16 px apart as in the mockup. */
export function Panel({ children }: { children: ReactNode }) {
  return <div className="ss-panel">{children}</div>;
}

/** A titled group of fields (the mockup's "Timing", "Text", "Output"). */
export function Section({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div className="ss-sec">
      {title && <span className="sec">{title}</span>}
      {children}
    </div>
  );
}

/** Two fields side by side (From / To, Resolution / Curve). */
export function Pair({ children }: { children: ReactNode }) {
  return <div className="ss-pair">{children}</div>;
}

export function Hint({ children, tone }: { children: ReactNode; tone?: "warning" | "destructive" }) {
  return (
    <span className={cx("ss-hint sm", tone === "warning" ? "warn" : tone === "destructive" ? "bad" : "t3")}>
      {children}
    </span>
  );
}

/** An accent text link inside a hint: Install, Add API key, Engines. */
export function HintLink({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return (
    <button type="button" className="ss-link" onClick={onClick}>
      {children}
    </button>
  );
}

/** Label over control. `group` renders a <div role=group> for controls
 *  that are button sets (a <label> would forward clicks on its text to the
 *  first button). `inline` puts the label and a small control on one row, as
 *  the mockup does for numbers. A text hint is the field's tooltip, as the
 *  mockup shows none; a hint with a link (Install, Choose…) shows under it. */
export function Field({
  label,
  hint,
  group,
  inline,
  plain,
  children,
  extra,
}: {
  label: string;
  hint?: ReactNode;
  group?: boolean;
  inline?: boolean;
  /** The label in the text colour, as the mockup has HDR's two levels. */
  plain?: boolean;
  children: ReactNode;
  extra?: ReactNode;
}) {
  const asTitle = typeof hint === "string";
  const title = asTitle ? (hint as string) : undefined;
  const heading = <span className={cx(inline && "grow", !plain && "t3")}>{label}</span>;
  const layout = inline ? "row ss-inline" : "ss-fld";
  const body = group ? (
    <div role="group" aria-label={label} className={layout} title={title}>
      {heading}
      {children}
    </div>
  ) : (
    <label className={layout} title={title}>
      {heading}
      {children}
    </label>
  );
  const shownHint = hint && !asTitle ? hint : null;
  if (!shownHint && !extra) return body;
  return (
    <div className="ss-fld">
      {body}
      {shownHint && <Hint>{shownHint}</Hint>}
      {extra}
    </div>
  );
}

/** Rarely needed settings, closed by default; `open` starts it open, for
 *  settings already changed from their defaults. */
export function Advanced({ children, title = "More options", open }: { children: ReactNode; title?: string; open?: boolean }) {
  return (
    <details className="ss-more" open={open || undefined}>
      <summary className="row t2">
        <span className="ss-chev" aria-hidden>
          <ChevronRightRegular />
        </span>
        {title}
      </summary>
      <div className="ss-more-b">{children}</div>
    </details>
  );
}

// ---------------------------------------------------------------- controls

export interface Choice<T extends string> {
  value: T;
  label: string;
  /** Tooltip, one per segment. */
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
    <span className="seg ss-seg" role="radiogroup" aria-label={label}>
      {choices.map((choice) => (
        <button
          key={choice.value}
          type="button"
          role="radio"
          aria-checked={value === choice.value}
          title={choice.title}
          disabled={disabled || choice.disabled}
          onClick={() => onChange(choice.value)}
          className={cx(value === choice.value && "on")}
        >
          {choice.label}
        </button>
      ))}
    </span>
  );
}

/** The combo box: a native <select> drawn as the mockup's combo. */
export function Select<T extends string>({
  value,
  choices,
  disabled,
  onChange,
  groups,
  ariaLabel,
  width = "100%",
}: {
  value: T;
  choices: Choice<T>[];
  disabled?: boolean;
  onChange: (value: T) => void;
  /** Optional <optgroup>s rendered after `choices`. */
  groups?: { label: string; choices: Choice<T>[] }[];
  ariaLabel?: string;
  width?: number | string;
}) {
  const option = (choice: Choice<T>) => (
    <option key={choice.value} value={choice.value} disabled={choice.disabled} title={choice.title}>
      {choice.label}
    </option>
  );
  return (
    <span className={cx("combo", disabled && "disabled")} style={{ width }}>
      <select
        className="combo-native"
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(event) => onChange(event.target.value as T)}
      >
        {choices.map(option)}
        {groups?.map((group) => (
          <optgroup key={group.label} label={group.label}>
            {group.choices.map(option)}
          </optgroup>
        ))}
      </select>
      <span className="ic" aria-hidden>
        <ChevronDownRegular />
      </span>
    </span>
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
  /** Label for an "auto" first option, e.g. "Detect". */
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

/** A number in a text box, with its unit inside the box. Out-of-range
 *  values are allowed while typing and pulled into range when the field
 *  loses focus. `nullable` fields treat an empty box as null ("use the
 *  preset"). */
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
  width = 110,
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
  width?: number | string;
}) {
  return (
    <span className="tbox ss-num" style={{ width }}>
      <input
        type="number"
        inputMode="decimal"
        className="num"
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
      />
      {unit && <span className="u">{unit}</span>}
    </span>
  );
}

/** The mockup's slider (rail, fill, thumb) over a native range input, with
 *  the value beside it. */
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
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <span className="row ss-sldrow">
      <span className={cx("sld", disabled && "disabled")}>
        <span className="r" />
        <span className="f" style={{ width: `${pct}%` }} />
        <span className="t" style={{ left: `${pct}%` }} />
        <input
          type="range"
          className="sld-native"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          aria-label={ariaLabel}
          aria-valuetext={display}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </span>
      <span className="num ss-sldv">{display}</span>
    </span>
  );
}

/** A single on/off setting: its name and a switch, on one row. The switch
 *  is a native checkbox (role="switch") under the mockup's toggle; the hint
 *  is the row's tooltip. */
export function Toggle({
  label,
  hint,
  checked,
  onChange,
  disabled,
  extra,
}: {
  label: string;
  hint?: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  extra?: ReactNode;
}) {
  const title = typeof hint === "string" ? hint : undefined;
  const row = (
    <label className={cx("row ss-tgl", disabled && "disabled")} title={title}>
      <span className="grow">{label}</span>
      <span className={cx("tgl", checked && "on")}>
        <input
          type="checkbox"
          role="switch"
          className="tgl-native"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="tr">
          <span className="kn" />
        </span>
      </span>
    </label>
  );
  const shownHint = hint && !title ? hint : null;
  if (!extra && !shownHint) return row;
  return (
    <div className="ss-fld">
      {row}
      {shownHint && <Hint>{shownHint}</Hint>}
      {extra}
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
    <span className="tbox" style={{ width: "100%" }}>
      <input
        type="text"
        className={cx(mono && "mono")}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        list={list}
        aria-label={ariaLabel}
        spellCheck={false}
        onChange={(event) => onChange(event.target.value)}
      />
    </span>
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
    <span className="tbox area" style={{ width: "100%" }}>
      <textarea
        className={cx(mono && "mono")}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        rows={rows}
        aria-label={ariaLabel}
        onChange={(event) => onChange(event.target.value)}
      />
    </span>
  );
}

// ---------------------------------------------------------------- engines

export interface EngineOption<T extends string> {
  id: T;
  label: string;
  /** One line: what it is good at. */
  description: string;
  /** What it needs from the user. */
  needs?: string;
}

/** What an engine lacks when a pack provides it, as the mockup says it:
 *  " needs Whisper (1.6 GB)." -- the name in the pack's brackets, when it
 *  has them. */
function needsText(status: EngineStatus, packs: readonly PackStatus[] | undefined): string | null {
  const pack = status.pack ? packs?.find((p) => p.id === status.pack) : undefined;
  if (!pack || pack.installed) return null;
  const name = /\(([^)]+)\)/.exec(pack.label)?.[1] ?? pack.label;
  const size = formatBytes(pack.downloadBytes);
  return ` needs ${name}${size ? ` (${size})` : ""}.`;
}

/** The engine combo box. An engine the capabilities report as unavailable
 *  is disabled in the list (unless it is the one chosen). The hint under it
 *  says why the chosen engine cannot run, or which others need something,
 *  with a link (Install, Add API key) that opens Preferences › Subtitles at
 *  the right place. While capabilities load, every engine is selectable. */
export function EnginePicker<T extends string>({
  title = "Engine",
  label,
  value,
  options,
  statuses,
  loading,
  disabled,
  keyed,
  packs,
  detail,
  onChange,
  onInstallPack,
}: {
  /** The visible label ("Engine", "Match against"). */
  title?: string;
  /** The combo's accessible name ("OCR engine"). */
  label: string;
  value: T;
  options: EngineOption<T>[];
  /** Status per option id (null = not reported). */
  statuses: Record<string, EngineStatus | null>;
  loading: boolean;
  disabled: boolean;
  /** Ids whose missing requirement is an API key. */
  keyed?: readonly string[];
  /** The packs, to name what an engine needs and its download. */
  packs?: readonly PackStatus[];
  /** Said after the chosen engine's name, as the mockup's "Claude ·
   *  claude-opus-5-5": its model. */
  detail?: string | null;
  onChange: (value: T) => void;
  onInstallPack: (packId: string) => void;
}) {
  const statusOf = (id: string) => statuses[id] ?? null;
  const selected = statusOf(value);

  let hint: ReactNode;
  if (loading) {
    hint = "Checking what is installed…";
  } else if (selected && !selected.available) {
    const action = installAction(selected, keyed?.includes(value));
    hint = (
      <>
        <span className="warn">{selected.reason || "Not available on this system."}</span>
        {action && (
          <>
            {" "}
            <HintLink onClick={() => onInstallPack(action.target)}>{action.label}</HintLink>
          </>
        )}
      </>
    );
  } else {
    // Other engines that need a pack, one line per pack, with its Install;
    // one the system lacks says so. A key is asked for only by the engine
    // chosen, and a pack this platform cannot run is not offered.
    const needs = new Map<string, { labels: string[]; reason: string; action: { label: string; target: string } | null }>();
    for (const option of options) {
      if (option.id === value) continue;
      const status = statusOf(option.id);
      if (!status || status.available) continue;
      const only = !status.pack && !keyed?.includes(option.id) ? /^only on (.+?)\.?$/i.exec(status.reason ?? "") : null;
      if (only) {
        needs.set(option.id, { labels: [option.label], reason: ` is only on ${only[1]}.`, action: null });
        continue;
      }
      const action = installAction(status);
      if (!action || packs?.find((p) => p.id === status.pack)?.supported === false) continue;
      const key = `${action.target}\n${status.reason ?? ""}`;
      const entry = needs.get(key);
      if (entry) entry.labels.push(option.label);
      else needs.set(key, { labels: [option.label], reason: needsText(status, packs) ?? `: ${status.reason || "Not installed."}`, action });
    }
    // Nothing else is said: each engine's description is its tooltip in the list.
    const lines = [...needs.values()].slice(0, 2);
    hint = lines.length
      ? lines.map((line, index) => (
          <span key={index} className="ss-hintline">
            {line.labels.join(", ")}
            {line.reason}
            {line.action && (
              <>
                {" "}
                <HintLink onClick={() => onInstallPack(line.action!.target)}>{line.action.label}</HintLink>
              </>
            )}
          </span>
        ))
      : null;
  }

  return (
    <div className="ss-fld">
      <label className="ss-fld">
        <span className="t3">{title}</span>
        <span className={cx("combo", disabled && "disabled")} style={{ width: "100%" }}>
          <select
            className="combo-native"
            aria-label={label}
            value={value}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value as T)}
          >
            {options.map((option) => {
              const status = statusOf(option.id);
              const off = status !== null && !status.available;
              return (
                <option
                  key={option.id}
                  value={option.id}
                  disabled={off && option.id !== value}
                  data-engine={option.id}
                  title={[option.description, off ? status?.reason : null].filter(Boolean).join(" ")}
                >
                  {option.id === value && detail ? `${option.label} · ${detail}` : option.label}
                </option>
              );
            })}
          </select>
          <span className="ic" aria-hidden>
            <ChevronDownRegular />
          </span>
        </span>
      </label>
      {hint && <span className="ss-hint sm t3">{hint}</span>}
    </div>
  );
}
