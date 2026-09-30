/** The UI kit: every control the app draws, in the look of the approved
 *  mockup (design/mockup/kit.tsx), but real — selects select, boxes type,
 *  switches switch, and all of them take keyboard focus.
 *
 *  Styling lives in ui.css; these components only pick classes. */

import {
  CheckmarkCircleFilled,
  CheckmarkFilled,
  ChevronDownRegular,
  ErrorCircleFilled,
  InfoFilled,
  WarningFilled,
} from "@fluentui/react-icons";
import {
  forwardRef,
  type ButtonHTMLAttributes,
  type CSSProperties,
  type InputHTMLAttributes,
  type ReactNode,
} from "react";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

type Fn = () => void;

/* --------------------------------- buttons --------------------------------- */

interface BtnProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  children?: ReactNode;
  accent?: boolean;
  icon?: ReactNode;
  kbd?: string;
}

/** A standard or accent (primary) button. */
export const Btn = forwardRef<HTMLButtonElement, BtnProps>(function Btn(
  { children, accent, icon, kbd, className, type = "button", ...rest },
  ref,
) {
  return (
    <button ref={ref} type={type} className={cx("btn", accent && "accent", className)} {...rest}>
      {icon && <span className="ic" aria-hidden>{icon}</span>}
      {children}
      {kbd && <span className="kbd">{kbd}</span>}
    </button>
  );
});

interface CmdProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  icon: ReactNode;
  children?: ReactNode;
  sm?: boolean;
}

/** A command-bar button: icon and label, no chrome until hovered. An icon-only
 *  command needs a title, which doubles as its accessible name. */
export const Cmd = forwardRef<HTMLButtonElement, CmdProps>(function Cmd(
  { icon, children, sm, className, title, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      title={title}
      aria-label={children ? undefined : title}
      className={cx("cmd", !children && "icon", sm && "sm", className)}
      {...rest}
    >
      <span className="ic" aria-hidden>{icon}</span>
      {children && <span className="lbl">{children}</span>}
    </button>
  );
});

/* --------------------------------- inputs --------------------------------- */

export interface Option<T extends string | number = string> {
  value: T;
  label: string;
  disabled?: boolean;
  /** A heading the option sits under in the list; runs of the same heading
   *  are grouped. */
  group?: string;
}

/** A ComboBox: the native select, drawn as the mockup's combo so the list it
 *  opens is the platform's own. */
export function Combo<T extends string | number>({
  value,
  options,
  onChange,
  w,
  sm,
  ghost,
  placeholder,
  disabled,
  label,
  style,
}: {
  value: T | null;
  options: Option<T>[];
  onChange?: (value: T) => void;
  w?: number | string;
  sm?: boolean;
  ghost?: boolean;
  placeholder?: string;
  disabled?: boolean;
  /** Accessible name when there is no visible label. */
  label?: string;
  style?: CSSProperties;
}) {
  const numeric = options.length > 0 && typeof options[0].value === "number";
  return (
    <span
      className={cx("combo", sm && "sm", ghost && "ghost", value === null && placeholder && "placeholder", disabled && "disabled")}
      style={{ width: w, ...style }}
    >
      <select
        className="combo-native"
        aria-label={label}
        disabled={disabled}
        value={value === null ? "" : String(value)}
        onChange={(event) => {
          const raw = event.target.value;
          onChange?.((numeric ? Number(raw) : raw) as T);
        }}
      >
        {value === null && placeholder && (
          <option value="" disabled>
            {placeholder}
          </option>
        )}
        {groupRuns(options).map(({ group, items }, i) => {
          const list = items.map((o) => (
            <option key={String(o.value)} value={String(o.value)} disabled={o.disabled}>
              {o.label}
            </option>
          ));
          return group ? <optgroup key={`${group}-${i}`} label={group}>{list}</optgroup> : list;
        })}
      </select>
      <span className="ic" aria-hidden><ChevronDownRegular /></span>
    </span>
  );
}

/** Consecutive options under the same heading, in order. */
function groupRuns<T extends string | number>(options: Option<T>[]): { group?: string; items: Option<T>[] }[] {
  const runs: { group?: string; items: Option<T>[] }[] = [];
  for (const o of options) {
    const last = runs[runs.length - 1];
    if (last && last.group === o.group) last.items.push(o);
    else runs.push({ group: o.group, items: [o] });
  }
  return runs;
}

interface TBoxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "value"> {
  value: string;
  onChange?: (value: string) => void;
  unit?: string;
  w?: number | string;
  mono?: boolean;
  /** Accessible name when there is no visible label. */
  label?: string;
}

/** A TextBox, with an optional unit after the value. */
export const TBox = forwardRef<HTMLInputElement, TBoxProps>(function TBox(
  { value, onChange, unit, w, mono, label, className, style, ...rest },
  ref,
) {
  return (
    <label className={cx("tbox", className)} style={{ width: w, ...style }}>
      <input
        ref={ref}
        className={cx(mono && "mono")}
        aria-label={label}
        value={value}
        onChange={(event) => onChange?.(event.target.value)}
        {...rest}
      />
      {unit && <span className="u">{unit}</span>}
    </label>
  );
});

/** A multi-line TextBox. */
export function TArea({
  value,
  onChange,
  placeholder,
  rows = 3,
  mono,
  label,
  disabled,
}: {
  value: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  rows?: number;
  mono?: boolean;
  label?: string;
  disabled?: boolean;
}) {
  return (
    <label className="tbox area">
      <textarea
        className={cx(mono && "mono")}
        aria-label={label}
        rows={rows}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => onChange?.(event.target.value)}
      />
    </label>
  );
}

/** A ToggleSwitch. With `label`, the state is spelt out beside it (On / Off),
 *  as Windows Settings does. */
export function Toggle({
  on,
  onChange,
  label,
  name,
  disabled,
}: {
  on: boolean;
  onChange?: (on: boolean) => void;
  label?: boolean;
  /** Accessible name. */
  name?: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={name}
      disabled={disabled}
      className={cx("tgl", on && "on")}
      onClick={() => onChange?.(!on)}
    >
      {label && <span style={{ minWidth: 22 }}>{on ? "On" : "Off"}</span>}
      <span className="tr"><span className="kn" /></span>
    </button>
  );
}

/** A CheckBox. */
export function Chk({
  on,
  onChange,
  name,
  disabled,
  children,
}: {
  on: boolean;
  onChange?: (on: boolean) => void;
  name?: string;
  disabled?: boolean;
  children?: ReactNode;
}) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={on}
      aria-label={children ? undefined : name}
      disabled={disabled}
      className="chkrow"
      onClick={(event) => {
        event.stopPropagation();
        onChange?.(!on);
      }}
    >
      <span className={cx("chk", on && "on")}>{on && <CheckmarkFilled />}</span>
      {children}
    </button>
  );
}

export const Radio = ({ on }: { on: boolean }) => <span className={cx("rad", on && "on")} />;

/** A segmented control: one of a few mutually exclusive choices. */
export function Seg<T extends string | number>({
  items,
  value,
  onChange,
  label,
  disabled,
}: {
  items: Option<T>[];
  value: T;
  onChange?: (value: T) => void;
  label?: string;
  disabled?: boolean;
}) {
  return (
    <span className="seg" role="radiogroup" aria-label={label}>
      {items.map((item) => (
        <button
          key={String(item.value)}
          type="button"
          role="radio"
          aria-checked={item.value === value}
          disabled={disabled || item.disabled}
          className={cx(item.value === value && "on")}
          onClick={() => onChange?.(item.value)}
        >
          {item.label}
        </button>
      ))}
    </span>
  );
}

/** A Slider: the mockup's rail, fill and thumb, driven by a native range
 *  input laid over them, which supplies dragging and the keyboard. */
export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  label,
  disabled,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange?: (value: number) => void;
  label?: string;
  disabled?: boolean;
}) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <span className={cx("sld", disabled && "disabled")}>
      <span className="r" />
      <span className="f" style={{ width: `${pct}%` }} />
      <span className="t" style={{ left: `${pct}%` }} />
      <input
        type="range"
        className="sld-native"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange?.(Number(event.target.value))}
      />
    </span>
  );
}

/* -------------------------------- progress -------------------------------- */

export const PBar = ({ pct, ind, w }: { pct?: number | null; ind?: boolean; w?: number | string }) => (
  <span
    className={cx("pbar", ind && "ind")}
    style={{ width: w ?? "100%", display: "block" }}
    role="progressbar"
    aria-valuenow={ind || pct == null ? undefined : Math.round(pct)}
    aria-valuemin={0}
    aria-valuemax={100}
  >
    <i style={ind ? undefined : { width: `${Math.max(0, Math.min(100, pct ?? 0))}%` }} />
  </span>
);

/** Fluent ProgressRing (indeterminate). */
export const Ring = ({ size = 16 }: { size?: number }) => (
  <svg className="ring" width={size} height={size} viewBox="0 0 16 16" aria-hidden>
    <circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--ac)" strokeWidth="1.6" strokeDasharray="18 40" strokeLinecap="round" />
  </svg>
);

/** Confidence colour: High ≥ 75, Medium ≥ 50, Low below. */
export const toneOf = (c: number) => (c >= 75 ? "var(--ok)" : c >= 50 ? "var(--warn)" : "var(--bad)");

export const Meter = ({ pct }: { pct: number }) => (
  <span className="meter" aria-hidden>
    <i style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: toneOf(pct) }} />
  </span>
);

export type St = "ready" | "wait" | "run" | "ok" | "warn" | "bad" | "written" | "writing";

/** One status vocabulary for every queue in the app. */
export function Status({ s, text, pct }: { s: St; text?: string; pct?: number | null }) {
  if (s === "run" || s === "writing")
    return (
      <span className="cell" style={{ width: "100%" }}>
        <PBar pct={pct ?? 0} ind={pct == null} w={64} />
        <span className="sm t2 num truncate">{text ?? `${Math.round(pct ?? 0)}%`}</span>
      </span>
    );
  const icon =
    s === "ok" || s === "written" ? (
      <CheckmarkCircleFilled className="ok" />
    ) : s === "warn" ? (
      <WarningFilled className="warn" />
    ) : s === "bad" ? (
      <ErrorCircleFilled className="bad" />
    ) : null;
  return (
    <span className={cx("cell", (s === "ready" || s === "wait") && "t3")}>
      {icon && <span style={{ fontSize: 16, display: "grid" }} aria-hidden>{icon}</span>}
      <span className="truncate">{text ?? { ready: "Ready", wait: "Waiting", ok: "Done", warn: "Check", bad: "Failed", written: "Written" }[s]}</span>
    </span>
  );
}

/* ---------------------------------- table ---------------------------------- */

export function Table({
  cols,
  head,
  children,
  style,
  label,
}: {
  cols: string;
  head: ReactNode[];
  children: ReactNode;
  style?: CSSProperties;
  label?: string;
}) {
  return (
    <div className="tbl" role="table" aria-label={label} style={{ ["--cols" as string]: cols, ...style }}>
      <div className="th" role="row">
        {head.map((h, i) => (
          <span key={i} role="columnheader" className={cx(typeof h === "string" && h.startsWith(" ") && "r")}>
            {typeof h === "string" ? h.trim() : h}
          </span>
        ))}
      </div>
      <div className="tb" role="rowgroup">{children}</div>
    </div>
  );
}

export function Tr({
  on,
  onClick,
  children,
  style,
  label,
}: {
  on?: boolean;
  onClick?: Fn;
  children: ReactNode;
  style?: CSSProperties;
  label?: string;
}) {
  return (
    <div
      className={cx("tr", on && "on")}
      role="row"
      aria-selected={on}
      aria-label={label}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={
        onClick
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      style={style}
    >
      {children}
    </div>
  );
}

/* ---------------------------------- panes ---------------------------------- */

export const DL = ({ rows }: { rows: [ReactNode, ReactNode][] }) => (
  <dl className="dl">
    {rows.map(([k, v], i) => (
      <div key={i} style={{ display: "contents" }}>
        <dt>{k}</dt>
        <dd>{v}</dd>
      </div>
    ))}
  </dl>
);

export function InfoBar({
  tone,
  children,
  actions,
  style,
}: {
  tone: "ok" | "warn" | "bad" | "info";
  children: ReactNode;
  actions?: ReactNode;
  style?: CSSProperties;
}) {
  const icon =
    tone === "ok" ? <CheckmarkCircleFilled className="ok" /> : tone === "warn" ? <WarningFilled className="warn" /> : tone === "bad" ? <ErrorCircleFilled className="bad" /> : <InfoFilled className="acc" />;
  return (
    <div className="info" role={tone === "bad" ? "alert" : "status"} style={style}>
      <span className="ic" aria-hidden>{icon}</span>
      <span className="grow">{children}</span>
      {actions && <span className="end">{actions}</span>}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <span className="glyph" aria-hidden>{icon}</span>
      <span className="h">{title}</span>
      {children && <div className="row" style={{ gap: 8, marginTop: 4 }}>{children}</div>}
    </div>
  );
}

/** Text links in a details pane: icon commands laid out in a row. */
export const Links = ({ children }: { children: ReactNode }) => <div className="links">{children}</div>;

/** Middle ellipsis that adapts to the column: the tail (release, extension)
 *  always shows, and it starts on a word boundary. */
export function MidText({ text, tail = 18, className }: { text: string; tail?: number; className?: string }) {
  let at = Math.max(0, text.length - tail);
  const b = Math.max(text.lastIndexOf(" ", at), text.lastIndexOf(".", at));
  if (b > 0 && at - b < 8) at = b + 1;
  if (at < 12) return <span className={cx("truncate", className)} title={text}>{text}</span>;
  return (
    <span className={cx("midtext", className)} title={text}>
      <span className="h">{text.slice(0, at).replace(/ $/, " ")}</span>
      {/* Clipped from its start (direction: rtl); the name itself still reads left to right. */}
      <span className="t"><bdi dir="ltr">{text.slice(at)}</bdi></span>
    </span>
  );
}

/** The base name of a path, on either platform. */
export const baseName = (path: string) => path.replace(/^.*[\\/]/, "");
