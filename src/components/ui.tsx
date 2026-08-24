/** Shared visual primitives.
 *
 *  The panels used to hand-roll the same long Tailwind strings, so a spacing or
 *  colour decision had to be re-made (and drifted) in every file. These are the
 *  vocabulary the redesign is written in: one definition per shape. */

import { forwardRef } from "react";

import { cx } from "@/lib/cx";

// --------------------------------------------------------------------- button

type ButtonVariant = "primary" | "default" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg" | "icon";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-primary-foreground border-primary hover:bg-primary/90 hover:border-primary/90 shadow-sm",
  default:
    "bg-card text-foreground border-border-strong hover:bg-secondary shadow-sm",
  ghost: "bg-transparent text-muted-foreground border-transparent hover:bg-secondary hover:text-foreground",
  danger: "bg-destructive/10 text-destructive border-destructive/30 hover:bg-destructive/15",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: "h-[30px] px-2.5 text-xs rounded-md gap-1.5",
  md: "h-9 px-4 text-[13px] rounded-lg gap-2",
  lg: "h-11 px-6 text-sm rounded-[11px] gap-2",
  icon: "h-[31px] w-[31px] rounded-md",
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "default", size = "md", className, type = "button", ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx(
        "inline-flex shrink-0 items-center justify-center border font-medium transition-colors",
        "disabled:pointer-events-none disabled:opacity-40 disabled:shadow-none",
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      {...props}
    />
  );
});

/** Header/toolbar icon button. `active` marks a toggled-on panel. */
interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  active?: boolean;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton({ label, active, className, type = "button", ...props }, ref) {
    return (
      <button
        ref={ref}
        type={type}
        title={label}
        aria-label={label}
        aria-pressed={active}
        className={cx(
          "inline-grid h-[31px] w-[31px] shrink-0 place-items-center rounded-md transition-colors",
          "disabled:pointer-events-none disabled:opacity-40",
          active
            ? "bg-accent text-accent-foreground"
            : "text-muted-foreground hover:bg-secondary hover:text-foreground",
          className,
        )}
        {...props}
      />
    );
  },
);

// ----------------------------------------------------------------------- pill

type PillTone = "neutral" | "accent" | "success" | "warning" | "destructive";

const PILL_TONES: Record<PillTone, string> = {
  neutral: "bg-sunken text-muted-foreground border-border",
  accent: "bg-accent text-accent-foreground border-primary/25",
  success: "bg-success/10 text-success border-success/25",
  warning: "bg-warning/10 text-warning border-warning/30",
  destructive: "bg-destructive/10 text-destructive border-destructive/25",
};

interface PillProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: PillTone;
}

export function Pill({ tone = "neutral", className, ...props }: PillProps) {
  return (
    <span
      className={cx(
        "inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5",
        "text-[11px] font-medium",
        PILL_TONES[tone],
        className,
      )}
      {...props}
    />
  );
}

// ----------------------------------------------------------------------- card

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cx("rounded-xl border border-border bg-card shadow-sm", className)}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cx("flex items-center gap-2.5 border-b border-border px-4 py-3", className)}
      {...props}
    />
  );
}

// --------------------------------------------------------------------- notice

interface NoticeProps extends React.HTMLAttributes<HTMLDivElement> {
  tone?: "warning" | "destructive";
  icon?: React.ReactNode;
}

export function Notice({ tone = "warning", icon, className, children, ...props }: NoticeProps) {
  return (
    <div
      role={tone === "destructive" ? "alert" : "status"}
      className={cx(
        "flex gap-2.5 rounded-lg border px-3.5 py-3 text-xs leading-relaxed",
        tone === "destructive"
          ? "border-destructive/30 bg-destructive/10 text-destructive"
          : "border-warning/30 bg-warning/10 text-warning",
        className,
      )}
      {...props}
    >
      {icon && <span className="mt-px shrink-0">{icon}</span>}
      <span className="min-w-0">{children}</span>
    </div>
  );
}

// ----------------------------------------------------------------- step header

interface StepHeaderProps {
  index: number;
  title: string;
  state: "todo" | "active" | "done";
  aside?: React.ReactNode;
}

/** The spine of the guided flow: Select -> Review -> Analyse -> Fix. */
export function StepHeader({ index, title, state, aside }: StepHeaderProps) {
  return (
    <div className="flex items-center gap-3">
      <span
        className={cx(
          "grid h-[21px] w-[21px] shrink-0 place-items-center rounded-full border text-[11px] font-semibold",
          state === "done" && "border-success bg-success text-success-foreground",
          state === "active" && "border-primary bg-primary text-primary-foreground",
          state === "todo" && "border-border-strong bg-card text-muted-foreground",
        )}
      >
        {state === "done" ? (
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M20 6 9 17l-5-5" />
          </svg>
        ) : (
          index
        )}
      </span>
      <h2
        className={cx(
          "text-xs font-semibold uppercase tracking-[0.06em]",
          state === "todo" ? "text-muted-foreground" : "text-foreground",
        )}
      >
        {title}
      </h2>
      <span className="h-px flex-1 bg-border" />
      {aside && <span className="shrink-0 text-[11px] text-muted-foreground">{aside}</span>}
    </div>
  );
}

// -------------------------------------------------------------------- overlays

interface ScrimProps {
  onClose: () => void;
  className?: string;
}

export function Scrim({ onClose, className }: ScrimProps) {
  return (
    <div
      className={cx("absolute inset-0 z-40 animate-fade-in bg-black/50 backdrop-blur-[2px]", className)}
      onMouseDown={onClose}
      aria-hidden
    />
  );
}

/** Checkbox styled to match the mockup. Native input kept for accessibility and
 *  keyboard behaviour; the visual is drawn on top of it. */
interface CheckboxProps {
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
  label: string;
  indeterminate?: boolean;
}

export function Checkbox({ checked, onChange, disabled, label, indeterminate }: CheckboxProps) {
  return (
    <span className="relative inline-flex h-[15px] w-[15px] shrink-0 align-middle">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        aria-label={label}
        className="peer absolute inset-0 z-10 m-0 cursor-pointer opacity-0 disabled:cursor-not-allowed"
      />
      <span
        aria-hidden
        className={cx(
          "pointer-events-none grid h-[15px] w-[15px] place-items-center rounded border-[1.5px] transition-colors",
          "peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-1 peer-focus-visible:ring-offset-card",
          checked || indeterminate
            ? "border-primary bg-primary text-primary-foreground"
            : "border-border-strong bg-card",
          disabled && "opacity-40",
        )}
      >
        {indeterminate && !checked ? (
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round">
            <path d="M6 12h12" />
          </svg>
        ) : checked ? (
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 6 9 17l-5-5" />
          </svg>
        ) : null}
      </span>
    </span>
  );
}

/** Determinate progress bar. */
export function ProgressBar({
  percent,
  thin,
  label,
}: {
  percent: number;
  thin?: boolean;
  label?: string;
}) {
  const clamped = Math.max(0, Math.min(100, percent));
  return (
    <div
      className={cx("overflow-hidden rounded-full bg-sunken", thin ? "h-1" : "h-1.5")}
      role="progressbar"
      aria-valuenow={Math.round(clamped)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div
        className={cx("h-full rounded-full transition-[width] duration-300", thin ? "bg-muted-foreground" : "bg-primary")}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cx(
        "inline-block shrink-0 animate-spin rounded-full border-2 border-border-strong border-t-primary",
        className ?? "h-4 w-4",
      )}
    />
  );
}
