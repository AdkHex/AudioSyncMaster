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

// ------------------------------------------------------------------------ tag

type TagTone = "neutral" | "success" | "warning" | "destructive";

const TAG_TONES: Record<TagTone, string> = {
  neutral: "text-muted-foreground",
  success: "text-success",
  warning: "text-warning",
  destructive: "text-destructive",
};

/** A status word carrying its meaning in colour alone.
 *
 *  A filled, bordered pill is three visual devices for one piece of
 *  information. In a list where most rows carry two of them, that reads as
 *  clutter; coloured text says the same thing and lets the filenames stay the
 *  loudest element in the row. */
export function Tag({
  tone = "neutral",
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: TagTone }) {
  return (
    <span
      className={cx("whitespace-nowrap text-[11.5px]", TAG_TONES[tone], className)}
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
