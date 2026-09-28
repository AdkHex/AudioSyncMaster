/** Small form pieces the Subsync workspace shares, drawn exactly like the
 *  Sidebar's own selects and checkboxes so the mode looks native. */

import { cx } from "@/lib/cx";

const SELECT_ARROW =
  "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23969696' stroke-width='2.5'><path d='m6 9 6 6 6-6'/></svg>\")";

const SELECT_STYLE: React.CSSProperties = {
  backgroundImage: SELECT_ARROW,
  backgroundRepeat: "no-repeat",
  backgroundPosition: "right 7px center",
};

export const SELECT_CLASS =
  "w-full appearance-none rounded-md bg-elevated px-2 py-1 pr-6 text-[12px] text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50";

export const INPUT_CLASS =
  "rounded-md border border-border-strong bg-input px-2 py-1 text-[12px] focus:outline-none focus:ring-2 focus:ring-ring/40 disabled:opacity-50";

export function SelectBox({
  className,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cx(SELECT_CLASS, className)} style={SELECT_STYLE} {...props} />;
}

export function CheckRow({
  checked,
  disabled,
  onChange,
  label,
  hint,
  className,
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  hint?: string;
  className?: string;
}) {
  return (
    <label className={cx("flex cursor-pointer items-start gap-2 text-[12px]", className)}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-[3px] h-[15px] w-[15px] shrink-0 accent-primary"
      />
      <span>
        {label}
        {hint && (
          <span className="block text-[10.5px] leading-snug text-muted-foreground/80">{hint}</span>
        )}
      </span>
    </label>
  );
}

export function SectionTitle({ children, id }: { children: React.ReactNode; id?: string }) {
  return (
    <h2 id={id} className="mb-2 text-[11px] font-semibold text-muted-foreground">
      {children}
    </h2>
  );
}

/** A short, centred explanation of what to do next (as in the other modes). */
export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="mx-auto mt-[12vh] max-w-[420px] px-4 text-center">
      <p className="text-[13px] font-semibold">{title}</p>
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted-foreground">{body}</p>
    </div>
  );
}

/** A text link styled like the FilePanel's Clear / Change actions. */
export function LinkButton({
  className,
  tone = "muted",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: "muted" | "primary" }) {
  return (
    <button
      type="button"
      className={cx(
        "text-[11.5px] transition-colors disabled:opacity-40",
        tone === "primary"
          ? "font-medium text-primary hover:opacity-80"
          : "text-muted-foreground hover:text-foreground",
        className,
      )}
      {...props}
    />
  );
}
