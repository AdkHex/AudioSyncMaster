/** Modal shell shared by settings, updates and the apply-progress dialog.
 *
 *  Handles the parts that are easy to get wrong once and then forget: Escape to
 *  close, a scrim click that only fires on the scrim itself, focus moved into
 *  the dialog on open and restored to the trigger on close, and body-level
 *  labelling for screen readers. */

import { X } from "lucide-react";
import { useEffect, useId, useRef } from "react";

import { IconButton } from "@/components/ui";
import { cx } from "@/lib/cx";

interface DialogProps {
  open: boolean;
  title: string;
  description?: string;
  onClose?: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  /** Hide the close affordances for a dialog the user must not dismiss. */
  dismissible?: boolean;
  className?: string;
}

export function Dialog({
  open,
  title,
  description,
  onClose,
  children,
  footer,
  dismissible = true,
  className,
}: DialogProps) {
  const ids = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;

    restoreRef.current = document.activeElement as HTMLElement | null;
    panelRef.current?.querySelector<HTMLElement>(
      "input:not([type=hidden]), button, select, textarea, [tabindex]",
    )?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && dismissible && onClose) {
        // Stop the app-level Escape handler from also cancelling a run.
        event.stopPropagation();
        event.preventDefault();
        onClose();
      }
    };

    window.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      restoreRef.current?.focus?.();
    };
  }, [open, onClose, dismissible]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-6 sm:p-10"
      onMouseDown={(event) => {
        if (dismissible && onClose && event.target === event.currentTarget) onClose();
      }}
    >
      <div className="fixed inset-0 animate-fade-in bg-black/50 backdrop-blur-[2px]" aria-hidden />

      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${ids}-title`}
        aria-describedby={description ? `${ids}-desc` : undefined}
        className={cx(
          "relative z-10 my-auto flex w-full max-w-lg animate-sheet-in flex-col overflow-hidden",
          "rounded-2xl border border-border bg-card shadow-2xl",
          className,
        )}
      >
        <header className="flex items-start gap-3 border-b border-border px-5 py-4">
          <div className="min-w-0 flex-1">
            <h2 id={`${ids}-title`} className="text-sm font-semibold tracking-tight">
              {title}
            </h2>
            {description && (
              <p id={`${ids}-desc`} className="mt-0.5 text-[11.5px] text-muted-foreground">
                {description}
              </p>
            )}
          </div>
          {dismissible && onClose && (
            <IconButton label="Close" onClick={onClose}>
              <X className="h-4 w-4" aria-hidden />
            </IconButton>
          )}
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-1">{children}</div>

        {footer && (
          <footer className="flex items-center gap-2.5 border-t border-border bg-elevated px-5 py-3.5">
            {footer}
          </footer>
        )}
      </div>
    </div>
  );
}
