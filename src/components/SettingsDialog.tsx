import { X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

import type { AppSettings, SyncMode } from "@/lib/types";

interface SettingsDialogProps {
  open: boolean;
  settings: AppSettings;
  mode: SyncMode;
  onChange: (settings: AppSettings) => void;
  onClose: () => void;
}

interface FieldProps {
  label: string;
  hint: string;
  children: React.ReactNode;
  htmlFor: string;
}

function Field({ label, hint, children, htmlFor }: FieldProps) {
  return (
    <div className="space-y-1">
      <label htmlFor={htmlFor} className="block text-xs font-medium text-foreground">
        {label}
      </label>
      {children}
      <p className="text-[11px] text-muted-foreground">{hint}</p>
    </div>
  );
}

export function SettingsDialog({
  open,
  settings,
  mode,
  onChange,
  onClose,
}: SettingsDialogProps) {
  const ids = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const [patternError, setPatternError] = useState<string | null>(null);

  // Escape closes, and focus moves into the dialog when it opens.
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    dialogRef.current?.querySelector<HTMLElement>("input, button")?.focus();
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const update = (patch: Partial<AppSettings>) => onChange({ ...settings, ...patch });

  const onPatternChange = (value: string) => {
    if (value.trim()) {
      try {
        const compiled = new RegExp(value, "i");
        // A pattern with no capture group can never produce a match key.
        const groups = new RegExp(`${compiled.source}|`).exec("")?.length ?? 1;
        setPatternError(groups - 1 < 1 ? "Add a capture group, e.g. S(\\d+)E(\\d+)" : null);
      } catch (error) {
        setPatternError(error instanceof Error ? error.message : "Invalid pattern");
      }
    } else {
      setPatternError(null);
    }
    update({ matchPattern: value });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${ids}-title`}
        className="w-full max-w-md rounded-lg border border-border bg-card shadow-lg"
      >
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 id={`${ids}-title`} className="text-sm font-medium text-foreground">
            Analysis settings
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            <X className="h-4 w-4" aria-hidden />
            <span className="sr-only">Close settings</span>
          </button>
        </header>

        <div className="max-h-[70vh] space-y-4 overflow-y-auto p-4">
          <Field
            label="Sample windows"
            htmlFor={`${ids}-count`}
            hint="How many points across each file to measure. More windows detect drift more reliably and resist a window landing on silence."
          >
            <div className="flex items-center gap-3">
              <input
                id={`${ids}-count`}
                type="range"
                min={2}
                max={12}
                step={1}
                value={settings.windowCount}
                onChange={(e) => update({ windowCount: Number(e.target.value) })}
                className="flex-1 accent-primary"
              />
              <span className="w-8 text-right font-mono text-xs text-foreground">
                {settings.windowCount}
              </span>
            </div>
          </Field>

          <Field
            label="Window length"
            htmlFor={`${ids}-window`}
            hint="Seconds of audio per window. Longer is more reliable on sparse dialogue but slower."
          >
            <div className="flex items-center gap-3">
              <input
                id={`${ids}-window`}
                type="range"
                min={10}
                max={180}
                step={5}
                value={settings.windowSeconds}
                onChange={(e) => update({ windowSeconds: Number(e.target.value) })}
                className="flex-1 accent-primary"
              />
              <span className="w-10 text-right font-mono text-xs text-foreground">
                {settings.windowSeconds}s
              </span>
            </div>
          </Field>

          <Field
            label="Maximum offset"
            htmlFor={`${ids}-offset`}
            hint="Alignments implying a larger shift than this are rejected. Bounding the search prevents distant false matches."
          >
            <div className="flex items-center gap-3">
              <input
                id={`${ids}-offset`}
                type="range"
                min={5}
                max={300}
                step={5}
                value={Math.round(settings.maxOffsetMs / 1000)}
                onChange={(e) => update({ maxOffsetMs: Number(e.target.value) * 1000 })}
                className="flex-1 accent-primary"
              />
              <span className="w-10 text-right font-mono text-xs text-foreground">
                {Math.round(settings.maxOffsetMs / 1000)}s
              </span>
            </div>
          </Field>

          <Field
            label="Parallel files"
            htmlFor={`${ids}-workers`}
            hint="Files analysed at once. Each one runs its own decoder, so high values compete for disk and memory."
          >
            <div className="flex items-center gap-3">
              <input
                id={`${ids}-workers`}
                type="range"
                min={1}
                max={8}
                step={1}
                value={settings.maxWorkers}
                onChange={(e) => update({ maxWorkers: Number(e.target.value) })}
                className="flex-1 accent-primary"
              />
              <span className="w-8 text-right font-mono text-xs text-foreground">
                {settings.maxWorkers}
              </span>
            </div>
          </Field>

          {mode === "series" && (
            <Field
              label="Match pattern (optional)"
              htmlFor={`${ids}-pattern`}
              hint="Leave empty to detect S01E01, 1x01 and similar automatically. Set a regex with capture groups to override."
            >
              <input
                id={`${ids}-pattern`}
                type="text"
                value={settings.matchPattern}
                onChange={(e) => onPatternChange(e.target.value)}
                placeholder="Automatic"
                spellCheck={false}
                aria-invalid={patternError ? true : undefined}
                className={`w-full rounded border bg-input px-3 py-2 font-mono text-xs text-foreground focus:outline-none focus:ring-1 ${
                  patternError
                    ? "border-destructive focus:ring-destructive"
                    : "border-border focus:ring-primary"
                }`}
              />
              {patternError && (
                <p className="text-[11px] text-destructive">{patternError}</p>
              )}
            </Field>
          )}

          <Field
            label="Output suffix"
            htmlFor={`${ids}-suffix`}
            hint="Appended to corrected files. Source files are never modified in place."
          >
            <input
              id={`${ids}-suffix`}
              type="text"
              value={settings.outputSuffix}
              onChange={(e) => update({ outputSuffix: e.target.value })}
              spellCheck={false}
              className="w-full rounded border border-border bg-input px-3 py-2 font-mono text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </Field>
        </div>

        <footer className="flex justify-end border-t border-border px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}
