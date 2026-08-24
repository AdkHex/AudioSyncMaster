import { RefreshCw } from "lucide-react";
import { useId, useState } from "react";

import { Dialog } from "@/components/Dialog";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui";
import { cx } from "@/lib/cx";
import { DEFAULT_SETTINGS, type AppSettings, type SyncMode } from "@/lib/types";

interface SettingsDialogProps {
  open: boolean;
  settings: AppSettings;
  mode: SyncMode;
  version: string;
  onChange: (settings: AppSettings) => void;
  onClose: () => void;
  onCheckForUpdate?: () => void;
  checkingUpdate?: boolean;
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-border py-4 last:border-0">
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">
        {title}
      </h3>
      <div className="space-y-0">{children}</div>
    </section>
  );
}

/** One setting: label and control on a line, explanation beneath. The hint
 *  always says what turning the value up costs, which the original never did. */
function Field({
  label,
  hint,
  htmlFor,
  control,
  children,
}: {
  label: string;
  hint: string;
  htmlFor?: string;
  control: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-border py-3 first:border-0">
      <label htmlFor={htmlFor} className="flex-1 text-[13px] font-medium">
        {label}
      </label>
      <div className="flex shrink-0 items-center gap-2.5">{control}</div>
      <p className="w-full max-w-[46ch] text-[11.5px] leading-relaxed text-muted-foreground">
        {hint}
      </p>
      {children}
    </div>
  );
}

function Slider({
  id,
  min,
  max,
  step,
  value,
  display,
  onChange,
}: {
  id: string;
  min: number;
  max: number;
  step: number;
  value: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-[132px] cursor-pointer"
      />
      <span className="tabular w-11 text-right font-mono text-xs font-semibold">{display}</span>
    </>
  );
}

export function SettingsDialog({
  open,
  settings,
  mode,
  version,
  onChange,
  onClose,
  onCheckForUpdate,
  checkingUpdate = false,
}: SettingsDialogProps) {
  const ids = useId();
  const [patternError, setPatternError] = useState<string | null>(null);

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

  const resetDefaults = () => {
    setPatternError(null);
    // Theme is owned by the ThemeProvider, so it is not reset here.
    onChange({ ...DEFAULT_SETTINGS, theme: settings.theme });
  };

  return (
    <Dialog
      open={open}
      title="Settings"
      description="Applies to the next analysis"
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" size="sm" onClick={resetDefaults}>
            Reset to defaults
          </Button>
          <span className="flex-1" />
          <Button variant="primary" size="sm" onClick={onClose} className="px-5">
            Done
          </Button>
        </>
      }
    >
      <Group title="Analysis">
        <Field
          label="Sample windows"
          htmlFor={`${ids}-count`}
          hint="How many points across each file to measure. More windows detect drift more reliably and resist a window landing on silence — but each one costs time."
          control={
            <Slider
              id={`${ids}-count`}
              min={2}
              max={12}
              step={1}
              value={settings.windowCount}
              display={String(settings.windowCount)}
              onChange={(windowCount) => update({ windowCount })}
            />
          }
        />

        <Field
          label="Window length"
          htmlFor={`${ids}-window`}
          hint="Seconds of audio per window. Longer is more reliable on sparse dialogue, and slower."
          control={
            <Slider
              id={`${ids}-window`}
              min={10}
              max={180}
              step={5}
              value={settings.windowSeconds}
              display={`${settings.windowSeconds}s`}
              onChange={(windowSeconds) => update({ windowSeconds })}
            />
          }
        />

        <Field
          label="Maximum offset"
          htmlFor={`${ids}-offset`}
          hint="Alignments implying a larger shift than this are rejected. Bounding the search prevents distant false matches."
          control={
            <Slider
              id={`${ids}-offset`}
              min={5}
              max={300}
              step={5}
              value={Math.round(settings.maxOffsetMs / 1000)}
              display={`${Math.round(settings.maxOffsetMs / 1000)}s`}
              onChange={(seconds) => update({ maxOffsetMs: seconds * 1000 })}
            />
          }
        />

        <Field
          label="Parallel files"
          htmlFor={`${ids}-workers`}
          hint="Files analysed at once. Each runs its own decoder, so high values compete for disk and memory."
          control={
            <Slider
              id={`${ids}-workers`}
              min={1}
              max={8}
              step={1}
              value={settings.maxWorkers}
              display={String(settings.maxWorkers)}
              onChange={(maxWorkers) => update({ maxWorkers })}
            />
          }
        />
      </Group>

      {mode === "series" && (
        <Group title="Matching · Series">
          <Field
            label="Match pattern"
            htmlFor={`${ids}-pattern`}
            hint="Leave empty to detect S01E01, 1x01 and similar automatically. Set a regex with capture groups to override."
            control={
              <input
                id={`${ids}-pattern`}
                type="text"
                value={settings.matchPattern}
                onChange={(event) => onPatternChange(event.target.value)}
                placeholder="Automatic"
                spellCheck={false}
                aria-invalid={patternError ? true : undefined}
                aria-errormessage={patternError ? `${ids}-pattern-error` : undefined}
                className={cx(
                  "w-[190px] rounded-md border bg-input px-2.5 py-1.5 font-mono text-xs",
                  "focus:outline-none focus:ring-2",
                  patternError
                    ? "border-destructive focus:ring-destructive/40"
                    : "border-border-strong focus:ring-ring/40",
                )}
              />
            }
          >
            {patternError && (
              <p
                id={`${ids}-pattern-error`}
                className="w-full text-[11.5px] text-destructive"
              >
                {patternError}
              </p>
            )}
          </Field>
        </Group>
      )}

      <Group title="Output">
        <Field
          label="Output suffix"
          htmlFor={`${ids}-suffix`}
          hint="Appended to corrected files. Source files are never modified in place."
          control={
            <input
              id={`${ids}-suffix`}
              type="text"
              value={settings.outputSuffix}
              onChange={(event) => update({ outputSuffix: event.target.value })}
              spellCheck={false}
              className="w-[130px] rounded-md border border-border-strong bg-input px-2.5 py-1.5 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-ring/40"
            />
          }
        />
      </Group>

      <Group title="Appearance">
        <Field
          label="Theme"
          hint="Auto follows the operating system setting."
          control={<ThemeToggle />}
        />
      </Group>

      <Group title="About">
        <Field
          label={`Version ${version}`}
          hint="Updates are checked automatically on launch."
          control={
            onCheckForUpdate ? (
              <Button size="sm" onClick={onCheckForUpdate} disabled={checkingUpdate}>
                <RefreshCw
                  className={cx("h-3.5 w-3.5", checkingUpdate && "animate-spin")}
                  aria-hidden
                />
                {checkingUpdate ? "Checking…" : "Check now"}
              </Button>
            ) : null
          }
        />
      </Group>
    </Dialog>
  );
}
