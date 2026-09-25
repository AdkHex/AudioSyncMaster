import { RefreshCw } from "lucide-react";
import { useEffect, useId, useState } from "react";

import { Dialog } from "@/components/Dialog";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button, ProgressBar } from "@/components/ui";
import * as api from "@/lib/api";
import { cx } from "@/lib/cx";
import {
  DEFAULT_SETTINGS,
  formatSize,
  type AppSettings,
  type SyncMode,
  type VoiceToolsStatus,
} from "@/lib/types";

interface SettingsDialogProps {
  open: boolean;
  settings: AppSettings;
  mode: SyncMode;
  version: string;
  /** A sync is running; disables installing or removing the voice tools. */
  busy?: boolean;
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
  busy = false,
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

      {mode === "dubsync" && (
        <Group title="Dub sync">
          <Field
            label="Replace stretches that did not correlate"
            htmlFor={`${ids}-fill-unmatched`}
            hint="Off, a passage where the dub is audible but its music and effects could not be matched is kept as the dub, provided the offset is the same either side, and the plan warns you where. On, it is filled from the original instead, and the plan marks those fills Replaced so they can be told from real cuts. Keep this off unless the result plays the wrong scene."
            control={
              <input
                id={`${ids}-fill-unmatched`}
                type="checkbox"
                checked={settings.dubFillUnmatched}
                onChange={(event) => update({ dubFillUnmatched: event.target.checked })}
                className="h-[15px] w-[15px] accent-primary"
              />
            }
          />

          <Field
            label="Check the dub's voices against the lips"
            htmlFor={`${ids}-fix-voices`}
            hint="Some dubs were cut apart from their music, so their voices land off the lips. The voice check finds those scenes and moves only the voices. Needs a one-time download."
            control={
              <input
                id={`${ids}-fix-voices`}
                type="checkbox"
                checked={settings.fixVoices}
                onChange={(event) => update({ fixVoices: event.target.checked })}
                className="h-[15px] w-[15px] accent-primary"
              />
            }
          >
            <VoiceToolsPanel busy={busy} />
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

/** Status of, and control over, the optional voice tools the voice check
 *  needs: PyTorch, Demucs and Silero VAD, downloaded once into the app's
 *  own data folder rather than bundled, since most runs never touch them. */
function VoiceToolsPanel({ busy }: { busy: boolean }) {
  const [status, setStatus] = useState<VoiceToolsStatus | null>(null);
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState<{ percent: number; stage: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!api.isDesktop()) return;
    let active = true;
    api
      .voiceTools("status")
      .then((result) => {
        if (active) setStatus(result.status);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  const refresh = async () => {
    try {
      const result = await api.voiceTools("status");
      setStatus(result.status);
    } catch {
      /* left as it was */
    }
  };

  const handleInstall = async () => {
    setInstalling(true);
    setError(null);
    setProgress({ percent: 0, stage: "Starting…" });
    const unlisten = await api.subscribeToVoiceToolsProgress((event) =>
      setProgress({ percent: event.percent, stage: event.stage }),
    );
    try {
      const result = await api.voiceTools("install");
      setStatus(result.status);
      if (result.error && result.error !== "cancelled") setError(result.error);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      unlisten();
      setInstalling(false);
      setProgress(null);
    }
  };

  const handleRemove = async () => {
    setError(null);
    try {
      const result = await api.voiceTools("remove");
      setStatus(result.status);
      if (result.error && result.error !== "cancelled") setError(result.error);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleStop = async () => {
    await api.cancelSync();
  };

  if (!api.isDesktop()) return null;

  const installed = status?.installed ?? false;
  const outdated = status?.outdated ?? false;

  return (
    <div className="mt-2 w-full">
      {installing ? (
        <div className="space-y-1.5">
          <ProgressBar percent={progress?.percent ?? 0} label="Installing voice tools" />
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-muted-foreground">
              {progress?.stage ?? "Working…"}
            </span>
            <Button size="sm" variant="ghost" onClick={() => void handleStop()}>
              Stop
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11.5px] text-muted-foreground">
            {outdated
              ? "Out of date"
              : installed
                ? `Installed (runs on ${status?.device === "cuda" || status?.device === "mps" ? "GPU" : "CPU"})`
                : "Not installed"}
            {installed && status?.sizeBytes ? ` · ${formatSize(status.sizeBytes)}` : ""}
          </span>
          {installed ? (
            <Button size="sm" variant="ghost" onClick={() => void handleRemove()} disabled={busy}>
              Remove
            </Button>
          ) : null}
          {!installed || outdated ? (
            <Button size="sm" onClick={() => void handleInstall()} disabled={busy}>
              {outdated ? "Reinstall" : "Install voice tools (about 1 GB)"}
            </Button>
          ) : null}
          <Button size="sm" variant="ghost" onClick={() => void refresh()} disabled={busy}>
            Refresh
          </Button>
        </div>
      )}
      {error && <p className="mt-1.5 text-[11.5px] text-destructive">{error}</p>}
    </div>
  );
}
