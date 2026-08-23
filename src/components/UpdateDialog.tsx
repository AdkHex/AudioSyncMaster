import { AlertCircle, Download, Sparkles, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  formatBytes,
  installUpdate,
  skipVersion,
  type DownloadProgress,
  type UpdateInfo,
} from "@/lib/updater";

interface UpdateDialogProps {
  update: UpdateInfo | null;
  onDismiss: () => void;
}

export function UpdateDialog({ update, onDismiss }: UpdateDialogProps) {
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  const installing = progress !== null;

  useEffect(() => {
    if (!update) return;
    // Escape dismisses, but not mid-install: interrupting a partly-written
    // installer is how you end up with a broken installation.
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !installing) {
        event.stopPropagation();
        onDismiss();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    dialogRef.current?.querySelector<HTMLButtonElement>("button[data-default]")?.focus();
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [update, installing, onDismiss]);

  if (!update) return null;

  const handleInstall = async () => {
    setError(null);
    setProgress({ downloaded: 0, total: null, percent: null });
    try {
      await installUpdate(update, setProgress);
      // If we get here the app is about to restart.
    } catch (err) {
      setProgress(null);
      setError(
        err instanceof Error
          ? err.message
          : "The update could not be installed. Try again, or download it manually.",
      );
    }
  };

  const handleSkip = () => {
    skipVersion(update.version);
    onDismiss();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !installing) onDismiss();
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="update-title"
        className="w-full max-w-md rounded-lg border border-border bg-card shadow-lg"
      >
        <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
          <div className="flex items-start gap-2.5">
            <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary/15">
              <Sparkles className="h-4 w-4 text-primary" aria-hidden />
            </span>
            <div>
              <h2 id="update-title" className="text-sm font-medium text-foreground">
                Version {update.version} is available
              </h2>
              <p className="text-[11px] text-muted-foreground">
                You have {update.currentVersion}
              </p>
            </div>
          </div>
          {!installing && (
            <button
              type="button"
              onClick={onDismiss}
              className="rounded p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            >
              <X className="h-4 w-4" aria-hidden />
              <span className="sr-only">Close</span>
            </button>
          )}
        </header>

        {update.notes && (
          <div className="max-h-52 overflow-y-auto border-b border-border px-4 py-3">
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
              {update.notes}
            </p>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 border-b border-border bg-destructive/10 px-4 py-2.5">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" aria-hidden />
            <p className="text-[11px] text-destructive">{error}</p>
          </div>
        )}

        {installing && (
          <div className="border-b border-border px-4 py-3">
            <div className="mb-1.5 flex items-center justify-between text-[11px]">
              <span className="text-foreground">
                {progress.percent === 100 ? "Installing…" : "Downloading…"}
              </span>
              <span className="text-muted-foreground">
                {progress.total
                  ? `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)}`
                  : formatBytes(progress.downloaded)}
              </span>
            </div>
            <div
              className="h-1.5 overflow-hidden rounded-full bg-secondary"
              role="progressbar"
              aria-valuenow={progress.percent ?? undefined}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Update download progress"
            >
              <div
                className={`h-full rounded-full bg-primary transition-[width] duration-200 ${
                  progress.percent === null ? "w-1/3 animate-pulse" : ""
                }`}
                style={progress.percent !== null ? { width: `${progress.percent}%` } : undefined}
              />
            </div>
            <p className="mt-2 text-[10px] text-muted-foreground">
              The app will restart when this finishes.
            </p>
          </div>
        )}

        <footer className="flex items-center justify-end gap-2 px-4 py-3">
          {!installing && (
            <>
              <button
                type="button"
                onClick={handleSkip}
                className="rounded px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              >
                Skip this version
              </button>
              <button
                type="button"
                onClick={onDismiss}
                className="rounded border border-border px-3 py-1.5 text-xs text-foreground transition-colors hover:bg-secondary"
              >
                Later
              </button>
              <button
                type="button"
                data-default
                onClick={() => void handleInstall()}
                className="flex items-center gap-1.5 rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                <Download className="h-3.5 w-3.5" aria-hidden />
                Install and restart
              </button>
            </>
          )}
        </footer>
      </div>
    </div>
  );
}
