import { AlertCircle, Download } from "lucide-react";
import { useState } from "react";

import { Dialog } from "@/components/Dialog";
import { Button, Notice, ProgressBar } from "@/components/ui";
import { cx } from "@/lib/cx";
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

  const installing = progress !== null;

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
    <Dialog
      open
      title={`Version ${update.version} is available`}
      description={`You have ${update.currentVersion}`}
      // Interrupting a partly-written installer is how you end up with a
      // broken installation, so the dialog locks down mid-install.
      dismissible={!installing}
      onClose={onDismiss}
      className="max-w-md"
      footer={
        installing ? (
          <p className="text-[11.5px] text-muted-foreground">
            The app will restart when this finishes.
          </p>
        ) : (
          <>
            <Button variant="ghost" size="sm" onClick={handleSkip}>
              Skip this version
            </Button>
            <span className="flex-1" />
            <Button size="sm" onClick={onDismiss}>
              Later
            </Button>
            <Button variant="primary" size="sm" onClick={() => void handleInstall()}>
              <Download className="h-3.5 w-3.5" aria-hidden />
              Update &amp; restart
            </Button>
          </>
        )
      }
    >
      <div className="space-y-3 py-3">
        {update.notes && (
          <div className="max-h-52 overflow-y-auto rounded-lg border border-border bg-sunken px-3.5 py-3">
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
              {update.notes}
            </p>
          </div>
        )}

        {error && (
          <Notice tone="destructive" icon={<AlertCircle className="h-3.5 w-3.5" aria-hidden />}>
            {error}
          </Notice>
        )}

        {installing && (
          <div>
            <div className="mb-2 flex items-center justify-between text-[11.5px]">
              <span className="font-medium">
                {progress.percent === 100 ? "Installing…" : "Downloading…"}
              </span>
              <span className="tabular font-mono text-muted-foreground">
                {progress.total
                  ? `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)}`
                  : formatBytes(progress.downloaded)}
              </span>
            </div>
            {progress.percent === null ? (
              <div className="h-1.5 overflow-hidden rounded-full bg-sunken">
                <div className={cx("h-full w-1/3 animate-pulse rounded-full bg-primary")} />
              </div>
            ) : (
              <ProgressBar percent={progress.percent} label="Update download progress" />
            )}
          </div>
        )}
      </div>
    </Dialog>
  );
}
