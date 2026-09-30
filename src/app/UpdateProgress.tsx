/** Installing an update: download, install, restart. The dialog cannot be
 *  dismissed while the installer runs — interrupting it is how an install
 *  ends up broken. */

import { useEffect, useRef, useState } from "react";

import { formatBytes, installUpdate, type DownloadProgress, type UpdateInfo } from "@/lib/updater";
import { Modal } from "@/ui/frame";
import { Btn, PBar } from "@/ui/kit";

const MB = 1024 * 1024;

export function UpdateProgress({ update, onClose }: { update: UpdateInfo; onClose: () => void }) {
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const started = useRef(-1);

  useEffect(() => {
    if (started.current === attempt) return;
    started.current = attempt;
    setError(null);
    setProgress({ downloaded: 0, total: null, percent: null });
    installUpdate(update, setProgress).catch((err) => {
      setProgress(null);
      setError(err instanceof Error ? err.message : "The update could not be installed. Try again, or download it manually.");
    });
  }, [attempt, update]);

  const done = progress?.percent !== null && progress?.percent !== undefined && progress.percent >= 100;
  return (
    <Modal onClose={error ? onClose : undefined}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={`Updating to ${update.version}`}>
        <div className="db">
          <div className="dt">Updating to {update.version}</div>
          {error ? (
            <div className="bad">{error}</div>
          ) : (
            <>
              <div className="row">
                <span className="grow">{done ? "Installing" : "Downloading"}</span>
                <span className="t3 num">
                  {progress && progress.total
                    ? `${(progress.downloaded / MB).toFixed(1)} of ${Math.round(progress.total / MB)} MB`
                    : progress
                      ? formatBytes(progress.downloaded)
                      : ""}
                </span>
              </div>
              <PBar pct={progress?.percent ?? null} ind={progress?.percent === null || progress?.percent === undefined} />
              <div className="t3">AudioSyncMaster restarts when it's done.</div>
            </>
          )}
        </div>
        {/* Nothing to press while it installs; the buttons come with a failure. */}
        {error && (
          <div className="df">
            <Btn accent onClick={() => setAttempt((n) => n + 1)}>Try again</Btn>
            <Btn onClick={onClose}>Close</Btn>
          </div>
        )}
      </div>
    </Modal>
  );
}
