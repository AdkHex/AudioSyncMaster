import { Check } from "lucide-react";
import { memo } from "react";

import { Card, ProgressBar, Spinner } from "@/components/ui";
import { cx } from "@/lib/cx";
import { formatDelay, type SyncResult } from "@/lib/types";

interface ProgressPanelProps {
  processed: number;
  total: number;
  currentFile: string | null;
  fileProgress: number;
  remainingMs: number | null;
  /** Results that have already streamed in, shown as the completed rows. */
  results: SyncResult[];
}

function formatRemaining(ms: number | null): string | null {
  if (ms === null) return null;
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `about ${seconds}s left`;
  const minutes = Math.floor(seconds / 60);
  return `about ${minutes}m ${String(seconds % 60).padStart(2, "0")}s left`;
}

export const ProgressPanel = memo(function ProgressPanel({
  processed,
  total,
  currentFile,
  fileProgress,
  remainingMs,
  results,
}: ProgressPanelProps) {
  // One number for the whole run. Counting only finished files leaves the bar
  // pinned at 0% for the entire first file, which reads as a hang; folding in
  // the active file's own progress makes it move continuously.
  const overall =
    total > 0
      ? Math.min(100, ((processed + (currentFile ? fileProgress / 100 : 0)) / total) * 100)
      : 0;

  const eta = formatRemaining(remainingMs);
  const recent = results.slice(-4);

  return (
    <Card className="px-4 py-4" aria-label="Analysis progress">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="text-[13px] font-medium">
          Analysing {total > 0 ? `${Math.min(processed + 1, total)} of ${total}` : "…"}
        </span>
        <span className="tabular shrink-0 font-mono text-[11.5px] text-muted-foreground">
          {/* Until there is a real estimate, show measured progress rather
              than a placeholder that lingers for the whole first file. */}
          {eta ?? `${Math.round(overall)}%`}
        </span>
      </div>

      <ProgressBar percent={overall} label="Overall progress" />

      {(recent.length > 0 || currentFile) && (
        <ul className="mt-3.5 flex flex-col gap-2 border-t border-border pt-3">
          {recent.map((result) => (
            <li
              key={result.primaryPath ?? result.videoFile}
              className="flex items-center gap-2.5 text-xs text-muted-foreground"
            >
              <span
                className={cx(
                  "grid h-[15px] w-[15px] shrink-0 place-items-center",
                  result.error ? "text-destructive" : "text-success",
                )}
              >
                {result.error ? (
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" aria-hidden>
                    <path d="M18 6 6 18M6 6l12 12" />
                  </svg>
                ) : (
                  <Check className="h-3.5 w-3.5" aria-hidden />
                )}
              </span>
              <span className="min-w-0 flex-1 truncate" title={result.videoFile}>
                {result.videoFile}
              </span>
              <span
                className={cx(
                  "tabular shrink-0 font-mono text-[11px]",
                  result.error && "text-destructive",
                )}
              >
                {result.error ? "failed" : formatDelay(result.delayMs)}
              </span>
            </li>
          ))}

          {/* The active file appears once, here -- not also in a header. */}
          {currentFile && (
            <li className="flex items-center gap-2.5 text-xs font-medium">
              <span className="grid h-[15px] w-[15px] shrink-0 place-items-center">
                <Spinner className="h-3 w-3 border-[1.8px]" />
              </span>
              <span className="min-w-0 flex-1 truncate" title={currentFile}>
                {currentFile}
              </span>
              <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
                {fileProgress}%
              </span>
            </li>
          )}
        </ul>
      )}
    </Card>
  );
});
