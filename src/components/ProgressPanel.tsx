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
  /** Results that have already streamed in, shown as the completed queue. */
  results: SyncResult[];
  workers: number;
}

function formatRemaining(ms: number | null): string {
  if (ms === null) return "estimating…";
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
  workers,
}: ProgressPanelProps) {
  const overall = total > 0 ? Math.round((processed / total) * 100) : 0;

  // The last few finished files, plus the one in flight. Older entries scroll
  // out rather than growing the panel without bound on a long series run.
  const recent = results.slice(-4);

  return (
    <Card className="px-4 py-4" aria-label="Analysis progress">
      <div className="mb-3.5 flex items-center gap-3">
        <Spinner className="h-[15px] w-[15px]" />
        <span
          className="min-w-0 flex-1 truncate text-[13px] font-medium"
          title={currentFile ?? undefined}
        >
          {currentFile ?? "Preparing…"}
        </span>
        <span className="tabular shrink-0 font-mono text-[11.5px] text-muted-foreground">
          {processed} of {total} · {formatRemaining(remainingMs)}
        </span>
      </div>

      <div className="mb-1.5 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>Overall</span>
        <span className="tabular font-mono">{overall}%</span>
      </div>
      <ProgressBar percent={overall} label="Overall progress" />

      <div className="mb-1.5 mt-3.5 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>Current file</span>
        <span className="tabular font-mono">{fileProgress}%</span>
      </div>
      <ProgressBar percent={fileProgress} thin label="Current file progress" />

      {(recent.length > 0 || currentFile) && (
        <ul className="mt-4 flex flex-col gap-2 border-t border-border pt-3">
          {recent.map((result) => (
            <li
              key={`${result.primaryPath ?? result.videoFile}`}
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

          {currentFile && (
            <li className="flex items-center gap-2.5 text-xs font-medium">
              <span className="grid h-[15px] w-[15px] shrink-0 place-items-center">
                <Spinner className="h-3 w-3 border-[1.8px]" />
              </span>
              <span className="min-w-0 flex-1 truncate" title={currentFile}>
                {currentFile}
              </span>
              <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
                measuring…
              </span>
            </li>
          )}
        </ul>
      )}

      {workers > 1 && (
        <p className="mt-3 text-[11px] text-muted-foreground">
          Analysing up to {workers} files at once.
        </p>
      )}
    </Card>
  );
});
