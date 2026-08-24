import { memo } from "react";

import { Spinner } from "@/components/ui";

interface ProgressPanelProps {
  processed: number;
  total: number;
  currentFile: string | null;
  fileProgress: number;
  remainingMs: number | null;
}

function formatRemaining(ms: number | null): string | null {
  if (ms === null) return null;
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `about ${seconds}s left`;
  const minutes = Math.floor(seconds / 60);
  return `about ${minutes}m ${String(seconds % 60).padStart(2, "0")}s left`;
}

/** One line above the results while a run is in flight.
 *
 *  The finished files used to be listed here as well as in the results table
 *  below, so each one was drawn twice and the whole page reflowed as rows
 *  streamed in. Results now stream straight into the list; this only has to
 *  say where the run is. */
export const ProgressPanel = memo(function ProgressPanel({
  processed,
  total,
  currentFile,
  fileProgress,
  remainingMs,
}: ProgressPanelProps) {
  // Counting only finished files pins the bar at 0% for the whole first file,
  // which reads as a hang; folding in the active file keeps it moving.
  const overall =
    total > 0
      ? Math.min(100, ((processed + (currentFile ? fileProgress / 100 : 0)) / total) * 100)
      : 0;

  const eta = formatRemaining(remainingMs);

  return (
    <div
      className="relative flex h-11 shrink-0 items-center gap-3 border-b border-border px-[18px]"
      aria-label="Analysis progress"
    >
      <Spinner className="h-3.5 w-3.5 shrink-0 border-[1.8px]" />

      <span className="tabular shrink-0 text-[12.5px] font-medium">
        {total > 0 ? `${Math.min(processed + 1, total)} of ${total}` : "Starting…"}
      </span>

      {currentFile && (
        <span
          className="min-w-0 flex-1 truncate text-[12.5px] text-muted-foreground"
          title={currentFile}
        >
          {currentFile}
        </span>
      )}
      {!currentFile && <span className="flex-1" />}

      <span className="tabular shrink-0 font-mono text-[11.5px] text-muted-foreground">
        {eta ?? `${Math.round(overall)}%`}
      </span>

      {/* The bar sits on the divider itself rather than taking a row of its
          own, so starting a run does not shift the results down. */}
      <span
        className="absolute inset-x-0 bottom-0 h-[2px] bg-primary transition-[width] duration-300"
        style={{ width: `${overall}%` }}
        aria-hidden
      />
    </div>
  );
});
