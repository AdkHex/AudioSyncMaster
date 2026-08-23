import { memo } from "react";

interface ProgressPanelProps {
  processed: number;
  total: number;
  currentFile: string | null;
  fileProgress: number;
  remainingMs: number | null;
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
}: ProgressPanelProps) {
  const overall = total > 0 ? Math.round((processed / total) * 100) : 0;

  return (
    <section
      className="rounded-lg border border-border bg-card px-4 py-3"
      aria-label="Analysis progress"
    >
      <div className="mb-2 flex items-center justify-between gap-3 text-[11px]">
        <span className="min-w-0 truncate text-foreground" title={currentFile ?? undefined}>
          {currentFile ?? "Preparing…"}
        </span>
        <span className="shrink-0 text-muted-foreground">
          {processed}/{total} · {formatRemaining(remainingMs)}
        </span>
      </div>

      <div
        className="h-1.5 overflow-hidden rounded-full bg-secondary"
        role="progressbar"
        aria-valuenow={overall}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Overall progress"
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-300"
          style={{ width: `${overall}%` }}
        />
      </div>

      <div className="mt-2 flex items-center justify-between text-[10px] text-muted-foreground">
        <span>Overall {overall}%</span>
        <span>Current file {fileProgress}%</span>
      </div>

      <div className="mt-1 h-1 overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full bg-success transition-[width] duration-300"
          style={{ width: `${fileProgress}%` }}
        />
      </div>
    </section>
  );
});
