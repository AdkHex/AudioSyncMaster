import {
  AlertTriangle,
  Check,
  Clipboard,
  Download,
  TrendingUp,
  Wand2,
  X,
} from "lucide-react";
import { memo, useMemo, useState } from "react";

import {
  confidenceLevel,
  ffmpegCommandFor,
  formatDelay,
  formatDrift,
  formatElapsed,
  resultKey,
  type ConfidenceLevel,
  type RunSummary,
  type SyncResult,
} from "@/lib/types";

type Filter = "all" | ConfidenceLevel | "drift" | "failed";

interface ResultsPanelProps {
  results: SyncResult[];
  summary: RunSummary | null;
  selectedKeys: Set<string>;
  onToggleSelection: (key: string) => void;
  onToggleAll: (keys: string[]) => void;
  onExportCsv: () => void;
  onExportJson: () => void;
  onApply: () => void;
  onCopy: (text: string) => void;
  applying: boolean;
}

const CONFIDENCE_STYLES: Record<ConfidenceLevel, string> = {
  high: "bg-success/15 text-success",
  medium: "bg-warning/15 text-warning",
  low: "bg-destructive/15 text-destructive",
};

export const ResultsPanel = memo(function ResultsPanel({
  results,
  summary,
  selectedKeys,
  onToggleSelection,
  onToggleAll,
  onExportCsv,
  onExportJson,
  onApply,
  onCopy,
  applying,
}: ResultsPanelProps) {
  const [filter, setFilter] = useState<Filter>("all");

  const filtered = useMemo(() => {
    switch (filter) {
      case "all":
        return results;
      case "drift":
        return results.filter((r) => r.hasSignificantDrift);
      case "failed":
        return results.filter((r) => r.error || r.delayMs === null);
      default:
        return results.filter((r) => !r.error && confidenceLevel(r) === filter);
    }
  }, [results, filter]);

  // Only results with a measurement can be applied.
  const applicableKeys = useMemo(
    () =>
      filtered
        .filter((r) => r.delayMs !== null && !r.error && r.primaryPath && r.secondaryPath)
        .map(resultKey),
    [filtered],
  );

  const counts = useMemo(
    () => ({
      all: results.length,
      high: results.filter((r) => !r.error && confidenceLevel(r) === "high").length,
      medium: results.filter((r) => !r.error && confidenceLevel(r) === "medium").length,
      low: results.filter((r) => !r.error && confidenceLevel(r) === "low").length,
      drift: results.filter((r) => r.hasSignificantDrift).length,
      failed: results.filter((r) => r.error || r.delayMs === null).length,
    }),
    [results],
  );

  if (results.length === 0) return null;

  const allSelected =
    applicableKeys.length > 0 && applicableKeys.every((key) => selectedKeys.has(key));
  const selectedCount = selectedKeys.size;

  const filters: { id: Filter; label: string; count: number }[] = [
    { id: "all", label: "All", count: counts.all },
    { id: "high", label: "High", count: counts.high },
    { id: "medium", label: "Medium", count: counts.medium },
    { id: "low", label: "Low", count: counts.low },
    { id: "drift", label: "Drift", count: counts.drift },
    { id: "failed", label: "Failed", count: counts.failed },
  ];

  return (
    <section className="rounded-lg border border-border bg-card" aria-label="Results">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium text-foreground">Results</h2>
          {summary && (
            <span className="text-[11px] text-muted-foreground">
              {summary.matched} matched · {summary.failed} failed
              {summary.drifting > 0 && ` · ${summary.drifting} drifting`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onApply}
            disabled={selectedCount === 0 || applying}
            className="flex items-center gap-1.5 rounded bg-primary px-2.5 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-40"
            title="Write corrected files for the selected results"
          >
            <Wand2 className="h-3.5 w-3.5" aria-hidden />
            {applying ? "Writing…" : `Fix ${selectedCount || ""}`.trim()}
          </button>
          <button
            type="button"
            onClick={onExportCsv}
            className="flex items-center gap-1.5 rounded px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            <Download className="h-3.5 w-3.5" aria-hidden />
            CSV
          </button>
          <button
            type="button"
            onClick={onExportJson}
            className="rounded px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            JSON
          </button>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-1 border-b border-border px-4 py-2">
        {filters.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => setFilter(entry.id)}
            disabled={entry.count === 0 && entry.id !== "all"}
            className={`rounded px-2 py-1 text-[11px] transition-colors disabled:opacity-30 ${
              filter === entry.id
                ? "bg-secondary font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {entry.label} {entry.count > 0 && <span className="opacity-60">{entry.count}</span>}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <caption className="sr-only">Measured sync offsets</caption>
          <thead>
            <tr className="border-b border-border bg-secondary/40 text-muted-foreground">
              <th scope="col" className="w-8 px-3 py-2">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={() => onToggleAll(applicableKeys)}
                  disabled={applicableKeys.length === 0}
                  className="h-3 w-3 accent-primary"
                  aria-label="Select all applicable results"
                />
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">Video</th>
              <th scope="col" className="px-3 py-2 text-left font-medium">Audio</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Delay</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Drift</th>
              <th scope="col" className="px-3 py-2 text-center font-medium">Confidence</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Time</th>
              <th scope="col" className="w-8 px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((result) => {
              const key = resultKey(result);
              const level = confidenceLevel(result);
              const applicable =
                result.delayMs !== null && !result.error && !!result.primaryPath;
              const command = ffmpegCommandFor(result);

              return (
                <tr key={key} className="border-b border-border/50 last:border-0">
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selectedKeys.has(key)}
                      onChange={() => onToggleSelection(key)}
                      disabled={!applicable}
                      className="h-3 w-3 accent-primary"
                      aria-label={`Select ${result.videoFile}`}
                    />
                  </td>
                  <td className="max-w-[220px] px-3 py-2">
                    <span className="block truncate text-foreground" title={result.videoFile}>
                      {result.videoFile}
                    </span>
                  </td>
                  <td className="max-w-[200px] px-3 py-2">
                    <span
                      className="block truncate text-muted-foreground"
                      title={result.audioFile}
                    >
                      {result.audioFile}
                    </span>
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-foreground">
                    {result.error ? (
                      <span className="text-destructive" title={result.error}>
                        --
                      </span>
                    ) : (
                      formatDelay(result.delayMs)
                    )}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-right font-mono">
                    {result.hasSignificantDrift ? (
                      <span
                        className="inline-flex items-center gap-1 text-warning"
                        title={`Total drift across the file: ${formatDelay(result.totalDriftMs)}`}
                      >
                        <TrendingUp className="h-3 w-3" aria-hidden />
                        {formatDrift(result.driftMsPerS)}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">--</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-center">
                    {result.error ? (
                      <span
                        className="inline-flex items-center gap-1 rounded bg-destructive/15 px-2 py-0.5 text-[10px] text-destructive"
                        title={result.error}
                      >
                        <X className="h-2.5 w-2.5" aria-hidden />
                        Failed
                      </span>
                    ) : (
                      <span
                        className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium ${CONFIDENCE_STYLES[level]}`}
                        title={
                          result.confidence !== null
                            ? `Match confidence ${(result.confidence * 100).toFixed(0)}% from ${result.windowsUsed}/${result.windowsTotal} windows`
                            : undefined
                        }
                      >
                        {level === "high" && <Check className="h-2.5 w-2.5" aria-hidden />}
                        {level === "low" && <AlertTriangle className="h-2.5 w-2.5" aria-hidden />}
                        {level[0].toUpperCase() + level.slice(1)}
                      </span>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[10px] text-muted-foreground">
                    {formatElapsed(result.elapsedMs)}
                  </td>
                  <td className="px-3 py-2">
                    {command && (
                      <button
                        type="button"
                        onClick={() => onCopy(command)}
                        className="rounded p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                        title="Copy the equivalent ffmpeg command"
                      >
                        <Clipboard className="h-3 w-3" aria-hidden />
                        <span className="sr-only">Copy ffmpeg command for {result.videoFile}</span>
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {filtered.length === 0 && (
        <p className="px-4 py-6 text-center text-xs text-muted-foreground">
          No results match this filter.
        </p>
      )}
    </section>
  );
});
