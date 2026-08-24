import { Clipboard, Download, Loader2, Play } from "lucide-react";
import { Fragment, memo, useMemo, useState } from "react";

import { ResultDetail } from "@/components/ResultDetail";
import { Tag } from "@/components/ui";
import { cx } from "@/lib/cx";
import {
  confidenceLevel,
  ffmpegCommandFor,
  formatDelay,
  formatElapsed,
  frameOffset,
  resultKey,
  type ConfidenceLevel,
  type RunSummary,
  type SyncResult,
} from "@/lib/types";

type Filter = "all" | ConfidenceLevel | "drift" | "cut" | "failed";

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
  onPreview: (result: SyncResult) => void;
  previewingKey: string | null;
  applying: boolean;
  outputSuffix: string;
}

/** Column template shared by the header and every row, so they cannot drift. */
const GRID = "34px minmax(0,1fr) 130px 120px 96px 60px";

const CONFIDENCE_TONE: Record<ConfidenceLevel, "success" | "warning" | "destructive"> = {
  high: "success",
  medium: "warning",
  low: "destructive",
};

/** The measured results, as a list rather than a table.
 *
 *  A table brought a sticky header, cell borders and hover-only icon buttons on
 *  every row. A grid gives the same alignment with one hairline between rows,
 *  and the row actions live inside the expanded detail so nothing appears and
 *  disappears under the pointer. */
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
  onPreview,
  previewingKey,
  applying,
  outputSuffix,
}: ResultsPanelProps) {
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggleExpanded = (key: string) =>
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const counts = useMemo(
    () => ({
      all: results.length,
      high: results.filter((r) => !r.error && confidenceLevel(r) === "high").length,
      medium: results.filter((r) => !r.error && confidenceLevel(r) === "medium").length,
      low: results.filter((r) => !r.error && confidenceLevel(r) === "low").length,
      drift: results.filter((r) => r.hasSignificantDrift && !r.isLikelyCut).length,
      cut: results.filter((r) => r.isLikelyCut).length,
      failed: results.filter((r) => r.error || r.delayMs === null).length,
    }),
    [results],
  );

  const filtered = useMemo(() => {
    switch (filter) {
      case "all":
        return results;
      case "drift":
        return results.filter((r) => r.hasSignificantDrift && !r.isLikelyCut);
      case "cut":
        return results.filter((r) => r.isLikelyCut);
      case "failed":
        return results.filter((r) => r.error || r.delayMs === null);
      default:
        return results.filter((r) => !r.error && confidenceLevel(r) === filter);
    }
  }, [results, filter]);

  // A different cut has no single delay that aligns it, so writing a "fixed"
  // copy would produce a file that is still out of sync.
  const applicableKeys = useMemo(
    () =>
      filtered
        .filter(
          (r) =>
            r.delayMs !== null && !r.error && !r.isLikelyCut && r.primaryPath && r.secondaryPath,
        )
        .map(resultKey),
    [filtered],
  );

  if (results.length === 0) return null;

  const selectedCount = selectedKeys.size;
  const allSelected =
    applicableKeys.length > 0 && applicableKeys.every((key) => selectedKeys.has(key));

  const filters: { id: Filter; label: string; count: number }[] = [
    { id: "all", label: "All", count: counts.all },
    { id: "high", label: "High", count: counts.high },
    { id: "medium", label: "Medium", count: counts.medium },
    { id: "low", label: "Low", count: counts.low },
    { id: "drift", label: "Drift", count: counts.drift },
    { id: "cut", label: "Different cut", count: counts.cut },
    { id: "failed", label: "Failed", count: counts.failed },
  ];

  return (
    <>
      <div className="flex h-11 shrink-0 items-center gap-4 border-b border-border px-[18px]">
        <h2 className="text-[13px] font-semibold">Results</h2>

        <div className="flex gap-[3px]" role="tablist" aria-label="Filter results">
          {filters.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              aria-selected={filter === entry.id}
              onClick={() => setFilter(entry.id)}
              disabled={entry.count === 0 && entry.id !== "all"}
              className={cx(
                "rounded-md px-2.5 py-1 text-xs transition-colors",
                "disabled:cursor-default disabled:opacity-40",
                filter === entry.id
                  ? "bg-elevated font-medium text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {entry.label}
              {entry.count > 0 && <span className="tabular ml-1.5">{entry.count}</span>}
            </button>
          ))}
        </div>

        <span className="flex-1" />

        {summary && (
          <span className="text-[11.5px] text-muted-foreground">
            {summary.matched} matched
            {summary.failed > 0 && ` · ${summary.failed} failed`}
          </span>
        )}

        <button
          type="button"
          onClick={onExportCsv}
          title="Export as CSV"
          className="grid h-7 w-7 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-elevated hover:text-foreground"
        >
          <Download className="h-[15px] w-[15px]" aria-hidden />
          <span className="sr-only">Export as CSV</span>
        </button>
        <button
          type="button"
          onClick={onExportJson}
          className="rounded-md px-1.5 py-1 text-[11.5px] text-muted-foreground transition-colors hover:text-foreground"
        >
          JSON
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div
          className="grid items-center gap-3.5 px-[18px] pb-2 pt-2.5 text-[11px] text-muted-foreground"
          style={{ gridTemplateColumns: GRID }}
        >
          <input
            type="checkbox"
            checked={allSelected}
            onChange={() => onToggleAll(applicableKeys)}
            disabled={applicableKeys.length === 0}
            className="h-[15px] w-[15px] accent-primary"
            aria-label="Select all applicable results"
          />
          <span>File</span>
          <span className="text-right">Delay</span>
          <span className="text-center">Drift</span>
          <span className="text-center">Confidence</span>
          <span className="text-right">Time</span>
        </div>

        {filtered.map((result) => {
          const key = resultKey(result);
          const level = confidenceLevel(result);
          const isExpanded = expanded.has(key);
          const frames = frameOffset(result.delayMs, result.primaryFps);
          const command = ffmpegCommandFor(result);
          const applicable =
            result.delayMs !== null && !result.error && !result.isLikelyCut && !!result.primaryPath;

          return (
            <Fragment key={key}>
              <div
                role="button"
                tabIndex={0}
                aria-expanded={isExpanded}
                onClick={() => toggleExpanded(key)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    toggleExpanded(key);
                  }
                }}
                className={cx(
                  "grid cursor-pointer items-center gap-3.5 border-t border-border px-[18px] py-3",
                  isExpanded ? "bg-primary/[0.055]" : "hover:bg-foreground/[0.022]",
                )}
                style={{ gridTemplateColumns: GRID }}
              >
                <input
                  type="checkbox"
                  checked={selectedKeys.has(key)}
                  onChange={() => onToggleSelection(key)}
                  onClick={(event) => event.stopPropagation()}
                  disabled={!applicable}
                  className="h-[15px] w-[15px] accent-primary"
                  aria-label={`Select ${result.videoFile}`}
                />

                <div className="min-w-0">
                  <p className="truncate text-[13px]" title={result.videoFile}>
                    {result.videoFile}
                  </p>
                  <p
                    className={cx(
                      "mt-[3px] truncate text-[11.5px]",
                      result.error ? "text-destructive" : "text-muted-foreground",
                    )}
                    title={result.error ?? result.audioFile}
                  >
                    {result.error ?? result.audioFile}
                  </p>
                </div>

                <div className="tabular text-right font-mono">
                  {result.error || result.delayMs === null ? (
                    <span className="text-muted-foreground">&mdash;</span>
                  ) : (
                    <>
                      <span className="text-sm font-semibold">
                        {formatDelay(result.delayMs)}
                      </span>
                      {frames !== null && (
                        <span className="block text-[10.5px] text-muted-foreground">
                          {frames > 0 ? "+" : ""}
                          {frames} frames
                        </span>
                      )}
                    </>
                  )}
                </div>

                <div className="text-center">
                  {result.isLikelyCut ? (
                    <Tag tone="destructive">Different cut</Tag>
                  ) : result.isRateMismatch && result.rateDiagnosis?.sourceFps ? (
                    <Tag tone="warning">
                      {result.rateDiagnosis.sourceFps} &rarr; {result.rateDiagnosis.targetFps} fps
                    </Tag>
                  ) : result.hasSignificantDrift ? (
                    <Tag tone="warning" className="tabular font-mono">
                      {result.driftMsPerS?.toFixed(3)} ms/s
                    </Tag>
                  ) : (
                    <span className="text-[11.5px] text-muted-foreground">&mdash;</span>
                  )}
                </div>

                <div className="text-center">
                  {result.error ? (
                    <Tag tone="destructive">Failed</Tag>
                  ) : (
                    <Tag tone={CONFIDENCE_TONE[level]}>
                      {level[0].toUpperCase() + level.slice(1)}
                      {result.confidence !== null && (
                        <span className="tabular">
                          {" · "}
                          {(result.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </Tag>
                  )}
                </div>

                <span className="tabular text-right font-mono text-[11.5px] text-muted-foreground">
                  {formatElapsed(result.elapsedMs)}
                </span>
              </div>

              {isExpanded && (
                <ResultDetail
                  result={result}
                  actions={
                    <>
                      {applicable && (
                        <button
                          type="button"
                          onClick={() => onPreview(result)}
                          disabled={previewingKey !== null}
                          className="flex items-center gap-1.5 rounded-[7px] border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:text-foreground disabled:opacity-40"
                        >
                          {previewingKey === key ? (
                            <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
                          ) : (
                            <Play className="h-3 w-3 fill-current" aria-hidden />
                          )}
                          Preview
                        </button>
                      )}
                      {command && (
                        <button
                          type="button"
                          onClick={() => onCopy(command)}
                          className="flex items-center gap-1.5 rounded-[7px] border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:text-foreground"
                        >
                          <Clipboard className="h-3 w-3" aria-hidden />
                          Copy ffmpeg command
                        </button>
                      )}
                    </>
                  }
                />
              )}
            </Fragment>
          );
        })}

        {filtered.length === 0 && (
          <p className="px-[18px] py-8 text-center text-xs text-muted-foreground">
            No results match this filter.
          </p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-3.5 border-t border-border px-[18px] py-3.5">
        <p className="flex-1 text-[12.5px] text-muted-foreground">
          {selectedCount > 0 ? (
            <>
              <b className="font-semibold text-foreground">{selectedCount} selected</b> &mdash; a
              corrected copy is written next to the original with{" "}
              <span className="font-mono">{outputSuffix}</span>. Sources are never changed.
            </>
          ) : (
            "Select the results you want corrected files for."
          )}
        </p>
        <button
          type="button"
          onClick={onApply}
          disabled={selectedCount === 0 || applying}
          className="rounded-lg bg-primary px-[18px] py-[9px] text-[12.5px] font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:bg-elevated disabled:text-muted-foreground"
        >
          {applying ? "Writing…" : selectedCount > 0 ? `Fix ${selectedCount}` : "Fix"}
        </button>
      </div>
    </>
  );
});
