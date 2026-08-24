import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
  Clipboard,
  Download,
  Gauge,
  Loader2,
  Play,
  Scissors,
  TrendingDown,
  TrendingUp,
  Wand2,
  X,
} from "lucide-react";
import { Fragment, memo, useMemo, useState } from "react";

import { ResultDetail } from "@/components/ResultDetail";
import { Button, Card, CardHeader, Checkbox, Pill } from "@/components/ui";
import { cx } from "@/lib/cx";
import {
  confidenceLevel,
  ffmpegCommandFor,
  formatDelay,
  formatDrift,
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
  /** Render and open a short aligned excerpt so a result can be judged by ear. */
  onPreview: (result: SyncResult) => void;
  /** Key of the result currently being rendered, if any. */
  previewingKey: string | null;
  applying: boolean;
  outputSuffix: string;
}

const CONFIDENCE_TONE = {
  high: "success",
  medium: "warning",
  low: "destructive",
} as const;

/** Summary tile. Reads at a glance before any row is examined. */
function Stat({
  label,
  value,
  tone,
  icon,
}: {
  label: string;
  value: number;
  tone: "success" | "warning" | "destructive" | "neutral";
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-3 shadow-sm">
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <span className="shrink-0" aria-hidden>
          {icon}
        </span>
        <span className="truncate">{label}</span>
      </div>
      <div
        className={cx(
          "tabular mt-1.5 text-[22px] font-bold leading-tight tracking-tight",
          tone === "success" && "text-success",
          tone === "warning" && "text-warning",
          tone === "destructive" && "text-destructive",
        )}
      >
        {value}
      </div>
    </div>
  );
}

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
  // Which rows have their measurement detail open. The engine reports far more
  // than fits a table row, and hiding it entirely meant the work was wasted.
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

  // Only measured results can be written.
  const applicableKeys = useMemo(
    () =>
      filtered
        // A different cut has no single delay that aligns it, so offering to
        // "fix" one would write a file that is still out of sync.
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
  const someSelected = applicableKeys.some((key) => selectedKeys.has(key));

  const filters: { id: Filter; label: string; count: number }[] = [
    { id: "all", label: "All", count: counts.all },
    { id: "high", label: "High", count: counts.high },
    { id: "medium", label: "Medium", count: counts.medium },
    { id: "low", label: "Low", count: counts.low },
    { id: "drift", label: "Drift", count: counts.drift },
    { id: "cut", label: "Different cut", count: counts.cut },
    { id: "failed", label: "Failed", count: counts.failed },
  ];

  const needsLook = counts.medium + counts.low;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        <Stat
          label="High confidence"
          value={counts.high}
          tone="success"
          icon={<Check className="h-3 w-3" />}
        />
        <Stat
          label="Needs a look"
          value={needsLook}
          tone={needsLook > 0 ? "warning" : "neutral"}
          icon={<AlertTriangle className="h-3 w-3" />}
        />
        <Stat
          label="Drifting"
          value={counts.drift}
          tone={counts.drift > 0 ? "warning" : "neutral"}
          icon={<TrendingUp className="h-3 w-3" />}
        />
        <Stat
          label="Failed"
          value={counts.failed}
          tone={counts.failed > 0 ? "destructive" : "neutral"}
          icon={<X className="h-3 w-3" />}
        />
      </div>

      <Card aria-label="Results">
        <CardHeader>
          <h2 className="text-[13px] font-semibold">Results</h2>
          {summary && (
            <span className="text-[11.5px] text-muted-foreground">
              {summary.matched} matched · {summary.failed} failed
            </span>
          )}
          <span className="flex-1" />
          <Button variant="ghost" size="sm" onClick={onExportCsv}>
            <Download className="h-3.5 w-3.5" aria-hidden />
            CSV
          </Button>
          <Button variant="ghost" size="sm" onClick={onExportJson}>
            JSON
          </Button>
        </CardHeader>

        <div
          className="flex items-center gap-1 overflow-x-auto border-b border-border px-3 py-2"
          role="tablist"
          aria-label="Filter results"
        >
          {filters.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              aria-selected={filter === entry.id}
              onClick={() => setFilter(entry.id)}
              disabled={entry.count === 0 && entry.id !== "all"}
              className={cx(
                "flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs transition-colors disabled:opacity-35",
                filter === entry.id
                  ? "bg-sunken font-semibold text-foreground"
                  : "text-muted-foreground hover:bg-sunken hover:text-foreground",
              )}
            >
              {entry.label}
              {entry.count > 0 && (
                <span className="tabular font-mono text-[10.5px] text-muted-foreground">
                  {entry.count}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full table-fixed text-xs">
            <caption className="sr-only">Measured sync offsets</caption>
            <colgroup>
              <col className="w-[38px]" />
              <col />
              <col className="w-[118px]" />
              <col className="w-[126px]" />
              <col className="w-[132px]" />
              <col className="w-[64px]" />
              <col className="w-[38px]" />
            </colgroup>
            <thead>
              <tr className="bg-elevated text-[10.5px] uppercase tracking-[0.05em] text-muted-foreground">
                <th scope="col" className="px-3 py-2.5 text-center font-semibold">
                  <Checkbox
                    checked={allSelected}
                    indeterminate={someSelected && !allSelected}
                    onChange={() => onToggleAll(applicableKeys)}
                    disabled={applicableKeys.length === 0}
                    label="Select all applicable results"
                  />
                </th>
                <th scope="col" className="px-3 py-2.5 text-left font-semibold">Pair</th>
                <th scope="col" className="px-3 py-2.5 text-right font-semibold">Delay</th>
                <th scope="col" className="px-3 py-2.5 text-right font-semibold">Drift</th>
                <th scope="col" className="px-3 py-2.5 text-center font-semibold">Confidence</th>
                <th scope="col" className="px-3 py-2.5 text-right font-semibold">Time</th>
                <th scope="col" className="px-3 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((result) => {
                const key = resultKey(result);
                const level = confidenceLevel(result);
                const applicable =
                  result.delayMs !== null &&
                  !result.error &&
                  !result.isLikelyCut &&
                  !!result.primaryPath;
                const command = ffmpegCommandFor(result);
                const selected = selectedKeys.has(key);
                const drifting = result.hasSignificantDrift;
                const DriftIcon = (result.driftMsPerS ?? 0) < 0 ? TrendingDown : TrendingUp;
                const isExpanded = expanded.has(key);
                const frames = frameOffset(result.delayMs, result.primaryFps);

                return (
                  <Fragment key={key}>
                  <tr
                    className={cx(
                      "group border-b border-border last:border-0",
                      selected ? "bg-accent" : "hover:bg-elevated",
                    )}
                  >
                    <td className="px-3 py-2.5 text-center">
                      <Checkbox
                        checked={selected}
                        onChange={() => onToggleSelection(key)}
                        disabled={!applicable}
                        label={`Select ${result.videoFile}`}
                      />
                    </td>

                    {/* Video and audio share one cell so the numbers get the width. */}
                    <td className="overflow-hidden px-3 py-2.5">
                      <span className="block truncate" title={result.videoFile}>
                        {result.videoFile}
                      </span>
                      {result.error ? (
                        <span
                          className="mt-0.5 block truncate text-[11px] text-destructive"
                          title={result.error}
                        >
                          {result.error}
                        </span>
                      ) : (
                        <span
                          className="mt-0.5 flex items-center gap-1 truncate text-[11px] text-muted-foreground"
                          title={result.audioFile}
                        >
                          <ArrowRight className="h-2.5 w-2.5 shrink-0" aria-hidden />
                          <span className="truncate">{result.audioFile}</span>
                        </span>
                      )}
                    </td>

                    <td className="tabular whitespace-nowrap px-3 py-2.5 text-right font-mono text-[13px] font-semibold">
                      {result.error ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        <>
                          {formatDelay(result.delayMs)}
                          {/* Frames are how editors judge whether an offset
                              matters; milliseconds alone carry no scale. */}
                          {frames !== null && (
                            <span className="block font-sans text-[10px] font-normal text-muted-foreground">
                              ≈ {frames > 0 ? "+" : ""}
                              {frames} frame{Math.abs(frames) === 1 ? "" : "s"}
                            </span>
                          )}
                        </>
                      )}
                    </td>

                    <td className="whitespace-nowrap px-3 py-2.5 text-right">
                      {result.isLikelyCut ? (
                        <Pill
                          tone="destructive"
                          title={
                            result.rateDiagnosis?.explanation ??
                            "The runtimes diverge too fast for a frame-rate difference."
                          }
                        >
                          <Scissors className="h-2.5 w-2.5" aria-hidden />
                          Different cut
                        </Pill>
                      ) : result.isRateMismatch ? (
                        <Pill
                          tone="warning"
                          title={result.rateDiagnosis?.explanation ?? undefined}
                        >
                          <Gauge className="h-2.5 w-2.5" aria-hidden />
                          {result.rateDiagnosis?.sourceFps && result.rateDiagnosis?.targetFps
                            ? `${result.rateDiagnosis.sourceFps}→${result.rateDiagnosis.targetFps}fps`
                            : "Speed"}
                        </Pill>
                      ) : drifting ? (
                        <Pill
                          tone="warning"
                          className="tabular font-mono"
                          title={`Total drift across the file: ${formatDelay(result.totalDriftMs)}`}
                        >
                          <DriftIcon className="h-2.5 w-2.5" aria-hidden />
                          {formatDrift(result.driftMsPerS)}
                        </Pill>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>

                    <td className="px-3 py-2.5 text-center">
                      {result.error ? (
                        <Pill tone="destructive" title={result.error}>
                          <X className="h-2.5 w-2.5" aria-hidden />
                          Failed
                        </Pill>
                      ) : (
                        <Pill
                          tone={CONFIDENCE_TONE[level]}
                          title={
                            result.confidence !== null
                              ? `Matched ${result.windowsUsed}/${result.windowsTotal} windows`
                              : undefined
                          }
                        >
                          {level === "high" && <Check className="h-2.5 w-2.5" aria-hidden />}
                          {level === "low" && <AlertTriangle className="h-2.5 w-2.5" aria-hidden />}
                          {level[0].toUpperCase() + level.slice(1)}
                          {result.confidence !== null && (
                            <span className="tabular font-mono">
                              · {(result.confidence * 100).toFixed(0)}%
                            </span>
                          )}
                        </Pill>
                      )}
                    </td>

                    <td className="tabular whitespace-nowrap px-3 py-2.5 text-right font-mono text-[11px] text-muted-foreground">
                      {formatElapsed(result.elapsedMs)}
                    </td>

                    <td className="px-3 py-2.5">
                      <div className="flex items-center justify-end gap-0.5">
                        {command && (
                          <button
                            type="button"
                            onClick={() => onCopy(command)}
                            title="Copy the equivalent ffmpeg command"
                            className="rounded p-1 text-muted-foreground opacity-0 transition hover:bg-sunken hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100"
                          >
                            <Clipboard className="h-3 w-3" aria-hidden />
                            <span className="sr-only">
                              Copy ffmpeg command for {result.videoFile}
                            </span>
                          </button>
                        )}
                        {applicable && (
                          <button
                            type="button"
                            onClick={() => onPreview(result)}
                            disabled={previewingKey !== null}
                            title="Play a short excerpt with this delay applied"
                            className="rounded p-1 text-muted-foreground opacity-0 transition hover:bg-sunken hover:text-foreground focus-visible:opacity-100 disabled:opacity-40 group-hover:opacity-100"
                          >
                            {previewingKey === key ? (
                              <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
                            ) : (
                              <Play className="h-3 w-3" aria-hidden />
                            )}
                            <span className="sr-only">
                              Preview {result.videoFile} with this delay
                            </span>
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => toggleExpanded(key)}
                          aria-expanded={isExpanded}
                          className="rounded p-1 text-muted-foreground transition hover:bg-sunken hover:text-foreground"
                        >
                          <ChevronDown
                            className={cx(
                              "h-3.5 w-3.5 transition-transform",
                              isExpanded && "rotate-180",
                            )}
                            aria-hidden
                          />
                          <span className="sr-only">
                            {isExpanded ? "Hide" : "Show"} details for {result.videoFile}
                          </span>
                        </button>
                      </div>
                    </td>
                  </tr>
                  {isExpanded && <ResultDetail result={result} columnCount={7} />}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        {filtered.length === 0 && (
          <p className="px-4 py-8 text-center text-xs text-muted-foreground">
            No results match this filter.
          </p>
        )}

        {/* The payoff action, given the weight it deserves. */}
        <div className="flex flex-wrap items-center gap-3 rounded-b-xl border-t border-border bg-elevated px-4 py-3">
          <p className="min-w-0 flex-1 text-[12.5px] text-muted-foreground">
            {selectedCount > 0 ? (
              <>
                <span className="font-semibold text-foreground">
                  {selectedCount} selected
                </span>{" "}
                — corrected copies are written alongside the originals with{" "}
                <span className="font-mono">{outputSuffix}</span>. Source files are never
                modified.
              </>
            ) : (
              "Select the results you want to write corrected files for."
            )}
          </p>
          <Button
            variant="primary"
            onClick={onApply}
            disabled={selectedCount === 0 || applying}
            title="Write corrected files for the selected results"
          >
            <Wand2 className="h-3.5 w-3.5" aria-hidden />
            {applying
              ? "Writing…"
              : selectedCount > 0
                ? `Fix ${selectedCount} file${selectedCount === 1 ? "" : "s"}`
                : "Fix"}
          </Button>
        </div>
      </Card>
    </div>
  );
});
