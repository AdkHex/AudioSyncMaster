import { Download, History, RotateCcw, Trash2 } from "lucide-react";
import { memo } from "react";

import { formatHistoryDate } from "@/lib/storage";
import type { HistoryEntry } from "@/lib/types";

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onLoad: (entry: HistoryEntry) => void;
  onExport: (entry: HistoryEntry) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
}

export const HistoryPanel = memo(function HistoryPanel({
  entries,
  onLoad,
  onExport,
  onDelete,
  onClear,
}: HistoryPanelProps) {
  return (
    <aside
      className="flex w-72 shrink-0 flex-col border-l border-border bg-card/60"
      aria-label="Run history"
    >
      <header className="flex items-center justify-between border-b border-border px-3 py-2.5">
        <div>
          <h2 className="text-sm font-medium text-foreground">History</h2>
          <p className="text-[10px] text-muted-foreground">
            {entries.length} run{entries.length === 1 ? "" : "s"}
          </p>
        </div>
        {entries.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            className="text-[11px] text-muted-foreground transition-colors hover:text-destructive"
          >
            Clear
          </button>
        )}
      </header>

      <div className="flex-1 overflow-y-auto">
        {entries.length === 0 ? (
          <div className="flex flex-col items-center gap-2 px-4 py-10 text-center">
            <History className="h-7 w-7 text-muted-foreground/30" aria-hidden />
            <p className="text-xs text-muted-foreground">
              Completed runs are saved here
            </p>
          </div>
        ) : (
          <ul className="space-y-1 p-2">
            {entries.map((entry) => {
              const high = entry.summary?.high ?? 0;
              return (
                <li
                  key={entry.id}
                  className="group rounded-lg bg-secondary/30 p-2.5 transition-colors hover:bg-secondary/50"
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="truncate text-[10px] text-muted-foreground">
                      {formatHistoryDate(entry.date)}
                    </span>
                    <span
                      className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] ${
                        entry.mode === "movie"
                          ? "bg-warning/15 text-warning"
                          : "bg-primary/15 text-primary"
                      }`}
                    >
                      {entry.mode === "movie" ? "Movie" : "Series"}
                    </span>
                  </div>

                  <p className="mb-2 text-xs text-foreground">
                    {entry.fileCount} file{entry.fileCount === 1 ? "" : "s"}
                    {high > 0 && (
                      <span className="text-muted-foreground"> · {high} high confidence</span>
                    )}
                  </p>

                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => onLoad(entry)}
                      className="flex flex-1 items-center justify-center gap-1 rounded py-1 text-[10px] text-primary transition-colors hover:bg-primary/10"
                    >
                      <RotateCcw className="h-3 w-3" aria-hidden />
                      Load
                    </button>
                    <button
                      type="button"
                      onClick={() => onExport(entry)}
                      className="flex flex-1 items-center justify-center gap-1 rounded py-1 text-[10px] text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                    >
                      <Download className="h-3 w-3" aria-hidden />
                      Export
                    </button>
                    <button
                      type="button"
                      onClick={() => onDelete(entry.id)}
                      className="rounded p-1 text-muted-foreground opacity-0 transition-all hover:bg-destructive/10 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
                    >
                      <Trash2 className="h-3 w-3" aria-hidden />
                      <span className="sr-only">Delete this run</span>
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
});
