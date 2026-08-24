import { Download, History, RotateCcw, Trash2, X } from "lucide-react";
import { memo } from "react";

import { Button, IconButton, Pill } from "@/components/ui";
import { formatHistoryDate } from "@/lib/storage";
import type { HistoryEntry } from "@/lib/types";

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onLoad: (entry: HistoryEntry) => void;
  onExport: (entry: HistoryEntry) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
  onClose: () => void;
}

/** Run history, as a drawer over the content rather than a permanent column. */
export const HistoryPanel = memo(function HistoryPanel({
  entries,
  onLoad,
  onExport,
  onDelete,
  onClear,
  onClose,
}: HistoryPanelProps) {
  return (
    <aside
      className="absolute inset-y-0 right-0 z-40 flex w-[288px] animate-drawer-in flex-col border-l border-border bg-card shadow-2xl"
      aria-label="Run history"
    >
      <header className="flex items-center gap-2 border-b border-border px-4 py-3.5">
        <div className="min-w-0 flex-1">
          <h2 className="text-[13px] font-semibold">History</h2>
          <p className="mt-px text-[11px] text-muted-foreground">
            {entries.length} run{entries.length === 1 ? "" : "s"}
          </p>
        </div>
        {entries.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            className="text-muted-foreground hover:text-destructive"
          >
            Clear
          </Button>
        )}
        <IconButton label="Close history" onClick={onClose}>
          <X className="h-4 w-4" aria-hidden />
        </IconButton>
      </header>

      <div className="flex-1 overflow-y-auto">
        {entries.length === 0 ? (
          <div className="flex flex-col items-center gap-2.5 px-6 py-12 text-center">
            <span className="grid h-[42px] w-[42px] place-items-center rounded-[11px] bg-sunken text-muted-foreground">
              <History className="h-5 w-5" aria-hidden />
            </span>
            <p className="text-[13px] font-semibold">No runs yet</p>
            <p className="max-w-[32ch] text-[11.5px] leading-relaxed text-muted-foreground">
              Completed analyses are saved here so you can reload or export them later.
            </p>
          </div>
        ) : (
          <ul className="flex flex-col gap-2 p-2.5">
            {entries.map((entry) => {
              const high = entry.summary?.high ?? 0;
              return (
                <li
                  key={entry.id}
                  className="group rounded-lg border border-border bg-card p-3 transition-colors hover:border-border-strong hover:shadow-sm"
                >
                  <div className="mb-1.5 flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate text-[11.5px] text-muted-foreground">
                      {formatHistoryDate(entry.date)}
                    </span>
                    <Pill tone={entry.mode === "movie" ? "warning" : "accent"}>
                      {entry.mode === "movie" ? "Movie" : "Series"}
                    </Pill>
                  </div>

                  <p className="mb-2.5 text-[12.5px] font-medium">
                    {entry.fileCount} file{entry.fileCount === 1 ? "" : "s"}
                    {high > 0 && (
                      <span className="font-normal text-muted-foreground">
                        {" "}
                        · {high} high confidence
                      </span>
                    )}
                  </p>

                  <div className="flex items-center gap-1.5">
                    <Button size="sm" onClick={() => onLoad(entry)} className="flex-1">
                      <RotateCcw className="h-3 w-3" aria-hidden />
                      Load
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => onExport(entry)}>
                      <Download className="h-3 w-3" aria-hidden />
                      Export
                    </Button>
                    <IconButton
                      label="Delete this run"
                      onClick={() => onDelete(entry.id)}
                      className="opacity-0 transition-opacity focus-visible:opacity-100 group-hover:opacity-100 hover:text-destructive"
                    >
                      <Trash2 className="h-3 w-3" aria-hidden />
                    </IconButton>
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
