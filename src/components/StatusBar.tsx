import { memo } from "react";

import type { RunSummary, SyncMode } from "@/lib/types";

interface StatusBarProps {
  mode: SyncMode;
  pairCount: number;
  resultCount: number;
  summary: RunSummary | null;
  busy: boolean;
  version: string;
}

const MODE_LABEL: Record<SyncMode, string> = {
  movie: "Movies",
  series: "Series",
  compare: "Compare",
};

/** The persistent footer: what is loaded, and the two shortcuts that matter.
 *
 *  Everything here was previously repeated inside the main column, where it
 *  competed with the results for attention. */
export const StatusBar = memo(function StatusBar({
  mode,
  pairCount,
  resultCount,
  summary,
  busy,
  version,
}: StatusBarProps) {
  const left: string[] = [MODE_LABEL[mode]];
  if (pairCount > 0) left.push(`${pairCount} pair${pairCount === 1 ? "" : "s"}`);
  if (resultCount > 0) {
    left.push(`${resultCount} result${resultCount === 1 ? "" : "s"}`);
    if (summary?.drifting) left.push(`${summary.drifting} drifting`);
    if (summary?.failed) left.push(`${summary.failed} failed`);
  }

  return (
    <footer className="flex h-[26px] shrink-0 items-center gap-3 border-t border-border px-[18px] text-[11px] text-muted-foreground">
      <span className="min-w-0 truncate">{left.join("  ·  ")}</span>
      <span className="flex-1" />
      <span className="hidden items-center gap-3 sm:flex">
        <Shortcut keys={busy ? "esc" : "↵"} label={busy ? "stop" : "analyse"} />
        <Shortcut keys="&#8963;," label="settings" />
      </span>
      <span className="tabular font-mono">v{version}</span>
    </footer>
  );
});

function Shortcut({ keys, label }: { keys: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <kbd className="rounded border border-border-strong bg-elevated px-1.5 py-px font-mono text-[10px]">
        {keys}
      </kbd>
      {label}
    </span>
  );
}
