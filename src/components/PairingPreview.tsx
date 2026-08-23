import { AlertTriangle, ArrowRight, Link2 } from "lucide-react";
import { memo } from "react";

import type { PairingReport } from "@/lib/types";

interface PairingPreviewProps {
  pairing: PairingReport | null;
  loading: boolean;
}

/** Shows exactly what will be compared, and how each pair was arrived at,
 *  before any work starts. */
export const PairingPreview = memo(function PairingPreview({
  pairing,
  loading,
}: PairingPreviewProps) {
  if (loading) {
    return (
      <section className="rounded-lg border border-border bg-card px-4 py-3">
        <p className="text-xs text-muted-foreground">Working out pairings…</p>
      </section>
    );
  }

  if (!pairing) return null;

  const { pairs, unmatchedPrimary, unmatchedSecondary, method, warning } = pairing;

  return (
    <section className="rounded-lg border border-border bg-card" aria-label="Pairing preview">
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Link2 className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
          <h2 className="text-sm font-medium text-foreground">Pairing preview</h2>
        </div>
        <span className="text-[11px] text-muted-foreground">
          {pairs.length} pair{pairs.length === 1 ? "" : "s"} · matched by {method}
        </span>
      </header>

      {warning && (
        <div className="flex items-start gap-2 border-b border-border bg-warning/10 px-4 py-2 text-[11px] text-warning">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
          <span>{warning}</span>
        </div>
      )}

      {pairs.length > 0 && (
        <ul className="max-h-52 divide-y divide-border/50 overflow-y-auto">
          {pairs.map((pair) => (
            <li
              key={`${pair.primaryPath}-${pair.secondaryPath}`}
              className="flex items-center gap-2 px-4 py-1.5 text-xs"
            >
              <span className="min-w-0 flex-1 truncate text-foreground" title={pair.primaryName}>
                {pair.primaryName}
              </span>
              <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
              <span
                className="min-w-0 flex-1 truncate text-muted-foreground"
                title={pair.secondaryName}
              >
                {pair.secondaryName}
              </span>
              {pair.score < 1 && (
                <span
                  className="shrink-0 rounded bg-warning/15 px-1.5 py-0.5 text-[10px] text-warning"
                  title="Paired by filename similarity rather than an episode number"
                >
                  {Math.round(pair.score * 100)}%
                </span>
              )}
            </li>
          ))}
        </ul>
      )}

      {(unmatchedPrimary.length > 0 || unmatchedSecondary.length > 0) && (
        <div className="space-y-1 border-t border-border px-4 py-2 text-[11px]">
          {unmatchedPrimary.length > 0 && (
            <p className="text-destructive">
              {unmatchedPrimary.length} video file
              {unmatchedPrimary.length === 1 ? "" : "s"} with no audio match:{" "}
              <span className="text-muted-foreground">
                {unmatchedPrimary.slice(0, 3).join(", ")}
                {unmatchedPrimary.length > 3 && ` +${unmatchedPrimary.length - 3} more`}
              </span>
            </p>
          )}
          {unmatchedSecondary.length > 0 && (
            <p className="text-muted-foreground">
              {unmatchedSecondary.length} audio file
              {unmatchedSecondary.length === 1 ? "" : "s"} unused
            </p>
          )}
        </div>
      )}
    </section>
  );
});
