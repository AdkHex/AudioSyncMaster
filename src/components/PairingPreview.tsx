import { AlertTriangle, ArrowRight } from "lucide-react";
import { memo } from "react";

import { Card, CardHeader, Notice, Pill, Spinner } from "@/components/ui";
import type { PairingReport } from "@/lib/types";

interface PairingPreviewProps {
  pairing: PairingReport | null;
  loading: boolean;
}

/** Shows exactly what will be compared, and how each pair was arrived at,
 *  before any work starts. In series mode this is where the risk lives, so
 *  episode keys and fuzzy scores are shown rather than hidden in tooltips. */
export const PairingPreview = memo(function PairingPreview({
  pairing,
  loading,
}: PairingPreviewProps) {
  if (loading) {
    return (
      <Card className="flex items-center gap-2.5 px-4 py-3">
        <Spinner className="h-3.5 w-3.5" />
        <p className="text-xs text-muted-foreground">Working out pairings…</p>
      </Card>
    );
  }

  if (!pairing) return null;

  const { pairs, unmatchedPrimary, unmatchedSecondary, method, warning } = pairing;

  return (
    <div className="space-y-3">
      {warning && (
        <Notice icon={<AlertTriangle className="h-3.5 w-3.5" aria-hidden />}>{warning}</Notice>
      )}

      <Card>
        <CardHeader>
          <h2 className="text-[13px] font-semibold">Pairing preview</h2>
          <span className="flex-1" />
          <span className="text-[11px] text-muted-foreground">matched by {method}</span>
          <Pill>
            {pairs.length} pair{pairs.length === 1 ? "" : "s"}
          </Pill>
        </CardHeader>

        {pairs.length > 0 && (
          <ul className="max-h-56 divide-y divide-border overflow-y-auto">
            {pairs.map((pair) => (
              <li
                key={`${pair.primaryPath}-${pair.secondaryPath}`}
                className="flex items-center gap-2.5 px-4 py-2.5 text-xs"
              >
                {/* The episode key is what the match was actually made on. */}
                {pair.key && (
                  <span className="shrink-0 rounded border border-border bg-sunken px-1.5 py-0.5 font-mono text-[10.5px] font-semibold text-secondary-foreground">
                    {pair.key}
                  </span>
                )}
                <span className="min-w-0 flex-1 truncate" title={pair.primaryName}>
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
                  <Pill
                    tone="warning"
                    title="Paired by filename similarity rather than an episode number"
                  >
                    {Math.round(pair.score * 100)}% name match
                  </Pill>
                )}
              </li>
            ))}
          </ul>
        )}

        {(unmatchedPrimary.length > 0 || unmatchedSecondary.length > 0) && (
          <div className="space-y-1 border-t border-border px-4 py-2.5 text-[11px]">
            {unmatchedPrimary.length > 0 && (
              <p className="text-destructive">
                Unpaired:{" "}
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
      </Card>
    </div>
  );
});
