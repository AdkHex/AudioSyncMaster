import { memo, useMemo, useState } from "react";

import { CheckRow } from "@/components/subsync/controls";
import { cx } from "@/lib/cx";
import { cueCps, formatCueTime, isLowConfidence, plainText } from "@/lib/subsync/format";
import type { CueData } from "@/lib/subsync/types";

const ROW = 42;
const OVERSCAN = 8;
const GRID = "44px 84px 84px 52px minmax(0,1fr) 40px";
/** Above this a line is hard to read for most audiences. */
const FAST_CPS = 20;

/** The cues of a subtitle: timing, text and reading speed, with the lines
 *  an engine was unsure of highlighted. Only the rows in view are drawn, so
 *  a feature-length file of thousands of cues scrolls as easily as ten. */
export const CueTable = memo(function CueTable({
  cues,
  total,
  height = 340,
  label = "Cues",
}: {
  cues: CueData[];
  /** Cues in the whole file, when only the first ones were sent. */
  total?: number;
  height?: number;
  label?: string;
}) {
  const [scrollTop, setScrollTop] = useState(0);
  const [flaggedOnly, setFlaggedOnly] = useState(false);

  const numbered = useMemo(() => cues.map((cue, index) => ({ cue, number: index + 1 })), [cues]);
  const flagged = useMemo(() => numbered.filter(({ cue }) => isLowConfidence(cue)), [numbered]);
  const rows = flaggedOnly ? flagged : numbered;

  const first = Math.max(0, Math.floor(scrollTop / ROW) - OVERSCAN);
  const last = Math.min(rows.length, Math.ceil((scrollTop + height) / ROW) + OVERSCAN);
  const visible = rows.slice(first, last);

  return (
    <div>
      <div className="mb-1.5 flex items-center gap-3 text-[11px] text-muted-foreground">
        <span>
          {cues.length.toLocaleString()} cue{cues.length === 1 ? "" : "s"}
          {total && total > cues.length ? ` (first of ${total.toLocaleString()})` : ""}
        </span>
        {flagged.length > 0 && (
          <CheckRow
            className="text-[11px]"
            label={`Only the ${flagged.length} to review`}
            checked={flaggedOnly}
            onChange={(value) => {
              setFlaggedOnly(value);
              setScrollTop(0);
            }}
          />
        )}
      </div>

      <div role="table" aria-label={label} aria-rowcount={rows.length + 1} className="overflow-hidden rounded-lg border border-border">
        <div role="rowgroup">
          <div
            role="row"
            className="grid border-b border-border bg-elevated px-2 py-1 text-[10.5px] font-medium text-muted-foreground"
            style={{ gridTemplateColumns: GRID }}
          >
            <span role="columnheader">#</span>
            <span role="columnheader">In</span>
            <span role="columnheader">Out</span>
            <span role="columnheader" className="text-right pr-2">Dur</span>
            <span role="columnheader">Text</span>
            <span role="columnheader" className="text-right">CPS</span>
          </div>
        </div>
        <div
          role="rowgroup"
          tabIndex={0}
          aria-label={`${label}, scrollable`}
          className="relative overflow-y-auto focus:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          style={{ height: Math.min(height, Math.max(ROW, rows.length * ROW)) }}
          onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        >
          <div style={{ height: rows.length * ROW }} />
          {visible.map(({ cue, number }, offset) => {
            const cps = cueCps(cue);
            const low = isLowConfidence(cue);
            const text = plainText(cue.text);
            return (
              <div
                key={number}
                role="row"
                aria-rowindex={first + offset + 2}
                className={cx(
                  "absolute inset-x-0 grid items-center border-b border-border/60 px-2 text-[11.5px]",
                  low && "bg-warning/10",
                )}
                style={{ top: (first + offset) * ROW, height: ROW, gridTemplateColumns: GRID }}
                title={low ? `Confidence ${Math.round((cue.confidence ?? 0) * 100)}%: worth checking` : undefined}
              >
                <span role="cell" className="tabular font-mono text-[10.5px] text-muted-foreground">{number}</span>
                <span role="cell" className="tabular font-mono text-[10.5px]">{formatCueTime(cue.start)}</span>
                <span role="cell" className="tabular font-mono text-[10.5px]">{formatCueTime(cue.end)}</span>
                <span role="cell" className="tabular pr-2 text-right font-mono text-[10.5px] text-muted-foreground">
                  {(cue.end - cue.start).toFixed(2)}
                </span>
                <span role="cell" className="line-clamp-2 whitespace-pre-line leading-[1.3]" title={text}>
                  {cue.forced && <span className="mr-1 text-[10px] text-primary">forced</span>}
                  {text}
                </span>
                <span
                  role="cell"
                  className={cx(
                    "tabular text-right font-mono text-[10.5px]",
                    cps !== null && cps > FAST_CPS ? "text-warning" : "text-muted-foreground",
                  )}
                >
                  {cps === null ? "—" : cps.toFixed(1)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
});
