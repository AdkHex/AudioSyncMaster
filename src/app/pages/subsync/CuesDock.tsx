/** The Cues tab of Subsync's dock: the selected file's cues (or, after a
 *  run, the result's), with the lines worth a second look flagged. OCR
 *  results list how sure the engine was of each line instead, and a line
 *  can be picked to check it in the side panel.
 *
 *  Only the rows in view are drawn, so a feature-length file of thousands
 *  of cues scrolls as easily as ten. */

import { WarningFilled } from "@fluentui/react-icons";
import { memo, useEffect, useMemo, useRef, useState } from "react";

import { cueCps, formatCueTime } from "@/lib/subsync/format";
import type { CueData } from "@/lib/subsync/types";
import { Tr, cx } from "@/ui/kit";

import { FAST_CPS, cueLine, needsCheck, type CueMode } from "./rows";

const OVERSCAN = 8;

export const CueTable = memo(function CueTable({
  cues,
  mode,
  minConfidence,
  onlyCheck,
  reviewed,
  edits,
  selected,
  onSelect,
  label = "Cues",
  viewport = 190,
  rowHeight: ROW = 30,
  reveal = null,
}: {
  cues: CueData[];
  mode: CueMode;
  minConfidence: number;
  onlyCheck: boolean;
  /** OCR: lines already confirmed. */
  reviewed?: ReadonlySet<number>;
  /** OCR: corrected text by line index. */
  edits?: Readonly<Record<number, string>>;
  selected?: number | null;
  onSelect?: (index: number) => void;
  label?: string;
  /** Height of the scrolling area before it is measured. */
  viewport?: number;
  /** 30 px in the dock; the OCR review's Lines list has the table's 36. */
  rowHeight?: number;
  /** A line to bring into view when the cues arrive: the first to check. */
  reveal?: number | null;
}) {
  const body = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(viewport);

  useEffect(() => {
    const el = body.current;
    if (!el) return;
    const measure = () => setHeight(el.clientHeight || viewport);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [viewport]);

  const rows = useMemo(() => {
    const all = cues.map((cue, index) => ({ cue, index }));
    if (!onlyCheck) return all;
    return all.filter(({ cue, index }) => !reviewed?.has(index) && needsCheck(cue, mode, minConfidence));
  }, [cues, onlyCheck, reviewed, mode, minConfidence]);

  // "Next to check" can pick a line out of view: bring it into the middle.
  const position = selected == null ? -1 : rows.findIndex((row) => row.index === selected);
  useEffect(() => {
    const el = body.current;
    if (!el || position < 0) return;
    const top = position * ROW;
    if (top < el.scrollTop || top + ROW > el.scrollTop + el.clientHeight) {
      el.scrollTop = Math.max(0, top - el.clientHeight / 2 + ROW / 2);
    }
  }, [position, ROW]);

  // New cues open on the first line worth a look, a few rows down.
  const revealRef = useRef(reveal);
  revealRef.current = reveal;
  useEffect(() => {
    const el = body.current;
    const at = revealRef.current;
    if (!el || at == null) return;
    el.scrollTop = Math.max(0, (at - 4) * ROW);
    setScrollTop(el.scrollTop);
  }, [cues, ROW]);

  const first = Math.max(0, Math.floor(scrollTop / ROW) - OVERSCAN);
  const last = Math.min(rows.length, Math.ceil((scrollTop + height) / ROW) + OVERSCAN);
  const visible = rows.slice(first, last);

  const ocr = mode === "ocr";
  const cols = ocr ? "44px 96px minmax(0,1fr) 64px" : "44px 96px 96px minmax(0,1fr) 48px";
  const head = ocr ? ["#", "In", "Text", " Sure"] : ["#", "In", "Out", "Text", " CPS"];

  return (
    <div className="tbl" role="table" aria-label={label} aria-rowcount={rows.length + 1} style={{ ["--cols" as string]: cols }}>
      <div className="th" role="row">
        {head.map((h) => (
          <span key={h} role="columnheader" className={cx(h.startsWith(" ") && "r")}>
            {h.trim()}
          </span>
        ))}
      </div>
      <div className="tb" role="rowgroup" ref={body} onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}>
        {rows.length === 0 ? (
          <div className="ss-none t3">{onlyCheck ? "No lines to check." : "No cues."}</div>
        ) : (
          <>
            <div style={{ height: first * ROW }} aria-hidden />
            {visible.map(({ cue, index }) => {
              const text = cueLine(edits?.[index] ?? cue.text);
              const flagged = !reviewed?.has(index) && needsCheck(cue, mode, minConfidence);
              if (ocr) {
                const sure = typeof cue.confidence === "number" ? Math.round(cue.confidence * 100) : null;
                return (
                  <Tr
                    key={index}
                    on={selected === index}
                    onClick={onSelect ? () => onSelect(index) : undefined}
                    style={{ height: ROW }}
                    label={`Line ${index + 1}`}
                  >
                    <span className="t3 num">{index + 1}</span>
                    <span className="num t2">{formatCueTime(cue.start)}</span>
                    <span className="cell">
                      {flagged && <WarningFilled className="warn" style={{ fontSize: 14 }} aria-label="To check" />}
                      <span className="truncate ss-cjk" title={text}>{text}</span>
                    </span>
                    <span className={cx("r num", flagged ? "warn" : "t3")} style={{ display: "flex" }}>
                      {sure === null ? "—" : `${sure}%`}
                    </span>
                  </Tr>
                );
              }
              const cps = cueCps(cue);
              const fast = cps !== null && cps > FAST_CPS;
              return (
                <Tr key={index} style={{ height: ROW }}>
                  <span className="t3 num">{index + 1}</span>
                  <span className="num">{formatCueTime(cue.start)}</span>
                  <span className="num t2">{formatCueTime(cue.end)}</span>
                  <span className="cell">
                    {flagged && <WarningFilled className="warn" style={{ fontSize: 14 }} aria-label="To check" />}
                    <span className="truncate" title={text}>
                      {text}
                      {cue.forced && <span className="t3"> (forced)</span>}
                    </span>
                  </span>
                  <span className={cx("r num", fast ? "warn" : "t3")} style={{ display: "flex" }}>
                    {cps === null ? "—" : cps.toFixed(1)}
                  </span>
                </Tr>
              );
            })}
            <div style={{ height: (rows.length - last) * ROW }} aria-hidden />
          </>
        )}
      </div>
    </div>
  );
});
