/** The bottom dock. Every page shares two of its tabs — Output (the engine's
 *  log) and History (past runs) — and may add its own in front of them:
 *  Timeline on Dub sync, Cues on Subsync. */

import { CopyRegular, DeleteRegular, SearchRegular } from "@fluentui/react-icons";
import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { parseHistoryDate } from "@/lib/storage";
import type { HistoryEntry, SyncMode } from "@/lib/types";
import { resultsOutcome } from "@/app/pages/analyse/views";
import { DockTabs } from "@/ui/frame";
import { Cmd, Combo, Status, Table, Tr, type St } from "@/ui/kit";

export type CommonTab = "output" | "history";

/** What the shell hands every page for its dock. */
export interface CommonDock {
  tab: CommonTab | null;
  setTab: (tab: CommonTab | null) => void;
  output: ReactNode;
  history: ReactNode;
  outputTools?: ReactNode;
  historyTools?: ReactNode;
}

export interface OwnTab<T extends string> {
  id: T;
  label: string;
  body: ReactNode;
  tools?: ReactNode;
}

/** A page's dock: its own tabs, then Output and History. Returns null when
 *  nothing is open, so the workspace gives the space back to the list. */
export function PageDock<T extends string>({
  own = [],
  ownTab,
  onOwnTab,
  common,
  height = 280,
}: {
  own?: OwnTab<T>[];
  /** The page's tab to show while Output and History are closed; null hides
   *  the dock unless one of them is open. */
  ownTab?: T | null;
  onOwnTab?: (tab: T) => void;
  common: CommonDock;
  height?: number;
}): ReactNode {
  const active: T | CommonTab | null = common.tab ?? ownTab ?? null;
  if (active === null) return null;
  const mine = own.find((t) => t.id === active);
  const tabs = [...own.map((t) => ({ id: t.id as T | CommonTab, label: t.label })), { id: "output" as const, label: "Output" }, { id: "history" as const, label: "History" }];
  return (
    <div style={{ height, display: "flex", flexDirection: "column" }}>
      <DockTabs<T | CommonTab>
        tabs={tabs}
        on={active}
        onTab={(tab) => {
          if (tab === "output" || tab === "history") common.setTab(tab as CommonTab);
          else {
            common.setTab(null);
            onOwnTab?.(tab as T);
          }
        }}
        tools={mine ? mine.tools : active === "output" ? common.outputTools : common.historyTools}
      />
      {mine ? mine.body : active === "output" ? common.output : common.history}
    </div>
  );
}

/* ---------------------------------------------------------------- output */

type Severity = "" | "ok" | "warn" | "bad";

/** Colour a log line by what it says. The engine emits plain strings, so
 *  severity is inferred here rather than plumbed through the whole pipeline. */
export function severityOf(line: string): Severity {
  const text = line.toLowerCase();
  if (/\b(error|failed|failure|no audio|no video|traceback|cannot|could not)\b/.test(text)) return "bad";
  if (/\b(warn|warning|low confidence|skipped|unmatched|unused)\b/.test(text)) return "warn";
  if (/\b(measured|wrote|written|done|complete|matched)\b/.test(text)) return "ok";
  return "";
}

/** The engine's log, newest at the bottom, following the end while you are
 *  there and staying put when you scroll up to read. */
export const OutputBody = memo(function OutputBody({ logs }: { logs: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const stamps = useRef<string[]>([]);
  if (logs.length < stamps.current.length) stamps.current = [];
  while (stamps.current.length < logs.length) {
    stamps.current.push(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }));
  }
  const atEnd = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && atEnd.current) el.scrollTop = el.scrollHeight;
  }, [logs.length]);
  return (
    <div
      className="log"
      ref={ref}
      role="log"
      aria-label="Output"
      onScroll={(event) => {
        const el = event.currentTarget;
        atEnd.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
    >
      {logs.length === 0 ? (
        <div className="t3">Engine output appears here during a run.</div>
      ) : (
        logs.map((line, i) => (
          <div key={i}>
            <span className="t">{stamps.current[i]}</span>
            <span className={severityOf(line) || undefined}>{line}</span>
          </div>
        ))
      )}
    </div>
  );
});

export function OutputTools({ logs, onCopy, onClear }: { logs: string[]; onCopy: (text: string) => void; onClear: () => void }) {
  return (
    <>
      <Cmd sm icon={<CopyRegular />} title="Copy the output" disabled={logs.length === 0} onClick={() => onCopy(logs.join("\n"))} />
      <Cmd sm icon={<DeleteRegular />} title="Clear the output" disabled={logs.length === 0} onClick={onClear} />
    </>
  );
}

/* --------------------------------------------------------------- history */

export const PAGE_LABEL: Record<SyncMode, string> = {
  movie: "Movies",
  series: "Series",
  compare: "Find match",
  dubsync: "Dub sync",
  subsync: "Subsync",
};

/** What a history row says happened, from the entry itself. */
export function historyOutcome(entry: HistoryEntry): { tone: St; text: string } {
  if (entry.outcome) return { tone: entry.outcome.tone, text: entry.outcome.text };
  // Movies and Series runs from before outcomes were saved.
  if (entry.mode !== "compare" && entry.results.length > 0) return resultsOutcome(entry.results);
  const s = entry.summary;
  if (!s) return { tone: "ok", text: `${entry.fileCount} file${entry.fileCount === 1 ? "" : "s"}` };
  const parts = [`${s.matched} matched`];
  if (s.drifting) parts.push(`${s.drifting} drifting`);
  if (s.cuts) parts.push(`${s.cuts} different cut${s.cuts === 1 ? "" : "s"}`);
  if (s.failed) parts.push(`${s.failed} failed`);
  return { tone: s.failed || s.cuts || s.drifting ? "warn" : "ok", text: parts.join(", ") };
}

/** A row's name: the entry's own, or the first file measured. */
export function historyName(entry: HistoryEntry): string {
  if (entry.name) return entry.name;
  const first = entry.results[0];
  const name = first ? first.videoFile.replace(/\.[^.]+$/, "") : "Run";
  return entry.fileCount > 1 ? `${name} and ${entry.fileCount - 1} more` : name;
}

/** When a run finished, as the design writes it: "Today 11:05",
 *  "Yesterday 22:40", "Sep 26 09:30", "Sep 26 2025". */
export function whenText(value: string, now = new Date()): string {
  const date = parseHistoryDate(value);
  if (!date) return "Unknown date";
  const time = `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(now) - day(date)) / 86_400_000);
  if (days === 0) return `Today ${time}`;
  if (days === 1) return `Yesterday ${time}`;
  const month = date.toLocaleString("en-US", { month: "short" });
  return date.getFullYear() === now.getFullYear() ? `${month} ${date.getDate()} ${time}` : `${month} ${date.getDate()} ${date.getFullYear()}`;
}

export interface HistoryState {
  query: string;
  page: SyncMode | "all";
}

export function useHistoryFilter() {
  return useState<HistoryState>({ query: "", page: "all" });
}

export function HistoryTools({
  filter,
  onFilter,
  onClear,
  empty,
}: {
  filter: HistoryState;
  onFilter: (filter: HistoryState) => void;
  onClear: () => void;
  empty: boolean;
}) {
  return (
    <>
      <Combo<SyncMode | "all">
        ghost
        sm
        w={112}
        label="Page"
        value={filter.page}
        options={[{ value: "all", label: "All pages" }, ...(["movie", "series", "compare", "dubsync", "subsync"] as SyncMode[]).map((m) => ({ value: m, label: PAGE_LABEL[m] }))]}
        onChange={(page) => onFilter({ ...filter, page })}
      />
      <label className="tbox" style={{ width: 200, height: 28 }}>
        <SearchRegular className="t3" aria-hidden />
        <input aria-label="Search history" placeholder="Search" value={filter.query} onChange={(event) => onFilter({ ...filter, query: event.target.value })} />
      </label>
      <Cmd sm icon={<DeleteRegular />} title="Clear history" disabled={empty} onClick={onClear} />
    </>
  );
}

export function HistoryBody({
  entries,
  filter,
  onOpen,
  onDelete,
}: {
  entries: HistoryEntry[];
  filter: HistoryState;
  onOpen: (entry: HistoryEntry) => void;
  onDelete: (id: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const shown = useMemo(() => {
    const q = filter.query.trim().toLowerCase();
    return entries.filter(
      (e) =>
        (filter.page === "all" || e.mode === filter.page) &&
        (!q || `${historyName(e)} ${historyOutcome(e).text} ${PAGE_LABEL[e.mode]}`.toLowerCase().includes(q)),
    );
  }, [entries, filter]);
  if (entries.length === 0) return <div className="log t3">Finished runs from every page are kept here.</div>;
  return (
    <Table cols="110px minmax(0,1fr) minmax(0,1.3fr) 120px" head={["Page", "Name", "Result", "When"]} label="History">
      {shown.map((entry) => {
        const outcome = historyOutcome(entry);
        return (
          <Tr
            key={entry.id}
            on={entry.id === selected}
            style={{ height: 32 }}
            onClick={() => {
              setSelected(entry.id);
              onOpen(entry);
            }}
          >
            <span className="t2">{PAGE_LABEL[entry.mode]}</span>
            <span className="truncate">{historyName(entry)}</span>
            {/* The design lists a run's outcome with commas. */}
            <Status s={outcome.tone} text={outcome.text.replace(/ · /g, ", ")} />
            <span className="t3 num truncate">{whenText(entry.date)}</span>
            {/* Shown on the row under the pointer or with focus. */}
            <span className="hdel">
              <button
                type="button"
                className="cmd sm icon"
                title="Delete this run"
                aria-label="Delete this run"
                onClick={(event) => {
                  event.stopPropagation();
                  onDelete(entry.id);
                }}
              >
                <span className="ic" aria-hidden><DeleteRegular /></span>
              </button>
            </span>
          </Tr>
        );
      })}
    </Table>
  );
}
