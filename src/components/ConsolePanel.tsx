import { ChevronDown } from "lucide-react";
import { memo, useEffect, useMemo, useRef } from "react";

import { Button, IconButton, Pill } from "@/components/ui";
import { cx } from "@/lib/cx";

interface ConsolePanelProps {
  logs: string[];
  onClear: () => void;
  onClose: () => void;
  onCopy: (text: string) => void;
}

type Severity = "info" | "success" | "warning" | "error";

/** Classify a log line for colour. The engine emits plain strings, so severity
 *  is inferred here rather than plumbed through the whole event pipeline. */
function severityOf(line: string): Severity {
  const text = line.toLowerCase();
  if (/\b(error|failed|failure|no audio|no video|traceback|cannot|could not)\b/.test(text)) {
    return "error";
  }
  if (/\b(warn|warning|low confidence|skipped|unmatched|unused)\b/.test(text)) {
    return "warning";
  }
  if (/\b(measured|wrote|written|done|complete|matched)\b/.test(text)) {
    return "success";
  }
  return "info";
}

const SEVERITY_CLASS: Record<Severity, string> = {
  info: "text-muted-foreground",
  success: "text-success",
  warning: "text-warning",
  error: "text-destructive",
};

export const ConsolePanel = memo(function ConsolePanel({
  logs,
  onClear,
  onClose,
  onCopy,
}: ConsolePanelProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Timestamps are captured as lines arrive. Keyed by index so re-renders do
  // not restamp existing lines; a cleared log resets the map.
  const stampsRef = useRef<string[]>([]);
  if (logs.length < stampsRef.current.length) stampsRef.current = [];
  while (stampsRef.current.length < logs.length) {
    stampsRef.current.push(
      new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }),
    );
  }
  const stamps = stampsRef.current;

  const lines = useMemo(
    () => logs.map((line, index) => ({
      line,
      time: stamps[index] ?? "",
      severity: severityOf(line),
    })),
    // stamps mutates in place alongside logs, so logs is the correct trigger.
    [logs, stamps],
  );

  // Follow new output only when already at the bottom, so scrolling back to
  // read something is not immediately undone by the next log line.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const atBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight < 40;
    if (atBottom) endRef.current?.scrollIntoView({ block: "end" });
  }, [logs]);

  return (
    <section className="shrink-0 border-t border-border bg-card" aria-label="Engine output">
      <header className="flex items-center gap-2.5 border-b border-border bg-elevated px-4 py-2">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">
          Console
        </h2>
        <Pill>
          {logs.length} line{logs.length === 1 ? "" : "s"}
        </Pill>
        <span className="flex-1" />
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onCopy(logs.join("\n"))}
          disabled={logs.length === 0}
        >
          Copy
        </Button>
        <Button variant="ghost" size="sm" onClick={onClear} disabled={logs.length === 0}>
          Clear
        </Button>
        <IconButton label="Hide console" onClick={onClose} className="h-6 w-6">
          <ChevronDown className="h-3.5 w-3.5" aria-hidden />
        </IconButton>
      </header>

      <div ref={containerRef} className="max-h-[150px] overflow-y-auto bg-sunken px-4 py-3">
        {lines.length === 0 ? (
          <p className="font-mono text-[11.5px] text-muted-foreground">
            Engine output appears here during a run.
          </p>
        ) : (
          <div className="font-mono text-[11.5px] leading-[1.75]">
            {lines.map(({ line, time, severity }, index) => (
              <div key={`${index}-${line.slice(0, 24)}`} className="flex gap-3">
                <span className="tabular shrink-0 text-muted-foreground/70">{time}</span>
                <span className={cx("min-w-0 whitespace-pre-wrap break-all", SEVERITY_CLASS[severity])}>
                  {line}
                </span>
              </div>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>
    </section>
  );
});
