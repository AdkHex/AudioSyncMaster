import { memo, useEffect, useRef } from "react";

interface ConsolePanelProps {
  logs: string[];
  onClear: () => void;
}

export const ConsolePanel = memo(function ConsolePanel({ logs, onClear }: ConsolePanelProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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
    <section className="rounded-lg border border-border bg-card" aria-label="Engine output">
      <header className="flex items-center justify-between border-b border-border px-4 py-2">
        <h2 className="text-[10px] uppercase tracking-wider text-muted-foreground">
          Console
        </h2>
        {logs.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            className="text-[10px] text-muted-foreground transition-colors hover:text-foreground"
          >
            Clear
          </button>
        )}
      </header>
      <div ref={containerRef} className="max-h-44 overflow-y-auto px-4 py-2">
        {logs.length === 0 ? (
          <p className="text-[11px] text-muted-foreground">
            Engine output appears here during a run.
          </p>
        ) : (
          <div className="space-y-0.5 font-mono text-[11px] text-muted-foreground">
            {logs.map((line, index) => (
              <p key={`${index}-${line.slice(0, 24)}`} className="whitespace-pre-wrap break-all">
                {line}
              </p>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>
    </section>
  );
});
