import { AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Clock, XCircle } from "lucide-react";
import { memo, useState } from "react";

import { Spinner, Tag } from "@/components/ui";
import { cx } from "@/lib/cx";
import { runProgress, type JobStatus, type RunJob, type RunState } from "@/lib/subsync/reducer";

import { OutcomeView, type OutcomeActions } from "./OutcomeView";

/** A batch on its way through the engine, then its outcomes. */
export const RunView = memo(function RunView({
  run,
  actions,
  onOpenConsole,
}: {
  run: RunState;
  actions: OutcomeActions;
  onOpenConsole: () => void;
}) {
  const running = run.status === "running";
  const progress = runProgress(run);
  const active = run.jobs.find((job) => job.status === "running");

  return (
    <div>
      {running && (
        <div className="relative flex h-10 items-center gap-3 border-b border-border px-[18px]" aria-live="polite">
          <Spinner className="h-3.5 w-3.5 shrink-0 border-[1.8px]" />
          <span className="min-w-0 flex-1 truncate text-[12px] text-muted-foreground">
            {active ? `${active.label}: ${active.stage ?? "starting"}` : progress.done > 0 ? `${progress.done} of ${progress.total} done` : "Starting…"}
          </span>
          <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
            {progress.done}/{progress.total}
          </span>
          <span
            className="absolute inset-x-0 bottom-0 h-[2px] bg-primary transition-[width] duration-300"
            style={{ width: `${progress.percent}%` }}
            aria-hidden
          />
        </div>
      )}

      {run.error && (
        <div className="border-b border-border px-[18px] py-3.5" role="alert">
          <p className="text-[13px] font-semibold text-destructive">The run did not finish.</p>
          <p className="mt-1 text-[12px] leading-relaxed text-muted-foreground">{run.error}</p>
          <button type="button" onClick={onOpenConsole} className="mt-1.5 text-[12px] font-medium text-primary hover:opacity-80">
            Open the console
          </button>
        </div>
      )}
      {run.status === "cancelled" && !run.error && (
        <p className="border-b border-border px-[18px] py-3 text-[12px] text-muted-foreground">Stopped before every job finished.</p>
      )}

      <ul aria-label="Jobs">
        {run.jobs.map((job) => (
          <JobRow key={job.id} job={job} defaultOpen={run.jobs.length === 1} actions={actions} onOpenConsole={onOpenConsole} />
        ))}
      </ul>
    </div>
  );
});

function JobRow({
  job,
  defaultOpen,
  actions,
  onOpenConsole,
}: {
  job: RunJob;
  defaultOpen: boolean;
  actions: OutcomeActions;
  onOpenConsole: () => void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const expandable = !!job.outcome || job.logs.length > 0;
  const error = job.outcome?.error;

  return (
    <li className="border-b border-border">
      <div className="flex items-center gap-2.5 px-[18px] py-2.5">
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          disabled={!expandable}
          aria-expanded={expandable ? open : undefined}
          aria-label={open ? `Hide details of ${job.label}` : `Show details of ${job.label}`}
          className="shrink-0 rounded text-muted-foreground hover:text-foreground disabled:opacity-30"
        >
          {open ? <ChevronDown className="h-3.5 w-3.5" aria-hidden /> : <ChevronRight className="h-3.5 w-3.5" aria-hidden />}
        </button>
        <StatusIcon status={job.status} />
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium" title={job.label}>
          {job.label}
        </span>
        {job.status === "running" && (
          <>
            <span className="max-w-[260px] truncate text-[11.5px] text-muted-foreground">{job.stage ?? "starting"}</span>
            <span className="tabular font-mono text-[11px] text-muted-foreground">{job.percent}%</span>
          </>
        )}
        {job.outcome?.summary && job.status === "done" && !open && (
          <span className="max-w-[320px] truncate text-[11.5px] text-muted-foreground">{job.outcome.summary}</span>
        )}
        <Tag tone={TONES[job.status]}>{LABELS[job.status]}</Tag>
      </div>

      {job.status === "running" && (
        <div className="px-[18px] pb-2.5">
          <div
            className="h-[3px] overflow-hidden rounded-full bg-elevated"
            role="progressbar"
            aria-label={`${job.label} progress`}
            aria-valuenow={job.percent}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="h-full bg-primary transition-[width] duration-300" style={{ width: `${job.percent}%` }} />
          </div>
        </div>
      )}

      {error && (
        <div className="px-[18px] pb-2.5 pl-[50px]">
          <p className="text-[12px] leading-relaxed text-muted-foreground">{error}</p>
          <button type="button" onClick={onOpenConsole} className="mt-1 text-[12px] font-medium text-primary hover:opacity-80">
            Open the console
          </button>
        </div>
      )}

      {open && expandable && (
        <div className={cx("px-[18px] pb-3.5 pl-[50px]")}>
          {job.outcome ? (
            <OutcomeView outcome={job.outcome} logs={job.logs} actions={actions} />
          ) : (
            <pre className="max-h-48 overflow-auto rounded-md bg-sunken px-2.5 py-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
              {job.logs.join("\n")}
            </pre>
          )}
        </div>
      )}
    </li>
  );
}

const LABELS: Record<JobStatus, string> = {
  queued: "Waiting",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Stopped",
};

const TONES: Record<JobStatus, "neutral" | "success" | "warning" | "destructive"> = {
  queued: "neutral",
  running: "neutral",
  done: "success",
  failed: "destructive",
  cancelled: "warning",
};

function StatusIcon({ status }: { status: JobStatus }) {
  switch (status) {
    case "running":
      return <Spinner className="h-3.5 w-3.5 border-[1.8px]" />;
    case "done":
      return <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success" aria-hidden />;
    case "failed":
      return <XCircle className="h-3.5 w-3.5 shrink-0 text-destructive" aria-hidden />;
    case "cancelled":
      return <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-warning" aria-hidden />;
    default:
      return <Clock className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />;
  }
}
