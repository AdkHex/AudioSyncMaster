import { AlertTriangle, CheckCircle2, ClipboardCopy, FolderOpen, Info, Play, Scissors, XCircle } from "lucide-react";
import { memo, useState } from "react";

import { Spinner, Tag } from "@/components/ui";
import { cx } from "@/lib/cx";
import { describePlan, describeStage, reportText, type DubQueueJob, type DubQueueState } from "@/lib/dubQueueReducer";
import {
  AUDIBLE_MS,
  formatMs,
  formatClock,
  formatSpan,
  isUnmatchedFill,
  type DubSegment,
  type DubSyncPlan,
  type DubVerification,
} from "@/lib/types";

interface DubQueuePanelProps {
  state: DubQueueState;
  onReveal: (path: string) => void;
  onOpen: (path: string) => void;
  onOpenConsole: () => void;
  /** Open the waveform editor on a finished job's cuts. Absent when editing
   *  is not possible (the browser preview). */
  onEdit?: (job: number) => void;
  /** The job whose waveforms are shown above the queue, and how to change it. */
  shown?: number | null;
  onShow?: (job: number) => void;
}

/** Column template shared by the header and every row, so they cannot drift. */
const GRID = "52px 190px 190px 92px 64px minmax(0,1fr)";

/** The dub sync queue: a season of episodes, or several movies, synced in
 *  parallel.
 *
 *  Three things stream per job, in the order they arrive: where the job is,
 *  its plan (every stretch of dub and every fill from the original, on the
 *  video's timeline), and then the measurement of the finished file against
 *  the video. A job's plan appears while its track is still being written,
 *  since it is the part worth reading. */
export const DubQueuePanel = memo(function DubQueuePanel({
  state,
  onReveal,
  onOpen,
  onOpenConsole,
  onEdit,
  shown = null,
  onShow,
}: DubQueuePanelProps) {
  const { status, jobs, done } = state;
  const running = status === "running";
  const active = jobs.find((job) => job.status === "running");
  const single = jobs.length === 1;

  return (
    <>
      {running && (
        <div
          className="relative flex h-11 shrink-0 items-center gap-3 border-b border-border px-[18px]"
          aria-label="Dub sync progress"
        >
          <Spinner className="h-3.5 w-3.5 shrink-0 border-[1.8px]" />
          <span className="min-w-0 flex-1 truncate text-[12.5px] text-muted-foreground">
            {active
              ? `${active.name} — ${describeStage(active.stage)}`
              : done > 0
                ? `${done} of ${jobs.length} done`
                : "Starting…"}
          </span>
          <span className="tabular shrink-0 font-mono text-[11.5px] text-muted-foreground">
            {done}/{jobs.length}
            {active && ` · ${active.percent}%`}
          </span>
          <span
            className="absolute inset-x-0 bottom-0 h-[2px] bg-primary transition-[width] duration-300"
            style={{ width: `${((done + (active ? active.percent / 100 : 0)) / jobs.length) * 100}%` }}
            aria-hidden
          />
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto">
        {state.error && (
          <div className="border-b border-border px-[18px] py-4">
            <p className="text-[13px] font-semibold text-destructive">
              The sync did not finish.
            </p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted-foreground">
              {state.error}
            </p>
            <button
              type="button"
              onClick={onOpenConsole}
              className="mt-2 text-[12.5px] font-medium text-primary hover:opacity-80"
            >
              Open the console
            </button>
          </div>
        )}

        {status === "cancelled" && !state.error && (
          <p className="border-b border-border px-[18px] py-4 text-[12.5px] text-muted-foreground">
            Stopped before every track was written.
          </p>
        )}

        <ul aria-label="Dub sync queue">
          {jobs.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              expanded={single}
              onReveal={onReveal}
              onOpen={onOpen}
              onOpenConsole={onOpenConsole}
              // The engine takes one command at a time, so the cuts can be
              // edited only once the queue has stopped.
              onEdit={onEdit && !running ? onEdit : undefined}
              shown={shown === job.id}
              onShow={onShow}
            />
          ))}
        </ul>
      </div>
    </>
  );
});

/** One pair of the queue, from queued to the verified track. */
function JobRow({
  job,
  expanded,
  onReveal,
  onOpen,
  onOpenConsole,
  onEdit,
  shown = false,
  onShow,
}: {
  job: DubQueueJob;
  expanded: boolean;
  onReveal: (path: string) => void;
  onOpen: (path: string) => void;
  onOpenConsole: () => void;
  onEdit?: (job: number) => void;
  shown?: boolean;
  onShow?: (job: number) => void;
}) {
  const { status, output, verification, plan } = job;
  const finished = status === "done" || status === "failed" || status === "cancelled";
  const [copied, setCopied] = useState(false);
  /** Everything about the job as text on the clipboard, for a message. */
  const copyReport = () => {
    void navigator.clipboard?.writeText(reportText(job)).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1800);
      },
      () => undefined,
    );
  };
  return (
    <li className={cx("border-b border-border", shown && "bg-primary/[0.04]")}>
      {/* The row's head picks the job whose waveforms are shown above. */}
      <div
        className={cx("flex items-center gap-3 px-[18px] py-3", onShow && "cursor-pointer")}
        onClick={onShow ? () => onShow(job.id) : undefined}
        aria-current={shown ? "true" : undefined}
      >
        <StatusIcon status={job.status} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-medium" title={job.name}>
            {job.name}
          </p>
          <p className="mt-[2px] truncate text-[11.5px] text-muted-foreground" title={job.dubName}>
            {job.dubName}
          </p>
        </div>

        <span className="flex shrink-0 items-center gap-3">
          {job.status === "running" && (
            <>
              <span className="min-w-0 max-w-[240px] truncate text-[11.5px] text-muted-foreground">
                {describeStage(job.stage)}
              </span>
              <span className="tabular font-mono text-[11.5px] text-muted-foreground">
                {job.percent}%
              </span>
            </>
          )}
          <Tag tone={statusTone(job.status)}>{statusLabel(job.status)}</Tag>
        </span>
      </div>

      {job.status === "running" && (
        <div className="px-[18px] pb-3">
          <div className="h-[3px] overflow-hidden rounded-full bg-elevated">
            <div
              className="h-full bg-primary transition-[width] duration-300"
              style={{ width: `${job.percent}%` }}
            />
          </div>
        </div>
      )}

      {job.status === "failed" && (
        <div className="px-[18px] pb-3">
          <p className="text-[12.5px] leading-relaxed text-muted-foreground">{job.error}</p>
          <button
            type="button"
            onClick={onOpenConsole}
            className="mt-1.5 text-[12.5px] font-medium text-primary hover:opacity-80"
          >
            Open the console
          </button>
        </div>
      )}

      {output && status === "done" && (
        <div className="flex items-center gap-3 px-[18px] pb-3">
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-[12px]" title={output.outputPath}>
              {basename(output.outputPath)}
            </p>
            <p className="mt-[3px] text-[11.5px] text-muted-foreground">
              {output.channels === 6 ? "5.1" : output.channels === 8 ? "7.1" : `${output.channels}ch`}
              {" · "}
              {output.sampleRate / 1000} kHz · {formatSpan(output.seconds)}
              {job.muxedPath && (
                <>
                  {" · "}
                  also muxed into{" "}
                  <span className="font-mono" title={job.muxedPath}>
                    {basename(job.muxedPath)}
                  </span>
                </>
              )}
            </p>
          </div>
          <button
            type="button"
            onClick={() => onOpen(job.muxedPath ?? output.outputPath)}
            className="flex items-center gap-1.5 rounded-[7px] border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:text-foreground"
          >
            <Play className="h-3 w-3 fill-current" aria-hidden />
            {job.muxedPath ? "Play the video" : "Play"}
          </button>
          <button
            type="button"
            onClick={() => onReveal(output.outputPath)}
            className="flex items-center gap-1.5 rounded-[7px] border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:text-foreground"
          >
            <FolderOpen className="h-3 w-3" aria-hidden />
            Show in folder
          </button>
          <button
            type="button"
            onClick={copyReport}
            title="The plan, what to check and how the track measured, as text"
            className="flex items-center gap-1.5 rounded-[7px] border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:text-foreground"
          >
            <ClipboardCopy className="h-3 w-3" aria-hidden />
            {copied ? "Copied" : "Copy report"}
          </button>
          {plan && plan.segments.length > 0 && (
            <button
              type="button"
              onClick={() => onEdit?.(job.id)}
              disabled={!onEdit}
              title={onEdit ? "See both waveforms and move the cuts by hand" : "Available once the queue has finished"}
              className="flex items-center gap-1.5 rounded-[7px] border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
            >
              <Scissors className="h-3 w-3" aria-hidden />
              Edit the cuts
            </button>
          )}
        </div>
      )}

      {finished && (plan || verification) && (
        <details
          className="group px-[18px] pb-3"
          open={expanded}
          aria-label={expanded ? undefined : `${job.name} details`}
        >
          <summary className="flex cursor-pointer select-none items-center gap-1.5 text-[11.5px] text-muted-foreground">
            <span className="transition-transform group-open:rotate-90">▸</span>
            {plan && plan.segments.length > 0
              ? describePlan(plan)
              : "No plan was made."}
          </summary>
          <div className="mt-2 rounded-lg border border-border">
            {plan && plan.segments.length > 0 && <Plan plan={plan} />}
            {verification && <Verification verification={verification} />}
            {(plan?.warnings.length ?? 0) + (output?.warnings.length ?? 0) + (plan?.notes?.length ?? 0) > 0 && (
              <ul className="space-y-1.5 px-[18px] py-4">
                {[...(plan?.warnings ?? []), ...(output?.warnings ?? [])].map((warning, index) => (
                  <li
                    key={`warning-${index}`}
                    className="flex items-start gap-1.5 text-[11.5px] leading-relaxed text-muted-foreground"
                  >
                    <AlertTriangle className="mt-[3px] h-3 w-3 shrink-0 text-warning" aria-hidden />
                    <span>{warning}</span>
                  </li>
                ))}
                {/* Notes are not warnings: nothing was changed and there is
                    nothing to check. A warning glyph on "kept the dub" read
                    as a flaw in a track that had none. */}
                {(plan?.notes ?? []).map((note, index) => (
                  <li
                    key={`note-${index}`}
                    className="flex items-start gap-1.5 text-[11.5px] leading-relaxed text-muted-foreground"
                  >
                    <Info className="mt-[3px] h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </details>
      )}
    </li>
  );
}

function StatusIcon({ status }: { status: DubQueueJob["status"] }) {
  switch (status) {
    case "done":
      return <CheckCircle2 className="h-4 w-4 shrink-0 text-success" aria-hidden />;
    case "failed":
      return <XCircle className="h-4 w-4 shrink-0 text-destructive" aria-hidden />;
    case "cancelled":
      return <XCircle className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />;
    case "running":
      return <Spinner className="h-3.5 w-3.5 shrink-0 border-[1.8px]" />;
    default:
      return <span className="h-4 w-4 shrink-0 rounded-full border border-border-strong" />;
  }
}

function statusTone(status: DubQueueJob["status"]): "success" | "destructive" | "warning" | "neutral" {
  switch (status) {
    case "done":
      return "success";
    case "failed":
      return "destructive";
    case "cancelled":
      return "neutral";
    case "running":
      return "warning";
    default:
      return "neutral";
  }
}

function statusLabel(status: DubQueueJob["status"]): string {
  switch (status) {
    case "queued":
      return "Waiting";
    case "running":
      return "Syncing";
    case "done":
      return "Done";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Stopped";
  }
}

/** What the engine is doing, in the user's terms. */

function basename(path: string): string {
  return path.replace(/^.*[\\/]/, "");
}

// ------------------------------------------------------------------- the plan

function Plan({ plan }: { plan: DubSyncPlan }) {
  const fills = plan.segments.filter((s) => s.kind === "fill");
  return (
    <section aria-label="Plan">
      <div className="flex h-11 items-center gap-4 border-b border-border px-[18px]">
        <h2 className="text-[13px] font-semibold">Plan</h2>
        <span className="min-w-0 flex-1 truncate text-[11.5px] text-muted-foreground">
          {describePlan(plan)}
          {fills.length > 0 && (
            <>
              {" · "}
              {formatSpan(plan.filledS)} from the original at{" "}
              <span className="tabular font-mono">
                {plan.fillGainDb >= 0 ? "+" : ""}
                {plan.fillGainDb.toFixed(1)} dB
              </span>
            </>
          )}
        </span>
      </div>

      <div
        className="grid items-center gap-3.5 px-[18px] pb-2 pt-2.5 text-[11px] text-muted-foreground"
        style={{ gridTemplateColumns: GRID }}
      >
        <span />
        <span>On the video</span>
        <span>Taken from</span>
        <span className="text-right">Offset</span>
        <span className="text-right">Match</span>
        <span />
      </div>

      {plan.segments.map((segment, index) => (
        <SegmentRow key={index} segment={segment} />
      ))}
    </section>
  );
}

function SegmentRow({ segment }: { segment: DubSegment }) {
  const fill = segment.kind === "fill";
  // A fill that replaced dub which was audible but could not be matched is
  // the one kind a reader should go and listen to: unlike a real cut, the
  // dub may well have had the right scene there.
  const replaced = isUnmatchedFill(segment);
  const length = segment.endS - segment.startS;
  return (
    <div
      className={cx(
        "grid items-center gap-3.5 border-t border-border px-[18px] py-2",
        replaced ? "bg-destructive/[0.05]" : fill && "bg-warning/[0.045]",
      )}
      style={{ gridTemplateColumns: GRID }}
    >
      <Tag tone={replaced ? "destructive" : fill ? "warning" : "neutral"} className="font-medium">
        {replaced ? "Replaced" : fill ? "Original" : "Dub"}
      </Tag>

      <span className="tabular font-mono text-[11.5px]">
        {formatClock(segment.startS)} &ndash; {formatClock(segment.endS)}
      </span>

      <span className="tabular font-mono text-[11.5px] text-muted-foreground">
        {fill ? "original" : "dub"} {formatClock(segment.sourceStartS)}
        <span className="ml-1.5 text-[10.5px]">({formatSpan(length)})</span>
      </span>

      <span className="tabular text-right font-mono text-[11.5px]">
        {segment.offsetS === null ? (
          <span className="text-muted-foreground">&mdash;</span>
        ) : (
          `${segment.offsetS >= 0 ? "+" : ""}${segment.offsetS.toFixed(3)}s`
        )}
      </span>

      <span className="tabular text-right font-mono text-[11.5px] text-muted-foreground">
        {segment.match === null ? "—" : segment.match.toFixed(2)}
      </span>

      <span
        className={cx("min-w-0 truncate text-[11.5px]", replaced ? "text-destructive" : "text-muted-foreground")}
        title={replaced ? "The dub was audible here but its music and effects could not be matched, and the setting asked for such stretches to be replaced. Listen to this spot: the dub may have had the right scene." : undefined}
      >
        {replaced ? "dub was audible but did not correlate — replaced; check this spot" : segment.note}
        {segment.uncertaintyS >= 0.05 && (
          <span className="tabular ml-1.5 font-mono text-[10.5px]">
            ±{segment.uncertaintyS.toFixed(1)}s
          </span>
        )}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------- the check

/** The finished file measured against the video.
 *
 *  This is the number that matters: not what the plan intended, but where the
 *  written track actually sits. Spots are long precise windows; the sweep is
 *  short windows every few seconds that catch a stretch the spots fell
 *  between. Fills are skipped in both, since they are the original itself. */
function Verification({ verification }: { verification: DubVerification }) {
  const measured = verification.spots.filter((s) => s.residualMs !== null);
  const worst = verification.worstMs;
  const share =
    verification.sweepMeasured > 0
      ? Math.round((100 * verification.sweepWithinAudible) / verification.sweepMeasured)
      : null;
  const audible = verification.stretches.length;
  // The verdict weighs the sweep as much as the spots. A dozen spots can
  // all land on fills or on the one stretch that happens to be right, and a
  // green "38 ms at worst" above a list of stretches that are 300 ms out is
  // worse than no badge: the sweep is what covers the whole runtime.
  const tone: "success" | "warning" | "destructive" =
    audible > 0 || (share !== null && share < 80)
      ? "destructive"
      : worst === null
        ? "warning"
        : worst <= AUDIBLE_MS / 2 && (share === null || share >= 95)
          ? "success"
          : "warning";

  return (
    <section aria-label="Verification" className="border-b border-border px-[18px] py-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-[13px] font-semibold">Checked against the video</h2>
        {audible > 0 ? (
          <Tag tone={tone}>
            {audible} stretch{audible === 1 ? "" : "es"} audibly out
            {share !== null && ` · ${share}% of the runtime within ${AUDIBLE_MS} ms`}
          </Tag>
        ) : measured.length > 0 && worst !== null ? (
          <Tag tone={tone}>
            {verification.typicalMs !== null && `${formatMs(verification.typicalMs)} ms typical`}
            {" · "}
            {formatMs(worst)} ms at worst
            {share !== null && share < 95 && ` · ${share}% of the runtime within ${AUDIBLE_MS} ms`}
          </Tag>
        ) : (
          <Tag tone="warning">could not be measured</Tag>
        )}
      </div>

      <p className="mt-1.5 max-w-[72ch] text-[11.5px] leading-relaxed text-muted-foreground">
        {measured.length > 0
          ? `Measured at ${measured.length} spot${measured.length === 1 ? "" : "s"} along the runtime. Lip-sync starts to show around ${AUDIBLE_MS} ms; the measurement resolves a fraction of a millisecond.`
          : "None of the spots could be measured: the shared music and effects were too quiet, or every spot fell on a fill."}
        {verification.sweepWindows > 0 && share !== null && (
          <>
            {" "}
            A sweep of {verification.sweepWindows} short windows measured{" "}
            {verification.sweepMeasured}, and {share}% of those sit within {AUDIBLE_MS} ms
            {verification.sweepWithin1Ms !== undefined && verification.sweepMeasured > 0 &&
              ` (${Math.round((100 * verification.sweepWithin1Ms) / verification.sweepMeasured)}% within 1 ms)`}
            {verification.sweepWorstMs !== null && `; worst ${formatMs(verification.sweepWorstMs)} ms`}
            .
          </>
        )}
      </p>

      {verification.lines && verification.lines.judged > 0 && (
        <p className="mt-1.5 max-w-[72ch] text-[11.5px] leading-relaxed text-muted-foreground">
          Lines: across {verification.lines.judged} of {verification.lines.windows.length} minute-long dialogue
          windows the dub&apos;s speech sits{" "}
          <span className="tabular font-mono">
            {(verification.lines.overallMs ?? 0) >= 0 ? "+" : ""}
            {(verification.lines.overallMs ?? 0).toFixed(0)} ms
          </span>{" "}
          from the original&apos;s, which is in sync with the lips; {verification.lines.withinTolerance} of{" "}
          {verification.lines.judged} windows within {verification.lines.toleranceMs} ms (one window reads to about a
          tenth of a second).
        </p>
      )}
      {verification.lines?.windows.some((w) => w.lagMs !== null && Math.abs(w.lagMs) > verification.lines!.toleranceMs) && (
        <ul className="mt-1.5 space-y-1">
          {verification.lines.windows
            .filter((w) => w.lagMs !== null && Math.abs(w.lagMs) > verification.lines!.toleranceMs)
            .map((w, index) => (
              <li key={index} className="flex items-baseline gap-2 text-[11.5px] text-warning">
                <AlertTriangle className="h-3 w-3 shrink-0 self-center" aria-hidden />
                <span className="tabular font-mono">
                  {formatClock(w.startS)} &ndash; {formatClock(w.endS)}
                </span>
                <span>
                  the lines sit {w.lagMs! >= 0 ? "+" : ""}
                  {w.lagMs!.toFixed(0)} ms from the original&apos;s &mdash; check the lips
                </span>
              </li>
            ))}
        </ul>
      )}

      {verification.stretches.length > 0 && (
        <ul className="mt-2.5 space-y-1">
          {verification.stretches.map((stretch, index) => (
            <li
              key={index}
              className="flex items-baseline gap-2 text-[11.5px] text-destructive"
            >
              <AlertTriangle className="h-3 w-3 shrink-0 self-center" aria-hidden />
              <span className="tabular font-mono">
                {formatClock(stretch.startS)} &ndash; {formatClock(stretch.endS)}
              </span>
              <span>
                out by{" "}
                <span className="tabular font-mono">
                  {stretch.residualMs >= 0 ? "+" : ""}
                  {formatMs(stretch.residualMs)} ms
                </span>{" "}
                over {stretch.windows} windows &mdash; would be audible
              </span>
            </li>
          ))}
        </ul>
      )}

      {measured.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1">
          {verification.spots.map((spot, index) => (
            <span
              key={index}
              className="tabular font-mono text-[10.5px] text-muted-foreground"
              title={spot.note || undefined}
            >
              {formatClock(spot.positionS).replace(/\.\d{3}$/, "")}{" "}
              {spot.residualMs === null ? (
                <span className="opacity-60">{spot.note.startsWith("filled") ? "fill" : "n/a"}</span>
              ) : (
                <span className={cx(Math.abs(spot.residualMs) > AUDIBLE_MS && "text-destructive")}>
                  {spot.residualMs >= 0 ? "+" : ""}
                  {formatMs(spot.residualMs)}
                </span>
              )}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
