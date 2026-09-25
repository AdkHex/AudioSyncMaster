/** Run state for the dub sync tab's queue.
 *
 *  The tab used to run exactly one pair; now it runs a queue of them -- a
 *  season of episodes, or several movies at once -- synced by the engine in
 *  parallel. Each job moves through the same stages the single run did
 *  (reading, finding the offsets, writing, checking) and ends with its own
 *  plan and verification, so the state is one row per job rather than one
 *  pair per run. */

import {
  AUDIBLE_MS,
  formatClock,
  formatFps,
  formatMs,
  formatSpan,
  isUnmatchedFill,
  type DubJobOutcome,
  type DubSyncPlan,
  type DubVerification,
  type DubOutput,
} from "./types";

export type DubQueueStatus = "idle" | "running" | "complete" | "cancelled" | "failed";

export type DubJobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

/** One pair of the queue, on its way through the engine. */
export interface DubQueueJob {
  /** The engine's job index; events and outcomes are routed on this. */
  id: number;
  /** Video (or original-language track) basename, for the row. */
  name: string;
  dubName: string;
  /** The files, for the waveform view before any plan names them. */
  videoPath: string;
  dubPath: string;
  videoTrack: number;
  dubTrack: number;
  status: DubJobStatus;
  /** 0-100 within the current stage; the engine restarts it per stage. */
  percent: number;
  stage: string | null;
  /** Arrives as soon as the analysis is done, before the track is written.
   *  After the cuts were edited by hand, the plan the track was written from. */
  plan: DubSyncPlan | null;
  /** The engine's own plan, kept when the cuts are edited so the editor can
   *  always go back to it. */
  enginePlan: DubSyncPlan | null;
  /** The plan as it stands mid-analysis, stage by stage, until the plan
   *  arrives; what the waveform view shows being laid down live. */
  draft: DubSyncPlan | null;
  output: DubOutput | null;
  verification: DubVerification | null;
  muxedPath: string | null;
  error: string | null;
}

export interface DubQueueState {
  status: DubQueueStatus;
  jobs: DubQueueJob[];
  /** Jobs that have finished or failed, for the overall progress. */
  done: number;
  startedAt: number | null;
  error: string | null;
  /** A finished job whose track is being written again from cuts edited by
   *  hand: the queue status to go back to once it is, and the row as it
   *  was, which a stopped or failed write goes back to since the track on
   *  disk is still the old one. The queue counts as running meanwhile: the
   *  engine takes one command at a time. */
  rerender: { job: number; resume: DubQueueStatus; previous: DubQueueJob } | null;
}

export const initialDubQueueState: DubQueueState = {
  status: "idle",
  jobs: [],
  done: 0,
  startedAt: null,
  error: null,
  rerender: null,
};

export type DubQueueAction =
  | {
      type: "queueStarted";
      jobs: {
        name: string;
        dubName: string;
        videoPath?: string;
        dubPath?: string;
        videoTrack?: number;
        dubTrack?: number;
      }[];
    }
  | { type: "jobStart"; job: number }
  | { type: "jobProgress"; job: number; percent: number; stage: string }
  | { type: "jobDraft"; job: number; plan: DubSyncPlan }
  | { type: "jobPlan"; job: number; plan: DubSyncPlan }
  | { type: "jobDone"; outcome: DubJobOutcome }
  | { type: "batchDone"; outcomes: DubJobOutcome[]; cancelled: boolean }
  | { type: "batchFailed"; message: string }
  | { type: "rerenderStart"; job: number; plan: DubSyncPlan }
  | { type: "rerenderDone"; outcome: DubJobOutcome }
  | { type: "reset" };

function updateJob(state: DubQueueState, job: number, patch: Partial<DubQueueJob>): DubQueueState {
  if (job < 0 || job >= state.jobs.length) return state;
  const jobs = state.jobs.slice();
  jobs[job] = { ...jobs[job], ...patch };
  return { ...state, jobs };
}

/** What one outcome says about the job it closes. */
function outcomeToJob(outcome: DubJobOutcome): Partial<DubQueueJob> {
  const error = outcome.error ?? outcome.plan?.error ?? null;
  return {
    status: outcome.cancelled ? "cancelled" : error ? "failed" : "done",
    plan: outcome.plan ?? null,
    enginePlan: outcome.plan ?? null,
    draft: null,
    output: outcome.output ?? null,
    verification: outcome.verification ?? null,
    muxedPath: outcome.muxedPath ?? null,
    error,
    percent: 100,
    stage: null,
  };
}

/** Jobs that have left the queue: done, failed, or cancelled. Derived rather
 *  than counted, so a duplicate completion can never move the bar twice. */
function countDone(jobs: DubQueueJob[]): number {
  return jobs.filter((job) => job.status !== "queued" && job.status !== "running").length;
}

export function dubQueueReducer(state: DubQueueState, action: DubQueueAction): DubQueueState {
  switch (action.type) {
    case "queueStarted":
      return {
        ...initialDubQueueState,
        status: "running",
        jobs: action.jobs.map((job, index) => ({
          id: index,
          name: job.name,
          dubName: job.dubName,
          videoPath: job.videoPath ?? "",
          dubPath: job.dubPath ?? "",
          videoTrack: job.videoTrack ?? 0,
          dubTrack: job.dubTrack ?? 0,
          status: "queued",
          percent: 0,
          stage: null,
          plan: null,
          enginePlan: null,
          draft: null,
          output: null,
          verification: null,
          muxedPath: null,
          error: null,
        })),
        startedAt: Date.now(),
      };

    case "jobStart":
      if (state.status !== "running") return state;
      return updateJob(state, action.job, { status: "running", percent: 0, stage: "starting" });

    case "jobProgress":
      if (state.status !== "running") return state;
      return updateJob(state, action.job, { percent: action.percent, stage: action.stage });

    case "jobDraft":
      if (state.status !== "running") return state;
      return updateJob(state, action.job, { draft: action.plan });

    case "jobPlan":
      // The plan supersedes every draft.
      return updateJob(state, action.job, { plan: action.plan, draft: null });

    case "jobDone": {
      const next = updateJob(state, action.outcome.job, outcomeToJob(action.outcome));
      return { ...next, done: countDone(next.jobs) };
    }

    case "batchDone": {
      // The engine's final list wins for every job, so a cancelled or
      // partially failed batch still shows what did finish. Jobs that never
      // produced an outcome keep their streamed state.
      let jobs = state.jobs.slice();
      for (const outcome of action.outcomes) {
        if (outcome.job < 0 || outcome.job >= jobs.length) continue;
        jobs[outcome.job] = { ...jobs[outcome.job], ...outcomeToJob(outcome) };
      }
      if (action.cancelled) {
        // Jobs still waiting when the batch stopped did not finish.
        jobs = jobs.map((job) =>
          job.status === "queued" || job.status === "running"
            ? { ...job, status: "cancelled" as const, percent: 0, stage: null }
            : job,
        );
      }
      return {
        ...state,
        status: action.cancelled ? "cancelled" : "complete",
        jobs,
        done: countDone(jobs),
        startedAt: null,
      };
    }

    case "batchFailed":
      return {
        ...state,
        status: "failed",
        error: action.message,
        startedAt: null,
      };

    case "rerenderStart": {
      // Only a finished job can be written again, and only while the engine
      // is free: it takes one command at a time.
      const job = state.jobs[action.job];
      if (state.status === "running" || !job || job.status !== "done") return state;
      const next = updateJob(state, action.job, {
        status: "running",
        percent: 0,
        stage: "writing the track with the edited cuts",
        plan: action.plan,
        error: null,
      });
      return {
        ...next,
        status: "running",
        done: countDone(next.jobs),
        rerender: { job: action.job, resume: state.status, previous: job },
      };
    }

    case "rerenderDone": {
      if (!state.rerender || state.rerender.job !== action.outcome.job) return state;
      const { job, resume, previous } = state.rerender;
      const failed = action.outcome.cancelled || !!(action.outcome.error ?? action.outcome.plan?.error);
      // The write is staged, so a stopped or failed one leaves the track
      // that was there: the row goes back to describing it. The editor
      // keeps the edited cuts on screen for another try.
      const next = failed
        ? updateJob(state, job, previous)
        : updateJob(state, job, { ...outcomeToJob(action.outcome), enginePlan: previous.enginePlan });
      return {
        ...next,
        status: resume,
        done: countDone(next.jobs),
        rerender: null,
      };
    }

    case "reset":
      return initialDubQueueState;

    default:
      return state;
  }
}

/** One line on what a plan will do, for a job card's summary. */
export function describePlan(plan: DubSyncPlan): string {
  const dubs = plan.segments.filter((s) => s.kind === "dub").length;
  const fills = plan.segments.filter((s) => s.kind === "fill").length;
  const replaced = plan.segments.filter(isUnmatchedFill).length;
  const parts = [
    `${dubs} stretch${dubs === 1 ? "" : "es"} of dub`,
    `${fills} fill${fills === 1 ? "" : "s"} from the original` +
      (replaced ? ` (${replaced} replacing dub that did not correlate)` : ""),
  ];
  // The frame-rate verdict, read off the video's own metadata rather than
  // left to the symptoms: matched, or mastered at another rate. dubRate is
  // rounded to 6 places, so equality is judged loosely.
  if (plan.videoFps !== null && plan.videoFps !== undefined) {
    if (plan.dubRate !== null && Math.abs(plan.dubRate - plan.videoFps) > 1e-6) {
      parts.push(
        `video ${formatFps(plan.videoFps)} fps, dub mastered at ${formatFps(plan.dubRate)} fps` +
          (Math.abs(plan.speed - 1) > 1e-9 ? ` (played at ${plan.speed.toFixed(6)}×)` : ""),
      );
    } else {
      parts.push(`video ${formatFps(plan.videoFps)} fps, dub at the same rate`);
    }
    if (plan.rateConfirmed === false) {
      parts[parts.length - 1] += " (not confirmed by the audio)";
    }
  } else if (Math.abs(plan.speed - 1) > 1e-9) {
    parts.push(`dub played at ${plan.speed.toFixed(6)}×`);
  }
  if (plan.summary?.dubUsedShare !== null && plan.summary?.dubUsedShare !== undefined) {
    parts.push(`${(100 * plan.summary.dubUsedShare).toFixed(1)}% of the dub used`);
  }
  return parts.join(", ");
}

/** Why the original plays, in the words the report uses. */
const REASON_WORDS: Record<string, string> = {
  head: "before the dub starts",
  tail: "after the dub ends",
  cut: "the dub lacks the scene",
  silent: "the dub is silent",
  unmatched: "dub replaced (did not correlate)",
  draft: "not placed yet",
};

/** Everything about one job as plain text, for pasting into a message:
 *  the files, what the plan does and why, every piece, what to check, and
 *  how the written track measured against the video. */
export function reportText(job: DubQueueJob): string {
  const lines: string[] = [`Dub sync report: ${job.name} + ${job.dubName}`, `video: ${job.videoPath}`, `dub: ${job.dubPath}`];
  const plan = job.plan;
  if (job.error) lines.push(`error: ${job.error}`);
  if (plan) {
    lines.push("", describePlan(plan));
    if (plan.summary) {
      const fills = Object.entries(plan.summary.fillS)
        .filter(([, seconds]) => seconds > 0)
        .map(([reason, seconds]) => `${formatSpan(seconds)} ${REASON_WORDS[reason] ?? reason}`);
      lines.push(
        `dub used: ${formatClock(plan.summary.dubUsedS)} of ${formatClock(plan.summary.dubDurationS)}` +
          (fills.length ? `; the original plays where ${fills.join("; ")}` : ""),
      );
    }
    lines.push("", "pieces:");
    for (const segment of plan.segments) {
      const span = `${formatClock(segment.startS)} - ${formatClock(segment.endS)}`;
      lines.push(
        segment.kind === "dub"
          ? `  dub   ${span}  offset ${(segment.offsetS ?? 0) >= 0 ? "+" : ""}${(segment.offsetS ?? 0).toFixed(4)}s  match ${(segment.match ?? 0).toFixed(2)}`
          : `  fill  ${span}  ${formatSpan(segment.endS - segment.startS)}  ${segment.note}`,
      );
    }
    if (plan.warnings.length) lines.push("", "to check:", ...plan.warnings.map((w) => `  ! ${w}`));
    if (plan.notes?.length) lines.push("", "notes:", ...plan.notes.map((n) => `  - ${n}`));
    if (plan.voicePieces?.length) {
      lines.push("", "voices moved:", ...plan.voicePieces.map((p) => `  - ${p.note}`));
    }
  }
  const check = job.verification;
  if (check) {
    lines.push("", "checked against the video:");
    lines.push(
      `  spots: typically ${check.typicalMs === null ? "?" : formatMs(check.typicalMs)} ms, worst ${check.worstMs === null ? "?" : formatMs(check.worstMs)} ms`,
      `  sweep: ${check.sweepMeasured} of ${check.sweepWindows} windows measured, ${check.sweepWithinAudible} within ${AUDIBLE_MS} ms` +
        (check.sweepWithin1Ms !== undefined ? `, ${check.sweepWithin1Ms} within 1 ms` : "") +
        (check.sweepWorstMs !== null ? `, worst ${formatMs(check.sweepWorstMs)} ms` : ""),
    );
    for (const stretch of check.stretches) {
      lines.push(`  ! ${formatClock(stretch.startS)} - ${formatClock(stretch.endS)} out by ${formatMs(stretch.residualMs)} ms`);
    }
    if (check.lines) {
      const l = check.lines;
      lines.push(
        `  lines: ${l.judged} of ${l.windows.length} dialogue windows judged` +
          (l.overallMs !== null && l.overallMs !== undefined ? `, overall ${l.overallMs >= 0 ? "+" : ""}${l.overallMs.toFixed(0)} ms from the original's` : "") +
          `, ${l.withinTolerance} within ${l.toleranceMs} ms`,
      );
      for (const w of l.windows) {
        if (w.lagMs !== null && Math.abs(w.lagMs) > l.toleranceMs) {
          lines.push(`  ! lines ${formatClock(w.startS)} - ${formatClock(w.endS)} at ${w.lagMs >= 0 ? "+" : ""}${w.lagMs.toFixed(0)} ms`);
        }
      }
    }
  }
  if (job.output) lines.push("", `written: ${job.output.outputPath}`);
  return lines.join("\n");
}

/** A stage name as the engine reports it, said the way the row says it. */
export function describeStage(stage: string | null): string {
  if (!stage) return "Starting…";
  const known: Record<string, string> = {
    starting: "Starting…",
    probing: "Reading the files",
    "reading the original": "Reading the original",
    "reading the dub": "Reading the dub",
    "checking the frame rate": "Checking the frame rate",
    "finding the offsets": "Finding where the dub belongs",
    "placing the cuts": "Placing the cuts",
    "measuring the offsets": "Measuring each stretch",
    "looking for dub inside the gaps": "Looking for dub inside the gaps",
    "assembling the plan": "Assembling the plan",
    "checking the finished track": "Checking the finished track against the video",
    "writing the track with the edited cuts": "Writing the track with the edited cuts",
    muxing: "Adding the track to a copy of the video",
    done: "Done",
  };
  if (known[stage]) return known[stage];
  if (stage.startsWith("writing ")) return `Writing ${stage.slice("writing ".length)}`;
  return stage[0].toUpperCase() + stage.slice(1);
}
