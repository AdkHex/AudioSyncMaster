/** Run state of the Subsync workspace: one batch of jobs at a time, each
 *  moving from queued to its outcome, plus the progress of engine packs
 *  being installed. Pure; events from the engine are fed straight in. */

import type { PackStatus, SubJobOutcome, SubTask, SubsyncEvent } from "./types";

export type RunStatus = "idle" | "running" | "complete" | "cancelled" | "failed";
export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

/** Log lines kept per job; the console has the full stream. */
const MAX_JOB_LOGS = 300;

export interface RunJob {
  /** Index in the batch; events are routed on it. */
  index: number;
  id: string;
  task: SubTask;
  label: string;
  status: JobStatus;
  percent: number;
  stage: string | null;
  logs: string[];
  outcome: SubJobOutcome | null;
}

export interface RunState {
  status: RunStatus;
  task: SubTask | null;
  jobs: RunJob[];
  error: string | null;
  startedAt: number | null;
}

export const initialRunState: RunState = {
  status: "idle",
  task: null,
  jobs: [],
  error: null,
  startedAt: null,
};

export type RunAction =
  | { type: "started"; task: SubTask; jobs: { id: string; label: string }[]; at: number }
  | { type: "event"; event: SubsyncEvent }
  | { type: "finished"; outcomes: SubJobOutcome[]; cancelled?: boolean; error?: string | null }
  | { type: "failed"; message: string }
  | { type: "reset" };

function patchJob(state: RunState, index: number, patch: Partial<RunJob>): RunState {
  if (index < 0 || index >= state.jobs.length) return state;
  const jobs = state.jobs.slice();
  jobs[index] = { ...jobs[index], ...patch };
  return { ...state, jobs };
}

function findIndex(state: RunState, outcome: { job: number; id?: string }): number {
  if (outcome.id) {
    const byId = state.jobs.findIndex((job) => job.id === outcome.id);
    if (byId >= 0) return byId;
  }
  return outcome.job;
}

function statusOf(outcome: SubJobOutcome): JobStatus {
  if (outcome.cancelled) return "cancelled";
  if (outcome.error) return "failed";
  return "done";
}

function applyOutcome(state: RunState, outcome: SubJobOutcome): RunState {
  const index = findIndex(state, outcome);
  return patchJob(state, index, {
    status: statusOf(outcome),
    outcome,
    percent: 100,
    stage: null,
  });
}

function isFinished(status: JobStatus): boolean {
  return status === "done" || status === "failed" || status === "cancelled";
}

function settle(state: RunState, cancelled: boolean, error: string | null): RunState {
  // Anything still open when the batch ends did not get to run.
  const jobs = state.jobs.map((job) =>
    isFinished(job.status) ? job : { ...job, status: "cancelled" as const, stage: null },
  );
  const status: RunStatus = error ? "failed" : cancelled ? "cancelled" : "complete";
  return { ...state, jobs, status, error };
}

export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "started":
      return {
        status: "running",
        task: action.task,
        error: null,
        startedAt: action.at,
        jobs: action.jobs.map((job, index) => ({
          index,
          id: job.id,
          task: action.task,
          label: job.label,
          status: "queued",
          percent: 0,
          stage: null,
          logs: [],
          outcome: null,
        })),
      };
    case "event": {
      const event = action.event;
      // Events that arrive after the batch settled belong to nothing here.
      if (state.status !== "running") return state;
      switch (event.type) {
        case "subsJobStart":
          return patchJob(state, findIndex(state, event), { status: "running", percent: 0, stage: null });
        case "subsJobProgress": {
          const job = state.jobs[event.job];
          if (!job || isFinished(job.status)) return state;
          return patchJob(state, event.job, {
            status: "running",
            percent: Math.max(0, Math.min(100, Math.round(event.percent))),
            stage: event.stage,
          });
        }
        case "subsJobLog": {
          const job = state.jobs[event.job];
          if (!job) return state;
          const logs = job.logs.length >= MAX_JOB_LOGS ? job.logs.slice(1 - MAX_JOB_LOGS) : job.logs.slice();
          logs.push(event.message);
          return patchJob(state, event.job, { logs });
        }
        case "subsJobDone":
          return applyOutcome(state, event);
        case "subsBatchDone":
          return runReducer(state, {
            type: "finished",
            outcomes: event.outcomes,
            cancelled: event.cancelled,
            error: event.error,
          });
        default:
          return state;
      }
    }
    case "finished": {
      if (state.status !== "running") return state;
      const withOutcomes = action.outcomes.reduce(applyOutcome, state);
      return settle(withOutcomes, !!action.cancelled, action.error ?? null);
    }
    case "failed":
      return settle({ ...state, status: "running" }, false, action.message);
    case "reset":
      return initialRunState;
  }
}

/** Jobs finished, for the overall bar. */
export function runProgress(state: RunState): { done: number; total: number; percent: number } {
  const total = state.jobs.length;
  if (total === 0) return { done: 0, total: 0, percent: 0 };
  let done = 0;
  let partial = 0;
  for (const job of state.jobs) {
    if (isFinished(job.status)) done += 1;
    else if (job.status === "running") partial += job.percent / 100;
  }
  return { done, total, percent: ((done + partial) / total) * 100 };
}

// ------------------------------------------------------------------ packs

export interface PackProgress {
  busy: boolean;
  action: "install" | "remove" | null;
  percent: number | null;
  stage: string | null;
  bytes: number | null;
  totalBytes: number | null;
  error: string | null;
}

export type PacksState = Record<string, PackProgress>;

export type PackAction =
  | { type: "begin"; pack: string; action: "install" | "remove" }
  | { type: "event"; event: SubsyncEvent }
  | { type: "done"; pack: string; ok: boolean; error?: string | null; status?: PackStatus };

const IDLE_PACK: PackProgress = {
  busy: false,
  action: null,
  percent: null,
  stage: null,
  bytes: null,
  totalBytes: null,
  error: null,
};

export function packsReducer(state: PacksState, action: PackAction): PacksState {
  switch (action.type) {
    case "begin":
      return { ...state, [action.pack]: { ...IDLE_PACK, busy: true, action: action.action } };
    case "event": {
      const event = action.event;
      if (event.type === "packProgress") {
        const current = state[event.pack] ?? { ...IDLE_PACK, busy: true, action: "install" };
        return {
          ...state,
          [event.pack]: {
            ...current,
            busy: true,
            percent: Math.max(0, Math.min(100, event.percent)),
            stage: event.stage,
            bytes: event.bytes ?? current.bytes,
            totalBytes: event.totalBytes ?? current.totalBytes,
          },
        };
      }
      if (event.type === "packDone") {
        return packsReducer(state, { type: "done", pack: event.pack, ok: event.ok, error: event.error });
      }
      return state;
    }
    case "done":
      return {
        ...state,
        [action.pack]: { ...IDLE_PACK, error: action.ok ? null : (action.error ?? "The install did not finish.") },
      };
  }
}
