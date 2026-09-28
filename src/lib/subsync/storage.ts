/** What the Subsync workspace remembers between launches: each task's
 *  settings and output choices, and the task last open.
 *
 *  Stored values are merged over the defaults field by field, so a setting
 *  added in a later release gets its default instead of `undefined`. API keys
 *  never pass through here: the desktop shell keeps them. */

import { DEFAULT_OUTPUT } from "./tools";
import { DEFAULT_OPTIONS, type OutputOptions, type SubTask, type TaskOptions } from "./types";

const KEY = "audiosync.subsync.v1";

export interface SubsyncPrefs {
  task: SubTask;
  options: TaskOptions;
  output: Record<SubTask, OutputOptions>;
}

const TASKS = Object.keys(DEFAULT_OPTIONS) as SubTask[];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A stored value can replace a default when it has a shape the field can
 *  hold. Unions are common ("auto" | number, number | null, number[] |
 *  "all"), so scalars and arrays pass among themselves; only booleans and
 *  objects must match exactly. */
function fits(fallback: unknown, value: unknown): boolean {
  if (value === undefined) return false;
  if (typeof fallback === "boolean") return typeof value === "boolean";
  if (isRecord(fallback)) return isRecord(value);
  if (typeof value === "boolean") return fallback === null || fallback === undefined;
  return Array.isArray(value) || ["string", "number"].includes(typeof value);
}

function mergeFields<T extends object>(defaults: T, stored: unknown): T {
  if (!isRecord(stored)) return { ...defaults };
  const out: Record<string, unknown> = { ...(defaults as Record<string, unknown>) };
  for (const [key, value] of Object.entries(stored)) {
    const fallback = (defaults as Record<string, unknown>)[key];
    if (fallback === null || fallback === undefined ? value !== undefined : fits(fallback, value)) {
      out[key] = value;
    }
  }
  return out as T;
}

export const DEFAULT_PREFS: SubsyncPrefs = {
  task: "sync",
  options: DEFAULT_OPTIONS,
  output: DEFAULT_OUTPUT,
};

/** File-specific values that must not outlive the session. */
function forgetFiles(options: TaskOptions): TaskOptions {
  return { ...options, sync: { ...options.sync, reference: null } };
}

export function loadPrefs(): SubsyncPrefs {
  let stored: unknown = null;
  try {
    const raw = localStorage.getItem(KEY);
    stored = raw ? JSON.parse(raw) : null;
  } catch {
    stored = null;
  }
  if (!isRecord(stored)) return DEFAULT_PREFS;

  const storedOptions = isRecord(stored.options) ? stored.options : {};
  const storedOutput = isRecord(stored.output) ? stored.output : {};
  const options = {} as Record<SubTask, unknown>;
  const output = {} as Record<SubTask, OutputOptions>;
  for (const task of TASKS) {
    options[task] = mergeFields(DEFAULT_OPTIONS[task], storedOptions[task]);
    output[task] = mergeFields(DEFAULT_OUTPUT[task], storedOutput[task]);
  }
  const task = TASKS.includes(stored.task as SubTask) ? (stored.task as SubTask) : DEFAULT_PREFS.task;
  return { task, options: forgetFiles(options as unknown as TaskOptions), output };
}

export function savePrefs(prefs: SubsyncPrefs): void {
  try {
    localStorage.setItem(
      KEY,
      JSON.stringify({ task: prefs.task, options: forgetFiles(prefs.options), output: prefs.output }),
    );
  } catch {
    /* over quota or unavailable: the defaults are fine next time */
  }
}
