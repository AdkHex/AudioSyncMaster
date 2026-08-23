/** localStorage persistence with quota safety.
 *
 *  History previously stored 20 full result sets and wrote them on every
 *  change with no error handling. A few large batches exceed the ~5MB quota,
 *  at which point setItem throws inside a useEffect and takes the render with
 *  it. Here writes are budgeted, trimmed on overflow, and never throw. */

import {
  DEFAULT_SETTINGS,
  type AppSettings,
  type HistoryEntry,
  type RunSummary,
  type SyncMode,
  type SyncResult,
} from "./types";

const HISTORY_KEY = "audiosync.history.v2";
const SETTINGS_KEY = "audiosync.settings.v2";
const FOLDERS_KEY = "audiosync.folders.v2";

const MAX_HISTORY_ENTRIES = 25;
/** Results kept per entry. Full sets from large batches are what blew the quota;
 *  the summary preserves the totals so trimmed entries still read correctly. */
const MAX_RESULTS_PER_ENTRY = 200;

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    // Corrupt or unreadable payload: fall back rather than break startup.
    return fallback;
  }
}

function writeJson(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

// -------------------------------------------------------------------- history

function trimEntry(entry: HistoryEntry): HistoryEntry {
  if (entry.results.length <= MAX_RESULTS_PER_ENTRY) return entry;
  return { ...entry, results: entry.results.slice(0, MAX_RESULTS_PER_ENTRY) };
}

export function loadHistory(): HistoryEntry[] {
  const entries = readJson<HistoryEntry[]>(HISTORY_KEY, []);
  if (!Array.isArray(entries)) return [];
  return entries.filter(
    (entry) => entry && typeof entry.id === "string" && Array.isArray(entry.results),
  );
}

/** Persist history, shedding the oldest entries until it fits the quota. */
export function saveHistory(entries: HistoryEntry[]): HistoryEntry[] {
  let candidate = entries.slice(0, MAX_HISTORY_ENTRIES).map(trimEntry);

  while (candidate.length > 0) {
    if (writeJson(HISTORY_KEY, candidate)) return candidate;
    // Over quota: drop the oldest and try again.
    candidate = candidate.slice(0, Math.floor(candidate.length / 2));
  }

  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    /* nothing further we can do */
  }
  return [];
}

export function createHistoryEntry(
  mode: SyncMode,
  results: SyncResult[],
  summary: RunSummary | null,
): HistoryEntry {
  return {
    id: `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    date: new Date().toISOString(),
    mode,
    results,
    summary,
    fileCount: results.length,
  };
}

/** Parse a stored ISO date defensively -- entries written by older builds may
 *  hold any shape, and an Invalid Date renders as "Invalid Date" in the UI. */
export function parseHistoryDate(value: string): Date | null {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatHistoryDate(value: string): string {
  const date = parseHistoryDate(value);
  if (!date) return "Unknown date";
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

// ------------------------------------------------------------------- settings

export function loadSettings(): AppSettings {
  const stored = readJson<Partial<AppSettings>>(SETTINGS_KEY, {});
  const merged = { ...DEFAULT_SETTINGS, ...stored };
  return {
    ...merged,
    // Clamp anything a hand-edited or stale payload might contain.
    windowSeconds: clamp(merged.windowSeconds, 5, 600, DEFAULT_SETTINGS.windowSeconds),
    windowCount: Math.round(clamp(merged.windowCount, 1, 20, DEFAULT_SETTINGS.windowCount)),
    maxOffsetMs: clamp(merged.maxOffsetMs, 1000, 600000, DEFAULT_SETTINGS.maxOffsetMs),
    maxWorkers: Math.round(clamp(merged.maxWorkers, 1, 16, DEFAULT_SETTINGS.maxWorkers)),
  };
}

export function saveSettings(settings: AppSettings): void {
  writeJson(SETTINGS_KEY, settings);
}

function clamp(value: number, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

// -------------------------------------------------------------- recent folders

export interface RecentFolders {
  video: string | null;
  audio: string | null;
}

export function loadRecentFolders(): RecentFolders {
  return readJson<RecentFolders>(FOLDERS_KEY, { video: null, audio: null });
}

export function saveRecentFolders(folders: RecentFolders): void {
  writeJson(FOLDERS_KEY, folders);
}
