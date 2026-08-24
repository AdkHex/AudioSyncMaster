/** Single reducer for all run state.
 *
 *  The original kept ~28 separate useState hooks and wrote the results array
 *  from three places (streaming events, the completion event, and the invoke
 *  return value), so a partial failure could wipe results that had already
 *  arrived. Here every transition is one reducer case, results have exactly one
 *  owner, and streamed results are merged by identity rather than appended
 *  blindly. */

import type {
  FileItem,
  PairingReport,
  ProcessingStatus,
  RunSummary,
  SyncMode,
  SyncResult,
} from "./types";

export interface SyncState {
  mode: SyncMode;
  status: ProcessingStatus;
  videoFiles: FileItem[];
  audioFiles: FileItem[];
  videoFolder: string | null;
  audioFolder: string | null;
  /** True when videos were chosen individually rather than via a folder. */
  explicitVideoSelection: boolean;
  results: SyncResult[];
  summary: RunSummary | null;
  pairing: PairingReport | null;
  progress: { processed: number; total: number };
  currentFile: string | null;
  fileProgress: number;
  logs: string[];
  startedAt: number | null;
  error: string | null;
}

export const initialSyncState: SyncState = {
  mode: "movie",
  status: "idle",
  videoFiles: [],
  audioFiles: [],
  videoFolder: null,
  audioFolder: null,
  explicitVideoSelection: false,
  results: [],
  summary: null,
  pairing: null,
  progress: { processed: 0, total: 0 },
  currentFile: null,
  fileProgress: 0,
  logs: [],
  startedAt: null,
  error: null,
};

const MAX_LOG_LINES = 500;

export type SyncAction =
  | { type: "setMode"; mode: SyncMode }
  | { type: "addFiles"; kind: "video" | "audio"; files: FileItem[]; folder?: string | null; explicit?: boolean }
  | { type: "replaceFiles"; kind: "video" | "audio"; files: FileItem[]; folder: string | null; explicit?: boolean }
  | { type: "removeFiles"; kind: "video" | "audio"; ids: string[] }
  | { type: "clearFiles"; kind: "video" | "audio" }
  | { type: "setPairing"; pairing: PairingReport | null }
  | { type: "runStarted"; total: number }
  | { type: "progress"; processed: number; total: number; current?: string }
  | { type: "fileStart"; file: string }
  | { type: "fileProgress"; file: string; percent: number }
  | { type: "result"; result: SyncResult }
  | { type: "runFinished"; results: SyncResult[]; summary: RunSummary | null; cancelled: boolean }
  | { type: "runFailed"; message: string }
  | { type: "log"; message: string }
  | { type: "loadResults"; results: SyncResult[]; summary: RunSummary | null; mode: SyncMode }
  | { type: "clearAll" }
  | { type: "clearLogs" };

/** Identity of a measured pair, used to merge streamed updates. */
function resultKey(result: SyncResult): string {
  return `${result.primaryPath ?? result.videoFile}::${result.secondaryPath ?? result.audioFile}`;
}

function appendLogs(existing: string[], message: string): string[] {
  const next = existing.length >= MAX_LOG_LINES
    ? existing.slice(existing.length - MAX_LOG_LINES + 1)
    : existing.slice();
  next.push(message);
  return next;
}

export function syncReducer(state: SyncState, action: SyncAction): SyncState {
  switch (action.type) {
    case "setMode": {
      if (action.mode === state.mode) return state;
      // Selections do not carry across modes: movie mode wants one audio file,
      // series mode wants a folder of them.
      return {
        ...initialSyncState,
        mode: action.mode,
        logs: state.logs,
      };
    }

    case "addFiles": {
      const key = action.kind === "video" ? "videoFiles" : "audioFiles";
      const existing = state[key];
      const seen = new Set(existing.map((file) => file.path));
      const additions = action.files.filter((file) => !seen.has(file.path));

      const next: SyncState = {
        ...state,
        [key]: [...existing, ...additions],
        pairing: null,
      } as SyncState;

      if (action.folder !== undefined) {
        if (action.kind === "video") next.videoFolder = action.folder;
        else next.audioFolder = action.folder;
      }
      if (action.kind === "video" && action.explicit !== undefined) {
        next.explicitVideoSelection = action.explicit;
      }
      return next;
    }

    case "replaceFiles": {
      const key = action.kind === "video" ? "videoFiles" : "audioFiles";
      const next: SyncState = {
        ...state,
        [key]: action.files,
        pairing: null,
      } as SyncState;
      if (action.kind === "video") {
        next.videoFolder = action.folder;
        next.explicitVideoSelection = action.explicit ?? false;
      } else {
        next.audioFolder = action.folder;
      }
      return next;
    }

    case "removeFiles": {
      const key = action.kind === "video" ? "videoFiles" : "audioFiles";
      const ids = new Set(action.ids);
      return {
        ...state,
        [key]: state[key].filter((file) => !ids.has(file.id)),
        pairing: null,
      } as SyncState;
    }

    case "clearFiles": {
      const key = action.kind === "video" ? "videoFiles" : "audioFiles";
      const next = { ...state, [key]: [], pairing: null } as SyncState;
      if (action.kind === "video") {
        next.videoFolder = null;
        next.explicitVideoSelection = false;
      } else {
        next.audioFolder = null;
      }
      return next;
    }

    case "setPairing":
      return { ...state, pairing: action.pairing };

    case "runStarted":
      return {
        ...state,
        status: "processing",
        results: [],
        summary: null,
        error: null,
        progress: { processed: 0, total: action.total },
        currentFile: null,
        fileProgress: 0,
        startedAt: Date.now(),
        logs: [],
      };

    case "progress":
      return {
        ...state,
        progress: { processed: action.processed, total: action.total },
        currentFile: action.current ?? state.currentFile,
      };

    case "fileStart":
      return {
        ...state,
        currentFile: action.file,
        fileProgress: 0,
        logs: appendLogs(state.logs, `Analysing ${action.file}`),
      };

    case "fileProgress":
      // Ignore progress for a file that is no longer the active one, so a
      // straggling event cannot rewind the bar.
      if (state.currentFile && state.currentFile !== action.file) return state;
      return { ...state, fileProgress: action.percent };

    case "result": {
      // Merge by identity: re-running or receiving a duplicate updates in place
      // rather than appending a second row for the same pair.
      const key = resultKey(action.result);
      const index = state.results.findIndex((r) => resultKey(r) === key);
      const results = state.results.slice();
      if (index >= 0) results[index] = action.result;
      else results.push(action.result);
      return { ...state, results };
    }

    case "runFinished": {
      // The engine's final list wins when it has content; otherwise keep what
      // streamed in, so a cancelled or partially failed run still shows work.
      const results = action.results.length > 0 ? action.results : state.results;
      return {
        ...state,
        status: action.cancelled ? "cancelled" : "complete",
        results,
        summary: action.summary,
        currentFile: null,
        fileProgress: 100,
        startedAt: null,
      };
    }

    case "runFailed":
      return {
        ...state,
        status: "idle",
        error: action.message,
        currentFile: null,
        fileProgress: 0,
        startedAt: null,
        logs: appendLogs(state.logs, `Error: ${action.message}`),
      };

    case "log":
      return { ...state, logs: appendLogs(state.logs, action.message) };

    case "loadResults":
      return {
        ...state,
        mode: action.mode,
        status: "complete",
        results: action.results,
        summary: action.summary,
        error: null,
      };

    case "clearAll":
      return { ...initialSyncState, mode: state.mode };

    case "clearLogs":
      return { ...state, logs: [] };

    default:
      return state;
  }
}

/** Whether the current selection can be analysed, and why not if it cannot. */
export function validateSelection(state: SyncState): { ok: boolean; reason?: string } {
  if (state.videoFiles.length === 0) {
    return { ok: false, reason: "Add at least one video file." };
  }
  if (state.audioFiles.length === 0) {
    return {
      ok: false,
      reason: state.mode === "movie"
        ? "Choose the audio file to sync against."
        : "Choose the folder containing the audio tracks.",
    };
  }
  if (state.mode === "series" && !state.audioFolder) {
    return { ok: false, reason: "Series mode needs an audio folder." };
  }
  if (state.mode === "series" && !state.videoFolder) {
    return { ok: false, reason: "Series mode needs a video folder." };
  }
  return { ok: true };
}

/** Remaining time from observed throughput, or null while unknown.
 *
 *  Counts the in-flight file's own progress, not just completed ones. Waiting
 *  for the first file to finish means no estimate at all for minutes on a
 *  movie -- the case where the user most wants one. A partly-decoded file is
 *  weak evidence, so it only counts once enough of it has been measured for
 *  the rate to mean anything. */
export function estimateRemainingMs(state: SyncState): number | null {
  if (state.status !== "processing" || !state.startedAt) return null;
  const { processed, total } = state.progress;
  if (total <= 0) return null;

  const partial = state.currentFile && state.fileProgress >= 10
    ? state.fileProgress / 100
    : 0;
  const done = processed + partial;
  if (done <= 0) return null;

  const elapsed = Date.now() - state.startedAt;
  const perItem = elapsed / done;
  return Math.max(0, perItem * (total - done));
}
