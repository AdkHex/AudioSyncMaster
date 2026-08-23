/** Typed wrapper over the Tauri command surface.
 *
 *  Every backend call goes through here so the component tree never touches
 *  `invoke` directly and there is one place to check whether the desktop shell
 *  is present. */

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";

import type {
  AnalyzeRequest,
  ApplyResult,
  CorrectionItem,
  FileItem,
  MediaProbe,
  PairingReport,
  PickResponse,
  SyncResult,
  SyncRun,
} from "./types";

export function isDesktop(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

class NotDesktopError extends Error {
  constructor(action: string) {
    super(`${action} is only available in the desktop app.`);
    this.name = "NotDesktopError";
  }
}

function requireDesktop(action: string) {
  if (!isDesktop()) throw new NotDesktopError(action);
}

let idCounter = 0;
/** Stable unique id. Date.now() collides when several files are added in the
 *  same millisecond, which produced duplicate React keys. */
function nextId(prefix: string): string {
  idCounter += 1;
  return `${prefix}-${idCounter}-${Math.random().toString(36).slice(2, 8)}`;
}

function withIds(files: PickResponse["files"], type: "video" | "audio"): FileItem[] {
  return files.map((file) => ({ ...file, type, id: nextId(type) }));
}

export async function pickVideoFolder(): Promise<{ folder: string | null; files: FileItem[] }> {
  requireDesktop("Choosing a folder");
  const response = await invoke<PickResponse>("pick_video_folder");
  return { folder: response.folder, files: withIds(response.files, "video") };
}

export async function pickAudioFolder(): Promise<{ folder: string | null; files: FileItem[] }> {
  requireDesktop("Choosing a folder");
  const response = await invoke<PickResponse>("pick_audio_folder");
  return { folder: response.folder, files: withIds(response.files, "audio") };
}

export async function pickAudioFile(): Promise<{ folder: string | null; files: FileItem[] }> {
  requireDesktop("Choosing a file");
  const response = await invoke<PickResponse>("pick_audio_file");
  return { folder: response.folder, files: withIds(response.files, "audio") };
}

/** Turn dropped OS paths into usable file entries, expanding folders. */
export async function resolveDroppedPaths(
  paths: string[],
  kind: "video" | "audio",
): Promise<FileItem[]> {
  requireDesktop("Drag and drop");
  const files = await invoke<Omit<FileItem, "id">[]>("resolve_dropped_paths", {
    paths,
    kind,
  });
  return files.map((file) => ({ ...file, id: nextId(kind) }));
}

export async function probeMedia(path: string): Promise<MediaProbe> {
  requireDesktop("Reading media information");
  return invoke<MediaProbe>("probe_media", { path });
}

export async function previewPairs(request: Partial<AnalyzeRequest>): Promise<PairingReport> {
  requireDesktop("Previewing pairs");
  return invoke<PairingReport>("preview_pairs", { request });
}

export async function startSync(request: AnalyzeRequest): Promise<SyncRun> {
  requireDesktop("Analysis");
  return invoke<SyncRun>("start_sync", { request });
}

export async function cancelSync(): Promise<void> {
  if (!isDesktop()) return;
  await invoke("cancel_sync");
}

export async function applyCorrections(
  items: CorrectionItem[],
  options: { outputDir?: string | null; suffix?: string; overwrite?: boolean } = {},
): Promise<ApplyResult> {
  requireDesktop("Writing corrected files");
  return invoke<ApplyResult>("apply_corrections", {
    request: {
      items,
      outputDir: options.outputDir ?? null,
      suffix: options.suffix ?? ".synced",
      overwrite: options.overwrite ?? false,
    },
  });
}

export async function exportCsv(results: SyncResult[]): Promise<string> {
  requireDesktop("Exporting");
  return invoke<string>("export_csv", { results });
}

export async function exportJson(results: SyncResult[]): Promise<string> {
  requireDesktop("Exporting");
  return invoke<string>("export_json", { results });
}

export async function revealPath(path: string): Promise<void> {
  requireDesktop("Opening a folder");
  await invoke("reveal_path", { path });
}

// ------------------------------------------------------------------- events

export interface ProgressEvent {
  processed: number;
  total: number;
  current?: string;
}

export interface FileProgressEvent {
  file: string;
  percent: number;
}

export interface ApplyProgressEvent {
  file?: string;
  output?: string;
  description?: string;
  done?: number;
  total?: number;
}

export interface SyncListeners {
  onLog?: (message: string) => void;
  onProgress?: (event: ProgressEvent) => void;
  onFileStart?: (file: string) => void;
  onFileProgress?: (event: FileProgressEvent) => void;
  onResult?: (result: SyncResult) => void;
  onDone?: (run: SyncRun) => void;
  onPairs?: (report: PairingReport) => void;
  onApplyProgress?: (event: ApplyProgressEvent) => void;
}

/** Subscribe to engine events. Returns a disposer that removes every listener,
 *  including any that resolved after the caller already unsubscribed. */
export async function subscribeToSync(listeners: SyncListeners): Promise<UnlistenFn> {
  if (!isDesktop()) return () => undefined;

  let disposed = false;
  const unlisteners: UnlistenFn[] = [];

  const add = async <T>(event: string, handler: (payload: T) => void) => {
    const unlisten = await listen<T>(event, (e) => handler(e.payload));
    if (disposed) {
      // The caller unsubscribed while this listener was still registering.
      unlisten();
      return;
    }
    unlisteners.push(unlisten);
  };

  await Promise.all([
    add<string>("sync-log", (message) => listeners.onLog?.(message)),
    add<ProgressEvent>("sync-progress", (payload) => listeners.onProgress?.(payload)),
    add<{ file: string }>("sync-file-start", (p) => listeners.onFileStart?.(p.file)),
    add<FileProgressEvent>("sync-file-progress", (p) => listeners.onFileProgress?.(p)),
    add<SyncResult>("sync-result", (payload) => listeners.onResult?.(payload)),
    add<SyncRun>("sync-done", (payload) => listeners.onDone?.(payload)),
    add<PairingReport>("sync-pairs", (payload) => listeners.onPairs?.(payload)),
    add<ApplyProgressEvent>("sync-apply-progress", (p) => listeners.onApplyProgress?.(p)),
  ]);

  return () => {
    disposed = true;
    unlisteners.forEach((unlisten) => unlisten());
    unlisteners.length = 0;
  };
}

/** Native OS drag-and-drop. The webview's File objects carry no filesystem
 *  path in Tauri v2, so drops must come from the window event instead. */
export async function subscribeToFileDrop(
  handler: (paths: string[]) => void,
  onHover?: (hovering: boolean) => void,
): Promise<UnlistenFn> {
  if (!isDesktop()) return () => undefined;

  const unlisten = await getCurrentWindow().onDragDropEvent((event) => {
    if (event.payload.type === "over") {
      onHover?.(true);
    } else if (event.payload.type === "drop") {
      onHover?.(false);
      handler(event.payload.paths);
    } else {
      onHover?.(false);
    }
  });
  return unlisten;
}
