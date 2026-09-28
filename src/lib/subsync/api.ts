/** Typed wrapper over the Subsync commands of the desktop shell.
 *
 *  Like src/lib/api.ts, this is the only place the Subsync UI reaches
 *  `invoke`, so the command names and argument shapes live in one file. */

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import { isDesktop } from "@/lib/api";

import type {
  Capabilities,
  CueData,
  CuePreview,
  MediaRef,
  PackStatus,
  ProbedFile,
  SecretName,
  SubJob,
  SubJobOutcome,
  SubsyncEvent,
} from "./types";

export { isDesktop };

/** What the pickers and a drop hand back: a file on disk, before probing. */
export interface PickedFile {
  name: string;
  path: string;
  type: "video" | "audio" | "subtitle";
  size: number | null;
}

export type PickAccept = "video" | "subtitle" | "any";

function requireDesktop(action: string) {
  if (!isDesktop()) throw new Error(`${action} is only available in the desktop app.`);
}

export async function pickFiles(
  accept: PickAccept,
): Promise<{ folder: string | null; files: PickedFile[] }> {
  requireDesktop("Choosing files");
  return invoke("subs_pick_files", { accept });
}

export async function pickFolder(): Promise<string | null> {
  requireDesktop("Choosing a folder");
  return invoke("subs_pick_folder");
}

/** Dropped OS paths as files; folders are expanded to the media and
 *  subtitle files inside them. */
export async function resolveDropped(paths: string[]): Promise<PickedFile[]> {
  requireDesktop("Drag and drop");
  return invoke("subs_resolve_dropped", { paths });
}

export async function probe(paths: string[]): Promise<ProbedFile[]> {
  if (!isDesktop() || paths.length === 0) return [];
  const response = await invoke<{ files: ProbedFile[] }>("subs_probe", { paths });
  return response.files;
}

export async function capabilities(): Promise<Capabilities | null> {
  if (!isDesktop()) return null;
  const response = await invoke<{ caps: Capabilities }>("subs_caps");
  return response.caps;
}

export async function loadCues(
  reference: MediaRef,
  limit = 5000,
): Promise<{ preview: CuePreview | null; error?: string }> {
  requireDesktop("Reading a subtitle");
  return invoke("subs_load", { reference, limit });
}

export async function saveCues(
  path: string,
  format: string,
  cues: CueData[],
  language: string | null,
): Promise<{ path?: string; error?: string }> {
  requireDesktop("Saving a subtitle");
  return invoke("subs_save", { path, format, cues, language });
}

export interface BatchResult {
  outcomes: SubJobOutcome[];
  cancelled?: boolean;
  error?: string;
}

/** Run a queue of jobs. Resolves once every job has finished; progress
 *  streams as `subsync-event`s meanwhile. API keys are added by the shell. */
export async function startBatch(jobs: SubJob[], maxWorkers: number): Promise<BatchResult> {
  requireDesktop("Running subtitle jobs");
  return invoke("start_subs_batch", { request: { jobs, maxWorkers } });
}

export async function cancel(): Promise<void> {
  if (!isDesktop()) return;
  await invoke("cancel_sync");
}

export interface PackResult {
  type: "packDone";
  pack: string;
  ok: boolean;
  error?: string | null;
  status?: PackStatus;
}

export async function pack(
  action: "install" | "remove",
  packId: string,
  model?: string,
): Promise<PackResult> {
  requireDesktop("Installing engines");
  return invoke("subs_pack", { action, pack: packId, model });
}

export async function secretStatus(): Promise<Record<SecretName, boolean>> {
  if (!isDesktop()) return { anthropic: false, openai: false, deepl: false, google: false };
  return invoke("subs_secret_status");
}

/** Store a key in the shell's own store; an empty value deletes it. */
export async function setSecret(name: SecretName, value: string): Promise<void> {
  requireDesktop("Saving a key");
  await invoke("subs_secret_set", { name, value });
}

export async function revealPath(path: string): Promise<void> {
  requireDesktop("Opening a folder");
  await invoke("reveal_path", { path });
}

export async function openPath(path: string): Promise<void> {
  requireDesktop("Opening a file");
  await invoke("open_path", { path });
}

/** Every Subsync engine event. Returns a disposer that is safe to call
 *  before the listener finished registering. */
export async function subscribe(handler: (event: SubsyncEvent) => void): Promise<UnlistenFn> {
  if (!isDesktop()) return () => undefined;
  return listen<SubsyncEvent>("subsync-event", (event) => handler(event.payload));
}
