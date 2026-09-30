/** Subsync's engines, packs and API keys: one store shared by the Subsync
 *  page (which engines its pickers offer) and Preferences › Subtitles
 *  (where packs are installed and keys saved), so installing a pack there
 *  updates the page at once.
 *
 *  A module-level store read with useSyncExternalStore. It starts reading
 *  the capabilities, the key status and the pack events the first time a
 *  component subscribes, and keeps them for the life of the app. */

import { useSyncExternalStore } from "react";

import * as subsApi from "@/lib/subsync/api";
import { packsReducer, type PackAction, type PacksState } from "@/lib/subsync/reducer";
import type { Capabilities, SecretName } from "@/lib/subsync/types";

export interface EnginesState {
  /** What the engine can run here; null until read (or in a browser). */
  caps: Capabilities | null;
  /** Install / remove progress per pack id. */
  packs: PacksState;
  /** Which API keys are saved; null until read. */
  secrets: Record<SecretName, boolean> | null;
  /** Where Preferences › Subtitles should scroll to next: a pack id, or
   *  "keys" for the API keys. Taken (cleared) by the section. */
  focus: string | null;
}

const INITIAL: EnginesState = { caps: null, packs: {}, secrets: null, focus: null };

let state: EnginesState = INITIAL;
const listeners = new Set<() => void>();
let started = false;

function set(patch: Partial<EnginesState>) {
  state = { ...state, ...patch };
  for (const listener of listeners) listener();
}

function packDispatch(action: PackAction) {
  set({ packs: packsReducer(state.packs, action) });
}

function start() {
  if (started) return;
  started = true;
  refreshCaps();
  refreshSecrets();
  void subsApi
    .subscribe((event) => {
      if (event.type === "packProgress" || event.type === "packDone") {
        packDispatch({ type: "event", event });
        if (event.type === "packDone") refreshCaps();
      }
    })
    .catch(() => undefined);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  start();
  return () => listeners.delete(listener);
}

const snapshot = () => state;

export function useEngines(): EnginesState {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

export function getEngines(): EnginesState {
  return state;
}

/** Whether a pack is being installed or removed (the engine is busy). */
export function packBusy(packs: PacksState = state.packs): boolean {
  return Object.values(packs).some((pack) => pack.busy);
}

export function refreshCaps(): void {
  void subsApi
    .capabilities()
    .then((caps) => set({ caps }))
    .catch(() => undefined);
}

export function refreshSecrets(): void {
  void subsApi
    .secretStatus()
    .then((secrets) => set({ secrets }))
    .catch(() => set({ secrets: null }));
}

/** Install or remove a pack (or one model inside it). Resolves with the
 *  error when it did not finish, null when it did. */
export async function runPack(action: "install" | "remove", pack: string, model?: string): Promise<string | null> {
  packDispatch({ type: "begin", pack, action });
  try {
    const result = await subsApi.pack(action, pack, model);
    packDispatch({ type: "done", pack, ok: result.ok, error: result.error, status: result.status });
    return result.ok ? null : (result.error ?? "That did not finish.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    packDispatch({ type: "done", pack, ok: false, error: message });
    return message;
  } finally {
    refreshCaps();
  }
}

/** Save a key in the shell's store (an empty value deletes it). Engines that
 *  need a key become available, so the capabilities are read again. */
export async function saveSecret(name: SecretName, value: string): Promise<void> {
  try {
    await subsApi.setSecret(name, value);
  } finally {
    refreshSecrets();
    refreshCaps();
  }
}

/** Ask Preferences › Subtitles to show a pack, or "keys". */
export function setFocus(focus: string | null): void {
  set({ focus });
}

/** For tests: back to nothing read. */
export function resetEngines(next: Partial<EnginesState> = {}): void {
  state = { ...INITIAL, ...next };
  for (const listener of listeners) listener();
}
