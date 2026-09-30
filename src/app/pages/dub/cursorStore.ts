/** The playhead, shared by the timeline, the player and the editor's keys.
 *
 *  While the player runs it moves every frame; held in React state on the
 *  page it would re-render the whole page sixty times a second. Here only
 *  what draws it (the head line, the player's timecode) subscribes. */

import { useSyncExternalStore } from "react";

export interface CursorStore {
  get(): number | null;
  set(timeS: number | null): void;
  subscribe(listener: () => void): () => void;
}

export function createCursorStore(initial: number | null = null): CursorStore {
  let value = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => value,
    set(next) {
      if (next === value) return;
      value = next;
      listeners.forEach((listener) => listener());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export function useCursor(store: CursorStore): number | null {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}
