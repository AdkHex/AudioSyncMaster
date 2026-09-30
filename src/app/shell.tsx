/** The shell's contract with the pages.
 *
 *  Every page (Movies, Series, Find match, Dub sync, Subsync) is its own
 *  container: it keeps its own state and stays mounted while hidden, so
 *  switching pages loses nothing. What the pages share lives here:
 *
 *  - the engine, which runs one command at a time. A page claims it for a
 *    run and releases it after; the engine's streamed events go to the page
 *    that holds it, wherever the user has navigated since.
 *  - the menus, the Enter / Esc keys and OS file drops, which act on the
 *    page that is showing, through the commands it registers.
 *  - settings, history, the Output log, the dock's shared tabs, and the
 *    screen-reader announcer. */

import { createContext, useContext, useEffect, useRef } from "react";

import type { SyncListeners } from "@/lib/api";
import type { Announcement } from "@/lib/announce";
import type { AppSettings, HistoryEntry, SyncMode } from "@/lib/types";

import type { CommonDock } from "./dock";

/** What a page can be asked to do from outside it: the menu bar, the
 *  keyboard, and files dropped on the window. Any may be absent. */
export interface PageCommands {
  addVideos?: () => void;
  addAudio?: () => void;
  addFolder?: () => void;
  exportResults?: () => void;
  removeSelected?: () => void;
  clear?: () => void;
  /** Enter: run, when the page can. */
  start?: () => void;
  /** Esc during a run: stop it. */
  stop?: () => void;
  /** Paths dropped on the window while this page shows. */
  drop?: (paths: string[]) => void;
  /** Export the results as JSON (File menu); CSV is exportResults. */
  exportJson?: () => void;
  /** Show a past run from History on this page. */
  loadRun?: (entry: HistoryEntry) => void;
  /** Labels for the File menu's first two items, which differ per page. */
  addVideosLabel?: string;
  addAudioLabel?: string;
  /** Commands that exist but cannot act right now (nothing to export, a run
   *  going). Absent means available. */
  disabled?: Partial<Record<"addVideos" | "addAudio" | "addFolder" | "exportResults" | "exportJson" | "removeSelected" | "clear" | "start", boolean>>;
}

export interface Shell {
  /** The page showing. */
  active: SyncMode;
  show: (page: SyncMode) => void;
  desktop: boolean;

  settings: AppSettings;
  updateSettings: (patch: Partial<AppSettings>) => void;

  /** The page whose run holds the engine, or null when it is free. */
  enginePage: SyncMode | null;
  /** Take the engine for a run; false when another page holds it. */
  claimEngine: (page: SyncMode) => boolean;
  releaseEngine: (page: SyncMode) => void;
  /** Where the engine's streamed events for a page go while it holds the
   *  engine. Log lines always go to Output as well. */
  setEngineListeners: (page: SyncMode, listeners: Partial<SyncListeners> | null) => void;

  log: (message: string) => void;
  clearLogs: () => void;
  /** Open the Output tab of the dock. */
  openOutput: () => void;
  dock: CommonDock;

  history: HistoryEntry[];
  addHistory: (entry: HistoryEntry) => void;
  /** A history row asked to be reopened on this page. */
  announce: (announcement: Announcement) => void;

  setCommands: (page: SyncMode, commands: PageCommands | null) => void;
  /** Open Preferences, optionally at a tab. */
  openPreferences: (tab?: "general" | "analysis" | "dub" | "subs" | "updates") => void;
  /** Copy text to the clipboard with a toast. */
  copy: (text: string) => void;
}

export const ShellContext = createContext<Shell | null>(null);

export function useShell(): Shell {
  const shell = useContext(ShellContext);
  if (!shell) throw new Error("useShell must be used inside the app shell");
  return shell;
}

/** Register a page's commands with the shell for as long as it is mounted.
 *  The latest closures are always used, without re-registering each render. */
export function usePageCommands(page: SyncMode, commands: PageCommands) {
  const { setCommands } = useShell();
  const ref = useRef(commands);
  ref.current = commands;
  useEffect(() => {
    const proxy: PageCommands = {};
    const keys = Object.keys(ref.current) as (keyof PageCommands)[];
    for (const key of keys) {
      if (key === "addVideosLabel" || key === "addAudioLabel" || key === "disabled") continue;
      (proxy as Record<string, unknown>)[key] = (...args: unknown[]) =>
        (ref.current[key] as ((...a: unknown[]) => void) | undefined)?.(...args);
    }
    setCommands(page, {
      ...proxy,
      get addVideosLabel() {
        return ref.current.addVideosLabel;
      },
      get addAudioLabel() {
        return ref.current.addAudioLabel;
      },
      get disabled() {
        return ref.current.disabled;
      },
    });
    return () => setCommands(page, null);
    // Registered once per page; the proxy reads the latest commands.
  }, [page, setCommands]);
}

/** Route the engine's events for this page while it holds the engine. */
export function useEngineListeners(page: SyncMode, listeners: Partial<SyncListeners>) {
  const { setEngineListeners } = useShell();
  const ref = useRef(listeners);
  ref.current = listeners;
  useEffect(() => {
    const proxy: Partial<SyncListeners> = {};
    for (const key of Object.keys(ref.current) as (keyof SyncListeners)[]) {
      (proxy as Record<string, unknown>)[key] = (...args: unknown[]) =>
        (ref.current[key] as ((...a: unknown[]) => void) | undefined)?.(...args);
    }
    setEngineListeners(page, proxy);
    return () => setEngineListeners(page, null);
  }, [page, setEngineListeners]);
}
