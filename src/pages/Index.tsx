/** The app shell: one window frame (title bar with the menu bar, the page
 *  bar along the bottom) around five pages that each keep their own state.
 *
 *  What the pages share lives here: settings, history, the Output log, the
 *  engine (one command at a time; its events are routed to the page that
 *  started the run), menus, keyboard shortcuts, OS file drops, Preferences
 *  and updates. See src/app/shell.tsx for the contract. */

import {
  ClosedCaptionFilled,
  ClosedCaptionRegular,
  ColumnDoubleCompareFilled,
  ColumnDoubleCompareRegular,
  FilmstripFilled,
  FilmstripRegular,
  HeadphonesSoundWaveFilled,
  HeadphonesSoundWaveRegular,
  MoviesAndTvFilled,
  MoviesAndTvRegular,
  VideoClipMultipleRegular,
} from "@fluentui/react-icons";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { toast } from "sonner";

import { HistoryBody, HistoryTools, OutputBody, OutputTools, PAGE_LABEL, useHistoryFilter, type CommonTab } from "@/app/dock";
import { AnalysePage } from "@/app/pages/analyse/AnalysePage";
import { DubPage } from "@/app/pages/DubPage";
import { SubsyncPage } from "@/app/pages/SubsyncPage";
import { EnginesSection } from "@/app/pages/subsync/EnginesSection";
// TEMPORARY while the Dub sync and Subsync pages are being built: load them
// only if present. Replaced by static imports once they exist.
import { PreferencesWindow, type PrefsTab } from "@/app/Preferences";
import { ShellContext, type PageCommands, type Shell } from "@/app/shell";
import { UpdateProgress } from "@/app/UpdateProgress";
import { LiveAnnouncer } from "@/components/LiveAnnouncer";
import type { Announcement } from "@/lib/announce";
import * as api from "@/lib/api";
import * as subsApi from "@/lib/subsync/api";
import { loadHistory, loadSettings, saveHistory, saveSettings } from "@/lib/storage";
import type { AppSettings, HistoryEntry, SyncMode } from "@/lib/types";
import { checkForUpdate, skipVersion, type UpdateInfo } from "@/lib/updater";
import { AppWindow, IS_MAC, PageBar, type Menu, type PageTab } from "@/ui/frame";

/** Injected from package.json at build time, so Preferences always reports
 *  the version CI tagged the release with. */
const APP_VERSION = __APP_VERSION__;

const MAX_LOG_LINES = 500;

const PAGES: PageTab<SyncMode>[] = [
  { id: "movie", label: "Movies", icon: <FilmstripRegular />, on: <FilmstripFilled />, shortcut: "Ctrl+1" },
  { id: "series", label: "Series", icon: <MoviesAndTvRegular />, on: <MoviesAndTvFilled />, shortcut: "Ctrl+2" },
  { id: "compare", label: "Find match", icon: <ColumnDoubleCompareRegular />, on: <ColumnDoubleCompareFilled />, shortcut: "Ctrl+3" },
  { id: "dubsync", label: "Dub sync", icon: <HeadphonesSoundWaveRegular />, on: <HeadphonesSoundWaveFilled />, shortcut: "Ctrl+4" },
  { id: "subsync", label: "Subsync", icon: <ClosedCaptionRegular />, on: <ClosedCaptionFilled />, shortcut: "Ctrl+5" },
];

function logReducer(state: string[], action: { type: "add"; message: string } | { type: "clear" }): string[] {
  if (action.type === "clear") return [];
  const next = state.length >= MAX_LOG_LINES ? state.slice(state.length - MAX_LOG_LINES + 1) : state.slice();
  next.push(action.message);
  return next;
}

const openReleaseNotes = () =>
  void import("@tauri-apps/api/core")
    .then(({ invoke }) => invoke("plugin:opener|open_url", { url: "https://github.com/AdkHex/AudioSyncMaster/releases" }))
    .catch(() => undefined);

const VIDEO_EXT = /\.(mkv|mp4|m4v|mov|avi|ts|m2ts|mts|webm|wmv|flv|mpg|mpeg|vob)$/i;
const DUB_EXT = /\.(ac3|eac3|ec3|aac|dts|thd|truehd|flac|wav|mka|mp3|m4a|opus|ogg)$/i;
const SUB_EXT = /\.(srt|ass|ssa|vtt|sub|sup|idx|ttml|sbv)$/i;

/** "Drop to add 6 videos": what the drop holds, in the words of the pages. */
function dropLabel(paths: string[]): string {
  const count = (n: number, one: string) => `${n} ${one}${n === 1 ? "" : "s"}`;
  const names = paths.map((p) => p.replace(/[\\/]+$/, "").replace(/^.*[\\/]/, ""));
  const videos = names.filter((n) => VIDEO_EXT.test(n)).length;
  const dubs = names.filter((n) => DUB_EXT.test(n)).length;
  const subs = names.filter((n) => SUB_EXT.test(n)).length;
  const folders = names.filter((n) => !/\.[a-z][a-z0-9]{1,4}$/i.test(n)).length;
  if (names.length === 0) return "Drop to add files";
  if (folders === names.length) return `Drop to add ${count(folders, "folder")}`;
  if (subs === names.length) return `Drop to add ${count(subs, "subtitle")}`;
  if (videos + dubs === names.length) {
    const parts = [videos && count(videos, "video"), dubs && count(dubs, "dub")].filter(Boolean);
    return `Drop to add ${parts.join(" and ")}`;
  }
  return `Drop to add ${count(names.length, "file")}`;
}

const isTyping = (target: EventTarget | null) => {
  const el = target as HTMLElement | null;
  return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
};

export default function Index() {
  const desktop = api.isDesktop();

  const [active, setActive] = useState<SyncMode>("movie");
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  const [logs, logDispatch] = useReducer(logReducer, []);
  const [enginePage, setEnginePage] = useState<SyncMode | null>(null);
  const [dockTab, setDockTab] = useState<CommonTab | null>(null);
  const [prefs, setPrefs] = useState<PrefsTab | null>(null);
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [installing, setInstalling] = useState(false);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [announcement, setAnnouncement] = useState<Announcement | null>(null);
  /** What is being dragged over the window: null when nothing is, [] when
   *  the paths are not known yet. */
  const [dragging, setDragging] = useState<string[] | null>(null);
  const [ffmpeg, setFfmpeg] = useState<string | null>(null);
  const [historyFilter, setHistoryFilter] = useHistoryFilter();

  const activeRef = useRef(active);
  activeRef.current = active;
  const engineRef = useRef<SyncMode | null>(null);
  const listenersRef = useRef<Partial<Record<SyncMode, Partial<api.SyncListeners>>>>({});
  const commandsRef = useRef<Partial<Record<SyncMode, PageCommands>>>({});
  const [, bumpCommands] = useReducer((n: number) => n + 1, 0);

  useEffect(() => saveSettings(settings), [settings]);

  // ------------------------------------------------------------- engine

  const claimEngine = useCallback((page: SyncMode) => {
    if (engineRef.current && engineRef.current !== page) return false;
    engineRef.current = page;
    setEnginePage(page);
    return true;
  }, []);

  const releaseEngine = useCallback((page: SyncMode) => {
    if (engineRef.current !== page) return;
    engineRef.current = null;
    setEnginePage(null);
  }, []);

  const setEngineListeners = useCallback((page: SyncMode, listeners: Partial<api.SyncListeners> | null) => {
    if (listeners) listenersRef.current[page] = listeners;
    else delete listenersRef.current[page];
  }, []);

  // One subscription for the whole app. Each event goes to the page holding
  // the engine; waveform reads belong to Dub sync whatever is running.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;
    const route = <K extends keyof api.SyncListeners>(key: K) =>
      ((...args: unknown[]) => {
        const page = key === "onWaveformProgress" ? "dubsync" : engineRef.current;
        const handler = page ? listenersRef.current[page]?.[key] : undefined;
        (handler as ((...a: unknown[]) => void) | undefined)?.(...args);
      }) as NonNullable<api.SyncListeners[K]>;
    api
      .subscribeToSync({
        onLog: (message) => {
          logDispatch({ type: "add", message });
          const page = engineRef.current;
          if (page) listenersRef.current[page]?.onLog?.(message);
        },
        onProgress: route("onProgress"),
        onFileStart: route("onFileStart"),
        onFileProgress: route("onFileProgress"),
        onResult: route("onResult"),
        onDone: route("onDone"),
        onPairs: route("onPairs"),
        onApplyProgress: route("onApplyProgress"),
        onDubSyncProgress: route("onDubSyncProgress"),
        onDubSyncDraft: route("onDubSyncDraft"),
        onDubSyncPlan: route("onDubSyncPlan"),
        onDubQueueJobStart: route("onDubQueueJobStart"),
        onDubQueueJobProgress: route("onDubQueueJobProgress"),
        onDubQueueJobDraft: route("onDubQueueJobDraft"),
        onDubQueueJobPlan: route("onDubQueueJobPlan"),
        onDubQueueJobDone: route("onDubQueueJobDone"),
        onWaveformProgress: route("onWaveformProgress"),
      })
      .then((unlisten) => {
        if (cancelled) unlisten();
        else dispose = unlisten;
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      dispose?.();
    };
  }, []);

  // ------------------------------------------------------------- pages

  const setCommands = useCallback((page: SyncMode, commands: PageCommands | null) => {
    if (commands) commandsRef.current[page] = commands;
    else delete commandsRef.current[page];
    bumpCommands();
  }, []);
  const command = useCallback(<K extends keyof PageCommands>(key: K) => commandsRef.current[activeRef.current]?.[key], []);

  const show = useCallback((page: SyncMode) => setActive(page), []);

  // The OS window's title follows the page, for the taskbar and Alt+Tab.
  useEffect(() => {
    if (!desktop) return;
    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().setTitle(`${PAGE_LABEL[active]} — AudioSyncMaster`))
      .catch(() => undefined);
  }, [active, desktop]);

  // OS file drops go to the page that shows.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;
    api
      .subscribeToFileDrop(
        (paths) => {
          setDragging(null);
          const drop = commandsRef.current[activeRef.current]?.drop;
          if (drop) drop(paths);
          else toast.error("This page does not take files.");
        },
        (hovering, paths) => setDragging((current) => (hovering ? (paths ?? current ?? []) : null)),
      )
      .then((unlisten) => {
        if (cancelled) unlisten();
        else dispose = unlisten;
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      dispose?.();
    };
  }, []);

  // ------------------------------------------------------------- history

  const persistHistory = useCallback((entries: HistoryEntry[]) => setHistory(saveHistory(entries)), []);
  const addHistory = useCallback((entry: HistoryEntry) => setHistory((current) => saveHistory([entry, ...current])), []);

  const openHistoryEntry = useCallback(
    (entry: HistoryEntry) => {
      setActive(entry.mode);
      if (entry.results.length > 0) {
        // Let the page mount its commands before asking it to load.
        window.setTimeout(() => commandsRef.current[entry.mode]?.loadRun?.(entry), 0);
      }
    },
    [],
  );

  // ------------------------------------------------------------- updates

  useEffect(() => {
    if (settings.autoUpdateCheck === false) return;
    const timer = window.setTimeout(() => {
      void checkForUpdate().then((found) => {
        if (!found) return;
        setUpdate(found);
        toast(`Version ${found.version} is ready`, {
          action: { label: "Update", onClick: () => setPrefs("updates") },
        });
      });
    }, 3000);
    return () => window.clearTimeout(timer);
    // Once, on launch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCheckForUpdate = useCallback(async () => {
    setCheckingUpdate(true);
    try {
      const found = await checkForUpdate(true);
      if (found) setUpdate(found);
      else toast.success("You are on the latest version.");
    } catch {
      toast.error("Could not check for updates.");
    } finally {
      setCheckingUpdate(false);
    }
  }, []);

  // The engine's FFmpeg, for the page bar and About.
  useEffect(() => {
    if (!desktop) return;
    void subsApi
      .capabilities()
      .then((caps) => caps?.ffmpeg.version && setFfmpeg(caps.ffmpeg.version.replace(/^n?(\d+(\.\d+)*).*$/, "$1")))
      .catch(() => undefined);
  }, [desktop]);

  // ------------------------------------------------------------- shell

  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied");
    } catch {
      toast.error("Could not copy");
    }
  }, []);

  const log = useCallback((message: string) => logDispatch({ type: "add", message }), []);
  const clearLogs = useCallback(() => logDispatch({ type: "clear" }), []);
  const openOutput = useCallback(() => setDockTab("output"), []);
  const announce = useCallback((a: Announcement) => setAnnouncement(a), []);
  const updateSettings = useCallback((patch: Partial<AppSettings>) => setSettings((current) => ({ ...current, ...patch })), []);
  const openPreferences = useCallback((tab: PrefsTab = "general") => setPrefs(tab), []);

  const dock = useMemo(
    () => ({
      tab: dockTab,
      setTab: setDockTab,
      output: <OutputBody logs={logs} />,
      history: (
        <HistoryBody
          entries={history}
          filter={historyFilter}
          onOpen={openHistoryEntry}
          onDelete={(id) => persistHistory(history.filter((entry) => entry.id !== id))}
        />
      ),
      outputTools: <OutputTools logs={logs} onCopy={(text) => void copy(text)} onClear={clearLogs} />,
      historyTools: <HistoryTools filter={historyFilter} onFilter={setHistoryFilter} empty={history.length === 0} onClear={() => persistHistory([])} />,
    }),
    [dockTab, logs, history, historyFilter, openHistoryEntry, persistHistory, copy, clearLogs, setHistoryFilter],
  );

  const shell: Shell = useMemo(
    () => ({
      active,
      show,
      desktop,
      settings,
      updateSettings,
      enginePage,
      claimEngine,
      releaseEngine,
      setEngineListeners,
      log,
      clearLogs,
      openOutput,
      dock,
      history,
      addHistory,
      announce,
      setCommands,
      openPreferences,
      copy: (text: string) => void copy(text),
    }),
    [active, show, desktop, settings, updateSettings, enginePage, claimEngine, releaseEngine, setEngineListeners, log, clearLogs, openOutput, dock, history, addHistory, announce, setCommands, openPreferences, copy],
  );

  // ------------------------------------------------------------- keys

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey;
      const key = event.key.toLowerCase();
      if (meta && key === "h") {
        event.preventDefault();
        setDockTab((tab) => (tab === "history" ? null : "history"));
      } else if (meta && key === "`") {
        event.preventDefault();
        setDockTab((tab) => (tab === "output" ? null : "output"));
      } else if (meta && key === ",") {
        event.preventDefault();
        setPrefs("general");
      } else if (meta && /^[1-5]$/.test(key)) {
        event.preventDefault();
        setActive(PAGES[Number(key) - 1].id);
      } else if (meta && key === "o" && !event.shiftKey) {
        event.preventDefault();
        command("addVideos")?.();
      } else if (meta && key === "o" && event.shiftKey) {
        event.preventDefault();
        command("addAudio")?.();
      } else if (meta && key === "e") {
        event.preventDefault();
        command("exportResults")?.();
      } else if (isTyping(event.target) || prefs || document.querySelector(".smoke")) {
        return;
      } else if (event.key === "Enter" && !event.defaultPrevented && (event.target as HTMLElement)?.tagName !== "BUTTON") {
        // Subsync runs from its own button: Enter stays the key that presses
        // whatever has focus there.
        if (activeRef.current === "subsync") return;
        event.preventDefault();
        command("start")?.();
      } else if (event.key === "Escape" && engineRef.current === activeRef.current) {
        event.preventDefault();
        command("stop")?.();
      } else if (event.key === "Delete" && !event.defaultPrevented) {
        command("removeSelected")?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [command, prefs]);

  // ------------------------------------------------------------- menus

  const buildMenus = (): Menu[] => {
  const commands = commandsRef.current[activeRef.current] ?? {};
  const off = (key: keyof NonNullable<PageCommands["disabled"]>) => !commands[key] || commands.disabled?.[key] === true;
  return [
    {
      label: "File",
      items: [
        { label: commands.addVideosLabel ?? "Add videos…", shortcut: "Ctrl+O", onSelect: commands.addVideos, disabled: off("addVideos") },
        { label: commands.addAudioLabel ?? "Add a dub…", shortcut: "Ctrl+Shift+O", onSelect: commands.addAudio, disabled: off("addAudio") },
        { label: "Add a folder…", onSelect: commands.addFolder, disabled: off("addFolder") },
        "separator",
        { label: "Export results…", shortcut: "Ctrl+E", onSelect: commands.exportResults, disabled: off("exportResults") },
        { label: "Export as JSON…", onSelect: commands.exportJson, disabled: off("exportJson") },
        "separator",
        { label: "Preferences…", shortcut: "Ctrl+,", onSelect: () => setPrefs("general") },
        "separator",
        {
          label: "Exit",
          shortcut: "Alt+F4",
          onSelect: () => void import("@tauri-apps/api/window").then(({ getCurrentWindow }) => getCurrentWindow().close()).catch(() => undefined),
        },
      ],
    },
    {
      label: "Edit",
      items: [
        { label: "Remove", shortcut: "Del", onSelect: commands.removeSelected, disabled: off("removeSelected") },
        { label: "Clear the list", onSelect: commands.clear, disabled: off("clear") },
      ],
    },
    {
      label: "View",
      items: [
        ...PAGES.map((p) => ({ label: p.label, shortcut: p.shortcut, checked: p.id === activeRef.current, onSelect: () => setActive(p.id) })),
        "separator" as const,
        { label: "Output", shortcut: "Ctrl+`", checked: dockTab === "output", onSelect: () => setDockTab((t) => (t === "output" ? null : "output")) },
        { label: "History", shortcut: "Ctrl+H", checked: dockTab === "history", onSelect: () => setDockTab((t) => (t === "history" ? null : "history")) },
      ],
    },
    {
      label: "Tools",
      items: [
        { label: "Engines and API keys…", onSelect: () => setPrefs("subs") },
        { label: "Voice tools…", onSelect: () => setPrefs("dub") },
        "separator",
        { label: "Preferences…", shortcut: "Ctrl+,", onSelect: () => setPrefs("general") },
      ],
    },
    {
      label: "Help",
      items: [
        { label: "Check for updates…", onSelect: () => setPrefs("updates") },
        { label: "Release notes", onSelect: openReleaseNotes },
        "separator",
        { label: `About AudioSyncMaster ${APP_VERSION}`, onSelect: () => setPrefs("updates") },
      ],
    },
  ];
  };

  // macOS: the native menus (src-tauri app_menu) send their item ids here.
  const menuActions = useRef<Record<string, () => void>>({});
  menuActions.current = {
    prefs: () => setPrefs("general"),
    "add-videos": () => command("addVideos")?.(),
    "add-audio": () => command("addAudio")?.(),
    "add-folder": () => command("addFolder")?.(),
    export: () => command("exportResults")?.(),
    "export-json": () => command("exportJson")?.(),
    remove: () => command("removeSelected")?.(),
    clear: () => command("clear")?.(),
    output: () => setDockTab((t) => (t === "output" ? null : "output")),
    history: () => setDockTab((t) => (t === "history" ? null : "history")),
    engines: () => setPrefs("subs"),
    "voice-tools": () => setPrefs("dub"),
    updates: () => setPrefs("updates"),
    "release-notes": openReleaseNotes,
    ...Object.fromEntries(PAGES.map((p) => [`page-${p.id}`, () => setActive(p.id)])),
  };
  useEffect(() => {
    if (!desktop || !IS_MAC) return;
    let dispose: (() => void) | undefined;
    let cancelled = false;
    void import("@tauri-apps/api/event")
      .then(({ listen }) => listen<string>("app-menu", (event) => menuActions.current[event.payload]?.()))
      .then((unlisten) => {
        if (cancelled) unlisten();
        else dispose = unlisten;
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      dispose?.();
    };
  }, [desktop]);

  // ------------------------------------------------------------- render

  const note = update ? (
    <button type="button" className="acc" onClick={() => setPrefs("updates")}>Version {update.version} is ready — Update</button>
  ) : ffmpeg ? (
    `FFmpeg ${ffmpeg}`
  ) : desktop ? null : (
    "Browser preview — files and runs need the desktop app"
  );

  const busyPage = enginePage && enginePage !== active ? enginePage : null;

  return (
    <ShellContext.Provider value={shell}>
      <AppWindow
        title={PAGE_LABEL[active]}
        menus={IS_MAC ? [] : buildMenus}
        footer={
          <PageBar
            pages={PAGES}
            page={active}
            onPage={setActive}
            busy={busyPage}
            note={note}
            history={dockTab === "history"}
            output={dockTab === "output"}
            onHistory={() => setDockTab((t) => (t === "history" ? null : "history"))}
            onOutput={() => setDockTab((t) => (t === "output" ? null : "output"))}
            onPreferences={() => setPrefs("general")}
          />
        }
        overlay={
          <>
            {dragging && (
              <div className="dropover" style={{ inset: "89px 7px 59px" }}>
                <span className="ic"><VideoClipMultipleRegular /></span>
                {dropLabel(dragging)}
              </div>
            )}
            {prefs && (
              <PreferencesWindow
                tab={prefs}
                onTab={setPrefs}
                settings={settings}
                onChange={setSettings}
                onClose={() => setPrefs(null)}
                version={APP_VERSION}
                busy={enginePage !== null}
                update={update}
                onInstallUpdate={() => {
                  setPrefs(null);
                  setInstalling(true);
                }}
                onSkipUpdate={() => {
                  if (update) skipVersion(update.version);
                  setUpdate(null);
                }}
                onCheckForUpdate={() => void handleCheckForUpdate()}
                checkingUpdate={checkingUpdate}
                ffmpeg={ffmpeg}
                subtitles={<EnginesSection />}
              />
            )}
            {installing && update && <UpdateProgress update={update} onClose={() => setInstalling(false)} />}
          </>
        }
      >
        <AnalysePage mode="movie" hidden={active !== "movie"} />
        <AnalysePage mode="series" hidden={active !== "series"} />
        <AnalysePage mode="compare" hidden={active !== "compare"} />
        <DubPage hidden={active !== "dubsync"} />
        <SubsyncPage hidden={active !== "subsync"} />
      </AppWindow>
      <LiveAnnouncer announcement={announcement} />
    </ShellContext.Provider>
  );
}
