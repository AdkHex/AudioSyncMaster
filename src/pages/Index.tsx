import {
  Film,
  History as HistoryIcon,
  Play,
  Settings2,
  Square,
  Terminal,
  Trash2,
  Tv,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { toast } from "sonner";

import { ConsolePanel } from "@/components/ConsolePanel";
import { FilePanel } from "@/components/FilePanel";
import { HistoryPanel } from "@/components/HistoryPanel";
import { PairingPreview } from "@/components/PairingPreview";
import { ProgressPanel } from "@/components/ProgressPanel";
import { ResultsPanel } from "@/components/ResultsPanel";
import { SettingsDialog } from "@/components/SettingsDialog";
import { ThemeToggle } from "@/components/ThemeToggle";
import * as api from "@/lib/api";
import {
  createHistoryEntry,
  loadHistory,
  loadRecentFolders,
  loadSettings,
  saveHistory,
  saveRecentFolders,
  saveSettings,
} from "@/lib/storage";
import {
  estimateRemainingMs,
  initialSyncState,
  syncReducer,
  validateSelection,
} from "@/lib/syncReducer";
import type {
  AnalyzeRequest,
  AppSettings,
  CorrectionItem,
  FileItem,
  HistoryEntry,
  MediaProbe,
  SyncMode,
  SyncResult,
} from "@/lib/types";
import { resultKey } from "@/lib/types";

export default function Index() {
  const desktop = api.isDesktop();

  const [state, dispatch] = useReducer(syncReducer, initialSyncState);
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  const [recentFolders, setRecentFolders] = useState(() => loadRecentFolders());
  const [probes, setProbes] = useState<Record<string, MediaProbe>>({});
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());

  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showConsole, setShowConsole] = useState(false);
  const [dragTarget, setDragTarget] = useState<"video" | "audio" | null>(null);
  const [pairingLoading, setPairingLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [remainingMs, setRemainingMs] = useState<number | null>(null);

  /** Mirrors state for callbacks that must not be re-created on every change.
   *  The original captured stale values here: the keyboard handler closed over
   *  an old handleProcess, so pressing Enter after changing a setting ran with
   *  the previous configuration. */
  const stateRef = useRef(state);
  stateRef.current = state;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  useEffect(() => saveSettings(settings), [settings]);
  useEffect(() => saveRecentFolders(recentFolders), [recentFolders]);

  const persistHistory = useCallback((entries: HistoryEntry[]) => {
    // saveHistory returns what actually fit within the storage quota.
    setHistory(saveHistory(entries));
  }, []);

  // ------------------------------------------------------------- engine events

  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;

    api
      .subscribeToSync({
        onLog: (message) => dispatch({ type: "log", message }),
        onProgress: (event) =>
          dispatch({
            type: "progress",
            processed: event.processed,
            total: event.total,
            current: event.current,
          }),
        onFileStart: (file) => dispatch({ type: "fileStart", file }),
        onFileProgress: (event) =>
          dispatch({ type: "fileProgress", file: event.file, percent: event.percent }),
        onResult: (result) => dispatch({ type: "result", result }),
        onPairs: (pairing) => dispatch({ type: "setPairing", pairing }),
        onApplyProgress: (event) => {
          if (event.file) {
            dispatch({ type: "log", message: `Writing ${event.file}` });
          }
        },
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

  // Live ETA while a run is in flight.
  useEffect(() => {
    if (state.status !== "processing") {
      setRemainingMs(null);
      return;
    }
    const tick = () => setRemainingMs(estimateRemainingMs(stateRef.current));
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [state.status]);

  // ------------------------------------------------------------------- probing

  const probeFiles = useCallback(async (paths: string[]) => {
    for (const path of paths) {
      try {
        const probe = await api.probeMedia(path);
        setProbes((prev) => ({ ...prev, [path]: probe }));
      } catch {
        // A probe failure is not fatal; the run will report it properly.
      }
    }
  }, []);

  // ---------------------------------------------------------------- selection

  const addFiles = useCallback(
    (kind: "video" | "audio", files: FileItem[], folder: string | null) => {
      if (files.length === 0) return;

      if (kind === "audio" && stateRef.current.mode === "movie") {
        // Movie mode compares many videos against exactly one audio track.
        dispatch({ type: "replaceFiles", kind, files: [files[0]], folder });
        if (files.length > 1) {
          toast.info("Movie mode uses one audio track. Kept the first.");
        }
      } else {
        dispatch({ type: "addFiles", kind, files, folder, explicit: !folder });
      }

      if (folder) {
        setRecentFolders((prev) => ({ ...prev, [kind]: folder }));
      }
      void probeFiles(files.map((file) => file.path));
    },
    [probeFiles],
  );

  const handleBrowse = useCallback(
    async (kind: "video" | "audio") => {
      if (!desktop) {
        toast.error("File selection needs the desktop app.");
        return;
      }
      try {
        const wantsSingleAudio = kind === "audio" && stateRef.current.mode === "movie";
        const response = wantsSingleAudio
          ? await api.pickAudioFile()
          : kind === "audio"
            ? await api.pickAudioFolder()
            : await api.pickVideoFolder();

        if (response.files.length === 0) return;

        dispatch({
          type: "replaceFiles",
          kind,
          files: response.files,
          folder: response.folder,
          explicit: false,
        });
        if (response.folder) {
          setRecentFolders((prev) => ({ ...prev, [kind]: response.folder }));
        }
        void probeFiles(response.files.map((file) => file.path));
        toast.success(
          `Added ${response.files.length} ${kind} file${response.files.length === 1 ? "" : "s"}`,
        );
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Could not open the picker");
      }
    },
    [desktop, probeFiles],
  );

  // Native OS drag-and-drop. Webview File objects carry no path in Tauri v2,
  // so drops are only usable through this window event.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;

    api
      .subscribeToFileDrop(
        (paths) => {
          const kind: "video" | "audio" = dragTargetRef.current ?? "video";
          void api
            .resolveDroppedPaths(paths, kind)
            .then((files) => {
              if (files.length === 0) {
                toast.error("No supported media files in that drop.");
                return;
              }
              const folder = files[0].path.replace(/[\\/][^\\/]+$/, "");
              addFiles(kind, files, folder);
              toast.success(`Added ${files.length} file${files.length === 1 ? "" : "s"}`);
            })
            .catch(() => toast.error("Could not read the dropped files."));
          setDragTarget(null);
        },
        (hovering) => {
          if (!hovering) setDragTarget(null);
        },
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
  }, [addFiles]);

  // Which panel the pointer is over, for drop targeting.
  const dragTargetRef = useRef<"video" | "audio" | null>(null);
  dragTargetRef.current = dragTarget;

  // ------------------------------------------------------------ pairing preview

  const buildRequest = useCallback((): AnalyzeRequest => {
    const current = stateRef.current;
    const config = settingsRef.current;
    return {
      mode: current.mode,
      videoFolder: current.videoFolder,
      audioFolder: current.mode === "series" ? current.audioFolder : null,
      audioFile: current.mode === "movie" ? (current.audioFiles[0]?.path ?? null) : null,
      videoFiles: current.videoFiles.map((file) => file.path),
      matchPattern:
        current.mode === "series" && config.matchPattern.trim()
          ? config.matchPattern
          : null,
      windowSeconds: config.windowSeconds,
      windowCount: config.windowCount,
      maxOffsetMs: config.maxOffsetMs,
      maxWorkers: config.maxWorkers,
    };
  }, []);

  // Refresh the preview when the selection settles, so the user sees what will
  // be compared before committing to a long run.
  useEffect(() => {
    if (!desktop) return;
    if (state.videoFiles.length === 0 || state.audioFiles.length === 0) {
      dispatch({ type: "setPairing", pairing: null });
      return;
    }
    if (state.status === "processing") return;

    let active = true;
    setPairingLoading(true);
    const timer = window.setTimeout(() => {
      api
        .previewPairs(buildRequest())
        .then((pairing) => {
          if (active) dispatch({ type: "setPairing", pairing });
        })
        .catch(() => undefined)
        .finally(() => {
          if (active) setPairingLoading(false);
        });
    }, 200);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [
    desktop,
    state.videoFiles,
    state.audioFiles,
    state.mode,
    state.status,
    settings.matchPattern,
    buildRequest,
  ]);

  // ---------------------------------------------------------------- the run

  const handleStart = useCallback(async () => {
    const current = stateRef.current;
    if (current.status === "processing") return;

    const check = validateSelection(current);
    if (!check.ok) {
      toast.error(check.reason ?? "Nothing to analyse.");
      return;
    }
    if (!desktop) {
      toast.error("Analysis needs the desktop app.");
      return;
    }

    setSelectedKeys(new Set());
    dispatch({ type: "runStarted", total: current.videoFiles.length });

    try {
      const run = await api.startSync(buildRequest());
      dispatch({
        type: "runFinished",
        results: run.results,
        summary: run.summary,
        cancelled: run.cancelled,
      });

      if (run.cancelled) {
        toast.info("Analysis cancelled.");
      } else {
        const matched = run.summary?.matched ?? run.results.length;
        toast.success(`Analysed ${run.results.length} file(s)`, {
          description: `${matched} matched${
            run.summary?.drifting ? ` · ${run.summary.drifting} with drift` : ""
          }`,
        });
      }

      if (run.results.length > 0) {
        persistHistory([
          createHistoryEntry(current.mode, run.results, run.summary),
          ...history,
        ]);
        // Pre-select confident results so the common case is one click.
        setSelectedKeys(
          new Set(
            run.results
              .filter((r) => !r.error && r.delayMs !== null && (r.confidence ?? 0) >= 0.75)
              .map(resultKey),
          ),
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      dispatch({ type: "runFailed", message });
      toast.error("Analysis failed", { description: message });
      setShowConsole(true);
    }
  }, [desktop, buildRequest, history, persistHistory]);

  const handleCancel = useCallback(async () => {
    try {
      await api.cancelSync();
      toast.info("Stopping…");
    } catch {
      toast.error("Could not stop the run.");
    }
  }, []);

  const handleApply = useCallback(async () => {
    const current = stateRef.current;
    const items: CorrectionItem[] = current.results
      .filter((result) => selectedKeys.has(resultKey(result)))
      .filter((result) => result.delayMs !== null && result.primaryPath && result.secondaryPath)
      .map((result) => ({
        videoPath: result.primaryPath as string,
        audioPath: result.secondaryPath as string,
        delayMs: result.delayMs as number,
        delayAtStartMs: result.delayAtStartMs ?? result.delayMs,
        driftMsPerS: result.hasSignificantDrift ? result.driftMsPerS : null,
      }));

    if (items.length === 0) {
      toast.error("Select at least one measured result to fix.");
      return;
    }

    setApplying(true);
    try {
      const outcome = await api.applyCorrections(items, {
        suffix: settingsRef.current.outputSuffix,
      });
      if (outcome.written.length > 0) {
        toast.success(`Wrote ${outcome.written.length} corrected file(s)`, {
          description: outcome.failed.length
            ? `${outcome.failed.length} failed — see the console`
            : undefined,
          action: {
            label: "Show",
            onClick: () => void api.revealPath(outcome.written[0]).catch(() => undefined),
          },
        });
      }
      outcome.failed.forEach((failure) =>
        dispatch({ type: "log", message: `${failure.video}: ${failure.error}` }),
      );
      if (outcome.written.length === 0 && outcome.failed.length > 0) {
        toast.error("No files could be written.");
        setShowConsole(true);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not write files");
    } finally {
      setApplying(false);
    }
  }, [selectedKeys]);

  const handleExport = useCallback(
    async (results: SyncResult[], format: "csv" | "json") => {
      if (results.length === 0) return;
      try {
        const path =
          format === "csv" ? await api.exportCsv(results) : await api.exportJson(results);
        toast.success("Exported", {
          action: { label: "Show", onClick: () => void api.revealPath(path).catch(() => undefined) },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!message.toLowerCase().includes("cancel")) toast.error("Export failed");
      }
    },
    [],
  );

  const handleCopy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied");
    } catch {
      toast.error("Could not copy");
    }
  }, []);

  // ------------------------------------------------------------------ shortcuts

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }

      const meta = event.metaKey || event.ctrlKey;
      if (meta && event.key.toLowerCase() === "h") {
        event.preventDefault();
        setShowHistory((open) => !open);
      } else if (meta && event.key.toLowerCase() === ",") {
        event.preventDefault();
        setShowSettings(true);
      } else if (event.key === "Enter" && stateRef.current.status !== "processing") {
        event.preventDefault();
        void handleStart();
      } else if (event.key === "Escape") {
        if (stateRef.current.status === "processing") {
          event.preventDefault();
          void handleCancel();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleStart, handleCancel]);

  // ---------------------------------------------------------------- derived

  const selection = useMemo(() => validateSelection(state), [state]);
  const busy = state.status === "processing";

  const toggleSelection = useCallback((key: string) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const toggleAll = useCallback((keys: string[]) => {
    setSelectedKeys((prev) => {
      const allSelected = keys.length > 0 && keys.every((key) => prev.has(key));
      return allSelected ? new Set() : new Set(keys);
    });
  }, []);

  const setMode = useCallback((mode: SyncMode) => {
    dispatch({ type: "setMode", mode });
    setSelectedKeys(new Set());
    setProbes({});
  }, []);

  return (
    <div className="flex h-screen flex-col bg-background">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-card px-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden className="text-primary-foreground">
              <path d="M9 18V5l12-2v13" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="6" cy="18" r="3" stroke="currentColor" strokeWidth="2.5" />
              <circle cx="18" cy="16" r="3" stroke="currentColor" strokeWidth="2.5" />
            </svg>
          </span>
          <h1 className="text-sm font-semibold text-foreground">AudioSyncMaster</h1>
        </div>

        <div
          className="flex items-center gap-0.5 rounded-md bg-secondary p-0.5"
          role="tablist"
          aria-label="Sync mode"
        >
          {(
            [
              { id: "movie" as const, label: "Movies", Icon: Film },
              { id: "series" as const, label: "Series", Icon: Tv },
            ]
          ).map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={state.mode === id}
              onClick={() => setMode(id)}
              disabled={busy}
              className={`flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40 ${
                state.mode === id
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Icon className="h-3.5 w-3.5" aria-hidden />
              {label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          <ThemeToggle />
          <button
            type="button"
            onClick={() => setShowConsole((open) => !open)}
            className={`rounded p-1.5 transition-colors ${
              showConsole ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-secondary"
            }`}
            title="Console"
          >
            <Terminal className="h-4 w-4" aria-hidden />
            <span className="sr-only">Toggle console</span>
          </button>
          <button
            type="button"
            onClick={() => setShowHistory((open) => !open)}
            className={`rounded p-1.5 transition-colors ${
              showHistory ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-secondary"
            }`}
            title="History (Ctrl+H)"
          >
            <HistoryIcon className="h-4 w-4" aria-hidden />
            <span className="sr-only">Toggle history</span>
          </button>
          <button
            type="button"
            onClick={() => setShowSettings(true)}
            className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-secondary"
            title="Settings (Ctrl+,)"
          >
            <Settings2 className="h-4 w-4" aria-hidden />
            <span className="sr-only">Open settings</span>
          </button>
        </div>
      </header>

      <main className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1 overflow-y-auto p-4">
          <div className="mx-auto max-w-4xl space-y-3">
            {!desktop && (
              <p className="rounded-lg border border-warning/30 bg-warning/10 px-4 py-2 text-xs text-warning">
                Running in a browser. File selection and analysis need the desktop app.
              </p>
            )}

            <div
              className="grid gap-3 md:grid-cols-2"
              onDragOver={(event) => event.preventDefault()}
            >
              <div onDragEnter={() => setDragTarget("video")}>
                <FilePanel
                  kind="video"
                  title="Video files"
                  hint="The files whose timing is correct"
                  files={state.videoFiles}
                  folder={state.videoFolder}
                  recentFolder={recentFolders.video}
                  probes={probes}
                  dragActive={dragTarget === "video"}
                  disabled={busy}
                  onBrowse={() => void handleBrowse("video")}
                  onRemove={(id) => dispatch({ type: "removeFiles", kind: "video", ids: [id] })}
                  onClear={() => dispatch({ type: "clearFiles", kind: "video" })}
                />
              </div>
              <div onDragEnter={() => setDragTarget("audio")}>
                <FilePanel
                  kind="audio"
                  title={state.mode === "movie" ? "Audio track" : "Audio files"}
                  hint={
                    state.mode === "movie"
                      ? "The track to align against the videos"
                      : "One track per episode"
                  }
                  files={state.audioFiles}
                  folder={state.audioFolder}
                  recentFolder={recentFolders.audio}
                  probes={probes}
                  dragActive={dragTarget === "audio"}
                  disabled={busy}
                  onBrowse={() => void handleBrowse("audio")}
                  onRemove={(id) => dispatch({ type: "removeFiles", kind: "audio", ids: [id] })}
                  onClear={() => dispatch({ type: "clearFiles", kind: "audio" })}
                />
              </div>
            </div>

            {state.status !== "processing" && (
              <PairingPreview pairing={state.pairing} loading={pairingLoading} />
            )}

            {busy && (
              <ProgressPanel
                processed={state.progress.processed}
                total={state.progress.total}
                currentFile={state.currentFile}
                fileProgress={state.fileProgress}
                remainingMs={remainingMs}
              />
            )}

            <div className="flex flex-wrap items-center gap-2">
              {busy ? (
                <button
                  type="button"
                  onClick={() => void handleCancel()}
                  className="flex items-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-secondary"
                >
                  <Square className="h-3.5 w-3.5" aria-hidden />
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => void handleStart()}
                  disabled={!selection.ok}
                  title={selection.reason}
                  className="flex items-center gap-2 rounded-md bg-primary px-5 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Play className="h-4 w-4" aria-hidden />
                  Analyse
                </button>
              )}

              {(state.videoFiles.length > 0 || state.results.length > 0) && !busy && (
                <button
                  type="button"
                  onClick={() => {
                    dispatch({ type: "clearAll" });
                    setSelectedKeys(new Set());
                    setProbes({});
                  }}
                  className="rounded p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                  title="Clear everything"
                >
                  <Trash2 className="h-4 w-4" aria-hidden />
                  <span className="sr-only">Clear everything</span>
                </button>
              )}

              {!selection.ok && state.videoFiles.length > 0 && (
                <span className="text-xs text-muted-foreground">{selection.reason}</span>
              )}
            </div>

            {state.error && (
              <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-xs text-destructive">
                {state.error}
              </p>
            )}

            <ResultsPanel
              results={state.results}
              summary={state.summary}
              selectedKeys={selectedKeys}
              onToggleSelection={toggleSelection}
              onToggleAll={toggleAll}
              onExportCsv={() => void handleExport(state.results, "csv")}
              onExportJson={() => void handleExport(state.results, "json")}
              onApply={() => void handleApply()}
              onCopy={(text) => void handleCopy(text)}
              applying={applying}
            />

            {showConsole && (
              <ConsolePanel
                logs={state.logs}
                onClear={() => dispatch({ type: "clearLogs" })}
              />
            )}
          </div>
        </div>

        {showHistory && (
          <HistoryPanel
            entries={history}
            onLoad={(entry) => {
              dispatch({
                type: "loadResults",
                results: entry.results,
                summary: entry.summary,
                mode: entry.mode,
              });
              setSelectedKeys(new Set());
              setShowHistory(false);
              toast.success("Loaded that run");
            }}
            onExport={(entry) => void handleExport(entry.results, "csv")}
            onDelete={(id) => persistHistory(history.filter((entry) => entry.id !== id))}
            onClear={() => persistHistory([])}
          />
        )}
      </main>

      <footer className="flex h-7 shrink-0 items-center justify-between border-t border-border bg-card/50 px-4 text-[10px] text-muted-foreground">
        <span>{state.mode === "movie" ? "Movie" : "Series"} mode</span>
        <span className="opacity-70">
          Enter analyse · Esc stop · Ctrl+H history · Ctrl+, settings
        </span>
      </footer>

      <SettingsDialog
        open={showSettings}
        settings={settings}
        mode={state.mode}
        onChange={setSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  );
}
