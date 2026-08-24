import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { toast } from "sonner";

import { AppHeader } from "@/components/AppHeader";
import { ApplyProgressDialog, type ApplyState } from "@/components/ApplyProgressDialog";
import { ConsolePanel } from "@/components/ConsolePanel";
import { HistoryPanel } from "@/components/HistoryPanel";
import { LiveAnnouncer } from "@/components/LiveAnnouncer";
import { PairingPreview } from "@/components/PairingPreview";
import { ProgressPanel } from "@/components/ProgressPanel";
import { ResultsPanel } from "@/components/ResultsPanel";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Sidebar } from "@/components/Sidebar";
import { StatusBar } from "@/components/StatusBar";
import { UpdateDialog } from "@/components/UpdateDialog";
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
  PairingReport,
  SyncMode,
  TrackListing,
  SyncResult,
} from "@/lib/types";
import { MAX_COMPARE_INPUTS, resultKey } from "@/lib/types";
import {
  announceApplyFinished,
  announceProgress,
  announceRunFailed,
  announceRunFinished,
  announceRunStarted,
  type Announcement,
} from "@/lib/announce";
import {
  applyOverrides,
  countManualPairs,
  pruneOverrides,
  type PairOverrides,
} from "@/lib/pairing";
import { checkForUpdate, type UpdateInfo } from "@/lib/updater";

/** Injected from package.json at build time, so Settings always reports the
 *  version CI actually tagged the release with. */
const APP_VERSION = __APP_VERSION__;

export default function Index() {
  const desktop = api.isDesktop();

  const [state, dispatch] = useReducer(syncReducer, initialSyncState);
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  const [recentFolders, setRecentFolders] = useState(() => loadRecentFolders());
  const [probes, setProbes] = useState<Record<string, MediaProbe>>({});
  // Audio streams of every selected file, keyed by path, and which stream to
  // compare for each. Files routinely carry several; without a choice every run
  // silently used the first, which on a disc rip is often a commentary track.
  // Keyed per file rather than per side, because a selection can mix sources
  // whose track layouts have nothing in common.
  const [listings, setListings] = useState<Record<string, TrackListing>>({});
  const [trackChoices, setTrackChoices] = useState<Record<string, number>>({});
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());

  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showConsole, setShowConsole] = useState(false);
  const [dragTarget, setDragTarget] = useState<"video" | "audio" | null>(null);
  const [pairingLoading, setPairingLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [previewingKey, setPreviewingKey] = useState<string | null>(null);
  // Hand corrections to the automatic pairing, keyed by video path so they
  // survive a re-match when the pattern or selection changes.
  const [pairOverrides, setPairOverrides] = useState<PairOverrides>({});
  const [applyState, setApplyState] = useState<ApplyState | null>(null);
  const [cancellingApply, setCancellingApply] = useState(false);
  const [remainingMs, setRemainingMs] = useState<number | null>(null);
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  // What a screen reader should say next. Analysis is long-running and almost
  // entirely visual, so without this its progress and outcome are invisible.
  const [announcement, setAnnouncement] = useState<Announcement | null>(null);

  /** Mirrors state for callbacks that must not be re-created on every change.
   *  The original captured stale values here: the keyboard handler closed over
   *  an old handleProcess, so pressing Enter after changing a setting ran with
   *  the previous configuration. */
  const stateRef = useRef(state);
  stateRef.current = state;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  // Read through refs: buildRequest must not be re-created on every track
  // change, or the pairing-preview effect that depends on it re-runs endlessly.
  const trackChoicesRef = useRef(trackChoices);
  trackChoicesRef.current = trackChoices;
  const overridesRef = useRef(pairOverrides);
  overridesRef.current = pairOverrides;
  // Read through a ref so buildRequest stays stable; it is called from the
  // keyboard handler, which must not close over a stale pairing.
  const effectivePairingRef = useRef<PairingReport | null>(null);

  useEffect(() => saveSettings(settings), [settings]);
  useEffect(() => saveRecentFolders(recentFolders), [recentFolders]);

  const persistHistory = useCallback((entries: HistoryEntry[]) => {
    // saveHistory returns what actually fit within the storage quota.
    setHistory(saveHistory(entries));
  }, []);

  // Check for a newer release shortly after launch. Delayed so it never
  // competes with first paint, and silent on failure so being offline or
  // behind a proxy cannot stop the app from starting.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void checkForUpdate().then((found) => {
        if (found) setUpdate(found);
      });
    }, 3000);
    return () => window.clearTimeout(timer);
  }, []);

  const handleCheckForUpdate = useCallback(async () => {
    setCheckingUpdate(true);
    try {
      const found = await checkForUpdate(true);
      if (found) {
        setUpdate(found);
      } else {
        toast.success("You are on the latest version.");
      }
    } catch {
      toast.error("Could not check for updates.");
    } finally {
      setCheckingUpdate(false);
    }
  }, []);

  // ------------------------------------------------------------- engine events

  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;

    api
      .subscribeToSync({
        onLog: (message) => dispatch({ type: "log", message }),
        onProgress: (event) => {
          dispatch({
            type: "progress",
            processed: event.processed,
            total: event.total,
            current: event.current,
          });
          const milestone = announceProgress(event.processed, event.total);
          if (milestone) setAnnouncement(milestone);
        },
        onFileStart: (file) => dispatch({ type: "fileStart", file }),
        onFileProgress: (event) =>
          dispatch({ type: "fileProgress", file: event.file, percent: event.percent }),
        onResult: (result) => dispatch({ type: "result", result }),
        onPairs: (pairing) => dispatch({ type: "setPairing", pairing }),
        onApplyProgress: (event) => {
          if (event.file) {
            dispatch({ type: "log", message: `Writing ${event.file}` });
          }
          // applyStart names the file about to be written; applyProgress
          // confirms one finished and carries the running totals.
          setApplyState((prev) => {
            const base: ApplyState = prev ?? {
              written: [],
              current: null,
              done: 0,
              total: 0,
            };
            const finished = typeof event.done === "number";
            return {
              written: finished && event.output ? [...base.written, event.output] : base.written,
              current: finished ? null : (event.file ?? base.current),
              done: event.done ?? base.done,
              total: event.total ?? base.total,
            };
          });
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

  // Read the audio streams of every selected file.
  //
  // This used to probe only the first file on each side and apply that one
  // choice to all of them, on the assumption that a season folder is encoded
  // consistently. Mixed sources break that: a REMUX with five language tracks
  // beside a WEB-DL with one leaves the second file's dropdown describing
  // streams it does not have, and no way to pick per file.
  useEffect(() => {
    const paths = [
      ...state.videoFiles.map((file) => file.path),
      ...state.audioFiles.map((file) => file.path),
    ];
    if (paths.length === 0) {
      setListings({});
      setTrackChoices({});
      return;
    }

    let active = true;
    void api
      .listAudioTracks(paths)
      .then((entries) => {
        if (!active) return;
        const byPath: Record<string, TrackListing> = {};
        entries.forEach((entry) => {
          byPath[entry.path] = entry;
        });
        setListings(byPath);
        // Drop choices that the newly probed files can no longer satisfy,
        // rather than sending the engine a stream index that does not exist.
        setTrackChoices((current) => {
          const next: Record<string, number> = {};
          Object.entries(current).forEach(([path, index]) => {
            const available = byPath[path]?.tracks.length ?? 0;
            if (index > 0 && index < available) next[path] = index;
          });
          return next;
        });
      })
      .catch(() => undefined);

    return () => {
      active = false;
    };
  }, [state.videoFiles, state.audioFiles]);

  // Drop corrections whose files are no longer selected, so the engine is
  // never asked to analyse something that has been removed.
  useEffect(() => {
    setPairOverrides((current) => {
      if (Object.keys(current).length === 0) return current;
      const pruned = pruneOverrides(
        current,
        state.videoFiles.map((file) => file.path),
        state.audioFiles.map((file) => file.path),
      );
      return Object.keys(pruned).length === Object.keys(current).length ? current : pruned;
    });
  }, [state.videoFiles, state.audioFiles]);

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
      audioFiles: current.audioFiles.map((file) => file.path),
      matchPattern:
        current.mode === "series" && config.matchPattern.trim()
          ? config.matchPattern
          : null,
      // Kept as the default for any pair that does not name its own stream.
      // Track 0 is the file's first audio stream, which is what the engine
      // would have used anyway.
      videoTrack: 0,
      audioTrack: 0,
      // Sent once the user has changed the matching or chosen a stream for any
      // file. Otherwise the engine should do its own matching, which stays
      // correct as the selection or pattern changes.
      //
      // Per-file stream choices travel on the pairs themselves: the engine
      // reads primaryTrack/secondaryTrack per pair, so a selection mixing a
      // five-track REMUX with a single-track WEB-DL is expressible.
      pairs:
        Object.keys(overridesRef.current).length > 0 ||
        Object.keys(trackChoicesRef.current).length > 0
          ? (effectivePairingRef.current?.pairs.map((pair) => ({
              ...pair,
              primaryTrack: trackChoicesRef.current[pair.primaryPath] ?? 0,
              secondaryTrack: trackChoicesRef.current[pair.secondaryPath] ?? 0,
            })) ?? null)
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
    setAnnouncement(
      announceRunStarted(
        stateRef.current.pairing?.pairs.length ?? current.videoFiles.length,
        current.mode,
      ),
    );

    try {
      const run = await api.startSync(buildRequest());
      dispatch({
        type: "runFinished",
        results: run.results,
        summary: run.summary,
        cancelled: run.cancelled,
      });
      setAnnouncement(announceRunFinished(run.results, run.summary, run.cancelled));

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
      setAnnouncement(announceRunFailed(message));
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
    setCancellingApply(false);
    setApplyState({ written: [], current: null, done: 0, total: items.length });
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
      setAnnouncement(
        announceApplyFinished(outcome.written.length, outcome.failed.length),
      );
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
      setApplyState(null);
      setCancellingApply(false);
    }
  }, [selectedKeys]);

  const handleCancelApply = useCallback(async () => {
    setCancellingApply(true);
    try {
      await api.cancelSync();
    } catch {
      toast.error("Could not stop writing.");
      setCancellingApply(false);
    }
  }, []);

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

  /** Render a short aligned excerpt and hand it to the user's own player.
   *
   *  A confidence score is an argument; hearing the dub land on the picture is
   *  the only thing that settles a borderline result. */
  const handlePreview = useCallback(async (result: SyncResult) => {
    if (!result.primaryPath || !result.secondaryPath || result.delayMs === null) return;
    const key = resultKey(result);
    setPreviewingKey(key);
    try {
      const path = await api.renderPreview({
        videoPath: result.primaryPath,
        audioPath: result.secondaryPath,
        delayMs: result.delayMs,
        driftMsPerS: result.hasSignificantDrift ? result.driftMsPerS : null,
        audioTrack: result.secondaryTrack ?? 0,
        durationSeconds: 12,
      });
      if (!path) {
        toast.error("Could not render the preview.");
        return;
      }
      await api.openPath(path);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not open the preview");
    } finally {
      setPreviewingKey(null);
    }
  }, []);

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

  /** What will actually be analysed: the engine's matching, plus any pair the
   *  user corrected by hand. */
  const effectivePairing = useMemo(
    () =>
      state.pairing
        ? applyOverrides(
            state.pairing,
            pairOverrides,
            state.audioFiles.map((file) => ({ path: file.path, name: file.name })),
          )
        : null,
    [state.pairing, pairOverrides, state.audioFiles],
  );
  const manualPairCount = useMemo(() => countManualPairs(pairOverrides), [pairOverrides]);
  effectivePairingRef.current = effectivePairing;
  // Count what will actually run, not what the matcher first proposed: an
  // excluded video must disappear from the button too.
  const pairCount = effectivePairing?.pairs.length ?? 0;
  const busy = state.status === "processing";
  const hasResults = state.results.length > 0;

  /** Choose which audio stream of one file to compare. Index 0 is the file's
   *  first stream, which is the default, so it is stored as an absence. */
  const handleTrackChange = useCallback((path: string, index: number) => {
    setTrackChoices((current) => {
      const next = { ...current };
      if (index > 0) next[path] = index;
      else delete next[path];
      return next;
    });
  }, []);

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

  /** What the run button says, so the sidebar does not have to know the modes. */
  const runLabel = pairCount > 0 ? `Analyse ${pairCount} pair${pairCount === 1 ? "" : "s"}` : "Analyse";

  return (
    <div className="flex h-screen flex-col bg-background">
      <AppHeader
        mode={state.mode}
        onModeChange={setMode}
        disabled={busy}
        showConsole={showConsole}
        showHistory={showHistory}
        onToggleConsole={() => setShowConsole((open) => !open)}
        onToggleHistory={() => setShowHistory((open) => !open)}
        onOpenSettings={() => setShowSettings(true)}
      />

      <div className="relative flex min-h-0 flex-1">
        <Sidebar
          mode={state.mode}
          videoFiles={state.videoFiles}
          audioFiles={state.audioFiles}
          videoFolder={state.videoFolder}
          audioFolder={state.audioFolder}
          recentFolders={recentFolders}
          probes={probes}
          dragTarget={dragTarget}
          busy={busy}
          listings={listings}
          trackChoices={trackChoices}
          onTrackChange={handleTrackChange}
          onBrowse={(kind) => void handleBrowse(kind)}
          onRemove={(kind, id) => dispatch({ type: "removeFiles", kind, ids: [id] })}
          onClear={(kind) => dispatch({ type: "clearFiles", kind })}
          onDragEnter={setDragTarget}
          runLabel={runLabel}
          canRun={selection.ok && desktop}
          runBlockedReason={
            !desktop ? "Analysis needs the desktop app." : (selection.reason ?? undefined)
          }
          onRun={() => void handleStart()}
          onStop={() => void handleCancel()}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          {busy && (
            <ProgressPanel
              processed={state.progress.processed}
              total={state.progress.total}
              currentFile={state.currentFile}
              fileProgress={state.fileProgress}
              remainingMs={remainingMs}
            />
          )}

          {hasResults ? (
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
              onPreview={(result) => void handlePreview(result)}
              previewingKey={previewingKey}
              applying={applying}
              outputSuffix={settings.outputSuffix}
            />
          ) : (
            /* Before a run, the main pane shows what will be compared. This is
               the one thing worth checking before committing to a long job,
               and it used to be squeezed between the inputs and the button. */
            <div className="min-h-0 flex-1 overflow-y-auto px-[18px] py-[18px]">
              {state.error ? (
                <div className="mx-auto mt-[12vh] max-w-[420px] text-center">
                  <p className="text-[13px] font-semibold text-destructive">
                    The analysis could not finish.
                  </p>
                  <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted-foreground">
                    {state.error}
                  </p>
                  <button
                    type="button"
                    onClick={() => setShowConsole(true)}
                    className="mt-3 text-[12.5px] font-medium text-primary hover:opacity-80"
                  >
                    Open the console
                  </button>
                </div>
              ) : !desktop ? (
                <EmptyState
                  title="Running in a browser"
                  body="File selection and analysis need the desktop app."
                />
              ) : state.videoFiles.length === 0 && state.audioFiles.length === 0 ? (
                <EmptyState
                  // The title names the question the mode answers. "Add files
                  // to begin" was the same in all three, which left the point
                  // of Find match discoverable only by trying it.
                  title={
                    state.mode === "compare"
                      ? "Find which release a dub matches"
                      : "Add files to begin"
                  }
                  body={
                    state.mode === "movie"
                      ? "Pick the videos whose timing is already correct, and the one audio track to align against them."
                      : state.mode === "compare"
                        ? `Every video is tested against every audio track, and the results rank which pairing actually lines up. Use this when you do not know which release a dub was timed for. Up to ${MAX_COMPARE_INPUTS} files per side.`
                        : "Pick the episode folder and the folder of dubs. They are matched by season and episode number."
                  }
                />
              ) : state.pairing || pairingLoading ? (
                <PairingPreview
                  pairing={effectivePairing}
                  loading={pairingLoading}
                  audioFiles={state.audioFiles}
                  videoFiles={state.videoFiles}
                  manualCount={manualPairCount}
                  disabled={busy}
                  onRepair={(videoPath, audioPath) =>
                    setPairOverrides((current) => ({ ...current, [videoPath]: audioPath }))
                  }
                  onResetRepairs={() => setPairOverrides({})}
                />
              ) : (
                <EmptyState
                  title={selection.ok ? "Ready" : "Not ready yet"}
                  body={
                    selection.ok
                      ? "Run the analysis to measure the delay for each pair."
                      : (selection.reason ?? "Add files on both sides.")
                  }
                />
              )}
            </div>
          )}
        </main>

        {showHistory && (
          <HistoryPanel
            entries={history}
            onClose={() => setShowHistory(false)}
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
      </div>

      {showConsole && (
        <ConsolePanel
          logs={state.logs}
          onClear={() => dispatch({ type: "clearLogs" })}
          onClose={() => setShowConsole(false)}
          onCopy={(text) => void handleCopy(text)}
        />
      )}

      <StatusBar
        mode={state.mode}
        pairCount={pairCount}
        resultCount={state.results.length}
        summary={state.summary}
        busy={busy}
        version={APP_VERSION}
      />

      <SettingsDialog
        open={showSettings}
        settings={settings}
        mode={state.mode}
        version={APP_VERSION}
        onChange={setSettings}
        onClose={() => setShowSettings(false)}
        onCheckForUpdate={() => void handleCheckForUpdate()}
        checkingUpdate={checkingUpdate}
      />

      <ApplyProgressDialog
        state={applying ? applyState : null}
        onCancel={() => void handleCancelApply()}
        cancelling={cancellingApply}
      />

      <UpdateDialog update={update} onDismiss={() => setUpdate(null)} />

      <LiveAnnouncer announcement={announcement} />
    </div>
  );
}

/** A short, centred explanation of what to do next.
 *
 *  Deliberately the only centred thing in the app: it marks a pane that has
 *  nothing in it, so there is no content for it to compete with. */
function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="mx-auto mt-[12vh] max-w-[420px] text-center">
      <p className="text-[13px] font-semibold">{title}</p>
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted-foreground">{body}</p>
    </div>
  );
}
