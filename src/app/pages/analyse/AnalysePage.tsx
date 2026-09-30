/** Movies, Series and Find match: one container, three pages.
 *
 *  Each is mounted once and kept while hidden, so every page has its own
 *  files, pairing, results and selection. The run holds the shared engine;
 *  its streamed events come back to the page that started it through the
 *  shell, wherever the user has gone since. */

import {
  AddRegular,
  ArrowExportRegular,
  ArrowSyncRegular,
  DeleteRegular,
  FolderOpenRegular,
  FolderRegular,
  MusicNote2Regular,
  PlayRegular,
  StopRegular,
} from "@fluentui/react-icons";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { toast } from "sonner";

import { announceApplyFinished, announceFilesAdded, announceProgress, announceRunFailed, announceRunFinished, announceRunStarted } from "@/lib/announce";
import * as api from "@/lib/api";
import { applyOverrides, countManualPairs, pruneOverrides, type PairOverrides } from "@/lib/pairing";
import { createHistoryEntry } from "@/lib/storage";
import { estimateRemainingMs, initialSyncState, syncReducer, validateSelection } from "@/lib/syncReducer";
import {
  MAX_COMPARE_INPUTS,
  ffmpegCommandFor,
  resultKey,
  type AnalyzeRequest,
  type FileItem,
  type HistoryEntry,
  type MediaProbe,
  type PairingReport,
  type SyncMode,
  type SyncResult,
  type TrackListing,
} from "@/lib/types";
import { useShell, usePageCommands, useEngineListeners } from "@/app/shell";
import { PageDock } from "@/app/dock";
import { PageView, type LcdProps } from "@/ui/frame";
import { Btn, Cmd, baseName } from "@/ui/kit";

import { AnalyseWorkspace, FilterMenu, commonTitle, episodeCode, releaseName, resultsOutcome, seriesTitle, tookText, withDetails, type Filter } from "./views";

type Mode = Exclude<SyncMode, "dubsync" | "subsync">;

const VIDEO_EXT = /\.(mkv|mp4|m4v|mov|avi|ts|m2ts|mts|webm|wmv|flv|mpg|mpeg|vob)$/i;

export interface ApplyState {
  /** Output paths written so far. */
  written: string[];
  /** The video being written now. */
  current: string | null;
  done: number;
  total: number;
  /** Result keys being written, in order. */
  keys: string[];
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export function AnalysePage({ mode, hidden }: { mode: Mode; hidden: boolean }) {
  const shell = useShell();
  const { desktop, settings } = shell;

  const [state, dispatch] = useReducer(syncReducer, { ...initialSyncState, mode });
  const [probes, setProbes] = useState<Record<string, MediaProbe>>({});
  const [listings, setListings] = useState<Record<string, TrackListing>>({});
  const [trackChoices, setTrackChoices] = useState<Record<string, number>>({});
  const [pairOverrides, setPairOverrides] = useState<PairOverrides>({});
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());
  const [pairingLoading, setPairingLoading] = useState(false);
  const [applyState, setApplyState] = useState<ApplyState | null>(null);
  const [written, setWritten] = useState<{ keys: Set<string>; paths: string[] } | null>(null);
  const [previewingKey, setPreviewingKey] = useState<string | null>(null);
  const [remainingMs, setRemainingMs] = useState<number | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [filterOpen, setFilterOpen] = useState(false);
  /** How long the last run took, for the status display; null for a run
   *  reopened from History. */
  const [runTook, setRunTook] = useState<number | null>(null);

  const stateRef = useRef(state);
  stateRef.current = state;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const trackChoicesRef = useRef(trackChoices);
  trackChoicesRef.current = trackChoices;
  const overridesRef = useRef(pairOverrides);
  overridesRef.current = pairOverrides;
  const effectivePairingRef = useRef<PairingReport | null>(null);
  const selectedRef = useRef(selectedKeys);
  selectedRef.current = selectedKeys;

  const running = state.status === "processing";
  const writing = applyState !== null;
  const busy = running || writing;

  // ------------------------------------------------------------ engine events

  useEngineListeners(mode, {
    onProgress: (event) => {
      dispatch({ type: "progress", processed: event.processed, total: event.total, current: event.current });
      const milestone = announceProgress(event.processed, event.total);
      if (milestone) shell.announce(milestone);
    },
    onFileStart: (file) => dispatch({ type: "fileStart", file }),
    onFileProgress: (event) => dispatch({ type: "fileProgress", file: event.file, percent: event.percent }),
    onResult: (result) => dispatch({ type: "result", result }),
    onPairs: (pairing) => dispatch({ type: "setPairing", pairing }),
    onApplyProgress: (event) => {
      if (event.file) shell.log(`Writing ${event.file}`);
      // applyStart names the file about to be written; applyProgress confirms
      // one finished and carries the running totals.
      setApplyState((prev) => {
        if (!prev) return prev;
        const finished = typeof event.done === "number";
        return {
          ...prev,
          written: finished && event.output ? [...prev.written, event.output] : prev.written,
          // The status display names the copy being written, not its source.
          current: finished ? null : (event.output ? baseName(event.output) : (event.file ?? prev.current)),
          done: event.done ?? prev.done,
          total: event.total ?? prev.total,
        };
      });
    },
  });

  // Live time left while a run is in flight.
  useEffect(() => {
    if (!running) {
      setRemainingMs(null);
      return;
    }
    const tick = () => setRemainingMs(estimateRemainingMs(stateRef.current));
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  // ------------------------------------------------------------ files

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

  // Read the audio streams of every selected file, so each can pick its own.
  useEffect(() => {
    const paths = [...state.videoFiles, ...state.audioFiles].map((file) => file.path);
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
        // Drop choices the newly probed files can no longer satisfy.
        setTrackChoices((current) => {
          const next: Record<string, number> = {};
          Object.entries(current).forEach(([path, index]) => {
            if (index > 0 && index < (byPath[path]?.tracks.length ?? 0)) next[path] = index;
          });
          return next;
        });
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [state.videoFiles, state.audioFiles]);

  // Drop corrections whose files are no longer selected.
  useEffect(() => {
    setPairOverrides((current) => {
      if (Object.keys(current).length === 0) return current;
      const pruned = pruneOverrides(
        current,
        state.videoFiles.map((f) => f.path),
        state.audioFiles.map((f) => f.path),
      );
      return Object.keys(pruned).length === Object.keys(current).length ? current : pruned;
    });
  }, [state.videoFiles, state.audioFiles]);

  const addFiles = useCallback(
    (kind: "video" | "audio", files: FileItem[], folder: string | null, replace = false) => {
      if (files.length === 0) return;
      setWritten(null);
      if (kind === "audio" && mode === "movie") {
        // Movies compares many videos against exactly one audio track.
        dispatch({ type: "replaceFiles", kind, files: [files[0]], folder });
        if (files.length > 1) toast.info("Movies uses one dub. Kept the first.");
      } else if (replace) {
        dispatch({ type: "replaceFiles", kind, files, folder, explicit: !folder });
      } else {
        dispatch({ type: "addFiles", kind, files, folder, explicit: !folder });
      }
      void probeFiles(files.map((file) => file.path));
    },
    [mode, probeFiles],
  );

  const pick = useCallback(
    async (kind: "video" | "audio", folderPick = false) => {
      if (!desktop) {
        toast.error("File selection needs the desktop app.");
        return;
      }
      try {
        // Series pairs folders; Movies takes one dub; the rest take files.
        const byFolder = folderPick || mode === "series";
        const response = byFolder
          ? kind === "audio"
            ? await api.pickAudioFolder()
            : await api.pickVideoFolder()
          : kind === "audio" && mode === "movie"
            ? await api.pickAudioFile()
            : await api.pickMediaFiles(kind);
        if (response.files.length === 0) return;
        addFiles(kind, response.files, response.folder, byFolder);
        // The list shows what arrived; only a screen reader needs telling.
        shell.announce(announceFilesAdded(response.files.length, kind));
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Could not open the picker");
      }
    },
    [addFiles, desktop, mode, shell],
  );

  /** Dropped paths: folders add what they hold; videos go on the video side,
   *  everything else is a dub. */
  const drop = useCallback(
    async (paths: string[]) => {
      try {
        const files = await api.resolveDroppedPaths(paths, "video", "media");
        if (files.length === 0) {
          toast.error("No supported media files in that drop.");
          return;
        }
        const videos = files.filter((f) => VIDEO_EXT.test(f.name));
        const audio = files.filter((f) => !VIDEO_EXT.test(f.name));
        const folderOf = (list: FileItem[]) => list[0]?.path.replace(/[\\/][^\\/]+$/, "") ?? null;
        if (videos.length) addFiles("video", videos.map((f) => ({ ...f, type: "video" })), folderOf(videos));
        if (audio.length) addFiles("audio", audio.map((f) => ({ ...f, type: "audio" })), folderOf(audio));
        if (videos.length) shell.announce(announceFilesAdded(videos.length, "video"));
        if (audio.length) shell.announce(announceFilesAdded(audio.length, "audio"));
      } catch {
        toast.error("Could not read the dropped files.");
      }
    },
    [addFiles, shell],
  );

  // ------------------------------------------------------------ pairing

  const buildRequest = useCallback((): AnalyzeRequest => {
    const current = stateRef.current;
    const config = settingsRef.current;
    return {
      mode,
      videoFolder: current.videoFolder,
      audioFolder: mode === "series" ? current.audioFolder : null,
      audioFile: mode === "movie" ? (current.audioFiles[0]?.path ?? null) : null,
      videoFiles: current.videoFiles.map((file) => file.path),
      audioFiles: current.audioFiles.map((file) => file.path),
      matchPattern: mode === "series" && config.matchPattern.trim() ? config.matchPattern : null,
      videoTrack: 0,
      audioTrack: 0,
      // Explicit pairs once the user has changed the matching or chosen a
      // stream for any file; otherwise the engine does its own matching.
      pairs:
        Object.keys(overridesRef.current).length > 0 || Object.keys(trackChoicesRef.current).length > 0
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
      findCuts: config.cutCheck,
      timeline: config.timelineCheck,
      findSpeed: config.rateCheck,
    };
  }, [mode]);

  // Refresh the pairing when the selection settles, so what will be compared
  // is visible before committing to a long run.
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
        .then((pairing) => active && dispatch({ type: "setPairing", pairing }))
        .catch(() => undefined)
        .finally(() => active && setPairingLoading(false));
    }, 200);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [desktop, state.videoFiles, state.audioFiles, state.status, settings.matchPattern, buildRequest]);

  const effectivePairing = useMemo(
    () =>
      state.pairing
        ? applyOverrides(state.pairing, pairOverrides, state.audioFiles.map((f) => ({ path: f.path, name: f.name })))
        : null,
    [state.pairing, pairOverrides, state.audioFiles],
  );
  effectivePairingRef.current = effectivePairing;
  const manualCount = useMemo(() => countManualPairs(pairOverrides), [pairOverrides]);
  const selection = useMemo(() => validateSelection(state), [state]);
  const pairCount = effectivePairing?.pairs.length ?? 0;

  // ------------------------------------------------------------ the run

  const start = useCallback(async () => {
    const current = stateRef.current;
    if (current.status === "processing" || applyState) return;
    const check = validateSelection(current);
    if (!check.ok) {
      toast.error(check.reason ?? "Nothing to analyse.");
      return;
    }
    if (!desktop) {
      toast.error("Analysis needs the desktop app.");
      return;
    }
    if (!shell.claimEngine(mode)) {
      toast.error("Another page is running. Wait for it to finish, or stop it there.");
      return;
    }
    setSelectedKeys(new Set());
    setWritten(null);
    setFocus(null);
    dispatch({ type: "runStarted", total: current.videoFiles.length });
    setRunTook(null);
    const began = Date.now();
    shell.announce(announceRunStarted(effectivePairingRef.current?.pairs.length ?? current.videoFiles.length, mode));
    try {
      const run = await api.startSync(buildRequest());
      dispatch({ type: "runFinished", results: run.results, summary: run.summary, cancelled: run.cancelled });
      setRunTook(Date.now() - began);
      shell.announce(announceRunFinished(run.results, run.summary, run.cancelled));
      if (run.cancelled) toast.info("Analysis stopped.");
      if (run.results.length > 0) {
        shell.addHistory({
          ...createHistoryEntry(mode, run.results, run.summary),
          name: runName(mode, stateRef.current),
          outcome: mode === "compare" ? matchOutcome(run.results, current.audioFiles.length) : resultsOutcome(run.results),
        });
        // Pre-select what is safe to fix: confident, not a different cut. In
        // Find match only each dub's best release.
        const confident = run.results.filter(
          (r) => !r.error && r.delayMs !== null && !r.isLikelyCut && (r.confidence ?? 0) >= 0.75,
        );
        const picks =
          mode === "compare"
            ? [...new Map(confident.sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)).map((r) => [r.secondaryPath ?? r.audioFile, r])).values()]
            : confident;
        setSelectedKeys(new Set(picks.map(resultKey)));
        setFocus(resultKey(run.results[0]));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      dispatch({ type: "runFailed", message });
      shell.announce(announceRunFailed(message));
      toast.error("Analysis failed", { description: message });
      shell.openOutput();
    } finally {
      shell.releaseEngine(mode);
    }
  }, [applyState, buildRequest, desktop, mode, shell]);

  const stop = useCallback(async () => {
    try {
      await api.cancelSync();
      toast.info("Stopping…");
    } catch {
      toast.error("Could not stop the run.");
    }
  }, []);

  const apply = useCallback(async () => {
    const chosen = stateRef.current.results
      .filter((r) => selectedRef.current.has(resultKey(r)))
      .filter((r) => r.delayMs !== null && r.primaryPath && r.secondaryPath);
    if (chosen.length === 0) {
      toast.error("Select at least one measured result to fix.");
      return;
    }
    if (!shell.claimEngine(mode)) {
      toast.error("Another page is running. Wait for it to finish, or stop it there.");
      return;
    }
    const keys = chosen.map(resultKey);
    setApplyState({ written: [], current: null, done: 0, total: chosen.length, keys });
    setWritten(null);
    try {
      const outcome = await api.applyCorrections(
        chosen.map((r) => ({
          videoPath: r.primaryPath as string,
          audioPath: r.secondaryPath as string,
          delayMs: r.delayMs as number,
          delayAtStartMs: r.delayAtStartMs ?? r.delayMs,
          driftMsPerS: r.hasSignificantDrift ? r.driftMsPerS : null,
        })),
        { suffix: settingsRef.current.outputSuffix, outputDir: settingsRef.current.outputDir },
      );
      shell.announce(announceApplyFinished(outcome.written.length, outcome.failed.length));
      outcome.failed.forEach((failure) => shell.log(`${failure.video}: ${failure.error}`));
      const failedVideos = new Set(outcome.failed.map((f) => f.video));
      setWritten({
        keys: new Set(chosen.filter((r) => !failedVideos.has(r.primaryPath as string) && !failedVideos.has(r.videoFile)).map(resultKey)),
        paths: outcome.written,
      });
      if (outcome.written.length === 0 && outcome.failed.length > 0) {
        toast.error("No files could be written.");
        shell.openOutput();
      } else if (outcome.failed.length > 0) {
        toast.error(`${plural(outcome.failed.length, "file")} could not be written`, { description: "See Output." });
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not write files");
    } finally {
      setApplyState(null);
      shell.releaseEngine(mode);
    }
  }, [mode, shell]);

  const exportResults = useCallback(async (format: "csv" | "json") => {
    const results = stateRef.current.results;
    if (results.length === 0) return;
    try {
      const path = format === "csv" ? await api.exportCsv(results) : await api.exportJson(results);
      toast.success("Exported", { action: { label: "Show", onClick: () => void api.revealPath(path).catch(() => undefined) } });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!message.toLowerCase().includes("cancel")) toast.error("Export failed");
    }
  }, []);

  const preview = useCallback(async (result: SyncResult) => {
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

  // The list shows its first row as selected until another is picked; Remove
  // acts on the row that shows.
  const target = focus ?? (mode === "compare" ? null : state.results[0] ? resultKey(state.results[0]) : mode === "movie" ? (state.videoFiles[0]?.path ?? null) : null);

  const removeFocused = useCallback(() => {
    if (busy || !target) return;
    // A result row: its video goes, and the video's results with it.
    const result = stateRef.current.results.find((r) => resultKey(r) === target);
    const focus = result ? (result.primaryPath ?? result.videoFile) : target;
    if (result) {
      const { results, summary } = stateRef.current;
      const gone = results.filter((r) => (r.primaryPath ?? r.videoFile) === focus).map(resultKey);
      dispatch({ type: "loadResults", results: results.filter((r) => !gone.includes(resultKey(r))), summary, mode });
      setSelectedKeys((prev) => new Set([...prev].filter((key) => !gone.includes(key))));
    }
    const video = stateRef.current.videoFiles.find((f) => f.path === focus || f.id === focus);
    if (result || video) {
      if (video) dispatch({ type: "removeFiles", kind: "video", ids: [video.id] });
      setFocus(null);
      return;
    }
    const audio = stateRef.current.audioFiles.find((f) => f.path === focus || f.id === focus);
    if (audio) {
      dispatch({ type: "removeFiles", kind: "audio", ids: [audio.id] });
      setFocus(null);
    }
  }, [busy, target, mode]);

  const clear = useCallback(() => {
    if (busy) return;
    dispatch({ type: "clearFiles", kind: "video" });
    dispatch({ type: "clearFiles", kind: "audio" });
    setPairOverrides({});
    setSelectedKeys(new Set());
    setWritten(null);
    setFocus(null);
  }, [busy]);

  const loadRun = useCallback((entry: HistoryEntry) => {
    dispatch({ type: "loadResults", results: entry.results, summary: entry.summary, mode });
    setSelectedKeys(new Set());
    setWritten(null);
    setFocus(entry.results[0] ? resultKey(entry.results[0]) : null);
    setRunTook(null);
  }, [mode]);

  usePageCommands(mode, {
    addVideos: () => void pick("video"),
    addAudio: () => void pick("audio"),
    addFolder: () => void pick("video", true),
    exportResults: () => void exportResults("csv"),
    removeSelected: removeFocused,
    clear,
    start: () => void start(),
    stop: () => void stop(),
    drop: (paths) => void drop(paths),
    loadRun,
    exportJson: () => void exportResults("json"),
    addVideosLabel: mode === "series" ? "Choose the episodes folder…" : "Add videos…",
    addAudioLabel: mode === "movie" ? "Add a dub…" : mode === "series" ? "Choose the dubs folder…" : "Add dubs…",
    disabled: {
      addVideos: busy,
      addAudio: busy,
      addFolder: busy,
      exportResults: state.results.length === 0,
      exportJson: state.results.length === 0,
      removeSelected: busy || !target,
      clear: busy || (state.videoFiles.length === 0 && state.audioFiles.length === 0),
      start: busy,
    },
  });

  // ------------------------------------------------------------ view

  const results = state.results;
  const hasFiles = state.videoFiles.length > 0 || state.audioFiles.length > 0;
  const hasResults = results.length > 0;
  const picked = results.filter((r) => selectedKeys.has(resultKey(r))).length;
  const lcd = statusDisplay({
    mode,
    state,
    pairCount,
    selectionOk: selection.ok,
    selectionReason: selection.reason ?? null,
    pairingLoading,
    effectivePairing,
    remainingMs,
    applyState,
    written,
    picked,
    suffix: settings.outputSuffix,
    outputDir: settings.outputDir,
    runTook,
  });

  const tools = (
    <>
      <Cmd icon={mode === "series" ? <FolderRegular /> : <AddRegular />} disabled={busy} onClick={() => void pick("video")}>
        {mode === "series" ? "Episodes" : "Add videos"}
      </Cmd>
      <Cmd icon={<MusicNote2Regular />} disabled={busy} onClick={() => void pick("audio")}>
        {mode === "series" ? "Dubs" : mode === "compare" ? "Add dubs" : state.audioFiles.length ? "Change dub" : "Add dub"}
      </Cmd>
      {/* Series leaves an episode out with "Skip this episode" in its pairing; Find match
          exports from the File menu. Edit › Remove, Del and Ctrl+E work on every page. */}
      {hasFiles && mode === "movie" && <Cmd icon={<DeleteRegular />} title="Remove" disabled={busy || !target} onClick={removeFocused} />}
      {hasResults && !running && mode !== "compare" && <Cmd icon={<ArrowExportRegular />} title="Export results (CSV)" onClick={() => void exportResults("csv")} />}
      {hasResults && !running && mode !== "compare" && (
        <FilterMenu results={results} value={filter} open={filterOpen} onOpen={setFilterOpen} onChange={setFilter} />
      )}
    </>
  );

  const primary = running || writing ? (
    <Btn icon={<StopRegular />} kbd="Esc" onClick={() => void stop()}>Stop</Btn>
  ) : written && written.paths.length > 0 ? (
    <>
      <Btn icon={<ArrowSyncRegular />} title="Analyse again" aria-label="Analyse again" onClick={() => void start()} />
      <Btn accent icon={<FolderOpenRegular />} onClick={() => void api.revealPath(written.paths[0]).catch(() => undefined)}>Open folder</Btn>
    </>
  ) : hasResults ? (
    <>
      <Btn icon={<ArrowSyncRegular />} title="Analyse again" aria-label="Analyse again" onClick={() => void start()} />
      <Btn accent disabled={picked === 0} onClick={() => void apply()}>
        {mode === "compare" ? `Fix ${plural(picked, "match", "matches")}` : `Fix ${picked}`}
      </Btn>
    </>
  ) : (
    <Btn accent icon={<PlayRegular />} kbd="Enter" disabled={!selection.ok || !desktop || pairingLoading} onClick={() => void start()}>
      {mode === "compare" ? "Find matches" : "Analyse"}
    </Btn>
  );

  const dock = PageDock({ common: shell.dock, height: 280 });

  return (
    <PageView
      hidden={hidden}
      tools={tools}
      lcd={lcd}
      primary={primary}
      cols={withDetails(mode, state) ? "minmax(0,1fr) 300px" : "minmax(0,1fr)"}
      dock={dock}
    >
      <AnalyseWorkspace
        mode={mode}
        state={state}
        pairing={effectivePairing}
        pairingLoading={pairingLoading}
        manualCount={manualCount}
        probes={probes}
        listings={listings}
        trackChoices={trackChoices}
        selectedKeys={selectedKeys}
        focus={focus}
        filter={filter}
        applyState={applyState}
        written={written}
        previewingKey={previewingKey}
        busy={busy}
        windowCount={settings.windowCount}
        onFilter={setFilter}
        onFocus={setFocus}
        onToggle={(key) =>
          setSelectedKeys((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
          })
        }
        onTrackChange={(path, index) =>
          setTrackChoices((current) => {
            const next = { ...current };
            if (index > 0) next[path] = index;
            else delete next[path];
            return next;
          })
        }
        onRepair={(videoPath, audioPath) => setPairOverrides((current) => ({ ...current, [videoPath]: audioPath }))}
        onResetRepairs={() => setPairOverrides({})}
        onAddVideos={() => void pick("video")}
        onAddAudio={() => void pick("audio")}
        onPreview={(r) => void preview(r)}
        onCopyCommand={(r) => {
          const command = ffmpegCommandFor(r);
          if (command) shell.copy(command);
        }}
        onOpenDubSync={() => shell.show("dubsync")}
        onOpenOutput={shell.openOutput}
      />
    </PageView>
  );
}

/* ------------------------------------------------------------ status display */

function statusDisplay({
  mode,
  state,
  pairCount,
  selectionOk,
  selectionReason,
  pairingLoading,
  effectivePairing,
  remainingMs,
  applyState,
  written,
  picked,
  suffix,
  outputDir,
  runTook,
}: {
  mode: Mode;
  state: ReturnType<typeof syncReducer>;
  pairCount: number;
  selectionOk: boolean;
  selectionReason: string | null;
  pairingLoading: boolean;
  effectivePairing: PairingReport | null;
  remainingMs: number | null;
  applyState: ApplyState | null;
  written: { keys: Set<string>; paths: string[] } | null;
  picked: number;
  suffix: string;
  outputDir: string | null;
  runTook: number | null;
}): LcdProps {
  const videos = state.videoFiles.length;
  const dubs = state.audioFiles.length;

  if (applyState) {
    const name = applyState.current ? baseName(applyState.current) : "Preparing…";
    const pct = applyState.total ? (applyState.done / applyState.total) * 100 : 0;
    return { icon: "run", l1: `Writing ${Math.min(applyState.done + 1, applyState.total)} of ${applyState.total}`, l2: name, pct };
  }
  if (written && written.paths.length > 0) {
    return {
      icon: "ok",
      l1: `${plural(written.paths.length, "corrected copy", "corrected copies")} written`,
      l2: `${outputDir ? `In ${baseName(outputDir.replace(/[\\/]+$/, ""))}` : "Beside the originals"}, ending ${suffix}`,
    };
  }
  if (state.status === "processing") {
    const { processed, total } = state.progress;
    const measuring = Object.values(state.inFlight);
    const partial = measuring.length ? measuring.reduce((sum, p) => sum + p, 0) / 100 : state.fileProgress / 100;
    const pct = total ? Math.min(100, ((processed + partial) / total) * 100) : null;
    const left = remainingMs && remainingMs > 0 ? `${formatLeft(remainingMs)} left` : undefined;
    return {
      icon: "run",
      l1: `${mode === "compare" ? "Testing" : "Analysing"} ${Math.min(processed + 1, Math.max(total, 1))} of ${total || pairCount}`,
      l2:
        mode === "compare"
          ? undefined
          : mode === "series"
          ? seriesTitle(state)
          : measuring.length > 1
            ? `${measuring.length} at a time`
            : state.currentFile
              ? baseName(state.currentFile)
              : `${plural(state.progress.total, "pair")}`,
      pct,
      time: left,
    };
  }
  if (state.error) return { icon: "bad", l1: "The analysis could not finish", l2: state.error };
  if (state.results.length > 0) {
    if (mode === "compare") {
      const matches = state.results.filter((r) => !r.error && (r.confidence ?? 0) >= 0.75);
      const dubsWithMatch = new Set(matches.map((r) => r.secondaryPath ?? r.audioFile)).size;
      // A run reopened from History has results but no files.
      const dubCount = dubs || new Set(state.results.map((r) => r.secondaryPath ?? r.audioFile)).size;
      // Releases no dub was timed for: "The Black and White release matches none".
      const title = commonTitle(state.videoFiles.map((v) => v.name));
      const idle = state.videoFiles.filter((v) => !matches.some((r) => (r.primaryPath ?? r.videoFile) === v.path || r.videoFile === v.name));
      const none =
        idle.length === 1
          ? `The ${releaseName(idle[0].name, title)} release matches none`
          : idle.length > 1
            ? `${idle.length} releases match none`
            : null;
      const tests = `${plural(state.results.length, "test")}${runTook ? ` in ${tookText(runTook)}` : ""}${state.status === "cancelled" ? " · stopped" : ""}`;
      return {
        icon: dubsWithMatch === dubCount ? "ok" : "warn",
        l1: dubsWithMatch === dubCount ? "Each dub has a match" : `${dubsWithMatch} of ${plural(dubCount, "dub")} matched`,
        l2: none ? `${none} · ${tests}` : tests,
      };
    }
    const outcome = resultsOutcome(state.results);
    return {
      icon: outcome.tone === "ok" ? "ok" : "warn",
      l1: outcome.text,
      l2: runTook
        ? `${state.results.length} analysed in ${tookText(runTook)}${state.status === "cancelled" ? " · stopped" : ""}`
        : `${plural(state.results.length, mode === "series" ? "episode" : "file")} analysed${state.status === "cancelled" ? " · stopped" : ""}`,
      time: picked ? `${picked} selected` : undefined,
    };
  }
  if (videos === 0 && dubs === 0) {
    return {
      l1:
        mode === "movie"
          ? "Drop videos and a dub to begin"
          : mode === "series"
            ? "Drop an episodes folder and a dubs folder"
            : `Drop up to ${MAX_COMPARE_INPUTS} videos and ${MAX_COMPARE_INPUTS} dubs`,
    };
  }
  if (pairingLoading) return { icon: "run", l1: "Working out the pairs…", pct: null };
  if (mode === "series" && effectivePairing) {
    // Counted from the pairs as edited: a dub chosen by hand pairs the episode.
    const pairedPaths = new Set(effectivePairing.pairs.map((p) => p.primaryPath));
    const unpaired = state.videoFiles.filter((f) => !pairedPaths.has(f.path)).map((f) => f.name);
    const unmatched = unpaired.length;
    const orderPairs = effectivePairing.pairs.filter((p) => p.method === "list order");
    const byOrder = orderPairs.length;
    // A few episodes by name ("E07 paired by list order"), more by count.
    const codes = (names: string[]) => {
      const list = names.map((name) => episodeCode(null, baseName(name)).replace(/^S\d+(?=E)/, ""));
      return list.length <= 2 && list.every((c) => c !== "—") ? list.join(", ") : null;
    };
    const orderCodes = codes(orderPairs.map((p) => p.primaryName || p.primaryPath));
    const missingCodes = codes(unpaired);
    const notes = [
      byOrder ? `${orderCodes ?? plural(byOrder, "episode")} paired by list order` : null,
      unmatched ? (missingCodes ? `${missingCodes} ${unmatched === 1 ? "has" : "have"} no dub` : `${plural(unmatched, "episode")} with no dub`) : null,
      effectivePairing.warning,
    ].filter(Boolean);
    return {
      icon: unmatched || byOrder || effectivePairing.warning ? "warn" : "idle",
      l1: `${pairCount} of ${plural(videos, "episode")} paired`,
      l2: notes.length ? notes.join(" · ") : "Ready to analyse",
    };
  }
  if (mode === "compare") {
    return selectionOk
      ? { l1: `${plural(videos, "video")} × ${plural(dubs, "dub")}`, l2: `${plural(videos * dubs, "test")}` }
      : { icon: "warn", l1: `${plural(videos, "video")} × ${plural(dubs, "dub")}`, l2: selectionReason ?? undefined };
  }
  const summary = `${plural(videos, "video")} · ${dubs ? "1 dub" : "no dub yet"}`;
  return selectionOk ? { l1: summary, l2: "Ready to analyse" } : { icon: "warn", l1: summary, l2: selectionReason ?? undefined };
}

/** A run's name in History: "Blade Runner 2049 (2017)", "Goblin · Season 1",
 *  "Parasite (2019) · 3 × 3". */
function runName(mode: Mode, state: ReturnType<typeof syncReducer>): string | undefined {
  const names = state.videoFiles.map((f) => f.name);
  if (mode === "series") return seriesTitle(state);
  const title = commonTitle(names);
  const known = title !== "Releases" ? title : names.length === 1 ? names[0].replace(/\.[^.]+$/, "") : undefined;
  if (mode === "compare") return known ? `${known} · ${state.videoFiles.length} × ${state.audioFiles.length}` : undefined;
  return known;
}

/** Find match in History: how many dubs found their release. */
function matchOutcome(results: SyncResult[], dubs: number): { tone: "ok" | "warn"; text: string } {
  const matched = new Set(results.filter((r) => !r.error && (r.confidence ?? 0) >= 0.75).map((r) => r.secondaryPath ?? r.audioFile)).size;
  return { tone: matched === dubs ? "ok" : "warn", text: `${matched} match${matched === 1 ? "" : "es"}${matched < dubs ? ` of ${dubs} dubs` : ""}` };
}

function formatLeft(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  const m = Math.round(s / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

