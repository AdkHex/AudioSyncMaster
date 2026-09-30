/** Dub sync: lay dubs onto their movies, a queue at a time.
 *
 *  Movies and their dubs are added in any order and paired by the engine
 *  ("auto": by episode number when that pairs every movie, by name
 *  otherwise); a pair can be corrected by hand. The run syncs the queue in
 *  parallel, each job streaming its stages, the plan as it is laid down,
 *  and the check of the written track. A finished job's cuts can then be
 *  edited in the Timeline and written again, and played back against the
 *  picture.
 *
 *  The page keeps its own state while hidden, and holds the shared engine
 *  for a run; its events come back here wherever the user has gone since. */

import {
  AddRegular,
  ArrowRedoRegular,
  ArrowUndoRegular,
  CutRegular,
  DeleteRegular,
  HeadphonesSoundWaveRegular,
  MergeRegular,
  MusicNote2Regular,
  PlayRegular,
  StopRegular,
  ZoomFitRegular,
  ZoomInRegular,
  ZoomOutRegular,
} from "@fluentui/react-icons";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { toast } from "sonner";

import { PageDock } from "@/app/dock";
import { useEngineListeners, usePageCommands, useShell } from "@/app/shell";
import { announceFilesAdded } from "@/lib/announce";
import * as api from "@/lib/api";
import { initialDubQueueState, reportText, type DubQueueJob } from "@/lib/dubQueueReducer";
import {
  asLoadedPlan,
  canMerge,
  codecOfPath,
  convertToDub,
  convertToFill,
  frameSeconds,
  mergeWithNext,
  nudgeOffset,
  segmentAt,
  setOffset,
  splitAt,
} from "@/lib/dubPlanEdit";
import { applyOverrides, countManualPairs, pruneOverrides, type PairOverrides } from "@/lib/pairing";
import { windowAround } from "@/lib/previewClock";
import type { MixMode } from "@/lib/previewMix";
import { createSummaryEntry } from "@/lib/storage";
import { initialSyncState, syncReducer, type SyncState } from "@/lib/syncReducer";
import type { AnalyzeRequest, DubSyncJob, FileItem, TrackListing } from "@/lib/types";
import { PageView } from "@/ui/frame";
import { Btn, Cmd, Empty, baseName } from "@/ui/kit";

import { createCursorStore } from "./dub/cursorStore";
import { FailedBox, DoneBox, OutputBox, StagesBox, StretchBox, type TrackChoice } from "./dub/DubInspector";
import { batchSummary, channelName, isVideoLike, pageQueueReducer, pageStatus, plural, shortReason, shortTitle, stretchName, buildRows, type QueueRow } from "./dub/dubModel";
import { DubPreview } from "./dub/DubPreview";
import { DubQueue, type Repair } from "./dub/DubQueue";
import { DubTimeline, TimelineMessage, type DubTimelineHandle, type TimelineMode } from "./dub/DubTimeline";
import { usePlanEditor } from "./dub/usePlanEditor";
import "./dub/dub.css";

const INITIAL_FILES: SyncState = { ...initialSyncState, mode: "dubsync", dubScope: "auto" };

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function DubPage({ hidden }: { hidden: boolean }) {
  const shell = useShell();
  const { desktop, settings, announce } = shell;

  const [files, dispatch] = useReducer(syncReducer, INITIAL_FILES);
  const [queue, queueDispatch] = useReducer(pageQueueReducer, initialDubQueueState);
  // Counts runs, so a job's edits never outlive the run that made its plan.
  const [batch, setBatch] = useState(0);
  const [runMux, setRunMux] = useState(false);
  // The jobs of the run under way, when it is some of the queue (a retry).
  const [runJobs, setRunJobs] = useState<number[] | null>(null);
  const [overrides, setOverrides] = useState<PairOverrides>({});
  const [listings, setListings] = useState<Record<string, TrackListing>>({});
  const [trackChoices, setTrackChoices] = useState<Record<string, number>>({});
  const [pairingLoading, setPairingLoading] = useState(false);
  // Files the waveform engine is reading for the first time: path to percent.
  const [reading, setReading] = useState<Record<string, number>>({});
  // Lengths of files read for their waveforms, by path#track.
  const [lengths, setLengths] = useState<Record<string, number>>({});
  const [lengthError, setLengthError] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [stretch, setStretch] = useState<number | null>(null);
  // The player: the window it plays, and the AudioContext made in the
  // gesture that opened it.
  const [player, setPlayer] = useState<{ window: { startS: number; endS: number }; context: AudioContext } | null>(null);
  const [mix, setMix] = useState<MixMode>("both");

  const cursor = useMemo(() => createCursorStore(), []);
  const timelineRef = useRef<DubTimelineHandle>(null);
  const requestedRef = useRef(new Set<string>());
  const mountedRef = useRef(true);
  // Engine job index to queue index, for a run of some of the queue's jobs.
  const jobMapRef = useRef<number[] | null>(null);

  const filesRef = useRef(files);
  filesRef.current = files;
  const queueRef = useRef(queue);
  queueRef.current = queue;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const trackChoicesRef = useRef(trackChoices);
  trackChoicesRef.current = trackChoices;
  const playerRef = useRef(player);
  playerRef.current = player;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const running = queue.status === "running";

  // ----------------------------------------------------------------- engine

  const toQueue = (job: number): number | undefined => {
    const map = jobMapRef.current;
    return map ? map[job] : job;
  };

  useEngineListeners("dubsync", {
    onDubQueueJobStart: (event) => {
      const job = toQueue(event.job);
      if (job !== undefined) queueDispatch({ type: "jobStart", job });
    },
    onDubQueueJobProgress: (event) => {
      const job = toQueue(event.job);
      if (job !== undefined) queueDispatch({ type: "jobProgress", job, percent: event.percent, stage: event.stage });
    },
    onDubQueueJobDraft: (event) => {
      const job = toQueue(event.job);
      if (job !== undefined) queueDispatch({ type: "jobDraft", job, plan: event.plan });
    },
    onDubQueueJobPlan: (event) => {
      const job = toQueue(event.job);
      if (job !== undefined) queueDispatch({ type: "jobPlan", job, plan: event.plan });
    },
    onDubQueueJobDone: (event) => {
      const job = toQueue(event.job);
      if (job === undefined) return;
      queueDispatch({
        type: "jobDone",
        outcome: {
          job,
          plan: event.plan ?? null,
          output: event.output ?? null,
          verification: event.verification ?? null,
          muxedPath: event.muxedPath ?? null,
          cancelled: event.cancelled,
          error: event.error ?? null,
        },
      });
    },
    // A single-job command is only ever a finished job being written again
    // from edited cuts; its progress belongs to that row.
    onDubSyncProgress: (event) => {
      const job = queueRef.current.rerender?.job;
      if (job !== undefined) queueDispatch({ type: "jobProgress", job, percent: event.percent, stage: event.stage });
    },
    onWaveformProgress: (event) =>
      setReading((current) => {
        if (event.percent >= 100) {
          if (!(event.path in current)) return current;
          const next = { ...current };
          delete next[event.path];
          return next;
        }
        return current[event.path] === event.percent ? current : { ...current, [event.path]: event.percent };
      }),
  });

  // ------------------------------------------------------------------ files

  /** Add files: a movie is anything with a picture, a dub anything else,
   *  unless the picker said which side they are for. */
  const addFiles = useCallback((kind: "video" | "audio" | "auto", list: FileItem[]) => {
    const videos = kind === "video" ? list : kind === "audio" ? [] : list.filter((f) => isVideoLike(f.name));
    const dubs = kind === "audio" ? list : kind === "video" ? [] : list.filter((f) => !isVideoLike(f.name));
    if (videos.length) dispatch({ type: "addFiles", kind: "video", files: videos.map((f) => ({ ...f, type: "video" as const })), explicit: true });
    if (dubs.length) dispatch({ type: "addFiles", kind: "audio", files: dubs.map((f) => ({ ...f, type: "audio" as const })) });
    // The queue shows what arrived; only a screen reader needs telling.
    if (videos.length) announce(announceFilesAdded(videos.length, "video"));
    if (dubs.length) announce(announceFilesAdded(dubs.length, "audio"));
    return videos.length + dubs.length;
  }, [announce]);

  const browse = useCallback(
    async (kind: "video" | "audio") => {
      if (!desktop) {
        toast.error("Adding files needs the desktop app.");
        return;
      }
      if (queueRef.current.status === "running") return;
      try {
        const picked = await api.pickMediaFiles(kind);
        addFiles(kind, picked.files);
      } catch (error) {
        toast.error(message(error) || "Could not open the picker");
      }
    },
    [desktop, addFiles],
  );

  /** Dropped paths or a folder: resolved by the engine, a folder adding
   *  what it holds. */
  const addPaths = useCallback(
    async (paths: string[]) => {
      if (!desktop) {
        toast.error("Adding files needs the desktop app.");
        return;
      }
      if (queueRef.current.status === "running") {
        toast.info("Add files once the sync has finished.");
        return;
      }
      try {
        const found = await api.resolveDroppedPaths(paths, "video", "media");
        if (addFiles("auto", found) === 0) toast.error("No supported media files there.");
      } catch {
        toast.error("Could not read those files.");
      }
    },
    [desktop, addFiles],
  );

  const addFolder = useCallback(async () => {
    if (!desktop) {
      toast.error("Adding files needs the desktop app.");
      return;
    }
    try {
      const picked = await api.pickVideoFolder();
      if (picked.folder) await addPaths([picked.folder]);
    } catch (error) {
      toast.error(message(error) || "Could not open the picker");
    }
  }, [desktop, addPaths]);

  // The audio streams of every file, so a file carrying several can be
  // told which to use.
  useEffect(() => {
    const paths = [...files.videoFiles, ...files.audioFiles].map((f) => f.path);
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
        setTrackChoices((current) => {
          const next: Record<string, number> = {};
          Object.entries(current).forEach(([path, index]) => {
            if ((byPath[path]?.tracks ?? []).some((t) => t.index === index)) next[path] = index;
          });
          return next;
        });
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [files.videoFiles, files.audioFiles]);

  // Pairs chosen by hand for files that are gone are dropped.
  useEffect(() => {
    setOverrides((current) => {
      if (Object.keys(current).length === 0) return current;
      const pruned = pruneOverrides(
        current,
        files.videoFiles.map((f) => f.path),
        files.audioFiles.map((f) => f.path),
      );
      return Object.keys(pruned).length === Object.keys(current).length ? current : pruned;
    });
  }, [files.videoFiles, files.audioFiles]);

  // The engine's pairing, refreshed once the selection settles. Not while
  // a run holds the engine; it is asked again when the run ends.
  useEffect(() => {
    if (!desktop || running) return;
    if (files.videoFiles.length === 0 || files.audioFiles.length === 0) {
      dispatch({ type: "setPairing", pairing: null });
      setPairingLoading(false);
      return;
    }
    let active = true;
    setPairingLoading(true);
    const timer = window.setTimeout(() => {
      const current = filesRef.current;
      const config = settingsRef.current;
      const request: Partial<AnalyzeRequest> = {
        mode: "dubsync",
        videoFolder: current.videoFolder,
        audioFolder: current.audioFolder,
        audioFile: null,
        videoFiles: current.videoFiles.map((f) => f.path),
        audioFiles: current.audioFiles.map((f) => f.path),
        matchPattern: null,
        // Episodes when the episode matcher pairs every movie, names otherwise.
        dubKind: "auto",
        videoTrack: 0,
        audioTrack: 0,
        pairs: null,
        windowSeconds: config.windowSeconds,
        windowCount: config.windowCount,
        maxOffsetMs: config.maxOffsetMs,
        maxWorkers: config.maxWorkers,
        findCuts: config.cutCheck,
        timeline: config.timelineCheck,
        findSpeed: config.rateCheck,
      };
      api
        .previewPairs(request)
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
  }, [desktop, running, files.videoFiles, files.audioFiles]);

  /** What will run: the engine's pairing with the corrections made by hand. */
  const pairing = useMemo(
    () =>
      files.pairing
        ? applyOverrides(
            files.pairing,
            overrides,
            files.audioFiles.map((f) => ({ path: f.path, name: f.name })),
          )
        : null,
    [files.pairing, overrides, files.audioFiles],
  );

  const rows = useMemo(
    () =>
      buildRows({
        running,
        jobs: queue.jobs,
        pairs: pairing?.pairs ?? null,
        videos: files.videoFiles,
        dubs: files.audioFiles,
        overrides,
        trackChoices,
      }),
    [running, queue.jobs, pairing, files.videoFiles, files.audioFiles, overrides, trackChoices],
  );
  const rowsRef = useRef(rows);
  rowsRef.current = rows;

  const shown: QueueRow | null = rows.find((r) => r.key === selectedKey) ?? rows[0] ?? null;
  const shownRef = useRef(shown);
  shownRef.current = shown;
  const job: DubQueueJob | null = shown?.job ?? null;
  const empty = files.videoFiles.length + files.audioFiles.length === 0;

  // ---------------------------------------------------------------- lengths

  // Read both files of the pair on screen for their waveforms: the engine
  // keeps what it read, so every view that follows is immediate, and their
  // lengths draw the pair as loaded.
  useEffect(() => {
    if (!desktop || !shown?.videoPath || !shown.dubPath) return;
    const want = [
      { path: shown.videoPath, track: shown.videoTrack },
      { path: shown.dubPath, track: shown.dubTrack },
    ];
    void (async () => {
      for (const file of want) {
        const key = `${file.path}#${file.track}`;
        if (requestedRef.current.has(key)) continue;
        requestedRef.current.add(key);
        try {
          const ready = await api.waveformBuild({ path: file.path, track: file.track });
          if (mountedRef.current) setLengths((current) => ({ ...current, [key]: ready.durationS }));
        } catch (error) {
          requestedRef.current.delete(key);
          if (mountedRef.current) setLengthError(message(error));
        }
      }
    })();
  }, [desktop, shown?.videoPath, shown?.dubPath, shown?.videoTrack, shown?.dubTrack]);

  const loadedPlan = useMemo(() => {
    if (!shown?.videoPath || !shown.dubPath) return null;
    const videoS = lengths[`${shown.videoPath}#${shown.videoTrack}`];
    const dubS = lengths[`${shown.dubPath}#${shown.dubTrack}`];
    if (videoS === undefined || dubS === undefined) return null;
    return asLoadedPlan({
      videoPath: shown.videoPath,
      dubPath: shown.dubPath,
      videoTrack: shown.videoTrack,
      dubTrack: shown.dubTrack,
      videoDurationS: videoS,
      dubDurationS: dubS,
    });
  }, [shown?.videoPath, shown?.dubPath, shown?.videoTrack, shown?.dubTrack, lengths]);

  // ---------------------------------------------------------------- editing

  const editorKey = job && job.status === "done" && job.plan ? `${batch}:${job.id}` : null;
  const editor = usePlanEditor({ key: editorKey, plan: editorKey ? job!.plan : null, enginePlan: editorKey ? job!.enginePlan : null });
  const editable = desktop && editorKey !== null && !running;
  const editing = editable && editor.key === editorKey && editor.shown !== null;
  const timelinePlan = editing ? editor.shown : (job?.plan ?? job?.draft ?? loadedPlan);
  const timelineMode: TimelineMode = job?.plan ? "done" : job?.draft ? "placing" : "loaded";
  const sel = editing && stretch !== null && stretch < editor.shown!.segments.length ? stretch : null;
  const frameS = timelinePlan ? frameSeconds(timelinePlan) : 1 / 24;
  const playable = desktop && job?.status === "done" && !!job.plan && !running;
  // A pair, or a new run of it, starts on the whole film.
  const viewKey = `${shown?.key ?? ""}|${batch}`;

  const closePlayer = useCallback(() => {
    const current = playerRef.current;
    if (!current) return;
    void current.context.close().catch(() => undefined);
    playerRef.current = null;
    setPlayer(null);
  }, []);

  // Another pair: its own stretches, playhead and player.
  useEffect(() => {
    setStretch(null);
    cursor.set(null);
    closePlayer();
  }, [shown?.key, cursor, closePlayer]);

  // The player stops for a run (the engine is taken) and when the page is put away.
  useEffect(() => {
    if (running || hidden) closePlayer();
  }, [running, hidden, closePlayer]);

  /** Open the player on the plan as it stands. The AudioContext is made
   *  here, in the gesture that opened it, which is what Web Audio asks for. */
  const openPlayer = () => {
    const plan = editing ? editor.plan : job?.plan;
    if (!plan || !playable || typeof AudioContext === "undefined") return;
    const context = playerRef.current?.context ?? new AudioContext();
    void context.resume().catch(() => undefined);
    if (cursor.get() === null) cursor.set(0);
    setPlayer({ window: windowAround(cursor.get() ?? 0, plan.videoDurationS), context });
  };

  /** Render a span around the playhead and hand it to the user's own player. */
  const openExternal = async () => {
    const plan = editing ? editor.plan : job?.plan;
    if (!plan) return;
    const startS = Math.max(0, (cursor.get() ?? 0) - 4);
    const endS = Math.min(plan.videoDurationS, startS + 12);
    try {
      const excerpt = await api.renderDubPreview({ plan, startS, endS, what: "both" });
      if (!excerpt) {
        toast.error("The preview could not be rendered.");
        return;
      }
      await api.openPath(excerpt.path);
    } catch (error) {
      toast.error("The preview could not be rendered.", { description: message(error) });
    }
  };

  const splitAtCursor = () => {
    const t = cursor.get();
    if (!editing || t === null) return;
    editor.change((plan) => splitAt(plan, t));
  };
  const nudge = (deltaS: number) => {
    if (sel === null) return;
    editor.change((plan) => nudgeOffset(plan, sel, deltaS));
  };
  const onSplitAt = (t: number) => {
    const index = editor.shown ? segmentAt(editor.shown, t) : null;
    editor.change((plan) => splitAt(plan, t));
    setStretch(index);
  };
  const canMergeSel = editing && sel !== null && canMerge(editor.shown!.segments[sel], editor.shown!.segments[sel + 1]);

  /** The editor's keys, while the Timeline or the stretch has focus. */
  const onEditorKey = (event: KeyboardEvent<HTMLElement>) => {
    const target = event.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT") return;
    const meta = event.metaKey || event.ctrlKey;
    const key = event.key.toLowerCase();
    if (meta && key === "z") {
      if (!editing) return;
      event.preventDefault();
      if (event.shiftKey) editor.redo();
      else editor.undo();
      return;
    }
    if (meta || event.altKey) return;
    if (event.key === " " && !player && playable) {
      event.preventDefault();
      openPlayer();
      return;
    }
    if (!editing) return;
    // Delete must not take the pair away from under the editor.
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      return;
    }
    // Nothing else while a stretch is being slipped: the slip commits from
    // the plan it started on.
    if (editor.slipping) return;
    if (event.key === "Escape" && sel !== null) {
      event.preventDefault();
      event.stopPropagation();
      setStretch(null);
    } else if (sel !== null && editor.shown!.segments[sel].kind === "dub" && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      event.preventDefault();
      const unit = event.shiftKey ? 0.01 : frameS;
      nudge(event.key === "ArrowLeft" ? -unit : unit);
    } else if (sel !== null && (event.key === "," || event.key === ".")) {
      // A millisecond at a time, on the frame-step keys of every editor.
      event.preventDefault();
      nudge(event.key === "," ? -0.001 : 0.001);
    } else if (key === "s") {
      event.preventDefault();
      splitAtCursor();
    }
  };

  // -------------------------------------------------------------------- run

  const claim = (): boolean => {
    if (shell.claimEngine("dubsync")) return true;
    toast.error("Another page is running.", { description: "Wait for it to finish, or stop it there." });
    return false;
  };

  /** Run jobs on the engine and settle the queue, the toasts, History. */
  const runEngine = async (jobs: DubSyncJob[], names: { name: string }[]) => {
    const config = settingsRef.current;
    try {
      const result = await api.startDubSyncBatch({
        jobs,
        codec: config.dubCodec,
        mux: config.dubMux,
        language: config.dubMux && config.dubLanguage.trim() ? config.dubLanguage.trim() : null,
        fillUnmatched: config.dubFillUnmatched,
        dubRate: config.dubRate,
        fixVoices: config.fixVoices,
        // Preferences › General › Save to; beside each dub when unset.
        outputDir: config.outputDir,
        // The outputs are this app's own files, named after the dubs; a
        // re-run is meant to replace them.
        overwrite: true,
        maxWorkers: config.maxWorkers,
      });
      const map = jobMapRef.current;
      queueDispatch({
        type: "batchDone",
        outcomes: map ? result.outcomes.map((o) => ({ ...o, job: map[o.job] ?? -1 })) : result.outcomes,
        cancelled: result.cancelled,
      });
      const summary = batchSummary(names, result.outcomes, result.cancelled);
      shell.addHistory(createSummaryEntry("dubsync", summary.name, { tone: summary.tone, text: summary.text }, jobs.length));

      const done = result.outcomes.filter((o) => !o.cancelled && !o.error && !o.plan?.error && o.output).length;
      const failed = result.outcomes.filter((o) => !o.cancelled).length - done;
      if (result.cancelled) {
        toast.info("Dub sync stopped.");
        shell.announce({ message: "Dub sync stopped.", politeness: "polite" });
      } else if (failed > 0) {
        // The first failure is what the user needs to see next.
        const first = result.outcomes.find((o) => !o.cancelled && (o.error || o.plan?.error || !o.output));
        const failedVideo = first ? jobs[first.job]?.videoPath : undefined;
        if (failedVideo) setSelectedKey(failedVideo);
        toast.error(`${plural(failed, "dub")} could not be synced`, { description: `${done} written. The Output says why.` });
        shell.announce({ message: `${plural(done, "dub")} written, ${failed} failed.`, politeness: "assertive" });
        shell.openOutput();
      } else if (done > 0) {
        const first = result.outcomes.find((o) => o.output)?.output?.outputPath;
        toast.success(`${plural(done, "synced track")} written`, {
          action: first ? { label: "Show", onClick: () => void api.revealPath(first).catch(() => undefined) } : undefined,
        });
        shell.announce({ message: `${plural(done, "synced track")} written.`, politeness: "polite" });
      }
    } catch (error) {
      const text = message(error);
      queueDispatch({ type: "batchFailed", message: text });
      shell.addHistory(createSummaryEntry("dubsync", batchSummary(names, [], false).name, { tone: "bad", text: `Failed — ${shortReason(text)}` }, jobs.length));
      shell.announce({ message: `Dub sync failed: ${text}`, politeness: "assertive" });
      toast.error("Dub sync failed", { description: text });
      shell.openOutput();
    } finally {
      jobMapRef.current = null;
      shell.releaseEngine("dubsync");
    }
  };

  /** Sync every pair in the queue, in parallel. */
  const runAll = async () => {
    if (!desktop) {
      toast.error("Dub sync needs the desktop app.");
      return;
    }
    if (queueRef.current.status === "running") return;
    const pairs = rowsRef.current.filter((r) => r.paired && r.videoPath && r.dubPath);
    if (pairs.length === 0) {
      toast.error("No pairs to sync.", { description: "Add the movies and their dubs." });
      return;
    }
    if (!claim()) return;
    closePlayer();
    const jobs: DubSyncJob[] = pairs.map((r) => ({ videoPath: r.videoPath!, dubPath: r.dubPath!, videoTrack: r.videoTrack, dubTrack: r.dubTrack }));
    const names = jobs.map((j) => ({ name: baseName(j.videoPath) }));
    jobMapRef.current = null;
    setRunJobs(null);
    setBatch((n) => n + 1);
    setRunMux(settingsRef.current.dubMux);
    setStretch(null);
    queueDispatch({
      type: "queueStarted",
      jobs: jobs.map((j, i) => ({ ...j, name: names[i].name, dubName: baseName(j.dubPath) })),
    });
    shell.clearLogs();
    shell.announce({ message: `Syncing ${plural(jobs.length, "dub")}.`, politeness: "polite" });
    await runEngine(jobs, names);
  };

  /** Run one finished job again, with its dub or another, keeping the rest. */
  const retry = async (row: QueueRow, dub?: FileItem) => {
    const j = row.job;
    if (!j || !desktop || queueRef.current.status === "running") return;
    if (!claim()) return;
    const run: DubSyncJob = {
      videoPath: j.videoPath,
      dubPath: dub?.path ?? j.dubPath,
      videoTrack: j.videoTrack,
      dubTrack: dub ? (trackChoicesRef.current[dub.path] ?? 0) : j.dubTrack,
    };
    jobMapRef.current = [j.id];
    setRunJobs([j.id]);
    setRunMux(settingsRef.current.dubMux);
    queueDispatch({ type: "retryStarted", jobs: [{ index: j.id, dubPath: run.dubPath, dubName: baseName(run.dubPath), dubTrack: run.dubTrack }] });
    shell.clearLogs();
    shell.announce({ message: `Syncing ${j.name} again.`, politeness: "polite" });
    await runEngine([run], [{ name: j.name }]);
  };

  /** Pick another dub for a failed pair and run it with that one. */
  const chooseAnotherDub = async (row: QueueRow) => {
    if (!desktop || !row.videoPath) return;
    try {
      const picked = await api.pickMediaFiles("audio");
      const chosen = picked.files[0];
      if (!chosen) return;
      const existing = filesRef.current.audioFiles.find((f) => f.path === chosen.path);
      if (!existing) addFiles("audio", [chosen]);
      const videoPath = row.videoPath;
      setOverrides((current) => ({ ...current, [videoPath]: chosen.path }));
      await retry(row, existing ?? chosen);
    } catch (error) {
      toast.error(message(error) || "Could not open the picker");
    }
  };

  /** Write a finished job's track again from cuts placed by hand: same
   *  file, same format, same mux as the engine's run. The write is staged,
   *  so stopping it keeps the track that was there. */
  const applyEdits = async () => {
    const j = job;
    const plan = editor.plan;
    if (!j || !plan || !editing || !editor.dirty || editor.problems.length > 0) return;
    if (j.status !== "done" || queueRef.current.status === "running") return;
    if (!claim()) return;
    closePlayer();
    const config = settingsRef.current;
    const outputPath = j.output?.outputPath ?? null;
    const mux = j.muxedPath !== null || config.dubMux;
    jobMapRef.current = null;
    setRunMux(mux);
    queueDispatch({ type: "rerenderStart", job: j.id, plan });
    shell.clearLogs();
    shell.announce({ message: `Writing ${j.name} again with the edited cuts.`, politeness: "polite" });
    try {
      const outcome = await api.startDubSync({
        videoPath: plan.videoPath,
        dubPath: plan.dubPath,
        videoTrack: plan.videoTrack,
        dubTrack: plan.dubTrack,
        codec: (outputPath && codecOfPath(outputPath)) || config.dubCodec,
        mux,
        language: mux && config.dubLanguage.trim() ? config.dubLanguage.trim() : null,
        fillUnmatched: config.dubFillUnmatched,
        fixVoices: config.fixVoices,
        overwrite: true,
        plan,
        outputPath,
        muxPath: j.muxedPath,
      });
      queueDispatch({
        type: "rerenderDone",
        outcome: {
          job: j.id,
          plan: outcome.plan ?? plan,
          output: outcome.output ?? null,
          verification: outcome.verification ?? null,
          muxedPath: outcome.muxedPath ?? null,
          cancelled: outcome.cancelled,
          error: outcome.error ?? null,
        },
      });
      if (outcome.cancelled) {
        toast.info("Stopped; the track that was there is kept.");
        shell.announce({ message: "Stopped; the track that was there is kept.", politeness: "polite" });
      } else if (outcome.error || outcome.plan?.error) {
        const text = outcome.error ?? outcome.plan?.error ?? "";
        toast.error("The track could not be written", { description: text });
        shell.announce({ message: `The track could not be written: ${text}`, politeness: "assertive" });
        shell.openOutput();
      } else {
        const written = outcome.output?.outputPath;
        toast.success("The track was written with your cuts", {
          action: written ? { label: "Show", onClick: () => void api.revealPath(written).catch(() => undefined) } : undefined,
        });
        shell.announce({ message: "The track was written with your cuts.", politeness: "polite" });
        setStretch(null);
      }
    } catch (error) {
      const text = message(error);
      queueDispatch({ type: "rerenderDone", outcome: { job: j.id, plan, output: null, verification: null, muxedPath: null, error: text } });
      toast.error("The track could not be written", { description: text });
      shell.announce({ message: `The track could not be written: ${text}`, politeness: "assertive" });
      shell.openOutput();
    } finally {
      shell.releaseEngine("dubsync");
    }
  };

  const stop = async () => {
    if (queueRef.current.status !== "running") return;
    try {
      await api.cancelSync();
      toast.info("Stopping…");
    } catch {
      toast.error("Could not stop the run.");
    }
  };

  /** Enter and the Sync button: edits not written yet are not thrown away
   *  without asking. */
  const start = () => {
    if (queueRef.current.status === "running") return;
    if (editing && editor.dirty) {
      toast.info(`${plural(editor.changes, "change")} to the cuts not applied`, {
        description: "Apply them, or sync again and lose them.",
        action: { label: "Sync again", onClick: () => void runAll() },
      });
      return;
    }
    void runAll();
  };

  // -------------------------------------------------------------- the list

  const repair: Repair = (videoPath, choice) =>
    setOverrides((current) => {
      const next = { ...current };
      if (choice === "auto") delete next[videoPath];
      else next[videoPath] = choice;
      return next;
    });

  const selectRow = (key: string) => {
    if (key === shownRef.current?.key) {
      // The row again: back to its summary.
      setStretch(null);
      closePlayer();
    }
    setSelectedKey(key);
  };

  const removeSelected = () => {
    const row = shownRef.current;
    if (!row || queueRef.current.status === "running") return;
    const all = rowsRef.current;
    const at = all.findIndex((r) => r.key === row.key);
    const next = all[at + 1] ?? all[at - 1] ?? null;
    const current = filesRef.current;
    const videoIds = current.videoFiles.filter((f) => f.path === row.videoPath).map((f) => f.id);
    const dubIds = current.audioFiles.filter((f) => f.path === row.dubPath).map((f) => f.id);
    if (videoIds.length) dispatch({ type: "removeFiles", kind: "video", ids: videoIds });
    if (dubIds.length) dispatch({ type: "removeFiles", kind: "audio", ids: dubIds });
    setSelectedKey(next?.key ?? null);
  };

  const clear = () => {
    if (queueRef.current.status === "running") return;
    closePlayer();
    dispatch({ type: "clearFiles", kind: "video" });
    dispatch({ type: "clearFiles", kind: "audio" });
    queueDispatch({ type: "reset" });
    setOverrides({});
    setSelectedKey(null);
  };

  usePageCommands("dubsync", {
    addVideos: () => void browse("video"),
    addAudio: () => void browse("audio"),
    addFolder: () => void addFolder(),
    removeSelected,
    clear,
    start,
    stop: () => void stop(),
    drop: (paths) => void addPaths(paths),
    addVideosLabel: "Add movies…",
    addAudioLabel: "Add dubs…",
  });

  // ---------------------------------------------------------------- display

  const onlyPair = rows.length === 1 && rows[0].paired ? rows[0] : null;
  const onlyLengths = (() => {
    if (!onlyPair?.videoPath || !onlyPair.dubPath) return null;
    const videoS = listings[onlyPair.videoPath]?.duration ?? lengths[`${onlyPair.videoPath}#${onlyPair.videoTrack}`];
    const dubS = listings[onlyPair.dubPath]?.duration ?? lengths[`${onlyPair.dubPath}#${onlyPair.dubTrack}`];
    return videoS != null && dubS != null ? { videoS, dubS } : null;
  })();

  const lcd = pageStatus({
    rows,
    queue,
    shown,
    videos: files.videoFiles.length,
    dubs: files.audioFiles.length,
    pairingLoading,
    lengths: onlyLengths,
    method: pairing?.method ?? null,
    manual: countManualPairs(overrides),
    edit:
      editing && editor.dirty
        ? {
            changes: editor.changes,
            stretch: sel !== null ? editor.shown!.segments[sel] : null,
            stretchName: sel !== null ? stretchName(editor.shown!, sel) : null,
          }
        : null,
    workers: settings.maxWorkers,
    now: Date.now(),
    runJobs,
  });

  const tools = (
    <>
      <Cmd icon={<AddRegular />} disabled={running} onClick={() => void browse("video")}>
        Add movies
      </Cmd>
      <Cmd icon={<MusicNote2Regular />} disabled={running} onClick={() => void browse("audio")}>
        Add dubs
      </Cmd>
      {!empty && <Cmd icon={<DeleteRegular />} disabled={running || !shown} title="Remove" onClick={removeSelected} />}
    </>
  );

  const canSync = desktop && rows.some((r) => r.paired);
  const primary = running ? (
    <Btn icon={<StopRegular />} kbd="Esc" onClick={() => void stop()}>
      Stop
    </Btn>
  ) : (
    <Btn accent icon={<PlayRegular />} kbd="Enter" disabled={!canSync} onClick={start}>
      {rows.some((r) => r.job) ? "Sync again" : "Sync"}
    </Btn>
  );

  // The details pane, for the selected pair.
  let inspector: ReactNode = null;
  if (shown) {
    const title = shortTitle(shown.videoName ?? shown.dubName ?? "");
    if (job && running && (job.status === "running" || job.status === "queued")) {
      inspector = <StagesBox title={title} job={job} mux={runMux} />;
    } else if (player && job?.plan) {
      inspector = (
        <DubPreview
          plan={editing && editor.plan ? editor.plan : job.plan}
          window={player.window}
          frameS={frameS}
          cursor={cursor}
          onNeedWindow={(aroundS) => setPlayer((current) => current && { ...current, window: windowAround(aroundS, job.plan!.videoDurationS) })}
          audioContext={player.context}
          mix={mix}
          onMix={setMix}
          active={!hidden}
          onOpenExternal={() => void openExternal()}
          render={api.renderDubPreview}
          read={api.readPreviewBytes}
        />
      );
    } else if (editing && sel !== null) {
      const segment = editor.shown!.segments[sel];
      inspector = (
        <StretchBox
          plan={editor.shown!}
          index={sel}
          frameS={frameS}
          problems={editor.problems}
          cursor={cursor}
          onNudge={nudge}
          onSetOffset={(value) => editor.change((plan) => setOffset(plan, sel, value))}
          onSplit={splitAtCursor}
          onUseOriginal={() => segment.kind === "dub" && editor.change((plan) => convertToFill(plan, sel))}
          onUseDub={() => segment.kind === "fill" && editor.change((plan) => convertToDub(plan, sel))}
          onBackToEngine={editor.isEngines ? undefined : editor.backToEngine}
          onKeyDown={onEditorKey}
        />
      );
    } else if (job?.status === "done") {
      inspector = (
        <DoneBox
          title={title}
          job={job}
          pieces={editing ? editor.shown : job.plan}
          selected={sel}
          problems={editing && editor.dirty ? editor.problems : []}
          onPlay={playable ? openPlayer : undefined}
          onShow={job.output ? () => void api.revealPath(job.output!.outputPath).catch(() => toast.error("That file no longer exists.")) : undefined}
          onReport={() => shell.copy(reportText(job))}
          onPiece={
            editing
              ? (index) => {
                  const s = editor.shown!.segments[index];
                  setStretch(index);
                  cursor.set(s.startS);
                  timelineRef.current?.show(s.startS, s.endS);
                }
              : undefined
          }
          onBackToEngine={editing && !editor.isEngines ? editor.backToEngine : undefined}
        />
      );
    } else if (job && (job.status === "failed" || job.status === "cancelled")) {
      const stopped = job.status === "cancelled";
      inspector = (
        <FailedBox
          title={title}
          stopped={stopped}
          reason={stopped ? "Stopped before it finished. Nothing was written." : (job.error ?? "The dub could not be synced.")}
          onRetry={desktop && !running ? () => void retry(shown) : undefined}
          onChooseDub={desktop && !running && shown.videoPath ? () => void chooseAnotherDub(shown) : undefined}
        />
      );
    } else {
      const dubListing = shown.dubPath ? listings[shown.dubPath] : undefined;
      const dubTrack = dubListing?.tracks.find((t) => t.index === shown.dubTrack) ?? dubListing?.tracks[0];
      const sameAs = dubTrack?.codec ? [dubTrack.codec.toUpperCase(), channelName(dubTrack.channels)].filter(Boolean).join(" ") : null;
      const trackChoice = (label: string, path: string | null, value: number): TrackChoice | null => {
        const tracks = path ? (listings[path]?.tracks ?? []) : [];
        if (!path || tracks.length < 2) return null;
        return {
          label,
          value,
          options: tracks.map((t) => ({ value: t.index, label: t.label })),
          onChange: (index) =>
            setTrackChoices((current) => {
              const next = { ...current };
              if (index > 0) next[path] = index;
              else delete next[path];
              return next;
            }),
        };
      };
      inspector = (
        <OutputBox
          settings={settings}
          onChange={shell.updateSettings}
          disabled={running}
          sameAs={sameAs}
          tracks={[trackChoice("Audio of the movie", shown.videoPath, shown.videoTrack), trackChoice("Audio of the dub", shown.dubPath, shown.dubTrack)].filter(
            (t): t is TrackChoice => t !== null,
          )}
        />
      );
    }
  }

  // The Timeline tab.
  const readingNote = (() => {
    if (!desktop) return "The timeline needs the desktop app.";
    if (!shown?.paired && !job) return shown?.videoPath ? "Choose a dub for this movie." : "Add the movie this dub belongs to.";
    if (lengthError) return `Could not read the waveforms: ${lengthError}`;
    const parts = Object.entries(reading)
      .filter(([path]) => path === shown?.videoPath || path === shown?.dubPath)
      .map(([path, pct]) => `${path === shown?.videoPath ? shown?.videoName : shown?.dubName} ${pct}%`);
    return parts.length ? `Reading the waveforms — ${parts.join(" · ")}` : "Reading the files…";
  })();

  const timelineTools = (
    <>
      {editing && editor.dirty && (
        <>
          <span className="t2 sm" style={{ marginRight: 8 }}>
            {plural(editor.changes, "change")}
          </span>
          <Btn accent style={{ height: 28, marginRight: 8 }} disabled={editor.problems.length > 0} onClick={() => void applyEdits()}>
            Apply
          </Btn>
        </>
      )}
      <Cmd sm icon={<ArrowUndoRegular />} title="Undo (Ctrl+Z)" disabled={!editing || !editor.canUndo} onClick={editor.undo} />
      <Cmd sm icon={<ArrowRedoRegular />} title="Redo (Ctrl+Shift+Z)" disabled={!editing || !editor.canRedo} onClick={editor.redo} />
      <span className="vsep" />
      <Cmd sm icon={<CutRegular />} title="Split (S)" disabled={!editing} onClick={splitAtCursor} />
      <Cmd sm icon={<MergeRegular />} title="Merge with next" disabled={!canMergeSel} onClick={() => sel !== null && editor.change((plan) => mergeWithNext(plan, sel))} />
      <span className="vsep" />
      <Cmd sm icon={<ZoomOutRegular />} title="Zoom out" disabled={!timelinePlan} onClick={() => timelineRef.current?.zoomBy(2)} />
      <Cmd sm icon={<ZoomInRegular />} title="Zoom in" disabled={!timelinePlan} onClick={() => timelineRef.current?.zoomBy(0.5)} />
      <Cmd sm icon={<ZoomFitRegular />} title="Whole film" disabled={!timelinePlan} onClick={() => timelineRef.current?.fit()} />
      <span className="vsep" />
      <Cmd
        sm
        icon={<PlayRegular />}
        title="Play (Space)"
        aria-pressed={player !== null}
        className={player ? "dub-on" : undefined}
        disabled={!playable}
        onClick={player ? closePlayer : openPlayer}
      />
    </>
  );

  const timelineBody = timelinePlan ? (
    <DubTimeline
      ref={timelineRef}
      plan={timelinePlan}
      mode={timelineMode}
      fetchPeaks={api.waveformPeaks}
      editable={editing}
      grips={editing && (sel !== null || editor.dirty)}
      selected={sel}
      onSelect={(index) => setStretch(index)}
      cursor={cursor}
      keepCursorInView={player !== null}
      onBoundaryDragStart={editor.dragStart}
      onBoundaryDrag={editor.drag}
      onBoundaryDragEnd={editor.dragEnd}
      onSplitAt={onSplitAt}
      onSlipStart={editor.slipStart}
      onSlip={editor.slip}
      onSlipEnd={editor.slipEnd}
      fetchShotCuts={desktop ? api.shotCuts : undefined}
      reading={reading}
      viewKey={viewKey}
      onKeyDown={onEditorKey}
    />
  ) : (
    <TimelineMessage>{readingNote}</TimelineMessage>
  );

  const pairs = rows.filter((r) => r.paired).length;
  const dock = PageDock({
    own: [{ id: "timeline" as const, label: "Timeline", body: timelineBody, tools: timelineTools }],
    ownTab: !empty && (pairs > 0 || rows.some((r) => r.job)) ? "timeline" : null,
    common: shell.dock,
    height: rows.length > 1 ? 250 : 300,
  });

  return (
    <PageView hidden={hidden} tools={tools} lcd={lcd} primary={primary} cols={empty ? "minmax(0,1fr)" : "minmax(0,1fr) 300px"} dock={dock}>
      {empty ? (
        <section className="box">
          <Empty icon={<HeadphonesSoundWaveRegular />} title="Drop movies and their dubs">
            <Btn icon={<AddRegular />} onClick={() => void browse("video")}>
              Add movies
            </Btn>
            <Btn icon={<MusicNote2Regular />} onClick={() => void browse("audio")}>
              Add dubs
            </Btn>
          </Empty>
        </section>
      ) : (
        <>
          <DubQueue rows={rows} selected={shown?.key ?? null} onSelect={selectRow} dubs={files.audioFiles} onRepair={running ? undefined : repair} />
          {inspector}
        </>
      )}
    </PageView>
  );
}
