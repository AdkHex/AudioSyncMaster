/** Subsync: the subtitle tools as one page of the workstation
 *  (design/mockup/screens/subs.tsx). A tool strip under the toolbar picks
 *  the tool; the Files list holds that tool's inputs and becomes the run and
 *  then its results; the side panel has the tool's settings (and, after a
 *  run, the selected file's result); the Cues tab of the dock shows the
 *  selected file's cues.
 *
 *  The page stays mounted while hidden, so its files, run and selection
 *  survive switching pages. The shell routes drops, menus and keys to it
 *  when it shows; Subsync streams its own "subsync-event" channel. */

import {
  AddRegular,
  ArrowLeftRegular,
  BrightnessHighRegular,
  CheckmarkRegular,
  ChevronDownRegular,
  DeleteRegular,
  DocumentMultipleRegular,
  DocumentTextRegular,
  FolderAddRegular,
  FolderOpenRegular,
  GaugeRegular,
  HdrRegular,
  MicRegular,
  PlayRegular,
  ScanTextRegular,
  StopRegular,
  TextEditStyleRegular,
  TimerRegular,
  TranslateRegular,
} from "@fluentui/react-icons";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type ComponentType, type ReactNode } from "react";
import { toast } from "sonner";

import { PageDock } from "@/app/dock";
import { usePageCommands, useShell } from "@/app/shell";
import { createSummaryEntry } from "@/lib/storage";
import * as subsApi from "@/lib/subsync/api";
import { LOW_CONFIDENCE } from "@/lib/subsync/format";
import {
  buildItems,
  buildJobs,
  emptyPools,
  firstVideoFps,
  pendingFile,
  poolsReducer,
  subtitleFileKind,
  type InputItem,
  type InputPool,
} from "@/lib/subsync/inputs";
import { initialRunState, runReducer, type RunJob } from "@/lib/subsync/reducer";
import { loadPrefs, savePrefs, type SubsyncPrefs } from "@/lib/subsync/storage";
import { TASK_LABELS, TOOLS, inputShape, toolOf, workersFor, type ToolId } from "@/lib/subsync/tools";
import type { CueData, CuePreview, MediaRef, OutputOptions, ProbedFile, SubTask, TaskOptions, TaskOutputFile } from "@/lib/subsync/types";
import { Box, PageView, type LcdProps } from "@/ui/frame";
import { Btn, Cmd, Empty, Links, Seg, TArea, TBox, Toggle, baseName, cx } from "@/ui/kit";

import { CueTable } from "./subsync/CuesDock";
import { packBusy, setFocus, useEngines } from "./subsync/engines";
import { FilesTable, MuxTable } from "./subsync/FileTable";
import { OutputSection } from "./subsync/OutputSection";
import { ResultView, type ResultActions } from "./subsync/ResultView";
import {
  STRIP,
  doneLcd,
  emptyTitle,
  historyLine,
  keyPath,
  needsCheck,
  readyLcd,
  runningLcd,
  stripLabel,
  type CueMode,
} from "./subsync/rows";
import { SETTINGS_PANELS, type SettingsPanelProps } from "./subsync/settings";
import { platformEngines } from "./subsync/settings/logic";
import { Field } from "./subsync/settings/fields";
import "./subsync/subsync.css";

const TOOL_ICON: Record<ToolId, ReactNode> = {
  sync: <TimerRegular />,
  ocr: <ScanTextRegular />,
  translate: <TranslateRegular />,
  fps: <GaugeRegular />,
  generate: <MicRegular />,
  style: <TextEditStyleRegular />,
  hdrSubs: <HdrRegular />,
  tonemap: <BrightnessHighRegular />,
  formats: <DocumentMultipleRegular />,
};

const FORMAT_TASKS: SubTask[] = ["convert", "extract", "mux"];

type CueLoad = { loading: boolean; preview: CuePreview | null; error: string | null };

/** OCR review of one job: corrected lines, lines confirmed, unsaved. */
interface OcrReview {
  edits: Record<number, string>;
  reviewed: ReadonlySet<number>;
  dirty: boolean;
}
const NO_REVIEW: OcrReview = { edits: {}, reviewed: new Set(), dirty: false };

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** The Tracks list's rows: the video, the subtitles, anything else added. */
function muxRowKeys(pool: InputPool, item: InputItem | undefined): string[] {
  const keys: string[] = [];
  if (item?.video) keys.push(item.video.path);
  for (const sub of item?.subtitles ?? []) keys.push(sub.path);
  for (const file of pool.files) if (file.kind !== "video" && file.kind !== "subtitle") keys.push(file.path);
  return keys;
}

/** The subtitle a row stands for, when it has readable text cues. */
function textSource(task: SubTask, pool: InputPool, item: InputItem | undefined, muxPath: string | null): MediaRef | null {
  const files = pool.files;
  if (inputShape(task) === "mux") {
    const file = muxPath ? files.find((f) => f.path === muxPath) : undefined;
    return file && file.kind === "subtitle" && !file.pending && subtitleFileKind(file) === "text" ? { path: file.path } : null;
  }
  if (inputShape(task) !== "subtitle" || !item?.subtitle) return null;
  const file = files.find((f) => f.path === item.subtitle!.path);
  if (!file || file.pending || file.error) return null;
  if (item.embedded) {
    const track = file.subtitleTracks.find((t) => t.index === item.subtitle!.track);
    return track?.kind === "text" ? item.subtitle : null;
  }
  return subtitleFileKind(file) === "text" ? item.subtitle : null;
}

export function SubsyncPage({ hidden }: { hidden: boolean }) {
  const shell = useShell();
  const engines = useEngines();
  const caps = engines.caps;
  const desktop = subsApi.isDesktop();

  const [prefs, setPrefs] = useState<SubsyncPrefs>(loadPrefs);
  const [pools, poolDispatch] = useReducer(poolsReducer, undefined, emptyPools);
  const [run, runDispatch] = useReducer(runReducer, initialRunState);
  /** The row key of each job of the run, by job index. */
  const [runKeys, setRunKeys] = useState<string[]>([]);
  /** The pool the run was started from: "Run again" while unchanged. */
  const [runPool, setRunPool] = useState<InputPool | null>(null);
  const [skipped, setSkipped] = useState(0);
  const [selected, setSelected] = useState<Partial<Record<SubTask, string>>>({});
  const [cuesOpen, setCuesOpen] = useState(false);
  const [pane, setPane] = useState<"settings" | "result">("settings");
  const [onlyCheck, setOnlyCheck] = useState(false);
  const [cueCache, setCueCache] = useState<Record<string, CueLoad>>({});
  const [reviews, setReviews] = useState<Record<string, OcrReview>>({});
  const [ocrLine, setOcrLine] = useState<number | null>(null);
  /** OCR review: the Lines list and the line in the side panel, in place of
   *  the Files list, its settings and the Cues dock. */
  const [reviewOpen, setReviewOpen] = useState(false);
  const [saving, setSaving] = useState(false);

  const task = prefs.task;
  const tool = toolOf(task);
  const shape = inputShape(task);
  const pool = pools[task];
  const options = prefs.options[task];
  const output = prefs.output[task];
  const busy = run.status === "running";
  const runHere = run.task === task && run.status !== "idle";
  const lockedHere = busy && run.task === task;

  useEffect(() => savePrefs(prefs), [prefs]);

  // Off macOS, the macOS engines the defaults name give way to ones that run.
  useEffect(() => {
    if (!caps) return;
    setPrefs((current) => {
      const options = platformEngines(current.options, caps.platform);
      return options === current.options ? current : { ...current, options };
    });
  }, [caps]);

  // Read through refs so the engine subscription is made once.
  const runRef = useRef(run);
  runRef.current = run;
  const taskRef = useRef(task);
  taskRef.current = task;
  const logRef = useRef(shell.log);
  logRef.current = shell.log;
  /** The Convert & mux tool reopens on the task used last; first on Mux. */
  const formatsTask = useRef<SubTask>(toolOf(task) === "formats" ? task : "mux");

  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;
    void subsApi
      .subscribe((event) => {
        runDispatch({ type: "event", event });
        if (event.type === "subsJobLog") {
          const label = runRef.current.jobs[event.job]?.label ?? `job ${event.job + 1}`;
          logRef.current(`${label}: ${event.message}`);
        } else if (event.type === "log") {
          logRef.current(event.message);
        }
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

  // ------------------------------------------------------------- files

  /** Add files to a task and probe them. */
  const addFiles = useCallback((target: SubTask, files: subsApi.PickedFile[]) => {
    if (files.length === 0) return;
    poolDispatch({ type: "add", task: target, files });
    subsApi
      .probe(files.map((file) => file.path))
      .then((probed) => {
        // A file the engine did not report on must not stay "Reading…".
        const seen = new Set(probed.map((file) => file.path));
        const missing = files
          .filter((file) => !seen.has(file.path))
          .map((file) => ({ ...pendingFile(file), error: "Could not read this file." }));
        poolDispatch({ type: "probed", files: [...probed, ...missing] });
      })
      .catch((error) => {
        const message = error instanceof Error ? error.message : "Could not read this file.";
        poolDispatch({ type: "probed", files: files.map((file) => ({ ...pendingFile(file), error: message })) });
      });
  }, []);

  const browse = async (folder: boolean) => {
    try {
      if (folder) {
        const path = await subsApi.pickFolder();
        if (path) addFiles(task, await subsApi.resolveDropped([path]));
      } else {
        const response = await subsApi.pickFiles(shape === "video" ? "video" : "any");
        addFiles(task, response.files);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not open the picker");
    }
  };

  const drop = (paths: string[]) => {
    subsApi
      .resolveDropped(paths)
      .then((files) => {
        if (files.length === 0) toast.error("No video or subtitle files in that drop.");
        else addFiles(taskRef.current, files);
      })
      .catch(() => toast.error("Could not read the dropped files."));
  };

  const setTask = useCallback((next: SubTask) => {
    if (toolOf(next) === "formats") formatsTask.current = next;
    setPrefs((current) => ({ ...current, task: next }));
    setOcrLine(null);
    setReviewOpen(false);
  }, []);

  const selectTool = (id: ToolId) =>
    setTask(id === "formats" ? formatsTask.current : (TOOLS.find((entry) => entry.id === id)?.tasks[0] ?? "sync"));

  // Bound to the task they were made for, so a patch that arrives late (a
  // reference picked after switching tools) cannot land on another task.
  const setOptions = useCallback(
    (patch: Partial<TaskOptions[SubTask]>) =>
      setPrefs((current) => ({
        ...current,
        options: { ...current.options, [task]: { ...current.options[task], ...patch } },
      })),
    [task],
  );
  const setOutput = useCallback(
    (patch: Partial<OutputOptions>) =>
      setPrefs((current) => ({ ...current, output: { ...current.output, [task]: { ...current.output[task], ...patch } } })),
    [task],
  );

  const chooseOutputFolder = async () => {
    try {
      const dir = await subsApi.pickFolder();
      if (dir) setOutput({ dir });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not open the picker");
    }
  };

  /** For reference inputs in the settings panels. */
  const pickReference = useCallback(async (accept: subsApi.PickAccept): Promise<ProbedFile | null> => {
    try {
      const response = await subsApi.pickFiles(accept);
      const file = response.files[0];
      if (!file) return null;
      const [probed] = await subsApi.probe([file.path]);
      return probed ?? pendingFile(file);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not open the picker");
      return null;
    }
  }, []);

  /** An Install / Add API key link: Preferences › Subtitles, at that row. */
  const openEngines = useCallback(
    (target: string) => {
      setFocus(target);
      shell.openPreferences("subs");
    },
    [shell],
  );

  // -------------------------------------------------------------- rows

  const items = useMemo(() => buildItems(task, pool, options), [task, pool, options]);
  const runnable = items.filter((item) => !item.problem).length;
  const rowKeys = shape === "mux" ? muxRowKeys(pool, items[0]) : items.map((item) => item.key);
  const chosen = selected[task];
  const current = chosen && rowKeys.includes(chosen) ? chosen : (rowKeys[0] ?? null);

  const jobs = useMemo(() => {
    const map = new Map<string, RunJob>();
    if (runHere) run.jobs.forEach((job, index) => runKeys[index] && map.set(runKeys[index], job));
    return map;
  }, [run, runHere, runKeys]);

  const currentItem = shape === "mux" ? items[0] : items.find((item) => item.key === current);
  const currentJob = (shape === "mux" ? jobs.get("mux") : current ? jobs.get(current) : undefined) ?? null;
  const hasOutcome = !!currentJob && (!!currentJob.outcome || currentJob.status === "failed" || currentJob.status === "cancelled");
  const shownPane = hasOutcome ? pane : "settings";

  const onSelect = (key: string) => {
    if (key === current && chosen === key) setCuesOpen((open) => !open);
    else {
      setSelected((all) => ({ ...all, [task]: key }));
      setCuesOpen(true);
    }
    setOcrLine(null);
  };

  const removeSelected = () => {
    if (lockedHere || !current) return;
    poolDispatch({ type: "remove", task, path: shape === "mux" ? current : keyPath(current) });
    setSelected((all) => ({ ...all, [task]: undefined }));
    setOcrLine(null);
  };

  // --------------------------------------------------------------- cues

  const resultPreview = currentJob?.outcome?.preview?.cues.length ? currentJob.outcome.preview : null;
  const source = resultPreview ? null : textSource(task, pool, currentItem, shape === "mux" ? current : null);
  const sourceKey = source ? `${source.path}#${source.track ?? ""}` : null;
  const sourceRef = useRef(source);
  sourceRef.current = source;
  const load = sourceKey ? cueCache[sourceKey] : undefined;

  useEffect(() => {
    const ref = sourceRef.current;
    if (!cuesOpen || !sourceKey || !ref || load || !desktop) return;
    setCueCache((cache) => ({ ...cache, [sourceKey]: { loading: true, preview: null, error: null } }));
    subsApi
      .loadCues(ref)
      .then((result) =>
        setCueCache((cache) => ({ ...cache, [sourceKey]: { loading: false, preview: result.preview, error: result.error ?? null } })),
      )
      .catch((error) =>
        setCueCache((cache) => ({ ...cache, [sourceKey]: { loading: false, preview: null, error: errorText(error) } })),
      );
  }, [cuesOpen, sourceKey, load, desktop]);

  const preview = resultPreview ?? load?.preview ?? null;
  const hasCues = !!resultPreview || !!source;
  const cueMode: CueMode = resultPreview && currentJob?.task === "ocr" ? "ocr" : "cps";
  const minConfidence = cueMode === "ocr" ? prefs.options.ocr.minConfidence : LOW_CONFIDENCE;

  // ---------------------------------------------------------- OCR review

  const ocrJob = cueMode === "ocr" ? currentJob : null;
  const review = ocrJob ? (reviews[ocrJob.id] ?? NO_REVIEW) : NO_REVIEW;
  const toCheck = useMemo(
    () => (preview ? preview.cues.flatMap((cue, index) => (!review.reviewed.has(index) && needsCheck(cue, cueMode, minConfidence) ? [index] : [])) : []),
    [preview, review, cueMode, minConfidence],
  );
  const reviewLine = ocrJob && preview && ocrLine !== null && preview.cues[ocrLine] ? ocrLine : null;
  /** Edits can be written back only when the engine sent every line. */
  const editable = !!preview && preview.count <= preview.cues.length;
  const ocrOutput = ocrJob?.outcome?.outputs.find((file) => file.kind === "subtitle") ?? null;
  const reviewing = reviewOpen && !!ocrJob && !!preview && !busy;
  const flaggedFolder = ocrJob?.outcome?.outputs.find((file) => file.kind === "folder") ?? null;

  const patchReview = (fn: (review: OcrReview) => OcrReview) => {
    if (!ocrJob) return;
    setReviews((all) => ({ ...all, [ocrJob.id]: fn(all[ocrJob.id] ?? NO_REVIEW) }));
  };

  const nextToCheck = (after: number | null = reviewLine, list: number[] = toCheck) => {
    if (list.length === 0) return;
    const next = list.find((index) => after === null || index > after) ?? list[0];
    setOcrLine(next);
    setReviewOpen(true);
  };

  const looksRight = () => {
    if (reviewLine === null) return;
    const line = reviewLine;
    patchReview((r) => ({ ...r, reviewed: new Set([...r.reviewed, line]) }));
    nextToCheck(line, toCheck.filter((index) => index !== line));
  };

  const saveReview = async () => {
    if (!ocrJob || !preview || !ocrOutput || !editable) return;
    setSaving(true);
    try {
      const cues: CueData[] = preview.cues.map((cue, index) => (index in review.edits ? { ...cue, text: review.edits[index] } : cue));
      const result = await subsApi.saveCues(ocrOutput.path, "", cues, ocrOutput.language ?? preview.language);
      if (result.error) toast.error(result.error);
      else {
        patchReview((r) => ({ ...r, dirty: false }));
        toast.success(`Saved ${baseName(ocrOutput.path)}`);
      }
    } catch (error) {
      toast.error(errorText(error));
    } finally {
      setSaving(false);
    }
  };

  // ---------------------------------------------------------------- run

  const engineElsewhere = shell.enginePage !== null && shell.enginePage !== "subsync";
  const installing = packBusy(engines.packs);
  const startRun = async () => {
    if (busy) return;
    if (!desktop) {
      toast.error("Subtitle tools need the desktop app.");
      return;
    }
    if (packBusy()) {
      toast.error("An engine is being installed. Run this once it has finished.");
      return;
    }
    const stamp = Date.now().toString(36);
    const built = buildJobs(task, items, options, output, (index) => `${task}-${stamp}-${index}`);
    if (built.jobs.length === 0) {
      toast.error("Nothing here can run yet.");
      return;
    }
    if (!shell.claimEngine("subsync")) {
      toast.error("Another page is running. Wait for it to finish, or stop it there.");
      return;
    }
    setRunKeys(built.keys);
    setRunPool(pool);
    setSkipped(shape === "mux" ? 0 : items.length - built.jobs.length);
    setOcrLine(null);
    setReviewOpen(false);
    runDispatch({
      type: "started",
      task,
      at: Date.now(),
      jobs: built.jobs.map((job, index) => ({ id: job.id, label: built.labels[index] })),
    });
    shell.log(`Subsync: ${TASK_LABELS[task]}, ${built.jobs.length} job${built.jobs.length === 1 ? "" : "s"}`);
    try {
      const result = await subsApi.startBatch(built.jobs, workersFor(task));
      runDispatch({ type: "finished", outcomes: result.outcomes ?? [], cancelled: result.cancelled, error: result.error });
    } catch (error) {
      runDispatch({ type: "failed", message: errorText(error) });
    } finally {
      shell.releaseEngine("subsync");
    }
  };

  const stop = () => {
    void subsApi.cancel().catch(() => toast.error("Could not stop the run."));
  };

  // A run settled: record it, select the first file and open its cues. The
  // side panel stays on Settings, as the mockup has it; the file's result is
  // a tab beside it.
  const settled = useRef(run.status);
  const prefsRef = useRef(prefs);
  prefsRef.current = prefs;
  const skippedRef = useRef(skipped);
  skippedRef.current = skipped;
  const runKeysRef = useRef(runKeys);
  runKeysRef.current = runKeys;
  const addHistory = shell.addHistory;
  useEffect(() => {
    const was = settled.current;
    settled.current = run.status;
    if (was !== "running" || run.status === "running" || run.status === "idle" || !run.task) return;
    const line = historyLine(run.task, run, skippedRef.current);
    addHistory(createSummaryEntry("subsync", line.name, { tone: line.tone, text: line.text }, run.jobs.length));
    const first = runKeysRef.current[0];
    const doneTask = run.task;
    if (first && first !== "mux") setSelected((all) => ({ ...all, [doneTask]: first }));
    setPane("settings");
    const cues = run.jobs[0]?.outcome?.preview?.cues ?? [];
    setCuesOpen(cues.length > 0);
    // OCR goes straight to its review, on the first line to check.
    if (doneTask === "ocr" && cues.length > 0) {
      const minimum = prefsRef.current.options.ocr.minConfidence;
      const first = cues.findIndex((cue) => needsCheck(cue, "ocr", minimum));
      setOcrLine(first >= 0 ? first : null);
      setReviewOpen(true);
    }
  }, [run, addHistory]);

  const resultActions: ResultActions = {
    onReveal: (path) => void subsApi.revealPath(path).catch(() => toast.error("That file no longer exists.")),
    onOpen: (path) => void subsApi.openPath(path).catch(() => toast.error("Could not open the file.")),
    onUseAs: (file: TaskOutputFile, target: SubTask) => {
      addFiles(target, [{ name: baseName(file.path), path: file.path, type: file.kind === "video" ? "video" : "subtitle", size: null }]);
      setTask(target);
      toast.success(`Added to ${TASK_LABELS[target]}`);
    },
    onOpenConsole: shell.openOutput,
  };

  usePageCommands("subsync", {
    drop,
    addVideos: () => void browse(false),
    addVideosLabel: "Add files…",
    addFolder: () => void browse(true),
    removeSelected,
    clear: () => {
      if (!lockedHere) poolDispatch({ type: "clear", task });
    },
    start: () => void startRun(),
    stop,
  });

  // --------------------------------------------------------------- view

  const empty = pool.files.length === 0;
  const again = runHere && !busy && runPool === pool;
  const label = stripLabel(tool);
  const lcd: LcdProps = busy
    ? runningLcd(stripLabel(toolOf(run.task!)), run)
    : again
      ? doneLcd(task, run, skipped)
      : readyLcd(label, shape, items, pool.files);

  const blocked = !desktop ? "Subtitle tools need the desktop app" : engineElsewhere ? "Another page is running" : installing ? "An engine is being installed" : runnable === 0 ? "Add files that can run" : undefined;
  const primary = busy ? (
    <Btn icon={<StopRegular />} onClick={stop}>
      Stop
    </Btn>
  ) : reviewing || (review.dirty && ocrJob) ? (
    <Btn accent icon={<CheckmarkRegular />} disabled={saving || !editable || !ocrOutput} onClick={() => void saveReview()}>
      Save
    </Btn>
  ) : (
    <Btn accent icon={<PlayRegular />} disabled={!!blocked || empty} title={blocked} onClick={() => void startRun()}>
      {task === "mux" ? "Mux" : again ? "Run again" : runnable > 1 ? `Run ${runnable}` : "Run"}
    </Btn>
  );

  // Reviewing OCR: the mockup's two commands. Otherwise the file commands,
  // and a way into the review while the toolbar has room only for its icon.
  const tools = reviewing ? (
    <>
      <Cmd icon={<AddRegular />} onClick={() => void browse(false)}>
        Add files
      </Cmd>
      <Cmd icon={<ChevronDownRegular />} disabled={toCheck.length === 0} onClick={() => nextToCheck()}>
        Next to check
      </Cmd>
    </>
  ) : (
    <>
      <Cmd icon={<AddRegular />} disabled={lockedHere} onClick={() => void browse(false)}>
        Add files
      </Cmd>
      <Cmd icon={<FolderAddRegular />} disabled={lockedHere} onClick={() => void browse(true)}>
        Add folder
      </Cmd>
      {!empty && <Cmd icon={<DeleteRegular />} disabled={lockedHere || !current} title="Remove" onClick={removeSelected} />}
      {ocrJob && !busy && <Cmd icon={<ChevronDownRegular />} title="Check the lines" onClick={() => nextToCheck()} />}
    </>
  );

  const strip = (
    <>
      {STRIP.map((entry) => (
        <button
          key={entry.id}
          type="button"
          className={cx("tool", entry.id === tool && "on")}
          aria-pressed={entry.id === tool}
          title={TOOLS.find((t) => t.id === entry.id)?.description}
          onClick={() => selectTool(entry.id)}
        >
          <span className="ic" aria-hidden>{TOOL_ICON[entry.id]}</span>
          {entry.label}
        </button>
      ))}
    </>
  );

  const taskSeg = (
    <Seg<SubTask>
      label="Task"
      value={task}
      disabled={lockedHere}
      items={FORMAT_TASKS.map((value) => ({ value, label: TASK_LABELS[value] }))}
      onChange={setTask}
    />
  );
  const Settings = SETTINGS_PANELS[task] as unknown as ComponentType<SettingsPanelProps<SubTask>>;
  const settingsBody = (
    <>
      {tool === "formats" && (
        <Field group label="Task">
          {taskSeg}
        </Field>
      )}
      {/* Open during a run, as the mockup has it: the run took its settings
          when it started, and these are for the next one. */}
      <Settings
        key={task}
        options={options}
        onChange={setOptions}
        caps={caps}
        disabled={false}
        onInstallPack={openEngines}
        pickFile={pickReference}
        videoFps={firstVideoFps(pool)}
      />
      <div className="hr" />
      <OutputSection task={task} output={output} disabled={false} onChange={setOutput} onChooseFolder={() => void chooseOutputFolder()} />
    </>
  );

  let side: ReactNode;
  if (reviewLine !== null && preview) {
    const cue = preview.cues[reviewLine];
    const sure = typeof cue.confidence === "number" ? Math.round(cue.confidence * 100) : null;
    const low = needsCheck(cue, "ocr", minConfidence);
    const text = review.edits[reviewLine] ?? cue.text;
    const setText = (value: string) =>
      patchReview((r) => ({ ...r, dirty: true, edits: { ...r.edits, [reviewLine]: value } }));
    side = (
      <Box
        title={`Line ${reviewLine + 1}`}
        label="Line to check"
        end={sure !== null && <span className={cx("sm", low ? "warn" : "t3")} style={{ paddingRight: 6 }}>{sure}% sure</span>}
      >
        {flaggedFolder && (
          <Links>
            <Cmd sm icon={<FolderOpenRegular />} onClick={() => resultActions.onReveal(flaggedFolder.path)}>
              Show the images of lines to check
            </Cmd>
          </Links>
        )}
        <div className="ss-fld">
          <span className="t3">Text</span>
          {text.includes("\n") ? (
            <TArea label="Text" rows={2} value={text} disabled={!editable} onChange={setText} />
          ) : (
            <TBox label="Text" value={text} disabled={!editable} onChange={setText} className="ss-cjk" />
          )}
          {!editable && (
            <span className="sm t3">
              Only the first {preview.cues.length.toLocaleString()} of {preview.count.toLocaleString()} lines came back from the engine, so corrections can't be saved here.
            </span>
          )}
        </div>
        <div className="row" style={{ gap: 8 }}>
          <Btn accent onClick={looksRight}>
            Looks right
          </Btn>
          <Btn disabled={toCheck.filter((index) => index !== reviewLine).length === 0} onClick={() => nextToCheck()}>
            Next
          </Btn>
        </div>
      </Box>
    );
  } else if (hasOutcome && currentJob) {
    side = (
      <section className="box" aria-label="Settings and result">
        <div className="box-h ss-tabs" role="tablist">
          {(["settings", "result"] as const).map((id) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={shownPane === id}
              className={cx("tab", shownPane === id && "on")}
              onClick={() => setPane(id)}
            >
              {id === "settings" ? "Settings" : "Result"}
            </button>
          ))}
        </div>
        <div className="box-b" role="tabpanel">
          {shownPane === "result" ? <ResultView job={currentJob} actions={resultActions} /> : settingsBody}
        </div>
      </section>
    );
  } else {
    side = (
      <Box title="Settings" label="Settings">
        {settingsBody}
      </Box>
    );
  }

  const cuesTools = preview ? (
    <>
      {preview.count > preview.cues.length && (
        <span className="t3 sm" style={{ marginRight: 12 }}>
          First {preview.cues.length.toLocaleString()} of {preview.count.toLocaleString()}
        </span>
      )}
      <span className="t2 sm" style={{ marginRight: 8 }}>
        Only lines to check ({toCheck.length.toLocaleString()})
      </span>
      <Toggle name="Only lines to check" on={onlyCheck} onChange={setOnlyCheck} />
    </>
  ) : undefined;

  const cuesBody = preview ? (
    <CueTable
      cues={preview.cues}
      mode={cueMode}
      minConfidence={minConfidence}
      onlyCheck={onlyCheck}
      reviewed={review.reviewed}
      edits={review.edits}
      selected={cueMode === "ocr" ? reviewLine : null}
      onSelect={cueMode === "ocr" ? setOcrLine : undefined}
      reveal={toCheck[0] ?? null}
      label="Cues"
    />
  ) : load?.error ? (
    <div className="log bad">{load.error}</div>
  ) : (
    <div className="log t3">Reading the subtitle…</div>
  );

  const dock = PageDock<"cues">({
    own: hasCues && !reviewing ? [{ id: "cues", label: "Cues", body: cuesBody, tools: cuesTools }] : [],
    ownTab: cuesOpen && hasCues && !reviewing ? "cues" : null,
    onOwnTab: () => setCuesOpen(true),
    common: shell.dock,
    height: 260,
  });

  return (
    <PageView
      hidden={hidden}
      tools={tools}
      lcd={lcd}
      primary={primary}
      strip={strip}
      cols={empty ? "minmax(0,1fr)" : "minmax(0,1fr) 300px"}
      dock={dock}
    >
      {empty ? (
        <section className="box">
          {desktop ? (
            <Empty icon={<DocumentTextRegular />} title={emptyTitle(shape)}>
              {/* Convert, Extract and Mux take different files: choose first. */}
              {tool === "formats" && taskSeg}
              <Btn icon={<AddRegular />} onClick={() => void browse(false)}>
                Add files
              </Btn>
            </Empty>
          ) : (
            <Empty icon={<DocumentTextRegular />} title="Subtitle tools need the desktop app" />
          )}
        </section>
      ) : (
        <>
          {reviewing ? (
            <Box
              body={false}
              title="Lines"
              sub={ocrJob!.label}
              label="Lines"
              end={<Cmd sm icon={<ArrowLeftRegular />} title="Back to the files" onClick={() => setReviewOpen(false)} />}
            >
              <CueTable
                cues={preview!.cues}
                mode="ocr"
                minConfidence={minConfidence}
                onlyCheck={false}
                reviewed={review.reviewed}
                edits={review.edits}
                selected={reviewLine}
                onSelect={setOcrLine}
                label="Lines"
                rowHeight={36}
                viewport={420}
              />
            </Box>
          ) : (
            <Box
              body={false}
              title={shape === "mux" ? "Tracks" : "Files"}
              sub={shape === "mux" ? `into one ${prefs.options.mux.container.toUpperCase()}` : String(items.length)}
            >
              {shape === "mux" ? (
                <MuxTable task={task} pool={pool} item={items[0]} current={current} onSelect={onSelect} disabled={lockedHere} dispatch={poolDispatch} />
              ) : (
                <FilesTable
                  task={task}
                  pool={pool}
                  items={items}
                  options={options}
                  jobs={jobs}
                  ranHere={runHere}
                  running={busy}
                  current={current}
                  onSelect={onSelect}
                  disabled={lockedHere}
                  dispatch={poolDispatch}
                />
              )}
            </Box>
          )}
          {side}
        </>
      )}
    </PageView>
  );
}
