import { Cpu, FilePlus, FolderPlus, Play, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { toast } from "sonner";

import { Dialog } from "@/components/Dialog";
import { CueTable } from "@/components/subsync/CueTable";
import { EmptyState, LinkButton, SectionTitle } from "@/components/subsync/controls";
import { EnginesPanel } from "@/components/subsync/EnginesPanel";
import { OutputSection } from "@/components/subsync/OutputSection";
import { QueueList } from "@/components/subsync/QueueList";
import { RunView } from "@/components/subsync/RunView";
import { TaskSettings } from "@/components/subsync/TaskSettings";
import { ToolRail } from "@/components/subsync/ToolRail";
import { Button, IconButton, Spinner } from "@/components/ui";
import { subscribeToFileDrop } from "@/lib/api";
import { cx } from "@/lib/cx";
import * as subsApi from "@/lib/subsync/api";
import { buildItems, buildJobs, emptyPools, firstVideoFps, pendingFile, poolsReducer } from "@/lib/subsync/inputs";
import { initialRunState, packsReducer, runReducer } from "@/lib/subsync/reducer";
import { loadPrefs, savePrefs, type SubsyncPrefs } from "@/lib/subsync/storage";
import { TASK_LABELS, TOOLS, inputShape, toolOf, workersFor, type ToolId } from "@/lib/subsync/tools";
import type {
  Capabilities,
  CuePreview,
  MediaRef,
  OutputOptions,
  ProbedFile,
  SubTask,
  TaskOptions,
  TaskOutputFile,
} from "@/lib/subsync/types";

interface SubsyncWorkspaceProps {
  /** Lines for the app console. */
  onLog: (message: string) => void;
  /** A batch started or ended; the header locks the mode switch meanwhile. */
  onBusyChange: (busy: boolean) => void;
  onOpenConsole: () => void;
}

/** The Subsync mode: a rail of subtitle tools, the chosen tool's settings,
 *  and its queue, which becomes the run and then its outcomes. */
export function SubsyncWorkspace(props: SubsyncWorkspaceProps) {
  if (!subsApi.isDesktop()) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto">
        <EmptyState
          title="Running in a browser"
          body="Subtitle files and engines need the desktop app."
        />
      </div>
    );
  }
  return <Workspace {...props} />;
}

type PreviewState = { name: string; loading: boolean; preview: CuePreview | null; error: string | null };

function Workspace({ onLog, onBusyChange, onOpenConsole }: SubsyncWorkspaceProps) {
  const [prefs, setPrefs] = useState<SubsyncPrefs>(loadPrefs);
  const [pools, poolDispatch] = useReducer(poolsReducer, undefined, emptyPools);
  const [run, runDispatch] = useReducer(runReducer, initialRunState);
  const [packs, packDispatch] = useReducer(packsReducer, {});
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [engines, setEngines] = useState<{ open: boolean; focus: string | null }>({ open: false, focus: null });
  const [preview, setPreview] = useState<PreviewState | null>(null);
  const [showRun, setShowRun] = useState(false);
  const [hovering, setHovering] = useState(false);

  const task = prefs.task;
  const tool = toolOf(task);
  const busy = run.status === "running";
  const pool = pools[task];
  const options = prefs.options[task];
  const output = prefs.output[task];

  useEffect(() => savePrefs(prefs), [prefs]);
  useEffect(() => onBusyChange(busy), [busy, onBusyChange]);
  useEffect(() => () => onBusyChange(false), [onBusyChange]);

  // Read through refs so the engine subscription is made once.
  const runRef = useRef(run);
  runRef.current = run;
  const taskRef = useRef(task);
  taskRef.current = task;
  const onLogRef = useRef(onLog);
  onLogRef.current = onLog;
  /** The Convert / Extract / Mux entry reopens on the one used last. */
  const formatsTask = useRef<SubTask>(toolOf(task) === "formats" ? task : "convert");

  const refreshCaps = useCallback(() => {
    void subsApi
      .capabilities()
      .then(setCaps)
      .catch(() => undefined);
  }, []);
  useEffect(refreshCaps, [refreshCaps]);

  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;
    void subsApi
      .subscribe((event) => {
        runDispatch({ type: "event", event });
        packDispatch({ type: "event", event });
        if (event.type === "subsJobLog") {
          const label = runRef.current.jobs[event.job]?.label ?? `job ${event.job + 1}`;
          onLogRef.current(`${label}: ${event.message}`);
        } else if (event.type === "log") {
          onLogRef.current(event.message);
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

  /** Add files to a task and probe them. */
  const addFiles = useCallback((target: SubTask, files: subsApi.PickedFile[]) => {
    if (files.length === 0) return;
    poolDispatch({ type: "add", task: target, files });
    const paths = files.map((file) => file.path);
    subsApi
      .probe(paths)
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
        poolDispatch({
          type: "probed",
          files: files.map((file) => ({ ...pendingFile(file), error: message })),
        });
      });
  }, []);

  // Native drops land on the tool that is open.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;
    subscribeToFileDrop(
      (paths) => {
        subsApi
          .resolveDropped(paths)
          .then((files) => {
            if (files.length === 0) toast.error("No video or subtitle files in that drop.");
            else addFiles(taskRef.current, files);
          })
          .catch(() => toast.error("Could not read the dropped files."));
      },
      setHovering,
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

  const items = useMemo(() => buildItems(task, pool, options), [task, pool, options]);
  const runnable = items.filter((item) => !item.problem).length;

  const setTask = useCallback((next: SubTask) => {
    if (toolOf(next) === "formats") formatsTask.current = next;
    setPrefs((current) => ({ ...current, task: next }));
    setShowRun(false);
  }, []);

  const selectTool = useCallback(
    (id: ToolId) => setTask(id === "formats" ? formatsTask.current : (TOOLS.find((entry) => entry.id === id)?.tasks[0] ?? "sync")),
    [setTask],
  );

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
      setPrefs((current) => ({
        ...current,
        output: { ...current.output, [task]: { ...current.output[task], ...patch } },
      })),
    [task],
  );

  const browse = async (folder: boolean) => {
    try {
      if (folder) {
        const path = await subsApi.pickFolder();
        if (path) addFiles(task, await subsApi.resolveDropped([path]));
      } else {
        const response = await subsApi.pickFiles(inputShape(task) === "video" ? "video" : "any");
        addFiles(task, response.files);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not open the picker");
    }
  };

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

  const handleRun = async () => {
    const stamp = Date.now().toString(36);
    const { jobs, labels } = buildJobs(task, items, options, output, (index) => `${task}-${stamp}-${index}`);
    if (jobs.length === 0) {
      toast.error("Nothing in the queue can run yet.");
      return;
    }
    runDispatch({
      type: "started",
      task,
      at: Date.now(),
      jobs: jobs.map((job, index) => ({ id: job.id, label: labels[index] })),
    });
    setShowRun(true);
    onLog(`Subsync: ${TASK_LABELS[task]}, ${jobs.length} job${jobs.length === 1 ? "" : "s"}`);
    try {
      const result = await subsApi.startBatch(jobs, workersFor(task));
      runDispatch({ type: "finished", outcomes: result.outcomes ?? [], cancelled: result.cancelled, error: result.error });
    } catch (error) {
      runDispatch({ type: "failed", message: error instanceof Error ? error.message : String(error) });
    }
  };

  const handleStop = () => {
    void subsApi.cancel().catch(() => toast.error("Could not stop the run."));
  };

  const openEngines = useCallback((focus: string | null = null) => setEngines({ open: true, focus }), []);

  const handlePack = async (action: "install" | "remove", pack: string, model?: string) => {
    packDispatch({ type: "begin", pack, action });
    try {
      const result = await subsApi.pack(action, pack, model);
      packDispatch({ type: "done", pack, ok: result.ok, error: result.error, status: result.status });
      if (!result.ok) toast.error(result.error ?? "That did not finish.");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      packDispatch({ type: "done", pack, ok: false, error: message });
      toast.error(message);
    } finally {
      refreshCaps();
    }
  };

  const openPreview = async (ref: MediaRef, name: string) => {
    setPreview({ name, loading: true, preview: null, error: null });
    try {
      const result = await subsApi.loadCues(ref);
      setPreview({ name, loading: false, preview: result.preview, error: result.error ?? null });
    } catch (error) {
      setPreview({ name, loading: false, preview: null, error: error instanceof Error ? error.message : String(error) });
    }
  };

  const outcomeActions = useMemo(
    () => ({
      onReveal: (path: string) => void subsApi.revealPath(path).catch(() => toast.error("That file no longer exists.")),
      onOpen: (path: string) => void subsApi.openPath(path).catch(() => toast.error("Could not open the file.")),
      onUseAs: (file: TaskOutputFile, target: SubTask) => {
        addFiles(target, [
          { name: file.path.replace(/^.*[\\/]/, ""), path: file.path, type: file.kind === "video" ? "video" : "subtitle", size: null },
        ]);
        setTask(target);
        toast.success(`Added to ${TASK_LABELS[target]}`);
      },
    }),
    [addFiles, setTask],
  );

  const runHere = run.status !== "idle" && run.task === task;
  const runElsewhere = busy && run.task !== task;
  const toolInfo = TOOLS.find((entry) => entry.id === tool)!;
  const runLabel =
    task === "mux" ? "Mux" : runnable > 0 ? `Run ${runnable} job${runnable === 1 ? "" : "s"}` : "Run";

  return (
    <div className="flex min-h-0 min-w-0 flex-1">
      <ToolRail tool={tool} runningTool={busy && run.task ? toolOf(run.task) : null} onSelect={selectTool} />

      <aside aria-label={`${toolInfo.label} settings`} className="flex w-[330px] shrink-0 flex-col border-r border-border">
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-[18px]">
          {tool === "formats" && <FormatsSwitch task={task} disabled={false} onChange={setTask} />}
          <section aria-label="Engine and settings">
            <SectionTitle>Engine & settings</SectionTitle>
            <TaskSettings
              key={task}
              task={task}
              options={options}
              onChange={setOptions}
              caps={caps}
              disabled={runHere && busy}
              onInstallPack={(pack) => openEngines(pack)}
              pickFile={pickReference}
              videoFps={firstVideoFps(pool)}
            />
          </section>
          <hr className="my-5 border-border" />
          <OutputSection task={task} output={output} disabled={runHere && busy} onChange={setOutput} onChooseFolder={() => void chooseOutputFolder()} />
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center gap-2 border-b border-border px-[18px]">
          <h1 className="min-w-0 flex-1 truncate text-[13px] font-semibold">
            {tool === "formats" ? TASK_LABELS[task] : toolInfo.label}
            <span className="ml-2 text-[11.5px] font-normal text-muted-foreground">
              {pool.files.length > 0 && `${pool.files.length} file${pool.files.length === 1 ? "" : "s"}`}
            </span>
          </h1>
          <Button size="sm" variant="ghost" disabled={runHere && busy} onClick={() => void browse(false)}>
            <FilePlus className="h-3.5 w-3.5" aria-hidden />
            Add files
          </Button>
          <Button size="sm" variant="ghost" disabled={runHere && busy} onClick={() => void browse(true)}>
            <FolderPlus className="h-3.5 w-3.5" aria-hidden />
            Add folder
          </Button>
          <IconButton label="Engines and API keys" onClick={() => openEngines(null)}>
            <Cpu className="h-4 w-4" aria-hidden />
          </IconButton>
          {busy && run.task === task ? (
            <Button size="sm" onClick={handleStop}>
              <Square className="h-3 w-3 fill-current" aria-hidden />
              Stop
            </Button>
          ) : (
            <Button
              size="sm"
              variant="primary"
              disabled={runElsewhere || runnable === 0}
              title={runElsewhere ? `${TASK_LABELS[run.task!]} is running` : runnable === 0 ? "Add inputs that can run" : undefined}
              onClick={() => void handleRun()}
            >
              <Play className="h-3 w-3 fill-current" aria-hidden />
              {runLabel}
            </Button>
          )}
        </div>

        <div
          className={cx(
            "min-h-0 flex-1 overflow-y-auto",
            hovering && !busy && "bg-primary/[0.03] ring-2 ring-inset ring-primary/40",
          )}
        >
          {runHere && (showRun || busy) ? (
            <>
              <RunView run={run} actions={outcomeActions} onOpenConsole={onOpenConsole} />
              {!busy && (
                <div className="px-[18px] py-3">
                  <LinkButton tone="primary" onClick={() => setShowRun(false)}>
                    Back to the queue
                  </LinkButton>
                </div>
              )}
            </>
          ) : (
            <>
              {runHere && (
                <div className="flex items-center gap-3 border-b border-border px-[18px] py-2 text-[11.5px] text-muted-foreground">
                  <span className="flex-1">The last run finished.</span>
                  <LinkButton tone="primary" onClick={() => setShowRun(true)}>
                    Show its results
                  </LinkButton>
                </div>
              )}
              {pool.files.length === 0 ? (
                <EmptyState title={emptyTitle(task)} body={emptyBody(task)} />
              ) : (
                <>
                  <div className="flex items-center justify-end gap-3 px-[18px] pt-2">
                    <span className="flex-1 text-[11px] text-muted-foreground">
                      {runnable} of {items.length} ready
                    </span>
                    <LinkButton disabled={runHere && busy} onClick={() => poolDispatch({ type: "clear", task })}>
                      Clear
                    </LinkButton>
                  </div>
                  <QueueList
                    task={task}
                    pool={pool}
                    items={items}
                    options={options}
                    disabled={runHere && busy}
                    dispatch={poolDispatch}
                    onPreview={(ref, name) => void openPreview(ref, name)}
                  />
                </>
              )}
            </>
          )}
        </div>
      </main>

      <EnginesPanel
        open={engines.open}
        focus={engines.focus}
        caps={caps}
        packs={packs}
        busy={busy}
        onPack={(action, pack, model) => void handlePack(action, pack, model)}
        onRefresh={refreshCaps}
        onClose={() => setEngines({ open: false, focus: null })}
      />

      <Dialog
        open={preview !== null}
        onClose={() => setPreview(null)}
        title={preview?.name ?? "Cues"}
        description={preview?.preview ? [preview.preview.format?.toUpperCase(), preview.preview.language].filter(Boolean).join(" · ") : undefined}
        className="max-w-3xl"
      >
        <div className="py-3">
          {preview?.loading ? (
            <p className="flex items-center gap-2 text-[12px] text-muted-foreground">
              <Spinner className="h-3.5 w-3.5" /> Reading the subtitle…
            </p>
          ) : preview?.error ? (
            <p className="text-[12px] text-destructive">{preview.error}</p>
          ) : preview?.preview ? (
            <CueTable cues={preview.preview.cues} total={preview.preview.count} height={420} label={preview.name} />
          ) : (
            <p className="text-[12px] text-muted-foreground">No cues.</p>
          )}
        </div>
      </Dialog>
    </div>
  );
}

function FormatsSwitch({ task, disabled, onChange }: { task: SubTask; disabled: boolean; onChange: (task: SubTask) => void }) {
  const options: SubTask[] = ["convert", "extract", "mux"];
  return (
    <div role="radiogroup" aria-label="What to do" className="mb-5 grid grid-cols-3 gap-1 rounded-[9px] bg-elevated p-1">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          role="radio"
          aria-checked={task === option}
          disabled={disabled}
          onClick={() => onChange(option)}
          className={cx(
            "rounded-[7px] px-3 py-1.5 text-[12px] font-medium transition-colors disabled:opacity-50",
            task === option ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {TASK_LABELS[option]}
        </button>
      ))}
    </div>
  );
}

function emptyTitle(task: SubTask): string {
  switch (task) {
    case "sync":
      return "Sync subtitles to their videos";
    case "generate":
      return "Make subtitles from speech";
    case "tonemap":
      return "Tone-map HDR video to SDR";
    case "extract":
      return "Pull subtitle tracks out of videos";
    case "mux":
      return "Add subtitles to a video";
    default:
      return `Add subtitles to ${TASK_LABELS[task].toLowerCase()}`;
  }
}

function emptyBody(task: SubTask): string {
  switch (task) {
    case "sync":
      return "Drop subtitle files and the videos they belong to, or click Add files. Movie.en.srt goes with Movie.mkv by name; a video with its own subtitle track can be synced on its own.";
    case "ocr":
      return "Drop Blu-ray .sup or DVD .idx files, or videos with image subtitle tracks.";
    case "generate":
    case "tonemap":
      return "Drop the videos, or click Add files. They run one at a time.";
    case "extract":
      return "Drop the videos; the tracks chosen in the settings are written beside each one.";
    case "mux":
      return "Drop one video and the subtitle files to add. Each gets its own language, title and flags.";
    default:
      return "Drop subtitle files, or videos with subtitle tracks, or click Add files. Add the videos too to write a copy with the result muxed in.";
  }
}
