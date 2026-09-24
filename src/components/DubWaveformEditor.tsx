/** The dub sync editor: both tracks as waveforms, the cuts as handles.
 *
 *  The drawing is DubWaveformView's; this owns the plan and everything
 *  that changes it. When the plan is right, the transients in the two
 *  lanes sit under each other; where they do not, the user drags the cut
 *  to where the scene really changes, or nudges a stretch by a frame
 *  until they do -- or slides it with Alt/Option-drag -- and applies. Everything the user can do here is a pure
 *  function on the plan (see dubPlanEdit.ts); this file only turns
 *  pointer and key events into those functions.
 */

import { ChevronsLeft, ChevronsRight, ExternalLink, Loader2, Music, Redo2, Undo2, Video, ZoomIn, ZoomOut } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DubPreviewPlayer } from "@/components/DubPreviewPlayer";
import { DubWaveformView, type DubWaveformViewHandle } from "@/components/DubWaveformView";
import { Button, Spinner, Tag } from "@/components/ui";
import { cx } from "@/lib/cx";
import type { DubExcerpt, DubPreviewRequest, WaveformPeaks, WaveformRequest } from "@/lib/api";
import {
  canMerge,
  convertToDub,
  convertToFill,
  frameSeconds,
  mergeWithNext,
  moveBoundary,
  nudgeOffset,
  samePlan,
  segmentAt,
  setOffset,
  splitAt,
  validatePlan,
} from "@/lib/dubPlanEdit";
import { windowAround } from "@/lib/previewClock";
import type { MixMode } from "@/lib/previewMix";
import { useExcerpt } from "@/lib/useExcerpt";
import { useShotCuts, type FetchShotCuts } from "@/lib/useShotCuts";
import { formatClock, formatSpan, type DubSegment, type DubSyncPlan } from "@/lib/types";
import type { View } from "@/lib/waveformView";

export interface DubWaveformEditorProps {
  /** The plan the track was written from, the starting point. */
  plan: DubSyncPlan;
  /** The engine's own plan, when the track has already been written from
   *  edited cuts once; "back to the engine's plan" goes here. */
  enginePlan?: DubSyncPlan | null;
  /** Fetches peaks; injected so the editor can be rendered without the engine. */
  fetchPeaks: (request: WaveformRequest) => Promise<WaveformPeaks>;
  /** Finds the video's picture cuts in a span, to mark on the ruler and
   *  snap dragged cuts to; without it there are neither. */
  fetchShotCuts?: FetchShotCuts;
  /** Renders a span of the edited plan and plays it; resolves when playback was handed off. */
  onPreview: (plan: DubSyncPlan, startS: number, endS: number) => Promise<void>;
  /** Renders a span of the edited plan for the in-app player. */
  renderDubPreview: (request: DubPreviewRequest) => Promise<DubExcerpt | null>;
  /** Bytes of a rendered excerpt, for the in-app player. */
  readPreviewBytes: (path: string) => Promise<ArrayBuffer>;
  /** Writes the track from the edited plan. */
  onApply: (plan: DubSyncPlan) => Promise<void>;
  /** Stops the write in progress; the file that was there is kept. */
  onStop?: () => void;
  onClose: () => void;
  /** A render is in progress: applying is disabled, progress is shown. */
  busy?: { percent: number; stage: string | null } | null;
  /** Files the engine is reading for the first time: path to percent. */
  reading?: Record<string, number>;
}

export function DubWaveformEditor({
  plan: initialPlan,
  enginePlan = null,
  fetchPeaks,
  fetchShotCuts,
  onPreview,
  renderDubPreview,
  readPreviewBytes,
  onApply,
  onStop,
  onClose,
  busy = null,
  reading,
}: DubWaveformEditorProps) {
  const duration = initialPlan.videoDurationS;
  const [plan, setPlan] = useState<DubSyncPlan>(initialPlan);
  const [past, setPast] = useState<DubSyncPlan[]>([]);
  const [future, setFuture] = useState<DubSyncPlan[]>([]);
  const [view, setView] = useState<View>({ startS: 0, endS: duration });
  const [selected, setSelected] = useState<number | null>(null);
  const [cursorS, setCursorS] = useState<number>(0);
  const [previewing, setPreviewing] = useState(false);
  const [offsetText, setOffsetText] = useState("");
  // The in-app player: which mode opened it, the window it is playing,
  // the mix it is playing, and the AudioContext made in the button press
  // that opened it (the user gesture Web Audio asks for).
  const [player, setPlayer] = useState<{ mode: "video" | "sample" } | null>(null);
  const [playerWindow, setPlayerWindow] = useState<{ startS: number; endS: number } | null>(null);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [mix, setMix] = useState<MixMode>("dub");

  const viewRef = useRef<DubWaveformViewHandle>(null);
  const planRef = useRef(plan);
  planRef.current = plan;
  const playerRef = useRef(player);
  playerRef.current = player;
  // A slip in progress: the plan it started from, and the plan as slipped
  // so far. The slipped plan is only drawn -- and shown in the offset box
  // -- until the pointer lets go; then it is committed once, as one step
  // to undo, and the player re-cuts its sound once rather than on every
  // pixel of the drag.
  const [slipPlan, setSlipPlan] = useState<DubSyncPlan | null>(null);
  const slipRef = useRef<{ index: number; base: DubSyncPlan; next: DubSyncPlan } | null>(null);
  const shownPlan = slipPlan ?? plan;

  const shotCuts = useShotCuts(initialPlan.videoPath, duration, view, fetchShotCuts);

  const frame = frameSeconds(plan);
  const problems = useMemo(() => validatePlan(plan), [plan]);
  const changed = !samePlan(plan, initialPlan);
  const original = enginePlan ?? initialPlan;
  const isEngines = samePlan(plan, original);
  const segment = selected !== null ? shownPlan.segments[selected] : undefined;

  // ----------------------------------------------------------------- history

  const commit = useCallback((next: DubSyncPlan) => {
    setPlan((current) => {
      if (next === current) return current;
      setPast((p) => [...p.slice(-99), current]);
      setFuture([]);
      return next;
    });
  }, []);

  const undo = useCallback(() => {
    setPast((p) => {
      if (p.length === 0) return p;
      const previous = p[p.length - 1];
      setPlan((current) => {
        setFuture((f) => [current, ...f]);
        return previous;
      });
      return p.slice(0, -1);
    });
  }, []);

  const redo = useCallback(() => {
    setFuture((f) => {
      if (f.length === 0) return f;
      const next = f[0];
      setPlan((current) => {
        setPast((p) => [...p, current]);
        return next;
      });
      return f.slice(1);
    });
  }, []);

  // A drag pushes the pre-drag plan once, at its start; if nothing moved,
  // the entry is popped again at its end.
  const onBoundaryDragStart = useCallback(() => {
    setPast((p) => [...p.slice(-99), planRef.current]);
    setFuture([]);
  }, []);
  const onBoundaryDrag = useCallback((boundary: number, timeS: number) => {
    setPlan((current) => moveBoundary(current, boundary, timeS));
  }, []);
  const onBoundaryDragEnd = useCallback(() => {
    setPast((p) => (p.length && samePlan(p[p.length - 1], planRef.current) ? p.slice(0, -1) : p));
  }, []);
  // A slip is measured from where it started, so the offset follows the
  // pointer exactly and never accumulates rounding.
  const onSlipStart = useCallback((index: number) => {
    slipRef.current = { index, base: planRef.current, next: planRef.current };
  }, []);
  const onSlip = useCallback((index: number, offsetDeltaS: number) => {
    const slip = slipRef.current;
    if (!slip || slip.index !== index) return;
    slip.next = nudgeOffset(slip.base, index, offsetDeltaS);
    setSlipPlan(slip.next);
  }, []);
  const onSlipEnd = useCallback(() => {
    const slip = slipRef.current;
    slipRef.current = null;
    setSlipPlan(null);
    if (slip && !samePlan(slip.next, slip.base)) commit(slip.next);
  }, [commit]);
  const onSplitAt = useCallback(
    (t: number) => {
      commit(splitAt(planRef.current, t));
      setSelected(segmentAt(planRef.current, t));
    },
    [commit],
  );

  // ------------------------------------------------------------------ player

  const closePlayer = useCallback(() => {
    setPlayer(null);
    setPlayerWindow(null);
    setAudioContext((context) => {
      void context?.close().catch(() => undefined);
      return null;
    });
  }, []);

  // ------------------------------------------------------------------- keys

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      const meta = event.metaKey || event.ctrlKey;
      if (meta && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
        return;
      }
      if (event.key === "Escape") {
        event.stopPropagation();
        event.preventDefault();
        // The player claims the first Escape; a second closes the editor.
        if (playerRef.current !== null) {
          closePlayer();
          return;
        }
        if (!busy) onClose();
        return;
      }
      // Nothing else while a stretch is being slipped: the slip commits
      // from the plan it started on, and would undo an edit made meanwhile.
      if (meta || slipRef.current) return;
      if (selected !== null && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
        event.preventDefault();
        const unit = event.shiftKey ? 0.01 : frame;
        commit(nudgeOffset(planRef.current, selected, event.key === "ArrowLeft" ? -unit : unit));
        return;
      }
      // A millisecond at a time on , and . -- the frame-step keys of every
      // editor; Alt+arrow is a browser gesture on some platforms.
      if (selected !== null && (event.key === "," || event.key === ".")) {
        event.preventDefault();
        commit(nudgeOffset(planRef.current, selected, event.key === "," ? -0.001 : 0.001));
        return;
      }
      if (event.key.toLowerCase() === "s") {
        commit(splitAt(planRef.current, cursorS));
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [selected, frame, cursorS, commit, undo, redo, onClose, busy, closePlayer]);

  useEffect(() => {
    setOffsetText(segment && segment.kind === "dub" ? (segment.offsetS ?? 0).toFixed(3) : "");
  }, [segment]);

  // ---------------------------------------------------------------- actions

  const nudge = (deltaS: number) => {
    if (selected === null) return;
    commit(nudgeOffset(plan, selected, deltaS));
  };

  const applyOffsetText = () => {
    if (selected === null) return;
    const value = Number(offsetText);
    if (Number.isFinite(value)) commit(setOffset(plan, selected, value));
  };

  /** The in-app player: create the AudioContext here, in the button press
   *  that opened it, and resume it in the same gesture. Web Audio refuses
   *  to run from a context made anywhere else. */
  const openPlayer = (mode: "video" | "sample") => {
    if (busy) return;
    const context = audioContext ?? new AudioContext();
    void context.resume().catch(() => undefined);
    setAudioContext(context);
    setPlayerWindow(windowAround(cursorS, duration));
    setPlayer({ mode });
  };

  /** Writing closes the player first: the render would swap the file under
   *  it, and stopping the sound keeps the transition simple. */
  const apply = () => {
    if (player) closePlayer();
    void onApply(plan);
  };

  const preview = async () => {
    const start = Math.max(0, cursorS - 4);
    const end = Math.min(duration, start + 12);
    setPreviewing(true);
    try {
      await onPreview(plan, start, end);
    } finally {
      setPreviewing(false);
    }
  };

  // The player's loaded window, for the plan as edited: every edit
  // re-cuts the sound only, after the loader's debounce.
  const sources = useExcerpt({
    plan,
    wantedStartS: playerWindow?.startS ?? null,
    wantedEndS: playerWindow?.endS ?? null,
    withPicture: player?.mode === "video",
    wantOriginal: mix !== "dub",
    audioContext,
    render: renderDubPreview,
    read: readPreviewBytes,
  });

  const showSegment = (index: number) => {
    setSelected(index);
    const s = plan.segments[index];
    setCursorS(s.startS);
    if (s.startS < view.startS || s.endS > view.endS) viewRef.current?.show(s.startS, s.endS);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-background"
      role="dialog"
      aria-modal="true"
      aria-label="Edit the cuts"
    >
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border px-4">
        <h2 className="text-[13px] font-semibold">Edit the cuts</h2>
        <span className="min-w-0 flex-1 truncate text-[11.5px] text-muted-foreground">
          Drag to scroll, Ctrl/⌘-wheel or pinch to zoom. Drag a cut to where the scene really changes (it snaps to the picture's cuts, ticked on the ruler; hold Alt/⌥ not to); Alt/⌥-drag a stretch
          to slide it under the original, or select it and use ← → to move it a frame (Shift: 10 ms; , and . : 1 ms).
          Double-click or S splits.
        </span>
        <Button size="sm" variant="ghost" onClick={onClose} disabled={!!busy}>
          Close
        </Button>
      </header>

      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <Button size="sm" variant="ghost" onClick={undo} disabled={past.length === 0} aria-label="Undo">
          <Undo2 className="h-3.5 w-3.5" aria-hidden /> Undo
        </Button>
        <Button size="sm" variant="ghost" onClick={redo} disabled={future.length === 0} aria-label="Redo">
          <Redo2 className="h-3.5 w-3.5" aria-hidden /> Redo
        </Button>
        <span className="mx-1 h-5 w-px bg-border" aria-hidden />
        <Button size="sm" variant="ghost" onClick={() => viewRef.current?.zoomBy(0.5)} aria-label="Zoom in">
          <ZoomIn className="h-3.5 w-3.5" aria-hidden />
        </Button>
        <Button size="sm" variant="ghost" onClick={() => viewRef.current?.zoomBy(2)} aria-label="Zoom out">
          <ZoomOut className="h-3.5 w-3.5" aria-hidden />
        </Button>
        <Button size="sm" variant="ghost" onClick={() => viewRef.current?.fit()}>
          Whole film
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => segment && viewRef.current?.show(segment.startS, segment.endS)}
          disabled={!segment}
        >
          Zoom to stretch
        </Button>
        <span className="tabular font-mono text-[11px] text-muted-foreground">
          {formatClock(view.startS)} – {formatClock(view.endS)} · {formatSpan(view.endS - view.startS)} shown
        </span>
        <span className="flex-1" />
        <Button size="sm" variant="default" onClick={() => openPlayer("video")} disabled={!!busy}>
          <Video className="h-3.5 w-3.5" aria-hidden /> Play video
        </Button>
        <Button size="sm" variant="default" onClick={() => openPlayer("sample")} disabled={!!busy}>
          <Music className="h-3.5 w-3.5" aria-hidden /> Play sample
        </Button>
        <Button size="sm" variant="ghost" onClick={() => void preview()} disabled={previewing || !!busy}>
          {previewing ? <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> : <ExternalLink className="h-3.5 w-3.5" aria-hidden />}
          Open in player
        </Button>
      </div>

      <div className="shrink-0 px-4 pt-3">
        <DubWaveformView
          ref={viewRef}
          plan={shownPlan}
          fetchPeaks={fetchPeaks}
          editable
          selected={selected}
          onSelect={setSelected}
          cursorS={cursorS}
          onCursor={setCursorS}
          keepCursorInView={player !== null}
          onBoundaryDragStart={onBoundaryDragStart}
          onBoundaryDrag={onBoundaryDrag}
          onBoundaryDragEnd={onBoundaryDragEnd}
          onSplitAt={onSplitAt}
          onSlipStart={onSlipStart}
          onSlip={onSlip}
          onSlipEnd={onSlipEnd}
          onViewChange={setView}
          shotCuts={shotCuts}
          laneHeight={110}
          reading={reading}
        />
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
        {segment ? (
          <>
            <Tag tone={segment.kind === "fill" ? "warning" : "neutral"}>{segment.kind === "fill" ? "Original" : "Dub"}</Tag>
            <span className="tabular font-mono text-[11.5px]">
              {formatClock(segment.startS)} – {formatClock(segment.endS)}
              <span className="ml-1.5 text-muted-foreground">({formatSpan(segment.endS - segment.startS)})</span>
            </span>
            {segment.kind === "dub" && (
              <>
                <span className="ml-2 text-[11.5px] text-muted-foreground">Offset</span>
                <Button size="sm" variant="ghost" onClick={() => nudge(-frame)} aria-label="Earlier by one frame">
                  <ChevronsLeft className="h-3.5 w-3.5" aria-hidden /> 1 frame
                </Button>
                <Button size="sm" variant="ghost" onClick={() => nudge(-0.01)} aria-label="Earlier by 10 ms">
                  −10 ms
                </Button>
                <Button size="sm" variant="ghost" onClick={() => nudge(-0.001)} aria-label="Earlier by 1 ms">
                  −1 ms
                </Button>
                <input
                  type="text"
                  inputMode="decimal"
                  value={offsetText}
                  onChange={(event) => setOffsetText(event.target.value)}
                  onBlur={applyOffsetText}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      applyOffsetText();
                      (event.target as HTMLInputElement).blur();
                    }
                  }}
                  aria-label="Offset in seconds"
                  className="w-[92px] rounded-md border border-border-strong bg-input px-2 py-1 text-right font-mono text-xs focus:outline-none focus:ring-2 focus:ring-ring/40"
                />
                <span className="text-[11px] text-muted-foreground">s</span>
                <Button size="sm" variant="ghost" onClick={() => nudge(0.001)} aria-label="Later by 1 ms">
                  +1 ms
                </Button>
                <Button size="sm" variant="ghost" onClick={() => nudge(0.01)} aria-label="Later by 10 ms">
                  +10 ms
                </Button>
                <Button size="sm" variant="ghost" onClick={() => nudge(frame)} aria-label="Later by one frame">
                  1 frame <ChevronsRight className="h-3.5 w-3.5" aria-hidden />
                </Button>
              </>
            )}
            <span className="mx-1 h-5 w-px bg-border" aria-hidden />
            <Button size="sm" variant="ghost" onClick={() => commit(splitAt(plan, cursorS))} disabled={segmentAt(plan, cursorS) === null}>
              Split at cursor
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => selected !== null && commit(mergeWithNext(plan, selected))}
              disabled={selected === null || !canMerge(plan.segments[selected], plan.segments[selected + 1])}
            >
              Merge with next
            </Button>
            {segment.kind === "dub" ? (
              <Button size="sm" variant="ghost" onClick={() => selected !== null && commit(convertToFill(plan, selected))}>
                Use the original here
              </Button>
            ) : (
              <Button size="sm" variant="ghost" onClick={() => selected !== null && commit(convertToDub(plan, selected))}>
                Use the dub here
              </Button>
            )}
            {segment.note && (
              <span className="min-w-0 truncate text-[11px] text-muted-foreground" title={segment.note}>
                {segment.note}
              </span>
            )}
          </>
        ) : (
          <span className="text-[11.5px] text-muted-foreground">
            Click a stretch to select it. {plan.segments.filter((s) => s.kind === "dub").length} stretches of dub,{" "}
            {formatSpan(plan.filledS)} from the original.
          </span>
        )}
      </div>

      <div className="flex min-h-0 flex-1">
        {player && audioContext && (
          <div className="w-[46%] min-w-[400px] shrink-0 overflow-y-auto border-r border-border px-4 py-3">
            <DubPreviewPlayer
              frameS={frame}
              cursorS={cursorS}
              onCursor={setCursorS}
              withPicture={player.mode === "video"}
              sources={sources}
              onNeedWindow={(aroundS) => setPlayerWindow(windowAround(aroundS, duration))}
              onClose={closePlayer}
              audioContext={audioContext}
              mix={mix}
              onMix={setMix}
            />
          </div>
        )}
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 text-[11.5px] text-muted-foreground">
          {problems.length > 0 && (
            <ul className="mb-3 space-y-1 text-destructive">
              {problems.map((problem) => (
                <li key={problem}>{problem}</li>
              ))}
            </ul>
          )}
          <SegmentList plan={plan} selected={selected} onSelect={showSegment} />
        </div>
      </div>

      <footer className="flex shrink-0 items-center gap-3 border-t border-border bg-elevated px-4 py-3">
        {busy ? (
          <>
            <span className="flex min-w-0 flex-1 items-center gap-2 text-[12px] text-muted-foreground">
              <Spinner className="h-3.5 w-3.5 border-[1.8px]" />
              <span className="truncate">{busy.stage ?? "Writing the track"}</span>
              <span className="tabular font-mono">{busy.percent}%</span>
            </span>
            {onStop && (
              <Button size="sm" variant="ghost" onClick={onStop}>
                Stop
              </Button>
            )}
          </>
        ) : (
          <span className="min-w-0 flex-1 truncate text-[12px] text-muted-foreground">
            {changed ? "The track will be written again with these cuts, and checked against the video." : "No changes yet."}
          </span>
        )}
        <Button size="sm" variant="ghost" onClick={() => commit(original)} disabled={isEngines || !!busy}>
          Back to the engine's plan
        </Button>
        <Button size="sm" variant="primary" onClick={apply} disabled={!changed || problems.length > 0 || !!busy}>
          Write the track with these cuts
        </Button>
      </footer>
    </div>
  );
}

function SegmentList({
  plan,
  selected,
  onSelect,
}: {
  plan: DubSyncPlan;
  selected: number | null;
  onSelect: (index: number) => void;
}) {
  return (
    <ol className="space-y-px" aria-label="Pieces of the plan">
      {plan.segments.map((s: DubSegment, index) => (
        <li key={index}>
          <button
            type="button"
            onClick={() => onSelect(index)}
            className={cx(
              "grid w-full grid-cols-[52px_190px_150px_90px_minmax(0,1fr)] items-center gap-3 rounded px-2 py-1 text-left font-mono text-[11px]",
              index === selected ? "bg-primary/10 text-foreground" : "hover:bg-secondary",
            )}
          >
            <Tag tone={s.kind === "fill" ? "warning" : "neutral"} className="font-medium">
              {s.kind === "fill" ? "Original" : "Dub"}
            </Tag>
            <span className="tabular">
              {formatClock(s.startS)} – {formatClock(s.endS)}
            </span>
            <span className="tabular text-muted-foreground">
              {s.kind === "fill" ? "original" : "dub"} {formatClock(s.sourceStartS)}
            </span>
            <span className="tabular text-right">
              {s.offsetS === null ? "—" : `${s.offsetS >= 0 ? "+" : ""}${s.offsetS.toFixed(3)}s`}
            </span>
            <span className="truncate text-muted-foreground">{s.note}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}
