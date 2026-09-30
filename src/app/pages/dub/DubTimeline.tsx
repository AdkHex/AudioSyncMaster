/** The Timeline: both tracks of a pair on the video's clock, the way an
 *  editor draws them.
 *
 *  The top lane is the video's own audio; the bottom lane is the synced
 *  track as the plan would write it -- each stretch of dub drawn from where
 *  the plan reads it, each stretch taken from the original hatched. When a
 *  stretch sits where it should, its peaks sit under the original's. The
 *  ruler, the lanes, the stretches, the grips, the playhead and the
 *  minimap are the mockup's own elements (ui.css); the canvases only draw
 *  the waveforms, each lane scaled to the loudest thing in sight.
 *
 *  One drawing serves every moment of a pair: as loaded, before any sync
 *  (the dub at the video's start); the analysis as it runs, redrawn from
 *  each draft the engine sends, the stretches not placed yet outlined; and
 *  the finished plan, which can be edited. Peaks come from the engine per
 *  visible span and are kept.
 *
 *  Getting around: drag to scroll, the wheel or two fingers to scroll,
 *  Ctrl/⌘-wheel or a pinch to zoom about the pointer, and the minimap to
 *  click or drag. A press that does not travel is a click: it places the
 *  playhead, and on the Dub lane selects the stretch. When editable, a cut
 *  is grabbed before anything else (it snaps to the picture's own cuts,
 *  ticked on the ruler, unless Alt/⌥ is held), Alt/⌥-dragging a stretch of
 *  dub slides it under the original, and a double-click splits. */

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";

import type { WaveformPeaks, WaveformRequest } from "@/lib/api";
import { segmentAt } from "@/lib/dubPlanEdit";
import { DRAFT_NOTE, formatClock, type DubSyncPlan } from "@/lib/types";
import { useShotCuts, type FetchShotCuts } from "@/lib/useShotCuts";
import {
  centreViewAt,
  clampView,
  followCursor,
  isDrag,
  MIN_VIEW_S,
  overviewBox,
  overviewDragView,
  overviewHit,
  overviewT,
  panView,
  peaksKey,
  slipOffsetDelta,
  snapToCuts,
  visibleRequests,
  wheelDeltaPx,
  wheelZoomFactor,
  zoomAround,
  type View,
} from "@/lib/waveformView";
import { cx } from "@/ui/kit";

import { useCursor, type CursorStore } from "./cursorStore";
import { rulerTicks } from "./dubModel";

/** How a pair is drawn: as loaded (the dub at the video's start), being
 *  placed (stretches not placed yet outlined), or as planned. */
export type TimelineMode = "loaded" | "placing" | "done";

export interface DubTimelineHandle {
  /** Zoom around the middle of the view; factors under 1 zoom in. */
  zoomBy(factor: number): void;
  /** The whole film. */
  fit(): void;
  /** Show a span, padded a little. */
  show(startS: number, endS: number): void;
}

export interface DubTimelineProps {
  plan: DubSyncPlan;
  mode: TimelineMode;
  fetchPeaks: (request: WaveformRequest) => Promise<WaveformPeaks>;
  /** Cuts can be dragged, stretches selected and slipped, double-click splits. */
  editable?: boolean;
  /** Draw the grips on the cuts. */
  grips?: boolean;
  selected?: number | null;
  onSelect?: (index: number | null) => void;
  cursor: CursorStore;
  /** While the player runs, keep the playhead on screen. */
  keepCursorInView?: boolean;
  onBoundaryDragStart?: (boundary: number) => void;
  onBoundaryDrag?: (boundary: number, timeS: number) => void;
  onBoundaryDragEnd?: () => void;
  onSplitAt?: (timeS: number) => void;
  /** Alt/⌥-drag on a stretch of dub: `onSlip` gets the change to its offset
   *  since the press, in seconds (dragging the sound right lowers it). */
  onSlipStart?: (index: number) => void;
  onSlip?: (index: number, offsetDeltaS: number) => void;
  onSlipEnd?: () => void;
  /** Finds the picture's cuts in the view, for the ruler and for snapping. */
  fetchShotCuts?: FetchShotCuts;
  /** The picture's cuts, when already known (overrides fetchShotCuts). */
  shotCuts?: readonly number[] | null;
  /** Files the engine is reading for the first time: path to percent. */
  reading?: Record<string, number>;
  /** A change of pair: the view goes back to the whole film. */
  viewKey?: string;
  onKeyDown?: (event: ReactKeyboardEvent<HTMLDivElement>) => void;
}

/** Pixels around a cut within which the pointer grabs it. */
const HANDLE_PX = 6;
/** The timeline's left padding and lane-label column (ui.css .tl, .lh). */
const PAD_PX = 14;
const LABEL_PX = 76;
/** The quietest a lane is ever scaled up from: -34 dBFS, so silence stays
 *  flat rather than blowing up into noise. */
const GAIN_FLOOR = 0.02;

type Drag =
  | { kind: "press"; startX: number; startY: number; startView: View; slip: number | null; inDub: boolean }
  | { kind: "pan"; startX: number; startView: View }
  | { kind: "boundary"; boundary: number }
  | { kind: "slip"; index: number; startX: number; startView: View }
  | { kind: "overview"; startX: number; startView: View };

/** WebKit's pinch (WKWebView, the Tauri window on macOS). */
type GestureEventLike = Event & { scale?: number; clientX?: number };

const loudest = (rows: (WaveformPeaks | null | undefined)[]) => {
  let peak = 0;
  for (const row of rows) {
    if (!row) continue;
    for (let ch = 0; ch < row.channels; ch++) {
      for (const v of row.max[ch] ?? []) if (v > peak) peak = v;
      for (const v of row.min[ch] ?? []) if (-v > peak) peak = -v;
    }
  }
  return peak;
};
const gainFor = (peak: number) => 1 / Math.max(peak, GAIN_FLOOR);

/** One track's min/max envelope across [x0, x1), every channel folded into
 *  one shape, as the mockup draws it. Silence stays a hairline. */
function drawEnvelope(ctx: CanvasRenderingContext2D, row: WaveformPeaks, x0: number, x1: number, top: number, height: number, gain: number, fill: string) {
  const n = row.buckets;
  if (n <= 0 || x1 <= x0) return;
  const step = (x1 - x0) / n;
  const mid = top + height / 2;
  const half = (height / 2) * 0.92;
  const channels = Math.max(1, row.channels);
  const hi: number[] = new Array(n);
  const lo: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let up = 0;
    let down = 0;
    for (let ch = 0; ch < channels; ch++) {
      const max = row.max[ch]?.[i] ?? 0;
      const min = row.min[ch]?.[i] ?? 0;
      if (max > up) up = max;
      if (min < down) down = min;
    }
    hi[i] = Math.max(0.6, Math.min(1, up * gain) * half);
    lo[i] = Math.max(0.6, Math.min(1, -down * gain) * half);
  }
  ctx.beginPath();
  ctx.moveTo(x0, mid - hi[0]);
  for (let i = 0; i < n; i++) ctx.lineTo(x0 + (i + 0.5) * step, mid - hi[i]);
  ctx.lineTo(x1, mid - hi[n - 1]);
  ctx.lineTo(x1, mid + lo[n - 1]);
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(x0 + (i + 0.5) * step, mid + lo[i]);
  ctx.lineTo(x0, mid + lo[0]);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

/** A canvas sized to its box at the screen's pixel ratio, cleared. */
function prepare(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const rect = canvas.getBoundingClientRect();
  const w = Math.round(rect.width);
  const h = Math.round(rect.height);
  if (w <= 0 || h <= 0) return null;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * ratio)) canvas.width = Math.round(w * ratio);
  if (canvas.height !== Math.round(h * ratio)) canvas.height = Math.round(h * ratio);
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/** The theme's wave colours, as the tokens hold them (hex or rgba), read
 *  as they are: an invalid fillStyle fails silently. */
function waveColours(element: Element | null): { a: string; b: string } {
  const styles = getComputedStyle(element ?? document.documentElement);
  return {
    a: styles.getPropertyValue("--wave-a").trim() || "rgba(255, 255, 255, 0.38)",
    b: styles.getPropertyValue("--wave-b").trim() || "#4cc2ff",
  };
}

export const DubTimeline = forwardRef<DubTimelineHandle, DubTimelineProps>(function DubTimeline(
  {
    plan,
    mode,
    fetchPeaks,
    editable = false,
    grips = false,
    selected = null,
    onSelect,
    cursor,
    keepCursorInView = false,
    onBoundaryDragStart,
    onBoundaryDrag,
    onBoundaryDragEnd,
    onSplitAt,
    onSlipStart,
    onSlip,
    onSlipEnd,
    fetchShotCuts,
    shotCuts: givenCuts,
    reading,
    viewKey,
    onKeyDown,
  },
  ref,
) {
  const duration = Math.max(MIN_VIEW_S, plan.videoDurationS);
  const [view, setViewState] = useState<View>({ startS: 0, endS: duration });
  const [width, setWidth] = useState(800);
  const [peaks, setPeaks] = useState<Map<string, WaveformPeaks | null>>(new Map());
  const [hoverBoundary, setHoverBoundary] = useState<number | null>(null);
  const [hoverSlip, setHoverSlip] = useState(false);
  const [hoverBox, setHoverBox] = useState(false);
  const [snappedShot, setSnappedShot] = useState<number | null>(null);
  const [drag, setDragState] = useState<Drag | null>(null);
  // Bumped when the theme changes, so the canvases pick up its colours.
  const [theme, setTheme] = useState(0);
  const [heights, setHeights] = useState("");

  const rootRef = useRef<HTMLDivElement>(null);
  const topLaneRef = useRef<HTMLDivElement>(null);
  const dubLaneRef = useRef<HTMLDivElement>(null);
  const topCanvasRef = useRef<HTMLCanvasElement>(null);
  const dubCanvasRef = useRef<HTMLCanvasElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const pendingRef = useRef<Set<string>>(new Set());
  const mountedRef = useRef(true);
  // The last rows drawn, drawn again where they belong while the view's
  // own rows are on their way, so a pan or a slip never blanks a lane.
  const lastTopRef = useRef<WaveformPeaks | null>(null);
  const lastDubRef = useRef<Map<number, WaveformPeaks>>(new Map());
  const viewRef = useRef(view);
  viewRef.current = view;
  // The pointer handlers read the gesture from a ref: several moves can
  // arrive before React renders the first one's update.
  const dragRef = useRef<Drag | null>(null);
  const setDrag = useCallback((next: Drag | null) => {
    dragRef.current = next;
    setDragState(next);
  }, []);

  const setView = useCallback(
    (next: View) => {
      const clamped = clampView(next, duration);
      viewRef.current = clamped;
      setViewState(clamped);
    },
    [duration],
  );

  // The native wheel and pinch listeners are attached once and read these.
  const widthRef = useRef(width);
  widthRef.current = width;
  const durationRef = useRef(duration);
  durationRef.current = duration;
  const setViewRef = useRef(setView);
  setViewRef.current = setView;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Another pair starts on its whole film; a film of another length keeps
  // the view where it still fits.
  const shownKey = useRef(viewKey);
  useEffect(() => {
    if (shownKey.current !== viewKey) {
      shownKey.current = viewKey;
      lastTopRef.current = null;
      lastDubRef.current = new Map();
      setViewState({ startS: 0, endS: duration });
      return;
    }
    setViewState((current) =>
      current.endS <= duration && current.startS < duration ? clampView(current, duration) : { startS: 0, endS: duration },
    );
  }, [duration, viewKey]);

  useImperativeHandle(
    ref,
    () => ({
      zoomBy(factor) {
        const current = viewRef.current;
        const span = current.endS - current.startS;
        const length = Math.min(duration, Math.max(MIN_VIEW_S, span * factor));
        // Around the playhead when it is in view, where it stays on screen;
        // around the middle otherwise.
        const head = cursor.get();
        const anchor = head !== null && head >= current.startS && head <= current.endS ? head : (current.startS + current.endS) / 2;
        const at = (anchor - current.startS) / span;
        setView({ startS: anchor - length * at, endS: anchor - length * at + length });
      },
      fit() {
        setView({ startS: 0, endS: duration });
      },
      show(startS, endS) {
        const pad = Math.max(1, (endS - startS) * 0.25);
        setView({ startS: startS - pad, endS: endS + pad });
      },
    }),
    [duration, setView, cursor],
  );

  // ------------------------------------------------------------------ size

  useEffect(() => {
    const top = topLaneRef.current;
    const bottom = dubLaneRef.current;
    if (!top || !bottom) return;
    const measure = () => {
      const w = Math.floor(top.getBoundingClientRect().width);
      if (w > 0) setWidth(w);
      setHeights(`${top.clientHeight}x${bottom.clientHeight}`);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(top);
    observer.observe(bottom);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (typeof MutationObserver === "undefined") return;
    const observer = new MutationObserver(() => setTheme((n) => n + 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  const span = view.endS - view.startS;
  const xOf = useCallback((t: number) => ((t - view.startS) / (view.endS - view.startS)) * width, [view, width]);
  const tOf = useCallback((x: number) => view.startS + (x / width) * (view.endS - view.startS), [view, width]);

  const fetchedCuts = useShotCuts(plan.videoPath, duration, view, editable ? fetchShotCuts : undefined);
  const shotCuts = givenCuts !== undefined ? givenCuts : fetchedCuts;

  // ----------------------------------------------------------------- peaks

  const requests = useMemo(() => visibleRequests(plan, view, width), [plan, view, width]);

  useEffect(() => {
    const missing = requests.filter((r) => {
      const key = peaksKey(r);
      return !peaks.has(key) && !pendingRef.current.has(key);
    });
    if (missing.length === 0) return;
    // Rows are keyed by what they show, so one that arrives after the view
    // moved on is still right and is kept.
    const timer = window.setTimeout(() => {
      missing.forEach((r) => pendingRef.current.add(peaksKey(r)));
      Promise.all(
        missing.map((r) =>
          fetchPeaks(r)
            .then((result) => [peaksKey(r), result] as const)
            .catch(() => [peaksKey(r), null] as const),
        ),
      )
        .then((results) => {
          if (!mountedRef.current) return;
          setPeaks((current) => {
            const next = new Map(current);
            for (const [key, values] of results) next.set(key, values);
            if (next.size > 400) {
              for (const key of Array.from(next.keys()).slice(0, next.size - 300)) next.delete(key);
            }
            return next;
          });
        })
        .finally(() => missing.forEach((r) => pendingRef.current.delete(peaksKey(r))));
    }, 30);
    return () => window.clearTimeout(timer);
  }, [requests, peaks, fetchPeaks]);

  // The visible stretches of dub, each with the row its request brings.
  const dubRows = useMemo(() => {
    const rows: { index: number; row: WaveformPeaks | null }[] = [];
    let k = 1;
    plan.segments.forEach((s, index) => {
      if (s.kind !== "dub" || Math.min(s.endS, view.endS) <= Math.max(s.startS, view.startS)) return;
      const request = requests[k++];
      rows.push({ index, row: request ? (peaks.get(peaksKey(request)) ?? null) : null });
    });
    return rows;
  }, [plan, view, requests, peaks]);
  const topRow = requests[0] ? (peaks.get(peaksKey(requests[0])) ?? null) : null;

  // ------------------------------------------------------------------ draw

  useEffect(() => {
    const canvas = topCanvasRef.current;
    if (!canvas) return;
    const prepared = prepare(canvas);
    if (!prepared) return;
    if (topRow) lastTopRef.current = topRow;
    const row = topRow ?? lastTopRef.current;
    if (!row || row.path !== plan.videoPath) return;
    const { ctx, h } = prepared;
    drawEnvelope(ctx, row, xOf(row.startS), xOf(row.endS), 0, h, gainFor(loudest([row])), waveColours(rootRef.current).a);
  }, [topRow, plan.videoPath, xOf, heights, theme]);

  useEffect(() => {
    const canvas = dubCanvasRef.current;
    if (!canvas) return;
    const prepared = prepare(canvas);
    if (!prepared) return;
    const { ctx, h } = prepared;
    const rows = dubRows.map(({ index, row }) => {
      if (row) lastDubRef.current.set(index, row);
      const shown = row ?? lastDubRef.current.get(index) ?? null;
      return { index, row: shown && shown.path === plan.dubPath ? shown : null };
    });
    const gain = gainFor(loudest(rows.map((r) => r.row)));
    // Inside a stretch (ui.css .segm: 6 px from the lane's edges), the wave
    // keeps 12% of the stretch's height clear above and below.
    const blockH = h - 12;
    const top = 6 + blockH * 0.12;
    const height = blockH * 0.76;
    const fill = waveColours(rootRef.current).b;
    ctx.globalAlpha = 0.85;
    for (const { index, row } of rows) {
      if (!row) continue;
      const s = plan.segments[index];
      const a = Math.max(s.startS, view.startS);
      const b = Math.min(s.endS, view.endS);
      // The row is on the dub's clock; the stretch plays it `offset` earlier.
      const offset = s.sourceStartS - s.startS;
      ctx.save();
      ctx.beginPath();
      ctx.rect(xOf(a), 0, xOf(b) - xOf(a), h);
      ctx.clip();
      drawEnvelope(ctx, row, xOf(row.startS - offset), xOf(row.endS - offset), top, height, gain, fill);
      ctx.restore();
    }
    ctx.globalAlpha = 1;
  }, [dubRows, plan, view, xOf, heights, theme]);

  // --------------------------------------------------------------- pointer

  const laneX = (event: { clientX: number }) => {
    const rect = topLaneRef.current?.getBoundingClientRect();
    return rect ? event.clientX - rect.left : 0;
  };
  const inDubLane = (event: { clientY: number }) => {
    const rect = dubLaneRef.current?.getBoundingClientRect();
    return rect ? event.clientY >= rect.top - 1 : false;
  };

  const boundaryNear = (x: number): number | null => {
    if (!editable) return null;
    let best: number | null = null;
    let bestDx = HANDLE_PX + 1;
    for (let i = 1; i < plan.segments.length; i++) {
      const dx = Math.abs(xOf(plan.segments[i].startS) - x);
      if (dx < bestDx) {
        bestDx = dx;
        best = i - 1;
      }
    }
    return best;
  };

  /** The stretch of dub an Alt-press at `x` would slip, if any. */
  const slipTarget = (x: number): number | null => {
    if (!editable || !onSlip) return null;
    const index = segmentAt(plan, tOf(x));
    return index !== null && plan.segments[index].kind === "dub" ? index : null;
  };

  const release = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.button !== 1) return;
    const x = laneX(event);
    const inDub = inDubLane(event);
    event.currentTarget.setPointerCapture?.(event.pointerId);
    if (event.button === 1 || event.shiftKey) {
      setDrag({ kind: "pan", startX: x, startView: view });
      return;
    }
    const boundary = inDub ? boundaryNear(x) : null;
    if (boundary !== null) {
      onBoundaryDragStart?.(boundary);
      setDrag({ kind: "boundary", boundary });
      onSelect?.(boundary + 1);
      return;
    }
    // Undecided until it moves or is released.
    setDrag({ kind: "press", startX: x, startY: event.clientY, startView: view, slip: event.altKey && inDub ? slipTarget(x) : null, inDub });
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const x = laneX(event);
    const current = dragRef.current;
    if (current === null) {
      const inDub = inDubLane(event);
      setHoverBoundary(inDub ? boundaryNear(x) : null);
      setHoverSlip(event.altKey && inDub && slipTarget(x) !== null);
      return;
    }
    switch (current.kind) {
      case "boundary": {
        const t = tOf(x);
        const snapped = shotCuts && !event.altKey ? snapToCuts(t, shotCuts, width / span) : t;
        setSnappedShot(snapped !== t ? snapped : null);
        onBoundaryDrag?.(current.boundary, snapped);
        return;
      }
      case "press": {
        if (!isDrag(x - current.startX, event.clientY - current.startY)) return;
        const delta = x - current.startX;
        if (current.slip !== null) {
          const index = current.slip;
          onSlipStart?.(index);
          onSelect?.(index);
          setDrag({ kind: "slip", index, startX: current.startX, startView: current.startView });
          onSlip?.(index, slipOffsetDelta(delta, current.startView, width));
        } else {
          setDrag({ kind: "pan", startX: current.startX, startView: current.startView });
          setView(panView(current.startView, delta, width));
        }
        return;
      }
      case "pan":
        setView(panView(current.startView, x - current.startX, width));
        return;
      case "slip":
        onSlip?.(current.index, slipOffsetDelta(x - current.startX, current.startView, width));
        return;
      default:
        return;
    }
  };

  const endGesture = (event: ReactPointerEvent<HTMLDivElement>, cancelled: boolean) => {
    release(event);
    const current = dragRef.current;
    setDrag(null);
    setSnappedShot(null);
    if (current === null) return;
    if (current.kind === "boundary") onBoundaryDragEnd?.();
    else if (current.kind === "slip") onSlipEnd?.();
    else if (current.kind === "press" && !cancelled) {
      // A click: the playhead goes there; on the Dub lane it picks the stretch.
      const t = Math.min(Math.max(tOf(current.startX), 0), duration);
      cursor.set(t);
      onSelect?.(editable && current.inDub ? segmentAt(plan, t) : null);
    }
  };

  const onDoubleClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (editable && inDubLane(event)) onSplitAt?.(tOf(laneX(event)));
  };

  // The minimap: a press on the box grabs it; anywhere else centres the view
  // there first and then grabs it.
  const minimapX = (event: { clientX: number }) => {
    const rect = minimapRef.current?.getBoundingClientRect();
    return rect ? event.clientX - rect.left : 0;
  };
  const onMinimapDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    const x = minimapX(event);
    event.currentTarget.setPointerCapture?.(event.pointerId);
    let start = view;
    if (!overviewHit(x, overviewBox(view, duration, width))) {
      start = centreViewAt(view, overviewT(x, duration, width), duration);
      setView(start);
    }
    setDrag({ kind: "overview", startX: x, startView: start });
  };
  const onMinimapMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const x = minimapX(event);
    const current = dragRef.current;
    if (current?.kind === "overview") setView(overviewDragView(current.startView, x - current.startX, duration, width));
    else if (current === null) setHoverBox(overviewHit(x, overviewBox(view, duration, width)));
  };
  const onMinimapUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    release(event);
    if (dragRef.current?.kind === "overview") setDrag(null);
  };

  // ---------------------------------------------------------- wheel, pinch

  // Native listeners: React's wheel listeners are passive, and the page
  // must not scroll under the timeline while it scrolls.
  useEffect(() => {
    const element = rootRef.current;
    if (!element) return;
    let pinch: { view: View; atS: number } | null = null;
    const anchorAt = (clientX: number | undefined) => {
      const current = viewRef.current;
      const lane = topLaneRef.current?.getBoundingClientRect();
      if (lane && clientX !== undefined && Number.isFinite(clientX) && clientX >= lane.left && clientX <= lane.right) {
        return current.startS + ((clientX - lane.left) / Math.max(1, widthRef.current)) * (current.endS - current.startS);
      }
      return (current.startS + current.endS) / 2;
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const current = viewRef.current;
      if (event.ctrlKey || event.metaKey) {
        if (pinch) return;
        setViewRef.current(zoomAround(current, anchorAt(event.clientX), wheelZoomFactor(event.deltaY, event.deltaMode), durationRef.current));
        return;
      }
      const delta = Math.abs(event.deltaX) >= Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
      setViewRef.current(panView(current, -wheelDeltaPx(delta, event.deltaMode) * 0.6, widthRef.current));
    };
    const onGestureStart = (event: Event) => {
      event.preventDefault();
      pinch = { view: viewRef.current, atS: anchorAt((event as GestureEventLike).clientX) };
    };
    const onGestureChange = (event: Event) => {
      event.preventDefault();
      const scale = (event as GestureEventLike).scale;
      if (!pinch || !scale || !Number.isFinite(scale) || scale <= 0) return;
      setViewRef.current(zoomAround(pinch.view, pinch.atS, 1 / scale, durationRef.current));
    };
    const onGestureEnd = (event: Event) => {
      event.preventDefault();
      pinch = null;
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    element.addEventListener("gesturestart", onGestureStart);
    element.addEventListener("gesturechange", onGestureChange);
    element.addEventListener("gestureend", onGestureEnd);
    return () => {
      element.removeEventListener("wheel", onWheel);
      element.removeEventListener("gesturestart", onGestureStart);
      element.removeEventListener("gesturechange", onGestureChange);
      element.removeEventListener("gestureend", onGestureEnd);
    };
  }, []);

  // ---------------------------------------------------------------- render

  const ticks = rulerTicks(view.startS, view.endS, width);
  const pointer =
    drag?.kind === "pan"
      ? "dub-grabbing"
      : drag?.kind === "slip" || (drag === null && hoverSlip)
        ? "dub-slip"
        : drag?.kind === "boundary" || (drag === null && hoverBoundary !== null)
          ? "dub-cut"
          : editable
            ? "dub-edit"
            : "dub-grab";
  const box = overviewBox(view, duration, width);
  const hotBoundary = drag?.kind === "boundary" ? drag.boundary : hoverBoundary;
  const readingTop = reading?.[plan.videoPath];
  const readingDub = reading?.[plan.dubPath];

  return (
    <div
      ref={rootRef}
      className="tl dub-tl"
      style={{ paddingTop: 2 }}
      tabIndex={0}
      role="group"
      aria-label="Timeline"
      onKeyDown={onKeyDown}
    >
      <div className="ruler">
        {ticks.map(({ t, label }) => (
          <span key={`l${t}`} style={{ left: xOf(t) }}>
            {label}
          </span>
        ))}
        {ticks.map(({ t }) => (
          <i key={`i${t}`} style={{ left: xOf(t) }} />
        ))}
        {shotCuts && shotCuts.length > 0 && (
          <div className="dub-shots" aria-hidden>
            {shotCuts.map((t) => {
              const x = Math.round(xOf(t));
              if (x < 0 || x > width) return null;
              return (
                <span
                  key={t}
                  data-shot-cut={t}
                  title={`Picture cut ${formatClock(t)}`}
                  className={cx("dub-shot", t === snappedShot && "on")}
                  style={{ left: x }}
                />
              );
            })}
          </div>
        )}
      </div>

      <div
        className={cx("dub-lanes", pointer)}
        role="img"
        aria-label="Waveforms of the original and the synced dub"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => endGesture(event, false)}
        onPointerCancel={(event) => endGesture(event, true)}
        onPointerLeave={() => {
          if (dragRef.current === null) {
            setHoverBoundary(null);
            setHoverSlip(false);
          }
        }}
        onDoubleClick={onDoubleClick}
      >
        <div className="lane" style={{ flex: 1, minHeight: 56 }}>
          <span className="lh">Original</span>
          <div className="lb" ref={topLaneRef}>
            <canvas ref={topCanvasRef} className="dub-wave top" aria-hidden />
            {readingTop !== undefined && !topRow && <span className="dub-reading">Reading {readingTop}%</span>}
          </div>
        </div>
        <div className="lane" style={{ flex: 1, minHeight: 56 }}>
          <span className="lh">Dub</span>
          <div className="lb" ref={dubLaneRef}>
            {plan.segments.map((s, i) => {
              if (s.endS <= view.startS || s.startS >= view.endS) return null;
              const kind = s.kind === "dub" ? "dub" : mode === "loaded" ? null : mode === "placing" && s.note === DRAFT_NOTE ? "pending" : "fill";
              if (!kind) return null;
              const x0 = xOf(Math.max(s.startS, view.startS));
              const x1 = xOf(Math.min(s.endS, view.endS));
              return <div key={i} className={cx("segm", kind, selected === i && "sel")} style={{ left: x0, width: Math.max(1, x1 - x0) }} />;
            })}
            <canvas ref={dubCanvasRef} className="dub-wave" aria-hidden />
            {editable &&
              grips &&
              plan.segments.map((s, i) =>
                i > 0 && s.startS > view.startS && s.startS < view.endS ? (
                  <span key={`g${i}`} className={cx("grip", hotBoundary === i - 1 && "hot")} style={{ left: xOf(s.startS) }} aria-hidden />
                ) : null,
              )}
            {readingDub !== undefined && dubRows.every((r) => !r.row) && <span className="dub-reading">Reading {readingDub}%</span>}
          </div>
        </div>
      </div>

      <HeadLine cursor={cursor} view={view} width={width} follow={keepCursorInView} dragging={drag !== null} onFollow={setView} />

      <div
        ref={minimapRef}
        className={cx("minimap", drag?.kind === "overview" ? "dub-grabbing" : hoverBox ? "dub-grab" : "dub-jump")}
        role="scrollbar"
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(view.startS)}
        aria-label="The whole film: click or drag to move the view"
        onPointerDown={onMinimapDown}
        onPointerMove={onMinimapMove}
        onPointerUp={onMinimapUp}
        onPointerCancel={onMinimapUp}
        onPointerLeave={() => setHoverBox(false)}
      >
        {plan.segments.map((s, i) =>
          s.kind === "fill" && s.note !== DRAFT_NOTE ? (
            <i key={i} style={{ left: `${(s.startS / duration) * 100}%`, width: `max(2px, ${((s.endS - s.startS) / duration) * 100}%)` }} />
          ) : null,
        )}
        <b style={{ left: box.x0, width: box.x1 - box.x0 }} />
      </div>
    </div>
  );
});

/** The playhead: a line over the lanes with a small triangle on top. On
 *  its own, so the playhead moving every frame redraws only itself. */
function HeadLine({
  cursor,
  view,
  width,
  follow,
  dragging,
  onFollow,
}: {
  cursor: CursorStore;
  view: View;
  width: number;
  follow: boolean;
  dragging: boolean;
  onFollow: (view: View) => void;
}) {
  const t = useCursor(cursor);
  // While the player runs, the view jumps to keep it on screen, a fifth of
  // the width in -- never in the middle of a gesture.
  useEffect(() => {
    if (!follow || t === null || dragging) return;
    const next = followCursor(view, t, 0.2);
    if (next) onFollow(next);
  }, [follow, t, view, dragging, onFollow]);
  if (t === null || t < view.startS || t > view.endS) return null;
  const x = ((t - view.startS) / (view.endS - view.startS)) * width;
  return <div className="head-line" style={{ left: PAD_PX + LABEL_PX + x }} aria-hidden />;
}

/** The Timeline's body while there is nothing to draw yet. */
export function TimelineMessage({ children }: { children: string }) {
  return (
    <div className="tl dub-tl-empty">
      <span className="t3">{children}</span>
    </div>
  );
}
