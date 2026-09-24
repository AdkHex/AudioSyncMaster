/** Both tracks of a dub sync, drawn the way an audio editor draws them.
 *
 *  Two lanes on one clock, the video's. The top lane is the video's own
 *  audio; the bottom lane is the synced track as the plan would write it:
 *  each stretch of dub drawn from where the plan reads it, each fill drawn
 *  as the original, hatched. Per channel, each pixel column is the lowest
 *  and highest sample in its span as a filled shape with the RMS inside it
 *  in a lighter tone, so a transient is a spike, and when a stretch sits
 *  where it should its spikes sit under the original's. Each lane is
 *  scaled to the loudest thing in sight, so a quiet scene reads as
 *  clearly as a loud one; the factor is shown in the corner.
 *
 *  The same drawing serves three moments: the pair as loaded, before any
 *  sync (the dub at the video's start); the analysis as it runs, redrawn
 *  from each draft the engine sends; and the finished plan, editable in
 *  the cut editor, which owns the plan and the keys and tells this
 *  component what to draw and what the pointer did. Peaks come from the
 *  engine per visible span and are kept.
 *
 *  Getting around is the same everywhere: drag to scroll, the wheel or two
 *  fingers to scroll, Ctrl/⌘-wheel or a pinch to zoom about the pointer,
 *  and a bar under the lanes with the whole film on it and the view as a
 *  box to click or drag. A press that does not travel is a click, and
 *  places the cursor. In the editor a cut is grabbed before anything else,
 *  and Alt/Option-dragging a stretch of dub slides it under the original.
 *  Where the editor knows the picture's own cuts (shot changes), they are
 *  ticks on the ruler, and a cut dragged near one lands on it.
 */

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";

import { cx } from "@/lib/cx";
import type { WaveformPeaks, WaveformRequest } from "@/lib/api";
import { segmentAt } from "@/lib/dubPlanEdit";
import { DRAFT_NOTE, formatClock, type DubSyncPlan } from "@/lib/types";
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
  overviewX,
  panView,
  peaksKey,
  slipOffsetDelta,
  snapToCuts,
  tickStep,
  visibleRequests,
  wheelDeltaPx,
  wheelZoomFactor,
  zoomAround,
  type View,
} from "@/lib/waveformView";

export interface WaveformStatus {
  /** What the engine is doing to this pair, shown over the ruler. */
  label: string;
  /** 0-100 within the stage, or null when nothing is measurable. */
  percent: number | null;
}

export interface DubWaveformViewHandle {
  /** Zoom around the middle of the view; factors under 1 zoom in. */
  zoomBy(factor: number): void;
  /** The whole film. */
  fit(): void;
  /** Show a span, padded a little. */
  show(startS: number, endS: number): void;
  view(): View;
}

export interface DubWaveformViewProps {
  plan: DubSyncPlan;
  fetchPeaks: (request: WaveformRequest) => Promise<WaveformPeaks>;
  /** Cuts can be dragged, pieces selected, double-click splits. */
  editable?: boolean;
  selected?: number | null;
  onSelect?: (index: number | null) => void;
  cursorS?: number | null;
  onCursor?: (timeS: number) => void;
  /** While the player runs, keep the cursor (the playhead) on screen:
   *  when it leaves the view, jump so it sits at a fifth of the width. */
  keepCursorInView?: boolean;
  onBoundaryDragStart?: (boundary: number) => void;
  onBoundaryDrag?: (boundary: number, timeS: number) => void;
  onBoundaryDragEnd?: () => void;
  onSplitAt?: (timeS: number) => void;
  /** Alt/Option-drag on a stretch of dub slips it: `onSlipStart` once the
   *  press becomes a drag, then `onSlip` with the change to the stretch's
   *  offset since the press (in seconds, to the millisecond; dragging the
   *  sound right lowers the offset), then `onSlipEnd` on release. Editable
   *  views only; without `onSlip` an Alt-drag pans like any other. */
  onSlipStart?: (index: number) => void;
  onSlip?: (index: number, offsetDeltaS: number) => void;
  onSlipEnd?: () => void;
  onViewChange?: (view: View) => void;
  /** The picture's cuts (shot changes) in the view, in order: ticks on the
   *  ruler, and where a dragged cut snaps to unless Alt/Option is held. */
  shotCuts?: readonly number[] | null;
  laneHeight?: number;
  status?: WaveformStatus | null;
  /** Files the engine is reading for the first time: path to percent. */
  reading?: Record<string, number>;
  className?: string;
}

/** Pixels around a cut within which the pointer grabs it. */
const HANDLE_PX = 6;
/** Pixels around a picture cut on the ruler within which it is named. */
const SHOT_HOVER_PX = 4;
const RULER_H = 20;
const LANE_GAP = 4;
/** The overview bar under the lanes: the whole film, the view as a box. */
const OVERVIEW_H = 14;
/** The quietest a lane is ever scaled up from: a floor of -34 dBFS, so
 *  silence stays flat rather than blowing up into noise. */
const GAIN_FLOOR = 0.02;

/** An audio editor's palette: the original green, the dub orange, the
 *  original where it fills a gap grey. Fixed rather than themed, since
 *  the lanes are dark in either theme and the two colours must never be
 *  confused. */
const ORIGINAL = { bg: "#173428", peak: "#3fcf8e", rms: "#b6f2d4", zero: "#2f6d52", text: "#8fe6bb" };
const DUB = { bg: "#3a2a15", peak: "#e8a33a", rms: "#f7dfb3", zero: "#7d5c2a", text: "#f3c476" };
const FILL = { peak: "#8f9088", rms: "#cfd0c8", hatch: "rgba(232, 163, 58, 0.28)", text: "#e0b06a" };
const DRAFT = { peak: "#5d5f5a", rms: "#8c8e88", hatch: "rgba(255, 255, 255, 0.10)", text: "#a0a29c" };

/** What the pointer is doing. A press is undecided until it travels past
 *  the drag threshold (then it pans, or slips a stretch when it began with
 *  Alt on one) or is released (then it was a click). A cut is grabbed at
 *  once, as is a Shift- or middle-button press, which always pans. */
type Drag =
  | { kind: "press"; startX: number; startY: number; startView: View; slip: number | null }
  | { kind: "pan"; startX: number; startView: View }
  | { kind: "boundary"; boundary: number }
  | { kind: "slip"; index: number; startX: number; startView: View; startOffsetS: number }
  | { kind: "overview"; startX: number; startView: View };

/** WebKit's pinch, which is how WKWebView (the Tauri window on macOS)
 *  reports a trackpad pinch instead of Chromium's Ctrl-wheel. Not in the
 *  DOM typings, since no other engine has it. */
type GestureEventLike = Event & { scale?: number; clientX?: number };

export const DubWaveformView = forwardRef<DubWaveformViewHandle, DubWaveformViewProps>(function DubWaveformView(
  {
    plan,
    fetchPeaks,
    editable = false,
    selected = null,
    onSelect,
    cursorS = null,
    onCursor,
    keepCursorInView = false,
    onBoundaryDragStart,
    onBoundaryDrag,
    onBoundaryDragEnd,
    onSplitAt,
    onSlipStart,
    onSlip,
    onSlipEnd,
    onViewChange,
    shotCuts = null,
    laneHeight = 96,
    status = null,
    reading,
    className,
  },
  ref,
) {
  const duration = Math.max(MIN_VIEW_S, plan.videoDurationS);
  const [view, setViewState] = useState<View>({ startS: 0, endS: duration });
  const [width, setWidth] = useState(800);
  const [peaks, setPeaks] = useState<Map<string, WaveformPeaks | null>>(new Map());
  const [loading, setLoading] = useState(0);
  const [hoverBoundary, setHoverBoundary] = useState<number | null>(null);
  const [hoverSlip, setHoverSlip] = useState(false);
  const [hoverBox, setHoverBox] = useState(false);
  // The picture cut named by the canvas's tooltip, and the one a dragged
  // cut has snapped to.
  const [hoverShot, setHoverShot] = useState<number | null>(null);
  const [snappedShot, setSnappedShot] = useState<number | null>(null);
  const [drag, setDragState] = useState<Drag | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overviewRef = useRef<HTMLCanvasElement>(null);
  const pendingRef = useRef<Set<string>>(new Set());
  const mountedRef = useRef(true);
  const viewRef = useRef(view);
  viewRef.current = view;
  // The pointer handlers read the gesture from a ref, not from state:
  // several pointer moves can arrive before React renders the first one's
  // update, and each must see what the one before it decided, or a press
  // would become a drag (and start a slip) more than once.
  const dragRef = useRef<Drag | null>(null);
  const setDrag = useCallback((next: Drag | null) => {
    dragRef.current = next;
    setDragState(next);
  }, []);

  const setView = useCallback(
    (next: View) => {
      const clamped = clampView(next, duration);
      // Straight into the ref too: wheel events come faster than renders,
      // and each must build on the last, not on the view last drawn.
      viewRef.current = clamped;
      setViewState(clamped);
      onViewChange?.(clamped);
    },
    [duration, onViewChange],
  );

  // The native wheel and pinch listeners are attached once; they read
  // what they need from here.
  const widthRef = useRef(800);
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

  // A film of another length: start over on the whole of it.
  useEffect(() => {
    setViewState((current) => {
      if (current.endS <= duration && current.startS < duration) return clampView(current, duration);
      return { startS: 0, endS: duration };
    });
  }, [duration]);

  // The playhead: when it leaves the view, jump so it lands a fifth of the
  // width in. Never while the pointer is dragging or panning -- that would
  // yank the view out from under the gesture.
  useEffect(() => {
    if (!keepCursorInView || cursorS === null || cursorS === undefined || drag !== null) return;
    const next = followCursor(view, cursorS, 0.2);
    if (next) setView(next);
  }, [keepCursorInView, cursorS, view, drag, setView]);

  useImperativeHandle(
    ref,
    () => ({
      zoomBy(factor) {
        const current = viewRef.current;
        const centre = (current.startS + current.endS) / 2;
        const length = Math.min(duration, Math.max(MIN_VIEW_S, (current.endS - current.startS) * factor));
        setView({ startS: centre - length / 2, endS: centre + length / 2 });
      },
      fit() {
        setView({ startS: 0, endS: duration });
      },
      show(startS, endS) {
        const pad = Math.max(1, (endS - startS) * 0.25);
        setView({ startS: startS - pad, endS: endS + pad });
      },
      view() {
        return viewRef.current;
      },
    }),
    [duration, setView],
  );

  // -------------------------------------------------------------------- size

  widthRef.current = width;

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const measure = () => {
      const w = Math.floor(element.getBoundingClientRect().width);
      if (w > 0) setWidth(w);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const originalTop = RULER_H;
  const dubTop = RULER_H + laneHeight + LANE_GAP;
  const height = RULER_H + laneHeight * 2 + LANE_GAP;

  const xOf = useCallback(
    (t: number) => ((t - view.startS) / (view.endS - view.startS)) * width,
    [view, width],
  );
  const tOf = useCallback(
    (x: number) => view.startS + (x / width) * (view.endS - view.startS),
    [view, width],
  );

  // ------------------------------------------------------------------- peaks

  const requests = useMemo(() => visibleRequests(plan, view, width), [plan, view, width]);

  useEffect(() => {
    const missing = requests.filter((r) => {
      const key = peaksKey(r);
      return !peaks.has(key) && !pendingRef.current.has(key);
    });
    if (missing.length === 0) return;
    // Rows are keyed by what they show, so one that arrives after the view
    // moved on is still right and is kept; only unmounting drops it.
    const timer = window.setTimeout(() => {
      missing.forEach((r) => pendingRef.current.add(peaksKey(r)));
      setLoading((n) => n + 1);
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
            // Keep the map from growing without bound across a long session.
            if (next.size > 400) {
              for (const key of Array.from(next.keys()).slice(0, next.size - 300)) next.delete(key);
            }
            return next;
          });
        })
        .finally(() => {
          missing.forEach((r) => pendingRef.current.delete(peaksKey(r)));
          if (mountedRef.current) setLoading((n) => Math.max(0, n - 1));
        });
    }, 30);
    return () => window.clearTimeout(timer);
  }, [requests, peaks, fetchPeaks]);

  // -------------------------------------------------------------------- draw

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    const styles = getComputedStyle(canvas);
    const themed = (name: string, alpha = 1) => `hsl(${styles.getPropertyValue(name).trim()} / ${alpha})`;
    const span = view.endS - view.startS;
    const mono = "10.5px ui-monospace, SFMono-Regular, Menlo, monospace";

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = ORIGINAL.bg;
    ctx.fillRect(0, originalTop, width, laneHeight);
    ctx.fillStyle = DUB.bg;
    ctx.fillRect(0, dubTop, width, laneHeight);

    const original = peaks.get(peaksKey(requests[0] ?? { path: "", track: 0, startS: 0, endS: 0, buckets: 0 })) ?? null;

    // Each lane is scaled to the loudest sample in sight.
    const loudest = (rows: (WaveformPeaks | null)[]) => {
      let peak = 0;
      for (const row of rows) {
        if (!row) continue;
        for (let ch = 0; ch < row.channels; ch++) {
          for (const v of row.max[ch]) if (v > peak) peak = v;
          for (const v of row.min[ch]) if (-v > peak) peak = -v;
        }
      }
      return peak;
    };
    const gainFor = (peak: number) => 1 / Math.max(peak, GAIN_FLOOR);
    const originalGain = gainFor(loudest([original]));
    const dubRows: (WaveformPeaks | null)[] = [];
    {
      let index = 1;
      for (const s of plan.segments) {
        if (s.kind !== "dub") continue;
        const a = Math.max(s.startS, view.startS);
        const b = Math.min(s.endS, view.endS);
        if (b <= a) continue;
        dubRows.push(peaks.get(peaksKey(requests[index] ?? { path: "", track: 0, startS: 0, endS: 0, buckets: 0 })) ?? null);
        index += 1;
      }
    }
    const dubGain = gainFor(loudest(dubRows));

    /** One track's shape across [x0, x1) of a lane: per channel, the
     *  min/max envelope filled, the RMS inside it. `slice` picks the
     *  buckets of `row` that belong to [x0, x1) when the row spans more. */
    const drawTrack = (
      row: WaveformPeaks,
      x0: number,
      x1: number,
      top: number,
      gain: number,
      colours: { peak: string; rms: string },
      slice?: { from: number; to: number },
    ) => {
      const channels = Math.max(1, row.channels);
      const laneH = laneHeight / channels;
      const from = slice?.from ?? 0;
      const to = slice?.to ?? row.buckets;
      const n = to - from;
      if (n <= 0 || x1 <= x0) return;
      const step = (x1 - x0) / n;
      const shape = (upper: number[], lower: number[], fill: string, mid: number, half: number) => {
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const y = mid - Math.max(-1, Math.min(1, upper[from + i] * gain)) * half;
          if (i === 0) ctx.moveTo(x0, y);
          ctx.lineTo(x0 + i * step, y);
          ctx.lineTo(x0 + (i + 1) * step, y);
        }
        for (let i = n - 1; i >= 0; i--) {
          const y = mid - Math.max(-1, Math.min(1, lower[from + i] * gain)) * half;
          ctx.lineTo(x0 + (i + 1) * step, y);
          ctx.lineTo(x0 + i * step, y);
        }
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      };
      for (let ch = 0; ch < channels; ch++) {
        const mid = top + laneH * ch + laneH / 2;
        const half = laneH / 2 - 1.5;
        const max = row.max[ch] ?? [];
        const min = row.min[ch] ?? [];
        const rms = row.rms[ch] ?? [];
        shape(max, min, colours.peak, mid, half);
        shape(rms, rms.map((v) => -v), colours.rms, mid, half);
      }
    };

    const zeroLines = (top: number, channels: number, colour: string) => {
      const laneH = laneHeight / Math.max(1, channels);
      ctx.fillStyle = colour;
      for (let ch = 0; ch < channels; ch++) {
        ctx.fillRect(0, Math.round(top + laneH * ch + laneH / 2) - 0.5, width, 1);
        if (ch > 0) {
          ctx.fillStyle = "rgba(0,0,0,0.25)";
          ctx.fillRect(0, Math.round(top + laneH * ch) - 0.5, width, 1);
          ctx.fillStyle = colour;
        }
      }
    };

    // The original, on its own lane.
    zeroLines(originalTop, original?.channels ?? 1, ORIGINAL.zero);
    if (original) drawTrack(original, 0, width, originalTop, originalGain, ORIGINAL);

    // The dub lane: fills are the original, dimmed and hatched; stretches
    // are the dub read from where the plan reads it.
    zeroLines(dubTop, dubRows.find((r) => r)?.channels ?? original?.channels ?? 1, DUB.zero);
    for (const s of plan.segments) {
      if (s.kind !== "fill") continue;
      const a = Math.max(s.startS, view.startS);
      const b = Math.min(s.endS, view.endS);
      if (b <= a) continue;
      const x0 = xOf(a);
      const x1 = xOf(b);
      const draft = s.note === DRAFT_NOTE;
      const colours = draft ? DRAFT : FILL;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, dubTop, x1 - x0, laneHeight);
      ctx.clip();
      if (original) {
        // The original's row covers the whole view; take the fill's share.
        const from = Math.round(((a - view.startS) / span) * original.buckets);
        const to = Math.round(((b - view.startS) / span) * original.buckets);
        drawTrack(original, xOf(view.startS + (from / original.buckets) * span), xOf(view.startS + (to / original.buckets) * span), dubTop, originalGain, colours, { from, to });
      }
      ctx.strokeStyle = colours.hatch;
      ctx.lineWidth = 1;
      for (let x = x0 - laneHeight; x < x1 + laneHeight; x += 9) {
        ctx.beginPath();
        ctx.moveTo(x, dubTop + laneHeight);
        ctx.lineTo(x + laneHeight, dubTop);
        ctx.stroke();
      }
      ctx.restore();
    }
    {
      let index = 0;
      for (const s of plan.segments) {
        if (s.kind !== "dub") continue;
        const a = Math.max(s.startS, view.startS);
        const b = Math.min(s.endS, view.endS);
        if (b <= a) continue;
        const row = dubRows[index];
        index += 1;
        if (row) drawTrack(row, xOf(a), xOf(b), dubTop, dubGain, DUB);
      }
    }

    // Selection.
    if (selected !== null && plan.segments[selected]) {
      const s = plan.segments[selected];
      const x0 = Math.max(0, xOf(s.startS));
      const x1 = Math.min(width, xOf(s.endS));
      ctx.fillStyle = "rgba(255,255,255,0.09)";
      ctx.fillRect(x0, originalTop, x1 - x0, height - originalTop);
    }

    /** Small text on a dark backdrop, so it reads over a waveform. */
    const badge = (text: string, x: number, y: number, colour: string, align: "left" | "right" = "left") => {
      const w = ctx.measureText(text).width + 8;
      const left = align === "left" ? x : x - w;
      ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
      ctx.fillRect(left, y - 2, w, 14);
      ctx.fillStyle = colour;
      ctx.textAlign = align;
      ctx.fillText(text, align === "left" ? x + 4 : x - 4, y);
      ctx.textAlign = "left";
      return w;
    };

    // Cuts: a line through both lanes, a grip on the dub lane when they
    // can be moved, and each piece's offset at its foot.
    ctx.font = mono;
    ctx.textBaseline = "top";
    plan.segments.forEach((s, index) => {
      if (index > 0 && s.startS >= view.startS && s.startS <= view.endS) {
        const x = Math.round(xOf(s.startS)) + 0.5;
        const hot = editable && (hoverBoundary === index - 1 || (drag?.kind === "boundary" && drag.boundary === index - 1));
        ctx.strokeStyle = hot ? "#ffffff" : "rgba(255,255,255,0.6)";
        ctx.lineWidth = hot ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, originalTop);
        ctx.lineTo(x, height);
        ctx.stroke();
        if (editable) {
          ctx.fillStyle = hot ? "#ffffff" : "rgba(255,255,255,0.8)";
          ctx.fillRect(x - 3, dubTop + laneHeight / 2 - 9, 6, 18);
        }
      }
      const a = Math.max(s.startS, view.startS);
      const b = Math.min(s.endS, view.endS);
      const offsetLabel = (offset: number) => `${offset >= 0 ? "+" : ""}${offset.toFixed(3)}s`;
      if (drag?.kind === "slip" && drag.index === index && b - a > 0) {
        // The stretch being slipped: its offset as it moves and how far it
        // has moved, always shown, bright, so the hand can stop on a number.
        const offset = s.offsetS ?? 0;
        const moved = offset - drag.startOffsetS;
        const text = `${offsetLabel(offset)}  (${moved >= 0 ? "+" : "−"}${Math.abs(moved).toFixed(3)})`;
        badge(text, Math.min(Math.max(xOf(a) + 4, 4), Math.max(4, width - ctx.measureText(text).width - 12)), dubTop + laneHeight - 15, "#ffffff");
      } else if (b - a > 0) {
        const label =
          s.kind === "fill"
            ? s.note === DRAFT_NOTE
              ? "not placed yet"
              : "original"
            : offsetLabel(s.offsetS ?? 0);
        // Only where it fits inside its own piece.
        if (ctx.measureText(label).width + 16 < xOf(b) - xOf(a)) {
          badge(label, xOf(a) + 4, dubTop + laneHeight - 15, s.kind === "fill" ? (s.note === DRAFT_NOTE ? DRAFT.text : FILL.text) : DUB.text);
        }
      }
    });

    // Lane names and gains.
    ctx.font = "600 9.5px ui-sans-serif, system-ui, sans-serif";
    badge("ORIGINAL", 4, originalTop + 4, ORIGINAL.text);
    badge("DUB, AS IT WILL BE WRITTEN", 4, dubTop + 4, DUB.text);
    ctx.font = mono;
    const gainLabel = (gain: number, top: number, colour: string, note?: string) => {
      const parts = [];
      if (note) parts.push(note);
      if (gain > 1.05) parts.push(`×${gain >= 10 ? gain.toFixed(0) : gain.toFixed(1)}`);
      if (parts.length === 0) return;
      badge(parts.join("  "), width - 4, top + 4, colour, "right");
    };
    const readingOf = (path: string) => (reading && path in reading ? `reading ${reading[path]}%` : undefined);
    gainLabel(original ? originalGain : 1, originalTop, ORIGINAL.text, readingOf(plan.videoPath));
    gainLabel(dubRows.some((r) => r) ? dubGain : 1, dubTop, DUB.text, readingOf(plan.dubPath));

    // Ruler.
    ctx.fillStyle = themed("--card");
    ctx.fillRect(0, 0, width, RULER_H);
    ctx.strokeStyle = themed("--border");
    ctx.beginPath();
    ctx.moveTo(0, RULER_H - 0.5);
    ctx.lineTo(width, RULER_H - 0.5);
    ctx.stroke();
    const step = tickStep(span, width);
    ctx.fillStyle = themed("--muted-foreground");
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (let t = Math.ceil(view.startS / step) * step; t <= view.endS; t += step) {
      const x = Math.round(xOf(t)) + 0.5;
      ctx.strokeStyle = themed("--border-strong");
      ctx.beginPath();
      ctx.moveTo(x, RULER_H - 6);
      ctx.lineTo(x, RULER_H);
      ctx.stroke();
      ctx.fillText(step >= 1 ? formatClock(t).replace(/\.\d{3}$/, "") : formatClock(t), x + 3, 4);
    }

    // What the engine is doing: a label over the ruler and, where the
    // stage has a measure, a bar along it.
    if (status) {
      ctx.textAlign = "right";
      ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
      const text = status.percent === null ? status.label : `${status.label} · ${status.percent}%`;
      const w = ctx.measureText(text).width + 12;
      ctx.fillStyle = themed("--card");
      ctx.fillRect(width - w - 2, 2, w + 2, RULER_H - 5);
      ctx.fillStyle = themed("--primary");
      ctx.fillText(text, width - 8, 4);
      ctx.textAlign = "left";
      if (status.percent !== null) {
        ctx.fillStyle = themed("--primary");
        ctx.fillRect(0, RULER_H - 2, (width * status.percent) / 100, 2);
      }
    }

    // Cursor.
    if (cursorS !== null && cursorS >= view.startS && cursorS <= view.endS) {
      const x = Math.round(xOf(cursorS)) + 0.5;
      ctx.strokeStyle = "rgba(255, 80, 80, 0.95)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }, [plan, view, width, height, laneHeight, originalTop, dubTop, peaks, requests, selected, cursorS, hoverBoundary, drag, editable, status, reading, xOf]);

  // ---------------------------------------------------------------- overview

  useEffect(() => {
    const canvas = overviewRef.current;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(OVERVIEW_H * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${OVERVIEW_H}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, OVERVIEW_H);
    ctx.fillStyle = "#1c1d1b";
    ctx.fillRect(0, 0, width, OVERVIEW_H);

    // Every piece of the plan at the bar's scale: dub orange, the original
    // grey, what the engine has not placed yet dimmer still. A piece is
    // never drawn thinner than a pixel, so short ones do not vanish.
    for (const s of plan.segments) {
      const x0 = overviewX(s.startS, duration, width);
      const x1 = Math.max(x0 + 1, overviewX(s.endS, duration, width));
      ctx.fillStyle =
        s.kind === "dub" ? "rgba(232, 163, 58, 0.65)" : s.note === DRAFT_NOTE ? "rgba(143, 144, 136, 0.22)" : "rgba(143, 144, 136, 0.55)";
      ctx.fillRect(x0, 3, x1 - x0, OVERVIEW_H - 6);
    }

    if (cursorS !== null) {
      const x = Math.round(overviewX(cursorS, duration, width)) + 0.5;
      ctx.strokeStyle = "rgba(255, 80, 80, 0.95)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, OVERVIEW_H);
      ctx.stroke();
    }

    const box = overviewBox(view, duration, width);
    const active = hoverBox || drag?.kind === "overview";
    ctx.fillStyle = active ? "rgba(255, 255, 255, 0.16)" : "rgba(255, 255, 255, 0.10)";
    ctx.fillRect(box.x0, 0, box.x1 - box.x0, OVERVIEW_H);
    ctx.strokeStyle = active ? "#ffffff" : "rgba(255, 255, 255, 0.8)";
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(box.x0) + 0.5, 0.5, Math.max(1, Math.round(box.x1 - box.x0) - 1), OVERVIEW_H - 1);
  }, [plan, view, width, duration, cursorS, hoverBox, drag]);

  // ---------------------------------------------------------------- pointer

  const boundaryNear = useCallback(
    (x: number): number | null => {
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
    },
    [editable, plan, xOf],
  );

  /** The stretch of dub an Alt-press at `x` would slip, if any. */
  const slipTarget = useCallback(
    (x: number): number | null => {
      if (!editable || !onSlip) return null;
      const index = segmentAt(plan, tOf(x));
      return index !== null && plan.segments[index].kind === "dub" ? index : null;
    },
    [editable, onSlip, plan, tOf],
  );

  const localX = (event: { clientX: number }, canvas: HTMLCanvasElement | null = canvasRef.current) => {
    const rect = canvas?.getBoundingClientRect();
    return rect ? event.clientX - rect.left : 0;
  };
  const localY = (event: { clientY: number }) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return rect ? event.clientY - rect.top : 0;
  };

  /** The picture cut under a point of the ruler, if any. */
  const shotNear = (x: number, y: number): number | null => {
    if (!shotCuts || y < 0 || y > RULER_H) return null;
    const at = tOf(x);
    const t = snapToCuts(at, shotCuts, width / (view.endS - view.startS), SHOT_HOVER_PX);
    return t !== at || shotCuts.includes(t) ? t : null;
  };

  const release = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    // The primary button and the middle one; the right one is left to the
    // context menu.
    if (event.button !== 0 && event.button !== 1) return;
    const x = localX(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.button === 1 || event.shiftKey) {
      setDrag({ kind: "pan", startX: x, startView: view });
      return;
    }
    const boundary = boundaryNear(x);
    if (boundary !== null) {
      onBoundaryDragStart?.(boundary);
      setDrag({ kind: "boundary", boundary });
      onSelect?.(boundary + 1);
      return;
    }
    // Undecided until it moves or is released: nothing happens yet, so a
    // drag that scrolls does not first move the cursor to where it began.
    setDrag({ kind: "press", startX: x, startY: event.clientY, startView: view, slip: event.altKey ? slipTarget(x) : null });
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const x = localX(event);
    const current = dragRef.current;
    if (current === null) {
      setHoverBoundary(boundaryNear(x));
      setHoverSlip(event.altKey && slipTarget(x) !== null);
      setHoverShot(shotNear(x, localY(event)));
      return;
    }
    switch (current.kind) {
      case "boundary": {
        // Onto the picture's cut when it is near, unless Alt/Option is held.
        const t = tOf(x);
        const snapped = shotCuts && !event.altKey ? snapToCuts(t, shotCuts, width / (view.endS - view.startS)) : t;
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
          setDrag({ kind: "slip", index, startX: current.startX, startView: current.startView, startOffsetS: plan.segments[index].offsetS ?? 0 });
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

  const endGesture = (event: React.PointerEvent<HTMLCanvasElement>, cancelled: boolean) => {
    release(event);
    const current = dragRef.current;
    setDrag(null);
    setSnappedShot(null);
    if (current === null) return;
    if (current.kind === "boundary") onBoundaryDragEnd?.();
    else if (current.kind === "slip") onSlipEnd?.();
    else if (current.kind === "press" && !cancelled) {
      // A press that never became a drag is a click, where it was pressed.
      const t = Math.min(Math.max(tOf(current.startX), 0), duration);
      onCursor?.(t);
      onSelect?.(segmentAt(plan, t));
    }
  };

  const onDoubleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editable) return;
    onSplitAt?.(tOf(localX(event)));
  };

  // The overview: a press on the box grabs it; a press anywhere else
  // centres the view there first and then grabs it, so a click jumps and
  // a click-and-drag jumps and keeps going.
  const onOverviewDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    const x = localX(event, overviewRef.current);
    event.currentTarget.setPointerCapture(event.pointerId);
    let start = view;
    if (!overviewHit(x, overviewBox(view, duration, width))) {
      start = centreViewAt(view, overviewT(x, duration, width), duration);
      setView(start);
    }
    setDrag({ kind: "overview", startX: x, startView: start });
  };

  const onOverviewMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const x = localX(event, overviewRef.current);
    const current = dragRef.current;
    if (current?.kind === "overview") {
      setView(overviewDragView(current.startView, x - current.startX, duration, width));
      return;
    }
    if (current === null) setHoverBox(overviewHit(x, overviewBox(view, duration, width)));
  };

  const onOverviewUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    release(event);
    if (dragRef.current?.kind === "overview") setDrag(null);
  };

  // ------------------------------------------------------------ wheel, pinch

  // Native listeners, not React's: React attaches wheel listeners as
  // passive, so its handler cannot stop the page scrolling under the
  // waveform while the waveform scrolls too. These are attached once and
  // read the view, width and length from refs.
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    // A WebKit pinch in progress: the view and the instant it started on.
    let pinch: { view: View; atS: number } | null = null;

    /** The instant a zoom holds still: the one under the pointer on the
     *  lanes, the middle of the view anywhere else (the overview bar). */
    const anchorAt = (clientX: number | undefined, target: EventTarget | null) => {
      const current = viewRef.current;
      const canvas = canvasRef.current;
      if (canvas && target === canvas && clientX !== undefined && Number.isFinite(clientX)) {
        const x = clientX - canvas.getBoundingClientRect().left;
        return current.startS + (x / Math.max(1, widthRef.current)) * (current.endS - current.startS);
      }
      return (current.startS + current.endS) / 2;
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const current = viewRef.current;
      if (event.ctrlKey || event.metaKey) {
        // WebKit may send a Ctrl-wheel alongside its own pinch events; the
        // pinch already zooms.
        if (pinch) return;
        const factor = wheelZoomFactor(event.deltaY, event.deltaMode);
        setViewRef.current(zoomAround(current, anchorAt(event.clientX, event.target), factor, durationRef.current));
        return;
      }
      // Whichever way the wheel or the fingers mostly went: a two-finger
      // swipe sideways, a plain wheel, or a Shift-wheel (sideways already).
      const delta = Math.abs(event.deltaX) >= Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
      setViewRef.current(panView(current, -wheelDeltaPx(delta, event.deltaMode) * 0.6, widthRef.current));
    };

    // WebKit's pinch: `scale` is relative to the gesture's start, so each
    // change zooms the view the gesture started on. Only WebKit sends
    // these; elsewhere the listeners sit idle.
    const onGestureStart = (event: Event) => {
      event.preventDefault();
      const gesture = event as GestureEventLike;
      pinch = { view: viewRef.current, atS: anchorAt(gesture.clientX, event.target) };
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

  const cursorClass =
    drag?.kind === "pan"
      ? "cursor-grabbing"
      : drag?.kind === "slip" || (drag === null && hoverSlip)
        ? "cursor-ew-resize"
        : drag?.kind === "boundary" || (drag === null && hoverBoundary !== null)
          ? "cursor-col-resize"
          : editable
            ? "cursor-crosshair"
            : "cursor-grab";

  return (
    <div ref={containerRef} className={cx("relative select-none", className)}>
      <canvas
        ref={canvasRef}
        className={cx("block w-full touch-none rounded-md", cursorClass)}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => endGesture(event, false)}
        onPointerCancel={(event) => endGesture(event, true)}
        onPointerLeave={() => {
          if (dragRef.current === null) {
            setHoverBoundary(null);
            setHoverSlip(false);
            setHoverShot(null);
          }
        }}
        onDoubleClick={onDoubleClick}
        title={hoverShot !== null ? `Picture cut ${formatClock(hoverShot)}` : undefined}
        aria-label="Waveforms of the original and the synced dub"
      />
      {shotCuts && shotCuts.length > 0 && (
        // The picture's cuts, over the ruler. Only marks: the pointer goes
        // through them to the canvas, which names the one it is over.
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 overflow-hidden" style={{ height: RULER_H }}>
          {shotCuts.map((t) => {
            const x = Math.round(xOf(t));
            if (x < 0 || x > width) return null;
            return (
              <span
                key={t}
                data-shot-cut={t}
                title={`Picture cut ${formatClock(t)}`}
                className={cx("absolute bottom-0 w-px", t === snappedShot ? "h-full bg-primary" : "h-[7px] bg-primary/60")}
                style={{ left: x }}
              />
            );
          })}
        </div>
      )}
      <canvas
        ref={overviewRef}
        className={cx(
          "mt-1 block w-full touch-none rounded-sm",
          drag?.kind === "overview" ? "cursor-grabbing" : hoverBox ? "cursor-grab" : "cursor-pointer",
        )}
        style={{ height: OVERVIEW_H }}
        onPointerDown={onOverviewDown}
        onPointerMove={onOverviewMove}
        onPointerUp={onOverviewUp}
        onPointerCancel={onOverviewUp}
        onPointerLeave={() => setHoverBox(false)}
        aria-label="The whole film: click or drag to move the view"
      />
      {loading > 0 && (
        <div className="pointer-events-none absolute right-2 top-[26px] text-[10px] text-white/60">reading…</div>
      )}
    </div>
  );
});
