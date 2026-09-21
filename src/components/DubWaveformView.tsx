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
import { clampView, followCursor, MIN_VIEW_S, peaksKey, tickStep, visibleRequests, type View } from "@/lib/waveformView";

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
  onViewChange?: (view: View) => void;
  laneHeight?: number;
  status?: WaveformStatus | null;
  /** Files the engine is reading for the first time: path to percent. */
  reading?: Record<string, number>;
  className?: string;
}

/** Pixels around a cut within which the pointer grabs it. */
const HANDLE_PX = 6;
const RULER_H = 20;
const LANE_GAP = 4;
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

interface Drag {
  kind: "boundary" | "pan";
  boundary?: number;
  startX?: number;
  startView?: View;
}

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
    onViewChange,
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
  const [drag, setDrag] = useState<Drag | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pendingRef = useRef<Set<string>>(new Set());
  const mountedRef = useRef(true);
  const viewRef = useRef(view);
  viewRef.current = view;

  const setView = useCallback(
    (next: View) => {
      const clamped = clampView(next, duration);
      setViewState(clamped);
      onViewChange?.(clamped);
    },
    [duration, onViewChange],
  );

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
      if (b - a > 0) {
        const label =
          s.kind === "fill"
            ? s.note === DRAFT_NOTE
              ? "not placed yet"
              : "original"
            : `${(s.offsetS ?? 0) >= 0 ? "+" : ""}${(s.offsetS ?? 0).toFixed(3)}s`;
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

  const localX = (event: React.PointerEvent | React.MouseEvent | React.WheelEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return rect ? event.clientX - rect.left : 0;
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
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
    const t = Math.min(Math.max(tOf(x), 0), duration);
    onCursor?.(t);
    onSelect?.(segmentAt(plan, t));
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const x = localX(event);
    if (drag?.kind === "boundary" && drag.boundary !== undefined) {
      onBoundaryDrag?.(drag.boundary, tOf(x));
      return;
    }
    if (drag?.kind === "pan" && drag.startX !== undefined && drag.startView) {
      const dt = ((drag.startX - x) / width) * (drag.startView.endS - drag.startView.startS);
      setView({ startS: drag.startView.startS + dt, endS: drag.startView.endS + dt });
      return;
    }
    setHoverBoundary(boundaryNear(x));
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.releasePointerCapture(event.pointerId);
    if (drag?.kind === "boundary") onBoundaryDragEnd?.();
    setDrag(null);
  };

  const onDoubleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editable) return;
    onSplitAt?.(tOf(localX(event)));
  };

  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    const x = localX(event);
    if (event.ctrlKey || event.metaKey) {
      const factor = event.deltaY > 0 ? 1.25 : 0.8;
      const at = tOf(x);
      const length = Math.min(duration, Math.max(MIN_VIEW_S, (view.endS - view.startS) * factor));
      const share = (at - view.startS) / (view.endS - view.startS);
      setView({ startS: at - share * length, endS: at - share * length + length });
    } else {
      const delta = (event.deltaX || event.deltaY) / width;
      const length = view.endS - view.startS;
      setView({ startS: view.startS + delta * length * 0.6, endS: view.endS + delta * length * 0.6 });
    }
  };

  return (
    <div ref={containerRef} className={cx("relative select-none", className)}>
      <canvas
        ref={canvasRef}
        className={cx(
          "block w-full touch-none rounded-md",
          drag?.kind === "pan" ? "cursor-grabbing" : hoverBoundary !== null ? "cursor-col-resize" : editable ? "cursor-crosshair" : "cursor-default",
        )}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={onDoubleClick}
        onWheel={onWheel}
        aria-label="Waveforms of the original and the synced dub"
      />
      {loading > 0 && (
        <div className="pointer-events-none absolute right-2 top-[26px] text-[10px] text-white/60">reading…</div>
      )}
    </div>
  );
});
