/** The waveform editor's view arithmetic, kept pure so it can be tested.
 *
 *  A view is a span of the video's clock; the editor maps it onto the
 *  canvas width. What is fetched for it is one row of peaks for the
 *  original and one per visible stretch of dub, each read from where the
 *  plan reads it, so the lanes show the track as it will be written. */

import type { WaveformRequest } from "./api";
import type { DubSyncPlan } from "./types";

export interface View {
  startS: number;
  endS: number;
}

/** The view never shows less than this: two frames a pixel is plenty. */
export const MIN_VIEW_S = 0.5;

/** Round-figure tick spacing for a ruler over `spanS` seconds on `width` px. */
export function tickStep(spanS: number, width: number): number {
  const target = spanS / Math.max(1, width / 90);
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  return steps.find((step) => step >= target) ?? 3600;
}

/** Clamp a view to the plan, keeping its length where it fits. */
export function clampView(view: View, durationS: number): View {
  const length = Math.min(Math.max(view.endS - view.startS, MIN_VIEW_S), durationS);
  const startS = Math.min(Math.max(view.startS, 0), Math.max(0, durationS - length));
  return { startS, endS: startS + length };
}

/** Move a view so `cursorS` sits at `share` of its width, keeping its
 *  length, when the cursor has left it; null while it is in view. The
 *  playhead auto-scroll passes this to the view, which clamps it. */
export function followCursor(view: View, cursorS: number, share: number): View | null {
  if (cursorS >= view.startS && cursorS <= view.endS) return null;
  const length = view.endS - view.startS;
  const startS = cursorS - length * share;
  return { startS, endS: startS + length };
}

/** One key per fetched row: which file, which span, how many pixels. */
export function peaksKey(request: WaveformRequest): string {
  return [
    request.path,
    request.track,
    request.startS.toFixed(3),
    request.endS.toFixed(3),
    request.buckets,
    request.speed ?? 1,
  ].join("|");
}

/** What the view needs read: the original across the whole view first,
 *  then the span of dub each visible stretch reads, in plan order. A
 *  stretch's span is on the dub's timeline as the plan plays it -- at the
 *  plan's speed -- so its peaks land under the original's. */
export function visibleRequests(plan: DubSyncPlan, view: View, width: number): WaveformRequest[] {
  const list: WaveformRequest[] = [];
  if (width <= 0 || view.endS <= view.startS) return list;
  const span = view.endS - view.startS;
  list.push({
    path: plan.videoPath,
    track: plan.videoTrack,
    startS: view.startS,
    endS: view.endS,
    buckets: width,
    speed: 1,
  });
  for (const s of plan.segments) {
    if (s.kind !== "dub") continue;
    const a = Math.max(s.startS, view.startS);
    const b = Math.min(s.endS, view.endS);
    if (b <= a) continue;
    list.push({
      path: plan.dubPath,
      track: plan.dubTrack,
      startS: s.sourceStartS + (a - s.startS),
      endS: s.sourceStartS + (b - s.startS),
      buckets: Math.max(1, Math.round(((b - a) / span) * width)),
      speed: plan.speed,
    });
  }
  return list;
}

// ------------------------------------------------------------- gestures

/** A press that travels further than this before it is released is a
 *  drag; anything less is a click. A hand on a trackpad wobbles a pixel or
 *  two on every click, and a click must still place the cursor. */
export const DRAG_THRESHOLD_PX = 4;

/** Whether a press has travelled far enough to be a drag. */
export function isDrag(dxPx: number, dyPx: number): boolean {
  return Math.hypot(dxPx, dyPx) > DRAG_THRESHOLD_PX;
}

/** The view dragged by `dxPx` like a sheet of paper: what was under the
 *  pointer at the press stays under it, so dragging right shows earlier
 *  time. Not clamped; the view clamps whatever it is given. */
export function panView(view: View, dxPx: number, width: number): View {
  const dt = (-dxPx / Math.max(1, width)) * (view.endS - view.startS);
  return { startS: view.startS + dt, endS: view.endS + dt };
}

/** Zoom by `factor` (under 1 zooms in) keeping the instant `atS` where it
 *  is on screen, so the thing under the pointer stays under it. */
export function zoomAround(view: View, atS: number, factor: number, durationS: number): View {
  const span = view.endS - view.startS;
  const length = Math.min(durationS, Math.max(MIN_VIEW_S, span * factor));
  const share = span > 0 ? Math.min(1, Math.max(0, (atS - view.startS) / span)) : 0.5;
  const startS = atS - share * length;
  return clampView({ startS, endS: startS + length }, durationS);
}

/** A wheel delta in pixels, whatever unit the device reported it in
 *  (a few mice on Windows and Linux still report lines). */
export function wheelDeltaPx(delta: number, deltaMode: number): number {
  if (deltaMode === 1) return delta * 16;
  if (deltaMode === 2) return delta * 400;
  return delta;
}

/** The zoom factor for one Ctrl/⌘-wheel event. Exponential in the delta,
 *  so a trackpad pinch -- which Chromium delivers as a stream of small
 *  Ctrl-wheel deltas -- zooms smoothly, and capped per event, so a mouse
 *  notch (about 100 px) is one step of about a quarter, as it always was. */
export function wheelZoomFactor(deltaY: number, deltaMode = 0): number {
  const delta = Math.min(25, Math.max(-25, wheelDeltaPx(deltaY, deltaMode)));
  return Math.exp(delta * 0.01);
}

/** How far a slip-drag of `dxPx` moves a stretch's offset, to the
 *  millisecond. Dragging the dub's sound to the right plays it later on
 *  the video's clock, which is reading the dub from earlier: the offset
 *  (dub time minus video time) goes down by the distance dragged. */
export function slipOffsetDelta(dxPx: number, view: View, width: number): number {
  const seconds = (dxPx / Math.max(1, width)) * (view.endS - view.startS);
  return Math.round(-seconds * 1000) / 1000 || 0; // and never -0
}

// ------------------------------------------------------------- overview

/** The overview bar draws the whole film on the canvas width; the view is
 *  a box on it never narrower than this, so an hour-long film zoomed to a
 *  second still shows (and can be grabbed by) where it is. */
export const OVERVIEW_MIN_BOX_PX = 6;

/** Where an instant falls on the overview bar. */
export function overviewX(timeS: number, durationS: number, width: number): number {
  return durationS > 0 ? (timeS / durationS) * width : 0;
}

/** The instant under a point of the overview bar, inside the film. */
export function overviewT(x: number, durationS: number, width: number): number {
  const t = width > 0 ? (x / width) * durationS : 0;
  return Math.min(Math.max(t, 0), Math.max(0, durationS));
}

/** The view's box on the overview bar, widened about its middle to the
 *  minimum and kept on the bar. */
export function overviewBox(view: View, durationS: number, width: number): { x0: number; x1: number } {
  let x0 = overviewX(view.startS, durationS, width);
  let x1 = overviewX(view.endS, durationS, width);
  const minimum = Math.min(OVERVIEW_MIN_BOX_PX, width);
  if (x1 - x0 < minimum) {
    const middle = (x0 + x1) / 2;
    x0 = Math.min(Math.max(middle - minimum / 2, 0), width - minimum);
    x1 = x0 + minimum;
  }
  return { x0, x1 };
}

/** Whether a press at `x` grabs the box, with a little slack either side. */
export function overviewHit(x: number, box: { x0: number; x1: number }, slackPx = 3): boolean {
  return x >= box.x0 - slackPx && x <= box.x1 + slackPx;
}

/** The view, same length, centred on `timeS` where the film allows. */
export function centreViewAt(view: View, timeS: number, durationS: number): View {
  const length = view.endS - view.startS;
  return clampView({ startS: timeS - length / 2, endS: timeS + length / 2 }, durationS);
}

/** The view as it was when the box was grabbed, moved with the pointer:
 *  a pixel of the bar is a whole film's width-th of time, not the view's. */
export function overviewDragView(startView: View, dxPx: number, durationS: number, width: number): View {
  const dt = (dxPx / Math.max(1, width)) * durationS;
  return clampView({ startS: startView.startS + dt, endS: startView.endS + dt }, durationS);
}

// ------------------------------------------------------------ picture cuts

/** The picture's cuts are marked on the ruler only while the view shows at
 *  most this much: finding them decodes the picture, a second or so for
 *  every 10-20 s of it. */
export const SHOT_CUTS_MAX_VIEW_S = 300;

/** They are read and kept in blocks of this much film, on round multiples
 *  of it, so a pan asks only for the blocks it brings into sight and
 *  panning back asks for nothing. */
export const SHOT_BLOCK_S = 60;

/** A dragged cut lands on a picture cut when it comes this near to one. */
export const SNAP_PX = 6;

/** What is known of a film's picture cuts: per block start, its cuts, or
 *  null where the picture could not be read. */
export type ShotBlocks = ReadonlyMap<number, number[] | null>;

/** The starts of the blocks that cover the view, inside the film. */
export function shotBlocks(view: View, durationS: number, blockS = SHOT_BLOCK_S): number[] {
  const starts: number[] = [];
  const end = Math.min(view.endS, durationS);
  for (let b = Math.max(0, Math.floor(view.startS / blockS) * blockS); b < end; b += blockS) starts.push(b);
  return starts;
}

/** The one span to ask the engine for so the view's cuts are all known:
 *  from the first block not read yet to the end of the last, or null when
 *  every block in sight is known. */
export function shotSpanToFetch(
  known: ShotBlocks,
  view: View,
  durationS: number,
  blockS = SHOT_BLOCK_S,
): View | null {
  const missing = shotBlocks(view, durationS, blockS).filter((b) => !known.has(b));
  if (missing.length === 0) return null;
  return { startS: missing[0], endS: Math.min(missing[missing.length - 1] + blockS, durationS) };
}

/** `known` with the engine's answer for [startS, endS] filed by block.
 *  Only blocks the answer covers whole are filed; a span the engine cut
 *  short leaves the rest to be asked for again. */
export function storeShotCuts(
  known: ShotBlocks,
  reply: { cuts: number[] | null; startS: number; endS: number },
  durationS: number,
  blockS = SHOT_BLOCK_S,
): Map<number, number[] | null> {
  const next = new Map(known);
  const slack = 1e-6;
  for (const b of shotBlocks(reply, durationS, blockS)) {
    if (b < reply.startS - slack || Math.min(b + blockS, durationS) > reply.endS + slack) continue;
    next.set(b, reply.cuts ? reply.cuts.filter((c) => c >= b && c < b + blockS) : null);
  }
  return next;
}

/** The known cuts in the view, in order. */
export function shotCutsIn(known: ShotBlocks, view: View, durationS: number, blockS = SHOT_BLOCK_S): number[] {
  const cuts: number[] = [];
  for (const b of shotBlocks(view, durationS, blockS)) {
    for (const c of known.get(b) ?? []) if (c >= view.startS && c <= view.endS) cuts.push(c);
  }
  return cuts;
}

/** `timeS` moved onto the nearest of `cuts` (in order) when that one is
 *  within `thresholdPx` on screen; `timeS` itself otherwise. */
export function snapToCuts(timeS: number, cuts: readonly number[], pxPerSecond: number, thresholdPx = SNAP_PX): number {
  if (cuts.length === 0 || !(pxPerSecond > 0)) return timeS;
  // The first cut at or after timeS; the nearest is it or the one before.
  let lo = 0;
  let hi = cuts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cuts[mid] < timeS) lo = mid + 1;
    else hi = mid;
  }
  let best: number | null = null;
  for (const c of [cuts[lo - 1], cuts[lo]]) {
    if (c === undefined) continue;
    if (best === null || Math.abs(c - timeS) < Math.abs(best - timeS)) best = c;
  }
  return best !== null && Math.abs(best - timeS) * pxPerSecond <= thresholdPx ? best : timeS;
}
