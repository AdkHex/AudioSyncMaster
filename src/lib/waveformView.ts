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
