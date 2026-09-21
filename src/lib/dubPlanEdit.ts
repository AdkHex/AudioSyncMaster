/** Editing a dub sync plan by hand.
 *
 *  The engine's plan is a list of pieces covering the video's timeline,
 *  each either a stretch of the dub read from some point in the dub or a
 *  stretch of the video's own audio. The editor lets the user move the cut
 *  between two pieces, change where a piece reads the dub from (its
 *  offset), split a piece, merge two, and turn a fill into dub or back.
 *  Every operation here returns a new plan that still covers the timeline
 *  contiguously, which is all the renderer asks of it; the renderer
 *  crossfades every seam it is given.
 *
 *  Times are seconds on the video's clock. A dub piece's `sourceStartS` is
 *  where it starts reading the dub, on the dub's timeline as the plan plays
 *  it (already stretched to the video's clock when the dub runs at another
 *  speed), and its offset is `sourceStartS - startS`: dub time minus video
 *  time, the number the plan's table shows. Moving a piece's boundary keeps
 *  its offset -- the scene stays in sync, more or less of it plays. Nudging
 *  its offset keeps its boundaries -- the same span of video gets a
 *  different span of dub. Those are the two things a cut can be wrong by.
 */

import type { DubCodec, DubSegment, DubSyncPlan } from "./types";

/** No piece may be shorter than this: a crossfade needs room. */
export const MIN_PIECE_S = 0.1;

/** Note carried by a piece the user placed, so the plan's table says so. */
export const HAND_NOTE = "set by hand";

/** One video frame, from the plan's frame rate; a film's frame otherwise. */
export function frameSeconds(plan: Pick<DubSyncPlan, "videoFps">): number {
  return plan.videoFps && plan.videoFps > 0 ? 1 / plan.videoFps : 1 / 24;
}

/** The dub's length on the plan's clock: its own length stretched by the
 *  speed it is played at. */
export function dubSpanS(plan: Pick<DubSyncPlan, "dubDurationS" | "speed">): number {
  return plan.dubDurationS * (plan.speed || 1);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000 || 0; // and never -0
}

function copySegment(segment: DubSegment, patch: Partial<DubSegment> = {}): DubSegment {
  return { ...segment, ...patch };
}

function byHand(segment: DubSegment): DubSegment {
  return copySegment(segment, { note: HAND_NOTE, match: null, uncertaintyS: 0 });
}

/** Seconds of the plan taken from the original, recomputed. */
export function filledSeconds(segments: DubSegment[]): number {
  return round(
    segments.filter((s) => s.kind === "fill").reduce((sum, s) => sum + (s.endS - s.startS), 0),
  );
}

function rebuilt(plan: DubSyncPlan, segments: DubSegment[]): DubSyncPlan {
  return { ...plan, segments, filledS: filledSeconds(segments) };
}

/** Index of the piece playing at `timeS`, or null outside the plan. */
export function segmentAt(plan: DubSyncPlan, timeS: number): number | null {
  const index = plan.segments.findIndex((s) => timeS >= s.startS && timeS < s.endS);
  if (index >= 0) return index;
  const last = plan.segments.length - 1;
  return last >= 0 && timeS === plan.segments[last].endS ? last : null;
}

/** Whatever a fill's or dub's source must be for it to start at `startS`
 *  and keep its offset (dub) or read the original in place (fill). */
function sourceFor(segment: DubSegment, startS: number): number {
  return segment.kind === "dub" ? startS + (segment.offsetS ?? 0) : startS;
}

/** Move the cut after `index` -- the end of piece `index` and the start of
 *  piece `index + 1` -- to `timeS`, keeping both pieces' offsets. Clamped so
 *  neither piece falls below MIN_PIECE_S. The last cut (the end of the plan)
 *  cannot move: the track is the video's length. */
export function moveBoundary(plan: DubSyncPlan, index: number, timeS: number): DubSyncPlan {
  const before = plan.segments[index];
  const after = plan.segments[index + 1];
  if (!before || !after) return plan;
  const lo = before.startS + MIN_PIECE_S;
  const hi = after.endS - MIN_PIECE_S;
  if (hi < lo) return plan;
  const at = round(Math.min(hi, Math.max(lo, timeS)));
  if (at === before.endS) return plan;
  const segments = plan.segments.slice();
  segments[index] = byHand(copySegment(before, { endS: at }));
  segments[index + 1] = byHand(copySegment(after, { startS: at, sourceStartS: round(sourceFor(after, at)) }));
  return rebuilt(plan, segments);
}

/** Give a dub piece a new offset: the same span of video, read from
 *  `offsetS` later in the dub. Clamped so the piece stays inside the dub. */
export function setOffset(plan: DubSyncPlan, index: number, offsetS: number): DubSyncPlan {
  const segment = plan.segments[index];
  if (!segment || segment.kind !== "dub") return plan;
  const lo = -segment.startS;
  const hi = dubSpanS(plan) - segment.endS;
  const offset = round(Math.min(Math.max(offsetS, lo), Math.max(lo, hi)));
  if (offset === segment.offsetS) return plan;
  const segments = plan.segments.slice();
  segments[index] = byHand(
    copySegment(segment, { offsetS: offset, sourceStartS: round(segment.startS + offset) }),
  );
  return rebuilt(plan, segments);
}

export function nudgeOffset(plan: DubSyncPlan, index: number, deltaS: number): DubSyncPlan {
  const segment = plan.segments[index];
  if (!segment || segment.kind !== "dub") return plan;
  return setOffset(plan, index, (segment.offsetS ?? 0) + deltaS);
}

/** Cut the piece playing at `timeS` in two, both halves as they were. Then
 *  one half can be nudged on its own. */
export function splitAt(plan: DubSyncPlan, timeS: number): DubSyncPlan {
  const index = segmentAt(plan, timeS);
  if (index === null) return plan;
  const segment = plan.segments[index];
  const at = round(timeS);
  if (at - segment.startS < MIN_PIECE_S || segment.endS - at < MIN_PIECE_S) return plan;
  const head = copySegment(segment, { endS: at });
  const tail = copySegment(segment, { startS: at, sourceStartS: round(sourceFor(segment, at)) });
  const segments = plan.segments.slice();
  segments.splice(index, 1, head, tail);
  return rebuilt(plan, segments);
}

/** Whether two adjacent pieces play the same audio without a seam: the
 *  same kind, and for dub the same offset to within a millisecond. */
export function canMerge(a: DubSegment | undefined, b: DubSegment | undefined): boolean {
  if (!a || !b || a.kind !== b.kind) return false;
  if (a.kind === "fill") return true;
  return Math.abs((a.offsetS ?? 0) - (b.offsetS ?? 0)) < 0.0015;
}

/** Join piece `index` with the one after it, when they are continuous. */
export function mergeWithNext(plan: DubSyncPlan, index: number): DubSyncPlan {
  const a = plan.segments[index];
  const b = plan.segments[index + 1];
  if (!canMerge(a, b)) return plan;
  const segments = plan.segments.slice();
  segments.splice(index, 2, copySegment(a, { endS: b.endS, note: a.note || b.note }));
  return rebuilt(plan, segments);
}

/** Replace a piece with the video's own audio. */
export function convertToFill(plan: DubSyncPlan, index: number): DubSyncPlan {
  const segment = plan.segments[index];
  if (!segment || segment.kind === "fill") return plan;
  const segments = plan.segments.slice();
  segments[index] = {
    kind: "fill",
    startS: segment.startS,
    endS: segment.endS,
    sourceStartS: segment.startS,
    offsetS: null,
    match: null,
    note: "filled by hand",
    uncertaintyS: 0,
  };
  return rebuilt(plan, segments);
}

/** The offset a piece would most plausibly take if it were dub: the nearest
 *  dub neighbour's, the one before it first. */
export function suggestedOffset(plan: DubSyncPlan, index: number): number {
  for (let i = index - 1; i >= 0; i--) {
    const s = plan.segments[i];
    if (s.kind === "dub" && s.offsetS !== null) return s.offsetS;
  }
  for (let i = index + 1; i < plan.segments.length; i++) {
    const s = plan.segments[i];
    if (s.kind === "dub" && s.offsetS !== null) return s.offsetS;
  }
  return 0;
}

/** Play the dub across a piece, at `offsetS` (a neighbour's by default). */
export function convertToDub(plan: DubSyncPlan, index: number, offsetS?: number): DubSyncPlan {
  const segment = plan.segments[index];
  if (!segment || segment.kind === "dub") return plan;
  const offset = round(offsetS ?? suggestedOffset(plan, index));
  const segments = plan.segments.slice();
  segments[index] = {
    kind: "dub",
    startS: segment.startS,
    endS: segment.endS,
    sourceStartS: round(segment.startS + offset),
    offsetS: offset,
    match: null,
    note: HAND_NOTE,
    uncertaintyS: 0,
  };
  return rebuilt(plan, segments);
}

/** Every way a plan can be unplayable, as sentences; empty when it is fine.
 *  The renderer would choke or, worse, quietly play the wrong thing. */
export function validatePlan(plan: DubSyncPlan): string[] {
  const problems: string[] = [];
  const { segments } = plan;
  if (segments.length === 0) return ["The plan has no pieces."];
  if (Math.abs(segments[0].startS) > 1e-6) problems.push("The first piece does not start at 0.");
  if (Math.abs(segments[segments.length - 1].endS - plan.videoDurationS) > 0.01) {
    problems.push("The last piece does not end where the video ends.");
  }
  const span = dubSpanS(plan);
  segments.forEach((s, i) => {
    if (s.endS - s.startS < MIN_PIECE_S - 1e-6) problems.push(`Piece ${i + 1} is shorter than ${MIN_PIECE_S}s.`);
    if (i > 0 && Math.abs(s.startS - segments[i - 1].endS) > 1e-6) {
      problems.push(`Piece ${i + 1} does not start where piece ${i} ends.`);
    }
    if (s.kind === "dub") {
      if (s.offsetS === null) problems.push(`Piece ${i + 1} is dub without an offset.`);
      if (s.sourceStartS < -1e-6) problems.push(`Piece ${i + 1} reads the dub before its start.`);
      if (s.sourceStartS + (s.endS - s.startS) > span + 1e-6) {
        problems.push(`Piece ${i + 1} reads the dub past its end.`);
      }
    }
  });
  return problems;
}

/** Whether two plans describe the same track. */
export function samePlan(a: DubSyncPlan, b: DubSyncPlan): boolean {
  if (a.segments.length !== b.segments.length) return false;
  return a.segments.every((s, i) => {
    const t = b.segments[i];
    return (
      s.kind === t.kind &&
      Math.abs(s.startS - t.startS) < 1e-6 &&
      Math.abs(s.endS - t.endS) < 1e-6 &&
      Math.abs(s.sourceStartS - t.sourceStartS) < 1e-6
    );
  });
}

/** The codec a written track is in, read off its extension, so writing it
 *  again with edited cuts keeps the format it was written in even if the
 *  codec setting has since changed; null when the extension says nothing. */
export function codecOfPath(path: string): DubCodec | null {
  const ext = /\.([a-z0-9]+)$/i.exec(path)?.[1]?.toLowerCase();
  switch (ext) {
    case "flac":
      return "flac";
    case "wav":
      return "wav";
    case "m4a":
    case "aac":
      return "aac";
    case "ac3":
      return "ac3";
    case "eac3":
    case "ec3":
      return "eac3";
    case "opus":
      return "opus";
    default:
      return null;
  }
}

/** The pair as loaded, before any sync: the dub laid at the video's start
 *  at its own speed, the video's tail filled where the dub is shorter.
 *  What the waveform view shows until the engine's first draft. */
export function asLoadedPlan(pair: {
  videoPath: string;
  dubPath: string;
  videoTrack: number;
  dubTrack: number;
  videoDurationS: number;
  dubDurationS: number;
}): DubSyncPlan {
  const videoS = Math.max(0, pair.videoDurationS);
  const dubS = Math.max(0, pair.dubDurationS);
  const covered = Math.min(videoS, dubS);
  const segments: DubSegment[] = [];
  if (covered > 0) {
    segments.push({ kind: "dub", startS: 0, endS: covered, sourceStartS: 0, offsetS: 0, match: null, note: "as loaded", uncertaintyS: 0 });
  }
  if (videoS > covered) {
    segments.push({ kind: "fill", startS: covered, endS: videoS, sourceStartS: covered, offsetS: null, match: null, note: "past dub end", uncertaintyS: 0 });
  }
  return {
    videoPath: pair.videoPath,
    dubPath: pair.dubPath,
    videoTrack: pair.videoTrack,
    dubTrack: pair.dubTrack,
    speed: 1,
    videoFps: null,
    dubRate: null,
    fillGainDb: 0,
    videoDurationS: videoS,
    dubDurationS: dubS,
    segments,
    warnings: [],
    notes: [],
    error: null,
    filledS: Math.max(0, videoS - covered),
  };
}
