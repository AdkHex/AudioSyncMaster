/** Dub sync plans, drafts and verification for the demo. Dev only. */

import {
  DRAFT_NOTE,
  formatClock,
  formatSpan,
  type DubCodec,
  type DubLineCheck,
  type DubOutput,
  type DubPlanSummary,
  type DubSegment,
  type DubSyncPlan,
  type DubVerification,
  type VoicePiece,
} from "@/lib/types";

import { SKYLINE_DUB, SKYLINE_VIDEO, dirName, extOf, stemOf, unit, type FileFx } from "./fixtures";

const NTSC_FILM = 24000 / 1001;

/** Stage names the engine reports, in its order (audiosync/dubsync.py:
 *  measuring at 60, placing at 70). The write, named after the file, comes
 *  between the plan and the check. */
export const DUB_STAGES = [
  "probing",
  "checking the frame rate",
  "finding the offsets",
  "measuring the offsets",
  "placing the cuts",
  "looking for dub inside the gaps",
  "assembling the plan",
  "checking the finished track",
] as const;

/** Milliseconds each stage takes; with the write they sum to 7 s. */
export const DUB_STAGE_MS = [600, 500, 1100, 1000, 1000, 700, 400, 900] as const;
export const DUB_WRITE_MS = 800;

const segment = (over: Partial<DubSegment> & Pick<DubSegment, "kind" | "startS" | "endS">): DubSegment => ({
  sourceStartS: over.startS,
  offsetS: null,
  match: null,
  note: "",
  uncertaintyS: 0.01,
  ...over,
});

/** Mockup PLAN: a 2:20:30 film, a 2:10:10 dub, five scenes cut from the dub.
 *  Offsets and source times are the mockup's own strings; the two are not
 *  exactly startS + offsetS there, and are kept as drawn. */
function skylineSegments(): DubSegment[] {
  const fill = (startS: number, endS: number, note: string, reason: string) =>
    segment({ kind: "fill", startS, endS, note, reason });
  const dub = (startS: number, endS: number, sourceStartS: number, offsetS: number, match: number) =>
    segment({ kind: "dub", startS, endS, sourceStartS, offsetS, match });
  return [
    fill(0, 1.21, "Dub starts late", "head"),
    dub(1.21, 1230, 0.404, 0.806, 0.12),
    fill(1230, 1361.5, "Cut from the dub", "cut"),
    dub(1361.5, 3070.25, 1229.194, -130.694, 0.09),
    fill(3070.25, 3182, "Cut from the dub", "cut"),
    dub(3182, 4724.8, 2939.556, -242.444, 0.11),
    fill(4724.8, 4883, "Cut from the dub", "cut"),
    dub(4883, 6545.6, 4482.356, -400.644, 0.13),
    fill(6545.6, 6642, "Cut from the dub", "cut"),
    dub(6642, 7590.85, 6144.956, -497.044, 0.08),
    fill(7590.85, 7713, "Cut from the dub", "cut"),
    dub(7713, 8427.4, 7033.806, -619.194, 0.1),
    fill(8427.4, 8430, "Past the dub's end", "tail"),
  ];
}

const VOICE_PIECES: VoicePiece[] = [
  {
    dubStartS: 3457.556,
    dubEndS: 3500.056,
    levelS: 0.18,
    shiftS: 0.18,
    joinEnd: true,
    note: "The voices sit 0.180 s early against the lips from 0:57:37.556 to 0:58:20.056; moved later by 0.180 s.",
    videoStartS: 3700,
    videoEndS: 3742.5,
  },
  {
    dubStartS: 5699.356,
    dubEndS: 5749.756,
    levelS: -0.12,
    shiftS: -0.12,
    joinEnd: true,
    note: "The voices sit 0.120 s late against the lips from 1:34:59.356 to 1:35:49.756; moved earlier by 0.120 s.",
    videoStartS: 6100,
    videoEndS: 6150.4,
  },
];

function summarize(dub: number, segments: DubSegment[]): DubPlanSummary {
  const fillS: Record<string, number> = {};
  let used = 0;
  segments.forEach((s) => {
    const length = s.endS - s.startS;
    if (s.kind === "dub") used += length;
    else fillS[s.reason ?? "cut"] = Math.round(((fillS[s.reason ?? "cut"] ?? 0) + length) * 1000) / 1000;
  });
  return { dubUsedS: Math.round(used * 1000) / 1000, dubDurationS: dub, dubUsedShare: used / dub, fillS, doubtS: 0 };
}

function header(video: FileFx, dub: FileFx): Omit<DubSyncPlan, "segments" | "summary" | "filledS" | "warnings" | "notes" | "voicePieces"> {
  return {
    videoPath: video.path,
    dubPath: dub.path,
    videoTrack: 0,
    dubTrack: 0,
    speed: 1,
    videoFps: video.fps ?? NTSC_FILM,
    dubRate: video.fps ?? NTSC_FILM,
    rateConfirmed: true,
    fillGainDb: 4.4,
    videoDurationS: video.durationS,
    dubDurationS: dub.durationS,
    error: null,
    timeline: "container",
  };
}

/** The Skyline Heist plan of the mockup, whichever files it is asked for. */
export function skylinePlan(video: FileFx = SKYLINE_VIDEO, dub: FileFx = SKYLINE_DUB): DubSyncPlan {
  const segments = skylineSegments();
  return {
    ...header(video, dub),
    videoDurationS: 8430,
    dubDurationS: 7810,
    segments,
    warnings: [],
    notes: ["5 scenes are missing from the dub; the original plays over them (10m 20.0s)."],
    // The mockup's "From original 10m 22.2s".
    filledS: 622.2,
    summary: summarize(7810, segments),
    voicePieces: VOICE_PIECES,
  };
}

/** A believable plan for any other pair: a late start, one to three cut
 *  scenes, a tail. Deterministic in the file names. */
export function generatedPlan(video: FileFx, dub: FileFx, warn = false): DubSyncPlan {
  const key = video.name + dub.name;
  const head = Math.round((0.4 + unit(`${key}h`) * 1.2) * 1000) / 1000;
  const cuts = 1 + Math.floor(unit(`${key}n`) * 3);
  const segments: DubSegment[] = [segment({ kind: "fill", startS: 0, endS: head, note: "dub starts late", reason: "head" })];
  let t = head;
  // dub time minus video time; the dub's own start sits at `head`
  let offset = -head;
  for (let i = 0; i < cuts; i += 1) {
    const at = Math.round((video.durationS * (0.15 + (0.7 * (i + unit(`${key}c${i}`))) / cuts)) * 1000) / 1000;
    const length = Math.round((40 + unit(`${key}l${i}`) * 100) * 1000) / 1000;
    segments.push(segment({ kind: "dub", startS: t, endS: at, sourceStartS: t + offset, offsetS: Math.round(offset * 1000) / 1000, match: 0.08 + unit(`${key}m${i}`) * 0.08 }));
    segments.push(segment({ kind: "fill", startS: at, endS: at + length, note: "cut from the dub", reason: "cut" }));
    t = at + length;
    offset -= length;
  }
  const dubEnd = Math.min(video.durationS, dub.durationS - offset);
  segments.push(segment({ kind: "dub", startS: t, endS: dubEnd, sourceStartS: t + offset, offsetS: Math.round(offset * 1000) / 1000, match: 0.1 }));
  if (video.durationS - dubEnd > 0.05) {
    segments.push(segment({ kind: "fill", startS: dubEnd, endS: video.durationS, note: "past the dub's end", reason: "tail" }));
  }
  const filled = segments.filter((s) => s.kind === "fill").reduce((sum, s) => sum + s.endS - s.startS, 0);
  const first = segments.find((s) => s.kind === "dub");
  return {
    ...header(video, dub),
    segments,
    warnings: warn && first ? [`The stretch from ${formatClock(first.startS)} correlates weakly (match ${(first.match ?? 0).toFixed(2)}); listen to it.`] : [],
    notes: [],
    filledS: Math.round(filled * 1000) / 1000,
    summary: summarize(dub.durationS, segments),
  };
}

/** The plan as it stands mid-analysis: `placed` segments settled, each of
 *  the rest a fill marked not placed yet, as the engine's drafts carry the
 *  coarse stretches it has not placed. */
export function draftPlan(plan: DubSyncPlan, placed: number): DubSyncPlan {
  const done = plan.segments.slice(0, Math.max(0, placed));
  const rest: DubSegment[] = plan.segments
    .slice(done.length)
    .map((s) => segment({ kind: "fill", startS: s.startS, endS: s.endS, note: DRAFT_NOTE, reason: "draft", uncertaintyS: 0 }));
  return {
    ...plan,
    segments: [...done, ...rest],
    warnings: [],
    notes: [],
    filledS: 0,
    summary: undefined,
    voicePieces: undefined,
  };
}

export function describePlan(plan: DubSyncPlan): string {
  const dubs = plan.segments.filter((s) => s.kind === "dub").length;
  const cuts = plan.segments.filter((s) => s.kind === "fill" && s.reason === "cut").length;
  return `${dubs} stretch${dubs === 1 ? "" : "es"} of the dub, ${cuts} cut${cuts === 1 ? "" : "s"} filled from the original (${formatSpan(plan.filledS)} in all)`;
}

// ------------------------------------------------------------ verification

export function verification(plan: DubSyncPlan, typicalMs: number, worstMs: number): DubVerification {
  const spotCount = 12;
  const spots = Array.from({ length: spotCount }, (_, i) => {
    const positionS = Math.round(((plan.videoDurationS * (i + 0.5)) / spotCount) * 1000) / 1000;
    const inside = plan.segments.find((s) => positionS >= s.startS && positionS < s.endS);
    if (!inside || inside.kind === "fill") return { positionS, residualMs: null, match: 0, note: "filled from the original" };
    const sign = unit(`${plan.dubPath}${i}`) > 0.5 ? 1 : -1;
    return {
      positionS,
      residualMs: Math.round(sign * typicalMs * (0.4 + unit(`${plan.dubPath}r${i}`) * 1.2) * 100) / 100,
      match: Math.round((0.08 + unit(`${plan.dubPath}q${i}`) * 0.08) * 100) / 100,
      note: "",
    };
  });
  const windows = Math.round(plan.videoDurationS / 54);
  const measured = Math.round(windows * 0.76);
  const lines: DubLineCheck = {
    windows: Array.from({ length: 6 }, (_, i) => ({
      startS: Math.round(plan.videoDurationS * (0.1 + i * 0.14)),
      endS: Math.round(plan.videoDurationS * (0.1 + i * 0.14)) + 90,
      lagMs: Math.round((2 + unit(`${plan.dubPath}L${i}`) * 14) * 10) / 10,
      z: 6 + i,
      note: "",
    })),
    judged: 6,
    overallMs: 2.8,
    typicalMs: 9.1,
    worstMs: 16.0,
    withinTolerance: 6,
    toleranceMs: 45,
  };
  return {
    spots,
    typicalMs,
    worstMs,
    sweepWindows: windows,
    sweepMeasured: measured,
    sweepWithinAudible: measured,
    sweepWithin1Ms: measured - 1,
    sweepWithin5Ms: measured,
    sweepTypicalMs: typicalMs,
    sweepWorstMs: worstMs,
    stretches: [],
    lines,
  };
}

export function verificationText(v: DubVerification): string {
  const measured = v.spots.filter((s) => s.residualMs !== null).length;
  const lines = v.spots.map((s) => {
    if (s.residualMs === null) return `  at ${formatClock(s.positionS)}: filled from the original`;
    const early = s.residualMs < 0;
    return `  at ${formatClock(s.positionS)}: ${Math.abs(s.residualMs).toFixed(1)}ms ${early ? "early" : "late"}, match ${s.match.toFixed(2)}`;
  });
  return [
    `the finished track sits ${v.typicalMs?.toFixed(1)}ms from the original typically and ${v.worstMs?.toFixed(1)}ms at its worst, measured at ${measured} spots. Lip-sync starts to show around 100ms; the measurement resolves 2ms.`,
    ...lines,
    `swept the whole runtime: ${v.sweepMeasured} of ${v.sweepWindows} windows could be measured, and every one of them is within 100ms of the original`,
    "no stretch is more than 200ms out",
  ].join("\n");
}

// ------------------------------------------------------------------ output

const CODEC_EXT: Record<Exclude<DubCodec, "same">, string> = { flac: "flac", eac3: "eac3", ac3: "ac3", aac: "m4a", opus: "opus", wav: "wav" };

export function outputFor(plan: DubSyncPlan, dub: FileFx, codec: DubCodec, seconds: number): DubOutput {
  const ext = codec === "same" ? extOf(dub.name) : CODEC_EXT[codec];
  return {
    outputPath: `${dirName(plan.dubPath)}\\${stemOf(dub.name)}.dubsynced.${ext}`,
    sampleRate: 48000,
    channels: 6,
    seconds,
    clippedSamples: 0,
    warnings: [],
  };
}

export function muxPathFor(plan: DubSyncPlan): string {
  return `${dirName(plan.videoPath)}\\${stemOf(plan.videoPath.replace(/^.*[\\/]/, ""))}.dubsynced.mkv`;
}
