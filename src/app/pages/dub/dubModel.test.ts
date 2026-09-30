import { describe, expect, it } from "vitest";

import { dubQueueReducer, initialDubQueueState, type DubQueueState } from "@/lib/dubQueueReducer";
import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import type { DubJobOutcome, DubSyncOutcome, DubSyncPlan, DubVerification, FileItem, MatchPair } from "@/lib/types";

import {
  batchSummary,
  buildRows,
  collectionTitle,
  formatOffset,
  gapNote,
  jobStatus,
  pageQueueReducer,
  pageStatus,
  parseOffset,
  planFacts,
  planLine,
  rowStatus,
  rulerTicks,
  shortReason,
  failWord,
  shortTitle,
  stageIndex,
  stageLabel,
  stagesFor,
  stretchName,
  timeLeft,
  verdict,
  verdictLine,
  type StatusInput,
} from "./dubModel";

/** The engine's real answer for the five-minute fixture: stereo AAC in an
 *  MKV against a raw 5.1 AC3 dub with cuts, captured from the bridge. */
const single = outcomeFixture as unknown as DubSyncOutcome;
const plan = single.plan as DubSyncPlan;
const check = single.verification as DubVerification;

function outcome(job: number, patch: Partial<DubSyncOutcome> = {}): DubJobOutcome {
  const source = { ...single, ...patch };
  return { job, plan: source.plan, output: source.output, verification: source.verification, muxedPath: source.muxedPath ?? null, cancelled: source.cancelled, error: source.error ?? null };
}

function queued(names: [string, string][]): DubQueueState {
  return dubQueueReducer(initialDubQueueState, {
    type: "queueStarted",
    jobs: names.map(([name, dubName]) => ({ name, dubName, videoPath: `/v/${name}`, dubPath: `/d/${dubName}` })),
  });
}

const file = (path: string, type: "video" | "audio"): FileItem => ({ id: path, path, name: path.replace(/^.*\//, ""), type });
const pair = (video: string, dub: string, method = "episode"): MatchPair => ({
  primaryPath: video,
  secondaryPath: dub,
  primaryName: video.replace(/^.*\//, ""),
  secondaryName: dub.replace(/^.*\//, ""),
  key: video,
  method,
  score: 1,
});

/** A check the verdict calls good: tight, and the whole sweep within bounds. */
const good: DubVerification = { ...check, sweepMeasured: 20, sweepWithinAudible: 20, stretches: [] };

describe("names", () => {
  it("keeps the title and drops the release", () => {
    expect(shortTitle("Skyline Heist (2023) 1080p BluRay.mkv")).toBe("Skyline Heist (2023)");
    expect(shortTitle("Skyline Heist (2023) Hindi TV rip.ac3")).toBe("Skyline Heist (2023)");
    expect(shortTitle("Goblin.S01E04.1080p.BluRay.mkv")).toBe("Goblin.S01E04");
    expect(shortTitle("Goblin.S01E01.hin.eac3")).toBe("Goblin.S01E01");
    expect(shortTitle("C:\\Films\\video.mkv")).toBe("video");
  });

  it("names a season by its series, and other lists by their first", () => {
    const season = ["Goblin.S01E01.1080p.BluRay.mkv", "Goblin.S01E02.1080p.BluRay.mkv", "Goblin.S01E03.1080p.BluRay.mkv"];
    expect(collectionTitle(season)).toBe("Goblin · Season 1");
    expect(collectionTitle(["Goblin.S01E01.mkv", "Goblin.S02E01.mkv"])).toBe("Goblin");
    expect(collectionTitle(["Skyline Heist (2023) 1080p.mkv", "Dune (2021).mkv", "Heat.mkv"])).toBe("Skyline Heist (2023) and 2 more");
    expect(collectionTitle(["Skyline Heist (2023) 1080p BluRay.mkv"])).toBe("Skyline Heist (2023)");
  });

  it("shortens an engine error for a status cell", () => {
    expect(shortReason("The dub ends 38 minutes before the video. It is probably a different edit.")).toBe("The dub ends 38 minutes before the video");
    expect(shortReason("no part of the dub could be matched")).toBe("No part of the dub could be matched");
    expect(shortReason(null)).toBe("Failed");
  });

  it("names the failures the engine names in a word or two", () => {
    expect(failWord("Goblin.S01E04: the dub ends 38 minutes before the video (a different edit?)")).toBe("Different edit");
    expect(failWord("No part of the dub could be matched to the video")).toBe("No match");
    expect(failWord("The dub's audio (x.ac3) is silent from start to end. It may be an empty or muted track.")).toBe("Silent audio");
    expect(failWord("x.ac3 has no audio")).toBe("No audio");
    expect(failWord("x.ac3 is only 1.5 s long; laying a dub on a video needs at least 30 s of audio.")).toBe("Too short");
    expect(failWord("ffmpeg exited with code 1: bad header")).toBe("Ffmpeg exited with code 1");
  });
});

describe("stages", () => {
  it("maps every engine stage onto the checklist, in the engine's order", () => {
    const labels = stagesFor(false).map((s) => s.label);
    expect(labels).toHaveLength(9);
    expect(stageLabel("probing")).toBe("Reading the files");
    expect(stageLabel("reading the dub")).toBe("Reading the files");
    expect(stageLabel("finding the offsets")).toBe("Finding where the dub belongs");
    expect(stageLabel("placing the cuts")).toBe("Placing the cuts");
    expect(stageLabel("writing dub.dubsynced.ac3")).toBe("Writing the track");
    expect(stageLabel("writing the track with the edited cuts")).toBe("Writing the track");
    expect(stageLabel("checking the finished track")).toBe("Checking against the video");
    expect(stageIndex("measuring the offsets")).toBeLessThan(stageIndex("placing the cuts"));
    expect(stageIndex("writing x")).toBeLessThan(stageIndex("checking the finished track"));
  });

  it("adds the mux stage only when the track goes into a copy of the video", () => {
    expect(stagesFor(true)).toHaveLength(10);
    expect(stageLabel("muxing")).toBe("Adding the track to a copy of the video");
    expect(stageIndex("muxing", stagesFor(false))).toBe(-1);
  });

  it("says an unknown stage the way the engine does", () => {
    expect(stageLabel("thinking hard")).toBe("Thinking hard");
    expect(stageLabel(null)).toBe("Starting…");
  });
});

describe("plans", () => {
  it("counts what the plan does", () => {
    const facts = planFacts(plan);
    expect(facts.stretches).toBe("4 of dub");
    // Two scenes the dub lacks; the head and tail fills are not cuts.
    expect(facts.cuts).toBe(2);
    expect(facts.fromOriginal).toBe("21.5s · 2 cuts");
    expect(facts.fillLevel).toBe("+4.4 dB");
    expect(facts.voices).toBe("None");
    expect(planLine(plan)).toBe("4 stretches · 2 cuts filled from the original");
  });

  it("gives the frame-rate verdict", () => {
    expect(planFacts({ ...plan, videoFps: 24000 / 1001, dubRate: 24000 / 1001 }).frameRate).toBe("23.976 fps, both");
    expect(planFacts({ ...plan, videoFps: 24000 / 1001, dubRate: 25, rateConfirmed: false }).frameRate).toBe("23.976 fps, dub 25 (assumed)");
  });

  it("counts a stretch among its own kind", () => {
    expect(stretchName(plan, 3)).toBe("Dub stretch 2 of 4");
    expect(stretchName(plan, 2)).toBe("Original stretch 2 of 4");
  });

  it("writes and reads offsets with a real minus sign", () => {
    expect(formatOffset(-400.644)).toBe("−400.644");
    expect(formatOffset(0.806)).toBe("+0.806");
    expect(formatOffset(-0)).toBe("+0.000");
    expect(parseOffset("−400.644")).toBe(-400.644);
    expect(parseOffset(" -1,5 s")).toBe(-1.5);
    expect(parseOffset("+0.806")).toBe(0.806);
    expect(parseOffset("abc")).toBeNull();
    expect(parseOffset("")).toBeNull();
  });
});

describe("verdict", () => {
  it("does not call the fixture good: a tenth of its sweep is out", () => {
    expect(verdict(check)).toEqual({ tone: "warn", share: 90, audible: 0 });
    expect(verdictLine(check)).toBe("0.2 ms typical, 0.4 ms at worst · 90% of the runtime within 45 ms");
  });

  it("calls a tight track good", () => {
    expect(verdict(good).tone).toBe("ok");
    expect(verdictLine(good)).toBe("0.2 ms typical, 0.4 ms at worst");
  });

  it("puts a stretch audibly out first", () => {
    const out = { ...check, stretches: [{ startS: 1285, endS: 1311, residualMs: 400, windows: 3 }] };
    expect(verdict(out).tone).toBe("bad");
    expect(verdictLine(out)).toBe("1 stretch audibly out · 90% within 45 ms");
  });

  it("calls half a sweep out bad even with tight spots", () => {
    expect(verdict({ ...check, sweepMeasured: 181, sweepWithinAudible: 92 }).tone).toBe("bad");
  });
});

describe("rows", () => {
  const videos = [file("/v/E01.mkv", "video"), file("/v/E02.mkv", "video"), file("/v/E03.mkv", "video")];
  const dubs = [file("/d/E01.eac3", "audio"), file("/d/E02.eac3", "audio")];
  const pairs = [pair("/v/E01.mkv", "/d/E01.eac3"), pair("/v/E02.mkv", "/d/E02.eac3")];

  it("lists every movie, paired or not, in the order they were added", () => {
    const rows = buildRows({ running: false, jobs: [], pairs, videos, dubs, overrides: {}, trackChoices: {} });
    expect(rows.map((r) => [r.videoName, r.dubName, r.paired])).toEqual([
      ["E01.mkv", "E01.eac3", true],
      ["E02.mkv", "E02.eac3", true],
      ["E03.mkv", null, false],
    ]);
    expect(rowStatus(rows[0], dubs.length)).toEqual({ s: "ready" });
    expect(rowStatus(rows[2], dubs.length)).toEqual({ s: "bad", text: "Not paired" });
  });

  it("joins the last run's job to the same movie and dub only", () => {
    let queue = dubQueueReducer(initialDubQueueState, {
      type: "queueStarted",
      jobs: [{ name: "E01.mkv", dubName: "E01.eac3", videoPath: "/v/E01.mkv", dubPath: "/d/E01.eac3" }],
    });
    queue = dubQueueReducer(queue, { type: "batchDone", outcomes: [outcome(0)], cancelled: false });
    const rows = buildRows({ running: false, jobs: queue.jobs, pairs, videos, dubs, overrides: {}, trackChoices: {} });
    expect(rows[0].job?.status).toBe("done");
    expect(rows[1].job).toBeNull();
    // Paired by hand to another dub: the result no longer describes it.
    const repaired = buildRows({ running: false, jobs: queue.jobs, pairs: [pair("/v/E01.mkv", "/d/E02.eac3")], videos, dubs, overrides: { "/v/E01.mkv": "/d/E02.eac3" }, trackChoices: {} });
    expect(repaired[0].job).toBeNull();
    expect(repaired[0].manual).toBe(true);
  });

  it("shows the queue's own jobs while it runs", () => {
    const queue = queued([["a.mkv", "a.ac3"], ["b.mkv", "b.ac3"]]);
    const rows = buildRows({ running: true, jobs: queue.jobs, pairs: [], videos: [], dubs: [], overrides: {}, trackChoices: {} });
    expect(rows.map((r) => r.key)).toEqual(["/v/a.mkv", "/v/b.mkv"]);
    expect(rowStatus(rows[0], 2)).toEqual({ s: "wait" });
  });

  it("says what is missing before anything pairs", () => {
    const pending = buildRows({ running: false, jobs: [], pairs: null, videos, dubs, overrides: {}, trackChoices: {} });
    expect(rowStatus(pending[0], 2)).toEqual({ s: "wait", text: "Pairing…" });
    const dubsOnly = buildRows({ running: false, jobs: [], pairs: null, videos: [], dubs, overrides: {}, trackChoices: {} });
    expect(rowStatus(dubsOnly[0], 2)).toEqual({ s: "warn", text: "Needs its movie" });
    const skipped = buildRows({ running: false, jobs: [], pairs: [], videos, dubs, overrides: { "/v/E01.mkv": null }, trackChoices: {} });
    expect(rowStatus(skipped[0], 2)).toEqual({ s: "wait", text: "Skipped" });
    const noDubs = buildRows({ running: false, jobs: [], pairs: null, videos, dubs: [], overrides: {}, trackChoices: {} });
    expect(rowStatus(noDubs[0], 0)).toEqual({ s: "wait", text: "Needs a dub" });
  });

  it("carries a chosen audio stream onto the pair", () => {
    const rows = buildRows({ running: false, jobs: [], pairs, videos, dubs, overrides: {}, trackChoices: { "/v/E01.mkv": 2, "/d/E01.eac3": 1 } });
    expect([rows[0].videoTrack, rows[0].dubTrack]).toEqual([2, 1]);
  });

  it("puts each job in the one status vocabulary", () => {
    let queue = queued([["a.mkv", "a.ac3"]]);
    queue = dubQueueReducer(queue, { type: "jobStart", job: 0 });
    queue = dubQueueReducer(queue, { type: "jobProgress", job: 0, percent: 42, stage: "placing the cuts" });
    expect(jobStatus(queue.jobs[0])).toEqual({ s: "run", pct: 42, text: "Placing the cuts" });
    const done = dubQueueReducer(queue, { type: "jobDone", outcome: outcome(0) });
    expect(jobStatus(done.jobs[0])).toEqual({ s: "ok", text: "Done" });
    const flagged = dubQueueReducer(queue, {
      type: "jobDone",
      outcome: outcome(0, { verification: { ...check, stretches: [{ startS: 1, endS: 9, residualMs: 80, windows: 2 }] } }),
    });
    expect(jobStatus(flagged.jobs[0])).toEqual({ s: "ok", text: "Done · 1 stretch to check" });
    const failed = dubQueueReducer(queue, {
      type: "jobDone",
      outcome: { job: 0, plan: null, output: null, verification: null, muxedPath: null, error: "the dub is a different edit. Nothing was written." },
    });
    expect(jobStatus(failed.jobs[0])).toEqual({ s: "bad", text: "Different edit", title: "The dub is a different edit. Nothing was written." });
  });
});

describe("status display", () => {
  const base: StatusInput = {
    rows: [],
    queue: initialDubQueueState,
    shown: null,
    videos: 0,
    dubs: 0,
    pairingLoading: false,
    lengths: null,
    method: null,
    manual: 0,
    edit: null,
    workers: 3,
    now: Date.now(),
  };
  const videos = [file("/v/Skyline Heist (2023) 1080p BluRay.mkv", "video")];
  const dubs = [file("/d/Skyline Heist (2023) Hindi TV rip.ac3", "audio")];
  const pairs = [pair(videos[0].path, dubs[0].path)];
  const readyRows = buildRows({ running: false, jobs: [], pairs, videos, dubs, overrides: {}, trackChoices: {} });

  it("asks for files on an empty page", () => {
    expect(pageStatus(base)).toEqual({ l1: "Drop movies and their dubs" });
  });

  it("says one pair is ready and how the lengths compare", () => {
    const status = pageStatus({ ...base, rows: readyRows, shown: readyRows[0], videos: 1, dubs: 1, lengths: { videoS: 8430, dubS: 7810 } });
    expect(status).toEqual({ l1: "1 pair ready", l2: "The dub is 10m 20s shorter than the video" });
  });

  it("says how several pairs were paired", () => {
    const vs = [file("/v/E01.mkv", "video"), file("/v/E02.mkv", "video")];
    const ds = [file("/d/E01.eac3", "audio"), file("/d/E02.eac3", "audio")];
    const rows = buildRows({ running: false, jobs: [], pairs: [pair(vs[0].path, ds[0].path), pair(vs[1].path, ds[1].path)], videos: vs, dubs: ds, overrides: {}, trackChoices: {} });
    expect(pageStatus({ ...base, rows, videos: 2, dubs: 2, method: "episode (mixed notation)", manual: 1 })).toEqual({
      l1: "2 pairs ready",
      l2: "Paired by episode number · 1 by hand",
    });
  });

  it("follows a single run's stage, with its percent", () => {
    let queue = dubQueueReducer(initialDubQueueState, { type: "queueStarted", jobs: [{ name: videos[0].name, dubName: dubs[0].name, videoPath: videos[0].path, dubPath: dubs[0].path }] });
    queue = dubQueueReducer(queue, { type: "jobStart", job: 0 });
    queue = dubQueueReducer(queue, { type: "jobProgress", job: 0, percent: 61, stage: "placing the cuts" });
    const rows = buildRows({ running: true, jobs: queue.jobs, pairs, videos, dubs, overrides: {}, trackChoices: {} });
    expect(pageStatus({ ...base, rows, queue, shown: rows[0], videos: 1, dubs: 1 })).toEqual({
      icon: "run",
      l1: "Placing the cuts",
      l2: "Skyline Heist (2023)",
      pct: 61,
      time: "61%",
    });
  });

  it("counts a season's run and how many run at once", () => {
    const names: [string, string][] = [1, 2, 3, 4, 5].map((n) => [`Goblin.S01E0${n}.1080p.BluRay.mkv`, `Goblin.S01E0${n}.hin.eac3`]);
    let queue = queued(names);
    queue = dubQueueReducer(queue, { type: "jobDone", outcome: outcome(0) });
    queue = dubQueueReducer(queue, { type: "jobDone", outcome: outcome(1) });
    queue = dubQueueReducer(queue, { type: "jobStart", job: 2 });
    queue = dubQueueReducer(queue, { type: "jobProgress", job: 2, percent: 50, stage: "placing the cuts" });
    const rows = buildRows({ running: true, jobs: queue.jobs, pairs: [], videos: [], dubs: [], overrides: {}, trackChoices: {} });
    const status = pageStatus({ ...base, rows, queue, shown: rows[2], videos: 5, dubs: 5 });
    expect(status.l1).toBe("Syncing 3 of 5 · Placing the cuts");
    expect(status.l2).toBe("Goblin · Season 1 · 3 at a time");
    expect(status.pct).toBeCloseTo(50);
  });

  it("puts edits not written yet first", () => {
    const status = pageStatus({
      ...base,
      rows: readyRows,
      videos: 1,
      dubs: 1,
      edit: { changes: 1, stretch: plan.segments[3], stretchName: "Dub stretch 2 of 4" },
    });
    expect(status).toEqual({ icon: "warn", l1: "1 change not applied", l2: "Dub stretch 2 of 4 · 0:01:15.010 – 0:02:59.996" });
  });

  it("gives a single result's verdict, and the plan in one line", () => {
    let queue = dubQueueReducer(initialDubQueueState, { type: "queueStarted", jobs: [{ name: videos[0].name, dubName: dubs[0].name, videoPath: videos[0].path, dubPath: dubs[0].path }] });
    queue = dubQueueReducer(queue, { type: "batchDone", outcomes: [outcome(0, { verification: good })], cancelled: false });
    const rows = buildRows({ running: false, jobs: queue.jobs, pairs, videos, dubs, overrides: {}, trackChoices: {} });
    expect(pageStatus({ ...base, rows, queue, shown: rows[0], videos: 1, dubs: 1 })).toEqual({
      icon: "ok",
      l1: "On the lips — 0.2 ms typical, 0.4 ms at worst",
      l2: "4 stretches · 2 cuts filled from the original",
    });
    // The fixture's own check is not good enough to say so.
    const warned = dubQueueReducer(queue, { type: "batchDone", outcomes: [outcome(0)], cancelled: false });
    const rows2 = buildRows({ running: false, jobs: warned.jobs, pairs, videos, dubs, overrides: {}, trackChoices: {} });
    expect(pageStatus({ ...base, rows: rows2, queue: warned, shown: rows2[0], videos: 1, dubs: 1 }).icon).toBe("warn");
  });

  it("names the first failure of a queue", () => {
    let queue = queued([
      ["Goblin.S01E03.mkv", "Goblin.S01E03.hin.eac3"],
      ["Goblin.S01E04.mkv", "Goblin.S01E04.hin.eac3"],
    ]);
    queue = dubQueueReducer(queue, {
      type: "batchDone",
      outcomes: [outcome(0), { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "The dub is a different edit" }],
      cancelled: false,
    });
    const vs = queue.jobs.map((j) => file(j.videoPath, "video"));
    const ds = queue.jobs.map((j) => file(j.dubPath, "audio"));
    const ps = queue.jobs.map((j) => pair(j.videoPath, j.dubPath));
    const rows = buildRows({ running: false, jobs: queue.jobs, pairs: ps, videos: vs, dubs: ds, overrides: {}, trackChoices: {} });
    expect(pageStatus({ ...base, rows, queue, videos: 2, dubs: 2 })).toEqual({
      icon: "bad",
      l1: "1 synced · 1 failed",
      l2: "Goblin.S01E04: the dub is a different edit",
    });
  });

  it("says when a batch did not finish at all", () => {
    const queue = pageQueueReducer(queued([["a.mkv", "a.ac3"]]), { type: "batchFailed", message: "The analysis engine stopped responding." });
    expect(pageStatus({ ...base, queue, videos: 1, dubs: 1 })).toEqual({ icon: "bad", l1: "The sync did not finish", l2: "The analysis engine stopped responding." });
  });

  it("counts the season's jobs started, several at a time", () => {
    const names: [string, string][] = [1, 2, 3, 4, 5].map((n) => [`Goblin.S01E0${n}.mkv`, `Goblin.S01E0${n}.hin.eac3`]);
    let queue = queued(names);
    for (const job of [0, 1, 2]) queue = dubQueueReducer(queue, { type: "jobStart", job });
    const rows = buildRows({ running: true, jobs: queue.jobs, pairs: [], videos: [], dubs: [], overrides: {}, trackChoices: {} });
    expect(pageStatus({ ...base, rows, queue, videos: 5, dubs: 5 }).l1).toBe("Syncing 3 of 5 · Reading the files");
  });

  it("follows a retried job on its own", () => {
    let queue = queued([["Goblin.S01E03.mkv", "a"], ["Goblin.S01E04.1080p.mkv", "b"]]);
    queue = dubQueueReducer(queue, { type: "batchDone", outcomes: [outcome(0), { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "x" }], cancelled: false });
    queue = pageQueueReducer(queue, { type: "retryStarted", jobs: [{ index: 1 }] });
    queue = pageQueueReducer(queue, { type: "jobStart", job: 1 });
    queue = pageQueueReducer(queue, { type: "jobProgress", job: 1, percent: 33, stage: "finding the offsets" });
    const rows = buildRows({ running: true, jobs: queue.jobs, pairs: [], videos: [], dubs: [], overrides: {}, trackChoices: {} });
    expect(pageStatus({ ...base, rows, queue, videos: 2, dubs: 2, runJobs: [1] })).toEqual({
      icon: "run",
      l1: "Finding where the dub belongs",
      l2: "Goblin.S01E04",
      pct: 33,
      time: "33%",
    });
  });

  it("guesses the time left from how far the run has come", () => {
    const t0 = 1_000_000;
    expect(timeLeft(t0, 0.5, t0 + 6 * 60_000)).toBe("6 min left");
    expect(timeLeft(t0, 0.9, t0 + 60_000)).toBe("Under a minute left");
    expect(timeLeft(null, 0.5, t0)).toBeUndefined();
    expect(timeLeft(t0, 0.01, t0 + 1000)).toBeUndefined();
  });

  it("compares lengths the way people say them", () => {
    expect(gapNote(100, 100.2)).toBe("The dub and the video are the same length");
    expect(gapNote(100, 160)).toBe("The dub is 1m 0s longer than the video");
    expect(gapNote(8430, 4200)).toBe("The dub is 1h 10m shorter than the video");
  });
});

describe("history", () => {
  it("records what a queue did", () => {
    const names = [{ name: "Goblin.S01E01.mkv" }, { name: "Goblin.S01E02.mkv" }];
    expect(batchSummary(names, [outcome(0), outcome(1)], false)).toEqual({ name: "Goblin · Season 1", tone: "ok", text: "2 synced" });
    const failed = { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "the dub is a different edit" };
    expect(batchSummary(names, [outcome(0), failed], false)).toEqual({ name: "Goblin · Season 1", tone: "warn", text: "1 synced, 1 failed" });
    expect(batchSummary([names[0]], [failed], false)).toEqual({ name: "Goblin.S01E01", tone: "bad", text: "Failed — different edit" });
    expect(batchSummary(names, [outcome(0), { ...failed, error: null, cancelled: true }], true).text).toBe("Stopped, 1 synced");
  });
});

describe("ruler", () => {
  it("labels a whole film in hours and minutes, ten ticks at most", () => {
    const ticks = rulerTicks(0, 8430, 1300);
    expect(ticks.map((t) => t.label)).toEqual(["0:00", "0:20", "0:40", "1:00", "1:20", "1:40", "2:00"]);
  });

  it("labels a zoomed view to the second, and not at its right edge", () => {
    const ticks = rulerTicks(4650, 4960, 1300);
    expect(ticks.map((t) => t.label)).toEqual(["1:18:00", "1:19:00", "1:20:00", "1:21:00", "1:22:00"]);
    expect(rulerTicks(10, 12, 800).every((t) => /\.\d$/.test(t.label))).toBe(true);
  });

  it("spaces the ticks for a narrow timeline", () => {
    expect(rulerTicks(0, 300, 300).length).toBeLessThanOrEqual(4);
  });
});

describe("pageQueueReducer", () => {
  it("runs one finished job again and keeps the others' results", () => {
    let queue = queued([["a.mkv", "a.ac3"], ["b.mkv", "b.ac3"]]);
    queue = dubQueueReducer(queue, {
      type: "batchDone",
      outcomes: [outcome(0), { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "different edit" }],
      cancelled: false,
    });
    const retried = pageQueueReducer(queue, { type: "retryStarted", jobs: [{ index: 1, dubPath: "/d/b2.ac3", dubName: "b2.ac3" }] });
    expect(retried.status).toBe("running");
    expect(retried.jobs[0].status).toBe("done");
    expect(retried.jobs[1]).toMatchObject({ status: "queued", dubPath: "/d/b2.ac3", dubName: "b2.ac3", error: null });
    // The engine's events for the retried job land on it.
    const started = pageQueueReducer(retried, { type: "jobStart", job: 1 });
    expect(started.jobs[1].status).toBe("running");
    // Not while something runs.
    expect(pageQueueReducer(retried, { type: "retryStarted", jobs: [{ index: 0 }] })).toBe(retried);
  });

  it("closes the jobs a failed batch never finished", () => {
    let queue = queued([["a.mkv", "a.ac3"], ["b.mkv", "b.ac3"]]);
    queue = dubQueueReducer(queue, { type: "jobDone", outcome: outcome(0) });
    const failed = pageQueueReducer(queue, { type: "batchFailed", message: "engine gone" });
    expect(failed.status).toBe("failed");
    expect(failed.jobs[0].status).toBe("done");
    expect(failed.jobs[1]).toMatchObject({ status: "failed", error: "engine gone" });
    expect(failed.done).toBe(2);
  });
});
