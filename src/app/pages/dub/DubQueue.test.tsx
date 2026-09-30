import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { dubQueueReducer, initialDubQueueState, type DubQueueState } from "@/lib/dubQueueReducer";
import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import type { DubJobOutcome, DubSyncOutcome, FileItem, MatchPair } from "@/lib/types";

import { DubQueue } from "./DubQueue";
import { buildRows, type QueueRow } from "./dubModel";

const single = outcomeFixture as unknown as DubSyncOutcome;

function outcome(job: number, patch: Partial<DubSyncOutcome> = {}): DubJobOutcome {
  const source = { ...single, ...patch };
  return { job, plan: source.plan, output: source.output, verification: source.verification, muxedPath: source.muxedPath ?? null, cancelled: source.cancelled, error: source.error ?? null };
}

const file = (path: string, type: "video" | "audio"): FileItem => ({ id: path, path, name: path.replace(/^.*\//, ""), type });
const pair = (video: string, dub: string): MatchPair => ({
  primaryPath: video,
  secondaryPath: dub,
  primaryName: video.replace(/^.*\//, ""),
  secondaryName: dub.replace(/^.*\//, ""),
  key: video,
  method: "episode",
  score: 1,
});

function queued(names: [string, string][]): DubQueueState {
  return dubQueueReducer(initialDubQueueState, {
    type: "queueStarted",
    jobs: names.map(([name, dubName]) => ({ name, dubName, videoPath: `/v/${name}`, dubPath: `/d/${dubName}` })),
  });
}

const noop = () => undefined;

function render(rows: QueueRow[], { dubs = [] as FileItem[], repair = true, selected = rows[0]?.key ?? null } = {}) {
  return renderToStaticMarkup(<DubQueue rows={rows} selected={selected} onSelect={noop} dubs={dubs} onRepair={repair ? noop : undefined} />);
}

const running = (state: DubQueueState) => buildRows({ running: true, jobs: state.jobs, pairs: [], videos: [], dubs: [], overrides: {}, trackChoices: {} });

describe("DubQueue", () => {
  it("lays out the mockup's table: pair, sync error, status", () => {
    const html = render(running(queued([["Skyline Heist (2023) 1080p BluRay.mkv", "Skyline Heist (2023) Hindi TV rip.ac3"]])));
    expect(html).toContain(">Queue<");
    expect(html).toContain(">1 pair<");
    expect(html).toContain("--cols:minmax(0,1fr) 84px 190px");
    expect(html).toMatch(/columnheader"[^>]*>Pair<.*columnheader" class="r">Sync error<.*columnheader"[^>]*>Status</);
    // Two lines a row: the movie, and its dub under it.
    expect(html).toContain("height:48px");
    expect(html).toContain('title="Skyline Heist (2023) 1080p BluRay.mkv"');
    expect(html).toContain('title="Skyline Heist (2023) Hindi TV rip.ac3"');
  });

  it("shows the stage and percent while a job is running, before any plan exists", () => {
    let state = queued([["Show.S01E01.mkv", "Show.S01E01.dub.eac3"]]);
    state = dubQueueReducer(state, { type: "jobStart", job: 0 });
    state = dubQueueReducer(state, { type: "jobProgress", job: 0, percent: 40, stage: "finding the offsets" });
    const html = render(running(state), { repair: false });
    expect(html).toContain("Show.S01E01.mkv");
    expect(html).toContain("Finding where the dub belongs");
    expect(html).toContain('aria-valuenow="40"');
    // No error measured yet.
    expect(html).toContain('<span class="t3">—</span>');
  });

  it("shows every job of a season, each with its own state", () => {
    let state = queued([
      ["Show.S01E01.mkv", "Show.S01E01.dub.eac3"],
      ["Show.S01E02.mkv", "Show.S01E02.dub.eac3"],
      ["Show.S01E03.mkv", "Show.S01E03.dub.eac3"],
    ]);
    state = dubQueueReducer(state, { type: "jobDone", outcome: outcome(0) });
    state = dubQueueReducer(state, {
      type: "jobDone",
      outcome: { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "No part of the dub could be matched" },
    });
    const html = render(running(state), { repair: false });
    expect(html).toContain(">3 pairs<");
    expect(html).toContain("Show.S01E03.mkv");
    // Job 0 measured and done, job 1 failed with its reason, job 2 waiting.
    expect(html).toContain("0.2 ms");
    expect(html).toContain(">Done<");
    expect(html).toContain(">No match<");
    expect(html).toContain('title="No part of the dub could be matched');
    expect(html).toContain(">Waiting<");
  });

  it("says a done job has stretches to check", () => {
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0, { verification: { ...single.verification!, stretches: [{ startS: 1285, endS: 1311, residualMs: 400, windows: 3 }] } }),
    });
    expect(render(running(state))).toContain("Done · 1 stretch to check");
  });

  it("offers a box to choose a dub for a movie the engine could not pair", () => {
    const videos = [file("/v/E07.mkv", "video"), file("/v/E08.mkv", "video")];
    const dubs = [file("/d/E07.eac3", "audio"), file("/d/Special.eac3", "audio")];
    const rows = buildRows({ running: false, jobs: [], pairs: [pair("/v/E07.mkv", "/d/E07.eac3")], videos, dubs, overrides: {}, trackChoices: {} });
    const html = render(rows, { dubs });
    expect(html).toContain('aria-label="Dub for E08.mkv"');
    expect(html).toContain(">Choose a dub<");
    expect(html).toContain(">Not paired<");
    expect(html).toContain(">Special.eac3<");
    expect(html).toContain(">Skip this movie<");
  });

  it("lets a paired dub be changed by hand, but not while the queue runs", () => {
    const videos = [file("/v/E07.mkv", "video")];
    const dubs = [file("/d/E07.eac3", "audio"), file("/d/Other.eac3", "audio")];
    const rows = buildRows({ running: false, jobs: [], pairs: [pair("/v/E07.mkv", "/d/E07.eac3")], videos, dubs, overrides: {}, trackChoices: {} });
    const html = render(rows, { dubs });
    expect(html).toMatch(/<select class="dubq-native" aria-label="Dub for E07.mkv"/);
    expect(html).toContain(">Ready<");
    expect(html).not.toContain(">Pair automatically<");
    expect(render(rows, { dubs, repair: false })).not.toContain("<select");
    // A pair made by hand can go back to the engine's.
    const manual = buildRows({ running: false, jobs: [], pairs: [pair("/v/E07.mkv", "/d/Other.eac3")], videos, dubs, overrides: { "/v/E07.mkv": "/d/Other.eac3" }, trackChoices: {} });
    expect(render(manual, { dubs })).toContain(">Pair automatically<");
  });

  it("marks the selected row", () => {
    const html = render(running(queued([["a.mkv", "a.ac3"], ["b.mkv", "b.ac3"]])), { selected: "/v/b.mkv" });
    expect(html.match(/class="tr on"/g)).toHaveLength(1);
    expect(html).toMatch(/class="tr on"[^>]*aria-label="b.mkv"/);
  });

  it("lists dubs waiting for their movies", () => {
    const rows = buildRows({ running: false, jobs: [], pairs: null, videos: [], dubs: [file("/d/x.ac3", "audio")], overrides: {}, trackChoices: {} });
    const html = render(rows);
    expect(html).toContain("No movie yet");
    expect(html).toContain("Needs its movie");
  });
});
