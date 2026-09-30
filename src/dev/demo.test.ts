/** Smoke test of the demo mocks through the real Tauri JS API, so a change
 *  to a command name or payload shape breaks here and not in a browser. */

import { beforeAll, describe, expect, it } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";

import type { WaveformPeaks } from "@/lib/api";
import type { AnalyzeRequest, DubSyncBatchResult, PairingReport, PickResponse, SyncRun } from "@/lib/types";

import { installDemo } from "./demo";

const shell = globalThis as unknown as Record<string, unknown>;

beforeAll(() => {
  shell.window = globalThis;
  shell.location = { search: "?demo=movies&speed=0.005&update=1" };
  shell.localStorage = { removeItem: () => undefined };
  installDemo("movies");
});

const record = async (events: string[]) => {
  const seen: { event: string; payload: unknown }[] = [];
  for (const event of events) await listen(event, (e) => seen.push({ event, payload: e.payload }));
  return seen;
};

const request = (over: Partial<AnalyzeRequest>): AnalyzeRequest => ({
  mode: "movie",
  videoFolder: null,
  audioFolder: null,
  audioFile: null,
  videoFiles: [],
  audioFiles: [],
  matchPattern: null,
  videoTrack: 0,
  audioTrack: 0,
  windowSeconds: 45,
  windowCount: 6,
  maxOffsetMs: 60000,
  maxWorkers: 3,
  findCuts: false,
  timeline: false,
  findSpeed: true,
  ...over,
});

describe("demo shell", () => {
  it("marks the window as the desktop shell", () => {
    expect("__TAURI_INTERNALS__" in globalThis).toBe(true);
    expect(getCurrentWindow().label).toBe("main");
  });

  it("analyses the six releases as the mockup shows them", async () => {
    const seen = await record(["sync-result", "sync-progress", "sync-pairs"]);
    const videos = await invoke<PickResponse>("pick_video_folder");
    const dub = await invoke<PickResponse>("pick_audio_file");
    expect(videos.files).toHaveLength(6);
    const run = await invoke<SyncRun>("start_sync", {
      request: request({ videoFiles: videos.files.map((f) => f.path), audioFiles: [dub.files[0].path], audioFile: dub.files[0].path }),
    });
    expect(run.cancelled).toBe(false);
    expect(run.results.map((r) => r.delayMs)).toEqual([317, 317, -1208, 84, 300, null]);
    expect(run.results.map((r) => r.confidence)).toEqual([0.94, 0.93, 0.68, 0.88, 0.31, null]);
    expect(run.results[2].hasSignificantDrift).toBe(true);
    expect(run.results[3].rateDiagnosis?.isRateMismatch).toBe(true);
    expect(run.results[4].isLikelyCut).toBe(true);
    expect(run.results[5].error).toBe("The video has no audio track to compare against.");
    expect(run.summary).toMatchObject({ total: 6, failed: 1, drifting: 1, cuts: 1, rateMismatches: 1 });
    expect(seen.filter((e) => e.event === "sync-result")).toHaveLength(6);
    expect(seen.filter((e) => e.event === "sync-progress")).toHaveLength(6);
    expect(seen.some((e) => e.event === "sync-pairs")).toBe(true);
  });

  it("pairs a season by episode, with the list-order and unmatched cases", async () => {
    const dir = "D:\\Media\\Goblin (2016)";
    const videos = Array.from({ length: 16 }, (_, i) => `${dir}\\Goblin.S01E${String(i + 1).padStart(2, "0")}.1080p.BluRay.x264.mkv`);
    const dubs = [
      ...videos.flatMap((_, i) => (i === 7 ? [] : [i === 6 ? `${dir}\\Hindi\\Goblin ep7 final (hin).eac3` : `${dir}\\Hindi\\Goblin.S01E${String(i + 1).padStart(2, "0")}.hin.eac3`])),
      `${dir}\\Hindi\\Goblin.S01.Special.hin.eac3`,
    ];
    const report = await invoke<PairingReport>("preview_pairs", {
      request: { mode: "series", videoFiles: videos, audioFiles: dubs },
    });
    expect(report.pairs).toHaveLength(15);
    expect(report.pairs[0].method).toBe("episode (S01E01)");
    expect(report.pairs.find((p) => p.key === "S01E07")?.method).toBe("list order");
    expect(report.unmatchedPrimary.map((p) => p.replace(/^.*\\/, ""))).toEqual(["Goblin.S01E08.1080p.BluRay.x264.mkv"]);
    expect(report.unmatchedSecondary).toHaveLength(1);
  });

  it("streams a dub sync job through its stages and ends with the plan", async () => {
    const seen = await record(["dubsync-job-start", "dubsync-job-progress", "dubsync-job-draft", "dubsync-job-plan", "dubsync-job-done"]);
    const v = "D:\\Media\\Skyline Heist (2023)\\Skyline Heist (2023) 1080p BluRay.mkv";
    const d = "D:\\Media\\Skyline Heist (2023)\\Skyline Heist (2023) Hindi TV rip.ac3";
    const batch = await invoke<DubSyncBatchResult>("start_dubsync_batch", {
      request: { jobs: [{ videoPath: v, dubPath: d, videoTrack: 0, dubTrack: 0 }], codec: "same", mux: false, language: null, fillUnmatched: false, overwrite: true, maxWorkers: 3 },
    });
    expect(batch.cancelled).toBe(false);
    const [outcome] = batch.outcomes;
    expect(outcome.plan?.segments).toHaveLength(13);
    expect(outcome.plan?.voicePieces).toHaveLength(2);
    expect(outcome.verification?.typicalMs).toBe(0.2);
    expect(outcome.verification?.worstMs).toBe(0.4);
    expect(outcome.output?.outputPath.endsWith(".dubsynced.ac3")).toBe(true);
    const stages = new Set(seen.filter((e) => e.event === "dubsync-job-progress").map((e) => (e.payload as { stage: string }).stage));
    expect(stages.has("looking for dub inside the gaps")).toBe(true);
    expect([...stages].some((s) => s.startsWith("writing "))).toBe(true);
    expect(seen.filter((e) => e.event === "dubsync-job-draft").length).toBeGreaterThan(2);
    expect(seen.filter((e) => e.event === "dubsync-job-done")).toHaveLength(1);
  });

  it("cancel_sync stops a run in flight and the command resolves cancelled", async () => {
    const { setTimeScale } = await import("./engine");
    setTimeScale(1);
    const pending = invoke<SyncRun>("start_sync", {
      request: request({ videoFiles: ["D:\\a\\A.mkv", "D:\\a\\B.mkv"], audioFiles: ["D:\\a\\dub.ac3"], audioFile: "D:\\a\\dub.ac3" }),
    });
    setTimeout(() => void invoke("cancel_sync"), 30);
    const run = await pending;
    expect(run.cancelled).toBe(true);
    expect(run.results.length).toBeLessThan(2);
    setTimeScale(0.005);
  });

  it("answers Subsync and the updater", async () => {
    const caps = await invoke<{ caps: { engines: { ocr: unknown[] } } }>("subs_caps");
    expect(caps.caps.engines.ocr.length).toBeGreaterThan(0);
    const probed = await invoke<{ files: { subtitle?: { cues: number } }[] }>("subs_probe", { paths: ["D:\\x\\Your.Name.2016.jpn.sup"] });
    expect(probed.files[0].subtitle?.cues).toBe(1204);
    const meta = await invoke<{ version: string } | null>("plugin:updater|check");
    expect(meta?.version).toBe("2.15.0");
    expect(await invoke("some_unknown_command")).toBeNull();
  });

  it("fails Goblin E04 in the season scenario with the dub-too-short message", async () => {
    shell.location = { search: "?demo=dub-season&speed=0.005" };
    installDemo("dub-season");
    const dir = "D:\\Media\\Goblin (2016)";
    const jobs = [3, 4].map((n) => ({ videoPath: `${dir}\\Goblin.S01E0${n}.1080p.BluRay.mkv`, dubPath: `${dir}\\Hindi\\Goblin.S01E0${n}.hin.eac3`, videoTrack: 0, dubTrack: 0 }));
    const batch = await invoke<DubSyncBatchResult>("start_dubsync_batch", {
      request: { jobs, codec: "same", mux: false, language: null, fillUnmatched: false, overwrite: true, maxWorkers: 3 },
    });
    expect(batch.outcomes[0].error).toBeNull();
    expect(batch.outcomes[1].error).toBe("The dub ends 38 minutes before the video. It is probably a different edit.");
    expect(batch.outcomes[1].plan).toBeNull();
  });

  it("serves speech-like peaks and cuts the canvas can draw", async () => {
    const peaks = await invoke<WaveformPeaks>("waveform_peaks", {
      request: { path: "D:\\Media\\Skyline Heist (2023)\\Skyline Heist (2023) 1080p BluRay.mkv", track: 0, startS: 1200, endS: 1500, buckets: 600 },
    });
    expect(peaks.channels).toBe(2);
    expect(peaks.min).toHaveLength(2);
    expect(peaks.max[0]).toHaveLength(600);
    expect(peaks.rms[1]).toHaveLength(600);
    expect(peaks.max[0].every((v, i) => v >= 0 && v <= 1 && peaks.min[0][i] <= 0 && peaks.min[0][i] >= -1)).toBe(true);
    expect(Math.max(...peaks.max[0])).toBeGreaterThan(0.3);
    expect(Math.min(...peaks.rms[0])).toBeLessThan(0.1);
    const cuts = await invoke<{ cuts: number[]; frameS: number }>("shot_cuts", { request: { path: "x.mkv", startS: 1200, endS: 1260 } });
    expect(cuts.cuts.length).toBeGreaterThan(5);
  });

  it("delivers a fake drop as the window's own drag-drop events", async () => {
    const seen: { type: string; paths?: string[]; x?: number }[] = [];
    await getCurrentWindow().onDragDropEvent((e) => {
      seen.push({ type: e.payload.type, paths: "paths" in e.payload ? e.payload.paths : undefined, x: "position" in e.payload ? e.payload.position.x : undefined });
    });
    await window.__demo!.drop(["D:\\a\\b.mkv"]);
    expect(seen.map((e) => e.type)).toEqual(["enter", "over", "drop"]);
    expect(seen[2].paths).toEqual(["D:\\a\\b.mkv"]);
    expect(seen[2].x).toBe(720);
  });
});
