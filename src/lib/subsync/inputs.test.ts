import { describe, expect, it } from "vitest";

import {
  autoPair,
  buildItems,
  buildJobs,
  emptyPools,
  metaFromName,
  poolsReducer,
  stemCandidates,
  type InputPools,
} from "./inputs";
import { DEFAULT_OPTIONS, type ProbedFile, type SubtitleTrackInfo } from "./types";

function video(path: string, tracks: Partial<SubtitleTrackInfo>[] = [], fps = 23.976): ProbedFile {
  return {
    path,
    name: path.replace(/^.*\//, ""),
    kind: "video",
    duration: 5400,
    video: {
      width: 3840, height: 2160, fps, codec: "hevc", hdr: "dv", dvProfile: 8, dvCompatibility: 1,
      maxCll: 1000, masteringPeak: 1000, transfer: "smpte2084", primaries: "bt2020",
    },
    audioTracks: [{ index: 0, codec: "eac3", language: "eng", title: null, channels: 6 }],
    subtitleTracks: tracks.map((track, index) => ({
      index, codec: "subrip", kind: "text", language: "eng", title: null, forced: false,
      default: false, hearingImpaired: false, events: 900, ...track,
    })),
    subtitle: null,
  };
}

function subtitle(path: string, kind: "text" | "image" = "text"): ProbedFile {
  return {
    path,
    name: path.replace(/^.*\//, ""),
    kind: "subtitle",
    duration: null,
    video: null,
    audioTracks: [],
    subtitleTracks: [],
    subtitle: { format: kind === "text" ? "srt" : "pgs", kind, cues: 900, language: "en", encoding: "utf-8" },
  };
}

/** Pools with the given files already probed into `task`. */
function poolsWith(task: keyof InputPools, files: ProbedFile[]): InputPools {
  let pools = poolsReducer(emptyPools(), {
    type: "add",
    task,
    files: files.map((file) => ({ name: file.name, path: file.path, type: file.kind === "unknown" ? "video" : file.kind, size: 1 })),
  });
  pools = poolsReducer(pools, { type: "probed", files });
  return pools;
}

let counter = 0;
const newId = (index: number) => `job-${index}-${counter++}`;

describe("stem pairing", () => {
  it("drops trailing language codes and tags, longest candidate first", () => {
    expect(stemCandidates("Movie.en.forced.srt")).toEqual(["Movie.en.forced", "Movie.en", "Movie"]);
    expect(stemCandidates("Show.S01E01.pt-br.srt")).toEqual(["Show.S01E01.pt-br", "Show.S01E01"]);
    expect(stemCandidates("Movie.2019.srt")).toEqual(["Movie.2019"]);
  });

  it("pairs Movie.en.srt with Movie.mkv, and episodes with their own video", () => {
    const pairs = autoPair(
      [subtitle("/s/Movie.en.srt"), subtitle("/s/Show.S01E02.eng.srt"), subtitle("/s/Other.srt")],
      [video("/v/Movie.mkv"), video("/v/Show.S01E01.mkv"), video("/v/Show.S01E02.mkv")],
    );
    expect(pairs["/s/Movie.en.srt"]).toBe("/v/Movie.mkv");
    expect(pairs["/s/Show.S01E02.eng.srt"]).toBe("/v/Show.S01E02.mkv");
    expect(pairs["/s/Other.srt"]).toBeNull();
  });

  it("matches a title whose last word looks like a code on its full name", () => {
    const pairs = autoPair([subtitle("/s/Up.Top.srt")], [video("/v/Up.mkv"), video("/v/Up.Top.mkv")]);
    expect(pairs["/s/Up.Top.srt"]).toBe("/v/Up.Top.mkv");
  });

  it("sends every subtitle to the only video when names do not match", () => {
    const pairs = autoPair([subtitle("/s/a.srt"), subtitle("/s/b.srt")], [video("/v/film.mkv")]);
    expect(pairs).toEqual({ "/s/a.srt": "/v/film.mkv", "/s/b.srt": "/v/film.mkv" });
  });

  it("reads language and flags from a subtitle name", () => {
    expect(metaFromName("Movie.pt-br.forced.srt")).toMatchObject({ language: "pt", forced: true });
    expect(metaFromName("Movie.eng.sdh.srt")).toMatchObject({ language: "eng", hearingImpaired: true });
    expect(metaFromName("The.End.srt").language).toBeUndefined();
  });
});

describe("sync items and jobs", () => {
  it("pairs files, and a video no file went with uses its own text track", () => {
    const pools = poolsWith("sync", [
      subtitle("/s/Movie.en.srt"),
      video("/v/Movie.mkv"),
      video("/v/Other.mkv", [{ kind: "image", codec: "hdmv_pgs_subtitle" }, { kind: "text" }]),
    ]);
    const items = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({ subtitle: { path: "/s/Movie.en.srt" }, video: { path: "/v/Movie.mkv" }, problem: null });
    expect(items[1]).toMatchObject({ embedded: true, subtitle: { path: "/v/Other.mkv", track: 1 }, problem: null });

    const { jobs, labels } = buildJobs("sync", items, DEFAULT_OPTIONS.sync, { dir: " ", suffix: ".synced" }, newId);
    expect(jobs).toHaveLength(2);
    expect(jobs[0].input).toEqual({ subtitle: { path: "/s/Movie.en.srt" }, video: { path: "/v/Movie.mkv", audioTrack: 0 } });
    expect(jobs[0].output.dir).toBeNull();
    expect(jobs[1].input.subtitle).toEqual({ path: "/v/Other.mkv", track: 1 });
    expect(labels[1]).toContain("Other.mkv");
  });

  it("needs a video for the audio engine but not for a reference subtitle", () => {
    const pools = poolsWith("sync", [subtitle("/s/a.srt"), subtitle("/s/b.srt"), video("/v/x.mkv"), video("/v/y.mkv")]);
    const audio = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
    expect(audio.filter((item) => item.problem === "Choose the video to sync it to.")).toHaveLength(2);
    const reference = buildItems("sync", pools.sync, { ...DEFAULT_OPTIONS.sync, engine: "subtitle" });
    expect(reference.filter((item) => !item.embedded).every((item) => item.problem === null)).toBe(true);
  });

  it("applies a manual pairing and a track choice", () => {
    let pools = poolsWith("sync", [
      subtitle("/s/a.srt"),
      video("/v/x.mkv", [{ language: "eng" }, { language: "jpn" }]),
      video("/v/y.mkv"),
    ]);
    pools = poolsReducer(pools, { type: "setPair", task: "sync", subtitle: "/s/a.srt", video: "/v/y.mkv" });
    pools = poolsReducer(pools, { type: "setTrack", task: "sync", video: "/v/x.mkv", track: 1 });
    const items = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
    expect(items[0].video?.path).toBe("/v/y.mkv");
    const embedded = items.find((item) => item.embedded);
    expect(embedded?.subtitle).toEqual({ path: "/v/x.mkv", track: 1 });
  });

  it("skips rows with a problem when building jobs", () => {
    const pools = poolsWith("sync", [subtitle("/s/a.srt"), video("/v/x.mkv"), video("/v/y.mkv")]);
    const items = buildItems("sync", pools.sync, DEFAULT_OPTIONS.sync);
    expect(buildJobs("sync", items, DEFAULT_OPTIONS.sync, {}, newId).jobs).toHaveLength(0);
  });
});

describe("other tools", () => {
  it("OCR takes image subtitles and image tracks, and refuses text", () => {
    const pools = poolsWith("ocr", [
      subtitle("/s/a.sup", "image"),
      subtitle("/s/b.srt"),
      video("/v/x.mkv", [{ kind: "text" }, { kind: "image", codec: "hdmv_pgs_subtitle" }]),
      video("/v/y.mkv", [{ kind: "text" }]),
    ]);
    const items = buildItems("ocr", pools.ocr, DEFAULT_OPTIONS.ocr);
    expect(items.map((item) => item.problem === null)).toEqual([true, false, true, false]);
    expect(items[1].problem).toMatch(/image subtitle/);
    expect(items[2].subtitle).toEqual({ path: "/v/x.mkv", track: 1 });
    expect(items[3].problem).toMatch(/No image subtitle tracks/);
  });

  it("generate makes one job per video with the chosen audio track", () => {
    let pools = poolsWith("generate", [video("/v/a.mkv"), video("/v/b.mkv")]);
    pools = poolsReducer(pools, { type: "setAudio", task: "generate", video: "/v/b.mkv", track: 2 });
    const items = buildItems("generate", pools.generate, DEFAULT_OPTIONS.generate);
    const { jobs } = buildJobs("generate", items, DEFAULT_OPTIONS.generate, {}, newId);
    expect(jobs.map((job) => job.input.video)).toEqual([
      { path: "/v/a.mkv", audioTrack: 0 },
      { path: "/v/b.mkv", audioTrack: 2 },
    ]);
  });

  it("mux makes one job from a video and every subtitle, with their flags", () => {
    let pools = poolsWith("mux", [video("/v/Movie.mkv"), subtitle("/s/Movie.en.srt"), subtitle("/s/Movie.fr.forced.srt")]);
    pools = poolsReducer(pools, { type: "setMuxMeta", task: "mux", path: "/s/Movie.en.srt", patch: { default: true, title: "English" } });
    const items = buildItems("mux", pools.mux, DEFAULT_OPTIONS.mux);
    const { jobs } = buildJobs("mux", items, DEFAULT_OPTIONS.mux, { suffix: ".muxed" }, newId);
    expect(jobs).toHaveLength(1);
    expect(jobs[0].input.video).toEqual({ path: "/v/Movie.mkv" });
    expect(jobs[0].input.subtitles).toEqual([
      { path: "/s/Movie.en.srt", language: "en", default: true, forced: false, hearingImpaired: false, title: "English" },
      { path: "/s/Movie.fr.forced.srt", language: "fr", default: false, forced: true, hearingImpaired: false },
    ]);
  });

  it("mux without a video is not runnable", () => {
    const pools = poolsWith("mux", [subtitle("/s/a.srt")]);
    const items = buildItems("mux", pools.mux, DEFAULT_OPTIONS.mux);
    expect(items[0].problem).toMatch(/video/);
  });
});

describe("poolsReducer", () => {
  it("ignores a file added twice and applies probes to every pool holding it", () => {
    let pools = poolsReducer(emptyPools(), { type: "add", task: "sync", files: [{ name: "a.srt", path: "/a.srt", type: "subtitle", size: 3 }] });
    pools = poolsReducer(pools, { type: "add", task: "sync", files: [{ name: "a.srt", path: "/a.srt", type: "subtitle", size: 3 }] });
    pools = poolsReducer(pools, { type: "add", task: "style", files: [{ name: "a.srt", path: "/a.srt", type: "subtitle", size: 3 }] });
    expect(pools.sync.files).toHaveLength(1);
    expect(pools.sync.files[0].pending).toBe(true);
    pools = poolsReducer(pools, { type: "probed", files: [subtitle("/a.srt")] });
    expect(pools.sync.files[0]).toMatchObject({ pending: false, size: 3, subtitle: { cues: 900 } });
    expect(pools.style.files[0].pending).toBe(false);
  });

  it("forgets the choices about a removed file", () => {
    let pools = poolsWith("sync", [subtitle("/s/a.srt"), video("/v/x.mkv")]);
    pools = poolsReducer(pools, { type: "setPair", task: "sync", subtitle: "/s/a.srt", video: "/v/x.mkv" });
    pools = poolsReducer(pools, { type: "setTrack", task: "sync", video: "/v/x.mkv", track: 0 });
    pools = poolsReducer(pools, { type: "remove", task: "sync", path: "/v/x.mkv" });
    expect(pools.sync.pairs).toEqual({});
    expect(pools.sync.tracks).toEqual({});
    expect(pools.sync.files).toHaveLength(1);
  });
});
