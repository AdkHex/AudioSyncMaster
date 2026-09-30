/* Sample data in the app's real formats: player sign convention for delays,
 * confidence bands High ≥ 75 / Medium ≥ 50 / Low, the dub-sync plan shape of
 * src/lib/__fixtures__/dubsync-outcome.json. Titles and numbers are invented. */

export interface Movie {
  name: string;
  dur: string;
  fps: string;
  audio: string;
  size: string;
  delay?: string;
  frames?: number;
  conf?: number;
  drift?: string;
  kind: "ok" | "drift" | "fps" | "cut" | "fail";
  start?: string;
  end?: string;
  win?: string;
  took?: string;
  err?: string;
  pick?: boolean;
}

export const DUB = { name: "Blade Runner 2049 (2017) Hindi DD5.1.ac3", meta: "AC3 5.1 · 448 kb/s · 2:43:48" };

export const MOVIES: Movie[] = [
  { name: "Blade Runner 2049 (2017) 2160p UHD BluRay REMUX.mkv", dur: "2:43:48", fps: "23.976", audio: "TrueHD 7.1 · English", size: "58.2 GB", delay: "−317.0", frames: 8, conf: 94, kind: "ok", start: "−317.0 ms", end: "−317.1 ms", win: "6 of 6", took: "41.2 s", pick: true },
  { name: "Blade Runner 2049 (2017) 1080p BluRay x264.mkv", dur: "2:43:48", fps: "23.976", audio: "DTS-HD MA 7.1", size: "14.1 GB", delay: "−317.0", frames: 8, conf: 93, kind: "ok", start: "−317.0 ms", end: "−316.9 ms", win: "6 of 6", took: "38.9 s", pick: true },
  { name: "Blade Runner 2049 (2017) 1080p WEB-DL DDP5.1.mkv", dur: "2:43:44", fps: "23.976", audio: "E-AC3 5.1", size: "7.9 GB", delay: "+1,208.0", frames: 29, conf: 68, drift: "0.412 ms/s", kind: "drift", start: "+1,206.9 ms", end: "+5,258.3 ms", win: "5 of 6", took: "44.0 s" },
  { name: "Blade Runner 2049 (2017) PAL DVD.mkv", dur: "2:37:13", fps: "25", audio: "AC3 5.1", size: "7.4 GB", delay: "−84.0", frames: 2, conf: 88, drift: "23.976 → 25 fps", kind: "fps", start: "−84.0 ms", end: "−84.2 ms", win: "6 of 6", took: "52.7 s", pick: true },
  { name: "Blade Runner 2049 (2017) Fan Edit.mkv", dur: "2:31:02", fps: "23.976", audio: "AAC 2.0", size: "4.2 GB", conf: 31, drift: "Different cut", kind: "cut", win: "2 of 6", took: "1m 12s" },
  { name: "Blade Runner 2049 (2017) Sample.mkv", dur: "0:01:00", fps: "23.976", audio: "None", size: "88 MB", kind: "fail", took: "0.4 s", err: "No audio track" },
];

export const EPISODES = Array.from({ length: 16 }, (_, i) => {
  const n = String(i + 1).padStart(2, "0");
  const conf = [94, 92, 95, 91, 63, 90, 88, 0, 93, 96, 71, 92, 94, 89, 91, 90][i];
  return {
    ep: `S01E${n}`,
    video: `Goblin.S01E${n}.1080p.BluRay.x264.mkv`,
    dub: i === 7 ? "" : i === 6 ? "Goblin ep7 final (hin).eac3" : `Goblin.S01E${n}.hin.eac3`,
    by: i === 7 ? "" : i === 6 ? "List order" : "Episode number",
    conf,
    delay: ["−317.0", "−317.0", "−320.2", "−317.0", "+42.0", "−317.0", "−316.8", "", "−317.0", "−317.1", "−1,540.0", "−317.0", "−317.0", "−318.4", "−317.0", "−317.0"][i],
    note: i === 10 ? "25 → 23.976 fps" : i === 4 ? "Low windows" : "",
  };
});

export const MATCH = {
  videos: ["BluRay", "WEB-DL", "Black & White"],
  vfiles: ["Parasite.2019.1080p.BluRay.x264.mkv", "Parasite.2019.1080p.WEB-DL.DDP5.1.mkv", "Parasite.2019.Black.and.White.1080p.mkv"],
  dubs: ["Hindi AC3", "Tamil AAC", "Telugu E-AC3"],
  conf: [
    [94, 38, 22],
    [41, 91, 88],
    [12, 18, 27],
  ],
  delay: [
    ["−317.0 ms", "", ""],
    ["", "+42.0 ms", "+41.8 ms"],
    ["", "", ""],
  ],
};

/* 2:20:30 film, 2:10:10 Hindi dub, five scenes cut from the dub. */
export const TOTAL = 8430;
export const PLAN: { k: "dub" | "fill"; a: number; b: number; src: string; off?: string; m?: string; note?: string }[] = [
  { k: "fill", a: 0, b: 1.21, src: "Original 0:00:00.000", note: "Dub starts late" },
  { k: "dub", a: 1.21, b: 1230, src: "Dub 0:00:00.404", off: "+0.806 s", m: "0.12" },
  { k: "fill", a: 1230, b: 1361.5, src: "Original 0:20:30.000", note: "Cut from the dub" },
  { k: "dub", a: 1361.5, b: 3070.25, src: "Dub 0:20:29.194", off: "−130.694 s", m: "0.09" },
  { k: "fill", a: 3070.25, b: 3182, src: "Original 0:51:10.250", note: "Cut from the dub" },
  { k: "dub", a: 3182, b: 4724.8, src: "Dub 0:48:59.556", off: "−242.444 s", m: "0.11" },
  { k: "fill", a: 4724.8, b: 4883, src: "Original 1:18:44.800", note: "Cut from the dub" },
  { k: "dub", a: 4883, b: 6545.6, src: "Dub 1:14:42.356", off: "−400.644 s", m: "0.13" },
  { k: "fill", a: 6545.6, b: 6642, src: "Original 1:49:05.600", note: "Cut from the dub" },
  { k: "dub", a: 6642, b: 7590.85, src: "Dub 1:42:24.956", off: "−497.044 s", m: "0.08" },
  { k: "fill", a: 7590.85, b: 7713, src: "Original 2:06:30.850", note: "Cut from the dub" },
  { k: "dub", a: 7713, b: 8427.4, src: "Dub 1:57:13.806", off: "−619.194 s", m: "0.10" },
  { k: "fill", a: 8427.4, b: 8430, src: "Original 2:20:27.400", note: "Past the dub's end" },
];

export const STAGES = [
  "Reading the files",
  "Checking the frame rate",
  "Finding where the dub belongs",
  "Placing the cuts",
  "Measuring each stretch",
  "Looking for dub inside the gaps",
  "Assembling the plan",
  "Checking against the video",
  "Writing the track",
];

export const LOG: [string, string, string][] = [
  ["14:22:07", "", "Dub sync: 5 pairs, 3 at a time"],
  ["14:22:07", "", "Reading Goblin.S01E03.1080p.BluRay.mkv — E-AC3 5.1, 48 kHz, 1:14:31"],
  ["14:22:09", "", "Frame rate: video 23.976 fps, dub mastered at 23.976 fps"],
  ["14:22:31", "", "Dub belongs at +0.806 s from 0:00:01.210"],
  ["14:23:02", "warn", "Scene cut from the dub at 0:20:30.000 (2m 11.5s) — filled from the original"],
  ["14:23:40", "ok", "Measured 6 stretches — 0.21 ms typical"],
  ["14:23:41", "bad", "Goblin.S01E04: the dub ends 38 minutes before the video (a different edit?)"],
  ["14:23:41", "", "Goblin.S01E04: source files left unchanged"],
];

export const CUES = [
  { n: 412, a: "00:31:02.480", b: "00:31:04.902", t: "Where are you going?", cps: 8.7 },
  { n: 413, a: "00:31:05.120", b: "00:31:07.300", t: "Somewhere the snow doesn't reach.", cps: 15.1 },
  { n: 414, a: "00:31:07.640", b: "00:31:08.850", t: "That's everywhere, K.", cps: 17.4 },
  { n: 415, a: "00:31:09.010", b: "00:31:12.760", t: "Then I'll find the place that isn't.", cps: 9.9 },
  { n: 416, a: "00:31:13.200", b: "00:31:14.100", t: "You always say that like it's easy to do.", cps: 45.6, warn: true },
  { n: 417, a: "00:31:15.400", b: "00:31:18.020", t: "[distant thunder]", cps: 6.5 },
  { n: 418, a: "00:31:18.300", b: "00:31:21.100", t: "I kept the wooden horse.", cps: 8.6 },
];

export const OCR = [
  { n: 88, a: "00:12:40.120", t: "どこへ行くの？", c: 97 },
  { n: 89, a: "00:12:43.100", t: "雪が届かないところへ。", c: 94 },
  { n: 90, a: "00:12:46.020", t: "そんな場所、ないよ。", c: 43 },
  { n: 91, a: "00:12:48.900", t: "じゃあ、探しに行こう。", c: 91 },
  { n: 92, a: "00:12:52.010", t: "三葉？ 三葉なの？", c: 52 },
  { n: 93, a: "00:12:55.300", t: "君の名前は…", c: 96 },
];

export const HISTORY = [
  { mode: "Dub sync", name: "Goblin · Season 1", res: "16 written, 1 to check", when: "Today 14:22", tone: "warn" },
  { mode: "Movies", name: "Blade Runner 2049", res: "4 matched, 1 different cut, 1 failed", when: "Today 11:05", tone: "ok" },
  { mode: "Subsync", name: "Blade.Runner.2049.en.srt", res: "Synced +3.250 s", when: "Yesterday 22:40", tone: "ok" },
  { mode: "Find match", name: "Parasite · 3 × 3", res: "3 matches", when: "Yesterday 19:12", tone: "ok" },
  { mode: "Series", name: "Goblin · Season 1", res: "14 matched, 1 rate change, 1 unpaired", when: "Sep 26 09:30", tone: "warn" },
  { mode: "Subsync", name: "Your.Name.2016.jpn.sup", res: "OCR 1,204 lines, 17 to check", when: "Sep 26 08:02", tone: "warn" },
  { mode: "Dub sync", name: "Train to Busan", res: "Failed — different edit", when: "Sep 25 21:47", tone: "bad" },
  { mode: "Movies", name: "Dune: Part Two", res: "2 matched", when: "Sep 25 18:10", tone: "ok" },
];
