/** Synthetic media data for the demo: speech-like waveform peaks, shot cuts
 *  and a silent preview clip. Everything is a pure function of time, so the
 *  picture is the same at every zoom level and after every reload. Dev only. */

import type { ShotCuts, WaveformPeaks, WaveformRequest } from "@/lib/api";

import { fileForPath, unit } from "./fixtures";

/** Smooth value noise in 0..1 with one lattice point per `period` seconds. */
function noise(t: number, period: number, seed: string): number {
  const x = t / period;
  const i = Math.floor(x);
  const f = x - i;
  const a = unit(`${seed}${i}`);
  const b = unit(`${seed}${i + 1}`);
  const s = f * f * (3 - 2 * f);
  return a + (b - a) * s;
}

/** Scenes of 1.5 to 8 minutes, each at its own loudness and some near
 *  silent: the shape a whole film shows, as the mockup draws it. */
const scenesByFilm = new Map<string, { at: number[]; level: number[] }>();
function sceneLevel(t: number, film: string): number {
  let scenes = scenesByFilm.get(film);
  if (!scenes) {
    const at = [0];
    const level: number[] = [];
    for (let i = 0; at[i] < 6 * 3600; i += 1) {
      at.push(at[i] + 90 + unit(`${film}L${i}`) * 390);
      level.push(unit(`${film}q${i}`) < 0.15 ? 0.08 : 0.35 + 0.65 * unit(`${film}v${i}`));
    }
    scenes = { at, level };
    scenesByFilm.set(film, scenes);
  }
  const { at, level } = scenes;
  let i = 0;
  while (i < level.length - 1 && at[i + 1] <= t) i += 1;
  // Two seconds to fade from one scene into the next.
  const into = t - at[i];
  const before = i > 0 ? level[i - 1] : level[i];
  return into < 2 ? before + ((level[i] - before) * into) / 2 : level[i];
}

/** Loudness envelope 0..1 at time t: scenes, phrases of speech and pauses
 *  inside them, words inside a phrase, syllables inside a word. `film`
 *  seeds it all so a film and its dub share it. */
export function envelope(t: number, film: string): number {
  const phrase = noise(t, 6.5, `${film}p`);
  const speaking = Math.min(1, Math.max(0, (phrase - 0.3) * 6));
  const words = 0.35 + 0.65 * noise(t, 0.55, `${film}w`);
  const syllables = 0.55 + 0.45 * Math.abs(Math.sin(t * Math.PI * 4.2 + noise(t, 1.3, `${film}s`) * 6));
  const score = 0.04 + 0.05 * noise(t, 9, `${film}m`);
  return Math.min(0.95, sceneLevel(t, film) * (score + speaking * words * syllables * (0.55 + 0.4 * noise(t, 17, `${film}v`))));
}

/** A title both a film and its dub reduce to: the first words of the name. */
function filmOf(path: string): string {
  const name = fileForPath(path).name.toLowerCase();
  return name.split(/[^a-z0-9]+/).slice(0, 2).join(" ");
}

export function synthPeaks(request: WaveformRequest): WaveformPeaks {
  const file = fileForPath(request.path);
  const film = filmOf(request.path);
  const buckets = Math.max(1, Math.round(request.buckets));
  const span = Math.max(1e-3, request.endS - request.startS);
  const step = span / buckets;
  const channels = 2;
  const min: number[][] = [];
  const max: number[][] = [];
  const rms: number[][] = [];
  for (let c = 0; c < channels; c += 1) {
    const lo: number[] = [];
    const hi: number[] = [];
    const rm: number[] = [];
    for (let i = 0; i < buckets; i += 1) {
      const t0 = request.startS + i * step;
      // Average the envelope across the bucket; at wide zoom this smooths
      // syllables into the phrase shape, as real peaks do.
      let level = 0;
      // Enough taps to average a wide bucket down to its scene's loudness.
      const taps = Math.min(48, Math.max(4, Math.round(step / 0.4)));
      for (let k = 0; k < taps; k += 1) level += envelope(t0 + ((k + 0.5) / taps) * step, film);
      level /= taps;
      // Seconds to a pixel: phrases blur into each scene's loudness, rising
      // and falling slowly, as a whole film's peaks do.
      const wide = Math.min(1, Math.max(0, (step - 0.5) / 2.5));
      if (wide > 0) {
        const mid = t0 + step / 2;
        level = (1 - wide) * level + wide * sceneLevel(mid, film) * (0.42 + 0.2 * noise(mid, 25, `${film}u`));
      }
      const fur = 1 - wide * 0.6;
      const jitter = 1 - 0.15 * fur + 0.3 * fur * unit(`${request.path}${c}${Math.round((t0 / step) * 7)}`);
      const value = Math.min(0.98, level * jitter);
      const past = t0 > file.durationS;
      const r = past ? 0 : value * 0.55;
      hi.push(past ? 0 : Math.min(0.99, value * (1.5 + 0.3 * fur * unit(`${request.path}h${c}${i}`))));
      lo.push(past ? 0 : -Math.min(0.99, value * (1.35 + 0.3 * fur * unit(`${request.path}l${c}${i}`))));
      rm.push(r);
    }
    min.push(lo);
    max.push(hi);
    rms.push(rm);
  }
  return {
    path: request.path,
    track: request.track,
    startS: request.startS,
    endS: request.endS,
    buckets,
    durationS: file.durationS,
    channels,
    sampleRate: 48000,
    min,
    max,
    rms,
    requestId: request.requestId,
  };
}

/** Shot changes every two to nine seconds, on frame boundaries. */
export function synthCuts(path: string, startS: number, endS: number): ShotCuts {
  const frameS = 1001 / 24000;
  const end = Math.min(endS, startS + 600);
  const cuts: number[] = [];
  const seed = fileForPath(path).name;
  let t = Math.floor(startS / 9) * 9;
  let i = Math.floor(startS / 9) * 3;
  while (t < end) {
    t += 2 + unit(`${seed}cut${i}`) * 7;
    i += 1;
    const snapped = Math.round(t / frameS) * frameS;
    if (snapped >= startS && snapped <= end) cuts.push(Math.round(snapped * 1000) / 1000);
  }
  return { cuts, frameS, startS, endS: end };
}

/** Two seconds of 8 kHz mono silence as a WAV, for `read_preview_bytes`. */
export function silentWav(seconds = 2): ArrayBuffer {
  const rate = 8000;
  const samples = rate * seconds;
  const buffer = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => [...value].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples * 2, true);
  return buffer;
}
