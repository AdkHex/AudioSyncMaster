/** Loading the player's 30 s window: the framework-free part of useExcerpt.
 *
 *  The player never plays the source files -- the webview cannot -- so it
 *  plays excerpts the engine renders: a 480p picture without sound, and
 *  the synced track as 16-bit WAV. The window's real start comes from the
 *  picture's reply (a picture is cut starting on a frame), and the sound
 *  is then asked for from exactly that start, so what is heard is what
 *  the written file would contain, frame-accurate against the picture.
 *
 *  All of that ordering lives in ExcerptLoader, a plain class with every
 *  side effect injected, so it can be tested without React or a browser.
 *  The hook is a thin subscription over it. */

import { useCallback, useEffect, useRef, useState } from "react";

import type { DubExcerpt, DubPreviewRequest } from "./api";
import type { DubSyncPlan } from "./types";
import { samePlan } from "./dubPlanEdit";

/** What the player can play right now. `startS`/`endS` are the span the
 *  picture reply reported -- the window's time 0 -- or the span asked for
 *  in sample mode. */
export interface ExcerptSources {
  startS: number | null;
  endS: number | null;
  /** A Blob URL for the picture, null in sample mode or when it failed. */
  pictureUrl: string | null;
  /** The synced track over the window, as the plan would write it. */
  dub: AudioBuffer | null;
  /** The video's own sound over the window, fetched on first need. */
  original: AudioBuffer | null;
  loading: "picture" | "audio" | "original" | null;
  error: string | null;
}

const EMPTY: ExcerptSources = {
  startS: null,
  endS: null,
  pictureUrl: null,
  dub: null,
  original: null,
  loading: null,
  error: null,
};

export interface ExcerptLoaderDeps {
  render: (request: DubPreviewRequest) => Promise<DubExcerpt | null>;
  read: (path: string) => Promise<ArrayBuffer>;
  /** Decodes one WAV excerpt; closes over the caller's AudioContext. */
  decodeAudio: (bytes: ArrayBuffer) => Promise<AudioBuffer>;
  makePictureUrl: (bytes: ArrayBuffer) => string;
  revokeUrl: (url: string) => void;
  /** How long to wait after a plan edit before re-rendering its sound. */
  debounceMs?: number;
  onChange: (sources: ExcerptSources) => void;
}

export class ExcerptLoader {
  private render: ExcerptLoaderDeps["render"];
  private read: ExcerptLoaderDeps["read"];
  private decodeAudio: ExcerptLoaderDeps["decodeAudio"];
  private makePictureUrl: ExcerptLoaderDeps["makePictureUrl"];
  private revokeUrl: ExcerptLoaderDeps["revokeUrl"];
  private debounceMs: number;
  private onChange: ExcerptLoaderDeps["onChange"];

  private plan: DubSyncPlan | null = null;
  private window: { startS: number; endS: number } | null = null;
  private windowKey: string | null = null;
  private sources: ExcerptSources = EMPTY;
  private notice: string | null = null;
  private originalRequested = false;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private seq = 0;
  private latest: Record<"picture" | "audio" | "original", number> = {
    picture: 0,
    audio: 0,
    original: 0,
  };
  private dead = false;

  constructor(deps: ExcerptLoaderDeps) {
    this.render = deps.render;
    this.read = deps.read;
    this.decodeAudio = deps.decodeAudio;
    this.makePictureUrl = deps.makePictureUrl;
    this.revokeUrl = deps.revokeUrl;
    this.debounceMs = deps.debounceMs ?? 300;
    this.onChange = deps.onChange;
  }

  updateDeps(deps: Pick<ExcerptLoaderDeps, "render" | "read" | "decodeAudio">) {
    this.render = deps.render;
    this.read = deps.read;
    this.decodeAudio = deps.decodeAudio;
  }

  /** A new window (or none): reload the picture and the sound. A plan edit
   *  with the same window does not come through here -- that is updatePlan,
   *  which re-renders the sound only. */
  updateWindow(startS: number | null, endS: number | null, withPicture: boolean) {
    if (startS === null || endS === null) {
      this.clearWindow();
      return;
    }
    const key = `${startS.toFixed(6)}|${endS.toFixed(6)}|${withPicture}`;
    if (key === this.windowKey && !this.dead) return;
    this.windowKey = key;
    this.window = { startS, endS };
    this.cancelDebounce();
    this.originalRequested = false;
    this.notice = null;
    this.revokePicture();
    this.sources = {
      ...EMPTY,
      loading: withPicture ? "picture" : "audio",
    };
    this.emit();
    if (withPicture) {
      void this.loadPicture(startS, endS);
    } else {
      void this.loadAudio(startS, endS);
    }
  }

  /** The plan changed (an edit): re-render the sound for the same window
   *  after the debounce. The original does not depend on the plan, so it
   *  is kept as it was. */
  updatePlan(plan: DubSyncPlan) {
    const changed = this.plan === null || !samePlan(this.plan, plan);
    this.plan = plan;
    if (!this.window || !changed || this.dead) return;
    this.cancelDebounce();
    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      if (this.dead || !this.window) return;
      // The sound is always re-cut from the window's real, picture-aligned
      // start, not the span the cursor first asked for.
      void this.loadAudio(
        this.sources.startS ?? this.window.startS,
        this.sources.endS ?? this.window.endS,
      );
    }, this.debounceMs);
  }

  /** The original is fetched once per window, the first time it is needed. */
  updateWantOriginal(want: boolean) {
    if (!want || this.originalRequested || !this.window || this.dead) return;
    this.originalRequested = true;
    void this.loadOriginal(this.sources.startS ?? this.window.startS, this.sources.endS ?? this.window.endS);
  }

  dispose() {
    this.dead = true;
    this.cancelDebounce();
    this.revokePicture();
  }

  private async loadPicture(startS: number, endS: number) {
    const id = this.bump("picture");
    this.set({ loading: "picture", error: null });
    try {
      const excerpt = await this.render({ plan: this.plan!, startS, endS, what: "picture" });
      if (this.stale("picture", id)) return;
      if (!excerpt) throw new Error("The picture could not be rendered.");
      const bytes = await this.read(excerpt.path);
      if (this.stale("picture", id)) return;
      const url = this.makePictureUrl(bytes);
      this.revokePicture();
      this.set({
        pictureUrl: url,
        startS: excerpt.startS,
        endS: excerpt.endS,
        loading: "audio",
      });
      // The sound is cut from the picture's own start, so the two are
      // frame-aligned even when the window began between frames.
      await this.loadAudio(excerpt.startS, excerpt.endS);
    } catch (error) {
      if (this.stale("picture", id)) return;
      // Fall back to sound-only playback over the span asked for.
      this.notice = `${messageOf(error)} Playing the sound only.`;
      this.set({
        pictureUrl: null,
        startS,
        endS,
        loading: "audio",
        error: this.notice,
      });
      await this.loadAudio(startS, endS);
    }
  }

  private async loadAudio(startS: number, endS: number) {
    const id = this.bump("audio");
    this.set({ loading: "audio" });
    try {
      const excerpt = await this.render({ plan: this.plan!, startS, endS, what: "audio" });
      if (this.stale("audio", id)) return;
      if (!excerpt) throw new Error("The sound could not be rendered.");
      const bytes = await this.read(excerpt.path);
      if (this.stale("audio", id)) return;
      const buffer = await this.decodeAudio(bytes);
      if (this.stale("audio", id)) return;
      this.set({ dub: buffer, startS, endS, loading: null, error: this.notice });
    } catch (error) {
      if (this.stale("audio", id)) return;
      this.set({ dub: null, loading: null, error: messageOf(error) });
    }
  }

  private async loadOriginal(startS: number, endS: number) {
    const id = this.bump("original");
    this.set({ loading: "original" });
    try {
      const excerpt = await this.render({ plan: this.plan!, startS, endS, what: "original" });
      if (this.stale("original", id)) return;
      if (!excerpt) throw new Error("The original's sound could not be rendered.");
      const bytes = await this.read(excerpt.path);
      if (this.stale("original", id)) return;
      const buffer = await this.decodeAudio(bytes);
      if (this.stale("original", id)) return;
      this.set({ original: buffer, loading: null });
    } catch (error) {
      if (this.stale("original", id)) return;
      this.set({ loading: null, error: messageOf(error) });
    }
  }

  private clearWindow() {
    this.cancelDebounce();
    this.window = null;
    this.windowKey = null;
    this.notice = null;
    this.originalRequested = false;
    this.revokePicture();
    this.sources = EMPTY;
    this.emit();
  }

  private bump(kind: "picture" | "audio" | "original"): number {
    this.seq += 1;
    this.latest[kind] = this.seq;
    return this.seq;
  }

  private stale(kind: "picture" | "audio" | "original", id: number): boolean {
    return this.dead || this.latest[kind] !== id;
  }

  private revokePicture() {
    if (this.sources.pictureUrl) this.revokeUrl(this.sources.pictureUrl);
    this.sources = { ...this.sources, pictureUrl: null };
  }

  private set(patch: Partial<ExcerptSources>) {
    this.sources = { ...this.sources, ...patch };
    this.emit();
  }

  private emit() {
    if (!this.dead) this.onChange(this.sources);
  }

  private cancelDebounce() {
    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** The loaded window for a plan, kept current as the window, the plan and
 *  the mix change. `audioContext` is created by the caller on the button
 *  press that opened the player, so decoding never starts from a
 *  non-gesture context. */
export function useExcerpt(args: {
  plan: DubSyncPlan;
  wantedStartS: number | null;
  wantedEndS: number | null;
  withPicture: boolean;
  wantOriginal: boolean;
  audioContext: AudioContext | null;
  render: (request: DubPreviewRequest) => Promise<DubExcerpt | null>;
  read: (path: string) => Promise<ArrayBuffer>;
}): ExcerptSources {
  const { plan, wantedStartS, wantedEndS, withPicture, wantOriginal, audioContext, render, read } = args;

  const [sources, setSources] = useState<ExcerptSources>(EMPTY);
  const loaderRef = useRef<ExcerptLoader | null>(null);
  if (loaderRef.current === null) {
    loaderRef.current = new ExcerptLoader({
      render,
      read,
      decodeAudio: (bytes) => decodeWith(audioContext, bytes),
      makePictureUrl: (bytes) => URL.createObjectURL(new Blob([bytes], { type: "video/mp4" })),
      revokeUrl: (url) => URL.revokeObjectURL(url),
      onChange: setSources,
    });
  }
  const loader = loaderRef.current;

  const decode = useCallback(
    (bytes: ArrayBuffer) => decodeWith(audioContext, bytes),
    [audioContext],
  );
  useEffect(() => {
    loader.updateDeps({ render, read, decodeAudio: decode });
  }, [loader, render, read, decode]);

  useEffect(() => {
    loader.updatePlan(plan);
  }, [loader, plan]);

  useEffect(() => {
    loader.updateWindow(wantedStartS, wantedEndS, withPicture);
  }, [loader, wantedStartS, wantedEndS, withPicture]);

  useEffect(() => {
    loader.updateWantOriginal(wantOriginal);
  }, [loader, wantOriginal]);

  useEffect(() => () => loader.dispose(), [loader]);

  return sources;
}

function decodeWith(context: AudioContext | null, bytes: ArrayBuffer): Promise<AudioBuffer> {
  if (!context) return Promise.reject(new Error("No audio yet."));
  return context.decodeAudioData(bytes);
}
