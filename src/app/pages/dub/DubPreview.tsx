/** The player in the details pane: what the plan would write, frame by
 *  frame against the picture, so lip-sync can be checked by eye.
 *
 *  A muted 480p excerpt of the 30 s around the playhead (the video
 *  element) and the synced track over the same window as 16-bit WAV (Web
 *  Audio), locked to the picture's clock. An edit to the cuts re-cuts only
 *  the sound, which is swapped in at the current position. The playhead in
 *  the Timeline follows the picture; moving it there is a seek.
 *
 *  While it is open and its page shows, the player claims Space (play and
 *  pause), [ and ] (a frame back and forward). The AudioContext is made by
 *  the page in the click or key press that opened the player -- a user
 *  gesture, which is what Web Audio asks for -- never here. */

import {
  ArrowRepeatAllRegular,
  NextFrameRegular,
  OpenRegular,
  PauseRegular,
  PlayRegular,
  PreviousFrameRegular,
} from "@fluentui/react-icons";
import { useEffect, useMemo, useRef, useState } from "react";

import type { DubExcerpt, DubPreviewRequest } from "@/lib/api";
import { frameIndex, inWindow, loopAround, needsResync, stepFrame, timecode } from "@/lib/previewClock";
import { MIX_GAINS, type MixMode } from "@/lib/previewMix";
import type { DubSyncPlan } from "@/lib/types";
import { useExcerpt, type ExcerptSources } from "@/lib/useExcerpt";
import { Box } from "@/ui/frame";
import { Cmd, Seg, Slider, cx } from "@/ui/kit";

import { useCursor, type CursorStore } from "./cursorStore";

const VOLUME_KEY = "dubsync-player-volume";

const MIXES: { value: MixMode; label: string }[] = [
  { value: "dub", label: "Dub" },
  { value: "original", label: "Original" },
  { value: "both", label: "Both" },
];

interface Common {
  /** One video frame, for stepping and the timecode. */
  frameS: number;
  cursor: CursorStore;
  /** Ask for another window: the playhead left this one. */
  onNeedWindow: (aroundS: number) => void;
  audioContext: AudioContext;
  mix: MixMode;
  onMix: (mix: MixMode) => void;
  /** Whether the page shows; the player's keys work only then. */
  active: boolean;
  /** Renders a span around the playhead and hands it to the user's player. */
  onOpenExternal?: () => void;
}

/** The player on a plan: loads its window and plays it. */
export function DubPreview({
  plan,
  window: span,
  render,
  read,
  ...rest
}: Common & {
  plan: DubSyncPlan;
  window: { startS: number; endS: number };
  render: (request: DubPreviewRequest) => Promise<DubExcerpt | null>;
  read: (path: string) => Promise<ArrayBuffer>;
}) {
  const sources = useExcerpt({
    plan,
    wantedStartS: span.startS,
    wantedEndS: span.endS,
    withPicture: true,
    wantOriginal: rest.mix !== "dub",
    audioContext: rest.audioContext,
    render,
    read,
  });
  return <PreviewPlayer {...rest} sources={sources} />;
}

interface Graph {
  master: GainNode;
  dub: GainNode;
  orig: GainNode;
  dubPan: StereoPannerNode;
  origPan: StereoPannerNode;
}

function buildGraph(ctx: AudioContext): Graph {
  const master = ctx.createGain();
  master.connect(ctx.destination);
  const dub = ctx.createGain();
  const dubPan = ctx.createStereoPanner();
  dub.connect(dubPan);
  dubPan.connect(master);
  const orig = ctx.createGain();
  const origPan = ctx.createStereoPanner();
  orig.connect(origPan);
  origPan.connect(master);
  return { master, dub, orig, dubPan, origPan };
}

function applyMix(graph: Graph, mix: MixMode) {
  const table = MIX_GAINS[mix];
  graph.dub.gain.value = table.dub;
  graph.orig.gain.value = table.original;
  graph.dubPan.pan.value = table.dubPan;
  graph.origPan.pan.value = table.originalPan;
}

/** The player's clock: the video element's, or the AudioContext's when
 *  there is no picture (one that failed to render). Both speak seconds
 *  since the window's start. */
interface Clock {
  now(): number;
  seek(t: number): void;
  play(): void;
  pause(): void;
}

function loadVolume(): number {
  if (typeof window === "undefined") return 100;
  const stored = Number(window.localStorage.getItem(VOLUME_KEY) ?? 100);
  return Number.isFinite(stored) && stored >= 0 && stored <= 100 ? stored : 100;
}

/** The transport over loaded sources. */
export function PreviewPlayer({
  frameS,
  cursor,
  sources,
  onNeedWindow,
  audioContext,
  mix,
  onMix,
  active,
  onOpenExternal,
}: Common & { sources: ExcerptSources }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [volume, setVolume] = useState(loadVolume);
  const cursorS = useCursor(cursor) ?? 0;

  const graphRef = useRef<Graph | null>(null);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const startedRef = useRef({ ctx: 0, media: 0 });
  const lastEmittedRef = useRef(0);
  const lastWindowStartRef = useRef<number | null>(null);
  const loopRef = useRef<{ a: number; b: number } | null>(null);
  const attemptStartRef = useRef(false);

  // A picture that loaded plays the video element; anything else ticks on
  // the AudioContext.
  const pictureReady = sources.pictureUrl !== null;
  const contextClock = useRef({ playing: false, startedAtCtx: 0, startedAtMedia: 0, pausedAt: 0 });

  const clock = useMemo<Clock>(() => {
    if (pictureReady) {
      return {
        now: () => videoRef.current?.currentTime ?? 0,
        seek: (t) => {
          if (videoRef.current) videoRef.current.currentTime = t;
        },
        play: () => {
          void videoRef.current?.play().catch(() => undefined);
        },
        pause: () => {
          videoRef.current?.pause();
        },
      };
    }
    const c = contextClock.current;
    return {
      now: () => (c.playing ? audioContext.currentTime - c.startedAtCtx + c.startedAtMedia : c.pausedAt),
      seek: (t) => {
        c.pausedAt = t;
        c.startedAtCtx = audioContext.currentTime;
        c.startedAtMedia = t;
      },
      play: () => {
        c.startedAtCtx = audioContext.currentTime;
        c.startedAtMedia = c.pausedAt;
        c.playing = true;
      },
      pause: () => {
        if (c.playing) c.pausedAt = audioContext.currentTime - c.startedAtCtx + c.startedAtMedia;
        c.playing = false;
      },
    };
  }, [pictureReady, audioContext]);

  // What the per-frame loop and the seek effect read, always current.
  const sourcesRef = useRef(sources);
  sourcesRef.current = sources;
  const onNeedWindowRef = useRef(onNeedWindow);
  onNeedWindowRef.current = onNeedWindow;
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const activeRef = useRef(active);
  activeRef.current = active;

  // -------------------------------------------------------------- the graph

  useEffect(() => {
    const graph = buildGraph(audioContext);
    graphRef.current = graph;
    applyMix(graph, mix);
    graph.master.gain.value = volume / 100;
    return () => {
      stopSources();
      graph.master.disconnect();
      graphRef.current = null;
    };
    // The graph is per context: the page makes one per opening.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioContext]);

  useEffect(() => {
    if (graphRef.current) applyMix(graphRef.current, mix);
  }, [mix]);

  useEffect(() => {
    if (graphRef.current) graphRef.current.master.gain.value = volume / 100;
    if (typeof window !== "undefined") window.localStorage.setItem(VOLUME_KEY, String(volume));
  }, [volume]);

  // Paused when its page is put away.
  useEffect(() => {
    if (!active && playingRef.current) {
      setPlaying(false);
      clock.pause();
      stopSources();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  // -------------------------------------------------------------- the sound

  function stopSources() {
    for (const source of activeSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // Stopped already; a seek can land on one twice.
      }
      source.disconnect();
    }
    activeSourcesRef.current = [];
  }

  function startSources(at?: number) {
    const graph = graphRef.current;
    if (!graph) return;
    stopSources();
    const t = at ?? clock.now();
    startedRef.current = { ctx: audioContext.currentTime, media: t };
    const entries: [AudioBuffer | null, GainNode][] = [
      [sourcesRef.current.dub, graph.dub],
      [sourcesRef.current.original, graph.orig],
    ];
    for (const [buffer, gain] of entries) {
      if (!buffer) continue;
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      source.start(0, Math.max(0, Math.min(t, Math.max(0, buffer.duration - 0.001))));
      activeSourcesRef.current.push(source);
    }
  }

  const togglePlay = () => {
    if (playingRef.current) {
      setPlaying(false);
      clock.pause();
      stopSources();
    } else {
      setPlaying(true);
      clock.play();
      startSources();
    }
  };

  /** One frame back or forward: pause, then land exactly on the frame. */
  const step = (direction: -1 | 1) => {
    const { startS, endS } = sourcesRef.current;
    if (startS === null || endS === null) return;
    setPlaying(false);
    clock.pause();
    stopSources();
    const next = stepFrame(lastEmittedRef.current, frameS, direction, startS, endS);
    // A quarter frame inside the target: seeking exactly onto a frame
    // boundary presents the frame before it.
    clock.seek(Math.min(next - startS + frameS * 0.25, endS - startS));
    lastEmittedRef.current = next;
    cursor.set(next);
  };

  const toggleLoop = () => {
    setLoop((on) => {
      const next = !on;
      const { startS, endS } = sourcesRef.current;
      loopRef.current = next && startS !== null && endS !== null ? loopAround(lastEmittedRef.current, startS, endS) : null;
      return next;
    });
  };

  // ------------------------------------------------------------- the clock

  const onFrame = (mediaTime: number) => {
    const { startS } = sourcesRef.current;
    if (startS === null) return;
    const filmTime = startS + mediaTime;
    lastEmittedRef.current = filmTime;
    cursor.set(filmTime);

    const region = loopRef.current;
    if (region && playingRef.current && filmTime >= region.b) {
      const at = region.a - startS;
      clock.seek(at);
      startSources(at);
      return;
    }
    // Once the picture and the sound are 20 ms apart, re-lock them.
    if (playingRef.current) {
      const expected = audioContext.currentTime - startedRef.current.ctx + startedRef.current.media;
      if (needsResync(mediaTime, expected)) startSources(mediaTime);
    }
  };

  useEffect(() => {
    if (!pictureReady) return;
    const video = videoRef.current;
    if (!video || typeof video.requestVideoFrameCallback !== "function") return;
    let handle: number | null = null;
    const tick = (_now: number, metadata: VideoFrameCallbackMetadata) => {
      onFrame(metadata.mediaTime);
      handle = video.requestVideoFrameCallback(tick);
    };
    handle = video.requestVideoFrameCallback(tick);
    return () => {
      if (handle !== null) video.cancelVideoFrameCallback(handle);
    };
    // onFrame reads refs only; re-subscribe when the picture appears.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pictureReady]);

  useEffect(() => {
    if (pictureReady) return;
    let raf = 0;
    const tick = () => {
      onFrame(clock.now());
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pictureReady, audioContext]);

  // The sound arriving is the moment to begin; a later edit swaps it in at
  // the current position without touching the picture.
  const ready =
    sources.startS !== null && sources.loading === null && sources.dub !== null && (sources.pictureUrl !== null || sources.error !== null);

  useEffect(() => {
    if (!playingRef.current || !sources.dub) return;
    startSources();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sources.dub]);

  useEffect(() => {
    if (!ready || !activeRef.current) return;
    setPlaying(true);
    attemptStartRef.current = true;
    clock.play();
    startSources();
    // Begin once, on the first load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // A playhead move this player did not make is a seek: inside the window
  // it jumps there, outside it asks for a new window.
  useEffect(() => {
    const { startS, endS } = sources;
    if (startS === null || endS === null) return;
    const windowChanged = lastWindowStartRef.current !== startS;
    if (!windowChanged && Math.abs(cursorS - lastEmittedRef.current) < 1e-9) return;
    lastWindowStartRef.current = startS;
    if (inWindow(cursorS, startS, endS)) {
      const at = cursorS - startS;
      clock.seek(at);
      if (playingRef.current) startSources(at);
      lastEmittedRef.current = cursorS;
    } else {
      onNeedWindowRef.current(cursorS);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursorS, sources.startS, sources.endS]);

  // --------------------------------------------------------------- the keys

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!activeRef.current || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable)) return;
      if (event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        togglePlay();
      } else if (event.key === "[") {
        event.preventDefault();
        event.stopPropagation();
        step(-1);
      } else if (event.key === "]") {
        event.preventDefault();
        event.stopPropagation();
        step(1);
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
    // The handlers read refs; the listener lives as long as the player.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const message =
    sources.loading === "picture"
      ? "Loading the picture…"
      : sources.loading === "audio"
        ? "Rendering the sound…"
        : sources.loading === "original"
          ? "Loading the original…"
          : sources.error;

  return (
    <Box
      title="Preview"
      label="Preview player"
      end={
        <span className="t3 num sm" style={{ paddingRight: 6 }} title={`Frame ${frameIndex(cursorS, frameS) + 1}`}>
          {timecode(cursorS, frameS)}
        </span>
      }
    >
      <div className="video dub-video">
        {pictureReady && (
          <video
            ref={videoRef}
            muted
            playsInline
            src={sources.pictureUrl ?? undefined}
            onLoadedData={() => {
              if (attemptStartRef.current) {
                attemptStartRef.current = false;
                clock.play();
              }
            }}
          />
        )}
      </div>
      {/* The transport; the volume and the user's own player sit at its end. */}
      <div className="row" style={{ gap: 2, marginTop: -8 }}>
        <Cmd icon={playing ? <PauseRegular /> : <PlayRegular />} title={playing ? "Pause (Space)" : "Play (Space)"} onClick={togglePlay} />
        <Cmd icon={<PreviousFrameRegular />} title="Back one frame ([)" onClick={() => step(-1)} />
        <Cmd icon={<NextFrameRegular />} title="Forward one frame (])" onClick={() => step(1)} />
        <Cmd icon={<ArrowRepeatAllRegular />} title="Loop 4 s" aria-pressed={loop} className={cx(loop && "dub-on")} onClick={toggleLoop} />
        <span className="grow" />
        <span className="dub-vol">
          <Slider value={volume} min={0} max={100} onChange={setVolume} label="Volume" />
        </span>
        {onOpenExternal && <Cmd icon={<OpenRegular />} title="Open in your player" onClick={onOpenExternal} />}
      </div>
      <div className="col" style={{ gap: 6 }}>
        <span className="t3">Listen to</span>
        <Seg<MixMode> items={MIXES} value={mix} onChange={onMix} label="What plays" />
      </div>
      {message && <span className="t3 sm">{message}</span>}
    </Box>
  );
}
