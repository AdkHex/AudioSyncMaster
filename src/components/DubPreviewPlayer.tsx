/** The in-app player for the dub sync waveform view.
 *
 *  Plays what the plan would write, so lip-sync can be checked frame by
 *  frame: a muted 480p picture excerpt of the 30 s window around the
 *  cursor (the video element) and the synced track over the same window
 *  as 16-bit WAV (Web Audio), locked to the picture's clock. The picture
 *  is never re-rendered for an edit -- a plan change re-cuts only the
 *  sound, which is swapped in at the current position. The waveform's
 *  cursor follows the playhead; the parent moves it through onCursor.
 *
 *  The player claims Space, [ and ] while it is mounted (capture-phase
 *  listener, stopping propagation), and the parent owns Escape: a first
 *  Escape closes the player, a second the editor. The AudioContext is
 *  created by the parent on the button press that opened the player --
 *  a user gesture -- so nothing here ever creates one. */

import { Pause, Play, Repeat, Volume2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui";
import { cx } from "@/lib/cx";
import type { ExcerptSources } from "@/lib/useExcerpt";
import { MIX_GAINS, type MixMode } from "@/lib/previewMix";
import {
  frameIndex,
  inWindow,
  loopAround,
  needsResync,
  stepFrame,
  timecode,
} from "@/lib/previewClock";

const VOLUME_KEY = "dubsync-player-volume";

export interface DubPreviewPlayerProps {
  /** One video frame, for stepping and the timecode. */
  frameS: number;
  /** The editor's or strip's cursor, on the film's clock. The playhead
   *  writes it (through onCursor); a change made elsewhere is a seek. */
  cursorS: number;
  onCursor: (timeS: number) => void;
  withPicture: boolean;
  sources: ExcerptSources;
  /** Ask the parent to move the window: the cursor left it. */
  onNeedWindow: (aroundS: number) => void;
  onClose: () => void;
  audioContext: AudioContext;
  mix: MixMode;
  onMix: (mix: MixMode) => void;
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

/** The player's clock: the video element's, or the AudioContext's when
 *  there is no picture (sample mode, or a picture that failed). Both
 *  speak one language: seconds since the window's start. */
interface Clock {
  now(): number;
  seek(t: number): void;
  play(): void;
  pause(): void;
}

function loadVolume(): number {
  if (typeof window === "undefined") return 100;
  const stored = Number(window.localStorage.getItem(VOLUME_KEY));
  return Number.isFinite(stored) && stored >= 0 && stored <= 100 ? stored : 100;
}

export function DubPreviewPlayer({
  frameS,
  cursorS,
  onCursor,
  withPicture,
  sources,
  onNeedWindow,
  onClose,
  audioContext,
  mix,
  onMix,
}: DubPreviewPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [volume, setVolume] = useState(loadVolume);

  const graphRef = useRef<Graph | null>(null);
  const activeRef = useRef<AudioBufferSourceNode[]>([]);
  const startedRef = useRef({ ctx: 0, media: 0 });
  const lastEmittedRef = useRef(0);
  const lastWindowStartRef = useRef<number | null>(null);
  const loopRef = useRef<{ a: number; b: number } | null>(null);
  const attemptStartRef = useRef(false);
  const timeRef = useRef<HTMLSpanElement>(null);

  // The clock's choice: a picture that loaded plays the video element;
  // anything else ticks on the AudioContext.
  const pictureReady = withPicture && sources.pictureUrl !== null;
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

  // Everything the per-frame loop and the seek effect read must be the
  // latest values without re-subscribing them, so they live in refs.
  const sourcesRef = useRef(sources);
  sourcesRef.current = sources;
  const onCursorRef = useRef(onCursor);
  onCursorRef.current = onCursor;
  const onNeedWindowRef = useRef(onNeedWindow);
  onNeedWindowRef.current = onNeedWindow;
  const playingRef = useRef(playing);
  playingRef.current = playing;

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
    // The graph is per context: the parent creates one per player.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioContext]);

  useEffect(() => {
    if (graphRef.current) applyMix(graphRef.current, mix);
  }, [mix]);

  useEffect(() => {
    if (graphRef.current) graphRef.current.master.gain.value = volume / 100;
    if (typeof window !== "undefined") window.localStorage.setItem(VOLUME_KEY, String(volume));
  }, [volume]);

  // -------------------------------------------------------------- the sound

  function stopSources() {
    for (const source of activeRef.current) {
      try {
        source.stop();
      } catch {
        // Stopped already; a seek can land on one twice.
      }
      source.disconnect();
    }
    activeRef.current = [];
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
      const offset = Math.max(0, Math.min(t, Math.max(0, buffer.duration - 0.001)));
      source.start(0, offset);
      activeRef.current.push(source);
    }
  }

  function updateTimecode(t: number) {
    const span = timeRef.current;
    if (!span) return;
    const text = `${timecode(t, frameS)} · frame ${frameIndex(t, frameS) + 1}`;
    if (span.textContent !== text) span.textContent = text;
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
    // Seek a quarter frame inside the target: seeking exactly onto a
    // frame boundary presents the frame before it, so a forward step
    // would not move. The displayed frame is still the target's -- a
    // quarter frame cannot cross into the next one.
    clock.seek(Math.min(next - startS + frameS * 0.25, endS - startS));
    lastEmittedRef.current = next;
    onCursorRef.current(next);
    updateTimecode(next);
  };

  const toggleLoop = () => {
    setLoop((on) => {
      const next = !on;
      if (next) {
        const { startS, endS } = sourcesRef.current;
        if (startS !== null && endS !== null) {
          loopRef.current = loopAround(lastEmittedRef.current, startS, endS);
        }
      } else {
        loopRef.current = null;
      }
      return next;
    });
  };

  // ------------------------------------------------------------- the clock

  const onFrame = (mediaTime: number) => {
    const { startS } = sourcesRef.current;
    if (startS === null) return;
    const filmTime = startS + mediaTime;
    lastEmittedRef.current = filmTime;
    onCursorRef.current(filmTime);
    updateTimecode(filmTime);

    // A loop: at its end, jump back to its start and keep going.
    const region = loopRef.current;
    if (region && playingRef.current && filmTime >= region.b) {
      const at = region.a - startS;
      clock.seek(at);
      startSources(at);
      return;
    }

    // Drift: once the picture and the sound are 20 ms apart, re-lock.
    if (playingRef.current) {
      const expected = audioContext.currentTime - startedRef.current.ctx + startedRef.current.media;
      if (needsResync(mediaTime, expected)) startSources(mediaTime);
    }
  };

  // The video's presented-frame callback; in sample mode, an rAF loop on
  // the AudioContext clock.
  useEffect(() => {
    if (!pictureReady) return;
    const video = videoRef.current;
    if (!video) return;
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

  // The sound arriving is the moment to begin; a later plan edit swaps it
  // in at the current position without touching the picture.
  const ready =
    sources.startS !== null &&
    sources.loading === null &&
    sources.dub !== null &&
    (withPicture ? sources.pictureUrl !== null || sources.error !== null : true);

  useEffect(() => {
    if (!playingRef.current || !sources.dub) return;
    // The picture already plays; only the sound needs restarting at the
    // position it was at.
    startSources();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sources.dub]);

  useEffect(() => {
    if (!ready) return;
    setPlaying(true);
    attemptStartRef.current = true;
    clock.play();
    startSources();
    // Begin once, on the first load; edits and window moves keep the
    // transport as it was.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // A cursor change this player did not make is a seek: inside the window
  // it jumps there, outside it asks the parent for a new window.
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
      updateTimecode(cursorS);
    } else {
      onNeedWindowRef.current(cursorS);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursorS, sources.startS, sources.endS]);

  // ---------------------------------------------------------------- the keys

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if (event.metaKey || event.ctrlKey) return;
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
    // The handlers read refs; the listener lives for the player's life.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadingLabel =
    sources.loading === "picture"
      ? "Loading the picture…"
      : sources.loading === "audio"
        ? "Rendering the sound…"
        : sources.loading === "original"
          ? "Loading the original…"
          : null;

  return (
    <div className="flex min-w-0 flex-col gap-2" aria-label="Preview player">
      {pictureReady && (
        <div className="w-full overflow-hidden rounded-md border border-border bg-black">
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
            className="mx-auto block h-[360px] w-auto max-w-full object-contain"
          />
        </div>
      )}
      <div className="flex h-[30px] shrink-0 items-center gap-1.5">
        <Button size="sm" variant="ghost" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <Pause className="h-3.5 w-3.5 fill-current" aria-hidden /> : <Play className="h-3.5 w-3.5 fill-current" aria-hidden />}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => step(-1)} aria-label="One frame back" className="font-mono">
          [
        </Button>
        <Button size="sm" variant="ghost" onClick={() => step(1)} aria-label="One frame forward" className="font-mono">
          ]
        </Button>
        <Button
          size="sm"
          variant={loop ? "default" : "ghost"}
          onClick={toggleLoop}
          aria-label="Loop four seconds"
          aria-pressed={loop}
        >
          <Repeat className="h-3.5 w-3.5" aria-hidden />
        </Button>
        <span ref={timeRef} className="tabular w-[168px] shrink-0 font-mono text-[11px] text-muted-foreground">
          0:00:00:00 · frame 1
        </span>
        <span className="mx-1 h-5 w-px shrink-0 bg-border" aria-hidden />
        <div className="flex shrink-0 overflow-hidden rounded-md border border-border-strong" role="group" aria-label="What plays">
          {(["dub", "original", "both"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => onMix(mode)}
              aria-pressed={mix === mode}
              className={cx(
                "h-[26px] px-2.5 text-[11px] font-medium transition-colors",
                mix === mode ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-secondary hover:text-foreground",
              )}
            >
              {mode === "dub" ? "Dub" : mode === "original" ? "Original" : "Both"}
            </button>
          ))}
        </div>
        <Volume2 className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
        <input
          type="range"
          min={0}
          max={100}
          value={volume}
          onChange={(event) => setVolume(Number(event.target.value))}
          aria-label="Volume"
          className="h-1 w-[90px] shrink-0 cursor-pointer accent-[hsl(var(--primary))]"
        />
        <span className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">
          {loadingLabel ?? sources.error ?? "What you hear is the track as it will be written."}
        </span>
        <Button size="sm" variant="ghost" onClick={onClose} aria-label="Close the player">
          <X className="h-3.5 w-3.5" aria-hidden />
        </Button>
      </div>
    </div>
  );
}

function applyMix(graph: Graph, mix: MixMode) {
  const table = MIX_GAINS[mix];
  graph.dub.gain.value = table.dub;
  graph.orig.gain.value = table.original;
  graph.dubPan.pan.value = table.dubPan;
  graph.origPan.pan.value = table.originalPan;
}
