/** The waveforms of one pair at the top of the Dub sync tab.
 *
 *  As soon as a video and its dub are paired, both are drawn: the dub laid
 *  at the video's start, as loaded, so the mismatch is there to see before
 *  anything runs. When the sync runs, the dub lane is redrawn from each
 *  draft the engine sends -- the coarse stretches, then the measured
 *  ones, then the cuts, then the gaps searched -- so the track can be
 *  watched being laid onto the picture; when it is done, the plan the
 *  track was written from stays up, and the cut editor opens on it.
 */

import { Play, Scissors } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { DubPreviewPlayer } from "@/components/DubPreviewPlayer";
import { DubWaveformView, type WaveformStatus } from "@/components/DubWaveformView";
import { Button } from "@/components/ui";
import type { DubExcerpt, DubPreviewRequest, WaveformPeaks, WaveformReady, WaveformRequest } from "@/lib/api";
import { asLoadedPlan, frameSeconds } from "@/lib/dubPlanEdit";
import { windowAround } from "@/lib/previewClock";
import type { MixMode } from "@/lib/previewMix";
import { useExcerpt } from "@/lib/useExcerpt";
import type { DubSyncPlan } from "@/lib/types";

export interface WaveformPair {
  videoPath: string;
  dubPath: string;
  videoTrack: number;
  dubTrack: number;
  name: string;
  dubName: string;
}

export interface DubWaveformStripProps {
  pair: WaveformPair;
  /** The plan to draw -- the engine's draft mid-analysis, or its plan --
   *  or null before any analysis, when the pair is drawn as loaded. */
  plan: DubSyncPlan | null;
  status: WaveformStatus | null;
  /** The pairs (or jobs) on offer, and which is shown. */
  choices: { name: string }[];
  index: number;
  onChoose: (index: number) => void;
  /** Opens the cut editor; present once the plan is final. */
  onEdit?: () => void;
  /** The plan is final and the queue is not running: the player may play it. */
  playable?: boolean;
  fetchPeaks: (request: WaveformRequest) => Promise<WaveformPeaks>;
  buildWaveform: (path: string, track: number) => Promise<WaveformReady>;
  /** Renders a span of the plan for the in-app player. */
  renderDubPreview: (request: DubPreviewRequest) => Promise<DubExcerpt | null>;
  /** Bytes of a rendered excerpt, for the in-app player. */
  readPreviewBytes: (path: string) => Promise<ArrayBuffer>;
  /** Files the engine is reading for the first time: path to percent. */
  reading: Record<string, number>;
}

export function DubWaveformStrip({
  pair,
  plan,
  status,
  choices,
  index,
  onChoose,
  onEdit,
  playable = false,
  fetchPeaks,
  buildWaveform,
  renderDubPreview,
  readPreviewBytes,
  reading,
}: DubWaveformStripProps) {
  // Lengths of files read so far, for drawing a pair as loaded.
  const [lengths, setLengths] = useState<Record<string, number>>({});
  const [failed, setFailed] = useState<string | null>(null);
  const requested = useRef<Set<string>>(new Set());
  // The in-app player, below the waveform. The strip has no sample mode:
  // Play sample is an editor thing.
  const [cursorS, setCursorS] = useState<number | null>(null);
  const [playerOpen, setPlayerOpen] = useState(false);
  const [playerWindow, setPlayerWindow] = useState<{ startS: number; endS: number } | null>(null);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [mix, setMix] = useState<MixMode>("dub");

  const closePlayer = useCallback(() => {
    setPlayerOpen(false);
    setPlayerWindow(null);
    setAudioContext((context) => {
      void context?.close().catch(() => undefined);
      return null;
    });
  }, []);

  // A different pair is a different film: the player's window means
  // nothing to it, so the player closes.
  useEffect(() => {
    closePlayer();
  }, [pair.videoPath, pair.dubPath, closePlayer]);

  // Read both files as soon as the pair is on screen: the engine keeps
  // what it read, so every view that follows is immediate.
  useEffect(() => {
    let cancelled = false;
    const files = [
      { path: pair.videoPath, track: pair.videoTrack },
      { path: pair.dubPath, track: pair.dubTrack },
    ];
    (async () => {
      for (const file of files) {
        const key = `${file.path}#${file.track}`;
        if (requested.current.has(key)) continue;
        requested.current.add(key);
        try {
          const ready = await buildWaveform(file.path, file.track);
          if (cancelled) return;
          setLengths((current) => ({ ...current, [file.path]: ready.durationS }));
        } catch (error) {
          requested.current.delete(key);
          if (!cancelled) setFailed(error instanceof Error ? error.message : String(error));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pair.videoPath, pair.dubPath, pair.videoTrack, pair.dubTrack, buildWaveform]);

  const videoS = lengths[pair.videoPath];
  const dubS = lengths[pair.dubPath];
  const shown =
    plan ??
    (videoS !== undefined && dubS !== undefined
      ? asLoadedPlan({
          videoPath: pair.videoPath,
          dubPath: pair.dubPath,
          videoTrack: pair.videoTrack,
          dubTrack: pair.dubTrack,
          videoDurationS: videoS,
          dubDurationS: dubS,
        })
      : null);

  const readingLine = Object.entries(reading)
    .filter(([path]) => path === pair.videoPath || path === pair.dubPath)
    .map(([path, percent]) => `${path === pair.videoPath ? pair.name : pair.dubName} ${percent}%`)
    .join(" · ");

  /** Open the player on the plan as it stands. The AudioContext is made
   *  here, in the button press, which is what Web Audio asks for. */
  const openPlayer = () => {
    if (!shown) return;
    const context = audioContext ?? new AudioContext();
    void context.resume().catch(() => undefined);
    setAudioContext(context);
    setCursorS((current) => current ?? 0);
    setPlayerWindow(windowAround(cursorS ?? 0, shown.videoDurationS));
    setPlayerOpen(true);
  };

  return (
    <section className="shrink-0 border-b border-border bg-card px-[18px] pb-3 pt-2" aria-label="Waveforms">
      <div className="mb-2 flex h-7 items-center gap-3">
        {choices.length > 1 ? (
          <select
            value={index}
            onChange={(event) => onChoose(Number(event.target.value))}
            aria-label="Pair shown"
            className="h-7 max-w-[360px] truncate rounded-md border border-border-strong bg-input px-2 text-[12px] focus:outline-none focus:ring-2 focus:ring-ring/40"
          >
            {choices.map((choice, i) => (
              <option key={i} value={i}>
                {choice.name}
              </option>
            ))}
          </select>
        ) : (
          <span className="truncate text-[12px] font-medium" title={pair.videoPath}>
            {pair.name}
          </span>
        )}
        <span className="min-w-0 flex-1 truncate text-[11.5px] text-muted-foreground">
          {plan
            ? status
              ? "The dub is being laid onto the video: every stretch moves into place as it is found."
              : "The synced track as it was written. Ctrl/⌘-wheel zooms, Shift-drag pans."
            : shown
              ? "As loaded: the dub at the video's start. Run the sync to lay it onto the picture."
              : failed
                ? `Could not read the waveforms: ${failed}`
                : readingLine
                  ? `Reading the waveforms — ${readingLine}`
                  : "Reading the waveforms…"}
        </span>
        {playable && shown && (
          <Button size="sm" variant="default" onClick={openPlayer} aria-pressed={playerOpen}>
            <Play className="h-3.5 w-3.5 fill-current" aria-hidden />
            Play video
          </Button>
        )}
        {onEdit && (
          <Button size="sm" variant="default" onClick={onEdit}>
            <Scissors className="h-3.5 w-3.5" aria-hidden />
            Edit the cuts
          </Button>
        )}
      </div>
      {shown ? (
        <DubWaveformView
          plan={shown}
          fetchPeaks={fetchPeaks}
          laneHeight={84}
          status={status}
          reading={reading}
          cursorS={cursorS}
          onCursor={setCursorS}
          keepCursorInView={playerOpen}
        />
      ) : (
        <div className="flex h-[192px] items-center justify-center rounded-md bg-sunken text-[11.5px] text-muted-foreground">
          {failed ? "No waveforms." : readingLine ? `Reading — ${readingLine}` : "Reading the files…"}
        </div>
      )}
      {playerOpen && shown && playerWindow && audioContext && (
        <div className="mt-3">
          <StripPlayer
            plan={shown}
            windowStart={playerWindow.startS}
            windowEnd={playerWindow.endS}
            cursorS={cursorS ?? 0}
            onCursor={setCursorS}
            onNeedWindow={(aroundS) => setPlayerWindow(windowAround(aroundS, shown.videoDurationS))}
            onClose={closePlayer}
            audioContext={audioContext}
            mix={mix}
            onMix={setMix}
            render={renderDubPreview}
            read={readPreviewBytes}
          />
        </div>
      )}
    </section>
  );
}

/** The strip's player, in its own component so the excerpt hook is only
 *  mounted while the player is open and a plan exists to play. */
function StripPlayer({
  plan,
  windowStart,
  windowEnd,
  cursorS,
  onCursor,
  onNeedWindow,
  onClose,
  audioContext,
  mix,
  onMix,
  render,
  read,
}: {
  plan: DubSyncPlan;
  windowStart: number;
  windowEnd: number;
  cursorS: number;
  onCursor: (timeS: number) => void;
  onNeedWindow: (aroundS: number) => void;
  onClose: () => void;
  audioContext: AudioContext;
  mix: MixMode;
  onMix: (mix: MixMode) => void;
  render: (request: DubPreviewRequest) => Promise<DubExcerpt | null>;
  read: (path: string) => Promise<ArrayBuffer>;
}) {
  const sources = useExcerpt({
    plan,
    wantedStartS: windowStart,
    wantedEndS: windowEnd,
    withPicture: true,
    wantOriginal: mix !== "dub",
    audioContext,
    render,
    read,
  });
  return (
    <DubPreviewPlayer
      frameS={frameSeconds(plan)}
      cursorS={cursorS}
      onCursor={onCursor}
      withPicture
      sources={sources}
      onNeedWindow={onNeedWindow}
      onClose={onClose}
      audioContext={audioContext}
      mix={mix}
      onMix={onMix}
    />
  );
}
