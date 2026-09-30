/** What Movies, Series and Find match draw: the list (or grid) of files and
 *  results on the left, the selected item's details on the right — exactly
 *  the mockup's screens/analyse.tsx, fed with real data. */

import {
  AddRegular,
  ArrowRightRegular,
  CheckmarkCircleFilled,
  CopyRegular,
  DismissRegular,
  FilterRegular,
  FolderRegular,
  MusicNote2Regular,
  PlayRegular,
  VideoClipMultipleRegular,
  VideoClipRegular,
} from "@fluentui/react-icons";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { SyncState } from "@/lib/syncReducer";
import {
  confidenceLevel,
  formatDuration,
  formatElapsed,
  formatFps,
  formatSize,
  frameOffset,
  playerDelayMs,
  resultKey,
  streamSummary,
  type AudioTrackInfo,
  type FileItem,
  type MatchPair,
  type MediaProbe,
  type PairingReport,
  type SyncMode,
  type SyncResult,
  type TrackListing,
} from "@/lib/types";
import { Box } from "@/ui/frame";
import {
  Btn,
  Chk,
  Cmd,
  Combo,
  DL,
  Empty,
  Links,
  Meter,
  MidText,
  Ring,
  Seg,
  Status,
  Table,
  Tr,
  baseName,
  cx,
  type St,
} from "@/ui/kit";
import { Wave } from "@/ui/wave";

import type { ApplyState } from "./AnalysePage";

type Mode = Exclude<SyncMode, "dubsync" | "subsync">;

export type Filter = "all" | "high" | "medium" | "low" | "drift" | "cut" | "failed";

/* ---------------------------------------------------------------- helpers */

const Dash = () => <span className="t3">—</span>;

/** A player delay as the design writes it: a true minus sign and thousands
 *  separators ("−317.0 ms", "+1,208.0 ms"). */
export function delayText(engineMs: number | null | undefined, unit = true): string {
  const ms = playerDelayMs(engineMs ?? null);
  if (ms === null || !Number.isFinite(ms)) return "—";
  const abs = Math.abs(ms).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const sign = abs === "0.0" ? "" : ms > 0 ? "+" : "\u2212";
  return `${sign}${abs}${unit ? " ms" : ""}`;
}

/** How long a run or a file took: "41.2 s", "1m 58s". */
export function tookText(ms: number | null | undefined): string {
  if (!ms || ms <= 0) return "—";
  if (ms < 60_000) return formatElapsed(ms);
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

/** A chart label in seconds, signed, without a unit: "−0.3", "+5.3", "+462". */
const chartSeconds = (playerMs: number) => {
  const s = playerMs / 1000;
  const abs = Math.abs(s) >= 100 ? Math.round(Math.abs(s)).toString() : Math.abs(s).toFixed(1);
  return `${s > 0 && abs !== "0.0" ? "+" : s < 0 && abs !== "0.0" ? "\u2212" : ""}${abs}`;
};

const CODEC_NAMES: Record<string, string> = { truehd: "TrueHD", eac3: "E-AC3", ac3: "AC3", dts: "DTS", aac: "AAC", flac: "FLAC", opus: "Opus", mp3: "MP3", pcm_s16le: "PCM", pcm_s24le: "PCM" };
const CHANNELS: Record<number, string> = { 1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1" };
const languages = typeof Intl !== "undefined" && "DisplayNames" in Intl ? new Intl.DisplayNames(["en"], { type: "language" }) : null;

/** An audio track the way the design lists it: "1 · English · TrueHD 7.1". */
export function trackText(t: AudioTrackInfo): string {
  let language: string | null = null;
  if (t.language && !/^(und|unknown|zxx|mis)$/i.test(t.language)) {
    try {
      language = languages?.of(t.language.toLowerCase()) ?? t.language.toUpperCase();
    } catch {
      language = t.language.toUpperCase();
    }
  }
  const codec = t.codec ? (CODEC_NAMES[t.codec.toLowerCase()] ?? t.codec.toUpperCase()) : null;
  const sound = [codec, t.channels ? (CHANNELS[t.channels] ?? `${t.channels}ch`) : null].filter(Boolean).join(" ");
  return [String(t.index + 1), language, t.title, sound || null].filter(Boolean).join(" · ");
}

function Name({ children, icon = <VideoClipRegular />, tail = 18 }: { children: string; icon?: ReactNode; tail?: number }) {
  return (
    <span className="cell">
      <span className="fi" aria-hidden>{icon}</span>
      <MidText text={children} tail={tail} />
    </span>
  );
}

function Conf({ r }: { r?: SyncResult }) {
  if (!r || r.error || r.confidence === null || r.delayMs === null) return <Dash />;
  const c = Math.round(r.confidence * 100);
  return (
    <span className="cell num">
      <Meter pct={c} />
      {c}%
    </span>
  );
}

function Delay({ r }: { r?: SyncResult }) {
  const ms = r ? playerDelayMs(r.delayMs) : null;
  return <span className="r num" style={{ display: "flex" }}>{ms === null || r?.isLikelyCut ? <Dash /> : delayText(r!.delayMs)}</span>;
}

/** Status word for a measured result. */
export function resultState(r: SyncResult): [St, string] {
  if (r.error || r.delayMs === null) return ["bad", shortError(r.error)];
  if (r.isLikelyCut) return ["warn", "Different cut"];
  // A frame-rate change drifts by design, and Fix corrects the speed with the
  // delay; the details say which rates. It is only worth a word when the
  // match is not a confident one.
  if (r.hasSignificantDrift && !r.isRateMismatch) return ["warn", "Drifting"];
  if (confidenceLevel(r) !== "high") return ["warn", r.isRateMismatch ? "Rate change" : "Low confidence"];
  return ["ok", "Matched"];
}

/** A run's outcome in words, counting each result once, as its row says. */
export function resultsOutcome(results: SyncResult[]): { tone: "ok" | "warn"; text: string } {
  let matched = 0, drifting = 0, cuts = 0, low = 0, rate = 0, failed = 0;
  for (const r of results) {
    const [tone, text] = resultState(r);
    if (tone === "bad") failed += 1;
    else if (text === "Different cut") cuts += 1;
    else if (text === "Drifting") drifting += 1;
    else if (text === "Rate change") rate += 1;
    else if (tone === "warn") low += 1;
    else matched += 1;
  }
  const parts = [`${matched} matched`];
  if (drifting) parts.push(`${drifting} drifting`);
  if (cuts) parts.push(`${cuts} different cut${cuts === 1 ? "" : "s"}`);
  if (low) parts.push(`${low} low confidence`);
  if (rate) parts.push(`${rate} rate change${rate === 1 ? "" : "s"}`);
  if (failed) parts.push(`${failed} failed`);
  return { tone: drifting || cuts || low || rate || failed ? "warn" : "ok", text: parts.join(" · ") };
}

function shortError(error: string | null): string {
  if (!error) return "Failed";
  if (/no audio/i.test(error)) return "No audio";
  if (/no video/i.test(error)) return "No video";
  return "Failed";
}

/** "s01e003" → "S01E03"; anything else as it is. */
export function episodeCode(key: string | null | undefined, name?: string): string {
  // "1x05" only as a word of its own, or "1920x1080" would read as S20E108.
  const m = key?.match(/^s(\d+)e(\d+)$/i) ?? name?.match(/s(\d{1,2})[ ._-]?e(\d{1,3})/i) ?? name?.match(/(?:^|[\s._-])(\d{1,2})x(\d{2,3})(?=$|[\s._-])/i);
  if (!m) return "—";
  return `S${m[1].padStart(2, "0")}E${String(Number(m[2])).padStart(2, "0")}`;
}

const pairedBy = (pair: MatchPair): { text: string; warn?: boolean } => {
  if (pair.method === "chosen by hand") return { text: "By hand" };
  if (pair.method.startsWith("episode")) return { text: "Episode number" };
  if (pair.method === "list order") return { text: "List order", warn: true };
  if (pair.method.includes("filename")) return { text: pair.score < 1 && pair.score > 0 ? `Name · ${Math.round(pair.score * 100)}%` : "Name", warn: pair.score > 0 && pair.score < 0.6 };
  return { text: pair.method };
};

/* ------------------------------------------------------------------ filter */

export function applyFilter(results: SyncResult[], filter: Filter): SyncResult[] {
  switch (filter) {
    case "all":
      return results;
    case "drift":
      return results.filter((r) => r.hasSignificantDrift && !r.isLikelyCut);
    case "cut":
      return results.filter((r) => r.isLikelyCut);
    case "failed":
      return results.filter((r) => r.error || r.delayMs === null);
    default:
      return results.filter((r) => !r.error && r.delayMs !== null && confidenceLevel(r) === filter);
  }
}

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All results" },
  { id: "high", label: "High confidence" },
  { id: "medium", label: "Medium confidence" },
  { id: "low", label: "Low confidence" },
  { id: "drift", label: "Drifting" },
  { id: "cut", label: "Different cut" },
  { id: "failed", label: "Failed" },
];

/** The toolbar's Filter command and the menu it opens. */
export function FilterMenu({
  results,
  value,
  open,
  onOpen,
  onChange,
}: {
  results: SyncResult[];
  value: Filter;
  open: boolean;
  onOpen: (open: boolean) => void;
  onChange: (filter: Filter) => void;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) onOpen(false);
    };
    const key = (event: KeyboardEvent) => event.key === "Escape" && onOpen(false);
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", key);
    };
  }, [open, onOpen]);
  return (
    <span ref={ref} style={{ position: "relative", display: "inline-flex" }}>
      <Cmd
        icon={<FilterRegular />}
        title={value === "all" ? "Filter" : `Filter: ${FILTERS.find((f) => f.id === value)?.label}`}
        aria-haspopup="menu"
        aria-expanded={open}
        className={cx(value !== "all" && "on")}
        onClick={() => onOpen(!open)}
      />
      {open && (
        <div className="fly" role="menu" style={{ top: 36, left: 0, width: 240 }}>
          {FILTERS.map((f) => {
            const count = applyFilter(results, f.id).length;
            return (
              <button
                key={f.id}
                type="button"
                role="menuitemradio"
                aria-checked={f.id === value}
                className={cx("mi", f.id === value && "on")}
                disabled={count === 0 && f.id !== "all"}
                onClick={() => {
                  onChange(f.id);
                  onOpen(false);
                }}
              >
                <span className="grow">{f.label}</span>
                <span className="t3 sm num">{count}</span>
              </button>
            );
          })}
        </div>
      )}
    </span>
  );
}

/* ---------------------------------------------------------------- charts */

/** Where the sample windows sit along the file (the engine spreads them
 *  evenly; six is the default). */
const windowsAt = (n: number) => (n === 6 ? WINDOWS : Array.from({ length: n }, (_, i) => 0.06 + (i * 0.88) / Math.max(1, n - 1)));

/** The delay across the file: the line fitted through the windows, from the
 *  start to the end, against the flat line a single delay would be. The
 *  engine reports the fit, not each window, so the windows are drawn on it. */
function DriftChart({ r }: { r: SyncResult }) {
  const start = playerDelayMs(r.startDelayMs ?? r.delayAtStartMs ?? r.delayMs) ?? 0;
  const end = playerDelayMs(r.endDelayMs ?? r.delayMs) ?? start;
  const W = 268, H = 110;
  const up = end >= start;
  const X = (x: number) => 30 + x * (W - 36);
  const Y = (x: number) => (up ? 90 - x * 70 : 20 + x * 70);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img" aria-label={`Delay goes from ${delayText(r.startDelayMs)} to ${delayText(r.endDelayMs)}`}>
      {[20, 55, 90].map((y) => <line key={y} x1="30" x2={W} y1={y} y2={y} stroke="var(--divider)" />)}
      <text x="24" y="24" fontSize="10" textAnchor="end" fill="var(--text-3)">{chartSeconds(Math.max(start, end))}</text>
      <text x="24" y="94" fontSize="10" textAnchor="end" fill="var(--text-3)">{chartSeconds(Math.min(start, end))}</text>
      <text x="30" y={H - 2} fontSize="10" fill="var(--text-3)">0:00</text>
      <text x={W} y={H - 2} fontSize="10" textAnchor="end" fill="var(--text-3)">{formatDuration(r.primaryDurationS).replace(/:\d\d$/, "")}</text>
      <line x1={X(0)} x2={X(1)} y1={Y(0)} y2={Y(0)} stroke="var(--text-4)" strokeDasharray="3 3" />
      <line x1={X(0)} x2={X(1)} y1={Y(0)} y2={Y(1)} stroke="var(--warn)" strokeWidth="1.5" />
      {windowsAt(r.windowsTotal ?? 6).map((x) => <circle key={x} cx={X(x)} cy={Y(x)} r="3" fill="var(--text)" stroke="var(--text)" />)}
    </svg>
  );
}

/** Where the scene is missing: the delay steps at the cut. A window that
 *  straddles the cut measures neither side, and is drawn between. */
function CutChart({ r }: { r: SyncResult }) {
  const W = 268, H = 110;
  const X = (x: number) => 30 + x * (W - 36);
  const dur = r.primaryDurationS ?? 0;
  const at = dur && r.cutPositionS ? Math.min(0.95, Math.max(0.05, r.cutPositionS / dur)) : 0.4;
  const before = playerDelayMs(r.startDelayMs);
  const after = playerDelayMs(r.endDelayMs);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }} role="img" aria-label="The delay steps where a scene is missing">
      {[20, 55, 90].map((y) => <line key={y} x1="30" x2={W} y1={y} y2={y} stroke="var(--divider)" />)}
      <text x="24" y="24" fontSize="10" textAnchor="end" fill="var(--text-3)">{before === null ? "" : chartSeconds(before)}</text>
      <text x="24" y="94" fontSize="10" textAnchor="end" fill="var(--text-3)">{after === null ? "" : chartSeconds(after)}</text>
      <text x="30" y={H - 2} fontSize="10" fill="var(--text-3)">0:00</text>
      <text x={W} y={H - 2} fontSize="10" textAnchor="end" fill="var(--text-3)">{formatDuration(r.primaryDurationS).replace(/:\d\d$/, "")}</text>
      <line x1={X(at)} x2={X(at)} y1="10" y2="96" stroke="var(--warn)" strokeDasharray="3 3" />
      <line x1={X(0)} x2={X(at)} y1="30" y2="30" stroke="var(--text-2)" strokeWidth="1.5" />
      <line x1={X(at)} x2={X(1)} y1="86" y2="86" stroke="var(--text-2)" strokeWidth="1.5" />
      {windowsAt(r.windowsTotal ?? 6).map((x) => {
        const straddles = Math.abs(x - at) < 0.03;
        return <circle key={x} cx={X(x)} cy={straddles ? 58 : x < at ? 30 : 86} r="3" fill="var(--text)" stroke="var(--text)" />;
      })}
    </svg>
  );
}

/** A picture of what the correction does: the dub moved onto the video's own
 *  audio by the measured delay. Drawn, not measured — the shape stands for
 *  the audio, the offset is the real one. */
function Alignment({ r, after }: { r: SyncResult; after: boolean }) {
  const seed = [...(r.primaryPath ?? r.videoFile)].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7) % 997;
  const ms = playerDelayMs(r.delayMs) ?? 0;
  const shift = after ? 0 : Math.max(-24, Math.min(24, -ms / 25));
  return (
    <div className="col" style={{ gap: 2 }} title="An illustration of the offset">
      <div style={{ height: 32 }}><Wave seed={seed} color="var(--wave-a)" /></div>
      <div style={{ height: 32, transform: `translateX(${shift}px)`, transition: "transform 250ms var(--ease)" }}><Wave seed={seed} color="var(--wave-b)" /></div>
    </div>
  );
}

/* --------------------------------------------------------------- details */

function ResultDetails({
  r,
  previewing,
  onPreview,
  onCopyCommand,
  onOpenDubSync,
  showSources,
  busy,
  episode,
}: {
  r: SyncResult;
  previewing: boolean;
  onPreview: () => void;
  onCopyCommand: () => void;
  onOpenDubSync: () => void;
  showSources?: boolean;
  /** A run is going: previews wait for the engine. */
  busy?: boolean;
  /** Series: the episode's code, shown with its dub's name. */
  episode?: string;
}) {
  const title = episode ? (
    <span>
      <span className="t2 num">{episode}</span> <span>{r.audioFile}</span>
    </span>
  ) : (
    <MidText text={r.videoFile} tail={22} />
  );
  const frames = frameOffset(playerDelayMs(r.delayMs), r.primaryFps);
  const rows: [ReactNode, ReactNode][] = [];
  const pct = r.confidence === null ? null : Math.round(r.confidence * 100);
  if (pct !== null) rows.push(["Confidence", `${pct}%`]);
  if (r.startDelayMs !== null) rows.push(["At the start", delayText(r.startDelayMs)]);
  if (r.endDelayMs !== null) rows.push(["At the end", delayText(r.endDelayMs)]);
  // The correction's own offset, when it reads differently from the first window's.
  if (r.delayAtStartMs !== null && delayText(r.delayAtStartMs) !== delayText(r.startDelayMs ?? r.delayMs)) rows.push(["From t=0", delayText(r.delayAtStartMs)]);
  if (r.windowsUsed !== null) rows.push(["Windows", `${r.windowsUsed} of ${r.windowsTotal ?? "?"}`]);
  if (r.elapsedMs) rows.push(["Time", tookText(r.elapsedMs)]);
  if (showSources) {
    rows.push(["Video", `${r.primaryCodec?.toUpperCase() ?? "—"}${r.primaryFps ? ` · ${formatFps(r.primaryFps)} fps` : ""}`]);
    rows.push(["Dub", r.audioFile]);
  }

  // Only what the figures above do not already say: the codec delay taken
  // out, and why a frame-rate change is corrected the way it is.
  const notes = (
    <>
      {!!r.codecDelayMs && (
        <span className="t3 sm">
          {delayText(-r.codecDelayMs).replace(/^\+/, "")} of codec delay was removed: one side is a raw {r.secondaryCodec?.toUpperCase() ?? "audio"} stream whose decoder
          priming would otherwise land in the measurement.
        </span>
      )}
    </>
  );
  const links = (
    <Links>
      <button type="button" className="cmd" onClick={onPreview} disabled={previewing || busy} title={busy ? "Available once the run finishes" : undefined}>
        <span className="ic" aria-hidden>{previewing ? <Ring size={16} /> : <PlayRegular />}</span>Preview
      </button>
      <button type="button" className="cmd" onClick={onCopyCommand}>
        <span className="ic" aria-hidden><CopyRegular /></span>Copy command
      </button>
    </Links>
  );

  if (r.error || r.delayMs === null)
    return (
      <Box title={title}>
        <div>
          <div className="t3">Result</div>
          <div className="big bad">Failed</div>
          <div className="t2">{r.error ?? "No delay could be measured."}</div>
        </div>
      </Box>
    );

  if (r.isLikelyCut)
    return (
      <Box title={title}>
        <div>
          <div className="t3">Result</div>
          <div className="big">Different cut</div>
          <div className="t2">
            {r.cutPositionS ? `A scene is missing near ${formatDuration(r.cutPositionS)}.` : "The windows land on different offsets: this is another edit."}
          </div>
        </div>
        <CutChart r={r} />
        <Btn icon={<ArrowRightRegular />} onClick={onOpenDubSync}>Open in Dub sync</Btn>
        {notes}
      </Box>
    );

  // A frame-rate change drifts, but Fix corrects the speed: it reads as one delay.
  const drifting = !!r.hasSignificantDrift && !r.isRateMismatch;
  const ratio = r.rateDiagnosis?.correctionRatio ?? (r.rateDiagnosis?.isRateMismatch ? r.rateDiagnosis.speedRatio : null);
  const rate = r.isRateMismatch && r.rateDiagnosis?.sourceFps && r.rateDiagnosis?.targetFps
    ? { sourceFps: r.rateDiagnosis.sourceFps, targetFps: r.rateDiagnosis.targetFps }
    : null;
  const slowed = ratio !== null && ratio < 1;

  // Series: a dub mastered at another frame rate, and what Fix does about it.
  if (episode && rate)
    return (
      <Box title={title}>
        <div>
          <div className="t3">Delay</div>
          <div className="big">
            {delayText(r.delayMs, false)}
            <small>ms</small>
          </div>
          <div className="warn">{ratio ? `After ${slowed ? "slowing" : "speeding up"} the dub ×${ratio.toFixed(6)}` : `${formatFps(rate.sourceFps)} → ${formatFps(rate.targetFps)} fps`}</div>
        </div>
        <DL
          rows={[
            ["Mastered at", `${formatFps(rate.sourceFps)} fps`],
            ["Video", `${formatFps(rate.targetFps)} fps`],
            ...rows.filter(([k]) => k === "Confidence" || k === "Windows"),
          ]}
        />
        <span className="t3 sm">
          {Math.abs((rate.sourceFps ?? 0) - 25) < 0.01 ? "Made for a PAL broadcast." : `Made for a ${formatFps(rate.sourceFps)} fps master.`} The fixed copy is{" "}
          {slowed ? "slowed" : "sped up"} to the video's speed, then shifted.
        </span>
        {notes}
        {links}
      </Box>
    );

  const sub = drifting
    ? `At the start · drifts ${r.driftMsPerS !== null ? `${Math.abs(r.driftMsPerS).toFixed(3)} ms/s` : ""}`
    : rate
      ? `${formatFps(rate.sourceFps)} → ${formatFps(rate.targetFps)} fps${ratio ? ` · ${slowed ? "slowed" : "sped up"} ×${ratio.toFixed(6)}` : ""}`
      : frames !== null
        ? `${Math.abs(frames)} frame${Math.abs(frames) === 1 ? "" : "s"} · constant`
        : "Constant across the file";

  return (
    <Box title={title}>
      <div>
        <div className="t3">Delay</div>
        <div className="big">
          {delayText(drifting ? (r.delayAtStartMs ?? r.startDelayMs ?? r.delayMs) : r.delayMs, false)}
          <small>ms</small>
        </div>
        <div className={drifting ? "warn" : "t2"}>{sub}</div>
      </div>
      {drifting ? (
        <DriftChart r={r} />
      ) : (
        <AlignmentBlock r={r} />
      )}
      <DL rows={rows} />
      {notes}
      {links}
    </Box>
  );
}

function AlignmentBlock({ r }: { r: SyncResult }) {
  const key = resultKey(r);
  return <AlignmentToggle key={key} r={r} />;
}

function AlignmentToggle({ r }: { r: SyncResult }) {
  const [after, setAfter] = useState(true);
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row">
        <span className="sec grow">Alignment</span>
        <Seg label="Alignment" items={[{ value: 0, label: "Before" }, { value: 1, label: "After" }]} value={after ? 1 : 0} onChange={(v) => setAfter(v === 1)} />
      </div>
      <Alignment r={r} after={after} />
    </div>
  );
}

function FileDetails({
  file,
  probe,
  listing,
  trackChoice,
  onTrackChange,
  kind,
}: {
  file: FileItem;
  probe?: MediaProbe;
  listing?: TrackListing;
  trackChoice: number;
  onTrackChange: (index: number) => void;
  kind: "video" | "audio";
}) {
  const tracks = listing?.tracks ?? probe?.audioTracks ?? [];
  const fps = listing?.fps ?? probe?.fps;
  const rows: [ReactNode, ReactNode][] = [
    ["Duration", formatDuration(listing?.duration ?? probe?.duration)],
  ];
  if (kind === "video") rows.push(["Frame rate", fps ? `${formatFps(fps)} fps` : "—"]);
  rows.push(["Size", formatSize(file.size)]);
  if (tracks.length > 1)
    rows.push([
      "Audio track",
      <Combo
        key="t"
        sm
        w="100%"
        label={`Audio track for ${file.name}`}
        value={trackChoice}
        options={tracks.map((t, i) => ({ value: i, label: trackText({ ...t, index: i }) || t.label || `Track ${i + 1}` }))}
        onChange={onTrackChange}
      />,
    ]);
  else rows.push(["Audio", streamSummary(probe, listing, kind) ?? (probe && !probe.hasAudio ? "None" : "—")]);
  return (
    <Box title={<MidText text={file.name} tail={22} />}>
      <DL rows={rows} />
      {probe && !probe.hasAudio && <span className="bad sm">This file has no audio track to compare.</span>}
      {listing?.error && <span className="bad sm">{listing.error}</span>}
    </Box>
  );
}

const WINDOWS = [0.08, 0.24, 0.4, 0.56, 0.72, 0.88];

function RunDetails({ name, pct, windows }: { name: string; pct: number; windows: number }) {
  const lit = Math.min(windows, Math.floor((pct / 100) * (windows + 1)));
  const at = windowsAt(windows);
  return (
    <Box title={<MidText text={name} tail={22} />}>
      <div className="row"><span className="grow">Measuring</span><span className="t3 num">window {Math.max(1, Math.min(windows, lit + 1))} of {windows}</span></div>
      <div style={{ position: "relative", height: 64 }}>
        <Wave seed={name.length * 17} color="var(--wave-a)" />
        {at.map((x, i) => (
          <span
            key={x}
            style={{
              position: "absolute", top: 0, bottom: 0, left: `${x * 100}%`, width: "6%", borderRadius: 2,
              background: i < lit ? "color-mix(in srgb, var(--ac) 28%, transparent)" : "transparent",
              boxShadow: "inset 0 0 0 1px var(--divider)",
            }}
          />
        ))}
      </div>
    </Box>
  );
}

/* ------------------------------------------------------------ workspace */

export interface WorkspaceProps {
  mode: Mode;
  state: SyncState;
  pairing: PairingReport | null;
  pairingLoading: boolean;
  manualCount: number;
  probes: Record<string, MediaProbe>;
  listings: Record<string, TrackListing>;
  trackChoices: Record<string, number>;
  selectedKeys: Set<string>;
  focus: string | null;
  filter: Filter;
  applyState: ApplyState | null;
  written: { keys: Set<string>; paths: string[] } | null;
  previewingKey: string | null;
  busy: boolean;
  windowCount: number;
  onFilter: (filter: Filter) => void;
  onFocus: (key: string | null) => void;
  onToggle: (key: string) => void;
  onTrackChange: (path: string, index: number) => void;
  onRepair: (videoPath: string, audioPath: string | null) => void;
  onResetRepairs: () => void;
  onAddVideos: () => void;
  onAddAudio: () => void;
  onPreview: (result: SyncResult) => void;
  onCopyCommand: (result: SyncResult) => void;
  onOpenDubSync: () => void;
  onOpenOutput: () => void;
}

/** Movies shows the selected file's details as soon as there are files;
 *  Series and Find match only once a run has results. */
export function withDetails(mode: Mode, state: SyncState): boolean {
  if (mode === "movie") return state.videoFiles.length > 0 || state.audioFiles.length > 0;
  return state.results.length > 0 && state.status !== "processing";
}

export function AnalyseWorkspace(props: WorkspaceProps) {
  const { mode, state } = props;
  if (state.videoFiles.length === 0 && state.audioFiles.length === 0 && state.results.length === 0) return <EmptyPage {...props} />;
  if (mode === "compare") return <MatchGrid {...props} />;
  return <ListPage {...props} />;
}

function EmptyPage({ mode, onAddVideos, onAddAudio }: WorkspaceProps) {
  return (
    <section className="box">
      {mode === "series" ? (
        <Empty icon={<FolderRegular />} title="Drop an episodes folder and a dubs folder">
          <Btn icon={<FolderRegular />} onClick={onAddVideos}>Choose episodes</Btn>
          <Btn icon={<MusicNote2Regular />} onClick={onAddAudio}>Choose dubs</Btn>
        </Empty>
      ) : (
        <Empty icon={<VideoClipMultipleRegular />} title={mode === "compare" ? "Drop up to 5 videos and 5 dubs" : "Drop videos and a dub"}>
          <Btn icon={<AddRegular />} onClick={onAddVideos}>Add videos</Btn>
          <Btn icon={<MusicNote2Regular />} onClick={onAddAudio}>{mode === "compare" ? "Add dubs" : "Add dub"}</Btn>
        </Empty>
      )}
    </section>
  );
}

/** Movies and Series: one row per video (or per result once measured). */
function ListPage(props: WorkspaceProps) {
  const {
    mode, state, pairing, focus, selectedKeys, filter, applyState, written, probes, listings, trackChoices, previewingKey,
    onFocus, onToggle, onTrackChange, onPreview, onCopyCommand, onOpenDubSync,
  } = props;
  const results = state.results;
  const hasResults = results.length > 0 && state.status !== "processing";
  const running = state.status === "processing";
  const byVideo = useMemo(() => new Map(results.map((r) => [r.primaryPath ?? r.videoFile, r])), [results]);
  const keyByVideo = useMemo(() => new Map((pairing?.pairs ?? []).map((p) => [p.primaryPath, p.key])), [pairing]);

  // Series before a run shows the pairing: every episode and its dub.
  if (mode === "series" && !hasResults && !running) return <PairingList {...props} />;

  const series = mode === "series";
  const withChk = hasResults;
  const cols = [withChk && "18px", series && "64px", "minmax(0,1fr)", "88px", "84px", series ? "140px" : "132px"].filter(Boolean).join(" ");
  const head = [withChk && "", series && "Episode", series ? "Video" : "Name", " Delay", "Confidence", "Status"].filter((h) => h !== false) as string[];

  // Rows: measured results once there are any, the videos before. Series
  // keeps every episode in its place, the ones with no dub among them.
  type Row = { key: string; video: string; name: string; r?: SyncResult; unpaired?: boolean };
  const measured: Row[] = applyFilter(results, filter).map((r) => ({ key: resultKey(r), video: r.primaryPath ?? r.videoFile, name: r.videoFile, r }));
  const paired = new Set((pairing?.pairs ?? []).map((p) => p.primaryPath));
  const rows: Row[] = !hasResults
    ? state.videoFiles.map((f) => ({ key: f.path, video: f.path, name: f.name, r: byVideo.get(f.path) }))
    : series && pairing && filter === "all"
      ? [
          ...state.videoFiles.flatMap((f): Row[] => {
            const own = measured.filter((row) => row.video === f.path);
            return own.length ? own : paired.has(f.path) ? [] : [{ key: f.path, video: f.path, name: f.name, unpaired: true }];
          }),
          ...measured.filter((row) => !state.videoFiles.some((f) => f.path === row.video)),
        ]
      : measured;

  // The percent of a row being measured now; undefined while it waits.
  const measuringPct = (row: (typeof rows)[number]): number | undefined => {
    if (row.r) return undefined;
    const pct = state.inFlight[row.name] ?? state.inFlight[baseName(row.video)] ?? state.inFlight[row.video];
    if (pct !== undefined) return pct;
    const current = state.currentFile && (state.currentFile === row.video || baseName(state.currentFile) === row.name);
    return current ? state.fileProgress : undefined;
  };
  // During a run the details follow the file being measured, until a row is
  // picked; a finished row then shows its result.
  // The oldest file still measuring, so the pane does not jump each time
  // another worker starts a file.
  const currentRow = running ? rows.find((row) => measuringPct(row) !== undefined) : undefined;
  const focused = rows.find((row) => row.key === focus) ?? currentRow ?? rows[0];

  const statusFor = (row: (typeof rows)[number]) => {
    if (applyState && row.r) {
      const i = applyState.keys.indexOf(row.key);
      if (i >= 0) return i < applyState.done ? <Status s="written" /> : i === applyState.done ? <Status s="writing" pct={null} text="Writing" /> : <Status s="wait" />;
    }
    if (written?.keys.has(row.key)) return <Status s="written" />;
    if (row.unpaired) return <Status s="bad" text="Not paired" />;
    if (row.r) {
      const [s, text] = resultState(row.r);
      return <Status s={s} text={text} />;
    }
    if (running) {
      const pct = measuringPct(row);
      return pct !== undefined ? <Status s="run" pct={pct} /> : <Status s="wait" />;
    }
    const probe = probes[row.video];
    if (probe && !probe.hasAudio) return <Status s="warn" text="No audio" />;
    return <Status s="ready" />;
  };

  const dub = state.audioFiles[0];
  const title = series ? seriesTitle(state) : "Videos";
  const sub = series ? seriesSub(state, listings) : `${state.videoFiles.length}`;

  let details: ReactNode = null;
  if (focused) {
    if (focused.r)
      details = (
        <ResultDetails
          r={focused.r}
          busy={props.busy}
          episode={series ? episodeCode(keyByVideo.get(focused.video), focused.name) : undefined}
          previewing={previewingKey === resultKey(focused.r)}
          onPreview={() => onPreview(focused.r!)}
          onCopyCommand={() => onCopyCommand(focused.r!)}
          onOpenDubSync={onOpenDubSync}
        />
      );
    else if (focused === currentRow)
      details = <RunDetails name={focused.name} pct={measuringPct(focused) ?? 0} windows={props.windowCount} />;
    else {
      const file = state.videoFiles.find((f) => f.path === focused.video);
      if (file)
        details = (
          <FileDetails
            file={file}
            kind="video"
            probe={probes[file.path]}
            listing={listings[file.path]}
            trackChoice={trackChoices[file.path] ?? 0}
            onTrackChange={(i) => onTrackChange(file.path, i)}
          />
        );
    }
  }
  if (!details && dub)
    details = (
      <FileDetails
        file={dub}
        kind="audio"
        probe={probes[dub.path]}
        listing={listings[dub.path]}
        trackChoice={trackChoices[dub.path] ?? 0}
        onTrackChange={(i) => onTrackChange(dub.path, i)}
      />
    );

  return (
    <>
      <Box
        body={false}
        title={title}
        sub={sub}
        end={
          mode === "movie" ? (
            <span className="cell t2" style={{ paddingRight: 6, maxWidth: 420 }}>
              <span className="fi" aria-hidden><MusicNote2Regular /></span>
              <span className="t3">Dub</span>
              {dub ? <MidText text={dub.name} tail={20} /> : <span className="t3">none yet</span>}
            </span>
          ) : undefined
        }
      >
        <Table cols={cols} head={head} label={title}>
          {rows.map((row) => (
            <Tr key={row.key} on={withDetails(mode, state) && row.key === focused?.key} onClick={() => onFocus(row.key)} label={row.name}>
              {withChk &&
                (row.r && !row.r.error && row.r.delayMs !== null && !row.r.isLikelyCut ? (
                  <Chk name={`Fix ${row.name}`} on={selectedKeys.has(row.key)} onChange={() => onToggle(row.key)} />
                ) : (
                  // Nothing to fix (a failure, another cut): the box stays empty.
                  <span className="chk" aria-hidden />
                ))}
              {series && <span className="num t2">{episodeCode(keyByVideo.get(row.video), row.name)}</span>}
              <Name>{row.name}</Name>
              <Delay r={row.r} />
              <Conf r={row.r} />
              {statusFor(row)}
            </Tr>
          ))}
        </Table>
      </Box>
      {withDetails(mode, state) && (details ?? <section className="box" />)}
    </>
  );
}

/** "Goblin · Season 1", from the episodes' names; the folder's name when
 *  they do not say. */
export function seriesTitle(state: SyncState): string {
  const parsed = state.videoFiles.map((f) => f.name.match(/^(.*?)[\s._-]*(?:S(\d{1,2})[\s._-]?E\d{1,3}|(\d{1,2})x\d{2,3}(?=$|[\s._-]))/i));
  const first = parsed[0];
  const show = first?.[1]?.replace(/[._]+/g, " ").trim();
  if (!first || !show) return folderTitle(state.videoFolder);
  const season = (m: RegExpMatchArray | null) => (m ? Number(m[2] ?? m[3]) : null);
  const one = parsed.every((m) => season(m) === season(first));
  return one ? `${show} · Season ${season(first)}` : show;
}

/** "D:\Media\Goblin (2016)  ·  Hindi dubs in \Hindi": where the episodes are,
 *  and where the dubs are, relative to them when they sit inside. */
function seriesSub(state: SyncState, listings: Record<string, TrackListing>): string {
  const { videoFolder, audioFolder } = state;
  if (!audioFolder) return videoFolder ?? "";
  const tongues = new Set(state.audioFiles.map((f) => listings[f.path]?.tracks[0]?.language?.toLowerCase() ?? ""));
  const [tongue] = [...tongues];
  let language: string | null = null;
  if (tongues.size === 1 && tongue && !/^(und|unknown)$/.test(tongue)) {
    try {
      language = languages?.of(tongue) ?? null;
    } catch {
      language = null;
    }
  }
  const inside = videoFolder && audioFolder.startsWith(videoFolder) && audioFolder.length > videoFolder.length;
  const where = inside ? audioFolder.slice(videoFolder.length) : audioFolder;
  const lead = videoFolder && videoFolder !== audioFolder ? `${videoFolder}  ·  ` : "";
  return `${lead}${language ? `${language} dubs` : lead ? "dubs" : "Dubs"} in ${where}`;
}

function folderTitle(folder: string | null): string {
  if (!folder) return "Episodes";
  return baseName(folder.replace(/[\\/]+$/, ""));
}

/** Series before a run: each episode, its dub, and how they were paired.
 *  An episode the matcher could not pair gets a dub picker in its row. */
function PairingList({ state, pairing, manualCount, onRepair, onResetRepairs, pairingLoading, listings }: WorkspaceProps) {
  const pairs = pairing?.pairs ?? [];
  const byVideo = new Map(pairs.map((p) => [p.primaryPath, p]));
  const used = new Set(pairs.map((p) => p.secondaryPath));
  const unused = state.audioFiles.filter((f) => !used.has(f.path));
  const rows = state.videoFiles.map((f) => ({ file: f, pair: byVideo.get(f.path) }));
  // The row's own dub, then the ones no episode uses (as the design's
  // picker lists them), then the ones paired elsewhere.
  const dubOptions = (current: string | null) => [
    ...(current ? state.audioFiles.filter((f) => f.path === current).map((f) => ({ value: f.path, label: f.name })) : []),
    ...unused.map((f) => ({ value: f.path, label: f.name, group: "Dubs not used" })),
    ...state.audioFiles.filter((f) => used.has(f.path) && f.path !== current).map((f) => ({ value: f.path, label: f.name, group: "Paired with another episode" })),
  ];

  return (
    <Box
      body={false}
      title={seriesTitle(state)}
      sub={seriesSub(state, listings)}
      end={
        manualCount > 0 ? (
          <Cmd sm icon={<DismissRegular />} onClick={onResetRepairs}>Reset {manualCount} edit{manualCount === 1 ? "" : "s"}</Cmd>
        ) : undefined
      }
    >
      <Table cols="64px minmax(0,1fr) minmax(0,1fr) 150px" head={["Episode", "Video", "Dub", "Paired by"]} label="Pairing">
        {rows.map(({ file, pair }) => {
          const by = pair ? pairedBy(pair) : null;
          const options = dubOptions(pair?.secondaryPath ?? null);
          return (
            <Tr key={file.path} label={file.name}>
              <span className="num t2">{episodeCode(pair?.key, file.name)}</span>
              <Name tail={20}>{file.name}</Name>
              {pair ? (
                <span className="cell pairdub">
                  <span className="fi" aria-hidden><MusicNote2Regular /></span>
                  <Combo
                    sm
                    ghost
                    w="100%"
                    label={`Dub for ${file.name}`}
                    value={pair.secondaryPath}
                    options={[...options, { value: "__skip__", label: "Skip this episode" }]}
                    onChange={(value) => onRepair(file.path, value === "__skip__" ? null : value)}
                  />
                </span>
              ) : (
                <Combo
                  sm
                  placeholder="Choose a dub"
                  w={200}
                  label={`Dub for ${file.name}`}
                  value={null}
                  options={options}
                  onChange={(value) => onRepair(file.path, value)}
                />
              )}
              {by ? (
                by.warn ? <Status s="warn" text={by.text} /> : <span className="t2 truncate">{by.text}</span>
              ) : (
                <Status s="bad" text="Not paired" />
              )}
            </Tr>
          );
        })}
        {pairingLoading && rows.length === 0 && (
          <div className="log t3"><Ring size={14} /> Working out the pairs…</div>
        )}
      </Table>
    </Box>
  );
}

/** Find match: videos down, dubs across; each dub's best release lit. */
/** The distinct files one side of the results names, in order. */
const sides = (results: SyncResult[], side: "video" | "dub"): Side[] => [
  ...new Map(
    results.map((r) => {
      const path = side === "video" ? (r.primaryPath ?? r.videoFile) : (r.secondaryPath ?? r.audioFile);
      return [path, { path, name: side === "video" ? r.videoFile : r.audioFile }] as const;
    }),
  ).values(),
];
type Side = Pick<FileItem, "path" | "name">;

function MatchGrid({ state, focus, onFocus, previewingKey, onPreview, onCopyCommand, onOpenDubSync }: WorkspaceProps) {
  // A run reopened from History brings its results, not the files: the grid
  // is then built from what the results name.
  const videos: Side[] = state.videoFiles.length ? state.videoFiles : sides(state.results, "video");
  const dubs: Side[] = state.audioFiles.length ? state.audioFiles : sides(state.results, "dub");
  const running = state.status === "processing";
  const byPair = useMemo(
    () => new Map(state.results.map((r) => [`${r.primaryPath ?? r.videoFile}::${r.secondaryPath ?? r.audioFile}`, r])),
    [state.results],
  );
  const get = (v: Side, d: Side) => byPair.get(`${v.path}::${d.path}`) ?? byPair.get(`${v.name}::${d.name}`);
  // Each dub's best release is lit once the tests are done.
  const best = new Map<string, string>();
  for (const d of running ? [] : dubs) {
    let top: SyncResult | undefined;
    for (const v of videos) {
      const r = get(v, d);
      if (r && !r.error && (r.confidence ?? 0) >= 0.75 && (!top || (r.confidence ?? 0) > (top.confidence ?? 0))) top = r;
    }
    if (top) best.set(d.path, resultKey(top));
  }
  const focusedResult = running
    ? undefined
    : (state.results.find((r) => resultKey(r) === focus) ?? (best.size ? state.results.find((r) => resultKey(r) === [...best.values()][0]) : undefined));
  const title = commonTitle(videos.map((v) => v.name));
  const dubNames = dubs.map((d) => d.name);
  const doneCount = state.results.length;

  return (
    <>
      <Box body={false} title={title} sub="videos down, dubs across">
        <div className="tbl" role="grid" aria-label="Matches" style={{ ["--cols" as string]: `minmax(0,1.2fr) repeat(${Math.max(1, dubs.length)}, minmax(0,1fr))` }}>
          <div className="th" style={{ height: 40 }} role="row">
            <span />
            {dubs.map((d) => (
              <span key={d.path} className="cell" style={{ color: "var(--text)" }} role="columnheader">
                <span className="fi" aria-hidden><MusicNote2Regular /></span>
                <MidText text={dubLabel(d.name, dubNames)} tail={14} />
              </span>
            ))}
          </div>
          <div className="tb" role="rowgroup">
            {videos.map((v) => (
              <div key={v.path} className="tr" style={{ height: 64 }} role="row">
                <span className="col" style={{ minWidth: 0 }}>
                  <span className="cell"><span className="fi" aria-hidden><VideoClipRegular /></span><span className="truncate">{releaseName(v.name, title)}</span></span>
                  <span className="t3 sm" style={{ paddingLeft: 24 }}><MidText text={v.name} tail={16} /></span>
                </span>
                {dubs.map((d) => {
                  const r = get(v, d);
                  const key = r ? resultKey(r) : `${v.path}::${d.path}`;
                  const isBest = r ? best.get(d.path) === key : false;
                  const c = r && r.confidence !== null && !r.error ? Math.round(r.confidence * 100) : null;
                  const on = !!r && focusedResult === r;
                  const testing = running && !r && state.currentFile && (state.currentFile === v.path || baseName(state.currentFile) === v.name);
                  return (
                    <button
                      key={d.path}
                      type="button"
                      role="gridcell"
                      className="col matchcell"
                      disabled={!r}
                      aria-label={`${v.name} with ${d.name}: ${c !== null ? `${c}%` : r ? "failed" : "not tested"}`}
                      onClick={() => r && onFocus(key)}
                      style={{
                        justifyContent: "center", height: 52, padding: "0 10px", borderRadius: 4, textAlign: "left",
                        background: isBest ? "var(--ac-wash)" : undefined,
                        boxShadow: on ? "inset 0 0 0 1.5px var(--ac)" : undefined,
                      }}
                    >
                      {r ? (
                        <>
                          <span className={cx("cell num", isBest ? "strong" : (c ?? 0) < 50 && "t3")} style={{ fontSize: 15 }}>
                            {isBest && <CheckmarkCircleFilled className="acc" style={{ fontSize: 16 }} aria-hidden />}
                            {c !== null ? `${c}%` : "—"}
                          </span>
                          <span className="t3 sm num">
                            {r.error ? shortError(r.error) : (c ?? 0) >= 50 ? delayText(r.delayMs) : "No match"}
                          </span>
                        </>
                      ) : testing ? (
                        <span className="cell t3"><Ring size={14} /> Testing</span>
                      ) : (
                        <Dash />
                      )}
                    </button>
                  );
                })}
              </div>
            ))}
            {doneCount > 0 && !running && (
              <div className="tr" style={{ height: 48, borderTop: "1px solid var(--divider)", borderRadius: 0, marginTop: 6 }} role="row">
                <span className="strong">Timed for</span>
                {dubs.map((d) => {
                  const r = state.results.find((x) => resultKey(x) === best.get(d.path));
                  return (
                    <span key={d.path} className={cx("strong truncate", !r && "t3")} style={{ paddingLeft: 10, fontWeight: r ? 600 : 400 }}>
                      {r ? releaseName(r.videoFile, title) : "No match"}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </Box>
      {!withDetails("compare", state) ? null : focusedResult ? (
        <MatchDetails
          r={focusedResult}
          title={`${dubLabel(focusedResult.audioFile, dubNames)} × ${releaseName(focusedResult.videoFile, title)}`}
          rank={rankOf(focusedResult, state.results)}
          previewing={previewingKey === resultKey(focusedResult)}
          onPreview={() => onPreview(focusedResult)}
          onCopyCommand={() => onCopyCommand(focusedResult)}
          onOpenDubSync={onOpenDubSync}
        />
      ) : (
        <section className="box" />
      )}
    </>
  );
}

function rankOf(r: SyncResult, results: SyncResult[]) {
  const same = results.filter((x) => (x.secondaryPath ?? x.audioFile) === (r.secondaryPath ?? r.audioFile));
  const sorted = [...same].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  return { place: sorted.indexOf(r) + 1, of: same.length };
}

function MatchDetails({
  r,
  title,
  rank,
  previewing,
  onPreview,
  onCopyCommand,
}: {
  r: SyncResult;
  title: string;
  rank: { place: number; of: number };
  previewing: boolean;
  onPreview: () => void;
  onCopyCommand: () => void;
  onOpenDubSync: () => void;
}) {
  const c = r.confidence === null ? null : Math.round(r.confidence * 100);
  return (
    <Box title={title}>
      <div>
        <div className="t3">Confidence</div>
        <div className={cx("big", r.error && "bad")}>{r.error ? "Failed" : c !== null ? `${c}%` : "—"}</div>
        <div className="t2">{r.error ?? (rank.place === 1 ? `Best of ${rank.of} releases for this dub` : `${ordinal(rank.place)} of ${rank.of} releases for this dub`)}</div>
      </div>
      <DL
        rows={[
          ["Delay", delayText(r.delayMs)],
          ["Drift", r.hasSignificantDrift && r.driftMsPerS !== null ? `${r.driftMsPerS.toFixed(3)} ms/s` : r.isLikelyCut ? "Different cut" : "None"],
          ["Windows", r.windowsUsed !== null ? `${r.windowsUsed} of ${r.windowsTotal ?? "?"}` : "—"],
          ["Video", <MidText key="v" text={r.videoFile} tail={14} />],
          ["Dub", <MidText key="d" text={r.audioFile} tail={14} />],
        ]}
      />
      {!r.error && r.delayMs !== null && (
        <Links>
          <button type="button" className="cmd" onClick={onPreview} disabled={previewing}>
            <span className="ic" aria-hidden>{previewing ? <Ring size={16} /> : <PlayRegular />}</span>Preview
          </button>
          <button type="button" className="cmd" onClick={onCopyCommand}>
            <span className="ic" aria-hidden><CopyRegular /></span>Copy command
          </button>
        </Links>
      )}
    </Box>
  );
}

const ordinal = (n: number) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : Math.min(n % 10, 4) % 4] ?? "th"}`;

/** A file name without the extension, for tight headers. */
const shortName = (name: string) => name.replace(/\.[^.]{2,4}$/, "");

/** Words in a release name that describe the format, not the release. */
const NOISE = /^(1080p|2160p|720p|480p|x264|x265|h264|h265|hevc|avc|remux|uhd|10bit|8bit|ddp5\.1|dd5\.1|aac|ac3|dts|truehd|atmos|hdr|hdr10|dv)$/i;

/** The part of the names every video shares ("Parasite (2019)"), for the title. */
export function commonTitle(names: string[]): string {
  if (names.length === 0) return "Releases";
  const words = names.map((n) => shortName(n).split(/[\s._]+/));
  const common: string[] = [];
  for (let i = 0; i < words[0].length; i++) {
    const w = words[0][i];
    if (words.every((ws) => ws[i]?.toLowerCase() === w.toLowerCase())) common.push(w);
    else break;
  }
  // Shared format words ("1080p") are not part of the title.
  while (common.length > 1 && NOISE.test(common[common.length - 1])) common.pop();
  // A trailing year reads as one ("Parasite 2019" → "Parasite (2019)"); a
  // number inside the title stays as it is ("Blade Runner 2049 (2017)").
  const joined = common.join(" ");
  const text = /\(\d{4}\)/.test(joined) ? joined : joined.replace(/\b((?:19|20)\d{2})$/, "($1)");
  return text.length >= 3 ? text : "Releases";
}

/** What tells one dub from the others ("Hindi AC3", "Telugu E-AC3"): its
 *  name without the words every dub's name starts with. */
export function dubLabel(name: string, all: string[]): string {
  const split = (n: string) => shortName(n).split(/[\s._]+/);
  const words = split(name);
  const others = all.map(split);
  let common = 0;
  while (all.length > 1 && common < words.length - 1 && others.every((ws) => ws[common]?.toLowerCase() === words[common].toLowerCase())) common += 1;
  return words.slice(common).join(" ") || shortName(name);
}

/** What distinguishes one release from the others ("BluRay", "WEB-DL"). */
export function releaseName(name: string, title: string): string {
  // "DDP5.1", "AAC2.0": a sound format, not a release, and its dot is no word break.
  const words = shortName(name).replace(/[\s._-]?\b(?:DDP?|DD\+|E-?AC-?3|AC-?3|AAC|DTS(?:-HD)?|TrueHD|FLAC|Opus|MA)\s?\d\.\d\b/gi, "").split(/[\s._]+/).filter(Boolean);
  const titleWords = title.replace(/[()]/g, "").split(/\s+/).map((w) => w.toLowerCase());
  const rest = words.filter((w, i) => i >= titleWords.length || w.toLowerCase().replace(/[()]/g, "") !== titleWords[i]);
  const tags = rest.filter((w) => !NOISE.test(w));
  return (tags.length ? tags : rest).slice(0, 3).join(" ") || shortName(name);
}
