/** The details pane: whatever the selected pair needs next.
 *
 *  Before a run, how the track is written. While it runs, the job's
 *  stages. After it, the error the check measured, the plan's numbers, and
 *  below them everything the engine said -- the check in full, warnings,
 *  notes, voices moved, every piece of the plan. A failed pair says why,
 *  with a retry. With a stretch selected in the Timeline, the stretch and
 *  its offset. */

import {
  ArrowResetRegular,
  CheckmarkCircleFilled,
  ChevronDoubleLeftRegular,
  ChevronDoubleRightRegular,
  ChevronLeftRegular,
  ChevronRightRegular,
  CircleRegular,
  CopyRegular,
  CutRegular,
  ErrorCircleFilled,
  FolderOpenRegular,
  InfoRegular,
  MusicNote2Regular,
  PlayRegular,
  VideoClipRegular,
  WarningFilled,
} from "@fluentui/react-icons";
import { useEffect, useState, type KeyboardEvent, type ReactNode } from "react";

import type { DubQueueJob } from "@/lib/dubQueueReducer";
import { HAND_NOTE } from "@/lib/dubPlanEdit";
import {
  AUDIBLE_MS,
  DUB_CODECS,
  DUB_RATES,
  formatClock,
  formatMs,
  formatSpan,
  isUnmatchedFill,
  type AppSettings,
  type DubSegment,
  type DubSyncPlan,
  type DubVerification,
} from "@/lib/types";
import { Box } from "@/ui/frame";
import { Btn, Cmd, Combo, DL, InfoBar, Links, Ring, TBox, Toggle, baseName, cx, type Option } from "@/ui/kit";

import { createCursorStore, useCursor, type CursorStore } from "./cursorStore";
import { channelName, formatLength, formatOffset, parseOffset, planFacts, sentence, stageIndex, stagesFor, stretchName } from "./dubModel";

/* ------------------------------------------------------------------ output */

export interface TrackChoice {
  label: string;
  options: Option<number>[];
  value: number;
  onChange: (index: number) => void;
}

/** How the track is written: the options that belong beside the files. */
export function OutputBox({
  settings,
  onChange,
  disabled,
  sameAs,
  tracks = [],
}: {
  settings: Pick<AppSettings, "dubCodec" | "dubRate" | "dubMux" | "dubLanguage">;
  onChange: (patch: Partial<AppSettings>) => void;
  disabled?: boolean;
  /** The dub's own format, for "Same as the dub (AC3 5.1)". */
  sameAs?: string | null;
  /** Audio streams to choose from, for files that carry several. */
  tracks?: TrackChoice[];
}) {
  const codecs = DUB_CODECS.map((c) => ({ value: c.id, label: c.id === "same" && sameAs ? `${c.label} (${sameAs})` : c.label }));
  const rates: Option<string>[] = [{ value: "auto", label: "Detect from the audio" }, ...DUB_RATES.map((r) => ({ value: String(r.value), label: r.label }))];
  return (
    <Box title="Output">
      <div className="col" style={{ gap: 6 }}>
        <span className="t3">Write the track as</span>
        <Combo value={settings.dubCodec} options={codecs} w="100%" label="Write the track as" disabled={disabled} onChange={(dubCodec) => onChange({ dubCodec })} />
      </div>
      <div className="col" style={{ gap: 6 }}>
        <span className="t3">The dub was mastered at</span>
        <Combo
          value={settings.dubRate === null ? "auto" : String(settings.dubRate)}
          options={rates}
          w="100%"
          label="The dub was mastered at"
          disabled={disabled}
          onChange={(value) => onChange({ dubRate: value === "auto" ? null : Number(value) })}
        />
      </div>
      <div className="row">
        <span className="grow">Add to a copy of the video</span>
        <Toggle on={settings.dubMux} name="Add to a copy of the video" disabled={disabled} onChange={(dubMux) => onChange({ dubMux })} />
      </div>
      <div className="row">
        <span className="grow t3">Language</span>
        <TBox
          value={settings.dubLanguage}
          w={80}
          maxLength={3}
          spellCheck={false}
          placeholder="none"
          label="Language of the added track"
          title="Three-letter language code for the added track, e.g. hin"
          disabled={disabled || !settings.dubMux}
          onChange={(value) => onChange({ dubLanguage: value.toLowerCase().replace(/[^a-z]/g, "") })}
        />
      </div>
      {tracks.length > 0 && <div className="hr" />}
      {tracks.map((track) => (
        <div key={track.label} className="col" style={{ gap: 6 }}>
          <span className="t3">{track.label}</span>
          <Combo<number> value={track.value} options={track.options} w="100%" label={track.label} disabled={disabled} onChange={track.onChange} />
        </div>
      ))}
    </Box>
  );
}

/* ------------------------------------------------------------------ stages */

/** A running job's checklist: done, doing (with its percent), to do. */
export function StagesBox({ title, job, mux }: { title: string; job: DubQueueJob; mux: boolean }) {
  const stages = stagesFor(mux);
  const cur = job.status === "queued" ? -1 : Math.max(0, stageIndex(job.stage, stages));
  return (
    <Box title={title}>
      <div className="col" aria-label="Stages">
        {stages.map((stage, i) => (
          <div key={stage.label} className="row" style={{ gap: 10, height: 29, color: i > cur ? "var(--text-3)" : undefined }}>
            <span style={{ width: 16, display: "grid", fontSize: 16 }} aria-hidden>
              {i < cur ? <CheckmarkCircleFilled className="ok" /> : i === cur ? <Ring size={16} /> : <CircleRegular style={{ color: "var(--text-4)" }} />}
            </span>
            <span className="grow truncate">{stage.label}</span>
            {i === cur && <span className="t3 num">{Math.round(job.percent)}%</span>}
          </div>
        ))}
      </div>
    </Box>
  );
}

/* -------------------------------------------------------------------- done */

/** A synced pair: how it measured, what the plan did, and everything else
 *  the engine said, below. */
export function DoneBox({
  title,
  job,
  pieces,
  selected = null,
  problems = [],
  onPlay,
  onShow,
  onReport,
  onPiece,
  onBackToEngine,
}: {
  title: string;
  job: DubQueueJob;
  /** The plan as it stands, edits included, for the list of pieces. */
  pieces?: DubSyncPlan | null;
  selected?: number | null;
  problems?: string[];
  onPlay?: () => void;
  onShow?: () => void;
  onReport?: () => void;
  onPiece?: (index: number) => void;
  onBackToEngine?: () => void;
}) {
  const plan = job.plan;
  const v = job.verification;
  const facts = plan ? planFacts(plan) : null;
  return (
    <Box title={title} label="Details">
      {problems.length > 0 && <Problems problems={problems} />}
      <div>
        <div className="t3">Sync error</div>
        {v && v.typicalMs !== null ? (
          <div className="big">
            {formatMs(v.typicalMs)}
            <small>ms</small>
          </div>
        ) : (
          <div className="big">—</div>
        )}
        <div className="t2">{v ? (v.worstMs !== null ? `Typical · ${formatMs(v.worstMs)} ms at worst` : "Typical") : "Not checked"}</div>
      </div>
      {facts && (
        <DL
          rows={[
            ["Stretches", facts.stretches],
            ["From original", facts.fromOriginal],
            ["Fill level", facts.fillLevel],
            ["Frame rate", facts.frameRate],
            ["Voices moved", facts.voices],
          ]}
        />
      )}
      <Links>
        <Cmd icon={<PlayRegular />} disabled={!onPlay} onClick={onPlay}>
          Play
        </Cmd>
        <Cmd icon={<FolderOpenRegular />} disabled={!onShow || !job.output} onClick={onShow}>
          Show
        </Cmd>
        <Cmd icon={<CopyRegular />} disabled={!onReport} onClick={onReport} title="The plan, what to check and how the track measured, as text">
          Report
        </Cmd>
        {onBackToEngine && (
          <Cmd icon={<ArrowResetRegular />} onClick={onBackToEngine}>
            Back to the engine&apos;s plan
          </Cmd>
        )}
      </Links>
      <div className="col dub-notes">
        {job.output && <OutputLine job={job} />}
        {v && <Verification verification={v} />}
        <Remarks warnings={[...(plan?.warnings ?? []), ...(job.output?.warnings ?? [])]} notes={plan?.notes ?? []} />
        {(plan?.voicePieces?.length ?? 0) > 0 && (
          <section className="col" style={{ gap: 4 }} aria-label="Voices moved">
            <span className="h">Voices moved ({plan!.voicePieces!.length})</span>
            {plan!.voicePieces!.map((piece, i) => (
              <span key={i}>{piece.note}</span>
            ))}
          </section>
        )}
        {(pieces ?? plan) && (pieces ?? plan)!.segments.length > 0 && <Pieces plan={(pieces ?? plan)!} selected={selected} onPiece={onPiece} />}
      </div>
    </Box>
  );
}

function OutputLine({ job }: { job: DubQueueJob }) {
  const output = job.output!;
  const parts = [channelName(output.channels), `${output.sampleRate / 1000} kHz`, formatLength(output.seconds)].filter(Boolean);
  return (
    <section className="col" style={{ gap: 2 }} aria-label="Written">
      <span className="h">Written</span>
      <span className="mono dub-path" title={output.outputPath}>
        {baseName(output.outputPath)}
      </span>
      <span>{parts.join(" · ")}</span>
      {job.muxedPath && (
        <span title={job.muxedPath}>
          Also added to <span className="mono">{baseName(job.muxedPath)}</span>
        </span>
      )}
    </section>
  );
}

/** The written file measured against the video, in full. Spots are long
 *  precise windows; the sweep is short windows every few seconds that
 *  catch a stretch the spots fell between. Fills are skipped in both,
 *  since they are the original itself. */
function Verification({ verification: v }: { verification: DubVerification }) {
  const measured = v.spots.filter((s) => s.residualMs !== null);
  const share = v.sweepMeasured > 0 ? Math.round((100 * v.sweepWithinAudible) / v.sweepMeasured) : null;
  const lines = v.lines;
  const offLines = lines ? lines.windows.filter((w) => w.lagMs !== null && Math.abs(w.lagMs) > lines.toleranceMs) : [];
  return (
    <section className="col" style={{ gap: 4 }} aria-label="Verification">
      <span className="h">Checked against the video</span>
      <span>
        {measured.length > 0
          ? `Measured at ${measured.length} spot${measured.length === 1 ? "" : "s"} along the runtime. Lip-sync starts to show around ${AUDIBLE_MS} ms; the measurement resolves a fraction of a millisecond.`
          : "None of the spots could be measured: the shared music and effects were too quiet, or every spot fell on a fill."}
        {v.sweepWindows > 0 && share !== null && (
          <>
            {" "}
            A sweep of {v.sweepWindows} short windows measured {v.sweepMeasured}, and {share}% of the runtime sits within {AUDIBLE_MS} ms
            {v.sweepWithin1Ms !== undefined && v.sweepMeasured > 0 && ` (${Math.round((100 * v.sweepWithin1Ms) / v.sweepMeasured)}% within 1 ms)`}
            {v.sweepWorstMs !== null && `; worst ${formatMs(v.sweepWorstMs)} ms`}.
          </>
        )}
      </span>
      {lines && lines.judged > 0 && (
        <span>
          Lines: across {lines.judged} of {lines.windows.length} minute-long dialogue windows the dub&apos;s speech sits{" "}
          <span className="num">
            {(lines.overallMs ?? 0) >= 0 ? "+" : ""}
            {(lines.overallMs ?? 0).toFixed(0)} ms
          </span>{" "}
          from the original&apos;s, which is in sync with the lips; {lines.withinTolerance} of {lines.judged} within {lines.toleranceMs} ms.
        </span>
      )}
      {offLines.map((w, i) => (
        <Remark key={`l${i}`} tone="warn">
          <span className="num">
            {formatClock(w.startS)} – {formatClock(w.endS)}
          </span>{" "}
          the lines sit {w.lagMs! >= 0 ? "+" : ""}
          {w.lagMs!.toFixed(0)} ms from the original&apos;s — check the lips
        </Remark>
      ))}
      {v.stretches.map((stretch, i) => (
        <Remark key={`s${i}`} tone="bad">
          <span className="num">
            {formatClock(stretch.startS)} – {formatClock(stretch.endS)}
          </span>{" "}
          out by{" "}
          <span className="num">
            {stretch.residualMs >= 0 ? "+" : ""}
            {formatMs(stretch.residualMs)} ms
          </span>{" "}
          over {stretch.windows} windows — would be audible
        </Remark>
      ))}
      {measured.length > 0 && (
        <span className="dub-spots">
          {v.spots.map((spot, i) => (
            <span key={i} className="num" title={spot.note || undefined}>
              {formatClock(spot.positionS).replace(/\.\d{3}$/, "")}{" "}
              {spot.residualMs === null ? (
                <span className="dub-dim">{spot.note.startsWith("filled") ? "fill" : "n/a"}</span>
              ) : (
                <span className={cx(Math.abs(spot.residualMs) > AUDIBLE_MS && "bad")}>
                  {spot.residualMs >= 0 ? "+" : ""}
                  {formatMs(spot.residualMs)}
                </span>
              )}
            </span>
          ))}
        </span>
      )}
    </section>
  );
}

function Remark({ tone, children }: { tone: "warn" | "bad" | "info"; children: ReactNode }) {
  return (
    <span className="dub-remark" data-tone={tone}>
      <span className="ic" aria-hidden>
        {tone === "warn" ? <WarningFilled className="warn" /> : tone === "bad" ? <ErrorCircleFilled className="bad" /> : <InfoRegular />}
      </span>
      <span className="grow">{children}</span>
    </span>
  );
}

/** Warnings are things to check; notes are worth knowing and need nothing. */
function Remarks({ warnings, notes }: { warnings: string[]; notes: string[] }) {
  if (warnings.length + notes.length === 0) return null;
  return (
    <section className="col" style={{ gap: 4 }} aria-label="Remarks">
      {warnings.map((w, i) => (
        <Remark key={`w${i}`} tone="warn">
          {w}
        </Remark>
      ))}
      {notes.map((n, i) => (
        <Remark key={`n${i}`} tone="info">
          {n}
        </Remark>
      ))}
    </section>
  );
}

/** Every piece of the plan on the video's timeline; one picks the stretch. */
function Pieces({ plan, selected, onPiece }: { plan: DubSyncPlan; selected: number | null; onPiece?: (index: number) => void }) {
  return (
    <section className="col" style={{ gap: 2 }} aria-label="Pieces of the plan">
      <span className="h">Pieces</span>
      {plan.segments.map((s, i) => {
        const replaced = isUnmatchedFill(s);
        return (
          <button
            key={i}
            type="button"
            className={cx("dub-piece", i === selected && "on")}
            disabled={!onPiece}
            onClick={() => onPiece?.(i)}
            title={replaced ? "The dub was audible here but could not be matched, and it was replaced. Listen to this spot: the dub may have had the right scene." : s.note || undefined}
          >
            <span className="k">{replaced ? "Replaced" : s.kind === "fill" ? "Original" : "Dub"}</span>
            <span className="num">
              {formatClock(s.startS)} – {formatClock(s.endS)}
            </span>
            <span className="num truncate">
              {s.kind === "dub"
                ? `${(s.offsetS ?? 0) >= 0 ? "+" : ""}${(s.offsetS ?? 0).toFixed(3)}s`
                : replaced
                  ? "did not correlate — replaced; check this spot"
                  : s.note}
              {s.uncertaintyS >= 0.05 && ` ±${s.uncertaintyS.toFixed(1)}s`}
            </span>
          </button>
        );
      })}
    </section>
  );
}

function Problems({ problems }: { problems: string[] }) {
  return (
    <InfoBar tone="bad" style={{ margin: 0 }}>
      {problems.length === 1 ? problems[0] : `${problems[0]} (${problems.length - 1} more)`}
    </InfoBar>
  );
}

/* ------------------------------------------------------------------ failed */

/** A pair that did not sync: why, and what to do about it. */
export function FailedBox({
  title,
  reason,
  stopped = false,
  onRetry,
  onChooseDub,
}: {
  title: string;
  reason: string;
  stopped?: boolean;
  onRetry?: () => void;
  onChooseDub?: () => void;
}) {
  return (
    <Box title={title} label="Details">
      <div>
        <div className="t3">Result</div>
        <div className={cx("big", !stopped && "bad")}>{stopped ? "Stopped" : "Not synced"}</div>
        <div className="t2">{sentence(reason)}</div>
      </div>
      <div className="row" style={{ gap: 8 }}>
        <Btn disabled={!onRetry} onClick={onRetry}>
          Retry
        </Btn>
        <Btn disabled={!onChooseDub} onClick={onChooseDub}>
          Choose another dub
        </Btn>
      </div>
    </Box>
  );
}

/* ----------------------------------------------------------------- stretch */

const NO_CURSOR = createCursorStore();

/** One stretch of the plan, selected in the Timeline: where it is, where
 *  it reads from, and its offset to nudge or type. */
export function StretchBox({
  plan,
  index,
  frameS,
  problems = [],
  cursor,
  canSplit,
  onNudge,
  onSetOffset,
  onSplit,
  onUseOriginal,
  onUseDub,
  onBackToEngine,
  onKeyDown,
}: {
  plan: DubSyncPlan;
  index: number;
  frameS: number;
  problems?: string[];
  /** The playhead: the stretch can be split where it is, inside it. */
  cursor?: CursorStore;
  /** Overrides what the playhead says. */
  canSplit?: boolean;
  onNudge: (deltaS: number) => void;
  onSetOffset: (offsetS: number) => void;
  onSplit: () => void;
  onUseOriginal: () => void;
  onUseDub: () => void;
  onBackToEngine?: () => void;
  onKeyDown?: (event: KeyboardEvent<HTMLDivElement>) => void;
}) {
  const segment: DubSegment = plan.segments[index];
  const dub = segment.kind === "dub";
  const playhead = useCursor(cursor ?? NO_CURSOR);
  const splittable = canSplit ?? (playhead !== null && playhead > segment.startS && playhead < segment.endS);
  const match = segment.match !== null ? segment.match.toFixed(2) : segment.note === HAND_NOTE ? "Set by hand" : "—";
  return (
    <div className="dub-insp" onKeyDown={onKeyDown}>
      <Box title={stretchName(plan, index)} label="Stretch">
        {problems.length > 0 && <Problems problems={problems} />}
        <DL
          rows={[
            ["Start", formatClock(segment.startS)],
            ["End", formatClock(segment.endS)],
            ["Length", formatSpan(segment.endS - segment.startS)],
            ["Taken from", `${dub ? "Dub" : "Original"} ${formatClock(segment.sourceStartS)}`],
            dub ? ["Match", match] : ["Why", segment.note || "—"],
          ] satisfies [string, string][]}
        />
        {dub && <OffsetBox offsetS={segment.offsetS ?? 0} frameS={frameS} onNudge={onNudge} onSetOffset={onSetOffset} />}
        <Links>
          <Cmd icon={<CutRegular />} disabled={!splittable} onClick={onSplit} title="Split at the playhead (S)">
            Split here
          </Cmd>
          {dub ? (
            <Cmd icon={<VideoClipRegular />} onClick={onUseOriginal}>
              Use the original
            </Cmd>
          ) : (
            <Cmd icon={<MusicNote2Regular />} onClick={onUseDub}>
              Use the dub
            </Cmd>
          )}
          {onBackToEngine && (
            <Cmd icon={<ArrowResetRegular />} onClick={onBackToEngine}>
              Back to the engine&apos;s plan
            </Cmd>
          )}
        </Links>
      </Box>
    </div>
  );
}

function OffsetBox({
  offsetS,
  frameS,
  onNudge,
  onSetOffset,
}: {
  offsetS: number;
  frameS: number;
  onNudge: (deltaS: number) => void;
  onSetOffset: (offsetS: number) => void;
}) {
  const [text, setText] = useState(formatOffset(offsetS));
  useEffect(() => setText(formatOffset(offsetS)), [offsetS]);
  const commit = () => {
    const value = parseOffset(text);
    if (value === null) setText(formatOffset(offsetS));
    else if (Math.abs(value - offsetS) >= 0.0005) onSetOffset(value);
  };
  // Shift makes any nudge a millisecond.
  const by = (event: { shiftKey: boolean }, step: number) => onNudge(event.shiftKey ? Math.sign(step) * 0.001 : step);
  return (
    <div className="col" style={{ gap: 6 }}>
      <span className="t3">Offset</span>
      <div className="row" style={{ gap: 2 }}>
        <Cmd icon={<ChevronDoubleLeftRegular />} title="−1 frame (Shift: −1 ms)" onClick={(event) => by(event, -frameS)} />
        <Cmd icon={<ChevronLeftRegular />} title="−10 ms (Shift: −1 ms)" onClick={(event) => by(event, -0.01)} />
        <TBox
          value={text}
          unit="s"
          w={116}
          inputMode="decimal"
          label="Offset in seconds"
          onChange={setText}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              event.stopPropagation();
              commit();
            } else if (event.key === "Escape") {
              event.stopPropagation();
              setText(formatOffset(offsetS));
              event.currentTarget.blur();
            }
          }}
        />
        <Cmd icon={<ChevronRightRegular />} title="+10 ms (Shift: +1 ms)" onClick={(event) => by(event, 0.01)} />
        <Cmd icon={<ChevronDoubleRightRegular />} title="+1 frame (Shift: +1 ms)" onClick={(event) => by(event, frameS)} />
      </div>
    </div>
  );
}
