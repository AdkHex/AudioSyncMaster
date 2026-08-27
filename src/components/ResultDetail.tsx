import { Info } from "lucide-react";
import type { ReactNode } from "react";

import {
  formatDelay,
  formatDrift,
  formatDuration,
  frameOffset,
  type SyncResult,
} from "@/lib/types";

interface ResultDetailProps {
  result: SyncResult;
  /** Row actions. They live here rather than in the row so nothing appears
   *  and disappears under the pointer while scanning the list. */
  actions?: ReactNode;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <dt className="shrink-0 text-[11px] text-muted-foreground">{label}</dt>
      <dd className="tabular text-right font-mono text-[11.5px] text-foreground">{children}</dd>
    </div>
  );
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h4 className="mb-1 text-[10px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">
        {title}
      </h4>
      <dl>{children}</dl>
    </div>
  );
}

/** Everything the engine measured for one pair.
 *
 *  All of this was already crossing the wire and being discarded. The
 *  per-window figures are what let a user judge a borderline result: a delay
 *  that is identical at the start and the end of a file is trustworthy in a way
 *  that a single number can never convey on its own.
 */
export function ResultDetail({ result, actions }: ResultDetailProps) {
  const frames = frameOffset(result.delayMs, result.primaryFps);

  return (
    <div className="border-t border-border bg-primary/[0.055] px-[18px] pb-4 pt-1">
      <div className="grid gap-x-9 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
        <Group title="Measurement">
          <Row label="Delay">
            {formatDelay(result.delayMs)}
            {frames !== null && (
              <span className="ml-1.5 font-sans text-[10.5px] text-muted-foreground">
                &asymp; {frames} frame{Math.abs(frames) === 1 ? "" : "s"}
              </span>
            )}
          </Row>
          <Row label="At file start">{formatDelay(result.startDelayMs)}</Row>
          <Row label="At file end">{formatDelay(result.endDelayMs)}</Row>
          {result.delayAtStartMs !== null && result.delayAtStartMs !== result.delayMs && (
            <Row label="Applied from t=0">{formatDelay(result.delayAtStartMs)}</Row>
          )}
          <Row label="Windows used">
            {result.windowsUsed ?? "--"} of {result.windowsTotal ?? "--"}
          </Row>
          <Row label="Confidence">
            {result.confidence === null ? "--" : `${(result.confidence * 100).toFixed(0)}%`}
          </Row>
        </Group>

        <Group title="Drift">
          {/* Two "--" rows under a heading take a third of the panel to say
              nothing. One line says the same thing. */}
          {result.driftMsPerS === null && result.totalDriftMs === null ? (
            <p className="py-1 text-[11.5px] text-muted-foreground">
              The offset is constant across the file.
            </p>
          ) : (
            <>
              <Row label="Rate">{formatDrift(result.driftMsPerS)}</Row>
              <Row label="Across the file">{formatDelay(result.totalDriftMs)}</Row>
            </>
          )}
          {result.rateDiagnosis?.sourceFps && result.rateDiagnosis?.targetFps && (
            <Row label="Conversion">
              {result.rateDiagnosis.sourceFps} &rarr; {result.rateDiagnosis.targetFps} fps
            </Row>
          )}
          {result.rateDiagnosis?.correctionRatio && (
            <Row label="Speed correction">
              {result.rateDiagnosis.correctionRatio.toFixed(6)}&times;
            </Row>
          )}
        </Group>

        <Group title="Sources">
          <Row label="Video">
            {result.primaryCodec?.toUpperCase() ?? "--"}
            {result.primaryFps
              ? ` · ${result.primaryFps.toFixed(3).replace(/\.?0+$/, "")} fps`
              : ""}
          </Row>
          <Row label="Audio">
            {result.secondaryCodec?.toUpperCase() ?? "--"}
            {result.secondaryFps
              ? ` · ${result.secondaryFps.toFixed(3).replace(/\.?0+$/, "")} fps`
              : ""}
          </Row>
          {(result.primaryTrack ?? 0) > 0 && (
            <Row label="Video track">#{(result.primaryTrack ?? 0) + 1}</Row>
          )}
          {(result.secondaryTrack ?? 0) > 0 && (
            <Row label="Audio track">#{(result.secondaryTrack ?? 0) + 1}</Row>
          )}
          <Row label="Video length">{formatDuration(result.primaryDurationS)}</Row>
          <Row label="Audio length">{formatDuration(result.secondaryDurationS)}</Row>
        </Group>
      </div>

      {/* A silently adjusted number is a number nobody can check. */}
      {!!result.codecDelayMs && (
        <p className="mt-3.5 flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground">
          <Info className="mt-[3px] h-3 w-3 shrink-0" aria-hidden />
          <span>
            <b className="font-semibold text-foreground">{formatDelay(result.codecDelayMs)}</b> of
            codec delay was removed. One side is a raw {result.primaryCodec?.toUpperCase()}/
            {result.secondaryCodec?.toUpperCase()} stream with no timestamps, so its decoder
            priming would otherwise land in the measurement. Files in a container carry
            timestamps and need no such adjustment.
          </span>
        </p>
      )}

      {result.rateDiagnosis?.explanation && (
        <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
          {result.rateDiagnosis.explanation}
        </p>
      )}

      {result.error && (
        <p className="mt-3 text-[11px] leading-relaxed text-destructive">{result.error}</p>
      )}

      {result.primaryPath && (
        <div className="mt-3.5 space-y-0.5">
          <p
            className="truncate font-mono text-[10.5px] text-muted-foreground"
            title={result.primaryPath}
          >
            {result.primaryPath}
          </p>
          {result.secondaryPath && (
            <p
              className="truncate font-mono text-[10.5px] text-muted-foreground"
              title={result.secondaryPath}
            >
              {result.secondaryPath}
            </p>
          )}
        </div>
      )}

      {actions && <div className="mt-3.5 flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}
