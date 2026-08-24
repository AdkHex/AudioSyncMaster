import { Info } from "lucide-react";

import { Pill } from "@/components/ui";
import {
  formatDelay,
  formatDrift,
  formatDuration,
  frameOffset,
  type SyncResult,
} from "@/lib/types";

interface ResultDetailProps {
  result: SyncResult;
  columnCount: number;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <dt className="shrink-0 text-[11px] text-muted-foreground">{label}</dt>
      <dd className="tabular text-right font-mono text-[11.5px] text-foreground">{children}</dd>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="mb-1 text-[10px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">
        {title}
      </h4>
      <dl className="divide-y divide-border/60">{children}</dl>
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
export function ResultDetail({ result, columnCount }: ResultDetailProps) {
  const frames = frameOffset(result.delayMs, result.primaryFps);

  return (
    <tr className="border-b border-border bg-sunken/60 last:border-0">
      <td colSpan={columnCount} className="px-4 py-3">
        <div className="grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
          <Group title="Measurement">
            <Row label="Delay">
              {formatDelay(result.delayMs)}
              {frames !== null && (
                <span className="ml-1.5 font-sans text-[10.5px] text-muted-foreground">
                  ≈ {frames} frame{Math.abs(frames) === 1 ? "" : "s"}
                </span>
              )}
            </Row>
            <Row label="At file start">{formatDelay(result.startDelayMs)}</Row>
            <Row label="At file end">{formatDelay(result.endDelayMs)}</Row>
            {result.delayAtStartMs !== null &&
              result.delayAtStartMs !== result.delayMs && (
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
            <Row label="Rate">{formatDrift(result.driftMsPerS)}</Row>
            <Row label="Across the file">{formatDelay(result.totalDriftMs)}</Row>
            {result.rateDiagnosis?.sourceFps && result.rateDiagnosis?.targetFps && (
              <Row label="Conversion">
                {result.rateDiagnosis.sourceFps} → {result.rateDiagnosis.targetFps} fps
              </Row>
            )}
            {result.rateDiagnosis?.correctionRatio && (
              <Row label="Speed correction">
                {result.rateDiagnosis.correctionRatio.toFixed(6)}×
              </Row>
            )}
          </Group>

          <Group title="Sources">
            <Row label="Video">
              {result.primaryCodec?.toUpperCase() ?? "--"}
              {result.primaryFps ? ` · ${result.primaryFps.toFixed(3).replace(/\.?0+$/, "")} fps` : ""}
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
          <p className="mt-3 flex items-start gap-1.5 border-t border-border pt-2.5 text-[11px] text-muted-foreground">
            <Info className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
            <span>
              <b className="font-semibold text-foreground">
                {formatDelay(result.codecDelayMs)}
              </b>{" "}
              of codec delay was removed. {result.primaryCodec?.toUpperCase()} and{" "}
              {result.secondaryCodec?.toUpperCase()} decode with different alignment, which
              would otherwise land in the measurement.
            </span>
          </p>
        )}

        {result.rateDiagnosis?.explanation && (
          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
            {result.rateDiagnosis.explanation}
          </p>
        )}

        {result.error && (
          <p className="mt-3 border-t border-border pt-2.5 text-[11px] text-destructive">
            {result.error}
          </p>
        )}

        {result.primaryPath && (
          <div className="mt-3 space-y-0.5 border-t border-border pt-2.5">
            <p className="truncate font-mono text-[10.5px] text-muted-foreground" title={result.primaryPath}>
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
      </td>
    </tr>
  );
}

/** Compact inline summary for the collapsed row. */
export function DetailHint({ result }: { result: SyncResult }) {
  const bits: string[] = [];
  if (result.windowsUsed !== null && result.windowsTotal !== null) {
    bits.push(`${result.windowsUsed}/${result.windowsTotal} windows`);
  }
  if (result.codecDelayMs) bits.push("codec-corrected");
  if (bits.length === 0) return null;
  return (
    <Pill tone="neutral" className="font-normal">
      {bits.join(" · ")}
    </Pill>
  );
}
