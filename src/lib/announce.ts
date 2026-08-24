/** Wording for screen-reader announcements.
 *
 *  Kept out of the component so the phrasing is testable on its own, and so a
 *  change to what gets said cannot accidentally change when it gets said.
 *
 *  Analysis is long-running and almost entirely visual: a progress bar, a table
 *  that fills in. Without an announcement a screen-reader user has no way to
 *  know a run finished, or that it found nothing.
 */

import type { RunSummary, SyncMode, SyncResult } from "./types";

/** How urgently an announcement interrupts. */
export type Politeness = "polite" | "assertive";

export interface Announcement {
  message: string;
  politeness: Politeness;
}

function plural(count: number, singular: string, suffix = "s"): string {
  return `${count} ${singular}${count === 1 ? "" : suffix}`;
}

export function announceRunStarted(pairCount: number, mode: SyncMode): Announcement {
  const label = mode === "compare" ? "combination" : "pair";
  return {
    message: `Analysis started. ${plural(pairCount, label)} queued.`,
    politeness: "polite",
  };
}

/** Summarise a finished run, leading with whatever needs attention.
 *
 *  A bare "analysis complete" is useless when half the files failed, so the
 *  counts that imply follow-up work come first.
 */
export function announceRunFinished(
  results: SyncResult[],
  summary: RunSummary | null,
  cancelled: boolean,
): Announcement {
  if (cancelled) {
    return {
      message: `Analysis stopped. ${plural(results.length, "result")} kept.`,
      politeness: "polite",
    };
  }

  if (results.length === 0) {
    return { message: "Analysis finished with no results.", politeness: "assertive" };
  }

  const matched = summary?.matched ?? results.length;
  const failed = summary?.failed ?? 0;

  const parts = [`Analysis complete. ${plural(matched, "file")} matched`];
  if (failed > 0) parts.push(`${failed} failed`);
  if (summary?.cuts) parts.push(`${plural(summary.cuts, "different cut")} detected`);
  if (summary?.rateMismatches) {
    parts.push(`${plural(summary.rateMismatches, "frame rate mismatch", "es")}`);
  }
  if (summary?.drifting) parts.push(`${plural(summary.drifting, "file")} drifting`);

  return {
    message: `${parts.join(", ")}.`,
    // Anything needing a decision interrupts; a clean run waits its turn.
    politeness: failed > 0 || summary?.cuts ? "assertive" : "polite",
  };
}

export function announceRunFailed(reason: string): Announcement {
  return { message: `Analysis failed. ${reason}`, politeness: "assertive" };
}

export function announceFilesAdded(count: number, kind: "video" | "audio"): Announcement {
  return {
    message: `${plural(count, `${kind} file`)} added.`,
    politeness: "polite",
  };
}

export function announceApplyFinished(written: number, failed: number): Announcement {
  if (written === 0) {
    return {
      message: `No files were written. ${plural(failed, "failure")}.`,
      politeness: "assertive",
    };
  }
  return {
    message:
      failed > 0
        ? `${plural(written, "corrected file")} written, ${failed} failed.`
        : `${plural(written, "corrected file")} written.`,
    politeness: failed > 0 ? "assertive" : "polite",
  };
}

/** Periodic progress, throttled to milestones.
 *
 *  Announcing every file would flood the screen reader on a 24-episode season
 *  and drown out everything else, so only quarter marks are spoken.
 */
export function announceProgress(
  processed: number,
  total: number,
): Announcement | null {
  if (total <= 0 || processed <= 0 || processed >= total) return null;
  const milestones = [0.25, 0.5, 0.75];
  const fraction = processed / total;
  const previous = (processed - 1) / total;
  const crossed = milestones.find((m) => previous < m && fraction >= m);
  if (crossed === undefined) return null;
  return {
    message: `${Math.round(crossed * 100)} percent complete. ${processed} of ${total} files.`,
    politeness: "polite",
  };
}
