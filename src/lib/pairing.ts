/** Manual corrections to an automatic pairing.
 *
 *  Matching is a guess, and on messy filenames it is sometimes a wrong one. The
 *  preview showed the mistake and then offered no way to correct it, so the only
 *  recourse was renaming files on disk and starting over.
 *
 *  An override is stored per video rather than as a rewritten pair list, so it
 *  survives a re-match: change the pattern, and any correction the user already
 *  made still applies to the same video.
 */

import type { MatchPair, PairingReport } from "./types";

/** videoPath -> the audio the user chose for it, or null to exclude it. */
export type PairOverrides = Record<string, string | null>;

/** Apply the user's corrections to a report from the engine. */
export function applyOverrides(
  report: PairingReport,
  overrides: PairOverrides,
  availableAudio: { path: string; name: string }[],
): PairingReport {
  if (Object.keys(overrides).length === 0) return report;

  const audioByPath = new Map(availableAudio.map((file) => [file.path, file.name]));
  const pairs: MatchPair[] = [];
  const handled = new Set<string>();

  for (const pair of report.pairs) {
    handled.add(pair.primaryPath);
    if (!(pair.primaryPath in overrides)) {
      pairs.push(pair);
      continue;
    }

    const chosen = overrides[pair.primaryPath];
    // An explicit null means "do not analyse this video at all".
    if (chosen === null) continue;

    pairs.push({
      ...pair,
      secondaryPath: chosen,
      secondaryName: audioByPath.get(chosen) ?? basename(chosen),
      method: "chosen by hand",
      score: 1,
    });
  }

  // A video the matcher gave up on can still be paired by hand.
  for (const [videoPath, chosen] of Object.entries(overrides)) {
    if (handled.has(videoPath) || chosen === null) continue;
    pairs.push({
      primaryPath: videoPath,
      secondaryPath: chosen,
      primaryName: basename(videoPath),
      secondaryName: audioByPath.get(chosen) ?? basename(chosen),
      key: basename(videoPath),
      method: "chosen by hand",
      score: 1,
      primaryTrack: 0,
      secondaryTrack: 0,
    });
  }

  pairs.sort((a, b) => a.primaryName.localeCompare(b.primaryName));

  const pairedVideos = new Set(pairs.map((pair) => pair.primaryPath));
  const pairedAudio = new Set(pairs.map((pair) => pair.secondaryPath));

  return {
    ...report,
    pairs,
    // Recompute rather than carry the engine's lists forward: a manual pairing
    // changes who is left over, and a stale "unmatched" entry is worse than none.
    unmatchedPrimary: report.unmatchedPrimary.filter(
      (name) => !pairs.some((pair) => pair.primaryName === name),
    ),
    unmatchedSecondary: availableAudio
      .filter((file) => !pairedAudio.has(file.path))
      .map((file) => file.name),
    method: hasManualPairs(pairs) ? `${report.method} + manual` : report.method,
    warning: pairedVideos.size === 0 ? "Every video has been excluded." : report.warning,
  };
}

function hasManualPairs(pairs: MatchPair[]): boolean {
  return pairs.some((pair) => pair.method === "chosen by hand");
}

function basename(path: string): string {
  const normalised = path.replace(/\\/g, "/");
  return normalised.slice(normalised.lastIndexOf("/") + 1);
}

/** Drop overrides whose files are no longer selected.
 *
 *  Without this, changing the selection leaves corrections pointing at files
 *  that are gone, and the engine is asked to analyse something that no longer
 *  exists.
 */
export function pruneOverrides(
  overrides: PairOverrides,
  videoPaths: string[],
  audioPaths: string[],
): PairOverrides {
  const videos = new Set(videoPaths);
  const audio = new Set(audioPaths);
  const pruned: PairOverrides = {};

  for (const [videoPath, chosen] of Object.entries(overrides)) {
    if (!videos.has(videoPath)) continue;
    if (chosen !== null && !audio.has(chosen)) continue;
    pruned[videoPath] = chosen;
  }
  return pruned;
}

export function countManualPairs(overrides: PairOverrides): number {
  return Object.values(overrides).filter((value) => value !== null).length;
}

export function countExcluded(overrides: PairOverrides): number {
  return Object.values(overrides).filter((value) => value === null).length;
}
