/** The player's mix switch: which track plays, and how, so a value stays
 *  out of the component file (components only export components) and the
 *  table is provable without a browser. */

export type MixMode = "dub" | "original" | "both";

/** Gains and pans for each mix. Both = original in the left ear, dub in
 *  the right, so the two tracks can be compared without switching. */
export const MIX_GAINS: Record<MixMode, { dub: number; original: number; dubPan: number; originalPan: number }> = {
  dub: { dub: 1, original: 0, dubPan: 0, originalPan: 0 },
  original: { dub: 0, original: 1, dubPan: 0, originalPan: 0 },
  both: { dub: 1, original: 1, dubPan: 1, originalPan: -1 },
};
