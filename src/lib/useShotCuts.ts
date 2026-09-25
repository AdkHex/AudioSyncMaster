/** The video's picture cuts in the editor's view, for the ruler and for
 *  snapping a dragged cut.
 *
 *  Asked for only while the view shows at most SHOT_CUTS_MAX_VIEW_S, a
 *  quarter of a second after it stops moving, one request at a time: the
 *  engine decodes the picture to answer, and a request made for every
 *  frame of a pan would queue minutes of work for views long gone. What
 *  comes back is kept in blocks of the film (see waveformView.ts), so
 *  panning back asks for nothing. */

import { useEffect, useMemo, useRef, useState } from "react";

import type { ShotCuts } from "./api";
import {
  SHOT_CUTS_MAX_VIEW_S,
  shotCutsIn,
  shotSpanToFetch,
  storeShotCuts,
  type ShotBlocks,
  type View,
} from "./waveformView";

export type FetchShotCuts = (videoPath: string, startS: number, endS: number) => Promise<ShotCuts>;

/** How long the view must hold still before its cuts are asked for. */
const SETTLE_MS = 250;

const NONE: ShotBlocks = new Map();

/** The cuts in `view`, or null while it is too wide to show them (or
 *  there is nothing to ask). */
export function useShotCuts(
  videoPath: string,
  durationS: number,
  view: View,
  fetchShotCuts?: FetchShotCuts,
): number[] | null {
  const [known, setKnown] = useState<{ path: string; blocks: ShotBlocks }>({ path: videoPath, blocks: NONE });
  // Bumped when a request lands, so the view as it is by then is looked at.
  const [landed, setLanded] = useState(0);
  const busyRef = useRef(false);
  const pathRef = useRef(videoPath);
  pathRef.current = videoPath;
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const blocks = known.path === videoPath ? known.blocks : NONE;
  const active = !!fetchShotCuts && view.endS - view.startS <= SHOT_CUTS_MAX_VIEW_S;

  useEffect(() => {
    if (!active || !fetchShotCuts) return;
    const span = shotSpanToFetch(blocks, view, durationS);
    if (!span) return;
    const timer = window.setTimeout(() => {
      // One at a time; the one in flight looks again when it lands.
      if (busyRef.current) return;
      busyRef.current = true;
      const path = videoPath;
      fetchShotCuts(path, span.startS, span.endS)
        // An engine that fails is not asked about the same film again.
        .catch((): ShotCuts => ({ cuts: null, frameS: null, ...span }))
        .then((reply) => {
          // An answer about a video no longer shown is dropped.
          if (!mountedRef.current || path !== pathRef.current) return;
          setKnown((current) => ({
            path,
            blocks: storeShotCuts(current.path === path ? current.blocks : NONE, reply, durationS),
          }));
        })
        .finally(() => {
          busyRef.current = false;
          if (mountedRef.current) setLanded((n) => n + 1);
        });
    }, SETTLE_MS);
    return () => window.clearTimeout(timer);
    // `landed` re-runs this once the request in flight is done.
  }, [active, fetchShotCuts, videoPath, durationS, view, blocks, landed]);

  return useMemo(() => (active ? shotCutsIn(blocks, view, durationS) : null), [active, blocks, view, durationS]);
}
