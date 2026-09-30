/** A waveform drawn as a filled min/max envelope (design/mockup/kit.tsx):
 *  deterministic from a seed, for illustrations and placeholders. Real
 *  waveforms come from the engine's peaks and are drawn by the timeline. */

import type { CSSProperties } from "react";

function rng(seed: number) {
  let s = (seed * 9301 + 49297) >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 10000) / 10000;
  };
}

/** Speech-like levels: phrases of energy separated by short pauses. */
export function envelope(seed: number, n: number): number[] {
  const r = rng(seed);
  const out: number[] = [];
  let lvl = 0.5;
  let left = 0;
  for (let i = 0; i < n; i++) {
    if (left-- <= 0) {
      left = 4 + Math.floor(r() * 18);
      lvl = r() < 0.15 ? 0.05 : 0.25 + r() * 0.7;
    }
    out.push(Math.max(0.03, Math.min(1, lvl * (0.6 + r() * 0.5))));
  }
  return out;
}

export function Wave({ seed, color, n = 240, style }: { seed: number; color: string; n?: number; style?: CSSProperties }) {
  const e = envelope(seed, n);
  const W = 1000;
  const top = e.map((v, i) => `${((i / (n - 1)) * W).toFixed(1)},${(50 - v * 46).toFixed(1)}`);
  const bot = e.map((v, i) => `${((i / (n - 1)) * W).toFixed(1)},${(50 + v * 46).toFixed(1)}`).reverse();
  return (
    <svg viewBox={`0 0 ${W} 100`} preserveAspectRatio="none" aria-hidden style={{ display: "block", width: "100%", height: "100%", ...style }}>
      <polygon points={[...top, ...bot].join(" ")} fill={color} />
    </svg>
  );
}
