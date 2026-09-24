import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DubWaveformView } from "./DubWaveformView";
import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import type { DubSyncOutcome, DubSyncPlan } from "@/lib/types";

const plan = (outcomeFixture as unknown as DubSyncOutcome).plan as DubSyncPlan;
const never = () => Promise.reject(new Error("not in this test"));

/** The canvas's opening tag with the given aria-label. */
function canvasTag(html: string, label: string): string {
  return html.match(new RegExp(`<canvas[^>]*aria-label="${label}"[^>]*>`))?.[0] ?? "";
}

describe("DubWaveformView", () => {
  it("draws the lanes and, under them, the whole film as an overview", () => {
    const html = renderToStaticMarkup(<DubWaveformView plan={plan} fetchPeaks={never} />);
    expect(canvasTag(html, "Waveforms of the original and the synced dub")).not.toBe("");
    const overview = canvasTag(html, "The whole film: click or drag to move the view");
    expect(overview).toContain("height:14px");
  });

  it("offers a hand to grab in the strip, a crosshair to place the cursor in the editor", () => {
    const strip = renderToStaticMarkup(<DubWaveformView plan={plan} fetchPeaks={never} />);
    expect(canvasTag(strip, "Waveforms of the original and the synced dub")).toContain("cursor-grab");
    const editor = renderToStaticMarkup(<DubWaveformView plan={plan} fetchPeaks={never} editable />);
    expect(canvasTag(editor, "Waveforms of the original and the synced dub")).toContain("cursor-crosshair");
  });

  it("ticks the picture's cuts on the ruler, named, and hidden from assistive tech", () => {
    // The whole five-minute film on the default 800 px.
    const html = renderToStaticMarkup(<DubWaveformView plan={plan} fetchPeaks={never} editable shotCuts={[12.5, 150, 299]} />);
    const ticks = html.match(/<span[^>]*data-shot-cut[^>]*>/g) ?? [];
    expect(ticks).toHaveLength(3);
    expect(ticks[0]).toContain('title="Picture cut 0:00:12.500"');
    expect(ticks[1]).toContain(`left:${Math.round((150 / plan.videoDurationS) * 800)}px`);
    expect(html).toMatch(/<div aria-hidden="true" class="pointer-events-none[^"]*"[^>]*><span[^>]*data-shot-cut/);
  });

  it("draws no ticks when the cuts are not known", () => {
    expect(renderToStaticMarkup(<DubWaveformView plan={plan} fetchPeaks={never} editable />)).not.toContain("data-shot-cut");
    expect(renderToStaticMarkup(<DubWaveformView plan={plan} fetchPeaks={never} editable shotCuts={[]} />)).not.toContain("data-shot-cut");
  });
});
