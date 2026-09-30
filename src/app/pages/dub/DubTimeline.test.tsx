import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import { asLoadedPlan } from "@/lib/dubPlanEdit";
import { DRAFT_NOTE, type DubSyncOutcome, type DubSyncPlan } from "@/lib/types";

import { createCursorStore } from "./cursorStore";
import { DubTimeline, TimelineMessage } from "./DubTimeline";

/** The engine's real plan for the five-minute fixture: four stretches of
 *  dub, two cuts filled from the original, a head and a tail. */
const plan = (outcomeFixture as unknown as DubSyncOutcome).plan as DubSyncPlan;
const never = () => Promise.reject(new Error("not in this test"));

function render(props: Partial<React.ComponentProps<typeof DubTimeline>> = {}): string {
  return renderToStaticMarkup(<DubTimeline plan={plan} mode="done" fetchPeaks={never} cursor={createCursorStore()} {...props} />);
}

const blocks = (html: string, kind: string) => html.match(new RegExp(`class="segm ${kind}(?: sel)?"`, "g")) ?? [];

describe("DubTimeline", () => {
  it("draws the mockup's timeline: ruler, two labelled lanes, the minimap", () => {
    const html = render();
    expect(html).toMatch(/^<div class="tl dub-tl" style="padding-top:2px" tabindex="0" role="group" aria-label="Timeline">/);
    expect(html).toContain('<div class="ruler">');
    expect(html).toContain('<span class="lh">Original</span>');
    expect(html).toContain('<span class="lh">Dub</span>');
    expect(html).toContain('aria-label="Waveforms of the original and the synced dub"');
    expect(html).toContain('class="minimap');
    expect(html).toContain('aria-label="The whole film: click or drag to move the view"');
  });

  it("rules a short film to the second, ten ticks at most", () => {
    const html = render();
    const labels = [...html.matchAll(/<span style="left:(?:0|[\d.]+px)">([^<]+)<\/span>/g)].map((m) => m[1]);
    // 300.01 s is just over ten half-minutes, so a tick a minute.
    expect(labels).toEqual(["0:00:00", "0:01:00", "0:02:00", "0:03:00", "0:04:00"]);
  });

  it("draws the stretches of dub tinted and the original's hatched, the selected one outlined", () => {
    const html = render({ selected: 3 });
    expect(blocks(html, "dub")).toHaveLength(4);
    expect(blocks(html, "fill")).toHaveLength(4);
    expect(html.match(/class="segm dub sel"/g)).toHaveLength(1);
    // The minimap marks every stretch taken from the original, and the view.
    expect(html.match(/<i style="left:[\d.]+%;width:max\(2px, [\d.]+%\)"><\/i>/g)).toHaveLength(4);
    expect(html).toMatch(/<b style="left:0;width:800px"><\/b>/);
  });

  it("outlines what the engine has not placed yet, while it places", () => {
    const draft: DubSyncPlan = { ...plan, segments: plan.segments.map((s, i) => (i >= 4 && s.kind === "fill" ? { ...s, note: DRAFT_NOTE } : s)) };
    const html = render({ plan: draft, mode: "placing" });
    expect(blocks(html, "pending")).toHaveLength(2);
    expect(blocks(html, "fill")).toHaveLength(2);
  });

  it("draws a pair as loaded: only the dub, at the video's start", () => {
    const loaded = asLoadedPlan({ videoPath: "/v.mkv", dubPath: "/d.ac3", videoTrack: 0, dubTrack: 0, videoDurationS: 8430, dubDurationS: 7810 });
    const html = render({ plan: loaded, mode: "loaded" });
    expect(blocks(html, "dub")).toHaveLength(1);
    expect(blocks(html, "fill")).toHaveLength(0);
    expect(html).toMatch(/class="segm dub" style="left:0;width:741\.1\d*px"/);
  });

  it("puts grips on the cuts only when they can be moved", () => {
    expect(render({ editable: true, grips: true }).match(/class="grip"/g)).toHaveLength(plan.segments.length - 1);
    expect(render({ editable: true })).not.toContain('class="grip"');
    expect(render({ grips: true })).not.toContain('class="grip"');
  });

  it("offers a hand to grab when read-only, a crosshair when the cuts can be edited", () => {
    expect(render()).toContain('class="dub-lanes dub-grab"');
    expect(render({ editable: true })).toContain('class="dub-lanes dub-edit"');
  });

  it("draws the playhead where the cursor is", () => {
    expect(render()).not.toContain("head-line");
    const html = render({ cursor: createCursorStore(150) });
    // 14 px padding and the 76 px lane labels, then 150 s of 300.01 on 800 px.
    expect(html).toMatch(/<div class="head-line" style="left:489\.9\d*px" aria-hidden="true"><\/div>/);
  });

  it("ticks the picture's cuts on the ruler, named, and hidden from assistive tech", () => {
    const html = render({ editable: true, shotCuts: [12.5, 150, 299] });
    const ticks = html.match(/<span[^>]*data-shot-cut[^>]*>/g) ?? [];
    expect(ticks).toHaveLength(3);
    expect(ticks[0]).toContain('title="Picture cut 0:00:12.500"');
    expect(ticks[1]).toContain(`left:${Math.round((150 / plan.videoDurationS) * 800)}px`);
    expect(html).toMatch(/<div class="dub-shots" aria-hidden="true"><span[^>]*data-shot-cut/);
  });

  it("draws no ticks when the cuts are not known", () => {
    expect(render({ editable: true })).not.toContain("data-shot-cut");
    expect(render({ editable: true, shotCuts: [] })).not.toContain("data-shot-cut");
  });

  it("asks for no picture cuts while the view is wider than five minutes", () => {
    const asked: number[][] = [];
    const html = render({
      editable: true,
      fetchShotCuts: (_path, startS, endS) => {
        asked.push([startS, endS]);
        return never();
      },
    });
    expect(html).not.toContain("data-shot-cut");
    expect(asked).toEqual([]);
  });

  it("says how far the engine has read a file it is reading for the first time", () => {
    const html = render({ reading: { [plan.dubPath]: 42 } });
    expect(html).toContain('<span class="dub-reading">Reading 42%</span>');
    expect(html.match(/dub-reading/g)).toHaveLength(1);
  });

  it("says why there is nothing to draw yet", () => {
    expect(renderToStaticMarkup(<TimelineMessage>Reading the files…</TimelineMessage>)).toContain("Reading the files…");
  });
});
