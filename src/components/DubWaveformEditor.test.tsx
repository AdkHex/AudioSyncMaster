import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DubWaveformEditor } from "./DubWaveformEditor";
import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import type { DubSyncOutcome, DubSyncPlan } from "@/lib/types";

/** The engine's real plan for the five-minute fixture: four cuts. */
const plan = (outcomeFixture as unknown as DubSyncOutcome).plan as DubSyncPlan;

const never = () => Promise.reject(new Error("not in this test"));

function render(props: Partial<React.ComponentProps<typeof DubWaveformEditor>> = {}): string {
  return renderToStaticMarkup(
    <DubWaveformEditor
      plan={plan}
      fetchPeaks={never}
      onPreview={never}
      renderDubPreview={never}
      readPreviewBytes={never}
      onApply={never}
      onClose={() => undefined}
      {...props}
    />,
  );
}

describe("DubWaveformEditor", () => {
  it("opens on the engine's plan with every piece listed and nothing to write yet", () => {
    const html = render();
    expect(html).toContain("Edit the cuts");
    expect(html).toContain("No changes yet.");
    // Every piece of the plan, with its offset to the millisecond.
    const rows = html.match(/<li>/g)?.length ?? 0;
    expect(rows).toBe(plan.segments.length);
    expect(html).toContain("+0.806s");
    expect(html).toContain("-14.194s");
    expect(html).toContain("dub is cut here");
    // Writing is offered only once something changed.
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Write the track with these cuts/);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Back to the engine&#x27;s plan/);
  });

  it("draws both lanes on the video's clock", () => {
    const html = render();
    expect(html).toContain('aria-label="Waveforms of the original and the synced dub"');
    expect(html).toContain("0:00:00.000 – 0:05:00.010");
  });

  it("says how to scroll, zoom, slip a stretch and split", () => {
    const html = render();
    expect(html).toContain("Drag to scroll, Ctrl/⌘-wheel or pinch to zoom.");
    expect(html).toContain("Alt/⌥-drag a stretch");
    expect(html).toContain("Double-click or S splits.");
    expect(html).toContain("it snaps to the picture&#x27;s cuts");
    expect(html).not.toContain("Shift-drag pans");
  });

  it("shows the write in progress and offers to stop it", () => {
    const html = render({ busy: { percent: 42, stage: "writing 6ch" }, onStop: () => undefined });
    expect(html).toContain("writing 6ch");
    expect(html).toContain("42%");
    expect(html).toContain(">Stop<");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Close/);
  });

  it("offers the engine's plan back when opened on cuts that were already edited", () => {
    const edited: DubSyncPlan = {
      ...plan,
      segments: plan.segments.map((s, i) => (i === 1 && s.kind === "dub" ? { ...s, offsetS: s.offsetS! + 0.01, sourceStartS: s.sourceStartS + 0.01 } : s)),
    };
    const html = render({ plan: edited, enginePlan: plan });
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Back to the engine&#x27;s plan/);
    expect(html).toContain("Back to the engine");
  });

  it("offers the player, hidden until a button opens it", () => {
    const html = render();
    expect(html).toContain("Play video");
    expect(html).toContain("Play sample");
    expect(html).toContain("Open in player");
    // The player itself stays closed until Play video or Play sample.
    expect(html).not.toContain('aria-label="Preview player"');
    expect(html).not.toContain("<video");
  });

  it("asks for no picture cuts while the view is wider than five minutes", () => {
    // The fixture is 5:00.010 long, and the editor opens on all of it.
    const asked: number[][] = [];
    const html = render({
      fetchShotCuts: (_path, startS, endS) => {
        asked.push([startS, endS]);
        return never();
      },
    });
    expect(html).not.toContain("data-shot-cut");
    expect(asked).toEqual([]);
  });
});
