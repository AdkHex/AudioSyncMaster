import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DubWaveformStrip } from "./DubWaveformStrip";
import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import { DRAFT_NOTE, type DubSyncOutcome, type DubSyncPlan } from "@/lib/types";

const plan = (outcomeFixture as unknown as DubSyncOutcome).plan as DubSyncPlan;
const never = () => Promise.reject(new Error("not in this test"));
const pair = { videoPath: plan.videoPath, dubPath: plan.dubPath, videoTrack: 0, dubTrack: 0, name: "video.mkv", dubName: "dub.ac3" };

function render(props: Partial<React.ComponentProps<typeof DubWaveformStrip>> = {}): string {
  return renderToStaticMarkup(
    <DubWaveformStrip
      pair={pair}
      plan={null}
      status={null}
      choices={[{ name: "video.mkv" }]}
      index={0}
      onChoose={() => undefined}
      fetchPeaks={never}
      buildWaveform={never}
      renderDubPreview={never}
      readPreviewBytes={never}
      reading={{}}
      {...props}
    />,
  );
}

describe("DubWaveformStrip", () => {
  it("says it is reading the files until both lengths are known", () => {
    const html = render();
    expect(html).toContain("Reading the waveforms");
    expect(html).not.toContain("<canvas");
  });

  it("names the file being read and how far along it is", () => {
    const html = render({ reading: { [plan.dubPath]: 42 } });
    expect(html).toContain("Reading the waveforms — dub.ac3 42%");
  });

  it("draws the engine's draft live, with the stage over the ruler", () => {
    const draft: DubSyncPlan = {
      ...plan,
      segments: plan.segments.map((s) => (s.kind === "fill" ? { ...s, note: DRAFT_NOTE } : s)),
    };
    const html = render({ plan: draft, status: { label: "Placing the cuts", percent: 72 } });
    expect(html).toContain("<canvas");
    expect(html).toContain("The dub is being laid onto the video");
    expect(html).not.toContain("Edit the cuts");
  });

  it("offers the cut editor once the plan is final", () => {
    const html = render({ plan, onEdit: () => undefined });
    expect(html).toContain("The synced track as it was written");
    expect(html).toContain("Edit the cuts");
  });

  it("offers Play video only once the plan is final and the queue is not running", () => {
    const html = render({ plan, playable: true });
    expect(html).toContain("Play video");
    // The player itself stays hidden until the button opens it.
    expect(html).not.toContain('aria-label="Preview player"');
    // Without playable -- a draft, or a queue still running -- no player.
    expect(render({ plan })).not.toContain("Play video");
  });

  it("says how to get around the waveforms whenever they are drawn, during a run as after it", () => {
    const hint = "Drag to scroll · Ctrl/⌘-wheel or pinch to zoom · click to place the cursor";
    const draft: DubSyncPlan = { ...plan, segments: plan.segments.map((s) => (s.kind === "fill" ? { ...s, note: DRAFT_NOTE } : s)) };
    expect(render({ plan: draft, status: { label: "Placing the cuts", percent: 72 } })).toContain(hint);
    expect(render({ plan })).toContain(hint);
    // Shift-drag is no longer the way to scroll, so it is no longer offered.
    expect(render({ plan })).not.toContain("Shift-drag pans");
    // Nothing to get around while the files are still being read.
    expect(render()).not.toContain(hint);
  });

  it("draws the whole film under the lanes, to click or drag the view along", () => {
    const html = render({ plan });
    expect(html).toContain('aria-label="The whole film: click or drag to move the view"');
  });

  it("lets one of several pairs be chosen", () => {
    const html = render({ choices: [{ name: "E01.mkv" }, { name: "E02.mkv" }], index: 1 });
    expect(html).toContain('aria-label="Pair shown"');
    expect(html).toContain("E02.mkv");
  });
});
