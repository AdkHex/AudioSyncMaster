import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DubQueuePanel } from "./DubQueuePanel";
import { dubQueueReducer, initialDubQueueState, type DubQueueState } from "@/lib/dubQueueReducer";
import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import { UNMATCHED_FILL_NOTE, type DubJobOutcome, type DubSyncOutcome } from "@/lib/types";

/** The engine's real answer for the five-minute fixture (an MKV with stereo
 *  AAC against a raw 5.1 AC3 dub with four cuts), captured from the bridge.
 *  Rendering it here is the check that the panel shows what the engine says,
 *  not what a hand-written stub happens to contain. */
const single = outcomeFixture as unknown as DubSyncOutcome;

const noop = () => undefined;

function render(state: DubQueueState, onEdit?: (job: number) => void): string {
  return renderToStaticMarkup(
    <DubQueuePanel state={state} onReveal={noop} onOpen={noop} onOpenConsole={noop} onEdit={onEdit} />,
  );
}

function queued(names: [string, string][]): DubQueueState {
  return dubQueueReducer(initialDubQueueState, {
    type: "queueStarted",
    jobs: names.map(([name, dubName]) => ({ name, dubName })),
  });
}

/** The fixture outcome, as the engine reports it for one job of a batch. */
function outcome(job: number, patch: Partial<DubSyncOutcome> = {}): DubJobOutcome {
  const source = { ...single, ...patch };
  return {
    job,
    plan: source.plan,
    output: source.output,
    verification: source.verification,
    muxedPath: source.muxedPath ?? null,
    cancelled: source.cancelled,
    error: source.error ?? null,
  };
}

describe("DubQueuePanel", () => {
  it("shows the stage and percent while a job is running, before any plan exists", () => {
    let state = queued([["Show.S01E01.mkv", "Show.S01E01.dub.eac3"]]);
    state = dubQueueReducer(state, { type: "jobStart", job: 0 });
    state = dubQueueReducer(state, { type: "jobProgress", job: 0, percent: 40, stage: "finding the offsets" });
    const html = render(state);
    expect(html).toContain("Show.S01E01.mkv");
    expect(html).toContain("Finding where the dub belongs");
    expect(html).toContain("40%");
    expect(html).toContain("Syncing");
    expect(html).not.toContain("Checked against the video");
  });

  it("lists every piece of the plan, on the video's timeline", () => {
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0),
    });
    const html = render(state);
    const dubs = single.plan!.segments.filter((s) => s.kind === "dub").length;
    const fills = single.plan!.segments.filter((s) => s.kind === "fill").length;
    expect(html).toContain(`${dubs} stretches of dub, ${fills} fills from the original`);
    // The cut the engine found at 3:00.0-3:02.5, filled from the original.
    expect(html).toContain("dub is cut here");
    expect(html).toMatch(/0:02:59\.99\d – 0:03:02\.49\d/);
    // Offsets to the millisecond, as measured.
    expect(html).toContain("-14.194s");
    expect(html).toContain("+0.806s");
  });

  it("reports the check of the written file against the video", () => {
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0),
    });
    const html = render(state);
    expect(html).toContain("Checked against the video");
    // Under ten milliseconds the tenth is shown: 0.2 is not 0.
    expect(html).toContain("0.2 ms typical");
    expect(html).toContain("0.4 ms at worst");
    expect(html).toContain("Measured at 10 spots");
    expect(html).not.toContain("would be audible");
  });

  it("names the output and offers to show it", () => {
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0),
    });
    const html = render(state);
    expect(html).toContain("dub.dubsynced.ac3");
    expect(html).toContain("5.1 · 48 kHz");
    expect(html).toContain("Show in folder");
    expect(html).toContain("Done");
  });

  it("offers to edit the cuts of a finished job, but not while the queue is still running", () => {
    let state = queued([["a.mkv", "a.ac3"], ["b.mkv", "b.ac3"]]);
    state = dubQueueReducer(state, { type: "jobDone", outcome: outcome(0) });
    // Job 1 is still queued: the engine is busy, the button is there but disabled.
    expect(render(state, noop)).toMatch(/<button[^>]*disabled=""[^>]*>[^<]*<svg[^>]*>[\s\S]*?<\/svg>Edit the cuts/);
    state = dubQueueReducer(state, { type: "batchDone", outcomes: [outcome(0), outcome(1)], cancelled: false });
    const html = render(state, noop);
    expect(html).toContain("Edit the cuts");
    expect(html).not.toMatch(/disabled=""[^>]*title="Available once/);
    // Without the desktop there is nothing to edit with.
    expect(render(state)).toMatch(/disabled=""[^>]*title="Available once the queue has finished"/);
  });

  it("shows every job of a season, each with its own state", () => {
    let state = queued([
      ["Show.S01E01.mkv", "Show.S01E01.dub.eac3"],
      ["Show.S01E02.mkv", "Show.S01E02.dub.eac3"],
      ["Show.S01E03.mkv", "Show.S01E03.dub.eac3"],
    ]);
    state = dubQueueReducer(state, { type: "jobDone", outcome: outcome(0) });
    state = dubQueueReducer(state, {
      type: "jobDone",
      outcome: { job: 1, plan: null, output: null, verification: null, muxedPath: null, error: "No part of the dub could be matched" },
    });
    const html = render(state);
    expect(html).toContain("Show.S01E01.mkv");
    expect(html).toContain("Show.S01E02.mkv");
    expect(html).toContain("Show.S01E03.mkv");
    // Job 0 finished with the fixture's plan, job 1 failed, job 2 is waiting.
    expect(html).toContain("dub.dubsynced.ac3");
    expect(html).toContain("No part of the dub could be matched");
    expect(html).toContain("Waiting");
    // Overall progress counts the two closed jobs.
    expect(html).toContain("2/3");
  });

  it("explains a failed batch and points at the console", () => {
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "batchFailed",
      message: "The analysis engine stopped responding.",
    });
    const html = render(state);
    expect(html).toContain("The sync did not finish.");
    expect(html).toContain("The analysis engine stopped responding.");
    expect(html).toContain("Open the console");
  });

  it("tells a fill that replaced unmatched dub from a real cut", () => {
    const plan = single.plan!;
    const replaced = { ...single, plan: { ...plan, segments: plan.segments.map((s, i) =>
      s.kind === "fill" && i === 2 ? { ...s, note: UNMATCHED_FILL_NOTE } : s,
    ) } };
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0, { plan: replaced.plan }),
    });
    const html = render(state);
    expect(html).toContain("Replaced");
    expect(html).toContain("did not correlate — replaced; check this spot");
    expect(html).toContain("(1 replacing dub that did not correlate)");
    // The other fills are still plain cuts.
    expect(html).toContain("dub is cut here");
  });

  it("shows a note as information, not as something to check", () => {
    const plan = single.plan!;
    const kept = "kept the dub across 0:01:40.000 - 0:04:40.000: it did not correlate there, but the offset is the same either side and the dub is not silent";
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0, { plan: { ...plan, warnings: [], notes: [kept] } }),
    });
    const html = render(state);
    expect(html).toContain(kept);
    // A note carries the information glyph; the warning glyph is reserved
    // for things to check, and a track with none shows none.
    expect(html).toContain("lucide-info");
    expect(html).not.toContain("lucide-triangle-alert");
  });

  it("flags a stretch the check found audibly out", () => {
    const worse = {
      ...single,
      verification: {
        ...single.verification!,
        worstMs: 400,
        stretches: [{ startS: 1285, endS: 1311, residualMs: 400, windows: 3 }],
      },
    };
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0, { verification: worse.verification }),
    });
    const html = render(state);
    expect(html).toContain("1 stretch audibly out");
    expect(html).not.toContain("400 ms at worst");
    expect(html).toContain("0:21:25.000 – 0:21:51.000");
    expect(html).toContain("would be audible");
  });

  it("does not call a track good when the sweep says half of it is out", () => {
    const halfOut = {
      ...single,
      verification: { ...single.verification!, sweepMeasured: 181, sweepWithinAudible: 92, stretches: [] },
    };
    const state = dubQueueReducer(queued([["film.mkv", "film.dub.ac3"]]), {
      type: "jobDone",
      outcome: outcome(0, { verification: halfOut.verification }),
    });
    const html = render(state);
    expect(html).toContain("51% of the runtime within 45 ms");
    // The verification badge must not read green (the job row's Done tag is
    // green for its own reasons).
    expect(html).not.toContain('text-success">0 ms typical');
  });
});
