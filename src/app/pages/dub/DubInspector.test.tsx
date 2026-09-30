import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { dubQueueReducer, initialDubQueueState, type DubQueueJob } from "@/lib/dubQueueReducer";
import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import { DEFAULT_SETTINGS, UNMATCHED_FILL_NOTE, type DubSyncOutcome, type DubSyncPlan } from "@/lib/types";

import { createCursorStore } from "./cursorStore";
import { DoneBox, FailedBox, OutputBox, StagesBox, StretchBox } from "./DubInspector";

/** The engine's real answer for the five-minute fixture. */
const single = outcomeFixture as unknown as DubSyncOutcome;
const plan = single.plan as DubSyncPlan;

const noop = () => undefined;

/** A job finished with the fixture's outcome, patched. */
function doneJob(patch: Partial<DubSyncOutcome> = {}): DubQueueJob {
  const source = { ...single, ...patch };
  let state = dubQueueReducer(initialDubQueueState, { type: "queueStarted", jobs: [{ name: "film.mkv", dubName: "film.dub.ac3" }] });
  state = dubQueueReducer(state, {
    type: "jobDone",
    outcome: { job: 0, plan: source.plan, output: source.output, verification: source.verification, muxedPath: source.muxedPath ?? null, error: source.error ?? null },
  });
  return state.jobs[0];
}

const done = (patch: Partial<DubSyncOutcome> = {}, props: Partial<React.ComponentProps<typeof DoneBox>> = {}) =>
  renderToStaticMarkup(<DoneBox title="film" job={doneJob(patch)} onPlay={noop} onShow={noop} onReport={noop} {...props} />);

describe("DoneBox", () => {
  it("leads with the sync error and the plan's numbers, as the mockup does", () => {
    const html = done();
    expect(html).toContain(">Sync error<");
    // Under ten milliseconds the tenth is shown: 0.2 is not 0.
    expect(html).toContain('<div class="big">0.2<small>ms</small></div>');
    expect(html).toContain("Typical · 0.4 ms at worst");
    expect(html).toMatch(/<dt>Stretches<\/dt><dd>4 of dub<\/dd>/);
    expect(html).toMatch(/<dt>From original<\/dt><dd>21.5s · 2 cuts<\/dd>/);
    expect(html).toMatch(/<dt>Fill level<\/dt><dd>\+4.4 dB<\/dd>/);
    expect(html).toMatch(/<dt>Voices moved<\/dt><dd>None<\/dd>/);
    expect(html).toContain(">Play<");
    expect(html).toContain(">Show<");
    expect(html).toContain(">Report<");
  });

  it("names the written track and what it is", () => {
    const html = done();
    expect(html).toContain("dub.dubsynced.ac3");
    expect(html).toContain("5.1 · 48 kHz · 5m 00.0s");
  });

  it("reports the check of the written file against the video in full", () => {
    const html = done();
    expect(html).toContain("Checked against the video");
    expect(html).toContain("Measured at 10 spots");
    expect(html).toContain("A sweep of 23 short windows measured 10, and 90% of the runtime sits within 45 ms");
    expect(html).not.toContain("would be audible");
  });

  it("lists every piece of the plan on the video's timeline, offsets to the millisecond", () => {
    const html = done();
    expect(html.match(/class="dub-piece/g)).toHaveLength(plan.segments.length);
    expect(html).toContain("dub is cut here");
    expect(html).toMatch(/0:02:59\.99\d – 0:03:02\.49\d/);
    expect(html).toContain("-14.194s");
    expect(html).toContain("+0.806s");
    // Pieces pick a stretch only while the cuts can be edited.
    expect(html).toMatch(/<button[^>]*class="dub-piece"[^>]*disabled=""/);
    expect(done({}, { onPiece: noop, selected: 3 })).toMatch(/class="dub-piece on"/);
  });

  it("flags a stretch the check found audibly out", () => {
    const html = done({ verification: { ...single.verification!, worstMs: 400, stretches: [{ startS: 1285, endS: 1311, residualMs: 400, windows: 3 }] } });
    expect(html).toContain("0:21:25.000 – 0:21:51.000");
    expect(html).toContain("would be audible");
    expect(html).toContain('data-tone="bad"');
  });

  it("says how much of the runtime is out when half the sweep is", () => {
    const html = done({ verification: { ...single.verification!, sweepMeasured: 181, sweepWithinAudible: 92, stretches: [] } });
    expect(html).toContain("51% of the runtime sits within 45 ms");
  });

  it("tells a fill that replaced unmatched dub from a real cut", () => {
    const replaced = { ...plan, segments: plan.segments.map((s, i) => (s.kind === "fill" && i === 2 ? { ...s, note: UNMATCHED_FILL_NOTE } : s)) };
    const html = done({ plan: replaced });
    expect(html).toContain(">Replaced<");
    expect(html).toContain("did not correlate — replaced; check this spot");
    expect(html).toContain("dub is cut here");
  });

  it("shows a note as information, not as something to check", () => {
    const kept = "kept the dub across 0:01:40.000 - 0:04:40.000: it did not correlate there, but the offset is the same either side and the dub is not silent";
    const html = done({ plan: { ...plan, warnings: [], notes: [kept] } });
    expect(html).toContain(kept);
    expect(html).toContain('data-tone="info"');
    expect(html).not.toContain('data-tone="warn"');
    expect(done({ plan: { ...plan, warnings: ["the dub drifts after 1:10:00"] } })).toContain('data-tone="warn"');
  });

  it("lists scenes where the voice check moved the dub's voices", () => {
    const voicePieces = [
      { dubStartS: 12.5, dubEndS: 18.2, levelS: 15, shiftS: 0.24, joinEnd: false, note: "Voices moved 240 ms later from 0:00:12.5 to 0:00:18.2 to match the lips.", videoStartS: 12.5, videoEndS: 18.2 },
      { dubStartS: 120, dubEndS: 126.4, levelS: 123, shiftS: -0.1, joinEnd: true, note: "Voices moved 100 ms earlier from 0:02:00.0 to 0:02:06.4 to match the lips.", videoStartS: 120, videoEndS: 126.4 },
    ];
    const html = done({ plan: { ...plan, voicePieces } });
    expect(html).toMatch(/<dt>Voices moved<\/dt><dd>2 scenes<\/dd>/);
    expect(html).toContain("Voices moved (2)");
    expect(html).toContain("Voices moved 240 ms later from 0:00:12.5 to 0:00:18.2 to match the lips.");
    expect(done()).not.toContain("Voices moved (");
  });

  it("says why edits cannot be written, and offers the engine's plan back", () => {
    const html = done({}, { problems: ["The first piece does not start at 0."], onBackToEngine: noop });
    expect(html).toContain('role="alert"');
    expect(html).toContain("The first piece does not start at 0.");
    expect(html).toContain("Back to the engine&#x27;s plan");
  });
});

describe("FailedBox", () => {
  it("says the pair was not synced, why, and offers another try", () => {
    const html = renderToStaticMarkup(
      <FailedBox title="Goblin.S01E04" reason="The dub ends 38 minutes before the video." onRetry={noop} onChooseDub={noop} />,
    );
    expect(html).toContain(">Goblin.S01E04<");
    expect(html).toContain('<div class="big bad">Not synced</div>');
    expect(html).toContain("The dub ends 38 minutes before the video.");
    expect(html).toMatch(/<button[^>]*class="btn"[^>]*>Retry<\/button>/);
    expect(html).toMatch(/<button[^>]*class="btn"[^>]*>Choose another dub<\/button>/);
  });

  it("says a stopped pair stopped, without the failure colour", () => {
    const html = renderToStaticMarkup(<FailedBox title="x" reason="Stopped before it finished." stopped />);
    expect(html).toContain('<div class="big">Stopped</div>');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Retry/);
  });
});

describe("OutputBox", () => {
  it("offers how the track is written, as the mockup lays it out", () => {
    const html = renderToStaticMarkup(<OutputBox settings={{ ...DEFAULT_SETTINGS, dubMux: true, dubLanguage: "hin" }} onChange={noop} sameAs="AC3 5.1" />);
    expect(html).toContain(">Output<");
    expect(html).toContain(">Write the track as<");
    expect(html).toContain(">Same as the dub (AC3 5.1)<");
    expect(html).toContain(">The dub was mastered at<");
    expect(html).toContain(">Detect from the audio<");
    expect(html).toContain(">25 fps<");
    expect(html).toContain('aria-label="Add to a copy of the video"');
    expect(html).toContain('aria-checked="true"');
    expect(html).toMatch(/<input[^>]*aria-label="Language of the added track"[^>]*value="hin"/);
  });

  it("leaves the language alone while no copy of the video is made", () => {
    const html = renderToStaticMarkup(<OutputBox settings={DEFAULT_SETTINGS} onChange={noop} />);
    expect(html).toMatch(/<input[^>]*aria-label="Language of the added track"[^>]*disabled=""/);
    expect(html).not.toContain('class="hr"');
  });

  it("offers a file's audio streams only when it carries several", () => {
    const html = renderToStaticMarkup(
      <OutputBox
        settings={DEFAULT_SETTINGS}
        onChange={noop}
        tracks={[{ label: "Audio of the movie", value: 1, onChange: noop, options: [{ value: 0, label: "1 · English · TrueHD 7.1" }, { value: 1, label: "2 · Commentary · AC3 2.0" }] }]}
      />,
    );
    expect(html).toContain('class="hr"');
    expect(html).toContain(">Audio of the movie<");
    expect(html).toContain("2 · Commentary · AC3 2.0");
  });
});

describe("StagesBox", () => {
  it("ticks the stages done, spins the current one with its percent", () => {
    let state = dubQueueReducer(initialDubQueueState, { type: "queueStarted", jobs: [{ name: "film.mkv", dubName: "film.ac3" }] });
    state = dubQueueReducer(state, { type: "jobStart", job: 0 });
    state = dubQueueReducer(state, { type: "jobProgress", job: 0, percent: 42, stage: "placing the cuts" });
    const html = renderToStaticMarkup(<StagesBox title="Skyline Heist (2023)" job={state.jobs[0]} mux={false} />);
    expect(html).toContain(">Skyline Heist (2023)<");
    expect(html.match(/class="row"/g)).toHaveLength(9);
    // Four done before it: reading, frame rate, finding, measuring.
    expect(html.match(/<svg[^>]*class="[^"]*\bok\b/g)).toHaveLength(4);
    expect(html).toContain('class="ring"');
    expect(html).toMatch(/Placing the cuts<\/span><span class="t3 num">42%<\/span>/);
    expect(renderToStaticMarkup(<StagesBox title="x" job={state.jobs[0]} mux />).match(/class="row"/g)).toHaveLength(10);
  });

  it("shows a waiting job's stages all still to do", () => {
    const state = dubQueueReducer(initialDubQueueState, { type: "queueStarted", jobs: [{ name: "film.mkv", dubName: "film.ac3" }] });
    const html = renderToStaticMarkup(<StagesBox title="x" job={state.jobs[0]} mux={false} />);
    expect(html).not.toContain('class="ring"');
    expect(html).not.toContain("%<");
  });
});

describe("StretchBox", () => {
  const stretch = (index: number, props: Partial<React.ComponentProps<typeof StretchBox>> = {}) =>
    renderToStaticMarkup(
      <StretchBox
        plan={plan}
        index={index}
        frameS={1 / 24}
        onNudge={noop}
        onSetOffset={noop}
        onSplit={noop}
        onUseOriginal={noop}
        onUseDub={noop}
        {...props}
      />,
    );

  it("shows a dub stretch with its offset to nudge or type", () => {
    const html = stretch(3);
    expect(html).toContain(">Dub stretch 2 of 4<");
    expect(html).toMatch(/<dt>Start<\/dt><dd>0:01:15.010<\/dd>/);
    expect(html).toMatch(/<dt>End<\/dt><dd>0:02:59.996<\/dd>/);
    expect(html).toMatch(/<dt>Length<\/dt><dd>1m 45.0s<\/dd>/);
    expect(html).toMatch(/<dt>Taken from<\/dt><dd>Dub 0:01:00.815<\/dd>/);
    expect(html).toMatch(/<dt>Match<\/dt><dd>0.07<\/dd>/);
    expect(html).toMatch(/<input[^>]*aria-label="Offset in seconds"[^>]*value="−14.194"/);
    for (const title of ["−1 frame (Shift: −1 ms)", "−10 ms (Shift: −1 ms)", "+10 ms (Shift: +1 ms)", "+1 frame (Shift: +1 ms)"]) {
      expect(html).toContain(`title="${title}"`);
    }
    expect(html).toContain(">Use the original<");
    expect(html).not.toContain(">Use the dub<");
    // Split here needs the playhead inside the stretch.
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>.*?Split here/);
    expect(stretch(3, { canSplit: true })).not.toMatch(/<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Split here/);
    expect(html).not.toContain("Back to the engine");
  });

  it("splits where the playhead is, only inside the stretch", () => {
    const splitDisabled = /<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Split here/;
    expect(stretch(3, { cursor: createCursorStore(100) })).not.toMatch(splitDisabled);
    expect(stretch(3, { cursor: createCursorStore(10) })).toMatch(splitDisabled);
    expect(stretch(3, { cursor: createCursorStore(null) })).toMatch(splitDisabled);
  });

  it("shows a stretch of the original with why it plays", () => {
    const html = stretch(2);
    expect(html).toContain(">Original stretch 2 of 4<");
    expect(html).toMatch(/<dt>Why<\/dt><dd>dub is cut here<\/dd>/);
    expect(html).toContain(">Use the dub<");
    expect(html).not.toContain("Offset in seconds");
  });

  it("says why the edited plan cannot be written, and offers the engine's back", () => {
    const html = stretch(3, { problems: ["Piece 4 reads the dub past its end.", "Piece 5 is shorter than 0.1s."], onBackToEngine: noop });
    expect(html).toContain('role="alert"');
    expect(html).toContain("Piece 4 reads the dub past its end. (1 more)");
    expect(html).toContain("Back to the engine&#x27;s plan");
  });
});
