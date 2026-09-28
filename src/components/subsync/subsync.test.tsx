import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { CueTable } from "./CueTable";
import { RunView } from "./RunView";
import { SubsyncWorkspace } from "./SubsyncWorkspace";
import { initialRunState, runReducer } from "@/lib/subsync/reducer";
import type { CueData } from "@/lib/subsync/types";

const noop = () => undefined;

describe("CueTable", () => {
  it("draws only the rows in view of a long file", () => {
    const cues: CueData[] = Array.from({ length: 5000 }, (_, i) => ({ start: i * 2, end: i * 2 + 1.5, text: `Line ${i}` }));
    const html = renderToStaticMarkup(<CueTable cues={cues} height={340} />);
    const rows = html.match(/role="row"/g) ?? [];
    expect(rows.length).toBeLessThan(30);
    expect(html).toContain("5,000 cues");
    expect(html).toContain("Line 0");
    expect(html).not.toContain("Line 4999");
  });

  it("highlights cues below 0.6 confidence and offers to show only them", () => {
    const cues: CueData[] = [
      { start: 0, end: 2, text: "sure", confidence: 0.95 },
      { start: 3, end: 5, text: "unsure", confidence: 0.4 },
    ];
    const html = renderToStaticMarkup(<CueTable cues={cues} />);
    expect(html.match(/bg-warning\/10/g)).toHaveLength(1);
    expect(html).toContain("Only the 1 to review");
  });
});

describe("RunView", () => {
  it("shows a finished job's summary, report and outputs", () => {
    let run = runReducer(initialRunState, { type: "started", task: "sync", at: 0, jobs: [{ id: "a", label: "Movie.en.srt" }] });
    run = runReducer(run, {
      type: "finished",
      outcomes: [
        {
          job: 0,
          id: "a",
          task: "sync",
          outputs: [{ path: "/out/Movie.synced.en.srt", kind: "subtitle", format: "srt", language: "en", label: null }],
          report: { offsetMs: 3250, method: "audio" },
          summary: "Shifted by +3.250 s",
          warnings: ["Two scenes were cut"],
          preview: null,
        },
      ],
    });
    const html = renderToStaticMarkup(
      <RunView run={run} actions={{ onReveal: noop, onOpen: noop, onUseAs: noop }} onOpenConsole={noop} />,
    );
    expect(html).toContain("Shifted by +3.250 s");
    expect(html).toContain("+3.250 s");
    expect(html).toContain("Two scenes were cut");
    expect(html).toContain("Movie.synced.en.srt");
    expect(html).toContain("Use as input for…");
  });
});

describe("SubsyncWorkspace", () => {
  it("explains that the browser build cannot run it", () => {
    const html = renderToStaticMarkup(<SubsyncWorkspace onLog={noop} onBusyChange={noop} onOpenConsole={noop} />);
    expect(html).toContain("Running in a browser");
  });
});
