import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DubPreviewPlayer } from "./DubPreviewPlayer";
import { MIX_GAINS } from "@/lib/previewMix";
import type { ExcerptSources } from "@/lib/useExcerpt";

const EMPTY_SOURCES: ExcerptSources = {
  startS: null,
  endS: null,
  pictureUrl: null,
  dub: null,
  original: null,
  loading: null,
  error: null,
};

// The graph is built in an effect, which SSR never runs, so a cast shell
// is enough for the markup.
const context = {} as unknown as AudioContext;

function render(props: Partial<React.ComponentProps<typeof DubPreviewPlayer>> = {}): string {
  return renderToStaticMarkup(
    <DubPreviewPlayer
      frameS={1 / 24}
      cursorS={100}
      onCursor={() => undefined}
      withPicture={false}
      sources={EMPTY_SOURCES}
      onNeedWindow={() => undefined}
      onClose={() => undefined}
      audioContext={context}
      mix="dub"
      onMix={() => undefined}
      {...props}
    />,
  );
}

describe("DubPreviewPlayer", () => {
  it("shows the whole transport, closed by default to just this row", () => {
    const html = render();
    expect(html).toContain('aria-label="Preview player"');
    expect(html).toContain('aria-label="Play"');
    expect(html).toContain('aria-label="One frame back"');
    expect(html).toContain('aria-label="One frame forward"');
    expect(html).toContain('aria-label="Loop four seconds"');
    expect(html).toContain("0:00:00:00 · frame 1");
    expect(html).toContain('aria-label="What plays"');
    expect(html).toContain(">Dub<");
    expect(html).toContain(">Original<");
    expect(html).toContain(">Both<");
    expect(html).toContain('aria-label="Volume"');
    expect(html).toContain('aria-label="Close the player"');
  });

  it("has no picture in sample mode", () => {
    expect(render({ withPicture: false })).not.toContain("<video");
  });

  it("shows the picture when one is loaded, and still no picture in sample mode", () => {
    const loaded: ExcerptSources = { ...EMPTY_SOURCES, pictureUrl: "blob:frame" };
    const html = render({ withPicture: true, sources: loaded });
    expect(html).toContain("<video");
    expect(html).toContain("blob:frame");
  });

  it("says what is loading and what failed", () => {
    expect(render({ sources: { ...EMPTY_SOURCES, loading: "audio" } })).toContain("Rendering the sound…");
    expect(render({ sources: { ...EMPTY_SOURCES, loading: "picture" } })).toContain("Loading the picture…");
    expect(render({ sources: { ...EMPTY_SOURCES, error: "The picture could not be rendered. Playing the sound only." } })).toContain(
      "Playing the sound only.",
    );
  });
});

describe("MIX_GAINS", () => {
  it("plays the dub alone by default, both ears", () => {
    expect(MIX_GAINS.dub).toEqual({ dub: 1, original: 0, dubPan: 0, originalPan: 0 });
  });

  it("plays the original alone", () => {
    expect(MIX_GAINS.original).toEqual({ dub: 0, original: 1, dubPan: 0, originalPan: 0 });
  });

  it("splits Both: original in the left ear, dub in the right", () => {
    expect(MIX_GAINS.both).toEqual({ dub: 1, original: 1, dubPan: 1, originalPan: -1 });
  });
});
