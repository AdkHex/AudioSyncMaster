import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { MIX_GAINS } from "@/lib/previewMix";
import type { ExcerptSources } from "@/lib/useExcerpt";

import { createCursorStore } from "./cursorStore";
import { PreviewPlayer } from "./DubPreview";

const EMPTY_SOURCES: ExcerptSources = {
  startS: null,
  endS: null,
  pictureUrl: null,
  dub: null,
  original: null,
  loading: null,
  error: null,
};

// The audio graph is built in an effect, which SSR never runs, so a cast
// shell is enough for the markup.
const context = {} as unknown as AudioContext;

function render(props: Partial<React.ComponentProps<typeof PreviewPlayer>> = {}): string {
  return renderToStaticMarkup(
    <PreviewPlayer
      frameS={1 / 24}
      cursor={createCursorStore(null)}
      sources={EMPTY_SOURCES}
      onNeedWindow={() => undefined}
      audioContext={context}
      mix="dub"
      onMix={() => undefined}
      active
      {...props}
    />,
  );
}

describe("PreviewPlayer", () => {
  it("shows the mockup's preview: the picture, the transport, what to listen to", () => {
    const html = render();
    expect(html).toContain('aria-label="Preview player"');
    expect(html).toContain(">Preview<");
    expect(html).toContain('class="video dub-video"');
    expect(html).toContain('aria-label="Play (Space)"');
    expect(html).toContain('aria-label="Back one frame ([)"');
    expect(html).toContain('aria-label="Forward one frame (])"');
    expect(html).toContain('aria-label="Loop 4 s"');
    expect(html).toContain('aria-pressed="false"');
    expect(html).toContain(">Listen to<");
    expect(html).toContain('aria-label="What plays"');
    expect(html).toContain(">Dub<");
    expect(html).toContain(">Original<");
    expect(html).toContain(">Both<");
    expect(html).toContain('aria-label="Volume"');
    // Closed from the Timeline's Play toggle; the mockup's header holds only the timecode.
    expect(html).not.toContain("Close the player");
    expect(html).not.toContain("Open in your player");
    expect(render({ onOpenExternal: () => undefined })).toContain('aria-label="Open in your player"');
  });

  it("shows the playhead as a timecode, frame and all", () => {
    expect(render()).toContain(">0:00:00:00<");
    const html = render({ cursor: createCursorStore(4724.8) });
    expect(html).toContain(">1:18:44:19<");
    expect(html).toContain('title="Frame 113396"');
  });

  it("marks what is playing", () => {
    expect(render({ mix: "both" })).toMatch(/aria-checked="true"[^>]*class="on"[^>]*>Both</);
  });

  it("has no picture until one is loaded", () => {
    expect(render()).not.toContain("<video");
    const html = render({ sources: { ...EMPTY_SOURCES, pictureUrl: "blob:frame" } });
    expect(html).toContain("<video");
    expect(html).toContain("blob:frame");
  });

  it("says what is loading and what failed", () => {
    expect(render({ sources: { ...EMPTY_SOURCES, loading: "audio" } })).toContain("Rendering the sound…");
    expect(render({ sources: { ...EMPTY_SOURCES, loading: "picture" } })).toContain("Loading the picture…");
    expect(render({ sources: { ...EMPTY_SOURCES, loading: "original" } })).toContain("Loading the original…");
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
