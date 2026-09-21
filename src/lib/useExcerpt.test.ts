import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ExcerptLoader, type ExcerptSources } from "./useExcerpt";
import type { DubPreviewRequest } from "./api";
import type { DubSyncPlan } from "./types";

const plan: DubSyncPlan = {
  videoPath: "/v.mkv",
  dubPath: "/d.mp4",
  videoTrack: 0,
  dubTrack: 0,
  speed: 1,
  videoFps: 24,
  dubRate: null,
  fillGainDb: 0,
  videoDurationS: 3600,
  dubDurationS: 3600,
  segments: [
    { kind: "dub", startS: 0, endS: 100, sourceStartS: 0, offsetS: 0, match: 1, note: "", uncertaintyS: 0 },
    { kind: "fill", startS: 100, endS: 3600, sourceStartS: 100, offsetS: null, match: null, note: "dub is cut here", uncertaintyS: 0 },
  ],
  warnings: [],
  error: null,
  filledS: 3500,
};

function editedPlan(): DubSyncPlan {
  return {
    ...plan,
    segments: plan.segments.map((s, i) =>
      i === 0 && s.kind === "dub" ? { ...s, offsetS: 0.01, sourceStartS: 0.01 } : s,
    ),
  };
}

/** Flush the promise chain a few turns, one per awaited step of a load. */
const flush = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};

function makeLoader(render: (request: DubPreviewRequest) => Promise<unknown>) {
  const events: ExcerptSources[] = [];
  const revoked: string[] = [];
  const loader = new ExcerptLoader({
    render: render as ExcerptLoader["render"],
    read: async () => new ArrayBuffer(8),
    decodeAudio: async () => ({ fake: true }) as unknown as AudioBuffer,
    makePictureUrl: () => `blob:fake`,
    revokeUrl: (url) => revoked.push(url),
    onChange: (sources) => events.push(sources),
  });
  const last = () => events[events.length - 1];
  return { loader, events, revoked, last };
}

/** A render that answers pictures and originals instantly and holds the
 *  audio back so each audio request can be resolved in order. */
function heldAudio() {
  const calls: DubPreviewRequest[] = [];
  const audio: { request: DubPreviewRequest; resolve: (excerpt: unknown) => void }[] = [];
  const render = (request: DubPreviewRequest): Promise<unknown> => {
    calls.push(request);
    if (request.what === "picture") {
      return Promise.resolve({ path: "/tmp/p.mp4", what: "picture", startS: 100.333, endS: 130.3 });
    }
    if (request.what === "original") {
      return Promise.resolve({ path: "/tmp/o.wav", what: "original", startS: 100.333, endS: 130.3 });
    }
    return new Promise((resolve) => audio.push({ request, resolve }));
  };
  return { calls, audio, render };
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("ExcerptLoader", () => {
  it("cuts the sound from the picture's returned start, not the one asked for", async () => {
    const { calls, audio, render } = heldAudio();
    const { loader, last } = makeLoader(render);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, true);
    await flush();

    expect(calls[0]).toMatchObject({ what: "picture", startS: 100, endS: 130 });
    // The picture began on the frame at 100.333; the sound must start there.
    expect(calls[1]).toMatchObject({ what: "audio", startS: 100.333, endS: 130.3 });
    expect(last().pictureUrl).toBe("blob:fake");
    expect(last().startS).toBe(100.333);
    expect(last().endS).toBe(130.3);
    expect(last().loading).toBe("audio");

    audio[0].resolve({ path: "/tmp/a.wav", what: "audio", startS: 100.333, endS: 130.3 });
    await flush();
    expect(last().dub).not.toBeNull();
    expect(last().loading).toBeNull();
    expect(last().error).toBeNull();
  });

  it("in sample mode loads the sound for the asked span without a picture", async () => {
    const { calls, render } = heldAudio();
    const { loader, last } = makeLoader(render);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, false);
    await flush();

    expect(calls.map((c) => c.what)).toEqual(["audio"]);
    expect(calls[0]).toMatchObject({ startS: 100, endS: 130 });
    expect(last().pictureUrl).toBeNull();
  });

  it("re-renders only the sound when the plan changes, for the same window, after the debounce", async () => {
    const { calls, render } = heldAudio();
    const { loader } = makeLoader(render);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, true);
    await flush();
    expect(calls).toHaveLength(2);

    loader.updatePlan(editedPlan());
    await vi.advanceTimersByTimeAsync(300);
    await flush();

    expect(calls).toHaveLength(3);
    expect(calls[2]).toMatchObject({ what: "audio", startS: 100.333, endS: 130.3 });
    // The picture is never re-rendered for an edit.
    expect(calls.filter((c) => c.what === "picture")).toHaveLength(1);
  });

  it("coalesces three quick edits into one sound re-render", async () => {
    const { calls, render } = heldAudio();
    const { loader } = makeLoader(render);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, true);
    await flush();
    const audioCalls = () => calls.filter((c) => c.what === "audio").length;
    expect(audioCalls()).toBe(1);

    loader.updatePlan(editedPlan());
    await vi.advanceTimersByTimeAsync(100);
    loader.updatePlan(editedPlan());
    await vi.advanceTimersByTimeAsync(100);
    loader.updatePlan(editedPlan());
    await vi.advanceTimersByTimeAsync(300);
    await flush();

    expect(audioCalls()).toBe(2);
  });

  it("ignores a stale sound reply when a newer one has won", async () => {
    const { audio, render } = heldAudio();
    const { loader, last } = makeLoader(render);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, true);
    await flush();
    const first = audio[0];

    // An edit re-renders the sound before the first render resolves.
    loader.updatePlan(editedPlan());
    await vi.advanceTimersByTimeAsync(300);
    await flush();
    const second = audio[1];

    second.resolve({ path: "/tmp/new.wav", what: "audio", startS: 100.333, endS: 130.3 });
    await flush();
    expect(last().dub).not.toBeNull();

    // The older reply lands afterwards and must not win.
    first.resolve({ path: "/tmp/old.wav", what: "audio", startS: 100.333, endS: 130.3 });
    await flush();
    expect(last().dub).not.toBeNull();
    expect(last().error).toBeNull();
  });

  it("fetches the original once, only when the mix wants it, and per window", async () => {
    const { calls, render } = heldAudio();
    const { loader, last } = makeLoader(render);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, true);
    await flush();
    expect(calls.some((c) => c.what === "original")).toBe(false);

    loader.updateWantOriginal(true);
    await flush();
    const originalCalls = calls.filter((c) => c.what === "original");
    expect(originalCalls).toHaveLength(1);
    // Asked from the picture-aligned start, like the sound.
    expect(originalCalls[0]).toMatchObject({ startS: 100.333, endS: 130.3 });
    expect(last().original).not.toBeNull();

    // Switching the mix off and back on does not fetch it again.
    loader.updateWantOriginal(false);
    loader.updateWantOriginal(true);
    await flush();
    expect(calls.filter((c) => c.what === "original")).toHaveLength(1);

    // A new window forgets it, and an edit does not.
    loader.updateWindow(200, 230, true);
    await flush();
    expect(calls.filter((c) => c.what === "original")).toHaveLength(1);
    loader.updateWantOriginal(true);
    await flush();
    expect(calls.filter((c) => c.what === "original")).toHaveLength(2);
  });

  it("falls back to sound-only playback with a notice when the picture fails", async () => {
    const { calls, render } = heldAudio();
    const failing = (request: DubPreviewRequest): Promise<unknown> =>
      request.what === "picture" ? Promise.resolve(null) : render(request);
    const { loader, last } = makeLoader(failing as never);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, true);
    await flush();

    expect(last().pictureUrl).toBeNull();
    expect(last().error).toContain("Playing the sound only");
    // The sound still loads, over the span asked for.
    expect(calls.find((c) => c.what === "audio")).toMatchObject({ startS: 100, endS: 130 });
  });

  it("revokes the picture's URL when the window moves on or the loader dies", async () => {
    const { render } = heldAudio();
    const { loader, revoked } = makeLoader(render);

    loader.updatePlan(plan);
    loader.updateWindow(100, 130, true);
    await flush();
    expect(revoked).toHaveLength(0);

    loader.updateWindow(200, 230, true);
    await flush();
    expect(revoked).toEqual(["blob:fake"]);

    loader.dispose();
    expect(revoked).toEqual(["blob:fake", "blob:fake"]);
  });
});
