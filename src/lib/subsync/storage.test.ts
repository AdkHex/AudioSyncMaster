import { beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_PREFS, loadPrefs, savePrefs } from "./storage";
import { DEFAULT_OPTIONS } from "./types";

class MemoryStorage {
  data = new Map<string, string>();
  getItem(key: string) {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.data.set(key, value);
  }
  removeItem(key: string) {
    this.data.delete(key);
  }
}

let storage: MemoryStorage;
beforeEach(() => {
  storage = new MemoryStorage();
  vi.stubGlobal("localStorage", storage);
});

describe("subsync prefs", () => {
  it("returns the defaults when nothing is stored or the payload is corrupt", () => {
    expect(loadPrefs()).toEqual(DEFAULT_PREFS);
    storage.setItem("audiosync.subsync.v1", "{not json");
    expect(loadPrefs()).toEqual(DEFAULT_PREFS);
  });

  it("round-trips options, output and the last task", () => {
    const prefs = {
      ...DEFAULT_PREFS,
      task: "translate" as const,
      options: { ...DEFAULT_OPTIONS, translate: { ...DEFAULT_OPTIONS.translate, target: "fr", glossary: "源氏 = Genji" } },
      output: { ...DEFAULT_PREFS.output, sync: { ...DEFAULT_PREFS.output.sync, dir: "/out", mux: true } },
    };
    savePrefs(prefs);
    expect(loadPrefs()).toEqual(prefs);
  });

  it("fills in fields a stored payload predates, and drops wrongly typed ones", () => {
    storage.setItem(
      "audiosync.subsync.v1",
      JSON.stringify({
        task: "nonsense",
        options: { sync: { maxOffsetS: 120, detectFramerate: "yes", engine: null }, fps: { from: "auto" } },
      }),
    );
    const prefs = loadPrefs();
    expect(prefs.task).toBe("sync");
    expect(prefs.options.sync.maxOffsetS).toBe(120);
    expect(prefs.options.sync.detectFramerate).toBe(true);
    expect(prefs.options.sync.engine).toBe("audio");
    expect(prefs.options.sync.allowSplits).toBe(DEFAULT_OPTIONS.sync.allowSplits);
    expect(prefs.options.fps.from).toBe("auto");
    expect(prefs.options.ocr).toEqual(DEFAULT_OPTIONS.ocr);
  });

  it("keeps union-typed values: overrides, track lists, auto rates", () => {
    storage.setItem(
      "audiosync.subsync.v1",
      JSON.stringify({ options: { style: { cps: 17 }, extract: { tracks: [0, 2] }, tonemap: { sourcePeak: 4000 } } }),
    );
    const prefs = loadPrefs();
    expect(prefs.options.style.cps).toBe(17);
    expect(prefs.options.extract.tracks).toEqual([0, 2]);
    expect(prefs.options.tonemap.sourcePeak).toBe(4000);
  });

  it("never persists a reference file", () => {
    savePrefs({
      ...DEFAULT_PREFS,
      options: { ...DEFAULT_OPTIONS, sync: { ...DEFAULT_OPTIONS.sync, engine: "subtitle", reference: { path: "/ref.srt" } } },
    });
    expect(storage.getItem("audiosync.subsync.v1")).not.toContain("/ref.srt");
    expect(loadPrefs().options.sync).toMatchObject({ engine: "subtitle", reference: null });
  });
});
