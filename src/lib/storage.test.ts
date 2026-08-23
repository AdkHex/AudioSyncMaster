import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createHistoryEntry,
  formatHistoryDate,
  loadHistory,
  loadSettings,
  parseHistoryDate,
  saveHistory,
  saveSettings,
} from "./storage";
import { DEFAULT_SETTINGS, type SyncResult } from "./types";

/** Minimal localStorage with an optional byte budget, to exercise quota paths. */
class MemoryStorage implements Storage {
  private data = new Map<string, string>();
  constructor(private budget = Infinity) {}

  get length() {
    return this.data.size;
  }
  clear() {
    this.data.clear();
  }
  key(index: number) {
    return Array.from(this.data.keys())[index] ?? null;
  }
  getItem(key: string) {
    return this.data.get(key) ?? null;
  }
  removeItem(key: string) {
    this.data.delete(key);
  }
  setItem(key: string, value: string) {
    let total = value.length;
    for (const [k, v] of this.data) if (k !== key) total += v.length;
    if (total > this.budget) {
      const error = new Error("QuotaExceededError");
      error.name = "QuotaExceededError";
      throw error;
    }
    this.data.set(key, value);
  }
}

function install(budget = Infinity) {
  const storage = new MemoryStorage(budget);
  vi.stubGlobal("localStorage", storage);
  return storage;
}

function result(name: string): SyncResult {
  return {
    videoFile: name,
    audioFile: "dub.ac3",
    delayMs: 120,
    delayAtStartMs: 120,
    confidence: 0.9,
    driftMsPerS: null,
    totalDriftMs: null,
    hasSignificantDrift: false,
    startDelayMs: 120,
    endDelayMs: 120,
    windowsUsed: 6,
    windowsTotal: 6,
    error: null,
    elapsedMs: 1000,
  };
}

beforeEach(() => {
  install();
});

describe("history", () => {
  it("round-trips an entry", () => {
    const entry = createHistoryEntry("movie", [result("a.mkv")], null);
    saveHistory([entry]);
    const loaded = loadHistory();
    expect(loaded).toHaveLength(1);
    expect(loaded[0].results[0].videoFile).toBe("a.mkv");
  });

  it("stores the date as a parseable ISO string", () => {
    const entry = createHistoryEntry("movie", [], null);
    expect(typeof entry.date).toBe("string");
    expect(parseHistoryDate(entry.date)).toBeInstanceOf(Date);
  });

  it("caps the number of retained entries", () => {
    const entries = Array.from({ length: 60 }, () =>
      createHistoryEntry("movie", [result("a.mkv")], null),
    );
    expect(saveHistory(entries).length).toBeLessThanOrEqual(25);
  });

  it("trims oversized result sets rather than failing to save", () => {
    const many = Array.from({ length: 900 }, (_, i) => result(`ep${i}.mkv`));
    const saved = saveHistory([createHistoryEntry("series", many, null)]);
    expect(saved).toHaveLength(1);
    expect(saved[0].results.length).toBeLessThanOrEqual(200);
  });

  it("sheds entries instead of throwing when the quota is exceeded", () => {
    install(4000); // tiny budget
    const entries = Array.from({ length: 20 }, (_, i) =>
      createHistoryEntry("series", [result(`long-name-${i}.mkv`)], null),
    );
    expect(() => saveHistory(entries)).not.toThrow();
    expect(loadHistory().length).toBeLessThan(20);
  });

  it("returns an empty list when stored data is corrupt", () => {
    localStorage.setItem("audiosync.history.v2", "{not json");
    expect(loadHistory()).toEqual([]);
  });

  it("discards malformed entries", () => {
    localStorage.setItem(
      "audiosync.history.v2",
      JSON.stringify([{ id: "ok", results: [] }, { nonsense: true }, null]),
    );
    expect(loadHistory()).toHaveLength(1);
  });

  it("renders an unparseable date without crashing", () => {
    expect(formatHistoryDate("not-a-date")).toBe("Unknown date");
    expect(parseHistoryDate("not-a-date")).toBeNull();
  });
});

describe("settings", () => {
  it("returns defaults when nothing is stored", () => {
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
  });

  it("round-trips saved settings", () => {
    saveSettings({ ...DEFAULT_SETTINGS, windowSeconds: 90, matchPattern: "E(\\d+)" });
    const loaded = loadSettings();
    expect(loaded.windowSeconds).toBe(90);
    expect(loaded.matchPattern).toBe("E(\\d+)");
  });

  it("clamps out-of-range values from stale payloads", () => {
    localStorage.setItem(
      "audiosync.settings.v2",
      JSON.stringify({ windowSeconds: 99999, windowCount: -5, maxWorkers: 500 }),
    );
    const loaded = loadSettings();
    expect(loaded.windowSeconds).toBeLessThanOrEqual(600);
    expect(loaded.windowCount).toBeGreaterThanOrEqual(1);
    expect(loaded.maxWorkers).toBeLessThanOrEqual(16);
  });

  it("ignores non-numeric values", () => {
    localStorage.setItem(
      "audiosync.settings.v2",
      JSON.stringify({ windowSeconds: "lots" }),
    );
    expect(loadSettings().windowSeconds).toBe(DEFAULT_SETTINGS.windowSeconds);
  });
});
