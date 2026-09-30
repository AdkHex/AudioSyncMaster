/** Timing and event plumbing for the demo engine: a cancellable run, sleeps
 *  that a cancel cuts short, and the mocked event bus. Dev only. */

import { emit } from "@tauri-apps/api/event";

/** `?speed=0.25` runs every streamed command four times faster (for tests
 *  and screenshots); `?speed=2` half as fast. */
let timeScale = 1;
export function setTimeScale(scale: number): void {
  timeScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
}

/** Emit through the mocked event system. The mock runs listeners
 *  synchronously inside `invoke`, so order is the order of the calls. */
export function send(event: string, payload?: unknown): void {
  void emit(event, payload);
}

export function log(message: string): void {
  send("sync-log", message);
}

/** A plain scaled delay, for commands that cannot be cancelled. */
export const nap = (ms: number): Promise<void> => new Promise((resolve) => window.setTimeout(resolve, ms * timeScale));

const runs = new Set<Run>();

/** One in-flight streamed command. `cancel_sync` cancels every live run. */
export class Run {
  cancelled = false;
  private wakers = new Set<() => void>();

  constructor() {
    runs.add(this);
  }

  /** Resolves true after `ms`, or false at once when the run was cancelled. */
  sleep(ms: number): Promise<boolean> {
    if (this.cancelled) return Promise.resolve(false);
    return new Promise((resolve) => {
      const wake = () => {
        window.clearTimeout(timer);
        this.wakers.delete(wake);
        resolve(false);
      };
      const timer = window.setTimeout(() => {
        this.wakers.delete(wake);
        resolve(!this.cancelled);
      }, ms * timeScale);
      this.wakers.add(wake);
    });
  }

  cancel(): void {
    this.cancelled = true;
    [...this.wakers].forEach((wake) => wake());
  }

  end(): void {
    runs.delete(this);
  }
}

export function beginRun(): Run {
  return new Run();
}

export function cancelAll(): number {
  const live = [...runs];
  live.forEach((run) => run.cancel());
  return live.length;
}

/** Run `count` tasks with at most `workers` at a time, in index order. */
export async function pool(count: number, workers: number, task: (index: number) => Promise<void>): Promise<void> {
  let next = 0;
  const lane = async () => {
    while (next < count) {
      const index = next++;
      await task(index);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(workers, count)) }, lane));
}

export const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
