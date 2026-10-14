/** The remote switch and the access password, as the desktop shell reports
 *  them (docs/REMOTE-STATUS.md). The shell decides and enforces the lock --
 *  every command is refused while it is on -- and this side only shows it. */

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import { isDesktop } from "@/lib/api";
import type { SyncMode } from "@/lib/types";

export type LockReason = "maintenance" | "update" | "password" | "connect";

/** Mirrors `ServiceLock` in src-tauri/src/remote_status.rs. */
export interface ServiceLock {
  locked: boolean;
  reason: LockReason | null;
  title: string | null;
  message: string | null;
  updateUrl: string | null;
  minVersion: string | null;
  appVersion: string;
  /** The switch is on in this build. */
  enabled: boolean;
  /** Unix ms of the last successful check. */
  checkedAt: number | null;
  lastError: string | null;
  /** Seconds the next password attempt must still wait. */
  retryIn: number | null;
  /** The work this lock stopped, by kind. */
  stopped: string[];
  /** On the event that lifts a lock: the work it had stopped, to resume. */
  resume: string[];
}

export const SERVICE_LOCK_EVENT = "service-lock-changed";

/** The kind of run each page's Start makes, as the shell reports it in
 *  `stopped` / `resume`: a page whose run a lock stopped starts it again
 *  when the lock lifts. */
export const RESUME_KIND: Record<SyncMode, string> = {
  movie: "sync",
  series: "sync",
  compare: "sync",
  dubsync: "dubsync",
  subsync: "subsync",
};

/** What the app is when there is no shell (the browser demo) or no answer. */
export const OPEN: ServiceLock = {
  locked: false,
  reason: null,
  title: null,
  message: null,
  updateUrl: null,
  minVersion: null,
  appVersion: "",
  enabled: false,
  checkedAt: null,
  lastError: null,
  retryIn: null,
  stopped: [],
  resume: [],
};

export async function serviceLock(): Promise<ServiceLock> {
  if (!isDesktop()) return OPEN;
  return (await invoke<ServiceLock | null>("service_lock")) ?? OPEN;
}

/** Fetch the status document now ("Check again", "Try again"). */
export async function checkServiceStatus(): Promise<ServiceLock> {
  if (!isDesktop()) return OPEN;
  return (await invoke<ServiceLock | null>("check_service_status")) ?? OPEN;
}

/** Rejects with "Wrong password", or with how long to wait after too many. */
export async function unlockAccess(password: string): Promise<ServiceLock> {
  return (await invoke<ServiceLock | null>("unlock_access", { password })) ?? OPEN;
}

/** Windows: the folder of an older copy installed for all users. */
export async function legacyInstall(): Promise<string | null> {
  if (!isDesktop()) return null;
  return (await invoke<string | null>("legacy_install")) ?? null;
}

export async function subscribeToServiceLock(listener: (lock: ServiceLock) => void): Promise<UnlistenFn> {
  if (!isDesktop()) return () => undefined;
  return listen<ServiceLock>(SERVICE_LOCK_EVENT, (event) => listener(event.payload));
}

export async function openExternal(url: string): Promise<void> {
  if (isDesktop()) await invoke("plugin:opener|open_url", { url });
  else window.open(url, "_blank", "noopener");
}
