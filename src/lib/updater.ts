/** Auto-update against GitHub Releases.
 *
 *  The app checks for a newer signed release on launch. Every update is
 *  verified against the public key embedded in tauri.conf.json before it is
 *  applied, so a tampered or unsigned artifact is rejected rather than
 *  installed.
 *
 *  Checks are deliberately non-fatal: a user offline, behind a proxy, or on a
 *  network that blocks GitHub must still get a working app, so every failure
 *  path here degrades to "no update available". */

import { check, type DownloadEvent, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

import { isDesktop } from "./api";

const SKIPPED_KEY = "audiosync.update.skipped";
const LAST_CHECK_KEY = "audiosync.update.lastCheck";

/** Minimum gap between automatic checks, so restarting repeatedly does not
 *  hammer the endpoint. Manual checks ignore this. */
const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000; // 6 hours

export interface UpdateInfo {
  version: string;
  currentVersion: string;
  notes: string | null;
  date: string | null;
  /** Handle used to download and install. Held so the dialog can act on it. */
  handle: Update;
}

export interface DownloadProgress {
  downloaded: number;
  total: number | null;
  percent: number | null;
}

function readSkipped(): string | null {
  try {
    return localStorage.getItem(SKIPPED_KEY);
  } catch {
    return null;
  }
}

/** Remember that the user does not want to be asked about this version again. */
export function skipVersion(version: string): void {
  try {
    localStorage.setItem(SKIPPED_KEY, version);
  } catch {
    /* storage unavailable; the prompt simply reappears next launch */
  }
}

function markChecked(): void {
  try {
    localStorage.setItem(LAST_CHECK_KEY, String(Date.now()));
  } catch {
    /* not important enough to surface */
  }
}

function checkedRecently(): boolean {
  try {
    const raw = localStorage.getItem(LAST_CHECK_KEY);
    if (!raw) return false;
    const last = Number(raw);
    if (!Number.isFinite(last)) return false;
    return Date.now() - last < CHECK_INTERVAL_MS;
  } catch {
    return false;
  }
}

/**
 * Look for a newer release.
 *
 * @param manual - true when the user explicitly asked. Manual checks bypass
 *   both the throttle and any previously skipped version, because the user is
 *   asking right now and expects an answer.
 */
export async function checkForUpdate(manual = false): Promise<UpdateInfo | null> {
  if (!isDesktop()) return null;
  if (!manual && checkedRecently()) return null;

  try {
    const update = await check();
    markChecked();
    if (!update) return null;

    if (!manual && readSkipped() === update.version) return null;

    return {
      version: update.version,
      currentVersion: update.currentVersion,
      notes: update.body?.trim() || null,
      date: update.date ?? null,
      handle: update,
    };
  } catch (error) {
    // Offline, blocked, or a malformed manifest. Never block startup for this.
    console.warn("Update check failed:", error);
    return null;
  }
}

/**
 * Download and install an update, then restart into it.
 *
 * @param onProgress - called as bytes arrive, so the UI can show a real bar
 *   rather than an indeterminate spinner on a 70MB installer.
 */
export async function installUpdate(
  info: UpdateInfo,
  onProgress?: (progress: DownloadProgress) => void,
): Promise<void> {
  let downloaded = 0;
  let total: number | null = null;

  await info.handle.downloadAndInstall((event: DownloadEvent) => {
    switch (event.event) {
      case "Started":
        total = event.data.contentLength ?? null;
        downloaded = 0;
        onProgress?.({ downloaded, total, percent: total ? 0 : null });
        break;
      case "Progress":
        downloaded += event.data.chunkLength;
        onProgress?.({
          downloaded,
          total,
          percent: total ? Math.min(100, Math.round((downloaded / total) * 100)) : null,
        });
        break;
      case "Finished":
        onProgress?.({ downloaded, total, percent: 100 });
        break;
    }
  });

  // On Windows the installer replaces the running exe and exits the app itself,
  // so relaunch may never return. Both outcomes are correct.
  await relaunch();
}

export function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 MB";
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(1)} MB`;
}
