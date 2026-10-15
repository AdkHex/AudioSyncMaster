/** The full-window lock screen of the remote switch and the access password
 *  (docs/REMOTE-STATUS.md). It cannot be dismissed: it goes when the desktop
 *  shell says the lock is off, and the shell refuses every command meanwhile,
 *  so this is the face of the lock, not the lock itself. */

import { useEffect, useRef, useState, type FormEvent } from "react";
import { createPortal } from "react-dom";

import { UpdateProgress } from "@/app/UpdateProgress";
import { checkServiceStatus, legacyInstall, openExternal, serviceLock, unlockAccess, type ServiceLock } from "@/lib/service";
import { checkForUpdate, type UpdateInfo } from "@/lib/updater";
import { AppIcon, CaptionButtons, IS_MAC, Modal, OS } from "@/ui/frame";
import { Btn, Ring, TBox, cx } from "@/ui/kit";

/** While the lock shows: the window underneath is inert, and its shortcuts
 *  (Ctrl+1-5, Ctrl+O, Enter to run…) never reach it. Typing, Enter on a
 *  button and paste still work inside the lock screen: they are the
 *  browser's own default actions, which this does not prevent. */
function useInertApp() {
  useEffect(() => {
    const root = document.getElementById("root");
    if (root) root.inert = true;
    // Whatever had focus on the page goes; the password field, which
    // focused itself a moment ago (children's effects run first), stays.
    const active = document.activeElement as HTMLElement | null;
    if (!active?.closest?.(".lock-screen")) active?.blur?.();
    const swallow = (event: KeyboardEvent) => {
      event.stopImmediatePropagation();
      if (!(event.target as HTMLElement | null)?.closest?.(".lock-screen")) event.preventDefault();
    };
    window.addEventListener("keydown", swallow, true);
    return () => {
      if (root) root.inert = false;
      window.removeEventListener("keydown", swallow, true);
    };
  }, []);
}

const clock = (ms: number | null) =>
  ms ? new Date(ms).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }) : "";

export function ServiceLockScreen({ lock, onLock }: { lock: ServiceLock; onLock: (lock: ServiceLock) => void }) {
  useInertApp();
  return createPortal(
    <div className="lock-screen" role="alertdialog" aria-modal="true" aria-labelledby="lock-title" aria-describedby="lock-body">
      {/* The lock covers the title bar: keep the window movable, and on
          Windows (no native frame) keep minimize, maximize and close. */}
      <div className={cx("lock-bar", IS_MAC && "mac")} data-tauri-drag-region>
        {OS === "windows" && <CaptionButtons />}
      </div>
      <div className="lock-body" data-tauri-drag-region>
        <div className="lock-card">
          <AppIcon size={48} />
          {lock.reason === "password" ? (
            <PasswordForm onLock={onLock} />
          ) : lock.reason === "connect" ? (
            <ConnectPanel lock={lock} onLock={onLock} />
          ) : (
            <PausedPanel lock={lock} onLock={onLock} />
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** "Check again" / "Try again": fetch now, and stay busy long enough to be seen. */
function useCheckAgain(onLock: (lock: ServiceLock) => void): [boolean, () => Promise<void>] {
  const [checking, setChecking] = useState(false);
  const check = async () => {
    setChecking(true);
    const began = Date.now();
    try {
      onLock(await checkServiceStatus());
    } catch {
      /* the last known state stays */
    }
    window.setTimeout(() => setChecking(false), Math.max(0, 600 - (Date.now() - began)));
  };
  return [checking, check];
}

function CheckButton({ checking, onCheck, accent, label }: { checking: boolean; onCheck: () => void; accent?: boolean; label: string }) {
  return (
    <Btn accent={accent} onClick={onCheck} disabled={checking}>
      {checking ? (
        <>
          <Ring size={14} /> Checking…
        </>
      ) : (
        label
      )}
    </Btn>
  );
}

/** Maintenance, or a build older than the document's minVersion. */
function PausedPanel({ lock, onLock }: { lock: ServiceLock; onLock: (lock: ServiceLock) => void }) {
  const [checking, check] = useCheckAgain(onLock);
  const update = lock.reason === "update";
  const [available, setAvailable] = useState<UpdateInfo | null>(null);
  const [installing, setInstalling] = useState(false);

  // The app's own updater keeps working while locked: for "Update required"
  // it is the way out, so offer it right here when GitHub has the version.
  useEffect(() => {
    if (!update) return;
    let alive = true;
    void checkForUpdate(true)
      .then((found) => alive && setAvailable(found))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [update]);

  const title = update ? "Update required" : lock.title || "Work is in progress";
  const message = update && !lock.message ? "This version of AudioSyncMaster is no longer supported. Please install the new version." : lock.message;
  return (
    <>
      <h1 id="lock-title" className="lock-title">
        {title}
      </h1>
      <div id="lock-body" className="lock-text">
        {message && <p className="lock-message">{message}</p>}
        <p className="lock-note">Your work is paused and will continue when AudioSyncMaster is back.</p>
      </div>
      <div className="lock-actions">
        {update && available && (
          <Btn accent onClick={() => setInstalling(true)}>
            Install version {available.version}
          </Btn>
        )}
        {update && !available && lock.updateUrl && (
          <Btn accent onClick={() => void openExternal(lock.updateUrl!).catch(() => undefined)}>
            Download the new version
          </Btn>
        )}
        <CheckButton checking={checking} onCheck={() => void check()} label="Check again" accent={!update} />
      </div>
      {lock.checkedAt !== null && (
        <p className="lock-meta">
          {update && lock.minVersion
            ? `Version ${lock.appVersion} · required ${lock.minVersion}`
            : `Last checked at ${clock(lock.checkedAt)}`}
        </p>
      )}
      {installing && available && <UpdateProgress update={available} onClose={() => setInstalling(false)} />}
    </>
  );
}

/** A build that requires access and has never read the status document. */
function ConnectPanel({ lock, onLock }: { lock: ServiceLock; onLock: (lock: ServiceLock) => void }) {
  const [checking, check] = useCheckAgain(onLock);
  // The first check is still on its way: say so, rather than flash
  // "Connect to the internet" for the second it takes.
  if (!lock.lastError) {
    return (
      <>
        <h1 id="lock-title" className="lock-title">
          Checking access…
        </h1>
        <div id="lock-body" className="lock-text">
          <Ring size={20} />
        </div>
      </>
    );
  }
  return (
    <>
      <h1 id="lock-title" className="lock-title">
        Connect to the internet once to verify access
      </h1>
      <div id="lock-body" className="lock-text">
        <p className="lock-message">AudioSyncMaster checks its access the first time it opens for your account. After that it also works offline.</p>
      </div>
      <div className="lock-actions">
        <CheckButton checking={checking} onCheck={() => void check()} label="Try again" accent />
      </div>
      {lock.lastError && <p className="lock-meta">{lock.lastError}</p>}
    </>
  );
}

/** The access password: asked once per OS user, then remembered by the
 *  shell in the OS credential store until the password changes. The
 *  password goes to the shell for hashing and is kept nowhere. */
function PasswordForm({ onLock }: { onLock: (lock: ServiceLock) => void }) {
  const [password, setPassword] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** When the next attempt may be made, after too many wrong ones. */
  const [waitUntil, setWaitUntil] = useState<number | null>(null);
  const [, tick] = useState(0);
  const field = useRef<HTMLInputElement>(null);

  useEffect(() => field.current?.focus(), []);
  useEffect(() => {
    if (waitUntil === null) return;
    const timer = window.setInterval(() => {
      if (Date.now() >= waitUntil) {
        setWaitUntil(null);
        setError(null);
        field.current?.focus();
      } else tick((n) => n + 1);
    }, 500);
    return () => window.clearInterval(timer);
  }, [waitUntil]);

  const waiting = waitUntil !== null && Date.now() < waitUntil;
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    // The field itself, not only React's copy: password managers and
    // assistive tools can fill it without the input event React follows.
    const entered = field.current?.value || password;
    if (checking || waiting) return;
    if (!entered) {
      field.current?.focus();
      return;
    }
    setChecking(true);
    setError(null);
    try {
      onLock(await unlockAccess(entered));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPassword("");
      if (field.current) field.current.value = "";
      const state = await serviceLock().catch(() => null);
      if (state?.retryIn) setWaitUntil(Date.now() + state.retryIn * 1000);
      window.setTimeout(() => field.current?.focus(), 0);
    } finally {
      setChecking(false);
    }
  };

  const secondsLeft = waiting ? Math.ceil((waitUntil! - Date.now()) / 1000) : 0;
  return (
    <form className="lock-form" onSubmit={(event) => void submit(event)}>
      <h1 id="lock-title" className="lock-title">
        This app is strictly prohibited to be used by others.
      </h1>
      <p id="lock-body" className="lock-message">
        If you are the trusted one, enter the password.
      </p>
      <TBox
        ref={field}
        type="password"
        value={password}
        onChange={(value) => {
          setPassword(value);
          if (!waiting) setError(null);
        }}
        label="Access password"
        placeholder="Password"
        autoComplete="off"
        spellCheck={false}
        disabled={checking || waiting}
        aria-invalid={error !== null}
        aria-describedby="lock-error"
      />
      <p id="lock-error" className="lock-error" role="alert">
        {waiting ? `Too many wrong passwords. Try again in ${secondsLeft} s.` : error}
      </p>
      <Btn accent type="submit" disabled={checking || waiting} className="lock-unlock">
        {checking ? (
          <>
            <Ring size={14} /> Checking…
          </>
        ) : (
          "Unlock"
        )}
      </Btn>
    </form>
  );
}

const LEGACY_KEY = "audiosync.legacyInstall.noticed";

/** Windows, once per user: an older copy installed for all users (Program
 *  Files) is still there. This per-user install cannot remove it; an
 *  administrator has to. */
export function LegacyInstallNotice() {
  const [path, setPath] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    void legacyInstall()
      .then((found) => {
        if (!alive || !found) return;
        if (localStorage.getItem(LEGACY_KEY) === found) return;
        setPath(found);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);
  if (!path) return null;
  const close = () => {
    localStorage.setItem(LEGACY_KEY, path);
    setPath(null);
  };
  return (
    <Modal onClose={close}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="legacy-title">
        <div className="db">
          <div className="dt" id="legacy-title">
            An all-users copy is still installed
          </div>
          <p className="lock-dialog-text">
            An all-users copy of AudioSyncMaster is installed in Program Files (<span className="lock-path">{path}</span>). Ask an administrator to
            uninstall it from Settings &gt; Apps.
          </p>
          <p className="lock-note">This copy is installed for your account only, and updates itself for your account only.</p>
        </div>
        <div className="df" style={{ gridTemplateColumns: "1fr" }}>
          <Btn accent onClick={close}>
            OK
          </Btn>
        </div>
      </div>
    </Modal>
  );
}
