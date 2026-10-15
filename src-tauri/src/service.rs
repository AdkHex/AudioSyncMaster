//! The remote switch and the access password, app side (docs/REMOTE-STATUS.md).
//! The logic -- the document, the lock it means, the password check, the
//! sticky state -- is in `remote_status.rs`; this is where it meets the app:
//!
//! - the gate every command checks first ([`guard`]), and the long work it
//!   knows is running ([`begin`]), so a lock can stop it and say what it
//!   stopped;
//! - stopping: every engine is told to cancel, and the conversions in the
//!   audio queue are remembered and queued again when the lock lifts; the
//!   pages restart their own runs from the `resume` list on that event;
//! - the poll loop, the `service-lock-changed` event, and the commands the
//!   lock screen uses.
//!
//! The URL and `requireAccess` are baked in at compile time from
//! `src-tauri/remote-status.json`. Debug builds also read
//! `AUDIOSYNC_STATUS_URL` (http:// allowed for 127.0.0.1 / localhost) and
//! `AUDIOSYNC_STATUS_POLL_SECS` for testing; release builds ignore both.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};

use crate::bridge::{BridgeHandle, ShotBridge, WaveformBridge};
use crate::encode::EncodeBridge;
use crate::remote_status::{
    validate_url, AccessStore, CoreConfig, Gate, ServiceCore, ServiceLock, Transition,
};
use crate::subsync::SubsInfoBridge;

pub(crate) const EVENT: &str = "service-lock-changed";
const BAKED: &str = include_str!("../remote-status.json");
/// The credential store's account; the service is the bundle identifier.
const CREDENTIAL_ACCOUNT: &str = "access";
const STORE_FILE: &str = "remote-status.json";

/// The lock in force for this process. Set only by the [`ServiceCore`].
pub(crate) static GATE: Gate = Gate::new();

/// Every command's first check: `Err` while the app is paused or locked.
pub(crate) fn guard() -> Result<(), String> {
    GATE.ensure_open()
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------- config

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct Baked {
    url: String,
    require_access: bool,
}

fn baked() -> Baked {
    serde_json::from_str(BAKED).unwrap_or_default()
}

/// The status URL this build polls, if the switch is on.
fn status_url(baked: &Baked) -> Option<String> {
    if cfg!(debug_assertions) {
        if let Some(u) = std::env::var("AUDIOSYNC_STATUS_URL")
            .ok()
            .filter(|u| !u.trim().is_empty())
        {
            return match validate_url(&u, true) {
                Ok(u) => Some(u),
                Err(e) => {
                    log::error!("AUDIOSYNC_STATUS_URL ignored: {e}");
                    None
                }
            };
        }
    }
    let raw = baked.url.trim();
    if raw.is_empty() {
        return None;
    }
    match validate_url(raw, false) {
        Ok(u) => Some(u),
        Err(e) => {
            log::error!("remote-status.json: {e}");
            None
        }
    }
}

fn poll_override() -> Option<Duration> {
    if !cfg!(debug_assertions) {
        return None;
    }
    std::env::var("AUDIOSYNC_STATUS_POLL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map(Duration::from_secs)
}

/// "This user has entered the password", in the OS credential store:
/// Windows Credential Manager or the macOS Keychain, per OS user.
struct KeychainStore {
    service: String,
}

impl AccessStore for KeychainStore {
    fn load(&self) -> Option<String> {
        keyring::Entry::new(&self.service, CREDENTIAL_ACCOUNT)
            .ok()?
            .get_password()
            .ok()
            .filter(|id| !id.is_empty())
    }

    fn save(&self, id: &str) -> Result<(), String> {
        keyring::Entry::new(&self.service, CREDENTIAL_ACCOUNT)
            .and_then(|entry| entry.set_password(id))
            .map_err(|e| e.to_string())
    }
}

// ------------------------------------------------------------ running work

static NEXT_WORK: AtomicU64 = AtomicU64::new(1);
/// Long work in flight, by kind: what a lock stops and reports.
static RUNNING: Mutex<Vec<(u64, &'static str)>> = Mutex::new(Vec::new());
/// What the current lock stopped, until it lifts.
static STOPPED: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SWEEPING: AtomicBool = AtomicBool::new(false);

/// A piece of long work, registered until dropped.
pub(crate) struct Work(u64);

impl Drop for Work {
    fn drop(&mut self) {
        lock(&RUNNING).retain(|(id, _)| *id != self.0);
    }
}

/// Start long work of `kind`: refused while locked, and known to the lock
/// until the returned guard is dropped. The gate is read under the same
/// lock a stopping lock reads the list under, so work is either refused or
/// seen -- never started unseen.
pub(crate) fn begin(kind: &'static str) -> Result<Work, String> {
    begin_on(&GATE, kind)
}

fn begin_on(gate: &Gate, kind: &'static str) -> Result<Work, String> {
    let mut running = lock(&RUNNING);
    gate.ensure_open()?;
    let id = NEXT_WORK.fetch_add(1, Ordering::SeqCst);
    running.push((id, kind));
    Ok(Work(id))
}

fn running_kinds() -> Vec<String> {
    let mut kinds: Vec<String> = Vec::new();
    for (_, kind) in lock(&RUNNING).iter() {
        if !kinds.iter().any(|k| k == kind) {
            kinds.push(kind.to_string());
        }
    }
    kinds
}

// ------------------------------------------------------- the audio queue

/// The conversions in the audio queue, by id, as they were asked for: what
/// a lock cancels and queues again when it lifts.
#[derive(Debug, Default)]
pub(crate) struct EncodeJobs {
    live: Vec<(String, Value)>,
    /// ids a lock cancelled whose `audioJobDone` has not come back yet; they
    /// are queued again only once it has, as the engine keeps a running
    /// job's id until it stops
    stopping: Vec<String>,
    /// the specs a lock cancelled, queued again when it lifts
    resume: Vec<Value>,
    next: u64,
}

impl EncodeJobs {
    /// Jobs on their way into the queue: give each one an id if it has none
    /// (the pages then learn it from the ack) and remember its spec.
    pub(crate) fn record(&mut self, jobs: &mut Value) {
        let Some(list) = jobs.as_array_mut() else {
            return;
        };
        for spec in list {
            let Some(object) = spec.as_object_mut() else {
                continue;
            };
            let id = match object
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                Some(id) => id.to_string(),
                None => {
                    self.next += 1;
                    let id = format!("asm-{}-{}", crate::remote_status::now_ms(), self.next);
                    object.insert("id".into(), Value::String(id.clone()));
                    id
                }
            };
            if !self.live.iter().any(|(live, _)| *live == id) {
                self.live.push((id, spec.clone()));
            }
        }
    }

    /// Follow the queue's events: a finished job leaves, a skipped one never came.
    pub(crate) fn observe(&mut self, event: &Value) {
        match event.get("type").and_then(Value::as_str) {
            Some("audioJobDone") => {
                if let Some(id) = event.get("jobId").and_then(Value::as_str) {
                    self.live.retain(|(live, _)| live != id);
                    self.stopping.retain(|stopping| stopping != id);
                }
            }
            Some("audioEnqueueAck") => {
                for skipped in event
                    .get("skipped")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    // "id already used": the job under that id is the one
                    // already known, so it stays.
                    if skipped.get("reason").and_then(Value::as_str) == Some("id already used") {
                        continue;
                    }
                    if let Some(id) = skipped.get("id").and_then(Value::as_str) {
                        self.live.retain(|(live, _)| live != id);
                    }
                }
            }
            _ => {}
        }
    }

    /// A lock: everything live is to be cancelled and queued again later.
    pub(crate) fn stop(&mut self) -> usize {
        let live = std::mem::take(&mut self.live);
        let n = live.len();
        for (id, spec) in live {
            self.stopping.push(id);
            self.resume.push(spec);
        }
        n
    }

    /// The specs to queue again, once every job the lock cancelled has
    /// reported back; None while some are still stopping.
    pub(crate) fn take_resume(&mut self) -> Option<Vec<Value>> {
        if self.resume.is_empty() {
            return Some(Vec::new());
        }
        if !self.stopping.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.resume))
    }
}

pub(crate) static ENCODE: Mutex<EncodeJobs> = Mutex::new(EncodeJobs {
    live: Vec::new(),
    stopping: Vec::new(),
    resume: Vec::new(),
    next: 0,
});

/// Send jobs to the running queue: refused while locked, and remembered.
/// The gate is read and the message written under the queue's lock, so a
/// lock that starts meanwhile cancels these too.
pub(crate) fn send_encode_jobs(handle: &BridgeHandle, mut message: Value) -> Result<(), String> {
    let mut jobs = lock(&ENCODE);
    guard()?;
    if let Some(list) = message.get_mut("jobs") {
        jobs.record(list);
    }
    handle.send_now(&message)
}

/// The jobs a queue session starts with.
pub(crate) fn record_encode_jobs(payload: &mut Value) -> Result<(), String> {
    let mut jobs = lock(&ENCODE);
    guard()?;
    if let Some(list) = payload.get_mut("jobs") {
        jobs.record(list);
    }
    Ok(())
}

/// An event from the queue session.
pub(crate) fn encode_event(event: &Value) {
    lock(&ENCODE).observe(event);
}

// ------------------------------------------------------- stop and resume

fn cancel_main_engine<R: Runtime>(app: &AppHandle<R>) {
    if let Some(handle) = app.try_state::<BridgeHandle>() {
        let _ = handle.send_now(&json!({ "command": "cancel" }));
    }
}

/// Stop everything: the run on each engine (sync, dub sync, subtitles,
/// installs; waveform and picture reads), and every job in the audio queue.
/// Returns the kinds of long work that were running.
fn stop_work<R: Runtime>(app: &AppHandle<R>) -> Vec<String> {
    let mut kinds = running_kinds();
    if lock(&ENCODE).stop() > 0 {
        kinds.push("encode".into());
    }
    cancel_main_engine(app);
    let cancel = json!({ "command": "cancel" });
    if let Some(handle) = app.try_state::<WaveformBridge>() {
        let _ = handle.0.send_now(&cancel);
    }
    if let Some(handle) = app.try_state::<ShotBridge>() {
        let _ = handle.0.send_now(&cancel);
    }
    if let Some(handle) = app.try_state::<SubsInfoBridge>() {
        let _ = handle.0.send_now(&cancel);
    }
    if let Some(handle) = app.try_state::<EncodeBridge>() {
        let _ = handle.0.send_now(&json!({ "command": "audioCancelAll" }));
    }
    kinds
}

/// A cancel can reach the engine a moment before the run it targets has
/// started, and miss it. While locked, and while work is still registered
/// as running, say it again.
fn spawn_sweeper<R: Runtime>(app: AppHandle<R>) {
    if SWEEPING.swap(true, Ordering::SeqCst) {
        return;
    }
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            if !GATE.is_locked() || lock(&RUNNING).is_empty() {
                break;
            }
            cancel_main_engine(&app);
        }
        SWEEPING.store(false, Ordering::SeqCst);
    });
}

/// The lock lifted: queue the conversions it stopped again (once they have
/// all stopped), and hand back what else it stopped for the pages.
fn resume_work<R: Runtime>(app: &AppHandle<R>) -> Vec<String> {
    let stopped = std::mem::take(&mut *lock(&STOPPED));
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let deadline = Instant::now() + Duration::from_secs(120);
        let specs = loop {
            if GATE.is_locked() {
                return; // locked again: the next unlock resumes them
            }
            if let Some(specs) = lock(&ENCODE).take_resume() {
                break specs;
            }
            if Instant::now() > deadline {
                log::warn!("remote status: the stopped conversions never reported back");
                let mut jobs = lock(&ENCODE);
                jobs.stopping.clear();
                break std::mem::take(&mut jobs.resume);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        };
        if specs.is_empty() {
            return;
        }
        let n = specs.len();
        let Some(handle) = app.try_state::<EncodeBridge>() else {
            return;
        };
        let message = json!({ "command": "audioEnqueue", "jobs": specs });
        match send_encode_jobs(&handle.0, message) {
            Ok(()) => log::info!("remote status: queued {n} conversion(s) again"),
            Err(e) => log::warn!("remote status: cannot queue the stopped conversions again: {e}"),
        }
    });
    stopped
}

// ------------------------------------------------------------ the service

pub(crate) struct Service {
    core: ServiceCore,
}

impl Service {
    fn snapshot(&self) -> ServiceLock {
        let mut lock_state = self.core.snapshot();
        lock_state.stopped = lock(&STOPPED).clone();
        lock_state
    }
}

#[derive(Clone)]
pub(crate) struct ServiceHandle(Arc<Service>);

fn emit<R: Runtime>(app: &AppHandle<R>, state: &ServiceLock) {
    let _ = app.emit(EVENT, state);
}

fn bring_to_front<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Act on what a check or an unlock did.
fn handle<R: Runtime>(app: &AppHandle<R>, svc: &Service, t: Transition) {
    match t {
        Transition::Engaged => {
            log::warn!("remote status: locking ({:?})", GATE.reason());
            let stopped = stop_work(app);
            {
                let mut list = lock(&STOPPED);
                for kind in stopped {
                    if !list.contains(&kind) {
                        list.push(kind);
                    }
                }
            }
            bring_to_front(app);
            spawn_sweeper(app.clone());
            emit(app, &svc.snapshot());
        }
        Transition::Released => {
            log::warn!("remote status: unlocking");
            let resume = resume_work(app);
            let mut state = svc.snapshot();
            state.resume = resume;
            emit(app, &state);
        }
        Transition::Changed => emit(app, &svc.snapshot()),
        Transition::Unchanged => {}
    }
}

/// At startup, before anything can run: apply the kept document (a lock
/// from the last run is back on at once), then poll in the background.
pub(crate) fn setup<R: Runtime>(app: &AppHandle<R>) {
    let baked = baked();
    let store = app
        .path()
        .app_data_dir()
        .ok()
        .map(|dir| dir.join(STORE_FILE));
    start(
        app,
        status_url(&baked),
        baked.require_access,
        store,
        poll_override(),
    );
}

/// [`setup`] with everything it reads from the build and the environment
/// given instead, for the engine tests.
pub(crate) fn start<R: Runtime>(
    app: &AppHandle<R>,
    url: Option<String>,
    require_access: bool,
    store: Option<std::path::PathBuf>,
    every: Option<Duration>,
) {
    let version = app.package_info().version.to_string();
    let core = ServiceCore::new(
        CoreConfig {
            app_version: version.clone(),
            url: url.clone(),
            require_access,
            store,
            user_agent: format!("AudioSyncMaster/{version}"),
        },
        Arc::new(KeychainStore {
            service: app.config().identifier.clone(),
        }),
        &GATE,
    );
    if core.restore() == Transition::Engaged {
        log::info!("remote status: locked from the start ({:?})", GATE.reason());
    }
    let svc = Arc::new(Service { core });
    app.manage(ServiceHandle(svc.clone()));
    let Some(url) = url else {
        log::info!("remote status: no status URL in this build; the switch is off");
        return;
    };
    // The gist's link stays out of a release build's log: anyone who has
    // it can read the document.
    if cfg!(debug_assertions) {
        log::info!("remote status: polling {url}");
    } else {
        log::info!("remote status: polling the status document");
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            let t = svc.core.check_now().await;
            handle(&app, &svc, t);
            let wait = every.unwrap_or_else(|| svc.core.interval());
            tokio::time::sleep(wait).await;
        }
    });
}

// --------------------------------------------------------------- commands

#[tauri::command]
pub(crate) fn service_lock(svc: State<'_, ServiceHandle>) -> ServiceLock {
    svc.0.snapshot()
}

/// "Check again" / "Try again" on the lock screen: fetch now.
#[tauri::command]
pub(crate) async fn check_service_status<R: Runtime>(
    app: AppHandle<R>,
    svc: State<'_, ServiceHandle>,
) -> Result<ServiceLock, String> {
    let svc = svc.0.clone();
    let t = svc.core.check_now().await;
    handle(&app, &svc, t);
    Ok(svc.snapshot())
}

/// Unlock on the password screen. The password is hashed here and never
/// stored or logged; on success this user's credential store keeps the
/// password's id, so the app does not ask again until the password changes.
#[tauri::command]
pub(crate) async fn unlock_access<R: Runtime>(
    app: AppHandle<R>,
    svc: State<'_, ServiceHandle>,
    password: String,
) -> Result<ServiceLock, String> {
    let svc = svc.0.clone();
    let worker = svc.clone();
    let t = tauri::async_runtime::spawn_blocking(move || worker.core.unlock(&password))
        .await
        .map_err(|e| e.to_string())??;
    handle(&app, &svc, t);
    Ok(svc.snapshot())
}

/// Windows: an older copy installed for all users (an MSI, or a per-machine
/// NSIS install), which this per-user install cannot remove. Its folder, for
/// a one-time notice; None everywhere else.
#[tauri::command]
pub(crate) fn legacy_install() -> Option<String> {
    #[cfg(windows)]
    {
        windows_legacy::find()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(windows)]
mod windows_legacy {
    use std::path::{Path, PathBuf};

    use winreg::enums::{HKEY_LOCAL_MACHINE, KEY_READ, KEY_WOW64_32KEY, KEY_WOW64_64KEY};
    use winreg::RegKey;

    const PRODUCT: &str = "AudioSyncMaster";
    const UNINSTALL: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall";

    fn same(a: &Path, b: &Path) -> bool {
        let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        canon(a) == canon(b)
    }

    pub(super) fn find() -> Option<String> {
        let running = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf));
        let is_running = |dir: &Path| running.as_deref().is_some_and(|r| same(r, dir));

        let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
        for view in [KEY_WOW64_64KEY, KEY_WOW64_32KEY] {
            let Ok(root) = hklm.open_subkey_with_flags(UNINSTALL, KEY_READ | view) else {
                continue;
            };
            for name in root.enum_keys().flatten() {
                let Ok(key) = root.open_subkey_with_flags(&name, KEY_READ | view) else {
                    continue;
                };
                let display: String = key.get_value("DisplayName").unwrap_or_default();
                if display.trim() != PRODUCT {
                    continue;
                }
                let location: String = key.get_value("InstallLocation").unwrap_or_default();
                let location = location.trim().trim_matches('"').to_string();
                if !location.is_empty() && is_running(Path::new(&location)) {
                    continue;
                }
                return Some(if location.is_empty() {
                    format!("{PRODUCT} (all users)")
                } else {
                    location
                });
            }
        }
        for var in ["ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"] {
            let Ok(base) = std::env::var(var) else {
                continue;
            };
            let dir = PathBuf::from(base).join(PRODUCT);
            let has_exe = std::fs::read_dir(&dir).is_ok_and(|entries| {
                entries.flatten().any(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("exe"))
                })
            });
            if has_exe && !is_running(&dir) {
                return Some(dir.to_string_lossy().into_owned());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build ships with the switch on, pointed at the gist over https,
    /// and with access required.
    #[test]
    fn the_baked_config_is_the_gist_with_access_required() {
        let b = baked();
        assert!(b.require_access, "requireAccess must be baked in");
        let url = validate_url(&b.url, false).expect("the baked URL must be https");
        assert!(
            url.starts_with("https://gist.githubusercontent.com/AdkHex/")
                && url.ends_with("/raw/audiosyncmaster-status.json"),
            "{url}"
        );
    }

    #[test]
    fn long_work_is_listed_while_it_runs_and_refused_while_locked() {
        let gate: &'static Gate = Box::leak(Box::new(Gate::new()));
        let work = begin_on(gate, "sync").expect("open");
        assert_eq!(running_kinds(), vec!["sync".to_string()]);
        drop(work);
        assert!(running_kinds().is_empty());

        // A build that requires access and has never read the document.
        let core = ServiceCore::new(
            CoreConfig {
                app_version: "2.10.0".into(),
                url: Some("https://example.invalid/status.json".into()),
                require_access: true,
                store: None,
                user_agent: "test".into(),
            },
            Arc::new(crate::remote_status::MemoryStore::default()),
            gate,
        );
        assert_eq!(core.restore(), Transition::Engaged);
        assert_eq!(
            begin_on(gate, "dubsync").err().as_deref(),
            Some(crate::remote_status::LOCKED_ERROR)
        );
        assert!(running_kinds().is_empty(), "refused work is not listed");
    }

    fn ack(added: &[&str], skipped: &[(&str, &str)]) -> Value {
        json!({
            "type": "audioEnqueueAck",
            "added": added,
            "skipped": skipped.iter().map(|(id, reason)| json!({ "id": id, "reason": reason })).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn the_queue_tracker_resumes_what_a_lock_cancelled() {
        let mut jobs = EncodeJobs::default();
        let mut sent = json!([
            { "id": "a", "path": "/m/a.mkv" },
            { "path": "/m/b.mkv" },
            { "id": "c", "path": "/m/c.mkv" },
        ]);
        jobs.record(&mut sent);
        let b_id = sent[1]["id"].as_str().expect("an id is given").to_string();
        assert!(b_id.starts_with("asm-"));
        jobs.observe(&ack(&["a", &b_id], &[("c", "already queued")]));
        jobs.observe(&json!({ "type": "audioJobDone", "jobId": "a", "state": "done" }));
        // Only b is live: a finished, c was never queued.
        assert_eq!(jobs.stop(), 1);
        // Not before b has reported its cancel back.
        assert_eq!(jobs.take_resume(), None);
        jobs.observe(&json!({ "type": "audioJobDone", "jobId": b_id, "state": "cancelled" }));
        let again = jobs.take_resume().expect("all stopped");
        assert_eq!(again, vec![json!({ "id": b_id, "path": "/m/b.mkv" })]);
        assert_eq!(jobs.take_resume(), Some(Vec::new()));
        // Queued again, it is live under the same id.
        let mut resent = Value::Array(again);
        jobs.record(&mut resent);
        assert_eq!(jobs.stop(), 1);
    }

    #[test]
    fn a_skipped_duplicate_id_keeps_the_known_job() {
        let mut jobs = EncodeJobs::default();
        jobs.record(&mut json!([{ "id": "a", "path": "/m/a.mkv" }]));
        jobs.record(&mut json!([{ "id": "a", "path": "/m/other.mkv" }]));
        jobs.observe(&ack(&[], &[("a", "id already used")]));
        assert_eq!(jobs.stop(), 1);
        jobs.observe(&json!({ "type": "audioJobDone", "jobId": "a", "state": "cancelled" }));
        assert_eq!(
            jobs.take_resume(),
            Some(vec![json!({ "id": "a", "path": "/m/a.mkv" })])
        );
    }
}
