//! Remote control for test builds: the "Work is in progress" switch and the
//! access password, both driven by one small JSON document on a secret
//! GitHub Gist. See docs/REMOTE-STATUS.md.
//!
//! This is the logic, with no Tauri in it, so the integration tests can run
//! it against a local server: parsing the document, the lock it means for
//! this build and this OS user, the password check, fetching, and the state
//! kept between runs. `service.rs` wires it into the app -- the poll loop,
//! the guard every command calls, stopping and resuming work, and the
//! commands the lock screen uses.
//!
//! The switch fails open and sticks: an unreachable or unusable document
//! changes nothing, and the last good one is kept on disk, so a lock
//! survives restarts and offline spells until a good `ok` arrives. Only a
//! build that requires access fails closed, and only before its first
//! successful check: a copy that has never read the document does not open.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// Larger documents are ignored.
pub const MAX_BYTES: usize = 64 * 1024;
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_CHECK_MINUTES: u32 = 5;
/// What every refused command answers while the switch says maintenance (or
/// the build is too old).
pub const PAUSED_ERROR: &str = "AudioSyncMaster is paused: work is in progress";
/// What every refused command answers while the password has not been entered.
pub const LOCKED_ERROR: &str = "AudioSyncMaster is locked";
pub const WRONG_PASSWORD: &str = "Wrong password";
pub const ACCESS_ALG: &str = "pbkdf2-sha256";
/// Wrong passwords in a row before the next attempt has to wait.
pub const FREE_ATTEMPTS: u32 = 5;
pub const THROTTLE: Duration = Duration::from_secs(30);
/// Round counts outside this range are refused: too few make the published
/// hash cheap to guess offline, too many would hang every unlock.
const MIN_ITERATIONS: u64 = 10_000;
const MAX_ITERATIONS: u64 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RemoteState {
    #[default]
    Ok,
    Maintenance,
}

/// The access password as the document publishes it: a salted PBKDF2-SHA256
/// hash, never the password itself.
#[derive(Clone, PartialEq, Eq)]
pub struct Access {
    iterations: u32,
    salt: Vec<u8>,
    hash: [u8; 32],
    id: String,
}

impl std::fmt::Debug for Access {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Access")
            .field("id", &self.id)
            .field("iterations", &self.iterations)
            .finish_non_exhaustive()
    }
}

impl Access {
    /// Which password this is. It changes whenever the password does, and it
    /// is all a user's credential store keeps once they have entered it.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Whether `password` is the one the hash was made from. Slow on purpose
    /// (PBKDF2), and compared in constant time.
    pub fn verify(&self, password: &str) -> bool {
        let mut out = [0u8; 32];
        pbkdf2::pbkdf2_hmac::<Sha256>(password.as_bytes(), &self.salt, self.iterations, &mut out);
        out.ct_eq(&self.hash).into()
    }
}

/// The first 16 hex digits of sha256(salt || hash): the password's id, the
/// same as `scripts/audiosyncmaster-status.mjs` publishes.
pub fn access_id(salt: &[u8], hash: &[u8]) -> String {
    Sha256::new()
        .chain_update(salt)
        .chain_update(hash)
        .finalize()
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A usable `access` block, or None. An `id` that does not match the salt
/// and hash is refused rather than trusted.
fn parse_access(v: &Value) -> Option<Access> {
    let o = v.as_object()?;
    if !o
        .get("alg")?
        .as_str()?
        .trim()
        .eq_ignore_ascii_case(ACCESS_ALG)
    {
        return None;
    }
    let iterations = o
        .get("iterations")?
        .as_u64()
        .filter(|n| (MIN_ITERATIONS..=MAX_ITERATIONS).contains(n))? as u32;
    let b64 = base64::engine::general_purpose::STANDARD;
    let salt = b64
        .decode(o.get("salt")?.as_str()?.trim())
        .ok()
        .filter(|s| (8..=64).contains(&s.len()))?;
    let hash: [u8; 32] = b64
        .decode(o.get("hash")?.as_str()?.trim())
        .ok()?
        .try_into()
        .ok()?;
    let id = access_id(&salt, &hash);
    match o.get("id") {
        None | Some(Value::Null) => {}
        Some(given) => {
            if !given.as_str()?.trim().eq_ignore_ascii_case(&id) {
                return None;
            }
        }
    }
    Some(Access {
        iterations,
        salt,
        hash,
        id,
    })
}

/// The remote document after validation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StatusDoc {
    pub status: RemoteState,
    pub title: Option<String>,
    pub message: Option<String>,
    pub min_version: Option<String>,
    pub update_url: Option<String>,
    pub check_minutes: Option<u32>,
    pub access: Option<Access>,
}

impl StatusDoc {
    pub fn interval(&self) -> Duration {
        Duration::from_secs(
            self.check_minutes
                .unwrap_or(DEFAULT_CHECK_MINUTES)
                .clamp(1, 60) as u64
                * 60,
        )
    }
}

// ------------------------------------------------------------------ parsing

fn clean_text(v: Option<&Value>, max_chars: usize) -> Option<String> {
    let s = v?.as_str()?.trim();
    if s.is_empty() {
        return None;
    }
    Some(s.chars().take(max_chars).collect())
}

/// Parse the document. `None` = no information (too large, not JSON, not an
/// object, no valid `status`, or an `access` block that cannot be checked):
/// the caller keeps the last known state. Unknown fields are ignored and
/// invalid optional fields dropped -- except `access`, since dropping a
/// broken password block would open the app to everyone.
pub fn parse(bytes: &[u8]) -> Option<StatusDoc> {
    parse_body(bytes).map(|(_, doc)| doc)
}

/// [`parse`], keeping the JSON as fetched (it is what the store keeps).
pub fn parse_body(bytes: &[u8]) -> Option<(Value, StatusDoc)> {
    if bytes.len() > MAX_BYTES {
        return None;
    }
    let v: Value = serde_json::from_slice(bytes).ok()?;
    let doc = parse_value(&v)?;
    Some((v, doc))
}

fn parse_value(v: &Value) -> Option<StatusDoc> {
    let o = v.as_object()?;
    let status = match o
        .get("status")?
        .as_str()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "ok" => RemoteState::Ok,
        "maintenance" => RemoteState::Maintenance,
        _ => return None,
    };
    let access = match o.get("access") {
        None | Some(Value::Null) => None,
        Some(a) => Some(parse_access(a)?),
    };
    let update_url = clean_text(o.get("updateUrl"), 2000).filter(|u| {
        url::Url::parse(u)
            .map(|p| matches!(p.scheme(), "https" | "http"))
            .unwrap_or(false)
    });
    let min_version = clean_text(o.get("minVersion"), 64).filter(|v| parse_version(v).is_some());
    let check_minutes = o
        .get("checkMinutes")
        .and_then(Value::as_f64)
        .filter(|x| x.is_finite())
        .map(|x| x.round().clamp(1.0, 60.0) as u32);
    Some(StatusDoc {
        status,
        title: clean_text(o.get("title"), 200),
        message: clean_text(o.get("message"), 2000),
        min_version,
        update_url,
        check_minutes,
        access,
    })
}

/// "1.2.3", "v1.2", "1.0.0-beta.2" -> numeric parts (pre-release and build
/// suffixes are ignored). None when the first part is not a number.
pub fn parse_version(v: &str) -> Option<Vec<u64>> {
    let v = v.trim().trim_start_matches(['v', 'V']);
    let core = v.split(['-', '+']).next().unwrap_or("");
    let mut out = Vec::new();
    for (i, part) in core.split('.').enumerate() {
        let digits: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
        match digits.parse::<u64>() {
            Ok(n) => out.push(n),
            Err(_) if i == 0 => return None,
            Err(_) => out.push(0),
        }
    }
    Some(out)
}

/// True when `app` is older than `min`. Unparsable versions never count as older.
pub fn version_older(app: &str, min: &str) -> bool {
    let (Some(mut a), Some(mut b)) = (parse_version(app), parse_version(min)) else {
        return false;
    };
    let n = a.len().max(b.len());
    a.resize(n, 0);
    b.resize(n, 0);
    a < b
}

// --------------------------------------------------------------- the lock

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LockReason {
    /// the document says "maintenance"
    Maintenance,
    /// this build is older than `minVersion`
    Update,
    /// a password is set and this OS user has not entered it (or it changed)
    Password,
    /// the build requires access and has never read the document
    Connect,
}

impl LockReason {
    /// What a refused command answers.
    pub fn error(self) -> &'static str {
        match self {
            LockReason::Maintenance | LockReason::Update => PAUSED_ERROR,
            LockReason::Password | LockReason::Connect => LOCKED_ERROR,
        }
    }

    fn code(reason: Option<Self>) -> u8 {
        match reason {
            None => 0,
            Some(LockReason::Maintenance) => 1,
            Some(LockReason::Update) => 2,
            Some(LockReason::Password) => 3,
            Some(LockReason::Connect) => 4,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(LockReason::Maintenance),
            2 => Some(LockReason::Update),
            3 => Some(LockReason::Password),
            4 => Some(LockReason::Connect),
            _ => None,
        }
    }
}

/// The lock a document means for this build and this user. Maintenance wins
/// over "update required", and both over the password screen.
///
/// `doc` is the last good document (None when this copy has never read
/// one); `granted` the password id this user's credential store holds.
pub fn lock_reason(
    doc: Option<&StatusDoc>,
    app_version: &str,
    granted: Option<&str>,
    require_access: bool,
) -> Option<LockReason> {
    let Some(doc) = doc else {
        return require_access.then_some(LockReason::Connect);
    };
    if doc.status == RemoteState::Maintenance {
        return Some(LockReason::Maintenance);
    }
    if let Some(min) = &doc.min_version {
        if version_older(app_version, min) {
            return Some(LockReason::Update);
        }
    }
    match &doc.access {
        Some(access) if granted != Some(access.id()) => Some(LockReason::Password),
        _ => None,
    }
}

/// Status URLs must be https; plain http only for a local test server, and
/// only when allowed.
pub fn validate_url(raw: &str, allow_local_http: bool) -> Result<String, String> {
    let u = url::Url::parse(raw.trim()).map_err(|e| format!("invalid status URL: {e}"))?;
    match u.scheme() {
        "https" => Ok(u.to_string()),
        "http" if allow_local_http && is_loopback(&u) => Ok(u.to_string()),
        "http" => Err(
            "the status URL must use https:// (http:// only for 127.0.0.1 / localhost in tests)"
                .into(),
        ),
        s => Err(format!("unsupported status URL scheme {s}://")),
    }
}

fn is_loopback(u: &url::Url) -> bool {
    match u.host() {
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}

/// The flag every command checks: the lock in force right now, if any. The
/// app has one for the whole process (`service::GATE`).
#[derive(Debug, Default)]
pub struct Gate(AtomicU8);

impl Gate {
    pub const fn new() -> Self {
        Self(AtomicU8::new(0))
    }

    pub fn reason(&self) -> Option<LockReason> {
        LockReason::from_code(self.0.load(Ordering::SeqCst))
    }

    pub fn is_locked(&self) -> bool {
        self.reason().is_some()
    }

    /// Set the lock; returns the one it replaces.
    fn set(&self, reason: Option<LockReason>) -> Option<LockReason> {
        LockReason::from_code(self.0.swap(LockReason::code(reason), Ordering::SeqCst))
    }

    /// `Err` with the lock's message while one is on.
    pub fn ensure_open(&self) -> Result<(), String> {
        match self.reason() {
            Some(reason) => Err(reason.error().to_string()),
            None => Ok(()),
        }
    }
}

/// Wrong passwords slow the next ones down: every [`FREE_ATTEMPTS`]th in a
/// row makes the next attempt wait [`THROTTLE`].
#[derive(Debug, Default)]
pub struct Throttle {
    failures: u32,
    until: Option<Instant>,
}

impl Throttle {
    /// How long the next attempt still has to wait, if at all.
    pub fn wait(&self, now: Instant) -> Option<Duration> {
        self.until
            .and_then(|until| until.checked_duration_since(now))
            .filter(|d| !d.is_zero())
    }

    pub fn failed(&mut self, now: Instant) {
        self.failures += 1;
        if self.failures % FREE_ATTEMPTS == 0 {
            self.until = Some(now + THROTTLE);
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

fn too_many(wait: Duration) -> String {
    let secs = wait.as_millis().div_ceil(1000);
    format!("Too many wrong passwords. Try again in {secs} s.")
}

/// Where "this user has entered the password" lives: the id of the password
/// they entered. The OS credential store in the app, so it is per OS user;
/// memory in tests.
pub trait AccessStore: Send + Sync {
    fn load(&self) -> Option<String>;
    fn save(&self, id: &str) -> Result<(), String>;
}

#[derive(Debug, Default)]
pub struct MemoryStore(Mutex<Option<String>>);

impl AccessStore for MemoryStore {
    fn load(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn save(&self, id: &str) -> Result<(), String> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(id.to_string());
        Ok(())
    }
}

/// What the lock screen shows (`service-lock-changed`, `service_lock`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ServiceLock {
    pub locked: bool,
    pub reason: Option<LockReason>,
    pub title: Option<String>,
    pub message: Option<String>,
    pub update_url: Option<String>,
    pub min_version: Option<String>,
    pub app_version: String,
    /// a status URL is configured (the switch is on in this build)
    pub enabled: bool,
    /// unix ms of the last successful fetch (this run or an earlier one)
    pub checked_at: Option<i64>,
    /// why the last fetch failed (cleared by the next success)
    pub last_error: Option<String>,
    /// seconds the next password attempt must still wait
    pub retry_in: Option<u64>,
    /// the work this lock stopped, by kind (filled in by the app)
    pub stopped: Vec<String>,
    /// on the event that lifts a lock: the work it had stopped, to resume
    pub resume: Vec<String>,
}

// ------------------------------------------------------------------ runtime

pub struct CoreConfig {
    pub app_version: String,
    /// the validated status URL; None = the switch is off in this build
    pub url: Option<String>,
    pub require_access: bool,
    /// the file the last good document is kept in (in the app data dir)
    pub store: Option<PathBuf>,
    pub user_agent: String,
}

/// What a check or an unlock did to the lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// nothing was locked and now something is: stop the work
    Engaged,
    /// the lock lifted: resume what it stopped
    Released,
    /// still locked (or still open), but what the screen says changed
    Changed,
    Unchanged,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Persisted {
    fetched_at: i64,
    document: Value,
}

#[derive(Default)]
struct Shared {
    doc: Option<StatusDoc>,
    fetched_at: Option<i64>,
    last_error: Option<String>,
    /// the password id this user's credential store holds, once read
    granted: Option<String>,
    granted_loaded: bool,
    throttle: Throttle,
}

/// The switch and the password for one process: the last good document,
/// this user's credentials, and the gate they decide.
pub struct ServiceCore {
    cfg: CoreConfig,
    creds: Arc<dyn AccessStore>,
    gate: &'static Gate,
    shared: Mutex<Shared>,
    /// serializes fetch + apply (the poll loop and "Check again")
    check: tokio::sync::Mutex<()>,
}

impl ServiceCore {
    pub fn new(cfg: CoreConfig, creds: Arc<dyn AccessStore>, gate: &'static Gate) -> Self {
        Self {
            cfg,
            creds,
            gate,
            shared: Mutex::new(Shared::default()),
            check: tokio::sync::Mutex::new(()),
        }
    }

    fn shared(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn enabled(&self) -> bool {
        self.cfg.url.is_some()
    }

    pub fn gate(&self) -> &'static Gate {
        self.gate
    }

    /// At startup, before anything can run: apply the last good document.
    /// A build with no status URL forgets any state an earlier build left,
    /// since a lock must never outlive the switch.
    pub fn restore(&self) -> Transition {
        if !self.enabled() {
            if let Some(path) = &self.cfg.store {
                let _ = std::fs::remove_file(path);
            }
            *self.shared() = Shared::default();
            return self.settle(false);
        }
        if let Some((doc, fetched_at)) = self.read_store() {
            let mut sh = self.shared();
            sh.doc = Some(doc);
            sh.fetched_at = Some(fetched_at);
        }
        self.settle(false)
    }

    /// Recompute the lock from what is known and set the gate.
    fn settle(&self, changed: bool) -> Transition {
        let want = if self.enabled() {
            let needs_creds = {
                let sh = self.shared();
                !sh.granted_loaded && sh.doc.as_ref().is_some_and(|d| d.access.is_some())
            };
            if needs_creds {
                // Read only once a password is actually set: a build whose
                // document has none never touches the credential store.
                let granted = self.creds.load();
                let mut sh = self.shared();
                sh.granted = granted;
                sh.granted_loaded = true;
            }
            let sh = self.shared();
            lock_reason(
                sh.doc.as_ref(),
                &self.cfg.app_version,
                sh.granted.as_deref(),
                self.cfg.require_access,
            )
        } else {
            None
        };
        let was = self.gate.set(want);
        match (was, want) {
            (None, Some(_)) => Transition::Engaged,
            (Some(_), None) => Transition::Released,
            (a, b) if a != b => Transition::Changed,
            _ if changed => Transition::Changed,
            _ => Transition::Unchanged,
        }
    }

    /// Apply a document as fetched: keep it, then lock, unlock or refresh.
    pub fn apply_body(&self, bytes: &[u8]) -> Transition {
        match parse_body(bytes) {
            Some((document, doc)) => {
                let now = now_ms();
                self.write_store(&Persisted {
                    fetched_at: now,
                    document,
                });
                let changed = {
                    let mut sh = self.shared();
                    let changed = sh.doc.as_ref() != Some(&doc) || sh.last_error.is_some();
                    sh.doc = Some(doc);
                    sh.fetched_at = Some(now);
                    sh.last_error = None;
                    changed
                };
                self.settle(changed)
            }
            None => self.note_error(
                "the status document is not valid; keeping the last known state".into(),
            ),
        }
    }

    /// A failed fetch changes nothing but the reason shown on the screen.
    fn note_error(&self, error: String) -> Transition {
        log::info!("remote status: {error}");
        let changed = {
            let mut sh = self.shared();
            let changed = sh.last_error.as_deref() != Some(error.as_str());
            sh.last_error = Some(error);
            changed
        };
        self.settle(changed && self.gate.is_locked())
    }

    /// Fetch now and apply the result.
    pub async fn check_now(&self) -> Transition {
        let Some(url) = self.cfg.url.clone() else {
            return Transition::Unchanged;
        };
        let _serial = self.check.lock().await;
        match fetch(&url, &self.cfg.user_agent).await {
            Ok(body) => self.apply_body(&body),
            Err(error) => self.note_error(error),
        }
    }

    /// The password screen's Unlock. Slow on purpose: call it off the UI
    /// thread. The password is only hashed, never kept or logged.
    pub fn unlock(&self, password: &str) -> Result<Transition, String> {
        self.unlock_at(password, Instant::now())
    }

    pub fn unlock_at(&self, password: &str, now: Instant) -> Result<Transition, String> {
        let access = {
            let sh = self.shared();
            if let Some(wait) = sh.throttle.wait(now) {
                return Err(too_many(wait));
            }
            match sh.doc.as_ref().and_then(|d| d.access.clone()) {
                Some(access) => access,
                // No password (any more): nothing to unlock.
                None => {
                    drop(sh);
                    return Ok(self.settle(false));
                }
            }
        };
        if !access.verify(password) {
            let mut sh = self.shared();
            sh.throttle.failed(now);
            return Err(match sh.throttle.wait(now) {
                Some(wait) => too_many(wait),
                None => WRONG_PASSWORD.to_string(),
            });
        }
        if let Err(error) = self.creds.save(access.id()) {
            // Unlocked for this run all the same; the next start asks again.
            log::warn!("remote status: cannot remember the access password: {error}");
        }
        {
            let mut sh = self.shared();
            sh.granted = Some(access.id().to_string());
            sh.granted_loaded = true;
            sh.throttle.reset();
        }
        Ok(self.settle(true))
    }

    pub fn snapshot(&self) -> ServiceLock {
        let sh = self.shared();
        let reason = self.gate.reason();
        let doc = sh.doc.clone().unwrap_or_default();
        ServiceLock {
            locked: reason.is_some(),
            reason,
            title: doc.title,
            message: doc.message,
            update_url: doc.update_url,
            min_version: doc.min_version,
            app_version: self.cfg.app_version.clone(),
            enabled: self.enabled(),
            checked_at: sh.fetched_at,
            last_error: sh.last_error.clone(),
            retry_in: sh
                .throttle
                .wait(Instant::now())
                .map(|d| d.as_millis().div_ceil(1000) as u64),
            stopped: Vec::new(),
            resume: Vec::new(),
        }
    }

    /// How long to wait before the next check: the document's `checkMinutes`.
    pub fn interval(&self) -> Duration {
        self.shared()
            .doc
            .as_ref()
            .map(StatusDoc::interval)
            .unwrap_or(Duration::from_secs(DEFAULT_CHECK_MINUTES as u64 * 60))
    }

    /// The kept document, if there is one and it still parses. A file that
    /// does not counts as never having checked.
    fn read_store(&self) -> Option<(StatusDoc, i64)> {
        let text = std::fs::read_to_string(self.cfg.store.as_ref()?).ok()?;
        let p: Persisted = serde_json::from_str(&text).ok()?;
        if p.fetched_at <= 0 {
            return None;
        }
        let doc = parse_value(&p.document)?;
        Some((doc, p.fetched_at))
    }

    fn write_store(&self, p: &Persisted) {
        let Some(path) = &self.cfg.store else { return };
        let write = || -> std::io::Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let staged = path.with_extension("json.tmp");
            std::fs::write(&staged, serde_json::to_vec_pretty(p)?)?;
            std::fs::rename(&staged, path)
        };
        if let Err(error) = write() {
            log::warn!("remote status: cannot keep the status document: {error}");
        }
    }
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ------------------------------------------------------------------ fetching

/// An error and its causes, without the URL (the gist link stays off the
/// screen): reqwest's own message names only the kind
/// ("error following redirect"), the reason is further down the chain.
fn causes(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let line = cause.to_string();
        if !text.contains(&line) {
            text.push_str(": ");
            text.push_str(&line);
        }
        source = cause.source();
    }
    text
}

/// GET the document: cache-busting query, no cookies, 10 s, at most 64 KB,
/// redirects only to https (or a loopback test server).
pub async fn fetch(url: &str, user_agent: &str) -> Result<Vec<u8>, String> {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let mut u = url::Url::parse(url).map_err(|e| e.to_string())?;
    u.query_pairs_mut()
        .append_pair("_hs", &now_ms().to_string());
    let policy = reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= 5 {
            attempt.error("too many redirects")
        } else if attempt.url().scheme() == "https" || is_loopback(attempt.url()) {
            attempt.follow()
        } else {
            attempt.error("redirect to a non-https address")
        }
    });
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .connect_timeout(FETCH_TIMEOUT)
        .redirect(policy)
        .user_agent(user_agent)
        .build()
        .map_err(|e| format!("cannot create the HTTP client: {e}"))?;
    let mut resp = client
        .get(u)
        .header(reqwest::header::CACHE_CONTROL, "no-cache")
        .header(reqwest::header::PRAGMA, "no-cache")
        .header(reqwest::header::ACCEPT, "application/json, text/plain, */*")
        .send()
        .await
        .map_err(|e| {
            format!(
                "cannot reach the status server: {}",
                causes(&e.without_url())
            )
        })?;
    if !resp.status().is_success() {
        return Err(format!("the status server answered {}", resp.status()));
    }
    let too_large = || "the status document is larger than 64 KB; keeping the last known state";
    if resp.content_length().is_some_and(|n| n > MAX_BYTES as u64) {
        return Err(too_large().into());
    }
    let mut body = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("the status download failed: {}", causes(&e.without_url())))?
    {
        body.extend_from_slice(&chunk);
        if body.len() > MAX_BYTES {
            return Err(too_large().into());
        }
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Made with Node, as the switch CLI does it:
    /// `crypto.pbkdf2Sync("correct horse battery staple", salt, 600000, 32, "sha256")`
    /// with salt = bytes 0..16.
    pub(crate) const NODE_SALT: &str = "AAECAwQFBgcICQoLDA0ODw==";
    pub(crate) const NODE_HASH: &str = "7xdxRO7JQgy8EJPSqLNEqSvFBtDU7JwCjdGfgyTYweY=";
    pub(crate) const NODE_ID: &str = "2e3c27cab79ad82a";
    pub(crate) const NODE_PASSWORD: &str = "correct horse battery staple";

    fn access_json() -> String {
        format!(
            r#"{{"alg":"pbkdf2-sha256","iterations":600000,"salt":"{NODE_SALT}","hash":"{NODE_HASH}","id":"{NODE_ID}"}}"#
        )
    }

    #[test]
    fn parses_ok_and_maintenance() {
        let d = parse(br#"{"status":"ok"}"#).unwrap();
        assert_eq!(d.status, RemoteState::Ok);
        assert_eq!(d.access, None);
        assert_eq!(d.interval(), Duration::from_secs(300));
        let d = parse(
            br#"{"status":"Maintenance","title":"  Work is in progress ","message":"Back soon","minVersion":"2.30.0",
                "updateUrl":"https://github.com/AdkHex/AudioSyncMaster/releases/latest","checkMinutes":2,"extra":{"ignored":true}}"#,
        )
        .unwrap();
        assert_eq!(d.status, RemoteState::Maintenance);
        assert_eq!(d.title.as_deref(), Some("Work is in progress"));
        assert_eq!(d.message.as_deref(), Some("Back soon"));
        assert_eq!(d.min_version.as_deref(), Some("2.30.0"));
        assert_eq!(
            d.update_url.as_deref(),
            Some("https://github.com/AdkHex/AudioSyncMaster/releases/latest")
        );
        assert_eq!(d.interval(), Duration::from_secs(120));
    }

    #[test]
    fn garbage_is_no_information() {
        for bad in [
            &b""[..],
            b"null",
            b"[]",
            b"<html>503</html>",
            br#"{"status":"locked"}"#,
            br#"{"status":1}"#,
            br#"{"title":"x"}"#,
        ] {
            assert_eq!(parse(bad), None, "{}", String::from_utf8_lossy(bad));
        }
        // Invalid optional fields are dropped; the document stays usable.
        let d = parse(
            br#"{"status":"ok","updateUrl":"javascript:alert(1)","minVersion":"latest","checkMinutes":999,"title":""}"#,
        )
        .unwrap();
        assert_eq!(
            (&d.update_url, &d.min_version, &d.title),
            (&None, &None, &None)
        );
        assert_eq!(d.interval(), Duration::from_secs(3600));
        assert_eq!(
            parse(br#"{"status":"ok","checkMinutes":0}"#)
                .unwrap()
                .interval(),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn oversize_is_rejected() {
        let mut big = br#"{"status":"maintenance","message":""#.to_vec();
        big.extend(std::iter::repeat_n(b'a', MAX_BYTES));
        big.extend_from_slice(br#""}"#);
        assert_eq!(parse(&big), None);
        // Long fields are cut, not rejected.
        let long = format!(
            r#"{{"status":"maintenance","title":"{}"}}"#,
            "t".repeat(1000)
        );
        assert_eq!(
            parse(long.as_bytes())
                .unwrap()
                .title
                .unwrap()
                .chars()
                .count(),
            200
        );
    }

    #[test]
    fn version_compare() {
        assert!(version_older("1.0.0", "1.0.10"));
        assert!(version_older("1.0.9", "1.0.10"));
        assert!(!version_older("1.0.10", "1.0.9"));
        assert!(!version_older("1.0.0", "1.0.0"));
        assert!(!version_older("1.0", "1.0.0"));
        assert!(version_older("1.0", "1.0.1"));
        assert!(version_older("0.9.9", "1"));
        assert!(version_older("2.9.0", "2.10.0"));
        assert!(!version_older("v2.21.0", "2.10.0"));
        assert!(!version_older("1.0.0-beta.1", "1.0.0"));
        // Unknown versions never lock.
        assert!(!version_older("test", "9.9.9"));
        assert!(!version_older("1.0.0", "garbage"));
    }

    #[test]
    fn url_rules() {
        assert!(validate_url("https://gist.githubusercontent.com/a/b/raw/s.json", false).is_ok());
        assert!(validate_url("http://example.com/s.json", false).is_err());
        assert!(validate_url("http://example.com/s.json", true).is_err());
        assert!(validate_url("http://127.0.0.1:8799/status.json", false).is_err());
        assert!(validate_url("http://127.0.0.1:8799/status.json", true).is_ok());
        assert!(validate_url("http://localhost:8799/status.json", true).is_ok());
        assert!(validate_url("file:///etc/passwd", true).is_err());
        assert!(validate_url("", true).is_err());
    }

    #[test]
    fn verifies_the_password_node_hashed() {
        let doc = parse(format!(r#"{{"status":"ok","access":{}}}"#, access_json()).as_bytes())
            .expect("the CLI's access block parses");
        let access = doc.access.expect("access");
        assert_eq!(access.id(), NODE_ID);
        assert!(access.verify(NODE_PASSWORD));
        assert!(!access.verify("correct horse battery stapl"));
        assert!(!access.verify(""));
        assert!(!access.verify("Correct horse battery staple"));
    }

    #[test]
    fn the_id_is_sha256_of_salt_and_hash() {
        let b64 = base64::engine::general_purpose::STANDARD;
        let salt = b64.decode(NODE_SALT).unwrap();
        let hash = b64.decode(NODE_HASH).unwrap();
        assert_eq!(access_id(&salt, &hash), NODE_ID);
        assert_eq!(NODE_ID.len(), 16);
        // A changed password (new salt) is a new id.
        assert_ne!(access_id(&[0u8; 16], &hash), NODE_ID);
    }

    #[test]
    fn a_broken_access_block_is_no_information() {
        let with = |access: &str| format!(r#"{{"status":"ok","access":{access}}}"#);
        let good = access_json();
        for bad in [
            good.replace("pbkdf2-sha256", "md5"),
            good.replace("600000", "10"),
            good.replace("600000", "\"600000\""),
            good.replace(NODE_HASH, "c2hvcnQ="),
            good.replace(NODE_SALT, "!!!"),
            good.replace(NODE_ID, "0000000000000000"),
            "\"letmein\"".to_string(),
            "{}".to_string(),
        ] {
            assert_eq!(parse(with(&bad).as_bytes()), None, "{bad}");
        }
        // No id is fine (it is computed), and null means no password.
        let no_id = good.replace(&format!(r#","id":"{NODE_ID}""#), "");
        assert_eq!(
            parse(with(&no_id).as_bytes()).unwrap().access.unwrap().id(),
            NODE_ID
        );
        assert_eq!(parse(with("null").as_bytes()).unwrap().access, None);
    }

    #[test]
    fn lock_reasons_and_their_order() {
        let mut d =
            parse(format!(r#"{{"status":"ok","access":{}}}"#, access_json()).as_bytes()).unwrap();
        // Never read a document: open, unless the build requires access.
        assert_eq!(lock_reason(None, "2.10.0", None, false), None);
        assert_eq!(
            lock_reason(None, "2.10.0", None, true),
            Some(LockReason::Connect)
        );
        // A password this user has not entered, or an older one.
        assert_eq!(
            lock_reason(Some(&d), "2.10.0", None, true),
            Some(LockReason::Password)
        );
        assert_eq!(
            lock_reason(Some(&d), "2.10.0", Some("0123456789abcdef"), true),
            Some(LockReason::Password)
        );
        assert_eq!(lock_reason(Some(&d), "2.10.0", Some(NODE_ID), true), None);
        // "Update required" wins over the password, maintenance over both.
        d.min_version = Some("2.11.0".into());
        assert_eq!(
            lock_reason(Some(&d), "2.10.0", None, true),
            Some(LockReason::Update)
        );
        assert_eq!(lock_reason(Some(&d), "2.11.0", Some(NODE_ID), true), None);
        d.status = RemoteState::Maintenance;
        assert_eq!(
            lock_reason(Some(&d), "2.11.0", None, true),
            Some(LockReason::Maintenance)
        );
        // No access block: no password.
        d.status = RemoteState::Ok;
        d.min_version = None;
        d.access = None;
        assert_eq!(lock_reason(Some(&d), "2.10.0", None, true), None);
    }

    #[test]
    fn the_gate_refuses_with_the_locks_message() {
        let gate = Gate::new();
        assert_eq!(gate.ensure_open(), Ok(()));
        for (reason, message) in [
            (LockReason::Maintenance, PAUSED_ERROR),
            (LockReason::Update, PAUSED_ERROR),
            (LockReason::Password, LOCKED_ERROR),
            (LockReason::Connect, LOCKED_ERROR),
        ] {
            gate.set(Some(reason));
            assert_eq!(gate.reason(), Some(reason));
            assert_eq!(gate.ensure_open(), Err(message.to_string()));
        }
        assert_eq!(gate.set(None), Some(LockReason::Connect));
        assert!(!gate.is_locked());
    }

    #[test]
    fn five_wrong_passwords_make_the_next_wait() {
        let mut t = Throttle::default();
        let start = Instant::now();
        for _ in 0..FREE_ATTEMPTS - 1 {
            t.failed(start);
            assert_eq!(t.wait(start), None);
        }
        t.failed(start);
        assert_eq!(t.wait(start), Some(THROTTLE));
        assert_eq!(
            t.wait(start + Duration::from_secs(10)),
            Some(Duration::from_secs(20))
        );
        assert_eq!(t.wait(start + THROTTLE), None);
        // The next four after the wait are free again, the fifth waits.
        let later = start + THROTTLE;
        for _ in 0..FREE_ATTEMPTS - 1 {
            t.failed(later);
            assert_eq!(t.wait(later), None);
        }
        t.failed(later);
        assert_eq!(t.wait(later), Some(THROTTLE));
        t.reset();
        assert_eq!(t.wait(later), None);
    }

    fn leaked_gate() -> &'static Gate {
        Box::leak(Box::new(Gate::new()))
    }

    fn core(store: Arc<dyn AccessStore>, require_access: bool) -> ServiceCore {
        ServiceCore::new(
            CoreConfig {
                app_version: "2.10.0".into(),
                url: Some("https://example.invalid/status.json".into()),
                require_access,
                store: None,
                user_agent: "test".into(),
            },
            store,
            leaked_gate(),
        )
    }

    #[test]
    fn unlock_refuses_a_wrong_password_and_remembers_the_right_one() {
        let creds: Arc<MemoryStore> = Arc::new(MemoryStore::default());
        let c = core(creds.clone(), true);
        assert_eq!(c.restore(), Transition::Engaged);
        assert_eq!(c.snapshot().reason, Some(LockReason::Connect));
        let doc = format!(r#"{{"status":"ok","access":{}}}"#, access_json());
        assert_eq!(c.apply_body(doc.as_bytes()), Transition::Changed);
        assert_eq!(c.snapshot().reason, Some(LockReason::Password));
        assert_eq!(c.unlock("nope"), Err(WRONG_PASSWORD.to_string()));
        assert!(c.gate().is_locked());
        assert_eq!(creds.load(), None);
        assert_eq!(c.unlock(NODE_PASSWORD), Ok(Transition::Released));
        assert!(!c.gate().is_locked());
        assert_eq!(creds.load().as_deref(), Some(NODE_ID));
    }

    #[test]
    fn unlock_waits_after_five_wrong_passwords() {
        let c = core(Arc::new(MemoryStore::default()), false);
        let doc = format!(r#"{{"status":"ok","access":{}}}"#, access_json());
        c.apply_body(doc.as_bytes());
        let now = Instant::now();
        for _ in 0..FREE_ATTEMPTS - 1 {
            assert_eq!(c.unlock_at("wrong", now), Err(WRONG_PASSWORD.to_string()));
        }
        let fifth = c.unlock_at("wrong", now).unwrap_err();
        assert!(fifth.starts_with("Too many wrong passwords"), "{fifth}");
        // Even the right password waits until the time is up.
        let early = c
            .unlock_at(NODE_PASSWORD, now + Duration::from_secs(1))
            .unwrap_err();
        assert!(early.contains("29 s"), "{early}");
        assert_eq!(
            c.unlock_at(NODE_PASSWORD, now + THROTTLE),
            Ok(Transition::Released)
        );
    }
}
