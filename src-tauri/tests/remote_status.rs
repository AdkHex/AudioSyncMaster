//! The remote switch and the access password end to end, against a real
//! HTTP server on 127.0.0.1: what a copy does when the document says
//! maintenance and then ok, that the lock sticks across a restart with the
//! server gone, the password from set to changed to off, a fresh copy that
//! requires access and cannot reach the server, and that bad answers never
//! lock anyone. See docs/REMOTE-STATUS.md.
//!
//!     cargo test --manifest-path src-tauri/Cargo.toml --test remote_status

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use audiosyncmaster_lib::remote_status::{
    AccessStore, CoreConfig, Gate, LockReason, MemoryStore, ServiceCore, Transition, LOCKED_ERROR,
    PAUSED_ERROR, WRONG_PASSWORD,
};

/// Made by Node the way `scripts/audiosyncmaster-status.mjs password` does
/// it: pbkdf2Sync(password, salt, 600000, 32, "sha256"), id =
/// sha256(salt || hash) to 16 hex digits.
const FIRST: Secret = Secret {
    password: "correct horse battery staple",
    salt: "AAECAwQFBgcICQoLDA0ODw==",
    hash: "7xdxRO7JQgy8EJPSqLNEqSvFBtDU7JwCjdGfgyTYweY=",
    id: "2e3c27cab79ad82a",
};
const SECOND: Secret = Secret {
    password: "a new passphrase for everyone",
    salt: "EBESExQVFhcYGRobHB0eHw==",
    hash: "PNpKKwO2tAgwtx9HW2/EoUUwjYsqKTmOexWl8RarcNg=",
    id: "c819fb8cc4727152",
};

struct Secret {
    password: &'static str,
    salt: &'static str,
    hash: &'static str,
    id: &'static str,
}

impl Secret {
    fn doc(&self, status: &str) -> String {
        format!(
            r#"{{"status":"{status}","checkMinutes":5,"access":{{"alg":"pbkdf2-sha256","iterations":600000,"salt":"{}","hash":"{}","id":"{}"}}}}"#,
            self.salt, self.hash, self.id
        )
    }
}

/// What the test server answers: a status and a body, or a redirect.
#[derive(Clone)]
enum Answer {
    Json(String),
    Status(u16),
    Redirect(String),
}

/// A one-file HTTP server whose answer the test changes as it goes, and
/// which keeps every request it was sent.
struct Server {
    url: String,
    answer: Arc<Mutex<Answer>>,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Server {
    fn start(first: Answer) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!(
            "http://127.0.0.1:{}/status.json",
            listener.local_addr().unwrap().port()
        );
        let answer = Arc::new(Mutex::new(first));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (a, r) = (answer.clone(), requests.clone());
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (a, r) = (a.clone(), r.clone());
                thread::spawn(move || serve(stream, &a, &r));
            }
        });
        Self {
            url,
            answer,
            requests,
        }
    }

    fn set(&self, answer: Answer) {
        *self.answer.lock().unwrap() = answer;
    }

    fn last_request(&self) -> String {
        self.requests
            .lock()
            .unwrap()
            .last()
            .cloned()
            .unwrap_or_default()
    }
}

fn serve(mut stream: TcpStream, answer: &Mutex<Answer>, requests: &Mutex<Vec<String>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut head = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        head.push_str(&line);
    }
    requests.lock().unwrap().push(head);
    let response = match answer.lock().unwrap().clone() {
        Answer::Json(body) => format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        ),
        Answer::Status(code) => {
            format!("HTTP/1.1 {code} Nope\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
        }
        Answer::Redirect(to) => format!(
            "HTTP/1.1 302 Found\r\nlocation: {to}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
        ),
    };
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    let _ = stream.read(&mut [0u8; 1]);
}

/// An app data dir of its own.
fn store_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("asm-remote-status-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One run of the app: its own gate, the user's credential store and data
/// dir shared with the runs before it.
fn launch(
    url: &str,
    dir: &std::path::Path,
    creds: Arc<dyn AccessStore>,
    require_access: bool,
) -> ServiceCore {
    let gate: &'static Gate = Box::leak(Box::new(Gate::new()));
    ServiceCore::new(
        CoreConfig {
            app_version: "2.22.0".into(),
            url: Some(url.into()),
            require_access,
            store: Some(dir.join("remote-status.json")),
            user_agent: "AudioSyncMaster/test".into(),
        },
        creds,
        gate,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn maintenance_locks_and_ok_releases() {
    let server = Server::start(Answer::Json(r#"{"status":"ok","checkMinutes":5}"#.into()));
    let dir = store_dir("maintenance");
    let app = launch(&server.url, &dir, Arc::new(MemoryStore::default()), true);
    assert_eq!(
        app.restore(),
        Transition::Engaged,
        "first use: connect once"
    );
    assert_eq!(app.gate().reason(), Some(LockReason::Connect));
    assert_eq!(app.check_now().await, Transition::Released);

    // The request: cache-busting, no-cache, no cookies.
    let request = server.last_request();
    assert!(request.starts_with("GET /status.json?_hs="), "{request}");
    let lower = request.to_ascii_lowercase();
    assert!(lower.contains("cache-control: no-cache"), "{request}");
    assert!(!lower.contains("cookie"), "{request}");

    server.set(Answer::Json(
        r#"{"status":"maintenance","title":"Work is in progress","message":"Back in ten minutes."}"#
            .into(),
    ));
    assert_eq!(app.check_now().await, Transition::Engaged);
    assert_eq!(app.gate().ensure_open(), Err(PAUSED_ERROR.to_string()));
    let shown = app.snapshot();
    assert!(shown.locked);
    assert_eq!(shown.reason, Some(LockReason::Maintenance));
    assert_eq!(shown.message.as_deref(), Some("Back in ten minutes."));
    // The same answer again changes nothing.
    assert_eq!(app.check_now().await, Transition::Unchanged);

    server.set(Answer::Json(r#"{"status":"ok"}"#.into()));
    assert_eq!(app.check_now().await, Transition::Released);
    assert_eq!(app.gate().ensure_open(), Ok(()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_lock_sticks_across_a_restart_with_the_server_gone() {
    let server = Server::start(Answer::Json(r#"{"status":"maintenance"}"#.into()));
    let dir = store_dir("sticky");
    let creds: Arc<dyn AccessStore> = Arc::new(MemoryStore::default());
    let first = launch(&server.url, &dir, creds.clone(), true);
    first.restore();
    assert_eq!(
        first.check_now().await,
        Transition::Changed,
        "connect -> maintenance"
    );
    assert_eq!(first.gate().reason(), Some(LockReason::Maintenance));

    // Restart with the status server unreachable: still locked, from disk,
    // before any check.
    let dead = "http://127.0.0.1:9/status.json";
    let second = launch(dead, &dir, creds.clone(), true);
    assert_eq!(second.restore(), Transition::Engaged);
    assert_eq!(second.gate().reason(), Some(LockReason::Maintenance));
    assert_eq!(
        second.check_now().await,
        Transition::Changed,
        "only the error shows"
    );
    assert_eq!(second.gate().reason(), Some(LockReason::Maintenance));
    assert!(second
        .snapshot()
        .last_error
        .unwrap()
        .contains("cannot reach the status server"));

    // Back online with ok: open, and the next start is open too.
    let third = launch(&server.url, &dir, creds.clone(), true);
    third.restore();
    server.set(Answer::Json(r#"{"status":"ok"}"#.into()));
    assert_eq!(third.check_now().await, Transition::Released);
    let fourth = launch(dead, &dir, creds, true);
    assert_eq!(fourth.restore(), Transition::Unchanged);
    assert!(!fourth.gate().is_locked());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_answers_never_lock_and_keep_the_last_state() {
    let server = Server::start(Answer::Json(r#"{"status":"ok"}"#.into()));
    let dir = store_dir("garbage");
    let app = launch(&server.url, &dir, Arc::new(MemoryStore::default()), false);
    app.restore();
    // The first document: new information, still open.
    assert_eq!(app.check_now().await, Transition::Changed);
    assert!(!app.gate().is_locked());

    let oversize = format!(
        r#"{{"status":"maintenance","message":"{}"}}"#,
        "a".repeat(70_000)
    );
    for answer in [
        Answer::Json("<html>503 Service Unavailable</html>".into()),
        Answer::Json(r#"{"status":"locked"}"#.into()),
        Answer::Json(oversize),
        Answer::Json(r#"{"status":"ok","access":{"alg":"md5","hash":"x"}}"#.into()),
        Answer::Status(500),
        Answer::Status(404),
        Answer::Redirect("http://example.com/status.json".into()),
    ] {
        app.check_now().await;
        assert!(!app.gate().is_locked(), "a bad answer locked the app");
        server.set(answer);
    }
    app.check_now().await;
    assert!(!app.gate().is_locked());
    let error = app.snapshot().last_error.unwrap_or_default();
    assert!(error.contains("non-https"), "{error}");

    // And while locked, a bad answer does not unlock either.
    server.set(Answer::Json(r#"{"status":"maintenance"}"#.into()));
    assert_eq!(app.check_now().await, Transition::Engaged);
    server.set(Answer::Status(500));
    app.check_now().await;
    assert_eq!(app.gate().reason(), Some(LockReason::Maintenance));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_password_from_set_to_changed_to_off() {
    let server = Server::start(Answer::Json(FIRST.doc("ok")));
    let dir = store_dir("password");
    let creds: Arc<MemoryStore> = Arc::new(MemoryStore::default());

    // Set: the app asks.
    let app = launch(&server.url, &dir, creds.clone(), true);
    app.restore();
    assert_eq!(
        app.check_now().await,
        Transition::Changed,
        "connect -> password"
    );
    assert_eq!(app.gate().reason(), Some(LockReason::Password));
    assert_eq!(app.gate().ensure_open(), Err(LOCKED_ERROR.to_string()));

    // A wrong password is refused; the right one unlocks and is remembered.
    assert_eq!(app.unlock("letmein"), Err(WRONG_PASSWORD.to_string()));
    assert!(app.gate().is_locked());
    assert_eq!(creds.load(), None, "nothing kept for a wrong password");
    assert_eq!(app.unlock(FIRST.password), Ok(Transition::Released));
    assert_eq!(
        creds.load().as_deref(),
        Some(FIRST.id),
        "only the id is kept"
    );

    // Restart: no prompt, even offline.
    let dead = "http://127.0.0.1:9/status.json";
    let again = launch(dead, &dir, creds.clone(), true);
    assert_eq!(again.restore(), Transition::Unchanged);
    assert!(!again.gate().is_locked());

    // The password changes: asked once more, and the old one no longer works.
    server.set(Answer::Json(SECOND.doc("ok")));
    let app = launch(&server.url, &dir, creds.clone(), true);
    app.restore();
    assert_eq!(app.check_now().await, Transition::Engaged);
    assert_eq!(app.gate().reason(), Some(LockReason::Password));
    assert_eq!(app.unlock(FIRST.password), Err(WRONG_PASSWORD.to_string()));
    assert_eq!(app.unlock(SECOND.password), Ok(Transition::Released));
    assert_eq!(creds.load().as_deref(), Some(SECOND.id));

    // Maintenance wins over the password screen.
    let other_user = launch(&server.url, &dir, Arc::new(MemoryStore::default()), true);
    other_user.restore();
    assert_eq!(
        other_user.gate().reason(),
        Some(LockReason::Password),
        "another OS user is asked"
    );
    server.set(Answer::Json(SECOND.doc("maintenance")));
    other_user.check_now().await;
    assert_eq!(other_user.gate().reason(), Some(LockReason::Maintenance));

    // Off: nobody is asked.
    server.set(Answer::Json(r#"{"status":"ok","checkMinutes":5}"#.into()));
    assert_eq!(other_user.check_now().await, Transition::Released);
    let fresh = launch(&server.url, &dir, Arc::new(MemoryStore::default()), true);
    assert_eq!(fresh.restore(), Transition::Unchanged);
    assert!(!fresh.gate().is_locked());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fresh_copy_that_requires_access_waits_to_connect_once() {
    let dir = store_dir("fresh");
    let dead = "http://127.0.0.1:9/status.json";
    let app = launch(dead, &dir, Arc::new(MemoryStore::default()), true);
    assert_eq!(app.restore(), Transition::Engaged);
    app.check_now().await;
    assert_eq!(app.gate().reason(), Some(LockReason::Connect));
    assert_eq!(app.gate().ensure_open(), Err(LOCKED_ERROR.to_string()));

    // Without requireAccess the same copy runs normally.
    let open = launch(dead, &dir, Arc::new(MemoryStore::default()), false);
    assert_eq!(open.restore(), Transition::Unchanged);
    assert!(!open.gate().is_locked());

    // One successful check and it opens.
    let server = Server::start(Answer::Json(r#"{"status":"ok"}"#.into()));
    let online = launch(&server.url, &dir, Arc::new(MemoryStore::default()), true);
    online.restore();
    assert_eq!(online.check_now().await, Transition::Released);
    let _ = std::fs::remove_dir_all(&dir);
}
