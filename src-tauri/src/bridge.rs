//! Long-lived connection to the Python analysis bridge.
//!
//! The original host spawned a fresh process per run, wrote the whole request
//! to stdin, then read stdout to EOF. That shape made real cancellation
//! impossible: the only way to stop work was to kill the process, which
//! orphaned every ffmpeg child it had spawned.
//!
//! Here the bridge is kept alive and spoken to line-by-line, so a `cancel`
//! command reaches the running batch and lets Python tear down its own
//! subprocesses in an orderly way.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A running bridge process plus the channel its stdout reader publishes to.
pub struct Bridge {
    child: Child,
    stdin: ChildStdin,
    events: Receiver<Value>,
}

impl Bridge {
    /// Start the bridge, preferring the bundled sidecar over a dev interpreter.
    pub fn spawn(app: &AppHandle) -> Result<Self, String> {
        let mut command = build_command(app)?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);

        let mut child = command
            .spawn()
            .map_err(|err| format!("Could not start the analysis engine: {err}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Could not open the engine's input stream".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Could not open the engine's output stream".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "Could not open the engine's error stream".to_string())?;

        let (sender, receiver): (Sender<Value>, Receiver<Value>) = channel();

        // stdout is drained on its own thread from the moment the process
        // starts, so writing a large request can never deadlock against a full
        // output pipe.
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(trimmed) {
                    Ok(value) => {
                        if sender.send(value).is_err() {
                            break; // Receiver dropped; the run is over.
                        }
                    }
                    Err(_) => {
                        let _ = sender.send(serde_json::json!({
                            "type": "log",
                            "message": format!("Unparsed engine output: {trimmed}"),
                        }));
                    }
                }
            }
        });

        let app_for_stderr = app.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                if !line.trim().is_empty() {
                    let _ = app_for_stderr.emit("sync-log", line);
                }
            }
        });

        Ok(Self {
            child,
            stdin,
            events: receiver,
        })
    }

    /// Send one command. Commands are newline-delimited JSON.
    pub fn send(&mut self, payload: &Value) -> Result<(), String> {
        let line = format!(
            "{}\n",
            serde_json::to_string(payload).map_err(|e| e.to_string())?
        );
        self.stdin
            .write_all(line.as_bytes())
            .map_err(|err| format!("Lost connection to the analysis engine: {err}"))?;
        self.stdin
            .flush()
            .map_err(|err| format!("Lost connection to the analysis engine: {err}"))
    }

    pub fn events(&self) -> &Receiver<Value> {
        &self.events
    }

    /// Stop the bridge, asking politely before killing it.
    pub fn shutdown(&mut self) {
        let _ = self.send(&serde_json::json!({ "command": "shutdown" }));
        // Closing stdin ends the read loop even if the command was not read.
        let _ = self.stdin.flush();
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            _ => {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Shared handle so commands from different invocations reach the same process.
#[derive(Clone, Default)]
pub struct BridgeHandle(Arc<Mutex<Option<Bridge>>>);

impl BridgeHandle {
    /// Run `action` against a live bridge, starting one if necessary.
    pub fn with<T>(
        &self,
        app: &AppHandle,
        action: impl FnOnce(&mut Bridge) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(Bridge::spawn(app)?);
        }
        let bridge = guard.as_mut().expect("bridge present");
        match action(bridge) {
            Ok(value) => Ok(value),
            Err(err) => {
                // A failed exchange usually means the process died; drop it so
                // the next call starts a healthy one rather than reusing a
                // half-broken pipe.
                *guard = None;
                Err(err)
            }
        }
    }

    /// Send a command without waiting for a reply. Used for cancellation, which
    /// must not queue behind the run it is trying to stop.
    pub fn send_now(&self, payload: &Value) -> Result<(), String> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())?;
        match guard.as_mut() {
            Some(bridge) => bridge.send(payload),
            None => Err("The analysis engine is not running".to_string()),
        }
    }

    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(mut bridge) = guard.take() {
                bridge.shutdown();
            }
        }
    }
}

/// Locate the sidecar, falling back to a development Python interpreter.
fn build_command(app: &AppHandle) -> Result<Command, String> {
    if let Some(sidecar) = find_sidecar(app) {
        let _ = app.emit("sync-log", format!("Engine: {}", sidecar.to_string_lossy()));
        return Ok(Command::new(sidecar));
    }

    let root = project_root(app);
    let bridge_script = root.join("python").join("bridge.py");
    if !bridge_script.exists() {
        return Err(
            "The analysis engine is missing. Build the sidecar with dev.sh (macOS/Linux) \
             or dev.bat (Windows), or run from a source checkout."
                .to_string(),
        );
    }

    let interpreter = find_python(&root);
    let _ = app.emit(
        "sync-log",
        format!(
            "Engine (development): {} {}",
            interpreter.to_string_lossy(),
            bridge_script.to_string_lossy()
        ),
    );

    let mut command = Command::new(interpreter);
    command.arg(bridge_script);
    // Run from the project root so the engine resolves its own package.
    command.current_dir(root);
    Ok(command)
}

/// Locate the packaged analysis engine.
///
/// The engine ships as a PyInstaller *directory* build under
/// `resources/engine/`, not as a single-file executable. A onefile build
/// re-extracts its entire ~47MB payload to a temp directory on every launch,
/// which measured 32-63 seconds per run on macOS once Gatekeeper rescanned the
/// unpacked copy. The directory build starts in ~0.12s because nothing is
/// unpacked at all.
fn find_sidecar(app: &AppHandle) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "audiosync-cli.exe"
    } else {
        "audiosync-cli"
    };

    let mut candidates: Vec<PathBuf> = Vec::new();

    // Packaged layout: resources/engine/audiosync-cli[.exe]
    if let Ok(dir) = app
        .path()
        .resolve("resources/engine", tauri::path::BaseDirectory::Resource)
    {
        candidates.push(dir.join(exe_name));
        // PyInstaller >= 6 nests the payload under _internal on some platforms.
        candidates.push(dir.join("audiosync-cli").join(exe_name));
    }

    // Development layout: built into src-tauri/resources/engine by dev.sh.
    if let Ok(exe) = std::env::current_exe() {
        let mut cursor = exe.parent().map(PathBuf::from);
        while let Some(dir) = cursor {
            candidates.push(
                dir.join("src-tauri")
                    .join("resources")
                    .join("engine")
                    .join(exe_name),
            );
            cursor = dir.parent().map(PathBuf::from);
        }
    }

    candidates.into_iter().find(|path| path.is_file())
}

/// Resolve the project root from the executable location rather than the
/// process CWD, which the original code depended on and which is whatever
/// directory the app happened to be launched from.
fn project_root(app: &AppHandle) -> PathBuf {
    if let Ok(dir) = std::env::var("AUDIOSYNC_PROJECT_ROOT") {
        let path = PathBuf::from(dir);
        if path.join("python").join("bridge.py").is_file() {
            return path;
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        let mut cursor = exe.parent().map(PathBuf::from);
        while let Some(dir) = cursor {
            if dir.join("python").join("bridge.py").is_file() {
                return dir;
            }
            cursor = dir.parent().map(PathBuf::from);
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("python").join("bridge.py").is_file() {
            return cwd;
        }
    }

    let _ = app;
    PathBuf::from(".")
}

fn find_python(root: &Path) -> PathBuf {
    let venv = if cfg!(windows) {
        root.join("python")
            .join(".venv")
            .join("Scripts")
            .join("python.exe")
    } else {
        root.join("python").join(".venv").join("bin").join("python")
    };
    if venv.is_file() {
        return venv;
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}
