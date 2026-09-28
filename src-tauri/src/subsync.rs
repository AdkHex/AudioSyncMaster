//! Subsync: the subtitle tools' commands.
//!
//! Long work -- a queue of jobs, installing an engine pack -- runs on the
//! main engine, the one the Stop button (`cancel_sync`) reaches. Quick
//! questions -- probing files, listing engines, loading a subtitle for the
//! review table -- go to a second engine process of their own, so adding a
//! file while an hour-long transcription runs answers at once instead of
//! queueing behind it.
//!
//! Every engine event that is not the terminal answer is forwarded to the
//! webview as `subsync-event`, unchanged; the frontend switches on `type`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, Runtime, State, Window};
use tauri_plugin_dialog::DialogExt;

use crate::bridge::BridgeHandle;
use crate::{file_item, FileItem, PickResponse, AUDIO_EXTENSIONS, EVENT_TIMEOUT, VIDEO_EXTENSIONS};

/// Subtitle files the tools take. `.sub` is both MicroDVD text and DVD
/// VobSub pictures; the engine tells them apart by content. `.xml` is
/// accepted when picked or dropped on its own (TTML often arrives so), but
/// not collected from a dropped folder, where it is usually something else.
pub(crate) const SUBTITLE_EXTENSIONS: &[&str] = &[
    "srt", "ass", "ssa", "vtt", "ttml", "dfxp", "sub", "idx", "sup", "sbv",
];

/// Services whose API keys the app stores for the translation and OCR
/// engines. Anything else is refused, so the store cannot become a general
/// key-value dump.
const SECRET_NAMES: &[&str] = &["anthropic", "openai", "deepl", "google"];

/// The second engine, for quick questions.
#[derive(Clone, Default)]
pub(crate) struct SubsInfoBridge(pub BridgeHandle);

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_lowercase()
}

/// "video", "audio" or "subtitle" for a file the tools can use, else None.
fn kind_of(path: &Path, from_folder: bool) -> Option<&'static str> {
    let ext = extension(path);
    let ext = ext.as_str();
    if VIDEO_EXTENSIONS.contains(&ext) {
        Some("video")
    } else if AUDIO_EXTENSIONS.contains(&ext) {
        Some("audio")
    } else if SUBTITLE_EXTENSIONS.contains(&ext) || (!from_folder && ext == "xml") {
        Some("subtitle")
    } else {
        None
    }
}

fn accepts(accept: &str, kind: &str) -> bool {
    match accept {
        "video" => kind == "video",
        "subtitle" => kind == "subtitle",
        "media" => kind == "video" || kind == "audio",
        _ => true,
    }
}

fn items_for(paths: impl IntoIterator<Item = PathBuf>, accept: &str) -> Vec<FileItem> {
    let mut items = Vec::new();
    for path in paths {
        if path.is_dir() {
            let Ok(entries) = fs::read_dir(&path) else {
                continue;
            };
            let mut inside: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            inside.sort();
            for child in inside {
                let hidden = child
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'));
                if hidden || !child.is_file() {
                    continue;
                }
                if let Some(kind) = kind_of(&child, true) {
                    if accepts(accept, kind) {
                        items.push(file_item(&child, kind));
                    }
                }
            }
        } else if path.is_file() {
            if let Some(kind) = kind_of(&path, false) {
                if accepts(accept, kind) {
                    items.push(file_item(&path, kind));
                }
            }
        }
    }
    items.dedup_by(|a, b| a.path == b.path);
    items
}

// ------------------------------------------------------------------ pickers

/// Several files at once. `accept` narrows the dialog: "video", "subtitle",
/// "media", or anything else for every file the tools can use.
#[tauri::command]
pub(crate) async fn subs_pick_files<R: Runtime>(
    window: Window<R>,
    accept: String,
) -> Result<PickResponse, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut dialog = window.dialog().file();
    match accept.as_str() {
        "video" => dialog = dialog.add_filter("Video", VIDEO_EXTENSIONS),
        "subtitle" => {
            let mut all: Vec<&str> = SUBTITLE_EXTENSIONS.to_vec();
            all.push("xml");
            dialog = dialog.add_filter("Subtitles", &all);
        }
        _ => {}
    }
    dialog.pick_files(move |paths| {
        let _ = tx.send(paths.unwrap_or_default());
    });
    let paths = tauri::async_runtime::spawn_blocking(move || rx.recv().ok().unwrap_or_default())
        .await
        .map_err(|err| err.to_string())?;
    let paths: Vec<PathBuf> = paths
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .collect();
    let folder = paths
        .first()
        .and_then(|p| p.parent())
        .map(|p| p.to_string_lossy().to_string());
    Ok(PickResponse {
        folder,
        files: items_for(paths, &accept),
    })
}

#[tauri::command]
pub(crate) async fn subs_pick_folder<R: Runtime>(
    window: Window<R>,
) -> Result<Option<String>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    window.dialog().file().pick_folder(move |path| {
        let _ = tx.send(path.and_then(|p| p.into_path().ok()));
    });
    let folder = tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
        .await
        .map_err(|err| err.to_string())?;
    Ok(folder.map(|p| p.to_string_lossy().to_string()))
}

/// Dropped paths, folders expanded, keeping only files the tools can use.
#[tauri::command]
pub(crate) fn subs_resolve_dropped(paths: Vec<String>) -> Vec<FileItem> {
    items_for(paths.into_iter().map(PathBuf::from), "any")
}

// --------------------------------------------------------- engine exchange

/// Send one command and pump its events until the terminal answer, which
/// is returned (and also forwarded, so listeners see every event). Logs go
/// to the console as well, prefixed with the job they belong to.
fn exchange<R: Runtime>(
    app: &AppHandle<R>,
    handle: &BridgeHandle,
    payload: Value,
    terminal: &str,
    timeout: Duration,
) -> Result<Value, String> {
    handle.with(app, |bridge| {
        bridge.send(&payload)?;
        loop {
            let event = match bridge.events().recv_timeout(timeout) {
                Ok(event) => event,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    return Err("The analysis engine stopped responding.".into());
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("The analysis engine exited unexpectedly.".into());
                }
            };
            let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
            match kind {
                "log" => {
                    if let Some(message) = event.get("message").and_then(Value::as_str) {
                        let _ = app.emit("sync-log", message);
                    }
                }
                "error" => {
                    let message = event
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("Unknown engine error");
                    let _ = app.emit("sync-log", format!("Error: {message}"));
                }
                "subsJobLog" => {
                    if let Some(message) = event.get("message").and_then(Value::as_str) {
                        let job = event.get("job").and_then(Value::as_u64).unwrap_or(0) + 1;
                        let _ = app.emit("sync-log", format!("[job {job}] {message}"));
                    }
                }
                _ => {}
            }
            let _ = app.emit("subsync-event", &event);
            if kind == terminal {
                return Ok(event);
            }
        }
    })
}

fn with_command(mut payload: Value, command: &str) -> Value {
    if !payload.is_object() {
        payload = json!({});
    }
    if let Some(object) = payload.as_object_mut() {
        object.insert("command".into(), Value::String(command.into()));
    }
    payload
}

async fn run_blocking<R: Runtime>(
    app: AppHandle<R>,
    handle: BridgeHandle,
    payload: Value,
    terminal: &'static str,
    timeout: Duration,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        exchange(&app, &handle, payload, terminal, timeout)
    })
    .await
    .map_err(|err| err.to_string())?
}

// ------------------------------------------------------------ quick queries

#[tauri::command]
pub(crate) async fn subs_probe<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    paths: Vec<String>,
) -> Result<Value, String> {
    let payload = json!({ "command": "subsProbe", "paths": paths });
    run_blocking(
        app,
        info.inner().0.clone(),
        payload,
        "subsProbeResult",
        Duration::from_secs(600),
    )
    .await
}

/// Which engines this machine can run. Keys are passed only as "present":
/// deciding availability needs to know a key exists, never its value.
#[tauri::command]
pub(crate) async fn subs_caps<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
) -> Result<Value, String> {
    let present: BTreeMap<String, String> = read_secrets(&app)
        .into_keys()
        .map(|name| (name, "present".to_string()))
        .collect();
    let payload = json!({ "command": "subsCaps", "secrets": present });
    run_blocking(
        app,
        info.inner().0.clone(),
        payload,
        "subsCapsResult",
        Duration::from_secs(120),
    )
    .await
}

#[tauri::command]
pub(crate) async fn subs_load<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    reference: Value,
    limit: Option<u32>,
) -> Result<Value, String> {
    let payload =
        json!({ "command": "subsLoad", "ref": reference, "limit": limit.unwrap_or(5000) });
    run_blocking(
        app,
        info.inner().0.clone(),
        payload,
        "subsLoadResult",
        Duration::from_secs(300),
    )
    .await
}

#[tauri::command]
pub(crate) async fn subs_save<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    path: String,
    format: Option<String>,
    cues: Value,
    language: Option<String>,
) -> Result<Value, String> {
    let payload = json!({
        "command": "subsSave", "path": path, "format": format, "cues": cues, "language": language,
    });
    run_blocking(
        app,
        info.inner().0.clone(),
        payload,
        "subsSaveResult",
        Duration::from_secs(120),
    )
    .await
}

// ---------------------------------------------------------------- long work

/// Run a queue of subtitle jobs. The stored API keys travel with this one
/// request and nowhere else; the engine never logs or writes them.
#[tauri::command]
pub(crate) async fn start_subs_batch<R: Runtime>(
    app: AppHandle<R>,
    handle: State<'_, BridgeHandle>,
    request: Value,
) -> Result<Value, String> {
    let mut payload = with_command(request, "subsBatch");
    let secrets: BTreeMap<String, String> = read_secrets(&app);
    if let Some(object) = payload.as_object_mut() {
        object.insert("secrets".into(), json!(secrets));
    }
    run_blocking(
        app,
        handle.inner().clone(),
        payload,
        "subsBatchDone",
        EVENT_TIMEOUT,
    )
    .await
}

/// Install or remove an engine pack (or one model inside it). Progress
/// arrives as `packProgress` events; Stop reaches an install through
/// `cancel_sync`, like any other long job.
#[tauri::command]
pub(crate) async fn subs_pack<R: Runtime>(
    app: AppHandle<R>,
    handle: State<'_, BridgeHandle>,
    action: String,
    pack: String,
    model: Option<String>,
) -> Result<Value, String> {
    let command = match action.as_str() {
        "install" => "packInstall",
        "remove" => "packRemove",
        other => return Err(format!("Unknown pack action: {other}")),
    };
    let payload = json!({ "command": command, "pack": pack, "model": model });
    run_blocking(
        app,
        handle.inner().clone(),
        payload,
        "packDone",
        EVENT_TIMEOUT,
    )
    .await
}

// ------------------------------------------------------------------ secrets
//
// API keys live in a file only this user can read, in the app's own config
// folder -- not in the webview's localStorage, where every script on the
// page could read them, and never in a log line.

fn secrets_path<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|dir| dir.join("secrets.json"))
}

fn read_secrets<R: Runtime>(app: &AppHandle<R>) -> BTreeMap<String, String> {
    let Some(path) = secrets_path(app) else {
        return BTreeMap::new();
    };
    let Ok(text) = fs::read_to_string(path) else {
        return BTreeMap::new();
    };
    let parsed: BTreeMap<String, String> = serde_json::from_str(&text).unwrap_or_default();
    parsed
        .into_iter()
        .filter(|(name, value)| SECRET_NAMES.contains(&name.as_str()) && !value.is_empty())
        .collect()
}

fn write_secrets<R: Runtime>(
    app: &AppHandle<R>,
    secrets: &BTreeMap<String, String>,
) -> Result<(), String> {
    let path = secrets_path(app).ok_or("The app's settings folder is unavailable")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let text = serde_json::to_string_pretty(secrets).map_err(|err| err.to_string())?;
    let staged = path.with_extension("json.tmp");
    fs::write(&staged, text).map_err(|err| err.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&staged, fs::Permissions::from_mode(0o600))
            .map_err(|err| err.to_string())?;
    }
    fs::rename(&staged, &path).map_err(|err| err.to_string())
}

/// Which services have a key stored. Never the keys themselves.
#[tauri::command]
pub(crate) fn subs_secret_status<R: Runtime>(app: AppHandle<R>) -> BTreeMap<String, bool> {
    let stored = read_secrets(&app);
    SECRET_NAMES
        .iter()
        .map(|name| (name.to_string(), stored.contains_key(*name)))
        .collect()
}

/// Store a key; an empty value deletes it.
#[tauri::command]
pub(crate) fn subs_secret_set<R: Runtime>(
    app: AppHandle<R>,
    name: String,
    value: Option<String>,
) -> Result<(), String> {
    if !SECRET_NAMES.contains(&name.as_str()) {
        return Err(format!("Unknown service: {name}"));
    }
    let mut secrets = read_secrets(&app);
    match value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        Some(value) => {
            secrets.insert(name, value);
        }
        None => {
            secrets.remove(&name);
        }
    }
    write_secrets(&app, &secrets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtitle_kinds_are_recognised() {
        assert_eq!(
            kind_of(Path::new("/x/Movie.ja.srt"), true),
            Some("subtitle")
        );
        assert_eq!(kind_of(Path::new("/x/Movie.SUP"), true), Some("subtitle"));
        assert_eq!(kind_of(Path::new("/x/Movie.mkv"), true), Some("video"));
        assert_eq!(kind_of(Path::new("/x/Movie.eac3"), true), Some("audio"));
        assert_eq!(kind_of(Path::new("/x/notes.txt"), false), None);
        // TTML as .xml: taken when chosen on purpose, not swept up from a folder.
        assert_eq!(kind_of(Path::new("/x/Movie.xml"), false), Some("subtitle"));
        assert_eq!(kind_of(Path::new("/x/Movie.xml"), true), None);
    }

    #[test]
    fn accept_filters_by_kind() {
        assert!(accepts("subtitle", "subtitle"));
        assert!(!accepts("subtitle", "video"));
        assert!(accepts("media", "audio"));
        assert!(accepts("any", "subtitle"));
    }
}
