mod bridge;
mod csv;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager, State, Window};
use tauri_plugin_dialog::DialogExt;

use bridge::BridgeHandle;

/// How long to wait for a single engine event before assuming it has stalled.
const EVENT_TIMEOUT: Duration = Duration::from_secs(1800);

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct FileItem {
    name: String,
    path: String,
    #[serde(rename = "type")]
    file_type: String,
    size: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct PickResponse {
    folder: Option<String>,
    files: Vec<FileItem>,
}

/// A measured pair. Field names are camelCase on the wire to match both the
/// Python engine and the TypeScript client exactly; the original code declared
/// camelCase Rust fields with no serde attribute, which silently dropped any
/// value the engine sent in snake_case.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct SyncResult {
    #[serde(default)]
    pub video_file: String,
    #[serde(default)]
    pub audio_file: String,
    #[serde(default)]
    pub primary_path: Option<String>,
    #[serde(default)]
    pub secondary_path: Option<String>,
    #[serde(default)]
    pub delay_ms: Option<f64>,
    #[serde(default)]
    pub delay_at_start_ms: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub drift_ms_per_s: Option<f64>,
    #[serde(default)]
    pub total_drift_ms: Option<f64>,
    #[serde(default)]
    pub has_significant_drift: Option<bool>,
    #[serde(default)]
    pub start_delay_ms: Option<f64>,
    #[serde(default)]
    pub end_delay_ms: Option<f64>,
    #[serde(default)]
    pub windows_used: Option<u32>,
    #[serde(default)]
    pub windows_total: Option<u32>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub elapsed_ms: Option<u64>,
    #[serde(default)]
    pub primary_duration_s: Option<f64>,
    #[serde(default)]
    pub secondary_duration_s: Option<f64>,
    #[serde(default)]
    pub primary_track: Option<u32>,
    #[serde(default)]
    pub secondary_track: Option<u32>,
    #[serde(default)]
    pub primary_fps: Option<f64>,
    #[serde(default)]
    pub secondary_fps: Option<f64>,
    #[serde(default)]
    pub is_likely_cut: Option<bool>,
    #[serde(default)]
    pub is_rate_mismatch: Option<bool>,
    #[serde(default)]
    pub codec_delay_ms: Option<f64>,
    #[serde(default)]
    pub primary_codec: Option<String>,
    #[serde(default)]
    pub secondary_codec: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SyncRun {
    results: Vec<SyncResult>,
    summary: Option<Value>,
    cancelled: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MediaProbe {
    has_audio: bool,
    has_video: bool,
    duration: Option<f64>,
    audio_codec: Option<String>,
    error: Option<String>,
}

// ---------------------------------------------------------------- file pickers

async fn pick_folder(window: Window) -> Option<PathBuf> {
    let (tx, rx) = std::sync::mpsc::channel();
    window.dialog().file().pick_folder(move |path| {
        let _ = tx.send(path.and_then(|p| p.into_path().ok()));
    });
    tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
        .await
        .ok()
        .flatten()
}

async fn pick_file(window: Window) -> Option<PathBuf> {
    let (tx, rx) = std::sync::mpsc::channel();
    window.dialog().file().pick_file(move |path| {
        let _ = tx.send(path.and_then(|p| p.into_path().ok()));
    });
    tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
        .await
        .ok()
        .flatten()
}

async fn pick_save_path(window: Window, default_name: &str) -> Option<PathBuf> {
    let (tx, rx) = std::sync::mpsc::channel();
    window
        .dialog()
        .file()
        .set_file_name(default_name)
        .save_file(move |path| {
            let _ = tx.send(path.and_then(|p| p.into_path().ok()));
        });
    tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
        .await
        .ok()
        .flatten()
}

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "webm", "avi", "mov", "m4v", "ts", "wmv", "flv",
];
const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "aac", "flac", "ogg", "opus", "m4a", "eac3", "ac3", "dts", "wma", "mka",
];

fn list_files(folder: &Path, extensions: Option<&[&str]>, kind: &str) -> Vec<FileItem> {
    let mut items = Vec::new();
    let Ok(entries) = fs::read_dir(folder) else {
        return items;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'))
        {
            continue;
        }
        if let Some(allowed) = extensions {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_lowercase();
            if !allowed.contains(&ext.as_str()) {
                continue;
            }
        }
        items.push(FileItem {
            name: path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            path: path.to_string_lossy().to_string(),
            file_type: kind.to_string(),
            size: fs::metadata(&path).map(|m| m.len()).ok(),
        });
    }
    items.sort_by(|a, b| a.name.cmp(&b.name));
    items
}

#[tauri::command]
async fn pick_video_folder(window: Window) -> Result<PickResponse, String> {
    let Some(folder) = pick_folder(window).await else {
        return Ok(PickResponse {
            folder: None,
            files: Vec::new(),
        });
    };
    let files = list_files(&folder, Some(VIDEO_EXTENSIONS), "video");
    Ok(PickResponse {
        folder: Some(folder.to_string_lossy().to_string()),
        files,
    })
}

#[tauri::command]
async fn pick_audio_folder(window: Window) -> Result<PickResponse, String> {
    let Some(folder) = pick_folder(window).await else {
        return Ok(PickResponse {
            folder: None,
            files: Vec::new(),
        });
    };
    let mut files = list_files(&folder, Some(AUDIO_EXTENSIONS), "audio");
    if files.is_empty() {
        // Dub tracks are often delivered inside video containers.
        files = list_files(&folder, None, "audio");
    }
    Ok(PickResponse {
        folder: Some(folder.to_string_lossy().to_string()),
        files,
    })
}

#[tauri::command]
async fn pick_audio_file(window: Window) -> Result<PickResponse, String> {
    let Some(file) = pick_file(window).await else {
        return Ok(PickResponse {
            folder: None,
            files: Vec::new(),
        });
    };
    Ok(PickResponse {
        folder: file.parent().map(|p| p.to_string_lossy().to_string()),
        files: vec![FileItem {
            name: file
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            path: file.to_string_lossy().to_string(),
            file_type: "audio".into(),
            size: fs::metadata(&file).map(|m| m.len()).ok(),
        }],
    })
}

/// Resolve dropped paths into file entries, expanding any dropped folders.
/// Drag-and-drop could never work before: the frontend read `File.path`, which
/// does not exist in a Tauri v2 webview, so every dropped file arrived as a
/// bare filename that no backend could open.
#[tauri::command]
fn resolve_dropped_paths(paths: Vec<String>, kind: String) -> Result<Vec<FileItem>, String> {
    let extensions = match kind.as_str() {
        "video" => Some(VIDEO_EXTENSIONS),
        "audio" => Some(AUDIO_EXTENSIONS),
        _ => None,
    };

    let mut items = Vec::new();
    for raw in paths {
        let path = PathBuf::from(&raw);
        if path.is_dir() {
            items.extend(list_files(&path, extensions, &kind));
        } else if path.is_file() {
            items.push(FileItem {
                name: path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                path: path.to_string_lossy().to_string(),
                file_type: kind.clone(),
                size: fs::metadata(&path).map(|m| m.len()).ok(),
            });
        }
    }
    items.sort_by(|a, b| a.name.cmp(&b.name));
    items.dedup_by(|a, b| a.path == b.path);
    Ok(items)
}

// ------------------------------------------------------------------- analysis

/// Pump engine events to the UI until a terminal event arrives.
///
/// Results accumulate as they stream, so a run that ends badly still returns
/// everything it managed to measure. The original returned `Err` on a non-zero
/// exit and discarded the entire batch.
fn drain_events(
    app: &AppHandle,
    bridge: &mut bridge::Bridge,
    terminal: &str,
) -> Result<(Vec<SyncResult>, Option<Value>, bool), String> {
    let mut results: Vec<SyncResult> = Vec::new();
    let mut summary = None;
    let mut cancelled = false;
    let mut fatal: Option<String> = None;

    loop {
        let event = match bridge.events().recv_timeout(EVENT_TIMEOUT) {
            Ok(event) => event,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err("The analysis engine stopped responding.".into());
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if results.is_empty() {
                    return Err("The analysis engine exited unexpectedly.".into());
                }
                break;
            }
        };

        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
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
                    .unwrap_or("Unknown engine error")
                    .to_string();
                let _ = app.emit("sync-log", format!("Error: {message}"));
                if event.get("fatal").and_then(Value::as_bool).unwrap_or(false) {
                    fatal = Some(message);
                }
            }
            "pairs" => {
                let _ = app.emit("sync-pairs", &event);
            }
            "progress" => {
                let _ = app.emit("sync-progress", &event);
            }
            "fileStart" => {
                let _ = app.emit("sync-file-start", &event);
            }
            "fileProgress" => {
                let _ = app.emit("sync-file-progress", &event);
            }
            "result" => match serde_json::from_value::<SyncResult>(event.clone()) {
                Ok(result) => {
                    let _ = app.emit("sync-result", &result);
                    results.push(result);
                }
                Err(err) => {
                    let _ = app.emit("sync-log", format!("Malformed result: {err}"));
                }
            },
            "applyStart" | "applyProgress" => {
                let _ = app.emit("sync-apply-progress", &event);
            }
            other if other == terminal => {
                cancelled = event
                    .get("cancelled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                summary = event.get("summary").cloned();

                // Prefer the engine's final list; fall back to what streamed in.
                if let Some(list) = event.get("results") {
                    if let Ok(final_results) =
                        serde_json::from_value::<Vec<SyncResult>>(list.clone())
                    {
                        if !final_results.is_empty() {
                            results = final_results;
                        }
                    }
                }
                break;
            }
            _ => {}
        }
    }

    if let Some(message) = fatal {
        if results.is_empty() {
            return Err(message);
        }
    }
    Ok((results, summary, cancelled))
}

#[tauri::command]
async fn start_sync(
    app: AppHandle,
    handle: State<'_, BridgeHandle>,
    request: Value,
) -> Result<SyncRun, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("analyze".into()));
            }
            bridge.send(&payload)?;
            let (results, summary, cancelled) = drain_events(&app_for_task, bridge, "done")?;
            let run = SyncRun {
                results,
                summary,
                cancelled,
            };
            let _ = app_for_task.emit("sync-done", &run);
            Ok(run)
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

/// List the selectable audio streams of each file.
///
/// A container often carries an original language, a dub and a commentary;
/// without this the UI cannot offer a choice and every comparison silently
/// uses the first stream.
#[tauri::command]
async fn list_audio_tracks(
    app: AppHandle,
    handle: State<'_, BridgeHandle>,
    paths: Vec<String>,
) -> Result<Value, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            bridge.send(&serde_json::json!({
                "command": "listTracks",
                "paths": paths,
            }))?;
            loop {
                match bridge.events().recv_timeout(Duration::from_secs(180)) {
                    Ok(event) => match event.get("type").and_then(Value::as_str) {
                        Some("tracks") => {
                            return Ok(event
                                .get("files")
                                .cloned()
                                .unwrap_or_else(|| Value::Array(Vec::new())));
                        }
                        Some("error") => {
                            return Err(event
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("Could not read audio tracks")
                                .to_string());
                        }
                        _ => {}
                    },
                    Err(_) => return Err("Timed out reading audio tracks.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

#[tauri::command]
async fn preview_pairs(
    app: AppHandle,
    handle: State<'_, BridgeHandle>,
    request: Value,
) -> Result<Value, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("previewPairs".into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(Duration::from_secs(120)) {
                    Ok(event) => {
                        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
                        if kind == "pairs" {
                            return Ok(event);
                        }
                        if kind == "error" {
                            return Err(event
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("Could not preview pairs")
                                .to_string());
                        }
                    }
                    Err(_) => return Err("Timed out building the pairing preview.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

#[tauri::command]
async fn apply_corrections(
    app: AppHandle,
    handle: State<'_, BridgeHandle>,
    request: Value,
) -> Result<Value, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("apply".into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(EVENT_TIMEOUT) {
                    Ok(event) => {
                        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
                        match kind {
                            "applyDone" => return Ok(event),
                            "applyStart" | "applyProgress" => {
                                let _ = app_for_task.emit("sync-apply-progress", &event);
                            }
                            "log" => {
                                if let Some(m) = event.get("message").and_then(Value::as_str) {
                                    let _ = app_for_task.emit("sync-log", m);
                                }
                            }
                            "error" => {
                                let _ = app_for_task.emit(
                                    "sync-log",
                                    event.get("message").and_then(Value::as_str).unwrap_or(""),
                                );
                            }
                            _ => {}
                        }
                    }
                    Err(_) => return Err("Timed out while writing corrected files.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

/// Cancel the running batch. Sent immediately rather than queued, so it reaches
/// the engine while the run it targets is still in flight.
#[tauri::command]
fn cancel_sync(handle: State<'_, BridgeHandle>) -> Result<(), String> {
    handle.send_now(&serde_json::json!({ "command": "cancel" }))
}

#[tauri::command]
async fn probe_media(
    app: AppHandle,
    handle: State<'_, BridgeHandle>,
    path: String,
) -> Result<MediaProbe, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            bridge.send(&serde_json::json!({ "command": "probe", "path": path }))?;
            loop {
                match bridge.events().recv_timeout(Duration::from_secs(120)) {
                    Ok(event) => {
                        if event.get("type").and_then(Value::as_str) == Some("probe") {
                            return Ok(MediaProbe {
                                has_audio: event
                                    .get("hasAudio")
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false),
                                has_video: event
                                    .get("hasVideo")
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false),
                                duration: event.get("duration").and_then(Value::as_f64),
                                audio_codec: event
                                    .get("audioCodec")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                                error: event
                                    .get("error")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                            });
                        }
                    }
                    Err(_) => return Err("Timed out reading media information.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

// --------------------------------------------------------------------- export

#[tauri::command]
async fn export_csv(window: Window, results: Vec<SyncResult>) -> Result<String, String> {
    let Some(path) = pick_save_path(window, "sync-results.csv").await else {
        return Err("Export cancelled".into());
    };
    let contents = csv::render(&results);
    fs::write(&path, contents.as_bytes()).map_err(|err| err.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
async fn export_json(window: Window, results: Vec<SyncResult>) -> Result<String, String> {
    let Some(path) = pick_save_path(window, "sync-results.json").await else {
        return Err("Export cancelled".into());
    };
    let contents = serde_json::to_string_pretty(&results).map_err(|err| err.to_string())?;
    fs::write(&path, contents.as_bytes()).map_err(|err| err.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err("That file no longer exists.".into());
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(format!("/select,{}", path.to_string_lossy()))
            .spawn()
            .map_err(|err| err.to_string())?;
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg("-R")
            .arg(&path)
            .spawn()
            .map_err(|err| err.to_string())?;
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        let folder = if path.is_dir() {
            path.clone()
        } else {
            path.parent().unwrap_or(Path::new(".")).to_path_buf()
        };
        Command::new("xdg-open")
            .arg(folder)
            .spawn()
            .map_err(|err| err.to_string())?;
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        );

    // Updating replaces the installed binary, which only applies on desktop.
    #[cfg(desktop)]
    {
        builder = builder
            .plugin(tauri_plugin_updater::Builder::new().build())
            .plugin(tauri_plugin_process::init());
    }

    builder
        .manage(BridgeHandle::default())
        .invoke_handler(tauri::generate_handler![
            pick_video_folder,
            pick_audio_folder,
            pick_audio_file,
            resolve_dropped_paths,
            preview_pairs,
            start_sync,
            cancel_sync,
            apply_corrections,
            probe_media,
            list_audio_tracks,
            export_csv,
            export_json,
            reveal_path,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(handle) = window.app_handle().try_state::<BridgeHandle>() {
                    handle.shutdown();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
