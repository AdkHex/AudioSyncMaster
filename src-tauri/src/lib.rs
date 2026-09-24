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

use bridge::{BridgeHandle, ShotBridge, WaveformBridge};

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
    /// The frame-rate explanation for any drift, passed through untouched.
    ///
    /// Kept as a raw value because the host never reads inside it: mirroring
    /// the engine's shape here would be a third copy of the same schema, and
    /// the whole struct exists because an undeclared field is dropped in
    /// silence rather than erroring -- which is exactly how this one went
    /// missing while Python emitted it and TypeScript expected it.
    #[serde(default)]
    pub rate_diagnosis: Option<Value>,
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

async fn pick_folder<R: tauri::Runtime>(window: Window<R>) -> Option<PathBuf> {
    let (tx, rx) = std::sync::mpsc::channel();
    window.dialog().file().pick_folder(move |path| {
        let _ = tx.send(path.and_then(|p| p.into_path().ok()));
    });
    tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
        .await
        .ok()
        .flatten()
}

async fn pick_file<R: tauri::Runtime>(window: Window<R>) -> Option<PathBuf> {
    let (tx, rx) = std::sync::mpsc::channel();
    window.dialog().file().pick_file(move |path| {
        let _ = tx.send(path.and_then(|p| p.into_path().ok()));
    });
    tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
        .await
        .ok()
        .flatten()
}

async fn pick_save_path<R: tauri::Runtime>(
    window: Window<R>,
    default_name: &str,
) -> Option<PathBuf> {
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
/// Every extension an external dub actually arrives as. Kept in step with
/// AUDIO_EXTENSIONS in audiosync/media.py: a `.ec3` or `.thd` dropped on the
/// window used to vanish without a word because only this shorter list was
/// consulted, while the engine itself would have decoded it happily.
#[rustfmt::skip]
const AUDIO_EXTENSIONS: &[&str] = &[
    // Dolby
    "ac3", "eac3", "ec3", "thd", "truehd", "mlp",
    // DTS
    "dts", "dtsma", "dtshd",
    // MPEG and friends
    "aac", "m4a", "m4b", "mp3", "mp2", "mpa",
    // Lossless and open formats
    "flac", "wav", "w64", "aiff", "aif", "caf", "alac",
    "ogg", "oga", "opus", "ape", "tak", "tta", "wv",
    // Containers that commonly hold nothing but audio
    "mka", "wma",
];

fn is_media_extension(ext: &str) -> bool {
    VIDEO_EXTENSIONS.contains(&ext) || AUDIO_EXTENSIONS.contains(&ext)
}

/// Which files a picker or a drop accepts.
#[derive(Clone, Copy)]
enum Accept {
    Video,
    Audio,
    /// Either: a dub is as likely to arrive inside an MKV as a bare .eac3.
    Media,
    Any,
}

impl Accept {
    fn from_name(name: &str) -> Self {
        match name {
            "video" => Accept::Video,
            "audio" => Accept::Audio,
            "media" => Accept::Media,
            _ => Accept::Any,
        }
    }

    fn allows(self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();
        match self {
            Accept::Video => VIDEO_EXTENSIONS.contains(&ext.as_str()),
            Accept::Audio => AUDIO_EXTENSIONS.contains(&ext.as_str()),
            Accept::Media => is_media_extension(&ext),
            Accept::Any => true,
        }
    }
}

fn file_item(path: &Path, kind: &str) -> FileItem {
    FileItem {
        name: path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        path: path.to_string_lossy().to_string(),
        file_type: kind.to_string(),
        size: fs::metadata(path).map(|m| m.len()).ok(),
    }
}

fn list_files(folder: &Path, accept: Accept, kind: &str) -> Vec<FileItem> {
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
        if !accept.allows(&path) {
            continue;
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
async fn pick_video_folder<R: tauri::Runtime>(window: Window<R>) -> Result<PickResponse, String> {
    let Some(folder) = pick_folder(window).await else {
        return Ok(PickResponse {
            folder: None,
            files: Vec::new(),
        });
    };
    let files = list_files(&folder, Accept::Video, "video");
    Ok(PickResponse {
        folder: Some(folder.to_string_lossy().to_string()),
        files,
    })
}

#[tauri::command]
async fn pick_audio_folder<R: tauri::Runtime>(window: Window<R>) -> Result<PickResponse, String> {
    let Some(folder) = pick_folder(window).await else {
        return Ok(PickResponse {
            folder: None,
            files: Vec::new(),
        });
    };
    let mut files = list_files(&folder, Accept::Audio, "audio");
    if files.is_empty() {
        // Dub tracks are often delivered inside video containers.
        files = list_files(&folder, Accept::Media, "audio");
    }
    Ok(PickResponse {
        folder: Some(folder.to_string_lossy().to_string()),
        files,
    })
}

#[tauri::command]
async fn pick_audio_file<R: tauri::Runtime>(window: Window<R>) -> Result<PickResponse, String> {
    pick_single(window, "audio").await
}

/// One file of either kind, for the sides of a dub sync: the "video" may be
/// a bare original-language track, and the dub may live inside an MKV.
#[tauri::command]
async fn pick_media_file<R: tauri::Runtime>(
    window: Window<R>,
    kind: String,
) -> Result<PickResponse, String> {
    pick_single(window, &kind).await
}

/// Several files of either kind at once, for the dub tab's Movies scope:
/// a movie list is rarely one folder, so the dialog allows multi-selection.
/// Anything media drops in, since either slot takes a dub inside an MKV as
/// readily as a bare track; non-media picks are discarded like stray drops.
#[tauri::command]
async fn pick_media_files<R: tauri::Runtime>(
    window: Window<R>,
    kind: String,
) -> Result<PickResponse, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    window.dialog().file().pick_files(move |paths| {
        let _ = tx.send(paths.unwrap_or_default());
    });
    let paths = tauri::async_runtime::spawn_blocking(move || rx.recv().ok().unwrap_or_default())
        .await
        .map_err(|err| err.to_string())?;

    let mut items: Vec<FileItem> = paths
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .filter(|p| p.is_file() && Accept::Media.allows(p))
        .map(|p| file_item(&p, &kind))
        .collect();
    items.sort_by(|a, b| a.name.cmp(&b.name));
    items.dedup_by(|a, b| a.path == b.path);
    Ok(PickResponse {
        folder: None,
        files: items,
    })
}

async fn pick_single<R: tauri::Runtime>(
    window: Window<R>,
    kind: &str,
) -> Result<PickResponse, String> {
    let Some(file) = pick_file(window).await else {
        return Ok(PickResponse {
            folder: None,
            files: Vec::new(),
        });
    };
    Ok(PickResponse {
        folder: file.parent().map(|p| p.to_string_lossy().to_string()),
        files: vec![file_item(&file, kind)],
    })
}

/// Resolve dropped paths into file entries, expanding any dropped folders.
/// Drag-and-drop could never work before: the frontend read `File.path`, which
/// does not exist in a Tauri v2 webview, so every dropped file arrived as a
/// bare filename that no backend could open.
///
/// `accept` narrows what a dropped folder contributes: the side's own kind by
/// default, or "media" for a slot that takes either, as both sides of a dub
/// sync do. A file dropped on its own only has to be media at all -- a dub
/// arrives inside an MKV as often as a bare .eac3, and the audio folder
/// picker already takes video containers for the same reason -- but it does
/// have to be media: a stray .srt used to land in the list as if it were audio.
#[tauri::command]
fn resolve_dropped_paths(
    paths: Vec<String>,
    kind: String,
    accept: Option<String>,
) -> Result<Vec<FileItem>, String> {
    let in_folders = Accept::from_name(accept.as_deref().unwrap_or(kind.as_str()));

    let mut items = Vec::new();
    for raw in paths {
        let path = PathBuf::from(&raw);
        if path.is_dir() {
            items.extend(list_files(&path, in_folders, &kind));
        } else if path.is_file() && Accept::Media.allows(&path) {
            items.push(file_item(&path, &kind));
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
fn drain_events<R: tauri::Runtime>(
    app: &AppHandle<R>,
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
async fn start_sync<R: tauri::Runtime>(
    app: AppHandle<R>,
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
async fn list_audio_tracks<R: tauri::Runtime>(
    app: AppHandle<R>,
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
async fn preview_pairs<R: tauri::Runtime>(
    app: AppHandle<R>,
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

/// Render a short aligned excerpt and return its path.
///
/// A confidence score is an argument; hearing the audio land settles it.
#[tauri::command]
async fn render_preview<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: State<'_, BridgeHandle>,
    request: Value,
) -> Result<Option<String>, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("preview".into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(Duration::from_secs(300)) {
                    Ok(event) => match event.get("type").and_then(Value::as_str) {
                        Some("previewDone") => {
                            return Ok(event
                                .get("path")
                                .and_then(Value::as_str)
                                .map(str::to_string));
                        }
                        Some("error") => {
                            let _ = app_for_task.emit(
                                "sync-log",
                                event.get("message").and_then(Value::as_str).unwrap_or(""),
                            );
                        }
                        _ => {}
                    },
                    Err(_) => return Err("Timed out rendering the preview.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

/// Waveform peaks of one track over a span, for the dub sync editor. The
/// engine keeps the decoded envelope, so after the first request for a file
/// the answer is immediate.
#[tauri::command]
async fn waveform_peaks<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: State<'_, WaveformBridge>,
    request: Value,
) -> Result<Value, String> {
    waveform_exchange(app, handle.inner().0.clone(), request, "waveformPeaks").await
}

/// Read a track's waveform into the engine's cache ahead of any view of it,
/// reporting progress as `waveform-progress` events. Resolves with the
/// track's length, channel count and rate.
#[tauri::command]
async fn waveform_build<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: State<'_, WaveformBridge>,
    request: Value,
) -> Result<Value, String> {
    waveform_exchange(app, handle.inner().0.clone(), request, "waveformBuild").await
}

/// One waveform command on the waveform engine: `waveformBuild` answers with
/// `waveformReady`, `waveformPeaks` with `waveformPeaks`; either may first
/// stream `waveformProgress` while a file is read for the first time.
async fn waveform_exchange<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: BridgeHandle,
    request: Value,
    command: &'static str,
) -> Result<Value, String> {
    let app_for_task = app.clone();
    let reply = if command == "waveformBuild" {
        "waveformReady"
    } else {
        "waveformPeaks"
    };

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String(command.into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(Duration::from_secs(600)) {
                    Ok(event) => match event.get("type").and_then(Value::as_str) {
                        Some(kind) if kind == reply => {
                            if let Some(message) = event.get("error").and_then(Value::as_str) {
                                return Err(message.to_string());
                            }
                            if event.get("cancelled").and_then(Value::as_bool) == Some(true) {
                                return Err("Reading the waveform was stopped.".into());
                            }
                            return Ok(event);
                        }
                        Some("waveformProgress") => {
                            let _ = app_for_task.emit("waveform-progress", &event);
                        }
                        Some("error") => {
                            let _ = app_for_task.emit(
                                "sync-log",
                                event.get("message").and_then(Value::as_str).unwrap_or(""),
                            );
                        }
                        _ => {}
                    },
                    Err(_) => return Err("Timed out reading the waveform.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

/// The video's shot changes across a span, for the ruler of the cut
/// editor: `{ cuts, frameS, startS, endS }`, `cuts` null when the picture
/// cannot be read. Served by an engine of its own, so decoding the picture
/// never holds up the waveforms.
#[tauri::command]
async fn shot_cuts<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: State<'_, ShotBridge>,
    request: Value,
) -> Result<Value, String> {
    let handle = handle.inner().0.clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("shotCuts".into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(Duration::from_secs(600)) {
                    Ok(event) => match event.get("type").and_then(Value::as_str) {
                        Some("shotCuts") => {
                            if let Some(message) = event.get("error").and_then(Value::as_str) {
                                return Err(message.to_string());
                            }
                            return Ok(event);
                        }
                        Some("log") | Some("error") => {
                            if let Some(m) = event.get("message").and_then(Value::as_str) {
                                let _ = app_for_task.emit("sync-log", m);
                            }
                        }
                        _ => {}
                    },
                    Err(_) => return Err("Timed out reading the picture.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

/// A short excerpt of a dub sync plan as edited in the app, rendered to
/// play: the picture, the synced sound, the original's sound, or the
/// picture with the sound under it (`what`). Returns the engine's reply --
/// the temporary file's path and the span actually cut, which for a
/// picture starts on a frame -- or None when nothing could be rendered.
#[tauri::command]
async fn render_dub_preview<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: State<'_, WaveformBridge>,
    request: Value,
) -> Result<Option<Value>, String> {
    let handle = handle.inner().0.clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("dubsyncPreview".into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(Duration::from_secs(600)) {
                    Ok(event) => match event.get("type").and_then(Value::as_str) {
                        Some("dubsyncPreviewDone") => {
                            return Ok(if event.get("path").and_then(Value::as_str).is_some() {
                                Some(event)
                            } else {
                                None
                            });
                        }
                        Some("log") => {
                            if let Some(m) = event.get("message").and_then(Value::as_str) {
                                let _ = app_for_task.emit("sync-log", m);
                            }
                        }
                        Some("error") => {
                            let _ = app_for_task.emit(
                                "sync-log",
                                event.get("message").and_then(Value::as_str).unwrap_or(""),
                            );
                        }
                        _ => {}
                    },
                    Err(_) => return Err("Timed out rendering the preview.".into()),
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

/// Bytes of one of the engine's preview excerpts, for the in-app player.
///
/// The webview cannot reach the OS filesystem, and the excerpts must not
/// be exposed through the asset protocol, so the bytes cross the IPC. Only
/// files the engine wrote for that purpose are served: inside the OS temp
/// dir and named `audiosync-dub-preview-*`, so a crafted path cannot be
/// used to read anything else on the disk.
#[tauri::command]
async fn read_preview_bytes(path: String) -> Result<tauri::ipc::Response, String> {
    let path = PathBuf::from(path);
    tauri::async_runtime::spawn_blocking(move || {
        let bytes = preview_bytes_guard(&path)?;
        Ok(tauri::ipc::Response::new(bytes))
    })
    .await
    .map_err(|err| err.to_string())?
}

/// The guard the preview reader enforces, separated so it can be tested
/// without an app. Both sides are canonicalised first: macOS reports its
/// temp dir as /var/folders/... while its real location is /private/var/...,
/// so a naive prefix check would refuse everything the engine wrote.
fn preview_bytes_guard(path: &Path) -> Result<Vec<u8>, String> {
    let canonical =
        fs::canonicalize(path).map_err(|_| "That preview no longer exists.".to_string())?;
    let temp = fs::canonicalize(std::env::temp_dir()).map_err(|err| err.to_string())?;
    if !canonical.starts_with(&temp) {
        return Err("That file is not a preview.".into());
    }
    let named = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("audiosync-dub-preview-"));
    if !named {
        return Err("That file is not a preview.".into());
    }
    fs::read(&canonical).map_err(|err| err.to_string())
}

#[tauri::command]
async fn apply_corrections<R: tauri::Runtime>(
    app: AppHandle<R>,
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

/// Lay a cut dub onto its video: plan, write, verify, and optionally mux.
///
/// The engine streams its progress and the plan as it goes, and the whole
/// outcome comes back at the end. Cancellation reaches it through
/// `cancel_sync`, the same as a batch.
#[tauri::command]
async fn start_dubsync<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: State<'_, BridgeHandle>,
    request: Value,
) -> Result<Value, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("dubsync".into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(EVENT_TIMEOUT) {
                    Ok(event) => {
                        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
                        match kind {
                            "dubsyncDone" => return Ok(event),
                            "dubsyncProgress" => {
                                let _ = app_for_task.emit("dubsync-progress", &event);
                            }
                            "dubsyncDraft" => {
                                let _ = app_for_task.emit("dubsync-draft", &event);
                            }
                            "dubsyncPlan" => {
                                let _ = app_for_task.emit("dubsync-plan", &event);
                            }
                            "log" => {
                                if let Some(m) = event.get("message").and_then(Value::as_str) {
                                    let _ = app_for_task.emit("sync-log", m);
                                }
                            }
                            "error" => {
                                let message = event
                                    .get("message")
                                    .and_then(Value::as_str)
                                    .unwrap_or("Unknown engine error");
                                let _ = app_for_task.emit("sync-log", format!("Error: {message}"));
                            }
                            _ => {}
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        return Err("The analysis engine stopped responding.".into());
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        return Err("The analysis engine exited unexpectedly.".into());
                    }
                }
            }
        })
    })
    .await
    .map_err(|err| err.to_string())?
}

/// Sync a queue of dub pairs in parallel: a season of episodes, or several
/// movies at once. Each job streams its own progress and plan, and the whole
/// batch resolves with one outcome per job. Cancellation reaches it through
/// `cancel_sync`, the same as a single dub sync.
#[tauri::command]
async fn start_dubsync_batch<R: tauri::Runtime>(
    app: AppHandle<R>,
    handle: State<'_, BridgeHandle>,
    request: Value,
) -> Result<Value, String> {
    let handle = handle.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        handle.with(&app_for_task, |bridge| {
            let mut payload = request.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("command".into(), Value::String("dubsyncBatch".into()));
            }
            bridge.send(&payload)?;
            loop {
                match bridge.events().recv_timeout(EVENT_TIMEOUT) {
                    Ok(event) => {
                        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
                        match kind {
                            "dubsyncBatchDone" => return Ok(event),
                            "dubsyncJobStart" => {
                                let _ = app_for_task.emit("dubsync-job-start", &event);
                            }
                            "dubsyncJobProgress" => {
                                let _ = app_for_task.emit("dubsync-job-progress", &event);
                            }
                            "dubsyncJobDraft" => {
                                let _ = app_for_task.emit("dubsync-job-draft", &event);
                            }
                            "dubsyncJobPlan" => {
                                let _ = app_for_task.emit("dubsync-job-plan", &event);
                            }
                            "dubsyncJobDone" => {
                                let _ = app_for_task.emit("dubsync-job-done", &event);
                            }
                            "log" | "dubsyncJobLog" => {
                                if let Some(m) = event.get("message").and_then(Value::as_str) {
                                    let _ = app_for_task.emit("sync-log", m);
                                }
                            }
                            "error" => {
                                let message = event
                                    .get("message")
                                    .and_then(Value::as_str)
                                    .unwrap_or("Unknown engine error");
                                let _ = app_for_task.emit("sync-log", format!("Error: {message}"));
                            }
                            _ => {}
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        return Err("The analysis engine stopped responding.".into());
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        return Err("The analysis engine exited unexpectedly.".into());
                    }
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
async fn probe_media<R: tauri::Runtime>(
    app: AppHandle<R>,
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
async fn export_csv<R: tauri::Runtime>(
    window: Window<R>,
    results: Vec<SyncResult>,
) -> Result<String, String> {
    let Some(path) = pick_save_path(window, "sync-results.csv").await else {
        return Err("Export cancelled".into());
    };
    let contents = csv::render(&results);
    fs::write(&path, contents.as_bytes()).map_err(|err| err.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
async fn export_json<R: tauri::Runtime>(
    window: Window<R>,
    results: Vec<SyncResult>,
) -> Result<String, String> {
    let Some(path) = pick_save_path(window, "sync-results.json").await else {
        return Err("Export cancelled".into());
    };
    let contents = serde_json::to_string_pretty(&results).map_err(|err| err.to_string())?;
    fs::write(&path, contents.as_bytes()).map_err(|err| err.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

/// Open a file with whatever the OS considers its default application.
///
/// Used for previews: the user's own player is better at playback than
/// anything embeddable, and it already knows their audio device.
#[tauri::command]
fn open_path<R: tauri::Runtime>(app: AppHandle<R>, path: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    if !PathBuf::from(&path).exists() {
        return Err("That file no longer exists.".into());
    }
    app.opener()
        .open_path(path, None::<&str>)
        .map_err(|err| err.to_string())
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

/// The app's context, generated once. `generate_context!` embeds a plist
/// symbol on macOS, so a second invocation anywhere in the crate -- a test,
/// say -- is a duplicate-symbol error; both the app and its tests go
/// through here.
fn context<R: tauri::Runtime>() -> tauri::Context<R> {
    tauri::generate_context!()
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
        .manage(WaveformBridge::default())
        .manage(ShotBridge::default())
        .invoke_handler(tauri::generate_handler![
            pick_video_folder,
            pick_audio_folder,
            pick_audio_file,
            pick_media_file,
            pick_media_files,
            resolve_dropped_paths,
            preview_pairs,
            start_sync,
            start_dubsync,
            start_dubsync_batch,
            cancel_sync,
            apply_corrections,
            probe_media,
            list_audio_tracks,
            render_preview,
            waveform_peaks,
            waveform_build,
            shot_cuts,
            render_dub_preview,
            read_preview_bytes,
            export_csv,
            export_json,
            reveal_path,
            open_path,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(handle) = window.app_handle().try_state::<BridgeHandle>() {
                    handle.shutdown();
                }
                if let Some(handle) = window.app_handle().try_state::<WaveformBridge>() {
                    handle.0.shutdown();
                }
                if let Some(handle) = window.app_handle().try_state::<ShotBridge>() {
                    handle.0.shutdown();
                }
            }
        })
        .run(context())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field the engine sends must survive the trip through SyncResult.
    ///
    /// Serde drops undeclared fields without complaining, so a field added to
    /// the Python engine and to the TypeScript client can still go missing
    /// here and the only symptom is a value that never renders. That is
    /// exactly what happened to `rateDiagnosis`: the engine classified a
    /// frame-rate conversion, the UI was ready to show it, and this struct
    /// silently discarded it in between.
    #[test]
    fn rate_diagnosis_survives_deserialization() {
        let payload = serde_json::json!({
            "videoFile": "a.mkv",
            "audioFile": "b.eac3",
            "delayMs": -17606.245,
            "driftMsPerS": -0.998,
            "isRateMismatch": true,
            "primaryFps": 23.976023976023978,
            "secondaryFps": null,
            "rateDiagnosis": {
                "driftMsPerS": -0.998,
                "speedRatio": 1.001,
                // As the engine sends them: 23.976 is 24000/1001, and the
                // correction ratio is 1000/1001 rather than a rounding of it.
                "sourceFps": 24.0,
                "targetFps": 23.976023976023978,
                "isRateMismatch": true,
                "isLikelyCut": false,
                "cutPositionS": null,
                "cutMagnitudeMs": null,
                "explanation": "The audio was timed against a 24fps source, but this video is 23.976fps. Resampling the audio corrects it exactly.",
                "correctionRatio": 0.999000999000999
            }
        });

        let result: SyncResult = serde_json::from_value(payload).expect("should deserialize");

        let diagnosis = result
            .rate_diagnosis
            .as_ref()
            .expect("rateDiagnosis must not be dropped");
        assert_eq!(diagnosis["sourceFps"], 24.0);
        assert_eq!(diagnosis["targetFps"], 23.976023976023978);
        assert_eq!(diagnosis["isRateMismatch"], true);

        // And it must still be there once re-serialized for the frontend.
        let out = serde_json::to_value(&result).expect("should serialize");
        assert_eq!(out["rateDiagnosis"]["targetFps"], 23.976023976023978);
    }

    /// A result with no diagnosis is normal, not an error.
    #[test]
    fn missing_rate_diagnosis_is_none() {
        let payload = serde_json::json!({ "videoFile": "a.mkv", "audioFile": "b.eac3" });
        let result: SyncResult = serde_json::from_value(payload).expect("should deserialize");
        assert!(result.rate_diagnosis.is_none());
    }

    /// The preview reader serves only what the engine wrote: a file in the
    /// temp dir named `audiosync-dub-preview-*` reads back, anything else
    /// -- another name, another directory, a missing file -- is refused.
    #[test]
    fn preview_reader_refuses_anything_but_the_engines_previews() {
        let inside = std::env::temp_dir().join(format!(
            "audiosync-dub-preview-guard-{}.bin",
            std::process::id()
        ));
        fs::write(&inside, b"frame bytes").expect("should write");
        let served = preview_bytes_guard(&inside).expect("a real preview is served");
        assert_eq!(served, b"frame bytes");
        let _ = fs::remove_file(&inside);

        // The right place, but not the engine's naming.
        let other = std::env::temp_dir().join(format!("guard-other-{}.bin", std::process::id()));
        fs::write(&other, b"nope").expect("should write");
        assert_eq!(
            preview_bytes_guard(&other),
            Err("That file is not a preview.".into())
        );
        let _ = fs::remove_file(&other);

        // The right naming, but outside the temp dir. The repo's own
        // directory stands in for anywhere else on the disk: it is always
        // writable where the tests run, unlike the temp dir's parent,
        // which is `/` on Linux.
        let outside_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("audiosync-guard-{}", std::process::id()));
        fs::create_dir_all(&outside_dir).expect("should create");
        let outside = outside_dir.join(format!("audiosync-dub-preview-{}.bin", std::process::id()));
        fs::write(&outside, b"nope").expect("should write");
        assert_eq!(
            preview_bytes_guard(&outside),
            Err("That file is not a preview.".into())
        );
        let _ = fs::remove_file(&outside);
        let _ = fs::remove_dir(&outside_dir);

        // And a path that no longer exists at all.
        let gone = std::env::temp_dir().join("audiosync-dub-preview-gone.mp4");
        assert_eq!(
            preview_bytes_guard(&gone),
            Err("That preview no longer exists.".into())
        );
    }

    /// The dub sync command, end to end: the request the webview sends, the
    /// real Python engine behind the bridge, and the outcome that comes back.
    ///
    /// Everything between the React panel and the engine is exercised here
    /// -- the command's argument shape, the bridge, the event forwarding,
    /// the terminal event -- on a synthetic pair whose cuts are known.
    ///
    /// Needs the Python environment and ffmpeg, which the Rust CI job does
    /// not have, so it runs only when AUDIOSYNC_E2E is set:
    ///
    ///     AUDIOSYNC_E2E=1 cargo test --manifest-path src-tauri/Cargo.toml
    #[test]
    fn start_dubsync_runs_the_engine_end_to_end() {
        if std::env::var_os("AUDIOSYNC_E2E").is_none() {
            eprintln!("skipped: set AUDIOSYNC_E2E=1 to run the engine end to end");
            return;
        }

        let app = tauri::test::mock_builder()
            .manage(BridgeHandle::default())
            .invoke_handler(tauri::generate_handler![start_dubsync])
            .build(context())
            .expect("app should build");
        let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
            .build()
            .expect("webview should build");

        // A synthetic pair with known cuts, from the Python test fixtures.
        let root = bridge::project_root(&app.handle().clone());
        let dir = std::env::temp_dir().join(format!("audiosync-e2e-{}", std::process::id()));
        let status = Command::new(bridge::find_python(&root))
            .arg(root.join("tests").join("test_dubsync.py"))
            .arg(&dir)
            .status()
            .expect("fixture script should run");
        assert!(status.success(), "fixture generation failed");
        let video = dir.join("org.wav");
        let dub = dir.join("dub.wav");
        let output = dir.join("synced.wav");

        let response = tauri::test::get_ipc_response(
            &webview,
            tauri::webview::InvokeRequest {
                cmd: "start_dubsync".into(),
                callback: tauri::ipc::CallbackFn(0),
                error: tauri::ipc::CallbackFn(1),
                url: "tauri://localhost".parse().unwrap(),
                body: tauri::ipc::InvokeBody::Json(serde_json::json!({
                    "request": {
                        "videoPath": video,
                        "dubPath": dub,
                        "videoTrack": 0,
                        "dubTrack": 0,
                        "codec": "wav",
                        "outputPath": output,
                        "mux": false,
                        "language": null,
                        "fillUnmatched": false,
                        "overwrite": true,
                    }
                })),
                headers: Default::default(),
                invoke_key: tauri::test::INVOKE_KEY.to_string(),
            },
        );
        app.state::<BridgeHandle>().shutdown();

        let outcome: Value = response
            .expect("the command should succeed")
            .deserialize()
            .expect("the outcome should be JSON");
        assert!(
            outcome["error"].is_null(),
            "engine error: {}",
            outcome["error"]
        );
        assert_eq!(outcome["cancelled"], false);

        // The fixture has four stretches of dub and cuts at 60-75s, 180-182.5s
        // and 297-300s; a dub-only insert at 240s leaves no gap in the video.
        let segments = outcome["plan"]["segments"]
            .as_array()
            .expect("plan should list segments");
        let dubs = segments.iter().filter(|s| s["kind"] == "dub").count();
        let fills: Vec<&Value> = segments.iter().filter(|s| s["kind"] == "fill").collect();
        assert_eq!(dubs, 4, "plan: {}", outcome["plan"]);
        assert!(
            fills
                .iter()
                .any(|f| (f["startS"].as_f64().unwrap() - 180.0).abs() < 0.05
                    && (f["endS"].as_f64().unwrap() - 182.5).abs() < 0.05),
            "the 2.5s cut at 180s was not filled: {}",
            outcome["plan"]
        );

        // The track was written, the video's length, and measured as in sync.
        assert!(output.is_file(), "no output at {}", output.display());
        assert_eq!(outcome["output"]["outputPath"].as_str(), output.to_str());
        let seconds = outcome["output"]["seconds"].as_f64().unwrap();
        assert!(
            (seconds - 300.0).abs() < 0.01,
            "output is {seconds}s, the video is 300s"
        );
        let worst = outcome["verification"]["worstMs"].as_f64().unwrap();
        assert!(worst <= 5.0, "finished track is {worst}ms out at worst");

        let _ = fs::remove_dir_all(&dir);
    }

    /// The pair the engine tests are written against, generated by the
    /// Python fixture script: DIR/org.wav and DIR/dub.wav, five minutes with
    /// four cuts, or `minutes` of one uncut stretch.
    fn synthetic_pair(
        app: &tauri::App<tauri::test::MockRuntime>,
        tag: &str,
        minutes: Option<u32>,
    ) -> PathBuf {
        let root = bridge::project_root(&app.handle().clone());
        let dir = std::env::temp_dir().join(format!("audiosync-e2e-{tag}-{}", std::process::id()));
        let mut command = Command::new(bridge::find_python(&root));
        command
            .arg(root.join("tests").join("test_dubsync.py"))
            .arg(&dir);
        if let Some(minutes) = minutes {
            command.arg(minutes.to_string());
        }
        let status = command.status().expect("fixture script should run");
        assert!(status.success(), "fixture generation failed");
        dir
    }

    fn invoke(
        webview: &tauri::WebviewWindow<tauri::test::MockRuntime>,
        cmd: &str,
        body: Value,
    ) -> Result<tauri::ipc::InvokeResponseBody, Value> {
        tauri::test::get_ipc_response(
            webview,
            tauri::webview::InvokeRequest {
                cmd: cmd.into(),
                callback: tauri::ipc::CallbackFn(0),
                error: tauri::ipc::CallbackFn(1),
                url: "tauri://localhost".parse().unwrap(),
                body: tauri::ipc::InvokeBody::Json(body),
                headers: Default::default(),
                invoke_key: tauri::test::INVOKE_KEY.to_string(),
            },
        )
    }

    /// Stop reaches a run in flight. The cancel used to queue behind the
    /// run's own lock, so it was written only once the run had finished
    /// on its own -- and, sent from the main thread, froze the window
    /// meanwhile. Here a five-minute pair's sync is stopped a moment after
    /// it starts and must come back cancelled within seconds, not minutes.
    #[test]
    fn cancel_reaches_a_dub_sync_in_flight() {
        if std::env::var_os("AUDIOSYNC_E2E").is_none() {
            eprintln!("skipped: set AUDIOSYNC_E2E=1 to run the engine end to end");
            return;
        }

        let app = tauri::test::mock_builder()
            .manage(BridgeHandle::default())
            .invoke_handler(tauri::generate_handler![start_dubsync, cancel_sync])
            .build(context())
            .expect("app should build");
        let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
            .build()
            .expect("webview should build");
        // Forty minutes: a sync the engine cannot finish in the moment
        // before it is stopped.
        let dir = synthetic_pair(&app, "cancel", Some(40));
        let request = serde_json::json!({
            "request": {
                "videoPath": dir.join("org.wav"),
                "dubPath": dir.join("dub.wav"),
                "videoTrack": 0,
                "dubTrack": 0,
                "codec": "wav",
                "outputPath": dir.join("synced.wav"),
                "mux": false,
                "language": null,
                "fillUnmatched": false,
                "overwrite": true,
            }
        });

        let runner = webview.clone();
        let started = std::time::Instant::now();
        let run = std::thread::spawn(move || invoke(&runner, "start_dubsync", request));
        // Let the engine start reading before stopping it.
        std::thread::sleep(Duration::from_secs(3));
        let cancel_sent = std::time::Instant::now();
        invoke(&webview, "cancel_sync", serde_json::json!({})).expect("cancel should be accepted");
        assert!(
            cancel_sent.elapsed() < Duration::from_secs(2),
            "cancel took {:?}: it waited for the run",
            cancel_sent.elapsed()
        );

        let outcome: Value = run
            .join()
            .expect("run thread")
            .expect("the command should resolve")
            .deserialize()
            .expect("the outcome should be JSON");
        app.state::<BridgeHandle>().shutdown();
        assert_eq!(outcome["cancelled"], true, "outcome: {outcome}");
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the run took {:?} to stop",
            started.elapsed()
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The waveform engine: a track read once with progress, then peaks per
    /// channel over any span of the plan's clock -- what the dub sync view
    /// draws, served by a process of its own so a sync in flight on the
    /// other cannot hold it up.
    #[test]
    fn waveform_engine_serves_peaks_per_channel() {
        if std::env::var_os("AUDIOSYNC_E2E").is_none() {
            eprintln!("skipped: set AUDIOSYNC_E2E=1 to run the engine end to end");
            return;
        }

        let app = tauri::test::mock_builder()
            .manage(BridgeHandle::default())
            .manage(WaveformBridge::default())
            .invoke_handler(tauri::generate_handler![waveform_build, waveform_peaks])
            .build(context())
            .expect("app should build");
        let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
            .build()
            .expect("webview should build");
        let dir = synthetic_pair(&app, "waveform", None);
        let path = dir.join("org.wav");

        let ready: Value = invoke(
            &webview,
            "waveform_build",
            serde_json::json!({ "request": { "path": path, "track": 0 } }),
        )
        .expect("build should succeed")
        .deserialize()
        .expect("ready should be JSON");
        assert_eq!(ready["type"], "waveformReady");
        assert!(
            (ready["durationS"].as_f64().unwrap() - 300.0).abs() < 0.1,
            "{ready}"
        );
        assert_eq!(ready["channels"], 1);

        let peaks: Value = invoke(
            &webview,
            "waveform_peaks",
            serde_json::json!({ "request": {
                "path": path, "track": 0, "startS": 0.0, "endS": 300.0, "buckets": 600, "speed": 1.0,
            } }),
        )
        .expect("peaks should succeed")
        .deserialize()
        .expect("peaks should be JSON");
        app.state::<WaveformBridge>().0.shutdown();
        let max = peaks["max"].as_array().expect("max per channel");
        assert_eq!(max.len(), 1, "{peaks}");
        let row = max[0].as_array().expect("one value per bucket");
        assert_eq!(row.len(), 600);
        let loudest = row.iter().filter_map(Value::as_f64).fold(0.0, f64::max);
        assert!(loudest > 0.1, "the fixture is not silent: {loudest}");
        let rms = peaks["rms"][0].as_array().expect("rms per bucket");
        assert!(rms.iter().filter_map(Value::as_f64).all(|v| v <= loudest));
        let _ = fs::remove_dir_all(&dir);
    }
}
