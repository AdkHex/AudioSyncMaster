use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tauri::{AppHandle, Emitter, Manager, State, Window};
use tauri::path::BaseDirectory;
use tauri_plugin_dialog::DialogExt;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FileItem {
  name: String,
  path: String,
  #[serde(rename = "type")]
  file_type: String,
  size: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PickResponse {
  folder: Option<String>,
  files: Vec<FileItem>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SyncRequest {
  mode: String,
  video_folder: Option<String>,
  audio_folder: Option<String>,
  audio_file: Option<String>,
  video_files: Option<Vec<String>>,
  segment_duration: f64,
  match_pattern: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SyncResult {
  videoFile: String,
  audioFile: String,
  startDelay: Option<f64>,
  endDelay: Option<f64>,
  error: Option<String>,
  elapsedMs: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum BridgeMessage {
  #[serde(rename = "progress")]
  Progress {
    processed: usize,
    total: usize,
    current: Option<String>,
  },
  #[serde(rename = "file_start")]
  FileStart { file: String },
  #[serde(rename = "file_end")]
  FileEnd { file: String, elapsed_ms: u64 },
  #[serde(rename = "file_progress")]
  FileProgress { file: String, percent: u8 },
  #[serde(rename = "log")]
  Log { message: String },
  #[serde(rename = "result")]
  Result {
    videoFile: String,
    audioFile: String,
    startDelay: Option<f64>,
    endDelay: Option<f64>,
    error: Option<String>,
    elapsed_ms: Option<u64>,
  },
  #[serde(rename = "done")]
  Done { results: Vec<SyncResult> },
}

#[derive(Clone)]
struct SyncState {
  cancel: Arc<AtomicBool>,
}

impl SyncState {
  fn new() -> Self {
    Self {
      cancel: Arc::new(AtomicBool::new(false)),
    }
  }
}

async fn pick_folder_async(window: Window) -> Option<PathBuf> {
  let (tx, rx) = std::sync::mpsc::channel::<Option<PathBuf>>();
  window.dialog().file().pick_folder(move |path| {
    let resolved = path.and_then(|p| p.into_path().ok());
    let _ = tx.send(resolved);
  });
  tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
    .await
    .ok()
    .flatten()
}

async fn pick_file_async(window: Window) -> Option<PathBuf> {
  let (tx, rx) = std::sync::mpsc::channel::<Option<PathBuf>>();
  window.dialog().file().pick_file(move |path| {
    let resolved = path.and_then(|p| p.into_path().ok());
    let _ = tx.send(resolved);
  });
  tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
    .await
    .ok()
    .flatten()
}

async fn save_file_async(window: Window, default_name: &str) -> Option<PathBuf> {
  let (tx, rx) = std::sync::mpsc::channel::<Option<PathBuf>>();
  window
    .dialog()
    .file()
    .set_file_name(default_name)
    .save_file(move |path| {
      let resolved = path.and_then(|p| p.into_path().ok());
      let _ = tx.send(resolved);
    });
  tauri::async_runtime::spawn_blocking(move || rx.recv().ok().flatten())
    .await
    .ok()
    .flatten()
}

#[tauri::command]
async fn pick_video_files(window: Window, mode: String) -> Result<PickResponse, String> {
  if mode != "movie" && mode != "series" {
    return Ok(PickResponse {
      folder: None,
      files: Vec::new(),
    });
  }

  let folder = pick_folder_async(window).await;
  let folder = match folder {
    Some(path) => path,
    None => {
      return Ok(PickResponse {
        folder: None,
        files: Vec::new(),
      })
    }
  };

  let files = if mode == "movie" {
    list_movie_videos(&folder)
  } else {
    list_folder_files(&folder)
  };

  Ok(PickResponse {
    folder: Some(folder.to_string_lossy().to_string()),
    files,
  })
}

#[tauri::command]
async fn pick_audio_files(window: Window, mode: String) -> Result<PickResponse, String> {
  if mode == "movie" {
    let file = pick_file_async(window).await;
    let file = match file {
      Some(path) => path,
      None => {
        return Ok(PickResponse {
          folder: None,
          files: Vec::new(),
        })
      }
    };
    let name = file
      .file_name()
      .map(|s| s.to_string_lossy().to_string())
      .unwrap_or_default();
    let size = fs::metadata(&file).map(|meta| meta.len()).ok();
    return Ok(PickResponse {
      folder: file.parent().map(|p| p.to_string_lossy().to_string()),
      files: vec![FileItem {
        name,
        path: file.to_string_lossy().to_string(),
        file_type: "audio".to_string(),
        size,
      }],
    });
  }

  let folder = pick_folder_async(window).await;
  let folder = match folder {
    Some(path) => path,
    None => {
      return Ok(PickResponse {
        folder: None,
        files: Vec::new(),
      })
    }
  };

  let files = list_folder_files(&folder)
    .into_iter()
    .map(|mut item| {
      item.file_type = "audio".to_string();
      item
    })
    .collect();

  Ok(PickResponse {
    folder: Some(folder.to_string_lossy().to_string()),
    files,
  })
}

#[tauri::command]
async fn start_sync(
  app: AppHandle,
  state: State<'_, SyncState>,
  request: SyncRequest,
) -> Result<Vec<SyncResult>, String> {
  state.cancel.store(false, Ordering::SeqCst);
  let handle = app.clone();
  let cancel = state.cancel.clone();
  tauri::async_runtime::spawn_blocking(move || run_bridge(handle, request, cancel))
    .await
    .map_err(|err| err.to_string())?
}

#[tauri::command]
async fn export_csv(window: Window, results: Vec<SyncResult>) -> Result<String, String> {
  let path = save_file_async(window, "sync-results.csv").await;
  let Some(path) = path else {
    return Err("Export canceled".to_string());
  };

  let mut csv = String::from("Video,Audio,Start Delay (ms),End Delay (ms),Elapsed (ms),Error\n");
  for result in results {
    let start = result.startDelay.map(|v| v.to_string()).unwrap_or_default();
    let end = result.endDelay.map(|v| v.to_string()).unwrap_or_default();
    let elapsed = result.elapsedMs.map(|v| v.to_string()).unwrap_or_default();
    let err = result.error.unwrap_or_default();
    csv.push_str(&format!(
      "\"{}\",\"{}\",{},{},{},\"{}\"\n",
      result.videoFile, result.audioFile, start, end, elapsed, err
    ));
  }

  fs::write(&path, csv.as_bytes()).map_err(|err| err.to_string())?;
  Ok(path.to_string_lossy().to_string())
}

fn list_movie_videos(folder: &Path) -> Vec<FileItem> {
  let mut items = Vec::new();
  let exts = ["mp4", "mkv", "webm", "avi", "mov"];
  if let Ok(entries) = fs::read_dir(folder) {
    for entry in entries.flatten() {
      let path = entry.path();
      if !path.is_file() {
        continue;
      }
      let size = fs::metadata(&path).map(|meta| meta.len()).ok();
      let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
      if !exts.contains(&ext.as_str()) {
        continue;
      }
      let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
      items.push(FileItem {
        name,
        path: path.to_string_lossy().to_string(),
        file_type: "video".to_string(),
        size,
      });
    }
  }
  items
}

fn list_folder_files(folder: &Path) -> Vec<FileItem> {
  let mut items = Vec::new();
  if let Ok(entries) = fs::read_dir(folder) {
    for entry in entries.flatten() {
      let path = entry.path();
      if !path.is_file() {
        continue;
      }
      let size = fs::metadata(&path).map(|meta| meta.len()).ok();
      let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
      items.push(FileItem {
        name,
        path: path.to_string_lossy().to_string(),
        file_type: "video".to_string(),
        size,
      });
    }
  }
  items
}

fn find_sidecar_path(app: &AppHandle) -> Option<PathBuf> {
  let mut candidates = vec![
    app.path().resolve("bin/audiosync-cli", BaseDirectory::Resource).ok(),
    app.path().resolve("bin/audiosync-cli.exe", BaseDirectory::Resource).ok(),
    app
      .path()
      .resolve("bin/audiosync-cli-x86_64-pc-windows-msvc.exe", BaseDirectory::Resource)
      .ok(),
    Some(PathBuf::from("bin/audiosync-cli.exe")),
    Some(PathBuf::from("bin/audiosync-cli-x86_64-pc-windows-msvc.exe")),
    Some(PathBuf::from("bin/audiosync-cli")),
  ];

  if let Ok(cwd) = std::env::current_dir() {
    candidates.push(Some(cwd.join("bin/audiosync-cli.exe")));
    candidates.push(Some(cwd.join("bin/audiosync-cli-x86_64-pc-windows-msvc.exe")));
    candidates.push(Some(cwd.join("bin/audiosync-cli")));
    if let Some(parent) = cwd.parent() {
      candidates.push(Some(parent.join("src-tauri/bin/audiosync-cli.exe")));
      candidates.push(Some(
        parent.join("src-tauri/bin/audiosync-cli-x86_64-pc-windows-msvc.exe"),
      ));
      candidates.push(Some(parent.join("src-tauri/bin/audiosync-cli")));
    }
  }

  for candidate in candidates.into_iter().flatten() {
    if candidate.exists() {
      return Some(candidate);
    }
  }
  None
}

fn find_python_exe() -> Option<PathBuf> {
  let mut candidates = vec![
    PathBuf::from("python/.venv/Scripts/python.exe"),
    PathBuf::from("../python/.venv/Scripts/python.exe"),
    PathBuf::from("python/.venv/bin/python"),
    PathBuf::from("../python/.venv/bin/python"),
  ];

  if let Ok(cwd) = std::env::current_dir() {
    candidates.push(cwd.join("python/.venv/Scripts/python.exe"));
    candidates.push(cwd.join("../python/.venv/Scripts/python.exe"));
    candidates.push(cwd.join("python/.venv/bin/python"));
    candidates.push(cwd.join("../python/.venv/bin/python"));
  }

  for candidate in candidates {
    if candidate.exists() {
      return Some(candidate);
    }
  }
  None
}

fn find_bridge_path() -> Option<PathBuf> {
  let candidates = [
    PathBuf::from("python/bridge.py"),
    PathBuf::from("../python/bridge.py"),
    PathBuf::from("../../python/bridge.py"),
  ];
  for candidate in candidates {
    if candidate.exists() {
      return Some(candidate);
    }
  }
  None
}

#[tauri::command]
fn cancel_sync(state: State<'_, SyncState>) -> Result<(), String> {
  state.cancel.store(true, Ordering::SeqCst);
  Ok(())
}

#[derive(Debug, Serialize)]
struct MediaProbe {
  has_audio: bool,
  has_video: bool,
  duration: Option<f64>,
}

#[tauri::command]
fn probe_media(path: String) -> Result<MediaProbe, String> {
  let output = Command::new("ffprobe")
    .args([
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-show_streams",
      "-of",
      "json",
      &path,
    ])
    .output()
    .map_err(|err| err.to_string())?;

  if !output.status.success() {
    return Err("ffprobe failed".to_string());
  }

  let value: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|err| err.to_string())?;
  let streams = value.get("streams").and_then(|v| v.as_array()).cloned().unwrap_or_default();
  let mut has_audio = false;
  let mut has_video = false;
  for stream in streams {
    if let Some(kind) = stream.get("codec_type").and_then(|v| v.as_str()) {
      if kind == "audio" {
        has_audio = true;
      } else if kind == "video" {
        has_video = true;
      }
    }
  }

  let duration = value
    .get("format")
    .and_then(|v| v.get("duration"))
    .and_then(|v| v.as_str())
    .and_then(|v| v.parse::<f64>().ok());

  Ok(MediaProbe {
    has_audio,
    has_video,
    duration,
  })
}

#[tauri::command]
fn open_output_folder(path: String) -> Result<(), String> {
  let path = PathBuf::from(path);
  if !path.exists() {
    return Err("Path not found".to_string());
  }

  #[cfg(target_os = "windows")]
  {
    Command::new("explorer")
      .arg("/select,")
      .arg(path)
      .spawn()
      .map_err(|err| err.to_string())?;
  }

  #[cfg(not(target_os = "windows"))]
  {
    let folder = path.parent().unwrap_or(Path::new("."));
    Command::new("open")
      .arg(folder)
      .spawn()
      .or_else(|_| Command::new("xdg-open").arg(folder).spawn())
      .map_err(|err| err.to_string())?;
  }

  Ok(())
}

fn run_bridge(
  app: AppHandle,
  request: SyncRequest,
  cancel: Arc<AtomicBool>,
) -> Result<Vec<SyncResult>, String> {
  let payload = serde_json::to_string(&request).map_err(|err| err.to_string())?;

  let mut command = if let Some(sidecar_path) = find_sidecar_path(&app) {
    let _ = app.emit(
      "sync-log",
      format!("Using sidecar: {}", sidecar_path.to_string_lossy()),
    );
    Command::new(sidecar_path)
  } else {
    let bridge_path = find_bridge_path().ok_or_else(|| "bridge.py not found".to_string())?;
    let python_exe = find_python_exe().unwrap_or_else(|| PathBuf::from("python"));
    let _ = app.emit(
      "sync-log",
      format!(
        "Sidecar not found. Falling back to python: {}",
        python_exe.to_string_lossy()
      ),
    );
    let mut cmd = Command::new(python_exe);
    cmd.arg(bridge_path);
    cmd
  };

  command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
  let mut child = match command.spawn() {
    Ok(child) => child,
    Err(err) => {
      let _ = app.emit("sync-log", format!("Failed to start process: {err}"));
      return Err(err.to_string());
    }
  };

  if let Some(mut stdin) = child.stdin.take() {
    stdin.write_all(payload.as_bytes()).map_err(|err| err.to_string())?;
  }

  let stdout = child.stdout.take().ok_or_else(|| "Failed to capture stdout".to_string())?;
  let stderr = child.stderr.take().ok_or_else(|| "Failed to capture stderr".to_string())?;

  let app_for_stderr = app.clone();
  std::thread::spawn(move || {
    let reader = BufReader::new(stderr);
    for line in reader.lines().flatten() {
      let _ = app_for_stderr.emit("sync-log", line);
    }
  });

  let mut results: Vec<SyncResult> = Vec::new();
  let reader = BufReader::new(stdout);
  for line in reader.lines().flatten() {
    if cancel.load(Ordering::SeqCst) {
      let _ = app.emit("sync-log", "Sync canceled by user.");
      let _ = child.kill();
      return Err("Canceled".to_string());
    }
    let line = line.trim();
    if line.is_empty() {
      continue;
    }
    let message: Result<BridgeMessage, _> = serde_json::from_str(line);
    match message {
      Ok(BridgeMessage::Progress { processed, total, current }) => {
        let _ = app.emit(
          "sync-progress",
          serde_json::json!({ "processed": processed, "total": total, "current": current }),
        );
      }
      Ok(BridgeMessage::FileStart { file }) => {
        let _ = app.emit("sync-file-start", serde_json::json!({ "file": file }));
      }
      Ok(BridgeMessage::FileEnd { file, elapsed_ms }) => {
        let _ = app.emit(
          "sync-file-end",
          serde_json::json!({ "file": file, "elapsed_ms": elapsed_ms }),
        );
      }
      Ok(BridgeMessage::FileProgress { file, percent }) => {
        let _ = app.emit(
          "sync-file-progress",
          serde_json::json!({ "file": file, "percent": percent }),
        );
      }
      Ok(BridgeMessage::Log { message }) => {
        let _ = app.emit("sync-log", message);
      }
      Ok(BridgeMessage::Result {
        videoFile,
        audioFile,
        startDelay,
        endDelay,
        error,
        elapsed_ms,
      }) => {
        let result = SyncResult {
          videoFile,
          audioFile,
          startDelay,
          endDelay,
          error,
          elapsedMs: elapsed_ms,
        };
        results.push(result.clone());
        let _ = app.emit("sync-result", result);
      }
      Ok(BridgeMessage::Done { results: final_results }) => {
        results = final_results;
        let _ = app.emit("sync-done", &results);
      }
      Err(err) => {
        let _ = app.emit("sync-log", format!("Invalid bridge message: {err}"));
      }
    }
  }

  let status = child.wait().map_err(|err| err.to_string())?;
  if !status.success() {
    return Err("Sync process failed".to_string());
  }

  Ok(results)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_log::Builder::default().level(log::LevelFilter::Info).build())
    .invoke_handler(tauri::generate_handler![
      pick_video_files,
      pick_audio_files,
      start_sync,
      cancel_sync,
      probe_media,
      open_output_folder,
      export_csv
    ])
    .manage(SyncState::new())
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
