//! Encode and FPSConv: the audio queue and its quick questions (Windows only,
//! see ENCODE.md).
//!
//! The queue runs in an engine process of its own, the EncodeBridge, so a
//! 40-minute Atmos encode never holds up Movies, Dub sync or Subsync. Its
//! `audioQueue` session opens on the first Start and stays open while the app
//! runs; jobs added later, cancels and worker changes reach it through
//! [`audio_send`], which writes straight to the process. Quick questions --
//! probing, previews, the doctor, settings -- go to Subsync's info engine.
//!
//! Every engine event is forwarded to the webview as `encode-event`,
//! unchanged; the frontend switches on `type`.

use std::time::Duration;

use serde_json::{json, Value};
use tauri::{AppHandle, Runtime, State};

use crate::bridge::BridgeHandle;
use crate::subsync::{exchange_on, SubsInfoBridge};
use crate::EVENT_TIMEOUT;

const WINDOWS_ONLY: &str = "Encode is available on Windows only";
const CHANNEL: &str = "encode-event";

/// Commands [`audio_send`] may write to the running queue. Anything else
/// would either be refused by the engine or, worse, queue behind the session.
const QUEUE_COMMANDS: &[&str] = &[
    "audioEnqueue",
    "audioCancelJob",
    "audioCancelAll",
    "audioSetWorkers",
];

/// The audio queue's own engine process.
#[derive(Clone, Default)]
pub(crate) struct EncodeBridge(pub BridgeHandle);

impl EncodeBridge {
    /// On app exit: cancel every job, so no DEE or ffmpeg outlives the app,
    /// then ask the engine to stop. Written without taking the bridge, which
    /// the queue session holds for as long as it is open.
    pub(crate) fn close(&self) {
        let _ = self.0.send_now(&json!({ "command": "audioCancelAll" }));
        let _ = self.0.send_now(&json!({ "command": "shutdown" }));
    }
}

/// Whether the encode pages may run here: Windows, or anywhere with
/// `AUDIOSYNC_ENCODE_ANY_OS=1` (the engine honours the same switch).
fn allowed(windows: bool, any_os: Option<&str>) -> Result<(), String> {
    if windows || any_os == Some("1") {
        Ok(())
    } else {
        Err(WINDOWS_ONLY.to_string())
    }
}

fn ensure_windows() -> Result<(), String> {
    allowed(
        cfg!(windows),
        std::env::var("AUDIOSYNC_ENCODE_ANY_OS").ok().as_deref(),
    )
}

/// The command of a message for the running queue, if it is one [`audio_send`] may send.
fn queue_command(message: &Value) -> Result<&str, String> {
    let command = message
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if QUEUE_COMMANDS.contains(&command) {
        Ok(command)
    } else {
        Err(format!("Not a queue command: {command}"))
    }
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

async fn run<R: Runtime>(
    app: AppHandle<R>,
    handle: BridgeHandle,
    payload: Value,
    terminal: &'static str,
    timeout: Duration,
) -> Result<Value, String> {
    ensure_windows()?;
    tauri::async_runtime::spawn_blocking(move || {
        exchange_on(&app, &handle, payload, terminal, timeout, CHANNEL)
    })
    .await
    .map_err(|err| err.to_string())?
}

// ---------------------------------------------------------------- the queue

/// Open the queue session and hold it until the app exits. Returns `Err` if
/// the engine process dies; the page then marks its live jobs as failed, and
/// the next Start launches a new process.
#[tauri::command]
pub(crate) async fn start_audio_queue<R: Runtime>(
    app: AppHandle<R>,
    handle: State<'_, EncodeBridge>,
    request: Value,
) -> Result<Value, String> {
    let payload = with_command(request, "audioQueue");
    run(
        app,
        handle.inner().0.clone(),
        payload,
        "audioQueueDone",
        EVENT_TIMEOUT,
    )
    .await
}

/// Add jobs to, cancel in, or resize the running queue. Answered by an
/// `…Ack` event on `encode-event`.
#[tauri::command]
pub(crate) fn audio_send(handle: State<'_, EncodeBridge>, message: Value) -> Result<(), String> {
    ensure_windows()?;
    queue_command(&message)?;
    handle.0.send_now(&message)
}

// ------------------------------------------------------------ quick queries

#[tauri::command]
pub(crate) async fn audio_probe<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    paths: Vec<String>,
) -> Result<Value, String> {
    let payload = json!({ "command": "audioProbe", "paths": paths });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "audioProbeResult",
        Duration::from_secs(600),
    )
    .await
}

#[tauri::command]
pub(crate) async fn audio_reference<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    path: String,
) -> Result<Value, String> {
    let payload = json!({ "command": "audioReference", "path": path });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "audioReferenceResult",
        Duration::from_secs(120),
    )
    .await
}

#[tauri::command]
pub(crate) async fn audio_suggest<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    reference: Value,
    items: Value,
) -> Result<Value, String> {
    let payload = json!({ "command": "audioSuggest", "reference": reference, "items": items });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "audioSuggestResult",
        Duration::from_secs(60),
    )
    .await
}

#[tauri::command]
pub(crate) async fn encode_resolve<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    streams: Value,
    encode: Value,
) -> Result<Value, String> {
    let payload = json!({ "command": "encodeResolve", "streams": streams, "encode": encode });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "encodeResolveResult",
        Duration::from_secs(60),
    )
    .await
}

#[tauri::command]
pub(crate) async fn encode_doctor<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
) -> Result<Value, String> {
    run(
        app,
        info.inner().0.clone(),
        json!({ "command": "encodeDoctor" }),
        "encodeDoctorResult",
        Duration::from_secs(180),
    )
    .await
}

#[tauri::command]
pub(crate) async fn encode_set_dee<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    dee_path: String,
) -> Result<Value, String> {
    let payload = json!({ "command": "encodeSetDee", "dee_path": dee_path });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "encodeSetDeeResult",
        Duration::from_secs(180),
    )
    .await
}

#[tauri::command]
pub(crate) async fn audio_mediainfo<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    path: String,
    folder: Option<String>,
    stem: Option<String>,
) -> Result<Value, String> {
    let payload =
        json!({ "command": "mediainfoReport", "path": path, "folder": folder, "stem": stem });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "mediainfoReportResult",
        Duration::from_secs(180),
    )
    .await
}

/// Look for FPSConv's settings (`apply: false`), or carry them over (`apply: true`).
#[tauri::command]
pub(crate) async fn fpsconv_import<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    apply: bool,
) -> Result<Value, String> {
    let payload = json!({ "command": "fpsconvImport", "apply": apply });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "fpsconvImportResult",
        Duration::from_secs(60),
    )
    .await
}

#[tauri::command]
pub(crate) async fn encode_settings_load<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
) -> Result<Value, String> {
    run(
        app,
        info.inner().0.clone(),
        json!({ "command": "encodeSettingsLoad" }),
        "encodeSettingsResult",
        Duration::from_secs(30),
    )
    .await
}

/// Save a partial of the engine's settings file; the reply carries the whole file.
#[tauri::command]
pub(crate) async fn encode_settings_save<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    partial: Value,
) -> Result<Value, String> {
    let payload = json!({ "command": "encodeSettingsSave", "partial": partial });
    run(
        app,
        info.inner().0.clone(),
        payload,
        "encodeSettingsResult",
        Duration::from_secs(30),
    )
    .await
}

/// The finished-jobs history: `load`, `remove` (one job) or `clear`.
#[tauri::command]
pub(crate) async fn audio_history<R: Runtime>(
    app: AppHandle<R>,
    info: State<'_, SubsInfoBridge>,
    action: String,
    job_id: Option<String>,
) -> Result<Value, String> {
    let payload = match action.as_str() {
        "load" => json!({ "command": "audioHistoryLoad" }),
        "remove" => json!({ "command": "audioHistoryRemove", "jobId": job_id }),
        "clear" => json!({ "command": "audioHistoryClear" }),
        other => return Err(format!("Unknown history action: {other}")),
    };
    run(
        app,
        info.inner().0.clone(),
        payload,
        "audioHistoryResult",
        Duration::from_secs(30),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_queue_commands_reach_the_running_queue() {
        for command in QUEUE_COMMANDS {
            assert_eq!(queue_command(&json!({ "command": command })), Ok(*command));
        }
        for command in [
            "audioQueue",
            "shutdown",
            "cancel",
            "analyze",
            "packInstall",
            "",
        ] {
            assert!(
                queue_command(&json!({ "command": command })).is_err(),
                "{command} was let through"
            );
        }
        assert!(queue_command(&json!({})).is_err());
        assert!(queue_command(&json!("audioEnqueue")).is_err());
    }

    #[test]
    fn the_guard_refuses_off_windows() {
        assert_eq!(allowed(true, None), Ok(()));
        assert_eq!(allowed(false, None), Err(WINDOWS_ONLY.to_string()));
        assert_eq!(allowed(false, Some("0")), Err(WINDOWS_ONLY.to_string()));
        assert_eq!(allowed(false, Some("1")), Ok(()));
    }

    #[test]
    fn this_host_is_guarded_as_its_os_says() {
        if std::env::var("AUDIOSYNC_ENCODE_ANY_OS").as_deref() == Ok("1") {
            return;
        }
        assert_eq!(ensure_windows().is_ok(), cfg!(windows));
    }

    #[test]
    fn requests_get_their_command() {
        assert_eq!(
            with_command(json!({ "workers": 2 }), "audioQueue"),
            json!({ "workers": 2, "command": "audioQueue" })
        );
        assert_eq!(
            with_command(json!(null), "audioQueue"),
            json!({ "command": "audioQueue" })
        );
    }
}
