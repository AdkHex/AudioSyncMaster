@echo off
REM Development setup for Windows.
REM Usage:  dev.bat [--sidecar]   (--sidecar also builds the frozen engine)
setlocal enabledelayedexpansion

cd /d "%~dp0"

set VENV=python\.venv
set BUILD_SIDECAR=0
if "%~1"=="--sidecar" set BUILD_SIDECAR=1

echo [1/5] Python environment
if not exist "%VENV%" (
  python -m venv "%VENV%"
  if errorlevel 1 (
    echo   ERROR: could not create the virtual environment. Is Python installed?
    exit /b 1
  )
)
"%VENV%\Scripts\pip" install --quiet --upgrade pip
"%VENV%\Scripts\pip" install --quiet -r python\requirements-dev.txt
if errorlevel 1 (
  echo   ERROR: could not install Python dependencies.
  exit /b 1
)

echo [2/5] Checking for FFmpeg
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo   WARNING: ffmpeg is not on PATH.
  echo   Install it, or place ffmpeg.exe and ffprobe.exe in
  echo   src-tauri\resources\ffmpeg\ to have them bundled with the app.
) else (
  echo   found on PATH
)

echo [3/5] Node dependencies
call npm install --no-audit --no-fund
if errorlevel 1 exit /b 1

if "%BUILD_SIDECAR%"=="1" (
  echo [4/5] Building the sidecar
  "%VENV%\Scripts\pip" install --quiet pyinstaller
  "%VENV%\Scripts\pyinstaller" --onefile --clean --noconfirm ^
    --name audiosync-cli ^
    --paths . ^
    --hidden-import audiosync ^
    --hidden-import audiosync.analyze ^
    --hidden-import audiosync.batch ^
    --hidden-import audiosync.correlate ^
    --hidden-import audiosync.matching ^
    --hidden-import audiosync.media ^
    --hidden-import audiosync.mux ^
    --collect-all numpy ^
    --collect-all scipy ^
    python\bridge.py
  if errorlevel 1 exit /b 1

  if not exist src-tauri\bin mkdir src-tauri\bin
  copy /Y dist\audiosync-cli.exe src-tauri\bin\audiosync-cli.exe >nul
  copy /Y dist\audiosync-cli.exe src-tauri\bin\audiosync-cli-x86_64-pc-windows-msvc.exe >nul
  echo   sidecar written to src-tauri\bin\
) else (
  echo [4/5] Skipping sidecar build ^(pass --sidecar to build it^)
)

echo [5/5] Starting Tauri
call npm run tauri:dev

endlocal
