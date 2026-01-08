@echo off
setlocal

echo [1/6] Creating venv...
if not exist python\.venv (
  python -m venv python\.venv
)

echo [2/6] Installing Python deps...
python\.venv\Scripts\pip install --upgrade pip
python\.venv\Scripts\pip install -r python\requirements.txt
python\.venv\Scripts\pip install pyinstaller

echo [3/7] Preparing ffmpeg...
if not exist resources\ffmpeg (
  mkdir resources\ffmpeg
)
for /f "delims=" %%F in ('where ffmpeg 2^>nul') do copy /Y "%%F" "resources\ffmpeg\ffmpeg.exe" >nul
for /f "delims=" %%F in ('where ffprobe 2^>nul') do copy /Y "%%F" "resources\ffmpeg\ffprobe.exe" >nul

echo [4/7] Building sidecar...
python\.venv\Scripts\pyinstaller --onefile --clean ^
  --add-data "Python Files;Python Files" ^
  --add-data "resources\ffmpeg;resources\ffmpeg" ^
  --hidden-import numpy ^
  --hidden-import scipy ^
  --hidden-import librosa ^
  --hidden-import soundfile ^
  --hidden-import numba ^
  --hidden-import llvmlite ^
  --hidden-import rich ^
  --hidden-import pymediainfo ^
  --hidden-import packaging ^
  --collect-all numpy ^
  --collect-all scipy ^
  --collect-all librosa ^
  --collect-all soundfile ^
  --collect-all rich ^
  --collect-all pymediainfo ^
  python\bridge.py -n audiosync-cli

echo [5/7] Copying sidecar...
if not exist src-tauri\bin (
  mkdir src-tauri\bin
)
copy /Y dist\audiosync-cli.exe src-tauri\bin\audiosync-cli-x86_64-pc-windows-msvc.exe

echo [6/7] Installing npm deps...
call npm install

echo [7/7] Starting Tauri dev...
call npm run tauri:dev

endlocal
