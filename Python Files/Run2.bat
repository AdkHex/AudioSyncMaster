@echo off
:: AUTO-SYNC FOR 'moviefiles' AND 'audiofiles'
setlocal enabledelayedexpansion

:: Set paths (NO NEED TO EDIT THESE)
set PYTHON_EXE=python
set SCRIPT_PATH=%~dp0main2.py
set VIDEO_FOLDER=%~dp0moviefiles
set AUDIO_FOLDER=%~dp0audiofiles
set PASSWORD=askvolx

:: Check folders
if not exist "%VIDEO_FOLDER%" (
    echo ERROR: Folder "moviefiles" not found!
    pause
    exit
)

if not exist "%AUDIO_FOLDER%" (
    echo ERROR: Folder "audiofiles" not found!
    pause
    exit
)

:: Find first audio file in 'audiofiles'
for %%a in ("%AUDIO_FOLDER%\*.*") do (
    set AUDIO_FILE=%%a
    goto :FOUND_AUDIO
)
echo ERROR: No audio file found in "audiofiles"!
pause
exit

:FOUND_AUDIO
echo Audio file: %AUDIO_FILE%

:: Count video files
set COUNT=0
for %%v in ("%VIDEO_FOLDER%\*.mp4", "%VIDEO_FOLDER%\*.mkv") do set /a COUNT+=1

if %COUNT% == 0 (
    echo ERROR: No videos found in "moviefiles"!
    pause
    exit
)

echo Found %COUNT% video files. Starting sync...
%PYTHON_EXE% "%SCRIPT_PATH%" "%VIDEO_FOLDER%" "%AUDIO_FILE%" --password %PASSWORD%

:: Save results with timestamp
set TIMESTAMP=%date:~-4%-%date:~3,2%-%date:~0,2%_%time:~0,2%-%time:~3,2%
%PYTHON_EXE% "%SCRIPT_PATH%" "%VIDEO_FOLDER%" "%AUDIO_FILE%" --password %PASSWORD% --output_csv "results_%TIMESTAMP%.csv"

echo Done! Check results_%TIMESTAMP%.csv
pause