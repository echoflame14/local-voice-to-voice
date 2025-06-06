@echo off
echo Converting josh.m4a to josh.wav...

REM Check if ffmpeg exists in current directory
if exist "ffmpeg.exe" (
    ffmpeg.exe -i voices\josh.m4a -ar 16000 -ac 1 -y voices\josh.wav
    echo Conversion complete!
) else (
    REM Try system ffmpeg
    ffmpeg -i voices\josh.m4a -ar 16000 -ac 1 -y voices\josh.wav 2>nul
    if %errorlevel% neq 0 (
        echo ERROR: ffmpeg not found. Please extract ffmpeg.zip first.
        echo You can download from: https://ffmpeg.org/download.html
    ) else (
        echo Conversion complete!
    )
)

pause