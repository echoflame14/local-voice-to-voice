@echo off
echo Installing Windows TTS support (pywin32)...
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Install pywin32
echo Installing pywin32...
pip install pywin32

echo.
echo Installation complete!
echo You can now test TTS with: python test_tts.py
echo.
pause