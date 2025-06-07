@echo off
REM Elegant Chatbot Installation Script for Windows

echo ================================
echo 🎨 Elegant Chatbot Installer (Windows)
echo ================================

REM Check Python version
python --version
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo 📦 Installing requirements...
pip install -r requirements.txt

REM Windows-specific PyAudio installation
echo.
echo 🔊 Installing PyAudio for Windows...
pip install pipwin
pipwin install pyaudio

REM Try to install Chatterbox TTS
echo.
echo 🎵 Installing Chatterbox TTS (optional)...
pip install git+https://github.com/resemble-ai/chatterbox.git
if errorlevel 1 (
    echo    ⚠️  Chatterbox installation failed, will use fallback TTS
)

REM Create directories
echo.
echo 📁 Creating directories...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "memories" mkdir memories
if not exist "voices" mkdir voices

REM Check for API key
echo.
echo 🔑 Checking API key...
if "%OPENAI_API_KEY%"=="" (
    echo    ⚠️  OPENAI_API_KEY not set!
    echo    Please set it with: set OPENAI_API_KEY=your-key-here
) else (
    echo    ✅ OpenAI API key found
)

REM Run tests
echo.
echo 🧪 Running basic tests...
python test_basic.py

echo.
echo ================================
echo ✅ Installation complete!
echo.
echo To run the chatbot:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Set API key: set OPENAI_API_KEY=your-key
echo   3. Run: python main.py
echo ================================
pause