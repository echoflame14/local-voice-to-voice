# Voice-to-Voice Setup Script
Write-Host "Setting up Voice-to-Voice Environment..." -ForegroundColor Cyan

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed. Please install Python 3.8 or higher and try again." -ForegroundColor Red
    exit 1
}

# Check Python version
$pythonVersion = (python --version 2>&1).ToString().Split(" ")[1]
$versionParts = $pythonVersion.Split(".")
$majorVersion = [int]$versionParts[0]
$minorVersion = [int]$versionParts[1]

if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
    Write-Host "Error: This project requires Python 3.8 or higher. Your version: $pythonVersion" -ForegroundColor Red
    Write-Host "Please install a compatible Python version and try again." -ForegroundColor Yellow
    exit 1
}

# Check if LM Studio server is running
Write-Host "Checking LM Studio server..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:1234/v1/models" -Method GET -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "LM Studio server is running" -ForegroundColor Green
    }
} catch {
    Write-Host "Error: LM Studio server is not running at http://localhost:1234" -ForegroundColor Red
    Write-Host "Please start LM Studio, load a model, and start the server before continuing." -ForegroundColor Yellow
    Write-Host "Press any key to continue once the server is running..."
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}

# Create and activate virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install audio dependencies first
Write-Host "Installing audio dependencies..." -ForegroundColor Yellow
pip install wheel setuptools

# Install PortAudio binary
Write-Host "Installing PortAudio..." -ForegroundColor Yellow
$portAudioUrl = "http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz"
$portAudioPath = "portaudio.tgz"

try {
    # First try pip install
    pip install PyAudio
} catch {
    Write-Host "Direct PyAudio installation failed, trying alternative method..." -ForegroundColor Yellow
    # Try pipwin as fallback
    pip install pipwin
    pipwin install pyaudio
}

# Clone and install Chatterbox
Write-Host "Setting up Chatterbox TTS..." -ForegroundColor Yellow
if (Test-Path "chatterbox") {
    Remove-Item -Recurse -Force "chatterbox"
}
git clone https://github.com/resemble-ai/chatterbox.git
Push-Location chatterbox
pip install -e .
Pop-Location

# Install project dependencies
Write-Host "Installing project dependencies..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Create .env file
Write-Host "Creating configuration file..." -ForegroundColor Yellow
$envContent = @"
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=not-needed

# Model Configuration
WHISPER_MODEL_SIZE=base
TTS_DEVICE=cuda

# Voice Configuration
VOICE_REFERENCE_PATH=./voices/default.wav
VOICE_EXAGGERATION=0.5
VOICE_CFG_WEIGHT=0.5

# Audio Settings
SAMPLE_RATE=16000
CHUNK_SIZE=1024
VAD_AGGRESSIVENESS=1

# Chat Settings
SYSTEM_PROMPT=You are a helpful voice assistant. Keep your responses concise and natural for speech.
MAX_RESPONSE_TOKENS=150
TEMPERATURE=0.7
"@

Set-Content -Path ".env" -Value $envContent -Encoding UTF8

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "voices" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "conversation_logs" | Out-Null

# Final check and instructions
Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "`nTo start the voice assistant:" -ForegroundColor Cyan
Write-Host "1. Ensure LM Studio server is running with a loaded model" -ForegroundColor White
Write-Host "2. Run: python main.py" -ForegroundColor White

Write-Host "`nFor troubleshooting and more information, please refer to the README.md" -ForegroundColor Yellow 