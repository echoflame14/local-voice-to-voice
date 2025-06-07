# Elegant Chatbot Installation Script for Windows (PowerShell)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "🎨 Elegant Chatbot Installer (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check Python version
Write-Host "`n📍 Python version:" -ForegroundColor Yellow
python --version

# Create virtual environment
Write-Host "`n📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`n📦 Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Windows-specific PyAudio installation
Write-Host "`n🔊 Installing PyAudio for Windows..." -ForegroundColor Yellow
pip install pipwin
pipwin install pyaudio

# Try to install Chatterbox TTS
Write-Host "`n🎵 Installing Chatterbox TTS (optional)..." -ForegroundColor Yellow
try {
    pip install git+https://github.com/resemble-ai/chatterbox.git
} catch {
    Write-Host "   ⚠️  Chatterbox installation failed, will use fallback TTS" -ForegroundColor Yellow
}

# Create directories
Write-Host "`n📁 Creating directories..." -ForegroundColor Yellow
@("data", "logs", "memories", "voices") | ForEach-Object {
    if (!(Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}

# Check for API key
Write-Host "`n🔑 Checking API key..." -ForegroundColor Yellow
if (!$env:OPENAI_API_KEY) {
    Write-Host "   ⚠️  OPENAI_API_KEY not set!" -ForegroundColor Red
    Write-Host "   Please set it with: `$env:OPENAI_API_KEY='your-key-here'" -ForegroundColor Yellow
} else {
    Write-Host "   ✅ OpenAI API key found" -ForegroundColor Green
}

# Run tests
Write-Host "`n🧪 Running basic tests..." -ForegroundColor Yellow
python test_basic.py

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host "`nTo run the chatbot:" -ForegroundColor Yellow
Write-Host "  1. Activate venv: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Set API key: `$env:OPENAI_API_KEY='your-key'" -ForegroundColor White
Write-Host "  3. Run: python main.py" -ForegroundColor White
Write-Host "================================" -ForegroundColor Cyan