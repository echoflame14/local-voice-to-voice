# PowerShell script to convert josh.m4a to josh.wav

Write-Host "🎤 Converting josh.m4a to josh.wav..." -ForegroundColor Cyan

$inputFile = "voices\josh.m4a"
$outputFile = "voices\josh.wav"

# Check if input file exists
if (-not (Test-Path $inputFile)) {
    Write-Host "❌ Error: $inputFile not found!" -ForegroundColor Red
    exit 1
}

# Try to find ffmpeg
$ffmpegPath = $null

# Check current directory
if (Test-Path "ffmpeg.exe") {
    $ffmpegPath = ".\ffmpeg.exe"
} elseif (Test-Path "ffmpeg\bin\ffmpeg.exe") {
    $ffmpegPath = ".\ffmpeg\bin\ffmpeg.exe"
} else {
    # Try system PATH
    $ffmpegPath = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegPath) {
        $ffmpegPath = "ffmpeg"
    }
}

if ($ffmpegPath) {
    Write-Host "✅ Found ffmpeg at: $ffmpegPath" -ForegroundColor Green
    
    # Convert with optimal settings for Chatterbox TTS
    $arguments = @(
        "-i", $inputFile,
        "-ar", "16000",     # 16kHz sample rate
        "-ac", "1",         # Mono channel
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-y",               # Overwrite output
        $outputFile
    )
    
    & $ffmpegPath $arguments
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Conversion successful!" -ForegroundColor Green
        Write-Host "📁 Output: $outputFile" -ForegroundColor Yellow
        Write-Host "🚀 You can now run: python main.py --use-gemini" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Conversion failed!" -ForegroundColor Red
    }
} else {
    Write-Host "❌ ffmpeg not found!" -ForegroundColor Red
    Write-Host "Please extract ffmpeg.zip or download from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    Write-Host "After extracting, run this script again." -ForegroundColor Yellow
}