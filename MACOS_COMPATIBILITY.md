# macOS Compatibility Report

## Overview
This document outlines the macOS compatibility issues found in the Voice-to-Voice Chatbot project and provides solutions for running it on macOS.

## Compatibility Analysis

### 1. **Requirements.txt - Compatible Packages**
All packages in requirements.txt are cross-platform compatible:
- ✅ **chatterbox-tts==0.1.2** - Pure Python, works on macOS
- ✅ **openai-whisper==20231117** - Cross-platform
- ✅ **pyaudio==0.2.13** - Works on macOS with PortAudio
- ✅ **webrtcvad==2.0.10** - Cross-platform
- ✅ **pynput==1.8.1** - macOS support for keyboard input
- ✅ **torch** - Supports MPS (Metal Performance Shaders) on Apple Silicon
- ✅ All other dependencies are platform-agnostic

### 2. **Audio Backend (PyAudio)**
- **Issue**: PyAudio requires PortAudio system library
- **Solution**: Install via Homebrew: `brew install portaudio`
- **Note**: macOS may require microphone permissions

### 3. **Installation Scripts**
- **Issue**: Only Windows scripts exist (install.bat, install.ps1)
- **Solution**: Created `install_macos.sh` with macOS-specific setup

### 4. **TTS Device Detection**
- **Current**: Auto-detects between cuda, mps, and cpu
- **macOS Support**: 
  - Apple Silicon (M1/M2/M3): Uses MPS acceleration
  - Intel Macs: Falls back to CPU
- **Code**: Already handles MPS in `src/tts/chatterbox_wrapper.py`:
  ```python
  if torch.backends.mps.is_available():
      device = "mps"
  ```

### 5. **File Path Handling**
- **Current**: Uses `pathlib.Path` throughout
- **Status**: ✅ Cross-platform compatible
- **Note**: Default voice path updated in config to use forward slashes

### 6. **Platform-Specific Code**
- **Finding**: No Windows-specific imports or code found
- **Keyboard handling**: `pynput` library is cross-platform
- **Audio**: PyAudio abstracts platform differences

## Required Changes for macOS

### 1. **Installation Script** ✅
Created `install_macos.sh` that:
- Checks for Python 3.8+
- Installs Homebrew if needed
- Installs system dependencies (portaudio, ffmpeg)
- Sets up virtual environment
- Installs PyTorch with MPS support for Apple Silicon
- Configures `.env` with macOS-appropriate settings

### 2. **Configuration Updates**
The `.env` template for macOS should use:
```
TTS_DEVICE=mps  # For Apple Silicon
# or
TTS_DEVICE=cpu  # For Intel Macs
```

### 3. **Audio Permissions**
macOS users need to grant microphone permissions when prompted by the system.

## Installation Instructions for macOS

1. **Make the script executable:**
   ```bash
   chmod +x install_macos.sh
   ```

2. **Run the installation:**
   ```bash
   ./install_macos.sh
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

4. **Configure API keys:**
   Edit `.env` and add your Gemini API key

5. **Run the application:**
   ```bash
   python main.py
   ```

## Known Issues and Workarounds

### 1. **Microphone Access**
- macOS will prompt for microphone permissions on first run
- Grant access in System Preferences > Security & Privacy > Microphone

### 2. **Audio Device Selection**
- Use `python fix_audio_device.py` to list and test audio devices
- Set specific device index if default doesn't work

### 3. **MPS Performance**
- Some operations may be slower on MPS than CUDA
- If experiencing issues, fall back to CPU: `--device cpu`

### 4. **Keyboard Permissions (Push-to-Talk)**
- macOS may require accessibility permissions for keyboard monitoring
- Grant in System Preferences > Security & Privacy > Accessibility

## Testing Recommendations

1. **Test audio input:**
   ```bash
   python test_mic_levels.py
   ```

2. **Test with CPU first:**
   ```bash
   python main.py --device cpu
   ```

3. **Then try MPS (Apple Silicon only):**
   ```bash
   python main.py --device mps
   ```

## Performance Considerations

- **Apple Silicon (M1/M2/M3)**: Use MPS for TTS acceleration
- **Intel Macs**: CPU-only, consider using smaller Whisper models
- **Whisper on CPU**: Already optimized in code (faster than GPU for real-time)

## Summary

The codebase is largely macOS-compatible with minimal changes needed:
1. ✅ All Python dependencies are cross-platform
2. ✅ Path handling uses pathlib (cross-platform)
3. ✅ Audio backend supports macOS with PortAudio
4. ✅ TTS supports MPS acceleration on Apple Silicon
5. ✅ No Windows-specific code found
6. ✅ Created macOS installation script

The main requirement is installing system dependencies (PortAudio) and using the provided installation script.