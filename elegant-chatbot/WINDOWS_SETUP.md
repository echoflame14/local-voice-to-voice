# Windows 11 Setup Guide for Elegant Chatbot 🪟

## Prerequisites

1. **Python 3.9 or higher**
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Git** (for installing Chatterbox)
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **OpenAI API Key**
   - Get from [platform.openai.com](https://platform.openai.com/api-keys)

## Installation Methods

### Method 1: Using PowerShell (Recommended)

1. Open PowerShell as Administrator
2. Navigate to the elegant-chatbot directory:
   ```powershell
   cd C:\Users\filiu\projects\startingOver\elegant-chatbot
   ```

3. Allow script execution (if needed):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. Run the installation script:
   ```powershell
   .\install.ps1
   ```

### Method 2: Using Command Prompt

1. Open Command Prompt as Administrator
2. Navigate to the elegant-chatbot directory:
   ```cmd
   cd C:\Users\filiu\projects\startingOver\elegant-chatbot
   ```

3. Run the batch file:
   ```cmd
   install.bat
   ```

### Method 3: Manual Installation

1. Open Command Prompt or PowerShell
2. Navigate to the elegant-chatbot directory
3. Run these commands:

```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install PyAudio for Windows
pip install pipwin
pipwin install pyaudio

# Create directories
mkdir data logs memories voices
```

## Setting the API Key

### PowerShell:
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### Command Prompt:
```cmd
set OPENAI_API_KEY=your-api-key-here
```

### Permanent (System Environment Variable):
1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Add new User variable:
   - Name: `OPENAI_API_KEY`
   - Value: `your-api-key-here`

## Running the Chatbot

### PowerShell:
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the chatbot
python main.py
```

### Command Prompt:
```cmd
# Activate virtual environment
venv\Scripts\activate

# Run the chatbot
python main.py
```

## Common Windows Issues & Solutions

### Issue: "pipwin install pyaudio" fails
**Solution**: Install PyAudio manually:
1. Download the appropriate `.whl` file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
2. Install with: `pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl`

### Issue: "git is not recognized"
**Solution**: Install Git for Windows or download Chatterbox manually

### Issue: Microphone not working
**Solution**: 
1. Check Windows Privacy Settings → Microphone
2. Allow Python to access microphone
3. Run the chatbot as Administrator

### Issue: PowerShell script execution disabled
**Solution**: Run this command:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Testing Installation

Run the test script:
```cmd
python test_basic.py
```

You should see:
```
🎨 Elegant Chatbot - Basic Tests
==================================
🔧 Testing Configuration...
   ✅ Configuration valid
🎤 Testing VAD...
   ✅ VAD working
...
```

## Quick Start Commands

After installation, you can create a batch file for easy startup:

Create `run_chatbot.bat`:
```batch
@echo off
call venv\Scripts\activate
set OPENAI_API_KEY=your-key-here
python main.py
```

Then just double-click `run_chatbot.bat` to start!

---

## Troubleshooting

If you encounter any issues:

1. Make sure Python is in your PATH:
   ```cmd
   python --version
   ```

2. Check that the virtual environment is activated:
   - You should see `(venv)` in your prompt

3. Verify API key is set:
   ```cmd
   echo %OPENAI_API_KEY%
   ```

4. Check audio devices:
   ```python
   python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_device_count(), 'audio devices found')"
   ```

Happy chatting! 🎤