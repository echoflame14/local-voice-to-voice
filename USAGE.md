# 🎤 Voice Assistant Usage Guide

## 🚀 Quick Start

### 1. Default VAD Mode (Hands-Free)
```bash
python main.py
```
Just start talking! The assistant will automatically detect when you're speaking.

### 2. Push-to-Talk Mode
```bash
python main.py --input-mode push_to_talk
```
Press and hold the spacebar to talk, release to process.

## 🎛️ Configuration Options

### Input Mode Settings

| Command | Description | Default |
|---------|-------------|---------|
| `--input-mode vad` | Voice Activity Detection (hands-free) | ✅ Default |
| `--input-mode push_to_talk` | Manual push-to-talk control | |
| `--vad-aggressiveness 0-3` | VAD sensitivity (0=least, 3=most) | 1 |
| `--ptt-key space` | Key for push-to-talk | space |

### Voice & Model Settings

| Command | Description | Default |
|---------|-------------|---------|
| `--model base` | Whisper model (tiny/base/small/medium/large) | base |
| `--voice path.wav` | Custom voice reference file | default.wav |
| `--device cuda` | TTS device (cpu/cuda/mps) | auto |

### Other Options

| Command | Description |
|---------|-------------|
| `--text-mode` | Text-only mode for testing |
| `--llm-url http://...` | Custom LLM server URL |
| `--system-prompt "..."` | Custom system prompt |

## 📖 Examples

### Basic Usage
```bash
# Default hands-free mode
python main.py

# More sensitive voice detection
python main.py --vad-aggressiveness 2

# Push-to-talk with Enter key
python main.py --input-mode push_to_talk --ptt-key enter

# High-quality setup
python main.py --model large --device cuda --voice my_voice.wav
```

### Testing & Development
```bash
# Text mode for testing
python main.py --text-mode

# Debug with basic settings
python main.py --model tiny --device cpu
```

## 🎯 Input Mode Comparison

### 🎤 VAD (Voice Activity Detection) - Default
**Best for:** Hands-free operation, natural conversations

✅ **Pros:**
- Natural, hands-free interaction
- No need to remember keys
- Great for long conversations
- Automatic speech detection

⚠️ **Considerations:**
- May pick up background noise
- Requires quiet environment for best results
- Uses some CPU for voice detection

**Settings:**
- `--vad-aggressiveness 0-3`: Higher = more sensitive
- Works best in quiet environments
- Automatically filters short sounds

### 🎮 Push-to-Talk (PTT)
**Best for:** Noisy environments, precise control

✅ **Pros:**
- Perfect for noisy environments
- Precise control over when to speak
- No false triggers
- Privacy control

⚠️ **Considerations:**
- Must remember to press/hold key
- Less natural interaction
- Requires hands near keyboard

**Settings:**
- `--ptt-key space`: Any key (space, enter, tab, etc.)
- Hold to talk, release to process
- Works in any environment

## 🔧 Environment Configuration

Create a `.env` file (copy from `env_template.txt`):

```bash
# Basic settings
INPUT_MODE=vad
VAD_AGGRESSIVENESS=1
PUSH_TO_TALK_KEY=space

# LLM settings  
LM_STUDIO_BASE_URL=http://localhost:1234/v1
MAX_RESPONSE_TOKENS=300

# Audio settings
SAMPLE_RATE=16000
CHUNK_SIZE=480
```

## 🎛️ VAD Fine-Tuning

### Aggressiveness Levels
- **0**: Least aggressive (miss some speech, fewer false positives)
- **1**: Balanced (recommended for most use cases) ✅
- **2**: More aggressive (catches quiet speech, more sensitive)
- **3**: Most aggressive (catches all speech, may have false positives)

### When to Adjust
- **Too many false triggers**: Lower aggressiveness (0-1)
- **Missing quiet speech**: Raise aggressiveness (2-3)
- **Noisy environment**: Use push-to-talk instead

## 🎯 Recommended Setups

### 🏠 Home/Office (Quiet)
```bash
python main.py --vad-aggressiveness 1
```

### 🏢 Office (Some Noise)
```bash
python main.py --vad-aggressiveness 2
```

### 🎵 Noisy Environment
```bash
python main.py --input-mode push_to_talk
```

### 🎮 Gaming Setup
```bash
python main.py --input-mode push_to_talk --ptt-key ctrl
```

### 💻 Development/Testing
```bash
python main.py --text-mode --model tiny
```

## 🆘 Troubleshooting

### VAD Issues
- **"Not detecting my voice"**: Increase `--vad-aggressiveness`
- **"Too many false triggers"**: Decrease `--vad-aggressiveness`
- **"Choppy detection"**: Check microphone levels, reduce background noise

### PTT Issues
- **"Key not working"**: Try `--ptt-key enter` or `--ptt-key tab`
- **"Need different key"**: Use any key name (space, enter, ctrl, alt, etc.)

### General Issues
- **"No audio"**: Check microphone permissions
- **"Slow responses"**: Use `--model tiny` or `--device cpu`
- **"LLM not connecting"**: Verify LM Studio is running on localhost:1234

## 🎯 Best Practices

1. **Start with defaults**: `python main.py` works great for most setups
2. **Quiet environment**: Use VAD mode for natural interaction
3. **Noisy environment**: Switch to push-to-talk
4. **Testing changes**: Use `--text-mode` to test without voice
5. **Performance issues**: Use smaller models (`--model tiny`)

---

**Happy chatting! 🎤🤖**

For more configuration options, see `env_template.txt` or run `python main.py --help`. 