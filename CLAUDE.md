# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Voice-to-Voice Chatbot that enables real-time voice conversations with AI models. It uses:
- **Speech-to-Text**: OpenAI Whisper 
- **LLM**: OpenAI-compatible API (via LM Studio) or Google Gemini
- **Text-to-Speech**: Chatterbox TTS (by Resemble AI) with voice cloning
- **Audio Pipeline**: VAD (Voice Activity Detection) or Push-to-Talk input

## Key Commands

### Running the Application
```bash
# Default run (VAD mode)
python main.py

# Push-to-talk mode  
python main.py --input-mode push_to_talk

# Text mode for testing
python main.py --text-mode

# With custom voice
python main.py --voice voices/my_voice.wav

# High quality setup
python main.py --model large --device cuda
```

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Installation
```bash
# Windows
install.bat  # or install.ps1

# Manual
pip install -r requirements.txt
```

## Architecture

The codebase follows a modular pipeline architecture:

```
src/
├── stt/              # Speech-to-Text (Whisper)
├── llm/              # LLM clients (OpenAI-compatible & Gemini)
├── tts/              # Text-to-Speech (Chatterbox wrapper)
├── audio/            # Audio I/O, VAD, sound effects
└── pipeline/         # Main VoiceAssistant orchestration
```

Key architectural patterns:
- **VoiceAssistant** (src/pipeline/voice_assistant.py:25-700): Central orchestrator managing the full pipeline
- **StreamManager** (src/audio/stream_manager.py): Handles real-time audio I/O and interruption logic
- **InputManager** (src/audio/input_manager.py): Manages VAD/PTT input modes
- **ChatterboxWrapper** (src/tts/chatterbox_wrapper.py): Wraps Chatterbox TTS with voice cloning support

## Configuration

The system uses environment variables (see env_template.txt) with validation in configs/config.py:
- LM_STUDIO_BASE_URL: Local LLM server (default: http://localhost:1234/v1)
- WHISPER_MODEL_SIZE: tiny/base/small/medium/large
- INPUT_MODE: vad (hands-free) or push_to_talk
- VOICE_REFERENCE_PATH: Path to voice sample for cloning

## Current Development Focus

Working on branch `reEnableAudioCuesAndInterrupts`:
- Implementing audio cues for better user feedback
- Enhancing interruption handling with grace periods
- Improving VAD sensitivity and reliability

See ENHANCEMENT_PLAN.md for the detailed roadmap of audio improvements.

## Important Considerations

1. **Audio Processing**: The system uses real-time audio streaming with PyAudio. Always test audio changes with actual microphone input.

2. **Interruption Logic**: Complex state management in StreamManager handles:
   - Grace periods (configurable delay before allowing interrupts)
   - Audio cue timing
   - Thread-safe state transitions

3. **Voice Cloning**: Chatterbox TTS requires:
   - Clear voice samples (10-30 seconds)
   - 16kHz+ sample rate
   - Minimal background noise

4. **Performance**: For low latency:
   - Whisper runs on CPU (faster for real-time)
   - TTS can use GPU (cuda/mps) 
   - Chunk size affects responsiveness (default: 480)

5. **LLM Integration**: Supports both:
   - OpenAI-compatible APIs (via LM Studio)
   - Google Gemini (with API key)