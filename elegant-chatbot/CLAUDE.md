# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The Elegant Chatbot is a simple, clean voice-to-voice chatbot that emphasizes radical simplicity, DRY principles, and progressive enhancement. It uses:
- **Speech-to-Text**: OpenAI Whisper
- **LLM**: OpenAI GPT-4.1-nano (primary), with support for Gemini and local models
- **Text-to-Speech**: pyttsx3 (fallback) or Chatterbox TTS (recommended)
- **Audio Pipeline**: PyAudio with VAD (Voice Activity Detection)

The core system is <500 lines of code with a plugin architecture for optional features.

## Key Commands

### Running the Application
```bash
# Basic usage
python main.py

# With memory enabled
python main.py --enable-memory

# Without interrupts
python main.py --no-interrupts

# With custom voice (Chatterbox TTS)
python main.py --voice voices/my_voice.wav
```

### Testing
```bash
# Run basic functionality tests
python test_basic.py

# Run full test suite (if available)
pytest tests/

# Run with coverage
pytest --cov=core tests/
```

### Installation
```bash
# Windows
install.bat  # or install.ps1

# Linux/Mac
./install.sh

# Manual
pip install -r requirements.txt
```

## Architecture

The codebase follows a modular architecture with clear separation:

```
elegant-chatbot/
├── config.py              # Single source of truth for ALL configuration
├── main.py               # Clean entry point with argument parsing
├── core/                 # Essential components only
│   ├── audio.py         # Unified audio I/O with VAD
│   ├── stt.py          # Speech-to-text (Whisper)
│   ├── llm.py          # LLM client (OpenAI/Gemini/Local)
│   ├── tts.py          # Text-to-speech
│   └── voice_loop.py   # Main VoiceAssistant class
├── features/            # Optional plugins
│   └── memory.py       # Conversation memory
└── utils/              # Utilities
```

Key architectural principles:
- **Radical Simplicity**: One clear path for each operation
- **DRY Configuration**: All settings in config.py with environment overrides
- **Progressive Enhancement**: Core system works alone, features are plugins
- **Direct Function Calls**: No callbacks, no event systems, just simple function calls

## Configuration

The system uses a sophisticated DRY configuration in `config.py`:
- All settings defined as dataclasses
- Environment variable overrides: `CHATBOT_SECTION_SETTING`
- Type validation and defaults
- Sections: AudioConfig, ModelConfig, FeatureConfig, PathConfig, RuntimeConfig

Example overrides:
```bash
export CHATBOT_AUDIO_SAMPLE_RATE=32000
export CHATBOT_MODEL_LLM_PROVIDER=gemini
export CHATBOT_FEATURES_ENABLE_MEMORY=true
```

## Current Development Focus

The project follows a phased approach:
- **Phase 1** ✓: Core voice loop (LISTENING → PROCESSING → SPEAKING)
- **Phase 2** (Current): Smart features (memory, interrupts, effects)
- **Phase 3**: Polish & performance optimization
- **Phase 4**: Advanced features based on feedback

## Important Considerations

1. **Simplicity First**: If something can be simpler, make it simpler. Each feature should be <100 lines.

2. **Audio Processing**: The system uses real-time audio with PyAudio. Test with actual microphone input.

3. **State Management**: Simple state machine in VoiceAssistant:
   - LISTENING: VAD monitors for speech
   - PROCESSING: STT → LLM → TTS pipeline
   - SPEAKING: Audio playback (can be interrupted)

4. **Performance Targets**:
   - Response time: <1 second
   - Memory usage: <200MB baseline
   - CPU usage: <10% idle
   - Startup time: <2 seconds

5. **Error Handling**: Graceful degradation - if TTS fails, print to console; if audio fails, provide clear troubleshooting.

## Development Guidelines

When adding features:
1. Check if it belongs in `core/` (essential) or `features/` (optional)
2. Follow existing patterns - look at how memory.py integrates
3. Add configuration to config.py if needed
4. Keep it simple - no unnecessary abstractions
5. Test with `test_basic.py` first