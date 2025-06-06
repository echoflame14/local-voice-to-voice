# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Voice-to-Voice Chatbot that enables real-time voice conversations with AI models. It uses:
- **Speech-to-Text**: OpenAI Whisper 
- **LLM**: Google Gemini 1.5 Flash or OpenAI-compatible API (via LM Studio)
- **Text-to-Speech**: Chatterbox TTS (by Resemble AI) with voice cloning and adaptive streaming
- **Audio Pipeline**: VAD (Voice Activity Detection) or Push-to-Talk input
- **Performance**: Adaptive streaming synthesis, performance monitoring, and optimizations

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

# With performance monitoring
python main.py --enable-performance-monitoring
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
├── llm/              # LLM clients (Gemini 2.0 & OpenAI-compatible)
├── tts/              # Text-to-Speech (Chatterbox with adaptive streaming)
├── audio/            # Audio I/O, VAD, sound effects
├── pipeline/         # Main VoiceAssistant orchestration
└── utils/            # Performance monitoring, logging, optimization
```

Key architectural patterns:
- **VoiceAssistantStreaming** (src/pipeline/voice_assistant_streaming.py): Enhanced streaming orchestrator with adaptive synthesis
- **StreamManager** (src/audio/stream_manager.py): Handles real-time audio I/O and interruption logic
- **InputManager** (src/audio/input_manager.py): Manages VAD/PTT input modes with confidence scoring
- **ChatterboxWrapper** (src/tts/chatterbox_wrapper.py): Wraps Chatterbox TTS with voice cloning support
- **AdaptiveStreamingSynthesis** (src/pipeline/adaptive_streaming.py): Intelligent chunk size adaptation for optimal latency
- **PerformanceMonitor** (src/utils/performance_monitor.py): Real-time performance tracking and metrics

## Configuration

The system uses environment variables (see env_template.txt) with validation in configs/config.py:
- GEMINI_API_KEY: Google Gemini API key (required)
- GEMINI_MODEL: Default is gemini-1.5-flash-latest
- LM_STUDIO_BASE_URL: Local LLM server (optional, default: http://localhost:1234/v1)
- WHISPER_MODEL_SIZE: tiny/base/small/medium/large
- INPUT_MODE: vad (hands-free) or push_to_talk
- VOICE_REFERENCE_PATH: Path to voice sample for cloning
- STREAMING_CHUNK_SIZE_WORDS: Words per streaming chunk (default: 3)
- ENABLE_HIGH_PERFORMANCE: Enable performance optimizations (default: True)
- MAX_HISTORY_MESSAGES: Max conversation history (default: 2000)

## Current Development Focus

Working on branch `CursorChanges`:
- Enhanced streaming synthesis with adaptive chunking
- Improved interrupt handling with confidence scoring
- Performance monitoring and optimization
- Voice sample conversion utilities
- Comprehensive test coverage

See:
- ENHANCEMENT_PLAN.md for the detailed roadmap
- TTS_IMPROVEMENTS.md for TTS-specific enhancements
- VAD_INTERRUPT_FIX_PLAN.md for VAD interrupt improvements

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
   - Google Gemini 1.5 Flash (primary)
   - OpenAI-compatible APIs (via LM Studio)

## Utility Scripts

### Voice Processing
- `convert_josh_voice.py`, `convert_josh_pydub.py`: Convert voice samples to compatible format
- `trim_voice_sample.py`, `trim_last_seconds.py`: Trim audio files for optimal voice cloning
- `pydub_convert_trim.py`: Combined conversion and trimming

### Testing & Analysis
- `test_streaming.py`, `test_adaptive_streaming.py`: Test streaming synthesis
- `test_chunking.py`, `test_3_sentence_chunking.py`: Test text chunking strategies
- `analyze_performance.py`: Analyze system performance metrics
- `test_optimizations.py`: Test performance optimizations

### Setup & Configuration
- `setup_gemini.py`, `upgrade_gemini.py`: Configure Gemini API
- `fix_audio_device.py`: Troubleshoot audio device issues
- `fix_cutoff.py`: Fix audio cutoff problems

## Testing Strategy

The project includes comprehensive tests:
- Unit tests for individual components
- Integration tests for the full pipeline
- Performance benchmarks
- Streaming synthesis tests
- Interrupt handling tests

Always run tests before committing changes:
```bash
pytest tests/ -v
```

## Lint and Type Checking

Before committing code, ensure it passes quality checks:
```bash
# Run linting (if configured)
# python -m pylint src/

# Run type checking (if configured)
# python -m mypy src/
```

Note: Check if linting/type checking commands are configured for this project.