# Configuration Guide

This guide explains all the configuration options available in the Voice Assistant. Most settings are in `configs/config.py` for easy editing, while sensitive information like API keys are in environment variables.

## Quick Setup

1. Copy the environment template and set your API key:
   ```bash
   cp env_template.txt .env
   ```

2. Edit `.env` and set your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. Optionally adjust other settings in `configs/config.py` as needed.

## Configuration Sections

### LLM Provider Settings (configs/config.py)

The system can use either Google Gemini or LM Studio as the language model provider.

#### Using Gemini (Recommended)
- Set `USE_GEMINI = True` in `configs/config.py`
- Get an API key from [Google AI Studio](https://ai.google.dev/)
- Set `GEMINI_API_KEY=your_key_here` in `.env`
- Choose model with `GEMINI_MODEL` in `config.py` (default: `models/gemini-1.5-flash`)

#### Using LM Studio (Local Alternative)
- Set `USE_GEMINI = False` in `configs/config.py`
- Ensure LM Studio is running on `http://localhost:1234`
- Load any compatible model in LM Studio

### Speech-to-Text Settings (configs/config.py)

- `WHISPER_MODEL_SIZE`: Model accuracy vs speed trade-off
  - `"tiny"`: Fastest, least accurate
  - `"base"`: Good balance (recommended)
  - `"small"`: Better accuracy, slower
  - `"medium"`: High accuracy, much slower
  - `"large"`: Best accuracy, very slow

- `WHISPER_DEVICE`: Processing device
  - `"cpu"`: Works everywhere (recommended)
  - `"cuda"`: Faster on NVIDIA GPUs
  - `"mps"`: Faster on Apple Silicon Macs

### Text-to-Speech Settings (configs/config.py)

- `VOICE_REFERENCE_PATH`: Path to voice sample file
- `VOICE_EXAGGERATION`: Emotion intensity (0.0-1.0)
- `VOICE_CFG_WEIGHT`: Voice guidance strength (0.0-1.0) 
- `VOICE_TEMPERATURE`: Voice randomness (0.0-1.0)

Lower values = faster generation, higher values = better quality.

### Input Mode Settings (configs/config.py)

#### Voice Activity Detection (VAD) - Hands-Free
- `INPUT_MODE = "vad"`
- `VAD_AGGRESSIVENESS`: Sensitivity (0-3, higher = more sensitive)
- `VAD_SPEECH_THRESHOLD`: When to start listening (0.0-1.0)
- `VAD_SILENCE_THRESHOLD`: When to stop listening (0.0-1.0)

#### Push-to-Talk - Manual Control
- `INPUT_MODE = "push_to_talk"`
- `PUSH_TO_TALK_KEY`: Key to hold while speaking

### Audio Settings (configs/config.py)

- `SAMPLE_RATE`: Audio quality (16000 recommended)
- `CHUNK_SIZE`: Processing buffer size (480 for real-time)
- `MIN_AUDIO_AMPLITUDE`: Noise gate threshold

### Sound Effects (configs/config.py)

- `ENABLE_SOUND_EFFECTS`: Master toggle for all sounds
- `ENABLE_INTERRUPTION_SOUND`: Play sound when interrupting
- `ENABLE_GENERATION_SOUND`: Play sound when starting to respond
- `SOUND_EFFECT_VOLUME`: Volume level (0.0-1.0)

### Conversation & Memory (configs/config.py)

- `LOG_CONVERSATIONS`: Save chat history to files
- `AUTO_SUMMARIZE_CONVERSATIONS`: Automatically create summaries
- `MAX_HISTORY_MESSAGES`: How much chat history to keep in memory
- `MAX_SUMMARIES_TO_LOAD`: How many conversation summaries to load

### Performance Settings (configs/config.py)

- `MAX_RESPONSE_TOKENS`: Maximum length of AI responses
- `LLM_TEMPERATURE`: AI creativity (0.0-2.0, higher = more creative)
- `SYNTHESIS_GRACE_PERIOD`: Delay before allowing interruption (seconds)
- `PROCESSING_TIMEOUT`: Maximum time to wait for AI response (seconds)

## Troubleshooting

### Common Issues

1. **"No Gemini API Key"**: Set `GEMINI_API_KEY` in your `.env` file
2. **"LM Studio not connecting"**: Ensure LM Studio is running and model is loaded
3. **Poor voice quality**: Increase `VOICE_EXAGGERATION`, `VOICE_CFG_WEIGHT`, and `VOICE_TEMPERATURE`
4. **VAD too sensitive**: Lower `VAD_AGGRESSIVENESS` and `VAD_SPEECH_THRESHOLD`
5. **VAD not detecting speech**: Increase `VAD_AGGRESSIVENESS` and lower `VAD_SILENCE_THRESHOLD`

### Performance Optimization

For better performance:
- Use `WHISPER_MODEL_SIZE=base` or `tiny`
- Set `WHISPER_DEVICE=cpu` for consistency
- Lower TTS quality settings for faster response
- Reduce `MAX_RESPONSE_TOKENS` for shorter responses

For better quality:
- Use `WHISPER_MODEL_SIZE=small` or `medium`
- Increase TTS quality settings
- Use `WHISPER_DEVICE=cuda` if you have a good GPU 