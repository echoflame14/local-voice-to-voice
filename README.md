# 🎤 Voice-to-Voice Chatbot

A fully offline, low-latency voice-to-voice chatbot built with state-of-the-art AI models:

- **Text-to-Speech**: [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI
- **Speech-to-Text**: OpenAI Whisper
- **Language Model**: Any OpenAI-compatible API (via LM Studio)

## ✨ Features

- **🎯 Fully Offline**: No internet required once models are downloaded
- **⚡ Low Latency**: Optimized for real-time conversation
- **🎭 Voice Cloning**: Use any reference voice with Chatterbox TTS
- **🎛️ Emotion Control**: Unique exaggeration control for expressive speech
- **🔌 Flexible LLM**: Works with any OpenAI-compatible API
- **🎤 Real-time VAD**: Voice Activity Detection for natural conversations
- **💬 Conversation Memory**: Maintains chat history during sessions
- **⚙️ Highly Configurable**: Customize all aspects via environment variables

## 🏗️ Architecture

```
[Microphone] → [VAD] → [Whisper STT] → [LLM] → [Chatterbox TTS] → [Speakers]
     ↑                                                                  ↓
     └─────────────── Real-time Audio Pipeline ──────────────────────────┘
```

## 📋 Prerequisites

### 1. LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a model (recommended: Llama 3.2 3B, Phi-3 Mini, or Mistral 7B)
3. Start the local server:
   - Open LM Studio
   - Go to "Local Server" tab
   - Load your model
   - Start server on `http://localhost:1234`

### 2. System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible GPU recommended (can run on CPU)
- **Memory**: 8GB+ RAM recommended
- **Audio**: Microphone and speakers/headphones

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <this-repo>
cd voice-chatbot
```

### 2. Install Chatterbox TTS

```bash
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
cd ..
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Additional Audio Dependencies

**Windows**:
```bash
# PyAudio wheel for Windows
pip install pipwin
pipwin install pyaudio
```

**macOS**:
```bash
brew install portaudio
pip install pyaudio
```

**Linux**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=not-needed

# Model Configuration
WHISPER_MODEL_SIZE=base
TTS_DEVICE=cuda  # cuda, mps, or cpu

# Voice Configuration
VOICE_REFERENCE_PATH=./voices/default.wav
VOICE_EXAGGERATION=0.5
VOICE_CFG_WEIGHT=0.5

# Audio Settings
SAMPLE_RATE=16000
CHUNK_SIZE=1024
VAD_AGGRESSIVENESS=1  # 0-3, higher is more aggressive

# Chat Settings
SYSTEM_PROMPT=You are a helpful voice assistant. Keep your responses concise and natural for speech.
MAX_RESPONSE_TOKENS=150
TEMPERATURE=0.7
```

## 🎯 Usage

### Quick Start

1. **Start LM Studio** with a loaded model
2. **Run the voice chatbot**:
   ```bash
   python main.py
   ```
3. **Start talking!** The assistant will respond automatically

### Command Line Options

```bash
# Basic usage
python main.py

# Use different Whisper model
python main.py --model large

# Use custom voice reference
python main.py --voice path/to/voice.wav

# Run in text mode for testing
python main.py --text-mode

# Use specific device
python main.py --device cuda

# Custom LLM URL
python main.py --llm-url http://localhost:8080/v1

# Custom system prompt
python main.py --system-prompt "You are a friendly AI assistant"
```

### Available Models

**Whisper Models** (size vs accuracy vs speed):
- `tiny`: Fastest, least accurate
- `base`: Good balance (recommended)
- `small`: Better accuracy, slower
- `medium`: High accuracy, much slower
- `large`: Best accuracy, slowest

**Recommended LLM Models for LM Studio**:
- **Fast**: Llama 3.2 1B/3B, Phi-3 Mini
- **Balanced**: Mistral 7B, Llama 3.1 8B
- **High Quality**: Llama 3.1 70B (requires powerful GPU)

## 🎭 Voice Customization

### Using Your Own Voice

1. Record a clear audio sample (10-30 seconds)
2. Save as WAV file in the `voices/` directory
3. Run with custom voice:
   ```bash
   python main.py --voice voices/my_voice.wav
   ```

### Voice Parameters

- **Exaggeration** (0.25-2.0): Controls emotion intensity
- **CFG Weight** (0.0-1.0): Controls pace and stability
- **Temperature** (0.1-1.0): Controls synthesis randomness

### Tips for Best Voice Quality

- Use clear, well-recorded audio
- 16kHz+ sample rate recommended
- Avoid background noise
- 10-30 seconds of speech ideal
- Multiple sentences work better than single words

## 🔧 Troubleshooting

### Common Issues

**1. "Could not connect to LLM server"**
- Ensure LM Studio is running on `http://localhost:1234`
- Check that a model is loaded in LM Studio
- Verify the URL with `--llm-url` parameter

**2. "No audio devices found"**
- Check microphone permissions
- Ensure audio drivers are installed
- Try different audio devices with device selection

**3. "CUDA out of memory"**
- Use smaller Whisper model: `--model tiny` or `--model base`
- Use CPU: `--device cpu`
- Reduce TTS batch size in configuration

**4. "PyAudio errors"**
- Install platform-specific audio dependencies (see Installation)
- Check microphone permissions
- Try different audio devices

**5. Poor voice quality**
- Use better reference voice (clear, longer recording)
- Adjust voice parameters in config
- Ensure reference voice matches target style

### Performance Optimization

**For Low Latency**:
- Use `tiny` or `base` Whisper models
- Reduce `chunk_size` to 512
- Use faster LLM (3B parameters or less)
- Lower `max_response_tokens`

**For High Quality**:
- Use `large` Whisper model
- Use larger LLM (7B+ parameters)
- Increase `max_response_tokens`
- Fine-tune voice parameters

## 📁 Project Structure

```
voice-chatbot/
├── src/
│   ├── stt/                 # Speech-to-Text (Whisper)
│   ├── llm/                 # Language Model (OpenAI-compatible)
│   ├── tts/                 # Text-to-Speech (Chatterbox)
│   ├── audio/               # Audio I/O and VAD
│   └── pipeline/            # Voice Assistant orchestration
├── configs/                 # Configuration management
├── voices/                  # Voice reference files
├── models/                  # Model cache directory
├── chatterbox/             # Chatterbox TTS submodule
├── requirements.txt        # Python dependencies
├── main.py                 # Main application
└── README.md              # This file
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to:
- Follow the existing code style
- Add tests if applicable
- Update documentation as needed
- Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification

## 🔬 Development Setup

1. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/yourusername/chatterbox.git
   cd chatterbox
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install development extras
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

For coverage report:
```bash
pytest --cov=src tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Resemble AI](https://github.com/resemble-ai) for the Chatterbox TTS system
- [OpenAI](https://github.com/openai/whisper) for the Whisper STT model
- [LM Studio](https://lmstudio.ai/) for the local LLM server
- All contributors and maintainers

## 📚 Additional Resources

- [Chatterbox TTS Documentation](https://github.com/resemble-ai/chatterbox)
- [Whisper Documentation](https://github.com/openai/whisper)
- [LM Studio Guide](https://lmstudio.ai/docs)

## ⭐ Support the Project

If you find this project useful, please consider:
- Giving it a star on GitHub
- Sharing it with others
- Contributing to its development
- Reporting issues or suggesting improvements

---
Built with ❤️ by the Chatterbox community

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Join community discussions

---

**Happy chatting! 🎤🤖** 