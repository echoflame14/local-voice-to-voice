"""Configuration settings for the voice assistant"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
VOICES_DIR = ROOT_DIR / "voices"

# Model settings
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny, base, small, medium, large
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "not-needed")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "models/gemini-1.5-flash"

# System prompt
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a helpful voice assistant. Keep your responses concise and natural for speech. Respond in a conversational tone and avoid overly long explanations unless specifically asked.""")

# TTS settings
TTS_DEVICE = os.getenv("TTS_DEVICE", None)  # Auto-detect (cpu, cuda, mps)
VOICE_REFERENCE_PATH = VOICES_DIR / "default.wav"  # Default voice
VOICE_EXAGGERATION = float(os.getenv("VOICE_EXAGGERATION", 0.5))  # Emotion exaggeration (0-1)
VOICE_CFG_WEIGHT = float(os.getenv("VOICE_CFG_WEIGHT", 0.5))  # Classifier-free guidance (0-1)
VOICE_TEMPERATURE = float(os.getenv("VOICE_TEMPERATURE", 0.8))  # Sampling temperature

# Audio settings
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 480))  # 30ms at 16kHz - matches VAD frame size
MIN_AUDIO_AMPLITUDE = float(os.getenv("MIN_AUDIO_AMPLITUDE", 0.015))  # Minimum amplitude threshold

# Input mode settings
INPUT_MODE = os.getenv("INPUT_MODE", "vad")  # "vad" or "push_to_talk"
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 2))  # 0-3: 0=least aggressive, 3=most aggressive
VAD_SPEECH_THRESHOLD = float(os.getenv("VAD_SPEECH_THRESHOLD", 0.6))  # Ratio of speech frames to trigger speech start
VAD_SILENCE_THRESHOLD = float(os.getenv("VAD_SILENCE_THRESHOLD", 0.4))  # Ratio of silence frames to trigger speech end
VAD_FRAME_DURATION_MS = int(os.getenv("VAD_FRAME_DURATION_MS", 30))  # Frame duration in milliseconds (10, 20, or 30)
PUSH_TO_TALK_KEY = os.getenv("PUSH_TO_TALK_KEY", "space")  # Keyboard key for push-to-talk

# Sound Effects
ENABLE_SOUND_EFFECTS = os.getenv("ENABLE_SOUND_EFFECTS", "True").lower() == "true"
SOUND_EFFECT_VOLUME = float(os.getenv("SOUND_EFFECT_VOLUME", 0.2))
ENABLE_INTERRUPTION_SOUND = os.getenv("ENABLE_INTERRUPTION_SOUND", "True").lower() == "true"
ENABLE_GENERATION_SOUND = os.getenv("ENABLE_GENERATION_SOUND", "True").lower() == "true"

# LLM settings
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", 300))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.85))

def validate():
    """Validate configuration settings"""
    # Check paths
    if not ROOT_DIR.exists():
        raise ValueError(f"Root directory not found: {ROOT_DIR}")
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True)
    if not VOICES_DIR.exists():
        VOICES_DIR.mkdir(parents=True)
    
    # Check model settings
    if WHISPER_MODEL_SIZE not in ["tiny", "base", "small", "medium", "large"]:
        raise ValueError(f"Invalid Whisper model size: {WHISPER_MODEL_SIZE}")
    
    # Check TTS settings
    if VOICE_REFERENCE_PATH and not Path(VOICE_REFERENCE_PATH).exists():
        print(f"Warning: Voice reference file not found: {VOICE_REFERENCE_PATH}")
    if not 0 <= VOICE_EXAGGERATION <= 1:
        raise ValueError(f"Voice exaggeration must be between 0 and 1")
    if not 0 <= VOICE_CFG_WEIGHT <= 1:
        raise ValueError(f"Voice CFG weight must be between 0 and 1")
    if not 0 <= VOICE_TEMPERATURE <= 1:
        raise ValueError(f"Voice temperature must be between 0 and 1")
    
    # Check audio settings
    if SAMPLE_RATE not in [8000, 16000, 32000, 48000]:
        raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000")
    if CHUNK_SIZE <= 0:
        raise ValueError(f"Chunk size must be positive")
    if MIN_AUDIO_AMPLITUDE < 0:
        raise ValueError(f"Minimum audio amplitude must be non-negative")
    
    # Check input mode settings
    if INPUT_MODE not in ["vad", "push_to_talk"]:
        raise ValueError(f"Input mode must be 'vad' or 'push_to_talk'")
    if not 0 <= VAD_AGGRESSIVENESS <= 3:
        raise ValueError(f"VAD aggressiveness must be between 0 and 3")
    if not 0 <= VAD_SPEECH_THRESHOLD <= 1:
        raise ValueError(f"VAD speech threshold must be between 0 and 1")
    if not 0 <= VAD_SILENCE_THRESHOLD <= 1:
        raise ValueError(f"VAD silence threshold must be between 0 and 1")
    
    # Check LLM settings
    if MAX_RESPONSE_TOKENS <= 0:
        raise ValueError(f"Max response tokens must be positive")
    if not 0 <= LLM_TEMPERATURE <= 2:
        raise ValueError(f"LLM temperature must be between 0 and 2")

# Validate configuration on import
validate() 