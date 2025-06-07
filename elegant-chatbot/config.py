"""
Elegant Chatbot Configuration System
Single source of truth for all settings
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os
import json
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    LOCAL = "local"


@dataclass
class AudioConfig:
    """Audio input/output configuration"""
    sample_rate: int = 16000
    chunk_size: int = 480  # 30ms at 16kHz
    channels: int = 1
    format: str = "int16"
    
    # VAD settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_aggressiveness: int = 2
    
    # Buffer settings  
    pre_buffer_ms: int = 300
    post_buffer_ms: int = 500


@dataclass
class ModelConfig:
    """AI model configuration"""
    # STT settings
    whisper_model: str = "base"
    whisper_device: str = "cpu"  # Faster for real-time
    
    # LLM settings
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4.1-nano"  # Exact model as requested
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150
    
    # TTS settings
    tts_voice: Optional[str] = None
    tts_speed: float = 1.0


@dataclass
class FeatureConfig:
    """Optional feature flags"""
    enable_interrupts: bool = True
    enable_memory: bool = False
    enable_effects: bool = True
    enable_analytics: bool = False
    
    # Memory settings
    memory_max_exchanges: int = 1000
    memory_context_size: int = 10
    
    # Effect settings
    effect_volume: float = 0.3


@dataclass
class PathConfig:
    """File and directory paths"""
    base_dir: Path = field(default_factory=lambda: Path.cwd() / "elegant-chatbot")
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "elegant-chatbot" / "data")
    log_dir: Path = field(default_factory=lambda: Path.cwd() / "elegant-chatbot" / "logs")
    memory_dir: Path = field(default_factory=lambda: Path.cwd() / "elegant-chatbot" / "memories")
    voice_dir: Path = field(default_factory=lambda: Path.cwd() / "elegant-chatbot" / "voices")
    
    def create_dirs(self):
        """Create all directories if they don't exist"""
        for path in [self.data_dir, self.log_dir, self.memory_dir, self.voice_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Config:
    """Master configuration class"""
    
    def __init__(self):
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.features = FeatureConfig()
        self.paths = PathConfig()
        
        # API keys
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }
        
        # Create directories
        self.paths.create_dirs()
        
        # Load overrides
        self._load_env_overrides()
        
    def _load_env_overrides(self):
        """Override settings from environment variables"""
        # Pattern: CHATBOT_SECTION_SETTING
        for key, value in os.environ.items():
            if key.startswith("CHATBOT_"):
                parts = key.split("_", 2)
                if len(parts) == 3:
                    _, section, setting = parts
                    section = section.lower()
                    setting = setting.lower()
                    
                    if hasattr(self, section):
                        config_section = getattr(self, section)
                        if hasattr(config_section, setting):
                            current_value = getattr(config_section, setting)
                            converted_value = self._convert_value(value, type(current_value))
                            setattr(config_section, setting, converted_value)
                            
    def _convert_value(self, value: str, target_type: type) -> Any:
        """Convert string to target type"""
        if target_type == bool:
            return value.lower() in ["true", "1", "yes"]
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        return value
        
    def validate(self) -> list:
        """Validate configuration"""
        errors = []
        
        # Check API key
        if self.model.llm_provider == LLMProvider.OPENAI and not self.api_keys["openai"]:
            errors.append("OpenAI API key not set (OPENAI_API_KEY)")
            
        return errors


# Global config instance
config = Config()