# DRY Configuration System Design 🔧

## Philosophy

"Configure once, use everywhere" - A single source of truth for all configuration with intelligent defaults, environment overrides, and runtime flexibility.

## Core Principles

1. **Single Configuration File**: All settings in one place
2. **Type Safety**: Use dataclasses and type hints
3. **Environment Override**: ENV vars override defaults
4. **Runtime Flexibility**: Change settings without code changes
5. **Validation**: Automatic validation of settings
6. **Documentation**: Self-documenting configuration

## Implementation

### Master Configuration Class

```python
# config.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import json
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    LOCAL = "local"

class AudioFormat(Enum):
    INT16 = "int16"
    FLOAT32 = "float32"

@dataclass
class AudioConfig:
    """Audio input/output configuration"""
    sample_rate: int = 16000
    chunk_size: int = 480  # 30ms at 16kHz
    channels: int = 1
    format: AudioFormat = AudioFormat.INT16
    device_index: Optional[int] = None
    
    # VAD settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_aggressiveness: int = 2
    
    # Buffer settings
    pre_buffer_ms: int = 300
    post_buffer_ms: int = 500
    
    def __post_init__(self):
        """Validate audio settings"""
        assert self.sample_rate in [8000, 16000, 32000, 48000], "Invalid sample rate"
        assert 0 <= self.vad_threshold <= 1, "VAD threshold must be between 0 and 1"
        assert 0 <= self.vad_aggressiveness <= 3, "VAD aggressiveness must be 0-3"

@dataclass
class ModelConfig:
    """AI model configuration"""
    # STT settings
    stt_provider: str = "whisper"
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_language: Optional[str] = None  # Auto-detect
    
    # LLM settings
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "GPT-4.1-nano"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150
    llm_timeout: int = 30
    
    # TTS settings
    tts_provider: str = "chatterbox"
    tts_voice: Optional[str] = None
    tts_speed: float = 1.0
    tts_pitch: float = 1.0
    
    def __post_init__(self):
        """Validate model settings"""
        assert 0 <= self.llm_temperature <= 2, "Temperature must be between 0 and 2"
        assert self.llm_max_tokens > 0, "Max tokens must be positive"
        assert 0.5 <= self.tts_speed <= 2.0, "TTS speed must be between 0.5 and 2.0"

@dataclass
class FeatureConfig:
    """Optional feature flags"""
    # Core features
    enable_interrupts: bool = True
    enable_memory: bool = False
    enable_effects: bool = True
    
    # Advanced features
    enable_analytics: bool = False
    enable_web_ui: bool = False
    enable_multi_language: bool = False
    enable_emotion_detection: bool = False
    
    # Memory settings (when enabled)
    memory_max_exchanges: int = 1000
    memory_context_size: int = 10
    memory_summarize_after: int = 100
    
    # Interrupt settings (when enabled)
    interrupt_grace_period_ms: int = 1000
    interrupt_cooldown_ms: int = 500
    
    # Effect settings (when enabled)
    effect_volume: float = 0.3
    effect_theme: str = "modern"

@dataclass
class PathConfig:
    """File and directory paths"""
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    log_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    memory_dir: Path = field(default_factory=lambda: Path.cwd() / "memories")
    voice_dir: Path = field(default_factory=lambda: Path.cwd() / "voices")
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / ".cache")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for path in [self.data_dir, self.log_dir, self.memory_dir, 
                     self.voice_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class RuntimeConfig:
    """Runtime behavior configuration"""
    debug: bool = False
    log_level: str = "INFO"
    performance_monitoring: bool = False
    auto_restart_on_error: bool = True
    max_retries: int = 3
    startup_sound: bool = True
    exit_sound: bool = True

class Config:
    """Master configuration class - single source of truth"""
    
    def __init__(self):
        # Initialize all sub-configs
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.features = FeatureConfig()
        self.paths = PathConfig()
        self.runtime = RuntimeConfig()
        
        # API keys from environment
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }
        
        # Load overrides
        self._load_env_overrides()
        self._load_file_overrides()
        
    def _load_env_overrides(self):
        """Override settings from environment variables"""
        # Pattern: CHATBOT_SECTION_SETTING
        # Example: CHATBOT_AUDIO_SAMPLE_RATE=32000
        
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
                            # Convert value to appropriate type
                            current_value = getattr(config_section, setting)
                            converted_value = self._convert_env_value(value, type(current_value))
                            setattr(config_section, setting, converted_value)
                            
    def _convert_env_value(self, value: str, target_type: type) -> Any:
        """Convert environment variable string to target type"""
        if target_type == bool:
            return value.lower() in ["true", "1", "yes", "on"]
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == Path:
            return Path(value)
        else:
            return value
            
    def _load_file_overrides(self, path: str = "config.json"):
        """Load configuration overrides from JSON file"""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                overrides = json.load(f)
                
            for section, settings in overrides.items():
                if hasattr(self, section):
                    config_section = getattr(self, section)
                    for key, value in settings.items():
                        if hasattr(config_section, key):
                            setattr(config_section, key, value)
                            
    def save(self, path: str = "config.json"):
        """Save current configuration to file"""
        config_dict = {
            "audio": asdict(self.audio),
            "model": asdict(self.model),
            "features": asdict(self.features),
            "paths": {k: str(v) for k, v in asdict(self.paths).items()},
            "runtime": asdict(self.runtime),
        }
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check API keys
        if self.model.llm_provider == LLMProvider.OPENAI and not self.api_keys["openai"]:
            errors.append("OpenAI API key not set (OPENAI_API_KEY)")
            
        # Check paths
        if self.features.enable_memory and not self.paths.memory_dir.exists():
            errors.append(f"Memory directory does not exist: {self.paths.memory_dir}")
            
        # Check model availability
        if self.model.stt_provider == "whisper":
            try:
                import whisper
            except ImportError:
                errors.append("Whisper not installed but selected as STT provider")
                
        return errors
        
    def __repr__(self) -> str:
        """Pretty print configuration"""
        return f"""
Elegant Chatbot Configuration:
==============================
Audio:
  Sample Rate: {self.audio.sample_rate}Hz
  Chunk Size: {self.audio.chunk_size}
  VAD: {'Enabled' if self.audio.vad_enabled else 'Disabled'}
  
Models:
  STT: {self.model.stt_provider} ({self.model.whisper_model})
  LLM: {self.model.llm_provider.value} ({self.model.llm_model})
  TTS: {self.model.tts_provider}
  
Features:
  Interrupts: {'✓' if self.features.enable_interrupts else '✗'}
  Memory: {'✓' if self.features.enable_memory else '✗'}
  Effects: {'✓' if self.features.enable_effects else '✗'}
  Analytics: {'✓' if self.features.enable_analytics else '✗'}
  
Runtime:
  Debug: {self.runtime.debug}
  Log Level: {self.runtime.log_level}
  Performance Monitoring: {'✓' if self.runtime.performance_monitoring else '✗'}
"""

# Global config instance
config = Config()
```

## Usage Patterns

### 1. Simple Access
```python
from config import config

# Direct access
sample_rate = config.audio.sample_rate
llm_model = config.model.llm_model

# Feature checking
if config.features.enable_memory:
    # Initialize memory system
    pass
```

### 2. Runtime Updates
```python
# Update configuration at runtime
config.audio.vad_threshold = 0.6
config.model.llm_temperature = 0.9

# Save updated config
config.save("my_config.json")
```

### 3. Environment Overrides
```bash
# Override via environment variables
export CHATBOT_AUDIO_SAMPLE_RATE=32000
export CHATBOT_MODEL_LLM_TEMPERATURE=0.5
export CHATBOT_FEATURES_ENABLE_MEMORY=true

python main.py
```

### 4. Configuration Profiles
```python
# config_profiles.py
class ConfigProfiles:
    @staticmethod
    def development():
        """Development configuration"""
        config.runtime.debug = True
        config.runtime.log_level = "DEBUG"
        config.features.enable_analytics = True
        
    @staticmethod
    def production():
        """Production configuration"""
        config.runtime.debug = False
        config.runtime.log_level = "WARNING"
        config.runtime.auto_restart_on_error = True
        
    @staticmethod
    def low_latency():
        """Optimized for minimal latency"""
        config.audio.chunk_size = 320  # 20ms
        config.audio.pre_buffer_ms = 100
        config.features.enable_effects = False
        config.model.whisper_model = "tiny"
```

### 5. Component Integration
```python
# core/audio.py
class AudioSystem:
    def __init__(self):
        # Use global config
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = config.audio.chunk_size
        self.vad_enabled = config.audio.vad_enabled
        
# core/llm.py
class LLMClient:
    def __init__(self):
        # Use global config
        self.provider = config.model.llm_provider
        self.model = config.model.llm_model
        self.api_key = config.api_keys[self.provider.value]
```

## Advanced Features

### 1. Configuration Validation
```python
# Validate on startup
errors = config.validate()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
```

### 2. Dynamic Reloading
```python
import watchdog

class ConfigWatcher:
    def __init__(self, config_path="config.json"):
        self.path = config_path
        self.observer = watchdog.observers.Observer()
        
    def on_modified(self, event):
        if event.src_path == self.path:
            config._load_file_overrides(self.path)
            print("Configuration reloaded")
```

### 3. Configuration CLI
```python
# config_cli.py
import click

@click.group()
def cli():
    """Configuration management CLI"""
    pass

@cli.command()
@click.option('--section', help='Config section')
@click.option('--key', help='Setting key')
@click.option('--value', help='New value')
def set(section, key, value):
    """Set a configuration value"""
    if hasattr(config, section):
        section_obj = getattr(config, section)
        if hasattr(section_obj, key):
            # Convert and set value
            current = getattr(section_obj, key)
            converted = config._convert_env_value(value, type(current))
            setattr(section_obj, key, converted)
            config.save()
            print(f"Set {section}.{key} = {converted}")

@cli.command()
def show():
    """Show current configuration"""
    print(config)

@cli.command()
def validate():
    """Validate configuration"""
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"❌ {error}")
    else:
        print("✅ Configuration valid")
```

## Testing Strategy

```python
# tests/test_config.py
import pytest
from config import Config, AudioConfig

def test_default_values():
    """Test default configuration values"""
    c = Config()
    assert c.audio.sample_rate == 16000
    assert c.model.llm_provider.value == "openai"
    
def test_env_override(monkeypatch):
    """Test environment variable overrides"""
    monkeypatch.setenv("CHATBOT_AUDIO_SAMPLE_RATE", "32000")
    c = Config()
    assert c.audio.sample_rate == 32000
    
def test_validation():
    """Test configuration validation"""
    c = Config()
    c.model.llm_temperature = 3.0  # Invalid
    errors = c.validate()
    assert len(errors) > 0
```

## Benefits of This Approach

1. **Single Source of Truth**: All configuration in one place
2. **Type Safety**: Dataclasses provide type checking
3. **Validation**: Automatic validation prevents errors
4. **Flexibility**: Easy to override via env vars or files
5. **Documentation**: Self-documenting with docstrings
6. **Testability**: Easy to mock and test
7. **Extensibility**: Easy to add new settings

## Migration Path

From scattered configuration:
```python
# OLD: Settings scattered across files
SAMPLE_RATE = 16000  # audio.py
MODEL_NAME = "whisper-base"  # stt.py
API_KEY = os.getenv("API_KEY")  # llm.py
```

To centralized configuration:
```python
# NEW: All settings in config.py
from config import config

sample_rate = config.audio.sample_rate
model_name = config.model.whisper_model
api_key = config.api_keys["openai"]
```

---

*One configuration to rule them all, one configuration to find them,
One configuration to bring them all, and in the chatbot bind them!*