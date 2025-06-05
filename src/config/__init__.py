"""
Configuration management for the voice-to-voice assistant.

This package provides dataclass-based configuration objects that replace
the long parameter lists in the VoiceAssistant class.
"""

from .assistant_config import (
    STTConfig,
    TTSConfig,
    LLMConfig,
    AudioConfig,
    InputConfig,
    SoundEffectsConfig,
    ConversationConfig,
    VoiceAssistantConfig
)

__all__ = [
    'STTConfig',
    'TTSConfig', 
    'LLMConfig',
    'AudioConfig',
    'InputConfig',
    'SoundEffectsConfig',
    'ConversationConfig',
    'VoiceAssistantConfig'
] 