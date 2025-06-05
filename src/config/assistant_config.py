"""
Configuration dataclasses for the Voice Assistant.

This module replaces the 27+ parameters in VoiceAssistant.__init__() with
well-organized, type-safe configuration objects.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    model_size: str = "base"
    device: Optional[str] = None


@dataclass  
class TTSConfig:
    """Text-to-Speech configuration."""
    device: Optional[str] = None
    voice_reference_path: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8


@dataclass
class LLMConfig:
    """Large Language Model configuration."""
    use_gemini: bool = True
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "models/gemini-1.5-flash"
    system_prompt: Optional[str] = None
    max_response_tokens: int = 5000
    temperature: float = 1.0


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    chunk_size: int = 1024
    min_amplitude: float = 0.015


@dataclass
class InputConfig:
    """Input mode and Voice Activity Detection configuration."""
    mode: str = "vad"  # "vad" or "push_to_talk"
    vad_aggressiveness: int = 1
    vad_speech_threshold: float = 0.3
    vad_silence_threshold: float = 0.8
    push_to_talk_key: str = "space"


@dataclass
class SoundEffectsConfig:
    """Sound effects configuration."""
    enable_sound_effects: bool = True
    sound_effect_volume: float = 0.2
    enable_interruption_sound: bool = True
    enable_generation_sound: bool = True


@dataclass
class ConversationConfig:
    """Conversation logging and history configuration."""
    log_conversations: bool = True
    conversation_log_dir: str = "conversation_logs"
    max_history_messages: int = 2000
    auto_summarize_conversations: bool = True
    max_summaries_to_load: int = 2000


@dataclass
class VoiceAssistantConfig:
    """Main configuration container for VoiceAssistant."""
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    input: InputConfig = field(default_factory=InputConfig)
    sound_effects: SoundEffectsConfig = field(default_factory=SoundEffectsConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)

    @classmethod
    def create_from_legacy_params(cls, **kwargs) -> 'VoiceAssistantConfig':
        """
        Create configuration from legacy VoiceAssistant parameters.
        
        This factory method allows gradual migration from the old parameter-heavy
        initialization to the new configuration-based approach.
        """
        # Extract STT parameters
        stt_config = STTConfig(
            model_size=kwargs.get('whisper_model_size', 'base'),
            device=kwargs.get('whisper_device')
        )
        
        # Extract TTS parameters
        tts_config = TTSConfig(
            device=kwargs.get('tts_device'),
            voice_reference_path=kwargs.get('voice_reference_path'),
            exaggeration=kwargs.get('voice_exaggeration', 0.5),
            cfg_weight=kwargs.get('voice_cfg_weight', 0.5),
            temperature=kwargs.get('voice_temperature', 0.8)
        )
        
        # Extract LLM parameters
        llm_config = LLMConfig(
            use_gemini=kwargs.get('use_gemini', True),
            base_url=kwargs.get('llm_base_url', 'http://localhost:1234/v1'),
            api_key=kwargs.get('llm_api_key', 'not-needed'),
            gemini_api_key=kwargs.get('gemini_api_key'),
            gemini_model=kwargs.get('gemini_model', 'models/gemini-1.5-flash'),
            system_prompt=kwargs.get('system_prompt'),
            max_response_tokens=kwargs.get('max_response_tokens', 5000),
            temperature=kwargs.get('llm_temperature', 1.0)
        )
        
        # Extract Audio parameters
        audio_config = AudioConfig(
            sample_rate=kwargs.get('sample_rate', 16000),
            chunk_size=kwargs.get('chunk_size', 1024),
            min_amplitude=kwargs.get('min_audio_amplitude', 0.015)
        )
        
        # Extract Input parameters
        input_config = InputConfig(
            mode=kwargs.get('input_mode', 'vad'),
            vad_aggressiveness=kwargs.get('vad_aggressiveness', 1),
            vad_speech_threshold=kwargs.get('vad_speech_threshold', 0.3),
            vad_silence_threshold=kwargs.get('vad_silence_threshold', 0.8),
            push_to_talk_key=kwargs.get('push_to_talk_key', 'space')
        )
        
        # Extract Sound Effects parameters
        sound_effects_config = SoundEffectsConfig(
            enable_sound_effects=kwargs.get('enable_sound_effects', True),
            sound_effect_volume=kwargs.get('sound_effect_volume', 0.2),
            enable_interruption_sound=kwargs.get('enable_interruption_sound', True),
            enable_generation_sound=kwargs.get('enable_generation_sound', True)
        )
        
        # Extract Conversation parameters
        conversation_config = ConversationConfig(
            log_conversations=kwargs.get('log_conversations', True),
            conversation_log_dir=kwargs.get('conversation_log_dir', 'conversation_logs'),
            max_history_messages=kwargs.get('max_history_messages', 2000),
            auto_summarize_conversations=kwargs.get('auto_summarize_conversations', True),
            max_summaries_to_load=kwargs.get('max_summaries_to_load', 2000)
        )
        
        return cls(
            stt=stt_config,
            tts=tts_config,
            llm=llm_config,
            audio=audio_config,
            input=input_config,
            sound_effects=sound_effects_config,
            conversation=conversation_config
        )

    def get_legacy_llm_params(self) -> dict:
        """
        Get LLM parameters in the format expected by legacy code.
        
        This helper method supports gradual migration.
        """
        return {
            'use_gemini': self.llm.use_gemini,
            'llm_base_url': self.llm.base_url,
            'llm_api_key': self.llm.api_key,
            'gemini_api_key': self.llm.gemini_api_key,
            'gemini_model': self.llm.gemini_model,
            'system_prompt': self.llm.system_prompt,
            'max_response_tokens': self.llm.max_response_tokens,
            'llm_temperature': self.llm.temperature
        }

    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate input mode
        if self.input.mode not in ['vad', 'push_to_talk']:
            raise ValueError(f"Invalid input mode: {self.input.mode}. Must be 'vad' or 'push_to_talk'")
        
        # Validate thresholds
        if not (0 <= self.input.vad_speech_threshold <= 1):
            raise ValueError(f"vad_speech_threshold must be between 0 and 1, got {self.input.vad_speech_threshold}")
        
        if not (0 <= self.input.vad_silence_threshold <= 1):
            raise ValueError(f"vad_silence_threshold must be between 0 and 1, got {self.input.vad_silence_threshold}")
        
        # Validate audio settings
        if self.audio.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.audio.sample_rate}")
        
        if self.audio.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.audio.chunk_size}")
        
        # Validate LLM settings (only warn, don't fail - allow for development scenarios)
        if self.llm.use_gemini and not self.llm.gemini_api_key:
            print("⚠️ Warning: gemini_api_key not provided, will fall back to local LLM")
        
        return True 