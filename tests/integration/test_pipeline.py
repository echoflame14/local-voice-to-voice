import pytest
from src.pipeline import VoicePipeline
from src.llm import LLMClient
from src.tts import TTSClient
from src.stt import STTClient

def test_voice_pipeline_initialization():
    """Test that the voice pipeline can be initialized with default settings."""
    pipeline = VoicePipeline()
    assert pipeline is not None
    assert isinstance(pipeline.llm_client, LLMClient)
    assert isinstance(pipeline.tts_client, TTSClient)
    assert isinstance(pipeline.stt_client, STTClient)

def test_voice_to_text_conversion(test_voice_path):
    """Test that voice can be converted to text."""
    pipeline = VoicePipeline()
    text = pipeline.transcribe_audio(test_voice_path)
    assert isinstance(text, str)
    assert len(text) > 0

def test_text_to_speech_generation(test_text):
    """Test that text can be converted to speech."""
    pipeline = VoicePipeline()
    audio = pipeline.synthesize_speech(test_text)
    assert audio is not None
    assert len(audio) > 0

def test_llm_response_generation(mock_llm_response, monkeypatch):
    """Test that LLM can generate responses."""
    def mock_generate(*args, **kwargs):
        return mock_llm_response["choices"][0]["message"]["content"]
    
    pipeline = VoicePipeline()
    monkeypatch.setattr(pipeline.llm_client, "generate", mock_generate)
    
    response = pipeline.generate_response("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_end_to_end_pipeline(test_voice_path, mock_llm_response, monkeypatch):
    """Test the complete pipeline from voice input to voice output."""
    def mock_generate(*args, **kwargs):
        return mock_llm_response["choices"][0]["message"]["content"]
    
    pipeline = VoicePipeline()
    monkeypatch.setattr(pipeline.llm_client, "generate", mock_generate)
    
    # Voice to text
    text = pipeline.transcribe_audio(test_voice_path)
    assert isinstance(text, str)
    
    # Text to LLM response
    response = pipeline.generate_response(text)
    assert isinstance(response, str)
    
    # Response to voice
    audio = pipeline.synthesize_speech(response)
    assert audio is not None
    assert len(audio) > 0 