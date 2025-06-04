import pytest
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def test_voice_path():
    """Path to a test voice file."""
    return os.path.join(os.path.dirname(__file__), 'data', 'test_voice.wav')

@pytest.fixture
def test_text():
    """Sample text for TTS testing."""
    return "This is a test of the voice synthesis system."

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "choices": [{
            "message": {
                "content": "This is a test response from the language model."
            }
        }]
    } 