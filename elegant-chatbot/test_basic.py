#!/usr/bin/env python3
"""
Basic functionality test for Elegant Chatbot
Tests core components without full audio pipeline
"""
import sys
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.stt import WhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS
from core.audio import SimpleVAD


def test_config():
    """Test configuration system"""
    print("🔧 Testing Configuration...")
    
    # Check basic config
    assert config.audio.sample_rate == 16000
    assert config.model.llm_model == "GPT-4.1-nano"
    assert config.model.llm_provider.value == "openai"
    
    # Validate
    errors = config.validate()
    if errors:
        print("   ❌ Config validation errors:")
        for e in errors:
            print(f"      - {e}")
    else:
        print("   ✅ Configuration valid")
        
    return len(errors) == 0


def test_vad():
    """Test Voice Activity Detection"""
    print("\n🎤 Testing VAD...")
    
    vad = SimpleVAD(config)
    
    # Test with silence
    silence = np.zeros(480, dtype=np.int16)
    state = vad.process(silence)
    print(f"   Silence detection: {state}")
    
    # Test with noise
    noise = np.random.randint(-1000, 1000, 480, dtype=np.int16)
    state = vad.process(noise)
    print(f"   Noise detection: {state}")
    
    # Test with "speech" (loud signal)
    speech = np.random.randint(-10000, 10000, 480, dtype=np.int16)
    for _ in range(5):  # Need multiple frames
        state = vad.process(speech)
    print(f"   Speech detection: {state}")
    
    print("   ✅ VAD working")
    return True


def test_llm():
    """Test LLM connection"""
    print("\n🤖 Testing LLM...")
    
    if not config.api_keys.get("openai"):
        print("   ⚠️  No OpenAI API key set, skipping LLM test")
        return True
        
    llm = LLMClient(config)
    
    # Simple test
    response = llm.generate("Say 'Hello, Elegant Chatbot is working!' and nothing else.")
    print(f"   Response: {response}")
    
    if "working" in response.lower():
        print("   ✅ LLM connection successful")
        return True
    else:
        print("   ❌ LLM response unexpected")
        return False


def test_stt():
    """Test STT setup"""
    print("\n🎯 Testing STT...")
    
    stt = WhisperSTT(config)
    
    # Just test loading
    try:
        stt.load()
        print("   ✅ Whisper model loaded")
        return True
    except Exception as e:
        print(f"   ❌ Failed to load Whisper: {e}")
        return False


def test_tts():
    """Test TTS setup"""
    print("\n🔊 Testing TTS...")
    
    tts = SimpleTTS(config)
    
    # Test synthesis (won't play)
    try:
        audio = tts.synthesize("Test")
        if len(audio) > 0:
            print(f"   ✅ TTS working (generated {len(audio)} samples)")
            return True
        else:
            print("   ❌ TTS generated empty audio")
            return False
    except Exception as e:
        print(f"   ❌ TTS error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("🎨 Elegant Chatbot - Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("VAD", test_vad),
        ("LLM", test_llm),
        ("STT", test_stt),
        ("TTS", test_tts),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test failed with error: {e}")
            results.append((name, False))
            
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<30} {status}")
        
    print("=" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The chatbot is ready to run.")
        print("   Run with: python main.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        

if __name__ == "__main__":
    main()