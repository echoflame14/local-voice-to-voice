#!/usr/bin/env python3
"""
Simple test script to verify all components work correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stt import WhisperSTT
from src.llm import OpenAICompatibleLLM
from src.tts import ChatterboxTTSWrapper
from src.audio import VoiceActivityDetector, AudioStreamManager

def test_stt():
    """Test Speech-to-Text"""
    print("🔤 Testing Whisper STT...")
    try:
        stt = WhisperSTT(model_size="tiny")  # Use tiny for faster testing
        print("✅ Whisper STT loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Whisper STT failed: {e}")
        return False

def test_llm():
    """Test Language Model"""
    print("🤖 Testing LLM connection...")
    try:
        llm = OpenAICompatibleLLM()
        response = llm.generate("Hello", max_tokens=5)
        
        if "sorry" in response.lower() and "trouble" in response.lower():
            print("❌ LLM connection failed - is LM Studio running?")
            return False
        
        print(f"✅ LLM response: {response}")
        return True
    except Exception as e:
        print(f"❌ LLM failed: {e}")
        return False

def test_tts():
    """Test Text-to-Speech"""
    print("🔊 Testing Chatterbox TTS...")
    try:
        tts = ChatterboxTTSWrapper(device="cpu")  # Use CPU for testing
        audio, sr = tts.synthesize("Hello, this is a test.", return_numpy=True)
        print(f"✅ TTS generated audio: {len(audio)} samples at {sr}Hz")
        return True
    except Exception as e:
        print(f"❌ TTS failed: {e}")
        return False

def test_audio():
    """Test Audio components"""
    print("🎤 Testing Audio components...")
    try:
        # Test VAD
        vad = VoiceActivityDetector()
        print("✅ VAD initialized")
        
        # Test Audio Manager
        audio_manager = AudioStreamManager()
        print("✅ Audio Manager initialized")
        audio_manager.cleanup()
        
        return True
    except Exception as e:
        print(f"❌ Audio components failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running component tests...\n")
    
    tests = [
        ("STT", test_stt),
        ("LLM", test_llm),
        ("TTS", test_tts),
        ("Audio", test_audio)
    ]
    
    results = {}
    
    for name, test_func in tests:
        print(f"\n{'='*50}")
        results[name] = test_func()
    
    print(f"\n{'='*50}")
    print("📊 Test Results:")
    print("='*50")
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:10} {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests passed! The voice assistant should work correctly.")
        print("\nNext steps:")
        print("1. Make sure LM Studio is running with a model loaded")
        print("2. Run: python main.py --text-mode  (for text testing)")
        print("3. Run: python main.py              (for voice mode)")
    else:
        print("⚠️  Some tests failed. Check the errors above and:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that LM Studio is running for LLM tests")
        print("3. Verify GPU/CPU settings for TTS tests")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 