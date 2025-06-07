#!/usr/bin/env python3
"""
Test TTS functionality
"""
import pyttsx3
import tempfile
import os
import time

def test_pyttsx3_direct():
    """Test pyttsx3 directly"""
    print("Testing pyttsx3 direct speech...")
    try:
        engine = pyttsx3.init()
        engine.say("Hello, this is a test")
        engine.runAndWait()
        print("✓ Direct speech worked")
    except Exception as e:
        print(f"✗ Direct speech failed: {e}")

def test_pyttsx3_file():
    """Test pyttsx3 file generation"""
    print("\nTesting pyttsx3 file generation...")
    try:
        engine = pyttsx3.init()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        print(f"Saving to: {tmp_path}")
        
        # Try to save
        engine.save_to_file("Hello, this is a file test", tmp_path)
        engine.runAndWait()
        
        # Wait for file
        time.sleep(0.5)
        
        # Check file
        if os.path.exists(tmp_path):
            size = os.path.getsize(tmp_path)
            print(f"✓ File created, size: {size} bytes")
            if size == 0:
                print("✗ But file is empty!")
        else:
            print("✗ File was not created")
        
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"✗ File generation failed: {e}")

def test_windows_sapi():
    """Test Windows SAPI directly"""
    print("\nTesting Windows SAPI...")
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak("Hello from Windows SAPI")
        print("✓ Windows SAPI worked")
    except Exception as e:
        print(f"✗ Windows SAPI failed: {e}")
        print("  You may need to: pip install pywin32")

def test_simple_tts():
    """Test our SimpleTTS class"""
    print("\nTesting SimpleTTS class...")
    try:
        from config import config
        from core.tts import SimpleTTS
        
        tts = SimpleTTS(config)
        audio = tts.synthesize("Testing SimpleTTS")
        print(f"✓ SimpleTTS worked, audio shape: {audio.shape}")
    except Exception as e:
        print(f"✗ SimpleTTS failed: {e}")

if __name__ == "__main__":
    print("=== TTS Testing ===\n")
    
    test_pyttsx3_direct()
    test_pyttsx3_file()
    test_windows_sapi()
    test_simple_tts()
    
    print("\nDone!")