#!/usr/bin/env python3
"""
Elegant Chatbot - Press Enter to stop recording version
Much simpler and more reliable
"""
import sys
import time
import numpy as np
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.audio_simple import SimpleAudioSystem
from core.stt import WhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS


def record_until_enter(audio):
    """Record audio until Enter is pressed"""
    print("\n🎤 Recording... Press ENTER to stop")
    
    # Start recording
    audio.start_recording()
    all_audio = []
    
    # Recording thread
    def record_loop():
        while audio.recording:
            chunk = audio.get_audio_chunk()
            if chunk is not None:
                all_audio.append(chunk)
            time.sleep(0.01)
    
    record_thread = threading.Thread(target=record_loop)
    record_thread.start()
    
    # Wait for Enter
    input()
    
    # Stop recording
    audio.stop_recording()
    record_thread.join()
    
    print("✅ Recording stopped!")
    
    if all_audio:
        return np.concatenate(all_audio)
    return None


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (Manual Recording Mode)")
    print("=" * 50)
    print("This mode lets you control exactly when to stop recording")
    print("=" * 50)
    
    # Use tiny model for speed
    config.model.whisper_model = "tiny"
    
    # Validate config
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    
    # Initialize components
    print("\nInitializing components...")
    
    audio = SimpleAudioSystem(config)
    
    print("Loading Whisper tiny model...")
    stt = WhisperSTT(config)
    stt.load()
    
    print("Initializing LLM...")
    llm = LLMClient(config)
    
    print("Initializing TTS...")
    tts = SimpleTTS(config)
    
    # Start audio
    try:
        audio.start()
    except Exception as e:
        print(f"\n❌ Audio initialization failed: {e}")
        return
    
    print("\n✅ Ready to chat!")
    print("=" * 50)
    
    try:
        while True:
            print("\n[Press ENTER to start recording]")
            input()
            
            # Record audio
            audio_data = record_until_enter(audio)
            
            if audio_data is not None and len(audio_data) > 0:
                duration = len(audio_data) / 16000
                print(f"\n📏 Recorded {duration:.1f} seconds of audio")
                
                # Check audio level
                max_level = np.max(np.abs(audio_data))
                print(f"📊 Max audio level: {max_level}")
                
                if max_level < 100:
                    print("⚠️  Very quiet recording, speak louder next time")
                
                # Transcribe
                print("\n🔄 Transcribing...")
                start_time = time.time()
                
                try:
                    text = stt.transcribe(audio_data)
                    transcribe_time = time.time() - start_time
                    print(f"⏱️  Transcription took {transcribe_time:.1f}s")
                    
                    if text:
                        print(f"\n👤 You said: \"{text}\"")
                        
                        # Generate response
                        print("\n🤖 Thinking...")
                        response = llm.generate(text)
                        print(f"💬 Response: \"{response}\"")
                        
                        # Speak response
                        print("\n🔊 Speaking...")
                        tts_audio = tts.synthesize(response)
                        audio.play_audio(tts_audio)
                        print("✓ Done!")
                    else:
                        print("❌ Couldn't transcribe audio (too quiet or unclear)")
                        
                except Exception as e:
                    print(f"❌ Error during processing: {e}")
                    
            else:
                print("❌ No audio recorded")
                
            print("\n" + "=" * 50)
                        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        audio.close()


if __name__ == "__main__":
    main()