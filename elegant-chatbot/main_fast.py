#!/usr/bin/env python3
"""
Elegant Chatbot - Fast version with tiny Whisper model
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.audio_simple import SimpleAudioSystem
from core.stt import WhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (Fast Mode)")
    print("=" * 50)
    
    # Use tiny model for speed
    config.model.whisper_model = "tiny"
    config.audio.vad_threshold = 0.2  # More sensitive
    
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
    
    print("Loading Whisper tiny model (faster)...")
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
    
    print("\n🎤 Ready! Using fast mode with tiny Whisper model.")
    print("   Speak clearly and pause for 2 seconds when done.")
    print("   Press Ctrl+C to exit.\n")
    
    # Main loop with longer buffers
    audio.start_recording()
    all_chunks = []  # Keep ALL audio
    silence_count = 0
    speaking = False
    max_recording_chunks = 500  # ~15 seconds max
    
    try:
        while True:
            # Get audio chunk
            chunk = audio.get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue
            
            # Always add to buffer
            all_chunks.append(chunk)
            
            # Limit buffer size
            if len(all_chunks) > max_recording_chunks:
                all_chunks.pop(0)
            
            # Check for speech
            if audio.is_speech(chunk):
                if not speaking:
                    print("🎯 Listening... (speak freely, I'll wait for a 2-second pause)")
                    speaking = True
                    # Mark where speech started
                    speech_start = max(0, len(all_chunks) - 20)  # Include pre-speech
                silence_count = 0
            else:
                if speaking:
                    silence_count += 1
                    
                    # Show progress
                    if silence_count % 10 == 0:
                        print(f"   ... waiting for pause ({silence_count/30:.1f}s of silence)")
                    
                    # After 2 seconds of silence, process
                    if silence_count > 60:
                        print("\n✅ Got it! Processing...\n")
                        
                        # Stop recording
                        audio.stop_recording()
                        
                        # Get speech portion (from start to now)
                        speech_chunks = all_chunks[speech_start:]
                        
                        if speech_chunks:
                            audio_data = np.concatenate(speech_chunks)
                            duration = len(audio_data) / 16000
                            print(f"📏 Captured {duration:.1f} seconds of audio")
                            
                            # Quick transcription with tiny model
                            print("🔄 Transcribing with tiny model...")
                            start_time = time.time()
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
                                print("✓ Done!\n")
                            else:
                                print("❌ Couldn't transcribe audio\n")
                        
                        # Reset
                        all_chunks = []
                        silence_count = 0
                        speaking = False
                        
                        # Resume recording
                        audio.start_recording()
                        print("🎤 Ready for next input...\n")
                        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        audio.close()


if __name__ == "__main__":
    main()