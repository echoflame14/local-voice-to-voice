#!/usr/bin/env python3
"""
Elegant Chatbot - Simplified Windows-friendly version
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
    print("🎨 Elegant Chatbot (Simple Mode)")
    print("=" * 50)
    print(f"LLM: {config.model.llm_model}")
    print(f"STT: Whisper {config.model.whisper_model}")
    print("=" * 50)
    
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
    
    print("Loading Whisper model...")
    stt = WhisperSTT(config)
    stt.load()  # Preload the model
    
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
    
    print("\n🎤 Ready! Speak and then pause for 1 second.")
    print("   Press Ctrl+C to exit.\n")
    
    # Main loop
    audio.start_recording()
    speech_buffer = []
    pre_speech_buffer = []  # Keep last few chunks before speech
    silence_count = 0
    speaking = False
    
    try:
        while True:
            # Get audio chunk
            chunk = audio.get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue
            
            # Check for speech
            if audio.is_speech(chunk):
                if not speaking:
                    print("🎯 Listening...", end="", flush=True)
                    speaking = True
                    # Add pre-speech buffer to capture beginning
                    speech_buffer.extend(pre_speech_buffer[-10:])  # Last ~300ms
                speech_buffer.append(chunk)
                silence_count = 0
            else:
                # Keep rolling buffer of recent chunks
                pre_speech_buffer.append(chunk)
                if len(pre_speech_buffer) > 20:
                    pre_speech_buffer.pop(0)
                    
                if speaking:
                    # Add silence to buffer to avoid cutting off
                    speech_buffer.append(chunk)
                    silence_count += 1
                    
                    # After ~1.5 seconds of silence, process
                    if silence_count > 45:
                        print(" Done!")
                        
                        # Stop recording briefly
                        audio.stop_recording()
                        
                        # Process speech
                        if speech_buffer:
                            audio_data = np.concatenate(speech_buffer)
                            
                            print("🔄 Processing...")
                            text = stt.transcribe(audio_data)
                            
                            if text:
                                print(f"👤 You: {text}")
                                
                                # Generate response
                                response = llm.generate(text)
                                print(f"🤖 Assistant: {response}")
                                
                                # Speak response
                                print("🔊 Speaking...")
                                tts_audio = tts.synthesize(response)
                                audio.play_audio(tts_audio)
                                print("✓ Done speaking\n")
                            else:
                                print("❌ Couldn't understand audio\n")
                        
                        # Reset
                        speech_buffer = []
                        silence_count = 0
                        speaking = False
                        
                        # Resume recording
                        audio.start_recording()
                        print("🎤 Ready for next input...")
                        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        audio.close()


if __name__ == "__main__":
    main()