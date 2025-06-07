#!/usr/bin/env python3
"""
Elegant Chatbot - With interrupt support
You can interrupt the assistant while it's speaking
"""
import sys
import time
import numpy as np
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.audio_simple import SimpleAudioSystem
from core.stt_faster import FasterWhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (With Interrupts)")
    print("=" * 50)
    
    # Enable interrupts
    config.features.enable_interrupts = True
    
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
    
    print("Loading faster-whisper model...")
    stt = FasterWhisperSTT(config)
    stt.load()
    
    print("Initializing LLM...")
    llm = LLMClient(config)
    llm.warm_up()  # Pre-warm the connection
    
    print("Initializing TTS...")
    tts = SimpleTTS(config)
    
    # Start audio
    try:
        audio.start()
    except Exception as e:
        print(f"\n❌ Audio initialization failed: {e}")
        return
    
    print("\n🎤 Ready! You can interrupt me while I'm speaking.")
    print("   Just start talking and I'll stop to listen.")
    print("   Press Ctrl+C to exit.\n")
    
    # Main loop
    audio.start_recording()
    speech_buffer = []
    pre_speech_buffer = []
    silence_count = 0
    speaking = False
    
    def monitor_for_interrupts():
        """Background thread to monitor for interrupts during playback"""
        interrupt_buffer = []
        while audio.is_playing:
            chunk = audio.get_audio_chunk()
            if chunk is not None and audio.is_speech(chunk):
                interrupt_buffer.append(chunk)
                if len(interrupt_buffer) > 3:  # ~90ms of speech
                    print("\n🖐️ Interrupt detected!")
                    audio.stop_playback()
                    # Save the interrupt audio
                    return interrupt_buffer
            else:
                interrupt_buffer = []
            time.sleep(0.01)
        return []
    
    try:
        while True:
            # Get audio chunk
            chunk = audio.get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue
            
            # Keep pre-speech buffer
            pre_speech_buffer.append(chunk)
            if len(pre_speech_buffer) > 20:
                pre_speech_buffer.pop(0)
            
            # Check for speech
            if audio.is_speech(chunk):
                if not speaking:
                    print("🎯 Listening...", end="", flush=True)
                    speaking = True
                    # Add pre-speech buffer
                    speech_buffer.extend(pre_speech_buffer)
                speech_buffer.append(chunk)
                silence_count = 0
            else:
                if speaking:
                    # Keep recording silence too
                    speech_buffer.append(chunk)
                    silence_count += 1
                    
                    # After 1.5 seconds of silence, process
                    if silence_count > 45:
                        print(" Done!")
                        
                        # Stop recording briefly
                        audio.stop_recording()
                        
                        # Process speech
                        if speech_buffer:
                            audio_data = np.concatenate(speech_buffer)
                            duration = len(audio_data) / 16000
                            print(f"📏 Captured {duration:.1f} seconds")
                            
                            print("🔄 Processing...")
                            text = stt.transcribe(audio_data)
                            
                            if text:
                                print(f"\n👤 You: {text}")
                                
                                # Generate response
                                print("🤖 Thinking...")
                                response = llm.generate(text)
                                print(f"💬 Assistant: {response}")
                                
                                # Speak response with interrupt monitoring
                                print("🔊 Speaking... (you can interrupt me)")
                                tts_audio = tts.synthesize(response)
                                
                                # Start monitoring for interrupts in background
                                interrupt_thread = threading.Thread(
                                    target=monitor_for_interrupts,
                                    daemon=True
                                )
                                interrupt_thread.start()
                                
                                # Play audio (can be interrupted)
                                audio.play_audio(tts_audio)
                                
                                # Wait for interrupt thread to finish
                                interrupt_thread.join(timeout=0.1)
                                
                                if audio.is_playing:
                                    print("✓ Finished speaking\n")
                                else:
                                    print("✓ Stopped for interrupt\n")
                                    # Continue listening immediately
                                    speech_buffer = []
                                    pre_speech_buffer = []
                                    silence_count = 0
                                    speaking = False
                                    audio.start_recording()
                                    continue
                                    
                            else:
                                print("❌ Couldn't understand audio\n")
                        
                        # Reset
                        speech_buffer = []
                        pre_speech_buffer = []
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