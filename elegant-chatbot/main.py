#!/usr/bin/env python3
"""
Elegant Chatbot - Final interrupt implementation
Properly handles interrupts with continuous recording
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.audio_interruptible import InterruptibleAudioSystem
from core.stt_faster import FasterWhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (Final Interrupt Support)")
    print("=" * 50)
    
    # Enable interrupts and adjust thresholds
    config.features.enable_interrupts = True
    config.audio.vad_threshold = 0.3  # Lower threshold for better detection
    
    # Validate config
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    
    # Initialize components
    print("\nInitializing components...")
    
    audio = InterruptibleAudioSystem(config)
    
    print("Loading faster-whisper model...")
    stt = FasterWhisperSTT(config)
    stt.load()
    
    print("Initializing LLM...")
    llm = LLMClient(config)
    llm.warm_up()
    
    print("Initializing TTS...")
    tts = SimpleTTS(config)
    
    # Start audio
    try:
        audio.start()
    except Exception as e:
        print(f"\n❌ Audio initialization failed: {e}")
        return
    
    print("\n🎤 Ready! You can interrupt me while I'm speaking.")
    print("   Just start talking and I'll stop immediately.")
    print("   Press Ctrl+C to exit.\n")
    
    # Main conversation loop
    audio.start_recording()
    speech_buffer = []
    pre_speech_buffer = []
    silence_count = 0
    speaking = False
    interrupt_audio = []
    
    def handle_interrupt(audio_chunks):
        """Handle interrupt audio"""
        nonlocal interrupt_audio
        interrupt_audio = audio_chunks
    
    try:
        while True:
            # Get audio chunk
            chunk = audio.get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue
            
            # Skip processing if we're playing (interrupt handler will catch speech)
            if audio.is_playing:
                continue
            
            # Keep pre-speech buffer
            pre_speech_buffer.append(chunk)
            if len(pre_speech_buffer) > 20:  # ~600ms
                pre_speech_buffer.pop(0)
            
            # Check for speech
            if audio.is_speech(chunk):
                if not speaking:
                    print("🎯 Listening...", end="", flush=True)
                    speaking = True
                    # Include any interrupt audio first
                    if interrupt_audio:
                        speech_buffer = interrupt_audio.copy()
                        interrupt_audio = []
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
                                
                                # Speak response with interrupt support
                                print("🔊 Speaking... (interrupt me anytime)")
                                tts_audio = tts.synthesize(response)
                                
                                # Play with interrupt detection
                                was_interrupted = audio.play_audio_interruptible(
                                    tts_audio, 
                                    on_interrupt=handle_interrupt
                                )
                                
                                if was_interrupted:
                                    print(" - Interrupted! Continuing...\n")
                                else:
                                    print(" - Complete\n")
                                    
                            else:
                                print("❌ Couldn't understand audio\n")
                        
                        # Reset for next input
                        speech_buffer = []
                        pre_speech_buffer = []
                        silence_count = 0
                        speaking = False
                        
                        print("🎤 Ready for next input...")
                        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        audio.close()


if __name__ == "__main__":
    main()