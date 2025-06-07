#!/usr/bin/env python3
"""
Elegant Chatbot - Using faster-whisper for much better performance
"""
print("Starting elegant chatbot...")

import sys
import time
import numpy as np
from pathlib import Path

print("Imports starting...")

sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import config
    print("Config imported successfully")
except Exception as e:
    print(f"Error importing config: {e}")
    sys.exit(1)
from core.audio_simple import SimpleAudioSystem
from core.stt_faster import FasterWhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (Faster-Whisper Edition)")
    print("=" * 50)
    
    # Allow model override via command line
    if len(sys.argv) > 1:
        model = sys.argv[1]
        print(f"Using model: {model}")
        config.model.llm_model = model
    
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
    
    print("\n🎤 Ready! Speak and then pause for 1-2 seconds.")
    print("   Using faster-whisper for better performance.")
    print("   Press Ctrl+C to exit.\n")
    
    # Main loop
    audio.start_recording()
    speech_buffer = []
    pre_speech_buffer = []
    silence_count = 0
    speaking = False
    
    try:
        while True:
            # Get audio chunk
            chunk = audio.get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
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
                            
                            print("🔄 Processing with faster-whisper...")
                            start_time = time.time()
                            text = stt.transcribe(audio_data)
                            process_time = time.time() - start_time
                            print(f"⏱️  Processed in {process_time:.1f}s")
                            
                            if text:
                                print(f"\n👤 You: {text}")
                                
                                # Add memory context if enabled
                                context = None
                                if hasattr(llm, 'get_context'):
                                    context = llm.get_context()
                                
                                # Generate response
                                print("🤖 Thinking...")
                                response = llm.generate(text, context)
                                print(f"💬 Assistant: {response}")
                                
                                # Speak response
                                print("🔊 Speaking...")
                                tts_audio = tts.synthesize(response)
                                audio.play_audio(tts_audio)
                                print("✓ Done\n")
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