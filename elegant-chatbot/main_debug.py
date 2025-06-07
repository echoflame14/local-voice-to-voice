#!/usr/bin/env python3
"""
Elegant Chatbot - Debug version with visual feedback
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
    print("🎨 Elegant Chatbot (Debug Mode)")
    print("=" * 50)
    
    # Lower the VAD threshold for better detection
    config.audio.vad_threshold = 0.3  # Lower threshold
    
    # Initialize components
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
    
    print("\n🎤 Ready! Speak and watch the level meter.")
    print("   The meter shows your audio level in real-time.")
    print("   Press Ctrl+C to exit.\n")
    
    # Main loop with visual feedback
    audio.start_recording()
    speech_buffer = []
    silence_count = 0
    speaking = False
    frame_count = 0
    
    try:
        while True:
            # Get audio chunk
            chunk = audio.get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Calculate energy level
            mean_square = np.mean(chunk.astype(np.float32)**2)
            energy = np.sqrt(mean_square) if mean_square > 0 else 0
            
            # Visual level meter
            bars = int(energy / 200)  # Scale for display
            level_display = f"[{'#' * min(bars, 30):<30}] {energy:.0f}"
            
            # Check for speech (with visual feedback)
            is_speech = energy > (config.audio.vad_threshold * 1000)
            
            if is_speech:
                if not speaking:
                    print(f"\n🎯 SPEECH DETECTED! Recording...")
                    speaking = True
                speech_buffer.append(chunk)
                silence_count = 0
                
                # Show level while speaking
                print(f"\r📊 Speaking: {level_display}", end="", flush=True)
            else:
                # Show level while silent
                if not speaking:
                    # Update every 10 frames to reduce flicker
                    if frame_count % 10 == 0:
                        print(f"\r💤 Waiting: {level_display} (threshold: {config.audio.vad_threshold * 1000:.0f})", end="", flush=True)
                
                if speaking:
                    silence_count += 1
                    print(f"\r🤫 Silence: {silence_count}/30 frames", end="", flush=True)
                    
                    # After ~1 second of silence, process
                    if silence_count > 30:
                        print("\n✅ Processing speech...\n")
                        
                        # Stop recording briefly
                        audio.stop_recording()
                        
                        # Process speech
                        if speech_buffer:
                            audio_data = np.concatenate(speech_buffer)
                            print(f"📏 Captured {len(audio_data)/16000:.1f} seconds of audio")
                            
                            print("🔄 Transcribing...")
                            text = stt.transcribe(audio_data)
                            
                            if text:
                                print(f"👤 You said: \"{text}\"")
                                
                                # Generate response
                                print("🤖 Thinking...")
                                response = llm.generate(text)
                                print(f"💬 Response: \"{response}\"")
                                
                                # Speak response
                                print("🔊 Speaking...")
                                tts_audio = tts.synthesize(response)
                                audio.play_audio(tts_audio)
                                print("✓ Done!\n")
                            else:
                                print("❌ Couldn't transcribe audio (too quiet or unclear)\n")
                        else:
                            print("❌ No audio captured\n")
                        
                        # Reset
                        speech_buffer = []
                        silence_count = 0
                        speaking = False
                        frame_count = 0
                        
                        # Resume recording
                        audio.start_recording()
                        print("🎤 Ready for next input...\n")
                        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        audio.close()


if __name__ == "__main__":
    main()