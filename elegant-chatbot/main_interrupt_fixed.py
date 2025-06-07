#!/usr/bin/env python3
"""
Elegant Chatbot - With FIXED interrupt support
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


class InterruptibleAudioPlayer:
    """Wrapper for audio playback with better interrupt support"""
    
    def __init__(self, audio_system):
        self.audio = audio_system
        self.interrupt_event = threading.Event()
        self.interrupt_audio = []
        
    def play_with_interrupt_detection(self, audio_data):
        """Play audio while monitoring for interrupts"""
        self.interrupt_event.clear()
        self.interrupt_audio = []
        
        # Start interrupt monitor thread
        monitor_thread = threading.Thread(
            target=self._monitor_interrupts,
            daemon=True
        )
        monitor_thread.start()
        
        # Play audio in small chunks for better responsiveness
        chunk_size = 480  # 30ms chunks for quick interrupt response
        self.audio.is_playing = True
        
        for i in range(0, len(audio_data), chunk_size):
            # Check for interrupt before each chunk
            if self.interrupt_event.is_set():
                self.audio.is_playing = False
                return True  # Was interrupted
                
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            try:
                self.audio.output_stream.write(chunk.tobytes())
            except:
                break
                
        self.audio.is_playing = False
        return False  # Completed without interrupt
        
    def _monitor_interrupts(self):
        """Monitor for speech during playback"""
        consecutive_speech = 0
        
        while self.audio.is_playing:
            chunk = self.audio.get_audio_chunk()
            if chunk is not None and self.audio.is_speech(chunk):
                consecutive_speech += 1
                self.interrupt_audio.append(chunk)
                
                # Require 3 consecutive chunks of speech (90ms)
                if consecutive_speech >= 3:
                    print("\n🛑 Interrupt detected!", end="", flush=True)
                    self.interrupt_event.set()
                    self.audio.stop_playback()
                    return
            else:
                consecutive_speech = 0
                if len(self.interrupt_audio) > 10:
                    # Keep only recent audio
                    self.interrupt_audio = self.interrupt_audio[-10:]
                    
            time.sleep(0.01)


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (With FIXED Interrupts)")
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
    player = InterruptibleAudioPlayer(audio)
    
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
    print("   Just start talking and I'll stop to listen.")
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
            
            # Skip if we're playing audio (handled by interrupt monitor)
            if audio.is_playing:
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
                        
                        # Process speech
                        if speech_buffer:
                            # Check if we have interrupt audio to prepend
                            if player.interrupt_audio:
                                print(" (including interrupt audio)")
                                speech_buffer = player.interrupt_audio + speech_buffer
                                player.interrupt_audio = []
                            
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
                                print("🔊 Speaking... (interrupt me anytime)")
                                tts_audio = tts.synthesize(response)
                                
                                # Play with interrupt detection
                                was_interrupted = player.play_with_interrupt_detection(tts_audio)
                                
                                if was_interrupted:
                                    print(" - Interrupted!\n")
                                    # The interrupt audio is saved in player.interrupt_audio
                                    # It will be included in the next transcription
                                else:
                                    print(" - Complete\n")
                                    
                            else:
                                print("❌ Couldn't understand audio\n")
                        
                        # Reset
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