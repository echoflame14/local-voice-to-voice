#!/usr/bin/env python3
"""
Elegant Chatbot - Full Debug Version
Shows everything that's happening with audio
"""
import sys
import time
import numpy as np
from pathlib import Path
import threading

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.audio_simple import SimpleAudioSystem
from core.stt_faster import FasterWhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (Full Debug Mode)")
    print("=" * 50)
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Sample rate: {config.audio.sample_rate}")
    print(f"  Chunk size: {config.audio.chunk_size}")
    print(f"  VAD threshold: {config.audio.vad_threshold}")
    print(f"  VAD enabled: {config.audio.vad_enabled}")
    
    # Lower threshold for testing
    config.audio.vad_threshold = 0.1  # Very sensitive
    print(f"  Adjusted VAD threshold: {config.audio.vad_threshold}")
    
    # Initialize
    print("\nInitializing components...")
    audio = SimpleAudioSystem(config)
    
    print("Loading STT...")
    stt = FasterWhisperSTT(config)
    stt.load()
    
    print("Loading LLM...")
    llm = LLMClient(config)
    
    print("Loading TTS...")
    tts = SimpleTTS(config)
    
    # Start audio
    try:
        audio.start()
    except Exception as e:
        print(f"\n❌ Audio initialization failed: {e}")
        return
    
    print("\n✅ Ready! Watch the debug output below:")
    print("=" * 50)
    
    # Test if we're getting audio at all
    print("\n📊 Testing audio input for 3 seconds...")
    audio.start_recording()
    
    max_level = 0
    avg_levels = []
    
    for i in range(100):  # ~3 seconds
        chunk = audio.get_audio_chunk()
        if chunk is not None:
            level = np.sqrt(np.mean(chunk.astype(np.float32)**2))
            max_level = max(max_level, level)
            avg_levels.append(level)
            
            # Show level bar
            bars = int(level / 100)
            print(f"\r[{'#' * min(bars, 40):<40}] Level: {level:.0f} | Max: {max_level:.0f}", end="", flush=True)
        else:
            print("\r⚠️  No audio chunk received!", end="", flush=True)
        
        time.sleep(0.03)
    
    print(f"\n\n📊 Audio test complete:")
    print(f"  Max level seen: {max_level:.0f}")
    print(f"  Avg level: {np.mean(avg_levels):.0f}")
    print(f"  VAD threshold: {config.audio.vad_threshold * 1000:.0f}")
    
    if max_level < 50:
        print("\n⚠️  WARNING: Very low audio levels detected!")
        print("  - Check your microphone is not muted")
        print("  - Check Windows sound settings")
        print("  - Try speaking louder")
    
    print("\n" + "=" * 50)
    print("🎤 Starting main loop with detailed debug info...")
    print("=" * 50 + "\n")
    
    # Main loop with extensive debugging
    speech_buffer = []
    pre_speech_buffer = []
    silence_count = 0
    speaking = False
    frame_count = 0
    last_levels = []
    
    try:
        while True:
            # Get audio chunk
            chunk = audio.get_audio_chunk()
            if chunk is None:
                print(f"\r⚠️  Frame {frame_count}: No audio chunk!", end="", flush=True)
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Calculate level
            level = np.sqrt(np.mean(chunk.astype(np.float32)**2))
            last_levels.append(level)
            if len(last_levels) > 10:
                last_levels.pop(0)
            
            # Keep pre-speech buffer
            pre_speech_buffer.append(chunk)
            if len(pre_speech_buffer) > 20:
                pre_speech_buffer.pop(0)
            
            # Check for speech
            is_speech = audio.is_speech(chunk)
            threshold = config.audio.vad_threshold * 1000
            
            # Debug output every 10 frames
            if frame_count % 10 == 0:
                avg_recent = np.mean(last_levels)
                status = "SPEAKING" if speaking else "WAITING"
                speech_indicator = "🔊" if is_speech else "🔇"
                print(f"\rFrame {frame_count} | {status} | Level: {level:.0f} (avg: {avg_recent:.0f}) | Threshold: {threshold:.0f} | {speech_indicator}", end="", flush=True)
            
            if is_speech:
                if not speaking:
                    print(f"\n\n🎯 SPEECH START at frame {frame_count}! Level: {level:.0f} > {threshold:.0f}")
                    speaking = True
                    speech_buffer.extend(pre_speech_buffer)
                speech_buffer.append(chunk)
                silence_count = 0
            else:
                if speaking:
                    speech_buffer.append(chunk)
                    silence_count += 1
                    
                    if silence_count % 10 == 0:
                        print(f"\n🤫 Silence count: {silence_count}/45 frames", end="", flush=True)
                    
                    # After 1.5 seconds of silence
                    if silence_count > 45:
                        print(f"\n\n✅ SPEECH END at frame {frame_count}!")
                        
                        # Stop recording
                        audio.stop_recording()
                        
                        # Process
                        if speech_buffer:
                            audio_data = np.concatenate(speech_buffer)
                            duration = len(audio_data) / 16000
                            max_amplitude = np.max(np.abs(audio_data))
                            
                            print(f"\n📏 Audio stats:")
                            print(f"  Duration: {duration:.1f}s")
                            print(f"  Samples: {len(audio_data)}")
                            print(f"  Max amplitude: {max_amplitude}")
                            
                            if max_amplitude < 100:
                                print("  ⚠️  Very quiet audio!")
                            
                            print("\n🔄 Transcribing...")
                            text = stt.transcribe(audio_data)
                            
                            if text:
                                print(f"\n✅ Transcription: \"{text}\"")
                                
                                print("\n🤖 Getting response...")
                                response = llm.generate(text)
                                print(f"💬 Response: \"{response}\"")
                                
                                print("\n🔊 Speaking...")
                                tts_audio = tts.synthesize(response)
                                audio.play_audio(tts_audio)
                                print("✓ Done")
                            else:
                                print("\n❌ No transcription result")
                        else:
                            print("\n❌ Empty speech buffer!")
                        
                        # Reset
                        print("\n" + "=" * 50)
                        print("🎤 Ready for next input...")
                        print("=" * 50 + "\n")
                        
                        speech_buffer = []
                        pre_speech_buffer = []
                        silence_count = 0
                        speaking = False
                        frame_count = 0
                        
                        # Resume
                        audio.start_recording()
                        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        audio.close()


if __name__ == "__main__":
    main()