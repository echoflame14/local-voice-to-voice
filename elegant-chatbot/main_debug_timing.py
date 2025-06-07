#!/usr/bin/env python3
"""
Elegant Chatbot - Debug version with detailed timing
Shows exactly where the lag is coming from
"""
import sys
import time
import numpy as np
from pathlib import Path

print(f"[STARTUP] Script started at {time.time():.3f}")
start_time = time.time()

sys.path.insert(0, str(Path(__file__).parent))

# Time imports
import_start = time.time()
from config import config
from core.audio_simple import SimpleAudioSystem
from core.stt_faster import FasterWhisperSTT
from core.llm import LLMClient
from core.tts import SimpleTTS
import_time = time.time() - import_start
print(f"[STARTUP] Imports took {import_time:.3f}s")


def main():
    print("=" * 50)
    print("🎨 Elegant Chatbot (Timing Debug Version)")
    print("=" * 50)
    
    # Validate config
    val_start = time.time()
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    val_time = time.time() - val_start
    print(f"[TIMING] Config validation took {val_time:.3f}s")
    
    # Initialize components
    print("\n[TIMING] Starting component initialization...")
    init_start = time.time()
    
    # Audio
    audio_start = time.time()
    print("Initializing audio...")
    audio = SimpleAudioSystem(config)
    audio_time = time.time() - audio_start
    print(f"[TIMING] Audio init took {audio_time:.3f}s")
    
    # STT
    stt_start = time.time()
    print("Loading faster-whisper model...")
    stt = FasterWhisperSTT(config)
    stt.load()
    stt_time = time.time() - stt_start
    print(f"[TIMING] STT init took {stt_time:.3f}s")
    
    # LLM
    llm_start = time.time()
    print("Initializing LLM...")
    llm = LLMClient(config)
    
    # Separate timing for warm-up
    warmup_start = time.time()
    llm.warm_up()
    warmup_time = time.time() - warmup_start
    
    llm_time = time.time() - llm_start
    print(f"[TIMING] LLM init took {llm_time:.3f}s (warm-up: {warmup_time:.3f}s)")
    
    # TTS
    tts_start = time.time()
    print("Initializing TTS...")
    tts = SimpleTTS(config)
    tts_time = time.time() - tts_start
    print(f"[TIMING] TTS init took {tts_time:.3f}s")
    
    total_init = time.time() - init_start
    print(f"\n[TIMING] Total initialization: {total_init:.3f}s")
    print(f"[TIMING] Breakdown: audio={audio_time:.3f}s, stt={stt_time:.3f}s, llm={llm_time:.3f}s, tts={tts_time:.3f}s")
    
    # Start audio
    audio_start_time = time.time()
    try:
        audio.start()
    except Exception as e:
        print(f"\n❌ Audio initialization failed: {e}")
        return
    print(f"[TIMING] Audio start took {time.time() - audio_start_time:.3f}s")
    
    total_startup = time.time() - start_time
    print(f"\n[TIMING] Total startup time: {total_startup:.3f}s")
    print("=" * 50)
    
    print("\n🎤 Ready! Timing debug enabled.")
    print("   Press Ctrl+C to exit.\n")
    
    # Main loop
    audio.start_recording()
    speech_buffer = []
    pre_speech_buffer = []
    silence_count = 0
    speaking = False
    first_response = True
    
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
                    print(f"🎯 Listening... (started at {time.time():.3f})")
                    speaking = True
                    speech_buffer.extend(pre_speech_buffer)
                speech_buffer.append(chunk)
                silence_count = 0
            else:
                if speaking:
                    speech_buffer.append(chunk)
                    silence_count += 1
                    
                    # After 1.5 seconds of silence, process
                    if silence_count > 45:
                        print(f" Done! (at {time.time():.3f})")
                        
                        process_start = time.time()
                        
                        # Stop recording briefly
                        audio.stop_recording()
                        
                        # Process speech
                        if speech_buffer:
                            audio_data = np.concatenate(speech_buffer)
                            duration = len(audio_data) / 16000
                            print(f"📏 Captured {duration:.1f} seconds")
                            
                            # Transcribe
                            trans_start = time.time()
                            print(f"🔄 Transcribing... (at {time.time():.3f})")
                            text = stt.transcribe(audio_data)
                            trans_time = time.time() - trans_start
                            print(f"[TIMING] Transcription took {trans_time:.3f}s")
                            
                            if text:
                                print(f"\n👤 You: {text}")
                                
                                # Generate response
                                gen_start = time.time()
                                print(f"🤖 Thinking... (at {time.time():.3f})")
                                
                                if first_response:
                                    print("[DEBUG] This is the FIRST response - may include connection overhead")
                                    first_response = False
                                
                                response = llm.generate(text)
                                gen_time = time.time() - gen_start
                                print(f"[TIMING] LLM generation took {gen_time:.3f}s")
                                print(f"💬 Assistant: {response}")
                                
                                # Speak response
                                speak_start = time.time()
                                print("🔊 Speaking...")
                                tts_audio = tts.synthesize(response)
                                audio.play_audio(tts_audio)
                                speak_time = time.time() - speak_start
                                print(f"[TIMING] TTS + playback took {speak_time:.3f}s")
                                print("✓ Done\n")
                                
                                total_process = time.time() - process_start
                                print(f"[TIMING] Total processing time: {total_process:.3f}s")
                                print(f"[TIMING] Breakdown: trans={trans_time:.3f}s, llm={gen_time:.3f}s, tts={speak_time:.3f}s")
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