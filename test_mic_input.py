#!/usr/bin/env python3
"""
Quick test to verify microphone input is working with the current configuration
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio import AudioStreamManager
from configs.config import SAMPLE_RATE, CHUNK_SIZE, INPUT_MODE, VAD_AGGRESSIVENESS


def main():
    print("🎤 Quick Microphone Test")
    print("=" * 60)
    
    print(f"\n📍 Current Configuration:")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Chunk Size: {CHUNK_SIZE} samples")
    print(f"   Input Mode: {INPUT_MODE}")
    if INPUT_MODE == "vad":
        print(f"   VAD Aggressiveness: {VAD_AGGRESSIVENESS}")
    
    print("\n🔧 Initializing AudioStreamManager...")
    
    try:
        # Initialize with verbose output
        manager = AudioStreamManager(
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
            enable_performance_logging=True,
            list_devices_on_init=True,  # This will list all devices
            input_mode=INPUT_MODE
        )
        
        print("\n✅ AudioStreamManager initialized successfully!")
        
        # Test raw audio input (bypass VAD)
        print("\n📊 Testing RAW audio input (bypassing VAD)...")
        print("🎤 Recording for 5 seconds... Make some noise!")
        
        audio_chunks = []
        chunk_count = 0
        max_amplitude = 0
        
        def raw_callback(audio_chunk):
            nonlocal chunk_count, max_amplitude
            chunk_count += 1
            amp = np.abs(audio_chunk).max()
            if amp > max_amplitude:
                max_amplitude = amp
            
            # Show level meter
            level = int(amp * 50)
            bar = "█" * level + "-" * (50 - level)
            print(f"\r📊 [{bar}] Amplitude: {amp:.3f} | Chunks: {chunk_count}", end="", flush=True)
            
            audio_chunks.append(audio_chunk)
        
        # Temporarily bypass VAD for raw input test
        original_mode = manager.input_manager.input_mode
        manager.input_manager.input_mode = "push_to_talk"
        manager.input_manager._push_to_talk_pressed = True  # Force capture
        
        manager.start_input_stream(callback=raw_callback)
        
        start_time = time.time()
        while time.time() - start_time < 5:
            time.sleep(0.1)
        
        manager.stop_input_stream()
        
        # Restore original mode
        manager.input_manager.input_mode = original_mode
        manager.input_manager._push_to_talk_pressed = False
        
        print(f"\n\n📊 Raw Input Test Results:")
        print(f"   Total chunks received: {chunk_count}")
        print(f"   Max amplitude: {max_amplitude:.3f}")
        print(f"   Average chunk rate: {chunk_count/5:.1f} chunks/second")
        
        if chunk_count == 0:
            print("\n❌ NO AUDIO CHUNKS RECEIVED!")
            print("   - Check if microphone is connected")
            print("   - Check system permissions")
            print("   - Try running as administrator")
        elif max_amplitude < 0.001:
            print("\n⚠️  Audio is being received but it's silent!")
            print("   - Check if microphone is muted")
            print("   - Increase microphone gain in system settings")
            print("   - Try speaking louder")
        else:
            print("\n✅ Microphone input is working!")
            
            # Now test with VAD if in VAD mode
            if INPUT_MODE == "vad":
                print("\n📊 Testing VAD detection...")
                print("🎤 Speak clearly for 5 seconds...")
                
                vad_chunks = []
                vad_detected = False
                
                def vad_callback(audio_chunk):
                    nonlocal vad_detected
                    vad_detected = True
                    vad_chunks.append(audio_chunk)
                    print(f"\n✅ VAD detected speech! Chunk size: {len(audio_chunk)}")
                
                manager.input_manager.input_mode = "vad"
                manager.start_input_stream(callback=vad_callback)
                
                start_time = time.time()
                while time.time() - start_time < 5:
                    time.sleep(0.1)
                
                manager.stop_input_stream()
                
                if vad_detected:
                    print(f"\n✅ VAD is working! Detected {len(vad_chunks)} speech chunks")
                else:
                    print("\n⚠️  VAD didn't detect any speech!")
                    print("   - Try speaking louder and clearer")
                    print("   - Reduce VAD aggressiveness: --vad-aggressiveness 0")
                    print("   - Try push-to-talk mode: --input-mode push_to_talk")
        
        manager.cleanup()
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting tips:")
        print("1. Run the debug tool for more details:")
        print("   python debug_microphone.py")
        print("\n2. List audio devices:")
        print("   python fix_audio_device.py")
        print("\n3. Try specifying a device:")
        print("   python main.py --input-device <number>")
        print("\n4. Check Windows audio settings")
        print("5. Restart your computer to reset audio subsystem")


if __name__ == "__main__":
    main()