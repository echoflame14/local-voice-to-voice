#!/usr/bin/env python3
"""Test microphone levels and find the optimal device"""

import pyaudio
import numpy as np
import time
import sys

def test_device(p, device_index, duration=3):
    """Test a specific audio device"""
    try:
        # Get device info
        info = p.get_device_info_by_index(device_index)
        
        # Only test input devices
        if info['maxInputChannels'] == 0:
            return None, 0
            
        print(f"\n🎤 Testing Device {device_index}: {info['name']}")
        print(f"   Channels: {info['maxInputChannels']}")
        print(f"   Default Sample Rate: {info['defaultSampleRate']}")
        
        # Try to open stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=min(1, info['maxInputChannels']),  # Use mono
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=480
        )
        
        print(f"   Recording for {duration} seconds... Make some noise!")
        
        max_amplitude = 0
        amplitudes = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                data = stream.read(480, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                amplitude = np.max(np.abs(audio_data))
                amplitudes.append(amplitude)
                max_amplitude = max(max_amplitude, amplitude)
                
                # Visual feedback
                bars = int(amplitude * 50)
                print(f"\r   Level: [{'█' * bars}{'-' * (50 - bars)}] {amplitude:.3f}", end='', flush=True)
                
            except Exception as e:
                print(f"\n   Error reading: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        avg_amplitude = np.mean(amplitudes) if amplitudes else 0
        print(f"\n   ✅ Max amplitude: {max_amplitude:.3f}, Average: {avg_amplitude:.3f}")
        
        return info['name'], max_amplitude
        
    except Exception as e:
        print(f"\n   ❌ Failed to test device {device_index}: {e}")
        return None, 0

def main():
    print("🎙️ Microphone Level Tester")
    print("=" * 60)
    
    p = pyaudio.PyAudio()
    
    # Test all devices
    device_results = []
    
    for i in range(p.get_device_count()):
        name, max_level = test_device(p, i, duration=2)
        if name and max_level > 0:
            device_results.append((i, name, max_level))
    
    # Show results
    print("\n\n📊 Results Summary:")
    print("=" * 60)
    
    if not device_results:
        print("❌ No working microphones found!")
    else:
        # Sort by max level
        device_results.sort(key=lambda x: x[2], reverse=True)
        
        print("Devices ranked by signal strength:")
        for idx, (dev_id, name, level) in enumerate(device_results):
            status = "🏆 BEST" if idx == 0 else "✅ Good" if level > 0.1 else "⚠️  Low"
            print(f"{status} Device {dev_id}: {name[:40]:40} Level: {level:.3f}")
        
        # Recommendation
        best_device = device_results[0]
        print(f"\n💡 Recommended device: {best_device[0]} ({best_device[1]})")
        print(f"\nTo use this device, run:")
        print(f"   python main.py --input-device {best_device[0]}")
        
        if best_device[2] < 0.1:
            print("\n⚠️  Warning: Even the best device has low signal levels.")
            print("   Consider:")
            print("   - Moving closer to the microphone")
            print("   - Increasing microphone gain in Windows settings")
            print("   - Using a different microphone")
    
    p.terminate()

if __name__ == "__main__":
    main()