#!/usr/bin/env python3
"""
Fix audio device selection by listing and testing devices
"""

import pyaudio
import numpy as np
import time

def list_audio_devices():
    """List all available audio devices"""
    pa = pyaudio.PyAudio()
    
    print("🎵 Available Audio Devices:")
    print("=" * 60)
    
    default_input = None
    default_output = None
    
    try:
        default_input = pa.get_default_input_device_info()['index']
        default_output = pa.get_default_output_device_info()['index']
    except:
        pass
    
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        
        # Mark default devices
        is_default_input = i == default_input
        is_default_output = i == default_output
        
        print(f"\nDevice {i}: {info['name']}")
        print(f"  Host API: {pa.get_host_api_info_by_index(info['hostApi'])['name']}")
        print(f"  Input Channels: {info['maxInputChannels']}")
        print(f"  Output Channels: {info['maxOutputChannels']}")
        print(f"  Sample Rate: {info['defaultSampleRate']} Hz")
        
        if is_default_input:
            print("  ⭐ DEFAULT INPUT DEVICE")
        if is_default_output:
            print("  ⭐ DEFAULT OUTPUT DEVICE")
            
    pa.terminate()
    
    return default_output

def test_audio_output(device_index=None):
    """Test audio output with a simple tone"""
    pa = pyaudio.PyAudio()
    
    print(f"\n🔊 Testing audio output (device: {device_index if device_index is not None else 'default'})...")
    
    # Generate a 440Hz tone
    sample_rate = 16000
    duration = 0.5  # seconds
    frequency = 440  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    
    try:
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=1024
        )
        
        print("  Playing test tone (440Hz)...")
        stream.write(tone.tobytes())
        
        stream.stop_stream()
        stream.close()
        
        print("  ✅ Audio output test successful!")
        return True
        
    except Exception as e:
        print(f"  ❌ Audio output test failed: {e}")
        return False
        
    finally:
        pa.terminate()

def suggest_fix():
    """Suggest how to fix the audio device issue"""
    print("\n💡 To fix audio device selection in your voice assistant:")
    print("\n1. Run this script to identify your preferred output device number")
    print("2. Then run main.py with the device specified:")
    print("   python main.py --use-gemini --output-device <device_number>")
    print("\n3. Or set it in your environment/config:")
    print("   - Add to .env: OUTPUT_DEVICE_INDEX=<device_number>")
    print("   - Or modify configs/config.py to include the device index")

if __name__ == "__main__":
    print("🎤 Audio Device Diagnostic Tool")
    print("=" * 60)
    
    # List devices
    default_device = list_audio_devices()
    
    # Test default device
    if default_device is not None:
        test_audio_output(default_device)
    
    # Suggest fix
    suggest_fix()
    
    # Interactive device testing
    print("\n🔧 Interactive Device Testing")
    print("Enter a device number to test, or 'q' to quit:")
    
    while True:
        choice = input("\nDevice number (or 'q'): ").strip()
        
        if choice.lower() == 'q':
            break
            
        try:
            device_num = int(choice)
            test_audio_output(device_num)
        except ValueError:
            print("❌ Please enter a valid device number or 'q' to quit")