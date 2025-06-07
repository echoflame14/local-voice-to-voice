#!/usr/bin/env python3
"""
Audio debugging script for Windows
"""
import pyaudio
import numpy as np
import time


def list_audio_devices():
    """List all audio devices"""
    p = pyaudio.PyAudio()
    
    print("=" * 60)
    print("AUDIO DEVICES")
    print("=" * 60)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"\nDevice {i}: {info['name']}")
        print(f"  - Max input channels: {info['maxInputChannels']}")
        print(f"  - Max output channels: {info['maxOutputChannels']}")
        print(f"  - Default sample rate: {info['defaultSampleRate']}")
        print(f"  - Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
        
        if info['maxInputChannels'] > 0:
            print("  ✓ Can be used for RECORDING")
        if info['maxOutputChannels'] > 0:
            print("  ✓ Can be used for PLAYBACK")
    
    # Get defaults
    try:
        default_input = p.get_default_input_device_info()
        print(f"\n🎤 Default INPUT device: {default_input['name']} (Device {default_input['index']})")
    except:
        print("\n❌ No default input device found!")
        
    try:
        default_output = p.get_default_output_device_info()
        print(f"🔊 Default OUTPUT device: {default_output['name']} (Device {default_output['index']})")
    except:
        print("❌ No default output device found!")
    
    p.terminate()
    print("=" * 60)


def test_recording(duration=3):
    """Test recording audio"""
    print(f"\n🎤 Testing recording for {duration} seconds...")
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=480
        )
        
        print("Recording... Speak now!")
        frames = []
        
        for _ in range(0, int(16000 / 480 * duration)):
            data = stream.read(480, exception_on_overflow=False)
            frames.append(data)
            
            # Show level meter
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_chunk**2))
            bars = int(energy / 500)
            print(f"\r[{'#' * bars:<20}] Level: {energy:.0f}", end="", flush=True)
        
        print("\n✓ Recording complete!")
        
        stream.stop_stream()
        stream.close()
        
        # Check if we got audio
        all_data = b''.join(frames)
        audio_array = np.frombuffer(all_data, dtype=np.int16)
        max_level = np.max(np.abs(audio_array))
        
        if max_level < 100:
            print("⚠️  Very low audio level detected. Check microphone.")
        else:
            print(f"✓ Good audio level: {max_level}")
            
    except Exception as e:
        print(f"❌ Recording failed: {e}")
    finally:
        p.terminate()


def test_playback():
    """Test audio playback"""
    print("\n🔊 Testing playback...")
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            output=True,
            frames_per_buffer=480
        )
        
        print("Playing test tone...")
        
        # Generate 1 second tone
        duration = 1.0
        frequency = 440  # A4 note
        sample_rate = 16000
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = np.sin(2 * np.pi * frequency * t)
        tone = (tone * 32767 * 0.3).astype(np.int16)  # 30% volume
        
        stream.write(tone.tobytes())
        
        print("✓ Playback complete! Did you hear the tone?")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"❌ Playback failed: {e}")
    finally:
        p.terminate()


def main():
    print("🎨 Elegant Chatbot - Audio Diagnostics")
    print("=" * 60)
    
    # List devices
    list_audio_devices()
    
    # Test recording
    input("\nPress Enter to test RECORDING...")
    test_recording()
    
    # Test playback
    input("\nPress Enter to test PLAYBACK...")
    test_playback()
    
    print("\n✅ Audio diagnostics complete!")
    print("\nIf audio is working here but not in the main app:")
    print("1. Try using main_simple.py instead of main.py")
    print("2. Check if Windows Defender is blocking Python")
    print("3. Run as Administrator")
    print("4. Check audio privacy settings in Windows")


if __name__ == "__main__":
    main()