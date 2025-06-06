#!/usr/bin/env python3
"""
Comprehensive microphone debugging tool for the voice chatbot
Helps diagnose why the microphone isn't being detected
"""

import pyaudio
import numpy as np
import time
import sys
import wave
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import SAMPLE_RATE, CHUNK_SIZE, INPUT_MODE
from src.audio.vad import VoiceActivityDetector


def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🎤 {text}")
    print('='*60)


def check_pyaudio_installation():
    """Check if PyAudio is properly installed"""
    print_header("Checking PyAudio Installation")
    
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        version = pyaudio.get_portaudio_version()
        print(f"✅ PyAudio is installed")
        print(f"   PortAudio version: {version}")
        pa.terminate()
        return True
    except Exception as e:
        print(f"❌ PyAudio installation error: {e}")
        print("\nTo fix:")
        print("  pip uninstall pyaudio")
        print("  pip install pyaudio")
        return False


def list_all_audio_devices():
    """List all available audio devices with detailed info"""
    print_header("Audio Device Information")
    
    try:
        pa = pyaudio.PyAudio()
        
        # Get default devices
        try:
            default_input = pa.get_default_input_device_info()
            print(f"🎯 Default Input Device: {default_input['name']} (Index: {default_input['index']})")
        except:
            print("⚠️  No default input device found!")
            default_input = None
            
        try:
            default_output = pa.get_default_output_device_info()
            print(f"🔊 Default Output Device: {default_output['name']} (Index: {default_output['index']})")
        except:
            print("⚠️  No default output device found!")
            default_output = None
        
        print(f"\n📋 Total devices found: {pa.get_device_count()}")
        
        input_devices = []
        output_devices = []
        
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                
                # Categorize device
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info))
                if info['maxOutputChannels'] > 0:
                    output_devices.append((i, info))
                    
            except Exception as e:
                print(f"⚠️  Error getting info for device {i}: {e}")
        
        # Print input devices
        print(f"\n🎤 INPUT DEVICES ({len(input_devices)} found):")
        print("-" * 50)
        for idx, info in input_devices:
            print(f"\nDevice {idx}: {info['name']}")
            print(f"  Channels: {info['maxInputChannels']}")
            print(f"  Sample Rate: {info['defaultSampleRate']} Hz")
            print(f"  Host API: {pa.get_host_api_info_by_index(info['hostApi'])['name']}")
            if default_input and idx == default_input['index']:
                print("  ⭐ DEFAULT INPUT")
        
        # Print output devices
        print(f"\n🔊 OUTPUT DEVICES ({len(output_devices)} found):")
        print("-" * 50)
        for idx, info in output_devices:
            print(f"\nDevice {idx}: {info['name']}")
            print(f"  Channels: {info['maxOutputChannels']}")
            print(f"  Sample Rate: {info['defaultSampleRate']} Hz")
            print(f"  Host API: {pa.get_host_api_info_by_index(info['hostApi'])['name']}")
            if default_output and idx == default_output['index']:
                print("  ⭐ DEFAULT OUTPUT")
        
        pa.terminate()
        return len(input_devices) > 0, input_devices
        
    except Exception as e:
        print(f"❌ Error listing devices: {e}")
        return False, []


def test_microphone_input(device_index=None, duration=3):
    """Test microphone input by recording audio"""
    print_header(f"Testing Microphone Input (Device: {device_index if device_index is not None else 'default'})")
    
    pa = pyaudio.PyAudio()
    
    try:
        # Try to open stream
        print(f"📍 Opening audio stream...")
        print(f"   Sample Rate: {SAMPLE_RATE} Hz")
        print(f"   Chunk Size: {CHUNK_SIZE}")
        print(f"   Format: Float32")
        print(f"   Channels: 1 (mono)")
        
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print(f"✅ Stream opened successfully!")
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        
        frames = []
        start_time = time.time()
        max_amplitude = 0
        
        while time.time() - start_time < duration:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                frames.append(audio_chunk)
                
                # Track max amplitude
                chunk_max = np.abs(audio_chunk).max()
                if chunk_max > max_amplitude:
                    max_amplitude = chunk_max
                
                # Show level meter
                level = int(chunk_max * 50)
                bar = "█" * level + "-" * (50 - level)
                print(f"\r📊 Level: [{bar}] {chunk_max:.3f}", end="", flush=True)
                
            except Exception as e:
                print(f"\n⚠️  Error reading audio: {e}")
        
        print(f"\n\n✅ Recording complete!")
        print(f"   Captured {len(frames)} chunks")
        print(f"   Max amplitude: {max_amplitude:.3f}")
        
        # Check if we got any audio
        audio_data = np.concatenate(frames)
        mean_amplitude = np.abs(audio_data).mean()
        
        print(f"   Mean amplitude: {mean_amplitude:.3f}")
        print(f"   Total samples: {len(audio_data)}")
        
        if max_amplitude < 0.001:
            print("\n⚠️  WARNING: No audio detected! Possible issues:")
            print("   - Microphone is muted")
            print("   - Wrong device selected")
            print("   - Permission issues")
            print("   - Hardware problem")
        elif max_amplitude < 0.01:
            print("\n⚠️  WARNING: Very low audio level detected!")
            print("   - Microphone gain might be too low")
            print("   - Speak louder or move closer to mic")
        else:
            print("\n✅ Audio levels look good!")
        
        # Save recording for inspection
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Convert float32 to int16 for WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"\n💾 Recording saved to: {temp_path}")
        print("   You can play this file to verify the recording")
        
        stream.stop_stream()
        stream.close()
        
        return True, max_amplitude > 0.001
        
    except Exception as e:
        print(f"\n❌ Error testing microphone: {e}")
        print("\nPossible causes:")
        print("  - Device doesn't support the requested format")
        print("  - Device is in use by another application")
        print("  - Permission denied (check system microphone permissions)")
        return False, False
        
    finally:
        pa.terminate()


def test_vad_functionality():
    """Test Voice Activity Detection"""
    print_header("Testing Voice Activity Detection (VAD)")
    
    try:
        print("🔧 Initializing VAD...")
        vad = VoiceActivityDetector(
            sample_rate=SAMPLE_RATE,
            aggressiveness=1,
            speech_threshold=0.3,
            silence_threshold=0.8
        )
        print("✅ VAD initialized successfully")
        
        print("\n📍 VAD Configuration:")
        print(f"   Frame size: {vad.frame_size} samples")
        print(f"   Frame duration: {vad.frame_duration_ms}ms")
        print(f"   Speech threshold: {vad.speech_threshold}")
        print(f"   Silence threshold: {vad.silence_threshold}")
        
        # Test with silence
        silence = np.zeros(vad.frame_size, dtype=np.int16)
        is_speech = vad._is_speech(silence)
        print(f"\n🔇 Silence test: {'❌ FAIL (detected as speech)' if is_speech else '✅ PASS (correctly identified)'}")
        
        # Test with noise
        noise = (np.random.random(vad.frame_size) * 1000).astype(np.int16)
        is_speech = vad._is_speech(noise)
        print(f"🔊 Noise test: {'✅ Detected as speech' if is_speech else '❌ Not detected as speech'}")
        
        return True
        
    except Exception as e:
        print(f"❌ VAD test failed: {e}")
        return False


def test_stream_manager():
    """Test the AudioStreamManager from the project"""
    print_header("Testing AudioStreamManager")
    
    try:
        from src.audio import AudioStreamManager
        
        print("🔧 Initializing AudioStreamManager...")
        manager = AudioStreamManager(
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
            enable_performance_logging=True,
            list_devices_on_init=True,  # This will list devices
            input_mode=INPUT_MODE
        )
        
        print("✅ AudioStreamManager initialized successfully")
        
        # Test input stream
        print("\n📍 Testing input stream...")
        
        received_audio = []
        
        def audio_callback(audio_chunk):
            received_audio.append(audio_chunk)
            max_amp = np.abs(audio_chunk).max()
            print(f"   Received chunk: {len(audio_chunk)} samples, max amplitude: {max_amp:.3f}")
        
        manager.start_input_stream(callback=audio_callback)
        
        print("🎤 Listening for 3 seconds... Speak now!")
        time.sleep(3)
        
        manager.stop_input_stream()
        
        if received_audio:
            print(f"\n✅ Received {len(received_audio)} audio chunks")
        else:
            print("\n⚠️  No audio chunks received!")
            print("   - Check if VAD is too aggressive")
            print("   - Try speaking louder")
            print("   - Check microphone selection")
        
        manager.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ AudioStreamManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def suggest_fixes(has_input_devices, devices_found):
    """Suggest fixes based on test results"""
    print_header("Suggested Fixes")
    
    if not has_input_devices:
        print("❌ NO INPUT DEVICES FOUND!")
        print("\nPossible solutions:")
        print("1. Check if your microphone is plugged in")
        print("2. Check Windows Sound Settings:")
        print("   - Right-click speaker icon > Sound settings")
        print("   - Make sure a microphone is selected")
        print("3. Update audio drivers")
        print("4. Try running as administrator")
        print("5. Check if Windows Privacy settings allow microphone access")
        
    else:
        print("✅ Input devices found. To use a specific device:")
        print("\n1. Note the device index from the list above")
        print("2. Run the voice assistant with:")
        print("   python main.py --input-device <device_number>")
        print("\n3. Or modify src/audio/stream_manager.py to set a default device")
        print("\n4. Check current audio settings:")
        print(f"   - Current input mode: {INPUT_MODE}")
        print(f"   - Sample rate: {SAMPLE_RATE} Hz")
        print(f"   - Chunk size: {CHUNK_SIZE}")
        
        print("\n5. If mic still doesn't work:")
        print("   - Try push-to-talk mode: python main.py --input-mode push_to_talk")
        print("   - Reduce VAD aggressiveness: python main.py --vad-aggressiveness 0")
        print("   - Check if another app is using the microphone")
        print("   - Restart the computer to reset audio subsystem")


def main():
    """Run all microphone diagnostics"""
    print("🎤 Microphone Diagnostic Tool for Voice Chatbot")
    print("=" * 60)
    
    # Track test results
    all_tests_passed = True
    
    # 1. Check PyAudio
    if not check_pyaudio_installation():
        return 1
    
    # 2. List devices
    has_devices, input_devices = list_all_audio_devices()
    if not has_devices:
        all_tests_passed = False
    
    # 3. Test default microphone
    if has_devices:
        success, has_audio = test_microphone_input(None, 3)
        if not success or not has_audio:
            all_tests_passed = False
            
            # Try other devices if default failed
            if not has_audio and len(input_devices) > 1:
                print("\n🔄 Default device failed. Testing other devices...")
                for idx, info in input_devices[:3]:  # Test up to 3 devices
                    print(f"\n🔄 Testing device {idx}: {info['name']}")
                    success, has_audio = test_microphone_input(idx, 2)
                    if success and has_audio:
                        print(f"\n✅ Device {idx} works! Use: python main.py --input-device {idx}")
                        break
    
    # 4. Test VAD
    if not test_vad_functionality():
        all_tests_passed = False
    
    # 5. Test StreamManager
    if has_devices:
        if not test_stream_manager():
            all_tests_passed = False
    
    # 6. Provide suggestions
    suggest_fixes(has_devices, input_devices)
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ All tests passed! Your microphone should work.")
    else:
        print("⚠️  Some tests failed. See suggestions above.")
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())