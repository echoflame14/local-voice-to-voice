#!/usr/bin/env python3
"""
Script to trim the last 3 seconds from JR_test.wav
"""

import librosa
import soundfile as sf
import numpy as np

def trim_last_seconds(input_path, output_path, seconds_to_trim=3):
    """
    Remove the last N seconds from an audio file
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        seconds_to_trim: Number of seconds to remove from the end
    """
    print(f"🎵 Loading audio: {input_path}")
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    original_duration = len(audio) / sr
    
    print(f"📏 Original duration: {original_duration:.2f} seconds")
    print(f"✂️  Trimming last {seconds_to_trim} seconds")
    
    if original_duration <= seconds_to_trim:
        print("❌ Audio is shorter than the amount to trim!")
        return None
    
    # Calculate how many samples to keep
    samples_to_trim = int(seconds_to_trim * sr)
    trimmed_audio = audio[:-samples_to_trim]
    
    new_duration = len(trimmed_audio) / sr
    
    # Apply fade out to avoid clicks
    fade_samples = int(0.05 * sr)  # 50ms fade
    if len(trimmed_audio) > fade_samples:
        fade_out = np.linspace(1, 0, fade_samples)
        trimmed_audio[-fade_samples:] *= fade_out
    
    # Save
    sf.write(output_path, trimmed_audio, sr)
    
    print(f"✅ Trimmed audio saved to: {output_path}")
    print(f"📊 New duration: {new_duration:.2f} seconds")
    print(f"🔊 Sample rate: {sr} Hz")
    
    return output_path

if __name__ == "__main__":
    input_file = "voices/JR_test.wav"
    output_file = "voices/JR_trimmed.wav"
    
    result = trim_last_seconds(input_file, output_file, 3)
    if result:
        print(f"\n🎯 Successfully trimmed audio!")
        print(f"📁 Original: {input_file}")
        print(f"✨ Trimmed: {output_file}") 