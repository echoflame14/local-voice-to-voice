#!/usr/bin/env python3
"""
Script to trim voice reference audio to optimal length for voice cloning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import librosa
import soundfile as sf
import numpy as np
from configs.config import config


def trim_voice_sample(input_path: str, output_path: str = None, target_duration: float = 12.0):
    """
    Trim voice sample to target duration, selecting the best middle portion
    
    Args:
        input_path: Path to input voice file
        output_path: Path for output (None to overwrite original)
        target_duration: Target duration in seconds
    """
    print(f"🎵 Trimming voice sample: {input_path}")
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    original_duration = len(audio) / sr
    
    print(f"📏 Original duration: {original_duration:.2f} seconds")
    print(f"🎯 Target duration: {target_duration:.2f} seconds")
    
    if original_duration <= target_duration:
        print("✅ Audio is already shorter than target duration!")
        return input_path
    
    # Calculate samples needed
    target_samples = int(target_duration * sr)
    
    # Find the best section (avoid silence at start/end)
    # Calculate RMS energy in overlapping windows to find the most active section
    window_size = target_samples
    hop_size = sr // 4  # 0.25 second hops
    
    best_start = 0
    best_energy = 0
    
    for start in range(0, len(audio) - window_size + 1, hop_size):
        window = audio[start:start + window_size]
        # Calculate RMS energy
        energy = np.sqrt(np.mean(window ** 2))
        
        if energy > best_energy:
            best_energy = energy
            best_start = start
    
    # Extract the best section
    trimmed_audio = audio[best_start:best_start + target_samples]
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.05 * sr)  # 50ms fade
    
    # Fade in
    fade_in = np.linspace(0, 1, fade_samples)
    trimmed_audio[:fade_samples] *= fade_in
    
    # Fade out
    fade_out = np.linspace(1, 0, fade_samples)
    trimmed_audio[-fade_samples:] *= fade_out
    
    # Normalize
    trimmed_audio = trimmed_audio / np.max(np.abs(trimmed_audio)) * 0.9
    
    # Save
    if output_path is None:
        output_path = input_path
    
    sf.write(output_path, trimmed_audio, sr)
    
    print(f"✅ Trimmed audio saved to: {output_path}")
    print(f"📊 New duration: {len(trimmed_audio) / sr:.2f} seconds")
    print(f"🔊 Sample rate: {sr} Hz")
    print(f"🎚️  Peak level: {np.max(np.abs(trimmed_audio)):.3f}")
    
    return output_path


def main():
    """Main function to trim the configured voice reference"""
    voice_path = config.VOICE_REFERENCE_PATH
    
    if not Path(voice_path).exists():
        print(f"❌ Voice file not found: {voice_path}")
        return 1
    
    print("🎭 Voice Cloning Audio Optimizer")
    print("=" * 50)
    
    # Create backup first
    backup_path = voice_path.replace('.wav', '_backup.wav')
    if not Path(backup_path).exists():
        print(f"💾 Creating backup: {backup_path}")
        import shutil
        shutil.copy2(voice_path, backup_path)
    
    # Trim the voice sample
    optimized_path = voice_path.replace('.wav', '_optimized.wav')
    trimmed_path = trim_voice_sample(voice_path, optimized_path, target_duration=12.0)
    
    print(f"\n🎯 Voice optimization complete!")
    print(f"📁 Original: {voice_path}")
    print(f"💾 Backup: {backup_path}")
    print(f"✨ Optimized: {optimized_path}")
    
    # Ask user if they want to replace the original
    response = input(f"\n❓ Replace original voice file with optimized version? (y/N): ")
    if response.lower().startswith('y'):
        import shutil
        shutil.copy2(optimized_path, voice_path)
        print(f"✅ Original voice file updated with optimized version")
        
        # Clean up optimized file since we moved it
        Path(optimized_path).unlink()
    else:
        print(f"💡 To use the optimized version, update your config to point to: {optimized_path}")
    
    print(f"\n🚀 Test the voice cloning with: python test_voice_cloning.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 