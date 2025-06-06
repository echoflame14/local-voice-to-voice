#!/usr/bin/env python3
"""
Convert JR.mp4 to JR.wav for voice cloning
"""

from pydub import AudioSegment
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np

def convert_mp4_to_wav():
    """Convert JR.mp4 to JR.wav with proper audio settings for voice cloning"""
    
    # File paths
    input_file = Path("voices/JR.mp4")
    output_file = Path("voices/JR.wav")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        return False
    
    try:
        print(f"Loading audio from {input_file}...")
        # Load MP4 file and extract audio
        audio = AudioSegment.from_file(str(input_file), format="mp4")
        
        print(f"Original audio info:")
        print(f"  - Duration: {len(audio)/1000:.2f} seconds")
        print(f"  - Sample rate: {audio.frame_rate} Hz")
        print(f"  - Channels: {audio.channels}")
        
        # Convert to mono if needed
        if audio.channels > 1:
            print("Converting to mono...")
            audio = audio.set_channels(1)
        
        # Set sample rate to 16kHz for voice cloning
        target_sr = 16000
        if audio.frame_rate != target_sr:
            print(f"Resampling from {audio.frame_rate} Hz to {target_sr} Hz...")
            audio = audio.set_frame_rate(target_sr)
        
        # Export as WAV
        print(f"Saving to {output_file}...")
        audio.export(str(output_file), format="wav")
        
        print(f"✅ Successfully converted {input_file} to {output_file}")
        print(f"Output file info:")
        print(f"  - Sample rate: {target_sr} Hz")
        print(f"  - Duration: {len(audio)/1000:.2f} seconds")
        print(f"  - Channels: 1 (mono)")
        print(f"  - Format: WAV")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return False

if __name__ == "__main__":
    convert_mp4_to_wav() 