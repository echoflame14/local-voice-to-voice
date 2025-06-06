#!/usr/bin/env python3
"""
Script to convert JR.mp4 to WAV and trim to exactly 9 seconds using pydub
"""

from pydub import AudioSegment
import os

def convert_and_trim_to_9_seconds(input_path, output_path):
    """
    Convert MP4 to WAV and trim to exactly 9 seconds using pydub
    """
    print(f"🎬 Processing {input_path}...")
    
    try:
        # Load audio from MP4
        audio = AudioSegment.from_file(input_path)
        
        original_duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        print(f"📏 Original duration: {original_duration:.2f} seconds")
        
        # Trim to 9 seconds (9000 milliseconds)
        target_duration_ms = 9 * 1000
        
        if len(audio) > target_duration_ms:
            print(f"✂️  Trimming to 9 seconds...")
            audio = audio[:target_duration_ms]
        else:
            print(f"✅ Audio is already {original_duration:.2f} seconds, keeping as is!")
        
        # Export as WAV
        audio.export(output_path, format="wav")
        
        final_duration = len(audio) / 1000.0
        print(f"✅ Audio saved to: {output_path}")
        print(f"📊 Final duration: {final_duration:.2f} seconds")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

def main():
    # Paths
    mp4_path = "voices/JR.mp4"
    output_path = "voices/JR_9sec.wav"
    
    print("🎭 Audio Conversion and Trimming Tool (Pydub)")
    print("=" * 50)
    
    # Check if MP4 exists
    if not os.path.exists(mp4_path):
        print(f"❌ MP4 file not found: {mp4_path}")
        return
    
    # Convert and trim in one step
    result = convert_and_trim_to_9_seconds(mp4_path, output_path)
    
    if result:
        print(f"\n🎯 Process complete!")
        print(f"📁 Original MP4: {mp4_path}")
        print(f"✨ Final 9-second WAV: {output_path}")
    else:
        print("❌ Process failed!")

if __name__ == "__main__":
    main() 