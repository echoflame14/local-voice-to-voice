#!/usr/bin/env python3
"""
Script to convert JR.mp4 to WAV and trim to exactly 9 seconds
"""

from moviepy.editor import VideoFileClip
import numpy as np
import soundfile as sf
import os

def convert_mp4_to_wav(input_path, output_path):
    """
    Convert MP4 to WAV using moviepy
    """
    print(f"🎬 Converting {input_path} to WAV...")
    
    try:
        # Load video file
        video = VideoFileClip(input_path)
        
        # Extract audio
        audio = video.audio
        
        # Write to WAV file
        audio.write_audiofile(output_path, verbose=False, logger=None)
        
        # Close to free memory
        audio.close()
        video.close()
        
        print(f"✅ Converted to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting: {e}")
        return False

def trim_to_duration_moviepy(input_path, output_path, target_duration=9):
    """
    Trim audio to exactly N seconds
    """
    print(f"✂️  Trimming {input_path} to {target_duration} seconds...")
    
    try:
        # Load audio file
        from moviepy.editor import AudioFileClip
        audio = AudioFileClip(input_path)
        
        original_duration = audio.duration
        print(f"📏 Original duration: {original_duration:.2f} seconds")
        print(f"🎯 Target duration: {target_duration} seconds")
        
        if original_duration <= target_duration:
            print(f"✅ Audio is already shorter than {target_duration} seconds, keeping as is!")
            # Just save it to the output path
            audio.write_audiofile(output_path, verbose=False, logger=None)
        else:
            # Trim to target duration (keep the first N seconds)
            trimmed_audio = audio.subclip(0, target_duration)
            
            # Write trimmed audio
            trimmed_audio.write_audiofile(output_path, verbose=False, logger=None)
            trimmed_audio.close()
        
        # Close to free memory
        audio.close()
        
        print(f"✅ Audio saved to: {output_path}")
        print(f"📊 Final duration: {min(original_duration, target_duration):.2f} seconds")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error trimming: {e}")
        return None

def main():
    # Paths
    mp4_path = "voices/asmr.mp4"
    wav_path = "voices/asmr.wav"
    final_path = "voices/asmr.wav"
    
    print("🎭 Audio Conversion and Trimming Tool")
    print("=" * 50)
    
    # Check if MP4 exists
    if not os.path.exists(mp4_path):
        print(f"❌ MP4 file not found: {mp4_path}")
        return
    
    # Step 1: Convert MP4 to WAV
    print("\n📋 Step 1: Converting MP4 to WAV")
    if convert_mp4_to_wav(mp4_path, wav_path):
        print("✅ Conversion successful!")
    else:
        print("❌ Conversion failed!")
        return
    
    # Step 2: Trim to 9 seconds
    # print("\n📋 Step 2: Trimming to 9 seconds")
    # result = trim_to_duration_moviepy(wav_path, final_path, 9)
    
    # if result:
    #     print(f"\n🎯 Process complete!")
    #     print(f"📁 Original MP4: {mp4_path}")
    #     print(f"🔄 Converted WAV: {wav_path}")
    #     print(f"✨ Final 9-second WAV: {final_path}")
        
    #     # Clean up intermediate file
    #     if os.path.exists(wav_path):
    #         os.remove(wav_path)
    #         print(f"🧹 Cleaned up intermediate file: {wav_path}")
    # else:
    #     print("❌ Trimming failed!")

if __name__ == "__main__":
    main() 