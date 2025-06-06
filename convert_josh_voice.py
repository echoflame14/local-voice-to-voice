#!/usr/bin/env python3
"""
Convert josh.m4a to josh.wav for Chatterbox TTS
"""

from moviepy.editor import AudioFileClip
import os

def convert_audio_to_wav(input_path, output_path):
    """Convert any audio file to WAV format"""
    print(f"🎵 Converting {input_path} to WAV...")
    
    try:
        # Load audio file
        audio = AudioFileClip(input_path)
        
        # Get duration
        duration = audio.duration
        print(f"📏 Duration: {duration:.2f} seconds")
        
        # Recommended duration for voice cloning is 10-30 seconds
        if duration < 10:
            print(f"⚠️  Warning: Audio is only {duration:.2f} seconds. Recommended: 10-30 seconds for best results.")
        elif duration > 30:
            print(f"ℹ️  Note: Audio is {duration:.2f} seconds. Consider trimming to 30 seconds or less.")
            
        # Write to WAV file (16kHz mono for Chatterbox)
        audio.write_audiofile(
            output_path, 
            fps=16000,  # 16kHz sample rate
            verbose=False, 
            logger=None
        )
        
        # Close to free memory
        audio.close()
        
        print(f"✅ Converted to: {output_path}")
        print(f"🎯 Ready for Chatterbox TTS!")
        return True
        
    except Exception as e:
        print(f"❌ Error converting: {e}")
        return False

def main():
    # Paths
    input_path = "voices/josh.m4a"
    output_path = "voices/josh.wav"
    
    print("🎤 Voice File Converter for Chatterbox TTS")
    print("=" * 50)
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Convert to WAV
    if convert_audio_to_wav(input_path, output_path):
        print(f"\n✨ Success! Your voice file is ready.")
        print(f"📁 Input: {input_path}")
        print(f"📁 Output: {output_path}")
        print(f"\n🚀 You can now run: python main.py --use-gemini")
    else:
        print("\n❌ Conversion failed!")

if __name__ == "__main__":
    main()