#!/usr/bin/env python3
"""
Convert josh.m4a to josh.wav using pydub
"""

try:
    from pydub import AudioSegment
    import os
    
    print("🎤 Converting josh.m4a to josh.wav...")
    
    # Convert m4a to wav
    audio = AudioSegment.from_file("voices/josh.m4a", "m4a")
    
    # Set to mono and 16kHz for Chatterbox
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    
    # Export as WAV
    audio.export("voices/josh.wav", format="wav")
    
    duration = len(audio) / 1000.0
    print(f"✅ Converted successfully!")
    print(f"📏 Duration: {duration:.2f} seconds")
    
    if duration < 10:
        print(f"⚠️  Warning: Audio is only {duration:.2f} seconds. Recommended: 10-30 seconds for best voice cloning results.")
    
    print(f"🚀 You can now run: python main.py --use-gemini")
    
except ImportError:
    print("❌ pydub not installed. Trying with ffmpeg directly...")
    import subprocess
    import os
    
    if os.path.exists("voices/josh.m4a"):
        cmd = [
            "ffmpeg", "-i", "voices/josh.m4a",
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # Mono
            "-y",            # Overwrite
            "voices/josh.wav"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Converted successfully using ffmpeg!")
            print("🚀 You can now run: python main.py --use-gemini")
        except subprocess.CalledProcessError:
            print("❌ ffmpeg conversion failed. Please install ffmpeg or pydub.")
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html")
    else:
        print("❌ voices/josh.m4a not found!")
        
except Exception as e:
    print(f"❌ Error: {e}")