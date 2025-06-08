#!/usr/bin/env python3
"""
Install Silero VAD for better voice activity detection
"""

import subprocess
import sys

print("🎯 Installing Silero VAD - a much better alternative to WebRTC VAD!")
print("\nThis will install:")
print("  - PyTorch (CPU version for fast inference)")
print("  - Silero VAD model")
print("\n" + "="*50 + "\n")

try:
    # Install PyTorch CPU version (faster for VAD)
    print("📦 Installing PyTorch CPU version...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    print("\n✅ PyTorch installed successfully!")
    
    # Test Silero VAD download
    print("\n📦 Downloading Silero VAD model...")
    test_code = """
import torch
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("✅ Silero VAD model downloaded successfully!")
"""
    
    subprocess.check_call([sys.executable, "-c", test_code])
    
    print("\n" + "="*50)
    print("✅ Installation complete!")
    print("\nSilero VAD advantages over WebRTC VAD:")
    print("  - Much better at filtering background noise")
    print("  - Trained on 6000+ hours of speech data")
    print("  - Processes chunks in <1ms on CPU")
    print("  - More accurate speech/silence detection")
    print("\nThe new VAD will automatically be used when you run the voice assistant.")
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Installation failed: {e}")
    print("\nFalling back to Energy-based VAD (still better than WebRTC!)")
    print("The assistant will work fine with the Energy-based VAD.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    print("\nYou can still use the Energy-based VAD as a fallback.")