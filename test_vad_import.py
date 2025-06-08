#!/usr/bin/env python3
"""
Test if the new VAD can be imported
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing VAD import...\n")

# Test 1: Can we import the new VAD directly?
try:
    from src.audio.vad_silero import VoiceActivityDetector, EnergyBasedVAD
    print("✅ Successfully imported vad_silero.py")
    print("   - VoiceActivityDetector available")
    print("   - EnergyBasedVAD available")
except ImportError as e:
    print(f"❌ Failed to import vad_silero.py: {e}")

# Test 2: Check what VAD is being used by the main vad.py
print("\nChecking main VAD module...")
try:
    from src.audio import vad
    if hasattr(vad, 'USE_NEW_VAD'):
        if vad.USE_NEW_VAD:
            print("✅ New VAD is being used!")
        else:
            print("❌ Still using WebRTC VAD")
    else:
        print("⚠️  Cannot determine which VAD is being used")
except Exception as e:
    print(f"❌ Error importing main VAD: {e}")

# Test 3: Try to create a VAD instance
print("\nTrying to create VAD instance...")
try:
    from src.audio.vad import VoiceActivityDetector
    vad = VoiceActivityDetector(use_silero=False)  # Force energy-based VAD
    print(f"✅ Created VAD instance: {vad.backend_type if hasattr(vad, 'backend_type') else 'unknown type'}")
except Exception as e:
    print(f"❌ Failed to create VAD instance: {e}")
    import traceback
    traceback.print_exc()

print("\nIf you see errors above, the new VAD may not be loading correctly.")