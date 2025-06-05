#!/usr/bin/env python3
"""Quick test script for VAD functionality"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio.input_manager import InputManager
import time
import numpy as np

def test_vad():
    """Test VAD functionality"""
    print("🧪 Testing VAD functionality...")
    
    # Create input manager in VAD mode
    input_manager = InputManager(
        input_mode="vad",
        vad_aggressiveness=2,
        vad_speech_threshold=0.6,
        vad_silence_threshold=0.4,
        sample_rate=16000
    )
    
    print("✅ InputManager created successfully")
    print(f"VAD frame size: {input_manager.vad.frame_size}")
    print(f"Speech threshold: {input_manager.vad.speech_threshold}")
    print(f"Silence threshold: {input_manager.vad.silence_threshold}")
    
    # Test with silence (zeros)
    silence_frame = np.zeros(480, dtype=np.int16)
    result = input_manager.process_audio(silence_frame)
    print(f"Silence test: {result} (should be False)")
    
    # Test with noise
    noise_frame = np.random.randint(-1000, 1000, 480, dtype=np.int16)
    result = input_manager.process_audio(noise_frame)
    print(f"Noise test: {result}")
    
    # Test with louder noise (simulating speech)
    loud_frame = np.random.randint(-5000, 5000, 480, dtype=np.int16)
    result = input_manager.process_audio(loud_frame)
    print(f"Loud noise test: {result}")
    
    input_manager.stop()
    print("🏁 VAD test complete")

if __name__ == "__main__":
    test_vad() 