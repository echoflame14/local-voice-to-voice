#!/usr/bin/env python3
"""Debug script to analyze interrupt audio buffer issues"""

import numpy as np
import time
import matplotlib.pyplot as plt
from src.audio import AudioStreamManager
from src.pipeline import VoiceAssistant

def analyze_interrupt_scenario():
    """Analyze what happens to audio buffers during interrupts"""
    
    print("🔍 Analyzing Interrupt Audio Buffer Behavior")
    print("=" * 60)
    
    # Create mock audio manager
    audio_manager = AudioStreamManager(
        sample_rate=16000,
        chunk_size=480,
        enable_performance_logging=False,
        list_devices_on_init=False,
        input_mode="vad"
    )
    
    print("\n📊 Buffer States:")
    print(f"- Pre-buffer size: {audio_manager.pre_buffer.maxlen} frames")
    print(f"- Pre-buffer duration: {audio_manager.pre_buffer_duration}s")
    print(f"- VAD audio buffer: {len(audio_manager.vad_audio_buffer)} frames")
    print(f"- Is speech active: {audio_manager.is_speech_active}")
    
    # Simulate speech detection
    print("\n🎤 Simulating speech detection...")
    audio_manager.is_speech_active = True
    audio_manager.speech_start_time = time.time()
    
    # Add some fake audio to buffers
    for i in range(50):  # 1.5 seconds of audio
        fake_audio = np.random.randn(480).astype(np.float32) * 0.1
        audio_manager.pre_buffer.append(fake_audio)
        if i > 25:  # Add to VAD buffer for last 0.75 seconds
            audio_manager.vad_audio_buffer.append(fake_audio)
    
    print(f"✅ Pre-buffer frames: {len(audio_manager.pre_buffer)}")
    print(f"✅ VAD buffer frames: {len(audio_manager.vad_audio_buffer)}")
    
    # Simulate interrupt
    print("\n🚨 Simulating interrupt...")
    print(f"- VAD buffer before interrupt: {len(audio_manager.vad_audio_buffer)} frames")
    print(f"- Pre-buffer before interrupt: {len(audio_manager.pre_buffer)} frames")
    
    # Check what happens after speech end
    print("\n🔇 Simulating speech end (what happens to buffers?)...")
    # This is what happens in the code at line 384:
    audio_manager.is_speech_active = False
    audio_manager.vad_audio_buffer = []  # <-- This clears the buffer!
    audio_manager.speech_start_time = None
    audio_manager.last_speech_time = None
    
    print(f"❌ VAD buffer after reset: {len(audio_manager.vad_audio_buffer)} frames")
    print(f"✅ Pre-buffer still has: {len(audio_manager.pre_buffer)} frames")
    
    print("\n💡 ISSUE IDENTIFIED:")
    print("- When speech ends, VAD buffer is cleared")
    print("- If interrupt happens during this, no audio for transcription")
    print("- Pre-buffer is preserved but may not be used for interrupts")
    
    print("\n🔧 POTENTIAL FIXES:")
    print("1. Don't clear VAD buffer immediately on interrupt")
    print("2. Use pre-buffer for interrupt transcription")
    print("3. Keep a separate interrupt audio buffer")
    print("4. Delay buffer clearing until after transcription")

if __name__ == "__main__":
    analyze_interrupt_scenario()