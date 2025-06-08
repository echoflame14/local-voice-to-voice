#!/usr/bin/env python3
"""
Simple test script to verify interrupt functionality works correctly.
Tests the simplified interrupt implementation.
"""

import time
import threading
import numpy as np
from src.pipeline.voice_assistant import VoiceAssistant
from src.llm.openai_compatible import OpenAICompatibleLLM
from src.stt.whisper_stt import WhisperSTT
from src.tts.chatterbox_wrapper import ChatterboxTTS

def simulate_user_speech(assistant, delay=1.5, duration=0.5):
    """Simulate user speaking after a delay"""
    print(f"\n⏰ Will simulate user speech in {delay} seconds...")
    time.sleep(delay)
    
    print("🎤 SIMULATING USER SPEECH NOW!")
    # Create a fake audio chunk (non-zero to trigger VAD)
    sample_rate = 16000
    samples = int(duration * sample_rate)
    # Generate audio with some energy
    audio = np.random.randn(samples).astype(np.float32) * 0.1
    
    # Trigger the interrupt callback directly
    if hasattr(assistant, '_interrupt_detection_callback'):
        assistant._interrupt_detection_callback(audio)
    
    print("✅ Simulated user speech sent")

def test_interrupt():
    """Test interrupt functionality"""
    print("="*60)
    print("🧪 INTERRUPT TEST - Simplified Implementation")
    print("="*60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    stt = WhisperSTT(model_size="base", device="cpu")
    llm = OpenAICompatibleLLM()
    tts = ChatterboxTTS(voice_reference_path="josh.wav")
    
    # Create assistant
    assistant = VoiceAssistant(
        stt=stt,
        llm=llm,
        tts=tts,
        enable_sound_effects=False,
        log_conversations=True
    )
    
    # Set up callbacks to track state
    def on_response(text):
        print(f"\n🤖 Assistant response: {text[:50]}...")
    
    assistant.on_response = on_response
    
    print("\n✅ Assistant initialized")
    
    # Test 1: Basic interrupt
    print("\n" + "="*40)
    print("TEST 1: Basic Interrupt")
    print("="*40)
    print("1. Assistant will start speaking")
    print("2. User interrupts after 1.5 seconds")
    print("3. Verify interrupt happens immediately")
    
    # Start assistant
    assistant.start()
    
    # Trigger a response
    print("\n📝 Triggering assistant response...")
    assistant._process_speech(np.ones(16000, dtype=np.float32) * 0.1)  # Fake audio
    
    # Simulate interrupt
    interrupt_thread = threading.Thread(
        target=simulate_user_speech,
        args=(assistant, 1.5, 0.5)
    )
    interrupt_thread.start()
    
    # Wait for test to complete
    time.sleep(5)
    
    # Check results
    print("\n📊 Test 1 Results:")
    print(f"- Last interrupt time: {assistant.last_interrupt_time}")
    print(f"- Is speaking: {assistant.is_speaking}")
    print(f"- Playback interrupted at: {assistant.playback_interrupted_at}")
    
    # Test 2: Interrupt cooldown
    print("\n" + "="*40)
    print("TEST 2: Interrupt Cooldown")
    print("="*40)
    print("Testing rapid interrupts to verify cooldown works")
    
    # Reset state
    assistant.last_interrupt_time = 0
    
    # Trigger response again
    print("\n📝 Triggering another assistant response...")
    assistant._process_speech(np.ones(16000, dtype=np.float32) * 0.1)
    
    # Try multiple interrupts
    for i in range(3):
        time.sleep(0.5)
        print(f"\n🎤 Interrupt attempt {i+1}")
        simulate_user_speech(assistant, 0, 0.1)
    
    time.sleep(3)
    
    # Test 3: No interrupt when not speaking
    print("\n" + "="*40)
    print("TEST 3: No Interrupt When Not Speaking")
    print("="*40)
    
    # Wait for assistant to finish
    time.sleep(2)
    assistant.is_speaking = False
    
    print("📝 Simulating user speech when assistant is NOT speaking...")
    simulate_user_speech(assistant, 0, 0.5)
    
    time.sleep(2)
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    assistant.stop()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_interrupt()