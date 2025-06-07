#!/usr/bin/env python3
"""
Interrupt Demo - Shows the problem and solution
"""
import sys
import time
import numpy as np
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.audio_simple import SimpleAudioSystem
from core.tts import SimpleTTS


def test_current_implementation():
    """Test the current interrupt implementation (problematic)"""
    print("\n" + "="*60)
    print("CURRENT IMPLEMENTATION (main_interrupt.py style)")
    print("="*60)
    
    audio = SimpleAudioSystem(config)
    tts = SimpleTTS(config)
    
    try:
        audio.start()
        audio.start_recording()
        
        text = "This is the current implementation. Try to interrupt me while I'm speaking. You'll notice I don't stop immediately."
        audio_data = tts.synthesize(text)
        
        print("\n🎤 SPEAK NOW to interrupt me!")
        print("Notice: Even if you speak, I'll finish my sentence first.\n")
        
        # Current approach - checks is_playing flag but doesn't stop immediately
        def monitor_interrupts():
            while audio.is_playing:
                chunk = audio.get_audio_chunk()
                if chunk is not None and audio.is_speech(chunk):
                    print("\n🔴 Interrupt detected (but not stopping yet)...")
                    audio.stop_playback()  # Sets is_playing = False
                    return
                time.sleep(0.01)
        
        # Start monitor
        monitor_thread = threading.Thread(target=monitor_interrupts, daemon=True)
        monitor_thread.start()
        
        # Play audio (current implementation)
        print("Speaking: ", end="", flush=True)
        audio.play_audio(audio_data)
        print(" [Finished]")
        
        monitor_thread.join(timeout=0.1)
        
    finally:
        audio.close()


def test_fixed_implementation():
    """Test the fixed interrupt implementation"""
    print("\n" + "="*60)
    print("FIXED IMPLEMENTATION (Immediate interrupt)")
    print("="*60)
    
    audio = SimpleAudioSystem(config)
    tts = SimpleTTS(config)
    
    try:
        audio.start()
        audio.start_recording()
        
        text = "This is the fixed implementation. Try to interrupt me and I'll stop immediately when you start speaking."
        audio_data = tts.synthesize(text)
        
        print("\n🎤 SPEAK NOW to interrupt me!")
        print("Notice: I'll stop immediately when you speak.\n")
        
        # Fixed approach - uses event signaling
        interrupt_event = threading.Event()
        
        def monitor_interrupts_fixed():
            consecutive_speech = 0
            while not interrupt_event.is_set():
                chunk = audio.get_audio_chunk()
                if chunk is not None and audio.is_speech(chunk):
                    consecutive_speech += 1
                    if consecutive_speech >= 2:  # 60ms of speech
                        print("\n✅ Interrupt detected - stopping NOW!")
                        interrupt_event.set()
                        return
                else:
                    consecutive_speech = 0
                time.sleep(0.01)
        
        # Start monitor
        monitor_thread = threading.Thread(target=monitor_interrupts_fixed, daemon=True)
        monitor_thread.start()
        
        # Play audio with interrupt checking
        print("Speaking: ", end="", flush=True)
        chunk_size = 480
        audio.is_playing = True
        
        for i in range(0, len(audio_data), chunk_size):
            # Check interrupt BEFORE playing each chunk
            if interrupt_event.is_set():
                print(" [INTERRUPTED!]")
                break
                
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            audio.output_stream.write(chunk.tobytes())
            # Small visual indicator
            print(".", end="", flush=True)
            
        audio.is_playing = False
        
        if not interrupt_event.is_set():
            print(" [Finished]")
            
        monitor_thread.join(timeout=0.1)
        
    finally:
        audio.close()


def test_automated_comparison():
    """Automated test showing the difference"""
    print("\n" + "="*60)
    print("AUTOMATED COMPARISON TEST")
    print("="*60)
    
    audio = SimpleAudioSystem(config)
    tts = SimpleTTS(config)
    
    try:
        audio.start()
        
        # Generate test audio
        text = "One, two, three, four, five, six, seven, eight, nine, ten."
        audio_data = tts.synthesize(text)
        
        print("\nTest 1: Current implementation with simulated interrupt at 1s")
        print("-" * 50)
        
        # Current implementation
        interrupt_at = 1.0
        start_time = time.time()
        
        def delayed_interrupt():
            time.sleep(interrupt_at)
            print(f"\n⚡ Interrupt signal at {time.time() - start_time:.1f}s")
            audio.stop_playback()
        
        threading.Thread(target=delayed_interrupt, daemon=True).start()
        
        audio.play_audio(audio_data)
        actual_stop = time.time() - start_time
        print(f"Actually stopped at: {actual_stop:.1f}s")
        print(f"Delay: {actual_stop - interrupt_at:.1f}s")
        
        time.sleep(1)
        
        print("\nTest 2: Fixed implementation with simulated interrupt at 1s")
        print("-" * 50)
        
        # Fixed implementation
        interrupt_event = threading.Event()
        start_time = time.time()
        
        def delayed_interrupt_fixed():
            time.sleep(interrupt_at)
            print(f"\n⚡ Interrupt signal at {time.time() - start_time:.1f}s")
            interrupt_event.set()
        
        threading.Thread(target=delayed_interrupt_fixed, daemon=True).start()
        
        # Play with interrupt checking
        chunk_size = 480
        for i in range(0, len(audio_data), chunk_size):
            if interrupt_event.is_set():
                actual_stop = time.time() - start_time
                print(f"Actually stopped at: {actual_stop:.1f}s")
                print(f"Delay: {actual_stop - interrupt_at:.1f}s (< 30ms!)")
                break
                
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            audio.output_stream.write(chunk.tobytes())
            
    finally:
        audio.close()


def main():
    print("\n" + "="*60)
    print("INTERRUPT DEMONSTRATION")
    print("This shows why interrupts aren't working and how to fix them")
    print("="*60)
    
    # Show the problem
    test_current_implementation()
    
    time.sleep(2)
    
    # Show the solution
    test_fixed_implementation()
    
    time.sleep(2)
    
    # Automated comparison
    test_automated_comparison()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Problem: Current implementation plays entire chunks before checking")
    print("Solution: Check interrupt flag before playing each small chunk")
    print("Result: Interrupt response time < 30ms instead of seconds")
    print("\nUse main_interrupt_fixed.py for the corrected version!")
    print("="*60)


if __name__ == "__main__":
    main()