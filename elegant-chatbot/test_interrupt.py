#!/usr/bin/env python3
"""
Test script for interrupt functionality
Tests both manual and automated interrupt scenarios
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


class InterruptTester:
    """Test harness for interrupt functionality"""
    
    def __init__(self):
        self.audio = SimpleAudioSystem(config)
        self.tts = SimpleTTS(config)
        self.interrupt_event = threading.Event()
        self.test_results = []
        
    def test_basic_playback(self):
        """Test 1: Basic playback without interrupts"""
        print("\n" + "="*50)
        print("TEST 1: Basic Playback (No Interrupts)")
        print("="*50)
        
        text = "This is a test of basic playback. One, two, three, four, five."
        audio_data = self.tts.synthesize(text)
        
        print(f"Audio duration: {len(audio_data)/16000:.1f}s")
        print("Playing...")
        
        start_time = time.time()
        self.audio.play_audio(audio_data)
        play_time = time.time() - start_time
        
        print(f"✓ Playback completed in {play_time:.1f}s")
        self.test_results.append(("Basic Playback", "PASSED", f"{play_time:.1f}s"))
        
    def test_interrupt_detection(self):
        """Test 2: Interrupt detection mechanism"""
        print("\n" + "="*50)
        print("TEST 2: Interrupt Detection")
        print("="*50)
        
        # Generate some test audio
        text = "Testing interrupt detection. This message should be interrupted halfway through if you speak."
        audio_data = self.tts.synthesize(text)
        
        print("Playing audio - SPEAK NOW to test interrupt!")
        print("(You have 3 seconds to interrupt)")
        
        # Play with interrupt monitoring
        interrupted = self._play_with_interrupt_monitor(audio_data, manual=True)
        
        if interrupted:
            print("✓ Interrupt detected successfully!")
            self.test_results.append(("Manual Interrupt", "PASSED", "User interrupted"))
        else:
            print("✗ No interrupt detected")
            self.test_results.append(("Manual Interrupt", "NO INTERRUPT", "Completed"))
            
    def test_interrupt_timing(self):
        """Test 3: Automated interrupt timing test"""
        print("\n" + "="*50)
        print("TEST 3: Automated Interrupt Timing")
        print("="*50)
        
        text = "This is an automated test. The system will simulate an interrupt after one second."
        audio_data = self.tts.synthesize(text)
        
        print("Playing with automated interrupt after 1 second...")
        
        # Start playback with automated interrupt
        interrupt_time = 1.0  # Interrupt after 1 second
        interrupted = self._play_with_timed_interrupt(audio_data, interrupt_time)
        
        if interrupted:
            print(f"✓ Successfully interrupted after ~{interrupt_time}s")
            self.test_results.append(("Timed Interrupt", "PASSED", f"{interrupt_time}s"))
        else:
            print("✗ Failed to interrupt")
            self.test_results.append(("Timed Interrupt", "FAILED", "No interrupt"))
            
    def test_interrupt_responsiveness(self):
        """Test 4: Measure interrupt response time"""
        print("\n" + "="*50)
        print("TEST 4: Interrupt Response Time")
        print("="*50)
        
        text = "Testing response time. How quickly can we stop playback after detecting an interrupt?"
        audio_data = self.tts.synthesize(text)
        
        print("Measuring interrupt response time...")
        
        # Test multiple times and average
        response_times = []
        for i in range(3):
            print(f"\nAttempt {i+1}/3:")
            response_time = self._measure_interrupt_response(audio_data)
            if response_time:
                response_times.append(response_time)
                print(f"  Response time: {response_time*1000:.0f}ms")
            else:
                print("  No interrupt")
                
        if response_times:
            avg_response = np.mean(response_times)
            print(f"\n✓ Average response time: {avg_response*1000:.0f}ms")
            self.test_results.append(("Response Time", "PASSED", f"{avg_response*1000:.0f}ms"))
        else:
            print("\n✗ Could not measure response time")
            self.test_results.append(("Response Time", "FAILED", "No data"))
            
    def _play_with_interrupt_monitor(self, audio_data, manual=False):
        """Play audio while monitoring for interrupts"""
        self.interrupt_event.clear()
        chunk_size = 480  # 30ms chunks
        
        # Start interrupt monitor
        if not manual:
            monitor_thread = threading.Thread(
                target=self._monitor_for_speech,
                daemon=True
            )
            monitor_thread.start()
        else:
            print("\n[Listening for your voice...]")
            
        # Visual progress bar
        total_chunks = len(audio_data) // chunk_size
        self.audio.is_playing = True
        
        for i in range(0, len(audio_data), chunk_size):
            # Check for interrupt
            if self.interrupt_event.is_set() or (manual and self._check_manual_interrupt()):
                self.audio.is_playing = False
                print("\n🛑 INTERRUPTED!")
                return True
                
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
            self.audio.output_stream.write(chunk.tobytes())
            
            # Progress indicator
            progress = int((i / len(audio_data)) * 40)
            print(f"\r[{'#' * progress}{'-' * (40 - progress)}]", end="", flush=True)
            
        self.audio.is_playing = False
        print("\r[" + "#"*40 + "] Complete")
        return False
        
    def _play_with_timed_interrupt(self, audio_data, interrupt_after):
        """Play audio with automated interrupt after specified time"""
        # Schedule interrupt
        timer = threading.Timer(interrupt_after, self._trigger_interrupt)
        timer.start()
        
        return self._play_with_interrupt_monitor(audio_data, manual=False)
        
    def _measure_interrupt_response(self, audio_data):
        """Measure how quickly playback stops after interrupt"""
        self.interrupt_event.clear()
        self.interrupt_time = None
        chunk_size = 480
        stop_time = None
        
        # Schedule interrupt after 0.5 seconds
        timer = threading.Timer(0.5, self._trigger_interrupt_with_timing)
        timer.start()
        
        self.audio.is_playing = True
        for i in range(0, len(audio_data), chunk_size):
            if self.interrupt_event.is_set():
                stop_time = time.time()
                self.audio.is_playing = False
                break
                
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self.audio.output_stream.write(chunk.tobytes())
            
        self.audio.is_playing = False
        timer.cancel()  # Cancel timer if not fired
        
        if self.interrupt_time and stop_time:
            return stop_time - self.interrupt_time
        return None
        
    def _check_manual_interrupt(self):
        """Check for manual interrupt (speech detection)"""
        # Make sure we're still recording during playback
        if not self.audio.recording:
            self.audio.start_recording()
            
        chunk = self.audio.get_audio_chunk()
        if chunk is not None:
            # Lower threshold for testing
            energy = np.sqrt(np.mean(chunk.astype(np.float32)**2))
            if energy > 300:  # Lower threshold for testing
                print(f"\n[Detected energy: {energy:.0f}]", end="", flush=True)
                return True
        return False
        
    def _monitor_for_speech(self):
        """Monitor for speech in background"""
        consecutive_speech = 0
        while self.audio.is_playing:
            chunk = self.audio.get_audio_chunk()
            if chunk is not None and self.audio.is_speech(chunk):
                consecutive_speech += 1
                if consecutive_speech >= 3:  # 90ms of speech
                    self.interrupt_event.set()
                    return
            else:
                consecutive_speech = 0
            time.sleep(0.01)
            
    def _trigger_interrupt(self):
        """Trigger an interrupt event"""
        print("\n⚡ Triggering interrupt...", end="", flush=True)
        self.interrupt_event.set()
        
    def _trigger_interrupt_with_timing(self):
        """Trigger interrupt and record time"""
        self.interrupt_time = time.time()
        self.interrupt_event.set()
        
    def run_all_tests(self):
        """Run all interrupt tests"""
        print("\n" + "="*60)
        print("INTERRUPT FUNCTIONALITY TEST SUITE")
        print("="*60)
        
        try:
            # Initialize audio
            print("\nInitializing audio system...")
            self.audio.start()
            self.audio.start_recording()
            
            # Run tests
            self.test_basic_playback()
            time.sleep(1)
            
            self.test_interrupt_timing()
            time.sleep(1)
            
            self.test_interrupt_responsiveness()
            time.sleep(1)
            
            self.test_interrupt_detection()
            
            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            print(f"{'Test':<20} {'Result':<15} {'Details':<20}")
            print("-"*60)
            for test, result, details in self.test_results:
                status_color = "✓" if result == "PASSED" else "✗"
                print(f"{test:<20} {status_color} {result:<13} {details:<20}")
                
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
        finally:
            self.audio.close()


def main():
    """Run interrupt tests"""
    tester = InterruptTester()
    tester.run_all_tests()
    
    print("\n" + "="*60)
    print("To fix interrupts in your main script, use the")
    print("InterruptibleAudioPlayer from main_interrupt_fixed.py")
    print("="*60)


if __name__ == "__main__":
    main()