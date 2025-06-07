"""
Interruptible audio system with proper recording during playback
"""
import numpy as np
import pyaudio
import queue
import threading
from typing import Optional, Callable
import time


class InterruptibleAudioSystem:
    """Audio system with proper interrupt support"""
    
    def __init__(self, config):
        self.config = config.audio
        self.pa = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = queue.Queue()
        self.recording = False
        self.running = True
        self.is_playing = False
        self.interrupt_callback = None
        
    def start(self):
        """Initialize audio streams"""
        print("Initializing interruptible audio...")
        
        # Open input stream
        try:
            self.input_stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            print("✓ Input stream opened")
        except Exception as e:
            print(f"❌ Failed to open input stream: {e}")
            raise
            
        # Open output stream
        try:
            self.output_stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=self.config.chunk_size
            )
            print("✓ Output stream opened")
        except Exception as e:
            print(f"❌ Failed to open output stream: {e}")
            raise
            
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        print("✓ Audio system ready\n")
        
    def _record_loop(self):
        """Background recording loop - ALWAYS recording"""
        while self.running:
            if self.input_stream:
                try:
                    data = self.input_stream.read(self.config.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Always put in queue
                    self.audio_queue.put(audio_chunk)
                    
                    # If playing and we have interrupt detection, check for speech
                    if self.is_playing and self.interrupt_callback:
                        if self.is_speech(audio_chunk):
                            self.interrupt_callback(audio_chunk)
                            
                except Exception as e:
                    pass
                    
    def start_recording(self):
        """Start recording (no-op since we're always recording)"""
        self.recording = True
        # Clear old audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
    def stop_recording(self):
        """Stop recording flag"""
        self.recording = False
        
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get single audio chunk if available"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
            
    def play_audio_interruptible(self, audio_data: np.ndarray, on_interrupt: Callable = None):
        """Play audio with interrupt support"""
        if not self.output_stream:
            return False
            
        # Set up interrupt detection
        interrupt_event = threading.Event()
        interrupt_buffer = []
        consecutive_speech_chunks = 0
        
        def interrupt_detector(chunk):
            nonlocal consecutive_speech_chunks
            consecutive_speech_chunks += 1
            interrupt_buffer.append(chunk)
            
            if consecutive_speech_chunks >= 2:  # 60ms of speech
                print("\n🛑 Interrupt detected!", end="", flush=True)
                interrupt_event.set()
                if on_interrupt:
                    on_interrupt(interrupt_buffer)
        
        self.interrupt_callback = interrupt_detector
        self.is_playing = True
        
        # Play in small chunks
        chunk_size = self.config.chunk_size
        start_idx = 0
        
        try:
            while start_idx < len(audio_data):
                # Check for interrupt FIRST
                if interrupt_event.is_set():
                    return True  # Was interrupted
                    
                # Get next chunk
                end_idx = min(start_idx + chunk_size, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                # Pad if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Play chunk
                self.output_stream.write(chunk.tobytes())
                start_idx = end_idx
                
                # Reset speech counter if no interrupt
                if not interrupt_event.is_set():
                    consecutive_speech_chunks = 0
                    
        finally:
            self.is_playing = False
            self.interrupt_callback = None
            
        return False  # Completed without interrupt
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Energy-based VAD with adaptive threshold"""
        if len(audio_chunk) == 0:
            return False
            
        # Calculate energy
        energy = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
        
        # Adaptive threshold - lower during playback for better interrupt detection
        threshold = self.config.vad_threshold * 500 if self.is_playing else self.config.vad_threshold * 1000
        
        return energy > threshold
        
    def close(self):
        """Clean up"""
        self.running = False
        self.recording = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        self.pa.terminate()