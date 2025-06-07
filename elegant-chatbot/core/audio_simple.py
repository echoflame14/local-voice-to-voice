"""
Simple blocking audio for Windows compatibility
"""
import numpy as np
import pyaudio
import queue
import threading
from typing import Optional


class SimpleAudioSystem:
    """Simple audio system using blocking mode"""
    
    def __init__(self, config):
        self.config = config.audio
        self.pa = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = queue.Queue()
        self.recording = False
        self.running = True
        self.is_playing = False
        
    def start(self):
        """Initialize audio streams"""
        print("Initializing audio...")
        
        # List available devices
        print("\nAvailable audio devices:")
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            print(f"  {i}: {info['name']} - {info['maxInputChannels']} in, {info['maxOutputChannels']} out")
        
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
        """Background recording loop"""
        while self.running:
            if self.recording and self.input_stream:
                try:
                    data = self.input_stream.read(self.config.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    self.audio_queue.put(audio_chunk)
                except Exception as e:
                    pass
                    
    def start_recording(self):
        """Start recording"""
        # Clear queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        self.recording = True
        
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio"""
        self.recording = False
        
        # Collect all audio from queue
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
            
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.int16)
        
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get single audio chunk if available"""
        if not self.audio_queue.empty():
            return self.audio_queue.get()
        return None
        
    def play_audio(self, audio_data: np.ndarray, interruptible=True):
        """Play audio through speakers with interrupt support"""
        if self.output_stream:
            self.is_playing = True
            
            # Play in chunks
            for i in range(0, len(audio_data), self.config.chunk_size):
                if not self.is_playing:  # Check for interrupt
                    print("\n🛑 Playback interrupted!")
                    break
                    
                chunk = audio_data[i:i + self.config.chunk_size]
                # Pad if necessary
                if len(chunk) < self.config.chunk_size:
                    chunk = np.pad(chunk, (0, self.config.chunk_size - len(chunk)))
                self.output_stream.write(chunk.tobytes())
                
            self.is_playing = False
            
    def stop_playback(self):
        """Stop current audio playback"""
        self.is_playing = False
                
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple energy-based VAD"""
        if len(audio_chunk) == 0:
            return False
        # Avoid sqrt of negative numbers
        mean_square = np.mean(audio_chunk.astype(np.float32)**2)
        if mean_square <= 0:
            return False
        energy = np.sqrt(mean_square)
        return energy > (self.config.vad_threshold * 1000)
        
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