"""
Unified audio handling module
Simple, efficient audio I/O
"""
import numpy as np
import pyaudio
from collections import deque
from typing import Optional, Callable
import threading
import time


class AudioSystem:
    """Simple, unified audio system"""
    
    def __init__(self, config):
        self.config = config.audio
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
        # Single circular buffer for all audio
        buffer_size = int(config.audio.sample_rate * 5)  # 5 seconds
        self.audio_buffer = deque(maxlen=buffer_size)
        
        # Simple state
        self.is_recording = False
        self.is_playing = False
        
    def start(self):
        """Initialize audio stream"""
        try:
            # Try with callback first
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                output=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"Callback mode failed: {e}")
            print("Trying blocking mode...")
            # Fall back to blocking mode
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                output=True,
                frames_per_buffer=self.config.chunk_size
            )
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Simple audio callback"""
        if self.is_recording and in_data:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.audio_buffer.extend(audio_chunk)
            
        return (in_data, pyaudio.paContinue)
        
    def start_recording(self):
        """Start recording audio"""
        self.audio_buffer.clear()
        self.is_recording = True
        
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data"""
        self.is_recording = False
        return np.array(self.audio_buffer, dtype=np.int16)
        
    def play_audio(self, audio_data: np.ndarray):
        """Play audio through speakers"""
        self.is_playing = True
        
        # Simple blocking playback
        for i in range(0, len(audio_data), self.config.chunk_size):
            if not self.is_playing:  # Allow interruption
                break
            chunk = audio_data[i:i + self.config.chunk_size]
            self.stream.write(chunk.tobytes())
            
        self.is_playing = False
        
    def stop_playback(self):
        """Stop current playback"""
        self.is_playing = False
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple energy-based speech detection"""
        if not self.config.vad_enabled:
            return False
            
        energy = np.sqrt(np.mean(audio_chunk**2))
        return energy > (self.config.vad_threshold * 1000)
        
    def close(self):
        """Clean up resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


class SimpleVAD:
    """Simple Voice Activity Detection"""
    
    def __init__(self, config):
        self.config = config.audio
        self.speech_frames = 0
        self.silence_frames = 0
        
    def process(self, audio_chunk: np.ndarray) -> str:
        """Process audio chunk and return state"""
        energy = np.sqrt(np.mean(audio_chunk**2))
        
        if energy > (self.config.vad_threshold * 1000):
            self.speech_frames += 1
            self.silence_frames = 0
            
            if self.speech_frames > 3:  # ~90ms of speech
                return "speech"
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            if self.silence_frames > 20:  # ~600ms of silence
                return "silence"
                
        return "unknown"