"""
Adaptive streaming synthesis with intelligent chunking based on synthesis speed
"""
import re
import threading
import queue
import time
from typing import List, Generator, Tuple, Optional, Dict
import numpy as np
from collections import deque

class AdaptiveStreamingSynthesizer:
    """
    Intelligent streaming synthesizer that adapts chunk sizes based on:
    1. Synthesis speed (roughly 1:1 with audio duration)
    2. Natural speech boundaries 
    3. Buffer levels to maintain smooth playback
    """
    
    def __init__(self, tts, audio_manager):
        self.tts = tts
        self.audio_manager = audio_manager
        
        # Adaptive chunking parameters
        self.min_chunk_words = 2  # Minimum for ultra-low latency first response
        self.max_chunk_words = 20  # Maximum chunk size
        self.target_buffer_seconds = 2.0  # Target audio buffer
        
        # Synthesis timing statistics
        self.synthesis_times = deque(maxlen=10)  # Track last 10 synthesis times
        self.audio_durations = deque(maxlen=10)   # Track last 10 audio durations
        
        # Queues
        self.text_buffer = []  # Buffer for incoming text
        self.synthesis_queue = queue.Queue(maxsize=3)
        
        # State tracking
        self.is_active = False
        self.should_stop = threading.Event()
        self.first_chunk_ready = threading.Event()
        self.total_audio_duration = 0.0
        self.playback_position = 0.0
        
        # Threads
        self.synthesis_thread = None
        self.playback_thread = None
        self.text_accumulator_thread = None
        
    def estimate_text_duration(self, text: str) -> float:
        """Estimate audio duration for text (rough approximation)"""
        # Average speaking rate: ~150 words per minute = 2.5 words per second
        word_count = len(text.split())
        return word_count / 2.5
        
    def get_synthesis_speed_ratio(self) -> float:
        """Calculate synthesis speed ratio (synthesis_time / audio_duration)"""
        if len(self.synthesis_times) < 2:
            return 1.0  # Assume 1:1 until we have data
            
        avg_synthesis_time = sum(self.synthesis_times) / len(self.synthesis_times)
        avg_audio_duration = sum(self.audio_durations) / len(self.audio_durations)
        
        if avg_audio_duration > 0:
            return avg_synthesis_time / avg_audio_duration
        return 1.0
        
    def calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on current conditions"""
        # If this is the first chunk, use minimum size for fastest response
        if len(self.synthesis_times) == 0:
            return self.min_chunk_words
            
        # Get current synthesis speed ratio
        speed_ratio = self.get_synthesis_speed_ratio()
        
        # Calculate buffer level (how much audio we have ready to play)
        buffer_level = self.total_audio_duration - self.playback_position
        
        # If buffer is low, use smaller chunks
        if buffer_level < 0.5:
            return self.min_chunk_words
        elif buffer_level < 1.0:
            return min(5, self.max_chunk_words)
        elif buffer_level < self.target_buffer_seconds:
            # Scale chunk size based on synthesis speed
            # If synthesis is fast (ratio < 1), we can use larger chunks
            if speed_ratio < 0.8:
                return min(15, self.max_chunk_words)
            elif speed_ratio < 1.0:
                return min(10, self.max_chunk_words)
            else:
                return min(7, self.max_chunk_words)
        else:
            # Buffer is healthy, use larger chunks for efficiency
            return self.max_chunk_words
            
    def find_natural_break(self, text: str, target_words: int) -> Tuple[str, str]:
        """Find natural break point in text near target word count"""
        words = text.split()
        
        if len(words) <= target_words:
            return text, ""
            
        # Look for punctuation marks that indicate good break points
        strong_breaks = ['.', '!', '?']
        medium_breaks = [',', ';', ':']
        weak_breaks = ['-', '—', '(', ')']
        
        # First, try to find a strong break near our target
        for i in range(min(target_words + 3, len(words) - 1), max(target_words - 2, 0), -1):
            if any(p in words[i] for p in strong_breaks):
                chunk = ' '.join(words[:i+1])
                remainder = ' '.join(words[i+1:])
                return chunk, remainder
                
        # Try medium breaks
        for i in range(min(target_words + 2, len(words) - 1), max(target_words - 2, 0), -1):
            if any(p in words[i] for p in medium_breaks):
                chunk = ' '.join(words[:i+1])
                remainder = ' '.join(words[i+1:])
                return chunk, remainder
                
        # Default: break at target word count
        chunk = ' '.join(words[:target_words])
        remainder = ' '.join(words[target_words:])
        return chunk, remainder
        
    def start_streaming(self, text_generator: Generator[str, None, None]):
        """Start adaptive streaming synthesis"""
        self.is_active = True
        self.should_stop.clear()
        self.first_chunk_ready.clear()
        
        # Reset statistics
        self.synthesis_times.clear()
        self.audio_durations.clear()
        self.total_audio_duration = 0.0
        self.playback_position = 0.0
        
        # Start threads
        self.text_accumulator_thread = threading.Thread(
            target=self._text_accumulator,
            args=(text_generator,),
            daemon=True,
            name="TextAccumulator"
        )
        
        self.synthesis_thread = threading.Thread(
            target=self._adaptive_synthesis_worker,
            daemon=True,
            name="AdaptiveSynthesis"
        )
        
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="AdaptivePlayback"
        )
        
        self.text_accumulator_thread.start()
        self.synthesis_thread.start()
        self.playback_thread.start()
        
    def _text_accumulator(self, text_generator: Generator[str, None, None]):
        """Accumulate incoming text"""
        try:
            for chunk in text_generator:
                if self.should_stop.is_set():
                    break
                self.text_buffer.append(chunk)
        finally:
            self.text_buffer.append(None)  # End signal
            
    def _adaptive_synthesis_worker(self):
        """Adaptive synthesis with intelligent chunking"""
        accumulated_text = ""
        end_of_stream = False
        
        while self.is_active and not self.should_stop.is_set():
            # Accumulate text from buffer
            while self.text_buffer:
                chunk = self.text_buffer.pop(0)
                if chunk is None:
                    end_of_stream = True
                    break
                accumulated_text += chunk
                
            if not accumulated_text and end_of_stream:
                # Signal end of synthesis
                self.synthesis_queue.put(None)
                break
                
            # Calculate optimal chunk size
            optimal_chunk_size = self.calculate_optimal_chunk_size()
            
            # Check if we have enough text for a chunk
            word_count = len(accumulated_text.split())
            
            # For first chunk, synthesize as soon as we have minimum words
            if len(self.synthesis_times) == 0 and word_count >= self.min_chunk_words:
                chunk_text, accumulated_text = self.find_natural_break(accumulated_text, self.min_chunk_words)
            # For subsequent chunks, wait for optimal size or end of stream
            elif word_count >= optimal_chunk_size or (end_of_stream and accumulated_text.strip()):
                chunk_text, accumulated_text = self.find_natural_break(accumulated_text, optimal_chunk_size)
            else:
                # Not enough text yet, wait for more
                if not end_of_stream:
                    time.sleep(0.05)
                    continue
                    
            # Synthesize chunk
            if 'chunk_text' in locals() and chunk_text.strip():
                start_time = time.time()
                
                try:
                    print(f"🎵 Synthesizing chunk ({len(chunk_text.split())} words): '{chunk_text[:50]}...'")
                    audio, sample_rate = self.tts.synthesize(chunk_text)
                    
                    synthesis_time = time.time() - start_time
                    audio_duration = len(audio) / sample_rate
                    
                    # Update statistics
                    self.synthesis_times.append(synthesis_time)
                    self.audio_durations.append(audio_duration)
                    self.total_audio_duration += audio_duration
                    
                    # Queue for playback
                    self.synthesis_queue.put({
                        'audio': audio,
                        'sample_rate': sample_rate,
                        'text': chunk_text,
                        'synthesis_time': synthesis_time,
                        'audio_duration': audio_duration,
                        'chunk_number': len(self.synthesis_times)
                    })
                    
                    # Signal first chunk is ready
                    if len(self.synthesis_times) == 1:
                        self.first_chunk_ready.set()
                        
                    print(f"   ✅ Chunk {len(self.synthesis_times)}: {synthesis_time:.2f}s synthesis, {audio_duration:.2f}s audio (ratio: {synthesis_time/audio_duration:.2f})")
                    
                except Exception as e:
                    print(f"❌ Synthesis error: {e}")
                    
    def _playback_worker(self):
        """Playback worker with position tracking"""
        first_chunk = True
        
        while self.is_active and not self.should_stop.is_set():
            try:
                # Get synthesized audio
                chunk_data = self.synthesis_queue.get(timeout=0.1)
                
                if chunk_data is None:  # End signal
                    break
                    
                if first_chunk:
                    # Start output stream on first chunk
                    self.audio_manager.start_output_stream()
                    first_chunk = False
                    print(f"🚀 First audio ready in {chunk_data['synthesis_time']:.2f}s")
                    
                # Resample if needed
                audio = chunk_data['audio']
                sample_rate = chunk_data['sample_rate']
                
                if sample_rate != self.audio_manager.output_sample_rate:
                    import librosa
                    audio = librosa.resample(
                        audio,
                        orig_sr=sample_rate,
                        target_sr=self.audio_manager.output_sample_rate
                    )
                    
                # Stream audio
                playback_start = time.time()
                self.audio_manager.stream_audio(audio)
                playback_duration = time.time() - playback_start
                
                # Update playback position
                self.playback_position += chunk_data['audio_duration']
                
                # Log buffer health
                buffer_level = self.total_audio_duration - self.playback_position
                print(f"   📊 Buffer: {buffer_level:.2f}s, Chunk size decision for next: {self.calculate_optimal_chunk_size()} words")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Playback error: {e}")
                
    def stop(self):
        """Stop streaming"""
        self.should_stop.set()
        self.is_active = False
        
    def wait_for_first_chunk(self, timeout: float = 5.0) -> bool:
        """Wait for first chunk to be ready"""
        return self.first_chunk_ready.wait(timeout)