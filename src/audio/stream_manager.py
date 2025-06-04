import pyaudio
import numpy as np
import queue
import threading
from typing import Callable, Optional
import time
import sounddevice as sd
import logging
from dataclasses import dataclass
from collections import deque


@dataclass
class AudioPerformanceMetrics:
    """Track audio performance metrics"""
    synthesis_start_time: float = 0.0
    synthesis_end_time: float = 0.0
    queue_start_time: float = 0.0
    playback_start_time: float = 0.0
    playback_end_time: float = 0.0
    audio_duration: float = 0.0
    queue_size_at_start: int = 0
    queue_size_at_end: int = 0
    
    @property
    def synthesis_time(self) -> float:
        return self.synthesis_end_time - self.synthesis_start_time
    
    @property
    def queue_wait_time(self) -> float:
        return self.playback_start_time - self.queue_start_time
    
    @property
    def total_latency(self) -> float:
        return self.playback_start_time - self.synthesis_start_time
    
    @property
    def playback_duration(self) -> float:
        return self.playback_end_time - self.playback_start_time


class AudioStreamManager:
    """Manages real-time audio input/output streams with low latency and performance logging"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        enable_performance_logging: bool = True
    ):
        """
        Initialize audio stream manager
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Samples per chunk (affects latency)
            input_device: Input device index (None for default)
            output_device: Output device index (None for default)
            enable_performance_logging: Whether to enable detailed performance logging
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.input_device = input_device
        self.output_device = output_device
        self.enable_performance_logging = enable_performance_logging
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=100)  # Keep last 100 audio chunks
        self.current_metrics = None
        self.total_audio_played = 0.0
        self.total_synthesis_time = 0.0
        self.total_queue_wait_time = 0.0
        
        # PyAudio instance
        self.pa = pyaudio.PyAudio()
        
        # Queues for audio data
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Stream handles
        self.input_stream = None
        self.output_stream = None
        
        # Callbacks
        self.input_callback = None
        self.output_callback = None
        
        # State
        self.is_recording = False
        self.is_playing = False
        self._stop_event = threading.Event()
        
        # Streaming control
        self.current_streaming_thread = None
        self.cancel_streaming = threading.Event()
        
        # Performance logging
        if self.enable_performance_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        
        # List available devices on init
        self.list_devices()
    
    def log_performance(self, message: str, level: str = "info"):
        """Log performance information"""
        if not self.enable_performance_logging:
            return
        
        timestamp = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
        if level == "info":
            print(f"🎵 [{timestamp}] {message}")
        elif level == "warning":
            print(f"⚠️  [{timestamp}] {message}")
        elif level == "error":
            print(f"❌ [{timestamp}] {message}")
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
    
    def start_audio_metrics(self, audio_duration: float) -> AudioPerformanceMetrics:
        """Start tracking metrics for a new audio chunk"""
        metrics = AudioPerformanceMetrics()
        metrics.synthesis_start_time = time.time()
        metrics.audio_duration = audio_duration
        metrics.queue_size_at_start = self.output_queue.qsize()
        return metrics
    
    def log_synthesis_complete(self, metrics: AudioPerformanceMetrics):
        """Log synthesis completion"""
        metrics.synthesis_end_time = time.time()
        if metrics.audio_duration > 0:
            ratio = metrics.synthesis_time / metrics.audio_duration
            self.log_performance(
                f"Synthesis complete: {metrics.synthesis_time:.3f}s for {metrics.audio_duration:.1f}s audio "
                f"(ratio: {ratio:.2f}x)"
            )
    
    def log_queue_start(self, metrics: AudioPerformanceMetrics):
        """Log when audio is queued for playback"""
        metrics.queue_start_time = time.time()
        self.log_performance(
            f"Audio queued: queue_size={metrics.queue_size_at_start}, "
            f"synthesis_to_queue={metrics.queue_start_time - metrics.synthesis_end_time:.3f}s"
        )
    
    def log_playback_start(self, metrics: AudioPerformanceMetrics):
        """Log when playback starts"""
        metrics.playback_start_time = time.time()
        metrics.queue_size_at_end = self.output_queue.qsize()
        self.log_performance(
            f"Playback started: queue_wait={metrics.queue_wait_time:.3f}s, "
            f"total_latency={metrics.total_latency:.3f}s"
        )
    
    def log_playback_complete(self, metrics: AudioPerformanceMetrics):
        """Log when playback completes"""
        metrics.playback_end_time = time.time()
        
        # Update totals
        self.total_audio_played += metrics.audio_duration
        self.total_synthesis_time += metrics.synthesis_time
        self.total_queue_wait_time += metrics.queue_wait_time
        
        # Store metrics
        self.performance_metrics.append(metrics)
        
        if metrics.playback_duration > 0 and metrics.audio_duration > 0:
            efficiency = (metrics.audio_duration / metrics.playback_duration) * 100
            self.log_performance(
                f"Playback complete: duration={metrics.playback_duration:.3f}s, "
                f"audio_length={metrics.audio_duration:.1f}s, "
                f"efficiency={efficiency:.1f}%",
                "success"
            )
    
    def get_performance_summary(self) -> dict:
        """Get performance summary statistics"""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = list(self.performance_metrics)
        
        avg_synthesis_time = sum(m.synthesis_time for m in recent_metrics) / len(recent_metrics)
        avg_queue_wait = sum(m.queue_wait_time for m in recent_metrics) / len(recent_metrics)
        avg_total_latency = sum(m.total_latency for m in recent_metrics) / len(recent_metrics)
        avg_audio_duration = sum(m.audio_duration for m in recent_metrics) / len(recent_metrics)
        
        synthesis_ratio = avg_synthesis_time / avg_audio_duration if avg_audio_duration > 0 else 0
        
        return {
            "total_audio_played": self.total_audio_played,
            "total_synthesis_time": self.total_synthesis_time,
            "total_queue_wait_time": self.total_queue_wait_time,
            "avg_synthesis_time": avg_synthesis_time,
            "avg_queue_wait": avg_queue_wait,
            "avg_total_latency": avg_total_latency,
            "avg_audio_duration": avg_audio_duration,
            "synthesis_speed_ratio": synthesis_ratio,
            "chunks_processed": len(recent_metrics)
        }
    
    def print_performance_summary(self):
        """Print a formatted performance summary"""
        summary = self.get_performance_summary()
        if not summary:
            print("📊 No performance data available")
            return
        
        print("\n" + "="*60)
        print("📊 AUDIO PERFORMANCE SUMMARY")
        print("="*60)
        print(f"🎵 Total audio played: {summary['total_audio_played']:.1f}s")
        print(f"⚡ Total synthesis time: {summary['total_synthesis_time']:.1f}s")
        print(f"⏱️  Total queue wait time: {summary['total_queue_wait_time']:.1f}s")
        print(f"📈 Chunks processed: {summary['chunks_processed']}")
        print()
        print("📊 AVERAGES:")
        print(f"   Synthesis time: {summary['avg_synthesis_time']:.3f}s")
        print(f"   Queue wait time: {summary['avg_queue_wait']:.3f}s")
        print(f"   Total latency: {summary['avg_total_latency']:.3f}s")
        print(f"   Audio duration: {summary['avg_audio_duration']:.1f}s")
        print()
        print(f"🚀 Synthesis speed: {summary['synthesis_speed_ratio']:.2f}x realtime")
        if summary['synthesis_speed_ratio'] < 1.0:
            print("   ✅ Faster than realtime (good)")
        else:
            print("   ⚠️  Slower than realtime (may cause delays)")
        print("="*60)
    
    def list_devices(self):
        """List available audio devices"""
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")
            print(f"  - Channels: {info['maxInputChannels']} in, {info['maxOutputChannels']} out")
            print(f"  - Sample Rate: {info['defaultSampleRate']} Hz")
            print()
    
    def start_input_stream(self, callback: Optional[Callable] = None):
        """
        Start audio input stream
        
        Args:
            callback: Optional callback function(audio_chunk) for processing
        """
        if self.input_stream is not None:
            self.stop_input_stream()
        
        self.input_callback = callback
        self.is_recording = True
        
        def stream_callback(in_data, frame_count, time_info, status):
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            
            # Add to queue
            self.input_queue.put(audio_chunk)
            
            # Call user callback if provided
            if self.input_callback:
                self.input_callback(audio_chunk)
            
            return (in_data, pyaudio.paContinue)
        
        # Open stream
        self.input_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.chunk_size,
            stream_callback=stream_callback
        )
        
        self.input_stream.start_stream()
        print("Input stream started")
    
    def stop_input_stream(self):
        """Stop audio input stream"""
        self.is_recording = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            print("Input stream stopped")
    
    def start_output_stream(self):
        """Start audio output stream"""
        if self.output_stream is not None:
            self.stop_output_stream()
        
        self.is_playing = True
        
        def stream_callback(in_data, frame_count, time_info, status):
            try:
                # Get audio from queue
                audio_chunk = self.output_queue.get_nowait()
                
                # Ensure correct size
                if len(audio_chunk) < frame_count:
                    # Pad with zeros
                    audio_chunk = np.pad(audio_chunk, (0, frame_count - len(audio_chunk)))
                elif len(audio_chunk) > frame_count:
                    # Trim
                    audio_chunk = audio_chunk[:frame_count]
                
                # Convert to bytes
                data = audio_chunk.astype(np.int16).tobytes()
                
            except queue.Empty:
                # Return silence if no data
                data = np.zeros(frame_count, dtype=np.int16).tobytes()
            
            return (data, pyaudio.paContinue)
        
        # Open stream
        self.output_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.output_device,
            frames_per_buffer=self.chunk_size,
            stream_callback=stream_callback
        )
        
        self.output_stream.start_stream()
        print("Output stream started")
    
    def stop_output_stream(self):
        """Stop audio output stream"""
        self.is_playing = False
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            print("Output stream stopped")
    
    def play_audio(self, audio: np.ndarray, block: bool = False):
        """
        Play audio through output stream
        
        Args:
            audio: Audio data as numpy array
            block: Whether to block until playback completes
        """
        # Ensure output stream is running
        if not self.output_stream:
            self.start_output_stream()
        
        # Convert to int16 if needed
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        
        # Add to output queue in chunks
        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            self.output_queue.put(chunk)
        
        if block:
            # Wait for queue to empty
            while not self.output_queue.empty():
                time.sleep(0.01)
            
            # Wait for last chunk to play
            time.sleep(self.chunk_size / self.sample_rate)
    
    def record_audio(self, duration: float) -> np.ndarray:
        """
        Record audio for specified duration
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio as numpy array
        """
        # Clear input queue
        while not self.input_queue.empty():
            self.input_queue.get()
        
        # Start recording if not already
        if not self.input_stream:
            self.start_input_stream()
        
        # Calculate number of chunks
        chunks_needed = int(duration * self.sample_rate / self.chunk_size)
        audio_chunks = []
        
        # Collect chunks
        for _ in range(chunks_needed):
            chunk = self.input_queue.get(timeout=2.0)
            audio_chunks.append(chunk)
        
        # Concatenate chunks
        audio = np.concatenate(audio_chunks)
        
        # Trim to exact duration
        target_samples = int(duration * self.sample_rate)
        audio = audio[:target_samples]
        
        return audio
    
    def get_input_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get a single chunk from input stream"""
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queues(self):
        """Clear all audio queues and cancel streaming"""
        try:
            # Cancel any ongoing streaming
            self.cancel_streaming.set()
            
            # Wait briefly for streaming thread to notice cancellation
            if self.current_streaming_thread and self.current_streaming_thread.is_alive():
                self.current_streaming_thread.join(timeout=0.5)
            
            # Clear queues
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"⚠️ Error clearing audio queues: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_input_stream()
        self.stop_output_stream()
        self.pa.terminate()
        print("Audio streams cleaned up")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    @staticmethod
    def play_audio_simple(audio: np.ndarray, sample_rate: int):
        """Simple blocking audio playback using sounddevice"""
        sd.play(audio, sample_rate)
        sd.wait()
    
    def play_audio_streaming(self, audio_generator, interrupt_current: bool = True):
        """
        Play audio from a generator/iterator, streaming chunks as they become available
        
        Args:
            audio_generator: Generator yielding (audio_chunk, sample_rate) tuples
            interrupt_current: Whether to clear current queue before starting
        """
        if interrupt_current:
            self.clear_queues()
        
        # Reset cancellation flag
        self.cancel_streaming.clear()
        
        # Ensure output stream is running
        if not self.output_stream:
            self.start_output_stream()
        
        def streaming_thread():
            """Background thread to handle streaming audio"""
            try:
                for audio_chunk, sample_rate in audio_generator:
                    # Check for cancellation
                    if self.cancel_streaming.is_set():
                        print("🚫 Audio streaming cancelled")
                        break
                    
                    # Resample if needed
                    if sample_rate != self.sample_rate:
                        try:
                            import librosa
                            audio_chunk = librosa.resample(
                                audio_chunk.astype(np.float32),
                                orig_sr=sample_rate,
                                target_sr=self.sample_rate
                            )
                        except Exception as e:
                            print(f"⚠️ Error resampling audio: {e}")
                            continue
                    
                    # Convert to int16 if needed
                    if audio_chunk.dtype == np.float32:
                        audio_chunk = (audio_chunk * 32767).astype(np.int16)
                    
                    # Add to output queue in chunks
                    for i in range(0, len(audio_chunk), self.chunk_size):
                        # Check for cancellation before adding each chunk
                        if self.cancel_streaming.is_set():
                            print("🚫 Audio streaming cancelled during chunk processing")
                            return
                        
                        chunk = audio_chunk[i:i + self.chunk_size]
                        try:
                            self.output_queue.put(chunk, timeout=1.0)
                        except queue.Full:
                            print("⚠️ Audio output queue full, dropping chunk")
                            continue
                        
                        # Small delay to prevent overwhelming the queue
                        time.sleep(0.001)
                
                if not self.cancel_streaming.is_set():
                    print("🔊 Streaming audio playback complete")
                
            except Exception as e:
                print(f"❌ Error in streaming audio playback: {e}")
            finally:
                # Ensure we clean up our reference
                if self.current_streaming_thread is threading.current_thread():
                    self.current_streaming_thread = None
        
        # Start streaming in background thread
        thread = threading.Thread(target=streaming_thread, daemon=True)
        thread.start()
        self.current_streaming_thread = thread  # Store reference
        return thread
    
    def queue_audio_chunk(self, audio: np.ndarray, sample_rate: Optional[int] = None):
        """
        Queue a single audio chunk for playback
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate (will resample if different from stream rate)
        """
        # Ensure output stream is running
        if not self.output_stream:
            self.start_output_stream()
        
        # Resample if needed
        if sample_rate and sample_rate != self.sample_rate:
            import librosa
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )
        
        # Convert to int16 if needed
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        
        # Add to output queue in chunks
        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            self.output_queue.put(chunk)
    
    def is_queue_empty(self) -> bool:
        """Check if output queue is empty"""
        return self.output_queue.empty()
    
    def get_queue_size(self) -> int:
        """Get current size of output queue"""
        return self.output_queue.qsize()
    
    def wait_for_playback_complete(self, timeout: float = 30.0):
        """
        Wait for all queued audio to finish playing
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        while not self.output_queue.empty():
            if time.time() - start_time > timeout:
                print("⚠️  Timeout waiting for playback to complete")
                break
            time.sleep(0.01)
        
        # Wait for the last chunk to play out
        time.sleep(self.chunk_size / self.sample_rate) 