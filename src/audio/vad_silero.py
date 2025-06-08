import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional, Union
import warnings

# Suppress torch warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SileroVAD:
    """Voice Activity Detection using Silero VAD
    
    Silero VAD is a pre-trained enterprise-grade Voice Activity Detector
    that's much more accurate than WebRTC VAD and specifically designed
    to handle background noise better.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        device: str = 'cpu'
    ):
        """
        Initialize Silero VAD
        
        Args:
            sample_rate: Audio sample rate (8000 or 16000 Hz)
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to trigger detection
            min_silence_duration_ms: Minimum silence duration to end speech
            window_size_samples: Window size for processing
            speech_pad_ms: Padding to add to speech segments
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        
        # Load Silero VAD model
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            self.model = self.model.to(device)
            self.model.eval()
            print("🎯 Silero VAD loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Silero VAD: {e}")
            raise
        
        # Get utility functions
        self.get_speech_timestamps = self.utils[0]
        
        # State tracking
        self.is_speech = False
        self.temp_end = 0
        self.current_speech_timestamp = 0
        self.speech_start_timestamp = 0
        self.speech_end_timestamp = 0
        
        # Convert durations to samples
        self.min_speech_samples = int(min_speech_duration_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        
        # Audio buffer for processing
        self.audio_buffer = []
        self.processed_samples = 0
        
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Process a single audio chunk and return speech probability
        
        Args:
            audio_chunk: Audio chunk as numpy array (int16 or float32)
            
        Returns:
            Tuple of (is_speech, speech_probability)
        """
        # Convert to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_float).to(self.device)
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        # Determine if speech based on threshold
        is_speech = speech_prob >= self.threshold
        
        return is_speech, speech_prob
    
    def process_audio_stream(self, audio_chunk: np.ndarray) -> Tuple[bool, bool, float]:
        """
        Process streaming audio with state management
        
        Args:
            audio_chunk: Audio chunk as numpy array
            
        Returns:
            Tuple of (is_speech, state_changed, confidence)
        """
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Process in windows
        results = []
        while len(self.audio_buffer) >= self.window_size_samples:
            window = np.array(self.audio_buffer[:self.window_size_samples])
            is_speech, confidence = self.process_chunk(window)
            results.append((is_speech, confidence))
            
            # Slide window
            self.audio_buffer = self.audio_buffer[self.window_size_samples//2:]
            self.processed_samples += self.window_size_samples//2
        
        if not results:
            return self.is_speech, False, 0.0
        
        # Average results
        avg_speech = sum(r[0] for r in results) / len(results) >= 0.5
        avg_confidence = sum(r[1] for r in results) / len(results)
        
        # State management
        old_state = self.is_speech
        current_timestamp = self.processed_samples
        
        if avg_speech:
            if not self.is_speech:
                # Potential speech start
                if self.speech_start_timestamp == 0:
                    self.speech_start_timestamp = current_timestamp
                
                # Check if we've had enough speech
                if current_timestamp - self.speech_start_timestamp >= self.min_speech_samples:
                    self.is_speech = True
                    self.temp_end = 0
                    self.speech_end_timestamp = 0
            else:
                # Continue speech
                self.temp_end = 0
                self.speech_end_timestamp = 0
        else:
            if self.is_speech:
                # Potential speech end
                if self.speech_end_timestamp == 0:
                    self.speech_end_timestamp = current_timestamp
                
                # Check if we've had enough silence
                if current_timestamp - self.speech_end_timestamp >= self.min_silence_samples:
                    self.is_speech = False
                    self.speech_start_timestamp = 0
            else:
                # Continue silence
                self.speech_start_timestamp = 0
        
        state_changed = old_state != self.is_speech
        
        return self.is_speech, state_changed, avg_confidence
    
    def reset(self):
        """Reset VAD state"""
        self.is_speech = False
        self.temp_end = 0
        self.current_speech_timestamp = 0
        self.speech_start_timestamp = 0
        self.speech_end_timestamp = 0
        self.audio_buffer = []
        self.processed_samples = 0


class EnergyBasedVAD:
    """Simple energy-based VAD as a fallback option
    
    Uses RMS energy and zero-crossing rate for voice detection.
    More reliable than WebRTC VAD for filtering background noise.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.02,
        zcr_threshold: float = 0.1,
        speech_frames_threshold: int = 3,
        silence_frames_threshold: int = 10
    ):
        """
        Initialize Energy-based VAD
        
        Args:
            sample_rate: Audio sample rate
            frame_duration_ms: Frame duration in milliseconds
            energy_threshold: RMS energy threshold for speech
            zcr_threshold: Zero-crossing rate threshold
            speech_frames_threshold: Consecutive frames to trigger speech
            silence_frames_threshold: Consecutive frames to trigger silence
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.speech_frames_threshold = speech_frames_threshold
        self.silence_frames_threshold = silence_frames_threshold
        
        # State tracking
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        
        # Adaptive threshold
        self.noise_frames = deque(maxlen=30)
        self.adaptive_threshold = energy_threshold
        
    def compute_rms(self, frame: np.ndarray) -> float:
        """Compute RMS energy of audio frame"""
        return np.sqrt(np.mean(frame.astype(np.float32) ** 2))
    
    def compute_zcr(self, frame: np.ndarray) -> float:
        """Compute zero-crossing rate"""
        signs = np.sign(frame)
        signs[signs == 0] = -1
        return np.sum(signs[:-1] != signs[1:]) / (2 * len(frame))
    
    def update_noise_profile(self, energy: float):
        """Update noise profile for adaptive thresholding"""
        if not self.is_speaking:
            self.noise_frames.append(energy)
            if len(self.noise_frames) > 10:
                noise_level = np.percentile(list(self.noise_frames), 90)
                self.adaptive_threshold = max(
                    self.energy_threshold,
                    noise_level * 2.5  # Speech should be 2.5x louder than noise
                )
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[bool, bool]:
        """
        Process a single audio frame
        
        Args:
            audio_frame: Audio frame as numpy array (int16)
            
        Returns:
            Tuple of (is_speech, state_changed)
        """
        # Ensure correct frame size
        if len(audio_frame) != self.frame_size:
            raise ValueError(f"Frame size must be {self.frame_size} samples")
        
        # Normalize to float32
        if audio_frame.dtype == np.int16:
            frame_float = audio_frame.astype(np.float32) / 32768.0
        else:
            frame_float = audio_frame.astype(np.float32)
        
        # Compute features
        energy = self.compute_rms(frame_float)
        zcr = self.compute_zcr(frame_float)
        
        # Update noise profile
        self.update_noise_profile(energy)
        
        # Detect speech
        is_speech = (
            energy > self.adaptive_threshold and
            zcr < self.zcr_threshold  # Speech has lower ZCR than noise
        )
        
        # State management
        old_state = self.is_speaking
        
        if is_speech:
            self.speech_frame_count += 1
            self.silence_frame_count = 0
            
            if self.speech_frame_count >= self.speech_frames_threshold:
                self.is_speaking = True
        else:
            self.silence_frame_count += 1
            self.speech_frame_count = 0
            
            if self.silence_frame_count >= self.silence_frames_threshold:
                self.is_speaking = False
        
        state_changed = old_state != self.is_speaking
        
        return self.is_speaking, state_changed
    
    def reset(self):
        """Reset VAD state"""
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.noise_frames.clear()
        self.adaptive_threshold = self.energy_threshold


# Backward compatibility wrapper
class VoiceActivityDetector:
    """Wrapper class to maintain compatibility with existing code"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 1,  # Ignored, for compatibility
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.8,  # Ignored, for compatibility
        ring_buffer_frames: int = None,  # Ignored, for compatibility
        use_silero: bool = True
    ):
        """Initialize VAD with Silero or Energy-based backend"""
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        if use_silero:
            try:
                # Try to use Silero VAD
                self.backend = SileroVAD(
                    sample_rate=sample_rate,
                    threshold=speech_threshold,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100,
                    window_size_samples=512 if sample_rate == 16000 else 256
                )
                self.backend_type = 'silero'
                print("✅ Using Silero VAD (much better than WebRTC!)")
            except Exception as e:
                print(f"⚠️ Silero VAD not available: {e}")
                print("🔄 Falling back to Energy-based VAD")
                self.backend = EnergyBasedVAD(
                    sample_rate=sample_rate,
                    frame_duration_ms=frame_duration_ms,
                    energy_threshold=0.02,
                    speech_frames_threshold=3,
                    silence_frames_threshold=10
                )
                self.backend_type = 'energy'
        else:
            # Use energy-based VAD
            self.backend = EnergyBasedVAD(
                sample_rate=sample_rate,
                frame_duration_ms=frame_duration_ms,
                energy_threshold=0.02,
                speech_frames_threshold=3,
                silence_frames_threshold=10
            )
            self.backend_type = 'energy'
        
        # Compatibility attributes
        self.is_speaking = False
        self.speech_buffer = []
        self.ring_buffer = deque(maxlen=20)  # For compatibility
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[bool, bool]:
        """Process a single audio frame"""
        if self.backend_type == 'silero':
            # Silero expects streaming chunks
            is_speech, state_changed, confidence = self.backend.process_audio_stream(audio_frame)
            self.is_speaking = is_speech
            return is_speech, state_changed
        else:
            # Energy-based VAD
            is_speech, state_changed = self.backend.process_frame(audio_frame)
            self.is_speaking = is_speech
            return is_speech, state_changed
    
    def process_audio(
        self,
        audio: np.ndarray,
        return_segments: bool = False
    ) -> Tuple[np.ndarray, Optional[list]]:
        """Process audio and extract speech segments (compatibility method)"""
        # For compatibility, just return the audio as-is
        # The real VAD happens in process_frame
        return audio, None
    
    def reset(self):
        """Reset VAD state"""
        self.backend.reset()
        self.is_speaking = False
        self.speech_buffer.clear()
        self.ring_buffer.clear()
    
    @staticmethod
    def resample_for_vad(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        """Resample audio for VAD processing (compatibility method)"""
        if orig_sr == target_sr:
            return audio
        
        # Simple decimation/interpolation
        if orig_sr > target_sr:
            # Downsample
            factor = orig_sr // target_sr
            return audio[::factor]
        else:
            # Upsample (simple repeat)
            factor = target_sr // orig_sr
            return np.repeat(audio, factor)