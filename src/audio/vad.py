import webrtcvad
import numpy as np
from collections import deque
from typing import Tuple, Optional
import librosa


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 1,
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.9,
        ring_buffer_frames: int = None  # NEW: configurable buffer size for silence detection
    ):
        """
        Initialize VAD
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in ms (10, 20, or 30)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            speech_threshold: Ratio of speech frames to trigger speech start
            silence_threshold: Ratio of silence frames to trigger speech end
            ring_buffer_frames: Optional ring buffer size for silence detection
        """
        # Validate parameters
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000")
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms")
        
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.aggressiveness = aggressiveness
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Ring buffer for smoothing decisions
        # If not specified, use the historical default of 10 frames (~300 ms)
        if ring_buffer_frames is None:
            ring_buffer_frames = 10

        self.ring_buffer_size = ring_buffer_frames
        self.ring_buffer = deque(maxlen=self.ring_buffer_size)
        
        # State tracking
        self.is_speaking = False
        self.speech_buffer = []
        
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[bool, bool]:
        """
        Process a single audio frame
        
        Args:
            audio_frame: Audio frame as int16 numpy array
            
        Returns:
            Tuple of (is_speech, state_changed)
        """
        # Ensure correct frame size
        if len(audio_frame) != self.frame_size:
            raise ValueError(f"Frame size must be {self.frame_size} samples")
        
        # Convert to bytes for WebRTC VAD
        audio_bytes = audio_frame.astype(np.int16).tobytes()
        
        # Detect speech
        is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
        
        # Add to ring buffer
        self.ring_buffer.append(1 if is_speech else 0)
        
        # Calculate speech ratio
        if len(self.ring_buffer) < self.ring_buffer_size:
            return is_speech, False
        
        speech_ratio = sum(self.ring_buffer) / len(self.ring_buffer)
        
        # Determine state change
        old_state = self.is_speaking
        
        if not self.is_speaking and speech_ratio > self.speech_threshold:
            self.is_speaking = True
        elif self.is_speaking and speech_ratio < (1 - self.silence_threshold):
            self.is_speaking = False
        
        state_changed = old_state != self.is_speaking
        
        return self.is_speaking, state_changed
    
    def process_audio(
        self,
        audio: np.ndarray,
        return_segments: bool = False
    ) -> Tuple[np.ndarray, Optional[list]]:
        """
        Process audio and extract speech segments
        
        Args:
            audio: Audio as numpy array (float32 or int16)
            return_segments: Whether to return segment timestamps
            
        Returns:
            Tuple of (speech_audio, segments) where segments is list of (start, end) tuples
        """
        # Convert to int16 if needed
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        
        # Pad audio to frame boundary
        remainder = len(audio) % self.frame_size
        if remainder != 0:
            pad_size = self.frame_size - remainder
            audio = np.pad(audio, (0, pad_size), mode='constant')
        
        # Process frames
        speech_frames = []
        segments = []
        current_segment_start = None
        
        for i in range(0, len(audio), self.frame_size):
            frame = audio[i:i + self.frame_size]
            is_speech, state_changed = self.process_frame(frame)
            
            if is_speech:
                speech_frames.append(frame)
                
                # Track segment start
                if state_changed and current_segment_start is None:
                    current_segment_start = i / self.sample_rate
            
            # Track segment end
            if state_changed and not is_speech and current_segment_start is not None:
                segments.append((current_segment_start, i / self.sample_rate))
                current_segment_start = None
        
        # Close final segment if needed
        if current_segment_start is not None:
            segments.append((current_segment_start, len(audio) / self.sample_rate))
        
        # Concatenate speech frames
        if speech_frames:
            speech_audio = np.concatenate(speech_frames)
        else:
            speech_audio = np.array([], dtype=np.int16)
        
        if return_segments:
            return speech_audio, segments
        else:
            return speech_audio, None
    
    def reset(self):
        """Reset VAD state"""
        self.ring_buffer.clear()
        self.is_speaking = False
        self.speech_buffer.clear()
    
    @staticmethod
    def resample_for_vad(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        """Resample audio for VAD processing"""
        if orig_sr == target_sr:
            return audio
        
        # Use librosa for high-quality resampling
        resampled = librosa.resample(
            audio.astype(np.float32),
            orig_sr=orig_sr,
            target_sr=target_sr
        )
        
        return resampled 