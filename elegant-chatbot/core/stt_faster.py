"""
Speech-to-Text module using faster-whisper
Much more efficient than OpenAI Whisper
"""
import numpy as np
from typing import Optional
from faster_whisper import WhisperModel


class FasterWhisperSTT:
    """Faster-whisper STT wrapper"""
    
    def __init__(self, config):
        self.model_name = config.model.whisper_model
        self.device = config.model.whisper_device
        self.model = None
        
    def load(self):
        """Load faster-whisper model"""
        print(f"Loading faster-whisper {self.model_name} model...")
        
        # Compute type based on device
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        self.model = WhisperModel(
            self.model_name, 
            device=self.device,
            compute_type=compute_type,
            num_workers=1,  # Single worker for lower latency
            cpu_threads=4   # Use 4 CPU threads
        )
        print(f"Faster-whisper model loaded! (using {compute_type} on {self.device})")
        
    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio to text using faster-whisper"""
        if self.model is None:
            self.load()
            
        if len(audio_data) == 0:
            return None
            
        # Convert to float32 [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        print(f"  Transcribing {len(audio_float)/16000:.1f}s of audio...")
        
        try:
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_float,
                beam_size=5,
                language="en",
                vad_filter=True,  # Use VAD to remove silence
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )
            )
            
            # Collect all text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
                
            text = " ".join(text_parts).strip()
            
            if text:
                print(f"  Transcription: '{text}'")
                print(f"  Language: {info.language} (probability: {info.language_probability:.2f})")
                
            return text if text else None
            
        except Exception as e:
            print(f"  Transcription error: {e}")
            return None