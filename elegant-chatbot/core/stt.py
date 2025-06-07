"""
Speech-to-Text module
Simple Whisper wrapper
"""
import numpy as np
import whisper
from typing import Optional


class WhisperSTT:
    """Simple Whisper STT wrapper"""
    
    def __init__(self, config):
        self.model_name = config.model.whisper_model
        self.device = config.model.whisper_device
        self.model = None
        
    def load(self):
        """Load Whisper model"""
        print(f"Loading Whisper {self.model_name} model...")
        self.model = whisper.load_model(self.model_name, device=self.device)
        print("Whisper model loaded!")
        
    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio to text"""
        if self.model is None:
            self.load()
            
        if len(audio_data) == 0:
            return None
            
        # Convert to float32 [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        print(f"  Audio shape: {audio_float.shape}, duration: {len(audio_float)/16000:.1f}s")
        print(f"  Using model: {self.model_name} on {self.device}")
        
        try:
            # Transcribe with verbose output
            result = self.model.transcribe(
                audio_float,
                language="en",
                fp16=False,
                verbose=False  # Set to True for more debug info
            )
            
            text = result["text"].strip()
            print(f"  Transcription result: '{text}'")
            return text if text else None
        except Exception as e:
            print(f"  Transcription error: {e}")
            return None