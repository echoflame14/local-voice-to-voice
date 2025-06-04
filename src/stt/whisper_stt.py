import whisper
import numpy as np
import torch
from typing import Union, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class WhisperSTT:
    """Speech-to-Text using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base", device: str = None):
        """
        Initialize Whisper STT
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, mps, cpu, or None for auto-detect)
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_size = model_size
        print(f"Loading Whisper {model_size} model on {device}...")
        
        self.model = whisper.load_model(model_size, device=device)
        print("Whisper model loaded successfully")
    
    def transcribe(
        self, 
        audio: Union[np.ndarray, str], 
        language: str = None,
        initial_prompt: str = None,
        temperature: Union[float, Tuple[float, ...]] = 0.0,
        condition_on_previous_text: bool = True,
        fp16: bool = True,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6
    ) -> dict:
        """
        Transcribe audio to text using OpenAI Whisper
        
        Args:
            audio: Audio array (float32) or path to audio file
            language: Language code (e.g., 'en', 'es'). None for auto-detect
            initial_prompt: Optional prompt to guide transcription
            temperature: Sampling temperature - can be float or tuple for progressive fallback
            condition_on_previous_text: Whether to use previous text as context
            fp16: Whether to use fp16 for inference (faster on GPU)
            compression_ratio_threshold: If gzip compression ratio > this, treat as failed
            logprob_threshold: If average log probability < this, treat as failed  
            no_speech_threshold: If no-speech probability > this, consider as silence
            
        Returns:
            Dictionary with transcription results including text, segments, and language info
        """
        # Prepare audio
        if isinstance(audio, np.ndarray):
            # Ensure float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed (int16 to float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            fp16=fp16,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold
        )
        
        return result
    
    def transcribe_with_timing(
        self,
        audio: Union[np.ndarray, str],
        **kwargs
    ) -> Tuple[str, list]:
        """
        Transcribe with word-level timestamps
        
        Returns:
            Tuple of (text, segments) where segments contain timing info
        """
        kwargs['word_timestamps'] = True
        result = self.transcribe(audio, **kwargs)
        
        return result['text'].strip(), result.get('segments', [])
    
    def detect_language(self, audio: Union[np.ndarray, str]) -> Tuple[str, float]:
        """
        Detect the language of the audio
        
        Returns:
            Tuple of (language_code, probability)
        """
        # Use first 30 seconds to detect language
        result = self.model.detect_language(audio)
        
        # Get the most likely language
        lang_code = max(result, key=result.get)
        probability = result[lang_code]
        
        return lang_code, probability 