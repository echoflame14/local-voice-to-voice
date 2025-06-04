import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import librosa
import time
from chatterbox.tts import ChatterboxTTS


class ChatterboxTTSWrapper:
    """Wrapper for Chatterbox TTS with voice management and optimization"""
    
    def __init__(
        self,
        device: str = None,
        voice_reference_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8
    ):
        """
        Initialize Chatterbox TTS wrapper
        
        Args:
            device: Device to run on (cuda, mps, cpu, or None for auto-detect)
            voice_reference_path: Path to reference voice audio
            exaggeration: Emotion exaggeration (0.25-2.0)
            cfg_weight: CFG/Pace control (0.0-1.0)
            temperature: Sampling temperature
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        
        print(f"Loading Chatterbox TTS on {device}...")
        self.model = ChatterboxTTS.from_pretrained(device=device)
        print("Chatterbox TTS loaded successfully")
        
        # Set voice reference if provided
        print(f"DEBUG: voice_reference_path = {voice_reference_path}")
        print(f"DEBUG: voice_reference_path exists = {Path(voice_reference_path).exists() if voice_reference_path else 'N/A'}")
        if voice_reference_path and Path(voice_reference_path).exists():
            self.set_voice(voice_reference_path)
        else:
            self.voice_reference_path = None
            print("No voice reference set - using default voice")
    
    def set_voice(self, voice_path: str, exaggeration: float = None):
        """
        Set reference voice for synthesis
        
        Args:
            voice_path: Path to reference voice audio file
            exaggeration: Override default exaggeration for this voice
        """
        if not Path(voice_path).exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        # Validate the voice file
        if not self.validate_voice_file(voice_path):
            print(f"⚠️  Warning: Voice file validation failed, but proceeding anyway: {voice_path}")
        
        self.voice_reference_path = voice_path
        if exaggeration is not None:
            self.exaggeration = exaggeration
        
        print(f"✅ Voice reference set to: {voice_path}")
    
    def synthesize(
        self,
        text: str,
        voice_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        return_numpy: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice_path: Override default voice reference
            exaggeration: Override default exaggeration
            cfg_weight: Override default CFG weight
            temperature: Override default temperature
            return_numpy: Return numpy array (True) or torch tensor (False)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Use provided values or defaults
        voice_path = voice_path or self.voice_reference_path
        exaggeration = exaggeration if exaggeration is not None else self.exaggeration
        cfg_weight = cfg_weight if cfg_weight is not None else self.cfg_weight
        temperature = temperature if temperature is not None else self.temperature
        
        # Time the synthesis
        start_time = time.time()
        
        # Add logging for voice cloning verification
        if voice_path:
            print(f"🎭 Using voice cloning with reference: {Path(voice_path).name}")
        else:
            print("🎤 Using default voice (no reference provided)")
        
        # Generate audio
        wav_tensor = self.model.generate(
            text,
            audio_prompt_path=voice_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature
        )
        
        synthesis_time = time.time() - start_time
        print(f"Synthesis completed in {synthesis_time:.2f}s")
        
        # Convert to numpy if requested
        if return_numpy:
            audio = wav_tensor.squeeze().numpy()
        else:
            audio = wav_tensor
        
        return audio, self.model.sr
    
    def synthesize_streaming(
        self,
        text: str,
        chunk_size: int = 50,
        **kwargs
    ):
        """
        Synthesize speech in chunks for lower latency
        
        Args:
            text: Text to synthesize
            chunk_size: Characters per chunk
            **kwargs: Additional arguments for synthesize()
            
        Yields:
            Tuple of (audio_chunk, sample_rate)
        """
        # Split text into chunks at sentence boundaries when possible
        chunks = self._split_text_chunks(text, chunk_size)
        
        for i, chunk in enumerate(chunks):
            # Add slight overlap context for better prosody
            if i > 0:
                context = chunks[i-1][-20:] + " "
                chunk_with_context = context + chunk
            else:
                chunk_with_context = chunk
            
            audio, sr = self.synthesize(chunk_with_context, **kwargs)
            
            # Trim context from audio if added
            if i > 0:
                # Estimate context duration and trim
                context_ratio = len(context) / len(chunk_with_context)
                trim_samples = int(len(audio) * context_ratio)
                audio = audio[trim_samples:]
            
            yield audio, sr
    
    def synthesize_sentences(
        self,
        text: str,
        **kwargs
    ):
        """
        Synthesize speech sentence by sentence for optimal streaming
        
        Args:
            text: Text to synthesize
            **kwargs: Additional arguments for synthesize()
            
        Yields:
            Tuple of (sentence_audio, sample_rate, sentence_text)
        """
        sentences = self._split_into_sentences(text)
        
        print(f"🔤 Streaming {len(sentences)} sentences...")
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            print(f"🎵 Synthesizing sentence {i+1}/{len(sentences)}: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
            
            # Add context from previous sentence for better prosody
            if i > 0 and len(sentences) > 1:
                # Add a bit of context from the previous sentence
                prev_words = sentences[i-1].split()[-3:] if sentences[i-1] else []
                context = " ".join(prev_words) + " " if prev_words else ""
                sentence_with_context = context + sentence
            else:
                sentence_with_context = sentence
            
            start_time = time.time()
            audio, sr = self.synthesize(sentence_with_context, **kwargs)
            synthesis_time = time.time() - start_time
            
            # Trim context if added
            if i > 0 and len(sentences) > 1:
                context_ratio = len(context) / len(sentence_with_context) if sentence_with_context else 0
                trim_samples = int(len(audio) * context_ratio)
                audio = audio[trim_samples:] if trim_samples > 0 else audio
            
            print(f"   ✅ Sentence {i+1} ready ({synthesis_time:.2f}s, {len(audio)/sr:.1f}s audio)")
            
            yield audio, sr, sentence
    
    def _split_text_chunks(self, text: str, chunk_size: int) -> list:
        """Split text into chunks, preferring sentence boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into individual sentences"""
        import re
        
        # More sophisticated sentence splitting
        # Handle common abbreviations and edge cases
        text = text.strip()
        if not text:
            return []
        
        # Replace common abbreviations to avoid false splits
        abbrevs = {
            'Mr.': 'Mr<DOT>',
            'Mrs.': 'Mrs<DOT>',
            'Dr.': 'Dr<DOT>',
            'Ms.': 'Ms<DOT>',
            'Prof.': 'Prof<DOT>',
            'vs.': 'vs<DOT>',
            'etc.': 'etc<DOT>',
            'i.e.': 'i.e<DOT>',
            'e.g.': 'e.g<DOT>',
        }
        
        for abbrev, replacement in abbrevs.items():
            text = text.replace(abbrev, replacement)
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Restore abbreviations and clean up
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Restore abbreviations
                for abbrev, replacement in abbrevs.items():
                    sentence = sentence.replace(replacement, abbrev)
                
                # Add appropriate ending punctuation if missing
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def validate_voice_file(self, voice_path: str) -> bool:
        """
        Validate that a voice file is compatible with the TTS model
        
        Args:
            voice_path: Path to voice file to validate
            
        Returns:
            True if voice file is valid and compatible
        """
        try:
            voice_file = Path(voice_path)
            if not voice_file.exists():
                print(f"❌ Voice file not found: {voice_path}")
                return False
            
            # Load and check audio properties
            audio, sr = librosa.load(voice_path, sr=None)
            duration = len(audio) / sr
            
            print(f"✅ Voice file validation for {voice_file.name}:")
            print(f"   📏 Duration: {duration:.2f} seconds")
            print(f"   🔊 Sample rate: {sr} Hz")
            print(f"   📊 Channels: {1 if audio.ndim == 1 else audio.shape[1]}")
            print(f"   💾 Size: {voice_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Check if duration is reasonable for voice cloning
            if duration < 3:
                print(f"⚠️  Warning: Voice sample is quite short ({duration:.1f}s). 5-15 seconds recommended.")
            elif duration > 30:
                print(f"⚠️  Warning: Voice sample is quite long ({duration:.1f}s). Consider trimming to 5-15 seconds.")
            else:
                print(f"✅ Voice sample length is optimal for cloning")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating voice file {voice_path}: {e}")
            return False
    
    def preload_voice(self, voice_path: str):
        """Preload voice embedding for faster first synthesis"""
        if self.validate_voice_file(voice_path):
            self.model.prepare_conditionals(voice_path, exaggeration=self.exaggeration)
            print(f"Voice preloaded: {voice_path}")
        else:
            print(f"❌ Failed to preload voice: {voice_path}")
    
    @property
    def sample_rate(self) -> int:
        """Get the model's output sample rate"""
        return self.model.sr 