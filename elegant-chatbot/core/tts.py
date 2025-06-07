"""
Text-to-Speech module
Simple TTS wrapper with optional Chatterbox support
"""
import numpy as np
from typing import Optional
import pyttsx3  # Fallback TTS


class SimpleTTS:
    """Simple TTS with fallback options"""
    
    def __init__(self, config):
        self.config = config
        self.voice_path = config.model.tts_voice
        self.speed = config.model.tts_speed
        
        # Try to load Chatterbox if available
        self.chatterbox = None
        try:
            from chatterbox import Chatterbox
            self.chatterbox = Chatterbox()
            if self.voice_path:
                self.chatterbox.set_voice(self.voice_path)
            print("Chatterbox TTS loaded")
        except ImportError:
            print("Chatterbox not available, trying fallbacks")
        
        # Try Windows SAPI on Windows
        self.windows_tts = None
        if not self.chatterbox:
            try:
                from .tts_windows import WindowsTTS
                self.windows_tts = WindowsTTS(config)
            except Exception as e:
                print(f"Windows TTS not available: {e}")
            
        # Fallback to pyttsx3
        self.engine = None
        if not self.chatterbox and not self.windows_tts:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', int(150 * self.speed))
                print("Using pyttsx3 TTS")
            except Exception as e:
                print(f"pyttsx3 init failed: {e}")
            
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio"""
        if self.chatterbox:
            return self._synthesize_chatterbox(text)
        elif self.windows_tts:
            return self.windows_tts.synthesize(text)
        elif self.engine:
            return self._synthesize_pyttsx3(text)
        else:
            print("No TTS engine available, returning silence")
            return np.zeros(int(self.config.audio.sample_rate * 0.5), dtype=np.int16)
            
    def _synthesize_chatterbox(self, text: str) -> np.ndarray:
        """Use Chatterbox for synthesis"""
        try:
            audio = self.chatterbox.synthesize(
                text,
                speed=self.speed,
                temperature=0.5
            )
            return audio
        except Exception as e:
            print(f"Chatterbox error: {e}, falling back")
            return self._synthesize_pyttsx3(text)
            
    def _synthesize_pyttsx3(self, text: str) -> np.ndarray:
        """Fallback TTS synthesis"""
        # Save to temp file and load
        import tempfile
        import wave
        import os
        import time
        
        tmp_path = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Generate audio
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()
            
            # Wait a bit for file to be written
            time.sleep(0.1)
            
            # Check if file exists and has content
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                print(f"Warning: TTS failed to create audio file")
                # Return silence
                return np.zeros(int(self.config.audio.sample_rate * 0.5), dtype=np.int16)
            
            # Load audio
            try:
                with wave.open(tmp_path, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)
                    return audio
            except Exception as e:
                print(f"Warning: Failed to read audio file: {e}")
                # Return silence
                return np.zeros(int(self.config.audio.sample_rate * 0.5), dtype=np.int16)
                
        except Exception as e:
            print(f"TTS error: {e}")
            # Return silence as fallback
            return np.zeros(int(self.config.audio.sample_rate * 0.5), dtype=np.int16)
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass