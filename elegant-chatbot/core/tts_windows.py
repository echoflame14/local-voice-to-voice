"""
Windows-specific TTS implementation using SAPI
"""
import numpy as np
import tempfile
import wave
import os
import time
from typing import Optional

class WindowsTTS:
    """Windows SAPI TTS implementation"""
    
    def __init__(self, config):
        self.config = config
        self.speaker = None
        
        try:
            import win32com.client
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
            print("Windows SAPI TTS initialized")
        except ImportError:
            print("pywin32 not installed. Run: pip install pywin32")
        except Exception as e:
            print(f"Failed to initialize Windows SAPI: {e}")
    
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio using Windows SAPI"""
        if not self.speaker:
            # Return silence if SAPI not available
            return np.zeros(int(self.config.audio.sample_rate * 0.5), dtype=np.int16)
        
        tmp_path = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Configure file stream
            self.file_stream.Open(tmp_path, 3)  # 3 = SSFMCreateForWrite
            
            # Set output to file
            old_output = self.speaker.AudioOutputStream
            self.speaker.AudioOutputStream = self.file_stream
            
            # Speak to file
            self.speaker.Speak(text)
            
            # Restore output and close file
            self.speaker.AudioOutputStream = old_output
            self.file_stream.Close()
            
            # Wait for file to be written
            time.sleep(0.1)
            
            # Load audio
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with wave.open(tmp_path, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)
                    
                    # Resample if needed (SAPI usually outputs 22050 Hz)
                    if wav.getframerate() != self.config.audio.sample_rate:
                        # Simple resampling
                        ratio = self.config.audio.sample_rate / wav.getframerate()
                        new_length = int(len(audio) * ratio)
                        indices = np.arange(new_length) / ratio
                        indices = indices.astype(int)
                        indices = np.clip(indices, 0, len(audio) - 1)
                        audio = audio[indices]
                    
                    return audio
            else:
                print("SAPI failed to create audio file")
                return np.zeros(int(self.config.audio.sample_rate * 0.5), dtype=np.int16)
                
        except Exception as e:
            print(f"Windows TTS error: {e}")
            return np.zeros(int(self.config.audio.sample_rate * 0.5), dtype=np.int16)
        finally:
            # Cleanup
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    def speak_direct(self, text: str):
        """Speak directly through speakers (for testing)"""
        if self.speaker:
            self.speaker.Speak(text)