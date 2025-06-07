"""
Windows Speech Recognition fallback
Uses Windows built-in speech recognition
"""
import numpy as np
from typing import Optional
import speech_recognition as sr


class WindowsSTT:
    """Windows Speech Recognition wrapper"""
    
    def __init__(self, config):
        self.recognizer = sr.Recognizer()
        print("Using Windows Speech Recognition")
        
    def load(self):
        """No loading needed for Windows SR"""
        pass
        
    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe using Windows Speech Recognition"""
        try:
            # Convert numpy array to AudioData
            # speech_recognition expects 16-bit PCM at 16kHz
            audio_bytes = audio_data.astype(np.int16).tobytes()
            
            # Create AudioData object
            audio = sr.AudioData(audio_bytes, 16000, 2)
            
            print("  Using Windows Speech Recognition...")
            # Try recognition
            try:
                # Try Google first (online)
                text = self.recognizer.recognize_google(audio)
                print(f"  Google result: '{text}'")
                return text
            except sr.UnknownValueError:
                print("  Google couldn't understand audio")
            except sr.RequestError as e:
                print(f"  Google error: {e}")
                
            # Fallback to offline Windows recognition
            try:
                text = self.recognizer.recognize_sphinx(audio)
                print(f"  Sphinx result: '{text}'")
                return text
            except sr.UnknownValueError:
                print("  Sphinx couldn't understand audio")
                return None
            except Exception as e:
                print(f"  Sphinx error: {e}")
                return None
                
        except Exception as e:
            print(f"  Windows STT error: {e}")
            return None