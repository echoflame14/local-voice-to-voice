import numpy as np
from typing import Tuple, Dict, Callable
import math


class SoundEffects:
    """Simple sound effect generator for UI feedback"""
    
    @staticmethod
    def generate_beep(
        frequency: float = 800.0,
        duration: float = 0.2,
        sample_rate: int = 16000,
        volume: float = 0.3
    ) -> np.ndarray:
        """Generate a simple beep tone"""
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Generate sine wave
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope to avoid clicks
        envelope = np.exp(-t * 5)  # Exponential decay
        wave = wave * envelope * volume
        
        return wave.astype(np.float32)
    
    @staticmethod
    def generate_chime(
        frequencies: Tuple[float, ...] = (523.25, 659.25, 783.99),  # C, E, G
        duration: float = 0.4,
        sample_rate: int = 16000,
        volume: float = 0.2
    ) -> np.ndarray:
        """Generate a pleasant chime sound"""
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Combine multiple frequencies
        wave = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            # Each frequency decays at different rate
            decay = np.exp(-t * (3 + i))
            wave += np.sin(2 * np.pi * freq * t) * decay
        
        # Normalize and apply volume
        wave = wave / len(frequencies) * volume
        
        return wave.astype(np.float32)
    
    @staticmethod
    def generate_notification(
        base_freq: float = 1000.0,
        duration: float = 0.15,
        sample_rate: int = 16000,
        volume: float = 0.25
    ) -> np.ndarray:
        """Generate a quick notification sound"""
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Rising frequency sweep
        freq_sweep = base_freq + (base_freq * 0.5 * t / duration)
        wave = np.sin(2 * np.pi * freq_sweep * t)
        
        # Quick fade envelope
        envelope = np.exp(-t * 10)
        wave = wave * envelope * volume
        
        return wave.astype(np.float32)
    
    @staticmethod
    def generate_interrupt_sound(sample_rate: int = 16000, duration: float = 0.15, volume: float = 0.2):
        """Generate a short interruption sound effect"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = np.linspace(1000, 500, len(t))  # Downward sweep
        sound = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        envelope = np.exp(-4 * t / duration)
        sound = sound * envelope * volume
        
        return sound.astype(np.float32)
    
    @staticmethod
    def generate_generation_start_sound(sample_rate: int = 16000, duration: float = 0.15, volume: float = 0.2):
        """Generate a short sound effect for generation start"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = np.linspace(500, 1000, len(t))  # Upward sweep
        sound = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        envelope = np.exp(-4 * t / duration)
        sound = sound * envelope * volume
        
        return sound.astype(np.float32)
    
    @staticmethod
    def generate_completion_sound(sample_rate: int = 16000, duration: float = 0.3, volume: float = 0.2):
        """Generate a pleasant completion sound effect"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create two tones for a pleasant chord
        f1, f2 = 800, 1000  # Frequencies for the two tones
        sound1 = np.sin(2 * np.pi * f1 * t)
        sound2 = np.sin(2 * np.pi * f2 * t)
        
        # Combine the tones
        sound = (sound1 + sound2) / 2
        
        # Apply a more musical envelope (ADSR-like)
        attack = 0.1  # Attack portion of the duration
        decay = 0.2   # Decay portion of the duration
        sustain = 0.4 # Sustain level
        
        envelope = np.ones_like(t)
        attack_samples = int(len(t) * attack)
        decay_samples = int(len(t) * decay)
        
        # Attack phase - linear ramp up
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase - exponential decay to sustain level
        decay_curve = np.exp(-3 * np.linspace(0, 1, decay_samples))
        decay_curve = (1 - sustain) * decay_curve + sustain
        envelope[attack_samples:attack_samples + decay_samples] = decay_curve
        
        # Release phase - exponential decay to zero
        release_curve = np.exp(-5 * np.linspace(0, 1, len(t) - attack_samples - decay_samples))
        envelope[attack_samples + decay_samples:] = sustain * release_curve
        
        # Apply envelope and volume
        sound = sound * envelope * volume
        
        return sound.astype(np.float32)
    
    @staticmethod
    def generate_processing_sound(sample_rate: int = 16000, duration: float = 0.2, volume: float = 0.2):
        """Generate a processing start sound effect"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Rising tone sequence for processing indication
        f1 = 600 + 200 * np.sin(2 * np.pi * 3 * t)  # Wobbling frequency
        sound = np.sin(2 * np.pi * f1 * t)
        
        # Apply envelope
        envelope = np.exp(-2 * t / duration) * (1 - np.exp(-10 * t / duration))
        sound = sound * envelope * volume
        
        return sound.astype(np.float32)
    
    @staticmethod
    def generate_transcription_complete_sound(sample_rate: int = 16000, duration: float = 0.2, volume: float = 0.15):
        """Generate a quick confirmation sound for transcription completion"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Quick ascending chirp to indicate completion
        freq = 800 + 400 * (t / duration)  # Rise from 800Hz to 1200Hz
        sound = np.sin(2 * np.pi * freq * t)
        
        # Sharp attack, quick decay envelope
        envelope = np.exp(-5 * t / duration) * (1 - np.exp(-30 * t / duration))
        sound = sound * envelope * volume
        
        return sound.astype(np.float32)
    
    @staticmethod
    def generate_contextual_sound(sound_type: str, theme: str = "modern", sample_rate: int = 16000, volume: float = 0.2):
        """Generate contextual sound effects based on type and theme"""
        
        if theme == "modern":
            if sound_type == "interrupt_gentle":
                return SoundEffects._generate_modern_interrupt_gentle(sample_rate, volume)
            elif sound_type == "interrupt_urgent":
                return SoundEffects._generate_modern_interrupt_urgent(sample_rate, volume)
            elif sound_type == "completion_success":
                return SoundEffects._generate_modern_completion_success(sample_rate, volume)
            elif sound_type == "processing_progress":
                return SoundEffects._generate_modern_processing_progress(sample_rate, volume)
            elif sound_type == "transcription_complete":
                return SoundEffects._generate_modern_transcription_complete(sample_rate, volume)
        
        elif theme == "classic":
            # Classic sounds (simpler tones)
            if sound_type == "interrupt_gentle":
                return SoundEffects.generate_beep(frequency=600, duration=0.1, sample_rate=sample_rate, volume=volume)
            elif sound_type == "completion_success":
                return SoundEffects.generate_chime(sample_rate=sample_rate, volume=volume)
            elif sound_type == "transcription_complete":
                return SoundEffects.generate_beep(frequency=1000, duration=0.1, sample_rate=sample_rate, volume=volume*0.7)
        
        elif theme == "minimal":
            # Minimal sounds (very brief)
            if sound_type == "interrupt_gentle":
                return SoundEffects.generate_beep(frequency=800, duration=0.05, sample_rate=sample_rate, volume=volume*0.5)
            elif sound_type == "completion_success":
                return SoundEffects.generate_beep(frequency=1000, duration=0.1, sample_rate=sample_rate, volume=volume*0.5)
            elif sound_type == "transcription_complete":
                return SoundEffects.generate_beep(frequency=900, duration=0.05, sample_rate=sample_rate, volume=volume*0.3)
        
        # Fallback to default sounds
        return SoundEffects.generate_beep(frequency=800, duration=0.1, sample_rate=sample_rate, volume=volume)
    
    @staticmethod
    def _generate_modern_interrupt_gentle(sample_rate: int, volume: float):
        """Modern gentle interrupt sound"""
        t = np.linspace(0, 0.15, int(sample_rate * 0.15))
        # Soft downward sweep
        freq = 900 - 300 * t / 0.15
        sound = np.sin(2 * np.pi * freq * t)
        # Smooth envelope
        envelope = np.exp(-3 * t / 0.15) * np.sin(np.pi * t / 0.15)
        return (sound * envelope * volume).astype(np.float32)
    
    @staticmethod
    def _generate_modern_interrupt_urgent(sample_rate: int, volume: float):
        """Modern urgent interrupt sound"""
        t = np.linspace(0, 0.2, int(sample_rate * 0.2))
        # Quick double beep
        freq1 = 1200 * (1 + 0.1 * np.sin(2 * np.pi * 20 * t))  # Slight tremolo
        sound = np.sin(2 * np.pi * freq1 * t)
        # Sharp envelope with two peaks
        envelope = np.exp(-8 * t / 0.2) * (1 + 0.5 * np.sin(2 * np.pi * 10 * t))
        return (sound * envelope * volume * 1.2).astype(np.float32)
    
    @staticmethod
    def _generate_modern_completion_success(sample_rate: int, volume: float):
        """Modern success completion sound"""
        t = np.linspace(0, 0.4, int(sample_rate * 0.4))
        # Rising triad
        f1, f2, f3 = 523, 659, 784  # C, E, G
        sound1 = np.sin(2 * np.pi * f1 * t) * np.exp(-t)
        sound2 = np.sin(2 * np.pi * f2 * t) * np.exp(-t * 0.8) * (t > 0.1)
        sound3 = np.sin(2 * np.pi * f3 * t) * np.exp(-t * 0.6) * (t > 0.2)
        sound = (sound1 + sound2 + sound3) / 3
        return (sound * volume).astype(np.float32)
    
    @staticmethod
    def _generate_modern_processing_progress(sample_rate: int, volume: float):
        """Modern processing progress indicator"""
        t = np.linspace(0, 0.1, int(sample_rate * 0.1))
        # Brief ascending chirp
        freq = 800 + 400 * t / 0.1
        sound = np.sin(2 * np.pi * freq * t)
        envelope = np.sin(np.pi * t / 0.1) ** 2
        return (sound * envelope * volume * 0.8).astype(np.float32)
    
    @staticmethod
    def _generate_modern_transcription_complete(sample_rate: int, volume: float):
        """Modern transcription complete sound - subtle confirmation"""
        t = np.linspace(0, 0.15, int(sample_rate * 0.15))
        # Quick ascending chirp
        freq = 900 + 300 * (t / 0.15)  # Rise from 900Hz to 1200Hz
        sound = np.sin(2 * np.pi * freq * t)
        # Quick attack, exponential decay
        envelope = np.exp(-6 * t / 0.15) * (1 - np.exp(-40 * t / 0.15))
        return (sound * envelope * volume * 0.8).astype(np.float32)  # Slightly quieter
    
    @staticmethod
    def apply_fade(sound: np.ndarray, fade_duration: float = 0.01, sample_rate: int = 16000):
        """Apply fade-in and fade-out to prevent audio clicks"""
        fade_samples = int(fade_duration * sample_rate)
        if fade_samples >= len(sound) // 2:
            fade_samples = len(sound) // 4
        
        if fade_samples > 0:
            # Fade in
            sound[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade out
            sound[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return sound 