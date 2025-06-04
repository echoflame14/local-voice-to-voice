import numpy as np
from typing import Tuple
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