"""Input mode management for voice interaction"""
import threading
from typing import Optional, Callable
import numpy as np
from pynput import keyboard
from .vad import VoiceActivityDetector
from configs.config import (
    INPUT_MODE, VAD_AGGRESSIVENESS, VAD_SPEECH_THRESHOLD,
    VAD_SILENCE_THRESHOLD, PUSH_TO_TALK_KEY, SAMPLE_RATE
)


class InputManager:
    """Manages voice input modes (VAD or push-to-talk)"""
    
    def __init__(
        self,
        input_mode: str = INPUT_MODE,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        vad_speech_threshold: float = VAD_SPEECH_THRESHOLD,
        vad_silence_threshold: float = VAD_SILENCE_THRESHOLD,
        push_to_talk_key: str = PUSH_TO_TALK_KEY,
        sample_rate: int = SAMPLE_RATE,
        on_input_start: Optional[Callable] = None,
        on_input_end: Optional[Callable] = None
    ):
        """
        Initialize input manager
        
        Args:
            input_mode: Input mode ("vad" or "push_to_talk")
            vad_aggressiveness: VAD aggressiveness (0-3)
            vad_speech_threshold: VAD speech threshold
            vad_silence_threshold: VAD silence threshold
            push_to_talk_key: Keyboard key for push-to-talk
            sample_rate: Audio sample rate
            on_input_start: Callback when input starts
            on_input_end: Callback when input ends
        """
        self.input_mode = input_mode
        self.push_to_talk_key = push_to_talk_key
        self.on_input_start = on_input_start
        self.on_input_end = on_input_end
        
        # Initialize VAD if needed
        self.vad = None
        if input_mode == "vad":
            self.vad = VoiceActivityDetector(
                sample_rate=sample_rate,
                aggressiveness=vad_aggressiveness,
                speech_threshold=vad_speech_threshold,
                silence_threshold=vad_silence_threshold
            )
        
        # State
        self.is_active = False
        self._stop_event = threading.Event()
        self._push_to_talk_pressed = False
        self.key_listener = None
        
        # Set up push-to-talk key listener if needed
        if input_mode == "push_to_talk":
            self._setup_key_listener()
        
        print(f"🎤 InputManager initialized in {input_mode} mode")
        if input_mode == "push_to_talk":
            print(f"🔑 Press '{push_to_talk_key}' to talk")
        else:
            print(f"🎙️ Voice activity detection enabled (aggressiveness: {vad_aggressiveness})")
    
    def _setup_key_listener(self):
        """Set up keyboard listener for push-to-talk"""
        try:
            self.key_listener = keyboard.Listener(
                on_press=self._handle_key_press,
                on_release=self._handle_key_release
            )
            self.key_listener.start()
        except Exception as e:
            print(f"⚠️ Failed to set up keyboard listener: {e}")
            print("🔄 Falling back to VAD mode")
            self.input_mode = "vad"
            self.vad = VoiceActivityDetector()
    
    def _handle_key_press(self, key):
        """Handle push-to-talk key press"""
        try:
            # Convert key to string representation
            key_str = self._key_to_string(key)
            
            # Check if it's our push-to-talk key
            if key_str == self.push_to_talk_key and not self._push_to_talk_pressed:
                self._push_to_talk_pressed = True
                self.is_active = True
                print("🎤 Recording started (PTT)")
                if self.on_input_start:
                    self.on_input_start()
        except Exception as e:
            pass  # Ignore key handling errors
    
    def _handle_key_release(self, key):
        """Handle push-to-talk key release"""
        try:
            # Convert key to string representation
            key_str = self._key_to_string(key)
            
            # Check if it's our push-to-talk key
            if key_str == self.push_to_talk_key and self._push_to_talk_pressed:
                self._push_to_talk_pressed = False
                self.is_active = False
                print("🎤 Recording stopped (PTT)")
                if self.on_input_end:
                    self.on_input_end()
        except Exception as e:
            pass  # Ignore key handling errors
    
    def _key_to_string(self, key) -> str:
        """Convert pynput key to string"""
        try:
            # Regular character keys
            if hasattr(key, 'char') and key.char:
                return key.char
            
            # Special keys
            if key == keyboard.Key.space:
                return "space"
            elif key == keyboard.Key.enter:
                return "enter"
            elif key == keyboard.Key.tab:
                return "tab"
            elif key == keyboard.Key.shift:
                return "shift"
            elif key == keyboard.Key.ctrl:
                return "ctrl"
            elif key == keyboard.Key.alt:
                return "alt"
            else:
                return str(key).replace('Key.', '')
        except:
            return str(key)
    
    def process_audio(self, audio: Optional[np.ndarray]) -> bool:
        """
        Process audio chunk and determine if it should be captured
        
        Args:
            audio: Audio chunk as numpy array (for VAD) or None (for PTT)
            
        Returns:
            Whether the audio should be captured
        """
        if self.input_mode == "vad":
            if audio is None or self.vad is None:
                return False
            
            try:
                is_speech, state_changed = self.vad.process_frame(audio)
                
                # Handle state changes
                if state_changed:
                    if is_speech and not self.is_active:
                        self.is_active = True
                        # Don't print here - let stream_manager handle logging
                        if self.on_input_start:
                            self.on_input_start()
                    elif not is_speech and self.is_active:
                        self.is_active = False
                        # Don't print here - let stream_manager handle logging  
                        if self.on_input_end:
                            self.on_input_end()
                
                return is_speech
            except Exception as e:
                print(f"⚠️ VAD processing error: {e}")
                return False
                
        else:  # push_to_talk
            return self._push_to_talk_pressed
    
    def stop(self):
        """Stop input manager and cleanup"""
        self._stop_event.set()
        if self.key_listener:
            try:
                self.key_listener.stop()
            except:
                pass
        if self.vad:
            self.vad.reset()
        print("🛑 InputManager stopped") 