"""
Main voice conversation loop
The heart of the elegant chatbot
"""
from enum import Enum
from typing import Optional
import numpy as np
import time

try:
    from .audio_simple import SimpleAudioSystem as AudioSystem
except ImportError:
    from .audio import AudioSystem
from .audio import SimpleVAD
from .stt import WhisperSTT
from .llm import LLMClient
from .tts import SimpleTTS

# Import features if available
try:
    from features.memory import ConversationMemory
except ImportError:
    ConversationMemory = None


class State(Enum):
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class VoiceAssistant:
    """Simple, elegant voice assistant"""
    
    def __init__(self, config):
        self.config = config
        
        # Core components
        self.audio = AudioSystem(config)
        self.vad = SimpleVAD(config)
        self.stt = WhisperSTT(config)
        self.llm = LLMClient(config)
        self.tts = SimpleTTS(config)
        
        # Optional features
        self.memory = None
        if config.features.enable_memory and ConversationMemory:
            self.memory = ConversationMemory(config)
        
        # State
        self.state = State.LISTENING
        self.running = True
        
        # Audio buffer for speech collection
        self.speech_buffer = []
        self.silence_count = 0
        
    def run(self):
        """Main conversation loop"""
        print("🎤 Starting Elegant Chatbot...")
        print("   Using GPT-4.1-nano")
        print("   Press Ctrl+C to exit\n")
        
        # Start audio system
        try:
            self.audio.start()
        except Exception as e:
            print(f"❌ Failed to initialize audio: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check if microphone is connected")
            print("2. Check Windows sound settings")
            print("3. Try running as Administrator")
            print("4. Install PyAudio manually if needed")
            raise
        
        # Simple greeting
        try:
            self._speak("Hello! I'm ready to chat.")
        except Exception as e:
            print(f"⚠️  TTS greeting failed: {e}")
            print("Continuing without greeting...")
        
        # Main loop
        while self.running:
            try:
                self._process_frame()
                time.sleep(0.01)  # Small delay to prevent CPU spinning
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                
        # Cleanup
        self.audio.close()
        
    def _process_frame(self):
        """Process a single frame of audio"""
        if self.state == State.LISTENING:
            # Start recording
            if hasattr(self.audio, 'recording') and not self.audio.recording:
                self.audio.start_recording()
            elif hasattr(self.audio, 'is_recording') and not self.audio.is_recording:
                self.audio.start_recording()
                
            # Get audio chunk
            chunk = None
            if hasattr(self.audio, 'get_audio_chunk'):
                chunk = self.audio.get_audio_chunk()
            elif hasattr(self.audio, 'audio_buffer') and len(self.audio.audio_buffer) >= self.config.audio.chunk_size:
                chunk = np.array(list(self.audio.audio_buffer)[-self.config.audio.chunk_size:])
            
            if chunk is not None:
                # Check for speech
                vad_state = self.vad.process(chunk)
                
                if vad_state == "speech":
                    self.speech_buffer.append(chunk)
                    self.silence_count = 0
                elif vad_state == "silence" and self.speech_buffer:
                    self.silence_count += 1
                    
                    # End of speech detected
                    if self.silence_count > 10:  # ~300ms of silence
                        self._handle_speech()
                        
        elif self.state == State.SPEAKING:
            # Check for interrupts if enabled
            if self.config.features.enable_interrupts:
                chunk = None
                if hasattr(self.audio, 'get_audio_chunk'):
                    chunk = self.audio.get_audio_chunk()
                elif hasattr(self.audio, 'audio_buffer') and len(self.audio.audio_buffer) >= self.config.audio.chunk_size:
                    chunk = np.array(list(self.audio.audio_buffer)[-self.config.audio.chunk_size:])
                    
                if chunk is not None and self.audio.is_speech(chunk):
                    print("🛑 Interrupted!")
                    if hasattr(self.audio, 'stop_playback'):
                        self.audio.stop_playback()
                    self.state = State.LISTENING
                        
    def _handle_speech(self):
        """Process collected speech"""
        if not self.speech_buffer:
            return
            
        # Stop recording and get audio
        audio_data = np.concatenate(self.speech_buffer)
        self.speech_buffer = []
        self.silence_count = 0
        
        # Change state
        self.state = State.PROCESSING
        
        # Transcribe
        print("🎯 Processing speech...")
        text = self.stt.transcribe(audio_data)
        
        if not text:
            self.state = State.LISTENING
            return
            
        print(f"👤 You: {text}")
        
        # Generate response
        context = None
        if self.memory:
            context = self.memory.get_context()
        elif self.config.features.enable_memory:
            context = self.llm.get_context()
            
        response = self.llm.generate(text, context)
        
        print(f"🤖 Assistant: {response}")
        
        # Save to memory if enabled
        if self.memory:
            self.memory.add_exchange(text, response)
        
        # Speak response
        self._speak(response)
        
        # Return to listening
        self.state = State.LISTENING
        
    def _speak(self, text: str):
        """Speak text through TTS"""
        self.state = State.SPEAKING
        
        # Clear audio buffer before speaking
        if hasattr(self.audio, 'audio_buffer'):
            self.audio.audio_buffer.clear()
        elif hasattr(self.audio, 'audio_queue'):
            # Clear the queue for SimpleAudioSystem
            while not self.audio.audio_queue.empty():
                self.audio.audio_queue.get()
        
        # Synthesize
        audio_data = self.tts.synthesize(text)
        
        # Play audio
        if self.config.features.enable_effects:
            self._play_effect("start")
            
        self.audio.play_audio(audio_data)
        
        if self.config.features.enable_effects:
            self._play_effect("end")
            
    def _play_effect(self, effect_type: str):
        """Play sound effect (if enabled)"""
        if not self.config.features.enable_effects:
            return
            
        # Simple tone generation
        duration = 0.1
        sample_rate = self.config.audio.sample_rate
        
        if effect_type == "start":
            # Rising tone
            freq = np.linspace(400, 800, int(duration * sample_rate))
        else:
            # Falling tone
            freq = np.linspace(800, 400, int(duration * sample_rate))
            
        t = np.linspace(0, duration, int(duration * sample_rate))
        tone = np.sin(2 * np.pi * freq * t)
        tone = (tone * 32767 * self.config.features.effect_volume).astype(np.int16)
        
        self.audio.play_audio(tone)