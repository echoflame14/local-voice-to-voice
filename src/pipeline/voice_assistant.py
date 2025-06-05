import numpy as np
import threading
import time
from typing import Optional, Callable, List, Dict
from collections import deque
import queue
import re
from datetime import datetime
from pynput import keyboard  # For handling keyboard events
from pynput.keyboard import Key, Listener as KeyboardListener # Added Listener
from pathlib import Path
import traceback

from ..stt import WhisperSTT
# (LLM imports performed lazily below to allow choosing provider)
from ..tts import ChatterboxTTSWrapper
from ..audio import AudioStreamManager, SoundEffects
# Conversation management now handled by ConversationManager
from ..utils.text_similarity import is_similar_text
from ..utils.text_cleaner import clean_text_for_tts
from ..config import VoiceAssistantConfig
from ..managers import ConversationManager


class VoiceAssistant:
    """Main voice assistant class orchestrating STT, LLM, and TTS"""
    
    def __init__(
        self,
        config: Optional[VoiceAssistantConfig] = None,
        # Legacy parameters for backward compatibility
        whisper_model_size: str = "base",
        whisper_device: str = None,
        use_gemini: bool = True,
        llm_base_url: str = "http://localhost:1234/v1",
        llm_api_key: str = "not-needed",
        gemini_api_key: str = None,
        gemini_model: str = "models/gemini-1.5-flash",
        system_prompt: str = None,
        tts_device: str = None,
        voice_reference_path: Optional[str] = None,
        voice_exaggeration: float = 0.5,
        voice_cfg_weight: float = 0.5,
        voice_temperature: float = 0.8,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        min_audio_amplitude: float = 0.015,
        input_mode: str = "vad",
        vad_aggressiveness: int = 1,
        vad_speech_threshold: float = 0.3,
        vad_silence_threshold: float = 0.8,
        push_to_talk_key: str = "space",
        enable_sound_effects: bool = True,
        sound_effect_volume: float = 0.2,
        enable_interruption_sound: bool = True,
        enable_generation_sound: bool = True,
        max_response_tokens: int = 5000,
        llm_temperature: float = 1,
        log_conversations: bool = True,
        conversation_log_dir: str = "conversation_logs",
        max_history_messages: int = 2000,
        auto_summarize_conversations: bool = True,
        max_summaries_to_load: int = 2000
    ):
        """Initialize Voice Assistant"""
        # Handle configuration - support both new config object and legacy parameters
        if config is None:
            # Create config from legacy parameters for backward compatibility
            config = VoiceAssistantConfig.create_from_legacy_params(
                whisper_model_size=whisper_model_size,
                whisper_device=whisper_device,
                use_gemini=use_gemini,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model,
                system_prompt=system_prompt,
                tts_device=tts_device,
                voice_reference_path=voice_reference_path,
                voice_exaggeration=voice_exaggeration,
                voice_cfg_weight=voice_cfg_weight,
                voice_temperature=voice_temperature,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                min_audio_amplitude=min_audio_amplitude,
                input_mode=input_mode,
                vad_aggressiveness=vad_aggressiveness,
                vad_speech_threshold=vad_speech_threshold,
                vad_silence_threshold=vad_silence_threshold,
                push_to_talk_key=push_to_talk_key,
                enable_sound_effects=enable_sound_effects,
                sound_effect_volume=sound_effect_volume,
                enable_interruption_sound=enable_interruption_sound,
                enable_generation_sound=enable_generation_sound,
                max_response_tokens=max_response_tokens,
                llm_temperature=llm_temperature,
                log_conversations=log_conversations,
                conversation_log_dir=conversation_log_dir,
                max_history_messages=max_history_messages,
                auto_summarize_conversations=auto_summarize_conversations,
                max_summaries_to_load=max_summaries_to_load
            )
        
        # Validate configuration
        config.validate()
        
        # Store configuration
        self.config = config
        
        # Quick access to frequently used values (for performance)
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = config.audio.chunk_size
        self.max_response_tokens = config.llm.max_response_tokens
        self.llm_temperature = config.llm.temperature
        self.min_audio_amplitude = config.audio.min_amplitude
        
        # Initialize components
        print("Initializing Voice Assistant components...")
        
        # STT
        print("Loading Whisper STT...")
        self.stt = WhisperSTT(
            model_size=config.stt.model_size, 
            device=config.stt.device
        )
        
        # LLM selection
        if config.llm.use_gemini and config.llm.gemini_api_key:
            from ..llm import GeminiLLM
            print("Connecting to Gemini...")
            self.llm = GeminiLLM(
                api_key=config.llm.gemini_api_key,
                model=config.llm.gemini_model,
                system_prompt=config.llm.system_prompt or "You are a helpful voice assistant. Keep your responses concise and natural for speech."
            )
            self.use_gemini = True
        else:
            from ..llm import OpenAICompatibleLLM
            print("Connecting to LM Studio...")
            self.llm = OpenAICompatibleLLM(
                base_url=config.llm.base_url,
                api_key=config.llm.api_key,
                model=None,
                system_prompt=config.llm.system_prompt or "You are a helpful voice assistant. Keep your responses concise and natural for speech."
            )
            self.use_gemini = False
        
        # Store LLM choice for summarizer (maintain compatibility with existing code)
        self.gemini_api_key = config.llm.gemini_api_key
        self.gemini_model = config.llm.gemini_model
        self.llm_base_url = config.llm.base_url
        self.llm_api_key = config.llm.api_key
        
        # TTS
        print("Loading Chatterbox TTS...")
        self.tts = ChatterboxTTSWrapper(
            device=config.tts.device,
            voice_reference_path=config.tts.voice_reference_path,
            exaggeration=config.tts.exaggeration,
            cfg_weight=config.tts.cfg_weight,
            temperature=config.tts.temperature
        )
        
        # Audio manager with input mode support
        # Use VAD-optimized chunk size for best performance
        vad_chunk_size = 480 if config.input.mode == "vad" else config.audio.chunk_size
        self.audio_manager = AudioStreamManager(
            sample_rate=config.audio.sample_rate,
            chunk_size=vad_chunk_size,
            enable_performance_logging=True,
            list_devices_on_init=False,
            input_mode=config.input.mode
        )
        
        # State management
        self.is_running = False
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.tentative_interruption = False
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(config.conversation, config.llm)
        
        # Interruption handling for speech continuation
        self.interrupted_user_text = ""
        self.is_continuation = False
        
        # Grace period to prevent immediate interruptions after starting synthesis
        from configs import config as app_config
        self.synthesis_grace_period = app_config.SYNTHESIS_GRACE_PERIOD
        self.synthesis_start_time = None
        
        # Audio buffers
        self.audio_buffer = []  # Simple list for PTT
        
        # Keyboard listener for PTT
        self.is_ctrl_pressed = False
        self.is_space_pressed = False
        self.ptt_active = False # Renamed from self.space_bar_pressed
        self._keyboard_listener = KeyboardListener(on_press=self._on_key_press, on_release=self._on_key_release)
        
        # Processing queue and control
        self.processing_queue = queue.Queue(maxsize=1)
        self.current_processing_thread = None
        self.cancel_processing = threading.Event()
        
        # Threading
        self.processing_thread = None
        self.audio_thread = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_synthesis_start: Optional[Callable] = None
        self.on_synthesis_end: Optional[Callable] = None
        
        # Sound Effects settings
        self.enable_sound_effects = config.sound_effects.enable_sound_effects
        self.sound_effect_volume = config.sound_effects.sound_effect_volume
        self.enable_interruption_sound = config.sound_effects.enable_interruption_sound
        self.enable_generation_sound = config.sound_effects.enable_generation_sound
        
        # Legacy conversation attributes for compatibility
        self.log_conversations = config.conversation.log_conversations
        self.auto_summarize = config.conversation.auto_summarize_conversations
        
        self.max_history_messages = max_history_messages
        
        print("Voice Assistant initialized successfully!")
    
    def _play_sound_effect(self, sound_type: str):
        """Play a sound effect in a non-blocking way"""
        if not self.enable_sound_effects:
            return
        
        try:
            # Generate sound based on type
            if sound_type == "interruption" and self.enable_interruption_sound:
                sound = SoundEffects.generate_interrupt_sound(
                    sample_rate=self.sample_rate,
                    volume=self.sound_effect_volume
                )
            elif sound_type == "generation_start" and self.enable_generation_sound:
                sound = SoundEffects.generate_generation_start_sound(
                    sample_rate=self.sample_rate,
                    volume=self.sound_effect_volume
                )
            elif sound_type == "completion":
                sound = SoundEffects.generate_completion_sound(
                    sample_rate=self.sample_rate,
                    volume=self.sound_effect_volume
                )
            elif sound_type == "processing":
                print("[DEBUG] Playing processing sound effect")
                # Always play a sound for processing, regardless of enable_generation_sound
                # Use a distinct sound if available, otherwise use completion sound as fallback
                if hasattr(SoundEffects, 'generate_processing_sound'):
                    sound = SoundEffects.generate_processing_sound(
                        sample_rate=self.sample_rate,
                        volume=self.sound_effect_volume
                    )
                else:
                    sound = SoundEffects.generate_completion_sound(
                        sample_rate=self.sample_rate,
                        volume=self.sound_effect_volume
                    )
            else:
                return
            
            # Play in separate thread to avoid blocking
            def play_sound():
                try:
                    AudioStreamManager.play_audio_simple(sound, self.sample_rate)
                except Exception as e:
                    print(f"Error playing sound effect: {e}")
            
            sound_thread = threading.Thread(target=play_sound, daemon=True)
            sound_thread.start()
            
        except Exception as e:
            print(f"Error generating sound effect: {e}")
    
    def start(self):
        """Start the voice assistant"""
        if self.is_running:
            print("Voice Assistant is already running")
            return
        
        print("Starting Voice Assistant...")
        self.is_running = True
        self._stop_event.clear()
        
        # Start audio input stream
        self.audio_manager.start_input_stream(callback=self._audio_callback)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start keyboard listener
        if not self._keyboard_listener.is_alive():
            self._keyboard_listener.start()
        
        print("Voice Assistant started! Listening for speech...")
    
    def stop(self):
        """Stop the voice assistant"""
        if not self.is_running:
            print("Voice Assistant is not running")
            return
        
        print("Stopping Voice Assistant...")
        self.is_running = False
        self._stop_event.set()
        
        # Cancel any ongoing processing - NEW
        self.cancel_processing.set()
        
        # Clear any interrupted speech state
        self.interrupted_user_text = ""
        self.is_continuation = False
        if hasattr(self, '_current_user_text'):
            self._current_user_text = ""
        
        # Stop audio streams
        self.audio_manager.stop_input_stream()
        self.audio_manager.stop_output_stream()
        
        # Wait for processing worker to finish - NEW
        if self.current_processing_thread and self.current_processing_thread.is_alive():
            self.current_processing_thread.join(timeout=2.0)
        
        # Wait for main processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Stop keyboard listener
        if self._keyboard_listener.is_alive():
            self._keyboard_listener.stop()
        
        print("Voice Assistant stopped")
    
    def _audio_callback(self, audio_chunk: np.ndarray):
        """Handle audio chunk from stream"""
        if not self.is_running:
            return
        
        # Handle audio based on input mode
        if self.audio_manager.input_manager.input_mode == "push_to_talk":
            # Check if we're in push-to-talk mode (Ctrl+Space)
            if self.ptt_active:
                if not self.is_listening:
                    self._on_speech_start()
                
                # Check audio amplitude
                audio_amplitude = np.abs(audio_chunk).mean()
                if audio_amplitude >= self.min_audio_amplitude:
                    self.audio_buffer.append(audio_chunk)
            elif self.is_listening:
                # Keys released, end speech
                self._on_speech_end()
        else:  # VAD mode
            # In VAD mode, we receive complete utterances from the stream manager
            if audio_chunk is not None and len(audio_chunk) > 0:
                try:
                    # Process the speech
                    self._process_speech(audio_chunk)
                except Exception as e:
                    print(f"[VA ERROR] Error processing speech: {e}")
                    traceback.print_exc()
    
    def _on_key_press(self, key):
        """Callback for key press events from pynput listener."""
        # Update individual key states
        if key == Key.ctrl_l or key == Key.ctrl_r:
            self.is_ctrl_pressed = True
        elif key == Key.space:
            self.is_space_pressed = True

        # If PTT condition is met and PTT is not already active
        if self.is_ctrl_pressed and self.is_space_pressed:
            if not self.ptt_active:
                self.ptt_active = True
                # _audio_callback will call _on_speech_start when audio arrives
    
    def _on_key_release(self, key):
        """Callback for key release events from pynput listener."""
        # Update individual key states
        if key == Key.ctrl_l or key == Key.ctrl_r:
            self.is_ctrl_pressed = False
        elif key == Key.space:
            self.is_space_pressed = False

        # If the PTT condition is no longer met and PTT was active
        if not (self.is_ctrl_pressed and self.is_space_pressed):
            if self.ptt_active: # If PTT *was* active
                self.ptt_active = False
                if self.is_listening: # And we were in a listening state
                    self._on_speech_end()
    
    def _on_speech_start(self):
        """Handle speech start"""
        print("🎤 Push-to-talk started...")
        
        # Check if we're in synthesis grace period
        if (self.is_speaking and self.synthesis_start_time is not None and 
            time.time() - self.synthesis_start_time < self.synthesis_grace_period):
            print(f"⏳ Ignoring PTT during synthesis grace period ({self.synthesis_grace_period}s)")
            return
        
        # If we're speaking, do a tentative interruption
        if self.is_speaking:
            print("⏸️ Tentative interruption...")
            self.tentative_interruption = True
            self._play_sound_effect("interruption")
        
        # If we're already processing, cancel it
        if self.is_processing:
            print("🚫 Cancelling current processing")
            if hasattr(self, '_current_user_text') and self._current_user_text:
                self.interrupted_user_text = self._current_user_text
                print(f"💾 Saving interrupted text: '{self.interrupted_user_text[:50]}{'...' if len(self.interrupted_user_text) > 50 else ''}'")
            self.cancel_processing.set()
            time.sleep(0.1)
        
        # Start new recording
        self.is_listening = True
        self.audio_buffer.clear()
        
        if self.on_speech_start:
            self.on_speech_start()
    
    def _on_speech_end(self):
        """Handle speech end"""
        if not self.is_listening:
            return
        
        print("🔇 Push-to-talk ended")
        self.is_listening = False
        
        if self.on_speech_end:
            self.on_speech_end()
        
        # Process collected speech
        if self.audio_buffer:
            speech_audio = np.concatenate(self.audio_buffer)
            self._queue_transcription(speech_audio)
            self.audio_buffer.clear()
        else:
            print("⚠️ No audio collected during PTT")
            if self.tentative_interruption:
                print("🔄 Reverting tentative interruption - no audio")
                self.tentative_interruption = False
    
    def _queue_transcription(self, audio: np.ndarray):
        """Queue audio for transcription - UPDATED with proper queuing and speech continuation"""
        # If we're already processing, save the current context for continuation
        if self.is_processing:
            print("⚠️  Interrupting current processing for new speech")
            
            # Save the current user text if we have it (for speech continuation)
            if hasattr(self, '_current_user_text') and self._current_user_text:
                self.interrupted_user_text = self._current_user_text
                print(f"💾 Saving interrupted speech for continuation: '{self.interrupted_user_text[:50]}{'...' if len(self.interrupted_user_text) > 50 else ''}'")
            
            self.cancel_processing.set()
            # Clear the queue to remove any pending items
            try:
                while True:
                    self.processing_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Wait briefly for cancellation
            time.sleep(0.1)
            
        try:
            # Try to add to queue (non-blocking)
            self.processing_queue.put_nowait(audio)
            
            # Start processing thread if not already running
            if self.current_processing_thread is None or not self.current_processing_thread.is_alive():
                self.current_processing_thread = threading.Thread(
                    target=self._processing_worker,
                    daemon=True
                )
                self.current_processing_thread.start()
                
        except queue.Full:
            print("⚠️  Processing queue full, dropping audio")
    
    def _processing_worker(self):
        """Worker thread that processes speech from the queue - NEW"""
        while self.is_running and not self._stop_event.is_set():
            try:
                # Get audio from queue (blocking with timeout)
                audio = self.processing_queue.get(timeout=1.0)
                
                # Reset cancellation flag
                self.cancel_processing.clear()
                
                # Process the speech
                try:
                    self._process_speech(audio)
                except Exception as e:
                    print(f"❌ Error processing speech: {e}")
                    # Continue processing other items instead of crashing
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                # No more items to process
                break
            except Exception as e:
                print(f"❌ Error in processing worker: {e}")
                # Continue processing to prevent complete failure
    
    def _prepare_messages_for_llm(self, user_text: str) -> List[Dict[str, str]]:
        """Prepare messages for LLM including system prompt, memory hierarchy, and conversation history"""
        try:
            messages = []
            
            # Add system prompt if configured
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            
            # Get context from conversation manager (includes memory hierarchy and recent history)
            context_messages = self.conversation_manager.get_context_for_llm(
                user_text, 
                self.config.conversation.max_summaries_to_load
            )
            messages.extend(context_messages)
            
            # Add current user message
            messages.append({"role": "user", "content": user_text})
            
            print(f"[VA DEBUG] Final message count: {len(messages)}")
            return messages

        except Exception as e:
            print(f"[VA ERROR] Error preparing messages: {e}")
            print(f"[VA ERROR] Traceback: {traceback.format_exc()}")
            # Fallback to basic message structure
            return [
                {"role": "system", "content": self.llm.system_prompt if self.llm else ""},
                {"role": "user", "content": user_text}
            ]

    def _process_speech(self, audio: np.ndarray):
        """Process recorded speech"""
        try:
            print("[VA DEBUG] Starting speech processing...")
            
            # Transcribe audio
            print("[VA DEBUG] Transcribing audio...")
            transcription_result = self.stt.transcribe(audio)
            # Extract the text from the transcription result dictionary
            user_text = transcription_result['text'].strip() if isinstance(transcription_result, dict) else str(transcription_result).strip()
            print(f"[VA DEBUG] Transcription complete: {user_text}")
            
            if not user_text:
                print("[VA DEBUG] No transcription produced")
                return
            
            # Call transcription callback
            if self.on_transcription:
                print("[VA DEBUG] Calling transcription callback...")
                self.on_transcription(user_text)
            
            # Add user message to conversation
            print("[VA DEBUG] Adding user message to conversation...")
            self.conversation_manager.add_message("user", user_text)
            
            # Generate response
            print("[VA DEBUG] Starting response generation...")
            messages = self._prepare_messages_for_llm(user_text)
            print(f"[VA DEBUG] Prepared {len(messages)} messages for LLM")
            
            try:
                print("[VA DEBUG] Calling LLM for response...")
                # Use chat method instead of generate for proper conversation handling
                response = self.llm.chat(
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=self.llm_temperature
                )
                print(f"[VA DEBUG] LLM response received: {response[:50]}...")
                
                # Add assistant response to conversation
                if response:
                    self.conversation_manager.add_message("assistant", response)
                
                # Call response callback
                if self.on_response:
                    print("[VA DEBUG] Calling response callback...")
                    self.on_response(response)
                
                # Note: Assistant message already logged by conversation_manager.add_message() above
                
                # Synthesize speech
                print("[VA DEBUG] Starting speech synthesis...")
                self.say(response)
                print("[VA DEBUG] Speech synthesis complete")
                
            except Exception as e:
                print(f"[VA ERROR] Error generating/synthesizing response: {e}")
                print(f"[VA ERROR] Traceback: {traceback.format_exc()}")
                error_msg = "I'm sorry, I'm having trouble generating a response right now."
                self.say(error_msg)
            
        except Exception as e:
            print(f"[VA ERROR] Error in speech processing: {e}")
            print(f"[VA ERROR] Traceback: {traceback.format_exc()}")
            error_msg = "I'm sorry, I encountered an error while processing your speech."
            self.say(error_msg)
    
    def _stream_generate_and_synthesize(self, messages: List[Dict[str, str]], user_text: str):
        """Stream generate response and synthesize with SEQUENTIAL processing for efficiency"""
        try:
            if self.on_synthesis_start:
                self.on_synthesis_start()
            
            print("🎵 Starting sequential streaming synthesis with longer chunks...")
            self.is_speaking = True
            self.tentative_interruption = False
            self.synthesis_start_time = time.time()  # Track when synthesis starts
            
            # Reset VAD state when starting synthesis
            if self.is_listening:
                self.is_listening = False
                self.audio_buffer.clear()
            
            full_response = []
            current_sentence = ""
            collected_sentences = []  # Accumulate sentences for longer chunks
            in_thinking_block = False
            chunk_count = 0
            
            # Chunk thresholds
            MIN_CHUNK_LENGTH = 50  # Minimum characters before considering synthesis
            MAX_SENTENCES_PER_CHUNK = 4  # Maximum sentences per chunk
            
            print("\n🤖 Assistant: ", end='', flush=True)
            
            # The 'messages' argument to this function already contains the full context:
            # 1. Comprehensive system prompt (base + LTM + STM + loaded summaries)
            # 2. Prior conversation turns
            # 3. The current user_text (which is the last message in the 'messages' list)
            
            # Debug output to show what we're sending (using the 'messages' argument)
            print("\n🔍 DEBUG: Messages being sent to LLM:")
            for i, msg in enumerate(messages): # Use 'messages' argument
                print(f"  {i+1}. {msg['role']}:")
                content_lines = msg['content'].split('\n')
                for line in content_lines:
                    if line.strip():  # Only print non-empty lines
                        print(f"         {line}")
                print()  # Add blank line between messages
            print("=" * 60)
            print("\n🤖 Assistant response:", flush=True)  # Changed to complete the line
            
            # Stream generate response using the correctly prepared 'messages' list
            for chunk in self.llm.stream_chat(
                messages=messages,  # Use the 'messages' argument directly
                max_tokens=self.max_response_tokens,
                temperature=self.llm_temperature
            ):
                # Check for cancellation
                if self.cancel_processing.is_set():
                    break
                
                # Filter out thinking blocks
                if "<think>" in chunk:
                    in_thinking_block = True
                    continue
                if "</think>" in chunk:
                    in_thinking_block = False
                    continue
                if in_thinking_block:
                    continue
                
                # Filter out stage directions
                if "*" in chunk:
                    chunk = re.sub(r'\*[^*]*\*', '', chunk)
                
                # Filter out emojis and special characters
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002702-\U000027B0"  # dingbats
                    u"\U000024C2-\U0001F251" 
                    u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
                    u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
                    "]+", flags=re.UNICODE)
                chunk = emoji_pattern.sub('', chunk)
                
                # Remove zero-width joiners and other invisible characters
                chunk = re.sub(r'[\u200B-\u200D\uFEFF]', '', chunk)
                
                # Skip empty chunks
                if not chunk.strip():
                    continue
                
                # Add chunk to response and display immediately
                full_response.append(chunk)
                print(chunk, end='', flush=True)
                
                # Accumulate text for synthesis
                current_sentence += chunk
                
                # Check for sentence boundaries for chunking
                sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
                
                for ending in sentence_endings:
                    if ending in current_sentence:
                        # Extract complete sentence
                        parts = current_sentence.split(ending, 1)
                        if len(parts) > 1:
                            complete_sentence = parts[0].strip() + ending.strip()
                            current_sentence = parts[1]  # Keep remainder
                            
                            # Add sentence to collection without cleaning
                            if complete_sentence:
                                collected_sentences.append(complete_sentence)
                                print(f"\n📝 Collected sentence {len(collected_sentences)}: '{complete_sentence[:40]}{'...' if len(complete_sentence) > 40 else ''}'")
                                
                                # Check if we should synthesize this chunk
                                chunk_text = " ".join(collected_sentences)
                                should_synthesize = (
                                    len(collected_sentences) >= MAX_SENTENCES_PER_CHUNK or
                                    len(chunk_text) >= MIN_CHUNK_LENGTH * 2  # Extra long chunk
                                )
                                
                                if should_synthesize:
                                    chunk_count += 1
                                    print(f"\n🎵 Synthesizing chunk {chunk_count} ({len(collected_sentences)} sentences, {len(chunk_text)} chars):")
                                    print(f"   📖 '{chunk_text[:60]}{'...' if len(chunk_text) > 60 else ''}'")
                                    
                                    # Synthesize accumulated sentences without cleaning
                                    start_time = time.time()
                                    audio, sample_rate = self.tts.synthesize(chunk_text)
                                    synthesis_time = time.time() - start_time
                                    audio_duration = len(audio) / sample_rate
                                    
                                    print(f"   ✅ Ready: {synthesis_time:.2f}s synthesis → {audio_duration:.1f}s audio")
                                    
                                    # Play immediately and wait for completion
                                    if not self.cancel_processing.is_set():
                                        playback_thread = self.audio_manager.play_audio_streaming(
                                            iter([(audio, sample_rate)]),
                                            interrupt_current=False
                                        )
                                        if playback_thread:
                                            # Use timeout to prevent indefinite blocking
                                            playback_thread.join(timeout=max(2.0, audio_duration + 1.0))
                                            if playback_thread.is_alive():
                                                print(f"   ⚠️ Chunk {chunk_count} playback timeout, continuing...")
                                            else:
                                                print(f"   🔊 Chunk {chunk_count} finished playing")
                                    
                                    # Clear collected sentences for next chunk
                                    collected_sentences = []
                            break
            
            # Handle any remaining sentences
            if collected_sentences:
                chunk_text = " ".join(collected_sentences)
                chunk_count += 1
                print(f"\n🎵 Final chunk {chunk_count} ({len(collected_sentences)} sentences, {len(chunk_text)} chars):")
                print(f"   📖 '{chunk_text[:60]}{'...' if len(chunk_text) > 60 else ''}'")
                
                start_time = time.time()
                # Synthesize without cleaning
                audio, sample_rate = self.tts.synthesize(chunk_text)
                synthesis_time = time.time() - start_time
                audio_duration = len(audio) / sample_rate
                
                print(f"   ✅ Ready: {synthesis_time:.2f}s synthesis → {audio_duration:.1f}s audio")
                
                if not self.cancel_processing.is_set():
                    playback_thread = self.audio_manager.play_audio_streaming(
                        iter([(audio, sample_rate)]),
                        interrupt_current=False
                    )
                    if playback_thread:
                        # Use timeout to prevent indefinite blocking
                        playback_thread.join(timeout=max(2.0, audio_duration + 1.0))
                        if playback_thread.is_alive():
                            print(f"   ⚠️ Final chunk playback timeout, continuing...")
                        else:
                            print(f"   🔊 Final chunk finished playing")
            
            # Handle any remaining partial sentence
            if current_sentence.strip():
                chunk_count += 1
                print(f"\n🎵 Partial sentence chunk {chunk_count}: '{current_sentence[:40]}{'...' if len(current_sentence) > 40 else ''}'")
                
                start_time = time.time()
                # Synthesize without cleaning
                audio, sample_rate = self.tts.synthesize(current_sentence)
                synthesis_time = time.time() - start_time
                audio_duration = len(audio) / sample_rate
                
                print(f"   ✅ Ready: {synthesis_time:.2f}s synthesis → {audio_duration:.1f}s audio")
                
                if not self.cancel_processing.is_set():
                    playback_thread = self.audio_manager.play_audio_streaming(
                        iter([(audio, sample_rate)]),
                        interrupt_current=False
                    )
                    if playback_thread:
                        # Use timeout to prevent indefinite blocking
                        playback_thread.join(timeout=max(2.0, audio_duration + 1.0))
                        if playback_thread.is_alive():
                            print(f"   ⚠️ Partial chunk playback timeout, continuing...")
                        else:
                            print(f"   🔊 Partial chunk finished playing")
            
            # Make sure all audio is finished
            self.audio_manager.wait_for_playback_complete()
            
            # Play completion sound effect
            self._play_sound_effect("completion")
            
            # Update conversation with streaming response
            full_response_text = "".join(full_response).strip()
            if full_response_text:
                # Handle continuations by updating the last user message
                if self.is_continuation:
                    if self.conversation_manager.update_last_user_message(user_text):
                        print(f"🔄 Updated last user message with continuation")
                    else:
                        # Fallback: add as new message
                        self.conversation_manager.add_message("user", user_text)
                else:
                    # Normal case: add new user message
                    self.conversation_manager.add_message("user", user_text)
                
                # Add the assistant response
                self.conversation_manager.add_message("assistant", full_response_text)
                
                # Show the raw response to user via callback
                if self.on_response:
                    self.on_response(full_response_text)
                
                # Clear continuation flag
                self.is_continuation = False
            
            print(f"\n✅ Sequential synthesis complete! Processed {chunk_count} chunk(s)")
            
        except Exception as e:
            print(f"❌ Error in streaming generation: {e}")
            # Make sure to cancel any ongoing audio
            try:
                self.audio_manager.clear_queues()
            except:
                pass
        finally:
            # Always reset speaking state and cancellation flag
            self.is_speaking = False
            self.tentative_interruption = False
            self.cancel_processing.clear()
            self.synthesis_start_time = None  # Reset synthesis timing
            if self.on_synthesis_end:
                self.on_synthesis_end()
    

    

    
    def _processing_loop(self):
        """Main processing loop (placeholder for future enhancements)"""
        while self.is_running and not self._stop_event.is_set():
            time.sleep(0.1)
    
    def say(self, text: str, interrupt: bool = True, use_streaming: bool = None):
        """
        Make the assistant say something
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
            use_streaming: Whether to use streaming synthesis (auto-detect if None)
        """
        if interrupt and self.is_speaking:
            self.audio_manager.clear_queues()
            self.is_speaking = False
        
        # Always use streaming for better responsiveness
        thread = threading.Thread(
            target=self._say_streaming,
            args=(text,),
            daemon=True
        )
        
        thread.start()
    
    def _say_streaming(self, text: str):
        """Simple streaming synthesis for the say method"""
        try:
            self.is_speaking = True
            
            # Split into sentences for better streaming
            sentences = self.tts._split_into_sentences(text)
            
            if not sentences:
                return
            
            for sentence in sentences:
                if self.cancel_processing.is_set():
                    break
                    
                print(f"🎵 Synthesizing: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
                
                # Synthesize this sentence
                audio, sample_rate = self.tts.synthesize(sentence)
                
                if not self.cancel_processing.is_set():
                    # Play immediately
                    playback_thread = self.audio_manager.play_audio_streaming(
                        iter([(audio, sample_rate)]),
                        interrupt_current=False
                    )
                    if playback_thread:
                        # Use timeout to prevent indefinite blocking
                        audio_duration = len(audio) / sample_rate
                        playback_thread.join(timeout=max(2.0, audio_duration + 1.0))
                        if playback_thread.is_alive():
                            print(f"   ⚠️ Say streaming playback timeout, continuing...")
            
            self.audio_manager.wait_for_playback_complete()
            self.is_speaking = False
            
        except Exception as e:
            print(f"Error in streaming say: {e}")
            self.is_speaking = False
    
    def set_voice(self, voice_path: str):
        """Change the assistant's voice"""
        self.tts.set_voice(voice_path)
        print(f"Voice changed to: {voice_path}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_manager.get_history()
    
    def clear_conversation_history(self):
        """Clear conversation history and start new log file"""
        self.conversation_manager.clear_history()
    
    def is_busy(self) -> bool:
        """Check if assistant is currently processing"""
        return self.is_listening or self.is_speaking or self.is_processing
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        self.audio_manager.cleanup()
    
    def load_conversation_history(self, filepath: str):
        """Load a previous conversation history"""
        return self.conversation_manager.load_conversation_file(filepath)
    
    def get_conversation_files(self) -> List[str]:
        """Get list of available conversation log files"""
        return self.conversation_manager.get_conversation_files()
    
    def _process_unsummarized_conversations(self):
        """Process any conversations that haven't been summarized yet"""
        # Delegate to conversation manager
        if hasattr(self.conversation_manager, '_process_unsummarized_conversations'):
            self.conversation_manager._process_unsummarized_conversations()
    
    def _load_hierarchical_memory(self):
        """Load hierarchical memory on startup"""
        # Delegate to conversation manager
        if hasattr(self.conversation_manager, '_load_hierarchical_memory'):
            self.conversation_manager._load_hierarchical_memory()