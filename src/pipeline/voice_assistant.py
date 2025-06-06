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
from .conversation_logger import ConversationLogger
from .conversation_summarizer import ConversationSummarizer
from .hierarchical_memory_manager import HierarchicalMemoryManager, NUM_CONV_SUMMARIES_FOR_STM, NUM_STM_FOR_LTM
from ..utils.logger import logger, log_performance
from ..utils.performance_monitor import perf_monitor, TimerContext


class VoiceAssistant:
    """Main voice assistant class orchestrating STT, LLM, and TTS"""
    
    def __init__(
        self,
        # STT settings
        whisper_model_size: str = "base",
        whisper_device: str = None,
        
        # LLM settings
        use_gemini: bool = False,  # NEW: Toggle for Gemini vs LM Studio
        llm_base_url: str = "http://localhost:1234/v1",
        llm_api_key: str = "not-needed",
        gemini_api_key: str = None,
        gemini_model: str = "models/gemini-1.5-flash",
        system_prompt: str = None,
        
        # TTS settings
        tts_device: str = None,
        voice_reference_path: Optional[str] = None,
        voice_exaggeration: float = 0.5,
        voice_cfg_weight: float = 0.5,
        voice_temperature: float = 0.8,
        
        # Audio settings
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        min_audio_amplitude: float = 0.015,  # Minimum amplitude threshold
        
        # Input mode settings  
        input_mode: str = "vad",  # "vad" or "push_to_talk"
        input_device: Optional[int] = None,  # Audio input device index
        vad_aggressiveness: int = 1,
        vad_speech_threshold: float = 0.3,
        vad_silence_threshold: float = 0.8,
        push_to_talk_key: str = "space",
        
        # Sound Effects settings
        enable_sound_effects: bool = True,
        sound_effect_volume: float = 0.2,
        enable_interruption_sound: bool = True,
        enable_generation_sound: bool = True,
        enable_transcription_sound: bool = True,
        
        # Conversation settings
        max_response_tokens: int = 5000,
        llm_temperature: float = 1,
        log_conversations: bool = True,
        conversation_log_dir: str = "conversation_logs",
        max_history_messages: int = 2000,
        auto_summarize_conversations: bool = True,
        max_summaries_to_load: int = 2000
    ):
        """Initialize Voice Assistant"""
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_response_tokens = max_response_tokens
        self.llm_temperature = llm_temperature
        self.min_audio_amplitude = min_audio_amplitude
        
        # Initialize components
        print("Initializing Voice Assistant components...")
        
        # STT
        print("Loading Whisper STT...")
        self.stt = WhisperSTT(model_size=whisper_model_size, device=whisper_device)
        
        # LLM selection
        self.use_gemini = use_gemini
        
        if use_gemini:
            from ..llm import GeminiLLM
            from configs.config import GEMINI_ENABLE_GROUNDING, GEMINI_GROUNDING_THRESHOLD
            if not gemini_api_key:
                raise ValueError("Gemini API key is required when use_gemini=True")
            print(f"Connecting to Gemini ({gemini_model}) with grounding...")
            self.llm = GeminiLLM(
                api_key=gemini_api_key,
                model=gemini_model,
                system_prompt=system_prompt or "You are a helpful voice assistant. Keep your responses concise and natural for speech.",
                enable_grounding=GEMINI_ENABLE_GROUNDING,
                grounding_threshold=GEMINI_GROUNDING_THRESHOLD
            )
        else:
            from ..llm import OpenAICompatibleLLM
            print("Connecting to LM Studio...")
            self.llm = OpenAICompatibleLLM(
                base_url=llm_base_url,  # Use direct parameter
                api_key=llm_api_key,    # Use direct parameter
                model=None,
                system_prompt=system_prompt or "You are a helpful voice assistant. Keep your responses concise and natural for speech."
            )
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.llm_base_url = llm_base_url # Assignment remains for other parts of the class
        self.llm_api_key = llm_api_key   # Assignment remains for other parts of the class
        
        # TTS
        print("Loading Chatterbox TTS...")
        self.tts = ChatterboxTTSWrapper(
            device=tts_device,
            voice_reference_path=voice_reference_path,
            exaggeration=voice_exaggeration,
            cfg_weight=voice_cfg_weight,
            temperature=voice_temperature
        )
        
        # Audio manager with input mode support
        # Use VAD-optimized chunk size for best performance
        vad_chunk_size = 480 if input_mode == "vad" else chunk_size
        self.audio_manager = AudioStreamManager(
            sample_rate=sample_rate,
            chunk_size=vad_chunk_size,
            enable_performance_logging=True,
            list_devices_on_init=False,
            input_mode=input_mode,
            input_device=input_device
        )
        
        # State management
        self.is_running = False
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.tentative_interruption = False
        self.conversation_history: List[Dict[str, str]] = []
        
        # Interruption handling for speech continuation
        self.interrupted_user_text = ""
        self.is_continuation = False
        
        # Grace period to prevent immediate interruptions after starting synthesis - OPTIMIZED
        from configs.config import INTERRUPT_GRACE_PERIOD, SOUND_THEME, SOUND_FADE_DURATION
        self.synthesis_grace_period = INTERRUPT_GRACE_PERIOD  # Reduced from 1.5s to 1.0s
        self.synthesis_start_time = None
        self.sound_theme = SOUND_THEME
        self.sound_fade_duration = SOUND_FADE_DURATION
        
        # TTS synthesis tracking for true interrupts
        self.active_synthesis_threads = []  # Track all active synthesis threads
        self.synthesis_lock = threading.Lock()  # Protect synthesis thread list
        self.synthesis_interrupted = threading.Event()  # Flag to tell synthesis threads not to play audio
        
        # Enhanced interrupt tracking for conversation logging accuracy
        self.current_response_text = ""      # Full response being synthesized
        self.spoken_response_text = ""       # Text that was actually heard by user  
        self.synthesis_progress = {}         # Track synthesis/playback progress per sentence
        self.played_audio_buffer = []        # Buffer to store actually played audio for Whisper transcription
        self.response_start_time = None      # When current response generation started
        self.playback_interrupted_at = None  # Timestamp when playback was interrupted
        self.sentence_timings = []           # Track when each sentence started/ended playing
        self._pending_response_log = None    # Track responses waiting to be logged
        self._whisper_transcription_used = False  # Flag to prevent overwriting Whisper results
        
        # Interrupt management and false positive prevention
        self.last_interrupt_time = 0         # Timestamp of last interrupt
        self.last_user_input = ""            # Track repeated inputs
        self.input_repeat_count = 0          # Count of repeated inputs
        
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
        self.enable_sound_effects = enable_sound_effects
        self.sound_effect_volume = sound_effect_volume
        self.enable_interruption_sound = enable_interruption_sound
        self.enable_generation_sound = enable_generation_sound
        self.enable_transcription_sound = enable_transcription_sound
        
        # Conversation logging and summarization
        self.log_conversations = log_conversations
        self.auto_summarize = auto_summarize_conversations
        self.max_summaries_to_load = max_summaries_to_load
        
        if log_conversations:
            self.conversation_logger = ConversationLogger(log_dir=conversation_log_dir)
            # Determine LLM for summarizer based on main LLM choice
            # Re-using the main LLM instance for summarization is generally not recommended
            # if it has a very specific system prompt or character.
            # It's better to use a dedicated summarization model or a general model with a summarization prompt.
            
            # Create a summarizer LLM using the same type as the main LLM
            if self.use_gemini:
                from ..llm import GeminiLLM
                from configs.config import GEMINI_ENABLE_GROUNDING, GEMINI_GROUNDING_THRESHOLD
                try:
                    summarizer_llm = GeminiLLM(
                        api_key=gemini_api_key,
                        model="gemini-1.5-flash",  # Use stable model for summarization
                        system_prompt="You are an expert conversation summarizer. Your task is to create concise, neutral, and informative summaries of conversations, focusing on key points, decisions, and outcomes. Preserve all critical context. For meta-summaries (summaries of summaries), synthesize the information into a coherent narrative, identifying overarching themes and long-term takeaways.",
                        enable_grounding=False,  # Disable grounding for summaries
                        grounding_threshold=GEMINI_GROUNDING_THRESHOLD
                    )
                    print("✅ Summarizer LLM initialized with Gemini 1.5 Flash")
                except Exception as e:
                    print(f"⚠️ Failed to initialize Gemini summarizer: {e}")
                    print("   Using fallback summarization...")
                    # Fallback to a minimal summarizer that doesn't use LLM
                    summarizer_llm = None
            else:
                from ..llm import OpenAICompatibleLLM
                summarizer_llm = OpenAICompatibleLLM(
                    base_url=llm_base_url,    # Use direct parameter
                    api_key=llm_api_key,      # Use direct parameter
                    model=None,           # Changed "local-model" to None
                    system_prompt="You are an expert conversation summarizer. Your task is to create concise, neutral, and informative summaries of conversations, focusing on key points, decisions, and outcomes. Preserve all critical context. For meta-summaries (summaries of summaries), synthesize the information into a coherent narrative, identifying overarching themes and long-term takeaways."
                )
            if summarizer_llm:
                self.conversation_summarizer = ConversationSummarizer(summarizer_llm)
                self.hierarchical_memory_manager = HierarchicalMemoryManager(
                    self.conversation_logger, 
                    self.conversation_summarizer
                )
            else:
                print("⚠️ Running without conversation summarization")
                self.conversation_summarizer = None
                self.hierarchical_memory_manager = None
            self.conversation_logger.start_new_conversation()
            
            # Process any unsummarized conversations and update memory hierarchy
            if auto_summarize_conversations and self.hierarchical_memory_manager: # This flag now implicitly covers hierarchical summarization
                print("🚀 Initializing memory system: Processing unsummarized conversations and updating hierarchy...")
                self._process_unsummarized_conversations() # Ensure all individual conversations are summarized first
                self.hierarchical_memory_manager.update_memory_hierarchy() # Build STMs and LTMs
                print("✅ Memory system initialized.")
            elif auto_summarize_conversations and not self.hierarchical_memory_manager:
                print("⚠️ Auto-summarization requested but summarizer unavailable - skipping")
            
            # Load hierarchical memory
            self._load_hierarchical_memory()
        
        self.max_history_messages = max_history_messages
        
        print("Voice Assistant initialized successfully!")
    
    def _capture_played_audio(self, audio_chunk: np.ndarray):
        """Callback to capture actually played audio for Whisper transcription"""
        try:
            # Only capture if we're currently speaking and no interrupt flags are set
            if (self.is_speaking and 
                not self.playback_interrupted_at and 
                not self.synthesis_interrupted.is_set() and 
                not self.cancel_processing.is_set()):
                
                self.played_audio_buffer.append(audio_chunk)
                
                # Limit buffer size to prevent memory issues (max ~10 seconds)
                max_buffer_size = int(10 * 16000 / 480)  # 10 seconds of audio at 16kHz with 480 samples per chunk
                if len(self.played_audio_buffer) > max_buffer_size:
                    self.played_audio_buffer = self.played_audio_buffer[-max_buffer_size:]
        except Exception as e:
            # Don't let capture errors affect main flow
            pass
    
    def _finalize_spoken_text(self):
        """Calculate what portion of the response was actually heard by the user using Whisper transcription"""
        if not self.current_response_text or not self.playback_interrupted_at:
            return
            
        try:
            # NEW APPROACH: Use Whisper to transcribe what was actually played
            if hasattr(self, 'played_audio_buffer') and self.played_audio_buffer:
                logger.debug(f"Using Whisper transcription on {len(self.played_audio_buffer)} audio chunks to determine actual spoken text", "VA")
                try:
                    # Concatenate played audio chunks into single array
                    played_audio = np.concatenate(self.played_audio_buffer)
                    
                    # Calculate how much audio should have been heard based on interrupt timing
                    if self.response_start_time and self.playback_interrupted_at:
                        elapsed_time = self.playback_interrupted_at - self.response_start_time
                        max_samples = int(elapsed_time * 16000)  # Convert to samples at 16kHz
                        
                        # Truncate audio to only what should have been heard
                        if len(played_audio) > max_samples:
                            played_audio = played_audio[:max_samples]
                            logger.debug(f"Truncated audio to {elapsed_time:.2f}s based on interrupt timing", "VA")
                    
                    logger.debug(f"Final audio length: {len(played_audio)} samples ({len(played_audio)/16000:.2f}s)", "VA")
                    
                    # Transcribe the actual played audio with enhanced settings for accuracy
                    transcribed_result = self.stt.transcribe(
                        played_audio,
                        language="en",  # Specify English for better accuracy
                        temperature=(0.0, 0.2, 0.4, 0.6, 0.8),  # Progressive fallback for better results
                        condition_on_previous_text=True,  # Use context for better accuracy
                        initial_prompt="The following is a conversation between a user and an AI assistant. The assistant was speaking when interrupted.",  # Context prompt
                        compression_ratio_threshold=2.4,  # Standard threshold
                        logprob_threshold=-1.0,  # Standard threshold
                        no_speech_threshold=0.6  # Standard threshold
                    )
                    transcribed_text = transcribed_result.get('text', '') if isinstance(transcribed_result, dict) else str(transcribed_result)
                    
                    if transcribed_text and transcribed_text.strip():
                        # Clean up the transcription
                        if isinstance(transcribed_text, dict) and 'text' in transcribed_text:
                            transcribed_text = transcribed_text['text']
                        
                        transcribed_text = transcribed_text.strip()
                        logger.debug(f"Whisper transcribed actual speech: '{transcribed_text}'", "VA")
                        
                        # Use the transcribed text as the spoken portion
                        self.spoken_response_text = transcribed_text
                        self._whisper_transcription_used = True  # Flag to prevent overwriting
                        
                        # Clear the audio buffer to free memory
                        self.played_audio_buffer = []
                        return
                        
                except Exception as e:
                    logger.debug(f"Whisper transcription failed, falling back to timing: {e}", "VA")
            
            # FALLBACK: Use timing-based estimation if Whisper fails or no audio buffer
            if not self.response_start_time:
                self.response_start_time = time.time() - 2.0  # Fallback estimate
            
            elapsed_time = self.playback_interrupted_at - self.response_start_time
            elapsed_time = max(0.5, elapsed_time)  # Ensure minimum positive time
            
            # Estimate words spoken based on timing and sentence progress
            # Average speaking rate is approximately 2-3 words per second for TTS
            estimated_words_spoken = max(1, int(elapsed_time * 2.5))  # Conservative estimate, minimum 1 word
            
            # Split response into words
            words = self.current_response_text.split()
            
            # Use sentence timing data if available for more accurate calculation
            spoken_sentences = []
            
            for timing in self.sentence_timings:
                if timing.get('play_start') and timing.get('play_start') < self.playback_interrupted_at:
                    if timing.get('play_end') and timing.get('play_end') <= self.playback_interrupted_at:
                        # Sentence completed before interrupt
                        spoken_sentences.append(timing['text'])
                    else:
                        # Sentence was partially spoken - use timing estimation
                        sentence_elapsed = self.playback_interrupted_at - timing['play_start']
                        sentence_duration = timing.get('duration', 1.0)
                        completion_ratio = min(sentence_elapsed / sentence_duration, 1.0)
                        
                        sentence_words = timing['text'].split()
                        partial_word_count = int(len(sentence_words) * completion_ratio)
                        
                        if partial_word_count > 0:
                            partial_sentence = ' '.join(sentence_words[:partial_word_count])
                            spoken_sentences.append(partial_sentence)
                        break
            
            # Apply fallback logic
            if spoken_sentences:
                self.spoken_response_text = ' '.join(spoken_sentences)
            elif estimated_words_spoken >= len(words):
                self.spoken_response_text = self.current_response_text
            else:
                self.spoken_response_text = ' '.join(words[:estimated_words_spoken])
            
            logger.debug(f"Response interrupted after {elapsed_time:.1f}s - estimated spoken: '{self.spoken_response_text[:50]}...'", "VA")
            
        except Exception as e:
            logger.error(f"Error calculating spoken text: {e}", "VA")
            # Ultimate fallback: assume first quarter was heard
            words = self.current_response_text.split()
            quarter_words = max(1, len(words) // 4)
            self.spoken_response_text = ' '.join(words[:quarter_words])
    
    def _update_conversation_log_with_actual_speech(self):
        """Update the conversation log to reflect what was actually heard vs the full response"""
        if not self.spoken_response_text or not self.current_response_text:
            return
            
        try:
            # Load current conversation
            conversation = self.conversation_logger._load_conversation()
            
            # Find the last assistant message
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i]['role'] == 'assistant' and conversation[i]['content'] == self.current_response_text:
                    # Create metadata about the interrupt
                    interrupt_info = {
                        'interrupted': True,
                        'full_response': self.current_response_text,
                        'spoken_portion': self.spoken_response_text,
                        'interrupted_at': self.playback_interrupted_at,
                        'response_duration': self.playback_interrupted_at - (self.response_start_time or time.time()),
                        'sentence_count': len(self.sentence_timings),
                        'completed_sentences': len([t for t in self.sentence_timings if t.get('play_end')])
                    }
                    
                    # Update the message content to show what was actually heard
                    conversation[i]['content'] = self.spoken_response_text
                    conversation[i]['metadata'] = interrupt_info
                    conversation[i]['timestamp'] = datetime.now().isoformat()
                    
                    # Also add a note about the interruption
                    conversation.append({
                        'role': 'system',
                        'content': f"[Response interrupted after {interrupt_info['response_duration']:.1f}s - {interrupt_info['completed_sentences']}/{interrupt_info['sentence_count']} sentences completed]",
                        'timestamp': datetime.now().isoformat(),
                        'type': 'interrupt_log'
                    })
                    
                    # Save updated conversation
                    self.conversation_logger._save_conversation(conversation)
                    logger.debug("Updated conversation log with actual spoken text", "VA")
                    break
                    
        except Exception as e:
            logger.error(f"Error updating conversation log with actual speech: {e}", "VA")
    
    def _add_interrupt_metadata_to_log(self):
        """Add interrupt metadata as a system message in the conversation log"""
        if not self.playback_interrupted_at or not self.response_start_time:
            return
            
        try:
            interrupt_duration = self.playback_interrupted_at - self.response_start_time
            completed_sentences = len([t for t in self.sentence_timings if t.get('play_end')])
            total_sentences = len(self.sentence_timings)
            
            metadata_message = (
                f"[INTERRUPT: Response cut off after {interrupt_duration:.1f}s, "
                f"{completed_sentences}/{total_sentences} sentences completed]"
            )
            
            # Add as system message for context
            conversation = self.conversation_logger._load_conversation()
            conversation.append({
                'role': 'system',
                'content': metadata_message,
                'timestamp': datetime.now().isoformat(),
                'type': 'interrupt_metadata'
            })
            self.conversation_logger._save_conversation(conversation)
            
        except Exception as e:
            logger.error(f"Error adding interrupt metadata: {e}", "VA")
    
    def _log_assistant_response(self, response: str):
        """Log the assistant response after synthesis is complete, accounting for interrupts"""
        if not self.log_conversations:
            return
            
        logger.info("🗂️ CONVERSATION LOGGING (after synthesis complete)...")
        logger.info(f"🗂️ Playback interrupted: {self.playback_interrupted_at is not None}")
        logger.info(f"🗂️ Spoken response text: '{self.spoken_response_text[:50] if self.spoken_response_text else 'None'}...'")
        logger.info(f"🗂️ Full response text: '{response[:50]}...'")
        logger.info(f"🗂️ Response start time: {self.response_start_time}")
        logger.info(f"🗂️ Current time: {time.time()}")
        
        # Always log something - either interrupted or complete response
        if self.playback_interrupted_at:
            # Log the interrupted (partial) response
            if self.spoken_response_text and len(self.spoken_response_text.strip()) > 0:
                logger.info(f"🗂️ Logging interrupted response: '{self.spoken_response_text[:50]}...'")
                self.conversation_logger.log_message("assistant", self.spoken_response_text, allow_incomplete=True)
            else:
                # If no spoken text calculated, use a fallback based on timing
                logger.info("🗂️ No spoken text calculated, using fallback for interrupted response")
                words = response.split()
                # Estimate based on interrupt timing (conservative)
                estimated_words = min(5, len(words) // 4)  # Assume ~25% was heard
                if estimated_words > 0:
                    fallback_text = ' '.join(words[:estimated_words]) + "..."
                    self.conversation_logger.log_message("assistant", fallback_text, allow_incomplete=True)
                else:
                    # At minimum, log the first few words
                    fallback_text = ' '.join(words[:2]) + "..." if len(words) > 1 else response[:20] + "..."
                    self.conversation_logger.log_message("assistant", fallback_text, allow_incomplete=True)
            
            # Add interrupt metadata
            self._add_interrupt_metadata_to_log()
        else:
            # Log the complete response
            logger.info(f"🗂️ Logging complete response: '{response[:50]}...'")
            self.conversation_logger.log_message("assistant", response)
    
    def _wait_for_synthesis_with_interrupt_detection(self, synthesis_thread):
        """Wait for synthesis to complete while allowing interrupt detection"""
        max_wait_time = 120  # Maximum wait time in seconds
        check_interval = 0.1  # Check every 100ms
        start_time = time.time()
        
        while synthesis_thread.is_alive() and (time.time() - start_time) < max_wait_time:
            # Check if we've been interrupted
            if self.playback_interrupted_at:
                logger.debug("Synthesis wait interrupted by user speech", "VA")
                # Don't break immediately - let the synthesis thread clean up
                # But we know the state has changed
                
            time.sleep(check_interval)
        
        # If synthesis thread is still alive after max wait, something is wrong
        if synthesis_thread.is_alive():
            logger.warning(f"Synthesis thread still alive after {max_wait_time}s timeout", "VA")
        else:
            logger.debug("Synthesis thread completed successfully", "VA")
    
    def _check_and_log_pending_response(self):
        """Check if there's a pending response to log and log it"""
        if hasattr(self, '_pending_response_log') and self._pending_response_log:
            pending = self._pending_response_log
            
            # Check if this is the synthesis thread that just completed
            current_thread = threading.current_thread()
            if pending['thread'] == current_thread or not pending['thread'].is_alive():
                logger.debug("Logging response after synthesis completion", "VA")
                self._log_assistant_response(pending['response'])
                
                # Clear the pending response
                self._pending_response_log = None
    
    def _play_sound_effect(self, sound_type: str, context: str = "normal"):
        """Play a sound effect in a non-blocking way with enhanced context awareness"""
        if not self.enable_sound_effects:
            print(f"🔇 Sound effects disabled - skipping {sound_type}")
            return
        
        try:
            sound = None
            print(f"🎵 Generating {sound_type} sound effect (theme: {self.sound_theme}, context: {context})")
            
            # Generate contextual sounds based on type and context
            if sound_type == "interruption" and self.enable_interruption_sound:
                if context == "gentle":
                    sound = SoundEffects.generate_contextual_sound(
                        "interrupt_gentle", self.sound_theme, self.sample_rate, self.sound_effect_volume
                    )
                elif context == "urgent":
                    sound = SoundEffects.generate_contextual_sound(
                        "interrupt_urgent", self.sound_theme, self.sample_rate, self.sound_effect_volume
                    )
                else:
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
                if context == "success":
                    sound = SoundEffects.generate_contextual_sound(
                        "completion_success", self.sound_theme, self.sample_rate, self.sound_effect_volume
                    )
                else:
                    sound = SoundEffects.generate_completion_sound(
                        sample_rate=self.sample_rate,
                        volume=self.sound_effect_volume
                    )
            
            elif sound_type == "processing":
                print(f"🎵 [DEBUG] Generating processing sound effect")
                sound = SoundEffects.generate_processing_sound(
                    sample_rate=self.sample_rate,
                    volume=self.sound_effect_volume
                )
            
            elif sound_type == "processing_progress":
                sound = SoundEffects.generate_contextual_sound(
                    "processing_progress", self.sound_theme, self.sample_rate, self.sound_effect_volume
                )
            
            elif sound_type == "transcription_complete" and self.enable_transcription_sound:
                sound = SoundEffects.generate_contextual_sound(
                    "transcription_complete", self.sound_theme, self.sample_rate, self.sound_effect_volume
                )
            
            else:
                print(f"⚠️ Unknown sound type: {sound_type}")
                return
            
            if sound is None:
                print(f"⚠️ Failed to generate sound for type: {sound_type}")
                return
            
            # Apply fade to prevent audio clicks
            sound = SoundEffects.apply_fade(sound, self.sound_fade_duration, self.sample_rate)
            
            print(f"🎵 Playing {sound_type} sound: {len(sound)} samples, {len(sound)/self.sample_rate:.2f}s duration")
            
            # Play in separate thread to avoid blocking
            def play_sound():
                try:
                    print(f"🔊 [DEBUG] Starting playback of {sound_type} sound")
                    AudioStreamManager.play_audio_simple(sound, self.sample_rate)
                    print(f"✅ [DEBUG] Completed playback of {sound_type} sound")
                except Exception as e:
                    print(f"❌ Error playing sound effect {sound_type}: {e}")
                    import traceback
                    traceback.print_exc()
            
            sound_thread = threading.Thread(target=play_sound, daemon=True)
            sound_thread.start()
            
        except Exception as e:
            print(f"❌ Error generating sound effect {sound_type}: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start the voice assistant"""
        if self.is_running:
            print("Voice Assistant is already running")
            return
        
        print("Starting Voice Assistant...")
        self.is_running = True
        self._stop_event.clear()
        
        # Start audio input stream with both callbacks
        self.audio_manager.start_input_stream(
            callback=self._audio_callback,  # For complete utterances  
            interrupt_callback=self._interrupt_detection_callback  # For immediate interrupt detection
        )
        
        # Set up audio capture for interrupt tracking
        self.audio_manager.set_audio_capture_callback(self._capture_played_audio)
        
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
            # Interrupt detection is now handled separately by _interrupt_detection_callback
            if audio_chunk is not None and len(audio_chunk) > 0:
                try:
                    logger.vad("Complete utterance received - processing for transcription...")
                    
                    # Process the complete speech utterance (interrupts already handled separately)
                    self._process_speech(audio_chunk)
                    
                except Exception as e:
                    print(f"[VA ERROR] Error processing speech: {e}")
                    traceback.print_exc()
    
    def _interrupt_detection_callback(self, audio_chunk: np.ndarray):
        """Handle immediate interrupt detection when speech starts (called from VAD on first speech frame)"""
        if not self.is_running:
            return
            
        try:
            current_time = time.time()
            
            # Check interrupt cooldown period
            from configs.config import INTERRUPT_COOLDOWN_PERIOD, MIN_RESPONSE_TIME_BEFORE_INTERRUPT
            if current_time - self.last_interrupt_time < INTERRUPT_COOLDOWN_PERIOD:
                logger.interrupt(f"Interrupt blocked - cooldown period active ({current_time - self.last_interrupt_time:.1f}s < {INTERRUPT_COOLDOWN_PERIOD}s)")
                return
            
            # Check minimum response time
            if self.response_start_time and (current_time - self.response_start_time) < MIN_RESPONSE_TIME_BEFORE_INTERRUPT:
                logger.interrupt(f"Interrupt blocked - response just started ({current_time - self.response_start_time:.1f}s < {MIN_RESPONSE_TIME_BEFORE_INTERRUPT}s)")
                return
            
            logger.interrupt("Immediate speech detection triggered!")
            logger.interrupt(f"Current state - is_speaking: {self.is_speaking}, is_processing: {self.is_processing}")
            
            # If we're speaking, IMMEDIATELY stop audio playback
            if self.is_speaking:
                # Update last interrupt time
                self.last_interrupt_time = current_time
                logger.interrupt("Stopping audio playback immediately!")
                self._play_sound_effect("interruption", "gentle")
                
                # Record interrupt timing for conversation logging
                self.playback_interrupted_at = time.time()
                logger.info(f"🚨 INTERRUPT DETECTED at {self.playback_interrupted_at}")
                logger.info(f"🚨 Current response text: '{self.current_response_text}'")
                logger.info(f"🚨 Current spoken text: '{self.spoken_response_text}'")
                logger.info(f"🚨 Response start time: {self.response_start_time}")
                logger.info(f"🚨 Sentence timings count: {len(self.sentence_timings)}")
                self._finalize_spoken_text()  # Calculate what was actually heard
                logger.info(f"🚨 FINALIZED spoken text: '{self.spoken_response_text}'")
                
                # 1. Stop audio playback RIGHT NOW
                try:
                    self.audio_manager.stop_output_stream()
                    self.audio_manager.clear_queues()
                    logger.interrupt("Audio stopped - old TTS can keep running in background")
                except Exception as e:
                    print(f"⚠️ [INTERRUPT] Error stopping audio: {e}")
                
                # 2. Set flag so background synthesis threads don't play their audio
                self.synthesis_interrupted.set()
                self.cancel_processing.set()  # Also cancel any ongoing synthesis
                logger.interrupt("Flagged background synthesis to not play audio and stop synthesis")
                
                # 3. Cancel all active synthesis threads
                with self.synthesis_lock:
                    for thread in self.active_synthesis_threads[:]:  # Copy to avoid modification during iteration
                        logger.interrupt(f"Cancelling synthesis thread: {thread.name}")
                    # Clear the list since we're cancelling everything
                    self.active_synthesis_threads.clear()
                
                # 4. Reset speaking state so new input can be processed
                self.is_speaking = False
                self.synthesis_start_time = None
                
                # 4. That's it! Let old TTS finish in background, we don't care
                logger.interrupt("Ready for new input (ignoring background synthesis)")
            else:
                logger.interrupt("Assistant not speaking - treating as normal speech input")
            
            logger.interrupt("Interrupt detection complete - waiting for complete utterance...")
            
        except Exception as e:
            print(f"❌ [INTERRUPT] Error in interrupt detection: {e}")
            import traceback
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
        
        # NO GRACE PERIOD FOR PTT EITHER - IMMEDIATE INTERRUPTS!
        # (Removed grace period check - if user presses PTT, they want to interrupt NOW!)
        
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
            # Start with base system prompt
            base_system_prompt = self.llm.system_prompt
            memory_introduction = "\n\n[EXTENDED CONTEXTUAL MEMORY FOLLOWS]:\n"
            final_memory_content_parts = []

            if self.log_conversations and hasattr(self, 'hierarchical_memory_manager'):
                # Only update memory hierarchy every 60 seconds to reduce overhead further
                current_time = time.time()
                if not hasattr(self, '_last_memory_update') or (current_time - self._last_memory_update) > 60:
                    with TimerContext("memory_update", log_result=False):
                        logger.debug("Updating memory hierarchy...", "VA")
                        self._process_unsummarized_conversations()
                        self.hierarchical_memory_manager.update_memory_hierarchy()
                        self._last_memory_update = current_time

                # Define how many of each to load
                MAX_LTMS = 3
                MAX_STMS = 3
                MAX_SUMMARIES = 3

                processed_stm_paths = set()
                processed_conv_summary_paths = set()

                # 1. Load Latest LTMs
                # Load LTMs silently
                latest_ltm_list = self.conversation_logger.get_latest_ltm_summaries(max_ltm_summaries=MAX_LTMS)
                if latest_ltm_list:
                    for ltm in reversed(latest_ltm_list):
                        ltm_content = ltm.get('content', '').strip()
                        if ltm_content:
                            final_memory_content_parts.append(f"[Long-Term Memory]:\n{ltm_content}")
                            for stm_path in ltm.get('constituent_stms', []):
                                processed_stm_paths.add(str(Path(stm_path)))

                # 2. Load Latest STMs
                # Load STMs silently
                stms = self.conversation_logger.get_latest_stm_summaries(max_stm_summaries=MAX_STMS * 2)
                for stm in reversed(stms):
                    stm_file_path = str(Path(stm.get('file_path', '')))
                    if stm_file_path not in processed_stm_paths:
                        stm_content = stm.get('content', '').strip()
                        if stm_content:
                            final_memory_content_parts.append(f"[Short-Term Memory]:\n{stm_content}")
                            for conv_sum_path in stm.get('constituent_summaries', []):
                                processed_conv_summary_paths.add(str(Path(conv_sum_path)))
                            if len([p for p in final_memory_content_parts if '[Short-Term Memory]' in p]) >= MAX_STMS:
                                break

                # 3. Load Latest Individual Summaries
                # Load conversation summaries silently
                summaries = self.conversation_logger.get_latest_summaries(max_summaries=MAX_SUMMARIES * 2)
                for summary in reversed(summaries):
                    original_file_path = str(Path(summary.get('file_path', '')))
                    if original_file_path not in processed_conv_summary_paths:
                        if summary.get('messages') and isinstance(summary['messages'], list):
                            for msg in summary['messages']:
                                if msg.get('role') == 'assistant' and msg.get('content'):
                                    final_memory_content_parts.append(f"[Recent Conversation]:\n{msg['content'].strip()}")
                                    break
                        if len([p for p in final_memory_content_parts if '[Recent Conversation]' in p]) >= MAX_SUMMARIES:
                            break

                # Construct comprehensive system prompt
                comprehensive_system_prompt = base_system_prompt
                if final_memory_content_parts:
                    comprehensive_system_prompt += memory_introduction + "\n\n".join(final_memory_content_parts)
                    logger.debug(f"Loaded {len(final_memory_content_parts)} memory entries", "VA")
            else:
                comprehensive_system_prompt = base_system_prompt
                logger.debug("Using base system prompt (no memory available)", "VA")

            # Start with system message
            messages = [{
                "role": "system",
                "content": comprehensive_system_prompt
            }]

            # Add recent conversation history from logged conversations (with accurate interrupt tracking)
            if self.log_conversations and hasattr(self, 'conversation_logger'):
                current_conversation = self.conversation_logger.get_current_conversation()
                if current_conversation:
                    # Skip system messages and get recent dialog
                    history_messages = [msg for msg in current_conversation if msg['role'] != 'system']
                    # Take last few turns (e.g., last 4 messages = 2 turns)
                    recent_history = history_messages[-4:]
                    messages.extend(recent_history)
                    logger.debug(f"Added {len(recent_history)} logged messages with interrupt tracking", "VA")
            elif self.conversation_history:
                # Fallback to in-memory history if logging unavailable
                history_messages = [msg for msg in self.conversation_history if msg['role'] != 'system']
                recent_history = history_messages[-4:]
                messages.extend(recent_history)
                logger.debug(f"Added {len(recent_history)} fallback messages", "VA")

            # Add current user message
            messages.append({
                "role": "user",
                "content": user_text
            })

            # Final message count (removed debug spam)
            return messages

        except Exception as e:
            print(f"[VA ERROR] Error preparing messages: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}", "VA")
            # Fallback to basic message structure
            return [
                {"role": "system", "content": self.llm.system_prompt},
                {"role": "user", "content": user_text}
            ]

    def _validate_transcription_quality(self, text: str, audio: np.ndarray) -> bool:
        """Validate transcription quality to prevent false positives and hallucinations"""
        from configs.config import (
            MIN_AUDIO_DURATION_FOR_TRANSCRIPTION, 
            WHISPER_CONFIDENCE_THRESHOLD
        )
        
        # Check audio duration
        audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
        if audio_duration < MIN_AUDIO_DURATION_FOR_TRANSCRIPTION:
            logger.debug(f"Audio too short: {audio_duration:.2f}s < {MIN_AUDIO_DURATION_FOR_TRANSCRIPTION}s", "VA")
            return False
        
        # Removed amplitude validation - if Whisper transcribed it, it's valid speech
        
        # Check for obvious hallucinations (common false positives)
        text_lower = text.lower().strip()
        hallucination_phrases = [
            "thank you", "thanks", "okay", "ok", "yes", "no", "mm-hmm", "uh-huh",
            "the", "a", "an", "and", "or", "but", "so", "well", "um", "uh"
        ]
        
        # If the entire transcription is just a single common word/phrase, it's likely a hallucination
        if text_lower in hallucination_phrases and audio_duration < 2.0:
            logger.debug(f"Likely hallucination detected: '{text}' from {audio_duration:.2f}s audio", "VA")
            return False
        
        return True
    
    def _is_repeated_input(self, text: str) -> bool:
        """Check if this is a repeated input (likely false positive)"""
        if text == self.last_user_input:
            self.input_repeat_count += 1
            if self.input_repeat_count >= 3:
                logger.warning(f"Repeated input detected {self.input_repeat_count} times: '{text}' - blocking", "VA")
                return True
        else:
            self.input_repeat_count = 0
            self.last_user_input = text
        return False
    
    def _filter_system_text_from_transcription(self, text: str) -> str:
        """Filter out system prompt text that might appear in transcription results"""
        # Clean the text
        cleaned_text = text.strip()
        
        # Check if we're at the start of a conversation (no logged messages yet)
        is_conversation_start = True
        if self.log_conversations and hasattr(self, 'conversation_logger'):
            try:
                current_conversation = self.conversation_logger.get_current_conversation()
                if current_conversation:
                    # Check if there are any user/assistant messages (not just system messages)
                    user_messages = [msg for msg in current_conversation if msg['role'] in ['user', 'assistant']]
                    is_conversation_start = len(user_messages) == 0
            except:
                is_conversation_start = True
        
        # Only filter system text at conversation start or if it's an exact match
        system_texts_to_filter = [
            "The following is a conversation between a user and an AI assistant",
            "The following is a conversation between a user and an AI assistant.",
            "[EXTENDED CONTEXTUAL MEMORY FOLLOWS]",
            "The assistant was speaking when interrupted",
            "The assistant was speaking when interrupted."
        ]
        
        # Check for exact system text matches (likely contamination)
        for system_text in system_texts_to_filter:
            # Only filter if:
            # 1. It's an exact match (entire transcription is just system text), OR
            # 2. It's at conversation start AND starts with system text
            if (cleaned_text.lower() == system_text.lower() or 
                (is_conversation_start and cleaned_text.lower().startswith(system_text.lower()))):
                
                # If exact match, likely contamination - filter completely
                if cleaned_text.lower() == system_text.lower():
                    logger.debug(f"Filtered exact system text match: '{system_text}'", "VA")
                    return ""
                
                # If starts with system text at conversation start, remove prefix
                if is_conversation_start:
                    remaining_text = cleaned_text[len(system_text):].strip()
                    remaining_text = remaining_text.lstrip('.,!?:;\n\t ')
                    if remaining_text and len(remaining_text) > 3:
                        logger.debug(f"Filtered system text prefix: '{system_text}'", "VA")
                        return remaining_text
                    else:
                        logger.debug(f"Filtered system text (no remaining content): '{system_text}'", "VA")
                        return ""
        
        # If we get here, it's either:
        # - Mid-conversation (user intentionally said these words)
        # - Not an exact system text match
        # - Contains additional content beyond system text
        # In all cases, preserve the user's speech
        return cleaned_text
    
    def _process_speech(self, audio: np.ndarray):
        """Process recorded speech"""
        try:
            logger.debug("Starting speech processing...", "VA")
            
            # Simple, elegant transcription - trust Whisper's defaults
            logger.debug("Transcribing audio...", "VA")
            transcription_result = self.stt.transcribe(audio)
            
            # Extract text simply
            user_text = transcription_result['text'].strip() if isinstance(transcription_result, dict) else str(transcription_result).strip()
            logger.debug(f"Transcription complete: {user_text}", "VA")
            
            # Simple validation - just check if we got text
            if not user_text:
                logger.debug("No transcription produced", "VA")
                return
            
            # Play transcription complete sound effect
            if user_text and self.enable_transcription_sound:  # Only play if we got actual text and sound is enabled
                self._play_sound_effect("transcription_complete")
            
            # Check for trigger phrase to activate new conversation
            if self._check_new_conversation_trigger(user_text):
                logger.info("New conversation trigger detected")
                self._activate_new_conversation()
                return
            
            # Call transcription callback
            if self.on_transcription:
                logger.debug("Calling transcription callback...", "VA")
                self.on_transcription(user_text)
            
            # Log user message (now with filtered text)
            if self.log_conversations:
                logger.debug("Logging filtered user message...", "VA")
                self.conversation_logger.log_message("user", user_text)
            
            # Generate response
            logger.debug("Starting response generation...", "VA")
            messages = self._prepare_messages_for_llm(user_text)
            # Prepared messages (removed duplicate debug)
            
            try:
                logger.debug("Calling LLM for response...", "VA")
                # Use chat method instead of generate for proper conversation handling
                with TimerContext("llm_generation"):
                    response = self.llm.chat(
                        messages=messages,
                        max_tokens=self.max_response_tokens,
                        temperature=self.llm_temperature
                    )
                logger.debug(f"LLM response received: {response[:50]}...", "VA")
                
                # Store the full response for interrupt tracking
                self.current_response_text = response
                self.spoken_response_text = ""  # Reset for new response
                self.response_start_time = time.time()
                self.playback_interrupted_at = None
                self.sentence_timings = []
                self.played_audio_buffer = []  # Reset audio buffer for new response
                self._whisper_transcription_used = False  # Reset flag for new response
                
                # Clear any previous synthesis operations before starting new response
                self.cancel_processing.set()
                with self.synthesis_lock:
                    self.active_synthesis_threads.clear()
                time.sleep(0.1)  # Brief pause to let previous operations stop
                self.cancel_processing.clear()  # Clear for new synthesis
                
                # Add messages to conversation history
                if response:
                    self.conversation_history.append({
                        "role": "user",
                        "content": user_text
                    })
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                
                # Call response callback
                if self.on_response:
                    logger.debug("Calling response callback...", "VA")
                    self.on_response(response)
                
                # Synthesize speech - logging will happen when synthesis is actually done
                with TimerContext("speech_synthesis"):
                    synthesis_thread = self.say(response)
                    
                    # Store the response and thread for later logging
                    self._pending_response_log = {
                        'response': response,
                        'thread': synthesis_thread,
                        'start_time': time.time()
                    }
                
            except Exception as e:
                logger.error(f"Error generating/synthesizing response: {e}", "VA")
                logger.error(f"Traceback: {traceback.format_exc()}", "VA")
                # Don't synthesize error messages - just log the error
                # This avoids the awkward situation of speaking error messages
                if self.on_response:
                    self.on_response("[Error: Unable to generate response]")
            
        except Exception as e:
            logger.error(f"Error in speech processing: {e}", "VA")
            logger.error(f"Traceback: {traceback.format_exc()}", "VA")
            error_msg = "I'm sorry, I encountered an error while processing your speech."
            self.say(error_msg)
    
    def _stream_generate_and_synthesize(self, messages: List[Dict[str, str]], user_text: str):
        """Stream generate response and synthesize with SEQUENTIAL processing for efficiency"""
        try:
            # Clear any previous cancellation and interruption flags
            self.cancel_processing.clear()
            self.synthesis_interrupted.clear()
            
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
            spoken_response = []  # Track only the text that was actually spoken
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
                # Check for cancellation or interruption
                if self.cancel_processing.is_set() or self.synthesis_interrupted.is_set():
                    print("🚫 [LLM] Stopping response generation due to interrupt")
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
                                    
                                    # Check for cancellation before synthesis
                                    if self.cancel_processing.is_set():
                                        print("🚫 [TTS] Synthesis cancelled before starting")
                                        break
                                    
                                    # Run synthesis in separate thread to allow interruption
                                    synthesis_result = {"audio": None, "sample_rate": None, "error": None}
                                    
                                    def synthesis_worker():
                                        try:
                                            start_time = time.time()
                                            
                                            # Validate text length for TTS
                                            if len(chunk_text.strip()) < 3:
                                                # Pad very short text to avoid kernel size errors
                                                padded_text = chunk_text.strip() + "."
                                                print(f"   🔧 Padding short text: '{chunk_text}' -> '{padded_text}'")
                                                chunk_text = padded_text
                                            
                                            audio, sample_rate = self.tts.synthesize(chunk_text)
                                            synthesis_time = time.time() - start_time
                                            synthesis_result["audio"] = audio
                                            synthesis_result["sample_rate"] = sample_rate
                                            print(f"   ✅ Synthesis completed in {synthesis_time:.2f}s")
                                            
                                        except Exception as e:
                                            # Handle specific TTS errors
                                            error_str = str(e)
                                            if "Kernel size can't be greater than actual input size" in error_str:
                                                print(f"   🔧 TTS kernel size error - retrying with padded text")
                                                try:
                                                    # Retry with padded text
                                                    padded_text = chunk_text + " (pause)."
                                                    audio, sample_rate = self.tts.synthesize(padded_text)
                                                    synthesis_result["audio"] = audio
                                                    synthesis_result["sample_rate"] = sample_rate
                                                    print(f"   ✅ Retry successful with padded text")
                                                except Exception as retry_error:
                                                    synthesis_result["error"] = retry_error
                                                    print(f"   ❌ Retry failed: {retry_error}")
                                            else:
                                                synthesis_result["error"] = e
                                                print(f"   ❌ Synthesis error: {e}")
                                    
                                    # Start synthesis in background thread
                                    synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True, name=f"TTS-Chunk-{chunk_count}")
                                    synthesis_thread.start()
                                    
                                    # Track the thread
                                    with self.synthesis_lock:
                                        self.active_synthesis_threads.append(synthesis_thread)
                                    
                                    # Wait for synthesis to complete or cancellation
                                    max_wait_time = 10.0  # Maximum wait time for synthesis
                                    wait_start = time.time()
                                    
                                    while synthesis_thread.is_alive() and time.time() - wait_start < max_wait_time:
                                        if self.cancel_processing.is_set():
                                            print("🚫 [TTS] Synthesis interrupted by user!")
                                            # Remove from tracking and break
                                            with self.synthesis_lock:
                                                if synthesis_thread in self.active_synthesis_threads:
                                                    self.active_synthesis_threads.remove(synthesis_thread)
                                            break
                                        time.sleep(0.1)  # Check every 100ms
                                    
                                    # Clean up thread tracking
                                    with self.synthesis_lock:
                                        if synthesis_thread in self.active_synthesis_threads:
                                            self.active_synthesis_threads.remove(synthesis_thread)
                                    
                                    # Check if we were cancelled or if synthesis failed
                                    if self.cancel_processing.is_set():
                                        print("🚫 [TTS] Synthesis cancelled - stopping chunk processing")
                                        break
                                    
                                    if synthesis_result["error"]:
                                        print(f"❌ [TTS] Synthesis failed: {synthesis_result['error']}")
                                        continue  # Skip this chunk and continue with next
                                    
                                    if synthesis_result["audio"] is None or len(synthesis_result["audio"]) == 0:
                                        print("⚠️ [TTS] No audio generated from synthesis")
                                        continue
                                    
                                    audio = synthesis_result["audio"]
                                    sample_rate = synthesis_result["sample_rate"]
                                    audio_duration = len(audio) / sample_rate
                                    
                                    print(f"   ✅ Ready: {synthesis_time:.2f}s synthesis → {audio_duration:.1f}s audio")
                                    
                                    # Play immediately and wait for completion
                                    if not self.cancel_processing.is_set() and not self.synthesis_interrupted.is_set():
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
                                    elif self.synthesis_interrupted.is_set():
                                        print("🚫 [CHUNK] Not playing audio - synthesis was interrupted")
                                    
                                    # Clear collected sentences for next chunk
                                    collected_sentences = []
                            break
            
            # Handle any remaining sentences
            if collected_sentences:
                chunk_text = " ".join(collected_sentences)
                chunk_count += 1
                print(f"\n🎵 Final chunk {chunk_count} ({len(collected_sentences)} sentences, {len(chunk_text)} chars):")
                print(f"   📖 '{chunk_text[:60]}{'...' if len(chunk_text) > 60 else ''}'")
                
                # Check for cancellation before synthesis
                if self.cancel_processing.is_set():
                    print("🚫 [TTS] Final chunk synthesis cancelled before starting")
                    return
                
                # Run final chunk synthesis in separate thread
                synthesis_result = {"audio": None, "sample_rate": None, "error": None}
                
                def synthesis_worker():
                    try:
                        start_time = time.time()
                        audio, sample_rate = self.tts.synthesize(chunk_text)
                        synthesis_time = time.time() - start_time
                        synthesis_result["audio"] = audio
                        synthesis_result["sample_rate"] = sample_rate
                        print(f"   ✅ Final chunk synthesis completed in {synthesis_time:.2f}s")
                    except Exception as e:
                        synthesis_result["error"] = e
                        print(f"   ❌ Final chunk synthesis error: {e}")
                
                synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True, name="TTS-FinalChunk")
                synthesis_thread.start()
                
                # Track the thread
                with self.synthesis_lock:
                    self.active_synthesis_threads.append(synthesis_thread)
                
                # Wait for synthesis with interruption check
                max_wait_time = 10.0
                wait_start = time.time()
                
                while synthesis_thread.is_alive() and time.time() - wait_start < max_wait_time:
                    if self.cancel_processing.is_set():
                        print("🚫 [TTS] Final chunk synthesis interrupted!")
                        with self.synthesis_lock:
                            if synthesis_thread in self.active_synthesis_threads:
                                self.active_synthesis_threads.remove(synthesis_thread)
                        return
                    time.sleep(0.1)
                
                # Clean up tracking
                with self.synthesis_lock:
                    if synthesis_thread in self.active_synthesis_threads:
                        self.active_synthesis_threads.remove(synthesis_thread)
                
                if self.cancel_processing.is_set():
                    return
                
                if synthesis_result["error"] or synthesis_result["audio"] is None:
                    print(f"❌ [TTS] Final chunk synthesis failed")
                    return
                
                audio = synthesis_result["audio"]
                sample_rate = synthesis_result["sample_rate"]
                audio_duration = len(audio) / sample_rate
                
                print(f"   ✅ Ready: {synthesis_time:.2f}s synthesis → {audio_duration:.1f}s audio")
                
                if not self.cancel_processing.is_set() and not self.synthesis_interrupted.is_set():
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
                elif self.synthesis_interrupted.is_set():
                    print("🚫 [FINAL] Not playing audio - synthesis was interrupted")
            
            # Handle any remaining partial sentence
            if current_sentence.strip():
                chunk_count += 1
                print(f"\n🎵 Partial sentence chunk {chunk_count}: '{current_sentence[:40]}{'...' if len(current_sentence) > 40 else ''}'")
                
                # Check for cancellation before synthesis
                if self.cancel_processing.is_set():
                    print("🚫 [TTS] Partial sentence synthesis cancelled before starting")
                    return
                
                # Run partial sentence synthesis in separate thread
                synthesis_result = {"audio": None, "sample_rate": None, "error": None}
                
                def synthesis_worker():
                    try:
                        start_time = time.time()
                        audio, sample_rate = self.tts.synthesize(current_sentence)
                        synthesis_time = time.time() - start_time
                        synthesis_result["audio"] = audio
                        synthesis_result["sample_rate"] = sample_rate
                        print(f"   ✅ Partial sentence synthesis completed in {synthesis_time:.2f}s")
                    except Exception as e:
                        synthesis_result["error"] = e
                        print(f"   ❌ Partial sentence synthesis error: {e}")
                
                synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True, name="TTS-PartialSentence")
                synthesis_thread.start()
                
                # Track the thread
                with self.synthesis_lock:
                    self.active_synthesis_threads.append(synthesis_thread)
                
                # Wait for synthesis with interruption check
                max_wait_time = 10.0
                wait_start = time.time()
                
                while synthesis_thread.is_alive() and time.time() - wait_start < max_wait_time:
                    if self.cancel_processing.is_set():
                        print("🚫 [TTS] Partial sentence synthesis interrupted!")
                        with self.synthesis_lock:
                            if synthesis_thread in self.active_synthesis_threads:
                                self.active_synthesis_threads.remove(synthesis_thread)
                        return
                    time.sleep(0.1)
                
                # Clean up tracking
                with self.synthesis_lock:
                    if synthesis_thread in self.active_synthesis_threads:
                        self.active_synthesis_threads.remove(synthesis_thread)
                
                if self.cancel_processing.is_set():
                    return
                
                if synthesis_result["error"] or synthesis_result["audio"] is None:
                    print(f"❌ [TTS] Partial sentence synthesis failed")
                    return
                
                audio = synthesis_result["audio"]
                sample_rate = synthesis_result["sample_rate"]
                audio_duration = len(audio) / sample_rate
                
                print(f"   ✅ Ready: {synthesis_time:.2f}s synthesis → {audio_duration:.1f}s audio")
                
                if not self.cancel_processing.is_set() and not self.synthesis_interrupted.is_set():
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
                elif self.synthesis_interrupted.is_set():
                    print("🚫 [PARTIAL] Not playing audio - synthesis was interrupted")
            
            # Make sure all audio is finished
            self.audio_manager.wait_for_playback_complete()
            
            # Play completion sound effect
            self._play_sound_effect("completion")
            
            # Update conversation history with raw text
            full_response_text = "".join(full_response).strip()
            if full_response_text:
                # Check for duplicate or incomplete messages
                should_log = True
                if self.conversation_history:
                    # Get last few messages
                    recent_messages = self.conversation_history[-2:] if len(self.conversation_history) >= 2 else self.conversation_history
                    
                    for msg in recent_messages:
                        if msg['role'] == 'user' and self._is_similar_text(msg['content'], user_text):
                            # Skip duplicate user message
                            print("⚠️ Skipping duplicate user message")
                            should_log = False
                            break
                        elif msg['role'] == 'assistant' and self._is_similar_text(msg['content'], full_response_text):
                            # Skip duplicate assistant message
                            print("⚠️ Skipping duplicate assistant message")
                            should_log = False
                            break
                
                if should_log:
                    # For continuations, replace the last user message instead of adding new one
                    if self.is_continuation and self.conversation_history and self.conversation_history[-2]["role"] == "user":
                        # Replace the last user message with the combined text
                        self.conversation_history[-2]["content"] = user_text
                        print(f"🔄 Updated conversation history with combined user text")
                    else:
                        # Normal case: add new user message
                        self.conversation_history.append({"role": "user", "content": user_text})
                    
                    # Add the raw assistant response
                    self.conversation_history.append({"role": "assistant", "content": full_response_text})
                    
                    # Log user message if enabled (assistant message will be logged after synthesis with interrupt tracking)
                    if self.log_conversations:
                        self.conversation_logger.log_message("user", user_text)
                    
                    # Keep conversation history manageable by truncating if needed
                    if len(self.conversation_history) > self.max_history_messages:
                        print(f"📝 Truncating conversation history to {self.max_history_messages} messages")
                        # Keep most recent messages
                        self.conversation_history = self.conversation_history[-self.max_history_messages:]
                    
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
    
    def _is_similar_text(self, text1: str, text2: str, similarity_threshold: float = 0.8) -> bool:
        """Check if two texts are similar enough to be considered duplicates"""
        if not text1 or not text2:
            return False
            
        # Clean and normalize texts
        text1 = re.sub(r'\s+', ' ', text1.strip().lower())
        text2 = re.sub(r'\s+', ' ', text2.strip().lower())
        
        # Quick length check
        if abs(len(text1) - len(text2)) / max(len(text1), len(text2)) > 0.2:
            return False
        
        # Check if one is a substring of the other
        if text1 in text2 or text2 in text1:
            return True
        
        # Calculate similarity using character-based comparison
        shorter = text1 if len(text1) <= len(text2) else text2
        longer = text2 if len(text1) <= len(text2) else text1
        
        # Use sliding window to find best match
        max_similarity = 0
        window_size = len(shorter)
        
        for i in range(len(longer) - window_size + 1):
            window = longer[i:i + window_size]
            matches = sum(1 for a, b in zip(shorter, window) if a == b)
            similarity = matches / window_size
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity >= similarity_threshold
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS synthesis"""
        original_text = text
        
        # Remove thinking blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove stage directions
        text = re.sub(r'\*[^*]*\*', '', text)
        
        # Filter out common noise patterns - Enhanced version
        noise_patterns = [
            r'\b(?:um+|uh+|ah+|er+)\b',  # Filler sounds
            r'\b(?:background\s+noise|static)\b',
            r'(?:\s*\.\s*){3,}',  # Multiple dots
            r'\b(?:silence|pause)\b',
            r'(?:\s*[^\w\s]\s*){3,}',  # Multiple punctuation marks
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove system prompt leakage patterns
        prompt_patterns = [
            r'You are.*?assistant.*?voice',
            r'CRITICAL.*?RULES.*?',
            r'VOICE OPTIMIZATION.*?',
            r'PERSONALITY.*?',
            r'EXAMPLE.*?RESPONSES.*?',
            r'AVOID.*?:',
            r'def \w+.*?\(.*?\):',  # Remove Python code
            r'print\(.*?\)',        # Remove print statements
            r'hello\(.*?\)',        # Remove function calls
            r'# prints.*?',         # Remove code comments
            r'pythonYou are.*?',    # Remove malformed prompt text
            r'```.*?```',           # Remove code blocks
            r'""".*?"""',           # Remove docstrings
        ]
        
        for pattern in prompt_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove extra whitespace and clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Additional noise checks
        if text:
            words = text.split()
            # Check for suspicious patterns, but preserve "between a"
            if (len(words) > 2 and
                text.lower() != "between a" and  # Preserve exact "between a"
                text.lower() != "between a." and  # Preserve with period
                (len(set(words)) < len(words) / 3 or  # Too much repetition
                 (sum(len(w) < 3 for w in words) > len(words) / 2 and "between a" not in text.lower()))):  # Too many short words, unless it's "between a"
                print(f"⚠️ Suspicious text pattern detected, using original: '{text}'")
                text = ""
        
        # Remove empty sentences
        if not text or len(text) < 3:
            # If we had text originally but it got cleaned away, return original for conversational responses
            if original_text and len(original_text.strip()) > 3:
                return original_text.strip()
            return ""
        
        return text
    
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
            
        Returns:
            threading.Thread: The synthesis thread (so caller can wait for completion)
        """
        if interrupt and self.is_speaking:
            # Stop current speech properly
            if self.audio_manager.output_stream:
                self.audio_manager.stop_output_stream()
            self.audio_manager.clear_queues()
            self.cancel_processing.set()
            self.is_speaking = False
            time.sleep(0.1)  # Brief pause for cleanup
        
        # Always use streaming for better responsiveness
        thread = threading.Thread(
            target=self._say_streaming,
            args=(text,),
            daemon=True,
            name="TTS-Main"
        )
        
        thread.start()
        return thread  # Return thread so caller can wait for completion
    
    def _say_streaming(self, text: str):
        """Simple streaming synthesis for the say method"""
        try:
            # Clear any previous cancellation and interruption flags
            self.cancel_processing.clear()
            self.synthesis_interrupted.clear()
            
            self.is_speaking = True
            self.synthesis_start_time = time.time()  # Track synthesis start for interrupts
            
            # Reset interrupt attempts when starting new speech
            if hasattr(self, '_continuous_interrupt_attempts'):
                self._continuous_interrupt_attempts = 0
            
            # Split into sentences for better streaming
            sentences = self.tts._split_into_sentences(text)
            
            if not sentences:
                return
            
            for sentence in sentences:
                if self.cancel_processing.is_set():
                    break
                    
                logger.synthesis(f"Synthesizing: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
                
                # Check for cancellation before synthesis
                if self.cancel_processing.is_set():
                    logger.synthesis("Sentence synthesis cancelled")
                    break
                
                # Run synthesis in separate thread
                synthesis_result = {"audio": None, "sample_rate": None, "error": None}
                
                def synthesis_worker():
                    try:
                        audio, sample_rate = self.tts.synthesize(sentence)
                        synthesis_result["audio"] = audio
                        synthesis_result["sample_rate"] = sample_rate
                    except Exception as e:
                        synthesis_result["error"] = e
                        print(f"   ❌ Say synthesis error: {e}")
                
                synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True, name="TTS-Say")
                synthesis_thread.start()
                
                # Track the thread
                with self.synthesis_lock:
                    self.active_synthesis_threads.append(synthesis_thread)
                
                # Wait for synthesis with interruption check
                max_wait_time = 10.0
                wait_start = time.time()
                
                while synthesis_thread.is_alive() and time.time() - wait_start < max_wait_time:
                    if self.cancel_processing.is_set():
                        logger.synthesis("Synthesis interrupted!")
                        with self.synthesis_lock:
                            if synthesis_thread in self.active_synthesis_threads:
                                self.active_synthesis_threads.remove(synthesis_thread)
                        break
                    time.sleep(0.1)
                
                # Clean up tracking
                with self.synthesis_lock:
                    if synthesis_thread in self.active_synthesis_threads:
                        self.active_synthesis_threads.remove(synthesis_thread)
                
                if self.cancel_processing.is_set():
                    break
                
                if synthesis_result["error"] or synthesis_result["audio"] is None:
                    continue  # Skip this sentence and continue with next
                
                audio = synthesis_result["audio"]
                sample_rate = synthesis_result["sample_rate"]
                
                # Check if synthesis was interrupted before playing
                if not self.cancel_processing.is_set() and not self.synthesis_interrupted.is_set():
                    # Record sentence timing
                    audio_duration = len(audio) / sample_rate
                    sentence_timing = {
                        'text': sentence,
                        'play_start': time.time(),
                        'duration': audio_duration,
                        'play_end': None
                    }
                    self.sentence_timings.append(sentence_timing)
                    
                    # Play immediately
                    playback_thread = self.audio_manager.play_audio_streaming(
                        iter([(audio, sample_rate)]),
                        interrupt_current=False
                    )
                    if playback_thread:
                        # Use timeout to prevent indefinite blocking
                        playback_thread.join(timeout=max(2.0, audio_duration + 1.0))
                        
                        # Update end timing ONLY if playback completed normally AND no interrupt occurred
                        if (not playback_thread.is_alive() and 
                            not self.synthesis_interrupted.is_set() and 
                            not self.cancel_processing.is_set()):
                            
                            # Double-check that we weren't interrupted during this sentence
                            current_time = time.time()
                            if (not self.playback_interrupted_at or 
                                sentence_timing['play_start'] < self.playback_interrupted_at):
                                
                                sentence_timing['play_end'] = current_time
                                
                                # Only track as spoken if sentence started before interrupt AND Whisper hasn't already set the result
                                if not self.playback_interrupted_at or sentence_timing['play_start'] < self.playback_interrupted_at:
                                    if not self._whisper_transcription_used:  # Don't overwrite Whisper results
                                        if self.spoken_response_text:
                                            self.spoken_response_text += " " + sentence
                                        else:
                                            self.spoken_response_text = sentence
                                        logger.debug(f"Sentence completed: '{sentence}' - total spoken: '{self.spoken_response_text[:50]}...'", "VA")
                                    else:
                                        logger.debug(f"Whisper result already set - not overwriting with timing estimation", "VA")
                                else:
                                    logger.debug(f"Sentence '{sentence}' started after interrupt - not counting as spoken", "VA")
                        elif playback_thread.is_alive():
                            print(f"   ⚠️ Say streaming playback timeout, continuing...")
                elif self.synthesis_interrupted.is_set():
                    logger.synthesis("Not playing audio - synthesis was interrupted")
            
            # Wait for all audio to complete
            self.audio_manager.wait_for_playback_complete()
            
            # Finalize response tracking if completed normally (no interrupt)
            if not self.cancel_processing.is_set() and not self.synthesis_interrupted.is_set():
                if not self.spoken_response_text and self.current_response_text:
                    # Response completed fully without interruption
                    self.spoken_response_text = self.current_response_text
                logger.synthesis("Speech completed - playing completion sound")
                self._play_sound_effect("completion", "success")
            
            self.is_speaking = False
            self.synthesis_start_time = None
            
            # Check if there's a pending response to log
            self._check_and_log_pending_response()
            
        except Exception as e:
            print(f"Error in streaming say: {e}")
            self.is_speaking = False
            self.synthesis_start_time = None
    
    def set_voice(self, voice_path: str):
        """Change the assistant's voice"""
        self.tts.set_voice(voice_path)
        print(f"Voice changed to: {voice_path}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history and start new log file"""
        self.conversation_history.clear()
        if self.log_conversations:
            self.conversation_logger.start_new_conversation()
        print("🧹 Conversation history cleared")
    
    def _check_new_conversation_trigger(self, user_text: str) -> bool:
        """Check if user said the trigger phrase to activate a new conversation"""
        # Normalize the text for comparison
        normalized_text = user_text.lower().strip()
        
        # Check for variations of the trigger phrase
        trigger_phrases = [
            "new conversation activate",
            "new conversation, activate", 
            "activate new conversation",
            "activate a new conversation",
            "start new conversation",
            "start a new conversation",
            "begin new conversation", 
            "begin a new conversation",
            "reset conversation",
            "reset the conversation",
            "new conversation reset"
        ]
        
        for phrase in trigger_phrases:
            if phrase in normalized_text:
                return True
        
        return False
    
    def _activate_new_conversation(self):
        """Activate a new conversation with auto-summarization"""
        try:
            # First, process any unsummarized conversations from previous sessions
            if self.log_conversations and hasattr(self, 'conversation_summarizer'):
                print("🔄 Processing any pending conversation summaries...")
                self._process_unsummarized_conversations()
                
                # Update memory hierarchy to include the latest summaries
                if hasattr(self, 'hierarchical_memory_manager'):
                    print("🧠 Updating memory hierarchy...")
                    self.hierarchical_memory_manager.update_memory_hierarchy()
            
            # Clear current conversation and start new log
            self.clear_conversation_history()
            
            # Reload hierarchical memory with updated summaries
            if self.log_conversations and hasattr(self, 'hierarchical_memory_manager'):
                print("🔄 Reloading hierarchical memory with latest summaries...")
                self._load_hierarchical_memory()
            
            # Provide audio feedback
            confirmation_msg = "New conversation activated. Auto-summary processing complete. Ready for a fresh start!"
            print(f"✅ {confirmation_msg}")
            
            # Synthesize confirmation
            self.say(confirmation_msg, interrupt=False)
            
            # Log the activation event
            if self.log_conversations:
                self.conversation_logger.log_message("system", "New conversation activated by user trigger phrase")
                
        except Exception as e:
            error_msg = f"Sorry, I encountered an error while activating the new conversation: {str(e)}"
            print(f"❌ Error in _activate_new_conversation: {e}")
            self.say(error_msg, interrupt=False)
    
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
        """Load and summarize a previous conversation history"""
        if not self.log_conversations:
            print("⚠️ Conversation logging is disabled")
            return
            
        try:
            # Load the conversation
            conversation = self.conversation_logger.load_conversation_file(filepath)
            if not conversation:
                print("⚠️ No conversation found in file")
                return
                
            print(f"📚 Loaded {len(conversation)} messages from {filepath}")
            
            # Summarize the conversation
            print("🤖 Generating conversation summary...")
            summarized = self.conversation_summarizer.summarize_conversation(conversation)
            
            # Update conversation history
            self.conversation_history = summarized
            print(f"✅ Conversation history updated with {len(summarized)} summarized messages")
            
        except Exception as e:
            print(f"❌ Error loading conversation history: {e}")
    
    def get_conversation_files(self) -> List[str]:
        """Get list of available conversation log files"""
        if not self.log_conversations:
            return []
        return self.conversation_logger.get_conversation_files()
    
    def _process_unsummarized_conversations(self):
        """Process any conversations that haven't been summarized yet"""
        try:
            unsummarized = self.conversation_logger.get_unsummarized_conversations()
            if unsummarized:
                print(f"🔄 Found {len(unsummarized)} new conversations to summarize individually")
                current_file = str(self.conversation_logger.current_log_file)
                
                for filepath in unsummarized:
                    # Skip the current conversation file
                    if filepath == current_file:
                        print(f"⏩ Skipping current active conversation: {filepath}")
                        continue
                        
                    try:
                        print(f"\n📝 Summarizing new conversation: {filepath}")
                        conversation = self.conversation_logger.load_conversation_file(filepath)
                        
                        # Only summarize if there are messages
                        if conversation:
                            print("🤖 Generating summary (streaming):")
                            print("=" * 60)
                            
                            # Stream the summary generation and collect chunks
                            summary_chunks = []
                            for chunk in self.conversation_summarizer.stream_summarize_conversation(conversation):
                                print(chunk, end='', flush=True)
                                summary_chunks.append(chunk)
                            print("\n" + "=" * 60)
                            
                            # Create summary message
                            summary_text = "".join(summary_chunks).strip()
                            if not summary_text:
                                print("⚠️ Empty summary generated, using fallback")
                                summary_messages = self.conversation_summarizer._fallback_summary(conversation)
                            else:
                                summary_messages = [{
                                    "role": "assistant",
                                    "content": summary_text
                                }]
                            
                            # Save the summary
                            summary_file = self.conversation_logger.save_conversation_summary(filepath, summary_messages)
                            print(f"✅ Created new summary: {summary_file}")
                            
                            # Debug output
                            print("\n[DEBUG] Generated summary:")
                            print(f"  {summary_text[:500]}{'...' if len(summary_text) > 500 else ''}")
                            
                    except Exception as e:
                        print(f"⚠️ Error summarizing {filepath}: {e}")
            else:
                print("✨ No new individual conversations to summarize")
        except Exception as e:
            print(f"❌ Error processing unsummarized conversations: {e}")
    
    def _load_hierarchical_memory(self):
        """Load the latest LTMs, STMs, and individual summaries into a comprehensive system prompt."""
        try:
            base_system_prompt = self.llm.system_prompt
            memory_introduction = "The following is a conversation between a user and an AI assistant.\n\n[EXTENDED CONTEXTUAL MEMORY FOLLOWS]:\n"
            final_memory_content_parts = []

            if not self.log_conversations or not hasattr(self, 'hierarchical_memory_manager') or not self.hierarchical_memory_manager:
                self.conversation_history = [{
                    "role": "system",
                    "content": base_system_prompt # Only base prompt if no logging/memory manager
                }]
                return

            print("Updating memory hierarchy before loading into prompt...")
            self._process_unsummarized_conversations()
            self.hierarchical_memory_manager.update_memory_hierarchy()
            print("Memory hierarchy updated.")

            processed_stm_paths = set()
            processed_conv_summary_paths = set()

            # Define how many of each to load into the prompt
            MAX_LTMS_TO_LOAD_IN_PROMPT = 5
            MAX_STMS_TO_LOAD_IN_PROMPT = 5
            MAX_CONV_SUMMARIES_TO_LOAD_IN_PROMPT = 5

            # 1. Load Latest LTMs
            latest_ltm_list = self.conversation_logger.get_latest_ltm_summaries(max_ltm_summaries=MAX_LTMS_TO_LOAD_IN_PROMPT)
            ltms_loaded_content = []
            if latest_ltm_list:
                for ltm in reversed(latest_ltm_list): # Newest LTM last (so it appears first after context header)
                    ltm_content = ltm.get('content', '').strip()
                    ltm_file_path = str(Path(ltm.get('file_path', ltm.get('summary_timestamp', 'unknown_ltm'))))
                    if ltm_content:
                        ltms_loaded_content.append(f"[Long-Term Memory Synthesis - ID: {Path(ltm_file_path).stem}]:\n{ltm_content}")
                        print(f"🧠 Adding LTM to prompt: {Path(ltm_file_path).stem}")
                        for stm_path in ltm.get('constituent_stms', []):
                            processed_stm_paths.add(str(Path(stm_path)))
            if ltms_loaded_content:
                final_memory_content_parts.extend(ltms_loaded_content)

            # 2. Load Latest STMs, excluding those already in LTMs
            # Fetch a bit more initially to allow filtering
            stms_to_potentially_load = self.conversation_logger.get_latest_stm_summaries(max_stm_summaries=MAX_STMS_TO_LOAD_IN_PROMPT * 2)
            stms_added_to_prompt_content = []
            for stm in reversed(stms_to_potentially_load): # Newest STM last
                stm_file_path = str(Path(stm.get('file_path', stm.get('summary_timestamp', 'unknown_stm'))))
                if stm_file_path not in processed_stm_paths:
                    stm_content = stm.get('content', '').strip()
                    if stm_content:
                        stms_added_to_prompt_content.append(f"[Short-Term Memory Snapshot - ID: {Path(stm_file_path).stem}]:\n{stm_content}")
                        print(f"🧠 Adding STM to prompt: {Path(stm_file_path).stem}")
                        for conv_sum_path in stm.get('constituent_summaries', []):
                            processed_conv_summary_paths.add(str(Path(conv_sum_path)))
                        if len(stms_added_to_prompt_content) >= MAX_STMS_TO_LOAD_IN_PROMPT:
                            break
            if stms_added_to_prompt_content:
                final_memory_content_parts.extend(stms_added_to_prompt_content)

            # 3. Load Latest Individual Conversation Summaries, excluding those already in STMs
            # Fetch a bit more initially
            conv_summaries_to_potentially_load = self.conversation_logger.get_latest_summaries(max_summaries=MAX_CONV_SUMMARIES_TO_LOAD_IN_PROMPT * 2)
            conv_summaries_added_to_prompt_content = []
            for conv_sum in reversed(conv_summaries_to_potentially_load): # Newest summary last
                original_file_path = str(Path(conv_sum.get('file_path', conv_sum.get('original_file', 'unknown_conv_summary'))))
                if original_file_path not in processed_conv_summary_paths:
                    summary_content = ""
                    if conv_sum.get('messages') and isinstance(conv_sum['messages'], list):
                        for msg in conv_sum['messages']:
                            if msg.get('role') == 'assistant' and msg.get('content'):
                                summary_content = msg['content'].strip()
                                break
                    elif conv_sum.get('content'): # Fallback
                        summary_content = conv_sum.get('content').strip()

                    if summary_content:
                        conv_summaries_added_to_prompt_content.append(f"[Previous Conversation Summary - ID: {Path(original_file_path).stem}]:\n{summary_content}")
                        print(f"🧠 Adding Conv Summary to prompt: {Path(original_file_path).stem}")
                        if len(conv_summaries_added_to_prompt_content) >= MAX_CONV_SUMMARIES_TO_LOAD_IN_PROMPT:
                            break
            if conv_summaries_added_to_prompt_content:
                final_memory_content_parts.extend(conv_summaries_added_to_prompt_content)

            # Construct the final system prompt
            comprehensive_system_prompt = base_system_prompt
            if final_memory_content_parts:
                comprehensive_system_prompt += "\n\n" + memory_introduction + "\n".join(final_memory_content_parts)
                print(f"✅ Incorporated {len(final_memory_content_parts)} memory entries into the system prompt.")
            else:
                print("No hierarchical memories available to add to system prompt.")

            self.conversation_history = [{
                "role": "system",
                "content": comprehensive_system_prompt
            }]
            # print(f"[DEBUG] Final System Prompt:\n{comprehensive_system_prompt}") # For debugging

        except Exception as e:
            print(f"❌ Error loading hierarchical memories into system prompt: {e}")
            # Fallback to just base system prompt if error
            self.conversation_history = [{
                "role": "system",
                "content": self.llm.system_prompt # Use the original base prompt
            }]
    