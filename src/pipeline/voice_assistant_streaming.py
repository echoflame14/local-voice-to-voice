"""
Enhanced Voice Assistant with aggressive progressive chunking for low-latency responses
"""
import re
import threading
import queue
import time
from typing import List, Generator, Tuple, Optional
import numpy as np

class StreamingSynthesisManager:
    """Manages progressive synthesis and playback with aggressive chunking"""
    
    def __init__(self, tts, audio_manager, chunk_size_words: int = 3):
        self.tts = tts
        self.audio_manager = audio_manager
        self.chunk_size_words = chunk_size_words
        
        # Queues for pipelining
        self.text_queue = queue.Queue(maxsize=10)
        self.synthesis_queue = queue.Queue(maxsize=5)
        self.playback_queue = queue.Queue(maxsize=5)
        
        # Control flags
        self.is_active = False
        self.should_stop = threading.Event()
        
        # Threads
        self.synthesis_thread = None
        self.playback_thread = None
        
    def start_streaming(self, text_generator: Generator[str, None, None]):
        """Start streaming synthesis from text generator"""
        self.is_active = True
        self.should_stop.clear()
        
        # Start worker threads
        self.synthesis_thread = threading.Thread(
            target=self._synthesis_worker,
            daemon=True,
            name="ProgressiveSynthesis"
        )
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="ProgressivePlayback"
        )
        
        self.synthesis_thread.start()
        self.playback_thread.start()
        
        # Start feeding text chunks
        threading.Thread(
            target=self._text_chunker,
            args=(text_generator,),
            daemon=True,
            name="TextChunker"
        ).start()
        
    def stop(self):
        """Stop all streaming operations"""
        self.should_stop.set()
        self.is_active = False
        
        # Clear queues
        self._clear_queue(self.text_queue)
        self._clear_queue(self.synthesis_queue)
        self._clear_queue(self.playback_queue)
        
    def _clear_queue(self, q: queue.Queue):
        """Clear a queue"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
            
    def _text_chunker(self, text_generator: Generator[str, None, None]):
        """Aggressively chunk incoming text for low latency"""
        buffer = ""
        word_count = 0
        
        try:
            for text_chunk in text_generator:
                if self.should_stop.is_set():
                    break
                    
                buffer += text_chunk
                words = buffer.split()
                
                # Aggressively send chunks as soon as we have enough words
                while len(words) >= self.chunk_size_words:
                    # Extract chunk
                    chunk_words = words[:self.chunk_size_words]
                    chunk_text = " ".join(chunk_words)
                    
                    # Look for natural break points (punctuation)
                    natural_break = False
                    for i, word in enumerate(chunk_words):
                        if any(p in word for p in ['.', '!', '?', ',', ';', ':']):
                            chunk_text = " ".join(chunk_words[:i+1])
                            words = words[i+1:]
                            natural_break = True
                            break
                    
                    if not natural_break:
                        words = words[self.chunk_size_words:]
                    
                    # Update buffer
                    buffer = " ".join(words)
                    
                    # Send chunk for synthesis
                    if chunk_text.strip():
                        self.text_queue.put(chunk_text.strip())
                        
            # Send any remaining text
            if buffer.strip() and not self.should_stop.is_set():
                self.text_queue.put(buffer.strip())
                
        finally:
            # Signal end of text
            self.text_queue.put(None)
            
    def _synthesis_worker(self):
        """Worker thread for synthesizing text chunks"""
        while self.is_active and not self.should_stop.is_set():
            try:
                # Get text chunk (with timeout)
                chunk = self.text_queue.get(timeout=0.1)
                
                if chunk is None:  # End signal
                    self.synthesis_queue.put(None)
                    break
                    
                # Synthesize chunk
                start_time = time.time()
                try:
                    audio, sample_rate = self.tts.synthesize(chunk)
                    synthesis_time = time.time() - start_time
                    
                    # Put synthesized audio in playback queue
                    self.synthesis_queue.put({
                        'audio': audio,
                        'sample_rate': sample_rate,
                        'text': chunk,
                        'synthesis_time': synthesis_time
                    })
                    
                except Exception as e:
                    print(f"❌ Synthesis error for chunk '{chunk}': {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Synthesis worker error: {e}")
                
    def _playback_worker(self):
        """Worker thread for playing synthesized audio"""
        first_chunk = True
        
        while self.is_active and not self.should_stop.is_set():
            try:
                # Get synthesized audio (with timeout)
                chunk_data = self.synthesis_queue.get(timeout=0.1)
                
                if chunk_data is None:  # End signal
                    break
                    
                if first_chunk:
                    # Start output stream on first chunk
                    self.audio_manager.start_output_stream()
                    first_chunk = False
                    print(f"🚀 First audio chunk ready in {chunk_data['synthesis_time']:.2f}s: '{chunk_data['text'][:30]}...'")
                    
                # Play audio
                audio = chunk_data['audio']
                sample_rate = chunk_data['sample_rate']
                
                # Resample if needed
                if sample_rate != self.audio_manager.output_sample_rate:
                    import librosa
                    audio = librosa.resample(
                        audio,
                        orig_sr=sample_rate,
                        target_sr=self.audio_manager.output_sample_rate
                    )
                    
                # Stream audio
                self.audio_manager.stream_audio(audio)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Playback worker error: {e}")


def create_streaming_voice_assistant_methods():
    """
    Returns enhanced methods for VoiceAssistant to support streaming
    These can be monkey-patched onto the existing VoiceAssistant class
    """
    
    def _process_speech_streaming(self, audio: np.ndarray):
        """Enhanced speech processing with streaming LLM and progressive TTS"""
        if not self.is_running:
            return
            
        try:
            # Transcribe speech
            print("[STREAMING] Transcribing audio...")
            transcription = self.stt.transcribe(audio)
            
            if not transcription or not transcription.strip():
                print("[STREAMING] No transcription produced")
                return
                
            user_text = transcription.strip()
            print(f"[STREAMING] Transcription: {user_text}")
            
            # Trigger callbacks
            if self.on_transcription:
                self.on_transcription(user_text)
                
            # Log user message
            if self.log_conversations:
                self.conversation_logger.log_message("user", user_text)
                
            # Prepare messages for LLM
            messages = self._prepare_messages_for_llm(user_text)
            
            # Start streaming manager
            streaming_manager = StreamingSynthesisManager(
                self.tts,
                self.audio_manager,
                chunk_size_words=3  # Aggressive chunking
            )
            
            try:
                # Check if LLM supports streaming
                if hasattr(self.llm, 'stream_chat'):
                    print("[STREAMING] Starting streaming LLM generation...")
                    
                    # Create text accumulator for logging
                    full_response = []
                    
                    def response_generator():
                        for chunk in self.llm.stream_chat(
                            messages=messages,
                            max_tokens=self.max_response_tokens,
                            temperature=self.llm_temperature
                        ):
                            if self.cancel_processing.is_set():
                                break
                            full_response.append(chunk)
                            yield chunk
                            
                    # Start streaming synthesis
                    self.is_speaking = True
                    self.synthesis_start_time = time.time()
                    
                    streaming_manager.start_streaming(response_generator())
                    
                    # Wait for completion or interruption
                    while streaming_manager.is_active and not self.cancel_processing.is_set():
                        time.sleep(0.1)
                        
                    # Get full response for logging
                    response_text = ''.join(full_response)
                    
                else:
                    # Fallback to non-streaming
                    print("[STREAMING] LLM doesn't support streaming, using standard generation...")
                    response_text = self.llm.chat(
                        messages=messages,
                        max_tokens=self.max_response_tokens,
                        temperature=self.llm_temperature
                    )
                    
                    # Use progressive synthesis even for non-streaming LLM
                    def response_generator():
                        # Simulate streaming by yielding words
                        words = response_text.split()
                        for i in range(0, len(words), 2):  # Yield 2 words at a time
                            yield ' '.join(words[i:i+2]) + ' '
                            
                    self.is_speaking = True
                    self.synthesis_start_time = time.time()
                    
                    streaming_manager.start_streaming(response_generator())
                    
                    # Wait for completion
                    while streaming_manager.is_active and not self.cancel_processing.is_set():
                        time.sleep(0.1)
                        
                # Log response
                if response_text and self.on_response:
                    self.on_response(response_text)
                    
                if self.log_conversations and response_text:
                    self.conversation_logger.log_message("assistant", response_text)
                    
            except Exception as e:
                print(f"[STREAMING ERROR] {e}")
                streaming_manager.stop()
                # Don't synthesize error messages
                return
                
            finally:
                self.is_speaking = False
                streaming_manager.stop()
                
        except Exception as e:
            print(f"[STREAMING ERROR] Speech processing error: {e}")
            self.is_speaking = False
    
    return {
        '_process_speech_streaming': _process_speech_streaming
    }