"""
True streaming TTS that synthesizes and plays audio in parallel
"""
import threading
import queue
import time
from typing import List, Optional, Tuple
import numpy as np
from src.utils.logger import logger

class StreamingTTSManager:
    """Manages parallel synthesis and playback for low-latency speech"""
    
    def __init__(self, tts, audio_manager, max_parallel_synthesis=2):
        self.tts = tts
        self.audio_manager = audio_manager
        self.max_parallel_synthesis = max_parallel_synthesis
        
        # Queues for pipeline
        self.text_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=5)
        
        # Control flags
        self.is_active = False
        self.should_stop = threading.Event()
        self.synthesis_interrupted = threading.Event()
        
        # Workers
        self.synthesis_workers = []
        self.playback_worker = None
        
        # Stats
        self.first_audio_ready_time = None
        self.start_time = None
        
    def start_streaming(self, text: str) -> bool:
        """Start streaming synthesis and playback"""
        self.is_active = True
        self.should_stop.clear()
        self.synthesis_interrupted.clear()
        self.start_time = time.time()
        
        # Split text into sentences
        sentences = self.tts._split_into_sentences(text)
        if not sentences:
            return False
            
        logger.synthesis(f"Starting streaming TTS for {len(sentences)} sentences")
        
        # Start playback worker first
        self.playback_worker = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="StreamingPlayback"
        )
        self.playback_worker.start()
        
        # Start synthesis workers
        for i in range(min(self.max_parallel_synthesis, len(sentences))):
            worker = threading.Thread(
                target=self._synthesis_worker,
                daemon=True,
                name=f"StreamingSynthesis-{i}"
            )
            self.synthesis_workers.append(worker)
            worker.start()
        
        # Feed sentences to synthesis queue
        threading.Thread(
            target=self._feed_sentences,
            args=(sentences,),
            daemon=True,
            name="SentenceFeeder"
        ).start()
        
        return True
        
    def _feed_sentences(self, sentences: List[str]):
        """Feed sentences to synthesis queue"""
        try:
            for i, sentence in enumerate(sentences):
                if self.should_stop.is_set():
                    break
                    
                self.text_queue.put((i, sentence))
                
            # Signal end of sentences
            for _ in range(self.max_parallel_synthesis):
                self.text_queue.put(None)
                
        except Exception as e:
            logger.error(f"Error feeding sentences: {e}")
            
    def _synthesis_worker(self):
        """Worker thread for synthesizing sentences"""
        while self.is_active and not self.should_stop.is_set():
            try:
                # Get sentence
                item = self.text_queue.get(timeout=0.1)
                if item is None:  # End signal
                    break
                    
                sentence_idx, sentence = item
                
                if self.synthesis_interrupted.is_set():
                    continue
                    
                # Synthesize
                start_time = time.time()
                logger.synthesis(f"[{sentence_idx}] Synthesizing: '{sentence[:30]}...'")
                
                try:
                    audio, sample_rate = self.tts.synthesize(sentence)
                    synthesis_time = time.time() - start_time
                    
                    # Queue for playback
                    if not self.synthesis_interrupted.is_set():
                        self.audio_queue.put({
                            'index': sentence_idx,
                            'audio': audio,
                            'sample_rate': sample_rate,
                            'text': sentence,
                            'synthesis_time': synthesis_time
                        })
                        
                        logger.synthesis(f"[{sentence_idx}] Synthesis complete: {synthesis_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"[{sentence_idx}] Synthesis error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Synthesis worker error: {e}")
                
    def _playback_worker(self):
        """Worker thread for playing audio in order"""
        expected_index = 0
        audio_buffer = {}  # Buffer for out-of-order audio
        first_chunk = True
        
        while self.is_active and not self.should_stop.is_set():
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:  # End signal
                    break
                    
                chunk_idx = chunk['index']
                
                # Buffer the chunk
                audio_buffer[chunk_idx] = chunk
                
                # Play chunks in order
                while expected_index in audio_buffer:
                    if self.should_stop.is_set():
                        break
                        
                    audio_chunk = audio_buffer.pop(expected_index)
                    
                    if first_chunk:
                        # Start output stream on first chunk
                        if not self.audio_manager.output_stream:
                            self.audio_manager.start_output_stream()
                        
                        self.first_audio_ready_time = time.time() - self.start_time
                        logger.synthesis(f"🚀 First audio ready in {self.first_audio_ready_time:.2f}s")
                        first_chunk = False
                    
                    # Play audio
                    audio = audio_chunk['audio']
                    sample_rate = audio_chunk['sample_rate']
                    
                    # Resample if needed
                    if sample_rate != self.audio_manager.output_sample_rate:
                        import librosa
                        audio = librosa.resample(
                            audio,
                            orig_sr=sample_rate,
                            target_sr=self.audio_manager.output_sample_rate
                        )
                    
                    # Stream audio (non-blocking)
                    self.audio_manager.stream_audio(audio)
                    
                    logger.synthesis(f"[{expected_index}] Playing: '{audio_chunk['text'][:30]}...'")
                    expected_index += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback worker error: {e}")
                
    def stop(self):
        """Stop streaming"""
        self.should_stop.set()
        self.is_active = False
        
        # Signal workers to stop
        try:
            self.audio_queue.put(None)
        except:
            pass
            
    def interrupt(self):
        """Interrupt current synthesis"""
        self.synthesis_interrupted.set()
        self.stop()
        
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Wait for streaming to complete"""
        start_time = time.time()
        
        while self.is_active and time.time() - start_time < timeout:
            # Check if all workers are done
            all_done = True
            
            for worker in self.synthesis_workers:
                if worker.is_alive():
                    all_done = False
                    break
                    
            if self.playback_worker and self.playback_worker.is_alive():
                all_done = False
                
            if all_done:
                return True
                
            time.sleep(0.1)
            
        return False


def patch_voice_assistant_with_streaming(voice_assistant):
    """Patch VoiceAssistant to use streaming TTS"""
    
    def _say_streaming_improved(self, text: str):
        """Improved streaming synthesis with parallel processing"""
        try:
            # Clear flags
            self.cancel_processing.clear()
            self.synthesis_interrupted.clear()
            
            self.is_speaking = True
            self.synthesis_start_time = time.time()
            
            # Create streaming manager
            streaming_manager = StreamingTTSManager(
                self.tts,
                self.audio_manager,
                max_parallel_synthesis=2  # Synthesize 2 sentences in parallel
            )
            
            # Start streaming
            if streaming_manager.start_streaming(text):
                # Wait for completion or interruption
                while streaming_manager.is_active and not self.cancel_processing.is_set():
                    if self.synthesis_interrupted.is_set():
                        logger.synthesis("Streaming interrupted by user")
                        streaming_manager.interrupt()
                        break
                    time.sleep(0.1)
                
                # Wait for completion
                streaming_manager.wait_for_completion(timeout=30.0)
            
            streaming_manager.stop()
            
        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
        finally:
            self.is_speaking = False
            
    # Patch the method
    voice_assistant._say_streaming = _say_streaming_improved.__get__(voice_assistant, type(voice_assistant))
    
    return voice_assistant