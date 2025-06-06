#!/usr/bin/env python3
"""
Test adaptive streaming synthesis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from configs import config
from src.pipeline.voice_assistant import VoiceAssistant
from src.pipeline.adaptive_streaming import AdaptiveStreamingSynthesizer
import time
import threading

def integrate_adaptive_streaming(voice_assistant):
    """Integrate adaptive streaming into voice assistant"""
    
    def _process_speech_adaptive(self, audio):
        """Enhanced speech processing with adaptive streaming"""
        if not self.is_running:
            return
            
        try:
            # Transcribe speech
            print("[ADAPTIVE] Transcribing audio...")
            transcription = self.stt.transcribe(audio)
            
            if not transcription or not transcription.strip():
                print("[ADAPTIVE] No transcription produced")
                return
                
            user_text = transcription.strip()
            print(f"[ADAPTIVE] Transcription: {user_text}")
            
            # Trigger callbacks
            if self.on_transcription:
                self.on_transcription(user_text)
                
            # Log user message
            if self.log_conversations:
                self.conversation_logger.log_message("user", user_text)
                
            # Prepare messages for LLM
            messages = self._prepare_messages_for_llm(user_text)
            
            # Create adaptive synthesizer
            synthesizer = AdaptiveStreamingSynthesizer(self.tts, self.audio_manager)
            
            try:
                # Start synthesis
                self.is_speaking = True
                self.synthesis_start_time = time.time()
                
                # Check if LLM supports streaming
                if hasattr(self.llm, 'stream_chat'):
                    print("[ADAPTIVE] Using streaming LLM...")
                    
                    # Collect full response for logging
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
                            
                    # Start adaptive streaming
                    synthesizer.start_streaming(response_generator())
                    
                    # Wait for first chunk
                    if synthesizer.wait_for_first_chunk(timeout=10.0):
                        print(f"[ADAPTIVE] First audio ready!")
                    
                    # Monitor progress
                    while synthesizer.is_active and not self.cancel_processing.is_set():
                        time.sleep(0.1)
                        
                    response_text = ''.join(full_response)
                    
                else:
                    print("[ADAPTIVE] Using non-streaming LLM...")
                    response_text = self.llm.chat(
                        messages=messages,
                        max_tokens=self.max_response_tokens,
                        temperature=self.llm_temperature
                    )
                    
                    # Stream the complete response
                    def response_generator():
                        # Feed text in small chunks to simulate streaming
                        words = response_text.split()
                        for i in range(0, len(words), 3):
                            yield ' '.join(words[i:i+3]) + ' '
                            
                    synthesizer.start_streaming(response_generator())
                    
                    # Wait for completion
                    while synthesizer.is_active and not self.cancel_processing.is_set():
                        time.sleep(0.1)
                        
                # Handle callbacks and logging
                if response_text and self.on_response:
                    self.on_response(response_text)
                    
                if self.log_conversations and response_text:
                    self.conversation_logger.log_message("assistant", response_text)
                    
            except Exception as e:
                print(f"[ADAPTIVE ERROR] {e}")
                synthesizer.stop()
                
            finally:
                self.is_speaking = False
                synthesizer.stop()
                
        except Exception as e:
            print(f"[ADAPTIVE ERROR] Speech processing error: {e}")
            self.is_speaking = False
    
    # Monkey patch the method
    voice_assistant._process_speech_adaptive = _process_speech_adaptive.__get__(voice_assistant, VoiceAssistant)
    return voice_assistant

def test_adaptive_streaming():
    """Test the adaptive streaming system"""
    print("🧪 Testing Adaptive Streaming Synthesis...")
    
    # Initialize voice assistant
    assistant = VoiceAssistant(
        whisper_model_size="base",
        whisper_device="cpu",
        use_gemini=True,
        gemini_api_key=config.GEMINI_API_KEY,
        gemini_model=config.GEMINI_MODEL,
        system_prompt=config.SYSTEM_PROMPT,
        voice_reference_path=str(config.VOICE_REFERENCE_PATH),
        enable_sound_effects=False,
        max_response_tokens=200,
        llm_temperature=0.7
    )
    
    # Integrate adaptive streaming
    assistant = integrate_adaptive_streaming(assistant)
    
    print("\n📊 Testing synthesis speed estimation...")
    
    # Test 1: Measure actual synthesis speed
    test_texts = [
        "Hello world.",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer sentence to test how the synthesis speed changes with different text lengths and complexity.",
    ]
    
    for text in test_texts:
        start = time.time()
        audio, sr = assistant.tts.synthesize(text)
        synthesis_time = time.time() - start
        audio_duration = len(audio) / sr
        
        print(f"\nText: '{text[:50]}...'")
        print(f"  Synthesis time: {synthesis_time:.2f}s")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Speed ratio: {synthesis_time/audio_duration:.2f}x")
    
    # Test 2: Simulate streaming with adaptive chunking
    print("\n\n📊 Testing adaptive chunking with simulated streaming...")
    
    synthesizer = AdaptiveStreamingSynthesizer(assistant.tts, assistant.audio_manager)
    
    # Simulate streaming text
    def text_generator():
        long_text = """
        Artificial intelligence has made remarkable progress in recent years. 
        Machine learning models can now understand and generate human language with impressive accuracy.
        The future of AI looks bright, with applications ranging from healthcare to creative arts.
        As we continue to develop these technologies, it's important to consider their ethical implications.
        """
        
        words = long_text.split()
        for i in range(0, len(words), 5):
            yield ' '.join(words[i:i+5]) + ' '
            time.sleep(0.2)  # Simulate LLM streaming delay
    
    print("\nStarting adaptive streaming synthesis...")
    start_time = time.time()
    
    synthesizer.start_streaming(text_generator())
    
    # Wait for first chunk
    if synthesizer.wait_for_first_chunk(timeout=5.0):
        first_chunk_time = time.time() - start_time
        print(f"\n🚀 First audio chunk ready in: {first_chunk_time:.2f}s")
    
    # Let it run for a bit
    time.sleep(10)
    synthesizer.stop()
    
    # Test 3: Real streaming with Gemini
    if hasattr(assistant.llm, 'stream_chat'):
        print("\n\n📊 Testing with real Gemini streaming...")
        
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": "Tell me a short story about a robot learning to paint. Make it about 3 sentences."}
        ]
        
        synthesizer = AdaptiveStreamingSynthesizer(assistant.tts, assistant.audio_manager)
        
        def llm_generator():
            for chunk in assistant.llm.stream_chat(messages, max_tokens=150, temperature=0.8):
                yield chunk
                
        print("\nStarting real LLM + adaptive synthesis...")
        start_time = time.time()
        
        synthesizer.start_streaming(llm_generator())
        
        if synthesizer.wait_for_first_chunk(timeout=10.0):
            first_chunk_time = time.time() - start_time
            print(f"\n🚀 First audio from LLM ready in: {first_chunk_time:.2f}s")
        
        # Monitor for a while
        for i in range(15):
            time.sleep(1)
            if not synthesizer.is_active:
                break
                
        synthesizer.stop()
    
    assistant.stop()
    print("\n\n✅ Adaptive streaming test complete!")

if __name__ == "__main__":
    test_adaptive_streaming()