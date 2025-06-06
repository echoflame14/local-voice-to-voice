#!/usr/bin/env python3
"""
Test script for streaming synthesis improvements
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from configs import config
from src.pipeline.voice_assistant import VoiceAssistant
from src.pipeline.voice_assistant_streaming import StreamingSynthesisManager, create_streaming_voice_assistant_methods
import time

def test_aggressive_chunking():
    """Test aggressive chunking for low latency"""
    print("🧪 Testing aggressive chunking for low-latency TTS...")
    
    # Initialize components
    assistant = VoiceAssistant(
        whisper_model_size="base",
        whisper_device="cpu",
        use_gemini=True,
        gemini_api_key=config.GEMINI_API_KEY,
        gemini_model=config.GEMINI_MODEL,
        system_prompt=config.SYSTEM_PROMPT,
        voice_reference_path=str(config.VOICE_REFERENCE_PATH),
        enable_sound_effects=False,  # Disable for testing
        max_response_tokens=config.MAX_RESPONSE_TOKENS,
        llm_temperature=config.LLM_TEMPERATURE
    )
    
    # Patch in streaming methods
    streaming_methods = create_streaming_voice_assistant_methods()
    for method_name, method in streaming_methods.items():
        setattr(assistant, method_name, method.__get__(assistant, VoiceAssistant))
    
    # Test streaming synthesis manager
    print("\n📊 Testing StreamingSynthesisManager with different chunk sizes...")
    
    test_text = "The quick brown fox jumps over the lazy dog. This is a test of progressive synthesis with aggressive chunking for lower latency."
    
    for chunk_size in [1, 3, 5]:
        print(f"\n🔹 Testing with chunk_size_words={chunk_size}")
        
        manager = StreamingSynthesisManager(
            assistant.tts,
            assistant.audio_manager,
            chunk_size_words=chunk_size
        )
        
        # Simulate streaming text
        def text_generator():
            words = test_text.split()
            for i in range(0, len(words), 2):
                yield ' '.join(words[i:i+2]) + ' '
                time.sleep(0.1)  # Simulate LLM delay
        
        start_time = time.time()
        manager.start_streaming(text_generator())
        
        # Wait for first audio
        first_audio_time = None
        while manager.synthesis_queue.qsize() == 0 and time.time() - start_time < 5:
            time.sleep(0.01)
        
        if manager.synthesis_queue.qsize() > 0:
            first_audio_time = time.time() - start_time
            print(f"   ⏱️  First audio ready in: {first_audio_time:.3f}s")
        
        # Let it run briefly then stop
        time.sleep(2)
        manager.stop()
        
    print("\n✅ Streaming synthesis test complete!")
    
    # Test with actual LLM streaming
    if hasattr(assistant.llm, 'stream_chat'):
        print("\n🔹 Testing with actual LLM streaming...")
        
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": "Count from 1 to 10 slowly."}
        ]
        
        manager = StreamingSynthesisManager(
            assistant.tts,
            assistant.audio_manager,
            chunk_size_words=2
        )
        
        start_time = time.time()
        
        def llm_response_generator():
            for chunk in assistant.llm.stream_chat(messages, max_tokens=100, temperature=0.5):
                yield chunk
                
        manager.start_streaming(llm_response_generator())
        
        # Wait for first audio
        while manager.synthesis_queue.qsize() == 0 and time.time() - start_time < 10:
            time.sleep(0.01)
            
        if manager.synthesis_queue.qsize() > 0:
            first_audio_time = time.time() - start_time
            print(f"   ⏱️  First audio from LLM in: {first_audio_time:.3f}s")
            
        # Let it run
        time.sleep(5)
        manager.stop()
        
    assistant.stop()
    print("\n🎉 All tests complete!")

if __name__ == "__main__":
    test_aggressive_chunking()