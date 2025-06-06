#!/usr/bin/env python3
"""
Compare sequential vs streaming TTS performance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from configs import config
from src.pipeline.voice_assistant import VoiceAssistant
from src.pipeline.streaming_tts import patch_voice_assistant_with_streaming
from src.utils.logger import logger
import time

def test_sequential_vs_streaming():
    """Compare sequential and streaming TTS"""
    
    test_text = """
    Hello there! This is a test of our streaming text-to-speech system. 
    We want to see how quickly we can get the first audio playing back to the user.
    The goal is to minimize the time between when the LLM finishes generating text and when the user hears the first words.
    This should demonstrate a significant improvement in perceived responsiveness.
    """
    
    print("🧪 Testing Sequential vs Streaming TTS Performance")
    print("=" * 60)
    
    # Test 1: Sequential (current implementation)
    print("\n📊 Test 1: Sequential TTS (current)")
    assistant_sequential = VoiceAssistant(
        whisper_model_size="base",
        whisper_device="cpu",
        use_gemini=True,
        gemini_api_key=config.GEMINI_API_KEY,
        gemini_model=config.GEMINI_MODEL,
        system_prompt=config.SYSTEM_PROMPT,
        voice_reference_path=str(config.VOICE_REFERENCE_PATH),
        enable_sound_effects=False,
        max_response_tokens=100,
        llm_temperature=0.7
    )
    
    start_time = time.time()
    logger.info("Starting sequential synthesis...")
    
    # Synthesize sequentially (current method)
    assistant_sequential.say(test_text)
    
    # Wait for completion
    while assistant_sequential.is_speaking:
        time.sleep(0.1)
        
    sequential_time = time.time() - start_time
    logger.success(f"Sequential synthesis completed in {sequential_time:.2f}s")
    
    assistant_sequential.stop()
    time.sleep(1)  # Brief pause
    
    # Test 2: Streaming (new implementation)
    print("\n📊 Test 2: Streaming TTS (new)")
    assistant_streaming = VoiceAssistant(
        whisper_model_size="base", 
        whisper_device="cpu",
        use_gemini=True,
        gemini_api_key=config.GEMINI_API_KEY,
        gemini_model=config.GEMINI_MODEL,
        system_prompt=config.SYSTEM_PROMPT,
        voice_reference_path=str(config.VOICE_REFERENCE_PATH),
        enable_sound_effects=False,
        max_response_tokens=100,
        llm_temperature=0.7
    )
    
    # Patch with streaming
    assistant_streaming = patch_voice_assistant_with_streaming(assistant_streaming)
    
    start_time = time.time()
    logger.info("Starting streaming synthesis...")
    
    # Synthesize with streaming
    assistant_streaming.say(test_text)
    
    # Wait for completion
    while assistant_streaming.is_speaking:
        time.sleep(0.1)
        
    streaming_time = time.time() - start_time
    logger.success(f"Streaming synthesis completed in {streaming_time:.2f}s")
    
    assistant_streaming.stop()
    
    # Results
    print("\n📈 Results:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Streaming:  {streaming_time:.2f}s")
    
    if streaming_time < sequential_time:
        improvement = ((sequential_time - streaming_time) / sequential_time) * 100
        print(f"  🚀 Streaming is {improvement:.1f}% faster!")
    else:
        print(f"  ⚠️  Sequential was faster (streaming needs optimization)")
        
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_sequential_vs_streaming()