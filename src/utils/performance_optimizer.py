"""
Performance optimization utilities for voice assistant
"""
import threading
import time
import asyncio
from typing import Optional, Dict, Any, Callable
from functools import lru_cache
from src.utils.logger import logger
from src.utils.performance_monitor import perf_monitor, TimerContext

class PerformanceOptimizer:
    """Implements various performance optimizations"""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.background_tasks = []
        
    @lru_cache(maxsize=128)
    def cached_function_call(self, func_name: str, *args, **kwargs) -> Any:
        """Cache function results for expensive operations"""
        # This is a template - specific implementations would go here
        pass
        
    def optimize_memory_processing(self, voice_assistant):
        """Optimize memory hierarchy processing"""
        # Skip memory updates if we're in rapid-fire conversation mode
        if hasattr(voice_assistant, '_last_user_input_time'):
            time_since_last = time.time() - voice_assistant._last_user_input_time
            if time_since_last < 5.0:  # Less than 5 seconds since last input
                logger.debug("Skipping memory update - rapid conversation mode", "PERF")
                return False
        return True
        
    def preload_models(self, voice_assistant):
        """Preload and warm up models in background"""
        def background_warmup():
            try:
                # Warm up TTS with a short phrase
                if hasattr(voice_assistant, 'tts'):
                    with TimerContext("model_warmup", log_result=False):
                        _, _ = voice_assistant.tts.synthesize("Hello")
                        logger.debug("TTS model warmed up", "PERF")
            except Exception as e:
                logger.debug(f"Model warmup failed: {e}", "PERF")
                
        warmup_thread = threading.Thread(target=background_warmup, daemon=True)
        warmup_thread.start()
        self.background_tasks.append(warmup_thread)

# Optimized configuration tweaks
PERFORMANCE_CONFIG = {
    # Reduce memory hierarchy update frequency
    'MEMORY_UPDATE_INTERVAL': 60,  # Update every 60 seconds instead of 30
    
    # Optimize audio settings for lower latency
    'OPTIMIZED_CHUNK_SIZE': 480,  # MUST match VAD frame size (480 samples = 30ms at 16kHz)
    'OPTIMIZED_SAMPLE_RATE': 16000,  # Standard rate
    
    # LLM optimization
    'MAX_RESPONSE_TOKENS': 150,  # Shorter responses for faster generation
    'LLM_TEMPERATURE': 0.7,  # Slightly more focused responses
    
    # Memory optimization
    'MAX_CONVERSATION_HISTORY': 10,  # Keep only recent messages
    'DISABLE_TRANSCRIPTION_SOUND': True,  # Remove unnecessary audio feedback
    
    # TTS optimization
    'TTS_BATCH_SIZE': 1,  # Process one sentence at a time for streaming
}

def apply_performance_optimizations(voice_assistant):
    """Apply all performance optimizations to voice assistant"""
    optimizer = PerformanceOptimizer()
    
    # 1. Optimize memory processing
    original_prepare_messages = voice_assistant._prepare_messages_for_llm
    
    def optimized_prepare_messages(user_text: str):
        if not optimizer.optimize_memory_processing(voice_assistant):
            # Skip expensive memory operations in rapid conversation
            base_system_prompt = voice_assistant.llm.system_prompt
            messages = [
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": user_text}
            ]
            logger.debug("Using fast-path message preparation", "PERF")
            return messages
        return original_prepare_messages(user_text)
    
    voice_assistant._prepare_messages_for_llm = optimized_prepare_messages
    
    # 2. Cache expensive operations
    if hasattr(voice_assistant, 'conversation_summarizer'):
        original_summarize = voice_assistant.conversation_summarizer.summarize_conversation
        
        @lru_cache(maxsize=32)
        def cached_summarize(conversation_hash: str, *args):
            return original_summarize(*args)
            
        # Note: This would need proper hash implementation for the conversation data
    
    # 3. Optimize audio processing
    if hasattr(voice_assistant, 'audio_manager'):
        # Use smaller buffer sizes for lower latency
        voice_assistant.audio_manager.chunk_size = PERFORMANCE_CONFIG['OPTIMIZED_CHUNK_SIZE']
    
    # 4. Preload models
    optimizer.preload_models(voice_assistant)
    
    # 5. Track input timing for rapid conversation detection
    original_process_speech = voice_assistant._process_speech
    
    def timed_process_speech(audio):
        voice_assistant._last_user_input_time = time.time()
        return original_process_speech(audio)
    
    voice_assistant._process_speech = timed_process_speech
    
    logger.info("🚀 Performance optimizations applied")
    return optimizer

def create_high_performance_config():
    """Create optimized configuration for maximum performance"""
    return {
        # Audio optimizations
        'chunk_size': 480,  # MUST match VAD frame size (480 samples)
        'sample_rate': 16000,
        'vad_aggressiveness': 2,  # More aggressive VAD for faster detection
        
        # LLM optimizations  
        'max_response_tokens': 150,  # Reasonable responses for Gemini's speed
        'llm_temperature': 1.0,  # Creative responses
        
        # Memory optimizations - UTILIZE GEMINI'S LARGE CONTEXT WINDOW
        'log_conversations': True,  # Full context logging
        'auto_summarize_conversations': True,  # Enable for rich context
        'max_history_messages': 2000,  # Full conversation history with Gemini's 2M token context
        
        # Sound optimizations
        'enable_sound_effects': False,  # Disable for maximum speed
        'enable_transcription_sound': False,
        'enable_interruption_sound': True,  # Keep only essential sounds
        'enable_generation_sound': False,
        
        # TTS optimizations
        'voice_temperature': 0.6,  # Faster synthesis
        'voice_cfg_weight': 0.3,   # Lower quality but faster
    }

def benchmark_performance(voice_assistant, test_phrases: list = None):
    """Benchmark voice assistant performance"""
    if test_phrases is None:
        test_phrases = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a short joke."
        ]
    
    results = []
    
    for phrase in test_phrases:
        logger.info(f"🧪 Benchmarking: '{phrase}'")
        
        start_time = time.time()
        
        # Simulate audio transcription (we'd need actual audio for real test)
        with TimerContext(f"benchmark_{phrase.replace(' ', '_')[:20]}"):
            # This would be the actual processing
            pass
            
        end_time = time.time()
        duration = end_time - start_time
        
        results.append({
            'phrase': phrase,
            'duration': duration,
            'tokens_per_second': len(phrase.split()) / duration if duration > 0 else 0
        })
        
        logger.info(f"   ⏱️  {duration:.2f}s ({len(phrase.split())} words)")
    
    # Summary
    avg_duration = sum(r['duration'] for r in results) / len(results)
    avg_tokens_per_sec = sum(r['tokens_per_second'] for r in results) / len(results)
    
    logger.info("📊 Performance Benchmark Results:")
    logger.info(f"   Average duration: {avg_duration:.2f}s")
    logger.info(f"   Average speed: {avg_tokens_per_sec:.1f} words/sec")
    
    return results