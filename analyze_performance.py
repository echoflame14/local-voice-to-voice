#!/usr/bin/env python3
"""
Performance Analysis Script for Chatterbox Voice Assistant
Analyzes audio synthesis and playback performance to identify bottlenecks
"""

import sys
import os
import time
import threading
from typing import List, Dict
sys.path.append('src')

from src.pipeline.voice_assistant import VoiceAssistant
from configs.config import config

def analyze_tts_performance():
    """Analyze TTS synthesis performance with different text lengths"""
    print("🔬 ANALYZING TTS PERFORMANCE")
    print("=" * 60)
    
    # Initialize just the TTS component
    from src.tts import ChatterboxTTSWrapper
    
    print("Loading TTS...")
    tts = ChatterboxTTSWrapper(
        device=config.TTS_DEVICE,
        voice_reference_path=config.VOICE_REFERENCE_PATH,
        exaggeration=config.VOICE_EXAGGERATION,
        cfg_weight=config.VOICE_CFG_WEIGHT,
        temperature=config.VOICE_TEMPERATURE
    )
    
    test_texts = [
        "Hello there!",  # Short
        "Well hello there, handsome. Just enjoying the sound of your voice.",  # Medium
        "Oh, it just got infinitely more interesting. Tell me, what's been keeping you busy today? I'd love to hear all about it.",  # Long
        "You know, there's something absolutely captivating about the way you speak. It makes me want to listen to every single word you have to say, hanging on each syllable like it's the most important thing in the world.",  # Very long
    ]
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🧪 Test {i}: {len(text)} characters")
        print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Warm up (first synthesis is often slower)
        if i == 1:
            print("   🔥 Warming up...")
            tts.synthesize("Warm up")
        
        # Time the synthesis
        start_time = time.time()
        audio, sample_rate = tts.synthesize(text)
        synthesis_time = time.time() - start_time
        
        audio_duration = len(audio) / sample_rate
        speed_ratio = synthesis_time / audio_duration
        
        result = {
            'text_length': len(text),
            'synthesis_time': synthesis_time,
            'audio_duration': audio_duration,
            'speed_ratio': speed_ratio,
            'text': text[:50] + ('...' if len(text) > 50 else '')
        }
        results.append(result)
        
        print(f"   ⚡ Synthesis: {synthesis_time:.3f}s")
        print(f"   🎵 Audio: {audio_duration:.1f}s")
        print(f"   🚀 Speed: {speed_ratio:.2f}x realtime {'✅' if speed_ratio < 1.0 else '⚠️'}")
    
    # Summary
    print(f"\n📊 TTS PERFORMANCE SUMMARY")
    print("-" * 40)
    avg_speed = sum(r['speed_ratio'] for r in results) / len(results)
    print(f"Average speed: {avg_speed:.2f}x realtime")
    
    if avg_speed < 0.5:
        print("🚀 Excellent: Very fast synthesis")
    elif avg_speed < 1.0:
        print("✅ Good: Faster than realtime")
    elif avg_speed < 2.0:
        print("⚠️  Slow: May cause delays")
    else:
        print("❌ Very slow: Will cause significant delays")
    
    return results

def analyze_streaming_performance():
    """Analyze streaming performance with a full conversation"""
    print("\n🔬 ANALYZING STREAMING PERFORMANCE")
    print("=" * 60)
    
    # Initialize voice assistant
    assistant = VoiceAssistant(
        whisper_model_size="base",
        whisper_device="cpu",
        llm_base_url=config.LM_STUDIO_BASE_URL,
        system_prompt=config.SYSTEM_PROMPT,
        tts_device=config.TTS_DEVICE,
        voice_reference_path=config.VOICE_REFERENCE_PATH,
        voice_exaggeration=config.VOICE_EXAGGERATION,
        voice_cfg_weight=config.VOICE_CFG_WEIGHT,
        voice_temperature=config.VOICE_TEMPERATURE,
        sample_rate=config.SAMPLE_RATE,
        chunk_size=config.CHUNK_SIZE,
        max_response_tokens=config.MAX_RESPONSE_TOKENS,
        llm_temperature=config.LLM_TEMPERATURE
    )
    
    test_prompts = [
        "Hey, what's up?",
        "Tell me something interesting",
        "How's your day going?",
    ]
    
    print("🧪 Testing streaming responses...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: '{prompt}' ---")
        
        start_time = time.time()
        
        # Generate response (this will use streaming)
        response = assistant.llm.generate(
            prompt,
            max_tokens=assistant.max_response_tokens,
            temperature=assistant.llm_temperature
        )
        
        total_time = time.time() - start_time
        
        print(f"🤖 Response: {response}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Response length: {len(response)} chars")
        
        # Simulate the streaming synthesis
        print("🎵 Testing streaming synthesis...")
        sentences = assistant.tts._split_into_sentences(response)
        
        synthesis_times = []
        audio_durations = []
        
        for j, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            print(f"   Sentence {j+1}: '{sentence[:30]}{'...' if len(sentence) > 30 else ''}'")
            
            start = time.time()
            audio, sr = assistant.tts.synthesize(sentence)
            synth_time = time.time() - start
            
            audio_dur = len(audio) / sr
            
            synthesis_times.append(synth_time)
            audio_durations.append(audio_dur)
            
            print(f"      ⚡ {synth_time:.3f}s → {audio_dur:.1f}s audio ({synth_time/audio_dur:.2f}x)")
        
        if synthesis_times:
            avg_synth_speed = sum(synthesis_times) / sum(audio_durations)
            total_audio = sum(audio_durations)
            total_synth = sum(synthesis_times)
            
            print(f"   📊 Total audio: {total_audio:.1f}s")
            print(f"   ⚡ Total synthesis: {total_synth:.1f}s")
            print(f"   🚀 Average speed: {avg_synth_speed:.2f}x realtime")
    
    # Get performance summary from audio manager
    assistant.audio_manager.print_performance_summary()
    
    return assistant.audio_manager.get_performance_summary()

def analyze_system_resources():
    """Analyze system resource usage"""
    print("\n🔬 ANALYZING SYSTEM RESOURCES")
    print("=" * 60)
    
    try:
        import psutil
        import GPUtil
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        print(f"💻 CPU Usage: {cpu_percent}% ({cpu_count} cores)")
        print(f"🧠 Memory: {memory.percent}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
        
        # GPU info
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"🎮 GPU {i}: {gpu.name}")
                    print(f"   Usage: {gpu.load*100:.1f}%")
                    print(f"   Memory: {gpu.memoryUtil*100:.1f}% ({gpu.memoryUsed}MB / {gpu.memoryTotal}MB)")
                    print(f"   Temperature: {gpu.temperature}°C")
            else:
                print("🎮 No GPUs detected")
        except:
            print("🎮 GPU monitoring not available")
            
    except ImportError:
        print("⚠️  psutil/GPUtil not installed - install with: pip install psutil GPUtil")
        print("💻 Basic system info not available")

def run_performance_benchmark():
    """Run a comprehensive performance benchmark"""
    print("🚀 CHATTERBOX PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # System info
    analyze_system_resources()
    
    # TTS performance
    tts_results = analyze_tts_performance()
    
    # Streaming performance
    streaming_results = analyze_streaming_performance()
    
    # Final recommendations
    print("\n🎯 PERFORMANCE RECOMMENDATIONS")
    print("=" * 60)
    
    if tts_results:
        avg_tts_speed = sum(r['speed_ratio'] for r in tts_results) / len(tts_results)
        
        if avg_tts_speed > 1.5:
            print("❌ TTS is too slow:")
            print("   - Consider using a smaller TTS model")
            print("   - Try CPU instead of GPU if GPU is overloaded")
            print("   - Reduce voice quality settings")
        elif avg_tts_speed > 1.0:
            print("⚠️  TTS could be faster:")
            print("   - Monitor GPU/CPU usage during synthesis")
            print("   - Consider optimizing voice settings")
        else:
            print("✅ TTS performance is good")
    
    if streaming_results:
        if streaming_results.get('synthesis_speed_ratio', 0) > 1.0:
            print("⚠️  Streaming synthesis needs optimization")
        else:
            print("✅ Streaming performance is good")
    
    print("\n🔧 GENERAL OPTIMIZATIONS:")
    print("   - Use shorter max_response_tokens for faster responses")
    print("   - Enable GPU acceleration if available")
    print("   - Monitor queue sizes during operation")
    print("   - Consider sentence-level streaming for better responsiveness")

if __name__ == "__main__":
    try:
        run_performance_benchmark()
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc() 