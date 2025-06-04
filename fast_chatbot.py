#!/usr/bin/env python3
"""
FAST Voice Chatbot - Optimized for Speed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import VoiceAssistant
from colorama import init, Fore, Style
import time

init()

def main():
    print(f"""
{Fore.CYAN}🚀 FAST Voice Chatbot - Speed Optimized 🚀{Style.RESET_ALL}

Optimizations:
- Tiny Whisper model (fastest STT)
- Short response limit (max 30 tokens)
- Optimized TTS settings
- CPU/GPU auto-detection
""")

    try:
        # Speed-optimized settings
        assistant = VoiceAssistant(
            # Fastest STT
            whisper_model_size="tiny",
            whisper_device="cpu",  # Tiny model is fast enough on CPU
            
            # LLM settings
            system_prompt="You are a helpful assistant. Keep all responses under 10 words. Be concise. Use natural punctuation: commas for pauses, periods for breaks. Spell out numbers.",
            max_response_tokens=30,  # Very short responses
            llm_temperature=0.3,     # Less randomness = faster
            
            # TTS settings - try GPU first, fallback to CPU
            tts_device="cuda",       # Will auto-fallback to CPU if needed
            voice_exaggeration=0.3,  # Less processing
            voice_cfg_weight=0.2,    # Faster generation
            voice_temperature=0.5,   # Less randomness
            
            # Audio settings
            sample_rate=16000,
            chunk_size=512,          # Smaller chunks = lower latency
            vad_aggressiveness=2     # More aggressive VAD
        )
        
        print(f"{Fore.GREEN}✅ Fast assistant ready!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Tip: Responses are limited to ~10 words for speed{Style.RESET_ALL}")
        print(f"{Fore.BLUE}📝 Type your message (or 'quit' to exit):{Style.RESET_ALL}")
        
        while True:
            try:
                user_input = input(f"\n{Fore.GREEN}👤 You: {Style.RESET_ALL}")
                
                if user_input.lower().strip() in ['quit', 'exit', 'bye']:
                    break
                
                if not user_input.strip():
                    continue
                
                # Time the response
                start_time = time.time()
                
                # Generate response
                print(f"{Fore.BLUE}🤖 Thinking...{Style.RESET_ALL}")
                response = assistant.llm.generate(
                    user_input,
                    max_tokens=30,
                    temperature=0.3
                )
                
                llm_time = time.time() - start_time
                print(f"{Fore.CYAN}🤖 Assistant: {response}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}⏱️  LLM: {llm_time:.2f}s{Style.RESET_ALL}")
                
                # Optional TTS
                speak = input(f"{Fore.YELLOW}🔊 Speak? (y/N): {Style.RESET_ALL}")
                if speak.lower().startswith('y'):
                    tts_start = time.time()
                    assistant.say(response, interrupt=False)
                    # Note: TTS runs in background thread, so this timing isn't accurate
                    print(f"{Fore.MAGENTA}🔊 TTS started...{Style.RESET_ALL}")
                
            except EOFError:
                break
    
    except KeyboardInterrupt:
        pass
    
    print(f"{Fore.GREEN}👋 Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 