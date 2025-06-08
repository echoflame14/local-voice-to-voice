#!/usr/bin/env python3
"""
Voice-to-Voice Chatbot using Chatterbox TTS and LM Studio

Usage:
    python main.py                          # Run with default settings
    python main.py --config path/to/config # Use custom config
    python main.py --help                  # Show help
"""

import argparse
import sys
import signal
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs import config
from src.pipeline import VoiceAssistant
from src.utils.logger import logger
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

# Check which VAD is being used
try:
    from src.audio import vad
    if hasattr(vad, 'USE_NEW_VAD') and vad.USE_NEW_VAD:
        print(f"{Fore.GREEN}✅ Using new VAD implementation (Silero/Energy-based){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Using WebRTC VAD (consider installing PyTorch for better VAD){Style.RESET_ALL}")
except Exception:
    pass



def print_banner():
    """Print application banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                  🎤 Voice Chatbot Assistant 🎤               ║
║                                                              ║
║  Powered by Chatterbox TTS + LM Studio + OpenAI Whisper     ║
║                                                              ║
║  Commands:                                                   ║
║    - Just speak naturally to chat                           ║
║    - Press Ctrl+C to exit                                   ║
║    - Type 'quit' or 'exit' to stop                         ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_status(message: str, status_type: str = "info"):
    """Print colored status message"""
    colors = {
        "info": Fore.BLUE,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED,
        "speech": Fore.MAGENTA
    }
    color = colors.get(status_type, Fore.WHITE)
    print(f"{color}[{status_type.upper()}] {message}{Style.RESET_ALL}")


def setup_signal_handlers(assistant: VoiceAssistant):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print_status("Shutting down gracefully...", "warning")
        
        # Show performance summary
        try:
            from src.utils.performance_monitor import perf_monitor
            perf_monitor.log_session_summary()
        except:
            pass
            
        assistant.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_callbacks(assistant: VoiceAssistant):
    """Setup event callbacks for the assistant"""
    
    def on_speech_start():
        print_status("🎤 Listening...", "speech")
    
    def on_speech_end():
        print_status("🔇 Processing...", "info")
    
    def on_transcription(text: str):
        print(f"{Fore.GREEN}👤 You: {text}{Style.RESET_ALL}")
    
    def on_response(text: str):
        print(f"{Fore.CYAN}🤖 Assistant: {text}{Style.RESET_ALL}")
    
    def on_synthesis_start():
        print_status("🔊 Speaking...", "speech")
    
    def on_synthesis_end():
        print_status("✅ Ready", "success")
    
    # Set callbacks
    assistant.on_speech_start = on_speech_start
    assistant.on_speech_end = on_speech_end
    assistant.on_transcription = on_transcription
    assistant.on_response = on_response
    assistant.on_synthesis_start = on_synthesis_start
    assistant.on_synthesis_end = on_synthesis_end


def run_interactive_mode(assistant: VoiceAssistant):
    """Run the assistant in interactive voice mode"""
    print_status("Voice mode active! Start speaking...", "success")
    print_status("Press Ctrl+C to exit", "info")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print_status("\nShutting down...", "info")
        assistant.stop()
        print_status("Goodbye!", "info")


def run_text_mode(assistant: VoiceAssistant):
    """Run the assistant in text mode for testing"""
    print_status("Text mode active! Type messages to chat.", "success")
    print_status("Type 'quit', 'exit', or press Ctrl+C to exit", "info")
    print_status(f"Voice reference: {assistant.tts.voice_reference_path}", "info")
    
    try:
        while True:
            try:
                user_input = input(f"\n{Fore.GREEN}👤 You: {Style.RESET_ALL}")
                
                if user_input.lower().strip() in ['quit', 'exit', 'bye']:
                    break
                
                if not user_input.strip():
                    continue
                
                # Generate response
                print_status("🤖 Thinking...", "info")
                response = assistant.llm.generate(
                    user_input,
                    max_tokens=assistant.max_response_tokens,
                    temperature=assistant.llm_temperature
                )
                
                print(f"{Fore.CYAN}🤖 Assistant: {response}{Style.RESET_ALL}")
                
                # Automatically synthesize speech
                assistant.say(response, interrupt=False)
                
            except EOFError:
                break
    
    except KeyboardInterrupt:
        pass
    
    print_status("Goodbye!", "info")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Voice-to-Voice Chatbot with Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Run with default VAD mode
    python main.py --input-mode push_to_talk    # Use push-to-talk mode
    python main.py --model large                # Use Whisper large model
    python main.py --voice path.wav             # Use custom voice
    python main.py --text-mode                  # Run in text mode for testing
    python main.py --vad-aggressiveness 2       # More aggressive voice detection
    python main.py --ptt-key enter              # Use Enter key for push-to-talk
    """
    )
    
    parser.add_argument(
        "--model", 
        choices=["tiny", "base", "small", "medium", "large"],
        default=config.WHISPER_MODEL_SIZE,
        help="Whisper model size"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        help="Path to reference voice file"
    )
    
    parser.add_argument(
        "--input-device",
        type=int,
        help="Audio input device index (use fix_audio_device.py to list devices)"
    )
    
    parser.add_argument(
        "--llm-url",
        default=config.LM_STUDIO_BASE_URL,
        help="LM Studio API URL"
    )
    
    parser.add_argument(
        "--system-prompt",
        help="Custom system prompt for the LLM"
    )
    
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Google Gemini instead of LM Studio"
    )
    
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API instead of LM Studio"
    )
    
    parser.add_argument(
        "--input-mode",
        choices=["vad", "push_to_talk"],
        default=config.INPUT_MODE,
        help="Input mode: 'vad' for hands-free voice detection, 'push_to_talk' for manual control (default: vad)"
    )
    
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        default=config.VAD_AGGRESSIVENESS,
        help="VAD aggressiveness level: 0=least aggressive, 3=most aggressive (default: 1)"
    )
    
    parser.add_argument(
        "--ptt-key",
        default=config.PUSH_TO_TALK_KEY,
        help="Push-to-talk key (default: space)"
    )
    
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Run in text mode for testing (no voice input)"
    )
    
    parser.add_argument(
        "--no-grace-period",
        action="store_true", 
        help="Disable interrupt grace period for immediate interrupts (allows interrupting right when AI starts speaking)"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming synthesis for lower latency (experimental)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Word chunk size for streaming synthesis (default: 3, lower = faster)"
    )
    
    parser.add_argument(
        "--high-performance",
        action="store_true",
        help="Enable high-performance mode with aggressive optimizations"
    )
    
    parser.add_argument(
        "--fast-tts",
        action="store_true",
        help="Enable ultra-fast TTS synthesis (lower quality but much faster)"
    )
    
    return parser.parse_args()


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print_status(f"CUDA available: {device_name}", "success")
        else:
            print_status("CUDA not available, using CPU", "warning")
    except ImportError:
        print_status("PyTorch not installed, using CPU", "warning")


def load_config():
    """Load and validate configuration"""
    try:
        config.validate()
        return config
    except Exception as e:
        print_status(f"Configuration error: {e}", "error")
        sys.exit(1)


def main():
    """Main application entry point"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    
    # Print startup banner
    print_banner()
    
    # Check CUDA availability
    check_cuda()
    
    # Initialize assistant
    try:
        logger.info("Initializing Voice Assistant...")
        
        # Determine grace period based on command line option
        grace_period = 0.0 if args.no_grace_period else config.INTERRUPT_GRACE_PERIOD
        
        # Initialize with performance tracking
        start_init = time.time()
        
        # Apply high-performance configuration if requested
        if args.high_performance:
            from src.utils.performance_optimizer import create_high_performance_config
            perf_config = create_high_performance_config()
            logger.info("🚀 High-performance mode enabled")
        else:
            perf_config = {}
        
        # Configure LLM settings based on flags
        if args.use_openai:
            llm_base_url = "https://api.openai.com/v1"
            llm_api_key = config.OPENAI_API_KEY if hasattr(config, 'OPENAI_API_KEY') else None
            if not llm_api_key:
                print_status("Error: OPENAI_API_KEY not found in config", "error")
                sys.exit(1)
        else:
            llm_base_url = args.llm_url
            llm_api_key = config.LM_STUDIO_API_KEY
        
        assistant = VoiceAssistant(
            whisper_model_size=args.model,
            whisper_device="cpu",  # Always use CPU for Whisper (faster for real-time)
            use_gemini=args.use_gemini,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model="gpt-4o-mini" if args.use_openai else None,  # Model for OpenAI API
            gemini_api_key=config.GEMINI_API_KEY if hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY else None,
            gemini_model=config.GEMINI_MODEL if hasattr(config, 'GEMINI_MODEL') else "models/gemini-1.5-flash",
            system_prompt=args.system_prompt or config.SYSTEM_PROMPT,
            voice_reference_path=args.voice or str(config.VOICE_REFERENCE_PATH),
            tts_device=args.device, # Pass device from command line
            enable_sound_effects=perf_config.get('enable_sound_effects', config.ENABLE_SOUND_EFFECTS),
            sound_effect_volume=config.SOUND_EFFECT_VOLUME,
            voice_exaggeration=config.VOICE_EXAGGERATION,
            voice_cfg_weight=config.VOICE_CFG_WEIGHT,
            voice_temperature=config.VOICE_TEMPERATURE,
            sample_rate=config.SAMPLE_RATE,
            chunk_size=perf_config.get('chunk_size', config.CHUNK_SIZE),
            min_audio_amplitude=config.MIN_AUDIO_AMPLITUDE,
            # Input mode settings
            input_mode=args.input_mode,
            input_device=args.input_device,  # Pass input device from command line
            vad_aggressiveness=perf_config.get('vad_aggressiveness', args.vad_aggressiveness),
            vad_speech_threshold=config.VAD_SPEECH_THRESHOLD,
            vad_silence_threshold=config.VAD_SILENCE_THRESHOLD,
            push_to_talk_key=args.ptt_key,
            enable_interruption_sound=perf_config.get('enable_interruption_sound', config.ENABLE_INTERRUPTION_SOUND),
            enable_generation_sound=perf_config.get('enable_generation_sound', config.ENABLE_GENERATION_SOUND),
            max_response_tokens=perf_config.get('max_response_tokens', config.MAX_RESPONSE_TOKENS),
            llm_temperature=perf_config.get('llm_temperature', config.LLM_TEMPERATURE),
            auto_summarize_conversations=perf_config.get('auto_summarize_conversations', True),
            max_history_messages=perf_config.get('max_history_messages', 2000)
        )
        
        init_time = time.time() - start_init
        logger.success(f"Voice Assistant initialized in {init_time:.2f}s")
        
        # Enable streaming TTS if requested
        if args.streaming:
            from src.pipeline.streaming_tts import patch_voice_assistant_with_streaming
            assistant = patch_voice_assistant_with_streaming(assistant)
            logger.info(f"Streaming TTS enabled with chunk size: {args.chunk_size} words")
        
        # Apply additional performance optimizations if requested
        if args.high_performance:
            from src.utils.performance_optimizer import apply_performance_optimizations
            apply_performance_optimizations(assistant)
        
        # Apply TTS optimizations if requested
        if args.fast_tts:
            from src.utils.tts_optimizer import apply_tts_optimizations
            apply_tts_optimizations(assistant.tts)
            logger.info("⚡ Ultra-fast TTS optimizations applied")
        
        # Override grace period if requested
        if args.no_grace_period:
            assistant.synthesis_grace_period = 0.0
            logger.warning("Grace period disabled - immediate interrupts enabled!")
        
        # Setup callbacks and signal handlers
        setup_callbacks(assistant)
        setup_signal_handlers(assistant)
        
        # Start the assistant
        logger.info("Starting Voice Assistant...")
        
        # Show appropriate instructions based on input mode
        if args.input_mode == "vad":
            print_status("🎤 Voice Activity Detection enabled - just start talking!", "success")
            print_status(f"VAD Sensitivity: {args.vad_aggressiveness}/3 (higher = more sensitive)", "info")
        else:
            print_status(f"🎮 Push-to-Talk mode - press '{args.ptt_key}' to talk", "success")
        
        if args.streaming:
            print_status("🚀 Streaming TTS enabled for low latency", "success")
        
        if args.high_performance:
            print_status("⚡ High-performance mode active", "success")
        
        if args.use_gemini:
            print_status("🧠 Gemini 2.0 Flash with Google Search grounding enabled", "success")
        elif args.use_openai:
            print_status("🤖 Using OpenAI API", "success")
        
        if args.fast_tts:
            print_status("⚡ Ultra-fast TTS mode active", "success")
        
        print_status("Press Ctrl+C to exit", "info")
        
        # Choose mode based on arguments
        if args.text_mode:
            run_text_mode(assistant)
        else:
            assistant.start()
            run_interactive_mode(assistant)
            
    except Exception as e:
        print_status(f"Error initializing assistant: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main()) 