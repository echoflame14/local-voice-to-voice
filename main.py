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
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs import config
from src.pipeline import VoiceAssistant
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()


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
    python main.py                    # Run with default settings
    python main.py --model large      # Use Whisper large model
    python main.py --voice path.wav   # Use custom voice
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
        "--llm-url",
        default=config.LM_STUDIO_BASE_URL,
        help="LM Studio API URL"
    )
    
    parser.add_argument(
        "--system-prompt",
        help="Custom system prompt for the LLM"
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
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Print startup banner
    print_banner()
    
    # Check CUDA availability
    check_cuda()
    
    # Initialize assistant
    try:
        print_status("Initializing Voice Assistant...", "info")
        
        assistant = VoiceAssistant(
            whisper_model_size=args.model,
            whisper_device="cpu",  # Always use CPU for Whisper (faster for real-time)
            llm_base_url=args.llm_url,
            llm_api_key=config.LM_STUDIO_API_KEY,
            gemini_api_key=config.GEMINI_API_KEY if config.GEMINI_API_KEY else None,
            gemini_model=config.GEMINI_MODEL,
            system_prompt=args.system_prompt or config.SYSTEM_PROMPT,
            voice_reference_path=str(config.VOICE_REFERENCE_PATH),  # Use command line arg or default WAV
            tts_device=args.device, # Pass device from command line
            enable_sound_effects=config.ENABLE_SOUND_EFFECTS,
            sound_effect_volume=config.SOUND_EFFECT_VOLUME,
            voice_exaggeration=config.VOICE_EXAGGERATION,
            voice_cfg_weight=config.VOICE_CFG_WEIGHT,
            voice_temperature=config.VOICE_TEMPERATURE,
            sample_rate=config.SAMPLE_RATE,
            chunk_size=config.CHUNK_SIZE,
            min_audio_amplitude=config.MIN_AUDIO_AMPLITUDE,
            enable_interruption_sound=config.ENABLE_INTERRUPTION_SOUND,
            enable_generation_sound=config.ENABLE_GENERATION_SOUND,
            max_response_tokens=config.MAX_RESPONSE_TOKENS,
            llm_temperature=config.LLM_TEMPERATURE
        )
        
        # Setup callbacks and signal handlers
        setup_callbacks(assistant)
        setup_signal_handlers(assistant)
        
        # Start the assistant
        print_status("\nStarting Voice Assistant...", "info")
        print_status("Press Spacebar to talk, release to process", "info")
        print_status("Press Ctrl+C to exit", "info")
        
        assistant.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print_status("\nShutting down...", "info")
            assistant.stop()
            sys.exit(0)
            
    except Exception as e:
        print_status(f"Error initializing assistant: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main()) 