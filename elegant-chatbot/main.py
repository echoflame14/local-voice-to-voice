#!/usr/bin/env python3
"""
Elegant Chatbot - Main Entry Point
A simple, elegant voice-to-voice chatbot using GPT-4.1-nano
"""
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from core.voice_loop import VoiceAssistant


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Elegant Voice Assistant - Simple, powerful, beautiful"
    )
    
    # Core options
    parser.add_argument(
        "--no-interrupts", 
        action="store_true",
        help="Disable interrupt detection"
    )
    parser.add_argument(
        "--enable-memory",
        action="store_true", 
        help="Enable conversation memory"
    )
    parser.add_argument(
        "--no-effects",
        action="store_true",
        help="Disable sound effects"
    )
    
    # Model options
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    parser.add_argument(
        "--voice",
        type=str,
        help="Path to voice sample for TTS"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply command line overrides
    if args.no_interrupts:
        config.features.enable_interrupts = False
    if args.enable_memory:
        config.features.enable_memory = True
    if args.no_effects:
        config.features.enable_effects = False
    if args.whisper_model:
        config.model.whisper_model = args.whisper_model
    if args.voice:
        config.model.tts_voice = args.voice
        
    # Validate configuration
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease set required environment variables.")
        sys.exit(1)
        
    # Print configuration
    print("=" * 50)
    print("🎨 Elegant Chatbot Configuration")
    print("=" * 50)
    print(f"LLM: {config.model.llm_model}")
    print(f"STT: Whisper {config.model.whisper_model}")
    print(f"Features:")
    print(f"  - Interrupts: {'✓' if config.features.enable_interrupts else '✗'}")
    print(f"  - Memory: {'✓' if config.features.enable_memory else '✗'}")
    print(f"  - Effects: {'✓' if config.features.enable_effects else '✗'}")
    print("=" * 50)
    print()
    
    # Initialize and run assistant
    try:
        assistant = VoiceAssistant(config)
        assistant.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()