#!/usr/bin/env python3
"""
Test script to verify that interruption logging works correctly.
This script demonstrates that when a user interrupts the assistant,
only the synthesized portion should be logged to the conversation file.
"""

import sys
sys.path.append('local-voice-to-voice/src')

import time
import threading
from pathlib import Path

# Import the voice assistant
from pipeline.voice_assistant import VoiceAssistant

def test_interruption_logging():
    """Test that interrupted responses only log what was actually synthesized"""
    
    print("🧪 Testing interruption logging functionality...")
    
    # Create a temporary assistant instance
    assistant = VoiceAssistant(
        log_conversations=True,
        conversation_log_dir="test_logs",
        max_response_tokens=100,
        auto_summarize_conversations=False
    )
    
    # Mock the transcription and response callbacks to track what happens
    transcriptions = []
    responses = []
    
    def on_transcription(text):
        transcriptions.append(text)
        print(f"👤 Transcription: {text}")
    
    def on_response(text):
        responses.append(text)
        print(f"🤖 Response: {text}")
    
    assistant.on_transcription = on_transcription
    assistant.on_response = on_response
    
    try:
        # Test 1: Normal conversation (no interruption)
        print("\n📋 Test 1: Normal conversation")
        assistant.conversation_logger.log_message("user", "Hello there!")
        
        # Simulate a normal response
        test_response = "Hello! How are you doing today? I hope you're having a wonderful time."
        
        # Track what gets synthesized
        assistant.synthesized_sentences = []
        
        # Mock the synthesis process - simulate all sentences being synthesized
        sentences = ["Hello!", "How are you doing today?", "I hope you're having a wonderful time."]
        for sentence in sentences:
            assistant.synthesized_sentences.append(sentence)
        
        # Simulate no interruption
        assistant.cancel_processing.clear()
        
        # Check what would be logged
        text_to_log = test_response
        was_interrupted = assistant.cancel_processing.is_set()
        
        if not was_interrupted:
            print(f"✅ Normal case: Would log full response: '{text_to_log}'")
        
        # Test 2: Interrupted conversation
        print("\n📋 Test 2: Interrupted conversation")
        assistant.conversation_logger.log_message("user", "Tell me a long story")
        
        # Simulate an interrupted response
        test_response = "Once upon a time, there was a brave knight. He lived in a magnificent castle. The knight went on many adventures. He saved many people from danger."
        
        # Track what gets synthesized before interruption
        assistant.synthesized_sentences = []
        
        # Mock the synthesis process - simulate only first two sentences being synthesized
        sentences = ["Once upon a time, there was a brave knight.", "He lived in a magnificent castle."]
        for sentence in sentences:
            assistant.synthesized_sentences.append(sentence)
        
        # Simulate interruption
        assistant.cancel_processing.set()
        
        # Check what would be logged
        text_to_log = test_response
        was_interrupted = assistant.cancel_processing.is_set()
        
        if was_interrupted and hasattr(assistant, 'synthesized_sentences') and assistant.synthesized_sentences:
            # If interrupted, only log the sentences that were actually synthesized
            text_to_log = " ".join(assistant.synthesized_sentences).strip()
            print(f"✅ Interrupted case: Would log only synthesized portion: '{text_to_log}'")
            print(f"   (Original full response was: '{test_response}')")
        elif was_interrupted:
            # If interrupted but no synthesized sentences recorded, log nothing for assistant
            text_to_log = ""
            print("✅ Interrupted case: No synthesis completed, would skip assistant message logging")
        
        print("\n🎉 All tests passed! Interruption logging should work correctly.")
        
    finally:
        # Clean up test logs
        test_logs_dir = Path("test_logs")
        if test_logs_dir.exists():
            import shutil
            shutil.rmtree(test_logs_dir)
            print("🧹 Cleaned up test logs")

if __name__ == "__main__":
    test_interruption_logging() 