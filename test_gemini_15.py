#!/usr/bin/env python3
"""Test script to verify Gemini 1.5 Flash integration"""

import os
from dotenv import load_dotenv
from src.llm.gemini_llm import GeminiLLM

def test_gemini_15():
    """Test Gemini 1.5 Flash functionality"""
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment")
        print("Please run 'python setup_gemini.py' first")
        return
    
    print("🔍 Testing Gemini 1.5 Flash integration...")
    print("-" * 50)
    
    try:
        # Initialize Gemini with 1.5 Flash
        llm = GeminiLLM(
            api_key=api_key,
            model="gemini-1.5-flash-latest",
            enable_grounding=False  # Disabled by default in config
        )
        
        print("✅ Successfully initialized Gemini 1.5 Flash")
        print(f"Model: {llm._model_name}")
        
        # Test simple generation
        print("\n📝 Testing simple generation...")
        response = llm.generate(
            "What is the capital of France? Answer in one sentence.",
            max_tokens=50,
            temperature=0.5
        )
        print(f"Response: {response}")
        
        # Test chat functionality
        print("\n💬 Testing chat functionality...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you tell me what 2+2 equals?"}
        ]
        chat_response = llm.chat(messages, max_tokens=50, temperature=0.5)
        print(f"Chat response: {chat_response}")
        
        # Test streaming
        print("\n🌊 Testing streaming functionality...")
        print("Streaming response: ", end="", flush=True)
        for chunk in llm.stream_chat(messages, max_tokens=50, temperature=0.5):
            print(chunk, end="", flush=True)
        print()
        
        print("\n✅ All tests passed! Gemini 1.5 Flash is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Please check your API key and internet connection")

if __name__ == "__main__":
    test_gemini_15()