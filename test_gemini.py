#!/usr/bin/env python3
"""
Test script to verify Gemini API integration
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gemini_api():
    """Test Gemini API connectivity and functionality"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please run: python setup_gemini.py")
        return False
    
    print(f"🔑 Found API key: {api_key[:10]}...")
    
    try:
        # Import and test Gemini LLM
        from src.llm import GeminiLLM
        
        print("🤖 Testing Gemini connection...")
        
        # Create Gemini LLM instance
        llm = GeminiLLM(
            api_key=api_key,
            model="gemini-1.5-flash-latest",
            system_prompt="You are a helpful assistant. Keep responses brief."
        )
        
        # Test basic generation
        print("🧪 Testing basic text generation...")
        response = llm.generate(
            "Say hello and tell me you're working correctly. Keep it brief.",
            max_tokens=100,
            temperature=0.7
        )
        
        if response:
            print(f"✅ Gemini response: {response}")
            
            # Test streaming
            print("\n🧪 Testing streaming generation...")
            messages = [
                {"role": "user", "content": "Count from 1 to 5, each number on a new line."}
            ]
            
            streaming_response = ""
            for chunk in llm.stream_chat(messages, max_tokens=50, temperature=0.3):
                streaming_response += chunk
                print(f"📝 Chunk: {chunk}")
            
            print(f"✅ Streaming complete: {streaming_response}")
            
            return True
        else:
            print("❌ Error: No response from Gemini")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Gemini API Test")
    print("=" * 30)
    
    if test_gemini_api():
        print("\n🎉 Gemini API test successful!")
        print("\nYou can now use Gemini with:")
        print("python main.py --use-gemini --text-mode")
    else:
        print("\n❌ Gemini API test failed!")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main() 