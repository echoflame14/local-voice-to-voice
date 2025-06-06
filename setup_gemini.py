#!/usr/bin/env python3
"""
Setup script to configure Gemini API for the voice assistant
"""

import os
import sys
from pathlib import Path

def create_env_file(api_key: str):
    """Create or update .env file with Gemini configuration"""
    env_path = Path(".env")
    
    # Read existing .env file or create new one
    env_content = ""
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_content = f.read()
    
    # Check if GEMINI_API_KEY already exists
    lines = env_content.split('\n')
    gemini_key_exists = False
    
    for i, line in enumerate(lines):
        if line.startswith('GEMINI_API_KEY='):
            lines[i] = f'GEMINI_API_KEY={api_key}'
            gemini_key_exists = True
            break
    
    if not gemini_key_exists:
        # Add Gemini configuration
        if env_content and not env_content.endswith('\n'):
            env_content += '\n'
        env_content += f'\n# === GEMINI SETTINGS ===\n'
        env_content += f'GEMINI_API_KEY={api_key}\n'
        env_content += f'GEMINI_MODEL=gemini-1.5-flash-latest\n'
    else:
        env_content = '\n'.join(lines)
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Updated .env file with Gemini API key")

def main():
    print("🤖 Gemini Setup for Voice Assistant")
    print("=" * 40)
    
    # Check if API key was provided as argument
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        print("\nTo get your Gemini API key:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Sign in with your Google account")
        print("3. Click 'Create API key' and copy it")
        print()
        
        api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ Error: API key is required")
        sys.exit(1)
    
    if not api_key.startswith('AIza'):
        print("⚠️  Warning: This doesn't look like a valid Gemini API key")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            sys.exit(1)
    
    try:
        create_env_file(api_key)
        
        print("\n🎉 Gemini setup complete!")
        print("\nTo use Gemini with your voice assistant:")
        print("python main.py --use-gemini")
        print("\nOr for text mode testing:")
        print("python main.py --use-gemini --text-mode")
        
    except Exception as e:
        print(f"❌ Error setting up Gemini: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 