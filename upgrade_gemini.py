#!/usr/bin/env python3
"""
Upgrade Google Generative AI library for grounding support
"""
import subprocess
import sys

def upgrade_gemini_library():
    """Upgrade to the latest version with grounding support"""
    commands = [
        ["pip", "install", "--upgrade", "google-generativeai>=0.3.0"],
        ["pip", "install", "--upgrade", "google-ai-generativelanguage"],
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(e.stderr)
    
    print("\n🚀 Library upgrade complete! Try running the voice assistant again.")

if __name__ == "__main__":
    upgrade_gemini_library()