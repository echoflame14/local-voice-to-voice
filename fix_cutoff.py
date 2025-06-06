#!/usr/bin/env python3
"""
Quick fix for audio cutoff issues
"""

import sys
from pathlib import Path

# Check current settings
config_file = Path("configs/config.py")

print("🔧 Checking current audio settings...")

# Read config
with open(config_file, 'r') as f:
    content = f.read()
    
# Check for sentence grouping settings
if "STREAMING_CHUNK_SIZE_WORDS" in content:
    import re
    match = re.search(r'STREAMING_CHUNK_SIZE_WORDS\s*=\s*(\d+)', content)
    if match:
        current_chunk = int(match.group(1))
        print(f"Current chunk size: {current_chunk} words")
        
# Suggestions
print("\n💡 To fix audio cutoff issues:")
print("\n1. Run with streaming mode and larger chunks:")
print("   python main.py --use-gemini --streaming --chunk-size 10")

print("\n2. Or disable streaming for full synthesis:")
print("   python main.py --use-gemini")

print("\n3. For debugging, run with verbose logging:")
print("   python main.py --use-gemini --list-devices")

print("\n4. Current volume boost is set to 7x in stream_manager.py")

print("\n📊 Summary of fixes applied:")
print("- ✅ Volume boost increased to 7x")
print("- ✅ Sentence grouping to 3 sentences per chunk")
print("- ✅ Audio device selection support added")
print("\n🚀 Try running with the suggested commands above!")