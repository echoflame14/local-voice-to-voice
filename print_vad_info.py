#!/usr/bin/env python3
"""
Add VAD info printing to the application startup
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Adding VAD info display to main.py...\n")

# Read main.py
main_path = Path("main.py")
with open(main_path, 'r') as f:
    content = f.read()

# Find where to insert the VAD info
insert_after = 'init()'
if insert_after in content:
    # Add VAD detection code after colorama init
    vad_info_code = '''

# Check which VAD is being used
try:
    from src.audio import vad
    if hasattr(vad, 'USE_NEW_VAD') and vad.USE_NEW_VAD:
        print(f"{Fore.GREEN}✅ Using new VAD implementation (Silero/Energy-based){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Using WebRTC VAD (consider installing PyTorch for better VAD){Style.RESET_ALL}")
except Exception:
    pass
'''
    
    # Insert the code
    insert_pos = content.find(insert_after) + len(insert_after)
    content = content[:insert_pos] + vad_info_code + content[insert_pos:]
    
    # Write back
    with open(main_path, 'w') as f:
        f.write(content)
    
    print("✅ Added VAD info display to main.py")
    print("Now when you run the app, it will show which VAD is being used.")
else:
    print("❌ Could not find insertion point in main.py")

# Also add a message when VAD is initialized
stream_manager_path = Path("src/audio/stream_manager.py")
if stream_manager_path.exists():
    with open(stream_manager_path, 'r') as f:
        sm_content = f.read()
    
    # Add logging after VAD initialization
    old_line = "self.input_manager = InputManager("
    if old_line in sm_content:
        new_line = """self.input_manager = InputManager(
        # Log which VAD backend is being used
        try:
            if hasattr(self.input_manager.vad, 'backend_type'):
                print(f"🎯 VAD Backend: {self.input_manager.vad.backend_type}")
        except:
            pass
        self.input_manager = InputManager("""
        
        # Don't modify for now, too complex
        print("  (Stream manager VAD logging would need more careful modification)")

print("\nDone! The app will now show which VAD is being used at startup.")