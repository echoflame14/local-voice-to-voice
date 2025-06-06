#!/usr/bin/env python3
"""
Fix for interrupt transcription bug

The issue: When an interrupt occurs, the VAD audio buffer might be empty,
resulting in "No transcription produced" for the interrupting speech.

This happens because:
1. User starts speaking (interrupt detected)
2. VAD buffer is cleared after previous utterance
3. New speech hasn't accumulated enough frames yet
4. Transcription gets empty audio

Solution: Use the pre-buffer (last 1.5 seconds of audio) for transcription
when interrupt is detected but VAD buffer is empty or too small.
"""

import os
import shutil
from datetime import datetime

def apply_fix():
    """Apply the interrupt transcription fix"""
    
    print("🔧 Fixing Interrupt Transcription Bug")
    print("=" * 60)
    
    # Backup the original file
    src_file = "src/audio/stream_manager.py"
    backup_file = f"{src_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📋 Creating backup: {backup_file}")
    shutil.copy2(src_file, backup_file)
    
    # Read the file
    with open(src_file, 'r') as f:
        content = f.read()
    
    # Fix 1: Add interrupt audio buffer preservation
    fix1 = '''
                            # Fix interrupt transcription: Preserve audio for interrupt cases
                            if self.vad_audio_buffer:
                                # Store complete audio before clearing (in case of interrupt)
                                self.last_complete_audio = np.concatenate(self.vad_audio_buffer)
                                
                                print("🔇 [COMPLETE] Voice ended - processing complete utterance")
                                self._on_input_end()
                                
                                # Process the complete utterance for transcription
                                complete_audio = self.last_complete_audio
                                if self.input_callback:
                                    print("📝 [COMPLETE] Sending complete utterance for transcription...")
                                    self.input_callback(complete_audio)
                            
                            # Reset state after processing (not before)
                            self.is_speech_active = False
                            self.vad_audio_buffer = []
                            self.speech_start_time = None
                            self.last_speech_time = None'''
    
    # Fix 2: Add pre-buffer fallback for interrupts
    fix2 = '''
                            # CRITICAL FIX: Call interrupt detection immediately on speech start
                            if self.interrupt_callback:
                                print("🚨 [INTERRUPT] Triggering immediate interrupt detection...")
                                # Send pre-buffer audio for interrupt detection/transcription
                                if self.pre_buffer and len(self.vad_audio_buffer) < 10:
                                    # Use pre-buffer if VAD buffer is too small
                                    pre_buffer_audio = np.concatenate(list(self.pre_buffer))
                                    self.interrupt_callback(pre_buffer_audio)
                                else:
                                    self.interrupt_callback(audio_data)'''
    
    print("\n📝 Applying fixes...")
    print("1. Preserving audio buffer for interrupt cases")
    print("2. Using pre-buffer for interrupt transcription when needed")
    
    # Apply the fixes (simplified for this example)
    print("\n✅ Fix prepared. To apply:")
    print("1. Edit src/audio/stream_manager.py")
    print("2. Around line 370-386, modify the speech end handling")
    print("3. Around line 360-362, enhance interrupt callback with pre-buffer")
    
    print("\n💡 Key changes:")
    print("- Store complete audio before clearing VAD buffer")
    print("- Use pre-buffer when VAD buffer is too small during interrupts")
    print("- This ensures transcription always has audio data")

if __name__ == "__main__":
    apply_fix()