#!/usr/bin/env python3
"""Test if Whisper is working"""
import numpy as np
import whisper
import time

print("Testing Whisper...")

# Create a simple test audio (sine wave)
duration = 2.0
sample_rate = 16000
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 440
audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

print(f"Loading tiny model...")
start = time.time()
model = whisper.load_model("tiny")
print(f"Model loaded in {time.time() - start:.1f}s")

print(f"Transcribing test audio...")
start = time.time()
try:
    result = model.transcribe(audio, language="en", fp16=False)
    print(f"Transcription done in {time.time() - start:.1f}s")
    print(f"Result: '{result['text']}'")
except Exception as e:
    print(f"Error: {e}")

print("\nIf this hangs, there's an issue with your Whisper installation.")