#!/usr/bin/env python3
"""Test chunking behavior"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.progressive_chunker import ProgressiveChunker

def test_chunking():
    pc = ProgressiveChunker()
    
    test_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five. This is sentence six. This is sentence seven."
    
    print("Testing progressive chunker with 3-sentence configuration...")
    print(f"Input text: {test_text}")
    print()
    
    # Simulate streaming text input
    chunks_emitted = []
    for word in test_text.split():
        chunk = pc.add_text(word + ' ')
        if chunk:
            chunks_emitted.append(chunk)
            sentences_in_chunk = len([s for s in chunk.split('.') if s.strip()])
            print(f"Chunk {len(chunks_emitted)}: {sentences_in_chunk} sentences")
            print(f"  Content: {repr(chunk)}")
            print()
    
    # Get any remaining text
    remaining = pc.get_remaining()
    if remaining:
        chunks_emitted.append(remaining)
        sentences_in_chunk = len([s for s in remaining.split('.') if s.strip()])
        print(f"Final chunk: {sentences_in_chunk} sentences")
        print(f"  Content: {repr(remaining)}")
        print()
    
    print(f"Total chunks emitted: {len(chunks_emitted)}")
    
    # Verify that later chunks have 3 sentences each (after the first chunk)
    if len(chunks_emitted) > 1:
        for i, chunk in enumerate(chunks_emitted[1:], 2):  # Skip first chunk
            sentences = len([s for s in chunk.split('.') if s.strip()])
            expected = 3
            status = "✅" if sentences == expected else "❌"
            print(f"Chunk {i}: {status} {sentences} sentences (expected {expected})")

if __name__ == "__main__":
    test_chunking() 