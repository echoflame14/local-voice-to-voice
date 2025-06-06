#!/usr/bin/env python3
"""
Test script to verify 3-sentence chunking is working correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tts.chatterbox_wrapper import ChatterboxTTSWrapper

def test_sentence_splitting():
    """Test that sentence splitting works correctly"""
    tts = ChatterboxTTSWrapper()
    
    # Test text with multiple sentences
    test_text = """
    This is the first sentence. This is the second sentence. This is the third sentence.
    This is the fourth sentence. This is the fifth sentence. This is the sixth sentence.
    This is the seventh sentence. This is the eighth sentence. This is the ninth sentence.
    """
    
    # Test sentence splitting
    sentences = tts._split_into_sentences(test_text)
    print(f"Original text split into {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}: {sentence}")
    
    # Test 3-sentence chunking
    chunk_size = 3
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        combined_text = ' '.join(chunk)
        if combined_text.strip():
            chunks.append(combined_text)
    
    print(f"\nGrouped into {len(chunks)} chunks of {chunk_size} sentences each:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {chunk}")
        print(f"    Length: {len(chunk)} characters")
        print(f"    Sentences in chunk: {len(chunk.split('. '))} (approximate)")
        print()

def test_progressive_chunker():
    """Test the progressive chunker configuration"""
    from src.pipeline.progressive_chunker import ProgressiveChunker
    
    print("Testing Progressive Chunker Configuration:")
    pc = ProgressiveChunker()
    print(f"  First chunk words: {pc.first_chunk_words}")
    print(f"  Second chunk sentences: {pc.second_chunk_sentences}")
    print(f"  Subsequent chunk sentences: {pc.subsequent_chunk_sentences}")
    
    # Test with sample text
    test_sentences = [
        "First sentence here.",
        "Second sentence here.", 
        "Third sentence here.",
        "Fourth sentence here.",
        "Fifth sentence here.",
        "Sixth sentence here.",
        "Seventh sentence here.",
        "Eighth sentence here.",
        "Ninth sentence here."
    ]
    
    print(f"\nSimulating progressive chunking with {len(test_sentences)} sentences:")
    
    pc = ProgressiveChunker()
    chunks_emitted = []
    
    # Simulate adding text word by word
    full_text = ' '.join(test_sentences)
    accumulated = ""
    
    for word in full_text.split():
        accumulated += word + " "
        chunk = pc.add_text(word + " ")
        if chunk:
            chunks_emitted.append(chunk)
            print(f"  Chunk {len(chunks_emitted)}: {repr(chunk[:60])}{'...' if len(chunk) > 60 else ''}")
    
    # Get remaining text
    remaining = pc.get_remaining()
    if remaining:
        chunks_emitted.append(remaining)
        print(f"  Final chunk: {repr(remaining[:60])}{'...' if len(remaining) > 60 else ''}")
    
    print(f"\nTotal chunks emitted: {len(chunks_emitted)}")

if __name__ == "__main__":
    print("🔬 Testing 3-Sentence Chunking System")
    print("=" * 50)
    
    print("\n1. Testing basic sentence splitting and chunking:")
    test_sentence_splitting()
    
    print("\n" + "=" * 50)
    print("2. Testing progressive chunker:")
    test_progressive_chunker()
    
    print("\n" + "=" * 50)
    print("✅ Test complete!") 