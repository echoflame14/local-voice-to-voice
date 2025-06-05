"""
Text similarity utilities for duplicate detection and comparison.

This module provides unified text similarity checking that supports both
word-based and character-based comparison methods.
"""

import re
from typing import Set


def is_similar_text(text1: str, text2: str, 
                   similarity_threshold: float = 0.8, 
                   method: str = "word") -> bool:
    """
    Unified text similarity checker supporting both word and character-based comparison.
    
    Args:
        text1, text2: Texts to compare
        similarity_threshold: Minimum similarity ratio (0-1)
        method: "word" for word-based, "char" for character-based comparison
        
    Returns:
        bool: True if texts are similar enough based on the threshold
        
    Examples:
        >>> is_similar_text("Hello world", "Hello there", method="word")
        False
        >>> is_similar_text("Hello world", "Hello world!", method="char")
        True
    """
    if not text1 or not text2:
        return False
    
    # Clean and normalize texts
    text1 = re.sub(r'\s+', ' ', text1.strip().lower())
    text2 = re.sub(r'\s+', ' ', text2.strip().lower())
    
    # Quick length check - if texts differ by more than 20% in length, likely not similar
    if abs(len(text1) - len(text2)) / max(len(text1), len(text2)) > 0.2:
        return False
    
    # Check if one is a substring of the other
    if text1 in text2 or text2 in text1:
        return True
    
    if method == "word":
        return _word_based_similarity(text1, text2, similarity_threshold)
    elif method == "char":
        return _character_based_similarity(text1, text2, similarity_threshold)
    else:
        raise ValueError(f"Unknown similarity method: {method}. Use 'word' or 'char'.")


def _word_based_similarity(text1: str, text2: str, threshold: float) -> bool:
    """
    Calculate similarity using word-based comparison (Jaccard similarity).
    
    This method compares the intersection and union of word sets.
    """
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return False
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    similarity = len(intersection) / len(union)
    return similarity >= threshold


def _character_based_similarity(text1: str, text2: str, threshold: float) -> bool:
    """
    Calculate similarity using character-based comparison with sliding window.
    
    This method uses a sliding window approach to find the best character match
    between the shorter and longer text.
    """
    shorter = text1 if len(text1) <= len(text2) else text2
    longer = text2 if len(text1) <= len(text2) else text1
    
    # Use sliding window to find best match
    max_similarity = 0
    window_size = len(shorter)
    
    for i in range(len(longer) - window_size + 1):
        window = longer[i:i + window_size]
        matches = sum(1 for a, b in zip(shorter, window) if a == b)
        similarity = matches / window_size
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity >= threshold


def clean_text_for_comparison(text: str) -> str:
    """
    Clean text for comparison by normalizing whitespace and case.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text with normalized whitespace and lowercase
    """
    if not text:
        return ""
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text 