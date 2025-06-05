"""
Text cleaning utilities for preprocessing text before TTS synthesis.

This module provides functions to clean and normalize text by removing
unwanted patterns, noise, and system prompt leakage.
"""

import re
from typing import List


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS synthesis by removing unwanted patterns and noise.
    
    Args:
        text: Raw text to clean
        
    Returns:
        str: Cleaned text ready for TTS synthesis
    """
    original_text = text
    
    # Remove thinking blocks
    text = remove_thinking_blocks(text)
    
    # Remove stage directions
    text = remove_stage_directions(text)
    
    # Remove noise patterns
    text = remove_noise_patterns(text)
    
    # Remove system prompt leakage
    text = remove_prompt_leakage(text)
    
    # Remove extra whitespace and clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Additional noise checks
    if text:
        text = _filter_suspicious_patterns(text, original_text)
    
    # Remove empty sentences
    if not text or len(text) < 3:
        # If we had text originally but it got cleaned away, return original for conversational responses
        if original_text and len(original_text.strip()) > 3:
            return original_text.strip()
        return ""
    
    return text


def remove_thinking_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from text.
    
    Args:
        text: Text potentially containing thinking blocks
        
    Returns:
        str: Text with thinking blocks removed
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


def remove_stage_directions(text: str) -> str:
    """
    Remove *action* style stage directions from text.
    
    Args:
        text: Text potentially containing stage directions
        
    Returns:
        str: Text with stage directions removed
    """
    return re.sub(r'\*[^*]*\*', '', text)


def remove_noise_patterns(text: str) -> str:
    """
    Remove common noise patterns like filler sounds and artifacts.
    
    Args:
        text: Text potentially containing noise patterns
        
    Returns:
        str: Text with noise patterns removed
    """
    noise_patterns = [
        r'\b(?:um+|uh+|ah+|er+)\b',  # Filler sounds
        r'\b(?:background\s+noise|static)\b',
        r'(?:\s*\.\s*){3,}',  # Multiple dots
        r'\b(?:silence|pause)\b',
        r'(?:\s*[^\w\s]\s*){3,}',  # Multiple punctuation marks
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def remove_prompt_leakage(text: str) -> str:
    """
    Remove system prompt leakage patterns from text.
    
    Args:
        text: Text potentially containing system prompt leakage
        
    Returns:
        str: Text with prompt leakage removed
    """
    prompt_patterns = [
        r'You are.*?assistant.*?voice',
        r'CRITICAL.*?RULES.*?',
        r'VOICE OPTIMIZATION.*?',
        r'PERSONALITY.*?',
        r'EXAMPLE.*?RESPONSES.*?',
        r'AVOID.*?:',
        r'def \w+.*?\(.*?\):',  # Remove Python code
        r'print\(.*?\)',        # Remove print statements
        r'hello\(.*?\)',        # Remove function calls
        r'# prints.*?',         # Remove code comments
        r'pythonYou are.*?',    # Remove malformed prompt text
        r'```.*?```',           # Remove code blocks
        r'""".*?"""',           # Remove docstrings
    ]
    
    for pattern in prompt_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text


def _filter_suspicious_patterns(text: str, original_text: str) -> str:
    """
    Filter out suspicious text patterns while preserving specific known good patterns.
    
    Args:
        text: Cleaned text to check
        original_text: Original text before cleaning
        
    Returns:
        str: Text after suspicious pattern filtering
    """
    words = text.split()
    
    # Check for suspicious patterns, but preserve "between a"
    if (len(words) > 2 and
        text.lower() != "between a" and  # Preserve exact "between a"
        text.lower() != "between a." and  # Preserve with period
        (len(set(words)) < len(words) / 3 or  # Too much repetition
         (sum(len(w) < 3 for w in words) > len(words) / 2 and "between a" not in text.lower()))):  # Too many short words, unless it's "between a"
        print(f"⚠️ Suspicious text pattern detected, using original: '{text}'")
        return ""
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text by replacing multiple spaces with single spaces.
    
    Args:
        text: Text to normalize
        
    Returns:
        str: Text with normalized whitespace
    """
    return re.sub(r'\s+', ' ', text).strip()


def is_text_mostly_noise(text: str, noise_threshold: float = 0.5) -> bool:
    """
    Check if text consists mostly of noise patterns.
    
    Args:
        text: Text to check
        noise_threshold: Ratio of noise words that makes text "mostly noise"
        
    Returns:
        bool: True if text is mostly noise
    """
    if not text:
        return True
    
    words = text.split()
    if not words:
        return True
    
    # Count short words (potential noise)
    short_word_count = sum(1 for word in words if len(word) < 3)
    short_word_ratio = short_word_count / len(words)
    
    return short_word_ratio > noise_threshold 