"""Language Model module using OpenAI-compatible API"""
from .openai_compatible import OpenAICompatibleLLM
from .gemini_llm import GeminiLLM

__all__ = [
    "OpenAICompatibleLLM",
    "GeminiLLM",
] 