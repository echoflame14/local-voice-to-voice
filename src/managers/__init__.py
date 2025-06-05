"""
Manager classes for the voice-to-voice assistant.

This package contains specialized manager classes extracted from the VoiceAssistant
god class to implement the Single Responsibility Principle.
"""

from .conversation_manager import ConversationManager

__all__ = [
    'ConversationManager'
] 