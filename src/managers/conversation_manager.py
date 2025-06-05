"""
Conversation Manager for handling conversation history, logging, and summarization.

This module extracts conversation-related functionality from VoiceAssistant
to implement the Single Responsibility Principle.
"""

import json
from typing import List, Dict, Optional
from ..config import ConversationConfig, LLMConfig
from ..pipeline.conversation_logger import ConversationLogger
from ..pipeline.conversation_summarizer import ConversationSummarizer
from ..pipeline.hierarchical_memory_manager import HierarchicalMemoryManager
from ..utils.text_similarity import is_similar_text


class ConversationManager:
    """
    Handles conversation history, logging, and summarization.
    
    This class encapsulates all conversation-related functionality that was
    previously scattered throughout the VoiceAssistant class.
    """
    
    def __init__(self, 
                 conversation_config: ConversationConfig,
                 llm_config: LLMConfig):
        """
        Initialize ConversationManager.
        
        Args:
            conversation_config: Configuration for conversation handling
            llm_config: LLM configuration for summarization
        """
        self.config = conversation_config
        self.history: List[Dict[str, str]] = []
        
        # Initialize conversation logging if enabled
        if conversation_config.log_conversations:
            self.logger = ConversationLogger(log_dir=conversation_config.conversation_log_dir)
            
            # Create summarizer LLM
            summarizer_llm = self._create_summarizer_llm(llm_config)
            self.summarizer = ConversationSummarizer(summarizer_llm)
            
            # Initialize hierarchical memory manager
            self.memory_manager = HierarchicalMemoryManager(
                self.logger, 
                self.summarizer
            )
            
            # Start new conversation
            self.logger.start_new_conversation()
            
            # Process unsummarized conversations if auto-summarization is enabled
            if conversation_config.auto_summarize_conversations:
                print("🚀 Initializing memory system: Processing unsummarized conversations...")
                self._process_unsummarized_conversations()
                self._load_hierarchical_memory()
        else:
            self.logger = None
            self.summarizer = None
            self.memory_manager = None
    
    def _create_summarizer_llm(self, llm_config: LLMConfig):
        """Create LLM instance for summarization."""
        from configs import config as app_config
        
        if llm_config.use_gemini and llm_config.gemini_api_key:
            from ..llm import GeminiLLM
            return GeminiLLM(
                api_key=llm_config.gemini_api_key,
                model=llm_config.gemini_model,
                system_prompt=app_config.SUMMARIZER_SYSTEM_PROMPT
            )
        else:
            from ..llm import OpenAICompatibleLLM
            return OpenAICompatibleLLM(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model=None,
                system_prompt=app_config.SUMMARIZER_SYSTEM_PROMPT
            )
    
    def add_message(self, role: str, content: str, check_duplicates: bool = True) -> bool:
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            check_duplicates: Whether to check for duplicate messages
            
        Returns:
            bool: True if message was added, False if skipped due to duplication
        """
        if not content or not content.strip():
            print("⚠️ Skipping empty message")
            return False
        
        # Check for duplicates if enabled
        if check_duplicates and self.history:
            recent_messages = self.history[-2:] if len(self.history) >= 2 else self.history
            
            for msg in recent_messages:
                if msg['role'] == role and is_similar_text(msg['content'], content, method="char"):
                    print(f"⚠️ Skipping duplicate {role} message")
                    return False
        
        # Add message to history
        message = {"role": role, "content": content.strip()}
        self.history.append(message)
        
        # Log message if logging is enabled
        if self.logger:
            self.logger.log_message(role, content.strip())
        
        # Keep history manageable
        if len(self.history) > self.config.max_history_messages:
            print(f"📝 Truncating conversation history to {self.config.max_history_messages} messages")
            self.history = self.history[-self.config.max_history_messages:]
        
        return True
    
    def update_last_user_message(self, content: str) -> bool:
        """
        Update the content of the last user message (for speech continuations).
        
        Args:
            content: New content for the last user message
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.history:
            return False
        
        # Find the last user message
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "user":
                self.history[i]["content"] = content.strip()
                print(f"🔄 Updated conversation history with combined user text")
                return True
        
        return False
    
    def get_context_for_llm(self, user_text: str, max_memory_summaries: int = 5) -> List[Dict[str, str]]:
        """
        Get conversation context for LLM including memory hierarchy and recent history.
        
        Args:
            user_text: Current user input
            max_memory_summaries: Maximum number of memory summaries to include
            
        Returns:
            List of message dictionaries for LLM
        """
        messages = []
        
        # Add memory hierarchy if available
        if self.logger and hasattr(self, 'memory_manager'):
            # Ensure unsummarized conversations are processed
            self._process_unsummarized_conversations()
            
            # Get memory hierarchy content
            memory_content = self._get_memory_hierarchy_content(max_memory_summaries)
            if memory_content:
                messages.append({
                    "role": "system",
                    "content": f"Context from previous conversations:\n{memory_content}"
                })
        
        # Add recent conversation history
        if self.history:
            # Filter out system messages and get recent history
            history_messages = [msg for msg in self.history if msg['role'] != 'system']
            recent_history = history_messages[-10:]  # Last 10 non-system messages
            messages.extend(recent_history)
        
        return messages
    
    def _get_memory_hierarchy_content(self, max_summaries: int) -> str:
        """Get memory hierarchy content for LLM context."""
        if not self.logger:
            return ""
        
        final_memory_parts = []
        
        # Constants for memory limits
        MAX_LTMS = 1
        MAX_STMS = max(1, max_summaries // 3)
        MAX_SUMMARIES = max_summaries - MAX_LTMS - MAX_STMS
        
        # Add Long-Term Memory summaries
        latest_ltm_list = self.logger.get_latest_ltm_summaries(max_ltm_summaries=MAX_LTMS)
        for ltm in latest_ltm_list:
            final_memory_parts.append(f"[Long-Term Memory]:\n{ltm['content'].strip()}")
        
        # Add Short-Term Memory summaries
        stms = self.logger.get_latest_stm_summaries(max_stm_summaries=MAX_STMS * 2)
        for stm in stms[:MAX_STMS]:
            final_memory_parts.append(f"[Short-Term Memory]:\n{stm['content'].strip()}")
        
        # Add recent conversation summaries
        summaries = self.logger.get_latest_summaries(max_summaries=MAX_SUMMARIES * 2)
        conversation_count = 0
        for summary in summaries:
            if conversation_count >= MAX_SUMMARIES:
                break
            
            for msg in summary.get('messages', []):
                if msg['role'] == 'assistant':
                    final_memory_parts.append(f"[Recent Conversation]:\n{msg['content'].strip()}")
                    conversation_count += 1
                    break
        
        return "\n\n".join(final_memory_parts)
    
    def clear_history(self):
        """Clear conversation history and start new log file."""
        self.history.clear()
        if self.logger:
            self.logger.start_new_conversation()
        print("🧹 Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get copy of conversation history."""
        return self.history.copy()
    
    def load_conversation_file(self, filepath: str) -> bool:
        """
        Load conversation from file.
        
        Args:
            filepath: Path to conversation file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not self.logger:
            print("⚠️ Conversation logging is disabled")
            return False
        
        try:
            conversation = self.logger.load_conversation_file(filepath)
            if not conversation:
                print("⚠️ No conversation found in file")
                return False
            
            # Process the conversation (implementation depends on requirements)
            # For now, just print summary
            print(f"📁 Loaded conversation with {len(conversation)} messages")
            return True
            
        except Exception as e:
            print(f"❌ Error loading conversation: {e}")
            return False
    
    def get_conversation_files(self) -> List[str]:
        """Get list of conversation log files."""
        if not self.logger:
            return []
        return self.logger.get_conversation_files()
    
    def _process_unsummarized_conversations(self):
        """Process any unsummarized conversations."""
        if not self.logger:
            return
        
        try:
            # Process unsummarized conversations using the conversation logger
            unsummarized = self.logger.get_unsummarized_conversations()
            if unsummarized:
                print(f"🔄 Found {len(unsummarized)} new conversations to summarize individually")
                current_file = str(self.logger.current_log_file)
                
                for filepath in unsummarized:
                    # Skip the current conversation file
                    if filepath == current_file:
                        print(f"⏩ Skipping current active conversation: {filepath}")
                        continue
                        
                    try:
                        print(f"\n📝 Summarizing new conversation: {filepath}")
                        conversation = self.logger.load_conversation_file(filepath)
                        
                        # Only summarize if there are messages
                        if conversation:
                            print("🤖 Generating summary (streaming):")
                            print("=" * 60)
                            
                            # Stream the summary generation and collect chunks
                            summary_chunks = []
                            for chunk in self.summarizer.stream_summarize_conversation(conversation):
                                print(chunk, end='', flush=True)
                                summary_chunks.append(chunk)
                            print("\n" + "=" * 60)
                            
                            # Create summary message
                            summary_text = "".join(summary_chunks).strip()
                            if not summary_text:
                                print("⚠️ Empty summary generated, using fallback")
                                summary_messages = self.summarizer._fallback_summary(conversation)
                            else:
                                summary_messages = [{
                                    "role": "assistant",
                                    "content": summary_text
                                }]
                            
                            # Save the summary
                            summary_file = self.logger.save_conversation_summary(filepath, summary_messages)
                            print(f"✅ Created new summary: {summary_file}")
                            
                            # Debug output
                            print("\n[DEBUG] Generated summary:")
                            print(f"  {summary_text[:500]}{'...' if len(summary_text) > 500 else ''}")
                        else:
                            print(f"⚠️ Skipping {filepath} - no valid conversation data found")
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Error parsing JSON in {filepath}: {e}")
                        print(f"   File may be corrupted. Consider moving it to a backup location.")
                    except Exception as e:
                        print(f"⚠️ Error summarizing {filepath}: {e}")
            else:
                print("✨ No new individual conversations to summarize")
                
        except Exception as e:
            print(f"⚠️ Error processing unsummarized conversations: {e}")
    
    def _load_hierarchical_memory(self):
        """Load hierarchical memory on startup."""
        if not self.memory_manager:
            return
        
        try:
            print("🧠 Loading hierarchical memory...")
            # Update memory hierarchy to build STMs and LTMs from existing summaries
            self.memory_manager.update_memory_hierarchy()
            print("✅ Memory hierarchy updated.")
        except Exception as e:
            print(f"⚠️ Error loading hierarchical memory: {e}")
    
    def is_logging_enabled(self) -> bool:
        """Check if conversation logging is enabled."""
        return self.logger is not None 