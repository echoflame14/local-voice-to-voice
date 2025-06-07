from typing import List, Dict, Iterator
import json
from datetime import datetime
import traceback

class ConversationSummarizer:
    """Handles summarization of conversation histories using Gemini"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def summarize_conversation(self, conversation: List[Dict]) -> List[Dict]:
        """
        Summarize a conversation into a condensed form while preserving context
        Returns a list of messages that can be used as conversation history
        """
        if not conversation:
            return []
            
        try:
            print("[SUMMARIZER DEBUG] Starting conversation summarization...")
            # Format conversation for the LLM
            formatted_convo = self._format_conversation_for_summary(conversation)
            
            if not formatted_convo:
                print("[SUMMARIZER WARNING] No valid messages to summarize")
                return self._fallback_summary(conversation)
            
            # Create summarization prompt
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a conversation summarizer. Create a concise but informative summary of the conversation. "
                        "Focus on key points, decisions, and outcomes. Write in a narrative style with clear paragraphs. "
                        "Include:\n"
                        "1. The main topic or purpose of the conversation\n"
                        "2. Key points discussed and any decisions made\n"
                        "3. The outcome or conclusion\n"
                        "Keep the summary focused and avoid unnecessary details."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please summarize this conversation in a few clear paragraphs. "
                        "Focus on the main points and outcomes:\n\n" +
                        formatted_convo
                    )
                }
            ]
            
            print("[SUMMARIZER DEBUG] Sending summarization request to LLM...")
            # Get summary from LLM using non-streaming method for reliability
            try:
                summary_text = self.llm.chat(
                    messages=prompt,
                    max_tokens=1000,
                    temperature=0.3
                ).strip()
                
                print(f"[SUMMARIZER DEBUG] Received complete summary: {len(summary_text)} chars")
                
                if not summary_text:
                    print("[SUMMARIZER WARNING] Empty summary received")
                    return self._fallback_summary(conversation)
                
                print(f"[SUMMARIZER DEBUG] Generated summary ({len(summary_text)} chars)")
                # Return as a single assistant message containing the narrative summary
                return [{
                    "role": "assistant",
                    "content": summary_text
                }]
                
            except Exception as e:
                print(f"[SUMMARIZER ERROR] Error during LLM summarization: {e}")
                print(f"[SUMMARIZER ERROR] Traceback: {traceback.format_exc()}")
                return self._fallback_summary(conversation)
                
        except Exception as e:
            print(f"[SUMMARIZER ERROR] Error in summarization process: {e}")
            print(f"[SUMMARIZER ERROR] Traceback: {traceback.format_exc()}")
            return self._fallback_summary(conversation)
    
    def _parse_summary_to_messages(self, summary: str) -> List[Dict]:
        """Parse summary into a list of messages with improved role detection"""
        messages = []
        current_role = None
        current_content = []
        
        # Split into lines and process
        for line in summary.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Look for role prefixes at start of line
            role_match = None
            if line.upper().startswith('USER:'):
                role_match = ('user', line[5:].strip())
            elif line.upper().startswith('ASSISTANT:'):
                role_match = ('assistant', line[10:].strip())
            
            if role_match:
                # Save previous message if exists
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": ' '.join(current_content).strip()
                    })
                    current_content = []
                
                # Start new message
                current_role = role_match[0]
                if role_match[1]:  # If there's content after the role prefix
                    current_content = [role_match[1]]
            elif current_role:  # Continue previous message
                current_content.append(line)
        
        # Add final message
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": ' '.join(current_content).strip()
            })
        
        return messages
    
    def fix_existing_summary(self, summary: List[Dict]) -> List[Dict]:
        """Fix an existing summary to ensure it's in the correct format"""
        if not summary:
            return []
        
        # Extract all meaningful content and rebuild with correct roles
        fixed_messages = []
        for msg in summary:
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()
            
            # Skip empty or system messages
            if not content or role == 'system':
                continue
            
            # Ensure role is either user or assistant
            if role not in ['user', 'assistant']:
                # Try to detect role from content
                if content.upper().startswith('USER:'):
                    role = 'user'
                    content = content[5:].strip()
                elif content.upper().startswith('ASSISTANT:'):
                    role = 'assistant'
                    content = content[10:].strip()
                else:
                    # Default to assistant if role can't be determined
                    role = 'assistant'
            
            if content:
                fixed_messages.append({
                    "role": role,
                    "content": content
                })
        
        return fixed_messages
    
    def _format_conversation_for_summary(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation messages for the LLM ensuring proper role alternation"""
        formatted = []
        
        try:
            for msg in conversation:
                # Handle missing role or content
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                # Handle dict content (like from Whisper)
                if isinstance(content, dict) and 'text' in content:
                    content = content['text']
                
                content = str(content).strip()
                
                # Skip empty messages
                if not content:
                    continue
                
                # Handle system messages specially
                if role == "system":
                    formatted.append(f"[SYSTEM CONTEXT]: {content}")
                    continue
                
                # Format with consistent role prefixes
                role_prefix = 'USER' if role == 'user' else 'ASSISTANT'
                formatted.append(f"{role_prefix}: {content}")
            
            return "\n\n".join(formatted)
            
        except Exception as e:
            print(f"[SUMMARIZER ERROR] Error formatting conversation: {e}")
            print(f"[SUMMARIZER ERROR] Traceback: {traceback.format_exc()}")
            return ""
    
    def _fallback_summary(self, conversation: List[Dict], max_messages: int = 5) -> List[Dict]:
        """Create a minimal fallback summary"""
        # Create a simple narrative summary
        summary = "This conversation contains several exchanges between the user and assistant. "
        
        if conversation:
            # Add first message context if it's a system message
            if conversation[0].get('role') == 'system':
                summary += f"Context: {conversation[0].get('content', '')}. "
            
            # Add basic info about the conversation
            summary += f"The conversation includes {len(conversation)} messages. "
            
            # Add note about fallback
            summary += "Note: This is a simplified summary due to processing limitations."
        
        return [{
            "role": "assistant",
            "content": summary.strip()
        }]
    
    def stream_summarize_conversation(self, conversation: List[Dict[str, str]]) -> Iterator[str]:
        """
        Stream the summarization of a conversation using the LLM
        
        Args:
            conversation: List of message dictionaries with 'role' and 'content'
            
        Yields:
            str: Chunks of the generated summary
        """
        # Load summarization model if available
        if hasattr(self.llm, 'load_summarization_model'):
            self.llm.load_summarization_model()
            
        try:
            # Format conversation for the LLM
            formatted_convo = self._format_conversation_for_summary(conversation)
            
            # Create the prompt for narrative summary
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a conversation summarizer. Create a concise but informative summary of the conversation. "
                        "Focus on key points, decisions, and outcomes. Write in a narrative style with clear paragraphs. "
                        "Include:\n"
                        "1. The main topic or purpose of the conversation\n"
                        "2. Key points discussed and any decisions made\n"
                        "3. The outcome or conclusion\n"
                        "Keep the summary focused and avoid unnecessary details."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please summarize this conversation in a few clear paragraphs. "
                        "Focus on the main points and outcomes:\n\n" +
                        formatted_convo
                    )
                }
            ]
            
            # Stream summary generation
            for chunk in self.llm.stream_chat(
                messages=prompt,
                max_tokens=1000,
                temperature=0.3
            ):
                yield chunk
                
        finally:
            # Restore original model if available
            if hasattr(self.llm, 'restore_original_model'):
                self.llm.restore_original_model() 