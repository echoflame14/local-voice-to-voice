import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ..utils.text_similarity import is_similar_text

class ConversationLogger:
    """Handles logging and loading of conversation histories"""
    
    def __init__(self, log_dir: str = "conversation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summaries directory for individual conversation summaries
        self.summaries_dir = self.log_dir / "summaries"
        self.summaries_dir.mkdir(exist_ok=True)

        # Create stm_summaries directory for Short-Term Memory summaries
        self.stm_summaries_dir = self.log_dir / "stm_summaries"
        self.stm_summaries_dir.mkdir(exist_ok=True)

        # Create ltm_summaries directory for Long-Term Memory summaries
        self.ltm_summaries_dir = self.log_dir / "ltm_summaries"
        self.ltm_summaries_dir.mkdir(exist_ok=True)
        
        self.current_log_file = None
        
    def start_new_conversation(self) -> str:
        """Start a new conversation log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        self.current_log_file = self.log_dir / filename
        
        # Initialize with empty conversation
        self._save_conversation([])
        return str(self.current_log_file)
    
    def log_message(self, role: str, content: str):
        """Log a single message to the current conversation"""
        if not self.current_log_file:
            self.start_new_conversation()
            
        # Validate message content
        if not content or not content.strip():
            print("⚠️ Skipping empty message")
            return
            
        # Check for incomplete sentences
        if content.strip()[-1] not in '.!?':
            print("⚠️ Skipping incomplete message")
            return
            
        # Load existing conversation
        conversation = self._load_conversation()
        
        # Check for duplicates
        if conversation:
            last_message = conversation[-1]
            if (last_message['role'] == role and 
                is_similar_text(last_message['content'], content, method="word")):
                print("⚠️ Skipping duplicate message")
                return
        
        # Add the message
        conversation.append({
            "role": role,
            "content": content.strip(),
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_conversation(conversation)
    

    
    def get_current_conversation(self) -> List[Dict]:
        """Get the current conversation history"""
        return self._load_conversation()
    
    def _save_conversation(self, conversation: List[Dict]):
        """Save conversation to JSON file"""
        with open(self.current_log_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self, summary: List[Dict], original_file: Path):
        """Save a conversation summary"""
        summary_file = self.summaries_dir / f"{original_file.stem}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "original_file": str(original_file),
                "summary_timestamp": datetime.now().isoformat(),
                "messages": summary
            }, f, indent=2, ensure_ascii=False)
        return summary_file

    def _save_stm_summary(self, summary_content: str, constituent_summary_files: List[str]) -> Path:
        """Save a Short-Term Memory (STM) summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.stm_summaries_dir / f"stm_{timestamp}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary_type": "STM",
                "summary_timestamp": datetime.now().isoformat(),
                "content": summary_content,
                "constituent_summaries": constituent_summary_files 
            }, f, indent=2, ensure_ascii=False)
        return summary_file

    def _save_ltm_summary(self, summary_content: str, constituent_stm_files: List[str]) -> Path:
        """Save a Long-Term Memory (LTM) summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.ltm_summaries_dir / f"ltm_{timestamp}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary_type": "LTM",
                "summary_timestamp": datetime.now().isoformat(),
                "content": summary_content,
                "constituent_stms": constituent_stm_files
            }, f, indent=2, ensure_ascii=False)
        return summary_file
    
    def _load_conversation(self) -> List[Dict]:
        """Load conversation from current JSON file"""
        if not self.current_log_file or not self.current_log_file.exists():
            return []
        
        with open(self.current_log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_conversation_files(self) -> List[str]:
        """Get list of all conversation log files"""
        return [str(f) for f in self.log_dir.glob("conversation_*.json") 
                if f.parent == self.log_dir]  # Exclude files in summaries dir
    
    def get_summary_files(self) -> List[str]:
        """Get list of all summary files"""
        return [str(f) for f in self.summaries_dir.glob("conversation_*_summary.json")]

    def get_stm_summary_files(self) -> List[str]:
        """Get list of all STM summary files."""
        return [str(f) for f in self.stm_summaries_dir.glob("stm_*_summary.json")]

    def get_ltm_summary_files(self) -> List[str]:
        """Get list of all LTM summary files."""
        return [str(f) for f in self.ltm_summaries_dir.glob("ltm_*_summary.json")]
    
    def has_summary(self, conversation_file: str) -> bool:
        """Check if a conversation already has a summary file"""
        conv_path = Path(conversation_file)
        expected_summary = self.summaries_dir / f"{conv_path.stem}_summary.json"
        return expected_summary.exists()
        
    def get_unsummarized_conversations(self) -> List[str]:
        """Get list of conversation files that don't have summaries yet"""
        all_convos = [f for f in self.log_dir.glob("conversation_*.json") 
                     if f.parent == self.log_dir]  # Exclude files in summaries dir
        
        # Only return conversations that don't have summary files
        return [str(f) for f in all_convos if not self.has_summary(f)]
    
    def load_conversation_file(self, filepath: str) -> List[Dict]:
        """Load a specific conversation file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_summary_file(self, filepath: str) -> Dict:
        """Load a specific summary file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_stm_summary_file(self, filepath: str) -> Dict:
        """Load a specific STM summary file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_ltm_summary_file(self, filepath: str) -> Dict:
        """Load a specific LTM summary file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_conversation_summary(self, filepath: str, summary: List[Dict]) -> str:
        """Save a summary for a specific conversation file"""
        original_file = Path(filepath)
        summary_file = self._save_summary(summary, original_file)
        return str(summary_file)
    
    def get_latest_summaries(self, max_summaries: int = 5) -> List[Dict]:
        """Get the most recent conversation summaries from both current and old directories"""
        # Get summaries from both current and old directories
        summary_files_paths = self.get_summary_files()
        
        # Also check old_summaries directory if it exists
        old_summaries_dir = self.log_dir / "old_summaries" # Maintained for backward compatibility
        if old_summaries_dir.exists():
            old_summary_files_paths = [str(f) for f in old_summaries_dir.glob("conversation_*_summary.json")]
            summary_files_paths.extend(old_summary_files_paths)
        
        # Remove duplicates that might arise from including old_summaries
        summary_files_paths = sorted(list(set(summary_files_paths)), key=lambda p: Path(p).name, reverse=True)

        if not summary_files_paths:
            return []
            
        # Load all summaries and sort by timestamp
        summaries = []
        for f_path_str in summary_files_paths:
            try:
                summary_data = self.load_summary_file(f_path_str)
                # Extract timestamp from the summary
                timestamp = summary_data.get('summary_timestamp')
                if timestamp:
                    summary_data['file_path'] = f_path_str # Inject file path
                    summaries.append(summary_data)
            except Exception as e:
                print(f"Error loading summary {f_path_str}: {e}")
                
        # Sort by timestamp (newest first) and take most recent
        # Already sorted by name which includes timestamp, but re-sorting by JSON timestamp is more robust
        summaries.sort(key=lambda x: x.get('summary_timestamp', ''), reverse=True)
        return summaries[:max_summaries]

    def get_latest_stm_summaries(self, max_stm_summaries: int = 5) -> List[Dict]:
        """Get the most recent STM summaries."""
        stm_summary_files_paths = self.get_stm_summary_files()
        # Sort by filename (which includes timestamp) to get newest first
        stm_summary_files_paths.sort(key=lambda p: Path(p).name, reverse=True)

        if not stm_summary_files_paths:
            return []
        
        stm_summaries = []
        for f_path_str in stm_summary_files_paths:
            try:
                stm_summary_data = self.load_stm_summary_file(f_path_str)
                if stm_summary_data.get('summary_timestamp'):
                    stm_summary_data['file_path'] = f_path_str # Inject file path
                    stm_summaries.append(stm_summary_data)
            except Exception as e:
                print(f"Error loading STM summary {f_path_str}: {e}")
        
        # Already sorted by name, but re-sorting by JSON timestamp is more robust
        stm_summaries.sort(key=lambda x: x.get('summary_timestamp', ''), reverse=True)
        return stm_summaries[:max_stm_summaries]

    def get_latest_ltm_summaries(self, max_ltm_summaries: int = 1) -> List[Dict]:
        """Get the most recent LTM summaries."""
        ltm_summary_files_paths = self.get_ltm_summary_files()
        # Sort by filename (which includes timestamp) to get newest first
        ltm_summary_files_paths.sort(key=lambda p: Path(p).name, reverse=True)

        if not ltm_summary_files_paths:
            return []

        ltm_summaries = []
        for f_path_str in ltm_summary_files_paths:
            try:
                ltm_summary_data = self.load_ltm_summary_file(f_path_str)
                if ltm_summary_data.get('summary_timestamp'):
                    ltm_summary_data['file_path'] = f_path_str # Inject file path
                    ltm_summaries.append(ltm_summary_data)
            except Exception as e:
                print(f"Error loading LTM summary {f_path_str}: {e}")
                
        # Already sorted by name, but re-sorting by JSON timestamp is more robust
        ltm_summaries.sort(key=lambda x: x.get('summary_timestamp', ''), reverse=True)
        return ltm_summaries[:max_ltm_summaries] 