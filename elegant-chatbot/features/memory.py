"""
Simple memory system for conversation context
Elegant and effective
"""
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path


class ConversationMemory:
    """Simple conversation memory system"""
    
    def __init__(self, config):
        self.enabled = config.features.enable_memory
        self.max_exchanges = config.features.memory_max_exchanges
        self.context_size = config.features.memory_context_size
        self.memory_dir = config.paths.memory_dir
        
        # In-memory storage
        self.exchanges = []
        self.session_file = None
        
        if self.enabled:
            # Create session file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_file = self.memory_dir / f"session_{timestamp}.json"
            
    def add_exchange(self, user_text: str, assistant_text: str):
        """Add a conversation exchange"""
        if not self.enabled:
            return
            
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_text,
            "assistant": assistant_text
        }
        
        self.exchanges.append(exchange)
        
        # Limit memory size
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges:]
            
        # Save periodically
        if len(self.exchanges) % 10 == 0:
            self.save()
            
    def get_context(self) -> List[Dict]:
        """Get recent context for LLM"""
        if not self.enabled or not self.exchanges:
            return []
            
        # Get last N exchanges
        recent = self.exchanges[-self.context_size:]
        
        # Format for LLM
        context = []
        for exchange in recent:
            context.append({"role": "user", "content": exchange["user"]})
            context.append({"role": "assistant", "content": exchange["assistant"]})
            
        return context
        
    def save(self):
        """Save memory to disk"""
        if not self.enabled or not self.session_file:
            return
            
        data = {
            "session_start": self.exchanges[0]["timestamp"] if self.exchanges else None,
            "last_update": datetime.now().isoformat(),
            "exchanges": self.exchanges
        }
        
        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=2)
            
    def search(self, query: str) -> List[Dict]:
        """Simple keyword search in memory"""
        if not self.enabled:
            return []
            
        results = []
        query_lower = query.lower()
        
        for exchange in self.exchanges:
            if (query_lower in exchange["user"].lower() or 
                query_lower in exchange["assistant"].lower()):
                results.append(exchange)
                
        return results