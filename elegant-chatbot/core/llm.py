"""
LLM module - OpenAI GPT-4.1-nano
Simple, direct API wrapper
"""
from typing import Optional, List, Dict
from openai import OpenAI
import httpx
from config import LLMProvider


class LLMClient:
    """Simple LLM client"""
    
    def __init__(self, config):
        self.config = config
        self.provider = config.model.llm_provider
        self.model = config.model.llm_model
        self.api_key = config.api_keys.get("openai")
        
        # Conversation history
        self.history = []
        
        # Initialize client with optimized settings
        self.client = None
        if self.provider == LLMProvider.OPENAI and self.api_key:
            # Create HTTP client with connection pooling
            try:
                # Try with HTTP/2 first
                http_client = httpx.Client(
                    timeout=httpx.Timeout(30.0, connect=5.0),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    http2=True  # Enable HTTP/2 for better performance
                )
            except ImportError:
                # Fall back to HTTP/1.1
                print("  HTTP/2 not available, using HTTP/1.1")
                http_client = httpx.Client(
                    timeout=httpx.Timeout(30.0, connect=5.0),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    http2=False
                )
            
            self.client = OpenAI(
                api_key=self.api_key,
                http_client=http_client,
                max_retries=2
            )
            
    def warm_up(self):
        """Pre-warm the API connection to avoid cold start latency"""
        import time
        print("  Warming up LLM connection...")
        start_time = time.time()
        try:
            # Make a minimal API call to establish connection
            api_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0
            )
            api_time = time.time() - api_start
            total_time = time.time() - start_time
            print(f"  ✓ LLM connection ready! (API: {api_time:.2f}s, Total: {total_time:.2f}s)")
            return True
        except Exception as e:
            failed_time = time.time() - start_time
            print(f"  ⚠️ LLM warm-up failed after {failed_time:.2f}s: {e}")
            return False
            
    def generate(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Generate response from LLM"""
        if self.provider == LLMProvider.OPENAI:
            return self._generate_openai(prompt, context)
        else:
            return "LLM provider not supported"
            
    def _generate_openai(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Generate using OpenAI API"""
        import time
        
        print(f"\n[DEBUG] Starting LLM generation at {time.time():.3f}")
        total_start = time.time()
        
        # Message preparation
        prep_start = time.time()
        messages = []
        
        # Add context if provided
        if context:
            messages.extend(context)
            print(f"[DEBUG] Added {len(context)} context messages")
            
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        prep_time = time.time() - prep_start
        print(f"[DEBUG] Message preparation took {prep_time:.3f}s")
        
        try:
            print(f"[DEBUG] Calling OpenAI API at {time.time():.3f}")
            print(f"[DEBUG] Model: {self.model}, Messages: {len(messages)}, Max tokens: {self.config.model.llm_max_tokens}")
            
            api_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.model.llm_temperature,
                max_tokens=self.config.model.llm_max_tokens,
                stream=False,
                timeout=10  # 10 second timeout
            )
            api_time = time.time() - api_start
            print(f"[DEBUG] API call completed in {api_time:.3f}s")
            
            # Extract result
            extract_start = time.time()
            result = response.choices[0].message.content
            extract_time = time.time() - extract_start
            print(f"[DEBUG] Response extraction took {extract_time:.3f}s")
            print(f"[DEBUG] Response length: {len(result)} chars")
            
            # Update history
            history_start = time.time()
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": result})
            
            # Keep history reasonable size
            if len(self.history) > 20:
                self.history = self.history[-20:]
            history_time = time.time() - history_start
            print(f"[DEBUG] History update took {history_time:.3f}s")
            
            total_time = time.time() - total_start
            print(f"[DEBUG] Total LLM generation time: {total_time:.3f}s")
            print(f"[DEBUG] Breakdown: prep={prep_time:.3f}s, api={api_time:.3f}s, extract={extract_time:.3f}s, history={history_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"  LLM error: {type(e).__name__}: {e}")
            if "timeout" in str(e).lower():
                return "Sorry, the AI service is taking too long to respond."
            elif "connection" in str(e).lower():
                return "Sorry, I can't connect to the AI service right now."
            else:
                return "I'm sorry, I couldn't process that request."
            
    def get_context(self, max_exchanges: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        # Return last N exchanges (user + assistant pairs)
        exchanges = max_exchanges * 2
        return self.history[-exchanges:] if self.history else []
        
    def clear_history(self):
        """Clear conversation history"""
        self.history = []