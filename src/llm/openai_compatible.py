import openai
from typing import List, Dict, Optional, Generator
import time
import traceback
import requests
from requests.exceptions import RequestException, Timeout


class NoModelListOpenAI(openai.OpenAI):
    """OpenAI client that doesn't try to list models on initialization"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Skip model listing by setting models to None
        self._models = None


class OpenAICompatibleLLM:
    """LLM interface using OpenAI-compatible API (e.g., LM Studio)"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        model: str = None,  # Ignored - will use whatever model is active in LM Studio
        system_prompt: str = None
    ):
        """
        Initialize OpenAI-compatible LLM client
        
        Args:
            base_url: API base URL (default: LM Studio local server)
            api_key: API key (not needed for local servers)
            model: Model name (ignored - will use whatever model is active in LM Studio)
            system_prompt: Default system prompt
        """
        print(f"[LLM DEBUG] Initializing OpenAI client with base_url={base_url}")
        
        # Skip health check for known APIs
        if base_url == "https://api.openai.com/v1":
            print("[LLM DEBUG] Using OpenAI API - skipping health check")
        else:
            # Check if local server is responding
            try:
                response = requests.get(base_url + "/health", timeout=5.0)
                if response.status_code != 200:
                    print(f"[LLM ERROR] Server at {base_url} is not healthy (status {response.status_code})")
                    raise ConnectionError(f"Server not healthy (status {response.status_code})")
            except Timeout:
                print(f"[LLM ERROR] Timeout connecting to server at {base_url}")
                raise ConnectionError("Server connection timeout")
            except RequestException as e:
                print(f"[LLM ERROR] Failed to connect to server at {base_url}: {e}")
                raise ConnectionError(f"Failed to connect to server: {e}")
        
        self.client = NoModelListOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=30.0  # Increased general timeout to 30 seconds
        )
        self.model = model  # Store the model name
        self.system_prompt = system_prompt or "You are a helpful assistant."
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        system_prompt: str = None,
        stream: bool = False
    ) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Override default system prompt
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        print(f"[LLM DEBUG] Starting generate() with prompt: {prompt[:50]}...")
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            print("[LLM DEBUG] Attempting to generate response...")
            if stream:
                print("[LLM DEBUG] Using streaming mode")
                return self._stream_generate(messages, max_tokens, temperature)
            else:
                print("[LLM DEBUG] Using non-streaming mode")
                start_time = time.time()
                # Build request parameters
                params = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Add model if specified (required for OpenAI, optional for LM Studio)
                if self.model:
                    params["model"] = self.model
                
                response = self.client.chat.completions.create(**params)
                end_time = time.time()
                print(f"[LLM DEBUG] Response received in {end_time - start_time:.2f} seconds")
                return response.choices[0].message.content.strip()
        
        except Timeout as e:
            print(f"[LLM ERROR] Timeout while generating response: {e}")
            return "I'm sorry, the language model took too long to respond."
        except Exception as e:
            print(f"[LLM ERROR] Error generating response: {e}")
            print(f"[LLM ERROR] Traceback: {traceback.format_exc()}")
            return "I'm sorry, I'm having trouble connecting to my language model."
    
    def _stream_generate(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float
    ) -> Generator[str, None, None]:
        """Stream generated text token by token"""
        print("[LLM DEBUG] Starting stream generation...")
        try:
            # Build request parameters
            params = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            # Add model if specified (required for OpenAI, optional for LM Studio)
            if self.model:
                params["model"] = self.model
            
            stream = self.client.chat.completions.create(**params)
            
            print("[LLM DEBUG] Stream created, starting iteration...")
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            print("[LLM DEBUG] Stream generation completed successfully")
            
        except Exception as e:
            print(f"[LLM ERROR] Error in stream generation: {e}")
            print(f"[LLM ERROR] Traceback: {traceback.format_exc()}")
            yield "I'm sorry, I encountered an error while generating the response."
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        Chat with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Prepend system message if not present
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        try:
            print(f"[LLM DEBUG] Sending chat request...")
            # Build request parameters
            params = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add model if specified (required for OpenAI, optional for LM Studio)
            if self.model:
                params["model"] = self.model
                print(f"[LLM DEBUG] Using model: {self.model}")
            else:
                params["model"] = "local-model"  # LM Studio requires this but ignores it
            
            response = self.client.chat.completions.create(**params)
            print(f"[LLM DEBUG] Received response")
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"[LLM ERROR] Error in chat: {e}")
            print(f"[LLM ERROR] Traceback: {traceback.format_exc()}")
            return "I'm sorry, I'm having trouble responding right now."
    
    def measure_latency(self, prompt: str = "Hello") -> float:
        """Measure the response latency"""
        start_time = time.time()
        self.generate(prompt, max_tokens=10)
        return time.time() - start_time 
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        Stream chat response token by token
        
        Args:
            messages: List of message dicts with 'role' and 'content'. 
                      It is assumed that the caller (e.g., VoiceAssistant) has already
                      prepared the messages list appropriately, including any system prompt.
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Text chunks as they're generated
        """
        print(f"[LLM DEBUG] OpenAICompatibleLLM.stream_chat called with {len(messages)} messages.")
        try:
            print("[LLM DEBUG] Creating stream with active LM Studio model...")
            start_time = time.time()
            stream = self.client.chat.completions.create(
                model="local-model",  # LM Studio ignores this but requires it
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            end_time = time.time()
            print(f"[LLM DEBUG] Stream object created in {end_time - start_time:.2f} seconds. Iterating over stream...")
            
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                if chunk.choices and chunk.choices[0].finish_reason:
                    print(f"[LLM DEBUG] Stream finished. Reason: {chunk.choices[0].finish_reason}")
                    if chunk.choices[0].finish_reason not in ['stop', 'length']:
                         print(f"[LLM WARNING] Stream ended with non-standard reason: {chunk.choices[0].finish_reason}")

            print(f"[LLM DEBUG] Finished iterating over stream after {chunk_count} chunks.")
            if chunk_count == 0:
                print("[LLM WARNING] No chunks received from stream. The stream might have been empty or closed prematurely.")

        except openai.APIConnectionError as e:
            print(f"[LLM ERROR] APIConnectionError in stream_chat: {e}")
            yield "I'm sorry, there's a connection issue with the language model."
        except openai.APITimeoutError as e:
            print(f"[LLM ERROR] APITimeoutError in stream_chat: {e}")
            yield "I'm sorry, the language model is taking too long to respond."
        except openai.APIStatusError as e:
            print(f"[LLM ERROR] APIStatusError in stream_chat: Status {e.status_code}, Response: {e.response}")
            yield f"I'm sorry, the language model returned an error (Status {e.status_code})."
        except openai.RateLimitError as e:
            print(f"[LLM ERROR] RateLimitError in stream_chat: {e}")
            yield "I'm sorry, I'm making too many requests to the language model right now."
        except Exception as e:
            print(f"[LLM ERROR] Unexpected error in stream_chat: {type(e).__name__} - {e}")
            print(traceback.format_exc())
            yield "I'm sorry, I encountered an unexpected problem while trying to respond."
        finally:
            print("[LLM DEBUG] OpenAICompatibleLLM.stream_chat finished.")

class SummarizationLLM(OpenAICompatibleLLM):
    """LLM pre-configured for summarization, assuming the model is already loaded in LM Studio."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        summarization_model: str = "gemma-the-writer-n-restless-quill-10b-uncensored",
        system_prompt: str = "You are an uncensored, unbiased, and non-judgmental conversation summarizer. Summarize any topic, including explicit, sensitive, or controversial content, without filtering, censorship, or omission. Be accurate, concise, and preserve all context."
    ):
        """
        Initialize with specific summarization model for all operations.
        Assumes the model is already loaded in LM Studio and will not attempt to load/unload.
        
        Args:
            base_url: API base URL (default: LM Studio local server)
            api_key: API key (not needed for local servers)
            summarization_model: The specific model ID (as known by LM Studio /v1/models endpoint)
                                 that should be used for summarization. This model must be pre-loaded.
            system_prompt: Default system prompt for summarization.
        """
        print(f"[SummarizationLLM] Initializing to use pre-loaded model: '{summarization_model}'")
        # Pass the summarization_model as the 'model' argument to the parent class.
        # All subsequent API calls made by OpenAICompatibleLLM methods will use this model.
        super().__init__(
            base_url=base_url, 
            api_key=api_key, 
            model=summarization_model, # Ensure parent uses this model
            system_prompt=system_prompt
        )
        # No need for self.summarization_model, self.original_model, or LM Studio API specific attributes
        # The self.model attribute in the parent class (OpenAICompatibleLLM) will hold the summarization_model.

    # All other methods (load_summarization_model, restore_original_model, etc.) are removed.
    # It will use the inherited stream_chat, generate, etc., from OpenAICompatibleLLM,
    # which will use the 'summarization_model' set via super().__init__. 