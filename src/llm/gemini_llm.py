import google.generativeai as genai
from typing import List, Dict, Generator
import google.generativeai.types as genai_types

class GeminiLLM:
    """LLM wrapper for Google's Gemini chat models (e.g., gemini-1.5-flash)."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest", system_prompt: str = None):
        genai.configure(api_key=api_key)
        self._model_name = model
        try:
            self._model = genai.GenerativeModel(self._model_name)
            print(f"Successfully loaded Gemini model: {self._model_name}")
        except Exception as e:
            simplified_model_name = self._model_name.split('/')[-1] if '/' in self._model_name else self._model_name
            print(f"Warning: could not load model '{self._model_name}' directly: {e}. Trying simplified name '{simplified_model_name}'...")
            try:
                self._model = genai.GenerativeModel(simplified_model_name)
                print(f"Successfully loaded Gemini model with simplified name: {simplified_model_name}")
                self._model_name = simplified_model_name
            except Exception as e2:
                print(f"Error: Could not load Gemini model '{self._model_name}' or '{simplified_model_name}': {e2}")
                raise
        self.system_prompt = system_prompt or "You are a helpful assistant."
        
        # Define less restrictive safety settings
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

    def _build_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Google API expects dicts with 'role' = 'user' | 'model'"""
        converted = []
        role_map = {"assistant": "model"}  # Only map assistant to model, system handled separately
        for msg in messages:
            r = msg["role"]
            if r != "system":  # Skip system messages - handled via system_instruction
                converted.append({
                    "role": role_map.get(r, r),
                    "parts": [msg["content"]]
                })
        return converted

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, system_prompt: str = None) -> str:
        model_to_use = genai.GenerativeModel(
            self._model_name,
            system_instruction=(system_prompt or self.system_prompt),
            safety_settings=self.safety_settings
        )
        generation_config = genai_types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        try:
            response = model_to_use.generate_content(
                prompt, 
                generation_config=generation_config
            )
            if response.text:
                return response.text.strip()
            else:
                # Check for empty response due to safety or other reasons
                print(f"Warning: Gemini generate() returned no text. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                for candidate in response.candidates:
                    print(f"Candidate: {candidate}") # Log candidate details
                return "" # Return empty string if no text
        except Exception as e:
            print(f"Error during Gemini generate(): {e}")
            # Attempt to get more details if it's a BlockedPromptException or similar
            if hasattr(e, 'response') and e.response:
                 print(f"Underlying response: {e.response}")
            return "I'm sorry, there was an issue generating a response."

    def stream_chat(self, messages: List[Dict[str, str]], max_tokens: int = 4096, temperature: float = 0.7) -> Generator[str, None, None]:
        # Extract system prompt if present
        current_system_prompt = self.system_prompt
        history_messages = messages

        # If first message is system message, use it as system instruction
        if messages and messages[0]["role"] == "system":
            current_system_prompt = messages[0]["content"]
            history_messages = messages[1:]  # Remove system message from history
        
        # Create model with system instruction
        model_to_use = genai.GenerativeModel(
            self._model_name,
            system_instruction=current_system_prompt,
            safety_settings=self.safety_settings
        )
        
        # Convert remaining messages to Gemini format
        gemini_history = self._build_messages(history_messages)
        
        generation_config = genai_types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        try:
            response_stream = model_to_use.generate_content(
                contents=gemini_history,
                generation_config=generation_config,
                stream=True
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except genai_types.BlockedPromptException as e:
            print(f"Error: Gemini stream_chat() prompt was blocked. {e}")
            yield "I am unable to respond to that due to safety guidelines."
        except Exception as e:
            print(f"Error streaming from Gemini: {e}")
            yield "I'm sorry, an error occurred while streaming the response." 