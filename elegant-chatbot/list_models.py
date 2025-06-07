#!/usr/bin/env python3
"""List available OpenAI models"""
import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set")
    exit(1)

print(f"Using API key: {api_key[:10]}...{api_key[-4:]}")

try:
    client = OpenAI(api_key=api_key)
    models = client.models.list()
    
    print("\nAvailable models:")
    chat_models = []
    for model in models.data:
        if "gpt" in model.id.lower():
            chat_models.append(model.id)
            
    chat_models.sort()
    for model in chat_models:
        print(f"  - {model}")
        
except Exception as e:
    print(f"Error: {e}")