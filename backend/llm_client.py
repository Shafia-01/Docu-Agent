# llm_client.py
"""
LLM Client Factory
Handles initialization for Gemini and Groq.
Add more providers here later (OpenAI, DeepSeek, etc.).
"""

import os
import google.generativeai as genai
from groq import Groq

def get_gemini_client():
    """Return configured Gemini client"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("⚠️ GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    return genai

def get_groq_client():
    """Return configured Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=api_key)
