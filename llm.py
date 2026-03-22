import os
import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")    

def chat(prompt: str, max_tokens: int = 500) -> str:
    if PROVIDER == "ollama":
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]
    elif PROVIDER == "anthropic":
        # Implement Anthropic API call here
        pass
    elif PROVIDER == "openai":
        # Implement OpenAI API call here
        pass
    else:
        raise ValueError(f"Unsupported LLM provider: {PROVIDER}")