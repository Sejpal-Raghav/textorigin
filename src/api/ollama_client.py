import os
import requests

# Default Ollama host, configurable via env var
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def polish_text(text: str, model: str = "llama3:8b-instruct") -> str | None:
    prompt = (
        "Rewrite the following passage so it sounds more like a human wrote it.\n"
        "Keep the original meaning and facts unchanged.\n"
        "Only make minimal stylistic changes (e.g., vary sentence length,\n"
        "replace rare words, add natural connectors).\n"
        "Return the revised text exactly, no explanations or conversational intro.\n\n"
        f"Text:\n{text}"
    )
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4
        }
    }
    
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None
