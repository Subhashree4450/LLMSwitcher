import requests

# Ollama-based models (local LLMs via HTTP API)
def ask_ollama_model(question: str, model: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": f"Q: {question}\nA:",
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "<no response>").strip()
    except requests.exceptions.RequestException as e:
        return f"[Error querying {model}]: {str(e)}"

# Wrapper for model selection
def get_answer(model_id: int, question: str) -> str:
    from core.config import PHI, LLAMA, GEMMA

    if model_id == PHI:
        return ask_ollama_model(question, model="phi3")        # use model name as per `ollama list`
    elif model_id == LLAMA:
        return ask_ollama_model(question, model="llama3")
    elif model_id == GEMMA:
        return ask_ollama_model(question, model="gemma:2b")
    else:
        return "Unknown model ID"
