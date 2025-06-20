import requests
from transformers import pipeline

# DistilBERT (local HF pipeline)
distilbert_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def ask_distilbert(question: str, context: str) -> str:
    result = distilbert_pipeline(question=question, context=context)
    return result['answer']

# Ollama-based models (local LLMs via HTTP API)
def ask_ollama_model(question: str, context: str, model: str) -> str:
    url = "http://localhost:11434/api/generate"

    # Clean the context input
    context = context.strip()

    # Dynamically build prompt
    if context:
        prompt = f"Q: {question}\nContext: {context}\nA:"
    else:
        prompt = f"Q: {question}\nA:"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "<no response>").strip()
    except requests.exceptions.RequestException as e:
        return f"[Error querying {model}]: {str(e)}"



# Wrappers

def get_answer(model_id: int, question: str, context: str) -> str:
    from core.config import DISTILBERT, PHI, LLAMA, GEMMA

    if model_id == DISTILBERT:
        return ask_distilbert(question, context)
    elif model_id == PHI:
        return ask_ollama_model(question, context, model="phi3")
    elif model_id == LLAMA:
        return ask_ollama_model(question, context, model="llama3")
    elif model_id == GEMMA:
        return ask_ollama_model(question, context, model="gemma:2b")
    else:
        return "Unknown model ID"
