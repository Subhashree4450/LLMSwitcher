import lorem
import time
from transformers import AutoTokenizer
from models.model_interface import get_answer
from core.config import PHI, LLAMA, GEMMA

# Define approximate context window limits for each model
CONTEXT_WINDOWS = {
    "Phi": (PHI, 2048),
    "LLaMA": (LLAMA, 8192),
    "Gemma": (GEMMA, 8192)
}

# Load a tokenizer to measure token count
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def generate_prompt(target_tokens):
    """Generates a prompt with approx. `target_tokens` number of tokens."""
    prompt = ""
    while len(tokenizer.tokenize(prompt)) < target_tokens:
        prompt += lorem.paragraph() + " "
    tokens = tokenizer.tokenize(prompt)
    return tokenizer.convert_tokens_to_string(tokens[:target_tokens])

# Test prompt length around context window (e.g., 50%, 100%, 125%)
ratios = [0.5, 1.0, 1.25]

for model_name, (model_id, max_ctx) in CONTEXT_WINDOWS.items():
    print(f"\n🔍 Testing Model: {model_name} (Context Limit ≈ {max_ctx} tokens)\n")
    for ratio in ratios:
        prompt_len = int(max_ctx * ratio)
        prompt = generate_prompt(prompt_len)
        actual_tokens = len(tokenizer.tokenize(prompt))

        print(f"➡️ Prompt Length: {prompt_len} requested | {actual_tokens} actual tokens")

        try:
            start_time = time.time()
            response = get_answer(model_id, prompt)
            duration = time.time() - start_time

            print(f"✅ Model Response (Time: {round(duration, 2)}s):")
            print(response[:300] + "...\n" if len(response) > 300 else response + "\n")

        except Exception as e:
            print(f"❌ Error from {model_name} at {actual_tokens} tokens: {str(e)}\n")
