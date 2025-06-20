# main.py

from strategies.mlfq import MLFQSelector
from strategies.ecomls import EpsilonGreedySelector
from strategies.naive import NaiveSelector
from strategies.rr import RoundRobinSelector
from models.model_interface import get_answer
from utils.tokenizer_utils import get_token_count
from core.config import *

# Mapping from model IDs to model names
MODEL_NAMES = {
    DISTILBERT: "DistilBERT",
    PHI: "Phi",
    LLAMA: "LLaMA",
    GEMMA: "Gemma"
}

mlfq = MLFQSelector()
ecomls = EpsilonGreedySelector()
naive = NaiveSelector()
rr = RoundRobinSelector()

first_question = True
user_feedback = 1.0  # default feedback

print("\n🧠 Multi-Strategy LLM Switcher 🧠")
print("Choose a mode: \n1. MLFQ\n2. Epsilon-Greedy\n3. Naive\n4. Round Robin\n5. Token-Based\n")

mode = input("Enter mode number: ").strip()
mode = int(mode) if mode.isdigit() else 5

print("\nStart asking questions. Type 'exit' to quit.\n")

while True:
    question = input("Q: ").strip()
    if question.lower() in ['exit', 'quit']:
        break

    context = input("Context: ").strip()
    if not context:
        print("⚠️ Context required for answering.")
        continue

    # Model selection logic
    if mode == 1:
        model_id = mlfq.select_model(context, user_feedback, first_question)
        model_name = MODEL_NAMES.get(model_id, "Unknown")
        print(f"🔄 Selected (MLFQ): {model_id} ({model_name})")
    elif mode == 2:
        model_id = ecomls.select_model(context, user_feedback, first_question)
        model_name = MODEL_NAMES.get(model_id, "Unknown")
        print(f"🔄 Selected (Epsilon-Greedy): {model_id} ({model_name})")
    elif mode == 3:
        model_id = naive.select_model(content=context)
        model_name = MODEL_NAMES.get(model_id, "Unknown")
        print(f"🔄 Selected (Naive): {model_id} ({model_name})")
    elif mode == 4:
        model_id = rr.get_next_model()
        model_name = MODEL_NAMES.get(model_id, "Unknown")
        print(f"🔄 Selected (Round Robin): {model_id} ({model_name})")
    else:
        token_count = get_token_count(context)
        model_id = GEMMA if token_count > TOKEN_THRESHOLD_HIGH else DISTILBERT
        model_name = MODEL_NAMES.get(model_id, "Unknown")
        print(f"🔄 Selected (Token-Based): {model_id} ({model_name})")

    print("🧠 Running inference...")
    response = get_answer(model_id, question, context)
    print(f"A: {response}\n")

    # Ask feedback for all except token-based (optional)
    if mode in [1, 2, 3, 4]:
        feedback_input = input("Was this answer helpful? (y/n): ").strip().lower()
        user_feedback = 1.0 if feedback_input == 'y' else 0.0
        first_question = False
