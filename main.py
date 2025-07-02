import json
import os
import re
from strategies.mlfq import MLFQSelector
from strategies.ecomls import EpsilonGreedySelector
from strategies.naive import NaiveSelector
from strategies.rr import RoundRobinSelector
from models.model_interface import get_answer
from core.config import *
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import bert_score

# 🔧 Normalize and tokenize
def normalize(text):
    return re.findall(r'\b\w+\b', text.lower())

# ✅ Find best matching question by keyword overlap
def find_best_match(user_question, dataset):
    user_keywords = set(normalize(user_question))
    best_match = None
    max_overlap = 0
    for question in dataset.keys():
        ref_keywords = set(normalize(question))
        overlap = len(user_keywords & ref_keywords)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = question
    return best_match if max_overlap >= 2 else None  # Threshold for relevance

# Load dataset
with open(os.path.join("data", "synthetic_qa.json"), "r", encoding="utf-8") as f:
    synthetic_qa = json.load(f)

# Model mapping
MODEL_NAMES = {
    PHI: "Phi",
    LLAMA: "LLaMA",
    GEMMA: "Gemma"
}

mlfq = MLFQSelector()
ecomls = EpsilonGreedySelector()
naive = NaiveSelector()
rr = RoundRobinSelector()

first_question = True
user_feedback = 1.0  # default

print("\n🧠 Multi-Strategy LLM Switcher 🧠")
print("Choose a mode: \n1. MLFQ\n2. Epsilon-Greedy\n3. Naive\n4. Round Robin\n")
mode = input("Enter mode number: ").strip()
mode = int(mode) if mode.isdigit() and 1 <= int(mode) <= 4 else 4

print("\nStart asking questions. Type 'exit' to quit.\n")

while True:
    question = input("Q: ").strip()
    if question.lower() in ['exit', 'quit']:
        break

    # Model selection
    if mode == 1:
        model_obj = mlfq.select_model(question, user_feedback, first_question)
        model_id = model_obj.model_id
        model_name = model_obj.name
        print(f"🔄 Selected (MLFQ): {model_id} ({model_name})")
    elif mode == 2:
        model_obj = ecomls.select_model(question, user_feedback, first_question)
        model_id = model_obj.model_id
        model_name = model_obj.name
        print(f"🔄 Selected (Epsilon-Greedy): {model_id} ({model_name})")
    elif mode == 3:
        model_id = naive.select_model(content=question)
        model_name = MODEL_NAMES.get(model_id, "Unknown")
        print(f"🔄 Selected (Naive): {model_id} ({model_name})")
    elif mode == 4:
        model_id = rr.get_next_model()
        model_name = MODEL_NAMES.get(model_id, "Unknown")
        print(f"🔄 Selected (Round Robin): {model_id} ({model_name})")

    print("🧠 Running inference...")
    response = get_answer(model_id, question)
    print(f"A: {response}")

    # Match with best reference for evaluation
    best_match = find_best_match(question, synthetic_qa)
    if best_match:
        reference = synthetic_qa[best_match]
        print("\n📊 Evaluation Scores:")
        print(f"✅ Matched Reference Q: {best_match}")
        print(f"✅ Reference A: {reference}")

        # BLEU
        bleu_score = sentence_bleu([reference.split()], response.split())
        print(f"BLEU: {bleu_score:.4f}")

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge = scorer.score(reference, response)
        print(f"ROUGE-L: {rouge['rougeL'].fmeasure:.4f}")

        # BERTScore
        P, R, F1 = bert_score.score([response], [reference], lang="en", verbose=False)
        print(f"BERTScore-F1: {F1[0].item():.4f}\n")
    else:
        print("\n📌 No reference available for evaluation.\n")

    feedback_input = input("Was this answer helpful? (y/n): ").strip().lower()
    user_feedback = 1.0 if feedback_input == 'y' else 0.0
    first_question = False
