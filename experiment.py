# import json
# import time
# import psutil
# import csv
# from codecarbon import EmissionsTracker
# from transformers import AutoTokenizer
# from bert_score import score as bert_score_fn
# from models.model_interface import get_answer
# from core.config import PHI, LLAMA, GEMMA

# from prometheus_client import start_http_server, Gauge

# # Start Prometheus metrics server         command prompt:prometheus --config.file=prometheus.yml
# start_http_server(8000)

# # Define per-model metrics with labels
# bert_score_metric = Gauge('bert_score_f1', 'BERT F1 Score', ['model', 'query'])
# token_count_metric = Gauge('token_count', 'Number of tokens', ['model', 'query'])
# energy_metric = Gauge('energy_consumption_kwh', 'Energy consumption in kWh', ['model', 'query'])
# cpu_metric = Gauge('cpu_used_percent', 'CPU usage', ['model', 'query'])
# memory_metric = Gauge('memory_used_mb', 'Memory used in MB', ['model', 'query'])
# inference_time_metric = Gauge('inference_time_seconds', 'Inference time in seconds', ['model', 'query'])

# # Load QA dataset
# with open("sample_qa_100.json", "r", encoding="utf-8") as f:
#     synthetic_qa = dict(list(json.load(f).items())[:50])

# print(f"📘 Loaded {len(synthetic_qa)} question-answer pairs.")

# # Load tokenizer
# default_tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")

# # Model registry
# MODELS = {
#     "Phi": PHI,
#     "LLaMA": LLAMA,
#     "Gemma": GEMMA
# }

# # CSV setup
# output_csv = "experiment_metrics_output.csv"
# with open(output_csv, mode="w", newline='', encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Query#", "Question", "Reference", "Model", "Answer", "BERT F1", "Tokens", "Energy (kWh)", "CPU (%)", "Memory (MB)", "Time (s)"])

#     for qnum, (question, reference) in enumerate(synthetic_qa.items(), start=1):
#         print(f"\n🟦 [{qnum}] Question: {question[:60]}...")
#         for model_name, model_id in MODELS.items():
#             try:
#                 print(f"🔸 Model: {model_name}")

#                 # Resource tracking
#                 tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
#                 cpu_before = psutil.cpu_percent(interval=None)
#                 memory_before = psutil.virtual_memory().used

#                 tracker.start()
#                 start_time = time.time()
#                 answer = get_answer(model_id, question)
#                 end_time = time.time()
#                 tracker.stop()

#                 cpu_after = psutil.cpu_percent(interval=None)
#                 memory_after = psutil.virtual_memory().used

#                 # Compute metrics
#                 inference_time = end_time - start_time
#                 cpu_used = abs(cpu_after - cpu_before)
#                 memory_diff = (memory_after - memory_before) / (1024 * 1024)
#                 token_count = len(default_tokenizer.tokenize(answer))
#                 emissions_data = tracker.final_emissions_data
#                 energy_kwh = emissions_data.energy_consumed if emissions_data else 0.0

#                 # BERTScore
#                 try:
#                     P, R, F1 = bert_score_fn([answer], [reference], lang="en", verbose=False)
#                     bert_f1 = round(F1[0].item(), 4)
#                 except Exception as e:
#                     print(f"⚠️ BERTScore failed: {e}")
#                     bert_f1 = 0.0

#                 query_str = str(qnum)

#                 # 🔁 Update Prometheus with labels
#                 bert_score_metric.labels(model=model_name, query=query_str).set(bert_f1)
#                 token_count_metric.labels(model=model_name, query=query_str).set(token_count)
#                 energy_metric.labels(model=model_name, query=query_str).set(energy_kwh)
#                 cpu_metric.labels(model=model_name, query=query_str).set(cpu_used)
#                 memory_metric.labels(model=model_name, query=query_str).set(memory_diff)
#                 inference_time_metric.labels(model=model_name, query=query_str).set(inference_time)

#                 # Write to CSV
#                 writer.writerow([qnum, question, reference, model_name, answer, bert_f1, token_count, energy_kwh, cpu_used, memory_diff, inference_time])
#                 file.flush()

#                 print(f"✅ {model_name} | Energy: {energy_kwh:.6f} kWh | Tokens: {token_count} | Time: {inference_time:.3f}s")

#             except Exception as e:
#                 print(f"❌ Error with {model_name}: {e}")



# import json
# import time
# import psutil
# import csv
# from codecarbon import EmissionsTracker
# from transformers import AutoTokenizer
# from bert_score import score as bert_score_fn
# from models.model_interface import get_answer
# from core.config import PHI, LLAMA, GEMMA

# from prometheus_client import start_http_server, Gauge

# # Start Prometheus metrics server
# start_http_server(8000)

# # Define metrics with labels for model and query number
# bert_score_metric = Gauge('bert_score_f1', 'BERT F1 Score', ['model', 'query'])
# token_count_metric = Gauge('token_count', 'Number of tokens in the generated answer', ['model', 'query'])
# energy_metric = Gauge('energy_consumption_kwh', 'Energy consumption in kWh', ['model', 'query'])
# cpu_metric = Gauge('cpu_used_percent', 'CPU usage during inference (%)', ['model', 'query'])
# memory_metric = Gauge('memory_used_mb', 'Memory usage during inference in MB', ['model', 'query'])
# inference_time_metric = Gauge('inference_time_seconds', 'Inference time in seconds', ['model', 'query'])

# # Load dataset
# with open("sample_qa_100.json", "r", encoding="utf-8") as f:
#     qa_data = dict(list(json.load(f).items())[:100])

# print(f"📘 Loaded {len(qa_data)} QA pairs.")

# # Load tokenizer (offline support)
# default_tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")

# # Define model registry
# MODELS = {
#     "Phi": PHI,
#     "LLaMA": LLAMA,
#     "Gemma": GEMMA
# }

# # Prepare CSV output
# csv_file = "experiment_metrics_output.csv"
# with open(csv_file, mode="w", newline='', encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Query#", "Question", "Reference", "Model", "Answer", "BERT F1", "Tokens", "Energy (kWh)", "CPU (%)", "Memory (MB)", "Time (s)"])

#     for q_num, (question, reference) in enumerate(qa_data.items(), start=1):
#         print(f"\n🟦 [{q_num}] Processing: {question[:70]}...")
#         query_id = str(q_num)

#         for model_name, model_id in MODELS.items():
#             try:
#                 print(f"🔸 Using model: {model_name}")
                
#                 tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
#                 cpu_before = psutil.cpu_percent(interval=None)
#                 mem_before = psutil.virtual_memory().used

#                 tracker.start()
#                 start_time = time.time()
#                 answer = get_answer(model_id, question)
#                 end_time = time.time()
#                 tracker.stop()

#                 cpu_after = psutil.cpu_percent(interval=None)
#                 mem_after = psutil.virtual_memory().used

#                 inference_time = end_time - start_time
#                 cpu_diff = abs(cpu_after - cpu_before)
#                 mem_diff = (mem_after - mem_before) / (1024 * 1024)  # MB
#                 tokens = len(default_tokenizer.tokenize(answer))
#                 emissions_data = tracker.final_emissions_data
#                 energy_kwh = emissions_data.energy_consumed if emissions_data else 0.0

#                 try:
#                     _, _, F1 = bert_score_fn([answer], [reference], lang="en", verbose=False)
#                     f1_score = round(F1[0].item(), 4)
#                 except Exception as e:
#                     print(f"⚠️ BERTScore failed: {e}")
#                     f1_score = 0.0

#                 # Update Prometheus
#                 bert_score_metric.labels(model=model_name, query=query_id).set(f1_score)
#                 token_count_metric.labels(model=model_name, query=query_id).set(tokens)
#                 energy_metric.labels(model=model_name, query=query_id).set(energy_kwh)
#                 cpu_metric.labels(model=model_name, query=query_id).set(cpu_diff)
#                 memory_metric.labels(model=model_name, query=query_id).set(mem_diff)
#                 inference_time_metric.labels(model=model_name, query=query_id).set(inference_time)

#                 # CSV logging
#                 writer.writerow([q_num, question, reference, model_name, answer, f1_score, tokens, energy_kwh, cpu_diff, mem_diff, inference_time])
#                 file.flush()

#                 print(f"✅ {model_name}: Energy={energy_kwh:.6f} kWh | Tokens={tokens} | Time={inference_time:.3f}s")

#             except Exception as e:
#                 print(f"❌ Error with {model_name}: {e}")



import json
import time
import psutil
import csv
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer
from bert_score import score as bert_score_fn
from models.model_interface import get_answer
from core.config import PHI, LLAMA, GEMMA

from prometheus_client import start_http_server, Gauge

# Start Prometheus metrics server
start_http_server(8000)

# Define metrics with labels for model and query number
bert_score_metric = Gauge('bert_score_f1', 'BERT F1 Score', ['model', 'query'])
token_count_metric = Gauge('token_count', 'Number of tokens in the generated answer', ['model', 'query'])
input_token_count_metric = Gauge('input_token_count', 'Number of tokens in the input question', ['model', 'query'])
energy_metric = Gauge('energy_consumption_kwh', 'Energy consumption in kWh', ['model', 'query'])
cpu_metric = Gauge('cpu_used_percent', 'CPU usage during inference (%)', ['model', 'query'])
memory_metric = Gauge('memory_used_mb', 'Memory usage during inference in MB', ['model', 'query'])
inference_time_metric = Gauge('inference_time_seconds', 'Inference time in seconds', ['model', 'query'])

# Load dataset
with open("sample_qa_100.json", "r", encoding="utf-8") as f:
    qa_data = dict(list(json.load(f).items())[:100])

print(f"📘 Loaded {len(qa_data)} QA pairs.")

# Define model registry and tokenizers
MODELS = {
    "Phi": {"id": PHI, "tokenizer": AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)},
    "LLaMA": {"id": LLAMA, "tokenizer": AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)},
    "Gemma": {"id": GEMMA, "tokenizer": AutoTokenizer.from_pretrained("google/gemma-7b", trust_remote_code=True)},
}

# Prepare CSV output
csv_file = "experiment_metrics_output.csv"
with open(csv_file, mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Query#", "Question", "Reference", "Model", "Answer",
        "BERT F1", "Input Tokens", "Output Tokens", 
        "Energy (kWh)", "CPU (%)", "Memory (MB)", "Time (s)"
    ])

    for q_num, (question, reference) in enumerate(qa_data.items(), start=1):
        print(f"\n🟦 [{q_num}] Processing: {question[:70]}...")
        query_id = str(q_num)

        for model_name, model_info in MODELS.items():
            model_id = model_info["id"]
            tokenizer = model_info["tokenizer"]

            try:
                print(f"🔸 Using model: {model_name}")
                
                tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
                cpu_before = psutil.cpu_percent(interval=None)
                mem_before = psutil.virtual_memory().used

                tracker.start()
                start_time = time.time()
                answer = get_answer(model_id, question)
                end_time = time.time()
                tracker.stop()

                cpu_after = psutil.cpu_percent(interval=None)
                mem_after = psutil.virtual_memory().used

                inference_time = end_time - start_time
                cpu_diff = abs(cpu_after - cpu_before)
                mem_diff = (mem_after - mem_before) / (1024 * 1024)  # MB

                # Count tokens
                input_tokens = len(tokenizer.tokenize(question))
                output_tokens = len(tokenizer.tokenize(answer))

                emissions_data = tracker.final_emissions_data
                energy_kwh = emissions_data.energy_consumed if emissions_data else 0.0

                try:
                    _, _, F1 = bert_score_fn([answer], [reference], lang="en", verbose=False)
                    f1_score = round(F1[0].item(), 4)
                except Exception as e:
                    print(f"⚠️ BERTScore failed: {e}")
                    f1_score = 0.0

                # Update Prometheus metrics
                bert_score_metric.labels(model=model_name, query=query_id).set(f1_score)
                token_count_metric.labels(model=model_name, query=query_id).set(output_tokens)
                input_token_count_metric.labels(model=model_name, query=query_id).set(input_tokens)
                energy_metric.labels(model=model_name, query=query_id).set(energy_kwh)
                cpu_metric.labels(model=model_name, query=query_id).set(cpu_diff)
                memory_metric.labels(model=model_name, query=query_id).set(mem_diff)
                inference_time_metric.labels(model=model_name, query=query_id).set(inference_time)

                # CSV logging
                writer.writerow([
                    q_num, question, reference, model_name, answer,
                    f1_score, input_tokens, output_tokens,
                    energy_kwh, cpu_diff, mem_diff, inference_time
                ])
                file.flush()

                print(f"✅ {model_name}: Input={input_tokens} | Output={output_tokens} | Energy={energy_kwh:.6f} kWh | Time={inference_time:.3f}s")

            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
