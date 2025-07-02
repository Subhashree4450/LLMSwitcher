# from transformers import AutoTokenizer

# # Download and save locally
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer.save_pretrained("./bert-base-uncased")



from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

# Load the token from .env file
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

tokenizers_to_download = {
    "phi": "microsoft/phi-2",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "gemma": "google/gemma-7b"
}

save_dir = "./tokenizers"

for name, model_id in tokenizers_to_download.items():
    local_path = os.path.join(save_dir, name)
    os.makedirs(local_path, exist_ok=True)

    print(f"🔽 Downloading tokenizer for {model_id}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_auth_token=HUGGINGFACE_TOKEN if "llama" in model_id else None
        )
        tokenizer.save_pretrained(local_path)
        print(f"✅ Saved to {local_path}")
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")

