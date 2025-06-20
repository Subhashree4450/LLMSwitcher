from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def get_token_count(text):
    return len(tokenizer.encode(text, add_special_tokens=False))
