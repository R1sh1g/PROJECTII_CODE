# src/utils.py

import torch
from transformers import AutoTokenizer
from .config import MAX_LEN

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def encode_batch(tokenizer, examples, device):
    """
    examples: list of dicts with keys sentence, aspect, label
    returns tensors input_ids, attention_mask, labels
    """
    texts = [ex["sentence"] for ex in examples]
    aspects = [ex["aspect"] for ex in examples]
    enc = tokenizer(
        texts,
        aspects,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), labels.to(device)
