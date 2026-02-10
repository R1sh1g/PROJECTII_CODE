# reviewOps/backend/models/asc_infer.py

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ACSCTransformer(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits


class ASCInfer:
    def __init__(self, ckpt_path: str, model_name: str = "roberta-base", max_len: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_len = max_len
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ACSCTransformer(self.model_name, 3).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, sentence: str, aspect: str):
        enc = self.tokenizer(
            [sentence],
            [aspect],
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids, attn)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))

        return {
            "sentiment": self.id2label[pred],
            "confidence": float(probs[pred]),
            "probs": probs.tolist(),
        }


def load_asc():
    # From reviewOps/backend -> go up 2 levels to PROJECTII_CODE root
    ckpt_path = "C:\\Users\\Rishi\\PROJECTII_CODE\\models\\maml_acsc_epoch4.pt"
    return ASCInfer(ckpt_path=ckpt_path, model_name="roberta-base", max_len=128)
