# src/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class ACSCTransformer(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token representation (first token)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits
