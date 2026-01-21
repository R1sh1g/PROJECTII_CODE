# src/maml.py

import torch
import torch.nn.functional as F
from collections import OrderedDict

def forward_with_weights(model, input_ids, attention_mask, weights: OrderedDict):
    """
    Forward pass using a set of 'fast weights' for classifier and encoder.
    For practical simplicity, we only adapt the classifier weights first.
    (This is still few-shot adaptation; and it actually works well.)
    """
    out = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls = out.last_hidden_state[:, 0]

    W = weights["classifier.weight"]
    b = weights["classifier.bias"]
    logits = F.linear(cls, W, b)
    return logits

def get_adaptable_weights(model):
    """
    Start by adapting only classifier head for stability/speed.
    Later you can extend to encoder layers (last k layers).
    """
    return OrderedDict({
        "classifier.weight": model.classifier.weight,
        "classifier.bias": model.classifier.bias
    })

def inner_adapt(model, weights, support_batch, inner_lr, inner_steps):
    input_ids, attn, labels = support_batch

    fast = OrderedDict((k, v) for k, v in weights.items())

    for _ in range(inner_steps):
        logits = forward_with_weights(model, input_ids, attn, fast)
        loss = F.cross_entropy(logits, labels)

        grads = torch.autograd.grad(loss, fast.values(), create_graph=False)
        fast = OrderedDict(
            (k, v - inner_lr * g)
            for (k, v), g in zip(fast.items(), grads)
        )
    return fast

def query_loss(model, fast_weights, query_batch):
    input_ids, attn, labels = query_batch
    logits = forward_with_weights(model, input_ids, attn, fast_weights)
    return F.cross_entropy(logits, labels)
