# ============================
# Cell 2 — Imports
# ============================

import os
#os.environ["HF_TOKEN"] = "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"

import random
from dataclasses import dataclass
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

# ============================
# Cell 3 — Seed + Device
# ============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================
# Cell 4 — Config
# ============================
@dataclass
class CFG:
    # Data
    train_csv: str = r"models/MAML Codes (ASC) ipynb/trainmaml.csv"  # change if needed
    test_csv: str  = r"models/MAML Codes (ASC) ipynb/testmaml.csv"   # change if needed

    # Model
    model_name: str = "roberta-base"
    max_len: int = 128
    num_labels: int = 3

    # Labels
    sentiment_map: dict = None
    id2label: dict = None

    # Episodes
    n_way: int = 3
    k_shot: int = 4
    q_query: int = 2

    # MAML
    inner_lr: float = 1e-2
    inner_steps: int = 3

    meta_lr: float = 2e-5
    tasks_per_meta_batch: int = 4

    epochs: int = 4
    meta_iters_per_epoch: int = 200

    # Eval
    eval_episodes: int = 200

    # Output
    ckpt_dir: str = r"C:\Users\Rishi\PROJECTII_CODE\models"


cfg = CFG(
    sentiment_map={"negative": 0, "neutral": 1, "positive": 2},
    id2label={0: "negative", 1: "neutral", 2: "positive"},
)

os.makedirs(cfg.ckpt_dir, exist_ok=True)
print("Checkpoint dir:", cfg.ckpt_dir)

# ============================
# Cell 5 — Load data
# ============================
train_df = pd.read_csv(cfg.train_csv)
test_df = pd.read_csv(cfg.test_csv)

req_cols = {"sentence", "aspect", "sentiment"}
assert req_cols.issubset(set(train_df.columns)), f"Train missing columns. Found: {train_df.columns}"
assert req_cols.issubset(set(test_df.columns)), f"Test missing columns. Found: {test_df.columns}"

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
train_df.head(3)

# ============================
# Prepare aspect list and index for episodes
# ============================
train_aspects = sorted(train_df["aspect"].unique())
test_aspects = sorted(test_df["aspect"].unique())

# Build train_index: aspect -> class -> list of (sentence, aspect, label)
train_index = defaultdict(lambda: defaultdict(list))
for _, row in train_df.iterrows():
    aspect = row["aspect"]
    label = cfg.sentiment_map[row["sentiment"]]
    train_index[aspect][label].append((row["sentence"], aspect, label))

# Build test_index: aspect -> class -> list of (sentence, aspect, label)
test_index = defaultdict(lambda: defaultdict(list))
for _, row in test_df.iterrows():
    aspect = row["aspect"]
    label = cfg.sentiment_map[row["sentiment"]]
    test_index[aspect][label].append((row["sentence"], aspect, label))


# ============================
# Cell 7 — Sample episode
# ============================

def sample_episode(idx, aspect: str):
    support, query = [], []
    need = cfg.k_shot + cfg.q_query

    for c in range(cfg.n_way):
        pool = idx[aspect][c]
        if len(pool) < need:
            return None, None  # not enough samples

        pool = pool.copy()
        random.shuffle(pool)
        support.extend(pool[: cfg.k_shot])
        query.extend(pool[cfg.k_shot : cfg.k_shot + cfg.q_query])

    random.shuffle(support)
    random.shuffle(query)
    return support, query


# ============================
# Cell 8 — Tokenizer + encode
# ============================

tok = AutoTokenizer.from_pretrained(cfg.model_name)


def encode_batch(examples):
    sents = [x[0] for x in examples]
    asps = [x[1] for x in examples]
    ys = torch.tensor([x[2] for x in examples], dtype=torch.long, device=device)

    enc = tok(
        sents,
        asps,
        padding=True,
        truncation=True,
        max_length=cfg.max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    return input_ids, attention_mask, ys



# ============================
# Cell 9 — Model (UPDATED)
# ============================
class ACSCTransformer(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [B, H]
        logits = self.classifier(cls)
        return logits


model = ACSCTransformer(cfg.model_name, cfg.num_labels).to(device)

# ✅ Freeze encoder to prevent huge second-order graphs & reduce VRAM
for p in model.encoder.parameters():
    p.requires_grad = False

print(model.__class__.__name__)
print("Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================
# Cell 10 — MAML helpers (UPDATED: FOMAML + encoder no_grad + AMP-friendly)
# ============================
from torch.cuda.amp import autocast, GradScaler

def get_adaptable_weights(model: ACSCTransformer):
    # only classifier is adapted
    return OrderedDict({
        "W": model.classifier.weight,
        "b": model.classifier.bias,
    })

def forward_with_weights(model: ACSCTransformer, input_ids, attention_mask, fast_weights: OrderedDict):
    # encoder is frozen -> no_grad + detach saves memory
    with torch.no_grad():
        out = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0].detach()
    logits = F.linear(cls, fast_weights["W"], fast_weights["b"])
    return logits

def inner_adapt(model: ACSCTransformer, support_examples):
    fast = get_adaptable_weights(model)

    for _ in range(cfg.inner_steps):
        input_ids, attn, y = encode_batch(support_examples)
        logits = forward_with_weights(model, input_ids, attn, fast)
        loss = F.cross_entropy(logits, y)

        # ✅ First-Order MAML: create_graph=False (prevents graph explosion)
        grads = torch.autograd.grad(loss, list(fast.values()), create_graph=False)
        fast = OrderedDict((k, v - cfg.inner_lr * g) for (k, v), g in zip(fast.items(), grads))

    return fast

def query_loss_and_acc(model: ACSCTransformer, fast_weights: OrderedDict, query_examples):
    input_ids, attn, y = encode_batch(query_examples)
    logits = forward_with_weights(model, input_ids, attn, fast_weights)
    loss = F.cross_entropy(logits, y)
    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    return loss, acc


# ============================
# Cell 11 — Train (UPDATED: optimize only classifier + AMP)
# ============================
# ✅ Optimizer only for trainable params (classifier)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg.meta_lr
)

scaler = GradScaler(enabled=(device.type == "cuda"))
history = []

# Optional: slightly safer CUDA behavior
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

for epoch in range(1, cfg.epochs + 1):
    model.train()

    pbar = tqdm(range(cfg.meta_iters_per_epoch), desc=f"Epoch {epoch}/{cfg.epochs}")
    for it in pbar:
        tasks = random.sample(train_aspects, k=min(cfg.tasks_per_meta_batch, len(train_aspects)))

        meta_loss = 0.0
        meta_acc = 0.0

        # meta batch over tasks
        for asp in tasks:
            support, query = sample_episode(train_index, asp)
            if support is None:
                continue
            fast = inner_adapt(model, support)
            qloss, qacc = query_loss_and_acc(model, fast, query)
            meta_loss = meta_loss + qloss
            meta_acc += qacc

        meta_loss = meta_loss / max(1, len(tasks))
        meta_acc = meta_acc / max(1, len(tasks))

        optimizer.zero_grad(set_to_none=True)

        # AMP backward
        if device.type == "cuda":
            scaler.scale(meta_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            meta_loss.backward()
            optimizer.step()

        history.append({
            "epoch": epoch,
            "iter": it,
            "meta_loss": float(meta_loss.item()),
            "meta_acc": float(meta_acc),
        })
        pbar.set_postfix({"loss": f"{meta_loss.item():.4f}", "acc": f"{meta_acc:.3f}"})

    ckpt_path = os.path.join(cfg.ckpt_dir, f"maml_acsc_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "history": history,
    }, ckpt_path)
    print("Saved checkpoint:", ckpt_path)

    
# ============================
# Cell 13 — Load ckpt
# ============================
ckpt_to_eval = os.path.join(cfg.ckpt_dir, f"maml_acsc_epoch{cfg.epochs}.pt")
ckpt = torch.load(ckpt_to_eval, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Loaded:", ckpt_to_eval)


# ============================
# Cell 14 — Evaluate
# ============================
losses, accs = [], []
for _ in tqdm(range(cfg.eval_episodes), desc="Eval episodes"):
    asp = random.choice(test_aspects)
    support, query = sample_episode(test_index, asp)
    if support is None:
        continue
    fast = inner_adapt(model, support)

    qloss, qacc = query_loss_and_acc(model, fast, query)
    losses.append(float(qloss.item()))
    accs.append(float(qacc))

print("\nEpisodic Test Results")
if len(losses) == 0:
    print("No valid eval episodes (not enough samples per class for k_shot+q_query).")
else:
    print("Mean loss:", sum(losses) / len(losses))
    print("Mean acc :", sum(accs) / len(accs))



def predict_sentiment(sentence: str, aspect: str, idx, support_k=None):
    """
    MAML-style prediction:
    - samples a small support set for the given aspect from `idx` (train_index or test_index)
    - adapts classifier weights using inner_adapt()
    - predicts on the given (sentence, aspect)
    """
    model.eval()
    support_k = support_k or cfg.k_shot

    support = []
    need = support_k
    for c in range(cfg.n_way):
        pool = idx[aspect][c]
        if len(pool) < need:
            return None, None, None
        pool = pool.copy()
        random.shuffle(pool)
        support.extend(pool[:need])

    fast = inner_adapt(model, support)

    enc = tok(
        [sentence], [aspect],
        padding=True, truncation=True,
        max_length=cfg.max_len,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = forward_with_weights(model, input_ids, attn, fast)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    return cfg.id2label[pred], float(probs[pred]), probs



demo_sentence = "The battery lasts long but the charger is terrible."
demo_aspect = "Battery"
label, conf, probs = predict_sentiment(demo_sentence, demo_aspect, test_index)
print("Sentence:", demo_sentence)
print("Aspect  :", demo_aspect)
print("Pred    :", label, "| conf:", conf)
print("Probs   :", probs)
