# asc_train_eval.py
# Version-safe (no Trainer/TrainingArguments). Uses GPU if available.
# Input CSV columns: id,sentence,aspect_category,sentiment

import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

TRAIN_CSV = "trainV2.csv"
TEST_CSV  = "testV2.csv"

MODEL_NAME = "roberta-base"
MAX_LEN = 192
BATCH = 16
EPOCHS = 3
LR = 2e-5
SEED = 42

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ASCDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.sents = df["sentence"].astype(str).tolist()
        self.asps  = df["aspect_category"].astype(str).tolist()
        self.y     = df["sentiment"].astype(str).str.lower().map(LABEL2ID).astype(int).tolist()
        self.tok = tokenizer

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        enc = self.tok(
            self.sents[i],
            self.asps[i],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.y[i], dtype=torch.long)
        return item

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=1).detach().cpu().numpy().tolist()
        labels = batch["labels"].detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
    }

def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available. Using CPU.")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    for df in (train_df, test_df):
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
        df["aspect_category"] = df["aspect_category"].astype(str).str.strip().str.lower()
        df.dropna(subset=["sentence", "aspect_category", "sentiment"], inplace=True)

    train_df = train_df[train_df["sentiment"].isin(LABEL2ID)].reset_index(drop=True)
    test_df  = test_df[test_df["sentiment"].isin(LABEL2ID)].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_loader = DataLoader(ASCDataset(train_df, tok), batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(ASCDataset(test_df, tok), batch_size=BATCH, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    global_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        epoch_start = time.time()
        n_steps = len(train_loader)

        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()

            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            total_loss += loss.item()

            # progress every 100 steps (adjust if you want)
            if step % 100 == 0 or step == n_steps:
                elapsed = time.time() - epoch_start
                sec_per_step = elapsed / max(1, step)
                eta = (n_steps - step) * sec_per_step
                print(
                    f"Epoch {epoch}/{EPOCHS} | step {step}/{n_steps} | "
                    f"loss={loss.item():.4f} | elapsed={fmt_time(elapsed)} | eta={fmt_time(eta)}"
                )

        metrics = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{EPOCHS} DONE | "
            f"train_loss={total_loss/len(train_loader):.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"f1_micro={metrics['f1_micro']:.4f} | "
            f"time={fmt_time(epoch_time)}"
        )

    total_time = time.time() - global_start
    print("\nFinal TEST:", evaluate(model, test_loader, device))
    print("Total time:", fmt_time(total_time))

if __name__ == "__main__":
    main()
