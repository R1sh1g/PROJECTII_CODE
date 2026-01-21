# acd.py
# Single-file ACD (Aspect Category Detection) trainer + predictor with HARD-CODED PATHS.
# Dataset format expected (as per your screenshot):
#   Columns: id, Sentence, labels
#   labels example: "['service', 'staff']"
#
# Run:
#   python acd.py
#
# Outputs:
#   outputs/acd_model/  (model + tokenizer + label_names.json)
#   outputs/train_used.csv, outputs/val_used.csv (if VAL_CSV_PATH is None)
#   outputs/test_with_predictions.csv (if TEST_CSV_PATH exists)

import os
import re
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

# ===============================
# HARD-CODED CONFIGURATION
# ===============================

# ---- DATA PATHS ----
TRAIN_CSV_PATH = "C:/Users/Rishi/PROJECTII_CODE/data/raw/MAMS-for-ABSA-master/MAMS-for-ABSA-master/data/train_format.csv"
VAL_CSV_PATH   = None               # set to "data/val.csv" if you have one
TEST_CSV_PATH  = "C:/Users/Rishi/PROJECTII_CODE/data/raw/MAMS-for-ABSA-master/MAMS-for-ABSA-master/data/test_format.csv"    # set to None if you don't want predictions

# ---- OUTPUT ----
OUTPUT_DIR = "outputs/acd_model"

# ---- DATASET COLUMNS ----
TEXT_COLUMN  = "Sentence"
LABEL_COLUMN = "labels"

# ---- MODEL ----
MODEL_NAME = "roberta-base"

# ---- TRAINING PARAMS ----
MAX_LENGTH = 192
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
SEED = 42
VAL_SPLIT = 0.1  # used only if VAL_CSV_PATH is None

# ---- PREDICTION ----
THRESHOLD = 0.5
PREDICTION_OUT_CSV = "outputs/test_with_predictions.csv"


# ----------------------------
# Parsing helpers
# ----------------------------

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


def parse_label_list(val) -> List[str]:
    """
    Robustly parse labels stored as:
      "['service', 'staff']"
      '["service","staff"]'
      "service" (fallback)
      "" or NaN -> []
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return []

    # bracketed list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s2 = s.strip("()")
        # Normalize single quotes to double quotes for JSON
        s2 = re.sub(r"'", '"', s2)
        try:
            out = json.loads(s2)
            if isinstance(out, list):
                return [str(x).strip() for x in out if str(x).strip() != ""]
        except Exception:
            # Fallback: split by commas
            inner = s.strip("[]() ").strip()
            if not inner:
                return []
            parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
            return [p for p in parts if p]

    # Single label fallback
    return [s]


def load_df(path: str, text_col: str, labels_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Found: {list(df.columns)}")
    if labels_col not in df.columns:
        raise ValueError(f"Missing labels column '{labels_col}'. Found: {list(df.columns)}")

    # Keep id if exists, plus required columns
    keep = []
    if "id" in df.columns:
        keep.append("id")
    keep += [text_col, labels_col]
    df = df[keep].copy()

    df[text_col] = df[text_col].astype(str)
    return df


def build_label_matrix(
    df: pd.DataFrame,
    text_col: str,
    labels_col: str,
    mlb: Optional[MultiLabelBinarizer] = None,
) -> Tuple[List[str], np.ndarray, MultiLabelBinarizer, List[str]]:
    texts = df[text_col].astype(str).tolist()
    y_list = [parse_label_list(x) for x in df[labels_col].tolist()]

    if mlb is None:
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y_list)
    else:
        Y = mlb.transform(y_list)

    label_names = list(mlb.classes_)
    return texts, Y.astype(np.float32), mlb, label_names


# ----------------------------
# Torch Dataset
# ----------------------------

class ACDDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


# ----------------------------
# Metrics (multi-label)
# ----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    p_micro = precision_score(labels, preds, average="micro", zero_division=0)
    r_micro = recall_score(labels, preds, average="micro", zero_division=0)
    subset_acc = accuracy_score(labels, preds)  # exact match

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "subset_accuracy": subset_acc,
    }


# ----------------------------
# Prediction helpers
# ----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@torch.no_grad()
def predict_text(
    model,
    tokenizer,
    label_names: List[str],
    text: str,
    max_length: int,
    threshold: float,
):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits.detach().cpu().numpy()[0]
    probs = sigmoid(logits)

    picked = [(label_names[i], float(probs[i])) for i in range(len(label_names)) if probs[i] >= threshold]
    picked.sort(key=lambda x: x[1], reverse=True)

    # If nothing crosses threshold, return top-1
    if not picked:
        i = int(np.argmax(probs))
        picked = [(label_names[i], float(probs[i]))]

    return picked


# ----------------------------
# Train
# ----------------------------

def train_acd():
    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PREDICTION_OUT_CSV), exist_ok=True)

    df = load_df(TRAIN_CSV_PATH, TEXT_COLUMN, LABEL_COLUMN)

    if VAL_CSV_PATH:
        train_df = df
        val_df = load_df(VAL_CSV_PATH, TEXT_COLUMN, LABEL_COLUMN)
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=VAL_SPLIT,
            random_state=SEED,
            shuffle=True,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # Save the exact split used
        train_df.to_csv(os.path.join("outputs", "train_used.csv"), index=False)
        val_df.to_csv(os.path.join("outputs", "val_used.csv"), index=False)

    train_texts, train_Y, mlb, label_names = build_label_matrix(
        train_df, TEXT_COLUMN, LABEL_COLUMN, mlb=None
    )
    val_texts, val_Y, _, _ = build_label_matrix(
        val_df, TEXT_COLUMN, LABEL_COLUMN, mlb=mlb
    )

    num_labels = int(train_Y.shape[1])
    if num_labels == 0:
        raise ValueError("No labels found after parsing. Check your 'labels' column formatting.")

    # Save label names for inference
    with open(os.path.join(OUTPUT_DIR, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = ACDDataset(train_texts, train_Y, tokenizer, MAX_LENGTH)
    val_ds = ACDDataset(val_texts, val_Y, tokenizer, MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[DONE] ACD model saved to: {OUTPUT_DIR}")
    print(f"[INFO] Labels ({len(label_names)}): {label_names}")


# ----------------------------
# Predict on TEST_CSV_PATH (optional)
# ----------------------------

def predict_acd_on_test():
    if not TEST_CSV_PATH:
        print("[INFO] TEST_CSV_PATH is None. Skipping prediction.")
        return
    if not os.path.exists(TEST_CSV_PATH):
        print(f"[INFO] Test CSV not found at {TEST_CSV_PATH}. Skipping prediction.")
        return

    label_path = os.path.join(OUTPUT_DIR, "label_names.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing {label_path}. Train first.")

    with open(label_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    df = pd.read_csv(TEST_CSV_PATH)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing text column '{TEXT_COLUMN}' in {TEST_CSV_PATH}. Found: {list(df.columns)}")

    outputs = []
    for t in df[TEXT_COLUMN].astype(str).tolist():
        outputs.append(predict_text(model, tokenizer, label_names, t, MAX_LENGTH, THRESHOLD))

    df["predicted_aspects"] = [json.dumps(x) for x in outputs]
    df.to_csv(PREDICTION_OUT_CSV, index=False)
    print(f"[DONE] Saved predictions to: {PREDICTION_OUT_CSV}")


# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":
    train_acd()
    predict_acd_on_test()