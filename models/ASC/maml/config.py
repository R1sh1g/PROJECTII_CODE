# src/config.py

MODEL_NAME = "distilbert-base-uncased"   # use same as baseline for comparability

MAX_LEN = 128

# Few-shot episode config
N_WAY = 3              # for sentiment (neg/neu/pos) typically 3
K_SHOT = 4             # support per class
Q_QUERY = 8            # query per class

TASKS_PER_META_BATCH = 4

# MAML
INNER_LR = 1e-3
INNER_STEPS = 3        # 1-5 typical
META_LR = 2e-5
EPOCHS = 3

DEVICE = "cuda"  # or "cpu"

SENTIMENT_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "neg": 0, "neu": 1, "pos": 2
}
