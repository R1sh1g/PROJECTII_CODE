# src/data_loader.py

import pandas as pd
from collections import defaultdict
from .config import SENTIMENT_MAP

def load_csv(path: str):
    df = pd.read_csv(path)

    # normalize column names if needed
    cols = {c.lower(): c for c in df.columns}
    # expected: sentence/aspect/sentiment
    sentence_col = cols.get("sentence", None)
    aspect_col = cols.get("aspect", None)
    sentiment_col = cols.get("sentiment", None)

    if sentence_col is None or aspect_col is None or sentiment_col is None:
        raise ValueError(f"CSV must contain columns: sentence, aspect, sentiment. Found: {list(df.columns)}")

    data = []
    for _, row in df.iterrows():
        sent = str(row[sentence_col])
        asp = str(row[aspect_col]).strip().lower()
        pol_raw = row[sentiment_col]
        if isinstance(pol_raw, str):
            pol_key = pol_raw.strip().lower()
            if pol_key not in SENTIMENT_MAP:
                continue
            pol = SENTIMENT_MAP[pol_key]
        else:
            pol = int(pol_raw)

        data.append({"sentence": sent, "aspect": asp, "label": pol})
    return data

def build_index(data):
    """
    index[aspect][label] = list of examples
    """
    index = defaultdict(lambda: defaultdict(list))
    for ex in data:
        index[ex["aspect"]][ex["label"]].append(ex)
    return index

def valid_aspects(index, n_way, k_shot, q_query):
    """
    Keep aspects that have at least (k_shot + q_query) examples for each required label.
    For sentiment, labels are [0..n_way-1].
    """
    good = []
    for asp, lab_map in index.items():
        ok = True
        for lab in range(n_way):
            if len(lab_map.get(lab, [])) < (k_shot + q_query):
                ok = False
                break
        if ok:
            good.append(asp)
    return good
