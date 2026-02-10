# asc_format_option_a.py
# Converts SemEval-style CSV with a column like:
# "[{'@category': 'food', '@polarity': 'positive'}, ...]"
# into Option A ASC format:
# id,sentence,aspect_category,sentiment   (sentiment ∈ {positive,negative,neutral})

import ast
import pandas as pd

INPUT_CSV  = "test.csv"   # change if needed
OUTPUT_CSV = "testV2.csv"

SRC_TEXT_COL = "text"
SRC_ASPECTS_COL = "aspectCategories.aspectCategory"

def parse_aspect_list(x):
    """
    Parses strings like:
    "[{'@category': 'food', '@polarity': 'positive'}, {'@category': 'service', '@polarity': 'negative'}]"
    Returns a list of dicts; returns [] on failure.
    """
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
        return []
    except Exception:
        return []

def norm_sentiment(p):
    p = "" if p is None else str(p).strip().lower()
    if p in {"positive", "negative", "neutral"}:
        return p
    # common aliases just in case
    alias = {"pos": "positive", "neg": "negative", "neu": "neutral"}
    return alias.get(p, None)

def main():
    df = pd.read_csv(INPUT_CSV)

    out_rows = []
    for i, row in df.iterrows():
        sentence = "" if pd.isna(row.get(SRC_TEXT_COL)) else str(row.get(SRC_TEXT_COL)).strip()
        aspects = parse_aspect_list(row.get(SRC_ASPECTS_COL))

        for j, a in enumerate(aspects):
            if not isinstance(a, dict):
                continue

            cat = a.get("@category") or a.get("category") or a.get("aspect")
            pol = a.get("@polarity") or a.get("polarity") or a.get("sentiment")

            aspect_category = "" if cat is None else str(cat).strip().lower()
            sentiment = norm_sentiment(pol)

            # keep only Option A labels
            if not sentence or not aspect_category or sentiment is None:
                continue

            out_rows.append({
                "id": f"{i}_{j}",                # unique per (sentence, aspect) pair
                "sentence": sentence,
                "aspect_category": aspect_category,
                "sentiment": sentiment
            })

    out_df = pd.DataFrame(out_rows, columns=["id", "sentence", "aspect_category", "sentiment"])
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(out_df)} rows to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
