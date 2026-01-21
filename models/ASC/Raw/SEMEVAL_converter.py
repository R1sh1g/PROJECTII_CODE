# make_option_a_from_term_level.py
# Input columns (example):
# id,Sentence,Aspect Term,polarity,from,to,mapped_category
# Output (Option A):
# id,sentence,aspect_category,sentiment

import pandas as pd

INPUT_CSV = "restaurants-trial_mapped.csv"          # change if needed
OUTPUT_CSV = "restaurants-trial_v3.csv"  # change if needed

def main():
    df = pd.read_csv(INPUT_CSV)

    required = {"id", "Sentence", "polarity", "mapped_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.loc[:, ["id", "Sentence", "mapped_category", "polarity"]].copy()

    # Rename to Option A schema
    out.rename(
        columns={
            "Sentence": "sentence",
            "mapped_category": "aspect_category",
            "polarity": "sentiment",
        },
        inplace=True,
    )

    # Normalize labels (optional but recommended)
    out["aspect_category"] = out["aspect_category"].astype(str).str.strip().str.lower()
    out["sentiment"] = out["sentiment"].astype(str).str.strip().str.lower()

    # Keep only Option A sentiments
    out = out[out["sentiment"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
