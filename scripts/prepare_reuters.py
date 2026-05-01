"""Prepare Reuters-21578 data: ModApte split CSVs → pool and validation CSVs.

Binary task: articles tagged with the 'earn' topic = 1, all others = 0.

Run from the repo root:
    python scripts/prepare_reuters.py
"""

import os
import re
import sys

import pandas as pd

REUTERS_DIR = "reuters-21578"
OUTPUT_DIR = "data_use_cases"


def parse_topics(s: str) -> list[str]:
    """Parse numpy-array repr like \"['earn' 'grain']\" into a Python list."""
    return re.findall(r"'([^']+)'", str(s))


def process_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["topics_list"] = df["topics"].apply(parse_topics)
    df["label"] = df["topics_list"].apply(lambda t: 1 if "earn" in t else 0)

    # Clean up id column
    if "new_id" in df.columns:
        df["id"] = df["new_id"].astype(str).str.strip('"')
    else:
        df["id"] = df.index.astype(str)

    # Fill missing titles
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    return df[["id", "title", "text", "label"]].copy()


def main() -> None:
    train_path = os.path.join(REUTERS_DIR, "ModApte_train.csv")
    test_path = os.path.join(REUTERS_DIR, "ModApte_test.csv")
    for p in (train_path, test_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: '{p}' not found. Run from repo root.")

    pool_df = process_split(train_path)
    val_df = process_split(test_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pool_path = os.path.join(OUTPUT_DIR, "reuters_pool.csv")
    val_path = os.path.join(OUTPUT_DIR, "reuters_validation.csv")
    pool_df.to_csv(pool_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Pool    → {pool_path}: {len(pool_df)} samples, {pool_df['label'].sum()} positive (earn)")
    print(f"Validation → {val_path}: {len(val_df)} samples, {val_df['label'].sum()} positive (earn)")


if __name__ == "__main__":
    main()
