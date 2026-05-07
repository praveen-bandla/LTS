"""Prepare Reuters-21578 data: ModApte split CSVs → pool and validation CSVs.

Multiclass task (9 classes — top-8 topics + other):
    earn=0  acq=1  money-fx=2  grain=3  crude=4  trade=5  interest=6  wheat=7  other=8

Label assignment: first match from TOP_TOPICS list; articles with none → other (8).

Run from the repo root:
    python scripts/prepare_reuters.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DATASET_PATHS

REUTERS_DIR = str(DATASET_PATHS.reuters_raw_dir)
OUTPUT_DIR = str(DATASET_PATHS.output_dir)

TOP_TOPICS = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "wheat"]
OTHER_LABEL = len(TOP_TOPICS)  # 8
TOPIC_TO_LABEL = {t: i for i, t in enumerate(TOP_TOPICS)}


def parse_topics(s: str) -> list[str]:
    """Parse numpy-array repr like \"['earn' 'grain']\" into a Python list."""
    return re.findall(r"'([^']+)'", str(s))


def assign_label(topics_list: list[str]) -> int:
    for topic in TOP_TOPICS:
        if topic in topics_list:
            return TOPIC_TO_LABEL[topic]
    return OTHER_LABEL


def process_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["topics_list"] = df["topics"].apply(parse_topics)
    df["label"] = df["topics_list"].apply(assign_label)

    if "new_id" in df.columns:
        df["id"] = df["new_id"].astype(str).str.strip('"')
    else:
        df["id"] = df.index.astype(str)

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

    print(f"Pool       → {pool_path}: {len(pool_df)} samples")
    print(f"Validation → {val_path}: {len(val_df)} samples")
    print(f"\nLabel distribution (pool):\n{pool_df['label'].value_counts().sort_index()}")
    topic_names = TOP_TOPICS + ["other"]
    print(f"\nClass names: {dict(enumerate(topic_names))}")


if __name__ == "__main__":
    main()
