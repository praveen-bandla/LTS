"""Prepare Emotions dataset: parse txt files → pool and validation CSVs.

Binary task: joy = 1, all other emotions = 0.

Run from the repo root:
    python scripts/prepare_emotions.py
"""

import os
import sys

import pandas as pd

EMOTIONS_DIR = "emotions_dataset_for_NLP"
OUTPUT_DIR = "data_use_cases"
POSITIVE_EMOTION = "joy"


def parse_emotions_file(filepath: str) -> list[dict]:
    records: list[dict] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(";", 1)
            if len(parts) != 2:
                continue
            text, emotion = parts[0].strip(), parts[1].strip()
            records.append({"text": text, "emotion": emotion})
    return records


def main() -> None:
    if not os.path.isdir(EMOTIONS_DIR):
        sys.exit(f"ERROR: '{EMOTIONS_DIR}' directory not found. Run from repo root.")

    pool_records = parse_emotions_file(os.path.join(EMOTIONS_DIR, "train.txt"))
    val_records = parse_emotions_file(os.path.join(EMOTIONS_DIR, "val.txt"))

    pool_df = pd.DataFrame(pool_records).reset_index(drop=True)
    val_df = pd.DataFrame(val_records).reset_index(drop=True)

    pool_df["id"] = pool_df.index.astype(str)
    val_df["id"] = val_df.index.astype(str)
    pool_df["label"] = (pool_df["emotion"] == POSITIVE_EMOTION).astype(int)
    val_df["label"] = (val_df["emotion"] == POSITIVE_EMOTION).astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pool_path = os.path.join(OUTPUT_DIR, "emotions_pool.csv")
    val_path = os.path.join(OUTPUT_DIR, "emotions_validation.csv")
    pool_df.to_csv(pool_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Pool    → {pool_path}: {len(pool_df)} samples, {pool_df['label'].sum()} positive ({POSITIVE_EMOTION})")
    print(f"Validation → {val_path}: {len(val_df)} samples, {val_df['label'].sum()} positive ({POSITIVE_EMOTION})")


if __name__ == "__main__":
    main()
