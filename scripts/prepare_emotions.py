"""Prepare Emotions dataset: parse txt files → pool and validation CSVs.

Multiclass task (6 classes):
    sadness=0  anger=1  fear=2  joy=3  love=4  surprise=5

Run from the repo root:
    python scripts/prepare_emotions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DATASET_PATHS

EMOTIONS_DIR = str(DATASET_PATHS.emotions_raw_dir)
OUTPUT_DIR = str(DATASET_PATHS.output_dir)

LABEL_MAP = {
    "sadness": 0,
    "anger": 1,
    "fear": 2,
    "joy": 3,
    "love": 4,
    "surprise": 5,
}


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
    pool_df["label"] = pool_df["emotion"].map(LABEL_MAP)
    val_df["label"] = val_df["emotion"].map(LABEL_MAP)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pool_path = os.path.join(OUTPUT_DIR, "emotions_pool.csv")
    val_path = os.path.join(OUTPUT_DIR, "emotions_validation.csv")
    pool_df.to_csv(pool_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Pool       → {pool_path}: {len(pool_df)} samples")
    print(f"Validation → {val_path}: {len(val_df)} samples")
    print(f"\nLabel distribution (pool):\n{pool_df['emotion'].value_counts()}")


if __name__ == "__main__":
    main()
