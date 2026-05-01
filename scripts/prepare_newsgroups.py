"""Prepare 20 Newsgroups data: parse local txt files → pool and validation CSVs.

Binary task: science-related posts (sci.*) = 1, all others = 0.

Run from the repo root:
    python scripts/prepare_newsgroups.py
"""

import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

NEWSGROUPS_DIR = "20_newsgroups"
OUTPUT_DIR = "data_use_cases"
RANDOM_SEED = 42
VAL_FRACTION = 0.2


def parse_newsgroup_file(filepath: str, newsgroup: str) -> list[dict]:
    with open(filepath, encoding="latin-1") as f:
        content = f.read()

    # Articles begin with 'From:' at the start of a line.
    raw_articles = re.split(r"(?=^From:)", content, flags=re.MULTILINE)

    records = []
    for article in raw_articles:
        article = article.strip()
        if not article:
            continue

        # Extract Subject header (first occurrence).
        subject_m = re.search(r"^Subject:\s*(.+)", article, re.MULTILINE)
        subject = subject_m.group(1).strip() if subject_m else ""

        # Body is everything after the first blank line (header/body separator).
        parts = re.split(r"\n\n", article, maxsplit=1)
        body = parts[1].strip() if len(parts) > 1 else article.strip()

        records.append({"newsgroup": newsgroup, "title": subject, "text": body})

    return records


def main() -> None:
    if not os.path.isdir(NEWSGROUPS_DIR):
        sys.exit(f"ERROR: '{NEWSGROUPS_DIR}' directory not found. Run from repo root.")

    all_records: list[dict] = []
    for fname in sorted(os.listdir(NEWSGROUPS_DIR)):
        if not fname.endswith(".txt"):
            continue
        newsgroup = fname[: -len(".txt")]
        filepath = os.path.join(NEWSGROUPS_DIR, fname)
        records = parse_newsgroup_file(filepath, newsgroup)
        all_records.extend(records)
        print(f"  {newsgroup}: {len(records)} articles")

    df = pd.DataFrame(all_records).reset_index(drop=True)
    df["id"] = df.index.astype(str)
    # sci.* = science-related = positive class
    df["label"] = df["newsgroup"].apply(lambda ng: 1 if ng.startswith("sci.") else 0)

    pool_df, val_df = train_test_split(
        df, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=df["label"]
    )
    pool_df = pool_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pool_path = os.path.join(OUTPUT_DIR, "newsgroups_pool.csv")
    val_path = os.path.join(OUTPUT_DIR, "newsgroups_validation.csv")
    pool_df.to_csv(pool_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nPool    → {pool_path}: {len(pool_df)} samples, {pool_df['label'].sum()} positive")
    print(f"Validation → {val_path}: {len(val_df)} samples, {val_df['label'].sum()} positive")


if __name__ == "__main__":
    main()
