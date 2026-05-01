"""Prepare CMU WebKB data: download + parse → pool and validation CSVs.

Binary task: student pages = 1, all other categories = 0.

The dataset is hosted at:
  http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz

Run from the repo root:
    python scripts/prepare_webkb.py

If the download fails, manually download webkb-data.gtar.gz and place it (or
the extracted 'webkb/' directory) in the repo root, then re-run this script.
"""

import csv
import io
import os
import re
import sys
import tarfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "data_use_cases"
LOCAL_ARCHIVE = "webkb-data.gtar.gz"
LOCAL_DIR = "webkb"
RANDOM_SEED = 42
VAL_FRACTION = 0.2

DOWNLOAD_URLS = [
    "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz",
]

CATEGORIES = {"student", "faculty", "staff", "department", "course", "project", "other"}
POSITIVE_CATEGORY = "student"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def extract_text(html_bytes: bytes) -> str:
    try:
        html = html_bytes.decode("latin-1")
    except Exception:
        html = html_bytes.decode("utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = parser.get_text()
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def records_from_dir(webkb_root: str) -> list[dict]:
    records: list[dict] = []
    root = Path(webkb_root)
    for university_dir in sorted(root.iterdir()):
        if not university_dir.is_dir():
            continue
        for category_dir in sorted(university_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name.lower()
            if category not in CATEGORIES:
                continue
            for page_file in sorted(category_dir.iterdir()):
                if not page_file.is_file():
                    continue
                try:
                    text = extract_text(page_file.read_bytes())
                except Exception:
                    continue
                if not text.strip():
                    continue
                records.append(
                    {
                        "university": university_dir.name,
                        "category": category,
                        "filename": page_file.name,
                        "text": text,
                    }
                )
    return records


def records_from_archive(archive_path: str) -> list[dict]:
    records: list[dict] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            # Structure: webkb/<category>/<university>/<filename>
            if len(parts) < 4:
                continue
            category = parts[1].lower()
            if category not in CATEGORIES:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text = extract_text(f.read())
            if not text.strip():
                continue
            records.append(
                {
                    "university": parts[2],
                    "category": category,
                    "filename": parts[-1],
                    "text": text,
                }
            )
    return records


def try_download(url: str, dest: str) -> bool:
    print(f"Downloading from {url} …")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
        return True
    except Exception as exc:
        print(f"  Failed: {exc}")
        return False


def main() -> None:
    records: list[dict] = []

    # 1. Try local extracted directory
    if os.path.isdir(LOCAL_DIR):
        print(f"Found local directory '{LOCAL_DIR}', reading …")
        records = records_from_dir(LOCAL_DIR)

    # 2. Try local archive
    elif os.path.exists(LOCAL_ARCHIVE):
        print(f"Found local archive '{LOCAL_ARCHIVE}', extracting …")
        records = records_from_archive(LOCAL_ARCHIVE)

    # 3. Try downloading
    else:
        for url in DOWNLOAD_URLS:
            if try_download(url, LOCAL_ARCHIVE):
                records = records_from_archive(LOCAL_ARCHIVE)
                break

    if not records:
        print(
            "\nERROR: Could not obtain WebKB data.\n"
            "Manual steps:\n"
            "  1. Download from:\n"
            "     http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz\n"
            "  2. Place 'webkb-data.gtar.gz' in the repo root, then re-run:\n"
            "     python scripts/prepare_webkb.py\n"
        )
        sys.exit(1)

    df = pd.DataFrame(records).reset_index(drop=True)
    df = df.dropna(subset=["category", "text"]).reset_index(drop=True)
    df["id"] = df.index.astype(str)
    df["label"] = (df["category"] == POSITIVE_CATEGORY).astype(int)

    pool_df, val_df = train_test_split(
        df, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=df["label"]
    )
    pool_df = pool_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    pool_df["label"] = pool_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pool_path = os.path.join(OUTPUT_DIR, "webkb_pool.csv")
    val_path = os.path.join(OUTPUT_DIR, "webkb_validation.csv")
    # QUOTE_ALL prevents HTTP-header content from breaking CSV row boundaries
    pool_df.to_csv(pool_path, index=False, quoting=csv.QUOTE_ALL)
    val_df.to_csv(val_path, index=False, quoting=csv.QUOTE_ALL)

    print(f"\nPool    → {pool_path}: {len(pool_df)} samples, {pool_df['label'].sum()} positive (student)")
    print(f"Validation → {val_path}: {len(val_df)} samples, {val_df['label'].sum()} positive (student)")
    print(f"\nCategory distribution:\n{df['category'].value_counts()}")


if __name__ == "__main__":
    main()
