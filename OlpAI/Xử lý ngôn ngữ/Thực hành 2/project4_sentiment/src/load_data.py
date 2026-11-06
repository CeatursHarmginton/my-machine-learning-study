"""
Load Stanford Large Movie Review Dataset (aclImdb) into pandas DataFrames.

Expected folder layout:
    aclImdb/
      train/pos/*.txt, train/neg/*.txt
      test/pos/*.txt,  test/neg/*.txt
"""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def default_data_dir() -> Path:
    """Resolve aclImdb relative to this project (project4_sentiment/data/aclImdb)."""
    return Path(__file__).resolve().parents[1] / "data" / "aclImdb"


def read_reviews(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Read all labeled reviews for one split ('train' or 'test').

    Labels: neg -> 0, pos -> 1.
    """
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    rows = []
    for label_name, label in [("neg", 0), ("pos", 1)]:
        folder = data_dir / split / label_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for path in folder.glob("*.txt"):
            rows.append(
                {
                    "text": path.read_text(encoding="utf-8"),
                    "label": label,
                    "file": path.name,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def clean_review(text: str) -> str:
    """
    Light cleaning only: decode HTML entities, remove <br> tags, normalize spaces.

    Do not strip stopwords here — words like 'not' / 'never' matter for sentiment.
    Lowercasing is left to the vectorizer.
    """
    text = unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_splits(
    data_dir: Path | None = None,
    valid_size: float = 0.20,
    random_state: int = 42,
    clean: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load official train/test, carve validation from train only, optionally clean text.

    Returns
    -------
    train, valid, test
    """
    train_full = read_reviews("train", data_dir=data_dir)
    test = read_reviews("test", data_dir=data_dir)

    train, valid = train_test_split(
        train_full,
        test_size=valid_size,
        random_state=random_state,
        stratify=train_full["label"],
    )

    # Avoid SettingWithCopyWarning when mapping cleaned text
    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    if clean:
        for df in (train, valid, test):
            df["text"] = df["text"].map(clean_review)

    return train, valid, test


if __name__ == "__main__":
    train_df, valid_df, test_df = load_splits()
    print("Train:", len(train_df), "Valid:", len(valid_df), "Test:", len(test_df))
    print("Train labels:\n", train_df["label"].value_counts().sort_index())
    print("Test labels:\n", test_df["label"].value_counts().sort_index())
