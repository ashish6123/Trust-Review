"""
Data Pipeline — Load, merge, harmonize, clean, and split all datasets.

Usage:
    cd "e:\Capstone Project"
    python -m training.data_pipeline
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `app` package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import (
    OPUS_CSV, AMAZON_CSV, YELP_TRAIN_CSV, YELP_TEST_CSV,
    PROCESSED_DIR, TRAIN_CSV, TEST_CSV,
)
from app.ml.preprocess import clean_text


def load_opus() -> pd.DataFrame:
    """Load OPUS Deceptive Opinion Spam Corpus → binary label."""
    df = pd.read_csv(OPUS_CSV)
    # label: deceptive → 1, truthful → 0
    df["label"] = df["deceptive"].apply(lambda x: 1 if "deceptive" in str(x).lower() else 0)
    df["source"] = "opus"
    df = df.rename(columns={"text": "text"})  # already named 'text'
    return df[["text", "label", "source"]]


def load_amazon() -> pd.DataFrame:
    """Load Amazon fake reviews dataset → binary label."""
    df = pd.read_csv(AMAZON_CSV, on_bad_lines="skip", engine="python")
    # label column has values like 'CG' (computer generated = fake) and 'OR' (original = real)
    df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().upper() == "CG" else 0)
    df["source"] = "amazon"
    text_col = "text_" if "text_" in df.columns else "text"
    df = df.rename(columns={text_col: "text"})
    return df[["text", "label", "source"]]


def load_yelp() -> pd.DataFrame:
    """Load Yelp dataset (train + test) → binary label."""
    dfs = []
    for path in [YELP_TRAIN_CSV, YELP_TEST_CSV]:
        if path.exists():
            df = pd.read_csv(path, on_bad_lines="skip", engine="python")
            dfs.append(df)
    if not dfs:
        print("⚠  Yelp CSVs not found — skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.concat(dfs, ignore_index=True)

    # Detect the text and label columns
    text_col = None
    for candidate in ["text", "reviewContent", "review", "text_"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # Fallback: pick the column with longest average string
        str_cols = df.select_dtypes(include="object").columns
        if len(str_cols):
            avg_lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
            text_col = max(avg_lens, key=avg_lens.get)

    label_col = None
    for candidate in ["label", "flagged", "fake", "isFake"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if text_col is None or label_col is None:
        print(f"⚠  Could not identify text/label columns in Yelp. Columns: {df.columns.tolist()}")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = df.rename(columns={text_col: "text", label_col: "label"})
    # Ensure binary label
    df["label"] = df["label"].apply(lambda x: int(x) if str(x).strip() in ("0", "1") else (1 if "fake" in str(x).lower() else 0))
    df["source"] = "yelp"
    return df[["text", "label", "source"]]


def main():
    print("═" * 60)
    print("  Trust Review — Data Pipeline")
    print("═" * 60)

    # ── Load each dataset ────────────────────────────────
    print("\n▸ Loading OPUS …")
    opus = load_opus()
    print(f"  {len(opus)} samples  |  labels: {opus['label'].value_counts().to_dict()}")

    print("▸ Loading Amazon …")
    amazon = load_amazon()
    print(f"  {len(amazon)} samples  |  labels: {amazon['label'].value_counts().to_dict()}")

    print("▸ Loading Yelp …")
    yelp = load_yelp()
    print(f"  {len(yelp)} samples  |  labels: {yelp['label'].value_counts().to_dict()}")

    # ── Merge ────────────────────────────────────────────
    merged = pd.concat([opus, amazon, yelp], ignore_index=True)
    merged = merged.dropna(subset=["text"])
    merged = merged[merged["text"].str.strip().str.len() > 0]
    print(f"\n▸ Merged dataset: {len(merged)} samples")
    print(f"  Label distribution:\n{merged['label'].value_counts().to_string()}")
    print(f"  Source distribution:\n{merged['source'].value_counts().to_string()}")

    # ── Clean text ───────────────────────────────────────
    print("\n▸ Cleaning text …")
    merged["text"] = merged["text"].apply(clean_text)
    merged = merged[merged["text"].str.len() > 10]  # drop very short after cleaning
    print(f"  After cleaning: {len(merged)} samples")

    # ── Stratified split ─────────────────────────────────
    # Stratify by both label and source
    merged["strat_key"] = merged["label"].astype(str) + "_" + merged["source"]
    train_df, test_df = train_test_split(
        merged, test_size=0.2, random_state=42, stratify=merged["strat_key"]
    )
    train_df = train_df.drop(columns=["strat_key"])
    test_df = test_df.drop(columns=["strat_key"])

    # ── Save ─────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"\n✓ Saved train.csv  ({len(train_df)} samples) → {TRAIN_CSV}")
    print(f"✓ Saved test.csv   ({len(test_df)} samples)  → {TEST_CSV}")
    print("═" * 60)


if __name__ == "__main__":
    main()
