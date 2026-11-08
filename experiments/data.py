"""Rebuild the exact merged dataset + 80/20 test split used by training/data_pipeline.py,
but keep Yelp reviewer metadata and carve a proper validation split out of train.

Test split is identical to the one the shipped DistilBERT was evaluated on (seed 42,
stratified by label+source), so numbers stay comparable with the 90.88% baseline.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.ml.preprocess import clean_text  # noqa: E402

DATA = Path(os.getenv("TRUST_REVIEW_DATASETS_DIR", "D:/Capstone Project/datasets"))
CACHE = Path(__file__).resolve().parent / "cache"

META = ["reviewerID", "date", "rating", "restaurantRating", "reviewUsefulCount", "yelpJoinDate",
        "friendCount", "reviewCount", "firstCount", "usefulCount", "coolCount", "funnyCount",
        "complimentCount", "tipCount", "fanCount"]


def _load_opus():
    df = pd.read_csv(DATA / "deceptive-opinion.csv")
    df["label"] = df["deceptive"].apply(lambda x: 1 if "deceptive" in str(x).lower() else 0)
    df["source"] = "opus"
    return df[["text", "label", "source"]]


def _load_amazon():
    df = pd.read_csv(DATA / "fake reviews dataset.csv", on_bad_lines="skip", engine="python")
    df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().upper() == "CG" else 0)
    df["source"] = "amazon"
    df = df.rename(columns={"text_" if "text_" in df.columns else "text": "text"})
    return df[["text", "label", "source"]]


def _load_yelp():
    dfs = [pd.read_csv(DATA / f, sep="\t", on_bad_lines="skip", engine="python")
           for f in ("yelp_data_train.csv", "yelp_data_test.csv")]
    df = pd.concat(dfs, ignore_index=True).rename(columns={"reviewContent": "text", "flagged": "label"})
    df["label"] = df["label"].apply(
        lambda x: int(x) if str(x).strip() in ("0", "1") else (1 if "fake" in str(x).lower() else 0))
    df["source"] = "yelp"
    return df[["text", "label", "source"] + META]


def _months_since(join, date):
    j = pd.to_datetime(join, format="%B %Y", errors="coerce")
    d = pd.to_datetime(date, format="%m/%d/%Y", errors="coerce")
    return ((d - j).dt.days / 30.44).clip(lower=0)


def behavioural(df: pd.DataFrame) -> pd.DataFrame:
    """Reviewer/behaviour features for Yelp rows (no text)."""
    f = pd.DataFrame(index=df.index)
    for c in ["rating", "restaurantRating", "reviewUsefulCount", "friendCount", "reviewCount", "firstCount",
              "usefulCount", "coolCount", "funnyCount", "complimentCount", "tipCount", "fanCount"]:
        f[c] = pd.to_numeric(df[c], errors="coerce")
    f["rating_dev"] = (f["rating"] - f["restaurantRating"]).abs()
    f["extreme_rating"] = f["rating"].isin([1, 5]).astype(int)
    f["account_age_months"] = _months_since(df["yelpJoinDate"], df["date"])
    f["text_len"] = df["text"].str.len()
    return f


def load_splits():
    """Returns train, val, test DataFrames with a 'split' column. Cached as parquet-free CSVs."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / "all.pkl"
    if p.exists():
        return pd.read_pickle(p)
    merged = pd.concat([_load_opus(), _load_amazon(), _load_yelp()], ignore_index=True)
    merged = merged.dropna(subset=["text"])
    merged = merged[merged["text"].str.strip().str.len() > 0]
    merged["text"] = merged["text"].apply(clean_text)
    merged = merged[merged["text"].str.len() > 10]
    merged["strat_key"] = merged["label"].astype(str) + "_" + merged["source"]
    train, test = train_test_split(merged, test_size=0.2, random_state=42, stratify=merged["strat_key"])
    train, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train["strat_key"])
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    out = out.drop(columns=["strat_key"]).reset_index(names="row_id")
    out.to_pickle(p)
    return out


ALIEN = Path(os.getenv("TRUST_REVIEW_ALIEN_DIR", str(ROOT / "datasets" / "alien")))


def load_splits_v2(fakes_per_generator=10000):
    """v1 splits + modern LLM fakes (DeepSeek-v4, LFM2.5) + GP Surgery healthcare reviews.
    GLM-5.2 appears ONLY in test (unseen generator). Every test/val set stays intact; any train row whose
    text appears in any val/test set is dropped, and any val row found in a test set is dropped."""
    p = CACHE / "all_v2.pkl"
    if p.exists():
        return pd.read_pickle(p)
    base = load_splits()[["row_id", "text", "label", "source", "split"]]
    parts = [base]
    for gen in ["deepseek", "lfm", "glm"]:
        for sp, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
            if gen == "glm" and name != "test":
                continue
            df = pd.read_parquet(ALIEN / f"{gen}_{sp}.parquet")
            df = pd.DataFrame({"text": df.text_.astype(str).map(clean_text),
                               "label": (df.label.astype(str).str.upper() == "CG").astype(int)})
            if name == "train":  # genuine half duplicates our Amazon OR rows; keep a capped sample of fakes only
                df = df[df.label == 1].sample(min(fakes_per_generator, (df.label == 1).sum()), random_state=42)
            parts.append(df.assign(source=f"llm_{gen}", split=name))
    for sp, name in [("train", "train"), ("val", "val"), ("test", "test")]:
        df = pd.read_csv(ALIEN / f"gp_{sp}.csv").dropna(subset=["free_text"])
        parts.append(pd.DataFrame({"text": df.free_text.astype(str).map(clean_text), "label": df.label.astype(int)})
                     .assign(source="gp", split=name))
    d = pd.concat(parts, ignore_index=True)
    d = d[d.text.str.len() > 10]
    test_txt = set(d[d.split == "test"].text)
    held = test_txt | set(d[d.split == "val"].text)
    d = d[~((d.split == "train") & d.text.isin(held))]
    d = d[~((d.split == "val") & d.text.isin(test_txt))]
    d = d.drop_duplicates(subset=["text", "split"])
    d["row_id"] = np.arange(len(d))
    d.to_pickle(p)
    return d


def metrics(y, prob, thr=0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y = np.asarray(y); prob = np.asarray(prob); pred = (prob >= thr).astype(int)
    auc = roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")
    return {"n": int(len(y)), "acc": round(float(accuracy_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4), "auc": round(float(auc), 4)}


def report(df, prob_col="prob"):
    """Overall + per-source metrics on a predictions frame."""
    r = {"overall": metrics(df["label"], df[prob_col])}
    for s, g in df.groupby("source"):
        r[s] = metrics(g["label"], g[prob_col])
    return r
