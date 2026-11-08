"""CPU experiments: per-source baselines, leave-one-source-out (LOSO), behavioural model, text+behaviour fusion.
Writes results/classical.json and results/beh_probs.pkl (behavioural probs for val/test Yelp, used by fusion.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import behavioural, load_splits, metrics, report  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SOURCES = ["amazon", "opus", "yelp"]


def tfidf_lr(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True, min_df=2)
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(vec.fit_transform(train["text"]), train["label"])
    return clf.predict_proba(vec.transform(test["text"]))[:, 1]


def main():
    d = load_splits()
    trn = d[d.split.isin(["train", "val"])]  # classical models do not need early stopping
    tst = d[d.split == "test"]
    res = {}

    # Reference: one TF-IDF model on the mixed data
    res["tfidf_mixed"] = report(tst.assign(prob=tfidf_lr(trn, tst)))

    # Fix 2: per-source models (train and test on the same source)
    res["tfidf_per_source"] = {s: metrics(tst[tst.source == s].label,
                                          tfidf_lr(trn[trn.source == s], tst[tst.source == s]))
                               for s in SOURCES}

    # Fix 6 (part): leave-one-source-out generalisation, tested on ALL rows of the held-out source
    res["tfidf_loso"] = {s: metrics(d[d.source == s].label, tfidf_lr(trn[trn.source != s], d[d.source == s]))
                         for s in SOURCES}

    # Fix 1: behavioural features on Yelp
    y = d[d.source == "yelp"].copy()
    X = behavioural(y)
    ytr, yval, yte = (y.split == "train"), (y.split == "val"), (y.split == "test")
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                        early_stopping=True, validation_fraction=0.1, random_state=42)
    gb.fit(X[ytr], y.label[ytr])
    beh = pd.Series(gb.predict_proba(X)[:, 1], index=y.index)
    res["yelp_behavioural"] = {"val": metrics(y.label[yval], beh[yval]), "test": metrics(y.label[yte], beh[yte])}

    # Reviewer leakage check: test rows whose reviewer also appears in train
    seen = set(y.loc[ytr, "reviewerID"])
    overlap = y.loc[yte, "reviewerID"].isin(seen)
    res["yelp_behavioural"]["test_unseen_reviewers"] = metrics(y.label[yte][~overlap], beh[yte][~overlap])
    res["yelp_behavioural"]["test_reviewer_overlap_rows"] = int(overlap.sum())

    # Text (Yelp-only TF-IDF) + behaviour fusion, weight tuned on val
    txt_tr = y[ytr]
    p_txt = pd.Series(tfidf_lr(txt_tr, y), index=y.index)
    best = max(np.linspace(0, 1, 21), key=lambda w: metrics(y.label[yval], w * p_txt[yval] + (1 - w) * beh[yval])["acc"])
    fused = best * p_txt + (1 - best) * beh
    res["yelp_tfidf_plus_behavioural"] = {"w_text": round(float(best), 2), "test": metrics(y.label[yte], fused[yte])}

    pd.DataFrame({"row_id": y.row_id, "split": y.split, "label": y.label, "beh": beh}).to_pickle(OUT / "beh_probs.pkl")
    (OUT / "classical.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
