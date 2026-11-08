"""Out-of-distribution test of the shipped DistilBERT on external labelled datasets it never saw.
  - GP Surgery reviews (healthcare; real NHS-style reviews vs DeepSeek-generated fakes)
  - Modern fake reviews (Amazon-style; genuine half = Salminen OR, fake half = DeepSeek-v4 generated)
Usage: python alien.py <gp_test.csv> <modern_test.parquet>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_splits, metrics  # noqa: E402
from app.ml.preprocess import clean_text  # noqa: E402

MODEL = HERE.parent / "models" / "distilbert"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()


def predict(texts):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=256, padding=True, return_tensors="pt")
            out += torch.softmax(model(**enc).logits, 1)[:, 1].tolist()
    return np.array(out)


res = {}
gp = pd.read_csv(sys.argv[1]).dropna(subset=["free_text"])
gp["text"] = gp.free_text.astype(str).map(clean_text)
p = predict(gp.text.tolist())
res["gp_surgery"] = {"all": metrics(gp.label, p),
                     "pred_fake_rate": round(float((p >= .5).mean()), 4),
                     "real_flagged_fake": round(float((p[gp.label.values == 0] >= .5).mean()), 4),
                     "fake_caught": round(float((p[gp.label.values == 1] >= .5).mean()), 4)}

m = pd.read_parquet(sys.argv[2])
m["y"] = (m.label.astype(str).str.upper() == "CG").astype(int)
m["text"] = m.text_.astype(str).map(clean_text)
seen = set(load_splits().text)  # everything in our train/val/test pool
m["seen"] = m.text.isin(seen)
p = predict(m.text.tolist())
unseen = ~m.seen.values
res["modern_llm_fakes"] = {
    "genuine_rows_already_in_our_data": int(m.seen[m.y == 0].sum()),
    "all_unseen_rows": metrics(m.y[unseen], p[unseen]),
    "modern_fakes_caught": round(float((p[m.y.values == 1] >= .5).mean()), 4),
    "genuine_unseen_flagged_fake": round(float((p[(m.y.values == 0) & unseen] >= .5).mean()), 4)}
(HERE / "results" / "alien.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
