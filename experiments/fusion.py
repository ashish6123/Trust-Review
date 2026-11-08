"""Fix 1 (full): late fusion of a DistilBERT text probability with the behavioural model on Yelp.
Fusion weight tuned on the Yelp val split; reported on Yelp test.  Usage: python fusion.py base [all ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import metrics  # noqa: E402

R = HERE / "results"
beh = pd.read_pickle(R / "beh_probs.pkl").set_index("row_id")
out = {}
for name in sys.argv[1:]:
    p = pd.read_pickle(R / f"preds_{name}.pkl")
    p = p[p.source == "yelp"].set_index("row_id").join(beh["beh"])
    va, te = p[p.split == "val"], p[p.split == "test"]
    w = max(np.linspace(0, 1, 21), key=lambda w: metrics(va.label, w * va.prob + (1 - w) * va.beh)["acc"])
    out[name] = {"w_text": round(float(w), 2), "text_only": metrics(te.label, te.prob),
                 "behavioural_only": metrics(te.label, te.beh),
                 "fused": metrics(te.label, w * te.prob + (1 - w) * te.beh)}
(R / "fusion.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
