"""DistilBERT fine-tuning variants, all selected on a proper validation split (test never used for selection).

  python dl.py --name base                       # fix 6: clean val/test protocol
  python dl.py --name weighted --weighted        # fix 3: balance sources in the loss
  python dl.py --name tagged --tag               # fix 4: prepend source tag to text
  python dl.py --name mlm --init cache/mlm_yelp  # fix 5: start from Yelp-adapted encoder
  python dl.py --name loso_yelp --exclude yelp   # fix 6: train without Yelp, test on Yelp
  python dl.py --name all --weighted --tag --init cache/mlm_yelp
Writes results/dl_<name>.json and results/preds_<name>.pkl (val+test probs).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_splits, load_splits_v2, metrics, report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--weighted", action="store_true")
ap.add_argument("--tag", action="store_true")
ap.add_argument("--init", default="distilbert-base-uncased")
ap.add_argument("--exclude", default=None)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--data", default="v1", choices=["v1", "v2"])
args = ap.parse_args()

torch.manual_seed(42); np.random.seed(42)
dev = "cuda"
d = load_splits_v2() if args.data == "v2" else load_splits()
if args.tag:
    d["text"] = d["source"] + " review : " + d["text"]
tr = d[d.split == "train"]; va = d[d.split == "val"]; te = d[d.split == "test"]
if args.exclude:
    tr = tr[tr.source != args.exclude]; va = va[va.source != args.exclude]
    te = d[d.source == args.exclude]  # every row of the unseen source
# Per-sample loss weights: each source contributes equally in total
w_src = (len(tr) / (tr.source.nunique() * tr.source.value_counts())).to_dict() if args.weighted else {}

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(args.init, num_labels=2).to(dev)


def loader(df, shuffle, bs):
    items = list(zip(df.text.astype(str), df.label.astype(int), df.source.map(lambda s: w_src.get(s, 1.0))))

    def collate(b):
        enc = tok([x[0] for x in b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([x[1] for x in b]); enc["w"] = torch.tensor([x[2] for x in b], dtype=torch.float)
        return enc
    return DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate)


def predict(df):
    model.eval(); out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for b in loader(df, False, 64):
            b = {k: v.to(dev) for k, v in b.items() if k not in ("labels", "w")}
            out += torch.softmax(model(**b).logits.float(), 1)[:, 1].tolist()
    return np.array(out)


BS, ACC = 16, 2
dl_tr = loader(tr, True, BS)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
steps = len(dl_tr) // ACC * args.epochs
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
ce = torch.nn.CrossEntropyLoss(reduction="none")
best, hist, t0 = -1, [], time.time()
ckpt = HERE / "cache" / f"ckpt_{args.name}"
for ep in range(args.epochs):
    model.train()
    for i, b in enumerate(dl_tr):
        b = {k: v.to(dev) for k, v in b.items()}
        w, y = b.pop("w"), b.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = (ce(model(**b).logits.float(), y) * w).sum() / w.sum() / ACC
        scaler.scale(loss).backward()
        if (i + 1) % ACC == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
    pv = predict(va); m = metrics(va.label, pv)
    # Select on macro F1 across sources so Amazon cannot dominate model choice
    macro = float(np.mean([metrics(g.label, pv[(va.source == s).values])["f1"] for s, g in va.groupby("source")]))
    hist.append({"epoch": ep + 1, "val": m, "val_macro_source_f1": round(macro, 4), "min": round((time.time() - t0) / 60, 1)})
    print(hist[-1], flush=True)
    if macro > best:
        best = macro; model.save_pretrained(ckpt)

model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(dev)
pv, pt = predict(va), predict(te)
res = {"args": vars(args), "history": hist, "val": report(va.assign(prob=pv)), "test": report(te.assign(prob=pt))}
(HERE / "results" / f"dl_{args.name}.json").write_text(json.dumps(res, indent=2))
pd.concat([va.assign(prob=pv), te.assign(prob=pt)])[["row_id", "split", "source", "label", "prob"]] \
    .to_pickle(HERE / "results" / f"preds_{args.name}.pkl")
print("TEST", json.dumps(res["test"]))
