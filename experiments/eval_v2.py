"""Score saved checkpoints on the v2 test set (old sources + modern LLM fakes + unseen GLM + GP healthcare).
Usage: python eval_v2.py base v2_all   (reads cache/ckpt_<name>)"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_splits_v2, report  # noqa: E402

te = load_splits_v2().query("split == 'test'")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
out = {}
for name in sys.argv[1:]:
    tag = "_tagged" in name or name in ("tagged", "all")
    texts = (te.source.where(~te.source.str.startswith("llm") & (te.source != "gp"), "amazon") + " review : " + te.text) if tag else te.text
    m = AutoModelForSequenceClassification.from_pretrained(HERE / "cache" / f"ckpt_{name}").cuda().eval()
    probs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        tl = texts.astype(str).tolist()
        for i in range(0, len(tl), 64):
            enc = tok(tl[i:i + 64], truncation=True, max_length=256, padding=True, return_tensors="pt").to("cuda")
            probs += torch.softmax(m(**enc).logits.float(), 1)[:, 1].tolist()
    r = report(te.assign(prob=np.array(probs)))
    for s, g in te.assign(prob=np.array(probs)).groupby("source"):
        r[s]["fake_caught"] = round(float((g.prob[g.label == 1] >= .5).mean()), 4)
    out[name] = r
    print(name, json.dumps(r), flush=True)
(HERE / "results" / "eval_v2.json").write_text(json.dumps(out, indent=2))
