"""Fix 5: domain-adaptive pretraining. Continue masked-LM training of distilbert-base-uncased on Yelp
TRAIN-split text only (labels unused, val/test never seen). Saves encoder to cache/mlm_yelp."""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_splits  # noqa: E402

torch.manual_seed(42)
d = load_splits()
texts = d[(d.source == "yelp") & (d.split == "train")].text.astype(str).tolist()
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").cuda()
enc = [tok(t, truncation=True, max_length=256) for t in texts]
coll = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
dl = DataLoader(enc, batch_size=16, shuffle=True, collate_fn=coll)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = torch.cuda.amp.GradScaler()
for ep in range(3):
    model.train(); tot = 0
    for b in dl:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**b).loss
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(); tot += loss.item()
    print(f"epoch {ep + 1} mlm loss {tot / len(dl):.3f}", flush=True)
model.save_pretrained(HERE / "cache" / "mlm_yelp")
