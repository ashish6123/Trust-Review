"""
Fine-tune DistilBERT for fake review classification.

Usage:
    cd "e:\Capstone Project"
    python -m training.train_dl
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, accuracy_score

from app.core.config import TRAIN_CSV, TEST_CSV, DISTILBERT_DIR, DL_MAX_LENGTH, DL_MODEL_NAME


# ── Dataset ──────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    print("═" * 60)
    print("  Trust Review — Fine-tune DistilBERT")
    print("═" * 60)

    # ── Config ───────────────────────────────────────────
    EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-5
    MAX_LEN = DL_MAX_LENGTH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Load data ────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # ── Tokenizer ────────────────────────────────────────
    print(f"\n▸ Loading tokenizer: {DL_MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(DL_MODEL_NAME)

    train_dataset = ReviewDataset(
        train_df["text"].astype(str).tolist(),
        train_df["label"].values,
        tokenizer, MAX_LEN,
    )
    test_dataset = ReviewDataset(
        test_df["text"].astype(str).tolist(),
        test_df["label"].values,
        tokenizer, MAX_LEN,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────
    print("▸ Loading pre-trained DistilBERT …")
    model = DistilBertForSequenceClassification.from_pretrained(
        DL_MODEL_NAME, num_labels=2,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # ── Training loop ────────────────────────────────────
    print(f"\n▸ Training for {EPOCHS} epochs …\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # ── Evaluate ─────────────────────────────────────────
    print("\n▸ Evaluating on test set …")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"  Test Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    # ── Save ─────────────────────────────────────────────
    DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(DISTILBERT_DIR))
    tokenizer.save_pretrained(str(DISTILBERT_DIR))
    print(f"\n✓ Model + tokenizer saved → {DISTILBERT_DIR}")
    print("═" * 60)


if __name__ == "__main__":
    main()
